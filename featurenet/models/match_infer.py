from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .infer import (
    decode_minutiae_rows,
    ensure_stdlib_copy_module,
    load_checkpoint_model,
    load_bgr_image,
    preprocess_input_bgr,
    preprocess_input_image,
    print_output_stats,
    run_inference,
    save_minutiae_csv,
    save_pose_sidecars,
    _resolve_device,
)

ensure_stdlib_copy_module()

import cv2


def _default_output_dir() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("match_outputs") / f"featurenet_mcc_{timestamp}"


def _save_mask_png(mask_tensor: Any, path: Path) -> None:
    mask_np = mask_tensor.detach().cpu().numpy()
    if mask_np.ndim != 4:
        raise ValueError(f"expected mask tensor shape [B,1,H,W], got {tuple(mask_np.shape)}")
    mask_img = (mask_np[0, 0] > 0.5).astype(np.uint8) * 255
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask_img):
        raise RuntimeError(f"failed to write mask image to {path}")


def _grayscale_bgr_with_black_background(bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        gray = gray.copy()
        gray[mask <= 0] = 0
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _foreground_mask_with_main(image_path: Path) -> np.ndarray:
    import main as crop_main

    full_bgr = crop_main.load_bgr_image(image_path)
    foreground_mask = crop_main.rembg_mask_from_bgr(full_bgr)
    crop_main.validate_foreground_area(foreground_mask, minimum_ratio=0.03)
    return foreground_mask


def _rotate_bound(array: np.ndarray, angle_degrees: float, interpolation: int, border_value: int = 0) -> np.ndarray:
    height, width = array.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_degrees), 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    matrix[0, 2] += (new_width / 2.0) - center[0]
    matrix[1, 2] += (new_height / 2.0) - center[1]
    return cv2.warpAffine(
        array,
        matrix,
        (new_width, new_height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _rotate_foreground_to_horizontal(bgr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    ys, xs = np.where(mask > 0)
    if xs.size < 2 or ys.size < 2:
        raise RuntimeError("cannot rotate foreground to horizontal from an empty mask")
    coords = np.column_stack([xs, ys]).astype(np.float32)
    coords -= np.mean(coords, axis=0, keepdims=True)
    covariance = np.cov(coords, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = float(np.degrees(np.arctan2(float(axis[1]), float(axis[0]))))
    if angle > 90.0:
        angle -= 180.0
    if angle < -90.0:
        angle += 180.0
    rotate_degrees = -angle
    rotated_bgr = _rotate_bound(bgr, rotate_degrees, cv2.INTER_LINEAR, border_value=0)
    rotated_mask = _rotate_bound(mask, rotate_degrees, cv2.INTER_NEAREST, border_value=0)
    return rotated_bgr, rotated_mask, rotate_degrees


def _crop_distal_phalanx_with_main(
    *,
    image_path: Path,
    crop_output_dir: Path,
) -> dict[str, Any]:
    import main as crop_main

    full_bgr = crop_main.load_bgr_image(image_path)
    coarse_mask = crop_main.rembg_mask_from_bgr(full_bgr)
    crop_main.validate_foreground_area(coarse_mask, minimum_ratio=0.03)

    contours, _ = cv2.findContours(coarse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("no contour available for crop")
    contour = max(contours, key=cv2.contourArea)

    axis_u, axis_v, length_l, _ = crop_main.estimate_finger_axis(contour)
    tip_center, tip_to_base = crop_main.select_fingertip_end(
        contour,
        axis_u,
        axis_v,
        length_l,
        coarse_mask.shape[:2],
    )
    width_near_tip = crop_main.local_width_from_mask(coarse_mask, tip_center, tip_to_base, axis_v, length_l, 0.16)
    width_stable = crop_main.local_width_from_mask(coarse_mask, tip_center, tip_to_base, axis_v, length_l, 0.28)
    distal_width = max(width_near_tip, 0.88 * width_stable)

    # Relaxed rectangular distal crop coefficients (match_infer specific).
    inward_extension = min(0.70 * length_l, max(2.25 * distal_width, 0.50 * length_l))
    outward_extension = min(0.08 * length_l, 0.36 * distal_width)
    half_width = 0.95 * distal_width
    margin_px = max(16, int(round(0.07 * max(length_l, distal_width))))

    outer_center = tip_center - (tip_to_base * outward_extension)
    inner_center = tip_center + (tip_to_base * inward_extension)
    crop_quad = np.array(
        [
            outer_center - (axis_v * half_width),
            outer_center + (axis_v * half_width),
            inner_center + (axis_v * half_width),
            inner_center - (axis_v * half_width),
        ],
        dtype=np.float32,
    )
    x_min = int(np.floor(np.min(crop_quad[:, 0]))) - margin_px
    y_min = int(np.floor(np.min(crop_quad[:, 1]))) - margin_px
    x_max = int(np.ceil(np.max(crop_quad[:, 0]))) + margin_px
    y_max = int(np.ceil(np.max(crop_quad[:, 1]))) + margin_px
    height, width = coarse_mask.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)
    if (x_max - x_min) < 32 or (y_max - y_min) < 32:
        raise RuntimeError("computed relaxed distal crop is too small")
    crop_bbox = (x_min, y_min, x_max, y_max)
    cropped_bgr = crop_main.crop_image(full_bgr, crop_bbox)

    fallback_reason: str | None = None
    crop_mode = "hybrid"
    try:
        cropped_fg_mask = crop_main.rembg_mask_from_bgr(cropped_bgr)
        crop_main.validate_foreground_area(cropped_fg_mask, minimum_ratio=0.06)
    except Exception:
        cropped_fg_mask = np.full(cropped_bgr.shape[:2], 255, dtype=np.uint8)
        fallback_reason = "cropped_rembg_failed"
        crop_mode = "fallback_relaxed_rect"

    tip_center_local = np.array([tip_center[0] - float(x_min), tip_center[1] - float(y_min)], dtype=np.float32)
    ys, xs = np.where(cropped_fg_mask > 0)
    if ys.size == 0:
        distal_mask = np.full(cropped_fg_mask.shape, 255, dtype=np.uint8)
        crop_mode = "fallback_relaxed_rect"
        fallback_reason = fallback_reason or "empty_cropped_foreground"
    else:
        points = np.stack([xs, ys], axis=1).astype(np.float32)
        relative = points - tip_center_local
        long_coord = relative @ tip_to_base
        cross_coord = relative @ axis_v

        long_min = -0.06 * length_l
        long_max = inward_extension * 1.03
        if long_max <= long_min + 1e-6:
            distal_mask = np.full(cropped_fg_mask.shape, 255, dtype=np.uint8)
            crop_mode = "fallback_relaxed_rect"
            fallback_reason = fallback_reason or "invalid_longitudinal_band"
        else:
            t = np.clip((long_coord - long_min) / (long_max - long_min), 0.0, 1.0)
            envelope = distal_width * (0.58 + (0.55 * t))
            keep = (long_coord >= long_min) & (long_coord <= long_max) & (np.abs(cross_coord) <= envelope)

            distal_mask = np.zeros_like(cropped_fg_mask, dtype=np.uint8)
            distal_mask[ys[keep], xs[keep]] = 255
            distal_mask = cv2.bitwise_and(distal_mask, cropped_fg_mask)

            if np.any(distal_mask > 0):
                distal_mask = cv2.morphologyEx(distal_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)
                distal_mask = cv2.morphologyEx(distal_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
                contours_mask, _ = cv2.findContours(distal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if contours_mask:
                    largest = max(contours_mask, key=cv2.contourArea)
                    refined = np.zeros_like(distal_mask)
                    cv2.drawContours(refined, [largest], -1, 255, thickness=cv2.FILLED)
                    distal_mask = refined

            fg_count = int(np.count_nonzero(cropped_fg_mask))
            distal_count = int(np.count_nonzero(distal_mask))
            distal_ratio = float(distal_count / max(fg_count, 1))
            if distal_count < 256 or distal_ratio < 0.18:
                distal_mask = np.full(cropped_fg_mask.shape, 255, dtype=np.uint8)
                crop_mode = "fallback_relaxed_rect"
                fallback_reason = fallback_reason or "distal_mask_too_small"

    cropped_distal = cropped_bgr.copy()
    if crop_mode == "hybrid":
        cropped_distal[distal_mask <= 0] = 0
    else:
        distal_mask = np.full(cropped_bgr.shape[:2], 255, dtype=np.uint8)

    crop_output_dir.mkdir(parents=True, exist_ok=True)
    coarse_mask_path = crop_output_dir / "coarse_mask.png"
    cropped_path = crop_output_dir / "cropped.png"
    distal_mask_path = crop_output_dir / "distal_mask.png"
    cropped_distal_path = crop_output_dir / "cropped_distal.png"
    crop_bbox_path = crop_output_dir / "crop_bbox.json"

    if not cv2.imwrite(str(coarse_mask_path), coarse_mask):
        raise RuntimeError(f"failed to write coarse mask image to {coarse_mask_path}")
    if not cv2.imwrite(str(cropped_path), cropped_bgr):
        raise RuntimeError(f"failed to write cropped image to {cropped_path}")
    if not cv2.imwrite(str(distal_mask_path), distal_mask):
        raise RuntimeError(f"failed to write distal mask image to {distal_mask_path}")
    if not cv2.imwrite(str(cropped_distal_path), cropped_distal):
        raise RuntimeError(f"failed to write masked cropped image to {cropped_distal_path}")

    x_min, y_min, x_max, y_max = crop_bbox
    coarse_count = int(np.count_nonzero(coarse_mask))
    distal_count = int(np.count_nonzero(distal_mask))
    crop_bbox_path.write_text(
        json.dumps(
            {
                "x_min": int(x_min),
                "y_min": int(y_min),
                "x_max": int(x_max),
                "y_max": int(y_max),
                "crop_mode": crop_mode,
                "fallback_reason": fallback_reason,
                "coarse_mask_pixels": coarse_count,
                "distal_mask_pixels": distal_count,
                "distal_to_coarse_ratio": float(distal_count / max(coarse_count, 1)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "full_bgr": full_bgr,
        "cropped_bgr": cropped_bgr,
        "inference_bgr": cropped_distal,
        "distal_mask": distal_mask,
        "crop_bbox": tuple(int(v) for v in crop_bbox),
        "crop_mode": crop_mode,
        "fallback_reason": fallback_reason,
        "coarse_mask_path": coarse_mask_path,
        "cropped_path": cropped_path,
        "distal_mask_path": distal_mask_path,
        "cropped_distal_path": cropped_distal_path,
        "crop_bbox_path": crop_bbox_path,
        "distal_mask_pixels": distal_count,
        "coarse_mask_pixels": coarse_count,
    }


def _fallback_distal_mask_with_main(
    *,
    image_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    crop_result = _crop_distal_phalanx_with_main(
        image_path=image_path,
        crop_output_dir=output_dir / "fallback_distal_crop",
    )
    full_bgr = crop_result["full_bgr"]
    full_mask = np.zeros(full_bgr.shape[:2], dtype=np.uint8)
    x_min, y_min, x_max, y_max = crop_result["crop_bbox"]
    local_mask = crop_result["distal_mask"]
    target_h = int(y_max) - int(y_min)
    target_w = int(x_max) - int(x_min)
    if local_mask.shape[:2] != (target_h, target_w):
        local_mask = cv2.resize(local_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    full_mask[int(y_min) : int(y_max), int(x_min) : int(x_max)] = np.maximum(
        full_mask[int(y_min) : int(y_max), int(x_min) : int(x_max)],
        np.where(local_mask > 0, 255, 0).astype(np.uint8),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fallback_mask_path = output_dir / "fallback_distal_mask.png"
    fallback_input_path = output_dir / "fallback_distal_black_gray.png"
    fallback_input = _grayscale_bgr_with_black_background(full_bgr, full_mask)
    cv2.imwrite(str(fallback_mask_path), full_mask)
    cv2.imwrite(str(fallback_input_path), fallback_input)
    return {
        "full_bgr": full_bgr,
        "fallback_mask": full_mask,
        "fallback_mask_path": fallback_mask_path,
        "fallback_input_path": fallback_input_path,
        "crop": {
            "crop_bbox": list(crop_result["crop_bbox"]),
            "crop_mode": crop_result["crop_mode"],
            "fallback_reason": crop_result["fallback_reason"],
            "coarse_mask_path": str(crop_result["coarse_mask_path"]),
            "cropped_path": str(crop_result["cropped_path"]),
            "distal_mask_path": str(crop_result["distal_mask_path"]),
            "cropped_distal_path": str(crop_result["cropped_distal_path"]),
            "crop_bbox_path": str(crop_result["crop_bbox_path"]),
        },
    }


def _run_single_image_inference(
    *,
    image_path: Path,
    label: str,
    model: Any,
    device: Any,
    score_threshold: float,
    apply_nms: bool,
    solov2_score_thr: float,
    solov2_input_mode: str,
    allow_distal_fallback: bool,
    image_output_dir: Path,
) -> dict[str, Any]:
    preprocess_dir = image_output_dir / "preprocess"
    solov2_input_artifacts: dict[str, Any]
    fallback_artifacts: dict[str, Any] | None = None
    fallback_mask: np.ndarray | None = None
    fallback_mask_source: str | None = None
    preprocess_bgr: np.ndarray | None = None

    if solov2_input_mode == "raw":
        preprocess_bgr = load_bgr_image(image_path)
        solov2_input_artifacts: dict[str, Any] = {"mode": "raw"}
    elif solov2_input_mode == "gray":
        full_bgr = load_bgr_image(image_path)
        solov2_bgr = _grayscale_bgr_with_black_background(full_bgr)
        solov2_input_path = image_output_dir / "solov2_input_gray.png"
        image_output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(solov2_input_path), solov2_bgr)
        preprocess_bgr = solov2_bgr
        solov2_input_artifacts = {"mode": "gray", "solov2_input_png": str(solov2_input_path)}
    elif solov2_input_mode == "foreground-black-gray":
        full_bgr = load_bgr_image(image_path)
        foreground_mask = _foreground_mask_with_main(image_path)
        solov2_bgr = _grayscale_bgr_with_black_background(full_bgr, foreground_mask)
        solov2_input_path = image_output_dir / "solov2_input_foreground_black_gray.png"
        foreground_mask_path = image_output_dir / "foreground_mask.png"
        image_output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(solov2_input_path), solov2_bgr)
        cv2.imwrite(str(foreground_mask_path), foreground_mask)
        preprocess_bgr = solov2_bgr
        solov2_input_artifacts = {
            "mode": "foreground-black-gray",
            "solov2_input_png": str(solov2_input_path),
            "foreground_mask_png": str(foreground_mask_path),
        }
    elif solov2_input_mode == "foreground-black-gray-horizontal":
        full_bgr = load_bgr_image(image_path)
        foreground_mask = _foreground_mask_with_main(image_path)
        horizontal_bgr, horizontal_mask, rotation_degrees = _rotate_foreground_to_horizontal(
            full_bgr,
            foreground_mask,
        )
        solov2_bgr = _grayscale_bgr_with_black_background(horizontal_bgr, horizontal_mask)
        solov2_input_path = image_output_dir / "solov2_input_foreground_black_gray_horizontal.png"
        foreground_mask_path = image_output_dir / "foreground_mask_horizontal.png"
        image_output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(solov2_input_path), solov2_bgr)
        cv2.imwrite(str(foreground_mask_path), horizontal_mask)
        preprocess_bgr = solov2_bgr
        solov2_input_artifacts = {
            "mode": "foreground-black-gray-horizontal",
            "solov2_input_png": str(solov2_input_path),
            "foreground_mask_png": str(foreground_mask_path),
            "rotation_degrees": float(rotation_degrees),
        }
    elif solov2_input_mode == "crop-black-gray":
        crop_result = _crop_distal_phalanx_with_main(
            image_path=image_path,
            crop_output_dir=image_output_dir / "solov2_input_crop",
        )
        solov2_bgr = _grayscale_bgr_with_black_background(
            crop_result["cropped_bgr"],
            crop_result["distal_mask"],
        )
        solov2_input_path = image_output_dir / "solov2_input_crop_black_gray.png"
        image_output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(solov2_input_path), solov2_bgr)
        preprocess_bgr = solov2_bgr
        solov2_input_artifacts = {
            "mode": "crop-black-gray",
            "solov2_input_png": str(solov2_input_path),
            "crop": {
                "crop_bbox": list(crop_result["crop_bbox"]),
                "crop_mode": crop_result["crop_mode"],
                "fallback_reason": crop_result["fallback_reason"],
                "coarse_mask_path": str(crop_result["coarse_mask_path"]),
                "cropped_path": str(crop_result["cropped_path"]),
                "distal_mask_path": str(crop_result["distal_mask_path"]),
                "cropped_distal_path": str(crop_result["cropped_distal_path"]),
                "crop_bbox_path": str(crop_result["crop_bbox_path"]),
            },
        }
    else:
        raise ValueError(f"unsupported --solov2-input-mode: {solov2_input_mode}")

    assert preprocess_bgr is not None
    try:
        image_tensor, mask_tensor, input_shape_hw = preprocess_input_bgr(
            preprocess_bgr,
            save_preprocess_dir=preprocess_dir,
            solov2_score_thr=solov2_score_thr,
        )
        mask_source = "solov2"
    except RuntimeError as exc:
        if not allow_distal_fallback or "distal phalanx" not in str(exc):
            raise
        fallback = _fallback_distal_mask_with_main(
            image_path=image_path,
            output_dir=image_output_dir,
        )
        fallback_mask = fallback["fallback_mask"]
        fallback_mask_source = "foreground_distal_fallback"
        image_tensor, mask_tensor, input_shape_hw = preprocess_input_bgr(
            fallback["full_bgr"],
            save_preprocess_dir=preprocess_dir,
            solov2_score_thr=solov2_score_thr,
            fallback_mask=fallback_mask,
            fallback_mask_source=fallback_mask_source,
        )
        fallback_artifacts = {
            "reason": str(exc),
            "mask_source": fallback_mask_source,
            "fallback_mask_png": str(fallback["fallback_mask_path"]),
            "fallback_input_png": str(fallback["fallback_input_path"]),
            "crop": fallback["crop"],
        }
        mask_source = fallback_mask_source
    outputs = run_inference(
        model=model,
        image_tensor=image_tensor,
        mask_tensor=mask_tensor,
        device=device,
    )
    print(f"[{label}] FeatureNet raw logit stats:")
    print_output_stats(outputs)

    minutiae_rows = decode_minutiae_rows(
        outputs=outputs,
        input_shape_hw=input_shape_hw,
        score_threshold=score_threshold,
        apply_nms=apply_nms,
    )

    minutiae_csv = image_output_dir / "minutiae.csv"
    mask_png = image_output_dir / "mask.png"
    save_minutiae_csv(minutiae_rows, minutiae_csv)
    orientation_npy, ridge_period_npy = save_pose_sidecars(outputs, image_output_dir)
    _save_mask_png(mask_tensor, mask_png)

    return {
        "minutiae_rows": minutiae_rows,
        "minutiae_csv": minutiae_csv,
        "mask_png": mask_png,
        "orientation_npy": orientation_npy,
        "ridge_period_npy": ridge_period_npy,
        "preprocess_dir": preprocess_dir,
        "inference_input_shape_hw": [int(input_shape_hw[0]), int(input_shape_hw[1])],
        "solov2_score_thr": float(solov2_score_thr),
        "solov2_input": solov2_input_artifacts,
        "mask_source": mask_source,
        "distal_fallback": fallback_artifacts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FeatureNet inference on two images and match decoded minutiae with MCC."
    )
    parser.add_argument("--image-a", type=Path, required=True, help="Path to first image.")
    parser.add_argument("--image-b", type=Path, required=True, help="Path to second image.")
    parser.add_argument("--weights-path", type=Path, default=Path("weights") / "best.pt", help="Path to checkpoint.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--method", type=str, default="LSA", help="MCC method passed to main.match_minutiae_csv.")
    parser.add_argument(
        "--minutia-score-threshold",
        type=float,
        default=0.6,
        help="Score threshold (after sigmoid) for minutia decoding.",
    )
    parser.add_argument(
        "--disable-minutia-nms",
        action="store_true",
        help="Disable 3x3 local-maximum suppression during minutia decoding.",
    )
    parser.add_argument(
        "--solov2-score-thr",
        type=float,
        default=0.15,
        help="Minimum SOLOv2 distal phalanx detection score for canonical preprocessing.",
    )
    parser.add_argument(
        "--solov2-input-mode",
        choices=("raw", "gray", "foreground-black-gray", "foreground-black-gray-horizontal", "crop-black-gray"),
        default="raw",
        help="Image preparation before SOLOv2. Use foreground-black-gray-horizontal for real-world photos that differ from black-background grayscale training images.",
    )
    parser.add_argument(
        "--allow-distal-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If SOLOv2 finds no distal phalanx, continue with a foreground-geometry distal mask fallback.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run output directory (default: match_outputs/featurenet_mcc_<timestamp>).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()

    image_a = args.image_a.resolve()
    image_b = args.image_b.resolve()
    weights_path = args.weights_path.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else _default_output_dir().resolve()

    if not image_a.exists():
        raise FileNotFoundError(f"image not found: {image_a}")
    if not image_b.exists():
        raise FileNotFoundError(f"image not found: {image_b}")
    if not weights_path.exists():
        raise FileNotFoundError(f"weights file not found: {weights_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    a_dir = output_dir / "a"
    b_dir = output_dir / "b"
    a_dir.mkdir(parents=True, exist_ok=True)
    b_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    model = load_checkpoint_model(weights_path, device)

    result_a = _run_single_image_inference(
        image_path=image_a,
        label="A",
        model=model,
        device=device,
        score_threshold=float(args.minutia_score_threshold),
        apply_nms=not bool(args.disable_minutia_nms),
        solov2_score_thr=float(args.solov2_score_thr),
        solov2_input_mode=str(args.solov2_input_mode),
        allow_distal_fallback=bool(args.allow_distal_fallback),
        image_output_dir=a_dir,
    )
    result_b = _run_single_image_inference(
        image_path=image_b,
        label="B",
        model=model,
        device=device,
        score_threshold=float(args.minutia_score_threshold),
        apply_nms=not bool(args.disable_minutia_nms),
        solov2_score_thr=float(args.solov2_score_thr),
        solov2_input_mode=str(args.solov2_input_mode),
        allow_distal_fallback=bool(args.allow_distal_fallback),
        image_output_dir=b_dir,
    )

    import main as mcc_main

    score, sim_matrix = mcc_main.match_minutiae_csv(
        path_a=result_a["minutiae_csv"],
        path_b=result_b["minutiae_csv"],
        method=args.method,
        mask_path_a=result_a["mask_png"],
        mask_path_b=result_b["mask_png"],
        overlap_mode="auto",
    )

    summary = {
        "image_a": str(image_a),
        "image_b": str(image_b),
        "weights_path": str(weights_path),
        "device": str(device),
        "method": args.method,
        "minutia_score_threshold": float(args.minutia_score_threshold),
        "solov2_score_thr": float(args.solov2_score_thr),
        "solov2_input_mode": str(args.solov2_input_mode),
        "allow_distal_fallback": bool(args.allow_distal_fallback),
        "minutia_nms_enabled": not bool(args.disable_minutia_nms),
        "mask_source_a": result_a["mask_source"],
        "mask_source_b": result_b["mask_source"],
        "minutiae_count_a": len(result_a["minutiae_rows"]),
        "minutiae_count_b": len(result_b["minutiae_rows"]),
        "mcc_score": float(score),
        "similarity_matrix_shape": list(np.asarray(sim_matrix).shape),
        "artifacts": {
            "run_dir": str(output_dir),
            "a_minutiae_csv": str(result_a["minutiae_csv"]),
            "a_mask_png": str(result_a["mask_png"]),
            "a_orientation_npy": str(result_a["orientation_npy"]),
            "a_ridge_period_npy": str(result_a["ridge_period_npy"]),
            "a_preprocess_dir": str(result_a["preprocess_dir"]),
            "b_minutiae_csv": str(result_b["minutiae_csv"]),
            "b_mask_png": str(result_b["mask_png"]),
            "b_orientation_npy": str(result_b["orientation_npy"]),
            "b_ridge_period_npy": str(result_b["ridge_period_npy"]),
            "b_preprocess_dir": str(result_b["preprocess_dir"]),
        },
        "preprocessing": {
            "a": {
                "canonical_preprocess": "preprocess.py",
                "solov2_score_thr": result_a["solov2_score_thr"],
                "solov2_input": result_a["solov2_input"],
                "mask_source": result_a["mask_source"],
                "distal_fallback": result_a["distal_fallback"],
                "inference_input_shape_hw": result_a["inference_input_shape_hw"],
            },
            "b": {
                "canonical_preprocess": "preprocess.py",
                "solov2_score_thr": result_b["solov2_score_thr"],
                "solov2_input": result_b["solov2_input"],
                "mask_source": result_b["mask_source"],
                "distal_fallback": result_b["distal_fallback"],
                "inference_input_shape_hw": result_b["inference_input_shape_hw"],
            },
        },
        "wall_seconds": round(time.time() - started_at, 3),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"MCC match score ({args.method}): {float(score):.6f}")
    print(f"Similarity matrix shape: {tuple(np.asarray(sim_matrix).shape)}")
    print(f"Saved run summary: {summary_path}")
    print(f"Saved A minutiae CSV: {result_a['minutiae_csv']}")
    print(f"Saved A mask: {result_a['mask_png']}")
    print(f"Saved A preprocess dir: {result_a['preprocess_dir']}")
    print(f"Saved B minutiae CSV: {result_b['minutiae_csv']}")
    print(f"Saved B mask: {result_b['mask_png']}")
    print(f"Saved B preprocess dir: {result_b['preprocess_dir']}")


if __name__ == "__main__":
    main()

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
    preprocess_input_bgr,
    print_output_stats,
    run_inference,
    save_minutiae_csv,
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


def _run_single_image_inference(
    *,
    image_path: Path,
    label: str,
    model: Any,
    device: Any,
    score_threshold: float,
    apply_nms: bool,
    image_output_dir: Path,
) -> dict[str, Any]:
    crop_dir = image_output_dir / "crop"
    crop_result = _crop_distal_phalanx_with_main(
        image_path=image_path,
        crop_output_dir=crop_dir,
    )

    preprocess_dir = image_output_dir / "preprocess"
    image_tensor, mask_tensor, input_shape_hw = preprocess_input_bgr(
        full_bgr=crop_result["inference_bgr"],
        save_preprocess_dir=preprocess_dir,
    )
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
    _save_mask_png(mask_tensor, mask_png)

    return {
        "minutiae_rows": minutiae_rows,
        "minutiae_csv": minutiae_csv,
        "mask_png": mask_png,
        "preprocess_dir": preprocess_dir,
        "crop_dir": crop_dir,
        "crop_bbox": crop_result["crop_bbox"],
        "crop_mode": crop_result["crop_mode"],
        "fallback_reason": crop_result["fallback_reason"],
        "coarse_mask_path": crop_result["coarse_mask_path"],
        "cropped_path": crop_result["cropped_path"],
        "distal_mask_path": crop_result["distal_mask_path"],
        "cropped_distal_path": crop_result["cropped_distal_path"],
        "crop_bbox_path": crop_result["crop_bbox_path"],
        "original_shape_hw": [int(crop_result["full_bgr"].shape[0]), int(crop_result["full_bgr"].shape[1])],
        "cropped_shape_hw": [int(crop_result["cropped_bgr"].shape[0]), int(crop_result["cropped_bgr"].shape[1])],
        "distal_mask_pixels": int(crop_result["distal_mask_pixels"]),
        "coarse_mask_pixels": int(crop_result["coarse_mask_pixels"]),
        "inference_input_shape_hw": [int(input_shape_hw[0]), int(input_shape_hw[1])],
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
        image_output_dir=a_dir,
    )
    result_b = _run_single_image_inference(
        image_path=image_b,
        label="B",
        model=model,
        device=device,
        score_threshold=float(args.minutia_score_threshold),
        apply_nms=not bool(args.disable_minutia_nms),
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
        "minutia_nms_enabled": not bool(args.disable_minutia_nms),
        "minutiae_count_a": len(result_a["minutiae_rows"]),
        "minutiae_count_b": len(result_b["minutiae_rows"]),
        "mcc_score": float(score),
        "similarity_matrix_shape": list(np.asarray(sim_matrix).shape),
        "artifacts": {
            "run_dir": str(output_dir),
            "a_minutiae_csv": str(result_a["minutiae_csv"]),
            "a_mask_png": str(result_a["mask_png"]),
            "a_crop_dir": str(result_a["crop_dir"]),
            "a_crop_coarse_mask_png": str(result_a["coarse_mask_path"]),
            "a_crop_cropped_png": str(result_a["cropped_path"]),
            "a_crop_distal_mask_png": str(result_a["distal_mask_path"]),
            "a_crop_cropped_distal_png": str(result_a["cropped_distal_path"]),
            "a_crop_bbox_json": str(result_a["crop_bbox_path"]),
            "a_preprocess_dir": str(result_a["preprocess_dir"]),
            "b_minutiae_csv": str(result_b["minutiae_csv"]),
            "b_mask_png": str(result_b["mask_png"]),
            "b_crop_dir": str(result_b["crop_dir"]),
            "b_crop_coarse_mask_png": str(result_b["coarse_mask_path"]),
            "b_crop_cropped_png": str(result_b["cropped_path"]),
            "b_crop_distal_mask_png": str(result_b["distal_mask_path"]),
            "b_crop_cropped_distal_png": str(result_b["cropped_distal_path"]),
            "b_crop_bbox_json": str(result_b["crop_bbox_path"]),
            "b_preprocess_dir": str(result_b["preprocess_dir"]),
        },
        "crop_stage": {
            "a": {
                "crop_bbox_xyxy": list(result_a["crop_bbox"]),
                "crop_mode": result_a["crop_mode"],
                "fallback_reason": result_a["fallback_reason"],
                "original_shape_hw": result_a["original_shape_hw"],
                "cropped_shape_hw": result_a["cropped_shape_hw"],
                "coarse_mask_pixels": result_a["coarse_mask_pixels"],
                "distal_mask_pixels": result_a["distal_mask_pixels"],
                "distal_to_coarse_ratio": float(result_a["distal_mask_pixels"] / max(result_a["coarse_mask_pixels"], 1)),
                "inference_input_shape_hw": result_a["inference_input_shape_hw"],
            },
            "b": {
                "crop_bbox_xyxy": list(result_b["crop_bbox"]),
                "crop_mode": result_b["crop_mode"],
                "fallback_reason": result_b["fallback_reason"],
                "original_shape_hw": result_b["original_shape_hw"],
                "cropped_shape_hw": result_b["cropped_shape_hw"],
                "coarse_mask_pixels": result_b["coarse_mask_pixels"],
                "distal_mask_pixels": result_b["distal_mask_pixels"],
                "distal_to_coarse_ratio": float(result_b["distal_mask_pixels"] / max(result_b["coarse_mask_pixels"], 1)),
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
    print(f"Saved A crop image: {result_a['cropped_path']}")
    print(f"Saved A distal mask: {result_a['distal_mask_path']}")
    print(f"Saved B minutiae CSV: {result_b['minutiae_csv']}")
    print(f"Saved B mask: {result_b['mask_png']}")
    print(f"Saved B crop image: {result_b['cropped_path']}")
    print(f"Saved B distal mask: {result_b['distal_mask_path']}")


if __name__ == "__main__":
    main()

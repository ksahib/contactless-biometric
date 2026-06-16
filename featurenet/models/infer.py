from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import pathlib
import sys
import sysconfig
from pathlib import Path
from typing import Any


def ensure_stdlib_copy_module() -> None:
    stdlib_copy = Path(sysconfig.get_paths()["stdlib"]) / "copy.py"
    spec = importlib.util.spec_from_file_location("copy", stdlib_copy)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not resolve stdlib copy module from {stdlib_copy}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = module
    spec.loader.exec_module(module)


ensure_stdlib_copy_module()

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import center_unwarping
import preprocess as solov2_preprocess
from generate_ground_truth import load_bgr_image

from .feature_extractor import FeatureExtractor


HEAD_KEYS = (
    "orientation",
    "ridge_period",
    "gradient",
    "minutia_orientation",
    "minutia_score",
    "minutia_x",
    "minutia_y",
)


def _resolve_device(device_arg: str) -> torch.device:
    normalized = device_arg.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda was requested but no CUDA device is available")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported --device value: {device_arg}")
    return torch.device(normalized)


def _torch_load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except NotImplementedError as exc:
        # Some checkpoints include pickled PosixPath values from Linux training runs.
        if "PosixPath" not in str(exc):
            raise
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc,assignment]
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint payload must be a dict, got {type(checkpoint).__name__}")
    return checkpoint


def load_checkpoint_model(weights_path: Path, device: torch.device) -> FeatureExtractor:
    checkpoint = _torch_load_checkpoint(weights_path, device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"missing model_state_dict in checkpoint: {weights_path}")

    model = FeatureExtractor().to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            "checkpoint is incompatible with the current FeatureExtractor architecture. "
            "This branch expects minutia_orientation to output 2 channels (cos/sin) and minutia_x/minutia_y "
            "to output 1 channel each (continuous offsets). "
            "Start a new training run with the updated model."
        ) from exc
    model.eval()
    return model


def preprocess_input_bgr(
    full_bgr: np.ndarray,
    save_preprocess_dir: Path | None = None,
    *,
    solov2_config: Path | None = None,
    solov2_checkpoint: Path | None = None,
    solov2_device: str | None = None,
    solov2_score_thr: float = 0.15,
    fallback_mask: np.ndarray | None = None,
    fallback_mask_source: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    raw_gray = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2GRAY)
    if fallback_mask is None:
        result = solov2_preprocess.run_preprocess_pipeline(
            full_bgr,
            score_thr=solov2_score_thr,
            device=solov2_device,
            model_config=solov2_config,
            checkpoint=solov2_checkpoint,
            target_period=10.0,
        )
        mask_source = "solov2"
    else:
        result = solov2_preprocess.run_preprocess_pipeline_from_mask(
            full_bgr,
            fallback_mask,
            target_period=10.0,
        )
        mask_source = fallback_mask_source or "fallback_mask"
    gray_image = result.rotated_image
    mask = result.rotated_mask
    masked_image = gray_image.copy()
    masked_image[mask <= 0] = 0

    image_float = masked_image.astype(np.float32)
    if image_float.max(initial=0.0) > 1.0:
        image_float /= 255.0
    mask_binary = (mask > 0).astype(np.float32)

    image_tensor = torch.from_numpy(image_float).unsqueeze(0).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).unsqueeze(0)

    if save_preprocess_dir is not None:
        save_preprocess_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_preprocess_dir / "raw_gray.png"), raw_gray)
        cv2.imwrite(str(save_preprocess_dir / "enhanced.png"), result.enhanced)
        cv2.imwrite(str(save_preprocess_dir / "full_mask.png"), result.full_mask)
        cv2.imwrite(str(save_preprocess_dir / "center_mask.png"), result.center_mask)
        cv2.imwrite(str(save_preprocess_dir / "scaled_image.png"), result.scaled_image)
        cv2.imwrite(str(save_preprocess_dir / "scaled_mask.png"), result.scaled_mask)
        cv2.imwrite(str(save_preprocess_dir / "preprocessed_gray.png"), gray_image)
        cv2.imwrite(str(save_preprocess_dir / "final_mask.png"), mask)
        cv2.imwrite(str(save_preprocess_dir / "masked_image.png"), masked_image)
        meta = {
            "canonical_preprocess": "preprocess.py",
            "ridge_period": float(result.ridge_period),
            "scale": float(result.scale),
            "yaw_angle": float(result.yaw_angle),
            "solov2_score_thr": float(solov2_score_thr),
            "solov2_device": solov2_device,
            "solov2_config": str(solov2_config) if solov2_config is not None else None,
            "solov2_checkpoint": str(solov2_checkpoint) if solov2_checkpoint is not None else None,
            "mask_source": mask_source,
        }
        (save_preprocess_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return image_tensor, mask_tensor, (int(masked_image.shape[0]), int(masked_image.shape[1]))


def preprocess_saved_masked_input(
    masked_image_path: Path,
    mask_path: Path,
    save_preprocess_dir: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    masked_image = cv2.imread(str(masked_image_path), cv2.IMREAD_GRAYSCALE)
    if masked_image is None:
        raise FileNotFoundError(f"unable to load masked image from {masked_image_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"unable to load mask from {mask_path}")
    if masked_image.shape != mask.shape:
        raise ValueError(
            "masked image and mask must have the same shape, "
            f"got {masked_image.shape} and {mask.shape}"
        )

    image_float = masked_image.astype(np.float32)
    if image_float.max(initial=0.0) > 1.0:
        image_float /= 255.0
    mask_binary = (mask > 0).astype(np.float32)

    image_tensor = torch.from_numpy(image_float).unsqueeze(0).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).unsqueeze(0)

    if save_preprocess_dir is not None:
        save_preprocess_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_preprocess_dir / "masked_image.png"), masked_image)
        cv2.imwrite(str(save_preprocess_dir / "mask.png"), mask)
        meta = {
            "canonical_preprocess": "generated_ground_truth_bundle",
            "masked_image_path": str(masked_image_path),
            "mask_path": str(mask_path),
        }
        (save_preprocess_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return image_tensor, mask_tensor, (int(masked_image.shape[0]), int(masked_image.shape[1]))


def preprocess_input_image(
    image_path: Path,
    save_preprocess_dir: Path | None = None,
    *,
    solov2_config: Path | None = None,
    solov2_checkpoint: Path | None = None,
    solov2_device: str | None = None,
    solov2_score_thr: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    full_bgr = load_bgr_image(image_path)
    return preprocess_input_bgr(
        full_bgr,
        save_preprocess_dir=save_preprocess_dir,
        solov2_config=solov2_config,
        solov2_checkpoint=solov2_checkpoint,
        solov2_device=solov2_device,
        solov2_score_thr=solov2_score_thr,
    )


@torch.no_grad()
def run_inference(
    model: FeatureExtractor,
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    image_tensor = image_tensor.to(device, non_blocking=True)
    mask_tensor = mask_tensor.to(device, non_blocking=True)
    outputs = model(image_tensor, mask=mask_tensor)
    missing = [key for key in HEAD_KEYS if key not in outputs]
    if missing:
        raise KeyError(f"model outputs missing expected heads: {missing}")
    return {key: outputs[key] for key in HEAD_KEYS}


def serialize_outputs(outputs: dict[str, torch.Tensor], output_npz: Path) -> None:
    arrays = {key: value.detach().cpu().numpy() for key, value in outputs.items()}
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)


def decode_orientation_field_radians(outputs: dict[str, torch.Tensor]) -> np.ndarray:
    logits = outputs["orientation"].detach().float()
    if logits.dim() != 4 or logits.shape[0] != 1:
        raise ValueError(f"expected orientation logits with shape [1,180,H,W], got {tuple(logits.shape)}")
    probabilities = torch.softmax(logits[0], dim=0)
    bins = torch.arange(
        probabilities.shape[0],
        device=probabilities.device,
        dtype=probabilities.dtype,
    ) * (torch.pi / float(probabilities.shape[0]))
    cos2 = (probabilities * torch.cos(2.0 * bins).view(-1, 1, 1)).sum(dim=0)
    sin2 = (probabilities * torch.sin(2.0 * bins).view(-1, 1, 1)).sum(dim=0)
    orientation = 0.5 * torch.atan2(sin2, cos2)
    orientation = torch.remainder(orientation, torch.pi)
    return orientation.cpu().numpy().astype(np.float32)


def decode_ridge_period_array(outputs: dict[str, torch.Tensor]) -> np.ndarray:
    ridge_period = outputs["ridge_period"].detach().float()
    if ridge_period.dim() == 4 and ridge_period.shape[0] == 1 and ridge_period.shape[1] == 1:
        ridge_period = ridge_period[0, 0]
    elif ridge_period.dim() == 3 and ridge_period.shape[0] == 1:
        ridge_period = ridge_period[0]
    elif ridge_period.dim() != 2:
        raise ValueError(f"expected ridge_period with shape [1,1,H,W], got {tuple(ridge_period.shape)}")
    return ridge_period.cpu().numpy().astype(np.float32)


def save_pose_sidecars(
    outputs: dict[str, torch.Tensor],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    orientation_path = output_dir / "orientation.npy"
    ridge_period_path = output_dir / "ridge_period.npy"
    np.save(orientation_path, decode_orientation_field_radians(outputs))
    np.save(ridge_period_path, decode_ridge_period_array(outputs))
    return orientation_path, ridge_period_path


def _orientation_vectors_to_radians(vectors: torch.Tensor) -> torch.Tensor:
    if vectors.dim() != 4 or vectors.shape[1] != 2:
        raise ValueError(f"expected minutia_orientation with shape [B,2,H,W], got {tuple(vectors.shape)}")
    unit = F.normalize(vectors.float(), dim=1, eps=1e-8)
    theta = torch.atan2(unit[:, 1], unit[:, 0])
    return theta


def decode_minutiae_rows(
    outputs: dict[str, torch.Tensor],
    input_shape_hw: tuple[int, int],
    score_threshold: float = 0.5,
    apply_nms: bool = True,
) -> list[dict[str, float]]:
    score = torch.sigmoid(outputs["minutia_score"].detach().float())
    x_offsets = torch.sigmoid(outputs["minutia_x"].detach().float())
    y_offsets = torch.sigmoid(outputs["minutia_y"].detach().float())
    ori_radians = _orientation_vectors_to_radians(outputs["minutia_orientation"].detach())

    if score.shape[0] != 1:
        raise ValueError(f"single-image inference expected batch size 1, got {score.shape[0]}")

    score_map = score[0, 0]
    x_map = x_offsets[0, 0]
    y_map = y_offsets[0, 0]
    angle_map = ori_radians[0]

    active = score_map >= float(score_threshold)
    if apply_nms:
        pooled = F.max_pool2d(score_map.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0)
        active = active & (score_map >= (pooled - 1e-8))

    indices = torch.nonzero(active, as_tuple=False)
    if indices.numel() == 0:
        return []

    out_h, out_w = int(score_map.shape[0]), int(score_map.shape[1])
    in_h, in_w = int(input_shape_hw[0]), int(input_shape_hw[1])
    scale_x = float(in_w) / max(float(out_w), 1.0)
    scale_y = float(in_h) / max(float(out_h), 1.0)

    rows: list[dict[str, float]] = []
    for row_col in indices:
        row = int(row_col[0].item())
        col = int(row_col[1].item())
        x = (float(col) + float(x_map[row, col].item())) * scale_x
        y = (float(row) + float(y_map[row, col].item())) * scale_y
        angle = float(angle_map[row, col].item())
        point_score = float(score_map[row, col].item())
        rows.append(
            {
                "x": x,
                "y": y,
                "angle": angle,
                "score": point_score,
            }
        )

    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows


def save_minutiae_csv(rows: list[dict[str, float]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y", "angle", "score"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _upsample_gradient_to_full(gradient_tensor: torch.Tensor, input_shape_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    grad = gradient_tensor.detach().float().cpu().numpy()
    if grad.ndim != 4 or grad.shape[0] != 1 or grad.shape[1] != 2:
        raise ValueError(f"expected gradient with shape [1,2,H,W], got {tuple(grad.shape)}")
    height, width = int(input_shape_hw[0]), int(input_shape_hw[1])
    gx = cv2.resize(grad[0, 0], (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    gy = cv2.resize(grad[0, 1], (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return gx, gy


def _sample_map_nearest(
    map_array: np.ndarray,
    valid_mask: np.ndarray,
    x: float,
    y: float,
) -> float | None:
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    height, width = map_array.shape
    xi = int(round(x))
    yi = int(round(y))
    if xi < 0 or yi < 0 or xi >= width or yi >= height:
        return None
    if not bool(valid_mask[yi, xi]):
        return None
    value = float(map_array[yi, xi])
    return value if math.isfinite(value) else None


def _warp_point(x_out: np.ndarray, y_out: np.ndarray, valid: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
    xu = _sample_map_nearest(x_out, valid, x, y)
    yu = _sample_map_nearest(y_out, valid, x, y)
    if xu is None or yu is None:
        return None
    return xu, yu


def unwarp_minutiae_rows(
    rows: list[dict[str, float]],
    gradient_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    input_shape_hw: tuple[int, int],
    gray_image: np.ndarray | None = None,
    orient_delta_px: float = 4.0,
) -> tuple[list[dict[str, float]], np.ndarray]:
    """Route A: warp decoded minutiae into the predicted-gradient unwarped frame.

    Uses the model's predicted shape (gradient) head to flatten the finger via
    ``center_unwarping.run_center_unwarping``, then moves each minutia's coordinate
    and orientation into the canonical (unwarped) frame. Extraction stays in the
    contactless view; only the geometry is transported, mirroring how the ground
    truth reprojects minutiae. Returns the warped rows and the unwarped mask that
    matches the new coordinate frame (for MCC overlap gating).
    """
    height, width = int(input_shape_hw[0]), int(input_shape_hw[1])
    mask = mask_tensor.detach().float().cpu().numpy()
    while mask.ndim > 2:
        mask = mask[0]
    mask_u8 = (mask > 0).astype(np.uint8)

    gx, gy = _upsample_gradient_to_full(gradient_tensor, input_shape_hw)
    gx = gx * mask_u8
    gy = gy * mask_u8

    if gray_image is not None:
        image = gray_image.astype(np.float32)
        if image.shape != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    else:
        image = np.zeros((height, width), dtype=np.float32)

    if int(mask_u8.sum()) == 0:
        return rows, mask_u8
    try:
        maps = center_unwarping.run_center_unwarping(image, mask_u8, gx, gy)
    except (ValueError, FloatingPointError):
        return rows, mask_u8

    x_out = np.asarray(maps["x_out"], dtype=np.float32)
    y_out = np.asarray(maps["y_out"], dtype=np.float32)
    valid = np.asarray(maps["valid_mask"], dtype=bool)
    unwarped_mask = np.asarray(maps["unwarped_mask"], dtype=np.uint8)

    warped_rows: list[dict[str, float]] = []
    for row in rows:
        x = float(row["x"])
        y = float(row["y"])
        angle = float(row.get("angle", 0.0))
        center = _warp_point(x_out, y_out, valid, x, y)
        if center is None:
            continue
        cxu, cyu = center
        forward = _warp_point(
            x_out, y_out, valid,
            x + math.cos(angle) * orient_delta_px,
            y + math.sin(angle) * orient_delta_px,
        )
        backward = _warp_point(
            x_out, y_out, valid,
            x - math.cos(angle) * orient_delta_px,
            y - math.sin(angle) * orient_delta_px,
        )
        if forward is not None and backward is not None:
            new_angle = math.atan2(forward[1] - backward[1], forward[0] - backward[0])
        elif forward is not None:
            new_angle = math.atan2(forward[1] - cyu, forward[0] - cxu)
        elif backward is not None:
            new_angle = math.atan2(cyu - backward[1], cxu - backward[0])
        else:
            new_angle = angle
        warped_rows.append(
            {
                "x": float(cxu),
                "y": float(cyu),
                "angle": float(new_angle),
                "score": float(row.get("score", 1.0)),
            }
        )
    return warped_rows, unwarped_mask


def print_output_stats(outputs: dict[str, torch.Tensor]) -> None:
    print("FeatureNet raw logit stats:")
    for key in HEAD_KEYS:
        tensor = outputs[key].detach().float().cpu()
        finite_mask = torch.isfinite(tensor)
        finite_count = int(finite_mask.sum().item())
        total_count = int(tensor.numel())

        if finite_count == 0:
            min_value = float("nan")
            max_value = float("nan")
            mean_value = float("nan")
        else:
            finite_values = tensor[finite_mask]
            min_value = float(finite_values.min().item())
            max_value = float(finite_values.max().item())
            mean_value = float(finite_values.mean().item())

        print(
            f"- {key}: shape={list(tensor.shape)} "
            f"min={min_value:.6f} max={max_value:.6f} mean={mean_value:.6f} "
            f"finite={finite_count}/{total_count}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image FeatureNet inference and save raw head logits.")
    parser.add_argument("--image-path", type=Path, required=True, help="Path to one input image.")
    parser.add_argument("--weights-path", type=Path, default=Path("weights") / "best.pt", help="Path to checkpoint.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help="Path to save raw logits .npz (default: <image_stem>_logits.npz in current directory).",
    )
    parser.add_argument(
        "--save-preprocess-dir",
        type=Path,
        default=None,
        help="Optional directory to save intermediate preprocessing images.",
    )
    parser.add_argument("--solov2-config", type=Path, default=None, help="SOLOv2 MMDetection config path.")
    parser.add_argument("--solov2-checkpoint", type=Path, default=None, help="SOLOv2 checkpoint path.")
    parser.add_argument("--solov2-device", default=None, help="SOLOv2 inference device, e.g. cpu or cuda:0.")
    parser.add_argument("--solov2-score-thr", type=float, default=0.15, help="Minimum SOLOv2 distal phalanx detection score.")
    parser.add_argument(
        "--output-minutiae-csv",
        type=Path,
        default=None,
        help="Path to save decoded minutiae CSV for MCC matching (default: <image_stem>_minutiae.csv in current directory).",
    )
    parser.add_argument(
        "--minutia-score-threshold",
        type=float,
        default=0.5,
        help="Score threshold (after sigmoid) for minutia point decoding.",
    )
    parser.add_argument(
        "--disable-minutia-nms",
        action="store_true",
        help="Disable 3x3 local-maximum suppression during minutia decoding.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image_path.resolve()
    weights_path = args.weights_path.resolve()
    output_npz = args.output_npz.resolve() if args.output_npz is not None else (Path.cwd() / f"{image_path.stem}_logits.npz")
    output_minutiae_csv = (
        args.output_minutiae_csv.resolve()
        if args.output_minutiae_csv is not None
        else (Path.cwd() / f"{image_path.stem}_minutiae.csv")
    )

    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"weights file not found: {weights_path}")

    device = _resolve_device(args.device)
    model = load_checkpoint_model(weights_path, device)
    image_tensor, mask_tensor, input_shape_hw = preprocess_input_image(
        image_path,
        save_preprocess_dir=args.save_preprocess_dir,
        solov2_config=args.solov2_config,
        solov2_checkpoint=args.solov2_checkpoint,
        solov2_device=args.solov2_device,
        solov2_score_thr=float(args.solov2_score_thr),
    )
    outputs = run_inference(model, image_tensor, mask_tensor, device)
    print_output_stats(outputs)
    serialize_outputs(outputs, output_npz)
    minutiae_rows = decode_minutiae_rows(
        outputs=outputs,
        input_shape_hw=input_shape_hw,
        score_threshold=float(args.minutia_score_threshold),
        apply_nms=not bool(args.disable_minutia_nms),
    )
    save_minutiae_csv(minutiae_rows, output_minutiae_csv)
    orientation_path, ridge_period_path = save_pose_sidecars(outputs, output_minutiae_csv.parent)
    print(f"Saved raw logits NPZ: {output_npz}")
    print(f"Saved decoded minutiae CSV: {output_minutiae_csv} (rows={len(minutiae_rows)})")
    print(f"Saved orientation sidecar: {orientation_path}")
    print(f"Saved ridge-period sidecar: {ridge_period_path}")


if __name__ == "__main__":
    main()

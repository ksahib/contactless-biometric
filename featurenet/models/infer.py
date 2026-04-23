from __future__ import annotations

import argparse
import csv
import importlib.util
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

from generate_ground_truth import (
    _normalize_ridge_frequency,
    load_bgr_image,
    normalise_brightness_array,
    rembg_mask_from_bgr,
    validate_foreground_area,
)

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
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    raw_gray = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2GRAY)
    initial_mask, _mask_source = rembg_mask_from_bgr(full_bgr)
    validate_foreground_area(initial_mask, minimum_ratio=0.03)
    normalized_gray = normalise_brightness_array(raw_gray, initial_mask)
    preprocessed_gray, final_mask, _ridge_scale_factor = _normalize_ridge_frequency(
        normalized_gray,
        initial_mask,
        target_spacing=10.0,
    )

    gray_image = preprocessed_gray
    mask = final_mask
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
        cv2.imwrite(str(save_preprocess_dir / "normalized_gray.png"), normalized_gray)
        cv2.imwrite(str(save_preprocess_dir / "initial_mask.png"), initial_mask)
        cv2.imwrite(str(save_preprocess_dir / "preprocessed_gray.png"), preprocessed_gray)
        cv2.imwrite(str(save_preprocess_dir / "final_mask.png"), final_mask)
        cv2.imwrite(str(save_preprocess_dir / "masked_image.png"), masked_image)

    return image_tensor, mask_tensor, (int(masked_image.shape[0]), int(masked_image.shape[1]))


def preprocess_input_image(
    image_path: Path,
    save_preprocess_dir: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    full_bgr = load_bgr_image(image_path)
    return preprocess_input_bgr(full_bgr, save_preprocess_dir=save_preprocess_dir)


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
    print(f"Saved raw logits NPZ: {output_npz}")
    print(f"Saved decoded minutiae CSV: {output_minutiae_csv} (rows={len(minutiae_rows)})")


if __name__ == "__main__":
    main()

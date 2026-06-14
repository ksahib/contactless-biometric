#!/usr/bin/env python
"""Detect and repair upside-down generated FeatureNet ground-truth bundles.

The repair is intentionally limited to sample bundle artifacts. Raw images and
pre-rotation images are left untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


POST_POSE_IMAGE_FILES = (
    "masked_image.png",
    "mask.png",
    "preprocessed_input.png",
    "preprocess_mask.png",
    "preprocess_pose_normalized.png",
    "preprocess_pose_mask.png",
    "minutiae_enhanced.png",
)
SPATIAL_NPY_FILES = ("orientation.npy", "ridge_period.npy")
VECTOR_NPY_FILES = ("gradient_visualization.npy",)
SCALAR_NPZ_KEYS = {
    "orientation",
    "ridge_period",
    "output_mask",
    "minutia_score",
    "minutia_score_weight_map",
    "minutia_score_ignore_mask",
    "minutia_score_center_map",
    "minutia_valid_mask",
}
LABEL_NPZ_KEYS = {
    "minutia_x",
    "minutia_y",
    "minutia_x_offset",
    "minutia_y_offset",
    "minutia_orientation",
    "minutia_orientation_vec",
}


@dataclass(slots=True)
class WidthStats:
    top_width: float
    bottom_width: float
    margin: float
    should_flip: bool


@dataclass(slots=True)
class PatchSummary:
    scanned: int = 0
    detected_upside_down: int = 0
    repaired: int = 0
    skipped_already_patched: int = 0
    skipped_missing_required: int = 0
    skipped_filter: int = 0
    errors: int = 0
    repaired_samples: list[str] = field(default_factory=list)
    upside_down_samples: list[str] = field(default_factory=list)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _as_uint8_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"expected 2D mask, got {mask.shape}")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(_as_uint8_mask(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("mask has no foreground component")
    largest = max(contours, key=cv2.contourArea)
    refined = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(refined, [largest], -1, 255, thickness=cv2.FILLED)
    return refined


def _foreground_band_width(mask: np.ndarray, rows: np.ndarray) -> float:
    widths: list[float] = []
    for row in rows:
        cols = np.flatnonzero(mask[int(row)] > 0)
        if cols.size > 0:
            widths.append(float(cols[-1] - cols[0] + 1))
    if not widths:
        return 0.0
    return float(np.median(np.asarray(widths, dtype=np.float32)))


def upside_down_width_stats(mask: np.ndarray) -> WidthStats:
    refined = _largest_component_mask(mask)
    ys, _ = np.where(refined > 0)
    if ys.size < 2:
        return WidthStats(top_width=0.0, bottom_width=0.0, margin=0.0, should_flip=False)

    y_min = int(ys.min())
    y_max = int(ys.max())
    height = y_max - y_min + 1
    if height < 8:
        return WidthStats(top_width=0.0, bottom_width=0.0, margin=0.0, should_flip=False)

    band_height = max(2, int(round(0.2 * height)))
    top_rows = np.arange(y_min, min(y_min + band_height, y_max + 1), dtype=np.int32)
    bottom_rows = np.arange(max(y_min, y_max - band_height + 1), y_max + 1, dtype=np.int32)
    top_width = _foreground_band_width(refined, top_rows)
    bottom_width = _foreground_band_width(refined, bottom_rows)
    margin = max(4.0, 0.08 * max(top_width, bottom_width))
    should_flip = top_width > 0.0 and bottom_width > 0.0 and (top_width - bottom_width) > margin
    return WidthStats(
        top_width=float(top_width),
        bottom_width=float(bottom_width),
        margin=float(margin),
        should_flip=bool(should_flip),
    )


def _flip_180(array: np.ndarray) -> np.ndarray:
    if array.ndim < 2:
        return array.copy()
    return np.ascontiguousarray(array[..., ::-1, ::-1])


def _flip_vector_180(array: np.ndarray) -> np.ndarray:
    flipped = _flip_180(array).astype(array.dtype, copy=False)
    if flipped.ndim == 3 and flipped.shape[0] == 2:
        return (-flipped).astype(array.dtype, copy=False)
    if flipped.ndim == 3 and flipped.shape[-1] == 2:
        return (-flipped).astype(array.dtype, copy=False)
    raise ValueError(f"expected vector map shape (2,H,W) or (H,W,2), got {array.shape}")


def _normalize_angle_2pi(angle: float) -> float:
    wrapped = float(angle) % (2.0 * math.pi)
    return wrapped if math.isfinite(wrapped) else 0.0


def _transform_minutia_record(record: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    transformed = dict(record)
    if "x" in transformed:
        transformed["x"] = float(width - 1) - float(transformed["x"])
    if "y" in transformed:
        transformed["y"] = float(height - 1) - float(transformed["y"])
    if "theta" in transformed:
        transformed["theta"] = _normalize_angle_2pi(float(transformed["theta"]) + math.pi)
    if "angle" in transformed:
        transformed["angle"] = _normalize_angle_2pi(float(transformed["angle"]) + math.pi)
    if "direction" in transformed:
        transformed["direction"] = _normalize_angle_2pi(float(transformed["direction"]) + math.pi)
    return transformed


def transform_minutiae_records(records: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    return [_transform_minutia_record(record, width=width, height=height) for record in records]


def _fallback_flip_local_labels(arrays: dict[str, np.ndarray]) -> None:
    active = arrays.get("minutia_valid_mask", np.zeros((1, 1, 1), dtype=np.float32)) > 0.5
    active_2d = active[0] if active.ndim == 3 else active
    if "minutia_x" in arrays:
        arrays["minutia_x"] = np.where(active_2d, 7 - arrays["minutia_x"], 0).astype(arrays["minutia_x"].dtype)
    if "minutia_y" in arrays:
        arrays["minutia_y"] = np.where(active_2d, 7 - arrays["minutia_y"], 0).astype(arrays["minutia_y"].dtype)
    if "minutia_x_offset" in arrays:
        arrays["minutia_x_offset"] = np.where(active, np.clip(1.0 - arrays["minutia_x_offset"], 0.0, 1.0 - 1e-6), 0.0).astype(np.float32)
    if "minutia_y_offset" in arrays:
        arrays["minutia_y_offset"] = np.where(active, np.clip(1.0 - arrays["minutia_y_offset"], 0.0, 1.0 - 1e-6), 0.0).astype(np.float32)


def _rasterize_flipped_minutia_labels(
    arrays: dict[str, np.ndarray],
    minutiae: list[dict[str, Any]],
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> None:
    target_height, target_width = target_shape
    source_height, source_width = source_shape
    output_mask = arrays.get("output_mask")
    if output_mask is None:
        allowed = np.ones((target_height, target_width), dtype=np.float32)
    else:
        allowed = output_mask[0] if output_mask.ndim == 3 else output_mask
        allowed = (allowed > 0).astype(np.float32)

    valid_mask = np.zeros((target_height, target_width), dtype=np.float32)
    minutia_x = np.zeros((target_height, target_width), dtype=np.int64)
    minutia_y = np.zeros((target_height, target_width), dtype=np.int64)
    minutia_x_offset = np.zeros((target_height, target_width), dtype=np.float32)
    minutia_y_offset = np.zeros((target_height, target_width), dtype=np.float32)
    minutia_orientation = np.zeros((target_height, target_width), dtype=np.int64)
    minutia_orientation_vec = np.zeros((2, target_height, target_width), dtype=np.float32)
    ownership = np.full((target_height, target_width), -1.0, dtype=np.float32)
    center_distance = np.full((target_height, target_width), np.inf, dtype=np.float32)
    cell_width = float(source_width) / float(max(target_width, 1))
    cell_height = float(source_height) / float(max(target_height, 1))

    for minutia in minutiae:
        try:
            x = float(minutia["x"])
            y = float(minutia["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        cell_x = int(np.clip(math.floor(x * target_width / max(source_width, 1)), 0, target_width - 1))
        cell_y = int(np.clip(math.floor(y * target_height / max(source_height, 1)), 0, target_height - 1))
        if allowed[cell_y, cell_x] <= 0:
            continue
        score = minutia.get("score")
        score_value = float(score) if score is not None and math.isfinite(float(score)) else 1.0
        score_value = float(np.clip(score_value, 0.0, 1.0))
        cell_origin_x = cell_x * cell_width
        cell_origin_y = cell_y * cell_height
        local_x = float(np.clip((x - cell_origin_x) / max(cell_width, 1e-6), 0.0, 1.0 - 1e-6))
        local_y = float(np.clip((y - cell_origin_y) / max(cell_height, 1e-6), 0.0, 1.0 - 1e-6))
        distance = float((local_x - 0.5) ** 2 + (local_y - 0.5) ** 2)
        if score_value < ownership[cell_y, cell_x]:
            continue
        if abs(score_value - ownership[cell_y, cell_x]) <= 1e-6 and distance >= center_distance[cell_y, cell_x]:
            continue

        ownership[cell_y, cell_x] = score_value
        center_distance[cell_y, cell_x] = distance
        valid_mask[cell_y, cell_x] = 1.0
        theta = _normalize_angle_2pi(float(minutia.get("theta", 0.0)))
        minutia_x[cell_y, cell_x] = int(np.clip(math.floor(local_x * 8.0), 0, 7))
        minutia_y[cell_y, cell_x] = int(np.clip(math.floor(local_y * 8.0), 0, 7))
        minutia_x_offset[cell_y, cell_x] = local_x
        minutia_y_offset[cell_y, cell_x] = local_y
        minutia_orientation[cell_y, cell_x] = int(np.clip(math.floor(theta * (360.0 / (2.0 * math.pi))), 0, 359))
        minutia_orientation_vec[0, cell_y, cell_x] = float(math.cos(theta))
        minutia_orientation_vec[1, cell_y, cell_x] = float(math.sin(theta))

    arrays["minutia_valid_mask"] = (valid_mask * allowed)[np.newaxis, ...].astype(np.float32)
    arrays["minutia_x"] = minutia_x.astype(np.int64)
    arrays["minutia_y"] = minutia_y.astype(np.int64)
    arrays["minutia_x_offset"] = minutia_x_offset[np.newaxis, ...].astype(np.float32)
    arrays["minutia_y_offset"] = minutia_y_offset[np.newaxis, ...].astype(np.float32)
    arrays["minutia_orientation"] = minutia_orientation.astype(np.int64)
    arrays["minutia_orientation_vec"] = minutia_orientation_vec.astype(np.float32)


def transform_featurenet_targets(
    arrays: dict[str, np.ndarray],
    *,
    minutiae: list[dict[str, Any]] | None,
    source_shape: tuple[int, int] | None,
) -> dict[str, np.ndarray]:
    transformed: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if key == "gradient":
            transformed[key] = _flip_vector_180(value).astype(value.dtype, copy=False)
        elif key == "minutia_orientation_vec":
            transformed[key] = _flip_vector_180(value).astype(value.dtype, copy=False)
        elif key == "minutia_orientation":
            flipped = _flip_180(value)
            active = _flip_180(arrays.get("minutia_valid_mask", np.zeros((1,) + value.shape, dtype=np.float32))) > 0.5
            active_2d = active[0] if active.ndim == 3 else active
            transformed[key] = np.where(active_2d, (flipped.astype(np.int64) + 180) % 360, 0).astype(value.dtype)
        elif key in SCALAR_NPZ_KEYS or key in LABEL_NPZ_KEYS:
            transformed[key] = _flip_180(value).astype(value.dtype, copy=False)
        elif value.ndim >= 2:
            transformed[key] = _flip_180(value).astype(value.dtype, copy=False)
        else:
            transformed[key] = value.copy()

    if minutiae and source_shape is not None and "output_mask" in transformed:
        mask = transformed["output_mask"]
        target_shape = tuple(int(v) for v in (mask.shape[-2], mask.shape[-1]))
        _rasterize_flipped_minutia_labels(
            transformed,
            minutiae,
            source_shape=source_shape,
            target_shape=target_shape,
        )
    else:
        _fallback_flip_local_labels(transformed)
    return transformed


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"unable to read image: {path}")
    return image


def _write_gray(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"unable to write image: {path}")


def _source_shape_from_meta(meta: dict[str, Any], fallback_image: np.ndarray) -> tuple[int, int]:
    shapes = meta.get("shapes")
    if isinstance(shapes, dict):
        input_shape = shapes.get("input_image")
        if isinstance(input_shape, list | tuple) and len(input_shape) >= 2:
            try:
                return int(input_shape[0]), int(input_shape[1])
            except (TypeError, ValueError):
                pass
    return int(fallback_image.shape[0]), int(fallback_image.shape[1])


def _repair_json_records(path: Path, width: int, height: int) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    records = _read_json(path)
    if not isinstance(records, list):
        return None
    transformed = transform_minutiae_records(records, width=width, height=height)
    _write_json(path, transformed)
    return transformed


def _already_patched(meta: dict[str, Any]) -> bool:
    patches = meta.get("patches")
    return isinstance(patches, dict) and isinstance(patches.get("upside_down_orientation_repair"), dict)


def _matches_filters(sample_dir: Path, meta: dict[str, Any], sample_ids: set[str] | None, raw_view_indices: set[int] | None) -> bool:
    sample_id = str(meta.get("sample_id") or sample_dir.name)
    if sample_ids is not None and sample_id not in sample_ids and sample_dir.name not in sample_ids:
        return False
    if raw_view_indices is not None:
        try:
            raw_view_index = int(meta.get("raw_view_index", -1))
        except (TypeError, ValueError):
            return False
        if raw_view_index not in raw_view_indices:
            return False
    return True


def repair_sample(sample_dir: Path, *, apply: bool) -> dict[str, Any]:
    meta_path = sample_dir / "meta.json"
    mask_path = sample_dir / "mask.png"
    targets_path = sample_dir / "featurenet_targets.npz"
    if not meta_path.exists() or not mask_path.exists() or not targets_path.exists():
        return {"sample_id": sample_dir.name, "status": "missing_required"}

    meta = _read_json(meta_path)
    if _already_patched(meta):
        return {"sample_id": sample_dir.name, "status": "already_patched"}

    mask = _read_gray(mask_path)
    before = upside_down_width_stats(mask)
    if not before.should_flip:
        return {
            "sample_id": str(meta.get("sample_id") or sample_dir.name),
            "status": "ok",
            "top_width": before.top_width,
            "bottom_width": before.bottom_width,
        }

    result: dict[str, Any] = {
        "sample_id": str(meta.get("sample_id") or sample_dir.name),
        "status": "would_repair" if not apply else "repaired",
        "top_width": before.top_width,
        "bottom_width": before.bottom_width,
    }
    if not apply:
        return result

    flipped_artifacts: list[str] = []
    for name in POST_POSE_IMAGE_FILES:
        path = sample_dir / name
        if not path.exists():
            continue
        image = _read_gray(path)
        _write_gray(path, _flip_180(image).astype(image.dtype, copy=False))
        flipped_artifacts.append(name)

    for name in SPATIAL_NPY_FILES:
        path = sample_dir / name
        if not path.exists():
            continue
        array = np.load(path)
        np.save(path, _flip_180(array).astype(array.dtype, copy=False))
        flipped_artifacts.append(name)

    for name in VECTOR_NPY_FILES:
        path = sample_dir / name
        if not path.exists():
            continue
        array = np.load(path)
        np.save(path, _flip_vector_180(array))
        flipped_artifacts.append(name)

    source_height, source_width = _source_shape_from_meta(meta, mask)
    transformed_minutiae = _repair_json_records(sample_dir / "minutiae.json", width=source_width, height=source_height)
    if transformed_minutiae is not None:
        flipped_artifacts.append("minutiae.json")
    if (sample_dir / "minutiae_single_source_candidates.json").exists():
        _repair_json_records(sample_dir / "minutiae_single_source_candidates.json", width=source_width, height=source_height)
        flipped_artifacts.append("minutiae_single_source_candidates.json")

    targets = transform_featurenet_targets(
        _load_npz_arrays(targets_path),
        minutiae=transformed_minutiae,
        source_shape=(source_height, source_width),
    )
    _save_npz(targets_path, targets)
    flipped_artifacts.append("featurenet_targets.npz")

    repaired_mask = _read_gray(mask_path)
    after = upside_down_width_stats(repaired_mask)
    patches = meta.setdefault("patches", {})
    patches["upside_down_orientation_repair"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "180_degree_in_place_flip_after_top_bottom_width_detection",
        "before": {
            "top_width": before.top_width,
            "bottom_width": before.bottom_width,
            "margin": before.margin,
        },
        "after": {
            "top_width": after.top_width,
            "bottom_width": after.bottom_width,
            "margin": after.margin,
        },
        "flipped_artifacts": flipped_artifacts,
    }
    _write_json(meta_path, meta)
    result["after_top_width"] = after.top_width
    result["after_bottom_width"] = after.bottom_width
    result["flipped_artifacts"] = flipped_artifacts
    return result


def _parse_int_set(values: list[str] | None) -> set[int] | None:
    if values is None:
        return None
    tokens: list[str] = []
    for value in values:
        tokens.extend(part.strip() for part in str(value).split(","))
    tokens = [token for token in tokens if token]
    if not tokens:
        return None
    return {int(token) for token in tokens}


def _iter_sample_dirs(samples_root: Path, sample_ids: set[str] | None) -> list[Path]:
    if sample_ids is not None:
        return sorted(samples_root / sample_id for sample_id in sample_ids if (samples_root / sample_id).is_dir())
    return sorted(path for path in samples_root.iterdir() if path.is_dir())


def patch_ground_truth(
    ground_truth_root: Path,
    *,
    apply: bool,
    limit: int | None = None,
    sample_ids: set[str] | None = None,
    raw_view_indices: set[int] | None = None,
) -> dict[str, Any]:
    samples_root = ground_truth_root / "samples"
    if not samples_root.exists():
        raise FileNotFoundError(f"missing samples directory: {samples_root}")
    summary = PatchSummary()
    details: list[dict[str, Any]] = []
    sample_dirs = _iter_sample_dirs(samples_root, sample_ids)
    if limit is not None:
        sample_dirs = sample_dirs[: int(limit)]

    for sample_dir in sample_dirs:
        summary.scanned += 1
        try:
            meta_path = sample_dir / "meta.json"
            meta = _read_json(meta_path) if meta_path.exists() else {}
            if not _matches_filters(sample_dir, meta, sample_ids, raw_view_indices):
                summary.skipped_filter += 1
                continue
            result = repair_sample(sample_dir, apply=apply)
            details.append(result)
            status = result.get("status")
            if status in {"would_repair", "repaired"}:
                summary.detected_upside_down += 1
                summary.upside_down_samples.append(str(result.get("sample_id", sample_dir.name)))
            if status == "repaired":
                summary.repaired += 1
                summary.repaired_samples.append(str(result.get("sample_id", sample_dir.name)))
            elif status == "already_patched":
                summary.skipped_already_patched += 1
            elif status == "missing_required":
                summary.skipped_missing_required += 1
        except Exception as exc:
            summary.errors += 1
            details.append({"sample_id": sample_dir.name, "status": "error", "error": str(exc)})

    return {
        "ground_truth_root": str(ground_truth_root),
        "apply": bool(apply),
        "summary": {
            "scanned": summary.scanned,
            "detected_upside_down": summary.detected_upside_down,
            "repaired": summary.repaired,
            "skipped_already_patched": summary.skipped_already_patched,
            "skipped_missing_required": summary.skipped_missing_required,
            "skipped_filter": summary.skipped_filter,
            "errors": summary.errors,
            "upside_down_samples": summary.upside_down_samples[:50],
            "repaired_samples": summary.repaired_samples[:50],
        },
        "details": details[:200],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch upside-down generated ground-truth sample bundles in place.")
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Rewrite affected sample bundles in place. Omit for dry-run audit.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-id", action="append", default=None, help="Sample id to include. Can be passed multiple times.")
    parser.add_argument("--raw-view-indices", nargs="+", default=None, help="Optional raw_view_index filter, e.g. 0 or '0 1 2'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    sample_ids = set(args.sample_id) if args.sample_id else None
    raw_view_indices = _parse_int_set(args.raw_view_indices)
    report = patch_ground_truth(
        args.ground_truth_root,
        apply=bool(args.apply),
        limit=args.limit,
        sample_ids=sample_ids,
        raw_view_indices=raw_view_indices,
    )
    print(json.dumps(report, indent=2), flush=True)
    if report["summary"]["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

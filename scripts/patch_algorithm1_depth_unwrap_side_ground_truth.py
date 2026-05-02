#!/usr/bin/env python
"""Patch side-view ground-truth minutiae with Algorithm-1 depth unwrap labels.

This is a surgical patcher for already generated FeatureNet ground-truth roots.
It rewrites only side-view minutiae labels and minutiae target arrays. Front
samples, dense orientation/ridge labels, gradients, masks, and images are left
unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

# Avoid repo-root copy.py shadowing stdlib copy while importing cv2/pyfing.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT]

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyfing  # noqa: E402

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import generate_ground_truth as gt  # noqa: E402


SIDE_ROLES = {1: "left", 2: "right"}
DEFAULT_DPI = 500
DEFAULT_ROW_PARAM_SMOOTH_WINDOW = 31
DEFAULT_MAP_SMOOTH_SIGMA_X = 2.0
DEFAULT_MAP_SMOOTH_SIGMA_Y = 5.0
DEFAULT_ORIENTATION_DELTA_PX = 4.0
MIN_ORIENTATION_BASELINE_PX = 1.5
MINUTIA_TARGET_KEYS = {
    "minutia_score",
    "minutia_valid_mask",
    "minutia_x",
    "minutia_y",
    "minutia_x_offset",
    "minutia_y_offset",
    "minutia_orientation",
    "minutia_orientation_vec",
}


@dataclass(slots=True)
class PatchStats:
    scanned: int = 0
    patched: int = 0
    dry_run_would_patch: int = 0
    skipped_not_side: int = 0
    skipped_missing_files: int = 0
    skipped_zero_after_patch: int = 0
    errors: int = 0


@dataclass(slots=True)
class RoleCache:
    role: str
    unwrap_image_path: Path
    unwrap_mask_path: Path
    unwrap_maps_path: Path
    extracted_minutiae_path: Path
    reprojected_pose_minutiae_path: Path
    extraction_count: int
    pose_reprojected_count: int
    orientation_projected_count: int
    orientation_fallback_count: int
    dense_orientation_stats: dict[str, Any]


@dataclass(slots=True)
class AcquisitionCache:
    acquisition_id: str
    reconstruction_dir: Path
    cache_dir: Path
    unwrap_report_path: Path
    roles: dict[str, RoleCache] = field(default_factory=dict)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_depth_unwrap_module() -> Any:
    module_path = REPO_ROOT / "scripts" / "algorithm1_side_depth_unwrap_fixed_v4.py"
    spec = importlib.util.spec_from_file_location("algorithm1_side_depth_unwrap_fixed_v4", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_angle_2pi(theta: float) -> float:
    value = float(theta) % (2.0 * math.pi)
    return value if value >= 0.0 else value + (2.0 * math.pi)


def _axial_angle_error_deg(theta_a: float, theta_b: float) -> float:
    diff = (float(theta_a) - float(theta_b) + math.pi / 2.0) % math.pi - math.pi / 2.0
    return abs(math.degrees(diff))


def _point_has_mask_support(mask: np.ndarray, x: float, y: float) -> bool:
    h, w = mask.shape[:2]
    ix = int(round(float(x)))
    iy = int(round(float(y)))
    return bool(0 <= ix < w and 0 <= iy < h and mask[iy, ix] > 0)


def _sample_2d_bilinear(array: np.ndarray, x: float, y: float, valid_mask: np.ndarray | None = None) -> float | None:
    if array.ndim != 2 or not (math.isfinite(x) and math.isfinite(y)):
        return None
    h, w = array.shape
    if x < 0.0 or y < 0.0 or x > float(w - 1) or y > float(h - 1):
        return None
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    samples: list[tuple[float, float]] = []
    for yy in (y0, y1):
        for xx in (x0, x1):
            if valid_mask is not None and not bool(valid_mask[yy, xx]):
                continue
            value = float(array[yy, xx])
            if not math.isfinite(value):
                continue
            weight = max(1.0 - abs(float(xx) - x), 0.0) * max(1.0 - abs(float(yy) - y), 0.0)
            if weight > 0.0:
                samples.append((weight, value))
    if not samples:
        return None
    weight_sum = sum(weight for weight, _value in samples)
    if weight_sum <= 1e-8:
        return None
    return float(sum(weight * value for weight, value in samples) / weight_sum)


def _map_chart_to_pose(
    x: float,
    y: float,
    source_x: np.ndarray,
    source_y: np.ndarray,
    valid_map: np.ndarray,
) -> tuple[float, float] | None:
    pose_x = _sample_2d_bilinear(source_x, x, y, valid_mask=valid_map)
    pose_y = _sample_2d_bilinear(source_y, x, y, valid_mask=valid_map)
    if pose_x is None or pose_y is None:
        return None
    return pose_x, pose_y


def _standardize_pyfing_minutiae(raw_minutiae: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in raw_minutiae:
        quality = getattr(item, "quality", None)
        rows.append(
            {
                "x": float(getattr(item, "x")),
                "y": float(getattr(item, "y")),
                "theta": _normalize_angle_2pi(float(getattr(item, "direction", getattr(item, "angle", 0.0)))),
                "score": float(quality) if quality is not None else 1.0,
                "type": str(getattr(item, "type", "")),
                "source": "pyfing_algorithm1_depth_unwrap_patch",
            }
        )
    return rows


def _write_minutiae_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y", "theta", "score", "type", "source"])
        writer.writeheader()
        writer.writerows(rows)


def _reproject_to_pose(
    minutiae: list[dict[str, Any]],
    source_x: np.ndarray,
    source_y: np.ndarray,
    unwrap_mask: np.ndarray,
    pose_mask: np.ndarray,
    *,
    orientation_delta_px: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_map = unwrap_mask.astype(bool) & np.isfinite(source_x) & np.isfinite(source_y)
    pose_h, pose_w = pose_mask.shape
    reprojected: list[dict[str, Any]] = []
    details = {
        "extracted_count": int(len(minutiae)),
        "dropped_outside_unwrap_mask": 0,
        "dropped_no_source_map": 0,
        "dropped_outside_pose_bounds": 0,
        "dropped_no_pose_mask_support": 0,
        "orientation_projected_count": 0,
        "orientation_fallback_count": 0,
        "reprojected_count": 0,
    }

    for item in minutiae:
        x = float(item["x"])
        y = float(item["y"])
        if not _point_has_mask_support(unwrap_mask, x, y):
            details["dropped_outside_unwrap_mask"] += 1
            continue
        center = _map_chart_to_pose(x, y, source_x, source_y, valid_map)
        if center is None:
            details["dropped_no_source_map"] += 1
            continue
        pose_x, pose_y = center
        if pose_x < 0.0 or pose_y < 0.0 or pose_x >= float(pose_w) or pose_y >= float(pose_h):
            details["dropped_outside_pose_bounds"] += 1
            continue
        if not _point_has_mask_support(pose_mask, pose_x, pose_y):
            details["dropped_no_pose_mask_support"] += 1
            continue

        theta = float(item["theta"])
        dx = math.cos(theta) * orientation_delta_px
        dy = math.sin(theta) * orientation_delta_px
        forward = _map_chart_to_pose(x + dx, y + dy, source_x, source_y, valid_map)
        backward = _map_chart_to_pose(x - dx, y - dy, source_x, source_y, valid_map)
        projected_points = [point for point in (forward, backward) if point is not None]

        theta_pose = theta
        if len(projected_points) == 2:
            dx_proj = float(projected_points[0][0] - projected_points[1][0])
            dy_proj = float(projected_points[0][1] - projected_points[1][1])
            if math.hypot(dx_proj, dy_proj) >= MIN_ORIENTATION_BASELINE_PX:
                theta_pose = math.atan2(dy_proj, dx_proj)
                details["orientation_projected_count"] += 1
            else:
                details["orientation_fallback_count"] += 1
        elif len(projected_points) == 1:
            dx_proj = float(projected_points[0][0] - pose_x)
            dy_proj = float(projected_points[0][1] - pose_y)
            if math.hypot(dx_proj, dy_proj) >= MIN_ORIENTATION_BASELINE_PX:
                theta_pose = math.atan2(dy_proj, dx_proj)
                details["orientation_projected_count"] += 1
            else:
                details["orientation_fallback_count"] += 1
        else:
            details["orientation_fallback_count"] += 1

        reprojected.append(
            {
                "x": float(pose_x),
                "y": float(pose_y),
                "theta": _normalize_angle_2pi(theta_pose),
                "score": item.get("score"),
                "type": item.get("type"),
                "source": str(item.get("source", "pyfing_algorithm1_depth_unwrap_patch")) + "_pose",
                "unwrap_x": x,
                "unwrap_y": y,
            }
        )

    details["reprojected_count"] = int(len(reprojected))
    return reprojected, details


def _dense_orientation_stats(
    role_cache_dir: Path,
    role: str,
    pose_image: np.ndarray,
    pose_mask: np.ndarray,
    minutiae: list[dict[str, Any]],
) -> dict[str, Any]:
    cache_path = role_cache_dir / f"{role}_pose_orientation_pyfing.npy"
    if cache_path.exists():
        dense = np.load(cache_path).astype(np.float32)
    else:
        dense = pyfing.orientation_field_estimation(
            pose_image,
            pose_mask.astype(np.uint8) * 255,
            dpi=DEFAULT_DPI,
            method="SNFOE",
        ).astype(np.float32)
        dense[~pose_mask.astype(bool)] = 0.0
        np.save(cache_path, dense)
    errors: list[float] = []
    h, w = pose_mask.shape
    for item in minutiae:
        x = int(round(float(item["x"])))
        y = int(round(float(item["y"])))
        if 0 <= x < w and 0 <= y < h and pose_mask[y, x]:
            errors.append(_axial_angle_error_deg(float(item["theta"]), float(dense[y, x])))
    values = np.asarray(errors, dtype=np.float32)
    return {
        "count": int(len(minutiae)),
        "sampled_count": int(values.size),
        "mean_error_deg": float(np.mean(values)) if values.size else 0.0,
        "median_error_deg": float(np.median(values)) if values.size else 0.0,
        "within_15_deg_count": int(np.count_nonzero(values <= 15.0)),
        "within_15_deg_fraction": float(np.count_nonzero(values <= 15.0) / max(values.size, 1)),
    }


def _scale_pose_minutiae_to_training(
    pose_minutiae: list[dict[str, Any]],
    source_x: np.ndarray,
    source_y: np.ndarray,
    unwrap_mask: np.ndarray,
    final_mask: np.ndarray,
    *,
    scale_x: float,
    scale_y: float,
    orientation_delta_px: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_map = unwrap_mask.astype(bool) & np.isfinite(source_x) & np.isfinite(source_y)
    height, width = final_mask.shape
    scaled: list[dict[str, Any]] = []
    details = {
        "pose_reprojected_count": int(len(pose_minutiae)),
        "dropped_out_of_training_bounds": 0,
        "dropped_no_final_mask_support": 0,
        "orientation_projected_count": 0,
        "orientation_fallback_count": 0,
        "training_minutiae_count": 0,
    }
    for item in pose_minutiae:
        x_train = float(item["x"]) * scale_x
        y_train = float(item["y"]) * scale_y
        if x_train < 0.0 or y_train < 0.0 or x_train >= float(width) or y_train >= float(height):
            details["dropped_out_of_training_bounds"] += 1
            continue
        if not _point_has_mask_support(final_mask, x_train, y_train):
            details["dropped_no_final_mask_support"] += 1
            continue

        theta = float(item["theta"])
        unwrap_x = float(item["unwrap_x"])
        unwrap_y = float(item["unwrap_y"])
        dx = math.cos(theta) * orientation_delta_px
        dy = math.sin(theta) * orientation_delta_px
        projected_points: list[tuple[float, float]] = []
        for candidate in (
            _map_chart_to_pose(unwrap_x + dx, unwrap_y + dy, source_x, source_y, valid_map),
            _map_chart_to_pose(unwrap_x - dx, unwrap_y - dy, source_x, source_y, valid_map),
        ):
            if candidate is not None:
                projected_points.append((candidate[0] * scale_x, candidate[1] * scale_y))

        theta_train = theta
        if len(projected_points) == 2:
            dx_proj = float(projected_points[0][0] - projected_points[1][0])
            dy_proj = float(projected_points[0][1] - projected_points[1][1])
            if math.hypot(dx_proj, dy_proj) >= MIN_ORIENTATION_BASELINE_PX:
                theta_train = math.atan2(dy_proj, dx_proj)
                details["orientation_projected_count"] += 1
            else:
                details["orientation_fallback_count"] += 1
        elif len(projected_points) == 1:
            dx_proj = float(projected_points[0][0] - x_train)
            dy_proj = float(projected_points[0][1] - y_train)
            if math.hypot(dx_proj, dy_proj) >= MIN_ORIENTATION_BASELINE_PX:
                theta_train = math.atan2(dy_proj, dx_proj)
                details["orientation_projected_count"] += 1
            else:
                details["orientation_fallback_count"] += 1
        else:
            details["orientation_fallback_count"] += 1

        scaled.append(
            {
                "x": float(x_train),
                "y": float(y_train),
                "theta": gt._normalize_angle_2pi_scalar(theta_train),
                "score": item.get("score"),
                "type": item.get("type"),
                "source": "pyfing_algorithm1_depth_unwrap_patch_training",
            }
        )
    details["training_minutiae_count"] = int(len(scaled))
    return scaled, details


def _patch_targets(existing_targets_path: Path, gray_shape: tuple[int, int], mask: np.ndarray, minutiae: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], int]:
    existing = gt._load_npz_arrays(existing_targets_path)
    output_shape = existing["output_mask"].shape[-2:]
    mask_small_points = gt._downsample_mask_for_points(mask, output_shape)
    minutia_targets = gt._rasterize_minutiae(minutiae, gray_shape, output_shape, mask_small_points)
    for key in MINUTIA_TARGET_KEYS:
        existing[key] = minutia_targets[key]
    return existing, int(np.count_nonzero(existing["minutia_valid_mask"]))


def _sample_has_nonfinite_targets(targets: dict[str, np.ndarray]) -> list[str]:
    return [
        key
        for key, value in targets.items()
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all()
    ]


def _resolve_reconstruction_dir(dataset_root: Path, meta: dict[str, Any]) -> Path | None:
    reconstruction = meta.get("multiview_reconstruction")
    if not isinstance(reconstruction, dict):
        return None
    acquisition_id = str(reconstruction.get("acquisition_id", ""))
    candidates: list[Path] = []
    if reconstruction.get("reconstruction_dir"):
        original = Path(str(reconstruction["reconstruction_dir"]))
        candidates.append(dataset_root / "reconstructions" / original.name)
        candidates.append(original)
    if acquisition_id:
        candidates.append(dataset_root / "reconstructions" / acquisition_id)
    for candidate in candidates:
        if (candidate / "debug_views").exists():
            return candidate.resolve()
    return None


def _training_scales(sample_dir: Path, meta: dict[str, Any]) -> tuple[float, float]:
    minutiae_gt = meta.get("minutiae_ground_truth") if isinstance(meta.get("minutiae_ground_truth"), dict) else {}
    if "scale_x" in minutiae_gt and "scale_y" in minutiae_gt:
        return float(minutiae_gt["scale_x"]), float(minutiae_gt["scale_y"])
    pose = _load_gray(sample_dir / "preprocess_pose_normalized.png")
    train = _load_gray(sample_dir / "preprocessed_input.png")
    return float(train.shape[1] / max(pose.shape[1], 1)), float(train.shape[0] / max(pose.shape[0], 1))


def _ensure_acquisition_cache(
    reconstruction_dir: Path,
    acquisition_id: str,
    cache_root: Path,
    params: argparse.Namespace,
    unwrap_module: Any,
    acquisition_cache: dict[str, AcquisitionCache],
) -> AcquisitionCache:
    key = str(reconstruction_dir.resolve())
    if key in acquisition_cache:
        return acquisition_cache[key]
    cache_dir = cache_root / acquisition_id
    cached_unwrap_report_path = cache_dir / "depth_unwrap" / "algorithm1_side_depth_unwrap_report.json"
    if cached_unwrap_report_path.exists():
        unwrap_report = _read_json(cached_unwrap_report_path)
    else:
        unwrap_report = unwrap_module.run(
            reconstruction_dir=reconstruction_dir,
            output_dir=cache_dir / "depth_unwrap",
            left_angle=params.left_angle,
            right_angle=params.right_angle,
            samples_per_pixel=params.samples_per_pixel,
            reverse_left_unwrap_x=params.reverse_left_unwrap_x,
            reverse_right_unwrap_x=params.reverse_right_unwrap_x,
            unwrap_width_scale=params.unwrap_width_scale,
            unwrap_gradient_clip=None if params.unwrap_gradient_clip <= 0 else float(params.unwrap_gradient_clip),
            row_param_smooth_window=params.row_param_smooth_window,
            map_smooth_sigma_x=params.map_smooth_sigma_x,
            map_smooth_sigma_y=params.map_smooth_sigma_y,
        )
    debug_dir = reconstruction_dir / "debug_views"
    cache = AcquisitionCache(
        acquisition_id=acquisition_id,
        reconstruction_dir=reconstruction_dir,
        cache_dir=cache_dir,
        unwrap_report_path=Path(unwrap_report.get("report_path", cached_unwrap_report_path)),
    )

    for role in ("left", "right"):
        role_dir = cache_dir / role
        role_dir.mkdir(parents=True, exist_ok=True)
        unwrap_image_path = cache_dir / "depth_unwrap" / role / f"{role}_depth_unwrapped.png"
        unwrap_mask_path = cache_dir / "depth_unwrap" / role / f"{role}_depth_unwrapped_mask.png"
        unwrap_maps_path = cache_dir / "depth_unwrap" / role / f"{role}_depth_unwarp_maps.npz"
        image = _load_gray(unwrap_image_path)
        unwrap_mask = _load_gray(unwrap_mask_path) > 0
        maps = np.load(unwrap_maps_path)
        source_x = maps["source_x_map"].astype(np.float32)
        source_y = maps["source_y_map"].astype(np.float32)
        pose_image = _load_gray(debug_dir / f"{role}_pose_normalized.png")
        pose_mask = _load_gray(debug_dir / f"{role}_pose_mask.png") > 0

        extracted_path = role_dir / f"{role}_depth_unwrapped_pyfing_minutiae.json"
        pose_path = role_dir / f"{role}_reprojected_pose_minutiae.json"
        if extracted_path.exists():
            extracted = _read_json(extracted_path)["minutiae"]
        else:
            extracted = _standardize_pyfing_minutiae(pyfing.minutiae_extraction(image, dpi=DEFAULT_DPI))
            _write_json(extracted_path, {"image": str(unwrap_image_path), "minutiae": extracted})
            _write_minutiae_csv(extracted, role_dir / f"{role}_depth_unwrapped_pyfing_minutiae.csv")
        if pose_path.exists():
            pose_payload = _read_json(pose_path)
            pose_minutiae = pose_payload["minutiae"]
            reproj = pose_payload.get("reprojection", {})
        else:
            pose_minutiae, reproj = _reproject_to_pose(
                extracted,
                source_x,
                source_y,
                unwrap_mask,
                pose_mask,
                orientation_delta_px=params.orientation_delta_px,
            )
            _write_json(pose_path, {"target": f"{role}_pose_normalized", "minutiae": pose_minutiae, "reprojection": reproj})
        dense_stats = _dense_orientation_stats(role_dir, role, pose_image, pose_mask, pose_minutiae)
        cache.roles[role] = RoleCache(
            role=role,
            unwrap_image_path=unwrap_image_path,
            unwrap_mask_path=unwrap_mask_path,
            unwrap_maps_path=unwrap_maps_path,
            extracted_minutiae_path=extracted_path,
            reprojected_pose_minutiae_path=pose_path,
            extraction_count=len(extracted),
            pose_reprojected_count=len(pose_minutiae),
            orientation_projected_count=int(reproj["orientation_projected_count"]),
            orientation_fallback_count=int(reproj["orientation_fallback_count"]),
            dense_orientation_stats=dense_stats,
        )
    acquisition_cache[key] = cache
    return cache


def _backup_sample_files(dataset_root: Path, sample_dir: Path, backup_dir: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    for name in ("meta.json", "minutiae.json", "featurenet_targets.npz"):
        src = sample_dir / name
        rel = src.relative_to(dataset_root)
        dst = backup_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied[name] = str(dst)
    return copied


def _front_hashes(dataset_root: Path, manifest: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    hashes: dict[str, dict[str, str]] = {}
    for row in manifest:
        if int(row.get("raw_view_index", -1)) in SIDE_ROLES:
            continue
        sample_id = row["sample_id"]
        sample_dir = dataset_root / "samples" / sample_id
        hashes[sample_id] = {
            name: _sha256(sample_dir / name)
            for name in ("meta.json", "minutiae.json", "featurenet_targets.npz")
            if (sample_dir / name).exists()
        }
    return hashes


def patch_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"missing dataset root: {dataset_root}")
    if not args.dry_run and not args.in_place:
        raise ValueError("--in-place is required for mutation; use --dry-run for inspection")

    manifest = _read_json(dataset_root / "manifest.json")
    stats = PatchStats()
    params = {
        "row_param_smooth_window": int(args.row_param_smooth_window),
        "map_smooth_sigma_x": float(args.map_smooth_sigma_x),
        "map_smooth_sigma_y": float(args.map_smooth_sigma_y),
        "orientation_delta_px": float(args.orientation_delta_px),
        "left_angle": float(args.left_angle),
        "right_angle": float(args.right_angle),
        "samples_per_pixel": float(args.samples_per_pixel),
        "unwrap_width_scale": float(args.unwrap_width_scale),
        "unwrap_gradient_clip": None if args.unwrap_gradient_clip <= 0 else float(args.unwrap_gradient_clip),
    }
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = args.backup_dir.resolve() if args.backup_dir else dataset_root / f"side_depth_unwrap_patch_backup_{timestamp}"
    cache_root = args.cache_root.resolve() if args.cache_root else REPO_ROOT / "tmp" / "algorithm1_depth_unwrap_patch_cache" / dataset_root.name
    unwrap_module = _load_depth_unwrap_module()
    acquisition_cache: dict[str, AcquisitionCache] = {}
    sample_reports: list[dict[str, Any]] = []
    role_aggregate: dict[str, dict[str, list[float] | int]] = {
        "left": {"patched": 0, "training_count": 0, "rasterized_count": 0},
        "right": {"patched": 0, "training_count": 0, "rasterized_count": 0},
    }
    front_before = _front_hashes(dataset_root, manifest)

    for row in manifest:
        stats.scanned += 1
        sample_id = row["sample_id"]
        raw_view_index = int(row.get("raw_view_index", -1))
        role = SIDE_ROLES.get(raw_view_index)
        if role is None:
            stats.skipped_not_side += 1
            continue
        sample_dir = dataset_root / "samples" / sample_id
        try:
            meta_path = sample_dir / "meta.json"
            required = [meta_path, sample_dir / "minutiae.json", sample_dir / "featurenet_targets.npz", sample_dir / "mask.png"]
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                stats.skipped_missing_files += 1
                sample_reports.append({"sample_id": sample_id, "status": "missing_files", "missing": missing})
                continue
            meta = _read_json(meta_path)
            reconstruction_dir = _resolve_reconstruction_dir(dataset_root, meta)
            if reconstruction_dir is None:
                stats.skipped_missing_files += 1
                sample_reports.append({"sample_id": sample_id, "status": "missing_reconstruction"})
                continue
            acquisition_id = str(meta.get("multiview_reconstruction", {}).get("acquisition_id") or reconstruction_dir.name)
            cache = _ensure_acquisition_cache(reconstruction_dir, acquisition_id, cache_root, args, unwrap_module, acquisition_cache)
            role_cache = cache.roles[role]
            maps = np.load(role_cache.unwrap_maps_path)
            source_x = maps["source_x_map"].astype(np.float32)
            source_y = maps["source_y_map"].astype(np.float32)
            unwrap_mask = _load_gray(role_cache.unwrap_mask_path) > 0
            pose_minutiae = _read_json(role_cache.reprojected_pose_minutiae_path)["minutiae"]
            final_mask = _load_gray(sample_dir / "mask.png")
            gray_path = sample_dir / "preprocessed_input.png"
            if not gray_path.exists():
                gray_path = sample_dir / "masked_image.png"
            gray_image = _load_gray(gray_path)
            scale_x, scale_y = _training_scales(sample_dir, meta)
            training_minutiae, training_details = _scale_pose_minutiae_to_training(
                pose_minutiae,
                source_x,
                source_y,
                unwrap_mask,
                final_mask,
                scale_x=scale_x,
                scale_y=scale_y,
                orientation_delta_px=args.orientation_delta_px,
            )
            patched_targets, rasterized_count = _patch_targets(
                sample_dir / "featurenet_targets.npz",
                gray_image.shape,
                final_mask,
                training_minutiae,
            )
            if rasterized_count == 0 and not args.allow_zero:
                stats.skipped_zero_after_patch += 1
                sample_reports.append({"sample_id": sample_id, "status": "zero_after_patch", **training_details})
                continue
            bad_targets = _sample_has_nonfinite_targets(patched_targets)
            if bad_targets:
                raise ValueError(f"non-finite patched targets: {bad_targets}")

            pre_meta = _read_json(meta_path)
            pre_minutiae = _read_json(sample_dir / "minutiae.json")
            pre_targets = gt._load_npz_arrays(sample_dir / "featurenet_targets.npz")
            pre_rasterized = int(np.count_nonzero(pre_targets["minutia_valid_mask"]))
            patch_record = {
                "patch_source": "algorithm1_depth_unwrap_side_patch",
                "parameters": params,
                "cache_dir": str(cache.cache_dir),
                "unwrap_maps": str(role_cache.unwrap_maps_path),
                "extracted_minutiae_path": str(role_cache.extracted_minutiae_path),
                "pose_reprojected_minutiae_path": str(role_cache.reprojected_pose_minutiae_path),
                "pre_patch_minutiae_count": int(len(pre_minutiae)),
                "post_patch_minutiae_count": int(len(training_minutiae)),
                "pre_patch_rasterized_minutiae_count": pre_rasterized,
                "post_patch_rasterized_minutiae_count": int(rasterized_count),
                "pose_dense_orientation_stats": role_cache.dense_orientation_stats,
                "pose_reprojection": {
                    "extraction_count": role_cache.extraction_count,
                    "pose_reprojected_count": role_cache.pose_reprojected_count,
                    "orientation_projected_count": role_cache.orientation_projected_count,
                    "orientation_fallback_count": role_cache.orientation_fallback_count,
                },
                "training_reprojection": training_details,
            }

            if args.dry_run:
                stats.dry_run_would_patch += 1
                sample_reports.append({"sample_id": sample_id, "role": role, "status": "would_patch", **patch_record})
                continue

            backup_files = _backup_sample_files(dataset_root, sample_dir, backup_dir)
            minutiae_gt = pre_meta.get("minutiae_ground_truth") if isinstance(pre_meta.get("minutiae_ground_truth"), dict) else {}
            pre_meta["minutiae_ground_truth"] = {
                **minutiae_gt,
                "mode": "reconstruction_backed",
                "canonical_source": "pyfing_algorithm1_depth_unwrap",
                "view_role": role,
                "patch_source": "algorithm1_depth_unwrap_side_patch",
                "scale_x": float(scale_x),
                "scale_y": float(scale_y),
                "reprojected_minutiae_count": int(len(training_minutiae)),
                "rasterized_minutiae_count": int(rasterized_count),
                **training_details,
            }
            pre_meta.setdefault("patches", []).append(patch_record)
            counts = pre_meta.setdefault("counts", {})
            counts["minutiae"] = int(len(training_minutiae))
            counts["minutia_support_pixels"] = int(rasterized_count)
            counts["rasterized_minutiae_count"] = int(rasterized_count)

            _write_json(sample_dir / "minutiae.json", training_minutiae)
            np.savez_compressed(sample_dir / "featurenet_targets.npz", **patched_targets)
            _write_json(meta_path, pre_meta)
            stats.patched += 1
            role_aggregate[role]["patched"] = int(role_aggregate[role]["patched"]) + 1
            role_aggregate[role]["training_count"] = int(role_aggregate[role]["training_count"]) + len(training_minutiae)
            role_aggregate[role]["rasterized_count"] = int(role_aggregate[role]["rasterized_count"]) + rasterized_count
            sample_reports.append({"sample_id": sample_id, "role": role, "status": "patched", "backup_files": backup_files, **patch_record})
        except Exception as exc:  # noqa: BLE001 - patch report should include per-sample errors.
            stats.errors += 1
            sample_reports.append({"sample_id": sample_id, "role": role, "status": "error", "error": str(exc)})

    front_after = _front_hashes(dataset_root, manifest)
    front_hashes_unchanged = front_before == front_after
    summary = {
        "dataset_root": str(dataset_root),
        "dry_run": bool(args.dry_run),
        "backup_dir": None if args.dry_run else str(backup_dir),
        "cache_root": str(cache_root),
        "parameters": params,
        "stats": asdict(stats),
        "front_hashes_unchanged": bool(front_hashes_unchanged),
        "role_aggregate": role_aggregate,
        "acquisitions": {
            key: {
                "acquisition_id": cache.acquisition_id,
                "reconstruction_dir": str(cache.reconstruction_dir),
                "cache_dir": str(cache.cache_dir),
                "unwrap_report_path": str(cache.unwrap_report_path),
                "roles": {role: asdict(role_cache) for role, role_cache in cache.roles.items()},
            }
            for key, cache in acquisition_cache.items()
        },
        "samples": sample_reports,
    }
    if not args.dry_run:
        _write_json(dataset_root / "algorithm1_depth_unwrap_side_patch_summary.json", summary)
    else:
        _write_json(cache_root / "algorithm1_depth_unwrap_side_patch_dry_run_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-zero", action="store_true")
    parser.add_argument("--row-param-smooth-window", type=int, default=DEFAULT_ROW_PARAM_SMOOTH_WINDOW)
    parser.add_argument("--map-smooth-sigma-x", type=float, default=DEFAULT_MAP_SMOOTH_SIGMA_X)
    parser.add_argument("--map-smooth-sigma-y", type=float, default=DEFAULT_MAP_SMOOTH_SIGMA_Y)
    parser.add_argument("--orientation-delta-px", type=float, default=DEFAULT_ORIENTATION_DELTA_PX)
    parser.add_argument("--left-angle", type=float, default=-45.0)
    parser.add_argument("--right-angle", type=float, default=45.0)
    parser.add_argument("--samples-per-pixel", type=float, default=2.0)
    parser.add_argument("--reverse-left-unwrap-x", action="store_true")
    parser.add_argument("--reverse-right-unwrap-x", action="store_true")
    parser.add_argument("--unwrap-width-scale", type=float, default=1.0)
    parser.add_argument("--unwrap-gradient-clip", type=float, default=3.0)
    args = parser.parse_args()
    summary = patch_dataset(args)
    print(json.dumps({"stats": summary["stats"], "front_hashes_unchanged": summary["front_hashes_unchanged"], "summary_path": str(Path(summary["cache_root"]) / "algorithm1_depth_unwrap_side_patch_dry_run_summary.json") if args.dry_run else str(Path(summary["dataset_root"]) / "algorithm1_depth_unwrap_side_patch_summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

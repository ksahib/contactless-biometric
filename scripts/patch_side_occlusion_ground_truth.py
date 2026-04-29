#!/usr/bin/env python
"""Patch existing FeatureNet ground truth with side-view occlusion gating.

This utility is intentionally conservative: it does not rerun segmentation,
unwarping, reconstruction, FingerFlow, or PyFing. It copies an existing ground
truth root to a new output root by default, then rewrites only side-view
reconstruction-backed minutiae labels and minutiae target arrays.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import generate_ground_truth as gt  # noqa: E402


SIDE_ROLES = {1: "left", 2: "right"}
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
    copied_samples: int = 0
    patched: int = 0
    unchanged: int = 0
    skipped_not_side: int = 0
    skipped_not_reconstruction_backed: int = 0
    skipped_missing_required_files: int = 0
    skipped_missing_dense_maps: int = 0
    skipped_zero_after_patch: int = 0
    errors: int = 0


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def _copy_limited_dataset(input_root: Path, output_root: Path, limit: int | None) -> list[dict[str, Any]]:
    manifest = _read_json(input_root / "manifest.json")
    if limit is not None:
        manifest = manifest[: max(0, int(limit))]

    (output_root / "samples").mkdir(parents=True, exist_ok=True)
    (output_root / "reconstructions").mkdir(parents=True, exist_ok=True)

    copied_reconstructions: set[str] = set()
    for row in manifest:
        sample_id = row["sample_id"]
        src_sample = input_root / "samples" / sample_id
        dst_sample = output_root / "samples" / sample_id
        if src_sample.exists():
            shutil.copytree(src_sample, dst_sample)

        meta_path = src_sample / "meta.json"
        if not meta_path.exists():
            continue
        meta = _read_json(meta_path)
        reconstruction = meta.get("multiview_reconstruction")
        if not isinstance(reconstruction, dict):
            continue
        reconstruction_dir = Path(reconstruction.get("reconstruction_dir", ""))
        if not reconstruction_dir.exists():
            reconstruction_dir = input_root / "reconstructions" / Path(str(reconstruction.get("acquisition_id", ""))).name
        if not reconstruction_dir.exists() or reconstruction_dir.name in copied_reconstructions:
            continue
        shutil.copytree(reconstruction_dir, output_root / "reconstructions" / reconstruction_dir.name)
        copied_reconstructions.add(reconstruction_dir.name)

    _write_json(output_root / "manifest.json", manifest)
    summary_path = input_root / "summary.json"
    if summary_path.exists():
        summary = _read_json(summary_path)
    else:
        summary = {}
    summary["patch_subset"] = {
        "source_root": str(input_root.resolve()),
        "limit": limit,
        "sample_count": len(manifest),
    }
    _write_json(output_root / "summary.json", summary)
    return manifest


def _prepare_output_root(input_root: Path, output_root: Path, limit: int | None, overwrite: bool) -> list[dict[str, Any]]:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output root already exists: {output_root}")
        shutil.rmtree(output_root)

    if limit is None:
        shutil.copytree(input_root, output_root)
        return _read_json(output_root / "manifest.json")

    output_root.mkdir(parents=True, exist_ok=False)
    return _copy_limited_dataset(input_root, output_root, limit)


def _resolve_output_reconstruction_path(output_root: Path, path_text: str | None, acquisition_id: str | None) -> Path | None:
    candidates: list[Path] = []
    if path_text:
        original = Path(path_text)
        if original.name:
            candidates.append(output_root / "reconstructions" / original.parent.name / original.name)
        candidates.append(original)
    if acquisition_id:
        candidates.append(output_root / "reconstructions" / acquisition_id / "reconstruction_maps.npz")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_reconstruction_file(
    output_root: Path,
    reconstruction: dict[str, Any],
    minutiae_gt: dict[str, Any],
    key: str,
    filename: str,
) -> Path | None:
    path_text = minutiae_gt.get(key) or reconstruction.get(key)
    candidates: list[Path] = []
    if path_text:
        original = Path(str(path_text))
        if original.name:
            candidates.append(output_root / "reconstructions" / original.parent.name / original.name)
        candidates.append(original)
    acquisition_id = reconstruction.get("acquisition_id")
    if acquisition_id:
        candidates.append(output_root / "reconstructions" / str(acquisition_id) / filename)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _build_side_visibility_mask(
    reconstruction_maps: dict[str, np.ndarray],
    role: str,
    *,
    depth_mode: str,
    depth_tolerance: float,
) -> np.ndarray:
    support = reconstruction_maps["support_mask"] > 0
    height, width = support.shape
    x_map_key = f"{role}_pose_x_map"
    y_map_key = f"{role}_pose_y_map"
    depth_key = f"depth_{role}"
    for key in (x_map_key, y_map_key, depth_key):
        if key not in reconstruction_maps:
            raise KeyError(f"missing {key} in reconstruction maps")

    ys, xs = np.nonzero(support)
    x_pose = reconstruction_maps[x_map_key][ys, xs]
    y_pose = reconstruction_maps[y_map_key][ys, xs]
    depth = reconstruction_maps[depth_key][ys, xs].astype(np.float32)
    xp = np.rint(x_pose).astype(np.int32)
    yp = np.rint(y_pose).astype(np.int32)
    valid = (
        np.isfinite(x_pose)
        & np.isfinite(y_pose)
        & np.isfinite(depth)
        & (xp >= 0)
        & (xp < width)
        & (yp >= 0)
        & (yp < height)
    )

    xs = xs[valid]
    ys = ys[valid]
    xp = xp[valid]
    yp = yp[valid]
    depth = depth[valid]

    side_flat = yp * width + xp
    if depth_mode == "min":
        z_buffer = np.full(height * width, np.inf, dtype=np.float32)
        np.minimum.at(z_buffer, side_flat, depth)
        visible_points = depth <= (z_buffer[side_flat] + float(depth_tolerance))
    elif depth_mode == "max":
        z_buffer = np.full(height * width, -np.inf, dtype=np.float32)
        np.maximum.at(z_buffer, side_flat, depth)
        visible_points = depth >= (z_buffer[side_flat] - float(depth_tolerance))
    else:
        raise ValueError(f"unsupported depth mode: {depth_mode}")

    visibility = np.zeros((height, width), dtype=np.uint8)
    visibility[ys[visible_points], xs[visible_points]] = 1
    return visibility


def _point_visible(visibility: np.ndarray, x: float, y: float, radius: int) -> bool:
    if not (math.isfinite(float(x)) and math.isfinite(float(y))):
        return False
    height, width = visibility.shape
    cx = int(round(float(x)))
    cy = int(round(float(y)))
    if cx < 0 or cy < 0 or cx >= width or cy >= height:
        return False
    radius = max(0, int(radius))
    y0 = max(0, cy - radius)
    y1 = min(height, cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(width, cx + radius + 1)
    return bool(np.any(visibility[y0:y1, x0:x1] > 0))


def _remap_with_side_visibility(
    canonical_minutiae: list[dict[str, Any]],
    unwarp_maps: dict[str, np.ndarray],
    reconstruction_maps: dict[str, np.ndarray],
    role: str,
    final_mask: np.ndarray,
    scale_x: float,
    scale_y: float,
    visibility: np.ndarray,
    *,
    visibility_radius: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_minutiae, counters = gt._canonical_minutiae_with_front_sources(canonical_minutiae, unwarp_maps)
    remapped: list[dict[str, Any]] = []
    height, width = final_mask.shape
    details: dict[str, Any] = {
        "view_role": role,
        "projection_map_mode": "dense_pose_map",
        "side_visibility_mode": "z_buffer_patch",
        "side_visibility_visible_source_count": int(np.count_nonzero(visibility)),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        **counters,
        "dropped_occluded_in_side_view": 0,
        "dropped_projection_failed": 0,
        "dropped_out_of_training_bounds": 0,
        "dropped_no_final_mask_support": 0,
        "orientation_projected_count": 0,
        "orientation_fallback_count": 0,
        "reprojected_minutiae_count": 0,
    }
    delta = 4.0
    min_orientation_baseline_px = 1.5

    for minutia, source_x, source_y in source_minutiae:
        if not _point_visible(visibility, source_x, source_y, visibility_radius):
            details["dropped_occluded_in_side_view"] += 1
            continue

        destination = gt._project_front_source_to_training_frame(
            reconstruction_maps,
            role,
            source_x,
            source_y,
            scale_x,
            scale_y,
        )
        if destination is None:
            details["dropped_projection_failed"] += 1
            continue
        x_dst, y_dst = destination
        if x_dst < 0.0 or y_dst < 0.0 or x_dst >= float(width) or y_dst >= float(height):
            details["dropped_out_of_training_bounds"] += 1
            continue
        if not gt._point_has_mask_support(final_mask, x_dst, y_dst):
            details["dropped_no_final_mask_support"] += 1
            continue

        theta = float(minutia["theta"])
        dx = math.cos(theta) * delta
        dy = math.sin(theta) * delta
        projected_points: list[tuple[float, float]] = []
        for candidate in (
            gt._map_unwarped_to_front_source(unwarp_maps, float(minutia["x"]) + dx, float(minutia["y"]) + dy),
            gt._map_unwarped_to_front_source(unwarp_maps, float(minutia["x"]) - dx, float(minutia["y"]) - dy),
        ):
            if candidate is None:
                continue
            projected = gt._project_front_source_to_training_frame(
                reconstruction_maps,
                role,
                candidate[0],
                candidate[1],
                scale_x,
                scale_y,
            )
            if projected is not None:
                projected_points.append(projected)

        theta_dst = theta
        if len(projected_points) == 2:
            dx_proj = float(projected_points[0][0] - projected_points[1][0])
            dy_proj = float(projected_points[0][1] - projected_points[1][1])
            if math.hypot(dx_proj, dy_proj) >= min_orientation_baseline_px:
                theta_dst = math.atan2(dy_proj, dx_proj)
                details["orientation_projected_count"] += 1
            else:
                details["orientation_fallback_count"] += 1
        elif len(projected_points) == 1:
            dx_proj = float(projected_points[0][0] - x_dst)
            dy_proj = float(projected_points[0][1] - y_dst)
            if math.hypot(dx_proj, dy_proj) >= min_orientation_baseline_px:
                theta_dst = math.atan2(dy_proj, dx_proj)
                details["orientation_projected_count"] += 1
            else:
                details["orientation_fallback_count"] += 1
        else:
            details["orientation_fallback_count"] += 1

        remapped.append(
            {
                "x": float(x_dst),
                "y": float(y_dst),
                "theta": gt._normalize_angle_2pi_scalar(theta_dst),
                "score": minutia.get("score"),
                "type": minutia.get("type"),
                "source": str(minutia.get("source", "reconstruction_unwarp")),
            }
        )

    details["reprojected_minutiae_count"] = len(remapped)
    return remapped, details


def _patch_targets(existing_targets_path: Path, gray_shape: tuple[int, int], mask: np.ndarray, minutiae: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], int]:
    existing = gt._load_npz_arrays(existing_targets_path)
    output_shape = existing["output_mask"].shape[-2:]
    mask_small_points = gt._downsample_mask_for_points(mask, output_shape)
    minutia_targets = gt._rasterize_minutiae(minutiae, gray_shape, output_shape, mask_small_points)
    for key in MINUTIA_TARGET_KEYS:
        existing[key] = minutia_targets[key]
    return existing, int(np.count_nonzero(existing["minutia_valid_mask"]))


def _sample_has_nonfinite_targets(targets: dict[str, np.ndarray]) -> list[str]:
    bad: list[str] = []
    for key, array in targets.items():
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            bad.append(key)
    return bad


def _patch_sample(
    output_root: Path,
    sample_id: str,
    *,
    depth_mode: str,
    depth_tolerance: float,
    visibility_radius: int,
    allow_zero: bool,
    visibility_cache: dict[tuple[str, str, str, float], np.ndarray],
) -> tuple[str, dict[str, Any]]:
    sample_dir = output_root / "samples" / sample_id
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        return "missing", {"reason": "missing_meta"}
    meta = _read_json(meta_path)
    raw_view_index = int(meta.get("raw_view_index", -1))
    role = SIDE_ROLES.get(raw_view_index)
    if role is None:
        return "not_side", {}

    minutiae_gt = meta.get("minutiae_ground_truth")
    if not isinstance(minutiae_gt, dict) or minutiae_gt.get("mode") != "reconstruction_backed":
        return "not_reconstruction_backed", {}

    reconstruction = meta.get("multiview_reconstruction")
    if not isinstance(reconstruction, dict):
        return "missing", {"reason": "missing_multiview_reconstruction"}

    reconstruction_maps_path = _resolve_reconstruction_file(
        output_root,
        reconstruction,
        minutiae_gt,
        "reconstruction_maps_path",
        "reconstruction_maps.npz",
    )
    unwarp_maps_path = _resolve_reconstruction_file(
        output_root,
        reconstruction,
        minutiae_gt,
        "center_unwarp_maps_path",
        "center_unwarp_maps.npz",
    )
    canonical_path = _resolve_reconstruction_file(
        output_root,
        reconstruction,
        minutiae_gt,
        "canonical_unwarped_minutiae_path",
        "canonical_unwarped_minutiae.json",
    )
    required = {
        "reconstruction_maps": reconstruction_maps_path,
        "center_unwarp_maps": unwarp_maps_path,
        "canonical_unwarped_minutiae": canonical_path,
        "targets": sample_dir / "featurenet_targets.npz",
        "mask": sample_dir / "mask.png",
        "minutiae": sample_dir / "minutiae.json",
    }
    missing = [name for name, path in required.items() if path is None or not Path(path).exists()]
    if missing:
        return "missing", {"reason": "missing_required_files", "missing": missing}

    reconstruction_maps = gt._load_npz_arrays(Path(reconstruction_maps_path))
    dense_required = [f"{role}_pose_x_map", f"{role}_pose_y_map", f"depth_{role}", "support_mask"]
    missing_dense = [key for key in dense_required if key not in reconstruction_maps]
    if missing_dense:
        return "missing_dense", {"missing_dense_maps": missing_dense}

    cache_key = (str(Path(reconstruction_maps_path).resolve()), role, depth_mode, float(depth_tolerance))
    visibility = visibility_cache.get(cache_key)
    if visibility is None:
        visibility = _build_side_visibility_mask(
            reconstruction_maps,
            role,
            depth_mode=depth_mode,
            depth_tolerance=depth_tolerance,
        )
        visibility_cache[cache_key] = visibility

    unwarp_maps = gt._load_npz_arrays(Path(unwarp_maps_path))
    canonical_minutiae = _read_json(Path(canonical_path))
    final_mask = _load_gray(sample_dir / "mask.png")
    scale_x = float(minutiae_gt.get("scale_x", 1.0))
    scale_y = float(minutiae_gt.get("scale_y", 1.0))
    patched_minutiae, remap_details = _remap_with_side_visibility(
        canonical_minutiae,
        unwarp_maps,
        reconstruction_maps,
        role,
        final_mask,
        scale_x,
        scale_y,
        visibility,
        visibility_radius=visibility_radius,
    )

    gray_path = sample_dir / "preprocessed_input.png"
    if not gray_path.exists():
        gray_path = sample_dir / "masked_image.png"
    gray_image = _load_gray(gray_path)
    patched_targets, rasterized_count = _patch_targets(
        sample_dir / "featurenet_targets.npz",
        gray_image.shape,
        final_mask,
        patched_minutiae,
    )
    if rasterized_count == 0 and not allow_zero:
        return "zero_after_patch", {
            "pre_patch_reprojected_minutiae_count": int(minutiae_gt.get("reprojected_minutiae_count", 0)),
            "post_patch_reprojected_minutiae_count": len(patched_minutiae),
            "dropped_occluded_in_side_view": remap_details["dropped_occluded_in_side_view"],
        }

    bad_targets = _sample_has_nonfinite_targets(patched_targets)
    if bad_targets:
        raise ValueError(f"non-finite patched targets for {sample_id}: {bad_targets}")

    pre_patch_reprojected = int(minutiae_gt.get("reprojected_minutiae_count", len(_read_json(sample_dir / "minutiae.json"))))
    pre_patch_rasterized = int(minutiae_gt.get("rasterized_minutiae_count", int(meta.get("counts", {}).get("rasterized_minutiae_count", 0))))
    patched_details = {
        **minutiae_gt,
        **remap_details,
        "mode": "reconstruction_backed",
        "canonical_source": minutiae_gt.get("canonical_source"),
        "view_role": role,
        "patch_source": "side_occlusion_visibility",
        "side_visibility_mode": "z_buffer_patch",
        "side_visibility_depth_mode": depth_mode,
        "side_visibility_depth_tolerance": float(depth_tolerance),
        "side_visibility_radius": int(visibility_radius),
        "pre_patch_reprojected_minutiae_count": pre_patch_reprojected,
        "pre_patch_rasterized_minutiae_count": pre_patch_rasterized,
        "post_patch_reprojected_minutiae_count": len(patched_minutiae),
        "post_patch_rasterized_minutiae_count": rasterized_count,
        "reprojected_minutiae_count": len(patched_minutiae),
        "rasterized_minutiae_count": rasterized_count,
    }
    meta["minutiae_ground_truth"] = patched_details
    meta.setdefault("patches", []).append(
        {
            "patch_source": "side_occlusion_visibility",
            "side_visibility_mode": "z_buffer_patch",
            "side_visibility_depth_mode": depth_mode,
            "side_visibility_depth_tolerance": float(depth_tolerance),
            "side_visibility_radius": int(visibility_radius),
            "pre_patch_reprojected_minutiae_count": pre_patch_reprojected,
            "post_patch_reprojected_minutiae_count": len(patched_minutiae),
            "dropped_occluded_in_side_view": remap_details["dropped_occluded_in_side_view"],
        }
    )
    counts = meta.setdefault("counts", {})
    counts["minutiae"] = len(patched_minutiae)
    counts["minutia_support_pixels"] = rasterized_count
    counts["rasterized_minutiae_count"] = rasterized_count

    _write_json(sample_dir / "minutiae.json", patched_minutiae)
    _save_npz(sample_dir / "featurenet_targets.npz", patched_targets)
    _write_json(meta_path, meta)
    return "patched", {
        "pre_patch_reprojected_minutiae_count": pre_patch_reprojected,
        "post_patch_reprojected_minutiae_count": len(patched_minutiae),
        "pre_patch_rasterized_minutiae_count": pre_patch_rasterized,
        "post_patch_rasterized_minutiae_count": rasterized_count,
        "dropped_occluded_in_side_view": remap_details["dropped_occluded_in_side_view"],
    }


def _write_patch_summary(output_root: Path, summary: dict[str, Any]) -> None:
    path = output_root / "side_occlusion_patch_summary.json"
    _write_json(path, summary)
    summary_path = output_root / "summary.json"
    if summary_path.exists():
        existing = _read_json(summary_path)
    else:
        existing = {}
    existing["side_occlusion_patch"] = summary
    _write_json(summary_path, existing)


def patch_dataset(args: argparse.Namespace) -> dict[str, Any]:
    input_root = args.input_root.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"missing input root: {input_root}")
    if not args.in_place and args.output_root is None:
        raise ValueError("--output-root is required unless --in-place is used")
    output_root = input_root if args.in_place else args.output_root.resolve()

    if args.in_place:
        manifest = _read_json(input_root / "manifest.json")
        if args.limit is not None:
            manifest = manifest[: max(0, int(args.limit))]
    else:
        manifest = _prepare_output_root(input_root, output_root, args.limit, args.overwrite)

    stats = PatchStats(scanned=len(manifest), copied_samples=len(manifest))
    visibility_cache: dict[tuple[str, str, str, float], np.ndarray] = {}
    patched_samples: list[dict[str, Any]] = []
    skipped_samples: list[dict[str, Any]] = []
    error_samples: list[dict[str, Any]] = []

    for row in manifest:
        sample_id = row["sample_id"]
        try:
            status, details = _patch_sample(
                output_root,
                sample_id,
                depth_mode=args.depth_mode,
                depth_tolerance=args.depth_tolerance,
                visibility_radius=args.visibility_radius,
                allow_zero=args.allow_zero,
                visibility_cache=visibility_cache,
            )
        except Exception as exc:
            stats.errors += 1
            error_samples.append({"sample_id": sample_id, "error": str(exc)})
            if not args.keep_going:
                raise
            continue

        if status == "patched":
            stats.patched += 1
            patched_samples.append({"sample_id": sample_id, **details})
        elif status == "not_side":
            stats.skipped_not_side += 1
        elif status == "not_reconstruction_backed":
            stats.skipped_not_reconstruction_backed += 1
        elif status == "missing":
            stats.skipped_missing_required_files += 1
            skipped_samples.append({"sample_id": sample_id, **details})
        elif status == "missing_dense":
            stats.skipped_missing_dense_maps += 1
            skipped_samples.append({"sample_id": sample_id, **details})
        elif status == "zero_after_patch":
            stats.skipped_zero_after_patch += 1
            skipped_samples.append({"sample_id": sample_id, **details})
        else:
            stats.unchanged += 1

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "in_place": bool(args.in_place),
        "limit": args.limit,
        "depth_mode": args.depth_mode,
        "depth_tolerance": float(args.depth_tolerance),
        "visibility_radius": int(args.visibility_radius),
        "stats": asdict(stats),
        "patched_samples": patched_samples[: args.report_sample_limit],
        "skipped_samples": skipped_samples[: args.report_sample_limit],
        "error_samples": error_samples[: args.report_sample_limit],
        "version": 1,
    }
    _write_patch_summary(output_root, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Patch only the first N manifest samples.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output root if it already exists.")
    parser.add_argument("--in-place", action="store_true", help="Patch the input root directly. Not recommended.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after per-sample patch errors.")
    parser.add_argument("--allow-zero", action="store_true", help="Allow a side sample to be rewritten with zero rasterized minutiae.")
    parser.add_argument("--depth-mode", choices=("max", "min"), default="max")
    parser.add_argument("--depth-tolerance", type=float, default=0.0)
    parser.add_argument("--visibility-radius", type=int, default=0)
    parser.add_argument("--report-sample-limit", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    summary = patch_dataset(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

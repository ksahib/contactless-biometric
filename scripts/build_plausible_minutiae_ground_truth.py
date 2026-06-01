#!/usr/bin/env python
"""Build a JSONL sidecar of plausible minutiae masks in parallel.

This walks the DS123 merged ground-truth tree, extracts plausible ridge
endings and bifurcations from the role-matched unwarped source view, projects
them back onto the FeatureNet output grid, and writes one JSONL row per sample.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))

import generate_ground_truth as gt  # noqa: E402
from featurenet.models import plausible_minutiae as pm  # noqa: E402


DEFAULT_GROUND_TRUTH_ROOT = REPO_ROOT / "ground_truth" / "DS123_merged_v5"
DEFAULT_OUTPUT_PATH = DEFAULT_GROUND_TRUTH_ROOT / "plausible_minutiae.jsonl"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_hw_shape(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        height = int(value[0])
        width = int(value[1])
    except (TypeError, ValueError):
        return None
    if height <= 0 or width <= 0:
        return None
    return height, width


def _relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path.resolve())


def _load_optional_gray(path: Path, fallback: np.ndarray | None = None) -> np.ndarray:
    if path.exists():
        return pm.load_gray_image(path)
    if fallback is None:
        raise FileNotFoundError(f"missing required image: {path}")
    return fallback.copy()


def _build_reconstruction_dataclass(reconstruction_dir: Path, reconstruction_meta: Mapping[str, Any]) -> gt.AcquisitionReconstructionResult:
    return gt.AcquisitionReconstructionResult(
        acquisition_id=str(reconstruction_meta.get("acquisition_id") or reconstruction_dir.name),
        reconstruction_dir=str(reconstruction_dir),
        depth_front_path=str(reconstruction_meta.get("depth_front_path", "")),
        depth_left_path=str(reconstruction_meta.get("depth_left_path", "")),
        depth_right_path=str(reconstruction_meta.get("depth_right_path", "")),
        depth_gradient_labels_path=str(reconstruction_meta.get("depth_gradient_labels_path", "")),
        reconstruction_maps_path=str(reconstruction_meta.get("reconstruction_maps_path", "")),
        support_mask_path=str(reconstruction_meta.get("support_mask_path", "")),
        row_measurements_path=str(reconstruction_meta.get("row_measurements_path", "")),
        meta_path=str(reconstruction_meta.get("meta_path", "")),
        preview_path=str(reconstruction_meta.get("preview_path", "")),
        center_unwarp_maps_path=str(reconstruction_meta.get("center_unwarp_maps_path", "")),
        center_unwarped_image_path=str(reconstruction_dir / "center_unwarped.png"),
        center_unwarped_mask_path=str(reconstruction_dir / "center_unwarped_mask.png"),
        surface_front_3d_html_path=str(reconstruction_meta.get("surface_front_3d_html_path", "")),
        surface_front_3d_png_path=str(reconstruction_meta.get("surface_front_3d_png_path", "")),
        surface_all_branches_3d_html_path=str(reconstruction_meta.get("surface_all_branches_3d_html_path", "")),
        surface_all_branches_3d_png_path=str(reconstruction_meta.get("surface_all_branches_3d_png_path", "")),
        reprojection_report_path=str(reconstruction_meta.get("reprojection_report_path", "")),
        reprojection_preview_path=str(reconstruction_meta.get("reprojection_preview_path", "")),
        valid_row_count=int(reconstruction_meta.get("valid_row_count", 1)),
        support_pixel_count=int(reconstruction_meta.get("support_pixel_count", 1)),
        input_view_paths={},
        debug_view_paths={},
    )


def _build_prepared_bundle(sample_dir: Path, meta: dict[str, Any]) -> tuple[gt.PreparedBundleArtifacts, dict[str, Any], np.ndarray, np.ndarray]:
    reconstruction_meta = meta.get("multiview_reconstruction")
    if not isinstance(reconstruction_meta, dict) or not reconstruction_meta.get("reconstruction_dir"):
        raise RuntimeError("sample metadata does not contain a reconstruction_dir")

    reconstruction_dir = Path(str(reconstruction_meta["reconstruction_dir"])).expanduser().resolve()
    if not reconstruction_dir.exists():
        raise FileNotFoundError(f"missing reconstruction directory: {reconstruction_dir}")

    raw_view_index = int(meta.get("raw_view_index", -1))
    role = pm.role_for_raw_view_index(raw_view_index)
    if role is None:
        raise RuntimeError(f"unsupported raw_view_index: {raw_view_index}")

    label_frame = pm.label_frame_for_role(reconstruction_dir, role)
    source_image = Path(label_frame["image_path"])
    source_mask = Path(label_frame["mask_path"])
    if not source_image.exists() or not source_mask.exists():
        raise FileNotFoundError(f"missing source-frame reconstruction artifacts for role={role}")

    source_gray = pm.load_gray_image(source_image)
    source_mask_gray = pm.load_gray_image(source_mask)

    raw_input_path = sample_dir / "raw_input.png"
    masked_image_path = sample_dir / "masked_image.png"
    mask_path = sample_dir / "mask.png"
    preprocessed_normalized_path = sample_dir / "preprocess_normalized.png"
    pose_normalized_path = sample_dir / "preprocess_pose_normalized.png"
    pose_mask_path = sample_dir / "preprocess_pose_mask.png"
    preprocessed_input_path = sample_dir / "preprocessed_input.png"
    preprocess_mask_path = sample_dir / "preprocess_mask.png"
    orientation_path = sample_dir / "orientation.npy"
    ridge_period_path = sample_dir / "ridge_period.npy"
    gradient_vis_path = sample_dir / "gradient_visualization.npy"

    masked_image = _load_optional_gray(masked_image_path, fallback=source_gray)
    raw_gray = _load_optional_gray(raw_input_path, fallback=masked_image)
    normalized_gray = _load_optional_gray(preprocessed_normalized_path, fallback=raw_gray)
    pose_normalized_gray = _load_optional_gray(pose_normalized_path, fallback=masked_image)
    pose_normalized_mask = _load_optional_gray(pose_mask_path, fallback=source_mask_gray)
    preprocessed_gray = _load_optional_gray(preprocessed_input_path, fallback=masked_image)
    final_mask = _load_optional_gray(
        preprocess_mask_path,
        fallback=_load_optional_gray(mask_path, fallback=np.where(masked_image > 0, 255, 0).astype(np.uint8)),
    )

    if orientation_path.exists():
        orientation = np.load(orientation_path).astype(np.float32)
    else:
        orientation = np.zeros_like(preprocessed_gray, dtype=np.float32)
    if ridge_period_path.exists():
        ridge_period = np.load(ridge_period_path).astype(np.float32)
    else:
        ridge_period = np.zeros_like(preprocessed_gray, dtype=np.float32)
    if gradient_vis_path.exists():
        visualization_gradient = np.load(gradient_vis_path).astype(np.float32)
    else:
        visualization_gradient = np.zeros((*preprocessed_gray.shape, 2), dtype=np.float32)

    raw_view_paths = [str(path) for path in meta.get("raw_view_paths", [])]
    variant_paths = dict(meta.get("variant_paths", {}))

    sample = gt.RawViewSample(
        sample_id=str(meta["sample_id"]),
        subject_id=int(meta["subject_id"]),
        subject_index=int(meta.get("subject_index", 0)),
        finger_id=int(meta["finger_id"]),
        acquisition_id=int(meta["acquisition_id"]),
        finger_class_id=int(meta["finger_class_id"]),
        raw_image_path=str(meta.get("raw_image_path", raw_input_path)),
        raw_view_index=raw_view_index,
        sire_path=meta.get("sire_path"),
        raw_view_paths=raw_view_paths,
        variant_paths=variant_paths,
        is_extra_acquisition=bool(meta.get("is_extra_acquisition", False)),
    )
    preprocessed = gt.PreprocessedContactlessImage(
        raw_gray=raw_gray,
        normalized_gray=normalized_gray,
        pose_normalized_gray=pose_normalized_gray,
        pose_normalized_mask=pose_normalized_mask,
        preprocessed_gray=preprocessed_gray,
        final_mask=final_mask,
        mask_source=str(meta.get("preprocessing", {}).get("masking", "bundle"))
        if isinstance(meta.get("preprocessing"), dict)
        else "bundle",
        pose_rotation_degrees=float(meta.get("preprocessing", {}).get("pose_normalization", {}).get("rotation_degrees", 0.0))
        if isinstance(meta.get("preprocessing"), dict)
        else 0.0,
        ridge_scale_factor=float(
            meta.get("preprocessing", {}).get("ridge_frequency_normalization", {}).get("scale_factor", 1.0)
        )
        if isinstance(meta.get("preprocessing"), dict)
        else 1.0,
    )
    reconstruction = _build_reconstruction_dataclass(reconstruction_dir, reconstruction_meta)
    prepared = gt.PreparedBundleArtifacts(
        sample=sample,
        bundle_dir=sample_dir,
        image_path=Path(sample.raw_image_path),
        preprocessed=preprocessed,
        gray_image=masked_image,
        mask=final_mask,
        orientation=orientation,
        ridge_period=ridge_period,
        visualization_gradient=visualization_gradient,
        reconstruction_gradient=None,
        masked_image=masked_image,
        enhanced_image=masked_image,
        visualize=False,
        reconstruction=reconstruction,
    )
    return prepared, label_frame, source_gray, source_mask_gray


def _project_candidates(
    prepared: gt.PreparedBundleArtifacts,
    accepted_candidates: list[dict[str, Any]],
    label_frame: dict[str, Any],
    source_mask: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reconstruction = prepared.reconstruction
    if reconstruction is None:
        return [], {"status": "missing_reconstruction"}

    role = pm.role_for_raw_view_index(prepared.sample.raw_view_index)
    if role is None:
        return [], {"status": "unsupported_raw_view_index"}

    if role == "front":
        unwarp_maps = gt._load_npz_arrays(Path(reconstruction.center_unwarp_maps_path))
        reconstruction_maps = gt._load_npz_arrays(Path(reconstruction.reconstruction_maps_path))
        reprojected, details = gt._remap_unwarped_minutiae_to_sample(
            accepted_candidates,
            unwarp_maps,
            reconstruction_maps,
            prepared.sample,
            prepared.preprocessed,
        )
    else:
        side_maps = gt._load_npz_arrays(Path(label_frame["maps_path"]))
        side_unwrapped_mask = np.asarray(source_mask > 0, dtype=np.uint8)
        reprojected, details = gt._remap_side_v4_unwrapped_minutiae_to_sample(
            accepted_candidates,
            side_maps,
            side_unwrapped_mask,
            prepared.sample,
            prepared.preprocessed,
        )
    return reprojected, details


def _suppress_projected_candidates(
    projected: list[dict[str, Any]],
    *,
    suppression_radius: int,
) -> list[dict[str, Any]]:
    if not projected:
        return []
    radius_sq = float(suppression_radius) * float(suppression_radius)
    kept: list[dict[str, Any]] = []
    for candidate in sorted(projected, key=lambda item: (-float(item.get("score", 0.0)), float(item["y"]), float(item["x"]))):
        if any((float(candidate["x"]) - float(existing["x"])) ** 2 + (float(candidate["y"]) - float(existing["y"])) ** 2 <= radius_sq for existing in kept):
            continue
        kept.append(candidate)
    return kept


def _build_row_for_sample(
    sample_dir: Path,
    root: Path,
    *,
    polarity: str,
    border_margin: int,
    min_branch_length: int,
    source_suppression_radius: int,
    projection_suppression_radius: int,
) -> dict[str, Any]:
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing metadata: {meta_path}")
    meta = _read_json(meta_path)
    prepared, label_frame, source_gray, source_mask = _build_prepared_bundle(sample_dir, meta)

    accepted_candidates, extraction_details = pm.extract_plausible_minutiae(
        source_gray,
        source_mask,
        polarity=polarity,
        border_margin=border_margin,
        min_branch_length=min_branch_length,
        suppression_radius=source_suppression_radius,
        include_endings=True,
        include_bifurcations=True,
    )
    reprojected, remap_details = _project_candidates(prepared, accepted_candidates, label_frame, source_mask)
    reprojected = _suppress_projected_candidates(reprojected, suppression_radius=projection_suppression_radius)

    output_shape = _coerce_hw_shape(meta.get("shapes", {}).get("featurenet_output")) if isinstance(meta.get("shapes"), dict) else None
    if output_shape is None:
        output_shape = tuple(int(v) for v in gt._compute_output_shape(*prepared.gray_image.shape))

    plausible_mask = np.zeros((1, int(output_shape[0]), int(output_shape[1])), dtype=np.float32)
    for minutia in reprojected:
        x = float(minutia.get("x", float("nan")))
        y = float(minutia.get("y", float("nan")))
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        cell_x = int(np.clip(math.floor(x), 0, plausible_mask.shape[2] - 1))
        cell_y = int(np.clip(math.floor(y), 0, plausible_mask.shape[1] - 1))
        plausible_mask[0, cell_y, cell_x] = 1.0

    return {
        "sample_id": str(meta["sample_id"]),
        "view_role": pm.role_for_raw_view_index(int(meta.get("raw_view_index", -1))),
        "source_frame": str(label_frame.get("label_frame")),
        "source_image_path": _relative_path(root, Path(label_frame["image_path"])),
        "source_mask_path": _relative_path(root, Path(label_frame["mask_path"])),
        "output_shape": [int(output_shape[0]), int(output_shape[1])],
        "mask_rle": pm.encode_binary_mask_rle(plausible_mask),
        "plausible_pixel_count": int(np.count_nonzero(plausible_mask)),
    }


def _build_zero_row(sample_dir: Path, root: Path, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    sample_id = str((meta or {}).get("sample_id", sample_dir.name))
    raw_view_index = int((meta or {}).get("raw_view_index", -1))
    view_role = pm.role_for_raw_view_index(raw_view_index)
    reconstruction_dir = None
    source_frame = None
    source_image_path = None
    source_mask_path = None
    if isinstance(meta, dict):
        reconstruction_info = meta.get("multiview_reconstruction")
        if isinstance(reconstruction_info, dict) and reconstruction_info.get("reconstruction_dir"):
            reconstruction_dir = Path(str(reconstruction_info["reconstruction_dir"])).expanduser().resolve()
            if view_role is not None:
                label_frame = pm.label_frame_for_role(reconstruction_dir, view_role)
                source_frame = str(label_frame.get("label_frame"))
                source_image_path = _relative_path(root, Path(label_frame["image_path"]))
                source_mask_path = _relative_path(root, Path(label_frame["mask_path"]))
    return {
        "sample_id": sample_id,
        "view_role": view_role,
        "source_frame": source_frame,
        "source_image_path": source_image_path,
        "source_mask_path": source_mask_path,
        "output_shape": [0, 0],
        "mask_rle": [],
        "plausible_pixel_count": 0,
    }


def _process_sample(
    sample_dir: Path,
    root: Path,
    *,
    polarity: str,
    border_margin: int,
    min_branch_length: int,
    source_suppression_radius: int,
    projection_suppression_radius: int,
) -> dict[str, Any]:
    meta: dict[str, Any] | None = None
    try:
        meta_path = sample_dir / "meta.json"
        if meta_path.exists():
            meta = _read_json(meta_path)
        return _build_row_for_sample(
            sample_dir,
            root,
            polarity=polarity,
            border_margin=border_margin,
            min_branch_length=min_branch_length,
            source_suppression_radius=source_suppression_radius,
            projection_suppression_radius=projection_suppression_radius,
        )
    except Exception as exc:
        print(f"[plausible_minutiae] sample {sample_dir.name} failed: {exc}", file=sys.stderr, flush=True)
        return _build_zero_row(sample_dir, root, meta=meta)


def _chunk_entries(entries: list[dict[str, Any]], chunk_count: int) -> list[list[dict[str, Any]]]:
    if chunk_count <= 1 or len(entries) <= 1:
        return [entries]
    chunk_count = min(chunk_count, len(entries))
    chunk_size = math.ceil(len(entries) / chunk_count)
    return [entries[index : index + chunk_size] for index in range(0, len(entries), chunk_size)]


def _resolve_worker_count(requested: int | None, sample_count: int) -> int:
    if sample_count <= 0:
        return 0
    if requested is not None:
        return max(1, min(int(requested), sample_count))
    cpu_count = os.cpu_count() or 1
    return max(1, min(int(cpu_count), sample_count))


def _worker_initializer() -> None:
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass


def _write_shard(
    shard_path: Path,
    chunk: list[dict[str, Any]],
    root: Path,
    *,
    polarity: str,
    border_margin: int,
    min_branch_length: int,
    source_suppression_radius: int,
    projection_suppression_radius: int,
) -> dict[str, Any]:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with shard_path.open("w", encoding="utf-8") as handle:
        for entry in chunk:
            sample_dir = root / "samples" / str(entry["sample_id"])
            row = _process_sample(
                sample_dir,
                root,
                polarity=polarity,
                border_margin=border_margin,
                min_branch_length=min_branch_length,
                source_suppression_radius=source_suppression_radius,
                projection_suppression_radius=projection_suppression_radius,
            )
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            rows_written += 1
    return {"shard_path": str(shard_path), "rows_written": rows_written}


def _merge_shards(shard_paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as destination:
        for shard_path in shard_paths:
            with shard_path.open("r", encoding="utf-8") as source:
                for line in source:
                    destination.write(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-root", type=Path, default=DEFAULT_GROUND_TRUTH_ROOT, help="Path to the DS123 merged ground-truth root.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Where to write the JSONL sidecar.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke testing.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override the automatic CPU worker count.")
    parser.add_argument("--polarity", type=str, default="dark", choices=("dark", "light", "auto"), help="Crossing-number binarization polarity.")
    parser.add_argument("--border-margin", type=int, default=10, help="Minimum distance from the ROI boundary before accepting a candidate.")
    parser.add_argument("--min-branch-length", type=int, default=1, help="Minimum traced branch length for an accepted candidate.")
    parser.add_argument("--source-suppression-radius", type=int, default=8, help="Suppression radius in the source frame.")
    parser.add_argument("--projection-suppression-radius", type=int, default=1, help="Suppression radius after projection to the FeatureNet grid.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.ground_truth_root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, list):
        raise ValueError(f"expected manifest.json to contain a list, got {type(manifest).__name__}")

    entries = list(manifest)
    if args.limit is not None:
        entries = entries[: max(0, int(args.limit))]
    if not entries:
        raise ValueError("no samples available to process")

    worker_count = _resolve_worker_count(args.num_workers, len(entries))
    chunks = _chunk_entries(entries, worker_count)
    if worker_count <= 1:
        with tempfile.TemporaryDirectory(prefix="plausible_minutiae_shards_", dir=str(root)) as shard_root:
            shard_root_path = Path(shard_root)
            shard_paths: list[Path] = []
            for index, chunk in enumerate(chunks):
                shard_path = shard_root_path / f"shard_{index:03d}.jsonl"
                _write_shard(
                    shard_path,
                    chunk,
                    root,
                    polarity=args.polarity,
                    border_margin=int(args.border_margin),
                    min_branch_length=int(args.min_branch_length),
                    source_suppression_radius=int(args.source_suppression_radius),
                    projection_suppression_radius=int(args.projection_suppression_radius),
                )
                shard_paths.append(shard_path)
            _merge_shards(shard_paths, args.output_path.resolve())
        total_rows = len(entries)
    else:
        ctx = mp.get_context("spawn")
        with tempfile.TemporaryDirectory(prefix="plausible_minutiae_shards_", dir=str(root)) as shard_root:
            shard_root_path = Path(shard_root)
            shard_paths = [shard_root_path / f"shard_{index:03d}.jsonl" for index in range(len(chunks))]
            futures = []
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=ctx,
                initializer=_worker_initializer,
            ) as executor:
                for index, chunk in enumerate(chunks):
                    futures.append(
                        executor.submit(
                            _write_shard,
                            shard_paths[index],
                            chunk,
                            root,
                            polarity=args.polarity,
                            border_margin=int(args.border_margin),
                            min_branch_length=int(args.min_branch_length),
                            source_suppression_radius=int(args.source_suppression_radius),
                            projection_suppression_radius=int(args.projection_suppression_radius),
                        )
                    )
                for future in as_completed(futures):
                    future.result()
            _merge_shards(shard_paths, args.output_path.resolve())
        total_rows = len(entries)

    print(
        json.dumps(
            {
                "ground_truth_root": str(root),
                "output_path": str(args.output_path.resolve()),
                "samples": total_rows,
                "num_workers": worker_count,
                "limit": args.limit,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

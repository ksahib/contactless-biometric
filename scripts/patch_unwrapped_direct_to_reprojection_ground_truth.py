#!/usr/bin/env python
"""Patch unwrapped-direct reconstruction GT into current reprojection GT.

This surgical patcher is for generated ground-truth roots whose reconstruction
samples were produced with ``minutiae_ground_truth.mode ==
"reconstruction_unwrapped_direct"``. It reuses each sample's existing
unwrapped-frame ``minutiae.json`` and rewrites only label artifacts/metadata
needed to match the current reconstruction-backed reprojection scheme.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

# Avoid repo-root copy.py shadowing stdlib copy while importing third-party modules.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT]

import cv2  # noqa: E402
import numpy as np  # noqa: E402


ROLE_BY_RAW_VIEW = {0: "front", 1: "left", 2: "right"}
PATCH_SOURCE = "unwrapped_direct_to_reprojection_patch"

_GT_MODULE: Any | None = None


@dataclass(slots=True)
class PatchCandidate:
    manifest_index: int
    sample_id: str
    raw_view_index: int
    role: str
    sample_dir: Path
    reconstruction_dir: Path
    manifest_row: dict[str, Any]


def _ensure_repo_on_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _gt() -> Any:
    global _GT_MODULE
    if _GT_MODULE is None:
        _ensure_repo_on_path()
        _GT_MODULE = __import__("generate_ground_truth")
    return _GT_MODULE


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"unable to load image: {path}")
    return image


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _role_from_index(raw_view_index: int) -> str | None:
    return ROLE_BY_RAW_VIEW.get(int(raw_view_index))


def _resolve_reconstruction_dir(dataset_root: Path, meta: dict[str, Any]) -> Path | None:
    reconstruction = meta.get("multiview_reconstruction")
    if not isinstance(reconstruction, dict):
        return None
    acquisition_id = str(reconstruction.get("acquisition_id") or "")
    if acquisition_id:
        candidate = dataset_root / "reconstructions" / acquisition_id
        if candidate.exists():
            return candidate
    raw_value = reconstruction.get("reconstruction_dir")
    if raw_value:
        path = Path(str(raw_value))
        if path.exists():
            return path
    return None


def _artifact_path(
    dataset_root: Path,
    meta: dict[str, Any],
    reconstruction_dir: Path,
    role: str,
    key: str,
) -> Path:
    minutiae_gt = meta.get("minutiae_ground_truth") if isinstance(meta.get("minutiae_ground_truth"), dict) else {}
    value = minutiae_gt.get(key)
    if value:
        path = Path(str(value))
        if path.exists():
            return path
        if not path.is_absolute() and (dataset_root / path).exists():
            return dataset_root / path
    if role == "front":
        defaults = {
            "label_image_path": reconstruction_dir / "center_unwarped.png",
            "label_mask_path": reconstruction_dir / "center_unwarped_mask.png",
            "label_unwarp_maps_path": reconstruction_dir / "center_unwarp_maps.npz",
            "label_depth_path": reconstruction_dir / "depth_front.npy",
        }
    else:
        side_dir = reconstruction_dir / "side_depth_unwrap_v4" / role
        defaults = {
            "label_image_path": side_dir / f"{role}_depth_unwrapped.png",
            "label_mask_path": side_dir / f"{role}_depth_unwrapped_mask.png",
            "label_unwarp_maps_path": side_dir / f"{role}_depth_unwarp_maps.npz",
            "label_depth_path": side_dir / f"{role}_depth.npy",
            "side_unwrap_v4_report_path": reconstruction_dir / "side_depth_unwrap_v4" / "algorithm1_side_depth_unwrap_report.json",
        }
    path = defaults[key]
    if not path.exists():
        raise FileNotFoundError(f"missing {key} for {role}: {path}")
    return path


def _load_preprocessed(sample_dir: Path, meta: dict[str, Any]) -> Any:
    gt = _gt()
    preprocessing = meta.get("preprocessing") if isinstance(meta.get("preprocessing"), dict) else {}
    pose = preprocessing.get("pose_normalization") if isinstance(preprocessing.get("pose_normalization"), dict) else {}
    ridge = preprocessing.get("ridge_frequency_normalization") if isinstance(preprocessing.get("ridge_frequency_normalization"), dict) else {}
    raw_gray = _load_gray(sample_dir / "raw_input.png") if (sample_dir / "raw_input.png").exists() else _load_gray(sample_dir / "preprocessed_input.png")
    normalized = (
        _load_gray(sample_dir / "preprocess_normalized.png")
        if (sample_dir / "preprocess_normalized.png").exists()
        else raw_gray
    )
    pose_gray = _load_gray(sample_dir / "preprocess_pose_normalized.png")
    pose_mask = np.where(_load_gray(sample_dir / "preprocess_pose_mask.png") > 0, 255, 0).astype(np.uint8)
    preprocessed_gray = _load_gray(sample_dir / "preprocessed_input.png")
    final_mask = np.where(_load_gray(sample_dir / "mask.png") > 0, 255, 0).astype(np.uint8)
    return gt.PreprocessedContactlessImage(
        raw_gray=raw_gray,
        normalized_gray=normalized,
        pose_normalized_gray=pose_gray,
        pose_normalized_mask=pose_mask,
        preprocessed_gray=preprocessed_gray,
        final_mask=final_mask,
        mask_source=str(preprocessing.get("masking", "patched_existing_mask")),
        pose_rotation_degrees=float(pose.get("rotation_degrees", 0.0)),
        ridge_scale_factor=float(ridge.get("scale_factor", 1.0)),
    )


def _build_sample(row: dict[str, Any]) -> Any:
    return _gt().RawViewSample(**row)


def _compute_dense_labels(gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gt = _gt()
    mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    orientation = gt.pyfing.orientation_field_estimation(gray, mask_u8, dpi=gt.DEFAULT_DPI, method="SNFOE")
    orientation = gt._normalize_angle_pi(orientation.astype(np.float32))
    orientation[mask_u8 <= 0] = 0.0
    ridge_period = gt.pyfing.frequency_estimation(gray, orientation, mask_u8, dpi=gt.DEFAULT_DPI, method="SNFFE")
    ridge_period = np.nan_to_num(ridge_period.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ridge_period = np.clip(ridge_period, 0.0, None)
    ridge_period[mask_u8 <= 0] = 0.0
    return orientation.astype(np.float32), ridge_period.astype(np.float32)


def _to_hw2_gradient(gradient_chw: np.ndarray) -> np.ndarray:
    if gradient_chw.ndim != 3 or gradient_chw.shape[0] != 2:
        raise ValueError(f"expected gradient shape (2,H,W), got {gradient_chw.shape}")
    gradient_hw2 = np.transpose(gradient_chw.astype(np.float32), (1, 2, 0))
    return np.nan_to_num(gradient_hw2, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _load_target_gradient(reconstruction_dir: Path, role: str) -> np.ndarray:
    if role in {"left", "right"}:
        v4_path = reconstruction_dir / "side_depth_unwrap_v4" / role / f"{role}_gradient.npy"
        if v4_path.exists():
            gradient = np.load(v4_path).astype(np.float32)
            if gradient.ndim != 3 or gradient.shape[2] != 2:
                raise ValueError(f"expected v4 gradient shape (H,W,2), got {gradient.shape}")
            return np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    key = {"front": "gradient_front", "left": "gradient_left", "right": "gradient_right"}[role]
    arrays = _load_npz_arrays(reconstruction_dir / "depth_gradient_labels.npz")
    return _to_hw2_gradient(arrays[key])


def _patch_reconstruction_gradient_cache(reconstruction_dir: Path, dry_run: bool) -> dict[str, Any]:
    path = reconstruction_dir / "depth_gradient_labels.npz"
    if not path.exists():
        return {"path": str(path), "status": "missing_depth_gradient_labels"}
    arrays = _load_npz_arrays(path)
    updates: dict[str, np.ndarray] = {}
    roles: dict[str, Any] = {}
    for role in ("left", "right"):
        v4_path = reconstruction_dir / "side_depth_unwrap_v4" / role / f"{role}_gradient.npy"
        if not v4_path.exists():
            roles[role] = {"status": "missing_v4_gradient", "path": str(v4_path)}
            continue
        gradient = np.load(v4_path).astype(np.float32)
        if gradient.ndim != 3 or gradient.shape[2] != 2:
            raise ValueError(f"expected {v4_path} shape (H,W,2), got {gradient.shape}")
        label = np.transpose(np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0), (2, 0, 1)).astype(np.float32)
        updates[f"gradient_{role}"] = label
        roles[role] = {"status": "would_patch" if dry_run else "patched", "source": str(v4_path), "shape": list(label.shape)}
    if updates and not dry_run:
        _save_npz(path, {**arrays, **updates})
    return {"path": str(path), "status": "would_patch" if dry_run else "patched", "roles": roles}


def _build_front_metadata(
    meta: dict[str, Any],
    reconstruction_dir: Path,
    source_count: int,
    reprojected_count: int,
    rasterized_count: int,
    remap_details: dict[str, Any],
) -> dict[str, Any]:
    old = meta.get("minutiae_ground_truth") if isinstance(meta.get("minutiae_ground_truth"), dict) else {}
    return {
        "mode": "reconstruction_backed",
        "patch_source": PATCH_SOURCE,
        "canonical_source": str(old.get("label_minutiae_extractor", "existing_unwrapped_minutiae")),
        "view_role": "front",
        "canonical_minutiae_count": int(source_count),
        "reprojected_minutiae_count": int(reprojected_count),
        "rasterized_minutiae_count": int(rasterized_count),
        "center_unwarped_image_path": str((reconstruction_dir / "center_unwarped.png").resolve()),
        "center_unwarped_mask_path": str((reconstruction_dir / "center_unwarped_mask.png").resolve()),
        "center_unwarp_maps_path": str((reconstruction_dir / "center_unwarp_maps.npz").resolve()),
        "reconstruction_maps_path": str((reconstruction_dir / "reconstruction_maps.npz").resolve()),
        **remap_details,
    }


def _build_side_metadata(
    meta: dict[str, Any],
    reconstruction_dir: Path,
    role: str,
    source_count: int,
    reprojected_count: int,
    rasterized_count: int,
    remap_details: dict[str, Any],
) -> dict[str, Any]:
    old = meta.get("minutiae_ground_truth") if isinstance(meta.get("minutiae_ground_truth"), dict) else {}
    side_dir = reconstruction_dir / "side_depth_unwrap_v4" / role
    return {
        "mode": "reconstruction_backed",
        "patch_source": PATCH_SOURCE,
        "side_reprojection_source": "algorithm1_v4_side_unwrapped",
        "canonical_source": str(old.get("label_minutiae_extractor", "existing_unwrapped_minutiae")),
        "view_role": role,
        "side_unwrapped_minutiae_count": int(source_count),
        "reprojected_minutiae_count": int(reprojected_count),
        "rasterized_minutiae_count": int(rasterized_count),
        "side_unwrapped_image_path": str((side_dir / f"{role}_depth_unwrapped.png").resolve()),
        "side_unwrapped_mask_path": str((side_dir / f"{role}_depth_unwrapped_mask.png").resolve()),
        "side_unwarp_maps_path": str((side_dir / f"{role}_depth_unwarp_maps.npz").resolve()),
        "side_depth_path": str((side_dir / f"{role}_depth.npy").resolve()),
        "side_gradient_path": str((side_dir / f"{role}_gradient.npy").resolve()),
        "side_unwrap_v4_report_path": str((reconstruction_dir / "side_depth_unwrap_v4" / "algorithm1_side_depth_unwrap_report.json").resolve()),
        "side_unwrap_v4_parameters": _gt()._side_unwrap_v4_parameters(),
        **remap_details,
    }


def _update_meta_common(
    meta: dict[str, Any],
    role: str,
    reconstruction_dir: Path,
    new_gt: dict[str, Any],
    patch_record: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(meta)
    updated["minutiae_ground_truth"] = new_gt
    updated.setdefault("patches", []).append(patch_record)
    methods = dict(updated.get("methods") or {})
    methods["orientation"] = "pyfing.orientation_field_estimation(method='SNFOE')"
    methods["ridge_period"] = "pyfing.frequency_estimation(method='SNFFE')"
    methods["gradient"] = "reconstruction depth partial derivatives resized to the FeatureNet output grid; side views use Algorithm-1 v4 side depth gradients when reconstruction-backed"
    methods["minutiae_ground_truth_pipeline"] = "reconstruction_unwarp_reproject"
    updated["methods"] = methods
    counts = dict(updated.get("counts") or {})
    counts["minutiae"] = int(new_gt["reprojected_minutiae_count"])
    counts["minutia_support_pixels"] = int(new_gt["rasterized_minutiae_count"])
    counts["rasterized_minutiae_count"] = int(new_gt["rasterized_minutiae_count"])
    updated["counts"] = counts
    reconstruction = dict(updated.get("multiview_reconstruction") or {})
    reconstruction["role"] = role
    reconstruction["reconstruction_dir"] = str(reconstruction_dir.resolve())
    reconstruction["depth_gradient_labels_path"] = str((reconstruction_dir / "depth_gradient_labels.npz").resolve())
    reconstruction["reconstruction_maps_path"] = str((reconstruction_dir / "reconstruction_maps.npz").resolve())
    reconstruction["center_unwarp_maps_path"] = str((reconstruction_dir / "center_unwarp_maps.npz").resolve())
    reconstruction["center_unwarped_image_path"] = str((reconstruction_dir / "center_unwarped.png").resolve())
    reconstruction["center_unwarped_mask_path"] = str((reconstruction_dir / "center_unwarped_mask.png").resolve())
    reconstruction["side_depth_unwrap_v4_dir"] = str((reconstruction_dir / "side_depth_unwrap_v4").resolve())
    reconstruction["side_depth_unwrap_v4_report_path"] = str((reconstruction_dir / "side_depth_unwrap_v4" / "algorithm1_side_depth_unwrap_report.json").resolve())
    updated["multiview_reconstruction"] = reconstruction
    return updated


def _finite_target_failures(targets: dict[str, np.ndarray]) -> list[str]:
    bad: list[str] = []
    for key, value in targets.items():
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            bad.append(key)
    return bad


def _patch_candidate_worker(payload: dict[str, Any]) -> dict[str, Any]:
    dataset_root = Path(payload["dataset_root"])
    sample_dir = Path(payload["sample_dir"])
    reconstruction_dir = Path(payload["reconstruction_dir"])
    row = payload["manifest_row"]
    sample_id = str(payload["sample_id"])
    role = str(payload["role"])
    dry_run = bool(payload["dry_run"])

    meta_path = sample_dir / "meta.json"
    meta = _read_json(meta_path)
    gt_old = meta.get("minutiae_ground_truth") if isinstance(meta.get("minutiae_ground_truth"), dict) else {}
    sample = _build_sample(row)
    preprocessed = _load_preprocessed(sample_dir, meta)
    gray = preprocessed.preprocessed_gray
    mask = preprocessed.final_mask
    source_minutiae = _read_json(sample_dir / "minutiae.json")

    if role == "front":
        unwarp_maps = _gt()._load_npz_arrays(_artifact_path(dataset_root, meta, reconstruction_dir, role, "label_unwarp_maps_path"))
        reconstruction_maps = _gt()._load_npz_arrays(reconstruction_dir / "reconstruction_maps.npz")
        reprojected, remap_details = _gt()._remap_unwarped_minutiae_to_sample(
            source_minutiae,
            unwarp_maps,
            reconstruction_maps,
            sample,
            preprocessed,
        )
    else:
        maps_path = _artifact_path(dataset_root, meta, reconstruction_dir, role, "label_unwarp_maps_path")
        mask_path = _artifact_path(dataset_root, meta, reconstruction_dir, role, "label_mask_path")
        side_maps = _gt()._load_npz_arrays(maps_path)
        side_mask = _load_gray(mask_path) > 0
        reprojected, remap_details = _gt()._remap_side_v4_unwrapped_minutiae_to_sample(
            source_minutiae,
            side_maps,
            side_mask,
            sample,
            preprocessed,
        )

    orientation, ridge_period = _compute_dense_labels(gray, mask)
    gradient = _load_target_gradient(reconstruction_dir, role)
    targets = _gt()._build_featurenet_targets(
        gray_image=gray,
        mask=mask,
        orientation=orientation,
        ridge_period=ridge_period,
        gradient=gradient,
        minutiae=reprojected,
    )
    bad = _finite_target_failures(targets)
    if bad:
        raise ValueError(f"non-finite target arrays for {sample_id}: {bad}")
    rasterized_count = int(np.count_nonzero(targets["minutia_valid_mask"]))
    if rasterized_count == 0:
        return {
            "sample_id": sample_id,
            "role": role,
            "status": "zero_after_patch",
            "source_minutiae_count": len(source_minutiae),
            "reprojected_minutiae_count": len(reprojected),
            **remap_details,
        }

    if role == "front":
        minutiae_gt = _build_front_metadata(
            meta,
            reconstruction_dir,
            len(source_minutiae),
            len(reprojected),
            rasterized_count,
            remap_details,
        )
    else:
        minutiae_gt = _build_side_metadata(
            meta,
            reconstruction_dir,
            role,
            len(source_minutiae),
            len(reprojected),
            rasterized_count,
            remap_details,
        )
    patch_record = {
        "patch_source": PATCH_SOURCE,
        "converted_from_unwrapped_direct": True,
        "view_role": role,
        "source_unwrapped_minutiae_count": int(len(source_minutiae)),
        "post_patch_minutiae_count": int(len(reprojected)),
        "post_patch_rasterized_minutiae_count": int(rasterized_count),
        "orientation_shape": list(orientation.shape),
        "ridge_period_shape": list(ridge_period.shape),
        "target_output_shape": list(targets["output_mask"].shape[-2:]),
    }
    updated_meta = _update_meta_common(meta, role, reconstruction_dir, minutiae_gt, patch_record)

    report = {
        "sample_id": sample_id,
        "role": role,
        "status": "would_patch" if dry_run else "patched",
        "previous_mode": gt_old.get("mode"),
        "source_minutiae_count": int(len(source_minutiae)),
        "reprojected_minutiae_count": int(len(reprojected)),
        "rasterized_minutiae_count": int(rasterized_count),
        "orientation_shape": list(orientation.shape),
        "ridge_period_shape": list(ridge_period.shape),
        "target_gradient_shape": list(targets.get("gradient", np.empty((0,))).shape),
        **remap_details,
    }
    if dry_run:
        return report

    np.save(sample_dir / "orientation.npy", orientation.astype(np.float32))
    np.save(sample_dir / "ridge_period.npy", ridge_period.astype(np.float32))
    _write_json(sample_dir / "minutiae.json", reprojected)
    _save_npz(sample_dir / "featurenet_targets.npz", targets)
    _write_json(meta_path, updated_meta)
    return report


def _build_candidates(dataset_root: Path, manifest: list[dict[str, Any]], limit: int | None) -> tuple[list[PatchCandidate], list[dict[str, Any]]]:
    candidates: list[PatchCandidate] = []
    skipped: list[dict[str, Any]] = []
    for index, row in enumerate(manifest):
        sample_id = str(row.get("sample_id"))
        role = _role_from_index(int(row.get("raw_view_index", -1)))
        if role is None:
            skipped.append({"sample_id": sample_id, "status": "skipped_not_reconstruction_view"})
            continue
        sample_dir = dataset_root / "samples" / sample_id
        required = [
            sample_dir / "meta.json",
            sample_dir / "minutiae.json",
            sample_dir / "featurenet_targets.npz",
            sample_dir / "preprocessed_input.png",
            sample_dir / "mask.png",
            sample_dir / "preprocess_pose_normalized.png",
            sample_dir / "preprocess_pose_mask.png",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            skipped.append({"sample_id": sample_id, "status": "skipped_missing_files", "missing": missing})
            continue
        meta = _read_json(sample_dir / "meta.json")
        gt_info = meta.get("minutiae_ground_truth") if isinstance(meta.get("minutiae_ground_truth"), dict) else {}
        if gt_info.get("mode") != "reconstruction_unwrapped_direct":
            skipped.append({"sample_id": sample_id, "status": "skipped_mode", "mode": gt_info.get("mode")})
            continue
        reconstruction_dir = _resolve_reconstruction_dir(dataset_root, meta)
        if reconstruction_dir is None:
            skipped.append({"sample_id": sample_id, "status": "skipped_missing_reconstruction"})
            continue
        candidates.append(
            PatchCandidate(
                manifest_index=index,
                sample_id=sample_id,
                raw_view_index=int(row.get("raw_view_index", -1)),
                role=role,
                sample_dir=sample_dir,
                reconstruction_dir=reconstruction_dir,
                manifest_row=row,
            )
        )
        if limit is not None and len(candidates) >= int(limit):
            break
    return candidates, skipped


def _run_sample_stage(candidates: list[PatchCandidate], dataset_root: Path, dry_run: bool, num_workers: int) -> list[dict[str, Any]]:
    payloads = [
        {
            "dataset_root": str(dataset_root),
            "sample_id": candidate.sample_id,
            "role": candidate.role,
            "sample_dir": str(candidate.sample_dir),
            "reconstruction_dir": str(candidate.reconstruction_dir),
            "manifest_row": candidate.manifest_row,
            "dry_run": bool(dry_run),
        }
        for candidate in candidates
    ]
    if not payloads:
        return []
    if num_workers <= 1:
        return [_patch_candidate_worker(payload) for payload in payloads]
    ctx = mp.get_context("spawn")
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = [executor.submit(_patch_candidate_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: str(item.get("sample_id", "")))
    return results


def patch_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"missing dataset root: {dataset_root}")
    if not args.dry_run:
        if not args.in_place:
            raise ValueError("--in-place is required for mutation; use --dry-run for inspection")
        if not args.no_backup:
            raise ValueError("--no-backup is required for this patcher because backups are intentionally disabled")

    manifest = _read_json(dataset_root / "manifest.json")
    candidates, skipped = _build_candidates(dataset_root, manifest, args.limit)
    reconstruction_dirs = sorted({candidate.reconstruction_dir for candidate in candidates}, key=lambda path: str(path))
    reconstruction_reports = [
        _patch_reconstruction_gradient_cache(reconstruction_dir, dry_run=bool(args.dry_run))
        for reconstruction_dir in reconstruction_dirs
    ]
    sample_reports = _run_sample_stage(
        candidates,
        dataset_root,
        dry_run=bool(args.dry_run),
        num_workers=max(1, int(args.num_workers)),
    )
    patched = sum(1 for item in sample_reports if item.get("status") == "patched")
    would_patch = sum(1 for item in sample_reports if item.get("status") == "would_patch")
    zero_after_patch = sum(1 for item in sample_reports if item.get("status") == "zero_after_patch")
    summary = {
        "patch_source": PATCH_SOURCE,
        "dataset_root": str(dataset_root),
        "dry_run": bool(args.dry_run),
        "in_place": bool(args.in_place),
        "no_backup": bool(args.no_backup),
        "num_workers": int(args.num_workers),
        "limit": None if args.limit is None else int(args.limit),
        "scanned_manifest_count": int(len(manifest)),
        "candidate_count": int(len(candidates)),
        "patched_count": int(patched),
        "would_patch_count": int(would_patch),
        "zero_after_patch_count": int(zero_after_patch),
        "skipped_count": int(len(skipped)),
        "reconstruction_reports": reconstruction_reports,
        "sample_reports": sample_reports,
        "skipped": skipped,
    }
    summary_path = (
        REPO_ROOT / "tmp" / f"unwrapped_direct_to_reprojection_patch_{dataset_root.name}_dry_run_summary.json"
        if args.dry_run
        else dataset_root / "unwrapped_direct_to_reprojection_patch_summary.json"
    )
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--num-workers", type=int, default=max(1, min(mp.cpu_count(), 4)))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    summary = patch_dataset(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

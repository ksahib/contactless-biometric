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
import os
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

# Avoid repo-root copy.py shadowing stdlib copy while importing third-party modules.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT]

from dataclasses import dataclass  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402


ROLE_BY_RAW_VIEW = {0: "front", 1: "left", 2: "right"}
PATCH_SOURCE = "unwrapped_direct_to_reprojection_patch"
PATCH_SCHEMA_VERSION = 2
DEFAULT_MAX_WORKERS = 24

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
        if os.environ.get("UNWRAPPED_REPROJECTION_PATCH_CPU_WORKER") == "1" and "pyfing" not in sys.modules:
            stub = types.ModuleType("pyfing")

            def _pyfing_unavailable(*_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError("pyfing is intentionally unavailable in CPU patch workers")

            stub.orientation_field_estimation = _pyfing_unavailable  # type: ignore[attr-defined]
            stub.frequency_estimation = _pyfing_unavailable  # type: ignore[attr-defined]
            stub.minutiae_extraction = _pyfing_unavailable  # type: ignore[attr-defined]
            sys.modules["pyfing"] = stub
        _GT_MODULE = __import__("generate_ground_truth")
    return _GT_MODULE


def _cpu_worker_initializer() -> None:
    # Worker processes must not become accidental owners of pyfing/TensorFlow GPU state.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["NVIDIA_VISIBLE_DEVICES"] = ""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
    os.environ["UNWRAPPED_REPROJECTION_PATCH_CPU_WORKER"] = "1"


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


def _safe_cache_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(":", "_")


def _sample_cache_dir(cache_root: Path, sample_id: str) -> Path:
    return cache_root / "samples" / _safe_cache_name(sample_id)


def _reconstruction_cache_dir(cache_root: Path, reconstruction_dir: Path) -> Path:
    return cache_root / "reconstructions" / _safe_cache_name(reconstruction_dir.name)


def _file_signature(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path.resolve()), "exists": False}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _base_patch_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": PATCH_SCHEMA_VERSION,
        "patch_source": PATCH_SOURCE,
        "model_worker_mode": str(args.model_worker_mode),
        "dense_orientation_method": "pyfing.orientation_field_estimation(method='SNFOE')",
        "dense_ridge_period_method": "pyfing.frequency_estimation(method='SNFFE')",
        "minutiae_extraction": "reuse_existing_unwrapped_minutiae_json",
        "fingerflow_extraction_ran": False,
    }


def _marker_matches(marker_path: Path, params: dict[str, Any], allowed_statuses: set[str]) -> bool:
    if not marker_path.exists():
        return False
    try:
        marker = _read_json(marker_path)
    except Exception:
        return False
    return marker.get("status") in allowed_statuses and marker.get("parameters") == params


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


def _dense_cache_parameters(sample_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
    return {
        **params,
        "stage": "dense_labels",
        "inputs": {
            "preprocessed_input": _file_signature(sample_dir / "preprocessed_input.png"),
            "mask": _file_signature(sample_dir / "mask.png"),
        },
    }


def _compute_dense_cache_for_candidate(
    candidate: PatchCandidate,
    cache_root: Path,
    params: dict[str, Any],
) -> dict[str, Any]:
    cache_dir = _sample_cache_dir(cache_root, candidate.sample_id)
    marker_path = cache_dir / "dense_done.json"
    orientation_path = cache_dir / "orientation.npy"
    ridge_period_path = cache_dir / "ridge_period.npy"
    dense_params = _dense_cache_parameters(candidate.sample_dir, params)
    if (
        _marker_matches(marker_path, dense_params, {"done"})
        and orientation_path.exists()
        and ridge_period_path.exists()
    ):
        orientation = np.load(orientation_path, mmap_mode="r")
        ridge_period = np.load(ridge_period_path, mmap_mode="r")
        return {
            "sample_id": candidate.sample_id,
            "role": candidate.role,
            "status": "cache_hit",
            "orientation_path": str(orientation_path),
            "ridge_period_path": str(ridge_period_path),
            "orientation_shape": list(orientation.shape),
            "ridge_period_shape": list(ridge_period.shape),
        }

    meta = _read_json(candidate.sample_dir / "meta.json")
    preprocessed = _load_preprocessed(candidate.sample_dir, meta)
    started_at = time.perf_counter()
    orientation, ridge_period = _compute_dense_labels(preprocessed.preprocessed_gray, preprocessed.final_mask)
    if not np.isfinite(orientation).all() or not np.isfinite(ridge_period).all():
        raise ValueError(f"non-finite dense labels for {candidate.sample_id}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(orientation_path, orientation.astype(np.float32))
    np.save(ridge_period_path, ridge_period.astype(np.float32))
    _write_json(
        marker_path,
        {
            "status": "done",
            "parameters": dense_params,
            "sample_id": candidate.sample_id,
            "role": candidate.role,
            "orientation_path": str(orientation_path),
            "ridge_period_path": str(ridge_period_path),
            "seconds": float(time.perf_counter() - started_at),
            "process_id": int(os.getpid()),
            "model_owner": "parent",
        },
    )
    return {
        "sample_id": candidate.sample_id,
        "role": candidate.role,
        "status": "computed",
        "orientation_path": str(orientation_path),
        "ridge_period_path": str(ridge_period_path),
        "orientation_shape": list(orientation.shape),
        "ridge_period_shape": list(ridge_period.shape),
    }


def _run_dense_label_stage(
    candidates: list[PatchCandidate],
    cache_root: Path,
    params: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        reports[candidate.sample_id] = _compute_dense_cache_for_candidate(candidate, cache_root, params)
    return reports


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


def _gradient_cache_parameters(reconstruction_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
    return {
        **params,
        "stage": "reconstruction_gradient_cache",
        "reconstruction_dir": str(reconstruction_dir.resolve()),
        "inputs": {
            "left_v4_gradient": _file_signature(reconstruction_dir / "side_depth_unwrap_v4" / "left" / "left_gradient.npy"),
            "right_v4_gradient": _file_signature(reconstruction_dir / "side_depth_unwrap_v4" / "right" / "right_gradient.npy"),
        },
    }


def _patch_reconstruction_gradient_cache_worker(payload: dict[str, Any]) -> dict[str, Any]:
    reconstruction_dir = Path(payload["reconstruction_dir"])
    cache_dir = Path(payload["cache_dir"])
    dry_run = bool(payload["dry_run"])
    params = dict(payload["params"])
    marker_path = cache_dir / "gradient_cache_done.json"
    gradient_params = _gradient_cache_parameters(reconstruction_dir, params)
    allowed_statuses = {"done", "dry_run"} if dry_run else {"done"}
    if _marker_matches(marker_path, gradient_params, allowed_statuses):
        marker = _read_json(marker_path)
        cached = dict(marker.get("result", {}))
        cached["status"] = "cache_hit"
        cached["cache_marker"] = str(marker_path)
        return cached

    started_at = time.perf_counter()
    result = _patch_reconstruction_gradient_cache(reconstruction_dir, dry_run=dry_run)
    result["cache_marker"] = str(marker_path)
    result["seconds"] = float(time.perf_counter() - started_at)
    _write_json(
        marker_path,
        {
            "status": "dry_run" if dry_run else "done",
            "parameters": gradient_params,
            "reconstruction_dir": str(reconstruction_dir.resolve()),
            "result": result,
        },
    )
    return result


def _run_reconstruction_gradient_stage(
    reconstruction_dirs: list[Path],
    cache_root: Path,
    dry_run: bool,
    num_workers: int,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    payloads = [
        {
            "reconstruction_dir": str(reconstruction_dir),
            "cache_dir": str(_reconstruction_cache_dir(cache_root, reconstruction_dir)),
            "dry_run": bool(dry_run),
            "params": params,
        }
        for reconstruction_dir in reconstruction_dirs
    ]
    if not payloads:
        return []
    if num_workers <= 1:
        return [_patch_reconstruction_gradient_cache_worker(payload) for payload in payloads]
    ctx = mp.get_context("spawn")
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx, initializer=_cpu_worker_initializer) as executor:
        futures = [executor.submit(_patch_reconstruction_gradient_cache_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: str(item.get("path", "")))
    return results


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
    orientation_path = Path(payload["orientation_path"])
    ridge_period_path = Path(payload["ridge_period_path"])
    sample_marker_path = Path(payload["sample_marker_path"])
    patch_params = dict(payload["params"])

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

    orientation = np.load(orientation_path).astype(np.float32)
    ridge_period = np.load(ridge_period_path).astype(np.float32)
    if orientation.shape != gray.shape or ridge_period.shape != gray.shape:
        raise ValueError(
            f"dense label shape mismatch for {sample_id}: "
            f"gray={gray.shape}, orientation={orientation.shape}, ridge_period={ridge_period.shape}"
        )
    if not np.isfinite(orientation).all() or not np.isfinite(ridge_period).all():
        raise ValueError(f"non-finite cached dense labels for {sample_id}")
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
            "dense_cache_orientation_path": str(orientation_path),
            "dense_cache_ridge_period_path": str(ridge_period_path),
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
        "dense_cache_orientation_path": str(orientation_path),
        "dense_cache_ridge_period_path": str(ridge_period_path),
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
        "dense_cache_orientation_path": str(orientation_path),
        "dense_cache_ridge_period_path": str(ridge_period_path),
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
    _write_json(
        sample_marker_path,
        {
            "status": "done",
            "parameters": patch_params,
            "sample_id": sample_id,
            "role": role,
            "sample_dir": str(sample_dir.resolve()),
            "result": report,
        },
    )
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


def _run_sample_stage(
    candidates: list[PatchCandidate],
    dataset_root: Path,
    dry_run: bool,
    num_workers: int,
    cache_root: Path,
    dense_reports: dict[str, dict[str, Any]],
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    payloads = [
        {
            "dataset_root": str(dataset_root),
            "sample_id": candidate.sample_id,
            "role": candidate.role,
            "sample_dir": str(candidate.sample_dir),
            "reconstruction_dir": str(candidate.reconstruction_dir),
            "manifest_row": candidate.manifest_row,
            "dry_run": bool(dry_run),
            "orientation_path": str(dense_reports[candidate.sample_id]["orientation_path"]),
            "ridge_period_path": str(dense_reports[candidate.sample_id]["ridge_period_path"]),
            "sample_marker_path": str(_sample_cache_dir(cache_root, candidate.sample_id) / "sample_patch_done.json"),
            "params": params,
        }
        for candidate in candidates
        if candidate.sample_id in dense_reports
    ]
    if not payloads:
        return []
    if num_workers <= 1:
        return [_patch_candidate_worker(payload) for payload in payloads]
    ctx = mp.get_context("spawn")
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx, initializer=_cpu_worker_initializer) as executor:
        futures = [executor.submit(_patch_candidate_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: str(item.get("sample_id", "")))
    return results


def patch_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"missing dataset root: {dataset_root}")
    if args.model_worker_mode != "single":
        raise ValueError("only --model-worker-mode single is currently supported")
    if not args.dry_run:
        if not args.in_place:
            raise ValueError("--in-place is required for mutation; use --dry-run for inspection")
        if not args.no_backup:
            raise ValueError("--no-backup is required for this patcher because backups are intentionally disabled")

    params = _base_patch_params(args)
    cache_root = args.cache_root.resolve() if args.cache_root else REPO_ROOT / "tmp" / "unwrapped_direct_to_reprojection_patch_cache" / dataset_root.name
    num_workers = max(1, int(args.num_workers))
    stage_timings: dict[str, float] = {}

    started_at = time.perf_counter()
    manifest = _read_json(dataset_root / "manifest.json")
    candidates, skipped = _build_candidates(dataset_root, manifest, args.limit)
    stage_timings["candidate_scan"] = float(time.perf_counter() - started_at)

    started_at = time.perf_counter()
    reconstruction_dirs = sorted({candidate.reconstruction_dir for candidate in candidates}, key=lambda path: str(path))
    reconstruction_reports = _run_reconstruction_gradient_stage(
        reconstruction_dirs,
        cache_root=cache_root,
        dry_run=bool(args.dry_run),
        num_workers=num_workers,
        params=params,
    )
    stage_timings["reconstruction_gradient_cache"] = float(time.perf_counter() - started_at)

    started_at = time.perf_counter()
    dense_reports = _run_dense_label_stage(candidates, cache_root=cache_root, params=params)
    stage_timings["dense_pyfing_model_owner"] = float(time.perf_counter() - started_at)

    started_at = time.perf_counter()
    sample_reports = _run_sample_stage(
        candidates,
        dataset_root,
        dry_run=bool(args.dry_run),
        num_workers=num_workers,
        cache_root=cache_root,
        dense_reports=dense_reports,
        params=params,
    )
    stage_timings["sample_patch"] = float(time.perf_counter() - started_at)

    patched = sum(1 for item in sample_reports if item.get("status") == "patched")
    would_patch = sum(1 for item in sample_reports if item.get("status") == "would_patch")
    zero_after_patch = sum(1 for item in sample_reports if item.get("status") == "zero_after_patch")
    dense_cache_hits = sum(1 for item in dense_reports.values() if item.get("status") == "cache_hit")
    dense_computed = sum(1 for item in dense_reports.values() if item.get("status") == "computed")
    gradient_cache_hits = sum(1 for item in reconstruction_reports if item.get("status") == "cache_hit")
    role_aggregate = {
        role: {
            "candidate_count": sum(1 for candidate in candidates if candidate.role == role),
            "patched_count": sum(1 for item in sample_reports if item.get("role") == role and item.get("status") == "patched"),
            "would_patch_count": sum(1 for item in sample_reports if item.get("role") == role and item.get("status") == "would_patch"),
            "zero_after_patch_count": sum(1 for item in sample_reports if item.get("role") == role and item.get("status") == "zero_after_patch"),
            "reprojected_minutiae_count": int(sum(int(item.get("reprojected_minutiae_count", 0)) for item in sample_reports if item.get("role") == role)),
            "rasterized_minutiae_count": int(sum(int(item.get("rasterized_minutiae_count", 0)) for item in sample_reports if item.get("role") == role)),
        }
        for role in ("front", "left", "right")
    }
    summary = {
        "patch_source": PATCH_SOURCE,
        "dataset_root": str(dataset_root),
        "dry_run": bool(args.dry_run),
        "in_place": bool(args.in_place),
        "no_backup": bool(args.no_backup),
        "cache_root": str(cache_root),
        "parameters": params,
        "parallel": {
            "num_workers": int(num_workers),
            "model_worker_mode": str(args.model_worker_mode),
            "cpu_parallel_stages": ["reconstruction_gradient_cache", "sample_patch"],
            "single_model_owner_stages": ["dense_pyfing_model_owner"],
        },
        "limit": None if args.limit is None else int(args.limit),
        "scanned_manifest_count": int(len(manifest)),
        "candidate_count": int(len(candidates)),
        "patched_count": int(patched),
        "would_patch_count": int(would_patch),
        "zero_after_patch_count": int(zero_after_patch),
        "skipped_count": int(len(skipped)),
        "cache_stats": {
            "dense_cache_hits": int(dense_cache_hits),
            "dense_computed": int(dense_computed),
            "gradient_cache_hits": int(gradient_cache_hits),
        },
        "stage_timings_seconds": stage_timings,
        "role_aggregate": role_aggregate,
        "minutiae_extraction": {
            "existing_unwrapped_minutiae_reused": True,
            "fingerflow_ran": False,
            "pyfing_minutiae_ran": False,
        },
        "reconstruction_reports": reconstruction_reports,
        "dense_reports": list(dense_reports.values()),
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
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=max(1, min(mp.cpu_count(), DEFAULT_MAX_WORKERS)))
    parser.add_argument("--model-worker-mode", choices=("single",), default="single")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    summary = patch_dataset(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Patch side-view FeatureNet gradient labels with Algorithm-1 v4 side gradients.

This surgical patcher rewrites only side-view gradient targets. It preserves
front samples, minutiae labels, dense orientation, ridge period, masks, images,
and all non-gradient target arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

# Avoid repo-root copy.py shadowing stdlib copy while importing third-party modules.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT]

import cv2  # noqa: E402
import numpy as np  # noqa: E402


SIDE_ROLES = {1: "left", 2: "right"}
DEFAULT_ROW_PARAM_SMOOTH_WINDOW = 31
DEFAULT_MAP_SMOOTH_SIGMA_X = 2.0
DEFAULT_MAP_SMOOTH_SIGMA_Y = 5.0
DEFAULT_UNWRAP_GRADIENT_CLIP = 3.0


@dataclass(slots=True)
class PatchStats:
    scanned: int = 0
    patched_samples: int = 0
    patched_reconstructions: int = 0
    dry_run_would_patch_samples: int = 0
    dry_run_would_patch_reconstructions: int = 0
    skipped_not_side: int = 0
    skipped_missing_files: int = 0
    resumed_already_done: int = 0
    errors: int = 0


@dataclass(slots=True)
class SideSampleRecord:
    manifest_index: int
    sample_id: str
    raw_view_index: int
    role: str
    sample_dir: Path
    reconstruction_dir: Path
    acquisition_id: str


@dataclass(slots=True)
class AcquisitionWorkUnit:
    acquisition_id: str
    reconstruction_dir: Path
    cache_dir: Path
    samples: list[SideSampleRecord] = field(default_factory=list)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_depth_unwrap_module() -> Any:
    module_path = REPO_ROOT / "scripts" / "algorithm1_side_depth_unwrap_fixed_v4.py"
    spec = importlib.util.spec_from_file_location("algorithm1_side_depth_unwrap_fixed_v4", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cpu_worker_initializer() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


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


def _parameter_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "row_param_smooth_window": int(args.row_param_smooth_window),
        "map_smooth_sigma_x": float(args.map_smooth_sigma_x),
        "map_smooth_sigma_y": float(args.map_smooth_sigma_y),
        "left_angle": float(args.left_angle),
        "right_angle": float(args.right_angle),
        "samples_per_pixel": float(args.samples_per_pixel),
        "unwrap_width_scale": float(args.unwrap_width_scale),
        "unwrap_gradient_clip": None if args.unwrap_gradient_clip <= 0 else float(args.unwrap_gradient_clip),
        "reverse_left_unwrap_x": bool(args.reverse_left_unwrap_x),
        "reverse_right_unwrap_x": bool(args.reverse_right_unwrap_x),
    }


def _gradient_stats(gradient: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    array = np.asarray(gradient, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 2:
        mag = np.linalg.norm(np.transpose(array, (1, 2, 0)), axis=2)
    elif array.ndim == 3 and array.shape[-1] == 2:
        mag = np.linalg.norm(array, axis=2)
    else:
        mag = np.abs(array)
    valid = np.isfinite(mag)
    if mask is not None:
        valid &= np.asarray(mask).squeeze() > 0
    values = mag[valid]
    return {
        "shape": list(array.shape),
        "finite": bool(np.isfinite(array).all()),
        "valid_count": int(values.size),
        "min_magnitude": float(values.min()) if values.size else 0.0,
        "mean_magnitude": float(values.mean()) if values.size else 0.0,
        "max_magnitude": float(values.max()) if values.size else 0.0,
    }


def _resize_gradient_to_target(gradient_hw2: np.ndarray, output_mask: np.ndarray) -> np.ndarray:
    if gradient_hw2.ndim != 3 or gradient_hw2.shape[2] != 2:
        raise ValueError(f"expected v4 gradient shape (H,W,2), got {gradient_hw2.shape}")
    gradient_hw2 = np.nan_to_num(gradient_hw2.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.asarray(output_mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"expected output mask shape (1,H,W) or (H,W), got {output_mask.shape}")
    shape = mask.shape
    grad_x = cv2.resize(gradient_hw2[:, :, 0].astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) * mask
    grad_y = cv2.resize(gradient_hw2[:, :, 1].astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) * mask
    patched = np.stack([grad_x, grad_y], axis=0).astype(np.float32)
    if not np.isfinite(patched).all():
        raise ValueError("patched sample gradient contains NaN/Inf")
    return patched


def _to_depth_gradient_label(gradient_hw2: np.ndarray) -> np.ndarray:
    if gradient_hw2.ndim != 3 or gradient_hw2.shape[2] != 2:
        raise ValueError(f"expected v4 gradient shape (H,W,2), got {gradient_hw2.shape}")
    gradient_hw2 = np.nan_to_num(gradient_hw2.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    patched = np.transpose(gradient_hw2, (2, 0, 1))
    if not np.isfinite(patched).all():
        raise ValueError("patched reconstruction gradient contains NaN/Inf")
    return patched


def _backup_file(dataset_root: Path, source: Path, backup_dir: Path) -> str:
    destination = backup_dir / source.relative_to(dataset_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


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


def _build_side_work_units(dataset_root: Path, manifest: list[dict[str, Any]], cache_root: Path) -> tuple[list[AcquisitionWorkUnit], list[dict[str, Any]], PatchStats]:
    stats = PatchStats(scanned=len(manifest))
    units_by_key: dict[str, AcquisitionWorkUnit] = {}
    early_reports: list[dict[str, Any]] = []
    for index, row in enumerate(manifest):
        sample_id = row["sample_id"]
        raw_view_index = int(row.get("raw_view_index", -1))
        role = SIDE_ROLES.get(raw_view_index)
        if role is None:
            stats.skipped_not_side += 1
            continue
        sample_dir = dataset_root / "samples" / sample_id
        meta_path = sample_dir / "meta.json"
        required = [meta_path, sample_dir / "featurenet_targets.npz"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            stats.skipped_missing_files += 1
            early_reports.append({"sample_id": sample_id, "status": "missing_files", "missing": missing})
            continue
        meta = _read_json(meta_path)
        reconstruction_dir = _resolve_reconstruction_dir(dataset_root, meta)
        if reconstruction_dir is None or not (reconstruction_dir / "depth_gradient_labels.npz").exists():
            stats.skipped_missing_files += 1
            early_reports.append({"sample_id": sample_id, "status": "missing_reconstruction_or_gradient_cache"})
            continue
        acquisition_id = str(meta.get("multiview_reconstruction", {}).get("acquisition_id") or reconstruction_dir.name)
        key = str(reconstruction_dir.resolve())
        if key not in units_by_key:
            units_by_key[key] = AcquisitionWorkUnit(
                acquisition_id=acquisition_id,
                reconstruction_dir=reconstruction_dir,
                cache_dir=cache_root / acquisition_id,
            )
        units_by_key[key].samples.append(
            SideSampleRecord(
                manifest_index=index,
                sample_id=sample_id,
                raw_view_index=raw_view_index,
                role=role,
                sample_dir=sample_dir,
                reconstruction_dir=reconstruction_dir,
                acquisition_id=acquisition_id,
            )
        )
    return list(units_by_key.values()), early_reports, stats


def _build_shard_spans(total: int, shard_count: int) -> list[tuple[int, int]]:
    shard_count = max(1, int(shard_count))
    base = total // shard_count
    remainder = total % shard_count
    spans: list[tuple[int, int]] = []
    start = 0
    for shard in range(shard_count):
        size = base + (1 if shard < remainder else 0)
        end = start + size
        spans.append((start, end))
        start = end
    return spans


def _select_sharded_units(units: list[AcquisitionWorkUnit], args: argparse.Namespace) -> tuple[list[AcquisitionWorkUnit], dict[str, Any]]:
    mode = str(args.shard_mode)
    if mode == "off" or len(units) <= 1:
        return units, {"mode": mode, "selected_start": 0, "selected_end": len(units), "total_units": len(units)}
    if mode == "auto":
        target = max(1, int(args.target_shard_size))
        shard_count = max(1, math.ceil(len(units) / target))
        shard_index = 0
    elif mode == "manual":
        shard_count = max(1, int(args.shard_count))
        shard_index = int(args.shard_index)
    else:
        raise ValueError(f"unsupported shard mode: {mode}")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"--shard-index must be in [0, {shard_count - 1}], got {shard_index}")
    spans = _build_shard_spans(len(units), shard_count)
    start, end = spans[shard_index]
    return units[start:end], {
        "mode": mode,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "selected_start": start,
        "selected_end": end,
        "total_units": len(units),
        "selected_units": end - start,
    }


def _run_v4_unwrap(unit: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    cache_dir = Path(unit["cache_dir"])
    unwrap_dir = cache_dir / "depth_unwrap"
    marker_path = cache_dir / "depth_unwrap_done.json"
    report_path = unwrap_dir / "algorithm1_side_depth_unwrap_report.json"
    if marker_path.exists() and report_path.exists():
        marker = _read_json(marker_path)
        if marker.get("parameters") == params:
            return {"status": "cached", "marker_path": str(marker_path), "report_path": str(report_path)}

    unwrap_module = _load_depth_unwrap_module()
    started_at = time.time()
    report = unwrap_module.run(
        reconstruction_dir=Path(unit["reconstruction_dir"]),
        output_dir=unwrap_dir,
        left_angle=float(params["left_angle"]),
        right_angle=float(params["right_angle"]),
        samples_per_pixel=float(params["samples_per_pixel"]),
        reverse_left_unwrap_x=bool(params["reverse_left_unwrap_x"]),
        reverse_right_unwrap_x=bool(params["reverse_right_unwrap_x"]),
        unwrap_width_scale=float(params["unwrap_width_scale"]),
        unwrap_gradient_clip=params["unwrap_gradient_clip"],
        row_param_smooth_window=int(params["row_param_smooth_window"]),
        map_smooth_sigma_x=float(params["map_smooth_sigma_x"]),
        map_smooth_sigma_y=float(params["map_smooth_sigma_y"]),
    )
    marker = {
        "stage": "depth_unwrap",
        "status": "done",
        "acquisition_id": unit["acquisition_id"],
        "parameters": params,
        "report_path": str(report.get("report_path", report_path)),
        "elapsed_sec": float(time.time() - started_at),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _write_json(marker_path, marker)
    return {"status": "done", "marker_path": str(marker_path), "report_path": str(marker["report_path"])}


def _patch_reconstruction_cache(unit: dict[str, Any], params: dict[str, Any], dry_run: bool, backup_dir: Path | None) -> dict[str, Any]:
    dataset_root = Path(unit["dataset_root"])
    reconstruction_dir = Path(unit["reconstruction_dir"])
    cache_dir = Path(unit["cache_dir"])
    gradient_path = reconstruction_dir / "depth_gradient_labels.npz"
    arrays = _load_npz_arrays(gradient_path)
    updates: dict[str, np.ndarray] = {}
    role_reports: dict[str, Any] = {}
    for role in ("left", "right"):
        v4_gradient = np.load(cache_dir / "depth_unwrap" / role / f"{role}_gradient.npy").astype(np.float32)
        label = _to_depth_gradient_label(v4_gradient)
        updates[f"gradient_{role}"] = label
        role_reports[role] = {
            "source_path": str(cache_dir / "depth_unwrap" / role / f"{role}_gradient.npy"),
            "old": _gradient_stats(arrays[f"gradient_{role}"]) if f"gradient_{role}" in arrays else None,
            "new": _gradient_stats(label),
        }
    patched = {**arrays, **updates}
    for key in ("gradient_front", "gradient_left", "gradient_right"):
        if key in patched and not np.isfinite(patched[key]).all():
            raise ValueError(f"non-finite values in patched {key}")
    if dry_run:
        return {"status": "would_patch", "path": str(gradient_path), "roles": role_reports}
    if backup_dir is None:
        raise ValueError("backup_dir is required for mutation")
    backup_path = _backup_file(dataset_root, gradient_path, backup_dir)
    _save_npz(gradient_path, patched)
    return {"status": "patched", "path": str(gradient_path), "backup_path": backup_path, "roles": role_reports}


def _patch_sample_gradient(unit: dict[str, Any], sample: dict[str, Any], params: dict[str, Any], dry_run: bool, backup_dir: Path | None) -> dict[str, Any]:
    dataset_root = Path(unit["dataset_root"])
    cache_dir = Path(unit["cache_dir"])
    sample_dir = Path(sample["sample_dir"])
    role = str(sample["role"])
    targets_path = sample_dir / "featurenet_targets.npz"
    meta_path = sample_dir / "meta.json"
    targets = _load_npz_arrays(targets_path)
    if "output_mask" not in targets:
        raise KeyError(f"missing output_mask in {targets_path}")
    old_gradient = targets.get("gradient")
    v4_gradient = np.load(cache_dir / "depth_unwrap" / role / f"{role}_gradient.npy").astype(np.float32)
    patched_gradient = _resize_gradient_to_target(v4_gradient, targets["output_mask"])
    patched_targets = dict(targets)
    patched_targets["gradient"] = patched_gradient
    if not np.isfinite(patched_gradient).all():
        raise ValueError(f"non-finite patched gradient for {sample['sample_id']}")

    patch_record = {
        "patch_source": "algorithm1_v4_side_gradient_patch",
        "parameters": params,
        "view_role": role,
        "cache_dir": str(cache_dir),
        "source_gradient_path": str(cache_dir / "depth_unwrap" / role / f"{role}_gradient.npy"),
        "pre_patch_gradient_stats": _gradient_stats(old_gradient, targets["output_mask"]) if old_gradient is not None else None,
        "post_patch_gradient_stats": _gradient_stats(patched_gradient, targets["output_mask"]),
    }
    if dry_run:
        return {"sample_id": sample["sample_id"], "role": role, "status": "would_patch", **patch_record}
    if backup_dir is None:
        raise ValueError("backup_dir is required for mutation")
    backup_files = {
        "meta.json": _backup_file(dataset_root, meta_path, backup_dir),
        "featurenet_targets.npz": _backup_file(dataset_root, targets_path, backup_dir),
    }
    meta = _read_json(meta_path)
    meta.setdefault("patches", []).append(patch_record)
    gradient_gt = meta.get("gradient_ground_truth") if isinstance(meta.get("gradient_ground_truth"), dict) else {}
    meta["gradient_ground_truth"] = {
        **gradient_gt,
        "patch_source": "algorithm1_v4_side_gradient_patch",
        "view_role": role,
        "source": "algorithm1_side_depth_unwrap_fixed_v4",
        "cache_dir": str(cache_dir),
        "target_shape": list(patched_gradient.shape),
        "finite": True,
    }
    _save_npz(targets_path, patched_targets)
    _write_json(meta_path, meta)
    return {"sample_id": sample["sample_id"], "role": role, "status": "patched", "backup_files": backup_files, **patch_record}


def _process_acquisition_worker(payload: dict[str, Any]) -> dict[str, Any]:
    _cpu_worker_initializer()
    unit = payload["unit"]
    params = payload["parameters"]
    dry_run = bool(payload["dry_run"])
    backup_dir = Path(payload["backup_dir"]) if payload.get("backup_dir") else None
    marker_path = Path(unit["cache_dir"]) / "gradient_patch_done.json"
    if not dry_run and marker_path.exists():
        marker = _read_json(marker_path)
        if marker.get("status") == "done" and marker.get("parameters") == params:
            return {
                "acquisition_id": unit["acquisition_id"],
                "status": "already_patched",
                "resumed_sample_count": len(unit["samples"]),
                "marker_path": str(marker_path),
                "previous": marker,
            }

    unwrap_result = _run_v4_unwrap(unit, params)
    reconstruction_result = _patch_reconstruction_cache(unit, params, dry_run, backup_dir)
    sample_results = [
        _patch_sample_gradient(unit, sample, params, dry_run, backup_dir)
        for sample in unit["samples"]
    ]
    result = {
        "acquisition_id": unit["acquisition_id"],
        "status": "dry_run" if dry_run else "patched",
        "reconstruction_dir": unit["reconstruction_dir"],
        "cache_dir": unit["cache_dir"],
        "unwrap": unwrap_result,
        "reconstruction": reconstruction_result,
        "samples": sample_results,
    }
    _write_json(
        marker_path,
        {
            "stage": "gradient_patch",
            "status": "dry_run" if dry_run else "done",
            "parameters": params,
            "result": result,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    return result


def _unit_payload(dataset_root: Path, unit: AcquisitionWorkUnit) -> dict[str, Any]:
    return {
        "dataset_root": str(dataset_root),
        "acquisition_id": unit.acquisition_id,
        "reconstruction_dir": str(unit.reconstruction_dir),
        "cache_dir": str(unit.cache_dir),
        "samples": [
            {
                "manifest_index": record.manifest_index,
                "sample_id": record.sample_id,
                "raw_view_index": record.raw_view_index,
                "role": record.role,
                "sample_dir": str(record.sample_dir),
            }
            for record in unit.samples
        ],
    }


def _run_acquisition_stage(units: list[AcquisitionWorkUnit], dataset_root: Path, params: dict[str, Any], args: argparse.Namespace, backup_dir: Path | None) -> list[dict[str, Any]]:
    payloads = [
        {
            "unit": _unit_payload(dataset_root, unit),
            "parameters": params,
            "dry_run": bool(args.dry_run),
            "backup_dir": None if args.dry_run else str(backup_dir),
        }
        for unit in units
    ]
    if not payloads:
        return []
    num_workers = max(1, int(args.num_workers))
    if num_workers <= 1:
        return [_process_acquisition_worker(payload) for payload in payloads]
    ctx = mp.get_context("spawn")
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx, initializer=_cpu_worker_initializer) as executor:
        futures = [executor.submit(_process_acquisition_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                if args.fail_fast:
                    raise
                results.append({"status": "error", "error": str(exc)})
    results.sort(key=lambda item: str(item.get("acquisition_id", "")))
    return results


def patch_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"missing dataset root: {dataset_root}")
    if not args.dry_run and not args.in_place:
        raise ValueError("--in-place is required for mutation; use --dry-run for inspection")

    manifest = _read_json(dataset_root / "manifest.json")
    params = _parameter_payload(args)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = args.backup_dir.resolve() if args.backup_dir else dataset_root / f"side_gradient_patch_backup_{timestamp}"
    cache_root = args.cache_root.resolve() if args.cache_root else REPO_ROOT / "tmp" / "algorithm1_side_gradient_patch_cache" / dataset_root.name
    all_units, early_reports, stats = _build_side_work_units(dataset_root, manifest, cache_root)
    selected_units, shard_info = _select_sharded_units(all_units, args)
    selected_side_ids = {record.sample_id for unit in selected_units for record in unit.samples}
    selected_manifest = [
        row
        for row in manifest
        if int(row.get("raw_view_index", -1)) not in SIDE_ROLES or row.get("sample_id") in selected_side_ids
    ]
    if args.shard_mode != "off":
        stats.scanned = len(selected_manifest)
        stats.skipped_not_side = len(selected_manifest) - len(selected_side_ids)

    front_before = _front_hashes(dataset_root, selected_manifest)
    acquisition_results = _run_acquisition_stage(selected_units, dataset_root, params, args, backup_dir)

    sample_reports = list(early_reports)
    reconstruction_reports: list[dict[str, Any]] = []
    for result in acquisition_results:
        status = result.get("status")
        if status == "error":
            stats.errors += 1
            reconstruction_reports.append(result)
            continue
        if status == "already_patched":
            stats.resumed_already_done += int(result.get("resumed_sample_count", 0))
            reconstruction_reports.append(result)
            continue
        reconstruction = result.get("reconstruction", {})
        reconstruction_reports.append(reconstruction)
        if reconstruction.get("status") == "patched":
            stats.patched_reconstructions += 1
        elif reconstruction.get("status") == "would_patch":
            stats.dry_run_would_patch_reconstructions += 1
        for sample in result.get("samples", []):
            sample_reports.append(sample)
            if sample.get("status") == "patched":
                stats.patched_samples += 1
            elif sample.get("status") == "would_patch":
                stats.dry_run_would_patch_samples += 1
            elif sample.get("status") == "error":
                stats.errors += 1

    front_after = _front_hashes(dataset_root, selected_manifest)
    summary = {
        "dataset_root": str(dataset_root),
        "dry_run": bool(args.dry_run),
        "backup_dir": None if args.dry_run else str(backup_dir),
        "cache_root": str(cache_root),
        "parameters": params,
        "parallel": {
            "num_workers": max(1, int(args.num_workers)),
            "shard": shard_info,
            "selected_acquisition_count": len(selected_units),
            "selected_side_sample_count": len(selected_side_ids),
        },
        "stats": asdict(stats),
        "front_hashes_unchanged": bool(front_before == front_after),
        "acquisitions": acquisition_results,
        "reconstructions": reconstruction_reports,
        "samples": sample_reports,
    }
    summary_path = (
        cache_root / "algorithm1_side_gradient_patch_dry_run_summary.json"
        if args.dry_run
        else dataset_root / "algorithm1_side_gradient_patch_summary.json"
    )
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 4))
    parser.add_argument("--shard-mode", choices=("off", "auto", "manual"), default="off")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--target-shard-size", type=int, default=500)
    parser.add_argument("--row-param-smooth-window", type=int, default=DEFAULT_ROW_PARAM_SMOOTH_WINDOW)
    parser.add_argument("--map-smooth-sigma-x", type=float, default=DEFAULT_MAP_SMOOTH_SIGMA_X)
    parser.add_argument("--map-smooth-sigma-y", type=float, default=DEFAULT_MAP_SMOOTH_SIGMA_Y)
    parser.add_argument("--left-angle", type=float, default=-45.0)
    parser.add_argument("--right-angle", type=float, default=45.0)
    parser.add_argument("--samples-per-pixel", type=float, default=2.0)
    parser.add_argument("--reverse-left-unwrap-x", action="store_true")
    parser.add_argument("--reverse-right-unwrap-x", action="store_true")
    parser.add_argument("--unwrap-width-scale", type=float, default=1.0)
    parser.add_argument("--unwrap-gradient-clip", type=float, default=DEFAULT_UNWRAP_GRADIENT_CLIP)
    args = parser.parse_args()
    summary = patch_dataset(args)
    print(
        json.dumps(
            {
                "stats": summary["stats"],
                "front_hashes_unchanged": summary["front_hashes_unchanged"],
                "summary_path": summary["summary_path"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import errno
import getpass
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from featurenet.models.augmentation import (  # noqa: E402
    AugmentationConfig,
    AugmentationParams,
    augment_sample,
    has_usable_reconstruction,
    sample_augmentation_params,
)
from featurenet.models.train import load_bundle_samples  # noqa: E402


SYNTHETIC_FILES = ("masked_image.png", "mask.png", "featurenet_targets.npz", "minutiae.json", "meta.json")
TRAINING_FILES = ("masked_image.png", "mask.png", "featurenet_targets.npz", "meta.json")
SAMPLE_PAYLOAD_FILES = ("masked_image.png", "mask.png", "featurenet_targets.npz", "minutiae.json")
STORAGE_DIR_SUFFIX = "_synthetic_storage"
SCRIPT_VERSION = 1


class NoStorageSpaceError(RuntimeError):
    pass


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    os.replace(tmp_path, path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _load_source_manifest_by_sample(root: Path) -> dict[str, dict[str, Any]]:
    manifest = _read_json(root / "manifest.json", default=[])
    if not isinstance(manifest, list):
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in manifest:
        if isinstance(row, Mapping) and row.get("sample_id") is not None:
            rows[str(row["sample_id"])] = dict(row)
    return rows


def _sample_complete(sample_dir: Path, required_files: Sequence[str] = TRAINING_FILES) -> bool:
    return sample_dir.is_dir() and all((sample_dir / filename).exists() for filename in required_files)


def _directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            total += child.stat().st_size
    return total


def _safe_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    os.symlink(source.resolve(), destination)


def _atomic_replace_dir(temp_dir: Path, final_dir: Path) -> None:
    if final_dir.exists() or final_dir.is_symlink():
        shutil.rmtree(final_dir)
    os.replace(temp_dir, final_dir)


def _link_logical_sample(
    physical_sample_dir: Path,
    logical_sample_dir: Path,
    *,
    copy_meta: bool = True,
    payload_files: Sequence[str] = SAMPLE_PAYLOAD_FILES,
) -> None:
    temp_dir = logical_sample_dir.with_name(f".{logical_sample_dir.name}.tmp.{os.getpid()}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    for filename in payload_files:
        source = physical_sample_dir / filename
        if source.exists():
            _safe_symlink(source, temp_dir / filename)
    if copy_meta:
        shutil.copy2(physical_sample_dir / "meta.json", temp_dir / "meta.json")
    else:
        _safe_symlink(physical_sample_dir / "meta.json", temp_dir / "meta.json")
    _atomic_replace_dir(temp_dir, logical_sample_dir)


def _link_original_sample(source_sample_dir: Path, logical_sample_dir: Path) -> None:
    temp_dir = logical_sample_dir.with_name(f".{logical_sample_dir.name}.tmp.{os.getpid()}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    for child in sorted(source_sample_dir.iterdir()):
        destination = temp_dir / child.name
        if child.name == "meta.json":
            shutil.copy2(child, destination)
        elif child.is_file() or child.is_symlink():
            _safe_symlink(child.resolve(), destination)
    _atomic_replace_dir(temp_dir, logical_sample_dir)


def _storage_dataset_root(storage_root: Path, output_root: Path) -> Path:
    storage_root = storage_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if storage_root == output_root:
        return output_root / f".{output_root.name}{STORAGE_DIR_SUFFIX}"
    return storage_root / f"{output_root.name}{STORAGE_DIR_SUFFIX}"


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        try:
            resolved = path.expanduser().resolve()
        except FileNotFoundError:
            resolved = path.expanduser().absolute()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def discover_media_storage_roots(min_free_bytes: int) -> list[Path]:
    user_media = Path("/media") / getpass.getuser()
    if not user_media.exists():
        return []
    roots: list[Path] = []
    for candidate in sorted(path for path in user_media.iterdir() if path.is_dir()):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            if os.access(candidate, os.W_OK) and shutil.disk_usage(candidate).free >= min_free_bytes:
                roots.append(candidate.resolve())
        except OSError:
            continue
    return roots


def _choose_storage_root(
    storage_dataset_roots: Sequence[Path],
    min_free_bytes: int,
    disk_usage_fn: Callable[[Path], Any] = shutil.disk_usage,
    excluded_roots: set[Path] | None = None,
) -> Path:
    excluded_roots = excluded_roots or set()
    for storage_dataset_root in storage_dataset_roots:
        if storage_dataset_root in excluded_roots:
            continue
        storage_dataset_root.mkdir(parents=True, exist_ok=True)
        if disk_usage_fn(storage_dataset_root).free >= min_free_bytes:
            return storage_dataset_root
    raise NoStorageSpaceError("no storage root has enough free space")


def _params_for_variant(seed: int, sample_index: int, variant_index: int, config: AugmentationConfig) -> AugmentationParams:
    rng = np.random.default_rng(
        int(seed)
        + (int(sample_index) + 1) * 1_000_003
        + int(variant_index) * 10_007
    )
    return sample_augmentation_params(rng, config)


def _make_augmented_meta(
    source_meta: Mapping[str, Any],
    *,
    sample_id: str,
    parent_sample_id: str,
    result: Any,
    seed: int,
    source_root: Path,
    variant_index: int,
) -> dict[str, Any]:
    meta = dict(source_meta)
    source_minutiae_count = int(source_meta.get("counts", {}).get("minutiae", 0)) if isinstance(source_meta.get("counts"), Mapping) else 0
    meta["sample_id"] = sample_id
    meta["parent_sample_id"] = parent_sample_id
    meta.pop("multiview_reconstruction", None)
    counts = dict(meta.get("counts", {})) if isinstance(meta.get("counts"), Mapping) else {}
    counts["minutiae"] = len(result.minutiae)
    meta["counts"] = counts
    meta["pregenerated_augmentation"] = {
        "version": SCRIPT_VERSION,
        "source_root": str(source_root.resolve()),
        "source_sample_id": parent_sample_id,
        "variant_index": int(variant_index),
        "seed": int(seed),
        "params": asdict(result.params),
        "mode": result.details.get("mode"),
        "details": result.details,
        "source_minutiae_count": source_minutiae_count,
        "augmented_minutiae_count": len(result.minutiae),
        "dropped_minutiae_count": max(0, source_minutiae_count - len(result.minutiae)),
    }
    return meta


def _write_synthetic_sample(
    sample: Mapping[str, Any],
    *,
    sample_index: int,
    variant_index: int,
    output_root: Path,
    storage_dataset_roots: Sequence[Path],
    config: AugmentationConfig,
    seed: int,
    min_free_bytes: int,
) -> dict[str, Any]:
    source_sample_id = str(sample["sample_id"])
    sample_id = f"{source_sample_id}_aug{variant_index:02d}"
    logical_sample_dir = output_root / "samples" / sample_id
    if _sample_complete(logical_sample_dir):
        return {
            "status": "skipped_existing",
            "sample_id": sample_id,
            "source_sample_id": source_sample_id,
            "variant_index": int(variant_index),
            "logical_sample_dir": str(logical_sample_dir),
        }

    started = time.perf_counter()
    params = _params_for_variant(seed, sample_index, variant_index, config)
    output_shape = tuple(int(value) for value in sample["output_shape_hw"])
    result = augment_sample(
        sample,
        params,
        output_shape=output_shape,
        reconstruction_cache_size=config.reconstruction_cache_size,
        sample_cache_size=config.sample_cache_size,
    )
    source_meta = _read_json(Path(sample["meta_path"]), default={})
    meta = _make_augmented_meta(
        source_meta,
        sample_id=sample_id,
        parent_sample_id=source_sample_id,
        result=result,
        seed=seed,
        source_root=Path(sample["meta_path"]).parents[2],
        variant_index=variant_index,
    )

    last_error: str | None = None
    tried_roots: list[str] = []
    excluded_roots: set[Path] = set()
    for _ in range(len(storage_dataset_roots)):
        try:
            storage_dataset_root = _choose_storage_root(storage_dataset_roots, min_free_bytes, excluded_roots=excluded_roots)
        except NoStorageSpaceError as exc:
            last_error = str(exc)
            break
        excluded_roots.add(storage_dataset_root)
        tried_roots.append(str(storage_dataset_root))
        physical_samples_root = storage_dataset_root / "samples"
        physical_samples_root.mkdir(parents=True, exist_ok=True)
        final_dir = physical_samples_root / sample_id
        temp_dir = physical_samples_root / f".{sample_id}.tmp.{os.getpid()}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        try:
            temp_dir.mkdir(parents=True)
            if not cv2.imwrite(str(temp_dir / "masked_image.png"), result.image):
                raise RuntimeError(f"failed to write image for {sample_id}")
            if not cv2.imwrite(str(temp_dir / "mask.png"), result.mask):
                raise RuntimeError(f"failed to write mask for {sample_id}")
            np.savez_compressed(temp_dir / "featurenet_targets.npz", **result.targets)
            _write_json(temp_dir / "minutiae.json", result.minutiae)
            _write_json(temp_dir / "meta.json", meta)
            _atomic_replace_dir(temp_dir, final_dir)
            _link_logical_sample(final_dir, logical_sample_dir)
            return {
                "status": "generated",
                "sample_id": sample_id,
                "source_sample_id": source_sample_id,
                "variant_index": int(variant_index),
                "logical_sample_dir": str(logical_sample_dir),
                "physical_sample_dir": str(final_dir),
                "storage_dataset_root": str(storage_dataset_root),
                "params": asdict(result.params),
                "mode": result.details.get("mode"),
                "seconds": round(time.perf_counter() - started, 3),
                "size_bytes": _directory_size_bytes(final_dir),
                "tried_storage_roots": tried_roots,
            }
        except OSError as exc:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            last_error = str(exc)
            if exc.errno in {errno.ENOSPC, errno.EDQUOT}:
                continue
            raise
        except Exception:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise
    raise NoStorageSpaceError(f"could not write {sample_id}; tried={tried_roots}; last_error={last_error}")


def _worker_generate(task: dict[str, Any]) -> dict[str, Any]:
    return _write_synthetic_sample(**task)


def _manifest_row_for_sample(
    sample: Mapping[str, Any],
    source_manifest_by_sample: Mapping[str, dict[str, Any]],
    *,
    synthetic_id: str | None = None,
    variant_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_sample_id = str(sample["sample_id"])
    row = dict(source_manifest_by_sample.get(source_sample_id, {}))
    if not row:
        meta = _read_json(Path(sample["meta_path"]), default={})
        row = {
            key: meta.get(key)
            for key in ("sample_id", "subject_id", "subject_index", "finger_id", "acquisition_id", "finger_class_id", "raw_view_index")
            if key in meta
        }
    if synthetic_id is None:
        row["sample_id"] = source_sample_id
        row["is_pregenerated_augmentation"] = False
    else:
        row["sample_id"] = synthetic_id
        row["parent_sample_id"] = source_sample_id
        row["is_pregenerated_augmentation"] = True
        if variant_record is not None:
            row["pregenerated_augmentation"] = {
                "source_sample_id": source_sample_id,
                "variant_index": variant_record.get("variant_index"),
                "params": variant_record.get("params"),
                "mode": variant_record.get("mode"),
            }
    return row


def _prepare_storage_roots(args: argparse.Namespace, output_root: Path, min_free_bytes: int) -> list[Path]:
    roots: list[Path] = []
    roots.extend(Path(value) for value in args.storage_root)
    if not roots:
        roots.append(output_root)
    if args.auto_discover_media_roots:
        roots.extend(discover_media_storage_roots(min_free_bytes))
    return [_storage_dataset_root(path, output_root) for path in _dedupe_paths(roots)]


def _build_config(args: argparse.Namespace) -> AugmentationConfig:
    translation_typical, translation_strong = sorted(float(value) for value in args.translation_jitter_px)
    _, pitch_roll_strong = sorted(float(value) for value in args.pitch_roll_jitter_deg)
    return AugmentationConfig(
        count=int(args.augmentation_count),
        translation_typical_px=translation_typical,
        translation_strong_px=translation_strong,
        yaw_deg=float(args.yaw_jitter_deg),
        pitch_roll_typical_deg=pitch_roll_strong,
        pitch_roll_strong_deg=max(25.0, pitch_roll_strong),
        missing_reconstruction=args.augmentation_missing_reconstruction,
        reconstruction_cache_size=int(args.augmentation_reconstruction_cache_size),
        sample_cache_size=int(args.augmentation_sample_cache_size),
    )


def _prepare_output_root(output_root: Path, args: argparse.Namespace, storage_dataset_roots: Sequence[Path]) -> None:
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        if not args.overwrite:
            raise RuntimeError(f"refusing to use non-empty output root without --resume or --overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "samples").mkdir(parents=True, exist_ok=True)
    if args.overwrite and not args.resume:
        for storage_dataset_root in storage_dataset_roots:
            if storage_dataset_root.exists():
                shutil.rmtree(storage_dataset_root)
    for storage_dataset_root in storage_dataset_roots:
        (storage_dataset_root / "samples").mkdir(parents=True, exist_ok=True)


def _link_originals(
    samples: Sequence[Mapping[str, Any]],
    output_root: Path,
    *,
    resume: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        source_sample_dir = Path(sample["meta_path"]).parent
        logical_sample_dir = output_root / "samples" / sample_id
        if resume and _sample_complete(logical_sample_dir):
            status = "skipped_existing"
        else:
            _link_original_sample(source_sample_dir, logical_sample_dir)
            status = "linked"
        records.append(
            {
                "status": status,
                "sample_id": sample_id,
                "source_sample_id": sample_id,
                "variant_index": 0,
                "logical_sample_dir": str(logical_sample_dir),
                "physical_sample_dir": str(source_sample_dir),
                "is_original": True,
            }
        )
    return records


def generate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_root = args.output_root.expanduser().resolve()
    source_root = args.ground_truth_root.expanduser().resolve()
    min_free_bytes = int(float(args.min_free_gb) * 1024**3)
    storage_dataset_roots = _prepare_storage_roots(args, output_root, min_free_bytes)
    _prepare_output_root(output_root, args, storage_dataset_roots)

    config = _build_config(args)
    samples = load_bundle_samples(
        source_root,
        limit=args.limit,
        strict_gradient_targets=args.strict_gradient_targets,
        strict_finite_targets=args.strict_finite_targets,
        skip_empty_minutia_support_with_minutiae=args.skip_empty_minutia_support_with_minutiae,
    )
    if config.missing_reconstruction == "skip":
        before = len(samples)
        samples = [sample for sample in samples if has_usable_reconstruction(sample)]
        skipped = before - len(samples)
        if skipped:
            print(f"[pregenerate] skipped {skipped} samples without usable reconstruction metadata", flush=True)
    if not samples:
        raise ValueError("no eligible samples to augment")

    original_records = _link_originals(samples, output_root, resume=bool(args.resume))
    source_manifest_by_sample = _load_source_manifest_by_sample(source_root)
    storage_records: list[dict[str, Any]] = list(original_records)
    manifest_rows: list[dict[str, Any]] = [
        _manifest_row_for_sample(sample, source_manifest_by_sample)
        for sample in samples
    ]

    tasks: list[dict[str, Any]] = []
    sample_by_id = {str(sample["sample_id"]): sample for sample in samples}
    for sample_index, sample in enumerate(samples):
        for variant_index in range(1, int(config.count) + 1):
            sample_id = f"{sample['sample_id']}_aug{variant_index:02d}"
            logical_sample_dir = output_root / "samples" / sample_id
            if args.resume and _sample_complete(logical_sample_dir):
                record = {
                    "status": "skipped_existing",
                    "sample_id": sample_id,
                    "source_sample_id": str(sample["sample_id"]),
                    "variant_index": variant_index,
                    "logical_sample_dir": str(logical_sample_dir),
                    "is_original": False,
                }
                storage_records.append(record)
                manifest_rows.append(_manifest_row_for_sample(sample, source_manifest_by_sample, synthetic_id=sample_id, variant_record=record))
                continue
            tasks.append(
                {
                    "sample": sample,
                    "sample_index": sample_index,
                    "variant_index": variant_index,
                    "output_root": output_root,
                    "storage_dataset_roots": storage_dataset_roots,
                    "config": config,
                    "seed": int(args.seed),
                    "min_free_bytes": min_free_bytes,
                }
            )

    print(
        "[pregenerate] "
        f"samples={len(samples)}, originals={len(original_records)}, pending_synthetic={len(tasks)}, "
        f"workers={args.workers}, storage_roots={[str(path) for path in storage_dataset_roots]}",
        flush=True,
    )

    errors: list[dict[str, Any]] = []
    completed = 0
    flush_every = max(1, int(args.manifest_flush_every))

    def record_result(record: dict[str, Any]) -> None:
        nonlocal completed
        storage_records.append(record)
        completed += 1
        source_sample = sample_by_id.get(str(record.get("source_sample_id")))
        if source_sample is not None:
            manifest_rows.append(
                _manifest_row_for_sample(
                    source_sample,
                    source_manifest_by_sample,
                    synthetic_id=str(record["sample_id"]),
                    variant_record=record,
                )
            )
        if completed % flush_every == 0:
            _write_storage_manifest(output_root, source_root, storage_dataset_roots, storage_records, errors, args, started)
            print(f"[pregenerate] completed {completed}/{len(tasks)} synthetic variants", flush=True)

    if tasks:
        if int(args.workers) <= 1:
            for task in tasks:
                try:
                    record_result(_worker_generate(task))
                except Exception as exc:  # noqa: BLE001 - summarize and keep manifest resumable.
                    errors.append({"task": {"sample_id": task["sample"]["sample_id"], "variant_index": task["variant_index"]}, "error": repr(exc)})
                    if isinstance(exc, NoStorageSpaceError):
                        break
        else:
            with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
                future_to_task = {executor.submit(_worker_generate, task): task for task in tasks}
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        record_result(future.result())
                    except Exception as exc:  # noqa: BLE001 - summarize and keep manifest resumable.
                        errors.append({"task": {"sample_id": task["sample"]["sample_id"], "variant_index": task["variant_index"]}, "error": repr(exc)})
                        if isinstance(exc, NoStorageSpaceError):
                            break

    manifest_rows = sorted(manifest_rows, key=lambda row: str(row.get("sample_id", "")))
    summary = _summary_payload(output_root, source_root, storage_dataset_roots, storage_records, errors, args, started)
    _write_json_atomic(output_root / "manifest.json", manifest_rows)
    _write_manifest_csv(output_root / "manifest.csv", manifest_rows)
    _write_json_atomic(output_root / "summary.json", summary)
    _write_storage_manifest(output_root, source_root, storage_dataset_roots, storage_records, errors, args, started)
    print(json.dumps(summary, indent=2, default=_json_default), flush=True)
    return summary


def _summary_payload(
    output_root: Path,
    source_root: Path,
    storage_dataset_roots: Sequence[Path],
    storage_records: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    started: float,
) -> dict[str, Any]:
    generated = [record for record in storage_records if record.get("status") == "generated"]
    linked = [record for record in storage_records if record.get("status") == "linked"]
    skipped = [record for record in storage_records if record.get("status") == "skipped_existing"]
    return {
        "version": SCRIPT_VERSION,
        "mode": "pregenerated_featurenet_augmentations",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "storage_dataset_roots": [str(path) for path in storage_dataset_roots],
        "augmentation_count": int(args.augmentation_count),
        "generated_synthetic_count": len(generated),
        "linked_original_count": len(linked),
        "skipped_existing_count": len(skipped),
        "error_count": len(errors),
        "total_size_bytes": sum(int(record.get("size_bytes", 0) or 0) for record in generated),
        "seconds": round(time.perf_counter() - started, 3),
        "errors": list(errors[:50]),
    }


def _write_storage_manifest(
    output_root: Path,
    source_root: Path,
    storage_dataset_roots: Sequence[Path],
    storage_records: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    started: float,
) -> None:
    payload = {
        "summary": _summary_payload(output_root, source_root, storage_dataset_roots, storage_records, errors, args, started),
        "records": list(storage_records),
    }
    _write_json_atomic(output_root / "storage_manifest.json", payload)


def _write_manifest_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys = [
        "sample_id",
        "parent_sample_id",
        "subject_id",
        "subject_index",
        "finger_id",
        "acquisition_id",
        "finger_class_id",
        "raw_view_index",
        "is_pregenerated_augmentation",
    ]
    lines = [",".join(keys)]
    for row in rows:
        values = []
        for key in keys:
            value = row.get(key, "")
            values.append(str(value).replace(",", " "))
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-generate compact FeatureNet augmentation bundles.")
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--augmentation-count", type=int, default=5)
    parser.add_argument("--translation-jitter-px", type=float, nargs=2, default=(16.0, 32.0), metavar=("TYPICAL", "STRONG"))
    parser.add_argument("--yaw-jitter-deg", type=float, default=5.0)
    parser.add_argument("--pitch-roll-jitter-deg", type=float, nargs=2, default=(5.0, 15.0), metavar=("TYPICAL_MIN", "TYPICAL_MAX"))
    parser.add_argument("--augmentation-missing-reconstruction", choices=("skip", "allow"), default="skip")
    parser.add_argument("--augmentation-reconstruction-cache-size", type=int, default=2)
    parser.add_argument("--augmentation-sample-cache-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--storage-root", type=Path, action="append", default=[])
    parser.add_argument("--auto-discover-media-roots", dest="auto_discover_media_roots", action="store_true", default=True)
    parser.add_argument("--no-auto-discover-media-roots", dest="auto_discover_media_roots", action="store_false")
    parser.add_argument("--manifest-flush-every", type=int, default=50)
    parser.add_argument("--strict-gradient-targets", action="store_true")
    parser.add_argument("--strict-finite-targets", action="store_true")
    parser.add_argument("--skip-empty-minutia-support-with-minutiae", action="store_true", default=True)
    parser.add_argument("--allow-empty-minutia-support-with-minutiae", dest="skip_empty_minutia_support_with_minutiae", action="store_false")
    args = parser.parse_args(argv)
    if args.augmentation_count < 0:
        parser.error("--augmentation-count must be non-negative")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.min_free_gb < 0.0:
        parser.error("--min-free-gb must be non-negative")
    if args.manifest_flush_every < 1:
        parser.error("--manifest-flush-every must be at least 1")
    if any(value < 0.0 for value in args.translation_jitter_px):
        parser.error("--translation-jitter-px values must be non-negative")
    if args.yaw_jitter_deg < 0.0:
        parser.error("--yaw-jitter-deg must be non-negative")
    if any(value < 0.0 for value in args.pitch_roll_jitter_deg):
        parser.error("--pitch-roll-jitter-deg values must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    generate_dataset(parse_args(argv))


if __name__ == "__main__":
    main()

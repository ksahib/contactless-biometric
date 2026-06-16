from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import sysconfig
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ARCHIVE_ROOT = Path("/media/milab-5/82002d9e-66a9-4739-925b-e2b789ec5641/archive")
DEFAULT_WEIGHTS_PATH = REPO_ROOT / "runs" / "featurenet_v4" / "best.pt"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "match_outputs"
CACHE_VERSION = 3
DEFAULT_MCC_METHODS = ("LSA", "LSA-R", "LSA-CENTROID")
DEFAULT_FEATURE_SCORE_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def ensure_stdlib_copy_module() -> None:
    """Avoid importing this repository's copy.py when libraries need stdlib copy."""
    stdlib_copy = Path(sysconfig.get_paths()["stdlib"]) / "copy.py"
    spec = importlib.util.spec_from_file_location("copy", stdlib_copy)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not resolve stdlib copy module from {stdlib_copy}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = module
    spec.loader.exec_module(module)


ensure_stdlib_copy_module()

Path("/tmp/contactless_matplotlib_config").mkdir(parents=True, exist_ok=True)
Path("/tmp/contactless_matplotlib_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/contactless_matplotlib_config")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/contactless_matplotlib_cache")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from dataclasses import asdict, dataclass


def prepend_workspace_site_packages() -> None:
    try:
        sys_prefix = Path(sys.prefix).resolve()
    except Exception:
        sys_prefix = None
    for site_packages in (
        REPO_ROOT / ".venv" / "Lib" / "site-packages",
        REPO_ROOT / ".venv" / "lib" / "site-packages",
    ):
        if not site_packages.exists():
            continue
        try:
            resolved = site_packages.resolve()
        except Exception:
            resolved = site_packages
        if sys_prefix is None or sys_prefix in resolved.parents:
            sys.path.insert(0, str(site_packages))


prepend_workspace_site_packages()


@dataclass(frozen=True)
class ImageRecord:
    dataset: str
    subject_id: int
    finger_id: int
    acquisition_id: int
    view_index: int
    image_path: str
    mask_path: str = ""
    source_kind: str = "archive"

    @property
    def identity_key(self) -> tuple[str, int, int]:
        return self.dataset, self.subject_id, self.finger_id

    @property
    def sample_uid(self) -> str:
        digest = hashlib.sha1(self.image_path.encode("utf-8", errors="replace")).hexdigest()[:12]
        return (
            f"{self.dataset}_s{self.subject_id:03d}_f{self.finger_id:02d}_"
            f"a{self.acquisition_id:02d}_v{self.view_index:02d}_{digest}"
        )

    @property
    def cache_key(self) -> str:
        return (
            f"{_slug(self.source_kind)}"
            f"/{_slug(self.dataset)}"
            f"/s{self.subject_id:03d}"
            f"/f{self.finger_id:02d}"
            f"/a{self.acquisition_id:02d}"
            f"/v{self.view_index:02d}_{hashlib.sha1(self.image_path.encode('utf-8')).hexdigest()[:10]}"
        )


@dataclass(frozen=True)
class PairSpec:
    label: str
    a: ImageRecord
    b: ImageRecord

    @property
    def same_identity(self) -> bool:
        return self.a.identity_key == self.b.identity_key

    @property
    def pair_key(self) -> tuple[str, str, str]:
        left, right = sorted((self.a.image_path, self.b.image_path))
        return self.label, left, right


@dataclass(frozen=True)
class ExtractedImage:
    sample_uid: str
    status: str
    minutiae_csv: str
    mask_png: str
    orientation_npy: str
    ridge_period_npy: str
    minutiae_count: int | None
    error: str


def _slug(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    return safe or "value"


def _threshold_label(value: float) -> str:
    return f"{float(value):.2f}"


def _method_label(value: str) -> str:
    return str(value).strip().upper()


def _method_slug(value: str) -> str:
    return _slug(_method_label(value).lower().replace("-", "_"))


def _unique_methods(values: Iterable[str]) -> list[str]:
    methods: list[str] = []
    seen: set[str] = set()
    for value in values:
        method = _method_label(value)
        if not method:
            continue
        if method in seen:
            continue
        seen.add(method)
        methods.append(method)
    if not methods:
        raise ValueError("--methods must contain at least one non-empty method")
    return methods


def _default_output_dir() -> Path:
    return DEFAULT_OUTPUT_ROOT / f"binary_roc_sweep_{time.strftime('%Y%m%d_%H%M%S')}"


def _parse_raw_image_path(dataset_dir: Path, raw_path: Path) -> ImageRecord | None:
    if raw_path.suffix.lower() != ".jpg":
        return None
    parts = raw_path.stem.split("_")
    if len(parts) != 4:
        return None
    try:
        subject_id, finger_id, acquisition_id, view_index = [int(part) for part in parts]
    except ValueError:
        return None
    return ImageRecord(
        dataset=dataset_dir.name,
        subject_id=subject_id,
        finger_id=finger_id,
        acquisition_id=acquisition_id,
        view_index=view_index,
        image_path=str(raw_path.resolve()),
    )


def discover_archive_images(
    archive_root: Path,
    *,
    side_views: set[int],
    front_view: int = 0,
) -> list[ImageRecord]:
    root = archive_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"archive root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"archive root is not a directory: {root}")

    allowed_views = {int(front_view)} | {int(view) for view in side_views}
    records: list[ImageRecord] = []
    seen_paths: set[str] = set()
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for raw_path in sorted(dataset_dir.glob("*/raw/*.jpg")):
            record = _parse_raw_image_path(dataset_dir, raw_path)
            if record is None or record.view_index not in allowed_views:
                continue
            if record.image_path in seen_paths:
                continue
            seen_paths.add(record.image_path)
            records.append(record)

    records.sort(
        key=lambda item: (
            item.dataset,
            item.subject_id,
            item.finger_id,
            item.acquisition_id,
            item.view_index,
            item.image_path,
        )
    )
    return records


def _dataset_from_bundle_meta(meta: dict[str, Any], sample_dir: Path, ground_truth_root: Path) -> str:
    sample_id = str(meta.get("sample_id") or sample_dir.name)
    prefix = sample_id.split("_", 1)[0].upper()
    if prefix.startswith("DS") and prefix[2:].isdigit():
        return prefix
    raw_path = str(meta.get("raw_image_path") or "")
    for part in Path(raw_path).parts:
        upper = part.upper()
        if upper.startswith("DS") and upper[2:].isdigit():
            return upper
    root_name = ground_truth_root.name.upper()
    return root_name or "GT"


def discover_ground_truth_bundle_images(
    ground_truth_root: Path,
    *,
    side_views: set[int],
    front_view: int = 0,
) -> list[ImageRecord]:
    root = ground_truth_root.resolve()
    samples_root = root / "samples"
    if not samples_root.exists():
        raise FileNotFoundError(f"ground-truth samples directory not found: {samples_root}")
    if not samples_root.is_dir():
        raise NotADirectoryError(f"ground-truth samples path is not a directory: {samples_root}")

    allowed_views = {int(front_view)} | {int(view) for view in side_views}
    records: list[ImageRecord] = []
    seen_paths: set[str] = set()
    for sample_dir in sorted(path for path in samples_root.iterdir() if path.is_dir()):
        meta_path = sample_dir / "meta.json"
        masked_image_path = sample_dir / "masked_image.png"
        mask_path = sample_dir / "mask.png"
        if not (meta_path.exists() and masked_image_path.exists() and mask_path.exists()):
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            subject_id = int(meta["subject_id"])
            finger_id = int(meta["finger_id"])
            acquisition_id = int(meta["acquisition_id"])
            view_index = int(meta["raw_view_index"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if view_index not in allowed_views:
            continue
        image_path = str(masked_image_path.resolve())
        if image_path in seen_paths:
            continue
        seen_paths.add(image_path)
        records.append(
            ImageRecord(
                dataset=_dataset_from_bundle_meta(meta, sample_dir, root),
                subject_id=subject_id,
                finger_id=finger_id,
                acquisition_id=acquisition_id,
                view_index=view_index,
                image_path=image_path,
                mask_path=str(mask_path.resolve()),
                source_kind="gt_bundle",
            )
        )

    records.sort(
        key=lambda item: (
            item.dataset,
            item.subject_id,
            item.finger_id,
            item.acquisition_id,
            item.view_index,
            item.image_path,
        )
    )
    return records


def _group_by_identity(records: Iterable[ImageRecord]) -> dict[tuple[str, int, int], list[ImageRecord]]:
    grouped: dict[tuple[str, int, int], list[ImageRecord]] = {}
    for record in records:
        grouped.setdefault(record.identity_key, []).append(record)
    return grouped


def sample_genuine_pairs(
    records: list[ImageRecord],
    *,
    side_views: set[int],
    count: int,
    rng: random.Random,
) -> list[PairSpec]:
    candidates: list[PairSpec] = []
    for group_records in _group_by_identity(records).values():
        fronts = [record for record in group_records if record.view_index == 0]
        sides = [record for record in group_records if record.view_index in side_views]
        for front in fronts:
            for side in sides:
                candidates.append(PairSpec(label="genuine", a=front, b=side))

    deduped = _dedupe_pairs(candidates)
    if len(deduped) < count:
        raise RuntimeError(f"not enough genuine front-side pairs: requested {count}, found {len(deduped)}")
    return rng.sample(deduped, count)


def sample_impostor_pairs(
    records: list[ImageRecord],
    *,
    count: int,
    rng: random.Random,
) -> list[PairSpec]:
    fronts = [record for record in records if record.view_index == 0]
    if len({record.identity_key for record in fronts}) < 2:
        raise RuntimeError("not enough distinct front-view identities for impostor sampling")

    pairs: list[PairSpec] = []
    seen: set[tuple[str, str, str]] = set()
    attempts = 0
    max_attempts = max(1000, count * 500)
    while len(pairs) < count and attempts < max_attempts:
        attempts += 1
        a, b = rng.sample(fronts, 2)
        if a.identity_key == b.identity_key:
            continue
        pair = PairSpec(label="impostor", a=a, b=b)
        if pair.pair_key in seen:
            continue
        seen.add(pair.pair_key)
        pairs.append(pair)

    if len(pairs) < count:
        raise RuntimeError(f"not enough impostor front-front pairs: requested {count}, sampled {len(pairs)}")
    return pairs


def sample_binary_pairs(
    records: list[ImageRecord],
    *,
    side_views: set[int],
    genuine_count: int,
    impostor_count: int,
    seed: int,
) -> list[PairSpec]:
    rng = random.Random(seed)
    genuine = sample_genuine_pairs(records, side_views=side_views, count=genuine_count, rng=rng)
    impostor = sample_impostor_pairs(records, count=impostor_count, rng=rng)
    pairs = genuine + impostor
    pairs.sort(key=lambda pair: (pair.label, pair.a.image_path, pair.b.image_path))
    return pairs


def _dedupe_pairs(pairs: Iterable[PairSpec]) -> list[PairSpec]:
    deduped: list[PairSpec] = []
    seen: set[tuple[str, str, str]] = set()
    for pair in pairs:
        if pair.pair_key in seen:
            continue
        seen.add(pair.pair_key)
        deduped.append(pair)
    return deduped


def unique_records_from_pairs(pairs: Iterable[PairSpec]) -> list[ImageRecord]:
    by_path: dict[str, ImageRecord] = {}
    for pair in pairs:
        by_path.setdefault(pair.a.image_path, pair.a)
        by_path.setdefault(pair.b.image_path, pair.b)
    return sorted(
        by_path.values(),
        key=lambda item: (
            item.dataset,
            item.subject_id,
            item.finger_id,
            item.acquisition_id,
            item.view_index,
            item.image_path,
        ),
    )


def _load_infer_helpers() -> dict[str, Any]:
    from featurenet.models.infer import (
        decode_minutiae_rows,
        load_checkpoint_model,
        preprocess_input_image,
        preprocess_saved_masked_input,
        run_inference,
        save_minutiae_csv,
        save_pose_sidecars,
        unwarp_minutiae_rows,
        _resolve_device,
    )
    from featurenet.models.match_infer import _save_mask_png, _save_mask_array_png

    return {
        "decode_minutiae_rows": decode_minutiae_rows,
        "load_checkpoint_model": load_checkpoint_model,
        "preprocess_input_image": preprocess_input_image,
        "preprocess_saved_masked_input": preprocess_saved_masked_input,
        "run_inference": run_inference,
        "save_minutiae_csv": save_minutiae_csv,
        "save_pose_sidecars": save_pose_sidecars,
        "unwarp_minutiae_rows": unwarp_minutiae_rows,
        "_resolve_device": _resolve_device,
        "_save_mask_png": _save_mask_png,
        "_save_mask_array_png": _save_mask_array_png,
    }


def _cache_files(cache_dir: Path) -> dict[str, Path]:
    return {
        "minutiae_csv": cache_dir / "minutiae.csv",
        "mask_png": cache_dir / "mask.png",
        "orientation_npy": cache_dir / "orientation.npy",
        "ridge_period_npy": cache_dir / "ridge_period.npy",
        "metadata_json": cache_dir / "metadata.json",
    }


def _count_minutiae_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def _cached_extraction_is_complete(
    files: dict[str, Path],
    *,
    record: ImageRecord,
    weights_path: Path,
    score_threshold: float,
    apply_nms: bool,
    solov2_score_thr: float,
    unwarp: str,
) -> tuple[bool, int | None]:
    required = ("minutiae_csv", "mask_png", "orientation_npy", "ridge_period_npy", "metadata_json")
    if not all(files[name].exists() and files[name].stat().st_size > 0 for name in required):
        return False, None
    try:
        metadata = json.loads(files["metadata_json"].read_text(encoding="utf-8"))
    except Exception:
        return False, None
    if metadata.get("cache_version") != CACHE_VERSION:
        return False, None
    if metadata.get("sample_uid") != record.sample_uid:
        return False, None
    if metadata.get("image_path") != record.image_path:
        return False, None
    if metadata.get("mask_path", "") != record.mask_path:
        return False, None
    if metadata.get("source_kind", "archive") != record.source_kind:
        return False, None
    if metadata.get("weights_path") != str(weights_path.resolve()):
        return False, None
    if float(metadata.get("feature_score_threshold", -1.0)) != float(score_threshold):
        return False, None
    if bool(metadata.get("minutia_nms_enabled")) != bool(apply_nms):
        return False, None
    if str(metadata.get("unwarp", "none")) != str(unwarp):
        return False, None
    if record.source_kind != "gt_bundle" and float(metadata.get("solov2_score_thr", -1.0)) != float(solov2_score_thr):
        return False, None
    return True, int(metadata.get("minutiae_count", _count_minutiae_csv_rows(files["minutiae_csv"])))


def _extraction_row(
    record: ImageRecord,
    *,
    feature_score_threshold: float,
    status: str,
    cache_hit: bool,
    cache_dir: Path,
    files: dict[str, Path],
    minutiae_count: int | None,
    error: str = "",
) -> dict[str, Any]:
    return {
        "feature_score_threshold": _threshold_label(feature_score_threshold),
        "sample_uid": record.sample_uid,
        "dataset": record.dataset,
        "subject_id": record.subject_id,
        "finger_id": record.finger_id,
        "acquisition_id": record.acquisition_id,
        "view_index": record.view_index,
        "image_path": record.image_path,
        "mask_path": record.mask_path,
        "source_kind": record.source_kind,
        "status": status,
        "cache_hit": str(cache_hit).lower(),
        "cache_dir": str(cache_dir),
        "minutiae_count": "" if minutiae_count is None else int(minutiae_count),
        "minutiae_csv": str(files["minutiae_csv"]),
        "mask_png": str(files["mask_png"]),
        "orientation_npy": str(files["orientation_npy"]),
        "ridge_period_npy": str(files["ridge_period_npy"]),
        "error": error,
    }


def extract_image(
    record: ImageRecord,
    *,
    helpers: dict[str, Any],
    model: Any,
    device: Any,
    weights_path: Path,
    score_threshold: float,
    solov2_score_thr: float,
    cache_root: Path,
    reuse_cache: bool,
    apply_nms: bool,
    unwarp: str = "none",
) -> dict[str, Any]:
    cache_dir = cache_root / f"score_{_threshold_label(score_threshold)}" / record.cache_key
    files = _cache_files(cache_dir)
    try:
        if reuse_cache:
            complete, cached_count = _cached_extraction_is_complete(
                files,
                record=record,
                weights_path=weights_path,
                score_threshold=score_threshold,
                apply_nms=apply_nms,
                solov2_score_thr=solov2_score_thr,
                unwarp=unwarp,
            )
            if complete:
                return _extraction_row(
                    record,
                    feature_score_threshold=score_threshold,
                    status="ok",
                    cache_hit=True,
                    cache_dir=cache_dir,
                    files=files,
                    minutiae_count=cached_count,
                )

        image_path = Path(record.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")
        mask_path = Path(record.mask_path) if record.mask_path else None
        if record.source_kind == "gt_bundle" and (mask_path is None or not mask_path.exists()):
            raise FileNotFoundError(f"GT bundle mask not found: {record.mask_path}")

        cache_dir.mkdir(parents=True, exist_ok=True)
        if record.source_kind == "gt_bundle":
            image_tensor, mask_tensor, input_shape_hw = helpers["preprocess_saved_masked_input"](
                masked_image_path=image_path,
                mask_path=mask_path,
                save_preprocess_dir=cache_dir / "preprocess",
            )
        else:
            image_tensor, mask_tensor, input_shape_hw = helpers["preprocess_input_image"](
                image_path=image_path,
                save_preprocess_dir=cache_dir / "preprocess",
                solov2_score_thr=solov2_score_thr,
            )
        outputs = helpers["run_inference"](
            model=model,
            image_tensor=image_tensor,
            mask_tensor=mask_tensor,
            device=device,
        )
        minutiae_rows = helpers["decode_minutiae_rows"](
            outputs=outputs,
            input_shape_hw=input_shape_hw,
            score_threshold=score_threshold,
            apply_nms=apply_nms,
        )
        unwarp_applied = False
        if unwarp == "gradient":
            gray_full = image_tensor.detach().cpu().numpy()[0, 0]
            warped_rows, unwarped_mask = helpers["unwarp_minutiae_rows"](
                rows=minutiae_rows,
                gradient_tensor=outputs["gradient"],
                mask_tensor=mask_tensor,
                input_shape_hw=input_shape_hw,
                gray_image=gray_full,
            )
            if warped_rows is not minutiae_rows:
                unwarp_applied = True
                minutiae_rows = warped_rows
                helpers["save_minutiae_csv"](minutiae_rows, files["minutiae_csv"])
                helpers["_save_mask_array_png"](unwarped_mask, files["mask_png"])
            else:
                helpers["save_minutiae_csv"](minutiae_rows, files["minutiae_csv"])
                helpers["_save_mask_png"](mask_tensor, files["mask_png"])
        else:
            helpers["save_minutiae_csv"](minutiae_rows, files["minutiae_csv"])
            helpers["_save_mask_png"](mask_tensor, files["mask_png"])
        orientation_npy, ridge_period_npy = helpers["save_pose_sidecars"](outputs, cache_dir)

        metadata = {
            "cache_version": CACHE_VERSION,
            "sample_uid": record.sample_uid,
            "record": asdict(record),
            "image_path": record.image_path,
            "mask_path": record.mask_path,
            "source_kind": record.source_kind,
            "weights_path": str(weights_path.resolve()),
            "feature_score_threshold": float(score_threshold),
            "solov2_score_thr": float(solov2_score_thr),
            "minutia_nms_enabled": bool(apply_nms),
            "unwarp": unwarp,
            "unwarp_applied": bool(unwarp_applied),
            "minutiae_count": len(minutiae_rows),
            "artifacts": {
                "minutiae_csv": str(files["minutiae_csv"].resolve()),
                "mask_png": str(files["mask_png"].resolve()),
                "orientation_npy": str(Path(orientation_npy).resolve()),
                "ridge_period_npy": str(Path(ridge_period_npy).resolve()),
            },
            "preprocessing": {
                "input_source": record.source_kind,
                "canonical_preprocess": (
                    "generated_ground_truth_bundle"
                    if record.source_kind == "gt_bundle"
                    else "preprocess.py"
                ),
                "input_shape_hw": [int(input_shape_hw[0]), int(input_shape_hw[1])],
            },
        }
        files["metadata_json"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return _extraction_row(
            record,
            feature_score_threshold=score_threshold,
            status="ok",
            cache_hit=False,
            cache_dir=cache_dir,
            files=files,
            minutiae_count=len(minutiae_rows),
        )
    except Exception as exc:
        return _extraction_row(
            record,
            feature_score_threshold=score_threshold,
            status="error",
            cache_hit=False,
            cache_dir=cache_dir,
            files=files,
            minutiae_count=None,
            error=str(exc),
        )


def run_inference_for_records(
    records: list[ImageRecord],
    *,
    helpers: dict[str, Any] | None,
    model: Any | None,
    device: Any | None,
    device_arg: str,
    weights_path: Path,
    score_threshold: float,
    solov2_score_thr: float,
    cache_root: Path,
    reuse_cache: bool,
    apply_nms: bool,
    unwarp: str = "none",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        cache_dir = cache_root / f"score_{_threshold_label(score_threshold)}" / record.cache_key
        files = _cache_files(cache_dir)
        if reuse_cache:
            complete, cached_count = _cached_extraction_is_complete(
                files,
                record=record,
                weights_path=weights_path,
                score_threshold=score_threshold,
                apply_nms=apply_nms,
                solov2_score_thr=solov2_score_thr,
                unwarp=unwarp,
            )
            if complete:
                rows.append(
                    _extraction_row(
                        record,
                        feature_score_threshold=score_threshold,
                        status="ok",
                        cache_hit=True,
                        cache_dir=cache_dir,
                        files=files,
                        minutiae_count=cached_count,
                    )
                )
                if index == 1 or index % 25 == 0 or index == len(records):
                    ok_count = sum(1 for row in rows if row.get("status") == "ok")
                    print(
                        f"[inference score={_threshold_label(score_threshold)}] "
                        f"processed {index}/{len(records)} ok={ok_count} errors={len(rows) - ok_count}",
                        flush=True,
                    )
                continue

        if helpers is None:
            helpers = _load_infer_helpers()
        if device is None:
            device = helpers["_resolve_device"](str(device_arg))
        if model is None:
            model = helpers["load_checkpoint_model"](weights_path, device)

        rows.append(
            extract_image(
                record,
                helpers=helpers,
                model=model,
                device=device,
                weights_path=weights_path,
                score_threshold=score_threshold,
                solov2_score_thr=solov2_score_thr,
                cache_root=cache_root,
                reuse_cache=reuse_cache,
                apply_nms=apply_nms,
                unwarp=unwarp,
            )
        )
        if index == 1 or index % 25 == 0 or index == len(records):
            ok_count = sum(1 for row in rows if row.get("status") == "ok")
            print(
                f"[inference score={_threshold_label(score_threshold)}] "
                f"processed {index}/{len(records)} ok={ok_count} errors={len(rows) - ok_count}",
                flush=True,
            )
    return rows


def _extracted_from_row(row: dict[str, Any]) -> ExtractedImage:
    count_value = str(row.get("minutiae_count", "")).strip()
    count = int(count_value) if count_value and count_value != "None" else None
    return ExtractedImage(
        sample_uid=str(row.get("sample_uid") or ""),
        status=str(row.get("status") or ""),
        minutiae_csv=str(row.get("minutiae_csv") or ""),
        mask_png=str(row.get("mask_png") or ""),
        orientation_npy=str(row.get("orientation_npy") or ""),
        ridge_period_npy=str(row.get("ridge_period_npy") or ""),
        minutiae_count=count,
        error=str(row.get("error") or ""),
    )


def _match_pair_task(job: dict[str, Any]) -> dict[str, Any]:
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        import main as mcc_main

        score, sim_matrix = mcc_main.match_minutiae_csv(
            path_a=Path(job["a_minutiae_csv"]),
            path_b=Path(job["b_minutiae_csv"]),
            method=str(job["method"]),
            mask_path_a=Path(job["a_mask_png"]),
            mask_path_b=Path(job["b_mask_png"]),
            orientation_path_a=Path(job["a_orientation_npy"]),
            orientation_path_b=Path(job["b_orientation_npy"]),
            ridge_period_path_a=Path(job["a_ridge_period_npy"]),
            ridge_period_path_b=Path(job["b_ridge_period_npy"]),
            overlap_mode="auto",
        )
        shape = getattr(sim_matrix, "shape", ())
        job.update(
            {
                "status": "ok",
                "score": float(score),
                "similarity_matrix_shape": "x".join(str(dim) for dim in shape),
                "error": "",
            }
        )
    except Exception as exc:
        job.update(
            {
                "status": "error",
                "score": "",
                "similarity_matrix_shape": "",
                "error": str(exc),
            }
        )
    return job


def _pair_row(pair: PairSpec, index: int, *, feature_score_threshold: float | None = None) -> dict[str, Any]:
    row = {
        "pair_id": f"pair_{index:06d}",
        "label": pair.label,
        "same_identity": str(pair.same_identity).lower(),
        "a_sample_uid": pair.a.sample_uid,
        "a_dataset": pair.a.dataset,
        "a_subject_id": pair.a.subject_id,
        "a_finger_id": pair.a.finger_id,
        "a_acquisition_id": pair.a.acquisition_id,
        "a_view_index": pair.a.view_index,
        "a_image_path": pair.a.image_path,
        "b_sample_uid": pair.b.sample_uid,
        "b_dataset": pair.b.dataset,
        "b_subject_id": pair.b.subject_id,
        "b_finger_id": pair.b.finger_id,
        "b_acquisition_id": pair.b.acquisition_id,
        "b_view_index": pair.b.view_index,
        "b_image_path": pair.b.image_path,
    }
    if feature_score_threshold is not None:
        row["feature_score_threshold"] = _threshold_label(feature_score_threshold)
    return row


def _make_match_jobs(
    pairs: list[PairSpec],
    extraction_rows: list[dict[str, Any]],
    *,
    method: str,
    feature_score_threshold: float,
    filtered_minutiae_root: Path,
    mcc_max_minutiae: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    extracted = {_extracted_from_row(row).sample_uid: _extracted_from_row(row) for row in extraction_rows}
    jobs: list[dict[str, Any]] = []
    immediate_rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs, start=1):
        row = _pair_row(pair, index, feature_score_threshold=feature_score_threshold)
        a = extracted.get(pair.a.sample_uid)
        b = extracted.get(pair.b.sample_uid)
        if a is None or b is None or a.status != "ok" or b.status != "ok":
            missing = []
            if a is None:
                missing.append("a_missing")
            elif a.status != "ok":
                missing.append(f"a_error:{a.error}")
            if b is None:
                missing.append("b_missing")
            elif b.status != "ok":
                missing.append(f"b_error:{b.error}")
            row.update(
                {
                    "method": method,
                    "status": "error",
                    "score": "",
                    "similarity_matrix_shape": "",
                    "a_minutiae_count": "" if a is None else a.minutiae_count,
                    "b_minutiae_count": "" if b is None else b.minutiae_count,
                    "a_minutiae_csv": "" if a is None else a.minutiae_csv,
                    "b_minutiae_csv": "" if b is None else b.minutiae_csv,
                    "a_mask_png": "" if a is None else a.mask_png,
                    "b_mask_png": "" if b is None else b.mask_png,
                    "error": "; ".join(missing),
                }
            )
            immediate_rows.append(row)
            continue

        a_minutiae_csv = _minutiae_csv_for_mcc(
            a,
            filtered_minutiae_root=filtered_minutiae_root,
            feature_score_threshold=feature_score_threshold,
            mcc_max_minutiae=mcc_max_minutiae,
        )
        b_minutiae_csv = _minutiae_csv_for_mcc(
            b,
            filtered_minutiae_root=filtered_minutiae_root,
            feature_score_threshold=feature_score_threshold,
            mcc_max_minutiae=mcc_max_minutiae,
        )

        row.update(
            {
                "method": method,
                "status": "pending",
                "score": "",
                "similarity_matrix_shape": "",
                "a_minutiae_count": a.minutiae_count,
                "b_minutiae_count": b.minutiae_count,
                "a_minutiae_csv": a_minutiae_csv,
                "b_minutiae_csv": b_minutiae_csv,
                "a_mask_png": a.mask_png,
                "b_mask_png": b.mask_png,
                "a_orientation_npy": a.orientation_npy,
                "b_orientation_npy": b.orientation_npy,
                "a_ridge_period_npy": a.ridge_period_npy,
                "b_ridge_period_npy": b.ridge_period_npy,
                "error": "",
            }
        )
        jobs.append(row)
    return jobs, immediate_rows


def _minutiae_csv_for_mcc(
    extracted: ExtractedImage,
    *,
    filtered_minutiae_root: Path,
    feature_score_threshold: float,
    mcc_max_minutiae: int | None,
) -> str:
    if mcc_max_minutiae is None:
        return extracted.minutiae_csv
    if mcc_max_minutiae <= 0:
        raise ValueError("--mcc-max-minutiae must be positive when provided")

    source = Path(extracted.minutiae_csv)
    target = (
        filtered_minutiae_root
        / f"score_{_threshold_label(feature_score_threshold)}"
        / _slug(extracted.sample_uid)
        / f"minutiae_top_{mcc_max_minutiae}.csv"
    )
    if target.exists() and target.stat().st_size > 0 and target.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return str(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", newline="", encoding="utf-8") as src, target.open("w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames or ["x", "y", "angle", "score"]
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        rows = list(reader)
        rows.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
        for row in rows[:mcc_max_minutiae]:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return str(target)


def _resolve_worker_count(value: str, *, default_auto: int) -> int:
    text = str(value).strip().lower()
    if text == "auto":
        return max(1, int(default_auto))
    count = int(text)
    if count < 1:
        raise ValueError("worker count must be at least 1")
    return count


def run_mcc_matching(
    pairs: list[PairSpec],
    extraction_rows: list[dict[str, Any]],
    *,
    method: str,
    feature_score_threshold: float,
    mcc_workers: int,
    filtered_minutiae_root: Path,
    mcc_max_minutiae: int | None,
) -> list[dict[str, Any]]:
    jobs, rows = _make_match_jobs(
        pairs,
        extraction_rows,
        method=method,
        feature_score_threshold=feature_score_threshold,
        filtered_minutiae_root=filtered_minutiae_root,
        mcc_max_minutiae=mcc_max_minutiae,
    )
    if not jobs:
        return rows

    if mcc_workers == 1:
        for index, job in enumerate(jobs, start=1):
            rows.append(_match_pair_task(job))
            if index == 1 or index % 25 == 0 or index == len(jobs):
                ok_count = sum(1 for row in rows if row.get("status") == "ok")
                print(
                    f"[mcc method={method} score={_threshold_label(feature_score_threshold)}] "
                    f"matched {index}/{len(jobs)} ok={ok_count} total_rows={len(rows)}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=mcc_workers) as executor:
            futures = [executor.submit(_match_pair_task, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if index == 1 or index % 25 == 0 or index == len(futures):
                    ok_count = sum(1 for row in rows if row.get("status") == "ok")
                    print(
                        f"[mcc method={method} score={_threshold_label(feature_score_threshold)}] "
                        f"matched {index}/{len(futures)} ok={ok_count} total_rows={len(rows)}",
                        flush=True,
                    )
    rows.sort(key=lambda row: str(row.get("pair_id") or ""))
    return rows


def compute_confusion_counts(match_rows: list[dict[str, Any]], matching_threshold: float) -> dict[str, Any]:
    tp = fn = fp = tn = 0
    for row in match_rows:
        if row.get("status") != "ok" or str(row.get("score", "")).strip() == "":
            continue
        score = float(row["score"])
        if not math.isfinite(score):
            continue
        actual_same = row.get("label") == "genuine"
        predicted_same = score >= float(matching_threshold)
        if actual_same and predicted_same:
            tp += 1
        elif actual_same and not predicted_same:
            fn += 1
        elif not actual_same and predicted_same:
            fp += 1
        else:
            tn += 1

    genuine_total = tp + fn
    impostor_total = fp + tn
    total = genuine_total + impostor_total
    return {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "TPR": float(tp / genuine_total) if genuine_total else None,
        "FPR": float(fp / impostor_total) if impostor_total else None,
        "TNR": float(tn / impostor_total) if impostor_total else None,
        "FNR": float(fn / genuine_total) if genuine_total else None,
        "accuracy": float((tp + tn) / total) if total else None,
    }


def compute_roc_counts(
    rows_by_feature_threshold: dict[float, list[dict[str, Any]]],
    *,
    matching_thresholds: Iterable[float] | None = None,
) -> list[dict[str, Any]]:
    thresholds = list(matching_thresholds) if matching_thresholds is not None else [index / 100.0 for index in range(1, 101)]
    rows: list[dict[str, Any]] = []
    for feature_score_threshold in sorted(rows_by_feature_threshold):
        match_rows = rows_by_feature_threshold[feature_score_threshold]
        for matching_threshold in thresholds:
            counts = compute_confusion_counts(match_rows, matching_threshold)
            rows.append(
                {
                    "feature_score_threshold": _threshold_label(feature_score_threshold),
                    "matching_threshold": _threshold_label(matching_threshold),
                    **counts,
                }
            )
    return rows


def _write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _ok_error_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "ok": sum(1 for row in rows if row.get("status") == "ok"),
        "error": sum(1 for row in rows if row.get("status") != "ok"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FeatureNet inference, MCC matching, and binary threshold ROC counts for sampled pairs."
    )
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument(
        "--ground-truth-root",
        type=Path,
        default=None,
        help="Generated ground-truth root. When provided, inference replays saved masked_image.png/mask.png bundles.",
    )
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument(
        "--feature-score-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_FEATURE_SCORE_THRESHOLDS),
    )
    parser.add_argument("--genuine-pairs", type=int, default=100)
    parser.add_argument("--impostor-pairs", type=int, default=100)
    parser.add_argument("--side-views", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="MCC methods to evaluate. Defaults to LSA, LSA-R, and LSA-CENTROID.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Deprecated single-method alias. Use --methods for one or more MCC methods.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--mcc-workers", type=str, default="auto")
    parser.add_argument(
        "--mcc-max-minutiae",
        type=int,
        default=None,
        help="Optional cap: match using only the top-scoring N minutiae per image.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-minutia-nms", action="store_true")
    parser.add_argument(
        "--unwarp",
        choices=("none", "gradient"),
        default="gradient",
        help="Route A: warp decoded minutiae into the predicted-gradient canonical frame before MCC.",
    )
    parser.add_argument(
        "--solov2-score-thr",
        type=float,
        default=0.15,
        help="SOLOv2 score threshold for archive/raw canonical preprocessing. Ignored in --ground-truth-root mode.",
    )
    return parser.parse_args()


def main() -> int:
    started_at = time.time()
    args = parse_args()

    archive_root = args.archive_root.resolve()
    ground_truth_root = args.ground_truth_root.resolve() if args.ground_truth_root is not None else None
    weights_path = args.weights_path.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else _default_output_dir().resolve()
    cache_root = args.cache_root.resolve() if args.cache_root is not None else output_dir / "cache"
    feature_score_thresholds = [float(value) for value in args.feature_score_thresholds]
    methods = _unique_methods(
        args.methods
        if args.methods is not None
        else ([args.method] if args.method is not None else DEFAULT_MCC_METHODS)
    )
    side_views = {int(view) for view in args.side_views}
    apply_nms = not bool(args.disable_minutia_nms)

    if not weights_path.exists():
        raise FileNotFoundError(f"weights file not found: {weights_path}")
    if args.genuine_pairs < 0 or args.impostor_pairs < 0:
        raise ValueError("--genuine-pairs and --impostor-pairs must be non-negative")
    if args.mcc_max_minutiae is not None and args.mcc_max_minutiae <= 0:
        raise ValueError("--mcc-max-minutiae must be positive when provided")
    if 0 in side_views:
        raise ValueError("--side-views must not include front view 0")
    if not feature_score_thresholds:
        raise ValueError("--feature-score-thresholds must include at least one value")
    for threshold in feature_score_thresholds:
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("--feature-score-thresholds values must be between 0 and 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    input_mode = "gt_bundle" if ground_truth_root is not None else "archive"
    if ground_truth_root is not None:
        records = discover_ground_truth_bundle_images(ground_truth_root, side_views=side_views)
        if not records:
            raise RuntimeError(f"no generated GT bundle images discovered under: {ground_truth_root}")
    else:
        records = discover_archive_images(archive_root, side_views=side_views)
        if not records:
            raise RuntimeError(f"no raw images discovered under archive root: {archive_root}")
    pairs = sample_binary_pairs(
        records,
        side_views=side_views,
        genuine_count=int(args.genuine_pairs),
        impostor_count=int(args.impostor_pairs),
        seed=int(args.seed),
    )
    unique_records = unique_records_from_pairs(pairs)

    sampled_pairs_path = output_dir / "sampled_pairs.csv"
    _write_csv_rows(sampled_pairs_path, [_pair_row(pair, index) for index, pair in enumerate(pairs, start=1)])

    print(f"Input mode: {input_mode}", flush=True)
    print(f"Discovered images: {len(records)}", flush=True)
    print(f"Sampled pairs: {len(pairs)}", flush=True)
    print(f"Unique images for inference: {len(unique_records)}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"MCC methods: {', '.join(methods)}", flush=True)

    helpers: dict[str, Any] | None = None
    device: Any | None = None
    model: Any | None = None
    auto_mcc = max(1, (os.cpu_count() or 2) - 1)
    mcc_workers = _resolve_worker_count(str(args.mcc_workers), default_auto=auto_mcc)
    print(f"Using device: {args.device}", flush=True)
    print(f"Using MCC workers: {mcc_workers}", flush=True)

    inference_rows_by_threshold: dict[float, list[dict[str, Any]]] = {}
    match_rows_by_method_threshold: dict[str, dict[float, list[dict[str, Any]]]] = {
        method: {} for method in methods
    }
    output_paths: dict[str, str] = {
        "output_dir": str(output_dir),
        "cache_root": str(cache_root),
        "filtered_minutiae_root": str(output_dir / "mcc_minutiae_inputs"),
        "sampled_pairs_csv": str(sampled_pairs_path),
        "summary_json": str(output_dir / "summary.json"),
        "roc_counts_csv": str(output_dir / "roc_counts.csv"),
    }
    single_method = len(methods) == 1

    for score_threshold in feature_score_thresholds:
        label = _threshold_label(score_threshold)
        inference_rows = run_inference_for_records(
            unique_records,
            helpers=helpers,
            model=model,
            device=device,
            device_arg=str(args.device),
            weights_path=weights_path,
            score_threshold=score_threshold,
            solov2_score_thr=float(args.solov2_score_thr),
            cache_root=cache_root,
            reuse_cache=bool(args.reuse_cache),
            apply_nms=apply_nms,
            unwarp=str(args.unwarp),
        )
        inference_rows_by_threshold[score_threshold] = inference_rows
        inference_manifest_path = output_dir / f"inference_manifest_score_{label}.csv"
        _write_csv_rows(inference_manifest_path, inference_rows)
        output_paths[f"inference_manifest_score_{label}_csv"] = str(inference_manifest_path)

        for method in methods:
            print(f"Running MCC method {method} at feature score {label}", flush=True)
            match_rows = run_mcc_matching(
                pairs,
                inference_rows,
                method=method,
                feature_score_threshold=score_threshold,
                mcc_workers=mcc_workers,
                filtered_minutiae_root=output_dir / "mcc_minutiae_inputs",
                mcc_max_minutiae=args.mcc_max_minutiae,
            )
            match_rows_by_method_threshold[method][score_threshold] = match_rows
            method_slug = _method_slug(method)
            mcc_scores_path = (
                output_dir / f"mcc_scores_score_{label}.csv"
                if single_method
                else output_dir / f"mcc_scores_{method_slug}_score_{label}.csv"
            )
            _write_csv_rows(mcc_scores_path, match_rows)
            output_key = (
                f"mcc_scores_score_{label}_csv"
                if single_method
                else f"mcc_scores_{method_slug}_score_{label}_csv"
            )
            output_paths[output_key] = str(mcc_scores_path)

    combined_roc_rows: list[dict[str, Any]] = []
    for method in methods:
        method_roc_rows = compute_roc_counts(match_rows_by_method_threshold[method])
        for row in method_roc_rows:
            combined_roc_rows.append({"method": method, **row})
        if not single_method:
            method_slug = _method_slug(method)
            method_roc_counts_path = output_dir / f"roc_counts_{method_slug}.csv"
            _write_csv_rows(
                method_roc_counts_path,
                [{"method": method, **row} for row in method_roc_rows],
                fieldnames=[
                    "method",
                    "feature_score_threshold",
                    "matching_threshold",
                    "TP",
                    "FN",
                    "FP",
                    "TN",
                    "TPR",
                    "FPR",
                    "TNR",
                    "FNR",
                    "accuracy",
                ],
            )
            output_paths[f"roc_counts_{method_slug}_csv"] = str(method_roc_counts_path)

    roc_counts_path = output_dir / "roc_counts.csv"
    _write_csv_rows(
        roc_counts_path,
        combined_roc_rows if not single_method else [{key: value for key, value in row.items() if key != "method"} for row in combined_roc_rows],
        fieldnames=(
            [
                "method",
                "feature_score_threshold",
                "matching_threshold",
                "TP",
                "FN",
                "FP",
                "TN",
                "TPR",
                "FPR",
                "TNR",
                "FNR",
                "accuracy",
            ]
            if not single_method
            else [
                "feature_score_threshold",
                "matching_threshold",
                "TP",
                "FN",
                "FP",
                "TN",
                "TPR",
                "FPR",
                "TNR",
                "FNR",
                "accuracy",
            ]
        ),
    )

    summary = {
        "config": {
            "input_mode": input_mode,
            "archive_root": str(archive_root),
            "ground_truth_root": None if ground_truth_root is None else str(ground_truth_root),
            "weights_path": str(weights_path),
            "feature_score_thresholds": [_threshold_label(value) for value in feature_score_thresholds],
            "genuine_pairs": int(args.genuine_pairs),
            "impostor_pairs": int(args.impostor_pairs),
            "side_views": sorted(side_views),
            "methods": methods,
            "method": methods[0] if single_method else None,
            "device": str(args.device),
            "mcc_workers": int(mcc_workers),
            "mcc_max_minutiae": args.mcc_max_minutiae,
            "seed": int(args.seed),
            "reuse_cache": bool(args.reuse_cache),
            "minutia_nms_enabled": bool(apply_nms),
            "unwarp": str(args.unwarp),
            "solov2_score_thr": float(args.solov2_score_thr),
        },
        "counts": {
            "discovered_image_count": len(records),
            "sampled_pair_count": len(pairs),
            "sampled_genuine_pair_count": sum(1 for pair in pairs if pair.label == "genuine"),
            "sampled_impostor_pair_count": sum(1 for pair in pairs if pair.label == "impostor"),
            "unique_inference_image_count": len(unique_records),
            "per_feature_score_threshold": {
                _threshold_label(threshold): {
                    "inference": _ok_error_counts(inference_rows_by_threshold[threshold]),
                    "mcc": {
                        method: _ok_error_counts(match_rows_by_method_threshold[method][threshold])
                        for method in methods
                    },
                }
                for threshold in feature_score_thresholds
            },
        },
        "errors": {
            _threshold_label(threshold): {
                "inference": [row for row in inference_rows_by_threshold[threshold] if row.get("status") != "ok"][:200],
                "mcc": {
                    method: [
                        row
                        for row in match_rows_by_method_threshold[method][threshold]
                        if row.get("status") != "ok"
                    ][:200]
                    for method in methods
                },
            }
            for threshold in feature_score_thresholds
        },
        "outputs": output_paths,
        "wall_seconds": round(time.time() - started_at, 3),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved sampled pairs: {sampled_pairs_path}", flush=True)
    print(f"Saved ROC counts: {roc_counts_path}", flush=True)
    print(f"Saved summary: {output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

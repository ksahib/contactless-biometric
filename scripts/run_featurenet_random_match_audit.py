from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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

import hashlib

import numpy as np
from dataclasses import asdict, dataclass

def prepend_workspace_site_packages() -> None:
    site_packages = REPO_ROOT / ".venv" / "lib" / "site-packages"
    if site_packages.exists():
        sys.path.insert(0, str(site_packages))


prepend_workspace_site_packages()

from featurenet.models.infer import (
    _resolve_device,
    decode_minutiae_rows,
    draw_minutiae_overlay,
    export_reconstruction_visualization,
    filter_minutiae_rows_for_overlay,
    load_checkpoint_model,
    mask_tensor_to_image,
    preprocess_input_bgr,
    run_inference,
    save_minutiae_csv,
    save_pose_sidecars,
)
from featurenet.models.match_infer import _crop_distal_phalanx_with_main, _save_mask_png


DEFAULT_WEIGHTS_PATH = REPO_ROOT / "runs" / "featurenet_run_v3_f1_hardneg_recipe_bf16_resume" / "best.pt"
DEFAULT_GROUND_TRUTH_ROOT = REPO_ROOT / "ground_truth" / "DS123_merged_v3"
DEFAULT_ARCHIVE_ROOT = Path("/media/milab-5/732c6478-0497-43ca-885a-72c45d10aea7/sahib/contactless-biometric/archive")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "match_outputs"
GENUINE_BUCKETS = (
    "front_side_same_finger",
    "side_side_same_finger",
)
IMPOSTOR_BUCKETS = (
    "front_front_other_finger_same_type",
    "front_front_other_finger_cross_type",
    "front_side_other_finger_same_type",
    "front_side_other_finger_cross_type",
    "side_side_other_finger_same_type",
    "side_side_other_finger_cross_type",
)
ALL_BUCKETS = GENUINE_BUCKETS + IMPOSTOR_BUCKETS


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    dataset: str
    subject_id: int
    finger_id: int
    acquisition_id: int
    raw_view_index: int
    raw_image_path: str
    role: str | None

    @property
    def image_path(self) -> str:
        return self.raw_image_path

    @property
    def view_group(self) -> str:
        return "front" if self.raw_view_index == 0 else "side"

    @property
    def identity_key(self) -> tuple[str, int, int]:
        return self.dataset, self.subject_id, self.finger_id

    @property
    def cache_key(self) -> str:
        digest = hashlib.sha1(self.raw_image_path.encode("utf-8")).hexdigest()[:10]
        return (
            f"{self.dataset.lower()}"
            f"/s{self.subject_id:03d}"
            f"/f{self.finger_id:02d}"
            f"/a{self.acquisition_id:02d}"
            f"/v{self.raw_view_index:02d}_{digest}"
        )


@dataclass(frozen=True)
class PairSpec:
    bucket: str
    label: str
    view_pair: str
    finger_type_relation: str
    a: SampleRecord
    b: SampleRecord

    @property
    def pair_key(self) -> tuple[str, str, str]:
        ordered = tuple(sorted((self.a.image_path, self.b.image_path)))
        return self.bucket, ordered[0], ordered[1]


@dataclass
class ExtractedImage:
    record: SampleRecord
    minutiae_csv: Path
    mask_png: Path
    cropped_overlay_png: Path
    reconstruction_artifacts: dict[str, str] | None
    orientation_npy: Path
    ridge_period_npy: Path
    metadata_json: Path
    minutiae_count: int


def _default_output_dir() -> Path:
    return DEFAULT_OUTPUT_ROOT / f"random_match_audit_{time.strftime('%Y%m%d_%H%M%S')}"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def discover_samples(
    ground_truth_root: Path,
    archive_root: Path,
    side_views: set[int],
) -> list[SampleRecord]:
    samples_dir = ground_truth_root / "samples"
    if not samples_dir.exists():
        raise FileNotFoundError(f"samples directory not found: {samples_dir}")

    archive_root_resolved = archive_root.resolve()
    discovered: list[SampleRecord] = []
    seen_sample_ids: set[str] = set()
    for meta_path in sorted(samples_dir.glob("*/meta.json")):
        try:
            meta = _read_json(meta_path)
            sample_id = str(meta["sample_id"])
            if sample_id in seen_sample_ids:
                continue
            raw_image_path = Path(str(meta["raw_image_path"])).resolve()
            raw_view_index = int(meta["raw_view_index"])
            if raw_view_index != 0 and raw_view_index not in side_views:
                continue
            if not raw_image_path.exists():
                continue
            if archive_root_resolved not in raw_image_path.parents:
                continue
            discovered.append(
                SampleRecord(
                    sample_id=sample_id,
                    dataset=_normalize_dataset_label(sample_id, meta),
                    subject_id=int(meta["subject_id"]),
                    finger_id=int(meta["finger_id"]),
                    acquisition_id=int(meta["acquisition_id"]),
                    raw_view_index=raw_view_index,
                    raw_image_path=str(raw_image_path),
                    role=_extract_role(meta),
                )
            )
            seen_sample_ids.add(sample_id)
        except Exception as exc:
            print(f"[warn] skipping malformed metadata {meta_path}: {exc}", file=sys.stderr)
    return discovered


def _normalize_dataset_label(sample_id: str, meta: dict[str, Any]) -> str:
    merge_source = meta.get("merge_source")
    if isinstance(merge_source, dict):
        label = merge_source.get("label")
        if isinstance(label, str) and label.strip():
            return label.strip().upper()
    if sample_id.startswith("ds"):
        return sample_id.split("_", 1)[0].upper()
    raise ValueError(f"could not infer dataset label for sample_id={sample_id}")


def _extract_role(meta: dict[str, Any]) -> str | None:
    reconstruction = meta.get("multiview_reconstruction")
    if not isinstance(reconstruction, dict):
        return None
    role = reconstruction.get("role")
    if role is None:
        return None
    return str(role)


def _group_by_identity(records: Iterable[SampleRecord]) -> dict[tuple[str, int, int], list[SampleRecord]]:
    groups: dict[tuple[str, int, int], list[SampleRecord]] = {}
    for record in records:
        groups.setdefault(record.identity_key, []).append(record)
    return groups


def _sample_pairs_from_candidates(
    candidates: list[PairSpec],
    *,
    bucket: str,
    rng: np.random.Generator,
    max_pairs: int,
) -> list[PairSpec]:
    deduped: list[PairSpec] = []
    seen: set[tuple[str, str, str]] = set()
    for pair in candidates:
        key = pair.pair_key
        if key in seen:
            continue
        seen.add(key)
        deduped.append(pair)
    if len(deduped) <= max_pairs:
        return deduped
    indices = rng.choice(len(deduped), size=max_pairs, replace=False)
    sampled = [deduped[int(index)] for index in sorted(indices)]
    print(f"[info] sampled {len(sampled)}/{len(deduped)} pairs for bucket={bucket}")
    return sampled


def build_genuine_pairs(
    records: list[SampleRecord],
    *,
    rng: np.random.Generator,
    max_pairs_per_bucket: int,
) -> dict[str, list[PairSpec]]:
    grouped = _group_by_identity(records)
    front_side_candidates: list[PairSpec] = []
    side_side_candidates: list[PairSpec] = []
    for group_records in grouped.values():
        fronts = [record for record in group_records if record.view_group == "front"]
        sides = [record for record in group_records if record.view_group == "side"]
        for front in fronts:
            for side in sides:
                front_side_candidates.append(
                    PairSpec(
                        bucket="front_side_same_finger",
                        label="genuine",
                        view_pair="front_side",
                        finger_type_relation="same_type",
                        a=front,
                        b=side,
                    )
                )
        for idx, side_a in enumerate(sides):
            for side_b in sides[idx + 1 :]:
                side_side_candidates.append(
                    PairSpec(
                        bucket="side_side_same_finger",
                        label="genuine",
                        view_pair="side_side",
                        finger_type_relation="same_type",
                        a=side_a,
                        b=side_b,
                    )
                )
    return {
        "front_side_same_finger": _sample_pairs_from_candidates(
            front_side_candidates, bucket="front_side_same_finger", rng=rng, max_pairs=max_pairs_per_bucket
        ),
        "side_side_same_finger": _sample_pairs_from_candidates(
            side_side_candidates, bucket="side_side_same_finger", rng=rng, max_pairs=max_pairs_per_bucket
        ),
    }


def _is_impostor_pair(a: SampleRecord, b: SampleRecord) -> bool:
    return not (a.subject_id == b.subject_id and a.finger_id == b.finger_id)


def _bucket_for_impostor_pair(a: SampleRecord, b: SampleRecord) -> tuple[str, str, str]:
    if not _is_impostor_pair(a, b):
        raise ValueError("pair is not an impostor")
    same_type = a.finger_id == b.finger_id
    relation = "same_type" if same_type else "cross_type"
    ordered_views = tuple(sorted((a.view_group, b.view_group)))
    if ordered_views == ("front", "front"):
        view_pair = "front_front"
    elif ordered_views == ("front", "side"):
        view_pair = "front_side"
    elif ordered_views == ("side", "side"):
        view_pair = "side_side"
    else:
        raise ValueError(f"unsupported view pairing: {ordered_views}")
    return f"{view_pair}_other_finger_{relation}", view_pair, relation


def build_impostor_pairs(
    records: list[SampleRecord],
    *,
    rng: np.random.Generator,
    max_pairs_per_bucket: int,
) -> dict[str, list[PairSpec]]:
    fronts = [record for record in records if record.view_group == "front"]
    sides = [record for record in records if record.view_group == "side"]
    fronts_by_finger = _group_by_finger_id(fronts)
    sides_by_finger = _group_by_finger_id(sides)
    all_finger_ids = sorted({record.finger_id for record in records})
    sampled: dict[str, list[PairSpec]] = {}

    sampled["front_front_other_finger_same_type"] = _sample_same_type_impostor_bucket(
        bucket="front_front_other_finger_same_type",
        left_by_finger=fronts_by_finger,
        right_by_finger=fronts_by_finger,
        same_pool=True,
        rng=rng,
        max_pairs=max_pairs_per_bucket,
    )
    sampled["front_side_other_finger_same_type"] = _sample_same_type_impostor_bucket(
        bucket="front_side_other_finger_same_type",
        left_by_finger=fronts_by_finger,
        right_by_finger=sides_by_finger,
        same_pool=False,
        rng=rng,
        max_pairs=max_pairs_per_bucket,
    )
    sampled["side_side_other_finger_same_type"] = _sample_same_type_impostor_bucket(
        bucket="side_side_other_finger_same_type",
        left_by_finger=sides_by_finger,
        right_by_finger=sides_by_finger,
        same_pool=True,
        rng=rng,
        max_pairs=max_pairs_per_bucket,
    )
    sampled["front_front_other_finger_cross_type"] = _sample_cross_type_impostor_bucket(
        bucket="front_front_other_finger_cross_type",
        left_records=fronts,
        right_records=fronts,
        all_finger_ids=all_finger_ids,
        same_pool=True,
        rng=rng,
        max_pairs=max_pairs_per_bucket,
    )
    sampled["front_side_other_finger_cross_type"] = _sample_cross_type_impostor_bucket(
        bucket="front_side_other_finger_cross_type",
        left_records=fronts,
        right_records=sides,
        all_finger_ids=all_finger_ids,
        same_pool=False,
        rng=rng,
        max_pairs=max_pairs_per_bucket,
    )
    sampled["side_side_other_finger_cross_type"] = _sample_cross_type_impostor_bucket(
        bucket="side_side_other_finger_cross_type",
        left_records=sides,
        right_records=sides,
        all_finger_ids=all_finger_ids,
        same_pool=True,
        rng=rng,
        max_pairs=max_pairs_per_bucket,
    )
    return sampled


def _group_by_finger_id(records: Iterable[SampleRecord]) -> dict[int, list[SampleRecord]]:
    grouped: dict[int, list[SampleRecord]] = {}
    for record in records:
        grouped.setdefault(record.finger_id, []).append(record)
    return grouped


def _sample_same_type_impostor_bucket(
    *,
    bucket: str,
    left_by_finger: dict[int, list[SampleRecord]],
    right_by_finger: dict[int, list[SampleRecord]],
    same_pool: bool,
    rng: np.random.Generator,
    max_pairs: int,
) -> list[PairSpec]:
    eligible_fingers = sorted(set(left_by_finger) & set(right_by_finger))
    results: list[PairSpec] = []
    seen: set[tuple[str, str, str]] = set()
    max_attempts = max(1000, max_pairs * 200)
    attempts = 0
    while len(results) < max_pairs and attempts < max_attempts and eligible_fingers:
        attempts += 1
        finger_id = eligible_fingers[int(rng.integers(0, len(eligible_fingers)))]
        left_pool = left_by_finger[finger_id]
        right_pool = right_by_finger[finger_id]
        if len(left_pool) < 1 or len(right_pool) < 1:
            continue
        left = left_pool[int(rng.integers(0, len(left_pool)))]
        if same_pool:
            if len(right_pool) < 2:
                continue
            right = right_pool[int(rng.integers(0, len(right_pool)))]
            if left.image_path == right.image_path:
                continue
        else:
            right = right_pool[int(rng.integers(0, len(right_pool)))]
        if not _is_impostor_pair(left, right):
            continue
        pair_bucket, view_pair, relation = _bucket_for_impostor_pair(left, right)
        pair = PairSpec(
            bucket=pair_bucket,
            label="impostor",
            view_pair=view_pair,
            finger_type_relation=relation,
            a=left,
            b=right,
        )
        if pair.pair_key in seen:
            continue
        seen.add(pair.pair_key)
        results.append(pair)
    return results


def _sample_cross_type_impostor_bucket(
    *,
    bucket: str,
    left_records: list[SampleRecord],
    right_records: list[SampleRecord],
    all_finger_ids: list[int],
    same_pool: bool,
    rng: np.random.Generator,
    max_pairs: int,
) -> list[PairSpec]:
    by_finger_right = _group_by_finger_id(right_records)
    results: list[PairSpec] = []
    seen: set[tuple[str, str, str]] = set()
    max_attempts = max(1000, max_pairs * 200)
    attempts = 0
    while len(results) < max_pairs and attempts < max_attempts and left_records:
        attempts += 1
        left = left_records[int(rng.integers(0, len(left_records)))]
        other_finger_ids = [finger_id for finger_id in all_finger_ids if finger_id != left.finger_id and by_finger_right.get(finger_id)]
        if not other_finger_ids:
            continue
        target_finger = other_finger_ids[int(rng.integers(0, len(other_finger_ids)))]
        right_pool = by_finger_right[target_finger]
        right = right_pool[int(rng.integers(0, len(right_pool)))]
        if same_pool and left.image_path == right.image_path:
            continue
        if not _is_impostor_pair(left, right):
            continue
        pair_bucket, view_pair, relation = _bucket_for_impostor_pair(left, right)
        pair = PairSpec(
            bucket=pair_bucket,
            label="impostor",
            view_pair=view_pair,
            finger_type_relation=relation,
            a=left,
            b=right,
        )
        if pair.pair_key in seen:
            continue
        seen.add(pair.pair_key)
        results.append(pair)
    return results


def _count_minutiae_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def _cache_files(cache_dir: Path) -> dict[str, Path]:
    return {
        "minutiae_csv": cache_dir / "minutiae.csv",
        "mask_png": cache_dir / "mask.png",
        "cropped_overlay_png": cache_dir / "cropped_minutiae_overlay.png",
        "orientation_npy": cache_dir / "orientation.npy",
        "ridge_period_npy": cache_dir / "ridge_period.npy",
        "metadata_json": cache_dir / "metadata.json",
    }


def _cached_extraction_is_complete(files: dict[str, Path]) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in files.values())


def extract_image(
    record: SampleRecord,
    *,
    cache_root: Path,
    model: Any,
    device: Any,
    score_threshold: float,
    reuse_cache: bool,
) -> ExtractedImage:
    cache_dir = cache_root / record.cache_key
    files = _cache_files(cache_dir)
    if reuse_cache and _cached_extraction_is_complete(files):
        try:
            metadata = _read_json(files["metadata_json"])
            minutiae_count = int(metadata.get("minutiae_count", _count_minutiae_rows(files["minutiae_csv"])))
        except Exception:
            minutiae_count = _count_minutiae_rows(files["minutiae_csv"])
        return ExtractedImage(
            record=record,
            minutiae_csv=files["minutiae_csv"],
            mask_png=files["mask_png"],
            cropped_overlay_png=files["cropped_overlay_png"],
            reconstruction_artifacts=metadata.get("reconstruction_artifacts") if isinstance(metadata.get("reconstruction_artifacts"), dict) else None,
            orientation_npy=files["orientation_npy"],
            ridge_period_npy=files["ridge_period_npy"],
            metadata_json=files["metadata_json"],
            minutiae_count=minutiae_count,
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    crop_result = _crop_distal_phalanx_with_main(
        image_path=Path(record.image_path),
        crop_output_dir=cache_dir / "crop",
    )
    image_tensor, mask_tensor, input_shape_hw = preprocess_input_bgr(
        full_bgr=crop_result["inference_bgr"],
        save_preprocess_dir=cache_dir / "preprocess",
    )
    outputs = run_inference(model=model, image_tensor=image_tensor, mask_tensor=mask_tensor, device=device)
    minutiae_rows = decode_minutiae_rows(
        outputs=outputs,
        input_shape_hw=input_shape_hw,
        score_threshold=score_threshold,
        apply_nms=True,
    )

    save_minutiae_csv(minutiae_rows, files["minutiae_csv"])
    orientation_npy, ridge_period_npy = save_pose_sidecars(outputs, cache_dir)
    _save_mask_png(mask_tensor, files["mask_png"])
    overlay_mask = mask_tensor_to_image(mask_tensor)
    overlay_rows = filter_minutiae_rows_for_overlay(minutiae_rows, min_score=0.8, mask=overlay_mask)
    cropped_overlay = draw_minutiae_overlay(crop_result["inference_bgr"], overlay_rows, mask=overlay_mask)
    if not cv2.imwrite(str(files["cropped_overlay_png"]), cropped_overlay):
        raise RuntimeError(f"failed to write cropped minutiae overlay to {files['cropped_overlay_png']}")
    reconstruction_artifacts = export_reconstruction_visualization(
        raw_image_path=Path(record.image_path),
        raw_image_shape_hw=(int(crop_result["full_bgr"].shape[0]), int(crop_result["full_bgr"].shape[1])),
        crop_bbox_xyxy=tuple(int(v) for v in crop_result["crop_bbox"]),
        minutiae_rows=overlay_rows,
        output_dir=cache_dir / "reconstruction",
    )

    metadata = {
        "record": asdict(record),
        "minutiae_count": len(minutiae_rows),
        "overlay_minutiae_count": len(overlay_rows),
        "overlay_minutia_score_threshold": 0.8,
        "minutia_score_threshold": float(score_threshold),
        "crop": {
            "crop_bbox_xyxy": list(crop_result["crop_bbox"]),
            "crop_mode": crop_result["crop_mode"],
            "fallback_reason": crop_result["fallback_reason"],
            "original_shape_hw": [int(crop_result["full_bgr"].shape[0]), int(crop_result["full_bgr"].shape[1])],
            "cropped_shape_hw": [int(crop_result["cropped_bgr"].shape[0]), int(crop_result["cropped_bgr"].shape[1])],
            "coarse_mask_pixels": int(crop_result["coarse_mask_pixels"]),
            "distal_mask_pixels": int(crop_result["distal_mask_pixels"]),
        },
        "artifacts": {
            "minutiae_csv": str(files["minutiae_csv"].resolve()),
            "mask_png": str(files["mask_png"].resolve()),
            "cropped_overlay_png": str(files["cropped_overlay_png"].resolve()),
            "orientation_npy": str(orientation_npy.resolve()),
            "ridge_period_npy": str(ridge_period_npy.resolve()),
        },
        "reconstruction_artifacts": reconstruction_artifacts,
    }
    files["metadata_json"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return ExtractedImage(
        record=record,
        minutiae_csv=files["minutiae_csv"],
        mask_png=files["mask_png"],
        cropped_overlay_png=files["cropped_overlay_png"],
        reconstruction_artifacts=reconstruction_artifacts,
        orientation_npy=orientation_npy,
        ridge_period_npy=ridge_period_npy,
        metadata_json=files["metadata_json"],
        minutiae_count=len(minutiae_rows),
    )


def _pair_row_base(pair: PairSpec) -> dict[str, Any]:
    return {
        "bucket": pair.bucket,
        "label": pair.label,
        "view_pair": pair.view_pair,
        "finger_type_relation": pair.finger_type_relation,
        "dataset_a": pair.a.dataset,
        "subject_id_a": pair.a.subject_id,
        "finger_id_a": pair.a.finger_id,
        "acquisition_id_a": pair.a.acquisition_id,
        "raw_view_index_a": pair.a.raw_view_index,
        "role_a": pair.a.role or "",
        "sample_id_a": pair.a.sample_id,
        "image_path_a": pair.a.image_path,
        "dataset_b": pair.b.dataset,
        "subject_id_b": pair.b.subject_id,
        "finger_id_b": pair.b.finger_id,
        "acquisition_id_b": pair.b.acquisition_id,
        "raw_view_index_b": pair.b.raw_view_index,
        "role_b": pair.b.role or "",
        "sample_id_b": pair.b.sample_id,
        "image_path_b": pair.b.image_path,
    }


def score_pairs(
    pairs: list[PairSpec],
    *,
    cache_root: Path,
    model: Any,
    device: Any,
    score_threshold: float,
    reuse_cache: bool,
    method: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import main as mcc_main

    extraction_cache: dict[str, ExtractedImage] = {}
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def get_extracted(record: SampleRecord) -> ExtractedImage:
        cached = extraction_cache.get(record.image_path)
        if cached is None:
            cached = extract_image(
                record,
                cache_root=cache_root,
                model=model,
                device=device,
                score_threshold=score_threshold,
                reuse_cache=reuse_cache,
            )
            extraction_cache[record.image_path] = cached
        return cached

    for index, pair in enumerate(pairs, start=1):
        row = _pair_row_base(pair)
        try:
            extracted_a = get_extracted(pair.a)
            extracted_b = get_extracted(pair.b)
            score, sim_matrix = mcc_main.match_minutiae_csv(
                path_a=extracted_a.minutiae_csv,
                path_b=extracted_b.minutiae_csv,
                method=method,
                mask_path_a=extracted_a.mask_png,
                mask_path_b=extracted_b.mask_png,
                orientation_path_a=extracted_a.orientation_npy,
                orientation_path_b=extracted_b.orientation_npy,
                ridge_period_path_a=extracted_a.ridge_period_npy,
                ridge_period_path_b=extracted_b.ridge_period_npy,
                overlap_mode="auto",
            )
            row.update(
                {
                    "status": "ok",
                    "score": float(score),
                    "method": method,
                    "similarity_matrix_shape": "x".join(str(dim) for dim in np.asarray(sim_matrix).shape),
                    "minutiae_count_a": extracted_a.minutiae_count,
                    "minutiae_count_b": extracted_b.minutiae_count,
                    "minutiae_csv_a": str(extracted_a.minutiae_csv.resolve()),
                    "minutiae_csv_b": str(extracted_b.minutiae_csv.resolve()),
                    "mask_png_a": str(extracted_a.mask_png.resolve()),
                    "mask_png_b": str(extracted_b.mask_png.resolve()),
                    "error": "",
                }
            )
        except Exception as exc:
            error = {
                "pair_index": index,
                "bucket": pair.bucket,
                "image_path_a": pair.a.image_path,
                "image_path_b": pair.b.image_path,
                "error": str(exc),
            }
            errors.append(error)
            row.update(
                {
                    "status": "error",
                    "score": "",
                    "method": method,
                    "similarity_matrix_shape": "",
                    "minutiae_count_a": "",
                    "minutiae_count_b": "",
                    "minutiae_csv_a": "",
                    "minutiae_csv_b": "",
                    "mask_png_a": "",
                    "mask_png_b": "",
                    "error": str(exc),
                }
            )
        rows.append(row)
        if index == 1 or index % 25 == 0 or index == len(pairs):
            ok_count = sum(1 for item in rows if item["status"] == "ok")
            print(f"[progress] scored {index}/{len(pairs)} pairs ok={ok_count} errors={len(errors)}")
    return rows, errors


def _finite_or_none(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return float(value)


def describe_scores(scores: Iterable[float]) -> dict[str, Any]:
    values = np.asarray(list(scores), dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _summarize_rows(rows: list[dict[str, Any]], *, predicate: Any) -> dict[str, Any]:
    scores = [float(row["score"]) for row in rows if row.get("status") == "ok" and predicate(row)]
    return describe_scores(scores)


def summarize_pair_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_stats = {bucket: _summarize_rows(rows, predicate=lambda row, bucket=bucket: row.get("bucket") == bucket) for bucket in ALL_BUCKETS}
    rollups = {
        "genuine_overall": _summarize_rows(rows, predicate=lambda row: row.get("label") == "genuine"),
        "impostor_overall": _summarize_rows(rows, predicate=lambda row: row.get("label") == "impostor"),
        "front_front_overall": _summarize_rows(rows, predicate=lambda row: row.get("view_pair") == "front_front"),
        "front_side_overall": _summarize_rows(rows, predicate=lambda row: row.get("view_pair") == "front_side"),
        "side_side_overall": _summarize_rows(rows, predicate=lambda row: row.get("view_pair") == "side_side"),
        "same_type_impostor_overall": _summarize_rows(
            rows,
            predicate=lambda row: row.get("label") == "impostor" and row.get("finger_type_relation") == "same_type",
        ),
        "cross_type_impostor_overall": _summarize_rows(
            rows,
            predicate=lambda row: row.get("label") == "impostor" and row.get("finger_type_relation") == "cross_type",
        ),
    }
    return {"buckets": bucket_stats, "rollups": rollups}


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _stats_line(name: str, stats: dict[str, Any]) -> str:
    return (
        f"{name}: count={stats.get('count')} "
        f"min={_finite_or_none_str(stats.get('min'))} "
        f"mean={_finite_or_none_str(stats.get('mean'))} "
        f"max={_finite_or_none_str(stats.get('max'))}"
    )


def _finite_or_none_str(value: Any) -> str:
    if value is None:
        return "None"
    return f"{float(value):.6f}"


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "Random FeatureNet Matching Audit",
        "",
        "Bucket Stats",
    ]
    for bucket in ALL_BUCKETS:
        lines.append(_stats_line(bucket, summary["stats"]["buckets"][bucket]))
    lines.extend(
        [
            "",
            "Rollups",
            _stats_line("genuine_overall", summary["stats"]["rollups"]["genuine_overall"]),
            _stats_line("impostor_overall", summary["stats"]["rollups"]["impostor_overall"]),
            _stats_line("front_front_overall", summary["stats"]["rollups"]["front_front_overall"]),
            _stats_line("front_side_overall", summary["stats"]["rollups"]["front_side_overall"]),
            _stats_line("side_side_overall", summary["stats"]["rollups"]["side_side_overall"]),
            _stats_line("same_type_impostor_overall", summary["stats"]["rollups"]["same_type_impostor_overall"]),
            _stats_line("cross_type_impostor_overall", summary["stats"]["rollups"]["cross_type_impostor_overall"]),
            "",
            f"scored_pairs={summary['counts']['scored_pair_count']}",
            f"errors={summary['counts']['error_pair_count']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a random FeatureNet matching audit over genuine and impostor pairs.")
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH, help="FeatureNet checkpoint path.")
    parser.add_argument("--ground-truth-root", type=Path, default=DEFAULT_GROUND_TRUTH_ROOT, help="Merged ground-truth root.")
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT, help="Archive root containing raw contactless images.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--method", type=str, default="LSA-R", help="MCC method passed to main.match_minutiae_csv.")
    parser.add_argument("--side-views", type=int, nargs="+", default=[1, 2], help="Side raw view indices.")
    parser.add_argument("--pairs-per-bucket", type=int, default=1000, help="Maximum sampled pairs per bucket.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for pair sampling.")
    parser.add_argument("--minutia-score-threshold", type=float, default=0.6, help="FeatureNet minutia score threshold.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: match_outputs/random_match_audit_<timestamp>.",
    )
    parser.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True, help="Reuse complete cached image extractions.")
    return parser.parse_args()


def main() -> int:
    started_at = time.time()
    args = parse_args()

    weights_path = args.weights_path.resolve()
    ground_truth_root = args.ground_truth_root.resolve()
    archive_root = args.archive_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else _default_output_dir().resolve()
    cache_root = output_dir / "cache"
    side_views = {int(view) for view in args.side_views}
    rng = np.random.default_rng(int(args.seed))

    if not weights_path.exists():
        raise FileNotFoundError(f"weights file not found: {weights_path}")
    if not ground_truth_root.exists():
        raise FileNotFoundError(f"ground-truth root not found: {ground_truth_root}")
    if not archive_root.exists():
        raise FileNotFoundError(f"archive root not found: {archive_root}")
    if args.pairs_per_bucket <= 0:
        raise ValueError("--pairs-per-bucket must be positive")
    if 0 in side_views:
        raise ValueError("--side-views must not include front view 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    records = discover_samples(ground_truth_root, archive_root, side_views)
    if not records:
        raise RuntimeError("no sample records were discovered")

    genuine_by_bucket = build_genuine_pairs(records, rng=rng, max_pairs_per_bucket=int(args.pairs_per_bucket))
    impostor_by_bucket = build_impostor_pairs(records, rng=rng, max_pairs_per_bucket=int(args.pairs_per_bucket))
    all_pairs = [pair for bucket in ALL_BUCKETS for pair in genuine_by_bucket.get(bucket, []) + impostor_by_bucket.get(bucket, [])]
    if not all_pairs:
        raise RuntimeError("no matching pairs were generated")

    print(f"Discovered samples: {len(records)}")
    for bucket in ALL_BUCKETS:
        count = len(genuine_by_bucket.get(bucket, [])) + len(impostor_by_bucket.get(bucket, []))
        print(f"Planned {bucket}: {count}")
    print(f"Total pairs to score: {len(all_pairs)}")
    print(f"Output directory: {output_dir}")

    device = _resolve_device(args.device)
    model = load_checkpoint_model(weights_path, device)

    pair_rows, errors = score_pairs(
        all_pairs,
        cache_root=cache_root,
        model=model,
        device=device,
        score_threshold=float(args.minutia_score_threshold),
        reuse_cache=bool(args.reuse_cache),
        method=str(args.method),
    )

    stats = summarize_pair_rows(pair_rows)
    ok_rows = [row for row in pair_rows if row.get("status") == "ok"]
    summary = {
        "config": {
            "weights_path": str(weights_path),
            "ground_truth_root": str(ground_truth_root),
            "archive_root": str(archive_root),
            "device": str(device),
            "method": str(args.method),
            "side_views": sorted(side_views),
            "pairs_per_bucket": int(args.pairs_per_bucket),
            "seed": int(args.seed),
            "minutia_score_threshold": float(args.minutia_score_threshold),
            "reuse_cache": bool(args.reuse_cache),
        },
        "counts": {
            "sample_count": len(records),
            "planned_pair_count": len(all_pairs),
            "scored_pair_count": len(ok_rows),
            "error_pair_count": len(errors),
            "planned_bucket_counts": {
                bucket: len(genuine_by_bucket.get(bucket, [])) + len(impostor_by_bucket.get(bucket, [])) for bucket in ALL_BUCKETS
            },
        },
        "stats": stats,
        "errors": errors[:200],
        "error_count_total": len(errors),
        "outputs": {
            "output_dir": str(output_dir),
            "cache_dir": str(cache_root),
            "pairs_csv": str((output_dir / "pairs.csv").resolve()),
            "summary_json": str((output_dir / "summary.json").resolve()),
            "report_txt": str((output_dir / "report.txt").resolve()),
        },
        "wall_seconds": round(time.time() - started_at, 3),
    }

    _write_csv_rows(output_dir / "pairs.csv", pair_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(output_dir / "report.txt", summary)
    print(f"Wrote summary to {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

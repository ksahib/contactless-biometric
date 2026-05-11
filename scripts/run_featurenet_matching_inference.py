from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset"
DEFAULT_GROUND_TRUTH_ROOT = REPO_ROOT / "ground_truth" / "DS123_merged_v5"
DEFAULT_WEIGHTS_PATH = REPO_ROOT / "runs" / "featurenet_run_v3_f1_hardneg_recipe" / "best.pt"
DEFAULT_MATCH_OUTPUTS_DIR = REPO_ROOT / "match_outputs"
DEFAULT_CACHE_ROOT = DEFAULT_MATCH_OUTPUTS_DIR / "featurenet_matching_cache"
CACHE_VERSION = 1


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    source_label: str
    subject_id: int
    finger_id: int
    acquisition_id: str
    raw_view_index: int
    raw_image_path: str
    image_exists: bool
    ground_truth_root: str

    @property
    def uid(self) -> str:
        return self.sample_id or self.raw_image_path

    @property
    def identity_key(self) -> tuple[str, int, int]:
        return self.source_label, self.subject_id, self.finger_id


@dataclass(frozen=True)
class PairSpec:
    label: str
    view_bucket: str
    finger_relation: str
    a: SampleRecord
    b: SampleRecord

    @property
    def bucket(self) -> str:
        if self.label == "genuine":
            return f"genuine_{self.view_bucket}"
        return f"impostor_{self.view_bucket}_{self.finger_relation}"


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


_INFERENCE_CONTEXT: dict[str, Any] = {}


def _slug(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    return safe or "value"


def _sha1_text(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest()[:length]


def _coerce_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text == "":
        return default
    try:
        return int(text)
    except ValueError:
        return default


def _path_name(path_text: Any) -> str:
    text = "" if path_text is None else str(path_text)
    if "\\" in text:
        return PureWindowsPath(text).name
    return Path(text).name


def _infer_source_label(meta: dict[str, Any], ground_truth_root: Path) -> str:
    merge_source = meta.get("merge_source")
    if isinstance(merge_source, dict) and merge_source.get("label"):
        return str(merge_source["label"])

    raw_path = str(meta.get("raw_image_path") or "")
    parts = PureWindowsPath(raw_path).parts if "\\" in raw_path else Path(raw_path).parts
    normalized = [part.lower() for part in parts]
    if "dataset" in normalized:
        index = normalized.index("dataset")
        if index + 1 < len(parts):
            return str(parts[index + 1])

    if ground_truth_root.name:
        return ground_truth_root.name
    return "dataset"


def _source_subject_id(meta: dict[str, Any]) -> int | None:
    merge_source = meta.get("merge_source")
    if isinstance(merge_source, dict):
        value = _coerce_int(merge_source.get("subject_id"))
        if value is not None:
            return value

    filename = _path_name(meta.get("raw_image_path"))
    first = filename.split("_", 1)[0]
    parsed = _coerce_int(first)
    if parsed is not None:
        return parsed
    return _coerce_int(meta.get("subject_id"))


def resolve_raw_image_path(
    *,
    raw_image_path: str,
    dataset_root: Path,
    source_label: str,
    source_subject_id: int | None,
    subject_id: int,
    finger_id: int,
    acquisition_id: str,
    raw_view_index: int,
) -> Path:
    raw_path = Path(raw_image_path)
    if raw_image_path and raw_path.exists():
        return raw_path.resolve()

    filename = _path_name(raw_image_path)
    if not filename:
        subject_for_name = source_subject_id if source_subject_id is not None else subject_id
        filename = f"{subject_for_name}_{finger_id}_{acquisition_id}_{raw_view_index}.jpg"

    subject_candidates: list[str] = []
    if source_subject_id is not None:
        subject_candidates.append(str(source_subject_id))
    subject_candidates.append(str(subject_id))
    subject_candidates = list(dict.fromkeys(subject_candidates))

    root = dataset_root.resolve()
    candidates: list[Path] = []
    for subject in subject_candidates:
        if source_label:
            candidates.append(root / source_label / subject / "raw" / filename)
        candidates.append(root / subject / "raw" / filename)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else raw_path.resolve()


def _record_from_meta(meta: dict[str, Any], ground_truth_root: Path, dataset_root: Path) -> SampleRecord | None:
    subject_id = _coerce_int(meta.get("subject_id"))
    finger_id = _coerce_int(meta.get("finger_id"))
    raw_view_index = _coerce_int(meta.get("raw_view_index"))
    if subject_id is None or finger_id is None or raw_view_index is None:
        return None

    acquisition_value = meta.get("acquisition_id")
    if acquisition_value is None or str(acquisition_value).strip() == "":
        sample_id = str(meta.get("sample_id") or "")
        acquisition_value = "unknown"
        if "_a" in sample_id:
            acquisition_value = sample_id.rsplit("_v", 1)[0].rsplit("_a", 1)[-1]
    acquisition_id = str(acquisition_value)
    source_label = _infer_source_label(meta, ground_truth_root)
    source_subject = _source_subject_id(meta)
    raw_path = resolve_raw_image_path(
        raw_image_path=str(meta.get("raw_image_path") or ""),
        dataset_root=dataset_root,
        source_label=source_label,
        source_subject_id=source_subject,
        subject_id=subject_id,
        finger_id=finger_id,
        acquisition_id=acquisition_id,
        raw_view_index=raw_view_index,
    )
    sample_id = str(meta.get("sample_id") or f"{source_label}_s{subject_id}_f{finger_id}_a{acquisition_id}_v{raw_view_index}")

    return SampleRecord(
        sample_id=sample_id,
        source_label=source_label,
        subject_id=int(subject_id),
        finger_id=int(finger_id),
        acquisition_id=acquisition_id,
        raw_view_index=int(raw_view_index),
        raw_image_path=str(raw_path),
        image_exists=raw_path.exists(),
        ground_truth_root=str(ground_truth_root.resolve()),
    )


def _read_manifest_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"manifest JSON must contain a list: {path}")
    return [item for item in payload if isinstance(item, dict)]


def _read_manifest_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_sample_meta_files(samples_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for sample_dir in sorted(path for path in samples_root.iterdir() if path.is_dir()):
        meta_path = sample_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"failed to parse {meta_path}: {exc}") from exc
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def discover_samples(
    ground_truth_root: Path,
    dataset_root: Path,
    *,
    front_views: set[int],
    side_views: set[int],
    max_images: int | None = None,
) -> list[SampleRecord]:
    root = ground_truth_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"ground-truth root not found: {root}")

    if (root / "manifest.json").exists():
        entries = _read_manifest_json(root / "manifest.json")
    elif (root / "manifest.csv").exists():
        entries = _read_manifest_csv(root / "manifest.csv")
    elif (root / "samples").exists():
        entries = _read_sample_meta_files(root / "samples")
    else:
        raise FileNotFoundError(f"no manifest.json, manifest.csv, or samples/ directory found in: {root}")

    allowed_views = set(front_views) | set(side_views)
    records: list[SampleRecord] = []
    seen: set[str] = set()
    for entry in entries:
        record = _record_from_meta(entry, root, dataset_root)
        if record is None or record.raw_view_index not in allowed_views:
            continue
        key = record.uid
        if key in seen:
            continue
        seen.add(key)
        records.append(record)

    records.sort(
        key=lambda item: (
            item.source_label,
            item.subject_id,
            item.finger_id,
            item.acquisition_id,
            item.raw_view_index,
            item.sample_id,
        )
    )
    if max_images is not None:
        records = records[: max(0, int(max_images))]
    return records


def _view_group(record: SampleRecord, front_views: set[int], side_views: set[int]) -> str | None:
    if record.raw_view_index in front_views:
        return "front"
    if record.raw_view_index in side_views:
        return "side"
    return None


def _pair_seen_key(a: SampleRecord, b: SampleRecord) -> tuple[str, str]:
    left, right = sorted((a.uid, b.uid))
    return left, right


def _same_identity(a: SampleRecord, b: SampleRecord) -> bool:
    return a.identity_key == b.identity_key


def _choose_two_distinct(pool: list[SampleRecord], rng: random.Random) -> tuple[SampleRecord, SampleRecord] | None:
    if len(pool) < 2:
        return None
    a, b = rng.sample(pool, 2)
    if a.uid == b.uid:
        return None
    return a, b


def _records_by_identity(
    records: Iterable[SampleRecord],
    front_views: set[int],
    side_views: set[int],
) -> dict[tuple[str, int, int], dict[str, list[SampleRecord]]]:
    grouped: dict[tuple[str, int, int], dict[str, list[SampleRecord]]] = {}
    for record in records:
        view = _view_group(record, front_views, side_views)
        if view is None:
            continue
        grouped.setdefault(record.identity_key, {"front": [], "side": []})[view].append(record)
    return grouped


def _sample_genuine_bucket(
    records: list[SampleRecord],
    *,
    view_bucket: str,
    target: int,
    rng: random.Random,
    front_views: set[int],
    side_views: set[int],
) -> list[PairSpec]:
    if target <= 0:
        return []

    grouped = _records_by_identity(records, front_views, side_views)
    if view_bucket == "front_side":
        eligible = [key for key, item in grouped.items() if item["front"] and item["side"]]
    elif view_bucket == "side_side":
        eligible = [key for key, item in grouped.items() if len(item["side"]) >= 2]
    else:
        raise ValueError(f"unsupported genuine view bucket: {view_bucket}")

    pairs: list[PairSpec] = []
    seen: set[tuple[str, str]] = set()
    stagnant_cycles = 0
    while len(pairs) < target and eligible and stagnant_cycles < 25:
        added = 0
        keys = eligible[:]
        rng.shuffle(keys)
        for key in keys:
            if len(pairs) >= target:
                break
            bucket = grouped[key]
            for _ in range(20):
                if view_bucket == "front_side":
                    a = rng.choice(bucket["front"])
                    b = rng.choice(bucket["side"])
                else:
                    chosen = _choose_two_distinct(bucket["side"], rng)
                    if chosen is None:
                        break
                    a, b = chosen
                if a.uid == b.uid:
                    continue
                seen_key = _pair_seen_key(a, b)
                if seen_key in seen:
                    continue
                seen.add(seen_key)
                pairs.append(PairSpec(label="genuine", view_bucket=view_bucket, finger_relation="same_identity", a=a, b=b))
                added += 1
                break
        stagnant_cycles = 0 if added else stagnant_cycles + 1
    return pairs


def _records_by_view_and_finger(
    records: Iterable[SampleRecord],
    front_views: set[int],
    side_views: set[int],
) -> dict[str, dict[int, list[SampleRecord]]]:
    grouped: dict[str, dict[int, list[SampleRecord]]] = {"front": {}, "side": {}}
    for record in records:
        view = _view_group(record, front_views, side_views)
        if view is None:
            continue
        grouped[view].setdefault(record.finger_id, []).append(record)
    return grouped


def _has_two_identities(pool: list[SampleRecord]) -> bool:
    return len({record.identity_key for record in pool}) >= 2


def _has_cross_identity(pool_a: list[SampleRecord], pool_b: list[SampleRecord]) -> bool:
    identities_b = {record.identity_key for record in pool_b}
    return any(record.identity_key not in identities_b or len(identities_b) > 1 for record in pool_a)


def _impostor_view_names(view_bucket: str) -> tuple[str, str]:
    if view_bucket == "front_front":
        return "front", "front"
    if view_bucket == "side_side":
        return "side", "side"
    if view_bucket == "front_side":
        return "front", "side"
    raise ValueError(f"unsupported impostor view bucket: {view_bucket}")


def _impostor_keys(
    grouped: dict[str, dict[int, list[SampleRecord]]],
    *,
    view_bucket: str,
    finger_relation: str,
) -> list[tuple[int, int]]:
    view_a, view_b = _impostor_view_names(view_bucket)
    fingers_a = sorted(grouped[view_a])
    fingers_b = sorted(grouped[view_b])
    keys: list[tuple[int, int]] = []

    if finger_relation == "same_finger_type":
        for finger in sorted(set(fingers_a) & set(fingers_b)):
            pool_a = grouped[view_a][finger]
            pool_b = grouped[view_b][finger]
            if view_a == view_b:
                if _has_two_identities(pool_a):
                    keys.append((finger, finger))
            elif _has_cross_identity(pool_a, pool_b):
                keys.append((finger, finger))
        return keys

    if finger_relation != "cross_finger_type":
        raise ValueError(f"unsupported impostor finger relation: {finger_relation}")

    for finger_a in fingers_a:
        for finger_b in fingers_b:
            if finger_a == finger_b:
                continue
            if view_a == view_b and finger_b < finger_a:
                continue
            keys.append((finger_a, finger_b))
    return keys


def _choose_impostor_pair(
    grouped: dict[str, dict[int, list[SampleRecord]]],
    *,
    view_bucket: str,
    finger_relation: str,
    finger_key: tuple[int, int],
    rng: random.Random,
) -> tuple[SampleRecord, SampleRecord] | None:
    view_a, view_b = _impostor_view_names(view_bucket)
    pool_a = grouped[view_a].get(finger_key[0], [])
    pool_b = grouped[view_b].get(finger_key[1], [])
    if not pool_a or not pool_b:
        return None

    for _ in range(50):
        if view_a == view_b and finger_key[0] == finger_key[1]:
            chosen = _choose_two_distinct(pool_a, rng)
            if chosen is None:
                return None
            a, b = chosen
        else:
            a = rng.choice(pool_a)
            b = rng.choice(pool_b)
            if a.uid == b.uid:
                continue

        if _same_identity(a, b):
            continue
        if finger_relation == "same_finger_type" and a.finger_id != b.finger_id:
            continue
        if finger_relation == "cross_finger_type" and a.finger_id == b.finger_id:
            continue
        return a, b
    return None


def _sample_impostor_bucket(
    records: list[SampleRecord],
    *,
    view_bucket: str,
    finger_relation: str,
    target: int,
    rng: random.Random,
    front_views: set[int],
    side_views: set[int],
) -> list[PairSpec]:
    if target <= 0:
        return []

    grouped = _records_by_view_and_finger(records, front_views, side_views)
    eligible = _impostor_keys(grouped, view_bucket=view_bucket, finger_relation=finger_relation)
    pairs: list[PairSpec] = []
    seen: set[tuple[str, str]] = set()
    stagnant_cycles = 0

    while len(pairs) < target and eligible and stagnant_cycles < 25:
        added = 0
        keys = eligible[:]
        rng.shuffle(keys)
        for finger_key in keys:
            if len(pairs) >= target:
                break
            chosen = _choose_impostor_pair(
                grouped,
                view_bucket=view_bucket,
                finger_relation=finger_relation,
                finger_key=finger_key,
                rng=rng,
            )
            if chosen is None:
                continue
            a, b = chosen
            seen_key = _pair_seen_key(a, b)
            if seen_key in seen:
                continue
            seen.add(seen_key)
            pairs.append(PairSpec(label="impostor", view_bucket=view_bucket, finger_relation=finger_relation, a=a, b=b))
            added += 1
        stagnant_cycles = 0 if added else stagnant_cycles + 1
    return pairs


def sample_all_pair_buckets(
    records: list[SampleRecord],
    *,
    front_views: set[int],
    side_views: set[int],
    pairs_per_bucket: int,
    seed: int,
) -> list[PairSpec]:
    rng = random.Random(seed)
    pairs: list[PairSpec] = []
    pairs.extend(
        _sample_genuine_bucket(
            records,
            view_bucket="front_side",
            target=pairs_per_bucket,
            rng=rng,
            front_views=front_views,
            side_views=side_views,
        )
    )
    pairs.extend(
        _sample_genuine_bucket(
            records,
            view_bucket="side_side",
            target=pairs_per_bucket,
            rng=rng,
            front_views=front_views,
            side_views=side_views,
        )
    )
    for view_bucket in ("front_front", "side_side", "front_side"):
        for finger_relation in ("same_finger_type", "cross_finger_type"):
            pairs.extend(
                _sample_impostor_bucket(
                    records,
                    view_bucket=view_bucket,
                    finger_relation=finger_relation,
                    target=pairs_per_bucket,
                    rng=rng,
                    front_views=front_views,
                    side_views=side_views,
                )
            )
    return pairs


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


def _sample_record_row(record: SampleRecord, front_views: set[int], side_views: set[int]) -> dict[str, Any]:
    return {
        "sample_id": record.sample_id,
        "source_label": record.source_label,
        "subject_id": record.subject_id,
        "finger_id": record.finger_id,
        "acquisition_id": record.acquisition_id,
        "raw_view_index": record.raw_view_index,
        "view_group": _view_group(record, front_views, side_views) or "",
        "raw_image_path": record.raw_image_path,
        "image_exists": str(record.image_exists).lower(),
    }


def _pair_row(pair: PairSpec, index: int) -> dict[str, Any]:
    return {
        "pair_id": f"{pair.bucket}_{index:06d}",
        "bucket": pair.bucket,
        "label": pair.label,
        "view_bucket": pair.view_bucket,
        "finger_relation": pair.finger_relation,
        "a_sample_id": pair.a.sample_id,
        "a_source_label": pair.a.source_label,
        "a_subject_id": pair.a.subject_id,
        "a_finger_id": pair.a.finger_id,
        "a_acquisition_id": pair.a.acquisition_id,
        "a_raw_view_index": pair.a.raw_view_index,
        "a_image_path": pair.a.raw_image_path,
        "b_sample_id": pair.b.sample_id,
        "b_source_label": pair.b.source_label,
        "b_subject_id": pair.b.subject_id,
        "b_finger_id": pair.b.finger_id,
        "b_acquisition_id": pair.b.acquisition_id,
        "b_raw_view_index": pair.b.raw_view_index,
        "b_image_path": pair.b.raw_image_path,
        "same_identity": str(_same_identity(pair.a, pair.b)).lower(),
        "same_finger_type": str(pair.a.finger_id == pair.b.finger_id).lower(),
    }


def _load_infer_helpers() -> dict[str, Any]:
    from featurenet.models.infer import (
        decode_minutiae_rows,
        load_checkpoint_model,
        preprocess_input_bgr,
        run_inference,
        save_minutiae_csv,
        save_pose_sidecars,
        _resolve_device,
    )
    from featurenet.models.match_infer import _crop_distal_phalanx_with_main, _save_mask_png

    return {
        "decode_minutiae_rows": decode_minutiae_rows,
        "load_checkpoint_model": load_checkpoint_model,
        "preprocess_input_bgr": preprocess_input_bgr,
        "run_inference": run_inference,
        "save_minutiae_csv": save_minutiae_csv,
        "save_pose_sidecars": save_pose_sidecars,
        "_resolve_device": _resolve_device,
        "_crop_distal_phalanx_with_main": _crop_distal_phalanx_with_main,
        "_save_mask_png": _save_mask_png,
    }


def _cache_namespace(weights_path: Path, score_threshold: float, apply_nms: bool) -> str:
    try:
        stat = weights_path.stat()
        weight_token = f"{weights_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    except FileNotFoundError:
        weight_token = str(weights_path.resolve())
    digest = _sha1_text(f"{weight_token}:score={score_threshold}:nms={apply_nms}", 14)
    return f"v{CACHE_VERSION}_{digest}"


def _cache_dir_for_record(cache_root: Path, namespace: str, record: SampleRecord) -> Path:
    digest = _sha1_text(record.raw_image_path, 10)
    return cache_root / namespace / _slug(record.source_label) / f"{_slug(record.sample_id)}_{digest}"


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
    record: SampleRecord,
    weights_path: str,
    score_threshold: float,
    apply_nms: bool,
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
    if metadata.get("sample_uid") != record.uid:
        return False, None
    if metadata.get("raw_image_path") != record.raw_image_path:
        return False, None
    if metadata.get("weights_path") != weights_path:
        return False, None
    if float(metadata.get("minutia_score_threshold", -1.0)) != float(score_threshold):
        return False, None
    if bool(metadata.get("minutia_nms_enabled")) != bool(apply_nms):
        return False, None
    return True, int(metadata.get("minutiae_count", _count_minutiae_csv_rows(files["minutiae_csv"])))


def _init_inference_worker(
    weights_path: str,
    device_arg: str,
    score_threshold: float,
    apply_nms: bool,
) -> None:
    helpers = _load_infer_helpers()
    device = helpers["_resolve_device"](device_arg)
    model = helpers["load_checkpoint_model"](Path(weights_path), device)
    _INFERENCE_CONTEXT.clear()
    _INFERENCE_CONTEXT.update(
        {
            "helpers": helpers,
            "device": device,
            "model": model,
            "weights_path": str(Path(weights_path).resolve()),
            "score_threshold": float(score_threshold),
            "apply_nms": bool(apply_nms),
        }
    )


def _extraction_row(
    record: SampleRecord,
    *,
    status: str,
    cache_hit: bool,
    cache_dir: Path,
    files: dict[str, Path],
    minutiae_count: int | None,
    error: str = "",
) -> dict[str, Any]:
    return {
        "sample_uid": record.uid,
        "sample_id": record.sample_id,
        "source_label": record.source_label,
        "subject_id": record.subject_id,
        "finger_id": record.finger_id,
        "acquisition_id": record.acquisition_id,
        "raw_view_index": record.raw_view_index,
        "raw_image_path": record.raw_image_path,
        "image_exists": str(record.image_exists).lower(),
        "status": status,
        "cache_hit": str(cache_hit).lower(),
        "cache_dir": str(cache_dir),
        "minutiae_count": "" if minutiae_count is None else int(minutiae_count),
        "minutiae_csv": str(files["minutiae_csv"]) if files else "",
        "mask_png": str(files["mask_png"]) if files else "",
        "orientation_npy": str(files["orientation_npy"]) if files else "",
        "ridge_period_npy": str(files["ridge_period_npy"]) if files else "",
        "error": error,
    }


def _extract_record_task(
    record_payload: dict[str, Any],
    cache_root: str,
    namespace: str,
    reuse_cache: bool,
) -> dict[str, Any]:
    record = SampleRecord(**record_payload)
    cache_dir = _cache_dir_for_record(Path(cache_root), namespace, record)
    files = _cache_files(cache_dir)
    try:
        if not _INFERENCE_CONTEXT:
            raise RuntimeError("inference worker was not initialized")
        weights_path = _INFERENCE_CONTEXT["weights_path"]
        score_threshold = float(_INFERENCE_CONTEXT["score_threshold"])
        apply_nms = bool(_INFERENCE_CONTEXT["apply_nms"])
        if reuse_cache:
            complete, cached_count = _cached_extraction_is_complete(
                files,
                record=record,
                weights_path=weights_path,
                score_threshold=score_threshold,
                apply_nms=apply_nms,
            )
            if complete:
                return _extraction_row(
                    record,
                    status="ok",
                    cache_hit=True,
                    cache_dir=cache_dir,
                    files=files,
                    minutiae_count=cached_count,
                )

        image_path = Path(record.raw_image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")

        helpers = _INFERENCE_CONTEXT["helpers"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        crop_result = helpers["_crop_distal_phalanx_with_main"](
            image_path=image_path,
            crop_output_dir=cache_dir / "crop",
        )
        image_tensor, mask_tensor, input_shape_hw = helpers["preprocess_input_bgr"](
            full_bgr=crop_result["inference_bgr"],
            save_preprocess_dir=cache_dir / "preprocess",
        )
        outputs = helpers["run_inference"](
            model=_INFERENCE_CONTEXT["model"],
            image_tensor=image_tensor,
            mask_tensor=mask_tensor,
            device=_INFERENCE_CONTEXT["device"],
        )
        minutiae_rows = helpers["decode_minutiae_rows"](
            outputs=outputs,
            input_shape_hw=input_shape_hw,
            score_threshold=score_threshold,
            apply_nms=apply_nms,
        )
        helpers["save_minutiae_csv"](minutiae_rows, files["minutiae_csv"])
        orientation_npy, ridge_period_npy = helpers["save_pose_sidecars"](outputs, cache_dir)
        helpers["_save_mask_png"](mask_tensor, files["mask_png"])

        metadata = {
            "cache_version": CACHE_VERSION,
            "sample_uid": record.uid,
            "record": asdict(record),
            "raw_image_path": record.raw_image_path,
            "weights_path": weights_path,
            "minutia_score_threshold": score_threshold,
            "minutia_nms_enabled": apply_nms,
            "minutiae_count": len(minutiae_rows),
            "artifacts": {
                "minutiae_csv": str(files["minutiae_csv"].resolve()),
                "mask_png": str(files["mask_png"].resolve()),
                "orientation_npy": str(Path(orientation_npy).resolve()),
                "ridge_period_npy": str(Path(ridge_period_npy).resolve()),
            },
            "crop": {
                "crop_bbox_xyxy": list(crop_result["crop_bbox"]),
                "crop_mode": crop_result["crop_mode"],
                "fallback_reason": crop_result["fallback_reason"],
                "coarse_mask_pixels": int(crop_result["coarse_mask_pixels"]),
                "distal_mask_pixels": int(crop_result["distal_mask_pixels"]),
            },
        }
        files["metadata_json"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return _extraction_row(
            record,
            status="ok",
            cache_hit=False,
            cache_dir=cache_dir,
            files=files,
            minutiae_count=len(minutiae_rows),
        )
    except Exception as exc:
        return _extraction_row(
            record,
            status="error",
            cache_hit=False,
            cache_dir=cache_dir,
            files=files,
            minutiae_count=None,
            error=str(exc),
        )


def _resolve_worker_count(value: str, *, default_auto: int) -> int:
    text = str(value).strip().lower()
    if text == "auto":
        return max(1, int(default_auto))
    count = int(text)
    if count < 1:
        raise ValueError("worker count must be at least 1")
    return count


def _bytes_to_gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / float(1024**3)


def estimate_gpu_capacity(
    *,
    weights_path: Path,
    device_arg: str,
    records: list[SampleRecord],
    score_threshold: float,
    apply_nms: bool,
    output_dir: Path,
    calibration_count: int,
    reserve_gb: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "available": False,
        "estimated_model_instances": 1,
        "reason": None,
    }
    try:
        helpers = _load_infer_helpers()
        import torch

        device = helpers["_resolve_device"](device_arg)
        report["device"] = str(device)
        if getattr(device, "type", str(device)) != "cuda":
            report["reason"] = "resolved device is not cuda"
            return report

        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = helpers["load_checkpoint_model"](weights_path, device)
        model_allocated = int(torch.cuda.memory_allocated(device))
        calibration_dir = output_dir / "gpu_calibration"
        calibrated = 0

        for record in records:
            if calibrated >= calibration_count:
                break
            image_path = Path(record.raw_image_path)
            if not image_path.exists():
                continue
            crop_result = helpers["_crop_distal_phalanx_with_main"](
                image_path=image_path,
                crop_output_dir=calibration_dir / _slug(record.sample_id),
            )
            image_tensor, mask_tensor, input_shape_hw = helpers["preprocess_input_bgr"](
                full_bgr=crop_result["inference_bgr"],
                save_preprocess_dir=None,
            )
            outputs = helpers["run_inference"](
                model=model,
                image_tensor=image_tensor,
                mask_tensor=mask_tensor,
                device=device,
            )
            helpers["decode_minutiae_rows"](
                outputs=outputs,
                input_shape_hw=input_shape_hw,
                score_threshold=score_threshold,
                apply_nms=apply_nms,
            )
            calibrated += 1

        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        reserve_bytes = int(max(0.0, reserve_gb) * (1024**3))
        usable_bytes = max(0, int(total_bytes) - reserve_bytes)
        per_instance_bytes = max(peak_allocated, model_allocated, 1)
        estimated = max(1, int(usable_bytes // per_instance_bytes))

        report.update(
            {
                "available": True,
                "reason": None,
                "cuda_free_gib_before": _bytes_to_gib(free_bytes),
                "cuda_total_gib": _bytes_to_gib(total_bytes),
                "model_allocated_gib": _bytes_to_gib(model_allocated),
                "peak_inference_allocated_gib": _bytes_to_gib(peak_allocated),
                "reserve_gib": float(reserve_gb),
                "calibration_images": int(calibrated),
                "estimated_model_instances": int(estimated),
            }
        )
        del model
        torch.cuda.empty_cache()
    except Exception as exc:
        report["reason"] = str(exc)
    return report


def run_inference_for_records(
    records: list[SampleRecord],
    *,
    weights_path: Path,
    device_arg: str,
    score_threshold: float,
    apply_nms: bool,
    cache_root: Path,
    namespace: str,
    reuse_cache: bool,
    gpu_workers: int,
) -> list[dict[str, Any]]:
    payloads = [asdict(record) for record in records]
    if gpu_workers == 1:
        _init_inference_worker(str(weights_path.resolve()), device_arg, score_threshold, apply_nms)
        rows: list[dict[str, Any]] = []
        for index, payload in enumerate(payloads, start=1):
            rows.append(_extract_record_task(payload, str(cache_root), namespace, reuse_cache))
            if index == 1 or index % 25 == 0 or index == len(payloads):
                ok_count = sum(1 for row in rows if row.get("status") == "ok")
                print(f"[inference] processed {index}/{len(payloads)} ok={ok_count} errors={len(rows) - ok_count}", flush=True)
        return rows

    rows = []
    with ProcessPoolExecutor(
        max_workers=gpu_workers,
        initializer=_init_inference_worker,
        initargs=(str(weights_path.resolve()), device_arg, score_threshold, apply_nms),
    ) as executor:
        futures = [
            executor.submit(_extract_record_task, payload, str(cache_root), namespace, reuse_cache)
            for payload in payloads
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index == 1 or index % 25 == 0 or index == len(futures):
                ok_count = sum(1 for row in rows if row.get("status") == "ok")
                print(f"[inference] processed {index}/{len(futures)} ok={ok_count} errors={len(rows) - ok_count}", flush=True)
    return rows


def _extracted_from_row(row: dict[str, Any]) -> ExtractedImage:
    count_value = row.get("minutiae_count")
    count = int(count_value) if str(count_value).strip() not in {"", "None"} else None
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


def _make_match_jobs(
    pairs: list[PairSpec],
    extraction_rows: list[dict[str, Any]],
    *,
    method: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    extracted = {_extracted_from_row(row).sample_uid: _extracted_from_row(row) for row in extraction_rows}
    jobs: list[dict[str, Any]] = []
    immediate_rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs, start=1):
        row = _pair_row(pair, index)
        a = extracted.get(pair.a.uid)
        b = extracted.get(pair.b.uid)
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
                    "error": "; ".join(missing),
                }
            )
            immediate_rows.append(row)
            continue

        row.update(
            {
                "method": method,
                "status": "pending",
                "score": "",
                "similarity_matrix_shape": "",
                "a_minutiae_count": a.minutiae_count,
                "b_minutiae_count": b.minutiae_count,
                "a_minutiae_csv": a.minutiae_csv,
                "b_minutiae_csv": b.minutiae_csv,
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


def run_mcc_matching(
    pairs: list[PairSpec],
    extraction_rows: list[dict[str, Any]],
    *,
    method: str,
    mcc_workers: int,
) -> list[dict[str, Any]]:
    jobs, rows = _make_match_jobs(pairs, extraction_rows, method=method)
    if not jobs:
        return rows

    with ProcessPoolExecutor(max_workers=mcc_workers) as executor:
        futures = [executor.submit(_match_pair_task, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index == 1 or index % 50 == 0 or index == len(futures):
                ok_count = sum(1 for row in rows if row.get("status") == "ok")
                print(f"[mcc] matched {index}/{len(futures)} worker_jobs ok={ok_count} total_rows={len(rows)}", flush=True)
    rows.sort(key=lambda row: str(row.get("pair_id") or ""))
    return rows


def score_stats(scores: Iterable[float]) -> dict[str, Any]:
    values = [float(score) for score in scores if math.isfinite(float(score))]
    if not values:
        return {"count": 0, "average": None, "lowest": None, "highest": None}
    return {
        "count": len(values),
        "average": float(sum(values) / len(values)),
        "lowest": float(min(values)),
        "highest": float(max(values)),
    }


def summarize_matches(match_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in match_rows if row.get("status") == "ok" and str(row.get("score", "")).strip() != ""]
    bucket_names = [
        "genuine_front_side",
        "genuine_side_side",
        "impostor_front_front_same_finger_type",
        "impostor_front_front_cross_finger_type",
        "impostor_side_side_same_finger_type",
        "impostor_side_side_cross_finger_type",
        "impostor_front_side_same_finger_type",
        "impostor_front_side_cross_finger_type",
    ]
    return {
        "overall": score_stats(float(row["score"]) for row in ok_rows),
        "by_label": {
            "genuine": score_stats(float(row["score"]) for row in ok_rows if row.get("label") == "genuine"),
            "impostor": score_stats(float(row["score"]) for row in ok_rows if row.get("label") == "impostor"),
        },
        "by_bucket": {
            bucket: score_stats(float(row["score"]) for row in ok_rows if row.get("bucket") == bucket)
            for bucket in bucket_names
        },
    }


def _write_summary_txt(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "FeatureNet Matching Inference Summary",
        "",
        f"Overall count/average/lowest/highest: {summary['overall']['count']} / {summary['overall']['average']} / {summary['overall']['lowest']} / {summary['overall']['highest']}",
        "",
        "Buckets:",
    ]
    for bucket, stats in summary["by_bucket"].items():
        lines.append(
            f"- {bucket}: count={stats['count']} average={stats['average']} lowest={stats['lowest']} highest={stats['highest']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_output_dir() -> Path:
    return DEFAULT_MATCH_OUTPUTS_DIR / f"featurenet_matching_{time.strftime('%Y%m%d_%H%M%S')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FeatureNet inference once per sample, then randomly sample and parallelize MCC matching."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--ground-truth-root", type=Path, default=DEFAULT_GROUND_TRUTH_ROOT)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--front-views", type=int, nargs="+", default=[0])
    parser.add_argument("--side-views", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--pairs-per-bucket", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--method", type=str, default="LSA-R")
    parser.add_argument("--minutia-score-threshold", type=float, default=0.6)
    parser.add_argument("--disable-minutia-nms", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-workers", type=str, default="1", help="1, auto, or an explicit worker count.")
    parser.add_argument("--mcc-workers", type=str, default="auto", help="auto or an explicit worker count.")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run-sampling", action="store_true")
    parser.add_argument("--gpu-calibration-images", type=int, default=2)
    parser.add_argument("--gpu-reserve-gb", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    started_at = time.time()
    args = parse_args()
    output_dir = (args.output_dir or _default_output_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    front_views = {int(view) for view in args.front_views}
    side_views = {int(view) for view in args.side_views}
    if front_views & side_views:
        raise ValueError("--front-views and --side-views must be disjoint")
    if args.pairs_per_bucket < 0:
        raise ValueError("--pairs-per-bucket must be non-negative")
    if args.max_images is not None and args.max_images < 0:
        raise ValueError("--max-images must be non-negative")

    records = discover_samples(
        args.ground_truth_root,
        args.dataset_root,
        front_views=front_views,
        side_views=side_views,
        max_images=args.max_images,
    )
    if not records:
        raise RuntimeError("no front/side samples discovered")

    pairs = sample_all_pair_buckets(
        records,
        front_views=front_views,
        side_views=side_views,
        pairs_per_bucket=int(args.pairs_per_bucket),
        seed=int(args.seed),
    )

    sampled_images_path = output_dir / "discovered_samples.csv"
    sampled_pairs_path = output_dir / "sampled_pairs.csv"
    _write_csv_rows(sampled_images_path, [_sample_record_row(record, front_views, side_views) for record in records])
    _write_csv_rows(sampled_pairs_path, [_pair_row(pair, index) for index, pair in enumerate(pairs, start=1)])

    print(f"Discovered samples: {len(records)}", flush=True)
    print(f"Sampled pairs: {len(pairs)}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)

    if args.dry_run_sampling:
        summary = {
            "config": {
                "dataset_root": str(args.dataset_root.resolve()),
                "ground_truth_root": str(args.ground_truth_root.resolve()),
                "front_views": sorted(front_views),
                "side_views": sorted(side_views),
                "pairs_per_bucket": int(args.pairs_per_bucket),
                "seed": int(args.seed),
                "max_images": args.max_images,
                "dry_run_sampling": True,
            },
            "counts": {
                "discovered_sample_count": len(records),
                "sampled_pair_count": len(pairs),
                "missing_image_count": sum(1 for record in records if not record.image_exists),
            },
            "outputs": {
                "discovered_samples_csv": str(sampled_images_path),
                "sampled_pairs_csv": str(sampled_pairs_path),
            },
            "wall_seconds": round(time.time() - started_at, 3),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Dry-run summary: {output_dir / 'summary.json'}", flush=True)
        return 0

    weights_path = args.weights_path.resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"weights file not found: {weights_path}")

    apply_nms = not bool(args.disable_minutia_nms)
    cache_root = args.cache_root.resolve()
    namespace = _cache_namespace(weights_path, float(args.minutia_score_threshold), apply_nms)

    gpu_report = estimate_gpu_capacity(
        weights_path=weights_path,
        device_arg=str(args.device),
        records=records,
        score_threshold=float(args.minutia_score_threshold),
        apply_nms=apply_nms,
        output_dir=output_dir,
        calibration_count=max(0, int(args.gpu_calibration_images)),
        reserve_gb=float(args.gpu_reserve_gb),
    )
    if str(args.gpu_workers).strip().lower() == "auto":
        gpu_workers = int(gpu_report.get("estimated_model_instances") or 1)
    else:
        gpu_workers = _resolve_worker_count(str(args.gpu_workers), default_auto=1)
    gpu_workers = max(1, gpu_workers)
    print(f"GPU capacity report: {json.dumps(gpu_report, sort_keys=True)}", flush=True)
    print(f"Using GPU inference workers: {gpu_workers}", flush=True)

    inference_rows = run_inference_for_records(
        records,
        weights_path=weights_path,
        device_arg=str(args.device),
        score_threshold=float(args.minutia_score_threshold),
        apply_nms=apply_nms,
        cache_root=cache_root,
        namespace=namespace,
        reuse_cache=bool(args.reuse_cache),
        gpu_workers=gpu_workers,
    )
    inference_manifest_path = output_dir / "inference_manifest.csv"
    inference_errors_path = output_dir / "inference_errors.csv"
    _write_csv_rows(inference_manifest_path, inference_rows)
    _write_csv_rows(inference_errors_path, [row for row in inference_rows if row.get("status") != "ok"])

    auto_mcc = max(1, (os.cpu_count() or 2) - 1)
    mcc_workers = _resolve_worker_count(str(args.mcc_workers), default_auto=auto_mcc)
    print(f"Using MCC workers: {mcc_workers}", flush=True)
    match_rows = run_mcc_matching(
        pairs,
        inference_rows,
        method=str(args.method),
        mcc_workers=mcc_workers,
    )
    matches_path = output_dir / "matches.csv"
    _write_csv_rows(matches_path, match_rows)

    score_summary = summarize_matches(match_rows)
    summary = {
        "config": {
            "dataset_root": str(args.dataset_root.resolve()),
            "ground_truth_root": str(args.ground_truth_root.resolve()),
            "weights_path": str(weights_path),
            "front_views": sorted(front_views),
            "side_views": sorted(side_views),
            "pairs_per_bucket": int(args.pairs_per_bucket),
            "seed": int(args.seed),
            "method": str(args.method),
            "minutia_score_threshold": float(args.minutia_score_threshold),
            "minutia_nms_enabled": apply_nms,
            "device": str(args.device),
            "gpu_workers": gpu_workers,
            "mcc_workers": mcc_workers,
            "cache_root": str(cache_root),
            "cache_namespace": namespace,
            "reuse_cache": bool(args.reuse_cache),
        },
        "counts": {
            "discovered_sample_count": len(records),
            "sampled_pair_count": len(pairs),
            "inference_ok_count": sum(1 for row in inference_rows if row.get("status") == "ok"),
            "inference_error_count": sum(1 for row in inference_rows if row.get("status") != "ok"),
            "match_ok_count": sum(1 for row in match_rows if row.get("status") == "ok"),
            "match_error_count": sum(1 for row in match_rows if row.get("status") != "ok"),
            "missing_image_count": sum(1 for record in records if not record.image_exists),
        },
        "gpu_capacity": gpu_report,
        "scores": score_summary,
        "outputs": {
            "discovered_samples_csv": str(sampled_images_path),
            "sampled_pairs_csv": str(sampled_pairs_path),
            "inference_manifest_csv": str(inference_manifest_path),
            "inference_errors_csv": str(inference_errors_path),
            "matches_csv": str(matches_path),
            "summary_json": str(output_dir / "summary.json"),
            "summary_txt": str(output_dir / "summary.txt"),
        },
        "wall_seconds": round(time.time() - started_at, 3),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_txt(output_dir / "summary.txt", score_summary)

    print(f"Saved inference manifest: {inference_manifest_path}", flush=True)
    print(f"Saved matches: {matches_path}", flush=True)
    print(f"Saved summary: {output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Audit MCC match scores after unwarping ground-truth samples and extracting minutiae."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import os
import math
import sys
import sysconfig
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT]


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

# Force CPU-only TensorFlow/pyfing/FingerFlow behavior in this audit pipeline.
# Spawned worker processes inherit these settings before importing ML libraries.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

from dataclasses import dataclass

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyfing  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
import main as mcc  # noqa: E402


def _load_unwarp_module() -> Any:
    module_name = "ground_truth_unwarp_pipeline"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = REPO_ROOT / "scripts" / "unwarp.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import unwarp module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


UNWARP = _load_unwarp_module()

DEFAULT_GROUND_TRUTH_ROOT = REPO_ROOT / "ground_truth" / "DS123_merged_v5"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "ground_truth_unwarp_mcc_audit"
DEFAULT_THRESHOLDS = (0.7, 0.8, 0.9)
DEFAULT_METHODS = ("LSA", "LSA-R", "LSA-CENTROID")
DEFAULT_EXTRACTORS = ("pyfing", "fingerflow")
FRONT_ROLE = "front"
SIDE_ROLES = {"left", "right"}
GENUINE_BUCKET = "front_side_same_finger"
IMPOSTOR_BUCKET = "front_front_other_finger"


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    subject_id: int
    finger_id: int
    acquisition_id: int
    view_role: str
    reconstruction_dir: str
    meta_path: str
    sample_dir: str

    @property
    def identity_key(self) -> tuple[int, int]:
        return self.subject_id, self.finger_id


@dataclass(frozen=True)
class PairSpec:
    bucket: str
    pair_index: int
    a: SampleRecord
    b: SampleRecord

    @property
    def pair_key(self) -> tuple[str, str]:
        ordered = tuple(sorted((self.a.sample_id, self.b.sample_id)))
        return ordered[0], ordered[1]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _load_sample_records(ground_truth_root: Path) -> list[SampleRecord]:
    samples_dir = ground_truth_root / "samples"
    if not samples_dir.exists():
        raise FileNotFoundError(f"samples directory not found: {samples_dir}")

    records: list[SampleRecord] = []
    for meta_path in sorted(samples_dir.glob("*/meta.json")):
        try:
            meta = _read_json(meta_path)
            gt = meta.get("minutiae_ground_truth")
            multiview = meta.get("multiview_reconstruction")
            if not isinstance(gt, dict) or not isinstance(multiview, dict):
                continue
            view_role = str(gt.get("view_role") or "").strip().lower()
            reconstruction_dir = multiview.get("reconstruction_dir")
            if view_role not in {FRONT_ROLE, *SIDE_ROLES} or not reconstruction_dir:
                continue
            records.append(
                SampleRecord(
                    sample_id=str(meta["sample_id"]),
                    subject_id=int(meta["subject_id"]),
                    finger_id=int(meta["finger_id"]),
                    acquisition_id=int(meta["acquisition_id"]),
                    view_role=view_role,
                    reconstruction_dir=_normalized_path(reconstruction_dir),
                    meta_path=_normalized_path(meta_path),
                    sample_dir=_normalized_path(meta_path.parent),
                )
            )
        except Exception as exc:
            print(f"[warn] skipping malformed sample metadata {meta_path}: {exc}", file=sys.stderr)
    return records


def _group_by_identity(records: Iterable[SampleRecord]) -> dict[tuple[int, int], list[SampleRecord]]:
    grouped: dict[tuple[int, int], list[SampleRecord]] = {}
    for record in records:
        grouped.setdefault(record.identity_key, []).append(record)
    return grouped


def _sample_unique_pair_specs_from_candidates(
    *,
    candidates: list[PairSpec],
    pair_count: int,
    rng: np.random.Generator,
) -> list[PairSpec]:
    if len(candidates) < pair_count:
        raise ValueError(f"insufficient candidate pairs: have {len(candidates)}, need {pair_count}")
    seen: set[tuple[str, str]] = set()
    sampled: list[PairSpec] = []
    attempts = 0
    max_attempts = max(1000, pair_count * 10_000)
    while len(sampled) < pair_count and attempts < max_attempts:
        attempts += 1
        pair = candidates[int(rng.integers(0, len(candidates)))]
        key = pair.pair_key
        if key in seen:
            continue
        seen.add(key)
        sampled.append(PairSpec(bucket=pair.bucket, pair_index=len(sampled), a=pair.a, b=pair.b))
    if len(sampled) < pair_count:
        raise RuntimeError(f"could not sample {pair_count} unique pairs from {len(candidates)} candidates")
    return sampled


def _sample_unique_impostor_pairs(
    *,
    front_by_identity: dict[tuple[int, int], list[SampleRecord]],
    pair_count: int,
    rng: np.random.Generator,
) -> list[PairSpec]:
    identities = [identity for identity, group in front_by_identity.items() if group]
    if len(identities) < 2:
        raise ValueError("insufficient front-view identities for impostor sampling")

    sampled: list[PairSpec] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = max(1000, pair_count * 20_000)
    while len(sampled) < pair_count and attempts < max_attempts:
        attempts += 1
        idx_a, idx_b = rng.choice(len(identities), size=2, replace=False)
        identity_a = identities[int(idx_a)]
        identity_b = identities[int(idx_b)]
        if identity_a == identity_b:
            continue
        front_a = front_by_identity[identity_a][int(rng.integers(0, len(front_by_identity[identity_a])))]
        front_b = front_by_identity[identity_b][int(rng.integers(0, len(front_by_identity[identity_b])))]
        pair_key = tuple(sorted((front_a.sample_id, front_b.sample_id)))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        sampled.append(
            PairSpec(
                bucket=IMPOSTOR_BUCKET,
                pair_index=len(sampled),
                a=front_a,
                b=front_b,
            )
        )
    if len(sampled) < pair_count:
        raise RuntimeError(f"could not sample {pair_count} unique impostor pairs")
    return sampled


def sample_pair_sets(
    records: list[SampleRecord],
    *,
    pair_count: int,
    seed: int,
) -> dict[str, list[PairSpec]]:
    rng = np.random.default_rng(seed)
    by_identity = _group_by_identity(records)
    front_by_identity: dict[tuple[int, int], list[SampleRecord]] = {}
    side_by_identity: dict[tuple[int, int], list[SampleRecord]] = {}
    for identity, group in by_identity.items():
        front_by_identity[identity] = [record for record in group if record.view_role == FRONT_ROLE]
        side_by_identity[identity] = [record for record in group if record.view_role in SIDE_ROLES]

    genuine_candidates: list[PairSpec] = []
    for identity in sorted(set(front_by_identity) & set(side_by_identity)):
        fronts = front_by_identity[identity]
        sides = side_by_identity[identity]
        for front in fronts:
            for side in sides:
                genuine_candidates.append(PairSpec(bucket=GENUINE_BUCKET, pair_index=0, a=front, b=side))

    return {
        GENUINE_BUCKET: _sample_unique_pair_specs_from_candidates(
            candidates=genuine_candidates,
            pair_count=pair_count,
            rng=rng,
        ),
        IMPOSTOR_BUCKET: _sample_unique_impostor_pairs(
            front_by_identity=front_by_identity,
            pair_count=pair_count,
            rng=rng,
        ),
    }


def _role_unwrapped_path(sample_cache_dir: Path, view_role: str) -> Path:
    if view_role == FRONT_ROLE:
        return sample_cache_dir / "front" / "front_algorithm3_unwrapped.png"
    return sample_cache_dir / view_role / f"{view_role}_canonical_chart_unwrapped.png"


def _role_unwarp_report_path(sample_cache_dir: Path, view_role: str) -> Path:
    if view_role == FRONT_ROLE:
        return sample_cache_dir / "front" / "front_algorithm3_unwrapped.png"
    return sample_cache_dir / view_role / f"{view_role}_canonical_chart_unwrapped.png"


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _write_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.clip(image, 0, 255).astype(np.uint8)):
        raise RuntimeError(f"failed to write image to {path}")


def _overlay_minutiae(image_path: Path, minutiae: list[dict[str, Any]]) -> np.ndarray:
    image = cv2.cvtColor(_read_gray(image_path), cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    for row in minutiae:
        x = int(round(float(row["x"])))
        y = int(round(float(row["y"])))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        angle = float(row.get("angle", 0.0))
        score = float(row.get("score", 0.0))
        color = (0, 255, 255) if score >= 0.5 else (0, 140, 255)
        x2 = int(round(x + 16 * math.cos(angle)))
        y2 = int(round(y + 16 * math.sin(angle)))
        cv2.circle(image, (x, y), 3, color, -1, cv2.LINE_AA)
        cv2.line(image, (x, y), (x2, y2), color, 1, cv2.LINE_AA)
    return image


def _standardize_pyfing_minutiae(raw_minutiae: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in raw_minutiae:
        x = float(getattr(item, "x"))
        y = float(getattr(item, "y"))
        theta = float(getattr(item, "direction", getattr(item, "angle", 0.0)))
        quality = getattr(item, "quality", None)
        rows.append(
            {
                "x": x,
                "y": y,
                "angle": theta,
                "score": float(quality) if quality is not None else 1.0,
                "type": str(getattr(item, "type", "")),
            }
        )
    return rows


def _write_minutiae_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y", "angle", "score", "type"])
        writer.writeheader()
        writer.writerows(rows)


def _load_minutiae_csv(csv_path: Path) -> list[dict[str, float]]:
    if not csv_path.exists():
        return []
    rows: list[dict[str, float]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                angle_value = row.get("angle", row.get("direction", 0.0))
                score_value = row.get("score", row.get("quality", 0.0))
                rows.append(
                    {
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "angle": float(angle_value),
                        "score": float(score_value) if score_value is not None else 0.0,
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def _load_minutiae_json(json_path: Path) -> list[dict[str, float]]:
    payload = _read_json(json_path)
    rows = payload.get("minutiae", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    normalized: list[dict[str, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            normalized.append(
                {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "angle": float(row.get("angle", row.get("direction", row.get("theta", 0.0)))),
                    "score": float(row.get("score", row.get("quality", 0.0))),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return normalized


def _extract_pyfing_from_unwrapped(unwrapped_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "minutiae.json"
    csv_path = output_dir / "minutiae.csv"
    overlay_path = output_dir / "minutiae_overlay.png"
    if json_path.exists() and csv_path.exists() and overlay_path.exists():
        rows = _load_minutiae_csv(csv_path)
        return {
            "source_image_path": str(unwrapped_path.resolve()),
            "minutiae_json_path": str(json_path.resolve()),
            "minutiae_csv_path": str(csv_path.resolve()),
            "overlay_path": str(overlay_path.resolve()),
            "minutiae_count": int(len(rows)),
        }

    image = _read_gray(unwrapped_path)
    raw_minutiae = pyfing.minutiae_extraction(image, dpi=500)
    rows = _standardize_pyfing_minutiae(raw_minutiae)
    payload = {"source_image": str(unwrapped_path.resolve()), "minutiae": rows}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_minutiae_csv(rows, csv_path)
    cv2.imwrite(str(overlay_path), _overlay_minutiae(unwrapped_path, rows))
    return {
        "source_image_path": str(unwrapped_path.resolve()),
        "minutiae_json_path": str(json_path.resolve()),
        "minutiae_csv_path": str(csv_path.resolve()),
        "overlay_path": str(overlay_path.resolve()),
        "minutiae_count": int(len(rows)),
    }


def _extract_fingerflow_from_unwrapped(
    unwrapped_path: Path,
    output_dir: Path,
    model_paths: tuple[Path, Path, Path, Path],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "minutiae.json"
    csv_path = output_dir / "minutiae.csv"
    core_csv_path = output_dir / "core.csv"
    overlay_path = output_dir / "minutiae_overlay.png"
    if json_path.exists() and csv_path.exists() and overlay_path.exists():
        rows = _load_minutiae_csv(csv_path)
        return {
            "source_image_path": str(unwrapped_path.resolve()),
            "minutiae_json_path": str(json_path.resolve()),
            "minutiae_csv_path": str(csv_path.resolve()),
            "core_csv_path": str(core_csv_path.resolve()) if core_csv_path.exists() else None,
            "overlay_path": str(overlay_path.resolve()),
            "minutiae_count": int(len(rows)),
        }

    count, _core_count = mcc.extract_minutiae_with_fingerflow(
        unwrapped_path.resolve(),
        unwrapped_path.resolve(),
        model_paths,
        json_path.resolve(),
        csv_path.resolve(),
        core_csv_path.resolve(),
    )
    rows = _load_minutiae_csv(csv_path)
    cv2.imwrite(str(overlay_path), _overlay_minutiae(unwrapped_path, rows))
    return {
        "source_image_path": str(unwrapped_path.resolve()),
        "minutiae_json_path": str(json_path.resolve()),
        "minutiae_csv_path": str(csv_path.resolve()),
        "core_csv_path": str(core_csv_path.resolve()),
        "overlay_path": str(overlay_path.resolve()),
        "minutiae_count": int(count),
    }


def _load_unwarp_report_path(sample_cache_dir: Path) -> Path:
    return sample_cache_dir / "algorithm1_depth_then_stitched_branch_midpoint_unwarp_report.json"


def _ensure_unwarp_cache(
    record: SampleRecord,
    cache_root: Path,
    *,
    left_angle: float,
    right_angle: float,
    samples_per_pixel: float,
    unwarp_output_height: int | None,
    unwarp_output_width: int | None,
) -> dict[str, Any]:
    sample_cache_dir = cache_root / "samples" / record.sample_id / "unwarp"
    report_path = _load_unwarp_report_path(sample_cache_dir)
    if report_path.exists():
        report = _read_json(report_path)
    else:
        report = UNWARP.run(
            reconstruction_dir=Path(record.reconstruction_dir),
            output_dir=sample_cache_dir,
            left_angle=left_angle,
            right_angle=right_angle,
            samples_per_pixel=samples_per_pixel,
            unwarp_output_height=unwarp_output_height,
            unwarp_output_width=unwarp_output_width,
        )
    unwrapped_path = _role_unwrapped_path(sample_cache_dir, record.view_role)
    if not unwrapped_path.exists():
        raise FileNotFoundError(f"missing unwrapped image for {record.sample_id}: {unwrapped_path}")
    selected_path = sample_cache_dir / "selected_unwrapped.png"
    if not selected_path.exists():
        _write_gray(selected_path, _read_gray(unwrapped_path))
    return {
        "sample_cache_dir": str(sample_cache_dir.resolve()),
        "report_path": str(report_path.resolve()),
        "report": report,
        "selected_unwrapped_path": str(selected_path.resolve()),
        "role_unwrapped_path": str(unwrapped_path.resolve()),
    }


def _prepare_sample_cache(
    record: SampleRecord,
    cache_root: Path,
    *,
    left_angle: float,
    right_angle: float,
    samples_per_pixel: float,
    unwarp_output_height: int | None,
    unwarp_output_width: int | None,
    model_paths: tuple[Path, Path, Path, Path],
) -> dict[str, Any]:
    unwarp_cache = _ensure_unwarp_cache(
        record,
        cache_root,
        left_angle=left_angle,
        right_angle=right_angle,
        samples_per_pixel=samples_per_pixel,
        unwarp_output_height=unwarp_output_height,
        unwarp_output_width=unwarp_output_width,
    )
    selected_unwrapped = Path(unwarp_cache["selected_unwrapped_path"])
    sample_root = cache_root / "samples" / record.sample_id / "extractors"
    pyfing_cache = _extract_pyfing_from_unwrapped(selected_unwrapped, sample_root / "pyfing")
    fingerflow_cache = _extract_fingerflow_from_unwrapped(selected_unwrapped, sample_root / "fingerflow", model_paths)
    return {
        "sample_id": record.sample_id,
        "view_role": record.view_role,
        "subject_id": int(record.subject_id),
        "finger_id": int(record.finger_id),
        "acquisition_id": int(record.acquisition_id),
        "reconstruction_dir": record.reconstruction_dir,
        "meta_path": record.meta_path,
        "sample_dir": record.sample_dir,
        "unwarp": unwarp_cache,
        "extractors": {
            "pyfing": pyfing_cache,
            "fingerflow": fingerflow_cache,
        },
    }


def _build_score_cache(sample_cache: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    score_cache: dict[str, dict[str, dict[str, Any]]] = {}
    for sample_id, sample_payload in sample_cache.items():
        score_cache[sample_id] = {}
        for extractor_name, extractor_payload in sample_payload["extractors"].items():
            score_cache[sample_id][extractor_name] = {
                "minutiae_csv_path": extractor_payload["minutiae_csv_path"],
                "minutiae_json_path": extractor_payload["minutiae_json_path"],
                "overlay_path": extractor_payload["overlay_path"],
                "core_csv_path": extractor_payload.get("core_csv_path"),
                "minutiae_count": extractor_payload.get("minutiae_count"),
            }
    return score_cache


def _prepare_sample_caches(
    records: list[SampleRecord],
    cache_root: Path,
    *,
    left_angle: float,
    right_angle: float,
    samples_per_pixel: float,
    unwarp_output_height: int | None,
    unwarp_output_width: int | None,
    model_paths: tuple[Path, Path, Path, Path],
    workers: int,
) -> dict[str, dict[str, Any]]:
    prepared: dict[str, dict[str, Any]] = {}
    # TensorFlow-backed minutiae extractors can be brittle when started in subprocesses.
    # Keep sample preparation in-process, and reserve process parallelism for the
    # lighter MCC scoring stage.
    for record in records:
        item = _prepare_sample_cache(
            record,
            cache_root,
            left_angle=left_angle,
            right_angle=right_angle,
            samples_per_pixel=samples_per_pixel,
            unwarp_output_height=unwarp_output_height,
            unwarp_output_width=unwarp_output_width,
            model_paths=model_paths,
        )
        prepared[str(item["sample_id"])] = item
    return prepared


def _load_sample_cache_from_existing(records: list[SampleRecord], cache_root: Path) -> dict[str, dict[str, Any]]:
    prepared: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_root = cache_root / "samples" / record.sample_id
        selected_unwrapped = sample_root / "unwarp" / "selected_unwrapped.png"
        report_path = sample_root / "unwarp" / "algorithm1_depth_then_stitched_branch_midpoint_unwarp_report.json"
        pyfing_root = sample_root / "extractors" / "pyfing"
        fingerflow_root = sample_root / "extractors" / "fingerflow"
        pyfing_csv = pyfing_root / "minutiae.csv"
        pyfing_json = pyfing_root / "minutiae.json"
        pyfing_overlay = pyfing_root / "minutiae_overlay.png"
        fingerflow_csv = fingerflow_root / "minutiae.csv"
        fingerflow_json = fingerflow_root / "minutiae.json"
        fingerflow_overlay = fingerflow_root / "minutiae_overlay.png"
        if not all(
            path.exists()
            for path in (
                selected_unwrapped,
                report_path,
                pyfing_csv,
                pyfing_json,
                pyfing_overlay,
                fingerflow_csv,
                fingerflow_json,
                fingerflow_overlay,
            )
        ):
            continue
        pyfing_rows = _load_minutiae_csv(pyfing_csv)
        fingerflow_rows = _load_minutiae_csv(fingerflow_csv)
        prepared[str(record.sample_id)] = {
            "sample_id": record.sample_id,
            "view_role": record.view_role,
            "subject_id": int(record.subject_id),
            "finger_id": int(record.finger_id),
            "acquisition_id": int(record.acquisition_id),
            "reconstruction_dir": record.reconstruction_dir,
            "meta_path": record.meta_path,
            "sample_dir": record.sample_dir,
            "unwarp": {
                "sample_cache_dir": str(sample_root.resolve()),
                "report_path": str(report_path.resolve()),
                "report": _read_json(report_path),
                "selected_unwrapped_path": str(selected_unwrapped.resolve()),
                "role_unwrapped_path": str(selected_unwrapped.resolve()),
            },
            "extractors": {
                "pyfing": {
                    "source_image_path": str(selected_unwrapped.resolve()),
                    "minutiae_json_path": str(pyfing_json.resolve()),
                    "minutiae_csv_path": str(pyfing_csv.resolve()),
                    "overlay_path": str(pyfing_overlay.resolve()),
                    "minutiae_count": int(len(pyfing_rows)),
                },
                "fingerflow": {
                    "source_image_path": str(selected_unwrapped.resolve()),
                    "minutiae_json_path": str(fingerflow_json.resolve()),
                    "minutiae_csv_path": str(fingerflow_csv.resolve()),
                    "overlay_path": str(fingerflow_overlay.resolve()),
                    "core_csv_path": str((fingerflow_root / "core.csv").resolve()) if (fingerflow_root / "core.csv").exists() else None,
                    "minutiae_count": int(len(fingerflow_rows)),
                },
            },
        }
    return prepared


def _filter_minutiae(rows: list[dict[str, Any]], threshold: float) -> pd.DataFrame:
    filtered = [row for row in rows if float(row.get("score", 0.0)) >= float(threshold)]
    if not filtered:
        return pd.DataFrame(columns=["x", "y", "angle", "score"])
    return pd.DataFrame(filtered, columns=["x", "y", "angle", "score"])


def _score_pair_for_extractor(
    pair: PairSpec,
    thresholds: tuple[float, ...],
    methods: tuple[str, ...],
    extractor_name: str,
    sample_cache: dict[str, dict[str, Any]],
    score_cache: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    sample_a = score_cache[pair.a.sample_id][extractor_name]
    sample_b = score_cache[pair.b.sample_id][extractor_name]
    rows_a = _load_minutiae_csv(Path(sample_a["minutiae_csv_path"]))
    rows_b = _load_minutiae_csv(Path(sample_b["minutiae_csv_path"]))
    threshold_scores: dict[str, dict[str, float]] = {}
    filtered_counts: dict[str, dict[str, int]] = {}
    for threshold in thresholds:
        threshold_key = f"{float(threshold):g}"
        frame_a = _filter_minutiae(rows_a, threshold)
        frame_b = _filter_minutiae(rows_b, threshold)
        method_scores: dict[str, float] = {}
        for method in methods:
            normalized_method = str(method).upper()
            if normalized_method in {"LSA-CENTROID", "LSA-R-CENTROID"}:
                score, _, _ = mcc.match_minutiae_csv_centroid_details(
                    frame_a,
                    frame_b,
                    method=normalized_method,
                )
            else:
                descriptors_a = mcc.build_descriptors(frame_a)
                descriptors_b = mcc.build_descriptors(frame_b)
                score, _ = mcc.match_descriptors(descriptors_a, descriptors_b, method=normalized_method)
            method_scores[method] = float(score)
        threshold_scores[threshold_key] = method_scores
        filtered_counts[threshold_key] = {"a": int(len(frame_a)), "b": int(len(frame_b))}
    return {
        "bucket": pair.bucket,
        "pair_index": pair.pair_index,
        "sample_a": {
            "sample_id": pair.a.sample_id,
            "subject_id": pair.a.subject_id,
            "finger_id": pair.a.finger_id,
            "acquisition_id": pair.a.acquisition_id,
            "view_role": pair.a.view_role,
            "reconstruction_dir": pair.a.reconstruction_dir,
            "unwarp_report_path": sample_cache[pair.a.sample_id]["unwarp"]["report_path"],
            "selected_unwrapped_path": sample_cache[pair.a.sample_id]["unwarp"]["selected_unwrapped_path"],
            "minutiae_json_path": sample_a["minutiae_json_path"],
            "minutiae_csv_path": sample_a["minutiae_csv_path"],
            "overlay_path": sample_a["overlay_path"],
        },
        "sample_b": {
            "sample_id": pair.b.sample_id,
            "subject_id": pair.b.subject_id,
            "finger_id": pair.b.finger_id,
            "acquisition_id": pair.b.acquisition_id,
            "view_role": pair.b.view_role,
            "reconstruction_dir": pair.b.reconstruction_dir,
            "unwarp_report_path": sample_cache[pair.b.sample_id]["unwarp"]["report_path"],
            "selected_unwrapped_path": sample_cache[pair.b.sample_id]["unwarp"]["selected_unwrapped_path"],
            "minutiae_json_path": sample_b["minutiae_json_path"],
            "minutiae_csv_path": sample_b["minutiae_csv_path"],
            "overlay_path": sample_b["overlay_path"],
        },
        "threshold_scores": threshold_scores,
        "filtered_counts": filtered_counts,
    }


def _score_pair_bucket(
    pair_specs: list[PairSpec],
    *,
    thresholds: tuple[float, ...],
    methods: tuple[str, ...],
    extractor_name: str,
    sample_cache: dict[str, dict[str, Any]],
    score_cache: dict[str, dict[str, dict[str, Any]]],
    workers: int,
) -> list[dict[str, Any]]:
    if not pair_specs:
        return []
    worker_count = max(1, min(int(workers), len(pair_specs)))
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp.get_context("spawn")) as executor:
        scored = list(
            executor.map(
                _score_pair_for_extractor,
                pair_specs,
                [thresholds] * len(pair_specs),
                [methods] * len(pair_specs),
                [extractor_name] * len(pair_specs),
                [sample_cache] * len(pair_specs),
                [score_cache] * len(pair_specs),
                chunksize=1,
            )
        )
    return scored


def _summarize_scores(scores: list[float]) -> dict[str, float | int | None]:
    if not scores:
        return {"count": 0, "average": None, "lowest": None, "highest": None}
    values = np.asarray(scores, dtype=np.float64)
    return {
        "count": int(values.size),
        "average": float(np.mean(values)),
        "lowest": float(np.min(values)),
        "highest": float(np.max(values)),
    }


def _summarize_bucket_results(
    pair_results: list[dict[str, Any]],
    thresholds: tuple[float, ...],
    methods: tuple[str, ...],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "pair_count": int(len(pair_results)),
        "pairs": pair_results,
        "methods": {},
    }
    for method in methods:
        method_summary: dict[str, Any] = {}
        for threshold in thresholds:
            threshold_key = f"{float(threshold):g}"
            scores = [float(pair["threshold_scores"][threshold_key][method]) for pair in pair_results]
            method_summary[threshold_key] = _summarize_scores(scores)
        summary["methods"][method] = method_summary
    return summary


def _build_extractor_report(
    *,
    extractor_name: str,
    pair_sets: dict[str, list[PairSpec]],
    pair_results_by_bucket: dict[str, list[dict[str, Any]]],
    thresholds: tuple[float, ...],
    methods: tuple[str, ...],
    sample_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "extractor": extractor_name,
        "pair_count": int(len(next(iter(pair_sets.values()), []))),
        "sample_count": int(len(sample_cache)),
        "buckets": {
            bucket: _summarize_bucket_results(pair_results_by_bucket[bucket], thresholds, methods)
            for bucket in pair_sets
        },
    }


def _default_output_path() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / timestamp / "report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ground-truth-root",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_ROOT,
        help="Ground-truth bundle root containing samples/*/meta.json.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Optional existing cache root containing samples/*/{unwarp,extractors}. If set, skip extraction and only run matching.",
    )
    parser.add_argument(
        "--pair-count",
        type=int,
        default=100,
        help="Number of unique pairs to sample per bucket.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        help="Minutiae score thresholds to evaluate.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="MCC matching methods to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for deterministic pair sampling.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of worker processes used for sample preparation and pair scoring.",
    )
    parser.add_argument(
        "--fingerflow-model-dir",
        type=Path,
        default=mcc.DEFAULT_FINGERFLOW_MODEL_DIR,
        help="Directory containing or caching FingerFlow model weights.",
    )
    parser.add_argument(
        "--left-angle",
        type=float,
        default=float(getattr(UNWARP, "DEFAULT_VIEW_ANGLES_DEG", {}).get("left", -45.0)),
        help="Left side camera angle in degrees for unwarping.",
    )
    parser.add_argument(
        "--right-angle",
        type=float,
        default=float(getattr(UNWARP, "DEFAULT_VIEW_ANGLES_DEG", {}).get("right", 45.0)),
        help="Right side camera angle in degrees for unwarping.",
    )
    parser.add_argument(
        "--samples-per-pixel",
        type=float,
        default=2.0,
        help="Canonical ellipse samples per front-view pixel for unwarping.",
    )
    parser.add_argument(
        "--unwarp-output-height",
        type=int,
        default=None,
        help="Optional fixed unwarp output height.",
    )
    parser.add_argument(
        "--unwarp-output-width",
        type=int,
        default=None,
        help="Optional fixed unwarp output width.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ground_truth_root = args.ground_truth_root.resolve()
    thresholds = tuple(float(value) for value in args.thresholds)
    methods = tuple(str(method).upper() for method in args.methods)
    if args.pair_count <= 0:
        raise ValueError("--pair-count must be positive")
    if not thresholds:
        raise ValueError("--thresholds must contain at least one value")
    if not methods:
        raise ValueError("--methods must contain at least one value")
    if (args.unwarp_output_height is None) != (args.unwarp_output_width is None):
        raise ValueError("--unwarp-output-height and --unwarp-output-width must be set together")

    records = _load_sample_records(ground_truth_root)
    output_root = (args.output.parent if args.output is not None else _default_output_path().parent).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.cache_root is not None:
        cache_root = args.cache_root.resolve()
        if not cache_root.exists():
            raise FileNotFoundError(f"cache root not found: {cache_root}")
        cached_records = []
        for record in records:
            sample_root = cache_root / "samples" / record.sample_id
            if not sample_root.exists():
                continue
            if not all(
                path.exists()
                for path in (
                    sample_root / "unwarp" / "selected_unwrapped.png",
                    sample_root / "unwarp" / "algorithm1_depth_then_stitched_branch_midpoint_unwarp_report.json",
                    sample_root / "extractors" / "pyfing" / "minutiae.csv",
                    sample_root / "extractors" / "pyfing" / "minutiae.json",
                    sample_root / "extractors" / "pyfing" / "minutiae_overlay.png",
                    sample_root / "extractors" / "fingerflow" / "minutiae.csv",
                    sample_root / "extractors" / "fingerflow" / "minutiae.json",
                    sample_root / "extractors" / "fingerflow" / "minutiae_overlay.png",
                )
            ):
                continue
            cached_records.append(record)
        sample_cache = _load_sample_cache_from_existing(cached_records, cache_root)
    else:
        pair_sets = sample_pair_sets(records, pair_count=int(args.pair_count), seed=int(args.seed))
        unique_records = {
            record.sample_id: record
            for pair_list in pair_sets.values()
            for pair in pair_list
            for record in (pair.a, pair.b)
        }
        unique_records_list = [unique_records[key] for key in sorted(unique_records)]
        cache_root = output_root / "cache"
        cache_root.mkdir(parents=True, exist_ok=True)

        model_paths = mcc.ensure_fingerflow_models(args.fingerflow_model_dir)
        sample_cache = _prepare_sample_caches(
            unique_records_list,
            cache_root,
            left_angle=float(args.left_angle),
            right_angle=float(args.right_angle),
            samples_per_pixel=float(args.samples_per_pixel),
            unwarp_output_height=args.unwarp_output_height,
            unwarp_output_width=args.unwarp_output_width,
            model_paths=model_paths,
            workers=int(args.workers),
        )
    score_cache = _build_score_cache(sample_cache)
    pair_sets = sample_pair_sets(
        [record for record in records if record.sample_id in sample_cache],
        pair_count=int(args.pair_count),
        seed=int(args.seed),
    )

    extractor_reports: dict[str, Any] = {}
    for extractor_name in DEFAULT_EXTRACTORS:
        pair_results_by_bucket = {
            bucket: _score_pair_bucket(
                pair_sets[bucket],
                thresholds=thresholds,
                methods=methods,
                extractor_name=extractor_name,
                sample_cache=sample_cache,
                score_cache=score_cache,
                workers=int(args.workers),
            )
            for bucket in pair_sets
        }
        extractor_reports[extractor_name] = _build_extractor_report(
            extractor_name=extractor_name,
            pair_sets=pair_sets,
            pair_results_by_bucket=pair_results_by_bucket,
            thresholds=thresholds,
            methods=methods,
            sample_cache=sample_cache,
        )

    report = {
        "ground_truth_root": str(ground_truth_root.resolve()),
        "seed": int(args.seed),
        "workers": int(args.workers),
        "pair_count": int(args.pair_count),
        "thresholds": [float(value) for value in thresholds],
        "methods": list(methods),
        "extractors": extractor_reports,
        "record_counts": {
            "total": int(len(records)),
            "front": int(sum(1 for record in records if record.view_role == FRONT_ROLE)),
            "side": int(sum(1 for record in records if record.view_role in SIDE_ROLES)),
        },
        "cache_root": str(cache_root.resolve()),
        "pair_sets": {
            bucket: [
                {
                    "pair_index": pair.pair_index,
                    "sample_a": pair.a.sample_id,
                    "sample_b": pair.b.sample_id,
                }
                for pair in pair_list
            ]
            for bucket, pair_list in pair_sets.items()
        },
    }

    output_path = args.output.resolve() if args.output is not None else _default_output_path().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"report": str(output_path), "ground_truth_root": str(ground_truth_root)}, indent=2))
    print()
    print("extractor | bucket | method | threshold | count | average | lowest | highest")
    print("-" * 88)
    for extractor_name, extractor_report in report["extractors"].items():
        for bucket_name, bucket_report in extractor_report["buckets"].items():
            for method, method_summary in bucket_report["methods"].items():
                for threshold, stats in method_summary.items():
                    average = "n/a" if stats["average"] is None else f"{stats['average']:.6f}"
                    lowest = "n/a" if stats["lowest"] is None else f"{stats['lowest']:.6f}"
                    highest = "n/a" if stats["highest"] is None else f"{stats['highest']:.6f}"
                    print(
                        f"{extractor_name} | {bucket_name} | {method} | {threshold} | {stats['count']} | {average} | {lowest} | {highest}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

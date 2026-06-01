#!/usr/bin/env python
"""Audit MCC match scores on ground-truth extracted minutiae."""

from __future__ import annotations

import argparse
import json
import importlib.util
import multiprocessing as mp
import os
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

from dataclasses import dataclass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
import main as mcc  # noqa: E402


DEFAULT_GROUND_TRUTH_ROOT = REPO_ROOT / "ground_truth" / "DS123_merged_v5"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "ground_truth_mcc_audit"
DEFAULT_THRESHOLDS = (0.7, 0.8, 0.9)
DEFAULT_METHODS = ("LSA", "LSA-R")
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
    raw_image_path: str
    sample_dir: str
    meta_path: str
    minutiae_json_path: str

    @property
    def identity_key(self) -> tuple[int, int]:
        return self.subject_id, self.finger_id

    @property
    def view_group(self) -> str:
        return FRONT_ROLE if self.view_role == FRONT_ROLE else "side"


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


def _normalized_record_path(path: str | Path) -> str:
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
            if not isinstance(gt, dict):
                continue
            view_role = str(gt.get("view_role") or "").strip().lower()
            if view_role not in {FRONT_ROLE, *SIDE_ROLES}:
                continue
            sample_dir = meta_path.parent
            minutiae_json_path = sample_dir / "minutiae.json"
            if not minutiae_json_path.exists():
                continue
            records.append(
                SampleRecord(
                    sample_id=str(meta["sample_id"]),
                    subject_id=int(meta["subject_id"]),
                    finger_id=int(meta["finger_id"]),
                    acquisition_id=int(meta["acquisition_id"]),
                    view_role=view_role,
                    raw_image_path=_normalized_record_path(meta["raw_image_path"]),
                    sample_dir=_normalized_record_path(sample_dir),
                    meta_path=_normalized_record_path(meta_path),
                    minutiae_json_path=_normalized_record_path(minutiae_json_path),
                )
            )
        except Exception as exc:
            print(f"[warn] skipping malformed sample metadata {meta_path}: {exc}", file=sys.stderr)
    return records


def _group_by_identity(records: Iterable[SampleRecord]) -> dict[tuple[int, int], list[SampleRecord]]:
    groups: dict[tuple[int, int], list[SampleRecord]] = {}
    for record in records:
        groups.setdefault(record.identity_key, []).append(record)
    return groups


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
            candidates=genuine_candidates, pair_count=pair_count, rng=rng
        ),
        IMPOSTOR_BUCKET: _sample_unique_impostor_pairs(
            front_by_identity=front_by_identity, pair_count=pair_count, rng=rng
        ),
    }


def _load_minutiae_rows(minutiae_json_path: str | Path) -> list[dict[str, Any]]:
    payload = _read_json(Path(minutiae_json_path))
    rows: Any
    if isinstance(payload, dict):
        rows = payload.get("minutiae", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        raise TypeError(f"unexpected minutiae JSON payload type: {type(payload).__name__}")

    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            x = float(row["x"])
            y = float(row["y"])
            angle_value = row.get("angle", row.get("theta"))
            if angle_value is None:
                continue
            angle = float(angle_value)
            score = float(row.get("score", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        normalized.append({"x": x, "y": y, "angle": angle, "score": score})
    return normalized


def _minutiae_frame(rows: list[dict[str, Any]], threshold: float) -> pd.DataFrame:
    filtered = [row for row in rows if float(row.get("score", 0.0)) >= float(threshold)]
    if not filtered:
        return pd.DataFrame(columns=["x", "y", "angle", "score"])
    return pd.DataFrame(filtered, columns=["x", "y", "angle", "score"])


def _score_pair(
    pair: PairSpec,
    thresholds: tuple[float, ...],
    methods: tuple[str, ...],
) -> dict[str, Any]:
    rows_a = _load_minutiae_rows(pair.a.minutiae_json_path)
    rows_b = _load_minutiae_rows(pair.b.minutiae_json_path)

    threshold_scores: dict[str, dict[str, float]] = {}
    filtered_counts: dict[str, dict[str, int]] = {}
    for threshold in thresholds:
        threshold_key = f"{float(threshold):g}"
        frame_a = _minutiae_frame(rows_a, threshold)
        frame_b = _minutiae_frame(rows_b, threshold)
        descriptors_a = mcc.build_descriptors(frame_a)
        descriptors_b = mcc.build_descriptors(frame_b)
        filtered_counts[threshold_key] = {
            "a": int(len(frame_a)),
            "b": int(len(frame_b)),
        }

        method_scores: dict[str, float] = {}
        for method in methods:
            score, _ = mcc.match_descriptors(descriptors_a, descriptors_b, method=method)
            method_scores[method] = float(score)
        threshold_scores[threshold_key] = method_scores

    return {
        "bucket": pair.bucket,
        "pair_index": pair.pair_index,
        "sample_a": {
            "sample_id": pair.a.sample_id,
            "subject_id": pair.a.subject_id,
            "finger_id": pair.a.finger_id,
            "acquisition_id": pair.a.acquisition_id,
            "view_role": pair.a.view_role,
            "minutiae_json_path": pair.a.minutiae_json_path,
        },
        "sample_b": {
            "sample_id": pair.b.sample_id,
            "subject_id": pair.b.subject_id,
            "finger_id": pair.b.finger_id,
            "acquisition_id": pair.b.acquisition_id,
            "view_role": pair.b.view_role,
            "minutiae_json_path": pair.b.minutiae_json_path,
        },
        "threshold_scores": threshold_scores,
        "filtered_counts": filtered_counts,
    }


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


def build_report(
    *,
    ground_truth_root: Path,
    records: list[SampleRecord],
    pair_sets: dict[str, list[PairSpec]],
    pair_results_by_bucket: dict[str, list[dict[str, Any]]],
    thresholds: tuple[float, ...],
    methods: tuple[str, ...],
    pair_count: int,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    return {
        "ground_truth_root": str(ground_truth_root.resolve()),
        "seed": int(seed),
        "workers": int(workers),
        "thresholds": [float(value) for value in thresholds],
        "methods": list(methods),
        "record_counts": {
            "total": int(len(records)),
            "front": int(sum(1 for record in records if record.view_role == FRONT_ROLE)),
            "side": int(sum(1 for record in records if record.view_role in SIDE_ROLES)),
        },
        "pair_count": int(pair_count),
        "buckets": {
            bucket: _summarize_bucket_results(pair_results_by_bucket[bucket], thresholds, methods)
            for bucket in pair_sets
        },
    }


def _run_parallel_scoring(
    pair_sets: dict[str, list[PairSpec]],
    thresholds: tuple[float, ...],
    methods: tuple[str, ...],
    workers: int,
) -> dict[str, list[dict[str, Any]]]:
    all_pairs = pair_sets[GENUINE_BUCKET] + pair_sets[IMPOSTOR_BUCKET]
    if not all_pairs:
        return {GENUINE_BUCKET: [], IMPOSTOR_BUCKET: []}

    worker_count = max(1, min(int(workers), len(all_pairs)))
    scored: list[dict[str, Any]]
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp.get_context("spawn")) as executor:
        scored = list(executor.map(_score_pair, all_pairs, [thresholds] * len(all_pairs), [methods] * len(all_pairs), chunksize=1))

    split = len(pair_sets[GENUINE_BUCKET])
    return {
        GENUINE_BUCKET: scored[:split],
        IMPOSTOR_BUCKET: scored[split:],
    }


def _print_summary_table(report: dict[str, Any]) -> None:
    print()
    print("bucket | method | threshold | count | average | lowest | highest")
    print("-" * 79)
    for bucket_name, bucket in report["buckets"].items():
        for method, method_summary in bucket["methods"].items():
            for threshold, stats in method_summary.items():
                average = "n/a" if stats["average"] is None else f"{stats['average']:.6f}"
                lowest = "n/a" if stats["lowest"] is None else f"{stats['lowest']:.6f}"
                highest = "n/a" if stats["highest"] is None else f"{stats['highest']:.6f}"
                print(
                    f"{bucket_name} | {method} | {threshold} | {stats['count']} | {average} | {lowest} | {highest}"
                )


def _default_output_path() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / timestamp / "report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ground-truth-root",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_ROOT,
        help="Ground-truth bundle root containing samples/*/meta.json and minutiae.json files.",
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
        help="Number of worker processes used for pair scoring.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path. Defaults to tmp/ground_truth_mcc_audit/<timestamp>/report.json.",
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

    records = _load_sample_records(ground_truth_root)
    pair_sets = sample_pair_sets(records, pair_count=int(args.pair_count), seed=int(args.seed))
    pair_results_by_bucket = _run_parallel_scoring(pair_sets, thresholds, methods, int(args.workers))
    report = build_report(
        ground_truth_root=ground_truth_root,
        records=records,
        pair_sets=pair_sets,
        pair_results_by_bucket=pair_results_by_bucket,
        thresholds=thresholds,
        methods=methods,
        pair_count=int(args.pair_count),
        seed=int(args.seed),
        workers=int(args.workers),
    )

    output_path = args.output.resolve() if args.output is not None else _default_output_path().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"report": str(output_path), "ground_truth_root": str(ground_truth_root)}, indent=2))
    _print_summary_table(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

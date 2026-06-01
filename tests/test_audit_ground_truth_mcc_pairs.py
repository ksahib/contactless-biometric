from __future__ import annotations

import importlib.util
import json
import sys
import sysconfig
from pathlib import Path


def _ensure_stdlib_copy_module() -> None:
    stdlib_copy = Path(sysconfig.get_paths()["stdlib"]) / "copy.py"
    spec = importlib.util.spec_from_file_location("copy", stdlib_copy)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not resolve stdlib copy module from {stdlib_copy}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = module
    spec.loader.exec_module(module)


_ensure_stdlib_copy_module()

import pandas as pd
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_ground_truth_mcc_pairs.py"
SPEC = importlib.util.spec_from_file_location("ground_truth_mcc_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def _record(sample_id: str, subject_id: int, finger_id: int, view_role: str, minutiae_path: Path) -> audit.SampleRecord:
    return audit.SampleRecord(
        sample_id=sample_id,
        subject_id=subject_id,
        finger_id=finger_id,
        acquisition_id=1,
        view_role=view_role,
        raw_image_path=f"/tmp/{sample_id}.png",
        sample_dir=str(minutiae_path.parent),
        meta_path=str(minutiae_path.parent / "meta.json"),
        minutiae_json_path=str(minutiae_path),
    )


def _write_minutiae_json(path: Path, rows: list[dict[str, float]]) -> None:
    path.write_text(json.dumps({"minutiae": rows}, indent=2), encoding="utf-8")


def test_sample_pair_sets_are_deterministic_and_role_correct(tmp_path: Path) -> None:
    records: list[audit.SampleRecord] = []
    for subject_id in (1, 2, 3):
        for finger_id in (1, 2):
            sample_root = tmp_path / f"s{subject_id}_f{finger_id}"
            sample_root.mkdir(parents=True, exist_ok=True)
            minutiae_path = sample_root / "minutiae.json"
            _write_minutiae_json(minutiae_path, [{"x": 1.0, "y": 2.0, "theta": 0.1, "score": 0.95}])
            records.append(_record(f"front_{subject_id}_{finger_id}", subject_id, finger_id, "front", minutiae_path))
            records.append(_record(f"left_{subject_id}_{finger_id}", subject_id, finger_id, "left", minutiae_path))
            records.append(_record(f"right_{subject_id}_{finger_id}", subject_id, finger_id, "right", minutiae_path))

    first = audit.sample_pair_sets(records, pair_count=5, seed=11)
    second = audit.sample_pair_sets(records, pair_count=5, seed=11)

    assert [pair.pair_key for pair in first[audit.GENUINE_BUCKET]] == [pair.pair_key for pair in second[audit.GENUINE_BUCKET]]
    assert [pair.pair_key for pair in first[audit.IMPOSTOR_BUCKET]] == [pair.pair_key for pair in second[audit.IMPOSTOR_BUCKET]]
    assert len({pair.pair_key for pair in first[audit.GENUINE_BUCKET]}) == 5
    assert len({pair.pair_key for pair in first[audit.IMPOSTOR_BUCKET]}) == 5

    for pair in first[audit.GENUINE_BUCKET]:
        assert pair.a.identity_key == pair.b.identity_key
        assert pair.a.view_role == "front"
        assert pair.b.view_role in {"left", "right"}
    for pair in first[audit.IMPOSTOR_BUCKET]:
        assert pair.a.identity_key != pair.b.identity_key
        assert pair.a.view_role == "front"
        assert pair.b.view_role == "front"


def test_threshold_filtering_happens_before_descriptor_building(tmp_path: Path) -> None:
    minu_a = tmp_path / "a" / "minutiae.json"
    minu_b = tmp_path / "b" / "minutiae.json"
    minu_a.parent.mkdir(parents=True, exist_ok=True)
    minu_b.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"x": 1.0, "y": 1.0, "theta": 0.1, "score": 0.6},
        {"x": 2.0, "y": 2.0, "theta": 0.2, "score": 0.8},
        {"x": 3.0, "y": 3.0, "theta": 0.3, "score": 0.95},
    ]
    _write_minutiae_json(minu_a, rows)
    _write_minutiae_json(minu_b, rows)
    pair = audit.PairSpec(
        bucket=audit.GENUINE_BUCKET,
        pair_index=0,
        a=_record("a", 1, 1, "front", minu_a),
        b=_record("b", 1, 1, "left", minu_b),
    )

    build_calls: list[int] = []

    def fake_build_descriptors(frame: pd.DataFrame) -> list[str]:
        build_calls.append(len(frame))
        return ["descriptor"]

    def fake_match_descriptors(descriptors_a, descriptors_b, method: str) -> tuple[float, object]:
        return float(len(descriptors_a) + len(descriptors_b)), object()

    with (
        mock.patch.object(audit.mcc, "build_descriptors", side_effect=fake_build_descriptors),
        mock.patch.object(audit.mcc, "match_descriptors", side_effect=fake_match_descriptors),
    ):
        result = audit._score_pair(pair, (0.7, 0.8, 0.9), ("LSA", "LSA-R"))

    assert build_calls == [2, 2, 2, 2, 1, 1]
    assert result["threshold_scores"]["0.7"]["LSA"] == 2.0
    assert result["threshold_scores"]["0.8"]["LSA-R"] == 2.0
    assert result["threshold_scores"]["0.9"]["LSA"] == 2.0


def test_report_aggregation_includes_buckets_methods_and_thresholds() -> None:
    pair_results = {
        audit.GENUINE_BUCKET: [
            {
                "pair_index": 0,
                "threshold_scores": {
                    "0.7": {"LSA": 0.1, "LSA-R": 0.2},
                    "0.8": {"LSA": 0.3, "LSA-R": 0.4},
                    "0.9": {"LSA": 0.5, "LSA-R": 0.6},
                },
            },
            {
                "pair_index": 1,
                "threshold_scores": {
                    "0.7": {"LSA": 0.2, "LSA-R": 0.3},
                    "0.8": {"LSA": 0.4, "LSA-R": 0.5},
                    "0.9": {"LSA": 0.6, "LSA-R": 0.7},
                },
            },
        ],
        audit.IMPOSTOR_BUCKET: [
            {
                "pair_index": 0,
                "threshold_scores": {
                    "0.7": {"LSA": 0.01, "LSA-R": 0.02},
                    "0.8": {"LSA": 0.03, "LSA-R": 0.04},
                    "0.9": {"LSA": 0.05, "LSA-R": 0.06},
                },
            },
            {
                "pair_index": 1,
                "threshold_scores": {
                    "0.7": {"LSA": 0.11, "LSA-R": 0.12},
                    "0.8": {"LSA": 0.13, "LSA-R": 0.14},
                    "0.9": {"LSA": 0.15, "LSA-R": 0.16},
                },
            },
        ],
    }
    pair_sets = {
        audit.GENUINE_BUCKET: [],
        audit.IMPOSTOR_BUCKET: [],
    }
    report = audit.build_report(
        ground_truth_root=Path("/tmp/ground_truth"),
        records=[],
        pair_sets=pair_sets,
        pair_results_by_bucket=pair_results,
        thresholds=(0.7, 0.8, 0.9),
        methods=("LSA", "LSA-R"),
        pair_count=2,
        seed=7,
        workers=2,
    )

    assert audit.GENUINE_BUCKET in report["buckets"]
    assert audit.IMPOSTOR_BUCKET in report["buckets"]
    assert set(report["buckets"][audit.GENUINE_BUCKET]["methods"]) == {"LSA", "LSA-R"}
    assert set(report["buckets"][audit.GENUINE_BUCKET]["methods"]["LSA"]) == {"0.7", "0.8", "0.9"}
    assert report["buckets"][audit.GENUINE_BUCKET]["methods"]["LSA"]["0.7"]["count"] == 2
    assert report["buckets"][audit.GENUINE_BUCKET]["methods"]["LSA"]["0.7"]["average"] == 0.15000000000000002
    assert report["buckets"][audit.IMPOSTOR_BUCKET]["methods"]["LSA-R"]["0.9"]["highest"] == 0.16

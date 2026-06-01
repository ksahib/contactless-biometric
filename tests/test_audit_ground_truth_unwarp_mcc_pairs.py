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

from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_ground_truth_unwarp_mcc_pairs.py"
SPEC = importlib.util.spec_from_file_location("ground_truth_unwarp_mcc_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def _sample_record(sample_id: str, subject_id: int, finger_id: int, view_role: str, reconstruction_dir: Path) -> audit.SampleRecord:
    return audit.SampleRecord(
        sample_id=sample_id,
        subject_id=subject_id,
        finger_id=finger_id,
        acquisition_id=1,
        view_role=view_role,
        reconstruction_dir=str(reconstruction_dir),
        meta_path=str(reconstruction_dir / "meta.json"),
        sample_dir=str(reconstruction_dir.parent),
    )


def test_sampler_is_deterministic_and_role_correct(tmp_path: Path) -> None:
    records: list[audit.SampleRecord] = []
    for subject_id in (1, 2, 3):
        for finger_id in (1, 2):
            reconstruction_dir = tmp_path / f"s{subject_id}_f{finger_id}"
            reconstruction_dir.mkdir(parents=True, exist_ok=True)
            records.append(_sample_record(f"front_{subject_id}_{finger_id}", subject_id, finger_id, "front", reconstruction_dir))
            records.append(_sample_record(f"left_{subject_id}_{finger_id}", subject_id, finger_id, "left", reconstruction_dir))
            records.append(_sample_record(f"right_{subject_id}_{finger_id}", subject_id, finger_id, "right", reconstruction_dir))

    first = audit.sample_pair_sets(records, pair_count=5, seed=11)
    second = audit.sample_pair_sets(records, pair_count=5, seed=11)

    assert [pair.pair_key for pair in first[audit.GENUINE_BUCKET]] == [pair.pair_key for pair in second[audit.GENUINE_BUCKET]]
    assert [pair.pair_key for pair in first[audit.IMPOSTOR_BUCKET]] == [pair.pair_key for pair in second[audit.IMPOSTOR_BUCKET]]

    for pair in first[audit.GENUINE_BUCKET]:
        assert pair.a.identity_key == pair.b.identity_key
        assert pair.a.view_role == "front"
        assert pair.b.view_role in {"left", "right"}
    for pair in first[audit.IMPOSTOR_BUCKET]:
        assert pair.a.identity_key != pair.b.identity_key
        assert pair.a.view_role == "front"
        assert pair.b.view_role == "front"


def test_unwarp_and_extraction_use_unwrapped_image(tmp_path: Path) -> None:
    reconstruction_dir = tmp_path / "recon"
    reconstruction_dir.mkdir(parents=True)
    record = _sample_record("front_1_1", 1, 1, "front", reconstruction_dir)
    cache_root = tmp_path / "cache"
    role_unwrapped = cache_root / "samples" / record.sample_id / "unwarp" / "front" / "front_algorithm3_unwrapped.png"
    role_unwrapped.parent.mkdir(parents=True, exist_ok=True)
    role_unwrapped.write_bytes(b"fake")

    with (
        mock.patch.object(audit.UNWARP, "run", return_value={"report_path": str(cache_root / "samples" / record.sample_id / "unwarp" / "algorithm1_depth_then_stitched_branch_midpoint_unwarp_report.json")}) as unwarp_run,
        mock.patch.object(audit, "_extract_pyfing_from_unwrapped", return_value={"minutiae_csv_path": str(tmp_path / "py.csv"), "minutiae_json_path": str(tmp_path / "py.json"), "overlay_path": str(tmp_path / "py.png"), "minutiae_count": 1}) as py_extract,
        mock.patch.object(audit, "_extract_fingerflow_from_unwrapped", return_value={"minutiae_csv_path": str(tmp_path / "ff.csv"), "minutiae_json_path": str(tmp_path / "ff.json"), "overlay_path": str(tmp_path / "ff.png"), "core_csv_path": str(tmp_path / "core.csv"), "minutiae_count": 1}) as ff_extract,
        mock.patch.object(audit, "_write_gray", autospec=True) as write_gray,
        mock.patch.object(audit, "_read_gray", return_value=audit.np.zeros((4, 4), dtype=audit.np.uint8)),
    ):
        result = audit._prepare_sample_cache(
            record,
            cache_root,
            left_angle=-45.0,
            right_angle=45.0,
            samples_per_pixel=2.0,
            unwarp_output_height=None,
            unwarp_output_width=None,
            model_paths=(Path("a"), Path("b"), Path("c"), Path("d")),
        )

    unwarp_run.assert_called_once()
    py_extract.assert_called_once()
    ff_extract.assert_called_once()
    assert Path(result["unwarp"]["selected_unwrapped_path"]).name == "selected_unwrapped.png"
    assert write_gray.called


def test_report_aggregation_includes_both_extractors_and_thresholds() -> None:
    pair_results = {
        audit.GENUINE_BUCKET: [
            {
                "pair_index": 0,
                "threshold_scores": {
                    "0.7": {"LSA": 0.1, "LSA-R": 0.2},
                    "0.8": {"LSA": 0.3, "LSA-R": 0.4},
                    "0.9": {"LSA": 0.5, "LSA-R": 0.6},
                },
                "sample_a": {},
                "sample_b": {},
            },
            {
                "pair_index": 1,
                "threshold_scores": {
                    "0.7": {"LSA": 0.2, "LSA-R": 0.3},
                    "0.8": {"LSA": 0.4, "LSA-R": 0.5},
                    "0.9": {"LSA": 0.6, "LSA-R": 0.7},
                },
                "sample_a": {},
                "sample_b": {},
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
                "sample_a": {},
                "sample_b": {},
            },
            {
                "pair_index": 1,
                "threshold_scores": {
                    "0.7": {"LSA": 0.11, "LSA-R": 0.12},
                    "0.8": {"LSA": 0.13, "LSA-R": 0.14},
                    "0.9": {"LSA": 0.15, "LSA-R": 0.16},
                },
                "sample_a": {},
                "sample_b": {},
            },
        ],
    }
    pair_sets = {audit.GENUINE_BUCKET: [], audit.IMPOSTOR_BUCKET: []}
    sample_cache = {"s1": {"unwarp": {"report_path": "u", "selected_unwrapped_path": "u"}, "extractors": {"pyfing": {}, "fingerflow": {}}}}
    report = audit._build_extractor_report(
        extractor_name="pyfing",
        pair_sets=pair_sets,
        pair_results_by_bucket=pair_results,
        thresholds=(0.7, 0.8, 0.9),
        methods=("LSA", "LSA-R"),
        sample_cache=sample_cache,
    )

    assert report["extractor"] == "pyfing"
    assert set(report["buckets"]) == {audit.GENUINE_BUCKET, audit.IMPOSTOR_BUCKET}
    assert set(report["buckets"][audit.GENUINE_BUCKET]["methods"]) == {"LSA", "LSA-R"}
    assert report["buckets"][audit.GENUINE_BUCKET]["methods"]["LSA"]["0.7"]["count"] == 2
    assert report["buckets"][audit.IMPOSTOR_BUCKET]["methods"]["LSA-R"]["0.9"]["highest"] == 0.16


def test_main_writes_two_extractor_branches(tmp_path: Path) -> None:
    records = [
        _sample_record("front_1_1", 1, 1, "front", tmp_path / "r1"),
        _sample_record("left_1_1", 1, 1, "left", tmp_path / "r1"),
        _sample_record("front_2_1", 2, 1, "front", tmp_path / "r2"),
        _sample_record("left_2_1", 2, 1, "left", tmp_path / "r2"),
    ]
    pair_sets = {
        audit.GENUINE_BUCKET: [audit.PairSpec(audit.GENUINE_BUCKET, 0, records[0], records[1])],
        audit.IMPOSTOR_BUCKET: [audit.PairSpec(audit.IMPOSTOR_BUCKET, 0, records[0], records[2])],
    }
    sample_cache = {
        "front_1_1": {
            "sample_id": "front_1_1",
            "view_role": "front",
            "subject_id": 1,
            "finger_id": 1,
            "acquisition_id": 1,
            "reconstruction_dir": str(tmp_path / "r1"),
            "meta_path": str(tmp_path / "r1" / "meta.json"),
            "sample_dir": str(tmp_path / "r1"),
            "unwarp": {"report_path": "u1", "selected_unwrapped_path": "u1"},
            "extractors": {"pyfing": {"minutiae_csv_path": "p1.csv", "minutiae_json_path": "p1.json", "overlay_path": "p1.png"}, "fingerflow": {"minutiae_csv_path": "f1.csv", "minutiae_json_path": "f1.json", "overlay_path": "f1.png"}},
        },
        "left_1_1": {
            "sample_id": "left_1_1",
            "view_role": "left",
            "subject_id": 1,
            "finger_id": 1,
            "acquisition_id": 1,
            "reconstruction_dir": str(tmp_path / "r1"),
            "meta_path": str(tmp_path / "r1" / "meta.json"),
            "sample_dir": str(tmp_path / "r1"),
            "unwarp": {"report_path": "u2", "selected_unwrapped_path": "u2"},
            "extractors": {"pyfing": {"minutiae_csv_path": "p2.csv", "minutiae_json_path": "p2.json", "overlay_path": "p2.png"}, "fingerflow": {"minutiae_csv_path": "f2.csv", "minutiae_json_path": "f2.json", "overlay_path": "f2.png"}},
        },
        "front_2_1": {
            "sample_id": "front_2_1",
            "view_role": "front",
            "subject_id": 2,
            "finger_id": 1,
            "acquisition_id": 1,
            "reconstruction_dir": str(tmp_path / "r2"),
            "meta_path": str(tmp_path / "r2" / "meta.json"),
            "sample_dir": str(tmp_path / "r2"),
            "unwarp": {"report_path": "u3", "selected_unwrapped_path": "u3"},
            "extractors": {"pyfing": {"minutiae_csv_path": "p3.csv", "minutiae_json_path": "p3.json", "overlay_path": "p3.png"}, "fingerflow": {"minutiae_csv_path": "f3.csv", "minutiae_json_path": "f3.json", "overlay_path": "f3.png"}},
        },
    }
    score_cache = {
        key: {
            "pyfing": {"minutiae_csv_path": f"py_{key}.csv", "minutiae_json_path": f"py_{key}.json", "overlay_path": f"py_{key}.png"},
            "fingerflow": {"minutiae_csv_path": f"ff_{key}.csv", "minutiae_json_path": f"ff_{key}.json", "overlay_path": f"ff_{key}.png"},
        }
        for key in sample_cache
    }

    class Args:
        ground_truth_root = tmp_path
        pair_count = 1
        thresholds = [0.7, 0.8, 0.9]
        methods = ["LSA", "LSA-R"]
        seed = 7
        workers = 1
        fingerflow_model_dir = tmp_path / "models"
        left_angle = -45.0
        right_angle = 45.0
        samples_per_pixel = 2.0
        unwarp_output_height = None
        unwarp_output_width = None
        output = tmp_path / "report.json"

    with (
        mock.patch.object(audit, "parse_args", return_value=Args()),
        mock.patch.object(audit, "_load_sample_records", return_value=records),
        mock.patch.object(audit, "sample_pair_sets", return_value=pair_sets),
        mock.patch.object(audit.mcc, "ensure_fingerflow_models", return_value=(Path("a"), Path("b"), Path("c"), Path("d"))),
        mock.patch.object(audit, "_prepare_sample_caches", return_value=sample_cache),
        mock.patch.object(audit, "_build_score_cache", return_value=score_cache),
        mock.patch.object(audit, "_score_pair_bucket", side_effect=lambda pair_specs, **kwargs: [
            {
                "bucket": pair_specs[0].bucket,
                "pair_index": pair_specs[0].pair_index,
                "threshold_scores": {
                    "0.7": {"LSA": 0.1, "LSA-R": 0.2},
                    "0.8": {"LSA": 0.3, "LSA-R": 0.4},
                    "0.9": {"LSA": 0.5, "LSA-R": 0.6},
                },
                "sample_a": {},
                "sample_b": {},
            }
        ]),
    ):
        assert audit.main() == 0

    report = json.loads((tmp_path / "report.json").read_text())
    assert set(report["extractors"]) == {"pyfing", "fingerflow"}
    assert set(report["extractors"]["pyfing"]["buckets"]) == {audit.GENUINE_BUCKET, audit.IMPOSTOR_BUCKET}
    assert set(report["extractors"]["fingerflow"]["buckets"]["front_side_same_finger"]["methods"]) == {"LSA", "LSA-R"}

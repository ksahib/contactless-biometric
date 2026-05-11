from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_featurenet_matching_inference.py"
SPEC = importlib.util.spec_from_file_location("featurenet_matching_script", MODULE_PATH)
matcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = matcher
assert SPEC.loader is not None
SPEC.loader.exec_module(matcher)


def _write_sample_meta(samples_root: Path, sample_id: str, payload: dict) -> None:
    sample_dir = samples_root / sample_id
    sample_dir.mkdir(parents=True)
    (sample_dir / "meta.json").write_text(json.dumps(payload), encoding="utf-8")


def _record(subject: int, finger: int, view: int, *, source: str = "DS1", acquisition: str = "1") -> matcher.SampleRecord:
    sample_id = f"{source}_s{subject:02d}_f{finger:02d}_a{acquisition}_v{view:02d}"
    return matcher.SampleRecord(
        sample_id=sample_id,
        source_label=source,
        subject_id=subject,
        finger_id=finger,
        acquisition_id=acquisition,
        raw_view_index=view,
        raw_image_path=f"/tmp/{sample_id}.jpg",
        image_exists=True,
        ground_truth_root="/tmp/gt",
    )


def test_discover_samples_reads_sample_meta_and_rewrites_stale_dataset_path(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    image_path = dataset_root / "DS1" / "1" / "raw" / "1_1_1_0.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"not-a-real-image-for-discovery-only")

    ground_truth_root = tmp_path / "ground_truth" / "DS1"
    samples_root = ground_truth_root / "samples"
    _write_sample_meta(
        samples_root,
        "s01_f01_a01_v00",
        {
            "sample_id": "s01_f01_a01_v00",
            "subject_id": 1,
            "finger_id": 1,
            "acquisition_id": 1,
            "raw_view_index": 0,
            "raw_image_path": r"Z:\old-pc\dataset\DS1\1\raw\1_1_1_0.jpg",
            "merge_source": {"label": "DS1", "subject_id": 1},
        },
    )

    records = matcher.discover_samples(
        ground_truth_root,
        dataset_root,
        front_views={0},
        side_views={1, 2},
    )

    assert len(records) == 1
    assert Path(records[0].raw_image_path) == image_path.resolve()
    assert records[0].image_exists is True
    assert records[0].source_label == "DS1"


def test_discover_samples_prefers_manifest_json_over_csv(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    json_image = dataset_root / "DS1" / "1" / "raw" / "1_1_1_0.jpg"
    csv_image = dataset_root / "DS1" / "1" / "raw" / "1_2_1_0.jpg"
    json_image.parent.mkdir(parents=True)
    json_image.write_bytes(b"x")
    csv_image.write_bytes(b"x")

    ground_truth_root = tmp_path / "ground_truth" / "DS1"
    ground_truth_root.mkdir(parents=True)
    (ground_truth_root / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "sample_id": "json_sample",
                    "subject_id": 1,
                    "finger_id": 1,
                    "acquisition_id": 1,
                    "raw_view_index": 0,
                    "raw_image_path": str(json_image),
                }
            ]
        ),
        encoding="utf-8",
    )
    with (ground_truth_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "subject_id", "finger_id", "acquisition_id", "raw_view_index", "raw_image_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "csv_sample",
                "subject_id": 1,
                "finger_id": 2,
                "acquisition_id": 1,
                "raw_view_index": 0,
                "raw_image_path": str(csv_image),
            }
        )

    records = matcher.discover_samples(
        ground_truth_root,
        dataset_root,
        front_views={0},
        side_views={1, 2},
    )

    assert [record.sample_id for record in records] == ["json_sample"]


def test_pair_sampler_covers_expected_buckets_and_prevents_duplicates() -> None:
    records = [
        _record(subject, finger, view)
        for subject in (1, 2, 3)
        for finger in (1, 2)
        for view in (0, 1, 2)
    ]

    pairs = matcher.sample_all_pair_buckets(
        records,
        front_views={0},
        side_views={1, 2},
        pairs_per_bucket=4,
        seed=7,
    )
    buckets = {pair.bucket for pair in pairs}

    assert "genuine_front_side" in buckets
    assert "genuine_side_side" in buckets
    assert "impostor_front_front_same_finger_type" in buckets
    assert "impostor_front_front_cross_finger_type" in buckets
    assert "impostor_side_side_same_finger_type" in buckets
    assert "impostor_side_side_cross_finger_type" in buckets
    assert "impostor_front_side_same_finger_type" in buckets
    assert "impostor_front_side_cross_finger_type" in buckets

    seen = set()
    for pair in pairs:
        key = matcher._pair_seen_key(pair.a, pair.b)
        assert key not in seen
        seen.add(key)
        if pair.label == "genuine":
            assert pair.a.identity_key == pair.b.identity_key
        else:
            assert pair.a.identity_key != pair.b.identity_key
        if pair.finger_relation == "same_finger_type":
            assert pair.a.finger_id == pair.b.finger_id
        if pair.finger_relation == "cross_finger_type":
            assert pair.a.finger_id != pair.b.finger_id


def test_summarize_matches_reports_count_average_lowest_highest() -> None:
    rows = [
        {"status": "ok", "label": "genuine", "bucket": "genuine_front_side", "score": 0.2},
        {"status": "ok", "label": "genuine", "bucket": "genuine_front_side", "score": 0.8},
        {"status": "ok", "label": "impostor", "bucket": "impostor_front_front_same_finger_type", "score": 0.1},
        {"status": "error", "label": "impostor", "bucket": "impostor_front_front_same_finger_type", "score": ""},
    ]

    summary = matcher.summarize_matches(rows)

    assert summary["overall"] == {"count": 3, "average": (0.2 + 0.8 + 0.1) / 3.0, "lowest": 0.1, "highest": 0.8}
    assert summary["by_bucket"]["genuine_front_side"] == {"count": 2, "average": 0.5, "lowest": 0.2, "highest": 0.8}
    assert summary["by_bucket"]["genuine_side_side"] == {"count": 0, "average": None, "lowest": None, "highest": None}

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_featurenet_binary_roc_sweep.py"
SPEC = importlib.util.spec_from_file_location("featurenet_binary_roc_sweep", MODULE_PATH)
sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sweep
assert SPEC.loader is not None
SPEC.loader.exec_module(sweep)


def _touch_raw(root: Path, dataset: str, subject: int, finger: int, acquisition: int, view: int) -> Path:
    path = root / dataset / str(subject) / "raw" / f"{subject}_{finger}_{acquisition}_{view}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image-placeholder")
    return path


def _record(dataset: str, subject: int, finger: int, acquisition: int, view: int) -> sweep.ImageRecord:
    return sweep.ImageRecord(
        dataset=dataset,
        subject_id=subject,
        finger_id=finger,
        acquisition_id=acquisition,
        view_index=view,
        image_path=f"/tmp/{dataset}_{subject}_{finger}_{acquisition}_{view}.jpg",
    )


def test_discover_archive_images_parses_ds_subject_raw_jpg_paths(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    front = _touch_raw(archive_root, "DS1", 1, 2, 1, 0)
    side = _touch_raw(archive_root, "DS1", 1, 2, 1, 2)
    ignored_view = _touch_raw(archive_root, "DS1", 1, 2, 1, 5)
    malformed = archive_root / "DS1" / "1" / "raw" / "not_a_sample.jpg"
    malformed.write_bytes(b"x")

    records = sweep.discover_archive_images(archive_root, side_views={1, 2})

    assert [Path(record.image_path) for record in records] == [front.resolve(), side.resolve()]
    assert all(Path(record.image_path) != ignored_view.resolve() for record in records)
    assert records[0].dataset == "DS1"
    assert records[0].subject_id == 1
    assert records[0].finger_id == 2
    assert records[0].acquisition_id == 1
    assert records[0].view_index == 0


def test_genuine_sampling_uses_front_side_same_finger_pairs() -> None:
    records = [
        _record("DS1", subject, finger, acquisition, view)
        for subject in (1, 2)
        for finger in (1, 2)
        for acquisition in (1, 2)
        for view in (0, 1, 2)
    ]
    rng = sweep.random.Random(7)

    pairs = sweep.sample_genuine_pairs(records, side_views={1, 2}, count=8, rng=rng)

    assert len(pairs) == 8
    for pair in pairs:
        assert pair.label == "genuine"
        assert pair.a.identity_key == pair.b.identity_key
        assert pair.a.view_index == 0
        assert pair.b.view_index in {1, 2}


def test_impostor_sampling_uses_front_front_different_identity_pairs() -> None:
    records = [
        _record("DS1", subject, finger, 1, view)
        for subject in (1, 2, 3)
        for finger in (1, 2)
        for view in (0, 1, 2)
    ]
    rng = sweep.random.Random(11)

    pairs = sweep.sample_impostor_pairs(records, count=10, rng=rng)

    assert len(pairs) == 10
    seen = set()
    for pair in pairs:
        assert pair.label == "impostor"
        assert pair.a.view_index == 0
        assert pair.b.view_index == 0
        assert pair.a.identity_key != pair.b.identity_key
        assert pair.pair_key not in seen
        seen.add(pair.pair_key)


def test_compute_roc_counts_uses_inclusive_match_threshold() -> None:
    rows = [
        {"status": "ok", "label": "genuine", "score": 0.50},
        {"status": "ok", "label": "genuine", "score": 0.20},
        {"status": "ok", "label": "impostor", "score": 0.50},
        {"status": "ok", "label": "impostor", "score": 0.10},
        {"status": "error", "label": "genuine", "score": ""},
    ]

    roc_rows = sweep.compute_roc_counts({0.8: rows}, matching_thresholds=[0.5])

    assert roc_rows == [
        {
            "feature_score_threshold": "0.80",
            "matching_threshold": "0.50",
            "TP": 1,
            "FN": 1,
            "FP": 1,
            "TN": 1,
            "TPR": 0.5,
            "FPR": 0.5,
            "TNR": 0.5,
            "FNR": 0.5,
            "accuracy": 0.5,
        }
    ]

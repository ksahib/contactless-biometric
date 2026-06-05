from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


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


def _touch_bundle(root: Path, sample_id: str, subject: int, finger: int, acquisition: int, view: int) -> Path:
    sample_dir = root / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "masked_image.png").write_bytes(b"masked")
    (sample_dir / "mask.png").write_bytes(b"mask")
    (sample_dir / "meta.json").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "subject_id": subject,
                "finger_id": finger,
                "acquisition_id": acquisition,
                "raw_view_index": view,
                "raw_image_path": f"/archive/DS9/{subject}/raw/{subject}_{finger}_{acquisition}_{view}.jpg",
            }
        ),
        encoding="utf-8",
    )
    return sample_dir


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


def test_discover_ground_truth_bundle_images_uses_saved_training_inputs(tmp_path: Path) -> None:
    gt_root = tmp_path / "gt"
    front = _touch_bundle(gt_root, "ds1_s01_f02_a01_v00", 1, 2, 1, 0)
    side = _touch_bundle(gt_root, "ds1_s01_f02_a01_v01", 1, 2, 1, 1)
    _touch_bundle(gt_root, "ds1_s01_f02_a01_v05", 1, 2, 1, 5)

    records = sweep.discover_ground_truth_bundle_images(gt_root, side_views={1, 2})

    assert [Path(record.image_path) for record in records] == [
        (front / "masked_image.png").resolve(),
        (side / "masked_image.png").resolve(),
    ]
    assert [Path(record.mask_path) for record in records] == [
        (front / "mask.png").resolve(),
        (side / "mask.png").resolve(),
    ]
    assert all(record.source_kind == "gt_bundle" for record in records)
    assert records[0].dataset == "DS1"
    assert records[0].identity_key == ("DS1", 1, 2)


def test_extract_gt_bundle_image_replays_saved_masked_input_without_crop(tmp_path: Path) -> None:
    sample_dir = _touch_bundle(tmp_path / "gt", "ds1_s01_f02_a01_v00", 1, 2, 1, 0)
    record = sweep.ImageRecord(
        dataset="DS1",
        subject_id=1,
        finger_id=2,
        acquisition_id=1,
        view_index=0,
        image_path=str((sample_dir / "masked_image.png").resolve()),
        mask_path=str((sample_dir / "mask.png").resolve()),
        source_kind="gt_bundle",
    )
    weights_path = tmp_path / "best.pt"
    weights_path.write_bytes(b"weights")
    calls: list[tuple[Path, Path]] = []

    def fake_preprocess_saved_masked_input(
        masked_image_path: Path,
        mask_path: Path,
        save_preprocess_dir: Path | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        calls.append((masked_image_path, mask_path))
        return torch.ones((1, 1, 8, 8)), torch.ones((1, 1, 8, 8)), (8, 8)

    def fake_run_inference(**kwargs: object) -> dict[str, torch.Tensor]:
        return {"dummy": torch.ones((1, 1, 1, 1))}

    def fake_decode_minutiae_rows(**kwargs: object) -> list[dict[str, float]]:
        return [{"x": 1.0, "y": 2.0, "angle": 0.0, "score": 0.9}]

    def fake_save_minutiae_csv(rows: list[dict[str, float]], output_csv: Path) -> None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_csv.write_text("x,y,angle,score\n1,2,0,0.9\n", encoding="utf-8")

    def fake_save_pose_sidecars(outputs: dict[str, torch.Tensor], output_dir: Path) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        orientation = output_dir / "orientation.npy"
        ridge = output_dir / "ridge_period.npy"
        orientation.write_bytes(b"orientation")
        ridge.write_bytes(b"ridge")
        return orientation, ridge

    def fake_save_mask_png(mask_tensor: torch.Tensor, path: Path) -> None:
        path.write_bytes(b"mask")

    row = sweep.extract_image(
        record,
        helpers={
            "preprocess_saved_masked_input": fake_preprocess_saved_masked_input,
            "run_inference": fake_run_inference,
            "decode_minutiae_rows": fake_decode_minutiae_rows,
            "save_minutiae_csv": fake_save_minutiae_csv,
            "save_pose_sidecars": fake_save_pose_sidecars,
            "_save_mask_png": fake_save_mask_png,
        },
        model=object(),
        device=torch.device("cpu"),
        weights_path=weights_path,
        score_threshold=0.5,
        solov2_score_thr=0.15,
        cache_root=tmp_path / "cache",
        reuse_cache=False,
        apply_nms=True,
    )

    assert row["status"] == "ok"
    assert row["source_kind"] == "gt_bundle"
    assert calls == [((sample_dir / "masked_image.png").resolve(), (sample_dir / "mask.png").resolve())]


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


def test_unique_methods_normalizes_and_deduplicates() -> None:
    assert sweep._unique_methods(["lsa", "lsa-r", "LSA-R", "LSA-CENTROID"]) == [
        "LSA",
        "LSA-R",
        "LSA-CENTROID",
    ]


def test_default_mcc_methods_include_lsa_variants_and_centroid() -> None:
    assert sweep.DEFAULT_MCC_METHODS == ("LSA", "LSA-R", "LSA-CENTROID")


def test_default_feature_score_thresholds_start_at_half() -> None:
    assert sweep.DEFAULT_FEATURE_SCORE_THRESHOLDS == (0.5, 0.6, 0.7, 0.8, 0.9)

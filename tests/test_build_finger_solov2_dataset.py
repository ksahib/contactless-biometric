from __future__ import annotations

import csv
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_finger_solov2_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_finger_solov2_dataset", MODULE_PATH)
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
assert SPEC.loader is not None
SPEC.loader.exec_module(builder)


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 8, 3), value, dtype=np.uint8)
    ok = cv2.imwrite(str(path), image)
    assert ok


def _build_dataset_root(tmp_path: Path, subjects_per_dataset: int = 14) -> Path:
    dataset_root = tmp_path / "dataset"
    for dataset_id, start_subject in (("DS1", 1), ("DS2", 101), ("DS3", 201)):
        for subject_offset in range(subjects_per_dataset):
            subject_id = start_subject + subject_offset
            raw_dir = dataset_root / dataset_id / str(subject_id) / "raw"
            for finger_id in range(1, 3):
                for acquisition_id in range(1, 3):
                    for view_index in range(6):
                        image_path = raw_dir / f"{subject_id}_{finger_id}_{acquisition_id}_{view_index}.jpg"
                        _write_image(image_path, value=(view_index * 30) + finger_id + acquisition_id)
    return dataset_root


def test_discover_raw_images_reads_ds1_ds2_ds3_raw_layout(tmp_path: Path) -> None:
    dataset_root = _build_dataset_root(tmp_path, subjects_per_dataset=1)

    records = builder.discover_raw_images(dataset_root)

    assert len(records) == 3 * 1 * 2 * 2 * 6
    categories = Counter(record.category for record in records)
    assert categories["front"] == 12
    assert categories["left_side"] == 12
    assert categories["right_side"] == 12
    assert categories["dark_low_contrast"] == 36


def test_sample_records_matches_counts_and_keeps_sources_unique(tmp_path: Path) -> None:
    dataset_root = _build_dataset_root(tmp_path)
    records = builder.discover_raw_images(dataset_root)

    selected = builder.sample_records(records, seed=builder.DEFAULT_SEED)

    assert len(selected) == 150
    split_category_counts = Counter((record.split, record.output_category) for record in selected)
    for split, categories in builder.SPLIT_COUNTS.items():
        for category, count in categories.items():
            assert split_category_counts[(split, category)] == count

    source_keys = [record.source.source_key for record in selected]
    assert len(source_keys) == len(set(source_keys))

    bright_records = [record for record in selected if record.output_category == "bright_overexposed"]
    assert len(bright_records) == 15
    assert all(record.source.category in builder.BRIGHT_SOURCE_CATEGORIES for record in bright_records)
    assert all(record.bright_transform_applied for record in bright_records)


def test_write_dataset_preserves_extensions_and_writes_manifest(tmp_path: Path) -> None:
    dataset_root = _build_dataset_root(tmp_path)
    records = builder.discover_raw_images(dataset_root)
    selected = builder.sample_records(records, seed=builder.DEFAULT_SEED)
    output_root = tmp_path / "finger_solov2_dataset"

    builder.prepare_output_root(output_root)
    manifest_path = builder.write_dataset(selected, output_root)

    train_files = sorted((output_root / "images" / "train").iterdir())
    val_files = sorted((output_root / "images" / "val").iterdir())
    assert len(train_files) == 120
    assert len(val_files) == 30
    assert train_files[0].name == "img_000001.jpg"
    assert train_files[-1].name == "img_000120.jpg"
    assert val_files[0].name == "img_000001.jpg"
    assert val_files[-1].name == "img_000030.jpg"

    rows = list(csv.DictReader(manifest_path.open("r", newline="", encoding="utf-8")))
    assert len(rows) == 150
    assert sum(row["split"] == "train" for row in rows) == 120
    assert sum(row["split"] == "val" for row in rows) == 30
    assert sum(row["bright_transform_applied"] == "true" for row in rows) == 15


def test_sample_records_is_reproducible_for_fixed_seed(tmp_path: Path) -> None:
    dataset_root = _build_dataset_root(tmp_path)
    records = builder.discover_raw_images(dataset_root)

    first = builder.sample_records(records, seed=builder.DEFAULT_SEED)
    second = builder.sample_records(records, seed=builder.DEFAULT_SEED)

    first_keys = [
        (record.split, record.output_category, record.source.source_key, record.bright_transform_applied)
        for record in first
    ]
    second_keys = [
        (record.split, record.output_category, record.source.source_key, record.bright_transform_applied)
        for record in second
    ]
    assert first_keys == second_keys

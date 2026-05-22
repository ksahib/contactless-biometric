from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import re
import shutil
from pathlib import Path
import sys
import sysconfig


def ensure_stdlib_copy_module() -> None:
    stdlib_copy = Path(sysconfig.get_paths()["stdlib"]) / "copy.py"
    spec = importlib.util.spec_from_file_location("copy", stdlib_copy)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not resolve stdlib copy module from {stdlib_copy}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = module
    spec.loader.exec_module(module)


ensure_stdlib_copy_module()

from dataclasses import dataclass

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "dataset"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "finger_solov2_dataset"
DEFAULT_SEED = 20260519

RAW_IMAGE_PATTERN = re.compile(
    r"^(?P<subject>\d+)_(?P<finger>\d+)_(?P<acquisition>\d+)_(?P<view>\d+)\.(?P<ext>[A-Za-z0-9]+)$"
)

CATEGORY_COUNTS = {
    "front": 50,
    "left_side": 35,
    "right_side": 35,
    "bright_overexposed": 15,
    "dark_low_contrast": 15,
}

SPLIT_COUNTS = {
    "train": {
        "front": 40,
        "left_side": 28,
        "right_side": 28,
        "bright_overexposed": 12,
        "dark_low_contrast": 12,
    },
    "val": {
        "front": 10,
        "left_side": 7,
        "right_side": 7,
        "bright_overexposed": 3,
        "dark_low_contrast": 3,
    },
}

VIEW_TO_CATEGORY = {
    0: "front",
    1: "left_side",
    2: "right_side",
    3: "dark_low_contrast",
    4: "dark_low_contrast",
    5: "dark_low_contrast",
}

BRIGHT_SOURCE_CATEGORIES = {"front", "left_side", "right_side"}


@dataclass(frozen=True)
class RawImageRecord:
    dataset_id: str
    subject_id: int
    finger_id: int
    acquisition_id: int
    view_index: int
    path: Path

    @property
    def category(self) -> str:
        return VIEW_TO_CATEGORY[self.view_index]

    @property
    def source_key(self) -> str:
        return f"{self.dataset_id}:{self.subject_id}:{self.finger_id}:{self.acquisition_id}:{self.view_index}"


@dataclass(frozen=True)
class SelectedImageRecord:
    source: RawImageRecord
    output_category: str
    split: str
    bright_transform_applied: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the finger_solov2_dataset split from DS1/DS2/DS3 raw captures."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def discover_raw_images(input_root: Path) -> list[RawImageRecord]:
    records: list[RawImageRecord] = []
    for dataset_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        raw_dirs = sorted(path for path in dataset_dir.glob("*/raw") if path.is_dir())
        for raw_dir in raw_dirs:
            for image_path in sorted(path for path in raw_dir.iterdir() if path.is_file()):
                match = RAW_IMAGE_PATTERN.match(image_path.name)
                if match is None:
                    continue
                view_index = int(match.group("view"))
                if view_index not in VIEW_TO_CATEGORY:
                    continue
                records.append(
                    RawImageRecord(
                        dataset_id=dataset_dir.name,
                        subject_id=int(match.group("subject")),
                        finger_id=int(match.group("finger")),
                        acquisition_id=int(match.group("acquisition")),
                        view_index=view_index,
                        path=image_path.resolve(),
                    )
                )
    return records


def sample_records(records: list[RawImageRecord], seed: int) -> list[SelectedImageRecord]:
    rng = random.Random(seed)
    used_source_keys: set[str] = set()
    selections_by_category: dict[str, list[RawImageRecord]] = {}

    for category in ("front", "left_side", "right_side", "dark_low_contrast"):
        pool = [record for record in records if record.category == category and record.source_key not in used_source_keys]
        count = CATEGORY_COUNTS[category]
        if len(pool) < count:
            raise ValueError(
                f"not enough images for category '{category}': required {count}, found {len(pool)}"
            )
        chosen = rng.sample(pool, count)
        selections_by_category[category] = chosen
        used_source_keys.update(record.source_key for record in chosen)

    bright_pool = [
        record
        for record in records
        if record.category in BRIGHT_SOURCE_CATEGORIES and record.source_key not in used_source_keys
    ]
    bright_count = CATEGORY_COUNTS["bright_overexposed"]
    if len(bright_pool) < bright_count:
        raise ValueError(
            f"not enough unused images for category 'bright_overexposed': required {bright_count}, found {len(bright_pool)}"
        )
    selections_by_category["bright_overexposed"] = rng.sample(bright_pool, bright_count)

    selected_records: list[SelectedImageRecord] = []
    for category, chosen in selections_by_category.items():
        shuffled = list(chosen)
        rng.shuffle(shuffled)
        train_count = SPLIT_COUNTS["train"][category]
        val_count = SPLIT_COUNTS["val"][category]
        if len(shuffled) != train_count + val_count:
            raise ValueError(f"split counts do not match sampled total for category '{category}'")
        for index, record in enumerate(shuffled):
            split = "train" if index < train_count else "val"
            selected_records.append(
                SelectedImageRecord(
                    source=record,
                    output_category=category,
                    split=split,
                    bright_transform_applied=(category == "bright_overexposed"),
                )
            )

    rng.shuffle(selected_records)
    return selected_records


def prepare_output_root(output_root: Path) -> None:
    output_root = output_root.resolve()
    repo_root = REPO_ROOT.resolve()
    if output_root == repo_root or output_root == repo_root.parent:
        raise ValueError(f"refusing to clear unsafe output root: {output_root}")
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "images" / "val").mkdir(parents=True, exist_ok=True)


def build_bright_variant(image_path: Path) -> cv2.typing.MatLike:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"failed to read image for bright transform: {image_path}")

    # Deterministic clipped gain plus offset to simulate overexposure.
    bright_image = cv2.convertScaleAbs(image, alpha=1.28, beta=42.0)
    return bright_image


def _write_image(selected: SelectedImageRecord, destination: Path) -> None:
    if selected.bright_transform_applied:
        bright_image = build_bright_variant(selected.source.path)
        if not cv2.imwrite(str(destination), bright_image):
            raise ValueError(f"failed to write transformed image: {destination}")
        return
    shutil.copy2(selected.source.path, destination)


def write_dataset(selected_records: list[SelectedImageRecord], output_root: Path) -> Path:
    manifest_path = output_root / "manifest.csv"
    fieldnames = [
        "output_filename",
        "split",
        "category",
        "dataset_id",
        "subject_id",
        "finger_id",
        "acquisition_id",
        "view_index",
        "source_path",
        "source_extension",
        "bright_transform_applied",
    ]

    counters = {"train": 1, "val": 1}
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for selected in selected_records:
            extension = selected.source.path.suffix.lower()
            output_filename = f"img_{counters[selected.split]:06d}{extension}"
            counters[selected.split] += 1
            destination = output_root / "images" / selected.split / output_filename
            _write_image(selected, destination)
            writer.writerow(
                {
                    "output_filename": output_filename,
                    "split": selected.split,
                    "category": selected.output_category,
                    "dataset_id": selected.source.dataset_id,
                    "subject_id": selected.source.subject_id,
                    "finger_id": selected.source.finger_id,
                    "acquisition_id": selected.source.acquisition_id,
                    "view_index": selected.source.view_index,
                    "source_path": str(selected.source.path),
                    "source_extension": extension,
                    "bright_transform_applied": str(selected.bright_transform_applied).lower(),
                }
            )
    return manifest_path


def summarize_selection(selected_records: list[SelectedImageRecord]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {
        "train": {},
        "val": {},
    }
    for split in summary:
        for category in CATEGORY_COUNTS:
            summary[split][category] = 0
    for selected in selected_records:
        summary[selected.split][selected.output_category] += 1
    return summary


def main() -> int:
    args = parse_args()
    records = discover_raw_images(args.input_root.resolve())
    if not records:
        raise ValueError(f"no raw images found under {args.input_root.resolve()}")

    selected_records = sample_records(records, seed=args.seed)
    prepare_output_root(args.output_root)
    manifest_path = write_dataset(selected_records, args.output_root)
    summary = summarize_selection(selected_records)

    print(f"Discovered raw images: {len(records)}")
    print(f"Wrote dataset root: {args.output_root.resolve()}")
    print(f"Wrote manifest: {manifest_path.resolve()}")
    print(f"Train images: {sum(summary['train'].values())}")
    print(f"Val images: {sum(summary['val'].values())}")
    for split in ("train", "val"):
        for category in CATEGORY_COUNTS:
            print(f"{split}.{category}: {summary[split][category]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

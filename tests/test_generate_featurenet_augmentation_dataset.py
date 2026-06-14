from __future__ import annotations

import importlib.util
import json
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


def _ensure_stdlib_copy_module() -> None:
    stdlib_copy = Path(sysconfig.get_paths()["stdlib"]) / "copy.py"
    spec = importlib.util.spec_from_file_location("copy", stdlib_copy)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not resolve stdlib copy module from {stdlib_copy}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = module
    spec.loader.exec_module(module)


_ensure_stdlib_copy_module()
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from featurenet.models.target_rasterization import build_featurenet_targets  # noqa: E402
from featurenet.models.train import load_bundle_samples  # noqa: E402


def _load_script_module():
    script_path = REPO_ROOT / "scripts" / "generate_featurenet_augmentation_dataset.py"
    spec = importlib.util.spec_from_file_location("generate_featurenet_augmentation_dataset", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_bundle(root: Path, sample_id: str = "sample_001") -> None:
    sample_dir = root / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    height = width = 16
    image = np.zeros((height, width), dtype=np.uint8)
    image[4:12, 4:12] = np.arange(64, dtype=np.uint8).reshape(8, 8) + 80
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[3:13, 3:13] = 255
    orientation = np.full((height, width), 0.25, dtype=np.float32)
    ridge = np.ones((height, width), dtype=np.float32) * 10.0
    gradient = np.zeros((height, width, 2), dtype=np.float32)
    minutiae = [{"x": 8.0, "y": 8.0, "theta": 0.25, "score": 1.0, "type": "ending"}]
    targets = build_featurenet_targets(image, mask, orientation, ridge, gradient, minutiae, output_shape=(2, 2))

    cv2.imwrite(str(sample_dir / "masked_image.png"), image)
    cv2.imwrite(str(sample_dir / "mask.png"), mask)
    np.save(sample_dir / "orientation.npy", orientation)
    np.save(sample_dir / "ridge_period.npy", ridge)
    np.savez_compressed(sample_dir / "featurenet_targets.npz", **targets)
    (sample_dir / "minutiae.json").write_text(json.dumps(minutiae), encoding="utf-8")

    yy, xx = np.indices((height, width), dtype=np.float32)
    reconstruction_dir = root / "reconstructions" / f"acq_{sample_id}"
    reconstruction_dir.mkdir(parents=True)
    reconstruction_maps_path = reconstruction_dir / "reconstruction_maps.npz"
    np.savez_compressed(
        reconstruction_maps_path,
        support_mask=(mask > 0).astype(np.uint8),
        front_pose_x_map=xx,
        front_pose_y_map=yy,
        depth_front=np.zeros((height, width), dtype=np.float32),
    )
    meta = {
        "sample_id": sample_id,
        "subject_id": 1,
        "subject_index": 0,
        "finger_id": 1,
        "acquisition_id": 1,
        "finger_class_id": 0,
        "raw_view_index": 0,
        "counts": {"minutiae": 1},
        "shapes": {"input_image": [height, width], "featurenet_output": [2, 2]},
        "multiview_reconstruction": {
            "role": "front",
            "reconstruction_dir": str(reconstruction_dir),
            "reconstruction_maps_path": str(reconstruction_maps_path),
        },
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    manifest = [
        {
            "sample_id": sample_id,
            "subject_id": 1,
            "subject_index": 0,
            "finger_id": 1,
            "acquisition_id": 1,
            "finger_class_id": 0,
            "raw_view_index": 0,
        }
    ]
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps({"generated_bundle_count": 1}), encoding="utf-8")


class GenerateFeatureNetAugmentationDatasetTests(unittest.TestCase):
    def test_compact_generated_root_loads_without_reconstructions(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            output_root = root / "logical_augmented"
            storage_root = root / "storage"
            _write_bundle(source_root)

            args = module.parse_args(
                [
                    "--ground-truth-root",
                    str(source_root),
                    "--output-root",
                    str(output_root),
                    "--storage-root",
                    str(storage_root),
                    "--no-auto-discover-media-roots",
                    "--augmentation-count",
                    "2",
                    "--translation-jitter-px",
                    "0",
                    "0",
                    "--yaw-jitter-deg",
                    "0",
                    "--pitch-roll-jitter-deg",
                    "0",
                    "0",
                    "--workers",
                    "1",
                    "--min-free-gb",
                    "0",
                    "--overwrite",
                    "--manifest-flush-every",
                    "1",
                ]
            )
            summary = module.generate_dataset(args)

            self.assertEqual(summary["generated_synthetic_count"], 2)
            self.assertFalse((output_root / "reconstructions").exists())
            samples = load_bundle_samples(output_root, strict_gradient_targets=True)
            self.assertEqual(len(samples), 3)

            original_dir = output_root / "samples" / "sample_001"
            synthetic_dir = output_root / "samples" / "sample_001_aug01"
            self.assertTrue((original_dir / "masked_image.png").is_symlink())
            self.assertFalse((original_dir / "meta.json").is_symlink())
            self.assertTrue((synthetic_dir / "masked_image.png").is_symlink())
            self.assertFalse((synthetic_dir / "meta.json").is_symlink())

            synthetic_meta = json.loads((synthetic_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(synthetic_meta["sample_id"], "sample_001_aug01")
            self.assertEqual(synthetic_meta["parent_sample_id"], "sample_001")
            self.assertIn("pregenerated_augmentation", synthetic_meta)
            self.assertNotIn("multiview_reconstruction", synthetic_meta)
            with np.load(synthetic_dir / "featurenet_targets.npz") as targets:
                self.assertIn("gradient", targets.files)
                self.assertIn("minutia_valid_mask", targets.files)

            storage_manifest = json.loads((output_root / "storage_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(storage_manifest["records"]), 3)

    def test_resume_skips_completed_samples(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            output_root = root / "logical_augmented"
            storage_root = root / "storage"
            _write_bundle(source_root)
            base_args = [
                "--ground-truth-root",
                str(source_root),
                "--output-root",
                str(output_root),
                "--storage-root",
                str(storage_root),
                "--no-auto-discover-media-roots",
                "--augmentation-count",
                "1",
                "--translation-jitter-px",
                "0",
                "0",
                "--yaw-jitter-deg",
                "0",
                "--pitch-roll-jitter-deg",
                "0",
                "0",
                "--workers",
                "1",
                "--min-free-gb",
                "0",
            ]
            module.generate_dataset(module.parse_args([*base_args, "--overwrite"]))
            resumed = module.generate_dataset(module.parse_args([*base_args, "--resume"]))

            self.assertEqual(resumed["generated_synthetic_count"], 0)
            self.assertEqual(resumed["skipped_existing_count"], 2)

    def test_storage_root_selection_skips_nearly_full_root(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            first.mkdir()
            second.mkdir()

            def fake_usage(path: Path) -> SimpleNamespace:
                return SimpleNamespace(free=0 if path == first else 100)

            selected = module._choose_storage_root([first, second], 50, disk_usage_fn=fake_usage)
            self.assertEqual(selected, second)


if __name__ == "__main__":
    unittest.main()

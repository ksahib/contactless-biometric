from __future__ import annotations

import importlib.util
import json
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path

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
sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2

from featurenet.models.train import FeatureNetDataset, load_bundle_samples


class FeatureNetLazyLoadingTests(unittest.TestCase):
    def _write_bundle(self, root: Path) -> Path:
        sample_dir = root / "samples" / "sample_001"
        sample_dir.mkdir(parents=True, exist_ok=True)

        image = np.full((8, 8), 127, dtype=np.uint8)
        mask = np.full((8, 8), 255, dtype=np.uint8)
        cv2.imwrite(str(sample_dir / "masked_image.png"), image)
        cv2.imwrite(str(sample_dir / "mask.png"), mask)

        np.savez(
            sample_dir / "featurenet_targets.npz",
            orientation=np.zeros((180, 1, 1), dtype=np.float32),
            ridge_period=np.ones((1, 1, 1), dtype=np.float32),
            gradient=np.zeros((2, 1, 1), dtype=np.float32),
            minutia_score=np.zeros((1, 1, 1), dtype=np.float32),
            minutia_valid_mask=np.zeros((1, 1, 1), dtype=np.float32),
            minutia_x_offset=np.zeros((1, 1, 1), dtype=np.float32),
            minutia_y_offset=np.zeros((1, 1, 1), dtype=np.float32),
            minutia_orientation_vec=np.zeros((2, 1, 1), dtype=np.float32),
            minutia_x=np.zeros((1, 1), dtype=np.int64),
            minutia_y=np.zeros((1, 1), dtype=np.int64),
            minutia_orientation=np.zeros((1, 1), dtype=np.int64),
        )

        meta = {
            "sample_id": "sample_001",
            "subject_id": 1,
            "subject_index": 0,
            "finger_id": 1,
            "acquisition_id": 1,
            "finger_class_id": 0,
            "raw_view_index": 0,
            "counts": {"minutiae": 0},
            "shapes": {"input_image": [8, 8], "featurenet_output": [1, 1]},
        }
        (sample_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return sample_dir

    def test_load_bundle_samples_keeps_targets_lazy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_dir = self._write_bundle(root)

            samples = load_bundle_samples(root, strict_gradient_targets=True)

            self.assertEqual(len(samples), 1)
            self.assertNotIn("targets", samples[0])
            self.assertEqual(samples[0]["targets_path"], sample_dir / "featurenet_targets.npz")
            self.assertEqual(samples[0]["raw_view_index"], 0)
            self.assertEqual(samples[0]["input_shape_hw"], (8, 8))
            self.assertEqual(samples[0]["output_shape_hw"], (1, 1))

    def test_dataset_getitem_lazy_loads_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_bundle(root)
            samples = load_bundle_samples(root, strict_gradient_targets=True)

            input_tensor, targets = FeatureNetDataset(samples)[0]

            self.assertEqual(tuple(input_tensor.shape), (2, 8, 8))
            self.assertEqual(tuple(targets["orientation"].shape), (180, 1, 1))
            self.assertEqual(tuple(targets["gradient"].shape), (2, 1, 1))
            self.assertEqual(int(targets["raw_view_index"].item()), 0)
            self.assertEqual(tuple(targets["input_shape_hw"].tolist()), (8, 8))
            self.assertEqual(tuple(targets["output_shape_hw"].tolist()), (1, 1))

    def test_dataset_getitem_preserves_in_memory_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_dir = root / "sample"
            sample_dir.mkdir()
            image = np.full((8, 8), 127, dtype=np.uint8)
            mask = np.full((8, 8), 255, dtype=np.uint8)
            cv2.imwrite(str(sample_dir / "masked_image.png"), image)
            cv2.imwrite(str(sample_dir / "mask.png"), mask)

            samples = [
                {
                    "sample_id": "memory_sample",
                    "masked_image": sample_dir / "masked_image.png",
                    "mask": sample_dir / "mask.png",
                    "targets": {"gradient": np.zeros((2, 1, 1), dtype=np.float32)},
                    "raw_view_index": 2,
                    "input_shape_hw": (8, 8),
                    "output_shape_hw": (1, 1),
                }
            ]

            _, targets = FeatureNetDataset(samples)[0]

            self.assertEqual(tuple(targets["gradient"].shape), (2, 1, 1))
            self.assertEqual(int(targets["raw_view_index"].item()), 2)


if __name__ == "__main__":
    unittest.main()

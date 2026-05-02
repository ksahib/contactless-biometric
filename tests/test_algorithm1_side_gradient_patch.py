from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "patch_algorithm1_side_gradients.py"
SPEC = importlib.util.spec_from_file_location("side_gradient_patch", MODULE_PATH)
side_gradient_patch = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = side_gradient_patch
SPEC.loader.exec_module(side_gradient_patch)


class Algorithm1SideGradientPatchTests(unittest.TestCase):
    def test_depth_gradient_label_transposes_hw2_to_2hw(self):
        gradient = np.zeros((3, 4, 2), dtype=np.float32)
        gradient[:, :, 0] = 7.0
        gradient[:, :, 1] = 11.0

        label = side_gradient_patch._to_depth_gradient_label(gradient)

        self.assertEqual(label.shape, (2, 3, 4))
        self.assertTrue(np.all(label[0] == 7.0))
        self.assertTrue(np.all(label[1] == 11.0))

    def test_resize_gradient_uses_output_mask_and_preserves_channels(self):
        gradient = np.zeros((4, 4, 2), dtype=np.float32)
        gradient[:, :, 0] = 2.0
        gradient[:, :, 1] = 5.0
        output_mask = np.ones((1, 2, 2), dtype=np.float32)
        output_mask[0, 0, 1] = 0.0

        resized = side_gradient_patch._resize_gradient_to_target(gradient, output_mask)

        self.assertEqual(resized.shape, (2, 2, 2))
        self.assertAlmostEqual(float(resized[0, 0, 0]), 2.0)
        self.assertAlmostEqual(float(resized[1, 0, 0]), 5.0)
        self.assertEqual(float(resized[:, 0, 1].sum()), 0.0)

    def test_manual_shard_selects_whole_acquisition_units(self):
        units = [
            side_gradient_patch.AcquisitionWorkUnit(
                acquisition_id=f"a{idx}",
                reconstruction_dir=Path(f"recon/{idx}"),
                cache_dir=Path(f"cache/{idx}"),
            )
            for idx in range(5)
        ]
        args = argparse.Namespace(shard_mode="manual", shard_count=2, shard_index=1, target_shard_size=10)

        selected, info = side_gradient_patch._select_sharded_units(units, args)

        self.assertEqual([unit.acquisition_id for unit in selected], ["a3", "a4"])
        self.assertEqual(info["selected_start"], 3)
        self.assertEqual(info["selected_end"], 5)


if __name__ == "__main__":
    unittest.main()

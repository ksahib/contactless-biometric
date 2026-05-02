from __future__ import annotations

import importlib.util
import argparse
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "patch_algorithm1_depth_unwrap_side_ground_truth.py"
SPEC = importlib.util.spec_from_file_location("side_depth_patch", MODULE_PATH)
side_depth_patch = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = side_depth_patch
SPEC.loader.exec_module(side_depth_patch)


class Algorithm1DepthUnwrapSidePatchTests(unittest.TestCase):
    def test_pose_to_training_scaling_preserves_target_frame(self):
        pose_minutiae = [{"x": 2.0, "y": 3.0, "theta": 0.0, "score": 1.0, "type": "E", "unwrap_x": 2.0, "unwrap_y": 3.0}]
        h, w = 8, 8
        source_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        source_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
        unwrap_mask = np.ones((h, w), dtype=bool)
        final_mask = np.ones((16, 24), dtype=np.uint8) * 255

        scaled, details = side_depth_patch._scale_pose_minutiae_to_training(
            pose_minutiae,
            source_x,
            source_y,
            unwrap_mask,
            final_mask,
            scale_x=3.0,
            scale_y=2.0,
            orientation_delta_px=2.0,
        )

        self.assertEqual(len(scaled), 1)
        self.assertAlmostEqual(scaled[0]["x"], 6.0)
        self.assertAlmostEqual(scaled[0]["y"], 6.0)
        self.assertEqual(details["training_minutiae_count"], 1)
        self.assertEqual(details["orientation_projected_count"], 1)

    def test_orientation_baseline_accounts_for_nonuniform_training_scale(self):
        pose_minutiae = [{"x": 5.0, "y": 5.0, "theta": np.pi / 4.0, "score": 1.0, "type": "E", "unwrap_x": 5.0, "unwrap_y": 5.0}]
        h, w = 12, 12
        source_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        source_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
        unwrap_mask = np.ones((h, w), dtype=bool)
        final_mask = np.ones((30, 30), dtype=np.uint8) * 255

        scaled, details = side_depth_patch._scale_pose_minutiae_to_training(
            pose_minutiae,
            source_x,
            source_y,
            unwrap_mask,
            final_mask,
            scale_x=2.0,
            scale_y=1.0,
            orientation_delta_px=2.0,
        )

        self.assertEqual(len(scaled), 1)
        self.assertEqual(details["orientation_projected_count"], 1)
        self.assertAlmostEqual(scaled[0]["theta"], np.arctan2(1.0, 2.0), places=5)

    def test_manual_shard_selects_whole_acquisition_units(self):
        units = [
            side_depth_patch.AcquisitionWorkUnit(
                acquisition_id=f"a{idx}",
                reconstruction_dir=Path(f"recon/{idx}"),
                cache_dir=Path(f"cache/{idx}"),
            )
            for idx in range(5)
        ]
        args = argparse.Namespace(shard_mode="manual", shard_count=2, shard_index=1, target_shard_size=10)

        selected, info = side_depth_patch._select_sharded_units(units, args)

        self.assertEqual([unit.acquisition_id for unit in selected], ["a3", "a4"])
        self.assertEqual(info["selected_start"], 3)
        self.assertEqual(info["selected_end"], 5)

    def test_auto_shard_keeps_first_chunk_deterministic(self):
        units = [
            side_depth_patch.AcquisitionWorkUnit(
                acquisition_id=f"a{idx}",
                reconstruction_dir=Path(f"recon/{idx}"),
                cache_dir=Path(f"cache/{idx}"),
            )
            for idx in range(5)
        ]
        args = argparse.Namespace(shard_mode="auto", shard_count=1, shard_index=0, target_shard_size=2)

        selected, info = side_depth_patch._select_sharded_units(units, args)

        self.assertEqual([unit.acquisition_id for unit in selected], ["a0", "a1"])
        self.assertEqual(info["shard_count"], 3)


if __name__ == "__main__":
    unittest.main()

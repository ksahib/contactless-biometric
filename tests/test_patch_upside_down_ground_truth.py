from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "patch_upside_down_ground_truth.py"
SPEC = importlib.util.spec_from_file_location("patch_upside_down_ground_truth", MODULE_PATH)
patch = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = patch
SPEC.loader.exec_module(patch)


def _tapered_mask(*, tip_at_top: bool) -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.uint8)
    points = np.array([[26, 6], [38, 6], [52, 58], [12, 58]], dtype=np.int32)
    cv2.fillConvexPoly(mask, points, 255)
    if not tip_at_top:
        mask = np.ascontiguousarray(mask[::-1, ::-1])
    return mask


class PatchUpsideDownGroundTruthTests(unittest.TestCase):
    def test_width_stats_detects_upside_down_tapered_mask(self) -> None:
        stats = patch.upside_down_width_stats(_tapered_mask(tip_at_top=False))

        self.assertTrue(stats.should_flip)
        self.assertGreater(stats.top_width, stats.bottom_width)

    def test_width_stats_keeps_tip_up_tapered_mask(self) -> None:
        stats = patch.upside_down_width_stats(_tapered_mask(tip_at_top=True))

        self.assertFalse(stats.should_flip)
        self.assertLess(stats.top_width, stats.bottom_width)

    def test_flip_vector_spatially_flips_and_negates_channels(self) -> None:
        gradient = np.zeros((2, 2, 3), dtype=np.float32)
        gradient[0] = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        gradient[1] = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)

        flipped = patch._flip_vector_180(gradient)

        np.testing.assert_array_equal(flipped[0], np.array([[-6, -5, -4], [-3, -2, -1]], dtype=np.float32))
        np.testing.assert_array_equal(flipped[1], np.array([[-60, -50, -40], [-30, -20, -10]], dtype=np.float32))

    def test_transform_minutiae_records_flips_xy_and_rotates_theta(self) -> None:
        transformed = patch.transform_minutiae_records(
            [{"x": 2.0, "y": 3.0, "theta": 0.25, "score": 0.9}],
            width=10,
            height=12,
        )

        self.assertAlmostEqual(transformed[0]["x"], 7.0)
        self.assertAlmostEqual(transformed[0]["y"], 8.0)
        self.assertAlmostEqual(transformed[0]["theta"], 0.25 + math.pi)
        self.assertAlmostEqual(transformed[0]["score"], 0.9)

    def test_transform_featurenet_targets_flips_maps_and_rebuilds_minutia_labels(self) -> None:
        arrays = {
            "output_mask": np.ones((1, 4, 4), dtype=np.float32),
            "ridge_period": np.arange(16, dtype=np.float32).reshape(1, 4, 4),
            "gradient": np.stack(
                [
                    np.arange(16, dtype=np.float32).reshape(4, 4),
                    np.arange(100, 116, dtype=np.float32).reshape(4, 4),
                ],
                axis=0,
            ),
            "minutia_score": np.zeros((1, 4, 4), dtype=np.float32),
            "minutia_score_weight_map": np.ones((1, 4, 4), dtype=np.float32),
            "minutia_score_ignore_mask": np.zeros((1, 4, 4), dtype=np.float32),
            "minutia_score_center_map": np.zeros((1, 4, 4), dtype=np.float32),
            "minutia_valid_mask": np.zeros((1, 4, 4), dtype=np.float32),
            "minutia_x": np.zeros((4, 4), dtype=np.int64),
            "minutia_y": np.zeros((4, 4), dtype=np.int64),
            "minutia_x_offset": np.zeros((1, 4, 4), dtype=np.float32),
            "minutia_y_offset": np.zeros((1, 4, 4), dtype=np.float32),
            "minutia_orientation": np.zeros((4, 4), dtype=np.int64),
            "minutia_orientation_vec": np.zeros((2, 4, 4), dtype=np.float32),
        }
        transformed_minutiae = [{"x": 13.0, "y": 11.0, "theta": math.pi, "score": 1.0}]

        transformed = patch.transform_featurenet_targets(
            arrays,
            minutiae=transformed_minutiae,
            source_shape=(16, 16),
        )

        self.assertAlmostEqual(float(transformed["ridge_period"][0, 0, 0]), 15.0)
        self.assertAlmostEqual(float(transformed["gradient"][0, 0, 0]), -15.0)
        self.assertAlmostEqual(float(transformed["gradient"][1, 0, 0]), -115.0)
        self.assertEqual(float(transformed["minutia_valid_mask"][0, 2, 3]), 1.0)
        self.assertEqual(int(transformed["minutia_x"][2, 3]), 2)
        self.assertEqual(int(transformed["minutia_y"][2, 3]), 6)
        self.assertAlmostEqual(float(transformed["minutia_x_offset"][0, 2, 3]), 0.25)
        self.assertAlmostEqual(float(transformed["minutia_y_offset"][0, 2, 3]), 0.75)
        self.assertEqual(int(transformed["minutia_orientation"][2, 3]), 180)
        self.assertAlmostEqual(float(transformed["minutia_orientation_vec"][0, 2, 3]), -1.0, places=6)
        self.assertAlmostEqual(float(transformed["minutia_orientation_vec"][1, 2, 3]), 0.0, places=6)

    def test_repair_sample_flips_bundle_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sample_dir = Path(directory) / "sample_001"
            sample_dir.mkdir()
            mask = _tapered_mask(tip_at_top=False)
            image = np.where(mask > 0, 120, 0).astype(np.uint8)
            for name in (
                "masked_image.png",
                "mask.png",
                "preprocessed_input.png",
                "preprocess_mask.png",
                "preprocess_pose_normalized.png",
                "preprocess_pose_mask.png",
            ):
                cv2.imwrite(str(sample_dir / name), mask if "mask" in name else image)
            cv2.imwrite(str(sample_dir / "raw_input.png"), np.full_like(mask, 77))
            cv2.imwrite(str(sample_dir / "preprocess_normalized.png"), np.full_like(mask, 88))
            np.save(sample_dir / "orientation.npy", np.arange(16, dtype=np.float32).reshape(4, 4))
            np.save(sample_dir / "ridge_period.npy", np.arange(16, dtype=np.float32).reshape(4, 4))
            np.save(sample_dir / "gradient_visualization.npy", np.dstack([np.ones((4, 4)), np.ones((4, 4)) * 2]).astype(np.float32))
            (sample_dir / "minutiae.json").write_text(
                json.dumps([{"x": 1.0, "y": 2.0, "theta": 0.0, "score": 1.0}]),
                encoding="utf-8",
            )
            (sample_dir / "meta.json").write_text(
                json.dumps({"sample_id": "sample_001", "raw_view_index": 0, "shapes": {"input_image": [64, 64]}}),
                encoding="utf-8",
            )
            np.savez_compressed(
                sample_dir / "featurenet_targets.npz",
                output_mask=np.ones((1, 8, 8), dtype=np.float32),
                ridge_period=np.ones((1, 8, 8), dtype=np.float32),
                gradient=np.ones((2, 8, 8), dtype=np.float32),
                minutia_score=np.zeros((1, 8, 8), dtype=np.float32),
                minutia_score_weight_map=np.ones((1, 8, 8), dtype=np.float32),
                minutia_score_ignore_mask=np.zeros((1, 8, 8), dtype=np.float32),
                minutia_score_center_map=np.zeros((1, 8, 8), dtype=np.float32),
                minutia_valid_mask=np.zeros((1, 8, 8), dtype=np.float32),
                minutia_x=np.zeros((8, 8), dtype=np.int64),
                minutia_y=np.zeros((8, 8), dtype=np.int64),
                minutia_x_offset=np.zeros((1, 8, 8), dtype=np.float32),
                minutia_y_offset=np.zeros((1, 8, 8), dtype=np.float32),
                minutia_orientation=np.zeros((8, 8), dtype=np.int64),
                minutia_orientation_vec=np.zeros((2, 8, 8), dtype=np.float32),
            )

            result = patch.repair_sample(sample_dir, apply=True)

            self.assertEqual(result["status"], "repaired")
            self.assertFalse(patch.upside_down_width_stats(cv2.imread(str(sample_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)).should_flip)
            self.assertEqual(int(cv2.imread(str(sample_dir / "raw_input.png"), cv2.IMREAD_GRAYSCALE)[0, 0]), 77)
            meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertIn("upside_down_orientation_repair", meta["patches"])
            minutiae = json.loads((sample_dir / "minutiae.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(minutiae[0]["x"], 62.0)
            self.assertAlmostEqual(minutiae[0]["y"], 61.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_multiview_side_texture_fusion.py"
SPEC = importlib.util.spec_from_file_location("side_fusion", MODULE_PATH)
side_fusion = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(side_fusion)


class MultiviewSideFusionReprojectionTests(unittest.TestCase):
    def _role(self, image: np.ndarray, source_x: np.ndarray | None = None) -> dict[str, np.ndarray]:
        image = image.astype(np.uint8)
        h, w = image.shape
        if source_x is None:
            source_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        source_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
        return {
            "image": image,
            "mask": np.ones((h, w), dtype=bool),
            "quality": np.ones((h, w), dtype=bool),
            "source_step": np.ones((h, w), dtype=np.float32),
            "source_x": source_x.astype(np.float32),
            "source_y": source_y.astype(np.float32),
        }

    def test_donor_source_maps_flip_with_donor_image(self):
        moving = self._role(np.array([[1, 2, 3]], dtype=np.uint8), source_x=np.array([[10, 20, 30]], dtype=np.float32))

        aligned = side_fusion._align_to_reference(moving, (1, 3), flip=True)

        np.testing.assert_array_equal(aligned["image"], np.array([[3, 2, 1]], dtype=np.uint8))
        np.testing.assert_array_equal(aligned["source_x"], np.array([[30, 20, 10]], dtype=np.float32))

    def test_fused_source_map_uses_selected_donor_pixel(self):
        reference = self._role(
            np.array([[50, 50, 50]], dtype=np.uint8),
            source_x=np.array([[1, 2, 3]], dtype=np.float32),
        )
        donor = self._role(
            np.array([[200, 200, 200]], dtype=np.uint8),
            source_x=np.array([[101, 102, 103]], dtype=np.float32),
        )
        reference["source_step"][:] = 0.2
        donor["source_step"][:] = 1.0

        fused = side_fusion._fuse_to_reference(reference, donor, False, "left", "right")

        np.testing.assert_array_equal(fused["selected_source_role"], np.array([[2, 2, 2]], dtype=np.uint8))
        np.testing.assert_array_equal(fused["source_x"], donor["source_x"])

    def test_shared_fusion_uses_donor_texture_but_target_geometry(self):
        target = {
            "image": np.array([[40, 40, 40]], dtype=np.uint8),
            "support": np.ones((1, 3), dtype=bool),
            "quality": np.ones((1, 3), dtype=bool),
            "source_step": np.full((1, 3), 0.4, dtype=np.float32),
            "source_x": np.array([[1, 2, 3]], dtype=np.float32),
            "source_y": np.array([[7, 7, 7]], dtype=np.float32),
        }
        donor = {
            "image": np.array([[200, 200, 200]], dtype=np.uint8),
            "support": np.ones((1, 3), dtype=bool),
            "quality": np.ones((1, 3), dtype=bool),
            "source_step": np.full((1, 3), 1.0, dtype=np.float32),
            "source_x": np.array([[101, 102, 103]], dtype=np.float32),
            "source_y": np.array([[17, 17, 17]], dtype=np.float32),
        }

        fused = side_fusion._fuse_shared_for_target(target, donor, "right", "left")

        np.testing.assert_array_equal(fused["image"], donor["image"])
        np.testing.assert_array_equal(fused["selected_source_role"], np.array([[2, 2, 2]], dtype=np.uint8))
        np.testing.assert_array_equal(fused["reference_source_x"], target["source_x"])
        np.testing.assert_array_equal(fused["reference_source_y"], target["source_y"])

    def test_quality_mask_rejects_minutia_before_source_mapping(self):
        minutiae = [{"x": 1.0, "y": 1.0, "theta": 0.0, "score": 1.0}]
        source_x = np.ones((3, 3), dtype=np.float32)
        source_y = np.ones((3, 3), dtype=np.float32)
        support = np.ones((3, 3), dtype=bool)
        quality = np.zeros((3, 3), dtype=bool)
        pose_mask = np.ones((4, 4), dtype=bool)

        reprojected, details = side_fusion._reproject_fused_minutiae_to_pose(
            minutiae,
            source_x,
            source_y,
            support,
            quality,
            np.ones((3, 3), dtype=np.float32),
            pose_mask,
        )

        self.assertEqual(reprojected, [])
        self.assertEqual(details["dropped_outside_quality_mask"], 1)

    def test_orientation_reprojects_from_forward_backward_baseline(self):
        minutiae = [{"x": 5.0, "y": 5.0, "theta": 0.0, "score": 1.0}]
        h, w = 11, 11
        x_grid = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        y_grid = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
        source_x = y_grid.copy()
        source_y = x_grid.copy()
        support = np.ones((h, w), dtype=bool)
        quality = np.ones((h, w), dtype=bool)
        pose_mask = np.ones((20, 20), dtype=bool)

        reprojected, details = side_fusion._reproject_fused_minutiae_to_pose(
            minutiae,
            source_x,
            source_y,
            support,
            quality,
            np.ones((h, w), dtype=np.float32),
            pose_mask,
        )

        self.assertEqual(len(reprojected), 1)
        self.assertEqual(details["orientation_projected_count"], 1)
        self.assertAlmostEqual(reprojected[0]["theta"], np.pi / 2.0, places=5)


if __name__ == "__main__":
    unittest.main()

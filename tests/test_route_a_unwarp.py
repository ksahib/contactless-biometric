from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from featurenet.models.infer import unwarp_minutiae_rows


def _zero_gradient(h8: int, w8: int) -> torch.Tensor:
    return torch.zeros((1, 2, h8, w8), dtype=torch.float32)


def _full_mask(height: int, width: int) -> torch.Tensor:
    return torch.ones((1, 1, height, width), dtype=torch.float32)


class RouteAUnwarpTest(unittest.TestCase):
    def test_zero_gradient_is_identity_up_to_translation(self) -> None:
        height = width = 64
        gradient = _zero_gradient(height // 8, width // 8)
        mask = _full_mask(height, width)
        rows = [
            {"x": 20.0, "y": 24.0, "angle": 0.3, "score": 0.9},
            {"x": 40.0, "y": 44.0, "angle": -1.1, "score": 0.8},
        ]

        warped, unwarped_mask = unwarp_minutiae_rows(
            rows=rows,
            gradient_tensor=gradient,
            mask_tensor=mask,
            input_shape_hw=(height, width),
        )

        self.assertEqual(len(warped), len(rows))
        self.assertIsInstance(unwarped_mask, np.ndarray)

        # With zero gradient the unwarp is a pure crop translation, so relative
        # geometry (point-to-point deltas) and angles are preserved.
        dx_in = rows[0]["x"] - rows[1]["x"]
        dy_in = rows[0]["y"] - rows[1]["y"]
        dx_out = warped[0]["x"] - warped[1]["x"]
        dy_out = warped[0]["y"] - warped[1]["y"]
        self.assertAlmostEqual(dx_in, dx_out, delta=1.5)
        self.assertAlmostEqual(dy_in, dy_out, delta=1.5)
        for original, moved in zip(rows, warped):
            self.assertAlmostEqual(original["angle"], moved["angle"], delta=0.15)
            self.assertAlmostEqual(original["score"], moved["score"], delta=1e-6)

    def test_constant_tilt_expands_coordinates_monotonically(self) -> None:
        height = width = 64
        mask = _full_mask(height, width)
        rows = [
            {"x": 16.0, "y": 32.0, "angle": 0.0, "score": 0.9},
            {"x": 48.0, "y": 32.0, "angle": 0.0, "score": 0.9},
        ]

        flat_warped, _ = unwarp_minutiae_rows(
            rows=rows,
            gradient_tensor=_zero_gradient(height // 8, width // 8),
            mask_tensor=mask,
            input_shape_hw=(height, width),
        )
        tilt_gradient = torch.zeros((1, 2, height // 8, width // 8), dtype=torch.float32)
        tilt_gradient[0, 0] = 1.0  # dz/dx = 1 -> foreshortening along x
        tilt_warped, _ = unwarp_minutiae_rows(
            rows=rows,
            gradient_tensor=tilt_gradient,
            mask_tensor=mask,
            input_shape_hw=(height, width),
        )

        self.assertEqual(len(flat_warped), 2)
        self.assertEqual(len(tilt_warped), 2)
        flat_span = abs(flat_warped[0]["x"] - flat_warped[1]["x"])
        tilt_span = abs(tilt_warped[0]["x"] - tilt_warped[1]["x"])
        # sqrt(1 + 1^2) ~= 1.41 arc-length stretch along the tilted axis.
        self.assertGreater(tilt_span, flat_span * 1.2)

    def test_minutiae_outside_valid_region_are_dropped(self) -> None:
        height = width = 64
        mask = torch.zeros((1, 1, height, width), dtype=torch.float32)
        mask[0, 0, 16:48, 16:48] = 1.0
        rows = [
            {"x": 32.0, "y": 32.0, "angle": 0.2, "score": 0.9},  # inside
            {"x": 4.0, "y": 4.0, "angle": 0.2, "score": 0.9},  # outside mask
        ]

        warped, _ = unwarp_minutiae_rows(
            rows=rows,
            gradient_tensor=_zero_gradient(height // 8, width // 8),
            mask_tensor=mask,
            input_shape_hw=(height, width),
        )
        self.assertEqual(len(warped), 1)

    def test_empty_mask_returns_rows_unchanged(self) -> None:
        height = width = 32
        mask = torch.zeros((1, 1, height, width), dtype=torch.float32)
        rows = [{"x": 10.0, "y": 10.0, "angle": 0.0, "score": 0.5}]
        warped, returned_mask = unwarp_minutiae_rows(
            rows=rows,
            gradient_tensor=_zero_gradient(height // 8, width // 8),
            mask_tensor=mask,
            input_shape_hw=(height, width),
        )
        self.assertIs(warped, rows)
        self.assertEqual(returned_mask.shape, (height, width))


if __name__ == "__main__":
    unittest.main()

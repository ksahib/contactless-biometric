from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

from featurenet.models.evaluate import compute_validation_metrics


class _SinglePointModel(nn.Module):
    def forward(self, image: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch_size, _, height, width = image.shape
        score = torch.full((batch_size, 1, height, width), -20.0, device=image.device)
        score[:, :, 4, 4] = 20.0

        x_offset = torch.full((batch_size, 1, height, width), 0.0, device=image.device)
        y_offset = torch.full((batch_size, 1, height, width), 0.0, device=image.device)
        orientation = torch.zeros((batch_size, 2, height, width), device=image.device)
        orientation[:, 0] = 1.0

        return {
            "minutia_score": score,
            "minutia_x": x_offset,
            "minutia_y": y_offset,
            "minutia_orientation": orientation,
            "orientation": torch.zeros((batch_size, 180, height, width), device=image.device),
            "ridge_period": torch.zeros((batch_size, 1, height, width), device=image.device),
            "gradient": torch.zeros((batch_size, 2, height, width), device=image.device),
        }


def _batch(*, include_center_map: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    height = width = 8
    inputs = torch.ones((1, 2, height, width), dtype=torch.float32)

    gaussian_score = torch.zeros((1, 1, height, width), dtype=torch.float32)
    gaussian_score[:, :, 3:6, 3:6] = torch.tensor(
        [
            [0.20, 0.35, 0.20],
            [0.35, 1.00, 0.35],
            [0.20, 0.35, 0.20],
        ],
        dtype=torch.float32,
    )
    center_map = torch.zeros_like(gaussian_score)
    center_map[:, :, 4, 4] = 1.0

    x_offset = torch.full((1, 1, height, width), 0.5, dtype=torch.float32)
    y_offset = torch.full((1, 1, height, width), 0.5, dtype=torch.float32)
    orientation_vec = torch.zeros((1, 2, height, width), dtype=torch.float32)
    orientation_vec[:, 0] = 1.0

    targets = {
        "mask": torch.ones((1, 1, height, width), dtype=torch.float32),
        "minutia_score": gaussian_score,
        "minutia_x_offset": x_offset,
        "minutia_y_offset": y_offset,
        "minutia_orientation_vec": orientation_vec,
        "minutia_valid_mask": center_map.clone(),
        "raw_view_index": torch.tensor([0], dtype=torch.long),
        "input_shape_hw": torch.tensor([[64, 64]], dtype=torch.long),
        "output_shape_hw": torch.tensor([[8, 8]], dtype=torch.long),
    }
    if include_center_map:
        targets["minutia_score_center_map"] = center_map

    return inputs, targets


class FeatureNetEvaluateCenterMapTest(unittest.TestCase):
    def test_center_map_counts_one_gt_point_for_gaussian_score_target(self) -> None:
        metrics = compute_validation_metrics(
            model=_SinglePointModel(),
            dataloader=[_batch(include_center_map=True)],
            device=torch.device("cpu"),
            score_thresholds=[0.5],
            target_threshold=0.5,
        )

        stats = metrics["score_thresholds"]["0.50"]["all"]
        self.assertEqual(stats["tp"], 1)
        self.assertEqual(stats["fp"], 0)
        self.assertEqual(stats["fn"], 0)
        self.assertEqual(metrics["best_score_f1"], 1.0)

    def test_falls_back_to_score_map_when_center_map_is_missing(self) -> None:
        metrics = compute_validation_metrics(
            model=_SinglePointModel(),
            dataloader=[_batch(include_center_map=False)],
            device=torch.device("cpu"),
            score_thresholds=[0.5],
            target_threshold=0.0,
        )

        stats = metrics["score_thresholds"]["0.50"]["all"]
        self.assertEqual(stats["tp"], 1)
        self.assertEqual(stats["fp"], 0)
        self.assertEqual(stats["fn"], 8)


if __name__ == "__main__":
    unittest.main()

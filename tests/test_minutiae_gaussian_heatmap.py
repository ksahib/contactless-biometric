from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.minutiae_gaussian_heatmap import rasterize_consensus_gaussian_heatmap


def test_exact_center_amplitude():
    maps = rasterize_consensus_gaussian_heatmap(
        [{"x": 64.0, "y": 48.0, "amplitude": 1.0, "sigma_cells": 1.0}],
        [],
        image_height=512,
        image_width=512,
        stride=8,
    )

    assert maps["score_target"][6, 8] == pytest.approx(1.0)
    assert maps["score_target"][6, 9] == pytest.approx(math.exp(-0.5))
    assert maps["score_center_map"][6, 8] == pytest.approx(1.0)


def test_fractional_center_does_not_snap():
    maps = rasterize_consensus_gaussian_heatmap(
        [{"x": 63.0, "y": 48.0, "amplitude": 1.0, "sigma_cells": 1.0}],
        [],
        image_height=512,
        image_width=512,
        stride=8,
    )

    nearest = maps["score_target"][6, 8]
    assert nearest < 1.0
    assert nearest > 0.98


def test_max_composition_not_sum():
    maps = rasterize_consensus_gaussian_heatmap(
        [
            {"x": 64.0, "y": 64.0, "amplitude": 1.0, "sigma_cells": 1.0},
            {"x": 72.0, "y": 64.0, "amplitude": 1.0, "sigma_cells": 1.0},
        ],
        [],
        image_height=512,
        image_width=512,
        stride=8,
    )

    assert maps["score_target"].max() <= 1.0
    assert maps["score_target"][8, 8] == pytest.approx(1.0)


def test_singleton_ignore_does_not_erase_positive():
    maps = rasterize_consensus_gaussian_heatmap(
        [{"x": 64.0, "y": 64.0, "amplitude": 1.0, "sigma_cells": 1.0}],
        [{"x": 66.0, "y": 64.0, "sigma_cells": 1.25}],
        image_height=512,
        image_width=512,
        stride=8,
    )

    assert maps["score_target"][8, 8] > 0.0
    assert maps["score_weight"][8, 8] > 0.0
    assert maps["score_ignore_mask"][8, 8] == 0


def test_singleton_creates_zero_weight_ambiguous_background():
    maps = rasterize_consensus_gaussian_heatmap(
        [],
        [{"x": 64.0, "y": 64.0, "sigma_cells": 1.25}],
        image_height=512,
        image_width=512,
        stride=8,
    )

    assert np.count_nonzero(maps["score_target"]) == 0
    assert np.count_nonzero(maps["score_ignore_mask"]) > 0
    assert maps["score_weight"][maps["score_ignore_mask"] > 0].max() == pytest.approx(0.0)


def test_requires_exact_divisibility():
    with pytest.raises(ValueError, match="divisible"):
        rasterize_consensus_gaussian_heatmap([], [], 511, 512, 8)

from __future__ import annotations

from math import ceil, floor
from typing import Any

import numpy as np


def _require_divisible(image_height: int, image_width: int, stride: int) -> tuple[int, int]:
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if image_height % stride != 0 or image_width % stride != 0:
        raise ValueError(
            "image size must be divisible by minutiae label stride: "
            f"height={image_height}, width={image_width}, stride={stride}"
        )
    return image_height // stride, image_width // stride


def _finite_float(value: Any, name: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def rasterize_consensus_gaussian_heatmap(
    positive_minutiae: list[dict[str, Any]],
    single_source_candidates: list[dict[str, Any]] | None,
    image_height: int,
    image_width: int,
    stride: int,
    *,
    gaussian_truncate_sigma: float = 3.0,
    single_source_ignore_sigma_cells: float = 1.25,
    single_source_ignore_truncate_sigma: float = 3.0,
    gaussian_positive_weight: float = 1.0,
    gaussian_shoulder_weight: float = 0.5,
    safe_negative_weight: float = 1.0,
    single_source_ignore_weight: float = 0.0,
    default_amplitude: float = 1.0,
    default_sigma_cells: float = 1.25,
) -> dict[str, np.ndarray]:
    """Rasterize consensus minutiae into a soft Gaussian score target."""

    h_out, w_out = _require_divisible(image_height, image_width, stride)

    if gaussian_truncate_sigma <= 0:
        raise ValueError("gaussian_truncate_sigma must be positive")
    if single_source_ignore_sigma_cells <= 0:
        raise ValueError("single_source_ignore_sigma_cells must be positive")
    if single_source_ignore_truncate_sigma <= 0:
        raise ValueError("single_source_ignore_truncate_sigma must be positive")

    score_target = np.zeros((h_out, w_out), dtype=np.float32)
    score_weight = np.full((h_out, w_out), float(safe_negative_weight), dtype=np.float32)
    score_ignore_mask = np.zeros((h_out, w_out), dtype=np.uint8)
    score_center_map = np.zeros((h_out, w_out), dtype=np.float32)

    for idx, minutia in enumerate(positive_minutiae):
        x = _finite_float(minutia["x"], f"positive_minutiae[{idx}].x")
        y = _finite_float(minutia["y"], f"positive_minutiae[{idx}].y")
        if x < 0 or x >= image_width or y < 0 or y >= image_height:
            continue

        amplitude = float(minutia.get("amplitude", default_amplitude))
        sigma_cells = float(minutia.get("sigma_cells", default_sigma_cells))
        if not np.isfinite(amplitude) or amplitude <= 0:
            continue
        if not np.isfinite(sigma_cells) or sigma_cells <= 0:
            raise ValueError(f"invalid sigma_cells for positive minutia {idx}: {sigma_cells}")
        amplitude = float(np.clip(amplitude, 0.0, 1.0))

        cx = x / float(stride)
        cy = y / float(stride)

        center_c = int(floor(cx))
        center_r = int(floor(cy))
        if 0 <= center_r < h_out and 0 <= center_c < w_out:
            score_center_map[center_r, center_c] = 1.0

        radius = int(ceil(float(gaussian_truncate_sigma) * sigma_cells))
        r0 = max(0, int(floor(cy - radius)))
        r1 = min(h_out - 1, int(ceil(cy + radius)))
        c0 = max(0, int(floor(cx - radius)))
        c1 = min(w_out - 1, int(ceil(cx + radius)))
        if r1 < r0 or c1 < c0:
            continue

        rr, cc = np.mgrid[r0 : r1 + 1, c0 : c1 + 1]
        dist2 = (cc.astype(np.float32) - cx) ** 2 + (rr.astype(np.float32) - cy) ** 2
        gaussian = amplitude * np.exp(-dist2 / (2.0 * sigma_cells * sigma_cells))
        gaussian = gaussian.astype(np.float32)

        window = score_target[r0 : r1 + 1, c0 : c1 + 1]
        np.maximum(window, gaussian, out=window)

    np.clip(score_target, 0.0, 1.0, out=score_target)

    core_mask = score_target >= 0.5
    shoulder_mask = (score_target > 0.0) & (score_target < 0.5)
    score_weight[core_mask] = float(gaussian_positive_weight)
    score_weight[shoulder_mask] = float(gaussian_shoulder_weight)

    for idx, minutia in enumerate(single_source_candidates or []):
        x = _finite_float(minutia["x"], f"single_source_candidates[{idx}].x")
        y = _finite_float(minutia["y"], f"single_source_candidates[{idx}].y")
        if x < 0 or x >= image_width or y < 0 or y >= image_height:
            continue

        cx = x / float(stride)
        cy = y / float(stride)
        sigma_cells = float(minutia.get("sigma_cells", single_source_ignore_sigma_cells))
        if not np.isfinite(sigma_cells) or sigma_cells <= 0:
            sigma_cells = float(single_source_ignore_sigma_cells)

        radius = int(ceil(float(single_source_ignore_truncate_sigma) * sigma_cells))
        r0 = max(0, int(floor(cy - radius)))
        r1 = min(h_out - 1, int(ceil(cy + radius)))
        c0 = max(0, int(floor(cx - radius)))
        c1 = min(w_out - 1, int(ceil(cx + radius)))
        if r1 < r0 or c1 < c0:
            continue

        rr, cc = np.mgrid[r0 : r1 + 1, c0 : c1 + 1]
        dist2 = (cc.astype(np.float32) - cx) ** 2 + (rr.astype(np.float32) - cy) ** 2
        support = dist2 <= float(radius * radius)

        target_window = score_target[r0 : r1 + 1, c0 : c1 + 1]
        weight_window = score_weight[r0 : r1 + 1, c0 : c1 + 1]
        ignore_window = score_ignore_mask[r0 : r1 + 1, c0 : c1 + 1]

        ambiguous = support & (target_window < 0.05)
        ignore_window[ambiguous] = 1
        weight_window[ambiguous] = float(single_source_ignore_weight)

    return {
        "score_target": score_target.astype(np.float32, copy=False),
        "score_weight": score_weight.astype(np.float32, copy=False),
        "score_ignore_mask": score_ignore_mask.astype(np.uint8, copy=False),
        "score_center_map": score_center_map.astype(np.float32, copy=False),
    }

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from scripts.minutiae_gaussian_heatmap import rasterize_consensus_gaussian_heatmap


@dataclass(slots=True)
class MinutiaeTargetConfig:
    minutiae_score_target: str = "gaussian"
    gaussian_truncate_sigma: float = 3.0
    gaussian_amp_3src: float = 1.0
    gaussian_amp_2src: float = 0.85
    gaussian_sigma_3src_cells: float = 1.0
    gaussian_sigma_2src_cells: float = 1.25
    single_source_ignore_sigma_cells: float = 1.25
    single_source_ignore_truncate_sigma: float = 3.0
    gaussian_shoulder_weight: float = 0.5
    gaussian_positive_weight: float = 1.0
    safe_negative_weight: float = 1.0
    single_source_ignore_weight: float = 0.0


def target_config_from_object(value: Any | None) -> MinutiaeTargetConfig:
    if value is None:
        return MinutiaeTargetConfig()
    defaults = MinutiaeTargetConfig()
    return MinutiaeTargetConfig(
        **{
            field: getattr(value, field, getattr(defaults, field))
            for field in defaults.__dataclass_fields__
        }
    )


def normalize_angle_pi(angle: np.ndarray) -> np.ndarray:
    return np.mod(angle, math.pi).astype(np.float32)


def normalize_angle_2pi_scalar(angle: float) -> float:
    value = math.fmod(float(angle), 2.0 * math.pi)
    if value < 0.0:
        value += 2.0 * math.pi
    return value


def resize_float(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(array.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(resized > 0, 1.0, 0.0).astype(np.float32)


def downsample_mask_for_points(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"expected 2D mask, got shape {mask.shape}")

    source_h, source_w = mask.shape
    target_h, target_w = target_shape
    target_h = max(1, int(target_h))
    target_w = max(1, int(target_w))
    mask_bool = mask > 0
    out = np.zeros((target_h, target_w), dtype=np.float32)

    for cy in range(target_h):
        y0 = int(math.floor(cy * source_h / target_h))
        y1 = int(math.ceil((cy + 1) * source_h / target_h))
        y0 = max(0, min(source_h, y0))
        y1 = max(y0 + 1, min(source_h, y1))
        for cx in range(target_w):
            x0 = int(math.floor(cx * source_w / target_w))
            x1 = int(math.ceil((cx + 1) * source_w / target_w))
            x0 = max(0, min(source_w, x0))
            x1 = max(x0 + 1, min(source_w, x1))
            if np.any(mask_bool[y0:y1, x0:x1]):
                out[cy, cx] = 1.0

    return out.astype(np.float32)


def compute_output_shape(height: int, width: int) -> tuple[int, int]:
    return max(1, int(height) // 8), max(1, int(width) // 8)


def build_orientation_one_hot(orientation: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bins = np.floor(np.clip(orientation, 0.0, math.pi - 1e-6) * (180.0 / math.pi)).astype(np.int64)
    one_hot = np.zeros((180, orientation.shape[0], orientation.shape[1]), dtype=np.float32)
    active = mask > 0
    ys, xs = np.nonzero(active)
    one_hot[bins[ys, xs], ys, xs] = 1.0
    return one_hot


def resize_orientation_for_model(orientation: np.ndarray, mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    cos2 = np.cos(2.0 * orientation).astype(np.float32) * mask.astype(np.float32)
    sin2 = np.sin(2.0 * orientation).astype(np.float32) * mask.astype(np.float32)
    cos2_small = resize_float(cos2, shape)
    sin2_small = resize_float(sin2, shape)
    orientation_small = 0.5 * np.arctan2(sin2_small, cos2_small)
    return normalize_angle_pi(orientation_small)


def _minutiae_label_stride(source_shape: tuple[int, int], target_shape: tuple[int, int]) -> int:
    source_height, source_width = source_shape
    target_height, target_width = target_shape
    if target_height <= 0 or target_width <= 0:
        raise ValueError(f"invalid target shape for minutiae labels: {target_shape}")
    if source_height % target_height != 0 or source_width % target_width != 0:
        raise ValueError("image size must be divisible by minutiae label stride")
    stride_y = source_height // target_height
    stride_x = source_width // target_width
    if stride_x != stride_y:
        raise ValueError(f"minutiae label stride must be isotropic, got x={stride_x}, y={stride_y}")
    return int(stride_x)


def rasterize_minutiae(
    minutiae: list[dict[str, Any]],
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    mask_small: np.ndarray,
) -> dict[str, np.ndarray]:
    target_height, target_width = target_shape
    source_height, source_width = source_shape
    score_map = np.zeros((target_height, target_width), dtype=np.float32)
    valid_mask = np.zeros((target_height, target_width), dtype=np.float32)
    minutia_x = np.zeros((target_height, target_width), dtype=np.int64)
    minutia_y = np.zeros((target_height, target_width), dtype=np.int64)
    minutia_x_offset = np.zeros((target_height, target_width), dtype=np.float32)
    minutia_y_offset = np.zeros((target_height, target_width), dtype=np.float32)
    minutia_orientation = np.zeros((target_height, target_width), dtype=np.int64)
    minutia_orientation_vec = np.zeros((2, target_height, target_width), dtype=np.float32)
    ownership = np.full((target_height, target_width), -1.0, dtype=np.float32)
    center_distance = np.full((target_height, target_width), np.inf, dtype=np.float32)
    cell_width = float(source_width) / float(max(target_width, 1))
    cell_height = float(source_height) / float(max(target_height, 1))

    for minutia in minutiae:
        x = float(minutia["x"])
        y = float(minutia["y"])
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if x < 0.0 or y < 0.0 or x >= float(source_width) or y >= float(source_height):
            continue
        cell_x = int(np.clip(math.floor(x * target_width / max(source_width, 1)), 0, target_width - 1))
        cell_y = int(np.clip(math.floor(y * target_height / max(source_height, 1)), 0, target_height - 1))
        if mask_small[cell_y, cell_x] <= 0:
            continue
        score = minutia.get("score")
        score_value = float(score) if score is not None and math.isfinite(float(score)) else 1.0
        score_value = float(np.clip(score_value, 0.0, 1.0))
        cell_origin_x = cell_x * cell_width
        cell_origin_y = cell_y * cell_height
        local_x = (x - cell_origin_x) / max(cell_width, 1e-6)
        local_y = (y - cell_origin_y) / max(cell_height, 1e-6)
        local_x = float(np.clip(local_x, 0.0, 1.0 - 1e-6))
        local_y = float(np.clip(local_y, 0.0, 1.0 - 1e-6))
        distance = float((local_x - 0.5) ** 2 + (local_y - 0.5) ** 2)
        replace = score_value > ownership[cell_y, cell_x]
        if abs(score_value - ownership[cell_y, cell_x]) <= 1e-6 and distance < center_distance[cell_y, cell_x]:
            replace = True
        if not replace:
            continue
        ownership[cell_y, cell_x] = score_value
        center_distance[cell_y, cell_x] = distance
        score_map[cell_y, cell_x] = 1.0
        valid_mask[cell_y, cell_x] = 1.0
        theta = normalize_angle_2pi_scalar(float(minutia["theta"]))
        minutia_x[cell_y, cell_x] = int(np.clip(math.floor(local_x * 8.0), 0, 7))
        minutia_y[cell_y, cell_x] = int(np.clip(math.floor(local_y * 8.0), 0, 7))
        minutia_x_offset[cell_y, cell_x] = local_x
        minutia_y_offset[cell_y, cell_x] = local_y
        minutia_orientation[cell_y, cell_x] = int(np.clip(math.floor(theta * (360.0 / (2.0 * math.pi))), 0, 359))
        minutia_orientation_vec[0, cell_y, cell_x] = float(math.cos(theta))
        minutia_orientation_vec[1, cell_y, cell_x] = float(math.sin(theta))

    valid_mask *= mask_small
    score_map *= mask_small
    return {
        "minutia_score": np.where(score_map > 0.0, 1.0, 0.0)[np.newaxis, ...].astype(np.float32),
        "minutia_valid_mask": np.where(valid_mask > 0.0, 1.0, 0.0)[np.newaxis, ...].astype(np.float32),
        "minutia_x": minutia_x.astype(np.int64),
        "minutia_y": minutia_y.astype(np.int64),
        "minutia_x_offset": minutia_x_offset[np.newaxis, ...].astype(np.float32),
        "minutia_y_offset": minutia_y_offset[np.newaxis, ...].astype(np.float32),
        "minutia_orientation": minutia_orientation.astype(np.int64),
        "minutia_orientation_vec": minutia_orientation_vec.astype(np.float32),
    }


def apply_gaussian_score_targets(
    minutia_targets: dict[str, np.ndarray],
    *,
    positive_minutiae: list[dict[str, Any]],
    single_source_candidates: list[dict[str, Any]] | None,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    mask_small: np.ndarray,
    extractor_config: Any | None = None,
) -> dict[str, np.ndarray]:
    config = target_config_from_object(extractor_config)
    if config.minutiae_score_target == "binary":
        score = minutia_targets["minutia_score"].astype(np.float32, copy=False)
        minutia_targets["minutia_score_weight_map"] = np.ones_like(score, dtype=np.float32)
        minutia_targets["minutia_score_ignore_mask"] = np.zeros_like(score, dtype=np.float32)
        minutia_targets["minutia_score_center_map"] = score.copy()
        return minutia_targets

    stride = _minutiae_label_stride(source_shape, target_shape)
    heatmap = rasterize_consensus_gaussian_heatmap(
        positive_minutiae=positive_minutiae,
        single_source_candidates=single_source_candidates or [],
        image_height=int(source_shape[0]),
        image_width=int(source_shape[1]),
        stride=stride,
        gaussian_truncate_sigma=float(config.gaussian_truncate_sigma),
        single_source_ignore_sigma_cells=float(config.single_source_ignore_sigma_cells),
        single_source_ignore_truncate_sigma=float(config.single_source_ignore_truncate_sigma),
        gaussian_positive_weight=float(config.gaussian_positive_weight),
        gaussian_shoulder_weight=float(config.gaussian_shoulder_weight),
        safe_negative_weight=float(config.safe_negative_weight),
        single_source_ignore_weight=float(config.single_source_ignore_weight),
        default_amplitude=float(config.gaussian_amp_2src),
        default_sigma_cells=float(config.gaussian_sigma_2src_cells),
    )
    point_mask = np.where(mask_small > 0.0, 1.0, 0.0).astype(np.float32)
    minutia_targets["minutia_score"] = (heatmap["score_target"] * point_mask)[np.newaxis, ...].astype(np.float32)
    minutia_targets["minutia_score_weight_map"] = (heatmap["score_weight"] * point_mask)[np.newaxis, ...].astype(np.float32)
    minutia_targets["minutia_score_ignore_mask"] = (heatmap["score_ignore_mask"] * point_mask.astype(np.uint8))[np.newaxis, ...].astype(np.float32)
    minutia_targets["minutia_score_center_map"] = (heatmap["score_center_map"] * point_mask)[np.newaxis, ...].astype(np.float32)
    return minutia_targets


def build_featurenet_targets(
    gray_image: np.ndarray,
    mask: np.ndarray,
    orientation: np.ndarray,
    ridge_period: np.ndarray,
    gradient: np.ndarray | None,
    minutiae: list[dict[str, Any]],
    single_source_candidates: list[dict[str, Any]] | None = None,
    extractor_config: Any | None = None,
    output_shape: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    if output_shape is None:
        output_shape = compute_output_shape(*gray_image.shape)
    mask_small_dense = resize_mask(mask, output_shape)
    mask_small_points = downsample_mask_for_points(mask, output_shape)
    orientation_small = resize_orientation_for_model(orientation, mask.astype(np.float32) / 255.0, output_shape)
    orientation_one_hot = build_orientation_one_hot(orientation_small, mask_small_dense)
    ridge_small = resize_float(ridge_period, output_shape) * mask_small_dense
    ridge_max = float(ridge_small.max(initial=0.0))
    if ridge_max > 0.0:
        ridge_small = ridge_small / ridge_max

    minutia_targets = rasterize_minutiae(minutiae, gray_image.shape, output_shape, mask_small_points)
    minutia_targets = apply_gaussian_score_targets(
        minutia_targets,
        positive_minutiae=minutiae,
        single_source_candidates=single_source_candidates,
        source_shape=gray_image.shape,
        target_shape=output_shape,
        mask_small=mask_small_points,
        extractor_config=extractor_config,
    )
    targets = {
        "orientation": orientation_one_hot.astype(np.float32),
        "ridge_period": ridge_small[np.newaxis, ...].astype(np.float32),
        "minutia_score": minutia_targets["minutia_score"],
        "minutia_score_weight_map": minutia_targets["minutia_score_weight_map"],
        "minutia_score_ignore_mask": minutia_targets["minutia_score_ignore_mask"],
        "minutia_score_center_map": minutia_targets["minutia_score_center_map"],
        "minutia_valid_mask": minutia_targets["minutia_valid_mask"],
        "minutia_x": minutia_targets["minutia_x"],
        "minutia_y": minutia_targets["minutia_y"],
        "minutia_x_offset": minutia_targets["minutia_x_offset"],
        "minutia_y_offset": minutia_targets["minutia_y_offset"],
        "minutia_orientation": minutia_targets["minutia_orientation"],
        "minutia_orientation_vec": minutia_targets["minutia_orientation_vec"],
        "output_mask": mask_small_dense[np.newaxis, ...].astype(np.float32),
    }
    if gradient is not None:
        grad_x_small = resize_float(gradient[:, :, 0], output_shape) * mask_small_dense
        grad_y_small = resize_float(gradient[:, :, 1], output_shape) * mask_small_dense
        targets["gradient"] = np.stack([grad_x_small, grad_y_small], axis=0).astype(np.float32)
    return targets

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import sysconfig
from pathlib import Path

def ensure_stdlib_copy_module() -> None:
    stdlib_copy = Path(sysconfig.get_paths()["stdlib"]) / "copy.py"
    spec = importlib.util.spec_from_file_location("copy", stdlib_copy)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not resolve stdlib copy module from {stdlib_copy}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = module
    spec.loader.exec_module(module)


ensure_stdlib_copy_module()

from dataclasses import dataclass

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SOLOV2_CONFIG = REPO_ROOT / "solov2_assets" / "solov2_distal_phalanx_infer.py"
DEFAULT_SOLOV2_CHECKPOINT = REPO_ROOT / "weights" / "solov2_distal_phalanx_best.pth"
SOLOV2_CONFIG_ENV = "FINGER_SOLOV2_CONFIG"
SOLOV2_CHECKPOINT_ENV = "FINGER_SOLOV2_CHECKPOINT"
SOLOV2_DEVICE_ENV = "FINGER_SOLOV2_DEVICE"

_MODEL_CACHE: dict[tuple[str, str, str], object] = {}


@dataclass(slots=True)
class PreprocessPipelineResult:
    enhanced: np.ndarray
    full_mask: np.ndarray
    center_mask: np.ndarray
    scaled_image: np.ndarray
    scaled_mask: np.ndarray
    ridge_period: float
    scale: float
    rotated_image: np.ndarray
    rotated_mask: np.ndarray
    yaw_angle: float


def _clahe_grid_from_tile_pixels(
    gray_image: np.ndarray, tile_size_pixels: int = 60
) -> tuple[int, int]:
    height, width = gray_image.shape[:2]
    grid_x = max(1, int(np.ceil(width / float(tile_size_pixels))))
    grid_y = max(1, int(np.ceil(height / float(tile_size_pixels))))
    return grid_x, grid_y


def _extract_gray_and_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if image.ndim == 2:
        return image, None
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY), image[:, :, 3]
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None
    raise ValueError(f"unsupported image shape for preprocessing: {image.shape}")


def _restore_output_channels(gray_image: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if alpha is None:
        return gray_image

    output = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGRA)
    output[:, :, 3] = alpha
    return output


def _extract_inference_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"unsupported image shape for segmentation: {image.shape}")


def _normalize_angle_pi(angle: float) -> float:
    wrapped = float(angle) % math.pi
    return wrapped if math.isfinite(wrapped) else 0.0


def _normalize_angle_degrees_180(angle_degrees: float) -> float:
    wrapped = (float(angle_degrees) + 180.0) % 360.0 - 180.0
    return wrapped if math.isfinite(wrapped) else 0.0


def _average_orientation(angles: np.ndarray) -> float | None:
    finite = angles[np.isfinite(angles)]
    if finite.size == 0:
        return None
    sin2 = float(np.mean(np.sin(2.0 * finite)))
    cos2 = float(np.mean(np.cos(2.0 * finite)))
    if abs(sin2) + abs(cos2) <= 1e-8:
        return None
    return _normalize_angle_pi(0.5 * math.atan2(sin2, cos2))


def _gaussian_kernel(size: int, sigma: float = 3.0) -> np.ndarray:
    if size <= 0 or size % 2 == 0:
        raise ValueError("Gaussian kernel size must be a positive odd integer")
    radius = size // 2
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-0.5 * (xx * xx + yy * yy) / float(sigma * sigma))
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0.0:
        raise ValueError("Gaussian kernel has zero integral")
    return (kernel / kernel_sum).astype(np.float32)


def _ridge_frequency_mask(gray_image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        roi_mask = gray_image > 0
    else:
        roi_mask = np.asarray(mask) > 0
        if roi_mask.shape != gray_image.shape:
            raise ValueError(
                f"mask shape {roi_mask.shape} does not match image shape {gray_image.shape}"
            )
    if not np.any(roi_mask):
        raise RuntimeError("cannot estimate ridge frequency from an empty ROI")
    return roi_mask


def _as_uint8_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {mask.shape}")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _estimate_dominant_ridge_orientation(gray_image: np.ndarray, mask: np.ndarray) -> float:
    image_float = gray_image.astype(np.float32)
    grad_x = cv2.Sobel(image_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_float, cv2.CV_32F, 0, 1, ksize=3)
    weights = mask.astype(np.float32)
    gxx = float(np.sum(weights * grad_x * grad_x))
    gyy = float(np.sum(weights * grad_y * grad_y))
    gxy = float(np.sum(weights * grad_x * grad_y))
    if gxx + gyy <= 1e-6:
        raise RuntimeError("cannot estimate ridge orientation from an image with no gradient energy")

    gradient_angle = 0.5 * math.atan2(2.0 * gxy, gxx - gyy)
    return _normalize_angle_pi(gradient_angle + math.pi / 2.0)


def _orientation_for_block(
    gray_image: np.ndarray,
    orientation_map: np.ndarray | None,
    orientation_scalar: float | None,
    mask: np.ndarray,
    center_y: int,
    center_x: int,
    block_size: int,
) -> float | None:
    if orientation_scalar is not None:
        return orientation_scalar

    if orientation_map is not None:
        half = block_size // 2
        y0 = max(0, center_y - half)
        y1 = min(mask.shape[0], center_y + half)
        x0 = max(0, center_x - half)
        x1 = min(mask.shape[1], center_x + half)

        local_mask = mask[y0:y1, x0:x1] > 0
        local_orientation = orientation_map[y0:y1, x0:x1]
        if not np.any(local_mask):
            return None

        return _average_orientation(local_orientation[local_mask])

    return _estimate_local_ridge_orientation(
        gray_image=gray_image,
        mask=mask,
        center_y=center_y,
        center_x=center_x,
        block_size=block_size,
    )


def _oriented_x_signature(
    gray_image: np.ndarray,
    mask: np.ndarray,
    center_y: int,
    center_x: int,
    orientation: float,
    window_length: int,
    window_width: int,
) -> np.ndarray | None:
    k_offsets = np.arange(window_length, dtype=np.float32) - (window_length - 1.0) / 2.0
    d_offsets = np.arange(window_width, dtype=np.float32) - (window_width - 1.0) / 2.0
    kk, dd = np.meshgrid(k_offsets, d_offsets, indexing="ij")

    cos_t = math.cos(orientation)
    sin_t = math.sin(orientation)
    x_map = center_x + dd * cos_t - kk * sin_t
    y_map = center_y + dd * sin_t + kk * cos_t

    sampled_mask = cv2.remap(
        mask.astype(np.float32),
        x_map.astype(np.float32),
        y_map.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    valid_samples = sampled_mask > 0.5
    if float(np.mean(valid_samples)) < 0.75:
        return None

    sampled_image = cv2.remap(
        gray_image.astype(np.float32),
        x_map.astype(np.float32),
        y_map.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    sampled_image[~valid_samples] = 0.0
    valid_counts = np.maximum(np.sum(valid_samples, axis=1), 1)
    return (np.sum(sampled_image, axis=1) / valid_counts).astype(np.float32)


def _ridge_period_from_signature(
    signature: np.ndarray,
    min_period: float,
    max_period: float,
) -> float | None:
    if signature.size < 3:
        return None

    padded = np.pad(signature.astype(np.float32), (1, 1), mode="edge")
    smoothed = np.convolve(
        padded,
        np.array([0.25, 0.5, 0.25], dtype=np.float32),
        mode="valid",
    )
    if float(smoothed.max(initial=0.0) - smoothed.min(initial=0.0)) <= 1e-3:
        return None

    threshold = float(np.mean(smoothed))
    peaks = [
        index
        for index in range(1, smoothed.size - 1)
        if smoothed[index] > smoothed[index - 1]
        and smoothed[index] >= smoothed[index + 1]
        and smoothed[index] >= threshold
    ]
    if len(peaks) < 2:
        return None

    distances = np.diff(np.asarray(peaks, dtype=np.float32))
    valid_distances = distances[(distances >= min_period) & (distances <= max_period)]
    if valid_distances.size == 0:
        return None
    return float(np.mean(valid_distances))


def _interpolate_frequency_grid(
    frequency: np.ndarray,
    active: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    result = frequency.copy()
    radius = kernel.shape[0] // 2
    max_iterations = max(1, int(np.count_nonzero(active)))

    for _ in range(max_iterations):
        invalid_points = np.argwhere(active & ~(result > 0.0))
        if invalid_points.size == 0:
            break

        updated = result.copy()
        changed = False
        for y, x in invalid_points:
            y0 = max(0, y - radius)
            y1 = min(result.shape[0], y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(result.shape[1], x + radius + 1)
            ky0 = radius - (y - y0)
            ky1 = ky0 + (y1 - y0)
            kx0 = radius - (x - x0)
            kx1 = kx0 + (x1 - x0)

            neighborhood = result[y0:y1, x0:x1]
            valid = active[y0:y1, x0:x1] & (neighborhood > 0.0)
            if not np.any(valid):
                continue

            weights = kernel[ky0:ky1, kx0:kx1] * valid.astype(np.float32)
            weight_sum = float(weights.sum())
            if weight_sum <= 0.0:
                continue
            updated[y, x] = float(np.sum(neighborhood * weights) / weight_sum)
            changed = True

        result = updated
        if not changed:
            break

    return result


def _smooth_frequency_grid(
    frequency: np.ndarray,
    active: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    smoothed = np.zeros_like(frequency, dtype=np.float32)
    radius = kernel.shape[0] // 2
    for y, x in np.argwhere(active & (frequency > 0.0)):
        y0 = max(0, y - radius)
        y1 = min(frequency.shape[0], y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(frequency.shape[1], x + radius + 1)
        ky0 = radius - (y - y0)
        ky1 = ky0 + (y1 - y0)
        kx0 = radius - (x - x0)
        kx1 = kx0 + (x1 - x0)

        neighborhood = frequency[y0:y1, x0:x1]
        valid = active[y0:y1, x0:x1] & (neighborhood > 0.0)
        if not np.any(valid):
            continue
        weights = kernel[ky0:ky1, kx0:kx1] * valid.astype(np.float32)
        weight_sum = float(weights.sum())
        if weight_sum > 0.0:
            smoothed[y, x] = float(np.sum(neighborhood * weights) / weight_sum)
    return smoothed


def _rotation_matrix_for_center(shape: tuple[int, ...], angle_degrees: float) -> np.ndarray:
    height, width = shape[:2]
    center = ((width - 1.0) / 2.0, (height - 1.0) / 2.0)
    return cv2.getRotationMatrix2D(center, float(angle_degrees), 1.0)


def _rotate_same_canvas(
    array: np.ndarray,
    angle_degrees: float,
    interpolation: int,
    border_value: int = 0,
) -> np.ndarray:
    matrix = _rotation_matrix_for_center(array.shape, angle_degrees)
    return cv2.warpAffine(
        array,
        matrix,
        (array.shape[1], array.shape[0]),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _principal_axis_angle_degrees(mask: np.ndarray) -> float:
    refined = _largest_component_mask(_as_uint8_mask(mask))
    ys, xs = np.where(refined > 0)
    if xs.size < 2 or ys.size < 2:
        raise RuntimeError("cannot estimate centerline from an empty or degenerate mask")

    coords = np.column_stack([xs, ys]).astype(np.float32)
    coords -= np.mean(coords, axis=0, keepdims=True)
    covariance = np.cov(coords, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    return _normalize_angle_degrees_180(angle)


def _foreground_band_width(mask: np.ndarray, rows: np.ndarray) -> float:
    widths: list[float] = []
    for row in rows:
        cols = np.flatnonzero(mask[int(row)] > 0)
        if cols.size > 0:
            widths.append(float(cols[-1] - cols[0] + 1))
    if not widths:
        return 0.0
    return float(np.median(np.asarray(widths, dtype=np.float32)))


def _should_flip_tip_to_top(mask: np.ndarray) -> bool:
    refined = _largest_component_mask(_as_uint8_mask(mask))
    ys, _ = np.where(refined > 0)
    if ys.size < 2:
        return False

    y_min = int(ys.min())
    y_max = int(ys.max())
    height = y_max - y_min + 1
    if height < 8:
        return False

    band_height = max(2, int(round(0.2 * height)))
    top_rows = np.arange(y_min, min(y_min + band_height, y_max + 1), dtype=np.int32)
    bottom_rows = np.arange(max(y_min, y_max - band_height + 1), y_max + 1, dtype=np.int32)
    top_width = _foreground_band_width(refined, top_rows)
    bottom_width = _foreground_band_width(refined, bottom_rows)
    if top_width <= 0.0 or bottom_width <= 0.0:
        return False

    min_margin = max(4.0, 0.08 * max(top_width, bottom_width))
    return (top_width - bottom_width) > min_margin


def _resolve_asset_path(
    explicit_path: str | Path | None,
    env_var_name: str,
    default_path: Path,
) -> Path:
    candidate = explicit_path
    if candidate is None:
        env_value = os.environ.get(env_var_name)
        if env_value:
            candidate = env_value
    path = Path(candidate) if candidate is not None else default_path
    resolved = path.expanduser()
    if not resolved.exists():
        raise FileNotFoundError(
            f"required SOLOv2 asset was not found: {resolved}"
            f" (set {env_var_name} or pass an explicit path)"
        )
    return resolved.resolve()


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    env_device = os.environ.get(SOLOV2_DEVICE_ENV)
    if env_device:
        return env_device

    try:
        import torch
    except Exception:
        return "cpu"

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_mmdet_apis() -> tuple[object, object]:
    try:
        from mmdet.apis import inference_detector, init_detector
    except Exception as exc:
        raise RuntimeError(
            "MMDetection inference dependencies are unavailable. "
            "Run this path under the Linux/WSL environment where SOLOv2 was installed."
        ) from exc
    return init_detector, inference_detector


def _get_detector(
    *,
    model_config: str | Path | None = None,
    checkpoint: str | Path | None = None,
    device: str | None = None,
) -> object:
    config_path = _resolve_asset_path(model_config, SOLOV2_CONFIG_ENV, DEFAULT_SOLOV2_CONFIG)
    checkpoint_path = _resolve_asset_path(
        checkpoint,
        SOLOV2_CHECKPOINT_ENV,
        DEFAULT_SOLOV2_CHECKPOINT,
    )
    resolved_device = _resolve_device(device)
    cache_key = (str(config_path), str(checkpoint_path), resolved_device)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    init_detector, _ = _load_mmdet_apis()
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    detector = init_detector(str(config_path), str(checkpoint_path), device=resolved_device)
    _MODEL_CACHE[cache_key] = detector
    return detector


def _to_numpy(array_like: object) -> np.ndarray:
    if hasattr(array_like, "detach"):
        array_like = array_like.detach()
    if hasattr(array_like, "cpu"):
        array_like = array_like.cpu()
    return np.asarray(array_like)


def _extract_result_mask(result: object, score_thr: float) -> np.ndarray:
    pred_instances = getattr(result, "pred_instances", None)
    if pred_instances is None:
        raise RuntimeError("SOLOv2 inference result did not include pred_instances")

    scores = _to_numpy(getattr(pred_instances, "scores", None))
    masks = _to_numpy(getattr(pred_instances, "masks", None))
    if scores.size == 0 or masks.size == 0:
        raise RuntimeError("no distal phalanx instance was detected")

    scores = scores.astype(np.float32).reshape(-1)
    candidate_indices = np.flatnonzero(scores >= float(score_thr))
    if candidate_indices.size == 0:
        raise RuntimeError(
            f"no distal phalanx instance met the score threshold {float(score_thr):.3f}"
        )

    best_index = int(candidate_indices[np.argmax(scores[candidate_indices])])
    mask = masks[best_index]
    if mask.ndim != 2:
        raise RuntimeError(f"unexpected SOLOv2 mask shape: {mask.shape}")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("segmentation mask did not contain a valid foreground contour")
    largest = max(contours, key=cv2.contourArea)
    refined = np.zeros_like(mask)
    cv2.drawContours(refined, [largest], -1, 255, thickness=cv2.FILLED)
    return refined


def _refine_mask(mask: np.ndarray) -> np.ndarray:
    refined = np.where(mask > 0, 255, 0).astype(np.uint8)
    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    )
    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    refined = _largest_component_mask(refined)
    if not np.any(refined > 0):
        raise RuntimeError("segmentation mask became empty after refinement")
    return refined


def _masked_clahe(gray_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        raise RuntimeError("cannot apply masked CLAHE to an empty segmentation mask")

    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    gray_roi = gray_image[y_min:y_max, x_min:x_max]
    mask_roi = mask[y_min:y_max, x_min:x_max] > 0

    clahe_input = gray_roi.copy()
    fill_value = int(np.median(gray_roi[mask_roi]))
    clahe_input[~mask_roi] = fill_value

    enhanced_roi = apply_clahe(clahe_input)
    enhanced = np.zeros_like(gray_image, dtype=np.uint8)
    target_roi = enhanced[y_min:y_max, x_min:x_max]
    target_roi[mask_roi] = enhanced_roi[mask_roi]
    return enhanced


def apply_clahe(image: np.ndarray) -> np.ndarray:
    gray_image, alpha = _extract_gray_and_alpha(image)
    clahe_grid = _clahe_grid_from_tile_pixels(gray_image, tile_size_pixels=60)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=clahe_grid)
    enhanced = clahe.apply(gray_image)
    return _restore_output_channels(enhanced, alpha)


def estimate_ridge_period(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    orientation: float | np.ndarray | None = None,
    *,
    block_size: int = 16,
    window_length: int = 32,
    window_width: int = 16,
    min_period: float = 3.0,
    max_period: float = 25.0,
    interpolation_kernel_size: int = 7,
    smoothing_kernel_size: int = 7,
) -> float:
    """Estimate the mean ridge period in pixels.

    This is the value needed by the paper's scaling preprocessing.
    The expected use is:

        center_mask = circular_mask(full_finger_mask)
        period = estimate_ridge_period(enhanced_image, mask=center_mask)
        scale = 10.0 / period

    The returned value is ridge period in pixels, not cycles per pixel.
    """

    gray_image, _ = _extract_gray_and_alpha(np.asarray(image))
    if gray_image.ndim != 2:
        raise ValueError(f"expected a 2D grayscale image after conversion, got {gray_image.shape}")

    block_size = int(block_size)
    window_length = int(window_length)
    window_width = int(window_width)
    interpolation_kernel_size = int(interpolation_kernel_size)
    smoothing_kernel_size = int(smoothing_kernel_size)
    min_period = float(min_period)
    max_period = float(max_period)

    if block_size <= 0 or window_length <= 2 or window_width <= 0:
        raise ValueError("block_size, window_length, and window_width must be positive")
    if not math.isfinite(min_period) or not math.isfinite(max_period) or min_period <= 0.0:
        raise ValueError("min_period and max_period must be finite positive values")
    if max_period <= min_period:
        raise ValueError("max_period must be greater than min_period")

    roi_mask = _ridge_frequency_mask(gray_image, mask)
    interpolation_kernel = _gaussian_kernel(interpolation_kernel_size)
    smoothing_kernel = _gaussian_kernel(smoothing_kernel_size)

    orientation_map: np.ndarray | None = None
    orientation_scalar: float | None = None

    if orientation is not None:
        orientation_array = np.asarray(orientation)
        if orientation_array.ndim == 0:
            orientation_scalar = _normalize_angle_pi(float(orientation_array))
        else:
            if orientation_array.shape != gray_image.shape:
                raise ValueError(
                    "orientation map shape "
                    f"{orientation_array.shape} does not match image shape {gray_image.shape}"
                )
            orientation_map = np.mod(orientation_array.astype(np.float32), np.float32(math.pi))

    centers_y = list(range(block_size // 2, gray_image.shape[0], block_size))
    centers_x = list(range(block_size // 2, gray_image.shape[1], block_size))
    if not centers_y or not centers_x:
        raise RuntimeError("image is too small for ridge-period block estimation")

    frequency_grid = np.zeros((len(centers_y), len(centers_x)), dtype=np.float32)
    active_grid = np.zeros_like(frequency_grid, dtype=bool)

    for gy, center_y in enumerate(centers_y):
        for gx, center_x in enumerate(centers_x):
            if not roi_mask[center_y, center_x]:
                continue

            active_grid[gy, gx] = True

            block_orientation = _orientation_for_block(
                gray_image=gray_image,
                orientation_map=orientation_map,
                orientation_scalar=orientation_scalar,
                mask=roi_mask,
                center_y=center_y,
                center_x=center_x,
                block_size=block_size,
            )
            if block_orientation is None:
                continue

            signature = _oriented_x_signature(
                gray_image=gray_image,
                mask=roi_mask,
                center_y=center_y,
                center_x=center_x,
                orientation=block_orientation,
                window_length=window_length,
                window_width=window_width,
            )
            if signature is None:
                continue

            period = _ridge_period_from_signature(
                signature=signature,
                min_period=min_period,
                max_period=max_period,
            )
            if period is not None:
                frequency_grid[gy, gx] = np.float32(1.0 / period)

    if not np.any(active_grid):
        raise RuntimeError("no ridge-period blocks fell inside the ROI")

    if not np.any(frequency_grid > 0.0):
        raise RuntimeError("no reliable ridge period could be estimated from the ROI")

    interpolated = _interpolate_frequency_grid(
        frequency=frequency_grid,
        active=active_grid,
        kernel=interpolation_kernel,
    )

    smoothed = _smooth_frequency_grid(
        frequency=interpolated,
        active=active_grid,
        kernel=smoothing_kernel,
    )

    valid_frequencies = smoothed[active_grid & (smoothed > 0.0)]
    if valid_frequencies.size == 0:
        raise RuntimeError("no reliable ridge period remained after interpolation")

    periods = 1.0 / valid_frequencies.astype(np.float32)
    valid_periods = periods[(periods >= min_period) & (periods <= max_period)]
    if valid_periods.size == 0:
        raise RuntimeError("estimated ridge periods are outside the valid range")

    mean_period = float(np.mean(valid_periods))
    if not math.isfinite(mean_period) or mean_period <= 0.0:
        raise RuntimeError("estimated ridge period is invalid")

    return mean_period


def estimate_ridge_frequency(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    orientation: float | np.ndarray | None = None,
    *,
    block_size: int = 16,
    window_length: int = 32,
    window_width: int = 16,
    min_period: float = 3.0,
    max_period: float = 25.0,
    interpolation_kernel_size: int = 7,
    smoothing_kernel_size: int = 7,
) -> float:
    """Estimate scalar ridge frequency in cycles per pixel.

    Kept for backward compatibility. For paper-style scaling,
    prefer estimate_ridge_period().
    """
    period = estimate_ridge_period(
        image=image,
        mask=mask,
        orientation=orientation,
        block_size=block_size,
        window_length=window_length,
        window_width=window_width,
        min_period=min_period,
        max_period=max_period,
        interpolation_kernel_size=interpolation_kernel_size,
        smoothing_kernel_size=smoothing_kernel_size,
    )
    return float(1.0 / period)

def _estimate_local_ridge_orientation(
    gray_image: np.ndarray,
    mask: np.ndarray,
    center_y: int,
    center_x: int,
    block_size: int,
) -> float | None:
    """Estimate local ridge orientation for one block.

    Returns ridge direction in radians in [0, pi).
    The structure tensor gives the dominant gradient direction;
    ridge direction is perpendicular to gradient direction.
    """
    half = block_size // 2
    y0 = max(0, center_y - half)
    y1 = min(gray_image.shape[0], center_y + half)
    x0 = max(0, center_x - half)
    x1 = min(gray_image.shape[1], center_x + half)

    local_mask = mask[y0:y1, x0:x1] > 0
    if np.count_nonzero(local_mask) < max(8, int(0.25 * block_size * block_size)):
        return None

    patch = gray_image[y0:y1, x0:x1].astype(np.float32)
    grad_x = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)

    weights = local_mask.astype(np.float32)
    gxx = float(np.sum(weights * grad_x * grad_x))
    gyy = float(np.sum(weights * grad_y * grad_y))
    gxy = float(np.sum(weights * grad_x * grad_y))

    if gxx + gyy <= 1e-6:
        return None

    gradient_angle = 0.5 * math.atan2(2.0 * gxy, gxx - gyy)
    ridge_angle = gradient_angle + math.pi / 2.0
    return _normalize_angle_pi(ridge_angle)

def segment_then_clahe(
    image: np.ndarray,
    *,
    score_thr: float = 0.3,
    device: str | None = None,
    model_config: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    detector = _get_detector(
        model_config=model_config,
        checkpoint=checkpoint,
        device=device,
    )
    _, inference_detector = _load_mmdet_apis()

    inference_bgr = _extract_inference_bgr(image)
    result = inference_detector(detector, inference_bgr)
    mask = _refine_mask(_extract_result_mask(result, score_thr=score_thr))
    gray_image, _ = _extract_gray_and_alpha(image)
    enhanced = _masked_clahe(gray_image, mask)
    enhanced[mask <= 0] = 0
    return enhanced.astype(np.uint8), mask.astype(np.uint8)


def segment_then_clahe_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    refined_mask = _refine_mask(_as_uint8_mask(mask))
    gray_image, _ = _extract_gray_and_alpha(image)
    if gray_image.shape != refined_mask.shape:
        raise ValueError(
            f"fallback mask shape {refined_mask.shape} does not match image shape {gray_image.shape}"
        )
    enhanced = _masked_clahe(gray_image, refined_mask)
    enhanced[refined_mask <= 0] = 0
    return enhanced.astype(np.uint8), refined_mask.astype(np.uint8)


def circular_mask(roi: np.ndarray) -> np.ndarray:
    """Return the central circular mask covering about 20% of the ROI area.

    This follows the paper's scaling preprocessing:
    - find geometric center of the ROI
    - draw a central circle covering ~20% of the total ROI area
    - use this central region for average ridge-period estimation

    Works correctly for both 0/1 and 0/255 masks.
    """
    roi_bool = np.asarray(roi) > 0
    ys, xs = np.where(roi_bool)
    if xs.size == 0 or ys.size == 0:
        raise RuntimeError("cannot create circular mask from an empty ROI")

    cx = float(xs.mean())
    cy = float(ys.mean())

    roi_area = int(np.count_nonzero(roi_bool))
    radius = math.sqrt((0.20 * roi_area) / math.pi)

    h, w = roi_bool.shape
    y_grid, x_grid = np.ogrid[:h, :w]
    circle = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= radius ** 2

    return (roi_bool & circle).astype(np.uint8)


def scale_to_paper_ridge_period(
    enhanced: np.ndarray,
    full_mask: np.ndarray,
    *,
    target_period: float = 10.0,
    orientation: float | np.ndarray | None = None,
    center_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Scale fingerprint so the central mean ridge period becomes target_period pixels.

    Returns:
        scaled_image: resized enhanced fingerprint
        scaled_mask: resized full finger mask
        ridge_period: estimated central ridge period before scaling
        scale: resize factor applied to image and mask
    """
    if center_mask is None:
        center_mask = circular_mask(full_mask)

    ridge_period = estimate_ridge_period(
        enhanced,
        mask=center_mask,
        orientation=orientation,
    )

    scale = float(target_period) / float(ridge_period)
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"invalid ridge-period scaling factor: {scale}")

    scaled_image = cv2.resize(
        enhanced,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LINEAR,
    )

    scaled_mask = cv2.resize(
        full_mask,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )

    return scaled_image.astype(np.uint8), _as_uint8_mask(scaled_mask), ridge_period, scale


def rotate_to_vertical_centerline(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotate the segmented finger so its mask centerline is vertical.

    The canvas size is intentionally preserved to match the requested pipeline.
    The returned yaw angle is the OpenCV rotation angle, in degrees, applied to
    both image and mask.
    """

    gray_image, _ = _extract_gray_and_alpha(np.asarray(image))
    mask_u8 = _as_uint8_mask(np.asarray(mask))
    if gray_image.shape != mask_u8.shape:
        raise ValueError(
            f"image shape {gray_image.shape} does not match mask shape {mask_u8.shape}"
        )

    axis_angle = _principal_axis_angle_degrees(mask_u8)
    yaw_candidates = (
        _normalize_angle_degrees_180(axis_angle - 90.0),
        _normalize_angle_degrees_180(axis_angle + 90.0),
    )
    yaw_angle = min(yaw_candidates, key=lambda value: (abs(value), value))

    rotated_image = _rotate_same_canvas(
        gray_image,
        yaw_angle,
        interpolation=cv2.INTER_LINEAR,
        border_value=0,
    )
    rotated_mask = _rotate_same_canvas(
        mask_u8,
        yaw_angle,
        interpolation=cv2.INTER_NEAREST,
        border_value=0,
    )
    if _should_flip_tip_to_top(rotated_mask):
        rotated_image = np.ascontiguousarray(np.flipud(np.fliplr(rotated_image)))
        rotated_mask = np.ascontiguousarray(np.flipud(np.fliplr(rotated_mask)))
        yaw_angle = _normalize_angle_degrees_180(yaw_angle + 180.0)
    rotated_image[rotated_mask <= 0] = 0
    return rotated_image.astype(np.uint8), _as_uint8_mask(rotated_mask), float(yaw_angle)


def run_preprocess_pipeline(
    image: np.ndarray,
    *,
    score_thr: float = 0.3,
    device: str | None = None,
    model_config: str | Path | None = None,
    checkpoint: str | Path | None = None,
    target_period: float = 10.0,
    orientation: float | np.ndarray | None = None,
) -> PreprocessPipelineResult:
    enhanced, full_mask = segment_then_clahe(
        image,
        score_thr=score_thr,
        device=device,
        model_config=model_config,
        checkpoint=checkpoint,
    )
    center_mask = circular_mask(full_mask)
    scaled_image, scaled_mask, ridge_period, scale = scale_to_paper_ridge_period(
        enhanced,
        full_mask,
        target_period=target_period,
        orientation=orientation,
        center_mask=center_mask,
    )
    rotated_image, rotated_mask, yaw_angle = rotate_to_vertical_centerline(
        scaled_image,
        scaled_mask,
    )
    return PreprocessPipelineResult(
        enhanced=enhanced,
        full_mask=_as_uint8_mask(full_mask),
        center_mask=_as_uint8_mask(center_mask),
        scaled_image=scaled_image,
        scaled_mask=scaled_mask,
        ridge_period=float(ridge_period),
        scale=float(scale),
        rotated_image=rotated_image,
        rotated_mask=rotated_mask,
        yaw_angle=float(yaw_angle),
    )


def run_preprocess_pipeline_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    target_period: float = 10.0,
    orientation: float | np.ndarray | None = None,
) -> PreprocessPipelineResult:
    enhanced, full_mask = segment_then_clahe_from_mask(image, mask)
    center_mask = circular_mask(full_mask)
    scaled_image, scaled_mask, ridge_period, scale = scale_to_paper_ridge_period(
        enhanced,
        full_mask,
        target_period=target_period,
        orientation=orientation,
        center_mask=center_mask,
    )
    rotated_image, rotated_mask, yaw_angle = rotate_to_vertical_centerline(
        scaled_image,
        scaled_mask,
    )
    return PreprocessPipelineResult(
        enhanced=enhanced,
        full_mask=_as_uint8_mask(full_mask),
        center_mask=_as_uint8_mask(center_mask),
        scaled_image=scaled_image,
        scaled_mask=scaled_mask,
        ridge_period=float(ridge_period),
        scale=float(scale),
        rotated_image=rotated_image,
        rotated_mask=rotated_mask,
        yaw_angle=float(yaw_angle),
    )


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


def _load_cli_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"unable to load image: {path}")
    return image


def _write_pipeline_outputs(
    result: PreprocessPipelineResult,
    output_dir: Path,
    *,
    input_path: Path,
    score_thr: float,
    device: str,
    model_config: Path,
    checkpoint: Path,
    target_period: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_image(output_dir / "01_enhanced.png", result.enhanced)
    _write_image(output_dir / "02_full_mask.png", result.full_mask)
    _write_image(output_dir / "03_center_mask.png", result.center_mask)
    _write_image(output_dir / "04_scaled_image.png", result.scaled_image)
    _write_image(output_dir / "05_scaled_mask.png", result.scaled_mask)
    _write_image(output_dir / "06_rotated_image.png", result.rotated_image)
    _write_image(output_dir / "07_rotated_mask.png", result.rotated_mask)

    meta = {
        "input": str(input_path.resolve()),
        "score_thr": float(score_thr),
        "device": device,
        "config": str(model_config),
        "checkpoint": str(checkpoint),
        "target_period": float(target_period),
        "ridge_period": float(result.ridge_period),
        "scale": float(result.scale),
        "yaw_angle": float(result.yaw_angle),
        "enhanced_shape": list(result.enhanced.shape),
        "scaled_shape": list(result.scaled_image.shape),
        "rotated_shape": list(result.rotated_image.shape),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SOLOv2 segmentation, CLAHE, paper ridge-period scaling, and vertical centerline rotation."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input finger image path.")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for stage PNGs and meta.json.",
    )
    parser.add_argument("--device", default=None, help="SOLOv2 device, e.g. cpu or cuda:0.")
    parser.add_argument("--score-thr", type=float, default=0.3, help="SOLOv2 score threshold.")
    parser.add_argument("--target-period", type=float, default=10.0, help="Target ridge period in pixels.")
    parser.add_argument("--model-config", type=Path, default=None, help="SOLOv2 MMDetection config path.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="SOLOv2 checkpoint path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    input_path = args.input.expanduser()
    output_dir = args.output_dir.expanduser()
    model_config = _resolve_asset_path(args.model_config, SOLOV2_CONFIG_ENV, DEFAULT_SOLOV2_CONFIG)
    checkpoint = _resolve_asset_path(args.checkpoint, SOLOV2_CHECKPOINT_ENV, DEFAULT_SOLOV2_CHECKPOINT)
    device = _resolve_device(args.device)

    image = _load_cli_image(input_path)
    result = run_preprocess_pipeline(
        image,
        score_thr=float(args.score_thr),
        device=device,
        model_config=model_config,
        checkpoint=checkpoint,
        target_period=float(args.target_period),
    )
    _write_pipeline_outputs(
        result,
        output_dir,
        input_path=input_path,
        score_thr=float(args.score_thr),
        device=device,
        model_config=model_config,
        checkpoint=checkpoint,
        target_period=float(args.target_period),
    )
    print(json.dumps({"output_dir": str(output_dir.resolve()), "meta": str((output_dir / "meta.json").resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

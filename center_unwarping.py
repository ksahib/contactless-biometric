from __future__ import annotations

import math
from typing import Any

import numpy as np


def summarize_values(values: np.ndarray) -> dict[str, float | None]:
    finite = values[np.isfinite(values)].astype(np.float32)
    if finite.size == 0:
        return {
            "min": None,
            "p01": None,
            "p05": None,
            "median": None,
            "mean": None,
            "max": None,
            "span": None,
        }
    return {
        "min": float(np.min(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p05": float(np.percentile(finite, 5)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
        "span": float(np.max(finite) - np.min(finite)),
    }


def summarize_masked(values: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    return summarize_values(values[np.asarray(mask, dtype=bool)])


def compute_center_point_from_gradients(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    mask: np.ndarray,
) -> tuple[int, int, dict[str, Any]]:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        raise ValueError("Mask is empty; cannot choose an unwarping center point.")
    grad_mag = np.sqrt(np.square(gradient_x.astype(np.float32)) + np.square(gradient_y.astype(np.float32))).astype(np.float32)
    masked_grad = np.where(mask_bool, grad_mag, np.inf)
    flat_index = int(np.argmin(masked_grad))
    cy, cx = np.unravel_index(flat_index, grad_mag.shape)
    return int(cx), int(cy), {
        "gradient_magnitude": float(grad_mag[cy, cx]),
        "grad_mag_stats": summarize_masked(grad_mag, mask_bool),
    }


def _integrate_rows(metric: np.ndarray, mask: np.ndarray, center_x: int) -> tuple[np.ndarray, list[int]]:
    height, width = metric.shape
    out = np.full((height, width), np.nan, dtype=np.float32)
    fallback_rows: list[int] = []
    for row in range(height):
        valid_cols = np.flatnonzero(mask[row])
        if valid_cols.size == 0:
            continue
        if center_x in valid_cols:
            seed = int(center_x)
        else:
            seed = int(valid_cols[np.argmin(np.abs(valid_cols - center_x))])
            fallback_rows.append(int(row))
        out[row, seed] = 0.0
        for col in range(seed, width - 1):
            if not (mask[row, col] and mask[row, col + 1]):
                break
            out[row, col + 1] = out[row, col] + 0.5 * (metric[row, col] + metric[row, col + 1])
        for col in range(seed, 0, -1):
            if not (mask[row, col] and mask[row, col - 1]):
                break
            out[row, col - 1] = out[row, col] - 0.5 * (metric[row, col] + metric[row, col - 1])
    return out.astype(np.float32), fallback_rows


def _integrate_cols(metric: np.ndarray, mask: np.ndarray, center_y: int) -> tuple[np.ndarray, list[int]]:
    height, width = metric.shape
    out = np.full((height, width), np.nan, dtype=np.float32)
    fallback_cols: list[int] = []
    for col in range(width):
        valid_rows = np.flatnonzero(mask[:, col])
        if valid_rows.size == 0:
            continue
        if center_y in valid_rows:
            seed = int(center_y)
        else:
            seed = int(valid_rows[np.argmin(np.abs(valid_rows - center_y))])
            fallback_cols.append(int(col))
        out[seed, col] = 0.0
        for row in range(seed, height - 1):
            if not (mask[row, col] and mask[row + 1, col]):
                break
            out[row + 1, col] = out[row, col] + 0.5 * (metric[row, col] + metric[row + 1, col])
        for row in range(seed, 0, -1):
            if not (mask[row, col] and mask[row - 1, col]):
                break
            out[row - 1, col] = out[row, col] - 0.5 * (metric[row, col] + metric[row - 1, col])
    return out.astype(np.float32), fallback_cols


def compute_arc_length_maps(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    mask: np.ndarray,
    center_point: tuple[int, int],
) -> dict[str, Any]:
    mask_bool = np.asarray(mask, dtype=bool)
    cx, cy = int(center_point[0]), int(center_point[1])
    s = np.sqrt(1.0 + np.square(gradient_x.astype(np.float32))).astype(np.float32)
    t = np.sqrt(1.0 + np.square(gradient_y.astype(np.float32))).astype(np.float32)
    u, fallback_rows = _integrate_rows(s, mask_bool, cx)
    v, fallback_cols = _integrate_cols(t, mask_bool, cy)
    return {
        "u": u.astype(np.float32),
        "v": v.astype(np.float32),
        "row_metric": s,
        "col_metric": t,
        "fallback_rows": fallback_rows,
        "fallback_cols": fallback_cols,
    }


def build_unwarp_coordinates(
    u: np.ndarray,
    v: np.ndarray,
    center_point: tuple[int, int],
    mask: np.ndarray,
) -> dict[str, Any]:
    mask_bool = np.asarray(mask, dtype=bool)
    cx, cy = int(center_point[0]), int(center_point[1])
    x_new = np.full(u.shape, np.nan, dtype=np.float32)
    y_new = np.full(v.shape, np.nan, dtype=np.float32)
    valid = mask_bool & np.isfinite(u) & np.isfinite(v)
    if not np.any(valid):
        raise ValueError("Unwarping coordinates could not be computed inside the mask.")
    x_new[valid] = (float(cx) + u[valid]).astype(np.float32)
    y_new[valid] = (float(cy) + v[valid]).astype(np.float32)
    x0 = int(math.floor(float(np.min(x_new[valid]))))
    x1 = int(math.ceil(float(np.max(x_new[valid]))))
    y0 = int(math.floor(float(np.min(y_new[valid]))))
    y1 = int(math.ceil(float(np.max(y_new[valid]))))
    x_out = np.full(u.shape, np.nan, dtype=np.float32)
    y_out = np.full(v.shape, np.nan, dtype=np.float32)
    x_out[valid] = (x_new[valid] - float(x0)).astype(np.float32)
    y_out[valid] = (y_new[valid] - float(y0)).astype(np.float32)
    return {
        "x_new": x_new.astype(np.float32),
        "y_new": y_new.astype(np.float32),
        "x_out": x_out.astype(np.float32),
        "y_out": y_out.astype(np.float32),
        "output_offset_x": int(x0),
        "output_offset_y": int(y0),
        "output_bounds": {
            "x_new_min": float(np.min(x_new[valid])),
            "x_new_max": float(np.max(x_new[valid])),
            "y_new_min": float(np.min(y_new[valid])),
            "y_new_max": float(np.max(y_new[valid])),
            "x0": int(x0),
            "x1": int(x1),
            "y0": int(y0),
            "y1": int(y1),
        },
        "output_shape": [int(y1 - y0 + 1), int(x1 - x0 + 1)],
        "valid_mask": valid.astype(bool),
    }


def _fill_small_holes(image: np.ndarray, mask: np.ndarray, max_iterations: int = 2) -> tuple[np.ndarray, int]:
    filled = image.astype(np.float32, copy=True)
    fill_count = 0
    for _ in range(max_iterations):
        holes = mask & (~np.isfinite(filled))
        if not np.any(holes):
            break
        updates: list[tuple[int, int, float]] = []
        rows, cols = np.nonzero(holes)
        for row, col in zip(rows.tolist(), cols.tolist()):
            neighbors: list[float] = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr = row + dr
                cc = col + dc
                if rr < 0 or rr >= filled.shape[0] or cc < 0 or cc >= filled.shape[1]:
                    continue
                if mask[rr, cc] and np.isfinite(filled[rr, cc]):
                    neighbors.append(float(filled[rr, cc]))
            if len(neighbors) >= 2:
                updates.append((row, col, float(np.mean(neighbors))))
        if not updates:
            break
        for row, col, value in updates:
            filled[row, col] = value
        fill_count += len(updates)
    return filled.astype(np.float32), int(fill_count)


def _candidate_fill_region(supported: np.ndarray) -> np.ndarray:
    region = supported.astype(bool).copy()
    if supported.shape[0] < 3 or supported.shape[1] < 3:
        return region
    neighbor_touch = (
        supported[:-2, 1:-1].astype(np.int32)
        + supported[2:, 1:-1].astype(np.int32)
        + supported[1:-1, :-2].astype(np.int32)
        + supported[1:-1, 2:].astype(np.int32)
    ) >= 2
    region[1:-1, 1:-1] |= neighbor_touch
    return region


def resample_unwarped_image(
    image: np.ndarray,
    mask: np.ndarray,
    x_out: np.ndarray,
    y_out: np.ndarray,
    output_shape: tuple[int, int] | list[int],
    method: str = "bilinear_forward",
) -> dict[str, Any]:
    if method != "bilinear_forward":
        raise ValueError(f"Unsupported resampling method: {method}")
    mask_bool = np.asarray(mask, dtype=bool)
    height_out, width_out = int(output_shape[0]), int(output_shape[1])
    accum = np.zeros((height_out, width_out), dtype=np.float32)
    weights = np.zeros((height_out, width_out), dtype=np.float32)

    valid = mask_bool & np.isfinite(x_out) & np.isfinite(y_out)
    src_y, src_x = np.nonzero(valid)
    sample_x = x_out[valid].astype(np.float32)
    sample_y = y_out[valid].astype(np.float32)
    intensities = image[valid].astype(np.float32)

    x0 = np.floor(sample_x).astype(np.int32)
    y0 = np.floor(sample_y).astype(np.int32)
    dx = (sample_x - x0).astype(np.float32)
    dy = (sample_y - y0).astype(np.float32)

    neighbor_specs = (
        (0, 0, (1.0 - dx) * (1.0 - dy)),
        (1, 0, dx * (1.0 - dy)),
        (0, 1, (1.0 - dx) * dy),
        (1, 1, dx * dy),
    )
    for off_x, off_y, w in neighbor_specs:
        tx = x0 + off_x
        ty = y0 + off_y
        inside = (tx >= 0) & (tx < width_out) & (ty >= 0) & (ty < height_out) & (w > 1e-8)
        if not np.any(inside):
            continue
        np.add.at(accum, (ty[inside], tx[inside]), intensities[inside] * w[inside])
        np.add.at(weights, (ty[inside], tx[inside]), w[inside])

    unwarped = np.full((height_out, width_out), np.nan, dtype=np.float32)
    supported = weights > 1e-6
    unwarped[supported] = (accum[supported] / weights[supported]).astype(np.float32)
    filled, fill_count = _fill_small_holes(unwarped, _candidate_fill_region(supported))
    unwarped_mask = np.isfinite(filled)
    filled[~np.isfinite(filled)] = 0.0
    return {
        "image": filled.astype(np.float32),
        "mask": unwarped_mask.astype(np.uint8),
        "weights": weights.astype(np.float32),
        "hole_fill_count": int(fill_count),
    }


def run_center_unwarping(
    image: np.ndarray,
    mask: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    center_point_mode: str = "min_gradient",
    resampling_method: str = "bilinear_forward",
) -> dict[str, Any]:
    if center_point_mode != "min_gradient":
        raise ValueError(f"Unsupported center point mode: {center_point_mode}")
    center_x, center_y, center_info = compute_center_point_from_gradients(gradient_x, gradient_y, mask)
    arc_maps = compute_arc_length_maps(gradient_x, gradient_y, mask, (center_x, center_y))
    coordinates = build_unwarp_coordinates(arc_maps["u"], arc_maps["v"], (center_x, center_y), mask)
    resampled = resample_unwarped_image(
        image=image,
        mask=coordinates["valid_mask"],
        x_out=coordinates["x_out"],
        y_out=coordinates["y_out"],
        output_shape=coordinates["output_shape"],
        method=resampling_method,
    )
    src_y, src_x = np.nonzero(coordinates["valid_mask"])
    sample_stride = max(1, int(math.ceil(math.sqrt(max(src_x.size, 1) / 900.0))))
    overlay_samples = {
        "source_x": src_x[::sample_stride].astype(np.int32),
        "source_y": src_y[::sample_stride].astype(np.int32),
        "target_x": coordinates["x_new"][coordinates["valid_mask"]][::sample_stride].astype(np.float32),
        "target_y": coordinates["y_new"][coordinates["valid_mask"]][::sample_stride].astype(np.float32),
    }

    warnings: list[str] = []
    if arc_maps["fallback_rows"]:
        warnings.append(f"Used nearest-mask seed fallback on {len(arc_maps['fallback_rows'])} rows.")
    if arc_maps["fallback_cols"]:
        warnings.append(f"Used nearest-mask seed fallback on {len(arc_maps['fallback_cols'])} columns.")
    if resampled["hole_fill_count"] > 0:
        warnings.append(f"Filled {resampled['hole_fill_count']} isolated unwarped output pixels after splatting.")

    return {
        "center_point": [int(center_x), int(center_y)],
        "center_info": center_info,
        "u": arc_maps["u"].astype(np.float32),
        "v": arc_maps["v"].astype(np.float32),
        "x_new": coordinates["x_new"].astype(np.float32),
        "y_new": coordinates["y_new"].astype(np.float32),
        "x_out": coordinates["x_out"].astype(np.float32),
        "y_out": coordinates["y_out"].astype(np.float32),
        "output_offset_x": int(coordinates["output_offset_x"]),
        "output_offset_y": int(coordinates["output_offset_y"]),
        "output_bounds": coordinates["output_bounds"],
        "output_shape": coordinates["output_shape"],
        "valid_mask": coordinates["valid_mask"].astype(np.uint8),
        "unwarped_image": resampled["image"].astype(np.float32),
        "unwarped_mask": resampled["mask"].astype(np.uint8),
        "overlay_samples": overlay_samples,
        "warnings": warnings,
        "u_stats": summarize_masked(arc_maps["u"], coordinates["valid_mask"]),
        "v_stats": summarize_masked(arc_maps["v"], coordinates["valid_mask"]),
        "x_new_stats": summarize_masked(coordinates["x_new"], coordinates["valid_mask"]),
        "y_new_stats": summarize_masked(coordinates["y_new"], coordinates["valid_mask"]),
    }

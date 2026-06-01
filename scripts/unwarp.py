#!/usr/bin/env python
"""Algorithm-1 side-view ground-truth depth rendering and unwrapping.

This script is intended to replace the current diagnostic-side-unwrap path that
builds an unwrap directly from stitched/monotonic side-surface curves.

Correct data flow implemented here:

    front/left/right pose-normalized masks
      -> Algorithm-1 row-wise canonical 3D ellipse surface
      -> fixed orthographic camera projection for front/left/right views
      -> z-buffer visibility to create per-view dense depth maps
      -> extrapolated/smoothed depth field inside the observed mask
      -> smoothed/extrapolated depth gradients
      -> canonical-surface-chart unwrapping using Algorithm-1 canonical coordinates

This version keeps depth reconstruction and unwarping separated. The reconstruction
stage outputs a continuous smoothed depth field and gx/gy gradients. It also
computes an occlusion-safe support mask from the canonical Algorithm-1 geometry.
The unwarper does not use side-image x/y as the chart. Instead it uses the
canonical Algorithm-1 surface coordinates (canonical_x, canonical_z, source_y_front)
to build a stable row-wise arc-length chart of the visible surface.

Expected reconstruction_dir layout:

    reconstruction_dir/debug_views/front_pose_normalized.png
    reconstruction_dir/debug_views/front_pose_mask.png
    reconstruction_dir/debug_views/left_pose_normalized.png
    reconstruction_dir/debug_views/left_pose_mask.png
    reconstruction_dir/debug_views/right_pose_normalized.png
    reconstruction_dir/debug_views/right_pose_mask.png

Example:

    python algorithm1_side_depth_unwrap_fixed_v4.py \
      --reconstruction-dir ground_truth_smoke_dense_fixed_limit6/reconstructions/s01_f01_a01 \
      --output-dir tmp/algorithm1_side_depth_unwrap_fixed_v4/s01_f01_a01
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SQRT2 = math.sqrt(2.0)
DEFAULT_VIEW_ANGLES_DEG = {
    "front": 0.0,
    # If your scanner convention is reversed, swap these two signs via CLI.
    "left": -45.0,
    "right": 45.0,
}


@dataclass(frozen=True)
class RowStats:
    left: np.ndarray
    right: np.ndarray
    center: np.ndarray
    width: np.ndarray
    valid_rows: np.ndarray


@dataclass(frozen=True)
class ViewData:
    role: str
    image: np.ndarray
    mask: np.ndarray
    stats: RowStats


@dataclass(frozen=True)
class RowSurfaceParams:
    y_front: int
    y_left: int
    y_right: int
    a: float
    b: float
    cz: float


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _write_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.clip(image, 0, 255).astype(np.uint8))


def _row_stats(mask_u8: np.ndarray) -> RowStats:
    mask = mask_u8 > 0
    h = mask.shape[0]
    left = np.full(h, np.nan, dtype=np.float32)
    right = np.full(h, np.nan, dtype=np.float32)
    center = np.full(h, np.nan, dtype=np.float32)
    width = np.zeros(h, dtype=np.float32)
    valid: list[int] = []
    for y in range(h):
        xs = np.flatnonzero(mask[y])
        if xs.size == 0:
            continue
        x0 = int(xs[0])
        x1 = int(xs[-1])
        left[y] = float(x0)
        right[y] = float(x1)
        center[y] = 0.5 * float(x0 + x1)
        width[y] = float(x1 - x0 + 1)
        valid.append(y)
    return RowStats(left, right, center, width, np.asarray(valid, dtype=np.int32))


def _load_views(reconstruction_dir: Path) -> dict[str, ViewData]:
    debug_dir = reconstruction_dir / "debug_views"
    views: dict[str, ViewData] = {}
    for role in ("front", "left", "right"):
        image = _read_gray(debug_dir / f"{role}_pose_normalized.png")
        mask = _read_gray(debug_dir / f"{role}_pose_mask.png")
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        views[role] = ViewData(role=role, image=image, mask=mask, stats=_row_stats(mask))
    return views


def _map_row(source_rows: np.ndarray, target_rows: np.ndarray, y_target: int | float) -> int:
    """Map a row by normalized position within each view's valid row span."""
    if source_rows.size == 0 or target_rows.size == 0:
        return -1
    if float(target_rows[-1]) == float(target_rows[0]):
        rel = 0.0
    else:
        rel = (float(y_target) - float(target_rows[0])) / float(target_rows[-1] - target_rows[0])
    y_source = float(source_rows[0]) + rel * float(source_rows[-1] - source_rows[0])
    return int(np.clip(round(y_source), int(source_rows[0]), int(source_rows[-1])))


def _finite_stats(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float32)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"min": 0.0, "p01": 0.0, "p05": 0.0, "median": 0.0, "mean": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "span": 0.0}
    return {
        "min": float(np.min(v)),
        "p01": float(np.percentile(v, 1)),
        "p05": float(np.percentile(v, 5)),
        "median": float(np.median(v)),
        "mean": float(np.mean(v)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "max": float(np.max(v)),
        "span": float(np.max(v) - np.min(v)),
    }


def _algorithm1_row_params(views: dict[str, ViewData], y_front: int) -> RowSurfaceParams | None:
    """Estimate Algorithm-1 row parameters in the canonical/front row frame.

    a: semi-major axis from front width.
    b: semi-minor axis from 45-degree side projected widths.
    cz: row-wise depth-center shift from side/front centerline differences.
    """
    front = views["front"].stats
    left = views["left"].stats
    right = views["right"].stats
    if y_front < 0 or y_front >= front.width.shape[0] or front.width[y_front] <= 0:
        return None

    y_left = _map_row(left.valid_rows, front.valid_rows, y_front)
    y_right = _map_row(right.valid_rows, front.valid_rows, y_front)
    if y_left < 0 or y_right < 0:
        return None
    if left.width[y_left] <= 0 or right.width[y_right] <= 0:
        return None

    a = max(float(front.width[y_front]) * 0.5, 1.0)
    d_left = float(left.width[y_left]) * 0.5
    d_right = float(right.width[y_right]) * 0.5
    b_left = math.sqrt(max(2.0 * d_left * d_left - a * a, 0.0))
    b_right = math.sqrt(max(2.0 * d_right * d_right - a * a, 0.0))
    b = 0.5 * (b_left + b_right)

    # Correct row-relative center-depth shift. Never use absolute center sums.
    shift_right = float(right.center[y_right] - front.center[y_front])
    shift_left = float(front.center[y_front] - left.center[y_left])
    cz = SQRT2 * 0.5 * (shift_right + shift_left)

    return RowSurfaceParams(y_front=y_front, y_left=y_left, y_right=y_right, a=a, b=b, cz=cz)


def _surface_samples_for_row(params: RowSurfaceParams, samples_per_pixel: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create canonical 3D samples for one Algorithm-1 ellipse row.

    Returns x, z, branch arrays. branch is +1 for upper/front surface and -1 for
    lower/rear surface. The visibility renderer decides which branch is visible
    in each camera; the unwarper never stitches these branches into one chart.
    """
    n = max(int(round(2.0 * params.a * samples_per_pixel)) + 1, 8)
    x = np.linspace(-params.a, params.a, n, dtype=np.float32)
    root = np.sqrt(np.maximum(1.0 - (x / max(params.a, 1e-6)) ** 2, 0.0)).astype(np.float32)
    z_radius = np.float32(params.b) * root
    z_upper = np.float32(params.cz) + z_radius
    z_lower = np.float32(params.cz) - z_radius
    x_all = np.concatenate([x, x]).astype(np.float32)
    z_all = np.concatenate([z_upper, z_lower]).astype(np.float32)
    branch = np.concatenate([np.ones_like(x, dtype=np.int8), -np.ones_like(x, dtype=np.int8)])
    return x_all, z_all, branch


def _rotate_xz_to_view(x: np.ndarray, z: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    x_view = x * c + z * s
    z_view = -x * s + z * c
    return x_view.astype(np.float32), z_view.astype(np.float32)


def _target_row_for_view(views: dict[str, ViewData], role: str, y_front: int) -> int:
    if role == "front":
        return y_front
    return _map_row(views[role].stats.valid_rows, views["front"].stats.valid_rows, y_front)


def _render_view_depth(
    views: dict[str, ViewData],
    role: str,
    angle_deg: float,
    samples_per_pixel: float = 2.0,
    smooth_depth: bool = True,
) -> dict[str, np.ndarray | dict[str, Any]]:
    """Project the canonical Algorithm-1 surface to one view and z-buffer it.

    Orthographic camera convention: larger z_view is closer/visible. This makes
    the upper/front surface visible in the front view and lets side views choose
    the visible branch by actual camera projection instead of chart heuristics.
    """
    view = views[role]
    h, w = view.mask.shape
    depth = np.full((h, w), np.nan, dtype=np.float32)
    branch_map = np.zeros((h, w), dtype=np.int8)
    canonical_x = np.full((h, w), np.nan, dtype=np.float32)
    canonical_z = np.full((h, w), np.nan, dtype=np.float32)
    source_y_front = np.full((h, w), np.nan, dtype=np.float32)
    sample_hits = np.zeros((h, w), dtype=np.uint16)

    front_rows = views["front"].stats.valid_rows
    row_param_values: list[list[float]] = []
    front_h = views["front"].mask.shape[0]
    row_a = np.full(front_h, np.nan, dtype=np.float32)
    row_b = np.full(front_h, np.nan, dtype=np.float32)
    row_cz = np.full(front_h, np.nan, dtype=np.float32)

    for y_front in front_rows:
        params = _algorithm1_row_params(views, int(y_front))
        if params is None:
            continue
        row_a[int(y_front)] = float(params.a)
        row_b[int(y_front)] = float(params.b)
        row_cz[int(y_front)] = float(params.cz)
        y_target = _target_row_for_view(views, role, int(y_front))
        if y_target < 0 or y_target >= h or view.stats.width[y_target] <= 0:
            continue

        x_canon, z_canon, branch = _surface_samples_for_row(params, samples_per_pixel=samples_per_pixel)
        x_view, z_view = _rotate_xz_to_view(x_canon, z_canon, angle_deg)

        # Align the projected row span to the measured view center. This keeps
        # Algorithm-1 silhouette fitting separate from visibility/depth rendering.
        projected_center = 0.5 * (float(np.nanmin(x_view)) + float(np.nanmax(x_view)))
        u_float = float(view.stats.center[y_target]) + (x_view - projected_center)

        # Bin by pixel column, keep nearest/visible sample (largest z_view).
        row_bins: dict[int, tuple[float, float, float, int]] = {}
        for u, zv, xc, zc, br in zip(u_float, z_view, x_canon, z_canon, branch):
            col = int(round(float(u)))
            if col < 0 or col >= w:
                continue
            if view.mask[y_target, col] <= 0:
                continue
            current = row_bins.get(col)
            if current is None or float(zv) > current[0]:
                row_bins[col] = (float(zv), float(xc), float(zc), int(br))

        if len(row_bins) < 2:
            continue

        cols = np.asarray(sorted(row_bins), dtype=np.int32)
        z_vals = np.asarray([row_bins[int(c)][0] for c in cols], dtype=np.float32)
        x_vals = np.asarray([row_bins[int(c)][1] for c in cols], dtype=np.float32)
        zc_vals = np.asarray([row_bins[int(c)][2] for c in cols], dtype=np.float32)
        br_vals = np.asarray([row_bins[int(c)][3] for c in cols], dtype=np.int8)

        # Interpolate to all observed columns inside this row's projected range.
        obs_cols = np.flatnonzero(view.mask[y_target] > 0)
        obs_cols = obs_cols[(obs_cols >= cols[0]) & (obs_cols <= cols[-1])]
        if obs_cols.size < 2:
            continue

        z_interp = np.interp(obs_cols.astype(np.float32), cols.astype(np.float32), z_vals).astype(np.float32)
        x_interp = np.interp(obs_cols.astype(np.float32), cols.astype(np.float32), x_vals).astype(np.float32)
        zc_interp = np.interp(obs_cols.astype(np.float32), cols.astype(np.float32), zc_vals).astype(np.float32)
        br_interp = np.interp(obs_cols.astype(np.float32), cols.astype(np.float32), br_vals.astype(np.float32))
        br_interp = np.where(br_interp >= 0.0, 1, -1).astype(np.int8)

        # Row interpolation already applied z-buffer before interpolation. If a
        # different front row maps to the same target row, retain nearest depth.
        old = depth[y_target, obs_cols]
        replace = ~np.isfinite(old) | (z_interp > old)
        replace_cols = obs_cols[replace]
        depth[y_target, replace_cols] = z_interp[replace]
        canonical_x[y_target, replace_cols] = x_interp[replace]
        canonical_z[y_target, replace_cols] = zc_interp[replace]
        source_y_front[y_target, replace_cols] = float(y_front)
        branch_map[y_target, replace_cols] = br_interp[replace]
        sample_hits[y_target, replace_cols] += 1

        row_param_values.append([params.a, params.b, params.cz])

    # Raw z-buffer support can contain holes and visibility discontinuities.
    # Do not differentiate that sparse/z-buffer field directly. First create a
    # continuous depth chart over the observed fingerprint mask, then compute
    # normal image gradients and smooth/extrapolate the gradient field.
    zbuffer_support = np.isfinite(depth) & (view.mask > 0)
    observed_support = view.mask > 0
    depth_filled = _extrapolate_scalar_field(depth, zbuffer_support, observed_support)
    if smooth_depth and np.any(observed_support):
        depth_filled = _masked_gaussian_smooth(depth_filled, observed_support, sigma_x=3.0, sigma_y=3.0)

    if role == "front":
        occlusion_safe_support = observed_support.copy()
    else:
        occlusion_safe_support = _occlusion_safe_mask_from_canonical_maps(
            canonical_x=canonical_x,
            canonical_z=canonical_z,
            row_a=row_a,
            row_b=row_b,
            row_cz=row_cz,
            source_y_front=source_y_front,
            base_mask=zbuffer_support,
            angle_deg=angle_deg,
            visibility_margin=0.03,
        )

    # IMPORTANT: Algorithm 3 integrates gx/gy along whole rows and columns from
    # a single zero point. If gx/gy only exist inside the occlusion-safe mask,
    # many integration paths break and the output collapses to a narrow strip.
    # Therefore gradients are kept finite over the full observed finger mask,
    # while occlusion_safe_support is used only as the source/painting mask for
    # Algorithm-3 unwarping (Option A: do not paint occluded pixels).
    # Compute gx/gy on the full observed mask, but seed them only from the
    # occlusion-safe interior. This removes the high-gradient roll-out band from
    # the gradient field itself, then extrapolates a smooth field across the full
    # observed mask so Algorithm-3 row/column integrations stay continuous.
    gx, gy, full_gradient_support = _smooth_depth_gradients(
        depth_filled,
        target_mask=observed_support,
        seed_mask=occlusion_safe_support,
        depth_gradient_sigma=1.5,
        gradient_sigma=5.0,
        erode_iterations=2,
        seed_gradient_percentile=95.0,
    )
    # Option A painting/source mask: do not paint pixels near/beyond occlusion.
    gradient_support = occlusion_safe_support
    report = {
        "role": role,
        "angle_deg": float(angle_deg),
        "zbuffer_support_pixels": int(np.count_nonzero(zbuffer_support)),
        "occlusion_safe_support_pixels": int(np.count_nonzero(occlusion_safe_support)),
        "full_gradient_domain_pixels": int(np.count_nonzero(full_gradient_support)),
        "algorithm3_source_support_pixels": int(np.count_nonzero(gradient_support)),
        "observed_mask_pixels": int(np.count_nonzero(observed_support)),
        "zbuffer_support_over_observed_mask": float(np.count_nonzero(zbuffer_support) / max(np.count_nonzero(observed_support), 1)),
        "occlusion_safe_support_over_observed_mask": float(np.count_nonzero(occlusion_safe_support) / max(np.count_nonzero(observed_support), 1)),
        "full_gradient_domain_over_observed_mask": float(np.count_nonzero(full_gradient_support) / max(np.count_nonzero(observed_support), 1)),
        "algorithm3_source_support_over_observed_mask": float(np.count_nonzero(gradient_support) / max(np.count_nonzero(observed_support), 1)),
        "depth_stats": _finite_stats(depth_filled[observed_support]),
        "gradient_x_stats_on_source_support": _finite_stats(gx[gradient_support]),
        "gradient_y_stats_on_source_support": _finite_stats(gy[gradient_support]),
        "gradient_x_stats_on_full_domain": _finite_stats(gx[full_gradient_support]),
        "gradient_y_stats_on_full_domain": _finite_stats(gy[full_gradient_support]),
        "row_param_stats": {
            "a": _finite_stats(np.asarray([v[0] for v in row_param_values], dtype=np.float32)),
            "b": _finite_stats(np.asarray([v[1] for v in row_param_values], dtype=np.float32)),
            "cz": _finite_stats(np.asarray([v[2] for v in row_param_values], dtype=np.float32)),
        },
    }
    return {
        "depth": depth_filled.astype(np.float32),
        "support": gradient_support.astype(np.uint8),
        "full_gradient_support": full_gradient_support.astype(np.uint8),
        "zbuffer_support": zbuffer_support.astype(np.uint8),
        "occlusion_safe_support": occlusion_safe_support.astype(np.uint8),
        "gx": gx.astype(np.float32),
        "gy": gy.astype(np.float32),
        "branch": branch_map.astype(np.int8),
        "canonical_x": canonical_x.astype(np.float32),
        "canonical_z": canonical_z.astype(np.float32),
        "source_y_front": source_y_front.astype(np.float32),
        "sample_hits": sample_hits.astype(np.uint16),
        "row_a": row_a.astype(np.float32),
        "row_b": row_b.astype(np.float32),
        "row_cz": row_cz.astype(np.float32),
        "report": report,
    }


def _extrapolate_scalar_field(values: np.ndarray, valid_mask: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    """Fill/extrapolate a scalar field inside target_mask from valid samples.

    This is used before differentiating depth and again after computing gradients.
    It avoids taking finite differences across NaNs, holes, or z-buffer gaps.
    """
    arr = np.asarray(values, dtype=np.float32)
    target = target_mask.astype(bool)
    valid = valid_mask.astype(bool) & target & np.isfinite(arr)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    if not np.any(target):
        return out
    if not np.any(valid):
        out[target] = 0.0
        return out

    # Row-wise interpolation first, because the finger chart is mostly row-like.
    out[valid] = arr[valid]
    h, _w = arr.shape
    for y in range(h):
        target_cols = np.flatnonzero(target[y])
        valid_cols = np.flatnonzero(valid[y])
        if target_cols.size == 0 or valid_cols.size < 2:
            continue
        cols = target_cols[(target_cols >= valid_cols[0]) & (target_cols <= valid_cols[-1])]
        if cols.size:
            out[y, cols] = np.interp(cols, valid_cols, arr[y, valid_cols]).astype(np.float32)

    current_valid = np.isfinite(out) & target
    missing = target & ~current_valid
    if np.any(missing):
        # Nearest-neighbor extrapolation for remaining holes and side gaps.
        from scipy import ndimage
        _dist, indices = ndimage.distance_transform_edt(~current_valid, return_indices=True)
        out[missing] = out[tuple(ind[missing] for ind in indices)]

    out[~target] = np.nan
    return out.astype(np.float32)


def _masked_gaussian_smooth(values: np.ndarray, mask: np.ndarray, sigma_x: float, sigma_y: float) -> np.ndarray:
    """Gaussian smooth inside a mask without bleeding zeros from background."""
    values = np.asarray(values, dtype=np.float32)
    valid = mask.astype(bool) & np.isfinite(values)
    out = values.copy()
    if not np.any(valid):
        return out
    num = np.where(valid, values, 0.0).astype(np.float32)
    den = valid.astype(np.float32)
    num_blur = cv2.GaussianBlur(num, (0, 0), sigmaX=float(sigma_x), sigmaY=float(sigma_y))
    den_blur = cv2.GaussianBlur(den, (0, 0), sigmaX=float(sigma_x), sigmaY=float(sigma_y))
    smoothed = np.divide(num_blur, np.maximum(den_blur, 1e-6)).astype(np.float32)
    out[valid] = smoothed[valid]
    out[~mask.astype(bool)] = np.nan
    return out


def _occlusion_safe_mask_from_canonical_maps(
    canonical_x: np.ndarray,
    canonical_z: np.ndarray,
    row_a: np.ndarray,
    row_b: np.ndarray,
    row_cz: np.ndarray,
    source_y_front: np.ndarray,
    base_mask: np.ndarray,
    angle_deg: float,
    visibility_margin: float = 0.03,
) -> np.ndarray:
    """Keep only pixels safely inside the visible side surface.

    Option A: do not use or unwrap pixels near/beyond the occlusion boundary.
    The boundary is estimated from the canonical ellipse normal and side-view
    direction. Points with n·view <= margin are treated as silhouette / roll-out
    regions and are excluded from the support mask.
    """
    safe = np.zeros(base_mask.shape, dtype=bool)
    valid = (
        base_mask.astype(bool)
        & np.isfinite(canonical_x)
        & np.isfinite(canonical_z)
        & np.isfinite(source_y_front)
    )
    if not np.any(valid):
        return safe

    theta = math.radians(float(angle_deg))
    view_x = -math.sin(theta)
    view_z = math.cos(theta)

    ys, xs = np.nonzero(valid)
    for y, x in zip(ys, xs):
        yf = int(round(float(source_y_front[y, x])))
        if yf < 0 or yf >= row_a.shape[0]:
            continue
        a = float(row_a[yf])
        b = float(row_b[yf])
        cz = float(row_cz[yf])
        if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(cz)):
            continue
        if a <= 1e-6 or b <= 1e-6:
            continue

        xc = float(canonical_x[y, x])
        zc = float(canonical_z[y, x])

        nx = xc / (a * a)
        nz = (zc - cz) / (b * b)
        norm = math.sqrt(nx * nx + nz * nz)
        if norm <= 1e-8:
            continue
        nx /= norm
        nz /= norm
        visibility_score = nx * view_x + nz * view_z
        if visibility_score > float(visibility_margin):
            safe[y, x] = True

    return safe


def _smooth_depth_gradients(
    depth: np.ndarray,
    target_mask: np.ndarray,
    seed_mask: np.ndarray | None = None,
    depth_gradient_sigma: float = 0.0,
    gradient_sigma: float = 2.0,
    erode_iterations: int = 2,
    seed_gradient_percentile: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a smooth, extrapolated gradient field from a continuous depth map.

    This replaces the old sparse z-buffer finite-difference gradients. The depth
    is already extrapolated over target_mask. We optionally ignore the unstable
    silhouette band while estimating gradients, smooth the interior gradients,
    and then extrapolate the missing gradient values back over the whole mask.
    """
    mask = target_mask.astype(bool)
    d = np.asarray(depth, dtype=np.float32)
    if not np.any(mask):
        nan = np.full_like(d, np.nan, dtype=np.float32)
        return nan, nan, mask

    d_work = d.copy()
    if depth_gradient_sigma > 0:
        d_work = _masked_gaussian_smooth(d_work, mask, depth_gradient_sigma, depth_gradient_sigma)

    # Fill outside mask with nearest valid values so Sobel/gradient does not see
    # a huge artificial jump at the mask boundary.
    d_for_grad = _extrapolate_scalar_field(d_work, np.isfinite(d_work) & mask, np.ones(mask.shape, dtype=bool))

    gx = cv2.Sobel(d_for_grad, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0)
    gy = cv2.Sobel(d_for_grad, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0)

    interior = mask.copy()
    if erode_iterations > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        interior = cv2.erode(mask.astype(np.uint8), kernel, iterations=int(erode_iterations)).astype(bool)
        if not np.any(interior):
            interior = mask

    if seed_mask is not None:
        seed = interior & seed_mask.astype(bool) & np.isfinite(gx) & np.isfinite(gy)
        if np.count_nonzero(seed) < 32:
            seed = interior & np.isfinite(gx) & np.isfinite(gy)
    else:
        seed = interior & np.isfinite(gx) & np.isfinite(gy)

    # Do not let remaining high-gradient edge/seam pixels seed the field.
    # They are exactly what creates Algorithm-3 fan-out because sqrt(1+g^2)
    # turns them into huge arc-length increments.
    if seed_gradient_percentile is not None and np.count_nonzero(seed) >= 32:
        grad_mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
        thresh = float(np.percentile(grad_mag[seed], float(seed_gradient_percentile)))
        seed = seed & (grad_mag <= thresh)
        if np.count_nonzero(seed) < 32:
            seed = interior & seed_mask.astype(bool) & np.isfinite(gx) & np.isfinite(gy) if seed_mask is not None else interior & np.isfinite(gx) & np.isfinite(gy)

    gx_seeded = np.where(seed, gx, np.nan).astype(np.float32)
    gy_seeded = np.where(seed, gy, np.nan).astype(np.float32)
    gx_full = _extrapolate_scalar_field(gx_seeded, seed, mask)
    gy_full = _extrapolate_scalar_field(gy_seeded, seed, mask)

    if gradient_sigma > 0:
        gx_full = _masked_gaussian_smooth(gx_full, mask, gradient_sigma, gradient_sigma)
        gy_full = _masked_gaussian_smooth(gy_full, mask, gradient_sigma, gradient_sigma)

    gx_full[~mask] = np.nan
    gy_full[~mask] = np.nan
    return gx_full.astype(np.float32), gy_full.astype(np.float32), mask

def _normalize_for_png(values: np.ndarray, mask: np.ndarray | None = None, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(arr)
    if mask is not None:
        valid &= mask.astype(bool)
    if not np.any(valid):
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.percentile(arr[valid], p_low))
    hi = float(np.percentile(arr[valid], p_high))
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    if mask is not None:
        out[~mask.astype(bool)] = 0.0
    return (out * 255.0).astype(np.uint8)


def _depth_color(depth: np.ndarray, support: np.ndarray) -> np.ndarray:
    norm = _normalize_for_png(depth, support)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[~support.astype(bool)] = 0
    return color






def contactless_fingerprint_unwarp_algorithm3(
    image: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    mask: np.ndarray | None = None,
    output_shape: tuple[int, int] | None = None,
    border_value: int | tuple[int, int, int] = 0,
) -> dict[str, Any]:
    """Algorithm 3: unwarp a contactless fingerprint from surface gradients.

    This function intentionally takes gradients as input. It does not consume
    depth and it does not recompute gradients from depth internally.

    Steps implemented:
      1. Find the zero point (cx, cy) with minimum gradient magnitude.
      2. Compute horizontal arc length u from cx using sqrt(1 + gx^2).
      3. Compute vertical arc length v from cy using sqrt(1 + gy^2).
      4. Forward map (x, y) to (u + cx, v + cy).
      5. Bilinear splat source pixels into the output image.
    """
    if image.ndim not in (2, 3):
        raise ValueError('image must be grayscale HxW or color HxWxC')

    h, w = image.shape[:2]
    if gx.shape != (h, w):
        raise ValueError(f'gx must have shape {(h, w)}, got {gx.shape}')
    if gy.shape != (h, w):
        raise ValueError(f'gy must have shape {(h, w)}, got {gy.shape}')

    gx = gx.astype(np.float32, copy=False)
    gy = gy.astype(np.float32, copy=False)
    finite = np.isfinite(gx) & np.isfinite(gy)
    valid_input = finite if mask is None else finite & mask.astype(bool)
    if not np.any(valid_input):
        raise ValueError('No valid pixels available for Algorithm-3 unwarping')

    if output_shape is None:
        out_h, out_w = h, w
    else:
        out_h, out_w = int(output_shape[0]), int(output_shape[1])

    # Algorithm 3 line 1: zero point / minimum gradient point.
    grad_mag = np.sqrt(gx * gx + gy * gy)
    grad_mag_search = np.where(valid_input, grad_mag, np.inf)
    cy, cx = np.unravel_index(np.argmin(grad_mag_search), grad_mag_search.shape)

    # Algorithm 3 lines 2-4: horizontal arc-length coordinate u.
    # Discrete left/right integration uses the pixel interval immediately crossed.
    horizontal_arc_increment = np.sqrt(1.0 + gx * gx).astype(np.float32)
    u = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        if cx + 1 < w:
            u[y, cx + 1:] = np.cumsum(horizontal_arc_increment[y, cx:w - 1])
        if cx > 0:
            left_steps = horizontal_arc_increment[y, 1:cx + 1][::-1]
            u[y, :cx] = -np.cumsum(left_steps)[::-1]

    # Algorithm 3 lines 5-6: vertical arc-length coordinate v.
    vertical_arc_increment = np.sqrt(1.0 + gy * gy).astype(np.float32)
    v = np.zeros((h, w), dtype=np.float32)
    for x in range(w):
        if cy + 1 < h:
            v[cy + 1:, x] = np.cumsum(vertical_arc_increment[cy:h - 1, x])
        if cy > 0:
            up_steps = vertical_arc_increment[1:cy + 1, x][::-1]
            v[:cy, x] = -np.cumsum(up_steps)[::-1]

    # Algorithm 3 line 7: forward coordinate correspondence.
    map_x = u + float(cx)
    map_y = v + float(cy)

    if image.ndim == 2:
        out = np.full((out_h, out_w), border_value, dtype=np.float32)
        accum = np.zeros((out_h, out_w), dtype=np.float32)
    else:
        channels = image.shape[2]
        if isinstance(border_value, tuple):
            fill_value = np.array(border_value, dtype=np.float32)
        else:
            fill_value = np.full((channels,), border_value, dtype=np.float32)
        out = np.zeros((out_h, out_w, channels), dtype=np.float32)
        out[:, :] = fill_value
        accum = np.zeros((out_h, out_w, channels), dtype=np.float32)
    weight_sum = np.zeros((out_h, out_w), dtype=np.float32)
    img_float = image.astype(np.float32)

    ys, xs = np.nonzero(valid_input)
    for y, x in zip(ys, xs):
        xo = float(map_x[y, x])
        yo = float(map_y[y, x])
        if not np.isfinite(xo) or not np.isfinite(yo):
            continue
        x0 = int(np.floor(xo))
        y0 = int(np.floor(yo))
        wx = xo - x0
        wy = yo - y0
        candidates = (
            (x0, y0, (1.0 - wx) * (1.0 - wy)),
            (x0 + 1, y0, wx * (1.0 - wy)),
            (x0, y0 + 1, (1.0 - wx) * wy),
            (x0 + 1, y0 + 1, wx * wy),
        )
        for xx, yy, ww in candidates:
            if 0 <= xx < out_w and 0 <= yy < out_h and ww > 0.0:
                if image.ndim == 2:
                    accum[yy, xx] += ww * img_float[y, x]
                else:
                    accum[yy, xx, :] += ww * img_float[y, x, :]
                weight_sum[yy, xx] += ww

    filled = weight_sum > 1e-8
    if image.ndim == 2:
        out[filled] = accum[filled] / weight_sum[filled]
    else:
        out[filled, :] = accum[filled, :] / weight_sum[filled, None]

    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        out = np.clip(out, info.min, info.max).astype(image.dtype)
    else:
        out = out.astype(image.dtype)

    return {
        'unwrapped': out,
        'cx': int(cx),
        'cy': int(cy),
        'u': u,
        'v': v,
        'map_x': map_x.astype(np.float32),
        'map_y': map_y.astype(np.float32),
        'valid': valid_input.astype(np.uint8),
        'filled': filled.astype(np.uint8),
        'report': {
            'algorithm': 'Algorithm 3 Contactless Fingerprint Unwarping',
            'zero_point_xy': [int(cx), int(cy)],
            'output_shape_hw': [int(out_h), int(out_w)],
            'valid_input_pixels': int(np.count_nonzero(valid_input)),
            'filled_output_pixels': int(np.count_nonzero(filled)),
            'filled_over_valid_input': float(np.count_nonzero(filled) / max(np.count_nonzero(valid_input), 1)),
        },
    }



def _fill_missing_output_rows(
    source_x: np.ndarray,
    source_y: np.ndarray,
    expected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Linearly fill fully-missing rows between valid rows in the output chart."""
    out_x = source_x.copy()
    out_y = source_y.copy()
    out_expected = expected.copy()
    valid_rows = np.flatnonzero(np.any(out_expected, axis=1))
    if valid_rows.size < 2:
        return out_x, out_y, out_expected, 0
    filled_count = 0
    for y0, y1 in zip(valid_rows[:-1], valid_rows[1:]):
        if y1 <= y0 + 1:
            continue
        gap = y1 - y0
        overlap = out_expected[y0] & out_expected[y1]
        if not np.any(overlap):
            continue
        for yy in range(y0 + 1, y1):
            t = float(yy - y0) / float(gap)
            out_x[yy, overlap] = (1.0 - t) * out_x[y0, overlap] + t * out_x[y1, overlap]
            out_y[yy, overlap] = (1.0 - t) * out_y[y0, overlap] + t * out_y[y1, overlap]
            out_expected[yy, overlap] = True
            filled_count += 1
    return out_x, out_y, out_expected, filled_count



def _largest_true_run_circular(flags: np.ndarray) -> tuple[int, int] | None:
    """Return start/end (end exclusive) for largest circular True run in flags."""
    f = np.asarray(flags, dtype=bool)
    n = int(f.size)
    if n == 0 or not np.any(f):
        return None
    if np.all(f):
        return 0, n
    ff = np.concatenate([f, f])
    best_start = -1
    best_len = 0
    i = 0
    while i < 2 * n:
        if not ff[i]:
            i += 1
            continue
        j = i
        while j < 2 * n and ff[j] and (j - i) < n:
            j += 1
        run_len = j - i
        if run_len > best_len and i < n:
            best_start = i
            best_len = run_len
        i = max(j, i + 1)
    if best_start < 0 or best_len < 2:
        return None
    return best_start, best_start + best_len


def _stitch_visible_branch_samples(
    a: float,
    b: float,
    cz: float,
    angle_deg: float,
    samples: int,
    visibility_margin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Sample the single visible Algorithm-1 ellipse branch for one row.

    This replaces z-buffered atan2 sorting. A dense canonical ellipse is sampled,
    the camera-facing half is selected by the surface-normal visibility test,
    and only the largest continuous visible interval is returned.
    """
    samples = max(int(samples), 128)
    t = np.linspace(-math.pi, math.pi, samples, endpoint=False, dtype=np.float32)
    x = (float(a) * np.cos(t)).astype(np.float32)
    z = (float(cz) + float(b) * np.sin(t)).astype(np.float32)

    alpha = math.radians(float(angle_deg))
    view_x = -math.sin(alpha)
    view_z = math.cos(alpha)

    # Ellipse normal proportional to gradient of x^2/a^2 + (z-cz)^2/b^2.
    nx = x / max(float(a) * float(a), 1e-6)
    nz = (z - float(cz)) / max(float(b) * float(b), 1e-6)
    norm = np.sqrt(nx * nx + nz * nz).astype(np.float32)
    norm = np.maximum(norm, 1e-8)
    nx = nx / norm
    nz = nz / norm
    visibility = nx * float(view_x) + nz * float(view_z)
    visible = visibility > float(visibility_margin)

    run = _largest_true_run_circular(visible)
    if run is None:
        return np.empty(0, np.float32), np.empty(0, np.float32), np.empty(0, np.float32), np.empty(0, np.float32), 0.0

    start, end = run
    idx = np.arange(start, end, dtype=np.int32) % samples
    # Append one extra endpoint just beyond the run when possible for stable interpolation.
    # The endpoint is still clipped to the visible run; no branch wrap stitching is done.
    x_run = x[idx]
    z_run = z[idx]
    t_run = t[idx]
    vis_run = visibility[idx]

    if x_run.size < 2:
        return np.empty(0, np.float32), np.empty(0, np.float32), np.empty(0, np.float32), np.empty(0, np.float32), 0.0

    ds = np.sqrt(np.diff(x_run) ** 2 + np.diff(z_run) ** 2).astype(np.float32)
    s_run = np.concatenate([[0.0], np.cumsum(ds)]).astype(np.float32)
    return x_run, z_run, t_run, s_run, float(s_run[-1])


def _canonical_stitched_branch_unwrap(
    image: np.ndarray,
    observed_mask: np.ndarray,
    view_stats: RowStats,
    front_valid_rows: np.ndarray,
    row_a: np.ndarray,
    row_b: np.ndarray,
    row_cz: np.ndarray,
    angle_deg: float,
    output_shape: tuple[int, int] | None = None,
    samples_per_row: int = 1600,
    visibility_margin: float = 0.015,
    smooth_map_sigma_x: float = 1.25,
    smooth_map_sigma_y: float = 0.75,
) -> dict[str, Any]:
    """Unwrap a side view with a global centered arc coordinate.

    Option 1 from the discussion: each row keeps its natural arc-length span,
    centered by branch midpoint. There is no per-row normalization to full width.

    For each front/canonical row:
      1. Build the single camera-facing branch directly from Algorithm-1 ellipse geometry.
      2. Project that branch into the side image.
      3. Truncate to the longest segment that is actually inside the observed side mask.
      4. Place that segment into one global output-x coordinate frame using
             x_out = center_x + (s - 0.5 * row_arc_len)
         so shorter fingertip rows naturally occupy fewer output columns.
    """
    h, w = image.shape[:2]
    mask = observed_mask.astype(bool)
    front_valid_rows = np.asarray(front_valid_rows, dtype=np.int32)
    dense_row_map = {int(yf): idx for idx, yf in enumerate(front_valid_rows.tolist())}

    row_records: list[dict[str, Any]] = []
    max_half_arc = 0.0
    max_arc = 0.0
    max_points = 0
    rejected_rows = 0

    for y_front in front_valid_rows:
        yf = int(y_front)
        if yf < 0 or yf >= int(row_a.shape[0]):
            continue
        a = float(row_a[yf])
        b = float(row_b[yf])
        cz = float(row_cz[yf])
        if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(cz)) or a <= 1e-6 or b <= 1e-6:
            rejected_rows += 1
            continue

        y_src = _map_row(view_stats.valid_rows, front_valid_rows, yf)
        if y_src < 0 or y_src >= h or view_stats.width[y_src] <= 0:
            rejected_rows += 1
            continue

        x_branch, z_branch, _t_branch, s_branch, arc_len = _stitch_visible_branch_samples(
            a=a,
            b=b,
            cz=cz,
            angle_deg=angle_deg,
            samples=samples_per_row,
            visibility_margin=visibility_margin,
        )
        if x_branch.size < 4 or arc_len <= 0.0:
            rejected_rows += 1
            continue

        theta = math.radians(float(angle_deg))
        c = math.cos(theta)
        ss = math.sin(theta)
        x_view = (x_branch * c + z_branch * ss).astype(np.float32)

        # Match _render_view_depth's projected-center alignment.
        tt_full = np.linspace(-math.pi, math.pi, max(samples_per_row, 512), endpoint=False, dtype=np.float32)
        x_full = (a * np.cos(tt_full)).astype(np.float32)
        z_full = (cz + b * np.sin(tt_full)).astype(np.float32)
        x_view_full = (x_full * c + z_full * ss).astype(np.float32)
        projected_center = 0.5 * (float(np.nanmin(x_view_full)) + float(np.nanmax(x_view_full)))
        u_float = float(view_stats.center[y_src]) + (x_view - projected_center)

        # Keep only the single longest observed segment, but do not reorder it.
        cols_round = np.round(u_float).astype(np.int32)
        in_img = (cols_round >= 0) & (cols_round < w)
        in_mask = np.zeros_like(in_img, dtype=bool)
        ok_cols = cols_round[in_img]
        if ok_cols.size:
            in_mask[in_img] = mask[y_src, ok_cols]
        if np.count_nonzero(in_mask) < 4:
            rejected_rows += 1
            continue

        best_i = best_j = -1
        best_len = 0
        i = 0
        n = int(in_mask.size)
        while i < n:
            if not in_mask[i]:
                i += 1
                continue
            j = i
            while j < n and in_mask[j]:
                j += 1
            if j - i > best_len:
                best_i, best_j, best_len = i, j, j - i
            i = j
        if best_len < 4:
            rejected_rows += 1
            continue

        x_src = u_float[best_i:best_j].astype(np.float32)
        s_src = s_branch[best_i:best_j].astype(np.float32)
        s_src = s_src - float(s_src[0])
        arc_len_seg = float(s_src[-1])
        if not math.isfinite(arc_len_seg) or arc_len_seg <= 0.0:
            rejected_rows += 1
            continue

        # Option 1: center by branch midpoint.
        center_s = 0.5 * arc_len_seg
        x_out_rel = s_src - center_s

        row_records.append({
            'y_out': int(dense_row_map[yf]),
            'y_front': yf,
            'y_src': y_src,
            'source_x': x_src,
            's': s_src,
            'x_out_rel': x_out_rel.astype(np.float32),
            'arc_len': arc_len_seg,
            'center_s': center_s,
            'num_points': int(x_src.size),
        })
        max_half_arc = max(max_half_arc, 0.5 * arc_len_seg)
        max_arc = max(max_arc, arc_len_seg)
        max_points = max(max_points, int(x_src.size))

    if not row_records:
        raise ValueError('No valid row records available for midpoint-centered stitched-branch unwrapping')

    if output_shape is None:
        out_h = int(len(front_valid_rows))
        # +4 gives a little integer rounding margin on both sides.
        out_w = max(int(math.ceil(2.0 * max_half_arc)) + 5, 16)
    else:
        out_h, out_w = int(output_shape[0]), int(output_shape[1])

    center_x = 0.5 * float(out_w - 1)
    source_x = np.full((out_h, out_w), np.nan, dtype=np.float32)
    source_y = np.full((out_h, out_w), np.nan, dtype=np.float32)
    expected = np.zeros((out_h, out_w), dtype=bool)

    rows_written = 0
    for rec in row_records:
        y_out = int(np.clip(rec['y_out'], 0, out_h - 1))
        # Global output columns that fall within this row's natural arc span.
        x0 = int(max(0, math.ceil(center_x - float(rec['center_s']))))
        x1 = int(min(out_w - 1, math.floor(center_x + (float(rec['arc_len']) - float(rec['center_s'])))))
        if x1 <= x0:
            continue
        cols = np.arange(x0, x1 + 1, dtype=np.int32)
        target_s = (cols.astype(np.float32) - np.float32(center_x) + np.float32(rec['center_s'])).astype(np.float32)
        valid_cols = (target_s >= 0.0) & (target_s <= float(rec['arc_len']))
        if np.count_nonzero(valid_cols) < 2:
            continue
        cols = cols[valid_cols]
        target_s = target_s[valid_cols]
        remap_cols = np.interp(target_s, rec['s'], rec['source_x']).astype(np.float32)
        source_x[y_out, cols] = remap_cols
        source_y[y_out, cols] = float(rec['y_src'])
        expected[y_out, cols] = True
        rows_written += 1

    source_x, source_y, expected, filled_missing_rows = _fill_missing_output_rows(source_x, source_y, expected)

    if smooth_map_sigma_x > 0 and np.any(expected):
        source_x = _masked_gaussian_smooth(source_x, expected, smooth_map_sigma_x, smooth_map_sigma_y)
        source_y = _masked_gaussian_smooth(source_y, expected, smooth_map_sigma_x, smooth_map_sigma_y)

    remap_x = np.where(expected, source_x, 0.0).astype(np.float32)
    remap_y = np.where(expected, source_y, 0.0).astype(np.float32)
    unwrapped = cv2.remap(image, remap_x, remap_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sampled_support = cv2.remap((mask.astype(np.uint8) * 255), remap_x, remap_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    filled = (expected & (sampled_support > 127))
    unwrapped = np.where(filled, unwrapped, 0).astype(image.dtype)

    return {
        'unwrapped': unwrapped,
        'valid': mask.astype(np.uint8),
        'filled': filled.astype(np.uint8),
        'expected': expected.astype(np.uint8),
        'source_x': source_x.astype(np.float32),
        'source_y': source_y.astype(np.float32),
        'report': {
            'algorithm': 'Algorithm-1 stitched visible branch unwrap with midpoint-centered global arc coordinates',
            'output_shape_hw': [int(out_h), int(out_w)],
            'valid_input_pixels': int(np.count_nonzero(mask)),
            'filled_output_pixels': int(np.count_nonzero(filled)),
            'filled_over_valid_input': float(np.count_nonzero(filled) / max(np.count_nonzero(mask), 1)),
            'rows_written': int(rows_written),
            'rows_with_valid_chart': int(len(row_records)),
            'rows_rejected': int(rejected_rows),
            'rows_filled_by_interpolation': int(filled_missing_rows),
            'max_row_arc_length': float(max_arc),
            'max_half_row_arc_length': float(max_half_arc),
            'max_row_points': int(max_points),
            'visibility_margin': float(visibility_margin),
            'row_centering': 'branch_midpoint',
            'per_row_width_normalization': False,
        },
    }

def unwarp_views_from_canonical_surface_chart(
    views: dict[str, ViewData],
    rendered: dict[str, dict[str, Any]],
    output_dir: Path | None = None,
    roles: tuple[str, ...] = ('left', 'right'),
    output_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Run side-view unwrapping using the canonical Algorithm-1 surface chart."""
    unwrapped: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {
        'stage': 'canonical_surface_chart_unwarping_only',
        'roles': {},
        'notes': [
            'Unwrapping constructs the visible row branch directly from Algorithm-1 ellipse geometry.',
            'No z-buffered atan2 sorting is used; each row uses the single camera-facing stitched branch.',
            'Rows are centered by their branch midpoint in one global arc-length coordinate frame.',
            'No per-row fixed-width normalization is applied, so the output support remains naturally finger-shaped.',
            'The side image is sampled through an inverse map from canonical arc length to projected source x.',
        ],
    }
    for role in roles:
        result = _canonical_stitched_branch_unwrap(
            image=views[role].image,
            observed_mask=views[role].mask,
            view_stats=views[role].stats,
            front_valid_rows=views['front'].stats.valid_rows,
            row_a=rendered[role]['row_a'],
            row_b=rendered[role]['row_b'],
            row_cz=rendered[role]['row_cz'],
            angle_deg=float(rendered[role]['report']['angle_deg']),
            output_shape=output_shape,
        )
        unwrapped[role] = result
        report['roles'][role] = result['report']

        if output_dir is not None:
            role_dir = output_dir / role
            role_dir.mkdir(parents=True, exist_ok=True)
            _write_gray(role_dir / f'{role}_canonical_chart_unwrapped.png', result['unwrapped'])
            _write_gray(role_dir / f'{role}_canonical_chart_valid_input.png', result['valid'] * 255)
            _write_gray(role_dir / f'{role}_canonical_chart_filled_output.png', result['filled'] * 255)
            _write_gray(role_dir / f'{role}_canonical_chart_expected.png', result['expected'] * 255)
            np.savez_compressed(
                role_dir / f'{role}_canonical_chart_unwrap_maps.npz',
                unwrapped=result['unwrapped'],
                valid=result['valid'],
                filled=result['filled'],
                expected=result['expected'],
                source_x=result['source_x'],
                source_y=result['source_y'],
            )
    return {'unwrapped': unwrapped, 'report': report}


def reconstruct_algorithm1_depth_and_gradients(
    reconstruction_dir: Path,
    output_dir: Path | None = None,
    left_angle: float = DEFAULT_VIEW_ANGLES_DEG['left'],
    right_angle: float = DEFAULT_VIEW_ANGLES_DEG['right'],
    samples_per_pixel: float = 2.0,
    smooth_depth: bool = True,
) -> dict[str, Any]:
    """Run only the Algorithm-1-style depth reconstruction stage.

    Inputs are front/left/right pose-normalized images and masks under
    reconstruction_dir/debug_views. Outputs are per-view depth maps, support
    masks, and gradients gx/gy. No unwarping is performed here.
    """
    reconstruction_dir = reconstruction_dir.resolve()
    views = _load_views(reconstruction_dir)
    angles = {'front': 0.0, 'left': float(left_angle), 'right': float(right_angle)}
    rendered: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {
        'input_reconstruction_dir': str(reconstruction_dir),
        'stage': 'depth_reconstruction_only',
        'method': 'Algorithm 1 row-wise ellipse surface, fixed-view projection, continuous depth extrapolation, occlusion-safe gradient seeding, smoothed/extrapolated depth gradients',
        'view_angles_deg': angles,
        'samples_per_pixel': float(samples_per_pixel),
        'roles': {},
    }

    for role in ('front', 'left', 'right'):
        rendered[role] = _render_view_depth(
            views,
            role=role,
            angle_deg=angles[role],
            samples_per_pixel=samples_per_pixel,
            smooth_depth=smooth_depth,
        )
        rendered_report = dict(rendered[role]['report'])
        rendered_report['support_vs_observed'] = _mask_iou(rendered[role]['support'], views[role].mask > 0)
        report['roles'][role] = {'rendered_depth': rendered_report}

        if output_dir is not None:
            role_dir = output_dir / role
            role_dir.mkdir(parents=True, exist_ok=True)
            np.save(role_dir / f'{role}_depth.npy', rendered[role]['depth'])
            np.save(role_dir / f'{role}_gradient.npy', np.stack([rendered[role]['gx'], rendered[role]['gy']], axis=-1).astype(np.float32))
            np.savez_compressed(
                role_dir / f'{role}_surface_maps.npz',
                depth=rendered[role]['depth'],
                support=rendered[role]['support'],
                gx=rendered[role]['gx'],
                gy=rendered[role]['gy'],
                branch=rendered[role]['branch'],
                canonical_x=rendered[role]['canonical_x'],
                canonical_z=rendered[role]['canonical_z'],
                source_y_front=rendered[role]['source_y_front'],
                row_a=rendered[role]['row_a'],
                row_b=rendered[role]['row_b'],
                row_cz=rendered[role]['row_cz'],
                full_gradient_support=rendered[role].get('full_gradient_support'),
                occlusion_safe_support=rendered[role].get('occlusion_safe_support'),
            )
            _write_gray(role_dir / f'{role}_support_mask.png', rendered[role]['support'] * 255)
            if 'zbuffer_support' in rendered[role]:
                _write_gray(role_dir / f'{role}_zbuffer_support_mask.png', rendered[role]['zbuffer_support'] * 255)
            if 'full_gradient_support' in rendered[role]:
                _write_gray(role_dir / f'{role}_full_gradient_support_mask.png', rendered[role]['full_gradient_support'] * 255)
            if 'occlusion_safe_support' in rendered[role]:
                _write_gray(role_dir / f'{role}_occlusion_safe_support_mask.png', rendered[role]['occlusion_safe_support'] * 255)
            cv2.imwrite(str(role_dir / f'{role}_depth_color.png'), _depth_color(rendered[role]['depth'], rendered[role]['support']))
            _write_gray(role_dir / f'{role}_gradient_x.png', _normalize_for_png(rendered[role]['gx'], rendered[role]['support']))
            _write_gray(role_dir / f'{role}_gradient_y.png', _normalize_for_png(rendered[role]['gy'], rendered[role]['support']))

    return {'views': views, 'rendered': rendered, 'report': report}


def unwarp_views_from_gradients_algorithm3(
    views: dict[str, ViewData],
    rendered: dict[str, dict[str, Any]],
    output_dir: Path | None = None,
    roles: tuple[str, ...] = ('left', 'right'),
    output_shape: tuple[int, int] | None = None,
    border_value: int | tuple[int, int, int] = 0,
) -> dict[str, Any]:
    """Run only Algorithm 3 unwarping from already-computed gradients."""
    unwrapped: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {
        'stage': 'algorithm3_unwarping_only',
        'roles': {},
        'notes': [
            'Unwarping consumes gx and gy directly.',
            'Unwarping uses a full-domain smoothed gradient field from reconstruction plus the occlusion-safe source mask.',
            'Pixels near/beyond the estimated occlusion boundary are excluded from painting only; gradients remain full-domain so Algorithm-3 integration paths do not collapse.',
        ],
    }
    for role in roles:
        result = contactless_fingerprint_unwarp_algorithm3(
            image=views[role].image,
            gx=rendered[role]['gx'],
            gy=rendered[role]['gy'],
            mask=rendered[role]['support'],
            output_shape=output_shape,
            border_value=border_value,
        )
        unwrapped[role] = result
        report['roles'][role] = result['report']

        if output_dir is not None:
            role_dir = output_dir / role
            role_dir.mkdir(parents=True, exist_ok=True)
            _write_gray(role_dir / f'{role}_algorithm3_unwrapped.png', result['unwrapped'])
            _write_gray(role_dir / f'{role}_algorithm3_valid_input.png', result['valid'] * 255)
            _write_gray(role_dir / f'{role}_algorithm3_filled_output.png', result['filled'] * 255)
            np.savez_compressed(
                role_dir / f'{role}_algorithm3_unwarp_maps.npz',
                unwrapped=result['unwrapped'],
                valid=result['valid'],
                filled=result['filled'],
                u=result['u'],
                v=result['v'],
                map_x=result['map_x'],
                map_y=result['map_y'],
                cx=np.asarray(result['cx'], dtype=np.int32),
                cy=np.asarray(result['cy'], dtype=np.int32),
            )
    return {'unwrapped': unwrapped, 'report': report}


def _mask_iou(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    aa = a.astype(bool)
    bb = b.astype(bool)
    inter = int(np.count_nonzero(aa & bb))
    union = int(np.count_nonzero(aa | bb))
    return {
        'iou': float(inter / union) if union else 0.0,
        'intersection': inter,
        'union': union,
        'a_pixels': int(np.count_nonzero(aa)),
        'b_pixels': int(np.count_nonzero(bb)),
    }


def _draw_label(panel: np.ndarray, label: str) -> np.ndarray:
    if panel.ndim == 2:
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (18, 18, 18), -1)
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return out


def _fit_panel(image: np.ndarray, size: tuple[int, int] = (260, 210)) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    target_w, target_h = size
    h, w = image.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _write_contact_sheet(
    out_dir: Path,
    views: dict[str, ViewData],
    rendered: dict[str, dict[str, Any]],
    canonical_unwrapped: dict[str, dict[str, Any]],
    front_algorithm3_unwrapped: dict[str, dict[str, Any]] | None = None,
) -> Path:
    rows: list[np.ndarray] = []
    roles = ('front', 'left', 'right') if front_algorithm3_unwrapped is not None else ('left', 'right')
    for role in roles:
        depth_color = _depth_color(rendered[role]['depth'], rendered[role]['support'])
        gx_png = _normalize_for_png(rendered[role]['gx'], rendered[role]['support'])
        gy_png = _normalize_for_png(rendered[role]['gy'], rendered[role]['support'])
        if role == 'front' and front_algorithm3_unwrapped is not None:
            role_unwrapped = front_algorithm3_unwrapped[role]
            unwrap_label = 'front Algorithm-3 unwrap'
        else:
            role_unwrapped = canonical_unwrapped[role]
            unwrap_label = f'{role} midpoint-centered unwrap'
        panels = [
            _draw_label(_fit_panel(views[role].image), f'{role} pose image'),
            _draw_label(_fit_panel(views[role].mask), f'{role} observed mask'),
            _draw_label(_fit_panel(rendered[role].get('zbuffer_support', rendered[role]['support']) * 255), f'{role} z-buffer support'),
            _draw_label(_fit_panel(rendered[role].get('occlusion_safe_support', rendered[role]['support']) * 255), f'{role} occlusion-safe support'),
            _draw_label(_fit_panel(depth_color), f'{role} Algorithm-1 depth'),
            _draw_label(_fit_panel(gx_png), f'{role} gx'),
            _draw_label(_fit_panel(gy_png), f'{role} gy'),
            _draw_label(_fit_panel(role_unwrapped['unwrapped']), unwrap_label),
            _draw_label(_fit_panel(role_unwrapped['filled'] * 255), f'{role} filled output'),
        ]
        rows.append(np.hstack(panels))
    sheet = np.vstack(rows)
    path = out_dir / 'algorithm1_depth_then_stitched_branch_midpoint_unwarp_contact_sheet.png'
    cv2.imwrite(str(path), sheet)
    return path


def run(
    reconstruction_dir: Path,
    output_dir: Path,
    left_angle: float,
    right_angle: float,
    samples_per_pixel: float,
    unwarp_output_height: int | None = None,
    unwarp_output_width: int | None = None,
) -> dict[str, Any]:
    """Pipeline wrapper that keeps reconstruction and unwarping separated."""
    output_dir.mkdir(parents=True, exist_ok=True)

    recon_result = reconstruct_algorithm1_depth_and_gradients(
        reconstruction_dir=reconstruction_dir,
        output_dir=output_dir,
        left_angle=left_angle,
        right_angle=right_angle,
        samples_per_pixel=samples_per_pixel,
        smooth_depth=True,
    )

    output_shape = None
    if unwarp_output_height is not None or unwarp_output_width is not None:
        if unwarp_output_height is None or unwarp_output_width is None:
            raise ValueError('Set both --unwarp-output-height and --unwarp-output-width, or neither.')
        output_shape = (int(unwarp_output_height), int(unwarp_output_width))

    unwrap_result = unwarp_views_from_canonical_surface_chart(
        views=recon_result['views'],
        rendered=recon_result['rendered'],
        output_dir=output_dir,
        roles=('left', 'right'),
        output_shape=output_shape,
    )

    front_unwrap_result = unwarp_views_from_gradients_algorithm3(
        views=recon_result['views'],
        rendered=recon_result['rendered'],
        output_dir=output_dir,
        roles=('front',),
        output_shape=output_shape,
    )

    contact_sheet = _write_contact_sheet(
        output_dir,
        recon_result['views'],
        recon_result['rendered'],
        unwrap_result['unwrapped'],
        front_unwrap_result['unwrapped'],
    )
    report = {
        'input_reconstruction_dir': str(reconstruction_dir.resolve()),
        'output_dir': str(output_dir.resolve()),
        'method': 'Algorithm 1 depth/gradient reconstruction plus stitched visible branch midpoint-centered natural-shape unwrapping with front Algorithm-3 output',
        'depth_reconstruction': recon_result['report'],
        'canonical_chart_unwarping': unwrap_result['report'],
        'front_algorithm3_unwarping': front_unwrap_result['report'],
        'contact_sheet': str(contact_sheet.resolve()),
    }
    report_path = output_dir / 'algorithm1_depth_then_stitched_branch_midpoint_unwarp_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    report['report_path'] = str(report_path.resolve())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reconstruction-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--left-angle', type=float, default=DEFAULT_VIEW_ANGLES_DEG['left'], help='Left side camera angle in degrees in the canonical x-z frame.')
    parser.add_argument('--right-angle', type=float, default=DEFAULT_VIEW_ANGLES_DEG['right'], help='Right side camera angle in degrees in the canonical x-z frame.')
    parser.add_argument('--samples-per-pixel', type=float, default=2.0, help='Canonical ellipse samples per front-view pixel.')
    parser.add_argument('--unwarp-output-height', type=int, default=None, help='Optional canonical-chart output height. Default: front image height.')
    parser.add_argument('--unwarp-output-width', type=int, default=None, help='Optional canonical-chart output width. Default: derived from canonical row arc lengths.')
    args = parser.parse_args()

    report = run(
        reconstruction_dir=args.reconstruction_dir,
        output_dir=args.output_dir,
        left_angle=args.left_angle,
        right_angle=args.right_angle,
        samples_per_pixel=args.samples_per_pixel,
        unwarp_output_height=args.unwarp_output_height,
        unwarp_output_width=args.unwarp_output_width,
    )
    print(json.dumps({
        'report_path': report['report_path'],
        'contact_sheet': report['contact_sheet'],
        'front_valid_input_pixels': report['front_algorithm3_unwarping']['roles']['front']['valid_input_pixels'],
        'front_filled_output_pixels': report['front_algorithm3_unwarping']['roles']['front']['filled_output_pixels'],
        'front_output_shape_hw': report['front_algorithm3_unwarping']['roles']['front']['output_shape_hw'],
        'left_filled_output_pixels': report['canonical_chart_unwarping']['roles']['left']['filled_output_pixels'],
        'right_filled_output_pixels': report['canonical_chart_unwarping']['roles']['right']['filled_output_pixels'],
        'left_rows_written': report['canonical_chart_unwarping']['roles']['left']['rows_written'],
        'right_rows_written': report['canonical_chart_unwarping']['roles']['right']['rows_written'],
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

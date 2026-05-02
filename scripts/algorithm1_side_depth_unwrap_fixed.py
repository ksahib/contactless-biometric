#!/usr/bin/env python
"""Algorithm-1 side-view ground-truth depth rendering and unwrapping.

This script is intended to replace the current diagnostic-side-unwrap path that
builds an unwrap directly from stitched/monotonic side-surface curves.

Correct data flow implemented here:

    front/left/right pose-normalized masks
      -> Algorithm-1 row-wise canonical 3D ellipse surface
      -> fixed orthographic camera projection for front/left/right views
      -> z-buffer visibility to create per-view dense depth maps
      -> depth-gradient surface-arc unwrap maps for each side view

The important difference from diagnose_algorithm1_surface_side_unwrap.py is
that side unwrapping consumes a rendered side depth map. It does not call
_stitched_side_surface, _dominant_monotonic_slice, or per-row chart selection.

Expected reconstruction_dir layout:

    reconstruction_dir/debug_views/front_pose_normalized.png
    reconstruction_dir/debug_views/front_pose_mask.png
    reconstruction_dir/debug_views/left_pose_normalized.png
    reconstruction_dir/debug_views/left_pose_mask.png
    reconstruction_dir/debug_views/right_pose_normalized.png
    reconstruction_dir/debug_views/right_pose_mask.png

Example:

    python algorithm1_side_depth_unwrap_fixed.py \
      --reconstruction-dir ground_truth_smoke_dense_fixed_limit6/reconstructions/s01_f01_a01 \
      --output-dir tmp/algorithm1_side_depth_unwrap_fixed/s01_f01_a01
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

    for y_front in front_rows:
        params = _algorithm1_row_params(views, int(y_front))
        if params is None:
            continue
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

    support = np.isfinite(depth) & (view.mask > 0)
    depth_filled = depth.copy()
    if smooth_depth and np.any(support):
        # Fill small holes row-wise before smoothing, then restore support mask.
        for y in range(h):
            cols = np.flatnonzero(support[y])
            if cols.size < 2:
                continue
            row = depth_filled[y]
            row_cols = np.arange(cols[0], cols[-1] + 1)
            valid = np.isfinite(row[row_cols])
            if np.count_nonzero(valid) >= 2:
                valid_cols = row_cols[valid]
                row[row_cols] = np.interp(row_cols, valid_cols, row[valid_cols]).astype(np.float32)
        clean = np.where(np.isfinite(depth_filled), depth_filled, 0.0).astype(np.float32)
        mask_f = support.astype(np.float32)
        clean_blur = cv2.GaussianBlur(clean * mask_f, (0, 0), sigmaX=1.2, sigmaY=1.2)
        mask_blur = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=1.2, sigmaY=1.2)
        smoothed = np.divide(clean_blur, np.maximum(mask_blur, 1e-6)).astype(np.float32)
        depth_filled[support] = smoothed[support]

    gx, gy = _depth_gradients(depth_filled, support)
    report = {
        "role": role,
        "angle_deg": float(angle_deg),
        "support_pixels": int(np.count_nonzero(support)),
        "observed_mask_pixels": int(np.count_nonzero(view.mask > 0)),
        "support_over_observed_mask": float(np.count_nonzero(support) / max(np.count_nonzero(view.mask > 0), 1)),
        "depth_stats": _finite_stats(depth_filled[support]),
        "gradient_x_stats": _finite_stats(gx[support]),
        "gradient_y_stats": _finite_stats(gy[support]),
        "row_param_stats": {
            "a": _finite_stats(np.asarray([v[0] for v in row_param_values], dtype=np.float32)),
            "b": _finite_stats(np.asarray([v[1] for v in row_param_values], dtype=np.float32)),
            "cz": _finite_stats(np.asarray([v[2] for v in row_param_values], dtype=np.float32)),
        },
    }
    return {
        "depth": depth_filled.astype(np.float32),
        "support": support.astype(np.uint8),
        "gx": gx.astype(np.float32),
        "gy": gy.astype(np.float32),
        "branch": branch_map.astype(np.int8),
        "canonical_x": canonical_x.astype(np.float32),
        "canonical_z": canonical_z.astype(np.float32),
        "source_y_front": source_y_front.astype(np.float32),
        "sample_hits": sample_hits.astype(np.uint16),
        "report": report,
    }


def _depth_gradients(depth: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(depth, dtype=np.float32)
    valid = support.astype(bool) & np.isfinite(d)
    filled = d.copy()
    if np.any(valid):
        median = float(np.nanmedian(filled[valid]))
    else:
        median = 0.0
    filled[~np.isfinite(filled)] = median
    gx = np.zeros_like(filled, dtype=np.float32)
    gy = np.zeros_like(filled, dtype=np.float32)
    gx[:, 1:-1] = 0.5 * (filled[:, 2:] - filled[:, :-2])
    gx[:, 0] = filled[:, 1] - filled[:, 0]
    gx[:, -1] = filled[:, -1] - filled[:, -2]
    gy[1:-1, :] = 0.5 * (filled[2:, :] - filled[:-2, :])
    gy[0, :] = filled[1, :] - filled[0, :]
    gy[-1, :] = filled[-1, :] - filled[-2, :]
    gx[~valid] = np.nan
    gy[~valid] = np.nan
    return gx, gy


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


def _unwrap_from_depth_arc(
    image: np.ndarray,
    support: np.ndarray,
    depth: np.ndarray,
    reverse_x: bool = False,
) -> dict[str, np.ndarray | dict[str, Any]]:
    """Unwarp one view from rendered depth using row-wise surface arc length.

    This deliberately uses the rendered side depth map as the geometry source.
    It does not use Algorithm-1 branch stitching as an unwrap chart. The output
    y coordinate is the pose-normalized image row; the output x coordinate is
    cumulative visible-surface arc length in each row.
    """
    h, w = image.shape
    support_bool = support.astype(bool) & np.isfinite(depth)
    row_records: list[dict[str, Any]] = []
    max_width = 1
    for y in range(h):
        cols = np.flatnonzero(support_bool[y])
        if cols.size < 2:
            continue
        # Split disjoint support chunks and keep the largest chunk. A correctly
        # rendered side depth map should have one dominant contiguous visible row.
        chunks = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
        cols = max(chunks, key=lambda c: c.size)
        if cols.size < 2:
            continue
        z = depth[y, cols].astype(np.float32)
        dz = np.diff(z)
        ds = np.sqrt(1.0 + dz * dz).astype(np.float32)
        s = np.concatenate([[0.0], np.cumsum(ds)]).astype(np.float32)
        if reverse_x:
            # Keep source order and reverse output parameterization, not source pixels.
            s = float(s[-1]) - s
            order = np.argsort(s)
            s = s[order]
            source_cols = cols[order].astype(np.float32)
        else:
            source_cols = cols.astype(np.float32)
        width = int(math.ceil(float(s[-1] - s[0]))) + 1
        max_width = max(max_width, width)
        row_records.append({"y": int(y), "cols": source_cols, "s": s, "s_min": float(s[0]), "s_max": float(s[-1]), "width": width})

    if not row_records:
        raise RuntimeError("No valid rows available for depth unwrap")

    y0 = min(r["y"] for r in row_records)
    y1 = max(r["y"] for r in row_records)
    out_h = y1 - y0 + 1
    out_w = max_width
    source_x = np.full((out_h, out_w), np.nan, dtype=np.float32)
    source_y = np.full((out_h, out_w), np.nan, dtype=np.float32)
    expected = np.zeros((out_h, out_w), dtype=np.uint8)

    for rec in row_records:
        row = int(rec["y"] - y0)
        s = np.asarray(rec["s"], dtype=np.float32)
        cols = np.asarray(rec["cols"], dtype=np.float32)
        if s.size < 2:
            continue
        # Center each row in the output canvas to preserve finger shape.
        width = int(rec["width"])
        offset = (out_w - width) // 2
        target_cols = np.arange(width, dtype=np.float32)
        target_s = target_cols + float(rec["s_min"])
        source_x[row, offset : offset + width] = np.interp(target_s, s, cols).astype(np.float32)
        source_y[row, offset : offset + width] = float(rec["y"])
        expected[row, offset : offset + width] = 1

    inside = (
        (expected > 0)
        & np.isfinite(source_x)
        & np.isfinite(source_y)
        & (source_x >= 0)
        & (source_x <= w - 1)
        & (source_y >= 0)
        & (source_y <= h - 1)
    )
    remap_x = np.where(inside, source_x, 0.0).astype(np.float32)
    remap_y = np.where(inside, source_y, 0.0).astype(np.float32)
    unwrapped = cv2.remap(image, remap_x, remap_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sampled_support = cv2.remap((support_bool.astype(np.uint8) * 255), remap_x, remap_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    unwrapped[~sampled_support] = 0

    report = {
        "output_shape_hw": [int(out_h), int(out_w)],
        "row_records": int(len(row_records)),
        "support_pixels": int(np.count_nonzero(sampled_support)),
        "expected_pixels": int(np.count_nonzero(expected)),
        "support_over_expected": float(np.count_nonzero(sampled_support) / max(np.count_nonzero(expected), 1)),
        "source_x_step_stats": _source_x_step_stats(source_x, sampled_support),
        "reverse_x": bool(reverse_x),
    }
    return {
        "unwrapped": unwrapped.astype(np.uint8),
        "mask": sampled_support.astype(np.uint8),
        "expected": expected.astype(np.uint8),
        "source_x": source_x.astype(np.float32),
        "source_y": source_y.astype(np.float32),
        "report": report,
    }


def _source_x_step_stats(source_x: np.ndarray, support: np.ndarray) -> dict[str, float]:
    steps: list[np.ndarray] = []
    valid = support.astype(bool) & np.isfinite(source_x)
    for y in range(source_x.shape[0]):
        cols = np.flatnonzero(valid[y])
        if cols.size < 2:
            continue
        breaks = np.flatnonzero(np.diff(cols) > 1) + 1
        for seg in np.split(cols, breaks):
            if seg.size < 2:
                continue
            steps.append(np.abs(np.gradient(source_x[y, seg].astype(np.float32))))
    if not steps:
        return _finite_stats(np.asarray([], dtype=np.float32))
    return _finite_stats(np.concatenate(steps))


def _mask_iou(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    aa = a.astype(bool)
    bb = b.astype(bool)
    inter = int(np.count_nonzero(aa & bb))
    union = int(np.count_nonzero(aa | bb))
    return {
        "iou": float(inter / union) if union else 0.0,
        "intersection": inter,
        "union": union,
        "a_pixels": int(np.count_nonzero(aa)),
        "b_pixels": int(np.count_nonzero(bb)),
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
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _write_contact_sheet(out_dir: Path, views: dict[str, ViewData], rendered: dict[str, dict[str, Any]], unwrapped: dict[str, dict[str, Any]]) -> Path:
    rows: list[np.ndarray] = []
    for role in ("left", "right"):
        depth_color = _depth_color(rendered[role]["depth"], rendered[role]["support"])
        gx_png = _normalize_for_png(rendered[role]["gx"], rendered[role]["support"])
        panels = [
            _draw_label(_fit_panel(views[role].image), f"{role} pose image"),
            _draw_label(_fit_panel(views[role].mask), f"{role} observed mask"),
            _draw_label(_fit_panel(rendered[role]["support"] * 255), f"{role} depth support"),
            _draw_label(_fit_panel(depth_color), f"{role} rendered depth"),
            _draw_label(_fit_panel(gx_png), f"{role} gx"),
            _draw_label(_fit_panel(unwrapped[role]["unwrapped"]), f"{role} depth unwrap"),
            _draw_label(_fit_panel(unwrapped[role]["mask"] * 255), f"{role} unwrap mask"),
        ]
        rows.append(np.hstack(panels))
    sheet = np.vstack(rows)
    path = out_dir / "algorithm1_side_depth_unwrap_contact_sheet.png"
    cv2.imwrite(str(path), sheet)
    return path


def run(
    reconstruction_dir: Path,
    output_dir: Path,
    left_angle: float,
    right_angle: float,
    samples_per_pixel: float,
    reverse_left_unwrap_x: bool,
    reverse_right_unwrap_x: bool,
) -> dict[str, Any]:
    reconstruction_dir = reconstruction_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    views = _load_views(reconstruction_dir)
    angles = {"front": 0.0, "left": float(left_angle), "right": float(right_angle)}

    rendered: dict[str, dict[str, Any]] = {}
    unwrapped: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {
        "input_reconstruction_dir": str(reconstruction_dir),
        "output_dir": str(output_dir.resolve()),
        "method": "Algorithm 1 canonical 3D surface -> fixed-view projection with z-buffer -> per-view depth maps -> depth-arc side unwrap",
        "view_angles_deg": angles,
        "samples_per_pixel": float(samples_per_pixel),
        "notes": [
            "Side depth maps are rendered from the canonical 3D model before unwrapping.",
            "No stitched side branch, per-row dominant segment, or chart-selection heuristic is used as geometry source.",
            "The support mask is a depth-rendering support mask, not a direct side-chart expected mask.",
        ],
        "roles": {},
    }

    for role in ("front", "left", "right"):
        rendered[role] = _render_view_depth(
            views,
            role=role,
            angle_deg=angles[role],
            samples_per_pixel=samples_per_pixel,
            smooth_depth=True,
        )
        role_dir = output_dir / role
        role_dir.mkdir(parents=True, exist_ok=True)
        np.save(role_dir / f"{role}_depth.npy", rendered[role]["depth"])
        np.save(role_dir / f"{role}_gradient.npy", np.stack([rendered[role]["gx"], rendered[role]["gy"]], axis=-1).astype(np.float32))
        np.savez_compressed(
            role_dir / f"{role}_surface_maps.npz",
            depth=rendered[role]["depth"],
            support=rendered[role]["support"],
            gx=rendered[role]["gx"],
            gy=rendered[role]["gy"],
            branch=rendered[role]["branch"],
            canonical_x=rendered[role]["canonical_x"],
            canonical_z=rendered[role]["canonical_z"],
            source_y_front=rendered[role]["source_y_front"],
        )
        _write_gray(role_dir / f"{role}_support_mask.png", rendered[role]["support"] * 255)
        cv2.imwrite(str(role_dir / f"{role}_depth_color.png"), _depth_color(rendered[role]["depth"], rendered[role]["support"]))
        _write_gray(role_dir / f"{role}_gradient_x.png", _normalize_for_png(rendered[role]["gx"], rendered[role]["support"]))
        _write_gray(role_dir / f"{role}_gradient_y.png", _normalize_for_png(rendered[role]["gy"], rendered[role]["support"]))
        rendered_report = rendered[role]["report"]
        rendered_report["support_vs_observed"] = _mask_iou(rendered[role]["support"], views[role].mask > 0)
        report["roles"][role] = {"rendered_depth": rendered_report}

    for role, reverse_x in (("left", reverse_left_unwrap_x), ("right", reverse_right_unwrap_x)):
        unwrapped[role] = _unwrap_from_depth_arc(
            views[role].image,
            rendered[role]["support"],
            rendered[role]["depth"],
            reverse_x=reverse_x,
        )
        role_dir = output_dir / role
        _write_gray(role_dir / f"{role}_depth_unwrapped.png", unwrapped[role]["unwrapped"])
        _write_gray(role_dir / f"{role}_depth_unwrapped_mask.png", unwrapped[role]["mask"] * 255)
        _write_gray(role_dir / f"{role}_depth_unwrapped_expected.png", unwrapped[role]["expected"] * 255)
        np.savez_compressed(
            role_dir / f"{role}_depth_unwarp_maps.npz",
            unwrapped=unwrapped[role]["unwrapped"],
            unwrapped_mask=unwrapped[role]["mask"],
            expected_mask=unwrapped[role]["expected"],
            source_x_map=unwrapped[role]["source_x"],
            source_y_map=unwrapped[role]["source_y"],
        )
        report["roles"][role]["depth_unwrap"] = unwrapped[role]["report"]

    contact_sheet = _write_contact_sheet(output_dir, views, rendered, unwrapped)
    report["contact_sheet"] = str(contact_sheet.resolve())
    report_path = output_dir / "algorithm1_side_depth_unwrap_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path.resolve())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--left-angle", type=float, default=DEFAULT_VIEW_ANGLES_DEG["left"], help="Left side camera angle in degrees in the canonical x-z frame.")
    parser.add_argument("--right-angle", type=float, default=DEFAULT_VIEW_ANGLES_DEG["right"], help="Right side camera angle in degrees in the canonical x-z frame.")
    parser.add_argument("--samples-per-pixel", type=float, default=2.0, help="Canonical ellipse samples per front-view pixel.")
    parser.add_argument("--reverse-left-unwrap-x", action="store_true", help="Reverse left unwrapped x orientation after depth rendering.")
    parser.add_argument("--reverse-right-unwrap-x", action="store_true", help="Reverse right unwrapped x orientation after depth rendering.")
    args = parser.parse_args()

    report = run(
        reconstruction_dir=args.reconstruction_dir,
        output_dir=args.output_dir,
        left_angle=args.left_angle,
        right_angle=args.right_angle,
        samples_per_pixel=args.samples_per_pixel,
        reverse_left_unwrap_x=args.reverse_left_unwrap_x,
        reverse_right_unwrap_x=args.reverse_right_unwrap_x,
    )
    print(json.dumps({
        "report_path": report["report_path"],
        "contact_sheet": report["contact_sheet"],
        "left_support_over_observed": report["roles"]["left"]["rendered_depth"]["support_over_observed_mask"],
        "right_support_over_observed": report["roles"]["right"]["rendered_depth"]["support_over_observed_mask"],
        "left_unwrap_shape": report["roles"]["left"]["depth_unwrap"]["output_shape_hw"],
        "right_unwrap_shape": report["roles"]["right"]["depth_unwrap"]["output_shape_hw"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

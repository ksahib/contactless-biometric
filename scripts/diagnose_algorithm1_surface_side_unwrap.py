#!/usr/bin/env python
"""Algorithm-1 surface-coordinate side unwrap diagnostic.

This is a temporary/research diagnostic. It does not modify training labels.

The script rebuilds the row-relative Algorithm 1 side geometry from the saved
pose-normalized smoke debug views, then unwraps each side image using a surface
arc coordinate for x and the side pose-image row for y. This avoids the
Algorithm 3 column/row integration collapse that produced side-view strips.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

# The repo has a top-level copy.py, which can shadow stdlib copy while importing
# cv2/matplotlib from ad-hoc scripts. Keep this diagnostic self-contained.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT]

import cv2  # noqa: E402
import numpy as np  # noqa: E402


SQRT2_INV = 1.0 / math.sqrt(2.0)
CHART_GAP_PX = 24
PROJECTION_SIGN_CANDIDATES = (-1.0, 1.0)
PRIMARY_BRANCH_MODE = "visible"
ALLOW_STITCHED_FALLBACK_IN_PRIMARY = False
REVERSE_OUTPUT_X_BY_ROLE = {"left": False, "right": False}
MIN_SOURCE_X_STEP_FOR_SHARP_DETAIL = 0.30
SEAM_DIAGNOSTIC_WIDTH_PX = 80


@dataclass(frozen=True)
class RowStats:
    left: np.ndarray
    right: np.ndarray
    center: np.ndarray
    width: np.ndarray
    valid_rows: np.ndarray


@dataclass(frozen=True)
class ViewData:
    image: np.ndarray
    mask: np.ndarray
    stats: RowStats


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


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
        left[y] = x0
        right[y] = x1
        center[y] = 0.5 * (x0 + x1)
        width[y] = float(x1 - x0 + 1)
        valid.append(y)
    return RowStats(left, right, center, width, np.asarray(valid, dtype=np.int32))


def _load_views(reconstruction_dir: Path) -> dict[str, ViewData]:
    debug_dir = reconstruction_dir / "debug_views"
    views: dict[str, ViewData] = {}
    for role in ("front", "left", "right"):
        image = _read_gray(debug_dir / f"{role}_pose_normalized.png")
        mask = _read_gray(debug_dir / f"{role}_pose_mask.png")
        views[role] = ViewData(image=image, mask=np.where(mask > 0, 255, 0).astype(np.uint8), stats=_row_stats(mask))
    return views


def _map_row(source_rows: np.ndarray, target_rows: np.ndarray, y_target: int) -> int:
    if len(source_rows) == 0 or len(target_rows) == 0:
        return -1
    if target_rows[-1] == target_rows[0]:
        rel = 0.0
    else:
        rel = (float(y_target) - float(target_rows[0])) / float(target_rows[-1] - target_rows[0])
    y_source = float(source_rows[0]) + rel * float(source_rows[-1] - source_rows[0])
    return int(np.clip(round(y_source), int(source_rows[0]), int(source_rows[-1])))


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {k: 0.0 for k in ("min", "p01", "p05", "median", "mean", "p95", "p99", "max", "span")}
    return {
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "span": float(np.max(values) - np.min(values)),
    }


def _algorithm1_row_params(
    views: dict[str, ViewData],
    role: str,
    y_side: int,
) -> tuple[int, int, float, float, float, float, float, float] | None:
    front_rows = views["front"].stats.valid_rows
    left_rows = views["left"].stats.valid_rows
    right_rows = views["right"].stats.valid_rows
    side_rows = views[role].stats.valid_rows
    y_front = _map_row(front_rows, side_rows, y_side)
    y_left = _map_row(left_rows, side_rows, y_side)
    y_right = _map_row(right_rows, side_rows, y_side)
    if min(y_front, y_left, y_right) < 0:
        return None

    front = views["front"].stats
    left = views["left"].stats
    right = views["right"].stats
    side = views[role].stats
    if front.width[y_front] <= 0 or left.width[y_left] <= 0 or right.width[y_right] <= 0 or side.width[y_side] <= 0:
        return None

    a = max(float(front.width[y_front]) * 0.5, 1.0)
    d_left = float(left.width[y_left]) * 0.5
    d_right = float(right.width[y_right]) * 0.5
    b_left = math.sqrt(max(2.0 * d_left * d_left - a * a, 0.0))
    b_right = math.sqrt(max(2.0 * d_right * d_right - a * a, 0.0))
    b = 0.5 * (b_left + b_right)

    # Algorithm-1-faithful row-relative center-depth shift. The important bit is
    # using side/front center differences, never absolute center sums.
    shift_right = float(right.center[y_right] - front.center[y_front])
    shift_left = float(front.center[y_front] - left.center[y_left])
    center_depth = math.sqrt(2.0) * 0.5 * (shift_right + shift_left)
    return y_front, y_left, y_right, a, b, center_depth, float(side.center[y_side]), float(side.width[y_side])


def _monotonicity_report(values: np.ndarray, eps: float = 1e-4) -> dict[str, Any]:
    diffs = np.diff(np.asarray(values, dtype=np.float32))
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return {
            "is_monotonic": False,
            "direction": "none",
            "turn_count": 0,
            "min_diff": 0.0,
            "max_diff": 0.0,
            "pos_steps": 0,
            "neg_steps": 0,
        }

    pos = int(np.count_nonzero(diffs > eps))
    neg = int(np.count_nonzero(diffs < -eps))
    direction = "increasing" if pos >= neg else "decreasing"
    is_monotonic = (neg == 0) if direction == "increasing" else (pos == 0)

    signs = np.sign(diffs)
    signs[np.abs(diffs) <= eps] = 0
    nonzero = signs[signs != 0]
    turn_count = int(np.count_nonzero(nonzero[1:] != nonzero[:-1])) if nonzero.size > 1 else 0
    return {
        "is_monotonic": bool(is_monotonic),
        "direction": direction,
        "turn_count": turn_count,
        "min_diff": float(np.min(diffs)),
        "max_diff": float(np.max(diffs)),
        "pos_steps": pos,
        "neg_steps": neg,
    }


def _split_into_monotonic_segments(x_pose: np.ndarray, min_len: int = 8, eps: float = 1e-4) -> list[slice]:
    diffs = np.diff(np.asarray(x_pose, dtype=np.float32))
    signs = np.sign(diffs)
    signs[np.abs(diffs) <= eps] = 0

    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    for i in range(len(signs) - 2, -1, -1):
        if signs[i] == 0:
            signs[i] = signs[i + 1]

    cut_points = [0]
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            cut_points.append(i + 1)
    cut_points.append(len(x_pose))

    segments: list[slice] = []
    for start, stop in zip(cut_points[:-1], cut_points[1:]):
        if stop - start >= min_len:
            segments.append(slice(start, stop))
    return segments


def _dominant_monotonic_slice(x_pose: np.ndarray, min_len: int = 8) -> slice:
    """Pick a deterministic visible chart from a projected top branch.

    The full top ellipse branch is usually folded in side-image x. For the
    primary diagnostic we do not choose per-row among many charts; we use the
    same deterministic rule for every row: the longest monotonic segment whose
    center is closest to the full branch center.
    """
    segments = _split_into_monotonic_segments(x_pose, min_len=min_len)
    if not segments:
        return slice(0, len(x_pose))
    branch_center = 0.5 * (float(np.nanmin(x_pose)) + float(np.nanmax(x_pose)))

    def score(segment: slice) -> tuple[float, float]:
        x_segment = np.asarray(x_pose[segment], dtype=np.float32)
        length = float(max(segment.stop - segment.start, 0))
        segment_center = 0.5 * (float(np.nanmin(x_segment)) + float(np.nanmax(x_segment)))
        return (length, -abs(segment_center - branch_center))

    return max(segments, key=score)


def _visible_side_surface(
    a: float,
    b: float,
    center_depth: float,
    role: str,
    projection_sign: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return a one-branch side-view surface candidate.

    This intentionally avoids stitching upper and lower ellipse branches into a
    single row. If the projected branch still folds near the silhouette, callers
    must split it into monotonic charts before remapping.
    """
    samples = max(int(round(2.0 * a)) + 1, 3)
    x = np.linspace(-a, a, samples, dtype=np.float32)
    z = b * np.sqrt(np.maximum(1.0 - (x / max(a, 1e-6)) ** 2, 0.0))
    z = z + center_depth

    x_rot = (x + projection_sign * z) * SQRT2_INV
    span_center = 0.5 * (float(np.nanmin(x_rot)) + float(np.nanmax(x_rot)))
    x_pose_offset = x_rot - span_center

    ds = np.sqrt(np.diff(x) ** 2 + np.diff(z) ** 2)
    s = np.concatenate([[0.0], np.cumsum(ds)]).astype(np.float32)
    s_rel = s - 0.5 * (float(s[0]) + float(s[-1]))
    segment = _dominant_monotonic_slice(x_pose_offset)
    x_pose_offset = x_pose_offset[segment]
    s_rel = s_rel[segment]
    s_rel = s_rel - 0.5 * (float(s_rel[0]) + float(s_rel[-1]))
    mono = _monotonicity_report(x_pose_offset)
    debug = {
        "projection_sign": float(projection_sign),
        "role": role,
        "branch": "top_visible_dominant_monotonic",
        "dominant_segment_start": int(segment.start),
        "dominant_segment_stop": int(segment.stop),
        "monotonicity": mono,
        "arc_length": float(s_rel[-1] - s_rel[0]),
        "x_pose_offset_min": float(np.nanmin(x_pose_offset)),
        "x_pose_offset_max": float(np.nanmax(x_pose_offset)),
    }
    return x_pose_offset.astype(np.float32), s_rel.astype(np.float32), debug


def _stitched_side_surface_candidate(
    a: float,
    b: float,
    center_depth: float,
    role: str,
    projection_sign: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the wider side-facing candidate path for split-chart fallback.

    This candidate may fold in side-image x; callers must split it into
    monotonic charts before using it for remapping.
    """
    samples = max(int(round(2.0 * a)) + 1, 3)
    x = np.linspace(-a, a, samples, dtype=np.float32)
    z_radius = b * np.sqrt(np.maximum(1.0 - (x / max(a, 1e-6)) ** 2, 0.0))
    z_top = z_radius + center_depth
    z_bottom = -z_radius + center_depth

    keep_bottom = x[::-1] >= 0.571 * a
    x_loop = np.concatenate([x, x[::-1][keep_bottom][1:]]).astype(np.float32)
    z_loop = np.concatenate([z_top, z_bottom[::-1][keep_bottom][1:]]).astype(np.float32)

    x_rot = (x_loop + projection_sign * z_loop) * SQRT2_INV
    span_center = 0.5 * (float(np.nanmin(x_rot)) + float(np.nanmax(x_rot)))
    x_pose_offset = x_rot - span_center

    ds = np.sqrt(np.diff(x_loop) ** 2 + np.diff(z_loop) ** 2)
    s = np.concatenate([[0.0], np.cumsum(ds)]).astype(np.float32)
    s_rel = s - 0.5 * (float(s[0]) + float(s[-1]))
    mono = _monotonicity_report(x_pose_offset)
    debug = {
        "projection_sign": float(projection_sign),
        "role": role,
        "branch": "stitched_split_fallback",
        "monotonicity": mono,
        "arc_length": float(s_rel[-1] - s_rel[0]),
        "x_pose_offset_min": float(np.nanmin(x_pose_offset)),
        "x_pose_offset_max": float(np.nanmax(x_pose_offset)),
    }
    return x_pose_offset.astype(np.float32), s_rel.astype(np.float32), debug


def _row_interval_iou(x_segments: list[np.ndarray], observed_x0: float, observed_x1: float, row_width: int) -> float:
    observed = np.zeros(row_width, dtype=bool)
    projected = np.zeros(row_width, dtype=bool)
    obs0 = int(max(0, math.floor(observed_x0)))
    obs1 = int(min(row_width - 1, math.ceil(observed_x1)))
    if obs1 >= obs0:
        observed[obs0 : obs1 + 1] = True
    for x_pose in x_segments:
        if x_pose.size == 0:
            continue
        x0 = int(max(0, math.floor(float(np.nanmin(x_pose)))))
        x1 = int(min(row_width - 1, math.ceil(float(np.nanmax(x_pose)))))
        if x1 >= x0:
            projected[x0 : x1 + 1] = True
    union = observed | projected
    if not np.any(union):
        return 0.0
    return float(np.count_nonzero(observed & projected) / np.count_nonzero(union))


def _segment_observed_fraction(x_pose: np.ndarray, observed_x0: float, observed_x1: float) -> float:
    x_pose = np.asarray(x_pose, dtype=np.float32)
    valid = np.isfinite(x_pose)
    if not np.any(valid):
        return 0.0
    inside = (x_pose[valid] >= observed_x0) & (x_pose[valid] <= observed_x1)
    return float(np.count_nonzero(inside) / np.count_nonzero(valid))


def _projected_mask_from_rows(
    views: dict[str, ViewData],
    role: str,
    row_records: list[dict[str, Any]],
) -> np.ndarray:
    h, w = views[role].mask.shape
    projected = np.zeros((h, w), dtype=np.uint8)
    for rec in row_records:
        y = int(rec["y_side"])
        x_pose = np.asarray(rec["x_pose"], dtype=np.float32)
        if x_pose.size == 0:
            continue
        x0 = int(max(0, math.floor(float(np.nanmin(x_pose)))))
        x1 = int(min(w - 1, math.ceil(float(np.nanmax(x_pose)))))
        if x1 >= x0:
            projected[y, x0 : x1 + 1] = 255
    return projected


def _reprojection_metrics(observed_u8: np.ndarray, projected_u8: np.ndarray) -> dict[str, Any]:
    observed = observed_u8 > 0
    projected = projected_u8 > 0
    inter = observed & projected
    union = observed | projected
    valid_rows = np.flatnonzero(observed.any(axis=1) | projected.any(axis=1))

    center_errors: list[float] = []
    width_errors: list[float] = []
    for y in valid_rows:
        obs_x = np.flatnonzero(observed[y])
        prj_x = np.flatnonzero(projected[y])
        if obs_x.size == 0 or prj_x.size == 0:
            continue
        obs_center = 0.5 * (int(obs_x[0]) + int(obs_x[-1]))
        prj_center = 0.5 * (int(prj_x[0]) + int(prj_x[-1]))
        center_errors.append(abs(prj_center - obs_center))
        width_errors.append(abs(float(prj_x[-1] - prj_x[0] + 1) - float(obs_x[-1] - obs_x[0] + 1)))

    rows = np.flatnonzero(observed.any(axis=1))
    if rows.size:
        y0 = int(rows[0] + 0.25 * (rows[-1] - rows[0]))
        y1 = int(rows[0] + 0.75 * (rows[-1] - rows[0]))
        central = slice(y0, y1 + 1)
        c_inter = int(np.count_nonzero(inter[central]))
        c_union = int(np.count_nonzero(union[central]))
    else:
        c_inter = c_union = 0

    inter_count = int(np.count_nonzero(inter))
    union_count = int(np.count_nonzero(union))
    observed_count = int(np.count_nonzero(observed))
    projected_count = int(np.count_nonzero(projected))
    return {
        "iou": float(inter_count / union_count) if union_count else 0.0,
        "central_iou": float(c_inter / c_union) if c_union else 0.0,
        "precision": float(inter_count / projected_count) if projected_count else 0.0,
        "recall": float(inter_count / observed_count) if observed_count else 0.0,
        "intersection_pixels": inter_count,
        "union_pixels": union_count,
        "observed_pixels": observed_count,
        "projected_pixels": projected_count,
        "false_positive_pixels": int(np.count_nonzero(projected & ~observed)),
        "false_negative_pixels": int(np.count_nonzero(observed & ~projected)),
        "overlap_rows": int(len(center_errors)),
        "row_center_mae": float(np.mean(center_errors)) if center_errors else 0.0,
        "row_width_mae": float(np.mean(width_errors)) if width_errors else 0.0,
    }


def _overlay(observed_u8: np.ndarray, projected_u8: np.ndarray) -> np.ndarray:
    observed = observed_u8 > 0
    projected = projected_u8 > 0
    out = np.zeros((*observed.shape, 3), dtype=np.uint8)
    out[observed & projected] = (0, 255, 255)
    out[observed & ~projected] = (0, 0, 255)
    out[projected & ~observed] = (0, 180, 0)
    return out


def _normalize_float_image(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(arr)
    if mask is not None:
        valid &= mask > 0
    if not np.any(valid):
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.percentile(arr[valid], 1))
    hi = float(np.percentile(arr[valid], 99))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return (out * 255.0).astype(np.uint8)


def _source_x_step_map(source_x: np.ndarray, expected_mask: np.ndarray) -> np.ndarray:
    """Estimate source-image x motion per output column.

    Values below about one third of a source pixel per output pixel indicate
    heavy horizontal magnification. That is where ridge detail tends to look
    soft even if the unwrap mapping is coherent.
    """
    source_x = np.asarray(source_x, dtype=np.float32)
    valid = (expected_mask > 0) & np.isfinite(source_x)
    step = np.full(source_x.shape, np.nan, dtype=np.float32)
    for y in range(source_x.shape[0]):
        cols = np.flatnonzero(valid[y])
        if cols.size < 2:
            continue
        breaks = np.flatnonzero(np.diff(cols) > 1) + 1
        for segment_cols in np.split(cols, breaks):
            if segment_cols.size < 2:
                continue
            values = source_x[y, segment_cols].astype(np.float32)
            if segment_cols.size == 2:
                segment_step = np.full(2, abs(float(values[1] - values[0])), dtype=np.float32)
            else:
                segment_step = np.abs(np.gradient(values)).astype(np.float32)
            step[y, segment_cols] = segment_step
    return step


def _scale_quality_metrics(
    unwrapped: np.ndarray,
    observed_support: np.ndarray,
    source_x_step: np.ndarray,
    threshold: float = MIN_SOURCE_X_STEP_FOR_SHARP_DETAIL,
) -> dict[str, Any]:
    valid = observed_support & np.isfinite(source_x_step)
    low_scale = valid & (source_x_step < threshold)

    seam_scale: dict[str, list[float]] = {"left": [], "right": [], "center": []}
    seam_contrast: dict[str, list[float]] = {"left": [], "right": [], "center": []}
    seam_low_fraction: dict[str, list[float]] = {"left": [], "right": [], "center": []}
    for y in range(valid.shape[0]):
        cols = np.flatnonzero(valid[y])
        if cols.size < max(16, SEAM_DIAGNOSTIC_WIDTH_PX):
            continue
        edge_n = min(SEAM_DIAGNOSTIC_WIDTH_PX, max(cols.size // 5, 8))
        center_n = min(SEAM_DIAGNOSTIC_WIDTH_PX, max(cols.size // 5, 8))
        center_mid = cols.size // 2
        slices = {
            "left": cols[:edge_n],
            "right": cols[-edge_n:],
            "center": cols[max(0, center_mid - center_n // 2) : min(cols.size, center_mid + center_n // 2)],
        }
        for name, region_cols in slices.items():
            if region_cols.size < 2:
                continue
            scale_values = source_x_step[y, region_cols]
            finite = np.isfinite(scale_values)
            if np.any(finite):
                seam_scale[name].append(float(np.median(scale_values[finite])))
                seam_low_fraction[name].append(float(np.count_nonzero(scale_values[finite] < threshold) / np.count_nonzero(finite)))
            seam_contrast[name].append(float(np.std(unwrapped[y, region_cols].astype(np.float32))))

    center_contrast = float(np.median(seam_contrast["center"])) if seam_contrast["center"] else 0.0
    right_contrast = float(np.median(seam_contrast["right"])) if seam_contrast["right"] else 0.0
    return {
        "min_source_x_step_for_sharp_detail": float(threshold),
        "valid_scale_pixels": int(np.count_nonzero(valid)),
        "low_source_x_step_pixels": int(np.count_nonzero(low_scale)),
        "low_source_x_step_fraction": float(np.count_nonzero(low_scale) / max(np.count_nonzero(valid), 1)),
        "source_x_step_stats": _stats(source_x_step[valid]) if np.any(valid) else _stats(np.asarray([], dtype=np.float32)),
        "left_seam_source_x_step_median": float(np.median(seam_scale["left"])) if seam_scale["left"] else 0.0,
        "center_source_x_step_median": float(np.median(seam_scale["center"])) if seam_scale["center"] else 0.0,
        "right_seam_source_x_step_median": float(np.median(seam_scale["right"])) if seam_scale["right"] else 0.0,
        "left_seam_low_source_x_step_fraction": (
            float(np.median(seam_low_fraction["left"])) if seam_low_fraction["left"] else 0.0
        ),
        "right_seam_low_source_x_step_fraction": (
            float(np.median(seam_low_fraction["right"])) if seam_low_fraction["right"] else 0.0
        ),
        "center_intensity_std_median": center_contrast,
        "right_seam_intensity_std_median": right_contrast,
        "right_seam_vs_center_contrast_ratio": float(right_contrast / center_contrast) if center_contrast > 0 else 0.0,
        "seam_diagnostic_width_px": int(SEAM_DIAGNOSTIC_WIDTH_PX),
    }


def _draw_label(panel: np.ndarray, text: str) -> np.ndarray:
    if panel.ndim == 2:
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
    panel = panel.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (18, 18, 18), -1)
    cv2.putText(panel, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return panel


def _fit_panel(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
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


def _resample_1d(values: np.ndarray, n: int = 128) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.zeros(n, dtype=np.float32)
    if values.size == 1:
        return np.full(n, float(values[0]), dtype=np.float32)
    old = np.linspace(0.0, 1.0, values.size, dtype=np.float32)
    new = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.interp(new, old, values).astype(np.float32)


def _row_coherence_metrics(row_records: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_records = sorted(row_records, key=lambda rec: int(rec["y_side"]))
    jumps: list[float] = []
    for prev, curr in zip(sorted_records[:-1], sorted_records[1:]):
        prev_x = _resample_1d(np.asarray(prev["x_pose"], dtype=np.float32))
        curr_x = _resample_1d(np.asarray(curr["x_pose"], dtype=np.float32))
        jumps.append(float(np.median(np.abs(curr_x - prev_x))))

    jump_values = np.asarray(jumps, dtype=np.float32)
    threshold = 25.0
    bad_count = int(np.count_nonzero(jump_values > threshold)) if jump_values.size else 0
    return {
        "adjacent_pairs": int(jump_values.size),
        "median_adjacent_source_x_jump": float(np.median(jump_values)) if jump_values.size else 0.0,
        "p95_adjacent_source_x_jump": float(np.percentile(jump_values, 95)) if jump_values.size else 0.0,
        "max_adjacent_source_x_jump": float(np.max(jump_values)) if jump_values.size else 0.0,
        "bad_row_jump_threshold_px": threshold,
        "bad_row_jump_count": bad_count,
        "bad_row_jump_fraction": float(bad_count / max(int(jump_values.size), 1)),
        "row_jump_values": jump_values,
    }


def _row_coherence_gate_passed(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["bad_row_jump_fraction"] <= 0.02
        and metrics["p95_adjacent_source_x_jump"] <= metrics["bad_row_jump_threshold_px"]
    )


def _build_primary_visible_row_records(
    views: dict[str, ViewData],
    role: str,
    side_start: int,
    side_end: int,
    projection_sign: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    side = views[role]
    row_records: list[dict[str, Any]] = []
    rows_missing_params = 0
    rows_skipped_nonmonotonic = 0
    reverse_output_x = bool(REVERSE_OUTPUT_X_BY_ROLE.get(role, False))

    for y_side in range(side_start, side_end + 1):
        params = _algorithm1_row_params(views, role, y_side)
        if params is None:
            rows_missing_params += 1
            continue
        _y_front, _y_left, _y_right, a, b, center_depth, side_center, _side_width = params
        x_offset, s_rel, surface_debug = _visible_side_surface(a, b, center_depth, role, projection_sign)
        x_pose = (side_center + x_offset).astype(np.float32)
        s_rel = (s_rel - 0.5 * (float(s_rel[0]) + float(s_rel[-1]))).astype(np.float32)
        if reverse_output_x:
            x_pose = x_pose[::-1].copy()
            s_rel = (-s_rel[::-1]).astype(np.float32)
            s_rel = s_rel - 0.5 * (float(s_rel[0]) + float(s_rel[-1]))

        mono = _monotonicity_report(x_pose)
        if not mono["is_monotonic"] or mono["turn_count"] != 0:
            rows_skipped_nonmonotonic += 1
            continue

        row_iou = _row_interval_iou(
            [x_pose],
            float(side.stats.left[y_side]),
            float(side.stats.right[y_side]),
            side.mask.shape[1],
        )
        support_fraction = _segment_observed_fraction(
            x_pose,
            float(side.stats.left[y_side]),
            float(side.stats.right[y_side]),
        )
        row_records.append(
            {
                "y_side": int(y_side),
                "chart_id": 0,
                "segment_id": 0,
                "branch_mode": PRIMARY_BRANCH_MODE,
                "projection_sign": float(projection_sign),
                "x_pose": x_pose,
                "s_rel": s_rel.astype(np.float32),
                "arc_length": float(s_rel[-1] - s_rel[0]),
                "row_interval_iou": float(row_iou),
                "segment_observed_fraction": float(support_fraction),
                "source_segment_count": 1,
                "interpolated_missing_row": False,
                "monotonicity": mono,
                "surface_debug": surface_debug,
                "center_depth": float(center_depth),
                "semi_major": float(a),
                "semi_minor": float(b),
            }
        )

    debug = {
        "projection_sign": float(projection_sign),
        "branch_mode": PRIMARY_BRANCH_MODE,
        "allow_stitched_fallback_in_primary": bool(ALLOW_STITCHED_FALLBACK_IN_PRIMARY),
        "reverse_output_x": reverse_output_x,
        "rows_total": int(side_end - side_start + 1),
        "rows_built": int(len(row_records)),
        "rows_missing_params": int(rows_missing_params),
        "rows_skipped_nonmonotonic": int(rows_skipped_nonmonotonic),
    }
    return row_records, debug


def _interpolate_row_record(prev: dict[str, Any], curr: dict[str, Any], y_side: int) -> dict[str, Any]:
    alpha = (float(y_side) - float(prev["y_side"])) / max(float(curr["y_side"] - prev["y_side"]), 1.0)
    n = max(len(prev["x_pose"]), len(curr["x_pose"]), 2)
    prev_x = _resample_1d(np.asarray(prev["x_pose"], dtype=np.float32), n)
    curr_x = _resample_1d(np.asarray(curr["x_pose"], dtype=np.float32), n)
    prev_s = _resample_1d(np.asarray(prev["s_rel"], dtype=np.float32), n)
    curr_s = _resample_1d(np.asarray(curr["s_rel"], dtype=np.float32), n)
    x_pose = ((1.0 - alpha) * prev_x + alpha * curr_x).astype(np.float32)
    s_rel = ((1.0 - alpha) * prev_s + alpha * curr_s).astype(np.float32)
    s_rel = s_rel - 0.5 * (float(s_rel[0]) + float(s_rel[-1]))
    mono = _monotonicity_report(x_pose)
    rec = dict(prev)
    rec.update(
        {
            "y_side": int(y_side),
            "x_pose": x_pose,
            "s_rel": s_rel,
            "arc_length": float(s_rel[-1] - s_rel[0]),
            "row_interval_iou": float((1.0 - alpha) * prev["row_interval_iou"] + alpha * curr["row_interval_iou"]),
            "segment_observed_fraction": float(
                (1.0 - alpha) * prev["segment_observed_fraction"] + alpha * curr["segment_observed_fraction"]
            ),
            "interpolated_missing_row": True,
            "monotonicity": mono,
            "center_depth": float((1.0 - alpha) * prev["center_depth"] + alpha * curr["center_depth"]),
            "semi_major": float((1.0 - alpha) * prev["semi_major"] + alpha * curr["semi_major"]),
            "semi_minor": float((1.0 - alpha) * prev["semi_minor"] + alpha * curr["semi_minor"]),
        }
    )
    return rec


def _fill_missing_row_records(
    row_records: list[dict[str, Any]],
    side_start: int,
    side_end: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not row_records:
        return [], {"missing_rows_filled": 0, "edge_missing_rows_left_blank": int(side_end - side_start + 1)}

    by_y = {int(rec["y_side"]): rec for rec in row_records}
    valid_rows = sorted(by_y)
    filled: list[dict[str, Any]] = []
    missing_rows_filled = 0

    first_valid = valid_rows[0]
    last_valid = valid_rows[-1]
    edge_missing = max(first_valid - side_start, 0) + max(side_end - last_valid, 0)
    edge_missing_left_blank = edge_missing if edge_missing > 5 else 0

    for y in range(side_start, side_end + 1):
        if y in by_y:
            filled.append(by_y[y])
            continue
        if y < first_valid:
            if first_valid - y <= 5:
                rec = dict(by_y[first_valid])
                rec["y_side"] = int(y)
                rec["interpolated_missing_row"] = True
                filled.append(rec)
                missing_rows_filled += 1
            continue
        if y > last_valid:
            if y - last_valid <= 5:
                rec = dict(by_y[last_valid])
                rec["y_side"] = int(y)
                rec["interpolated_missing_row"] = True
                filled.append(rec)
                missing_rows_filled += 1
            continue

        prev_y = max(v for v in valid_rows if v < y)
        next_y = min(v for v in valid_rows if v > y)
        filled.append(_interpolate_row_record(by_y[prev_y], by_y[next_y], y))
        missing_rows_filled += 1

    return sorted(filled, key=lambda rec: int(rec["y_side"])), {
        "missing_rows_filled": int(missing_rows_filled),
        "edge_missing_rows_left_blank": int(edge_missing_left_blank),
        "first_valid_input_row": int(first_valid),
        "last_valid_input_row": int(last_valid),
        "rows_after_fill": int(len(filled)),
    }


def _select_projection_sign_for_role(
    views: dict[str, ViewData],
    role: str,
    side_start: int,
    side_end: int,
) -> tuple[float, dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for sign in PROJECTION_SIGN_CANDIDATES:
        rows, build_debug = _build_primary_visible_row_records(views, role, side_start, side_end, sign)
        filled_rows, fill_debug = _fill_missing_row_records(rows, side_start, side_end)
        projected = _projected_mask_from_rows(views, role, filled_rows)
        reproj = _reprojection_metrics(views[role].mask, projected)
        coherence = _row_coherence_metrics(filled_rows)
        score = (
            3.0 * reproj["iou"]
            + 2.0 * reproj["central_iou"]
            - 0.01 * reproj["row_width_mae"]
            - 0.01 * reproj["row_center_mae"]
            - 0.50 * coherence["bad_row_jump_fraction"]
        )
        coherence_report = dict(coherence)
        coherence_report.pop("row_jump_values", None)
        trials.append(
            {
                "sign": float(sign),
                "score": float(score),
                "reprojection_metrics": reproj,
                "coherence_metrics": coherence_report,
                "rows_used": int(len(filled_rows)),
                "row_build_debug": build_debug,
                "missing_row_fill_debug": fill_debug,
            }
        )

    selected = max(trials, key=lambda item: item["score"])
    return float(selected["sign"]), {"selected_sign": float(selected["sign"]), "trials": trials}


def _write_row_debug_images(
    out_dir: Path,
    role: str,
    row_records: list[dict[str, Any]],
    side_start: int,
    side_end: int,
    row_jump_values: np.ndarray,
) -> tuple[Path, Path]:
    h = side_end - side_start + 1
    jump_col = np.zeros((h, 48), dtype=np.uint8)
    values = np.asarray(row_jump_values, dtype=np.float32)
    for idx, value in enumerate(values, start=1):
        if idx >= h:
            break
        jump_col[idx, :] = int(np.clip(value / 50.0, 0.0, 1.0) * 255)

    sign_col = np.zeros((h, 48), dtype=np.uint8)
    for rec in row_records:
        y_out = int(rec["y_side"] - side_start)
        if 0 <= y_out < h:
            sign_col[y_out, :] = 64 if float(rec["projection_sign"]) < 0 else 192

    jump_path = out_dir / f"{role}_surface_row_jump_debug.png"
    sign_path = out_dir / f"{role}_surface_projection_sign_debug.png"
    cv2.imwrite(str(jump_path), jump_col)
    cv2.imwrite(str(sign_path), sign_col)
    return jump_path, sign_path


def _write_contact_sheet(out_dir: Path, views: dict[str, ViewData], role_reports: dict[str, Any]) -> Path:
    panels: list[np.ndarray] = []
    panel_size = (260, 210)
    for role in ("left", "right"):
        info = role_reports[role]
        baseline_path = Path(info["algorithm3_strip_baseline"])
        baseline = _read_gray(baseline_path) if baseline_path.exists() else np.zeros((64, 128), dtype=np.uint8)
        role_panels = [
            _draw_label(_fit_panel(views[role].image, panel_size), f"{role} pose image"),
            _draw_label(_fit_panel(views[role].mask, panel_size), f"{role} observed mask"),
            _draw_label(_fit_panel(_read_gray(Path(info["surface_projected_mask"])), panel_size), f"{role} projected IoU {info['reprojection_metrics']['iou']:.3f}"),
            _draw_label(_fit_panel(cv2.imread(info["surface_reprojection_overlay"], cv2.IMREAD_COLOR), panel_size), f"{role} overlay"),
            _draw_label(_fit_panel(baseline, panel_size), f"{role} alg3 strip"),
            _draw_label(
                _fit_panel(_read_gray(Path(info["surface_unwrapped"])), panel_size),
                f"{role} surface unwrap mono {int(info['monotonicity_gate_passed'])}",
            ),
            _draw_label(
                _fit_panel(_read_gray(Path(info["surface_unwrapped_mask"])), panel_size),
                f"{role} mask {info['surface_metrics']['mask_coverage_over_expected']:.3f}",
            ),
            _draw_label(
                _fit_panel(_read_gray(Path(info["surface_source_x_step"])), panel_size),
                f"{role} source scale",
            ),
            _draw_label(
                _fit_panel(_read_gray(Path(info["surface_scale_quality_preview"])), panel_size),
                f"{role} quality preview",
            ),
        ]
        panels.append(np.hstack(role_panels))
    sheet = np.vstack(panels)
    path = out_dir / "surface_unwrap_contact_sheet.png"
    cv2.imwrite(str(path), sheet)
    return path


def _unwrap_role(
    views: dict[str, ViewData],
    reconstruction_dir: Path,
    out_dir: Path,
    role: str,
) -> dict[str, Any]:
    side = views[role]
    side_rows = side.stats.valid_rows
    if side_rows.size == 0:
        raise ValueError(f"{role} side mask has no valid rows")
    side_start = int(side_rows[0])
    side_end = int(side_rows[-1])
    out_h = side_end - side_start + 1

    selected_sign, sign_debug = _select_projection_sign_for_role(views, role, side_start, side_end)
    row_records, row_build_debug = _build_primary_visible_row_records(views, role, side_start, side_end, selected_sign)
    row_records, missing_row_fill_debug = _fill_missing_row_records(row_records, side_start, side_end)
    if not row_records:
        raise ValueError(f"No usable {role} surface rows")

    row_coherence = _row_coherence_metrics(row_records)
    row_coherence_gate_passed = _row_coherence_gate_passed(row_coherence)

    chart_bounds: dict[int, list[float]] = {}
    for rec in row_records:
        chart_id = int(rec["chart_id"])
        s_rel = np.asarray(rec["s_rel"], dtype=np.float32)
        if chart_id not in chart_bounds:
            chart_bounds[chart_id] = [float(s_rel[0]), float(s_rel[-1])]
        else:
            chart_bounds[chart_id][0] = min(chart_bounds[chart_id][0], float(s_rel[0]))
            chart_bounds[chart_id][1] = max(chart_bounds[chart_id][1], float(s_rel[-1]))

    chart_offsets: dict[int, int] = {}
    cursor = 0
    for chart_id in sorted(chart_bounds):
        chart_offsets[chart_id] = cursor
        width = int(math.ceil(chart_bounds[chart_id][1] - chart_bounds[chart_id][0])) + 2
        cursor += width + CHART_GAP_PX
    out_w = max(1, cursor - CHART_GAP_PX)
    source_x = np.full((out_h, out_w), np.nan, dtype=np.float32)
    source_y = np.full((out_h, out_w), np.nan, dtype=np.float32)
    expected_mask = np.zeros((out_h, out_w), dtype=np.uint8)

    for rec in row_records:
        y_out = int(rec["y_side"] - side_start)
        s_rel = np.asarray(rec["s_rel"], dtype=np.float32)
        x_pose = np.asarray(rec["x_pose"], dtype=np.float32)
        chart_id = int(rec["chart_id"])
        chart_min = chart_bounds[chart_id][0]
        chart_offset = chart_offsets[chart_id]
        col0 = max(0, chart_offset + int(math.ceil(float(s_rel[0] - chart_min))))
        col1 = min(out_w - 1, chart_offset + int(math.floor(float(s_rel[-1] - chart_min))))
        if col1 < col0:
            continue
        cols = np.arange(col0, col1 + 1, dtype=np.float32)
        target_s = (cols - float(chart_offset)) + float(chart_min)
        source_x[y_out, col0 : col1 + 1] = np.interp(target_s, s_rel, x_pose).astype(np.float32)
        source_y[y_out, col0 : col1 + 1] = float(rec["y_side"])
        expected_mask[y_out, col0 : col1 + 1] = 1

    h, w = side.image.shape
    inside = (
        expected_mask.astype(bool)
        & np.isfinite(source_x)
        & np.isfinite(source_y)
        & (source_x >= 0)
        & (source_x <= w - 1)
        & (source_y >= 0)
        & (source_y <= h - 1)
    )
    remap_x = np.where(inside, source_x, 0).astype(np.float32)
    remap_y = np.where(inside, source_y, 0).astype(np.float32)
    unwrapped = cv2.remap(side.image, remap_x, remap_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sampled_mask = cv2.remap(side.mask, remap_x, remap_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    observed_support = inside & (sampled_mask > 0)
    unwrapped[~observed_support] = 0
    source_x_step = _source_x_step_map(source_x, expected_mask)
    source_scale_quality = observed_support & np.isfinite(source_x_step) & (source_x_step >= MIN_SOURCE_X_STEP_FOR_SHARP_DETAIL)
    quality_masked_preview = unwrapped.copy()
    quality_masked_preview[~source_scale_quality] = 0
    scale_quality_metrics = _scale_quality_metrics(unwrapped, observed_support, source_x_step)

    projected_mask = _projected_mask_from_rows(views, role, row_records)
    overlay = _overlay(side.mask, projected_mask)
    reproj = _reprojection_metrics(side.mask, projected_mask)

    stem = f"{role}_surface"
    unwrapped_path = out_dir / f"{stem}_unwrapped.png"
    unwrapped_mask_path = out_dir / f"{stem}_unwrapped_mask.png"
    expected_mask_path = out_dir / f"{stem}_expected_mask.png"
    source_x_path = out_dir / f"{stem}_source_x.png"
    source_y_path = out_dir / f"{stem}_source_y.png"
    source_x_step_path = out_dir / f"{stem}_source_x_step.png"
    scale_quality_mask_path = out_dir / f"{stem}_scale_quality_mask.png"
    scale_quality_preview_path = out_dir / f"{stem}_scale_quality_preview.png"
    debug_path = out_dir / f"{stem}_coordinate_debug.npz"
    projected_path = out_dir / f"{stem}_projected_mask.png"
    overlay_path = out_dir / f"{stem}_reprojection_overlay.png"

    cv2.imwrite(str(unwrapped_path), unwrapped)
    cv2.imwrite(str(unwrapped_mask_path), (observed_support.astype(np.uint8) * 255))
    cv2.imwrite(str(expected_mask_path), (expected_mask.astype(np.uint8) * 255))
    cv2.imwrite(str(source_x_path), _normalize_float_image(source_x, expected_mask))
    cv2.imwrite(str(source_y_path), _normalize_float_image(source_y, expected_mask))
    cv2.imwrite(str(source_x_step_path), _normalize_float_image(source_x_step, observed_support.astype(np.uint8)))
    cv2.imwrite(str(scale_quality_mask_path), (source_scale_quality.astype(np.uint8) * 255))
    cv2.imwrite(str(scale_quality_preview_path), quality_masked_preview)
    cv2.imwrite(str(projected_path), projected_mask)
    cv2.imwrite(str(overlay_path), overlay)
    row_jump_path, projection_sign_debug_path = _write_row_debug_images(
        out_dir,
        role,
        row_records,
        side_start,
        side_end,
        row_coherence["row_jump_values"],
    )
    row_projection_signs = np.asarray([float(rec["projection_sign"]) for rec in row_records], dtype=np.float32)
    row_monotonic_flags = np.asarray([bool(rec["monotonicity"]["is_monotonic"]) for rec in row_records], dtype=np.uint8)
    row_turn_counts = np.asarray([int(rec["monotonicity"]["turn_count"]) for rec in row_records], dtype=np.int32)
    row_arc_lengths = np.asarray([float(rec["arc_length"]) for rec in row_records], dtype=np.float32)
    row_y_side_values = np.asarray([int(rec["y_side"]) for rec in row_records], dtype=np.int32)
    row_chart_ids = np.asarray([int(rec["chart_id"]) for rec in row_records], dtype=np.int32)
    row_interval_ious = np.asarray([float(rec["row_interval_iou"]) for rec in row_records], dtype=np.float32)
    row_segment_observed_fractions = np.asarray(
        [float(rec["segment_observed_fraction"]) for rec in row_records],
        dtype=np.float32,
    )
    row_interpolated_flags = np.asarray([bool(rec.get("interpolated_missing_row", False)) for rec in row_records], dtype=np.uint8)
    row_segment_ids = np.asarray([int(rec["segment_id"]) for rec in row_records], dtype=np.int32)
    x_pose_min_diffs = np.asarray([float(rec["monotonicity"]["min_diff"]) for rec in row_records], dtype=np.float32)
    x_pose_max_diffs = np.asarray([float(rec["monotonicity"]["max_diff"]) for rec in row_records], dtype=np.float32)
    arc_lengths = np.asarray([float(rec["arc_length"]) for rec in row_records], dtype=np.float32)
    center_depths = np.asarray([float(rec["center_depth"]) for rec in row_records], dtype=np.float32)
    semi_major = np.asarray([float(rec["semi_major"]) for rec in row_records], dtype=np.float32)
    semi_minor = np.asarray([float(rec["semi_minor"]) for rec in row_records], dtype=np.float32)
    np.savez_compressed(
        debug_path,
        unwrapped_mask=observed_support.astype(np.uint8),
        expected_mask=expected_mask.astype(np.uint8),
        source_x_map=source_x.astype(np.float32),
        source_y_map=source_y.astype(np.float32),
        source_x_step_map=source_x_step.astype(np.float32),
        source_scale_quality_mask=source_scale_quality.astype(np.uint8),
        global_s_min=np.float32(min(bounds[0] for bounds in chart_bounds.values())),
        global_s_max=np.float32(max(bounds[1] for bounds in chart_bounds.values())),
        side_start=np.int32(side_start),
        side_end=np.int32(side_end),
        projected_mask=projected_mask.astype(np.uint8),
        row_projection_signs=row_projection_signs,
        row_monotonic_flags=row_monotonic_flags,
        row_turn_counts=row_turn_counts,
        row_arc_lengths=row_arc_lengths,
        row_y_side_values=row_y_side_values,
        row_chart_ids=row_chart_ids,
        row_segment_ids=row_segment_ids,
        row_interval_ious=row_interval_ious,
        row_segment_observed_fractions=row_segment_observed_fractions,
        row_interpolated_flags=row_interpolated_flags,
        row_source_x_jump_values=row_coherence["row_jump_values"].astype(np.float32),
    )

    expected_pixels = int(np.count_nonzero(expected_mask))
    support_pixels = int(np.count_nonzero(observed_support))
    rows_used = len(set(int(rec["y_side"]) for rec in row_records))
    nonmonotonic_records = int(np.count_nonzero((row_monotonic_flags == 0) | (row_turn_counts != 0)))
    max_turn_count = int(np.max(row_turn_counts)) if row_turn_counts.size else 0
    projection_signs_used = sorted(float(v) for v in set(row_projection_signs.tolist()))
    branch_modes_used = sorted(str(v) for v in set(rec["branch_mode"] for rec in row_records))
    rows_split_into_charts = int(np.count_nonzero(row_segment_ids != 0) + np.count_nonzero(row_chart_ids != 0))
    surface_chart_metrics = {
        "selected_projection_sign": float(selected_sign),
        "projection_signs_used": projection_signs_used,
        "branch_modes_used": branch_modes_used,
        "rows_total": int(out_h),
        "rows_used": int(rows_used),
        "rows_skipped_nonmonotonic": int(row_build_debug["rows_skipped_nonmonotonic"]),
        "rows_split_into_charts": int(rows_split_into_charts),
        "record_count": int(len(row_records)),
        "chart_count": int(len(chart_bounds)),
        "nonmonotonic_fraction": float(nonmonotonic_records / max(len(row_records), 1)),
        "max_turn_count": max_turn_count,
        "mean_arc_length": float(np.mean(arc_lengths)) if arc_lengths.size else 0.0,
        "min_arc_length": float(np.min(arc_lengths)) if arc_lengths.size else 0.0,
        "max_arc_length": float(np.max(arc_lengths)) if arc_lengths.size else 0.0,
        "x_pose_min_diff_p01": float(np.percentile(x_pose_min_diffs, 1)) if x_pose_min_diffs.size else 0.0,
        "x_pose_max_diff_p99": float(np.percentile(x_pose_max_diffs, 99)) if x_pose_max_diffs.size else 0.0,
        "mean_row_interval_iou": float(np.mean(row_interval_ious)) if row_interval_ious.size else 0.0,
        "min_row_interval_iou": float(np.min(row_interval_ious)) if row_interval_ious.size else 0.0,
        "mean_segment_observed_fraction": (
            float(np.mean(row_segment_observed_fractions)) if row_segment_observed_fractions.size else 0.0
        ),
        "min_segment_observed_fraction_used": (
            float(np.min(row_segment_observed_fractions)) if row_segment_observed_fractions.size else 0.0
        ),
    }
    monotonicity_gate_passed = bool(
        surface_chart_metrics["nonmonotonic_fraction"] <= 0.01
        and surface_chart_metrics["max_turn_count"] == 0
    )
    chart_coherence_gate_passed = bool(
        surface_chart_metrics["chart_count"] == 1
        and surface_chart_metrics["rows_split_into_charts"] == 0
        and len(projection_signs_used) == 1
        and projection_signs_used[0] == float(selected_sign)
        and branch_modes_used == [PRIMARY_BRANCH_MODE]
        and monotonicity_gate_passed
    )
    surface_metrics = {
        "output_shape_hw": [int(out_h), int(out_w)],
        "side_valid_row_span": int(out_h),
        "height_ratio_vs_side_valid_span": float(out_h / max(out_h, 1)),
        "expected_surface_pixels": expected_pixels,
        "unwrapped_mask_pixels": support_pixels,
        "mask_coverage_over_expected": float(support_pixels / expected_pixels) if expected_pixels else 0.0,
        "row_records": int(len(row_records)),
        "global_s_min": float(min(bounds[0] for bounds in chart_bounds.values())),
        "global_s_max": float(max(bounds[1] for bounds in chart_bounds.values())),
        "global_surface_width": float(sum(bounds[1] - bounds[0] for bounds in chart_bounds.values())),
        "chart_gap_px": int(CHART_GAP_PX),
    }
    surface_metrics["surface_gate_passed"] = bool(
        surface_metrics["height_ratio_vs_side_valid_span"] >= 0.90
        and surface_metrics["mask_coverage_over_expected"] >= 0.95
    )

    primary_gate = bool(
        reproj["iou"] >= 0.80 and reproj["central_iou"] >= 0.90 and reproj["row_center_mae"] <= 10.0
    )
    baseline = reconstruction_dir.parents[2] if len(reconstruction_dir.parents) > 2 else REPO_ROOT
    alg3 = REPO_ROOT / "tmp" / "algorithm1_faithful_side_geometry" / reconstruction_dir.name / f"{role}_algorithm3_unwarped.png"
    return {
        "input_image": str((reconstruction_dir / "debug_views" / f"{role}_pose_normalized.png").resolve()),
        "input_mask": str((reconstruction_dir / "debug_views" / f"{role}_pose_mask.png").resolve()),
        "surface_unwrapped": str(unwrapped_path.resolve()),
        "surface_unwrapped_mask": str(unwrapped_mask_path.resolve()),
        "surface_expected_mask": str(expected_mask_path.resolve()),
        "surface_source_x": str(source_x_path.resolve()),
        "surface_source_y": str(source_y_path.resolve()),
        "surface_source_x_step": str(source_x_step_path.resolve()),
        "surface_scale_quality_mask": str(scale_quality_mask_path.resolve()),
        "surface_scale_quality_preview": str(scale_quality_preview_path.resolve()),
        "surface_coordinate_debug": str(debug_path.resolve()),
        "surface_projected_mask": str(projected_path.resolve()),
        "surface_reprojection_overlay": str(overlay_path.resolve()),
        "surface_row_jump_debug": str(row_jump_path.resolve()),
        "surface_projection_sign_debug": str(projection_sign_debug_path.resolve()),
        "algorithm3_strip_baseline": str(alg3.resolve()),
        "reprojection_metrics": reproj,
        "primary_gate_passed": primary_gate,
        "surface_metrics": surface_metrics,
        "scale_quality_metrics": scale_quality_metrics,
        "surface_chart_metrics": surface_chart_metrics,
        "selected_projection_sign": float(selected_sign),
        "projection_sign_selection": sign_debug,
        "row_build_debug": row_build_debug,
        "missing_row_fill_debug": missing_row_fill_debug,
        "row_coherence_metrics": {k: v for k, v in row_coherence.items() if k != "row_jump_values"},
        "row_coherence_gate_passed": row_coherence_gate_passed,
        "chart_coherence_gate_passed": chart_coherence_gate_passed,
        "monotonicity_gate_passed": monotonicity_gate_passed,
        "all_gates_passed": bool(
            primary_gate
            and surface_metrics["surface_gate_passed"]
            and monotonicity_gate_passed
            and chart_coherence_gate_passed
            and row_coherence_gate_passed
        ),
        "arc_length_stats": _stats(arc_lengths),
        "center_depth_stats": _stats(center_depths),
        "semi_major_stats": _stats(semi_major),
        "semi_minor_stats": _stats(semi_minor),
    }


def _compare_reports(new_report: dict[str, Any], compare_dir: Path) -> dict[str, Any]:
    old_path = compare_dir / "surface_unwrap_report.json"
    if not old_path.exists():
        return {"compare_dir": str(compare_dir.resolve()), "available": False}
    old = json.loads(old_path.read_text())
    diffs: dict[str, Any] = {"compare_dir": str(compare_dir.resolve()), "available": True, "roles": {}}
    for role in ("left", "right"):
        role_diff: dict[str, Any] = {}
        for metric in ("iou", "central_iou", "row_center_mae", "row_width_mae"):
            old_v = old["roles"][role]["reprojection_metrics"][metric]
            new_v = new_report["roles"][role]["reprojection_metrics"][metric]
            role_diff[f"reprojection_{metric}_delta"] = float(new_v - old_v)
        for metric in ("output_shape_hw", "mask_coverage_over_expected", "expected_surface_pixels", "unwrapped_mask_pixels"):
            old_v = old["roles"][role]["surface_metrics"][metric]
            new_v = new_report["roles"][role]["surface_metrics"][metric]
            if isinstance(old_v, list):
                role_diff[f"surface_{metric}_same"] = bool(old_v == new_v)
            else:
                role_diff[f"surface_{metric}_delta"] = float(new_v - old_v)
        old_chart = old["roles"][role].get("surface_chart_metrics", {})
        new_chart = new_report["roles"][role].get("surface_chart_metrics", {})
        for metric in ("nonmonotonic_fraction", "max_turn_count", "rows_split_into_charts", "chart_count"):
            if metric in old_chart and metric in new_chart:
                role_diff[f"chart_{metric}_delta"] = float(new_chart[metric] - old_chart[metric])
            elif metric in new_chart:
                role_diff[f"chart_{metric}_new"] = new_chart[metric]
        diffs["roles"][role] = role_diff
    return diffs


def run(reconstruction_dir: Path, output_dir: Path, compare_dir: Path | None = None) -> dict[str, Any]:
    reconstruction_dir = reconstruction_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    views = _load_views(reconstruction_dir)

    roles = {
        "left": _unwrap_role(views, reconstruction_dir, output_dir, "left"),
        "right": _unwrap_role(views, reconstruction_dir, output_dir, "right"),
    }
    contact_sheet = _write_contact_sheet(output_dir, views, roles)

    report: dict[str, Any] = {
        "output_dir": str(output_dir.resolve()),
        "input_reconstruction_dir": str(reconstruction_dir),
        "method": (
            "Algorithm 1 side unwrap using cross-section surface arc length for x "
            "and side pose row for y; no Algorithm 3 row/column seed integration."
        ),
        "acceptance_criteria": {
            "reprojection": {"iou_min": 0.8, "central_iou_min": 0.9, "row_center_mae_max_px": 10.0},
            "surface_unwrap": {"height_ratio_min": 0.9, "mask_coverage_min": 0.95},
            "row_coherence": {"bad_row_jump_fraction_max": 0.02, "p95_adjacent_source_x_jump_max_px": 25.0},
            "chart_coherence": {"chart_count": 1, "rows_split_into_charts": 0, "projection_signs_used": 1},
        },
        "baseline_algorithm3_strip": {
            "directory": str((REPO_ROOT / "tmp" / "algorithm1_faithful_side_geometry" / reconstruction_dir.name).resolve()),
            "known_shapes": {"left": [221, 867], "right": [148, 862]},
        },
        "geometry": {
            "shape_hw": list(map(int, views["front"].mask.shape)),
            "valid_all_rows": int(
                len(
                    set(map(int, views["front"].stats.valid_rows))
                    & set(map(int, views["left"].stats.valid_rows))
                    & set(map(int, views["right"].stats.valid_rows))
                )
            ),
            "center_depth_stats": _stats(
                np.concatenate(
                    [
                        np.asarray(roles["left"]["center_depth_stats"]["mean"], dtype=np.float32).reshape(1),
                        np.asarray(roles["right"]["center_depth_stats"]["mean"], dtype=np.float32).reshape(1),
                    ]
                )
            ),
            "semi_major_stats": roles["left"]["semi_major_stats"],
            "semi_minor_stats": roles["left"]["semi_minor_stats"],
        },
        "roles": roles,
        "contact_sheet": str(contact_sheet.resolve()),
    }
    warnings: list[str] = []
    if roles["left"]["selected_projection_sign"] == roles["right"]["selected_projection_sign"]:
        warnings.append(
            "Left and right selected the same projection sign; verify camera handedness and row-center convention."
        )
    report["primary_gate_passed_all"] = bool(all(roles[r]["primary_gate_passed"] for r in ("left", "right")))
    report["surface_gate_passed_all"] = bool(all(roles[r]["surface_metrics"]["surface_gate_passed"] for r in ("left", "right")))
    report["monotonicity_gate_passed_all"] = bool(all(roles[r]["monotonicity_gate_passed"] for r in ("left", "right")))
    report["chart_coherence_gate_passed_all"] = bool(all(roles[r]["chart_coherence_gate_passed"] for r in ("left", "right")))
    report["row_coherence_gate_passed_all"] = bool(all(roles[r]["row_coherence_gate_passed"] for r in ("left", "right")))
    report["warnings"] = warnings
    report["all_gates_passed_all"] = bool(all(roles[r]["all_gates_passed"] for r in ("left", "right")))
    report["conclusion"] = (
        "Surface-coordinate unwrap cleared all numeric gates."
        if report["all_gates_passed_all"]
        else "Surface-coordinate unwrap did not clear all numeric gates."
    )
    if compare_dir is not None:
        report["comparison_to_previous"] = _compare_reports(report, compare_dir)

    (output_dir / "surface_unwrap_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reconstruction-dir",
        type=Path,
        default=REPO_ROOT / "ground_truth_smoke_dense_fixed_limit6" / "reconstructions" / "s01_f01_a01",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "tmp" / "algorithm1_surface_unwrap" / "s01_f01_a01",
    )
    parser.add_argument("--compare-dir", type=Path, default=None)
    args = parser.parse_args()

    report = run(args.reconstruction_dir, args.output_dir, args.compare_dir)
    print(json.dumps(
        {
            "output_dir": report["output_dir"],
            "contact_sheet": report["contact_sheet"],
            "primary_gate_passed_all": report["primary_gate_passed_all"],
            "surface_gate_passed_all": report["surface_gate_passed_all"],
            "monotonicity_gate_passed_all": report["monotonicity_gate_passed_all"],
            "chart_coherence_gate_passed_all": report["chart_coherence_gate_passed_all"],
            "row_coherence_gate_passed_all": report["row_coherence_gate_passed_all"],
            "all_gates_passed_all": report["all_gates_passed_all"],
            "warnings": report["warnings"],
            "left": {
                "shape": report["roles"]["left"]["surface_metrics"]["output_shape_hw"],
                "iou": report["roles"]["left"]["reprojection_metrics"]["iou"],
                "central_iou": report["roles"]["left"]["reprojection_metrics"]["central_iou"],
                "coverage": report["roles"]["left"]["surface_metrics"]["mask_coverage_over_expected"],
                "selected_projection_sign": report["roles"]["left"]["selected_projection_sign"],
                "nonmonotonic_fraction": report["roles"]["left"]["surface_chart_metrics"]["nonmonotonic_fraction"],
                "max_turn_count": report["roles"]["left"]["surface_chart_metrics"]["max_turn_count"],
                "bad_row_jump_fraction": report["roles"]["left"]["row_coherence_metrics"]["bad_row_jump_fraction"],
            },
            "right": {
                "shape": report["roles"]["right"]["surface_metrics"]["output_shape_hw"],
                "iou": report["roles"]["right"]["reprojection_metrics"]["iou"],
                "central_iou": report["roles"]["right"]["reprojection_metrics"]["central_iou"],
                "coverage": report["roles"]["right"]["surface_metrics"]["mask_coverage_over_expected"],
                "selected_projection_sign": report["roles"]["right"]["selected_projection_sign"],
                "nonmonotonic_fraction": report["roles"]["right"]["surface_chart_metrics"]["nonmonotonic_fraction"],
                "max_turn_count": report["roles"]["right"]["surface_chart_metrics"]["max_turn_count"],
                "bad_row_jump_fraction": report["roles"]["right"]["row_coherence_metrics"]["bad_row_jump_fraction"],
            },
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

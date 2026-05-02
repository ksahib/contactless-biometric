#!/usr/bin/env python
"""Fuse left/right side surface unwraps by choosing the better-sampled view."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT]

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyfing  # noqa: E402


MIN_SOURCE_X_STEP = 0.30
SEAM_WIDTH = 80
DEFAULT_DPI = 500
ORIENTATION_DELTA_PX = 4.0
MIN_ORIENTATION_BASELINE_PX = 1.5
SOURCE_STEP_SWEEP_THRESHOLDS = (0.30, 0.45, 0.60, 0.75, 0.90)
SHARED_S_REVERSE_BY_ROLE = {"left": False, "right": True}
TARGET_REPROJECTION_SOURCE_STEP_BY_ROLE = {"left": 0.30, "right": 0.60}


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _load_role(input_dir: Path, role: str) -> dict[str, np.ndarray]:
    debug = np.load(input_dir / f"{role}_surface_coordinate_debug.npz")
    return {
        "image": _read_gray(input_dir / f"{role}_surface_unwrapped.png"),
        "mask": _read_gray(input_dir / f"{role}_surface_unwrapped_mask.png") > 0,
        "quality": _read_gray(input_dir / f"{role}_surface_scale_quality_mask.png") > 0,
        "source_step": debug["source_x_step_map"].astype(np.float32),
        "source_x": debug["source_x_map"].astype(np.float32),
        "source_y": debug["source_y_map"].astype(np.float32),
    }


def _load_surface_unwrap_module():
    module_path = REPO_ROOT / "scripts" / "diagnose_algorithm1_surface_side_unwrap.py"
    spec = importlib.util.spec_from_file_location("surface_side_unwrap", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_x_step_map(source_x: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    source_x = np.asarray(source_x, dtype=np.float32)
    valid = valid_mask.astype(bool) & np.isfinite(source_x)
    step = np.full(source_x.shape, np.nan, dtype=np.float32)
    for y in range(source_x.shape[0]):
        cols = np.flatnonzero(valid[y])
        if cols.size < 2:
            continue
        breaks = np.flatnonzero(np.diff(cols) > 1) + 1
        for segment_cols in np.split(cols, breaks):
            if segment_cols.size < 2:
                continue
            values = source_x[y, segment_cols]
            if segment_cols.size == 2:
                segment_step = np.full(2, abs(float(values[1] - values[0])), dtype=np.float32)
            else:
                segment_step = np.abs(np.gradient(values)).astype(np.float32)
            step[y, segment_cols] = segment_step
    return step


def _finite_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _safe_divide(num: np.ndarray, den: np.ndarray, default: float = np.nan) -> np.ndarray:
    out = np.full(np.broadcast_shapes(num.shape, den.shape), default, dtype=np.float32)
    valid = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-8)
    out[valid] = (num[valid] / den[valid]).astype(np.float32)
    return out


def _local_std_map(values: np.ndarray, valid: np.ndarray, ksize: int = 7) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid_f = valid.astype(np.float32)
    clean = np.where(valid, values, 0.0).astype(np.float32)
    kernel = (ksize, ksize)
    count = cv2.blur(valid_f, kernel)
    mean = _safe_divide(cv2.blur(clean, kernel), count, default=np.nan)
    mean_sq = _safe_divide(cv2.blur(clean * clean, kernel), count, default=np.nan)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)).astype(np.float32)


def _jacobian_geometry_quality_maps(
    source_x: np.ndarray,
    source_y: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    if source_x.shape[0] < 2 or source_x.shape[1] < 2:
        nan_map = np.full(source_x.shape, np.nan, dtype=np.float32)
        return {
            "dx_dchart_x": nan_map.copy(),
            "dx_dchart_y": nan_map.copy(),
            "dy_dchart_x": nan_map.copy(),
            "dy_dchart_y": nan_map.copy(),
            "scale_x": nan_map.copy(),
            "scale_y": nan_map.copy(),
            "area": nan_map.copy(),
            "area_signed": nan_map.copy(),
            "anisotropy": nan_map.copy(),
            "condition": nan_map.copy(),
            "jacobian_change": nan_map.copy(),
            "local_jacobian_std": nan_map.copy(),
            "jacobian_valid": np.zeros(source_x.shape, dtype=bool),
        }
    sx = np.where(valid_mask, source_x, np.nan).astype(np.float32)
    sy = np.where(valid_mask, source_y, np.nan).astype(np.float32)
    dx_dchart_y, dx_dchart_x = np.gradient(sx)
    dy_dchart_y, dy_dchart_x = np.gradient(sy)
    derivative_valid = (
        valid_mask
        & np.isfinite(dx_dchart_x)
        & np.isfinite(dx_dchart_y)
        & np.isfinite(dy_dchart_x)
        & np.isfinite(dy_dchart_y)
    )

    scale_x = np.sqrt(dx_dchart_x * dx_dchart_x + dy_dchart_x * dy_dchart_x).astype(np.float32)
    scale_y = np.sqrt(dx_dchart_y * dx_dchart_y + dy_dchart_y * dy_dchart_y).astype(np.float32)
    area_signed = (dx_dchart_x * dy_dchart_y - dx_dchart_y * dy_dchart_x).astype(np.float32)
    area = np.abs(area_signed).astype(np.float32)
    anisotropy = _safe_divide(np.maximum(scale_x, scale_y), np.minimum(scale_x, scale_y), default=np.nan)

    # Singular values of a 2x2 Jacobian from eig(J^T J). This is stricter than
    # scale_x/scale_y because it also catches shear.
    frob_sq = (
        dx_dchart_x * dx_dchart_x
        + dx_dchart_y * dx_dchart_y
        + dy_dchart_x * dy_dchart_x
        + dy_dchart_y * dy_dchart_y
    ).astype(np.float32)
    discr = np.maximum(frob_sq * frob_sq - 4.0 * area * area, 0.0)
    sigma_max_sq = 0.5 * (frob_sq + np.sqrt(discr))
    sigma_min_sq = 0.5 * (frob_sq - np.sqrt(discr))
    sigma_max = np.sqrt(np.maximum(sigma_max_sq, 0.0)).astype(np.float32)
    sigma_min = np.sqrt(np.maximum(sigma_min_sq, 0.0)).astype(np.float32)
    condition = _safe_divide(sigma_max, sigma_min, default=np.nan)

    rapid_terms: list[np.ndarray] = []
    for component in (dx_dchart_x, dx_dchart_y, dy_dchart_x, dy_dchart_y):
        comp_dy, comp_dx = np.gradient(component.astype(np.float32))
        rapid_terms.extend([comp_dx * comp_dx, comp_dy * comp_dy])
    jacobian_change = np.sqrt(np.nansum(np.stack(rapid_terms, axis=0), axis=0)).astype(np.float32)
    local_std_terms = [
        _local_std_map(component.astype(np.float32), derivative_valid)
        for component in (dx_dchart_x, dx_dchart_y, dy_dchart_x, dy_dchart_y)
    ]
    local_jacobian_std = np.sqrt(np.nansum(np.stack([term * term for term in local_std_terms], axis=0), axis=0)).astype(np.float32)

    for arr in (scale_x, scale_y, area, area_signed, anisotropy, condition, jacobian_change, local_jacobian_std):
        arr[~derivative_valid] = np.nan

    return {
        "dx_dchart_x": dx_dchart_x.astype(np.float32),
        "dx_dchart_y": dx_dchart_y.astype(np.float32),
        "dy_dchart_x": dy_dchart_x.astype(np.float32),
        "dy_dchart_y": dy_dchart_y.astype(np.float32),
        "scale_x": scale_x,
        "scale_y": scale_y,
        "area": area,
        "area_signed": area_signed,
        "anisotropy": anisotropy,
        "condition": condition,
        "jacobian_change": jacobian_change,
        "local_jacobian_std": local_jacobian_std,
        "jacobian_valid": derivative_valid,
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return {"n": int(x.size), "pearson": 0.0, "spearman": 0.0, "abs_pearson": 0.0, "abs_spearman": 0.0}
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])
    return {
        "n": int(x.size),
        "pearson": pearson,
        "spearman": spearman,
        "abs_pearson": abs(pearson),
        "abs_spearman": abs(spearman),
    }


def _infer_reconstruction_dir(input_dir: Path) -> Path:
    candidates = [
        REPO_ROOT / "ground_truth_smoke_dense_fixed_limit6" / "reconstructions" / "s01_f01_a01",
        input_dir.parents[1] / "ground_truth_smoke_dense_fixed_limit6" / "reconstructions" / "s01_f01_a01"
        if len(input_dir.parents) > 1
        else input_dir,
    ]
    for candidate in candidates:
        if (candidate / "debug_views" / "left_pose_normalized.png").exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not locate smoke reconstruction debug_views for s01_f01_a01")


def _build_role_surface_records(surface_module: Any, views: dict[str, Any], role: str) -> dict[str, Any]:
    side = views[role]
    rows = side.stats.valid_rows
    if rows.size == 0:
        raise ValueError(f"{role} side mask has no valid rows")
    side_start = int(rows[0])
    side_end = int(rows[-1])
    selected_sign, sign_debug = surface_module._select_projection_sign_for_role(views, role, side_start, side_end)
    row_records, row_build_debug = surface_module._build_primary_visible_row_records(
        views,
        role,
        side_start,
        side_end,
        selected_sign,
    )
    row_records, missing_row_fill_debug = surface_module._fill_missing_row_records(row_records, side_start, side_end)
    if not row_records:
        raise ValueError(f"No usable {role} shared-surface rows")
    row_records = sorted(row_records, key=lambda rec: int(rec["y_side"]))
    return {
        "role": role,
        "side_start": side_start,
        "side_end": side_end,
        "row_records": row_records,
        "by_y": {int(rec["y_side"]): rec for rec in row_records},
        "selected_projection_sign": float(selected_sign),
        "projection_sign_selection": sign_debug,
        "row_build_debug": row_build_debug,
        "missing_row_fill_debug": missing_row_fill_debug,
    }


def _shared_chart_geometry(role_records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    s_values: list[float] = []
    max_span = 0
    for role, info in role_records.items():
        max_span = max(max_span, int(info["side_end"] - info["side_start"] + 1))
        for rec in info["row_records"]:
            s_rel = _shared_s_rel_for_role(role, rec)
            if s_rel.size:
                s_values.extend([float(np.nanmin(s_rel)), float(np.nanmax(s_rel))])
    if not s_values:
        raise ValueError("No surface coordinate values found")
    s_min = math.floor(min(s_values))
    s_max = math.ceil(max(s_values))
    return {
        "s_min": float(s_min),
        "s_max": float(s_max),
        "shape_hw": [int(max_span), int(max(1, s_max - s_min + 1))],
    }


def _shared_s_rel_for_role(role: str, rec: dict[str, Any]) -> np.ndarray:
    s_rel = np.asarray(rec["s_rel"], dtype=np.float32)
    if bool(SHARED_S_REVERSE_BY_ROLE.get(role, False)):
        return (-s_rel).astype(np.float32)
    return s_rel.astype(np.float32)


def _record_for_shared_row(info: dict[str, Any], y_out: int, out_h: int) -> dict[str, Any]:
    if out_h <= 1:
        y_side = int(info["side_start"])
    else:
        rel = float(y_out) / float(out_h - 1)
        y_side = int(round(float(info["side_start"]) + rel * float(info["side_end"] - info["side_start"])))
    y_side = int(np.clip(y_side, int(info["side_start"]), int(info["side_end"])))
    by_y = info["by_y"]
    if y_side in by_y:
        return by_y[y_side]
    keys = np.asarray(sorted(by_y), dtype=np.int32)
    nearest = int(keys[int(np.argmin(np.abs(keys - y_side)))])
    return by_y[nearest]


def _render_role_to_shared_chart(views: dict[str, Any], role: str, info: dict[str, Any], chart: dict[str, Any]) -> dict[str, np.ndarray]:
    out_h, out_w = (int(v) for v in chart["shape_hw"])
    s_min = float(chart["s_min"])
    side = views[role]
    source_x = np.full((out_h, out_w), np.nan, dtype=np.float32)
    source_y = np.full((out_h, out_w), np.nan, dtype=np.float32)
    expected = np.zeros((out_h, out_w), dtype=bool)

    cols = np.arange(out_w, dtype=np.float32)
    target_s = cols + s_min
    shared_y_to_side_y = np.zeros(out_h, dtype=np.int32)
    for y_out in range(out_h):
        rec = _record_for_shared_row(info, y_out, out_h)
        shared_y_to_side_y[y_out] = int(rec["y_side"])
        s_rel = _shared_s_rel_for_role(role, rec)
        x_pose = np.asarray(rec["x_pose"], dtype=np.float32)
        finite = np.isfinite(s_rel) & np.isfinite(x_pose)
        if np.count_nonzero(finite) < 2:
            continue
        s_valid = s_rel[finite]
        x_valid = x_pose[finite]
        order = np.argsort(s_valid)
        s_valid = s_valid[order]
        x_valid = x_valid[order]
        unique_s, unique_idx = np.unique(s_valid, return_index=True)
        if unique_s.size < 2:
            continue
        x_unique = x_valid[unique_idx]
        row_valid = (target_s >= float(unique_s[0])) & (target_s <= float(unique_s[-1]))
        source_x[y_out, row_valid] = np.interp(target_s[row_valid], unique_s, x_unique).astype(np.float32)
        source_y[y_out, row_valid] = float(rec["y_side"])
        expected[y_out, row_valid] = True

    h, w = side.image.shape
    inside = (
        expected
        & np.isfinite(source_x)
        & np.isfinite(source_y)
        & (source_x >= 0)
        & (source_x <= w - 1)
        & (source_y >= 0)
        & (source_y <= h - 1)
    )
    remap_x = np.where(inside, source_x, 0.0).astype(np.float32)
    remap_y = np.where(inside, source_y, 0.0).astype(np.float32)
    image = cv2.remap(side.image, remap_x, remap_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sampled_mask = cv2.remap(side.mask, remap_x, remap_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    support = inside & (sampled_mask > 0)
    image[~support] = 0
    source_step = _source_x_step_map(source_x, support)
    quality = support & np.isfinite(source_step) & (source_step >= MIN_SOURCE_X_STEP)
    return {
        "image": image,
        "support": support,
        "quality": quality,
        "source_x": source_x,
        "source_y": source_y,
        "source_step": source_step,
        "expected": expected,
        "shared_y_to_side_y": shared_y_to_side_y,
    }


def _build_shared_surface_sources(reconstruction_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    surface_module = _load_surface_unwrap_module()
    views = surface_module._load_views(reconstruction_dir)
    role_records = {role: _build_role_surface_records(surface_module, views, role) for role in ("left", "right")}
    chart = _shared_chart_geometry(role_records)
    sources = {
        role: _render_role_to_shared_chart(views, role, role_records[role], chart)
        for role in ("left", "right")
    }
    debug = {
        "chart": chart,
        "roles": {
            role: {
                "side_start": int(info["side_start"]),
                "side_end": int(info["side_end"]),
                "row_count": int(len(info["row_records"])),
                "selected_projection_sign": float(info["selected_projection_sign"]),
                "shared_s_reversed": bool(SHARED_S_REVERSE_BY_ROLE.get(role, False)),
                "projection_sign_selection": info["projection_sign_selection"],
                "row_build_debug": info["row_build_debug"],
                "missing_row_fill_debug": info["missing_row_fill_debug"],
            }
            for role, info in role_records.items()
        },
    }
    return views, sources, debug


def _fuse_shared_for_target(
    target: dict[str, np.ndarray],
    donor: dict[str, np.ndarray],
    target_name: str,
    donor_name: str,
) -> dict[str, np.ndarray | dict[str, Any] | str]:
    target_reprojection_threshold = float(TARGET_REPROJECTION_SOURCE_STEP_BY_ROLE.get(target_name, MIN_SOURCE_X_STEP))
    target_step = np.where(target["support"] & np.isfinite(target["source_step"]), target["source_step"], -1.0)
    donor_step = np.where(donor["support"] & np.isfinite(donor["source_step"]), donor["source_step"], -1.0)
    target_score = target_step + 0.01 * _local_contrast(target["image"])
    donor_score = donor_step + 0.01 * _local_contrast(donor["image"])
    choose_donor = donor["support"] & ((donor_score > target_score) | ~target["support"])
    choose_target = target["support"] & ~choose_donor
    support = choose_target | choose_donor

    fused = np.zeros_like(target["image"])
    fused[choose_target] = target["image"][choose_target]
    fused[choose_donor] = donor["image"][choose_donor]
    selected_step = np.where(choose_donor, donor_step, target_step).astype(np.float32)
    selected_step[~support] = np.nan
    selected_source_role = np.where(choose_donor, 2, np.where(choose_target, 1, 0)).astype(np.uint8)
    selection = np.zeros_like(target["image"], dtype=np.uint8)
    selection[choose_target] = 85
    selection[choose_donor] = 200

    target_geometry_valid = (
        support
        & target["quality"]
        & np.isfinite(target["source_x"])
        & np.isfinite(target["source_y"])
        & np.isfinite(target["source_step"])
        & (target["source_step"] >= target_reprojection_threshold)
    )
    geometry_quality_maps = _jacobian_geometry_quality_maps(
        target["source_x"].astype(np.float32),
        target["source_y"].astype(np.float32),
        target["support"],
    )
    metrics = _metrics(
        fused,
        support,
        target_geometry_valid,
        selected_step,
        choose_target,
        choose_donor,
        target_name,
        donor_name,
    )
    metrics["target_geometry_valid_pixels"] = int(np.count_nonzero(target_geometry_valid))
    metrics["target_geometry_valid_fraction_of_support"] = float(np.count_nonzero(target_geometry_valid) / max(np.count_nonzero(support), 1))
    metrics["target_reprojection_source_step_threshold"] = float(target_reprojection_threshold)
    metrics["target_source_step_stats"] = _finite_stats(target["source_step"][support & np.isfinite(target["source_step"])])
    return {
        "image": fused,
        "support": support,
        "quality": target_geometry_valid,
        "selected_step": selected_step,
        "source_x": target["source_x"].astype(np.float32).copy(),
        "source_y": target["source_y"].astype(np.float32).copy(),
        "reference_source_x": target["source_x"].astype(np.float32).copy(),
        "reference_source_y": target["source_y"].astype(np.float32).copy(),
        "reference_source_step": target["source_step"].astype(np.float32).copy(),
        "reference_geometry_valid": target_geometry_valid,
        "geometry_quality_maps": geometry_quality_maps,
        "selection": selection,
        "selected_source_role": selected_source_role,
        "target_name": target_name,
        "donor_name": donor_name,
        "metrics": metrics,
    }


def _resize_to(arr: np.ndarray, shape: tuple[int, int], is_mask: bool = False) -> np.ndarray:
    h, w = shape
    if arr.shape[:2] == (h, w):
        return arr.copy()
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    out = cv2.resize(arr.astype(np.float32), (w, h), interpolation=interp)
    if is_mask:
        return out > 0.5
    return out.astype(arr.dtype if arr.dtype != bool else np.float32)


def _normalize(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = values.astype(np.float32)
    valid = np.isfinite(arr)
    if mask is not None:
        valid &= mask
    if not np.any(valid):
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.percentile(arr[valid], 1))
    hi = float(np.percentile(arr[valid], 99))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return (out * 255).astype(np.uint8)


def _local_contrast(image: np.ndarray) -> np.ndarray:
    image_f = image.astype(np.float32)
    mean = cv2.blur(image_f, (17, 17))
    mean_sq = cv2.blur(image_f * image_f, (17, 17))
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)).astype(np.float32)


def _align_to_reference(moving: dict[str, np.ndarray], ref_shape: tuple[int, int], flip: bool) -> dict[str, np.ndarray]:
    aligned = {
        "image": _resize_to(moving["image"], ref_shape),
        "mask": _resize_to(moving["mask"], ref_shape, is_mask=True),
        "quality": _resize_to(moving["quality"], ref_shape, is_mask=True),
        "source_step": _resize_to(moving["source_step"], ref_shape),
        "source_x": _resize_to(moving["source_x"], ref_shape),
        "source_y": _resize_to(moving["source_y"], ref_shape),
    }
    if flip:
        for key in aligned:
            aligned[key] = np.fliplr(aligned[key])
    return aligned


def _alignment_score(reference: dict[str, np.ndarray], moving: dict[str, np.ndarray], flip_moving: bool) -> float:
    h, w = reference["image"].shape
    aligned = _align_to_reference(moving, (h, w), flip_moving)
    overlap = reference["mask"] & aligned["mask"]
    if np.count_nonzero(overlap) < 100:
        return -1.0
    l = reference["image"][overlap].astype(np.float32)
    r = aligned["image"][overlap].astype(np.float32)
    l -= float(np.mean(l))
    r -= float(np.mean(r))
    denom = float(np.linalg.norm(l) * np.linalg.norm(r))
    return float(np.dot(l, r) / denom) if denom > 0 else -1.0


def _fuse_to_reference(
    reference: dict[str, np.ndarray],
    donor: dict[str, np.ndarray],
    flip_donor: bool,
    reference_name: str,
    donor_name: str,
) -> dict[str, np.ndarray | dict[str, Any]]:
    h, w = reference["image"].shape
    donor_aligned = _align_to_reference(donor, (h, w), flip_donor)

    reference_step = np.where(reference["mask"] & np.isfinite(reference["source_step"]), reference["source_step"], -1.0)
    donor_step = np.where(donor_aligned["mask"] & np.isfinite(donor_aligned["source_step"]), donor_aligned["source_step"], -1.0)
    reference_contrast = _local_contrast(reference["image"])
    donor_contrast = _local_contrast(donor_aligned["image"])

    # Prefer geometrically sharper samples. Contrast breaks ties where both views
    # are above the minimum scale.
    reference_score = reference_step + 0.01 * reference_contrast
    donor_score = donor_step + 0.01 * donor_contrast
    choose_donor = donor_aligned["mask"] & ((donor_score > reference_score) | ~reference["mask"])
    choose_reference = reference["mask"] & ~choose_donor

    fused = np.zeros_like(reference["image"])
    fused[choose_reference] = reference["image"][choose_reference]
    fused[choose_donor] = donor_aligned["image"][choose_donor]
    support = choose_reference | choose_donor
    selected_step = np.where(choose_donor, donor_step, reference_step)
    source_x = np.where(choose_donor, donor_aligned["source_x"], reference["source_x"]).astype(np.float32)
    source_y = np.where(choose_donor, donor_aligned["source_y"], reference["source_y"]).astype(np.float32)
    source_x[~support] = np.nan
    source_y[~support] = np.nan
    reference_source_x = reference["source_x"].astype(np.float32).copy()
    reference_source_y = reference["source_y"].astype(np.float32).copy()
    reference_source_step = reference_step.astype(np.float32).copy()
    reference_source_x[~support] = np.nan
    reference_source_y[~support] = np.nan
    reference_source_step[~support] = np.nan
    reference_geometry_valid = (
        support
        & reference["mask"]
        & np.isfinite(reference_source_x)
        & np.isfinite(reference_source_y)
        & np.isfinite(reference_source_step)
        & (reference_source_step >= MIN_SOURCE_X_STEP)
    )
    quality = support & (selected_step >= MIN_SOURCE_X_STEP)
    selection = np.zeros_like(reference["image"], dtype=np.uint8)
    selection[choose_reference] = 85
    selection[choose_donor] = 200

    metrics = _metrics(
        fused,
        support,
        quality,
        selected_step,
        choose_reference,
        choose_donor,
        reference_name,
        donor_name,
    )
    return {
        "image": fused,
        "support": support,
        "quality": quality,
        "selected_step": selected_step.astype(np.float32),
        "source_x": source_x,
        "source_y": source_y,
        "reference_source_x": reference_source_x,
        "reference_source_y": reference_source_y,
        "reference_source_step": reference_source_step,
        "reference_geometry_valid": reference_geometry_valid,
        "selection": selection,
        "selected_source_role": np.where(choose_donor, 2, np.where(choose_reference, 1, 0)).astype(np.uint8),
        "donor_aligned_image": donor_aligned["image"],
        "reference_name": reference_name,
        "donor_name": donor_name,
        "flip_donor": flip_donor,
        "metrics": metrics,
    }


def _metrics(
    image: np.ndarray,
    support: np.ndarray,
    quality: np.ndarray,
    selected_step: np.ndarray,
    choose_reference: np.ndarray,
    choose_donor: np.ndarray,
    reference_name: str,
    donor_name: str,
) -> dict[str, Any]:
    valid = support & np.isfinite(selected_step)
    h, _w = support.shape
    seam_stats: dict[str, Any] = {}
    for name, selector in {
        "left_seam": lambda xs: xs[: min(SEAM_WIDTH, xs.size)],
        "right_seam": lambda xs: xs[-min(SEAM_WIDTH, xs.size) :],
        "center": lambda xs: xs[max(0, xs.size // 2 - SEAM_WIDTH // 2) : min(xs.size, xs.size // 2 + SEAM_WIDTH // 2)],
    }.items():
        steps: list[float] = []
        contrasts: list[float] = []
        low: list[float] = []
        for y in range(h):
            xs = np.flatnonzero(valid[y])
            if xs.size < 16:
                continue
            region = selector(xs)
            step = selected_step[y, region]
            finite = np.isfinite(step)
            if not np.any(finite):
                continue
            steps.append(float(np.median(step[finite])))
            contrasts.append(float(np.std(image[y, region].astype(np.float32))))
            low.append(float(np.count_nonzero(step[finite] < MIN_SOURCE_X_STEP) / np.count_nonzero(finite)))
        seam_stats[name] = {
            "source_x_step_median": float(np.median(steps)) if steps else 0.0,
            "intensity_std_median": float(np.median(contrasts)) if contrasts else 0.0,
            "low_source_step_fraction_median": float(np.median(low)) if low else 0.0,
        }
    return {
        "support_pixels": int(np.count_nonzero(support)),
        "quality_pixels": int(np.count_nonzero(quality)),
        "quality_fraction": float(np.count_nonzero(quality) / max(np.count_nonzero(support), 1)),
        f"{reference_name}_selected_fraction": float(np.count_nonzero(choose_reference) / max(np.count_nonzero(support), 1)),
        f"{donor_name}_selected_fraction": float(np.count_nonzero(choose_donor) / max(np.count_nonzero(support), 1)),
        "reference_selected_fraction": float(np.count_nonzero(choose_reference) / max(np.count_nonzero(support), 1)),
        "donor_selected_fraction": float(np.count_nonzero(choose_donor) / max(np.count_nonzero(support), 1)),
        "selected_source_step_median": float(np.nanmedian(selected_step[valid])) if np.any(valid) else 0.0,
        "selected_source_step_p05": float(np.nanpercentile(selected_step[valid], 5)) if np.any(valid) else 0.0,
        "selected_source_step_p95": float(np.nanpercentile(selected_step[valid], 95)) if np.any(valid) else 0.0,
        "seam_stats": seam_stats,
    }


def _normalize_angle_2pi(theta: float) -> float:
    value = float(theta) % (2.0 * math.pi)
    return value if value >= 0.0 else value + (2.0 * math.pi)


def _point_has_mask_support(mask: np.ndarray, x: float, y: float) -> bool:
    if mask.ndim != 2 or not (math.isfinite(x) and math.isfinite(y)):
        return False
    h, w = mask.shape
    cx = int(round(x))
    cy = int(round(y))
    return bool(0 <= cx < w and 0 <= cy < h and mask[cy, cx] > 0)


def _sample_2d_bilinear(array: np.ndarray, x: float, y: float, valid_mask: np.ndarray | None = None) -> float | None:
    if array.ndim != 2 or not (math.isfinite(x) and math.isfinite(y)):
        return None
    h, w = array.shape
    if x < 0.0 or y < 0.0 or x > float(w - 1) or y > float(h - 1):
        return None
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    samples: list[tuple[float, float]] = []
    for yy in (y0, y1):
        for xx in (x0, x1):
            if valid_mask is not None and not bool(valid_mask[yy, xx]):
                continue
            value = float(array[yy, xx])
            if not math.isfinite(value):
                continue
            weight = max(1.0 - abs(float(xx) - x), 0.0) * max(1.0 - abs(float(yy) - y), 0.0)
            if weight > 0.0:
                samples.append((weight, value))
    if not samples:
        return None
    weight_sum = sum(weight for weight, _value in samples)
    if weight_sum <= 1e-8:
        return None
    return float(sum(weight * value for weight, value in samples) / weight_sum)


def _map_fused_chart_to_pose(
    x: float,
    y: float,
    source_x_map: np.ndarray,
    source_y_map: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[float, float] | None:
    source_x = _sample_2d_bilinear(source_x_map, x, y, valid_mask=valid_mask)
    source_y = _sample_2d_bilinear(source_y_map, x, y, valid_mask=valid_mask)
    if source_x is None or source_y is None:
        return None
    return float(source_x), float(source_y)


def _standardize_pyfing_minutiae(raw_minutiae: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in raw_minutiae:
        theta = float(getattr(item, "direction", getattr(item, "angle", 0.0)))
        quality = getattr(item, "quality", None)
        rows.append(
            {
                "x": float(getattr(item, "x")),
                "y": float(getattr(item, "y")),
                "theta": _normalize_angle_2pi(theta),
                "score": float(quality) if quality is not None else 1.0,
                "type": str(getattr(item, "type", "")),
                "source": "pyfing_multiview_fused",
            }
        )
    return rows


def _write_minutiae_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y", "theta", "score", "type", "source"])
        writer.writeheader()
        writer.writerows(rows)


def _draw_minutiae_overlay(image: np.ndarray, minutiae: list[dict[str, Any]], mask: np.ndarray | None = None) -> np.ndarray:
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()
    if mask is not None and mask.shape == canvas.shape[:2]:
        tint = np.zeros_like(canvas)
        tint[:, :, 1] = np.where(mask > 0, 70, 0).astype(np.uint8)
        canvas = cv2.addWeighted(canvas, 1.0, tint, 0.45, 0)
    h, w = canvas.shape[:2]
    for item in minutiae:
        x = int(round(float(item["x"])))
        y = int(round(float(item["y"])))
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        theta = float(item["theta"])
        x2 = int(round(x + 16.0 * math.cos(theta)))
        y2 = int(round(y + 16.0 * math.sin(theta)))
        cv2.circle(canvas, (x, y), 3, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.line(canvas, (x, y), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _extract_pyfing_minutiae(gray: np.ndarray) -> list[dict[str, Any]]:
    return _standardize_pyfing_minutiae(pyfing.minutiae_extraction(gray, dpi=DEFAULT_DPI))


def _reproject_fused_minutiae_to_pose(
    minutiae: list[dict[str, Any]],
    source_x_map: np.ndarray,
    source_y_map: np.ndarray,
    support_mask: np.ndarray,
    quality_mask: np.ndarray,
    source_step_map: np.ndarray,
    pose_mask: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_map = support_mask & quality_mask & np.isfinite(source_x_map) & np.isfinite(source_y_map)
    pose_h, pose_w = pose_mask.shape
    reprojected: list[dict[str, Any]] = []
    accepted_steps: list[float] = []
    details: dict[str, Any] = {
        "extracted_count": int(len(minutiae)),
        "dropped_nonfinite_xy": 0,
        "dropped_outside_fused_mask": 0,
        "dropped_outside_quality_mask": 0,
        "dropped_no_source_map": 0,
        "dropped_outside_side_pose_bounds": 0,
        "dropped_no_side_mask_support": 0,
        "orientation_projected_count": 0,
        "orientation_fallback_count": 0,
        "reprojected_count": 0,
        "accepted_source_step_mean": 0.0,
        "accepted_source_step_median": 0.0,
    }

    for item in minutiae:
        x = float(item.get("x", float("nan")))
        y = float(item.get("y", float("nan")))
        if not (math.isfinite(x) and math.isfinite(y)):
            details["dropped_nonfinite_xy"] += 1
            continue
        if not _point_has_mask_support(support_mask, x, y):
            details["dropped_outside_fused_mask"] += 1
            continue
        if not _point_has_mask_support(quality_mask, x, y):
            details["dropped_outside_quality_mask"] += 1
            continue

        center = _map_fused_chart_to_pose(x, y, source_x_map, source_y_map, valid_map)
        if center is None:
            details["dropped_no_source_map"] += 1
            continue
        x_pose, y_pose = center
        if x_pose < 0.0 or y_pose < 0.0 or x_pose >= float(pose_w) or y_pose >= float(pose_h):
            details["dropped_outside_side_pose_bounds"] += 1
            continue
        if not _point_has_mask_support(pose_mask, x_pose, y_pose):
            details["dropped_no_side_mask_support"] += 1
            continue

        theta = float(item.get("theta", 0.0))
        dx = math.cos(theta) * ORIENTATION_DELTA_PX
        dy = math.sin(theta) * ORIENTATION_DELTA_PX
        forward = _map_fused_chart_to_pose(x + dx, y + dy, source_x_map, source_y_map, valid_map)
        backward = _map_fused_chart_to_pose(x - dx, y - dy, source_x_map, source_y_map, valid_map)
        projected_points = [point for point in (forward, backward) if point is not None]

        theta_pose = theta
        if len(projected_points) == 2:
            dx_proj = float(projected_points[0][0] - projected_points[1][0])
            dy_proj = float(projected_points[0][1] - projected_points[1][1])
            baseline = math.hypot(dx_proj, dy_proj)
            if baseline >= MIN_ORIENTATION_BASELINE_PX:
                theta_pose = math.atan2(dy_proj, dx_proj)
                details["orientation_projected_count"] += 1
            else:
                details["orientation_fallback_count"] += 1
        elif len(projected_points) == 1:
            dx_proj = float(projected_points[0][0] - x_pose)
            dy_proj = float(projected_points[0][1] - y_pose)
            baseline = math.hypot(dx_proj, dy_proj)
            if baseline >= MIN_ORIENTATION_BASELINE_PX:
                theta_pose = math.atan2(dy_proj, dx_proj)
                details["orientation_projected_count"] += 1
            else:
                details["orientation_fallback_count"] += 1
        else:
            details["orientation_fallback_count"] += 1

        step = _sample_2d_bilinear(source_step_map, x, y, valid_mask=support_mask)
        if step is not None:
            accepted_steps.append(float(step))
        reprojected.append(
            {
                "x": float(x_pose),
                "y": float(y_pose),
                "theta": _normalize_angle_2pi(theta_pose),
                "score": item.get("score"),
                "type": item.get("type"),
                "source": str(item.get("source", "pyfing_multiview_fused")) + "_reprojected_pose",
                "fused_x": x,
                "fused_y": y,
            }
        )

    details["reprojected_count"] = int(len(reprojected))
    if accepted_steps:
        step_values = np.asarray(accepted_steps, dtype=np.float32)
        details["accepted_source_step_mean"] = float(np.mean(step_values))
        details["accepted_source_step_median"] = float(np.median(step_values))
    return reprojected, details


def _axial_angle_error_deg(theta_a: np.ndarray, theta_b: np.ndarray) -> np.ndarray:
    diff = (theta_a - theta_b + math.pi / 2.0) % math.pi - math.pi / 2.0
    return np.degrees(np.abs(diff)).astype(np.float32)


def _dense_orientation_for_pose(output_dir: Path, role: str, pose_image: np.ndarray, pose_mask: np.ndarray) -> np.ndarray:
    cache_path = output_dir / f"{role}_pose_orientation_pyfing.npy"
    if cache_path.exists():
        return np.load(cache_path).astype(np.float32)
    orientation = pyfing.orientation_field_estimation(
        pose_image,
        pose_mask.astype(np.uint8) * 255,
        dpi=DEFAULT_DPI,
        method="SNFOE",
    ).astype(np.float32)
    orientation[~pose_mask.astype(bool)] = 0.0
    np.save(cache_path, orientation)
    return orientation


def _dense_vs_minutia_orientation_stats(
    output_dir: Path,
    role: str,
    pose_image: np.ndarray,
    pose_mask: np.ndarray,
    minutiae: list[dict[str, Any]],
) -> dict[str, Any]:
    if not minutiae:
        return {
            "count": 0,
            "sampled_count": 0,
            "mean_error_deg": 0.0,
            "median_error_deg": 0.0,
            "within_15_deg_fraction": 0.0,
        }
    dense = _dense_orientation_for_pose(output_dir, role, pose_image, pose_mask)
    errors: list[float] = []
    h, w = pose_mask.shape
    for item in minutiae:
        x = int(round(float(item.get("x", float("nan")))))
        y = int(round(float(item.get("y", float("nan")))))
        if x < 0 or y < 0 or x >= w or y >= h or not pose_mask[y, x]:
            continue
        theta = float(item.get("theta", 0.0))
        errors.append(float(_axial_angle_error_deg(np.asarray([theta]), np.asarray([float(dense[y, x])]))[0]))
    values = np.asarray(errors, dtype=np.float32)
    return {
        "count": int(len(minutiae)),
        "sampled_count": int(values.size),
        "mean_error_deg": float(np.mean(values)) if values.size else 0.0,
        "median_error_deg": float(np.median(values)) if values.size else 0.0,
        "within_15_deg_fraction": float(np.count_nonzero(values <= 15.0) / max(values.size, 1)),
    }


def _annotate_minutiae_with_geometry(
    minutiae: list[dict[str, Any]],
    dense_orientation: np.ndarray,
    pose_mask: np.ndarray,
    geometry_maps: dict[str, np.ndarray],
    source_step_map: np.ndarray,
    selected_source_role: np.ndarray,
    support_mask: np.ndarray,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    pose_h, pose_w = pose_mask.shape
    for item in minutiae:
        row = dict(item)
        pose_x = int(round(float(row.get("x", float("nan")))))
        pose_y = int(round(float(row.get("y", float("nan")))))
        fused_x = float(row.get("fused_x", float("nan")))
        fused_y = float(row.get("fused_y", float("nan")))
        if 0 <= pose_x < pose_w and 0 <= pose_y < pose_h and pose_mask[pose_y, pose_x]:
            error = float(_axial_angle_error_deg(np.asarray([row["theta"]]), np.asarray([float(dense_orientation[pose_y, pose_x])]))[0])
        else:
            error = float("nan")

        geometry: dict[str, Any] = {"orientation_error_deg": error}
        for key in ("scale_x", "scale_y", "area", "anisotropy", "condition", "jacobian_change", "local_jacobian_std"):
            value = _sample_2d_bilinear(geometry_maps[key], fused_x, fused_y, valid_mask=geometry_maps["jacobian_valid"])
            geometry[key] = float(value) if value is not None else None
        step = _sample_2d_bilinear(source_step_map, fused_x, fused_y, valid_mask=support_mask)
        geometry["source_step"] = float(step) if step is not None else None
        if math.isfinite(fused_x) and math.isfinite(fused_y):
            fx = int(round(fused_x))
            fy = int(round(fused_y))
            if 0 <= fy < selected_source_role.shape[0] and 0 <= fx < selected_source_role.shape[1]:
                geometry["selected_source_role"] = int(selected_source_role[fy, fx])
            else:
                geometry["selected_source_role"] = None
        else:
            geometry["selected_source_role"] = None
        row["geometry_quality"] = geometry
        annotated.append(row)
    return annotated


def _geometry_error_correlation(annotated_minutiae: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("source_step", "scale_x", "scale_y", "area", "anisotropy", "condition", "jacobian_change", "local_jacobian_std")
    errors: list[float] = []
    values: dict[str, list[float]] = {key: [] for key in metrics}
    selected_roles: list[int] = []
    for item in annotated_minutiae:
        geometry = item.get("geometry_quality", {})
        error = geometry.get("orientation_error_deg")
        if error is None or not math.isfinite(float(error)):
            continue
        errors.append(float(error))
        for key in metrics:
            value = geometry.get(key)
            values[key].append(float(value) if value is not None and math.isfinite(float(value)) else float("nan"))
        role = geometry.get("selected_source_role")
        if role is not None:
            selected_roles.append(int(role))

    error_array = np.asarray(errors, dtype=np.float32)
    correlations = {
        key: _correlation(np.asarray(vals, dtype=np.float32), error_array)
        for key, vals in values.items()
    }
    ranked = sorted(correlations, key=lambda key: correlations[key]["abs_spearman"], reverse=True)
    by_source_role: dict[str, Any] = {}
    if selected_roles and len(selected_roles) == len(error_array):
        roles = np.asarray(selected_roles, dtype=np.int32)
        for role_value, label in ((1, "target_texture"), (2, "donor_texture")):
            role_errors = error_array[roles == role_value]
            by_source_role[label] = {
                "count": int(role_errors.size),
                "mean_error_deg": float(np.mean(role_errors)) if role_errors.size else 0.0,
                "within_15_deg_fraction": float(np.count_nonzero(role_errors <= 15.0) / max(role_errors.size, 1)),
            }
    return {
        "count": int(error_array.size),
        "error_stats": {
            "mean": float(np.mean(error_array)) if error_array.size else 0.0,
            "median": float(np.median(error_array)) if error_array.size else 0.0,
            "p90": float(np.percentile(error_array, 90)) if error_array.size else 0.0,
            "within_15_deg_fraction": float(np.count_nonzero(error_array <= 15.0) / max(error_array.size, 1)),
        },
        "correlations": correlations,
        "ranked_by_abs_spearman": ranked,
        "best_proxy_by_abs_spearman": ranked[0] if ranked else None,
        "by_selected_source_role": by_source_role,
    }


def _combine_mismatch_stats(role_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    # Keep the aggregate count-weighted. The per-role stats remain the source of
    # truth when one side has much fewer accepted minutiae after strict geometry.
    sampled = sum(int(stats["sampled_count"]) for stats in role_stats.values())
    if sampled == 0:
        return {"sampled_count": 0, "mean_error_deg": 0.0, "within_15_deg_fraction": 0.0}
    mean = sum(float(stats["mean_error_deg"]) * int(stats["sampled_count"]) for stats in role_stats.values()) / sampled
    within = sum(float(stats["within_15_deg_fraction"]) * int(stats["sampled_count"]) for stats in role_stats.values()) / sampled
    return {"sampled_count": int(sampled), "mean_error_deg": float(mean), "within_15_deg_fraction": float(within)}


def _threshold_sweep_for_role(
    output_dir: Path,
    role: str,
    extracted_minutiae: list[dict[str, Any]],
    fused: dict[str, Any],
    pose_image: np.ndarray,
    pose_mask: np.ndarray,
    geometry_maps: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dense_orientation = _dense_orientation_for_pose(output_dir, role, pose_image, pose_mask)
    for threshold in SOURCE_STEP_SWEEP_THRESHOLDS:
        quality = (
            fused["support"]
            & np.isfinite(fused["reference_source_step"])
            & (fused["reference_source_step"] >= float(threshold))
        )
        reprojected, details = _reproject_fused_minutiae_to_pose(
            extracted_minutiae,
            fused["reference_source_x"],
            fused["reference_source_y"],
            fused["support"],
            quality,
            fused["reference_source_step"],
            pose_mask,
        )
        mismatch = _dense_vs_minutia_orientation_stats(output_dir, role, pose_image, pose_mask, reprojected)
        annotated = _annotate_minutiae_with_geometry(
            reprojected,
            dense_orientation,
            pose_mask,
            geometry_maps,
            fused["reference_source_step"],
            fused["selected_source_role"],
            fused["support"],
        )
        rows.append(
            {
                "target_source_step_threshold": float(threshold),
                "reprojection": details,
                "dense_vs_minutia_orientation": mismatch,
                "geometry_error_correlation": _geometry_error_correlation(annotated),
            }
        )
    return rows


def _draw_label(panel: np.ndarray, text: str) -> np.ndarray:
    if panel.ndim == 2:
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
    panel = panel.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (18, 18, 18), -1)
    cv2.putText(panel, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return panel


def _fit(image: np.ndarray, size: tuple[int, int] = (320, 250)) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    tw, th = size
    h, w = image.shape[:2]
    scale = min(tw / max(w, 1), th / max(h, 1))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((th, tw, 3), dtype=np.uint8)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def run(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_dir = _infer_reconstruction_dir(input_dir)
    debug_views_dir = reconstruction_dir / "debug_views"

    views, shared_sources, shared_debug = _build_shared_surface_sources(reconstruction_dir)
    fused_by_frame = {
        "left": _fuse_shared_for_target(shared_sources["left"], shared_sources["right"], "left", "right"),
        "right": _fuse_shared_for_target(shared_sources["right"], shared_sources["left"], "right", "left"),
    }

    source_outputs: dict[str, dict[str, str]] = {}
    for role, source in shared_sources.items():
        prefix = f"{role}_shared_surface"
        image_path = output_dir / f"{prefix}.png"
        mask_path = output_dir / f"{prefix}_mask.png"
        quality_path = output_dir / f"{prefix}_quality_mask.png"
        step_path = output_dir / f"{prefix}_source_step.png"
        coord_path = output_dir / f"{prefix}_coordinate_debug.npz"
        cv2.imwrite(str(image_path), source["image"])
        cv2.imwrite(str(mask_path), source["support"].astype(np.uint8) * 255)
        cv2.imwrite(str(quality_path), source["quality"].astype(np.uint8) * 255)
        cv2.imwrite(str(step_path), _normalize(source["source_step"], source["support"]))
        np.savez_compressed(
            coord_path,
            source_x_map=source["source_x"].astype(np.float32),
            source_y_map=source["source_y"].astype(np.float32),
            source_step_map=source["source_step"].astype(np.float32),
            support_mask=source["support"].astype(np.uint8),
            quality_mask=source["quality"].astype(np.uint8),
            expected_mask=source["expected"].astype(np.uint8),
            shared_y_to_side_y=source["shared_y_to_side_y"].astype(np.int32),
            shared_s_min=np.float32(shared_debug["chart"]["s_min"]),
            shared_s_max=np.float32(shared_debug["chart"]["s_max"]),
            shared_s_reversed=np.asarray([bool(SHARED_S_REVERSE_BY_ROLE.get(role, False))], dtype=np.uint8),
        )
        source_outputs[role] = {
            "shared_surface": str(image_path),
            "support_mask": str(mask_path),
            "quality_mask": str(quality_path),
            "source_step": str(step_path),
            "coordinate_debug": str(coord_path),
        }

    legacy_alignment_scores: dict[str, Any] = {"available": False}
    try:
        left_legacy = _load_role(input_dir, "left")
        right_legacy = _load_role(input_dir, "right")
        legacy_alignment_scores = {
            "available": True,
            "left_frame_right_donor": {
                "donor_as_is": _alignment_score(left_legacy, right_legacy, False),
                "donor_flipped": _alignment_score(left_legacy, right_legacy, True),
            },
            "right_frame_left_donor": {
                "donor_as_is": _alignment_score(right_legacy, left_legacy, False),
                "donor_flipped": _alignment_score(right_legacy, left_legacy, True),
            },
        }
    except FileNotFoundError:
        pass

    outputs: dict[str, dict[str, str]] = {}
    mismatch_by_role: dict[str, dict[str, Any]] = {}
    geometry_correlation_by_role: dict[str, dict[str, Any]] = {}
    threshold_sweep: dict[str, list[dict[str, Any]]] = {}
    panel_rows: list[np.ndarray] = []
    for frame, fused in fused_by_frame.items():
        donor = str(fused["donor_name"])
        prefix = f"{frame}_shared_surface_fused"
        legacy_prefix = f"{frame}_multiview_side_fused"
        fused_path = output_dir / f"{prefix}.png"
        legacy_fused_path = output_dir / f"{legacy_prefix}.png"
        support_path = output_dir / f"{prefix}_mask.png"
        legacy_support_path = output_dir / f"{legacy_prefix}_mask.png"
        quality_path = output_dir / f"{prefix}_quality_mask.png"
        legacy_quality_path = output_dir / f"{legacy_prefix}_quality_mask.png"
        step_path = output_dir / f"{prefix}_source_step.png"
        legacy_step_path = output_dir / f"{legacy_prefix}_source_step.png"
        selection_path = output_dir / f"{prefix}_selection_map.png"
        legacy_selection_path = output_dir / f"{legacy_prefix}_selection_map.png"
        jacobian_area_path = output_dir / f"{prefix}_jacobian_area.png"
        jacobian_anisotropy_path = output_dir / f"{prefix}_jacobian_anisotropy.png"
        jacobian_condition_path = output_dir / f"{prefix}_jacobian_condition.png"
        jacobian_change_path = output_dir / f"{prefix}_jacobian_change.png"
        coord_debug_path = output_dir / f"{prefix}_coordinate_debug.npz"
        legacy_coord_debug_path = output_dir / f"{legacy_prefix}_coordinate_debug.npz"
        minutiae_json_path = output_dir / f"{frame}_shared_surface_fused_minutiae.json"
        legacy_minutiae_json_path = output_dir / f"{frame}_multiview_fused_minutiae.json"
        minutiae_csv_path = output_dir / f"{frame}_shared_surface_fused_minutiae.csv"
        legacy_minutiae_csv_path = output_dir / f"{frame}_multiview_fused_minutiae.csv"
        minutiae_overlay_path = output_dir / f"{frame}_shared_surface_fused_minutiae_overlay.png"
        legacy_minutiae_overlay_path = output_dir / f"{frame}_multiview_fused_minutiae_overlay.png"
        reprojected_json_path = output_dir / f"{frame}_shared_surface_reprojected_minutiae.json"
        legacy_reprojected_json_path = output_dir / f"{frame}_reprojected_minutiae.json"
        reprojected_overlay_path = output_dir / f"{frame}_shared_surface_reprojected_minutiae_overlay.png"
        legacy_reprojected_overlay_path = output_dir / f"{frame}_reprojected_minutiae_overlay.png"
        cv2.imwrite(str(fused_path), fused["image"])
        cv2.imwrite(str(legacy_fused_path), fused["image"])
        cv2.imwrite(str(support_path), fused["support"].astype(np.uint8) * 255)
        cv2.imwrite(str(legacy_support_path), fused["support"].astype(np.uint8) * 255)
        cv2.imwrite(str(quality_path), fused["quality"].astype(np.uint8) * 255)
        cv2.imwrite(str(legacy_quality_path), fused["quality"].astype(np.uint8) * 255)
        cv2.imwrite(str(step_path), _normalize(fused["selected_step"], fused["support"]))
        cv2.imwrite(str(legacy_step_path), _normalize(fused["selected_step"], fused["support"]))
        cv2.imwrite(str(selection_path), fused["selection"])
        cv2.imwrite(str(legacy_selection_path), fused["selection"])
        geometry_maps = fused["geometry_quality_maps"]
        jacobian_valid = geometry_maps["jacobian_valid"]
        cv2.imwrite(str(jacobian_area_path), _normalize(geometry_maps["area"], jacobian_valid))
        cv2.imwrite(str(jacobian_anisotropy_path), _normalize(geometry_maps["anisotropy"], jacobian_valid))
        cv2.imwrite(str(jacobian_condition_path), _normalize(geometry_maps["condition"], jacobian_valid))
        cv2.imwrite(str(jacobian_change_path), _normalize(geometry_maps["jacobian_change"], jacobian_valid))
        for path in (coord_debug_path, legacy_coord_debug_path):
            np.savez_compressed(
                path,
                source_x_map=fused["source_x"].astype(np.float32),
                source_y_map=fused["source_y"].astype(np.float32),
                source_step_map=fused["selected_step"].astype(np.float32),
                target_source_x_map=fused["reference_source_x"].astype(np.float32),
                target_source_y_map=fused["reference_source_y"].astype(np.float32),
                target_source_step_map=fused["reference_source_step"].astype(np.float32),
                reference_source_x_map=fused["reference_source_x"].astype(np.float32),
                reference_source_y_map=fused["reference_source_y"].astype(np.float32),
                reference_source_step_map=fused["reference_source_step"].astype(np.float32),
                left_source_x_map=shared_sources["left"]["source_x"].astype(np.float32),
                left_source_y_map=shared_sources["left"]["source_y"].astype(np.float32),
                left_source_step_map=shared_sources["left"]["source_step"].astype(np.float32),
                right_source_x_map=shared_sources["right"]["source_x"].astype(np.float32),
                right_source_y_map=shared_sources["right"]["source_y"].astype(np.float32),
                right_source_step_map=shared_sources["right"]["source_step"].astype(np.float32),
                jacobian_scale_x_map=geometry_maps["scale_x"].astype(np.float32),
                jacobian_scale_y_map=geometry_maps["scale_y"].astype(np.float32),
                jacobian_area_map=geometry_maps["area"].astype(np.float32),
                jacobian_area_signed_map=geometry_maps["area_signed"].astype(np.float32),
                jacobian_anisotropy_map=geometry_maps["anisotropy"].astype(np.float32),
                jacobian_condition_map=geometry_maps["condition"].astype(np.float32),
                jacobian_change_map=geometry_maps["jacobian_change"].astype(np.float32),
                jacobian_local_std_map=geometry_maps["local_jacobian_std"].astype(np.float32),
                jacobian_valid_mask=geometry_maps["jacobian_valid"].astype(np.uint8),
                support_mask=fused["support"].astype(np.uint8),
                quality_mask=fused["quality"].astype(np.uint8),
                reference_geometry_valid_mask=fused["reference_geometry_valid"].astype(np.uint8),
                selected_source_role=fused["selected_source_role"].astype(np.uint8),
                selected_source_role_legend=np.asarray(["none", str(frame), str(donor)]),
                shared_chart_shape_hw=np.asarray(shared_debug["chart"]["shape_hw"], dtype=np.int32),
                shared_s_min=np.float32(shared_debug["chart"]["s_min"]),
                shared_s_max=np.float32(shared_debug["chart"]["s_max"]),
                target_shared_s_reversed=np.asarray([bool(SHARED_S_REVERSE_BY_ROLE.get(frame, False))], dtype=np.uint8),
                fusion_mode=np.asarray(["true_shared_surface_coordinate_chart"]),
            )

        extracted_minutiae = _extract_pyfing_minutiae(fused["image"])
        minutiae_payload = {"image": str(fused_path), "minutiae": extracted_minutiae}
        minutiae_json_path.write_text(json.dumps(minutiae_payload, indent=2), encoding="utf-8")
        legacy_minutiae_json_path.write_text(json.dumps(minutiae_payload, indent=2), encoding="utf-8")
        _write_minutiae_csv(extracted_minutiae, minutiae_csv_path)
        _write_minutiae_csv(extracted_minutiae, legacy_minutiae_csv_path)
        minutiae_overlay = _draw_minutiae_overlay(fused["image"], extracted_minutiae, fused["quality"].astype(np.uint8))
        cv2.imwrite(str(minutiae_overlay_path), minutiae_overlay)
        cv2.imwrite(str(legacy_minutiae_overlay_path), minutiae_overlay)

        pose_image = _read_gray(debug_views_dir / f"{frame}_pose_normalized.png")
        pose_mask = _read_gray(debug_views_dir / f"{frame}_pose_mask.png") > 0
        reprojected_minutiae, reproject_details = _reproject_fused_minutiae_to_pose(
            extracted_minutiae,
            fused["reference_source_x"],
            fused["reference_source_y"],
            fused["support"],
            fused["reference_geometry_valid"],
            fused["reference_source_step"],
            pose_mask,
        )
        reproject_details["geometry_policy"] = "target_role_shared_surface_geometry_for_all_texture_sources"
        dense_orientation = _dense_orientation_for_pose(output_dir, frame, pose_image, pose_mask)
        reprojected_minutiae = _annotate_minutiae_with_geometry(
            reprojected_minutiae,
            dense_orientation,
            pose_mask,
            geometry_maps,
            fused["reference_source_step"],
            fused["selected_source_role"],
            fused["support"],
        )
        mismatch = _dense_vs_minutia_orientation_stats(output_dir, frame, pose_image, pose_mask, reprojected_minutiae)
        mismatch_by_role[frame] = mismatch
        geometry_correlation_by_role[frame] = _geometry_error_correlation(reprojected_minutiae)
        threshold_sweep[frame] = _threshold_sweep_for_role(
            output_dir,
            frame,
            extracted_minutiae,
            fused,
            pose_image,
            pose_mask,
            geometry_maps,
        )
        reproject_payload = {
            "target": f"{frame}_pose_normalized",
            "geometry_policy": reproject_details["geometry_policy"],
            "minutiae": reprojected_minutiae,
        }
        reprojected_json_path.write_text(json.dumps(reproject_payload, indent=2), encoding="utf-8")
        legacy_reprojected_json_path.write_text(json.dumps(reproject_payload, indent=2), encoding="utf-8")
        reprojected_overlay = _draw_minutiae_overlay(pose_image, reprojected_minutiae, pose_mask.astype(np.uint8))
        cv2.imwrite(str(reprojected_overlay_path), reprojected_overlay)
        cv2.imwrite(str(legacy_reprojected_overlay_path), reprojected_overlay)
        outputs[frame] = {
            "fused": str(fused_path),
            "legacy_fused": str(legacy_fused_path),
            "support_mask": str(support_path),
            "quality_mask": str(quality_path),
            "source_step": str(step_path),
            "selection_map": str(selection_path),
            "jacobian_area": str(jacobian_area_path),
            "jacobian_anisotropy": str(jacobian_anisotropy_path),
            "jacobian_condition": str(jacobian_condition_path),
            "jacobian_change": str(jacobian_change_path),
            "coordinate_debug": str(coord_debug_path),
            "fused_minutiae_json": str(minutiae_json_path),
            "fused_minutiae_csv": str(minutiae_csv_path),
            "fused_minutiae_overlay": str(minutiae_overlay_path),
            "reprojected_minutiae_json": str(reprojected_json_path),
            "reprojected_minutiae_overlay": str(reprojected_overlay_path),
        }
        fused["metrics"]["minutiae_reprojection"] = reproject_details
        target_source = shared_sources[frame]
        donor_source = shared_sources[donor]
        selection_color = cv2.applyColorMap(fused["selection"], cv2.COLORMAP_VIRIDIS)
        selection_color[fused["selected_source_role"] == 0] = 0
        panel_rows.append(
            np.hstack(
                [
                    _draw_label(_fit(views[frame].image), f"{frame} pose"),
                    _draw_label(_fit(views[donor].image), f"{donor} pose"),
                    _draw_label(_fit(target_source["image"]), f"{frame} shared chart"),
                    _draw_label(_fit(donor_source["image"]), f"{donor} shared chart"),
                    _draw_label(_fit(fused["image"]), f"{frame} shared fused"),
                    _draw_label(_fit(selection_color), f"dark={frame} bright={donor}"),
                    _draw_label(_fit(fused["quality"].astype(np.uint8) * 255), f"{frame} target geom"),
                    _draw_label(_fit(reprojected_overlay), f"{frame} reproj n={len(reprojected_minutiae)}"),
                ]
            )
        )

    contact_path = output_dir / "multiview_side_fusion_contact_sheet.png"
    shared_contact_path = output_dir / "shared_surface_side_fusion_contact_sheet.png"
    cv2.imwrite(str(contact_path), np.vstack(panel_rows))
    cv2.imwrite(str(shared_contact_path), np.vstack(panel_rows))
    dense_mismatch = {**mismatch_by_role, "all": _combine_mismatch_stats(mismatch_by_role)}

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "reconstruction_dir": str(reconstruction_dir),
        "fusion_mode": "true_shared_surface_coordinate_chart",
        "legacy_flip_resize_alignment_scores": legacy_alignment_scores,
        "alignment_scores": legacy_alignment_scores,
        "donor_flip_by_frame": {"left": False, "right": False},
        "min_source_x_step": MIN_SOURCE_X_STEP,
        "target_reprojection_source_step_by_role": TARGET_REPROJECTION_SOURCE_STEP_BY_ROLE,
        "source_step_sweep_thresholds": list(SOURCE_STEP_SWEEP_THRESHOLDS),
        "shared_chart": shared_debug,
        "outputs": {
            **outputs,
            "shared_sources": source_outputs,
            "contact_sheet": str(contact_path),
            "shared_contact_sheet": str(shared_contact_path),
        },
        "metrics": {frame: fused["metrics"] for frame, fused in fused_by_frame.items()},
        "dense_vs_minutia_orientation": dense_mismatch,
        "geometry_error_correlation": geometry_correlation_by_role,
        "threshold_sweep": threshold_sweep,
        "warnings": [],
    }
    if (
        shared_debug["roles"]["left"]["selected_projection_sign"]
        == shared_debug["roles"]["right"]["selected_projection_sign"]
    ):
        report["warnings"].append("Both roles selected the same projection sign for the shared chart.")
    report_path = output_dir / "multiview_side_fusion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "tmp" / "algorithm1_surface_unwrap_fixed" / "s01_f01_a01")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "tmp" / "multiview_side_texture_fusion" / "s01_f01_a01")
    args = parser.parse_args()
    report = run(args.input_dir, args.output_dir)
    print(json.dumps({
        "output_dir": report["output_dir"],
        "contact_sheet": report["outputs"]["contact_sheet"],
        "left_fused": report["outputs"]["left"]["fused"],
        "right_fused": report["outputs"]["right"]["fused"],
        "fusion_mode": report["fusion_mode"],
        "dense_vs_minutia_orientation": report["dense_vs_minutia_orientation"],
        "metrics": report["metrics"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

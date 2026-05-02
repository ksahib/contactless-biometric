#!/usr/bin/env python
"""Pyfing minutiae reprojection diagnostic for Algorithm-1 depth side unwraps.

This script intentionally does not modify algorithm1_side_depth_unwrap_fixed_v4.
It imports that diagnostic and calls its run(...) function as-is, then extracts
pyfing minutiae on the generated side depth unwraps and reprojects them back to
left/right pose-normalized side coordinates through the saved unwrap maps.
"""

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

# Avoid the repo's copy.py shadowing stdlib copy while importing cv2/pyfing.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT]

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyfing  # noqa: E402


DEFAULT_DPI = 500
ORIENTATION_DELTA_PX = 4.0
MIN_ORIENTATION_BASELINE_PX = 1.5


def _load_depth_unwrap_module() -> Any:
    module_path = REPO_ROOT / "scripts" / "algorithm1_side_depth_unwrap_fixed_v4.py"
    spec = importlib.util.spec_from_file_location("algorithm1_side_depth_unwrap_fixed_v4", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _normalize_angle_2pi(theta: float) -> float:
    value = float(theta) % (2.0 * math.pi)
    return value if value >= 0.0 else value + (2.0 * math.pi)


def _axial_angle_error_deg(theta_a: float, theta_b: float) -> float:
    diff = (float(theta_a) - float(theta_b) + math.pi / 2.0) % math.pi - math.pi / 2.0
    return abs(math.degrees(diff))


def _point_has_mask_support(mask: np.ndarray, x: float, y: float) -> bool:
    if mask.ndim != 2 or not (math.isfinite(x) and math.isfinite(y)):
        return False
    h, w = mask.shape
    ix = int(round(x))
    iy = int(round(y))
    return bool(0 <= ix < w and 0 <= iy < h and mask[iy, ix] > 0)


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


def _map_chart_to_pose(
    x: float,
    y: float,
    source_x: np.ndarray,
    source_y: np.ndarray,
    valid_map: np.ndarray,
) -> tuple[float, float] | None:
    pose_x = _sample_2d_bilinear(source_x, x, y, valid_mask=valid_map)
    pose_y = _sample_2d_bilinear(source_y, x, y, valid_mask=valid_map)
    if pose_x is None or pose_y is None:
        return None
    return pose_x, pose_y


def _standardize_pyfing_minutiae(raw_minutiae: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in raw_minutiae:
        quality = getattr(item, "quality", None)
        rows.append(
            {
                "x": float(getattr(item, "x")),
                "y": float(getattr(item, "y")),
                "theta": _normalize_angle_2pi(float(getattr(item, "direction", getattr(item, "angle", 0.0)))),
                "score": float(quality) if quality is not None else 1.0,
                "type": str(getattr(item, "type", "")),
                "source": "pyfing_algorithm1_depth_unwrap",
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


def _score_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    scores = np.asarray([float(row.get("score", 0.0)) for row in rows], dtype=np.float32)
    if scores.size == 0:
        return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "p25": float(np.percentile(scores, 25)),
        "p75": float(np.percentile(scores, 75)),
    }


def _extract_pyfing_minutiae(image: np.ndarray) -> list[dict[str, Any]]:
    return _standardize_pyfing_minutiae(pyfing.minutiae_extraction(image, dpi=DEFAULT_DPI))


def _reproject_minutiae(
    minutiae: list[dict[str, Any]],
    source_x: np.ndarray,
    source_y: np.ndarray,
    unwrap_mask: np.ndarray,
    pose_mask: np.ndarray,
    orientation_delta_px: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_map = unwrap_mask.astype(bool) & np.isfinite(source_x) & np.isfinite(source_y)
    pose_h, pose_w = pose_mask.shape
    reprojected: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "extracted_count": int(len(minutiae)),
        "dropped_nonfinite_xy": 0,
        "dropped_outside_unwrap_mask": 0,
        "dropped_no_source_map": 0,
        "dropped_outside_pose_bounds": 0,
        "dropped_no_pose_mask_support": 0,
        "orientation_projected_count": 0,
        "orientation_fallback_count": 0,
        "orientation_delta_px": float(orientation_delta_px),
        "reprojected_count": 0,
    }

    for item in minutiae:
        x = float(item.get("x", float("nan")))
        y = float(item.get("y", float("nan")))
        if not (math.isfinite(x) and math.isfinite(y)):
            details["dropped_nonfinite_xy"] += 1
            continue
        if not _point_has_mask_support(unwrap_mask, x, y):
            details["dropped_outside_unwrap_mask"] += 1
            continue
        center = _map_chart_to_pose(x, y, source_x, source_y, valid_map)
        if center is None:
            details["dropped_no_source_map"] += 1
            continue
        pose_x, pose_y = center
        if pose_x < 0.0 or pose_y < 0.0 or pose_x >= float(pose_w) or pose_y >= float(pose_h):
            details["dropped_outside_pose_bounds"] += 1
            continue
        if not _point_has_mask_support(pose_mask, pose_x, pose_y):
            details["dropped_no_pose_mask_support"] += 1
            continue

        theta = float(item.get("theta", 0.0))
        dx = math.cos(theta) * float(orientation_delta_px)
        dy = math.sin(theta) * float(orientation_delta_px)
        forward = _map_chart_to_pose(x + dx, y + dy, source_x, source_y, valid_map)
        backward = _map_chart_to_pose(x - dx, y - dy, source_x, source_y, valid_map)
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
            dx_proj = float(projected_points[0][0] - pose_x)
            dy_proj = float(projected_points[0][1] - pose_y)
            baseline = math.hypot(dx_proj, dy_proj)
            if baseline >= MIN_ORIENTATION_BASELINE_PX:
                theta_pose = math.atan2(dy_proj, dx_proj)
                details["orientation_projected_count"] += 1
            else:
                details["orientation_fallback_count"] += 1
        else:
            details["orientation_fallback_count"] += 1

        reprojected.append(
            {
                "x": float(pose_x),
                "y": float(pose_y),
                "theta": _normalize_angle_2pi(theta_pose),
                "score": item.get("score"),
                "type": item.get("type"),
                "source": str(item.get("source", "pyfing_algorithm1_depth_unwrap")) + "_reprojected_pose",
                "unwrap_x": x,
                "unwrap_y": y,
            }
        )

    details["reprojected_count"] = int(len(reprojected))
    return reprojected, details


def _dense_orientation_stats(
    output_dir: Path,
    role: str,
    pose_image: np.ndarray,
    pose_mask: np.ndarray,
    minutiae: list[dict[str, Any]],
) -> dict[str, Any]:
    cache_path = output_dir / f"{role}_pose_orientation_pyfing.npy"
    if cache_path.exists():
        dense = np.load(cache_path).astype(np.float32)
    else:
        dense = pyfing.orientation_field_estimation(
            pose_image,
            pose_mask.astype(np.uint8) * 255,
            dpi=DEFAULT_DPI,
            method="SNFOE",
        ).astype(np.float32)
        dense[~pose_mask.astype(bool)] = 0.0
        np.save(cache_path, dense)

    errors: list[float] = []
    h, w = pose_mask.shape
    for item in minutiae:
        x = int(round(float(item.get("x", float("nan")))))
        y = int(round(float(item.get("y", float("nan")))))
        if x < 0 or y < 0 or x >= w or y >= h or not pose_mask[y, x]:
            continue
        error = _axial_angle_error_deg(float(item.get("theta", 0.0)), float(dense[y, x]))
        errors.append(error)
        item["dense_orientation_error_deg"] = float(error)

    values = np.asarray(errors, dtype=np.float32)
    return {
        "count": int(len(minutiae)),
        "sampled_count": int(values.size),
        "mean_error_deg": float(np.mean(values)) if values.size else 0.0,
        "median_error_deg": float(np.median(values)) if values.size else 0.0,
        "within_15_deg_count": int(np.count_nonzero(values <= 15.0)),
        "within_15_deg_fraction": float(np.count_nonzero(values <= 15.0) / max(values.size, 1)),
        "p90_error_deg": float(np.percentile(values, 90)) if values.size else 0.0,
    }


def _extraction_quality(rows: list[dict[str, Any]], image_shape: tuple[int, int], support_mask: np.ndarray) -> dict[str, Any]:
    h, w = image_shape
    inside = 0
    inside_support = 0
    for row in rows:
        x = int(round(float(row.get("x", float("nan")))))
        y = int(round(float(row.get("y", float("nan")))))
        if 0 <= x < w and 0 <= y < h:
            inside += 1
            if support_mask[y, x] > 0:
                inside_support += 1
    return {
        "minutiae_count": int(len(rows)),
        "inside_image_count": int(inside),
        "inside_unwrap_mask_count": int(inside_support),
        "inside_unwrap_mask_fraction": float(inside_support / max(len(rows), 1)),
        "score_stats": _score_stats(rows),
    }


def _fit_panel(image: np.ndarray, size: tuple[int, int] = (320, 250)) -> np.ndarray:
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


def _draw_label(panel: np.ndarray, text: str) -> np.ndarray:
    if panel.ndim == 2:
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
    panel = panel.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (18, 18, 18), -1)
    cv2.putText(panel, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return panel


def run(
    reconstruction_dir: Path,
    output_dir: Path,
    left_angle: float,
    right_angle: float,
    samples_per_pixel: float,
    reverse_left_unwrap_x: bool,
    reverse_right_unwrap_x: bool,
    unwrap_width_scale: float,
    unwrap_gradient_clip: float | None,
    row_param_smooth_window: int,
    map_smooth_sigma_x: float,
    map_smooth_sigma_y: float,
    orientation_delta_px: float = ORIENTATION_DELTA_PX,
) -> dict[str, Any]:
    reconstruction_dir = reconstruction_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrap_dir = output_dir / "depth_unwrap"

    unwrap_module = _load_depth_unwrap_module()
    unwrap_report = unwrap_module.run(
        reconstruction_dir=reconstruction_dir,
        output_dir=unwrap_dir,
        left_angle=left_angle,
        right_angle=right_angle,
        samples_per_pixel=samples_per_pixel,
        reverse_left_unwrap_x=reverse_left_unwrap_x,
        reverse_right_unwrap_x=reverse_right_unwrap_x,
        unwrap_width_scale=unwrap_width_scale,
        unwrap_gradient_clip=unwrap_gradient_clip,
        row_param_smooth_window=row_param_smooth_window,
        map_smooth_sigma_x=map_smooth_sigma_x,
        map_smooth_sigma_y=map_smooth_sigma_y,
    )

    debug_dir = reconstruction_dir / "debug_views"
    report: dict[str, Any] = {
        "input_reconstruction_dir": str(reconstruction_dir),
        "output_dir": str(output_dir),
        "depth_unwrap_output_dir": str(unwrap_dir),
        "depth_unwrap_report": unwrap_report,
        "method": "Algorithm-1 depth unwrap v4 as-is -> pyfing minutiae -> chart-to-pose reprojection with forward/backward orientation baseline",
        "orientation_delta_px": float(orientation_delta_px),
        "roles": {},
    }
    panels: list[np.ndarray] = []

    for role in ("left", "right"):
        role_dir = output_dir / role
        role_dir.mkdir(parents=True, exist_ok=True)
        unwrap_image_path = unwrap_dir / role / f"{role}_depth_unwrapped.png"
        unwrap_mask_path = unwrap_dir / role / f"{role}_depth_unwrapped_mask.png"
        unwrap_maps_path = unwrap_dir / role / f"{role}_depth_unwarp_maps.npz"
        unwrap_image = _read_gray(unwrap_image_path)
        unwrap_mask = _read_gray(unwrap_mask_path) > 0
        maps = np.load(unwrap_maps_path)
        source_x = maps["source_x_map"].astype(np.float32)
        source_y = maps["source_y_map"].astype(np.float32)

        pose_image = _read_gray(debug_dir / f"{role}_pose_normalized.png")
        pose_mask = _read_gray(debug_dir / f"{role}_pose_mask.png") > 0

        minutiae = _extract_pyfing_minutiae(unwrap_image)
        extracted_json = role_dir / f"{role}_depth_unwrapped_pyfing_minutiae.json"
        extracted_csv = role_dir / f"{role}_depth_unwrapped_pyfing_minutiae.csv"
        extracted_overlay = role_dir / f"{role}_depth_unwrapped_pyfing_minutiae_overlay.png"
        extracted_json.write_text(json.dumps({"image": str(unwrap_image_path), "minutiae": minutiae}, indent=2), encoding="utf-8")
        _write_minutiae_csv(minutiae, extracted_csv)
        cv2.imwrite(str(extracted_overlay), _draw_minutiae_overlay(unwrap_image, minutiae, unwrap_mask.astype(np.uint8)))

        reprojected, reproject_details = _reproject_minutiae(
            minutiae,
            source_x,
            source_y,
            unwrap_mask,
            pose_mask,
            orientation_delta_px,
        )
        orientation_stats = _dense_orientation_stats(output_dir, role, pose_image, pose_mask, reprojected)
        reprojected_json = role_dir / f"{role}_reprojected_minutiae.json"
        reprojected_overlay = role_dir / f"{role}_reprojected_minutiae_overlay.png"
        reprojected_json.write_text(
            json.dumps(
                {
                    "target": f"{role}_pose_normalized",
                    "geometry_policy": "algorithm1_depth_unwrap_v4_source_maps",
                    "minutiae": reprojected,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        cv2.imwrite(str(reprojected_overlay), _draw_minutiae_overlay(pose_image, reprojected, pose_mask.astype(np.uint8)))

        extraction_quality = _extraction_quality(minutiae, unwrap_image.shape[:2], unwrap_mask)
        report["roles"][role] = {
            "unwrap_image": str(unwrap_image_path.resolve()),
            "unwrap_mask": str(unwrap_mask_path.resolve()),
            "unwrap_maps": str(unwrap_maps_path.resolve()),
            "extraction_quality": extraction_quality,
            "reprojection": reproject_details,
            "dense_vs_minutia_orientation": orientation_stats,
            "outputs": {
                "extracted_minutiae_json": str(extracted_json),
                "extracted_minutiae_csv": str(extracted_csv),
                "extracted_minutiae_overlay": str(extracted_overlay),
                "reprojected_minutiae_json": str(reprojected_json),
                "reprojected_minutiae_overlay": str(reprojected_overlay),
            },
        }
        panels.append(
            np.hstack(
                [
                    _draw_label(_fit_panel(unwrap_image), f"{role} depth unwrap"),
                    _draw_label(_fit_panel(unwrap_mask.astype(np.uint8) * 255), f"{role} unwrap mask"),
                    _draw_label(_fit_panel(cv2.imread(str(extracted_overlay), cv2.IMREAD_COLOR)), f"{role} pyfing n={len(minutiae)}"),
                    _draw_label(_fit_panel(cv2.imread(str(reprojected_overlay), cv2.IMREAD_COLOR)), f"{role} reproj n={len(reprojected)}"),
                ]
            )
        )

    left_stats = report["roles"]["left"]["dense_vs_minutia_orientation"]
    right_stats = report["roles"]["right"]["dense_vs_minutia_orientation"]
    total = int(left_stats["sampled_count"] + right_stats["sampled_count"])
    report["all_dense_vs_minutia_orientation"] = {
        "sampled_count": total,
        "mean_error_deg": float(
            (
                left_stats["mean_error_deg"] * left_stats["sampled_count"]
                + right_stats["mean_error_deg"] * right_stats["sampled_count"]
            )
            / max(total, 1)
        ),
        "within_15_deg_count": int(left_stats["within_15_deg_count"] + right_stats["within_15_deg_count"]),
        "within_15_deg_fraction": float(
            (left_stats["within_15_deg_count"] + right_stats["within_15_deg_count"]) / max(total, 1)
        ),
    }

    contact_sheet = output_dir / "algorithm1_depth_unwrap_pyfing_reprojection_contact_sheet.png"
    cv2.imwrite(str(contact_sheet), np.vstack(panels))
    report["contact_sheet"] = str(contact_sheet)
    report_path = output_dir / "algorithm1_depth_unwrap_pyfing_reprojection_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
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
        default=REPO_ROOT / "tmp" / "algorithm1_depth_unwrap_pyfing_reprojection" / "s01_f01_a01",
    )
    parser.add_argument("--left-angle", type=float, default=-45.0)
    parser.add_argument("--right-angle", type=float, default=45.0)
    parser.add_argument("--samples-per-pixel", type=float, default=2.0)
    parser.add_argument("--reverse-left-unwrap-x", action="store_true")
    parser.add_argument("--reverse-right-unwrap-x", action="store_true")
    parser.add_argument("--unwrap-width-scale", type=float, default=1.0)
    parser.add_argument("--unwrap-gradient-clip", type=float, default=3.0)
    # Defaults match the currently inspected v4 smoke diagnostic report.
    parser.add_argument("--row-param-smooth-window", type=int, default=31)
    parser.add_argument("--map-smooth-sigma-x", type=float, default=6.0)
    parser.add_argument("--map-smooth-sigma-y", type=float, default=5.0)
    parser.add_argument("--orientation-delta-px", type=float, default=ORIENTATION_DELTA_PX)
    args = parser.parse_args()

    unwrap_gradient_clip = None if args.unwrap_gradient_clip <= 0 else float(args.unwrap_gradient_clip)
    report = run(
        reconstruction_dir=args.reconstruction_dir,
        output_dir=args.output_dir,
        left_angle=args.left_angle,
        right_angle=args.right_angle,
        samples_per_pixel=args.samples_per_pixel,
        reverse_left_unwrap_x=args.reverse_left_unwrap_x,
        reverse_right_unwrap_x=args.reverse_right_unwrap_x,
        unwrap_width_scale=args.unwrap_width_scale,
        unwrap_gradient_clip=unwrap_gradient_clip,
        row_param_smooth_window=args.row_param_smooth_window,
        map_smooth_sigma_x=args.map_smooth_sigma_x,
        map_smooth_sigma_y=args.map_smooth_sigma_y,
        orientation_delta_px=args.orientation_delta_px,
    )
    print(
        json.dumps(
            {
                "report_path": report["report_path"],
                "contact_sheet": report["contact_sheet"],
                "left": {
                    "extracted": report["roles"]["left"]["extraction_quality"]["minutiae_count"],
                    "reprojected": report["roles"]["left"]["reprojection"]["reprojected_count"],
                    "mean_error_deg": report["roles"]["left"]["dense_vs_minutia_orientation"]["mean_error_deg"],
                    "within_15_deg_fraction": report["roles"]["left"]["dense_vs_minutia_orientation"]["within_15_deg_fraction"],
                },
                "right": {
                    "extracted": report["roles"]["right"]["extraction_quality"]["minutiae_count"],
                    "reprojected": report["roles"]["right"]["reprojection"]["reprojected_count"],
                    "mean_error_deg": report["roles"]["right"]["dense_vs_minutia_orientation"]["mean_error_deg"],
                    "within_15_deg_fraction": report["roles"]["right"]["dense_vs_minutia_orientation"]["within_15_deg_fraction"],
                },
                "all": report["all_dense_vs_minutia_orientation"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Sweep Algorithm-1 depth unwrap smoothing and orientation reprojection delta."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT]


def _load_diagnostic_module() -> Any:
    module_path = REPO_ROOT / "scripts" / "diagnose_algorithm1_depth_unwrap_pyfing_reprojection.py"
    spec = importlib.util.spec_from_file_location("depth_unwrap_pyfing_reprojection", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _case_name(sigma_x: float, sigma_y: float, delta: float) -> str:
    return f"sx{sigma_x:g}_sy{sigma_y:g}_d{delta:g}".replace(".", "p")


def _summarize_case(report: dict[str, Any], sigma_x: float, sigma_y: float, delta: float, case_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case": _case_name(sigma_x, sigma_y, delta),
        "case_dir": str(case_dir),
        "map_smooth_sigma_x": float(sigma_x),
        "map_smooth_sigma_y": float(sigma_y),
        "orientation_delta_px": float(delta),
        "all_mean_error_deg": float(report["all_dense_vs_minutia_orientation"]["mean_error_deg"]),
        "all_within_15_fraction": float(report["all_dense_vs_minutia_orientation"]["within_15_deg_fraction"]),
        "all_sampled_count": int(report["all_dense_vs_minutia_orientation"]["sampled_count"]),
    }
    for role in ("left", "right"):
        orient = report["roles"][role]["dense_vs_minutia_orientation"]
        extract = report["roles"][role]["extraction_quality"]
        reproj = report["roles"][role]["reprojection"]
        row[f"{role}_extracted"] = int(extract["minutiae_count"])
        row[f"{role}_reprojected"] = int(reproj["reprojected_count"])
        row[f"{role}_mean_error_deg"] = float(orient["mean_error_deg"])
        row[f"{role}_median_error_deg"] = float(orient["median_error_deg"])
        row[f"{role}_within_15_fraction"] = float(orient["within_15_deg_fraction"])
        row[f"{role}_within_15_count"] = int(orient["within_15_deg_count"])
        row[f"{role}_score_median"] = float(extract["score_stats"]["median"])
    return row


def _report_for_existing_unwrap(
    diagnostic: Any,
    reconstruction_dir: Path,
    base_report: dict[str, Any],
    base_case_dir: Path,
    delta: float,
) -> dict[str, Any]:
    debug_dir = reconstruction_dir / "debug_views"
    report: dict[str, Any] = {
        "all_dense_vs_minutia_orientation": {},
        "roles": {},
    }
    for role in ("left", "right"):
        role_base = base_report["roles"][role]
        with Path(role_base["outputs"]["extracted_minutiae_json"]).open(encoding="utf-8") as handle:
            minutiae = json.load(handle)["minutiae"]
        maps = np.load(role_base["unwrap_maps"])
        source_x = maps["source_x_map"].astype(np.float32)
        source_y = maps["source_y_map"].astype(np.float32)
        unwrap_mask = cv2.imread(role_base["unwrap_mask"], cv2.IMREAD_GRAYSCALE) > 0
        pose_image = cv2.imread(str(debug_dir / f"{role}_pose_normalized.png"), cv2.IMREAD_GRAYSCALE)
        pose_mask = cv2.imread(str(debug_dir / f"{role}_pose_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
        reprojected, reproj = diagnostic._reproject_minutiae(
            minutiae,
            source_x,
            source_y,
            unwrap_mask,
            pose_mask,
            float(delta),
        )
        orient = diagnostic._dense_orientation_stats(base_case_dir, role, pose_image, pose_mask, reprojected)
        report["roles"][role] = {
            "extraction_quality": role_base["extraction_quality"],
            "reprojection": reproj,
            "dense_vs_minutia_orientation": orient,
        }

    left = report["roles"]["left"]["dense_vs_minutia_orientation"]
    right = report["roles"]["right"]["dense_vs_minutia_orientation"]
    total = int(left["sampled_count"] + right["sampled_count"])
    report["all_dense_vs_minutia_orientation"] = {
        "sampled_count": total,
        "mean_error_deg": float(
            (left["mean_error_deg"] * left["sampled_count"] + right["mean_error_deg"] * right["sampled_count"])
            / max(total, 1)
        ),
        "within_15_deg_count": int(left["within_15_deg_count"] + right["within_15_deg_count"]),
        "within_15_deg_fraction": float((left["within_15_deg_count"] + right["within_15_deg_count"]) / max(total, 1)),
    }
    return report


def run(
    reconstruction_dir: Path,
    output_dir: Path,
    sigma_x_values: list[float],
    sigma_y_values: list[float],
    delta_values: list[float],
    row_param_smooth_window: int,
    samples_per_pixel: float,
    unwrap_width_scale: float,
    unwrap_gradient_clip: float | None,
    left_angle: float,
    right_angle: float,
    reverse_left_unwrap_x: bool,
    reverse_right_unwrap_x: bool,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic = _load_diagnostic_module()
    rows: list[dict[str, Any]] = []

    first_delta = float(delta_values[0]) if delta_values else 4.0
    for sigma_x in sigma_x_values:
        for sigma_y in sigma_y_values:
            base_case_dir = output_dir / f"sx{sigma_x:g}_sy{sigma_y:g}".replace(".", "p")
            print(f"Running unwrap/extraction case {base_case_dir.name}", flush=True)
            base_report = diagnostic.run(
                reconstruction_dir=reconstruction_dir,
                output_dir=base_case_dir,
                left_angle=left_angle,
                right_angle=right_angle,
                samples_per_pixel=samples_per_pixel,
                reverse_left_unwrap_x=reverse_left_unwrap_x,
                reverse_right_unwrap_x=reverse_right_unwrap_x,
                unwrap_width_scale=unwrap_width_scale,
                unwrap_gradient_clip=unwrap_gradient_clip,
                row_param_smooth_window=row_param_smooth_window,
                map_smooth_sigma_x=sigma_x,
                map_smooth_sigma_y=sigma_y,
                orientation_delta_px=first_delta,
            )
            for delta in delta_values:
                if float(delta) == first_delta:
                    report = base_report
                else:
                    report = _report_for_existing_unwrap(
                        diagnostic,
                        reconstruction_dir,
                        base_report,
                        base_case_dir,
                        float(delta),
                    )
                rows.append(_summarize_case(report, sigma_x, sigma_y, delta, base_case_dir))

    ranked_all = sorted(rows, key=lambda item: (-item["all_within_15_fraction"], item["all_mean_error_deg"]))
    ranked_left = sorted(rows, key=lambda item: (-item["left_within_15_fraction"], item["left_mean_error_deg"]))
    ranked_right = sorted(rows, key=lambda item: (-item["right_within_15_fraction"], item["right_mean_error_deg"]))
    report = {
        "reconstruction_dir": str(reconstruction_dir.resolve()),
        "output_dir": str(output_dir),
        "sigma_x_values": sigma_x_values,
        "sigma_y_values": sigma_y_values,
        "orientation_delta_values": delta_values,
        "row_param_smooth_window": int(row_param_smooth_window),
        "cases": rows,
        "best_all": ranked_all[0] if ranked_all else None,
        "best_left": ranked_left[0] if ranked_left else None,
        "best_right": ranked_right[0] if ranked_right else None,
    }

    csv_path = output_dir / "algorithm1_depth_unwrap_parameter_sweep.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    report["csv"] = str(csv_path)
    report_path = output_dir / "algorithm1_depth_unwrap_parameter_sweep_report.json"
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
        default=REPO_ROOT / "tmp" / "algorithm1_depth_unwrap_parameter_sweep" / "s01_f01_a01",
    )
    parser.add_argument("--sigma-x-values", default="2.0,4.0,6.0")
    parser.add_argument("--sigma-y-values", default="1.2,3.0,5.0")
    parser.add_argument("--orientation-delta-values", default="2.0,4.0,6.0,8.0")
    parser.add_argument("--row-param-smooth-window", type=int, default=31)
    parser.add_argument("--samples-per-pixel", type=float, default=2.0)
    parser.add_argument("--unwrap-width-scale", type=float, default=1.0)
    parser.add_argument("--unwrap-gradient-clip", type=float, default=3.0)
    parser.add_argument("--left-angle", type=float, default=-45.0)
    parser.add_argument("--right-angle", type=float, default=45.0)
    parser.add_argument("--reverse-left-unwrap-x", action="store_true")
    parser.add_argument("--reverse-right-unwrap-x", action="store_true")
    args = parser.parse_args()

    unwrap_gradient_clip = None if args.unwrap_gradient_clip <= 0 else float(args.unwrap_gradient_clip)
    report = run(
        reconstruction_dir=args.reconstruction_dir,
        output_dir=args.output_dir,
        sigma_x_values=_parse_float_list(args.sigma_x_values),
        sigma_y_values=_parse_float_list(args.sigma_y_values),
        delta_values=_parse_float_list(args.orientation_delta_values),
        row_param_smooth_window=args.row_param_smooth_window,
        samples_per_pixel=args.samples_per_pixel,
        unwrap_width_scale=args.unwrap_width_scale,
        unwrap_gradient_clip=unwrap_gradient_clip,
        left_angle=args.left_angle,
        right_angle=args.right_angle,
        reverse_left_unwrap_x=args.reverse_left_unwrap_x,
        reverse_right_unwrap_x=args.reverse_right_unwrap_x,
    )
    print(
        json.dumps(
            {
                "report_path": report["report_path"],
                "csv": report["csv"],
                "best_all": report["best_all"],
                "best_left": report["best_left"],
                "best_right": report["best_right"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

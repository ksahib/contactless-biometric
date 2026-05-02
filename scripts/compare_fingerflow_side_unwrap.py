#!/usr/bin/env python
"""Compare FingerFlow extraction on raw/preprocessed side views vs side unwraps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

# The repo has a top-level copy.py, which can shadow stdlib copy while importing
# cv2/matplotlib. Import third-party modules with the repo root hidden first.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
import main as mcc_main  # noqa: E402


ROLE_SAMPLE_IDS = {"left": "s01_f01_a01_v01", "right": "s01_f01_a01_v02"}
INPUT_KINDS = ("raw", "pose_normalized", "unwrapped")


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


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


def _draw_label(panel: np.ndarray, text: str) -> np.ndarray:
    panel = panel.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (18, 18, 18), -1)
    cv2.putText(panel, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return panel


def _load_minutiae(csv_path: Path) -> list[dict[str, float]]:
    if not csv_path.exists():
        return []
    rows: list[dict[str, float]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append(
                    {
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "angle": float(row.get("angle", 0.0)),
                        "score": float(row.get("score", 0.0)),
                        "class": float(row.get("class", 0.0)),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def _overlay_minutiae(image_path: Path, minutiae: list[dict[str, float]], mask_path: Path | None = None) -> np.ndarray:
    image = _read_color(image_path)
    if mask_path is not None and mask_path.exists():
        mask = _read_gray(mask_path)
        if mask.shape == image.shape[:2]:
            tint = np.zeros_like(image)
            tint[:, :, 1] = np.where(mask > 0, 80, 0).astype(np.uint8)
            image = cv2.addWeighted(image, 1.0, tint, 0.45, 0)

    h, w = image.shape[:2]
    for row in minutiae:
        x = int(round(row["x"]))
        y = int(round(row["y"]))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        score = float(row.get("score", 0.0))
        color = (0, 255, 255) if score >= 0.5 else (0, 140, 255)
        angle = float(row.get("angle", 0.0))
        length = 16
        x2 = int(round(x + length * math.cos(angle)))
        y2 = int(round(y + length * math.sin(angle)))
        cv2.circle(image, (x, y), 3, color, -1, cv2.LINE_AA)
        cv2.line(image, (x, y), (x2, y2), color, 1, cv2.LINE_AA)
    return image


def _summarize_minutiae(
    minutiae: list[dict[str, float]],
    image_shape: tuple[int, int],
    support_mask_path: Path | None,
    quality_mask_path: Path | None,
) -> dict[str, Any]:
    h, w = image_shape
    scores = np.asarray([row["score"] for row in minutiae], dtype=np.float32)
    inside_bounds = 0
    inside_support = 0
    inside_quality = 0
    support_mask = _read_gray(support_mask_path) > 0 if support_mask_path is not None and support_mask_path.exists() else None
    quality_mask = _read_gray(quality_mask_path) > 0 if quality_mask_path is not None and quality_mask_path.exists() else None

    for row in minutiae:
        x = int(round(row["x"]))
        y = int(round(row["y"]))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        inside_bounds += 1
        if support_mask is not None and support_mask.shape == (h, w) and support_mask[y, x]:
            inside_support += 1
        if quality_mask is not None and quality_mask.shape == (h, w) and quality_mask[y, x]:
            inside_quality += 1

    return {
        "minutiae_count": int(len(minutiae)),
        "inside_image_count": int(inside_bounds),
        "inside_support_mask_count": int(inside_support) if support_mask is not None else None,
        "inside_scale_quality_mask_count": int(inside_quality) if quality_mask is not None else None,
        "mean_score": float(np.mean(scores)) if scores.size else 0.0,
        "median_score": float(np.median(scores)) if scores.size else 0.0,
        "score_p25": float(np.percentile(scores, 25)) if scores.size else 0.0,
        "score_p75": float(np.percentile(scores, 75)) if scores.size else 0.0,
    }


def _input_spec(gt_root: Path, unwrap_dir: Path, role: str, kind: str) -> dict[str, Path | None]:
    sample_dir = gt_root / "samples" / ROLE_SAMPLE_IDS[role]
    if kind == "raw":
        return {
            "image": sample_dir / "raw_input.png",
            "support_mask": sample_dir / "mask.png",
            "quality_mask": None,
        }
    if kind == "preprocessed":
        return {
            "image": sample_dir / "preprocessed_input.png",
            "support_mask": sample_dir / "preprocess_mask.png",
            "quality_mask": None,
        }
    if kind == "pose_normalized":
        return {
            "image": gt_root / "reconstructions" / "s01_f01_a01" / "debug_views" / f"{role}_pose_normalized.png",
            "support_mask": gt_root / "reconstructions" / "s01_f01_a01" / "debug_views" / f"{role}_pose_mask.png",
            "quality_mask": None,
        }
    if kind == "unwrapped":
        return {
            "image": unwrap_dir / f"{role}_surface_unwrapped.png",
            "support_mask": unwrap_dir / f"{role}_surface_unwrapped_mask.png",
            "quality_mask": unwrap_dir / f"{role}_surface_scale_quality_mask.png",
        }
    raise ValueError(f"unknown input kind: {kind}")


def _run_one(
    model_paths: tuple[Path, Path, Path, Path],
    image_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    minutiae_json = output_dir / "minutiae.json"
    minutiae_csv = output_dir / "minutiae.csv"
    core_csv = output_dir / "core.csv"
    if minutiae_csv.exists() and minutiae_json.exists():
        return minutiae_json, minutiae_csv, core_csv, len(_load_minutiae(minutiae_csv)), 0
    print(f"Running FingerFlow: {image_path} -> {output_dir}", flush=True)
    count, core_count = mcc_main.extract_minutiae_with_fingerflow(
        image_path.resolve(),
        image_path.resolve(),
        model_paths,
        minutiae_json.resolve(),
        minutiae_csv.resolve(),
        core_csv.resolve(),
    )
    return minutiae_json, minutiae_csv, core_csv, count, core_count


def run(gt_root: Path, unwrap_dir: Path, output_dir: Path, model_dir: Path) -> dict[str, Any]:
    gt_root = gt_root.resolve()
    unwrap_dir = unwrap_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_paths = mcc_main.ensure_fingerflow_models(model_dir)

    report: dict[str, Any] = {
        "gt_root": str(gt_root),
        "unwrap_dir": str(unwrap_dir),
        "output_dir": str(output_dir),
        "roles": {},
    }
    contact_rows: list[np.ndarray] = []
    panel_size = (320, 250)
    for role in ("left", "right"):
        role_report: dict[str, Any] = {}
        panels: list[np.ndarray] = []
        for kind in INPUT_KINDS:
            spec = _input_spec(gt_root, unwrap_dir, role, kind)
            image_path = Path(spec["image"])
            case_dir = output_dir / role / kind
            minutiae_json, minutiae_csv, core_csv, _count, core_count = _run_one(model_paths, image_path, case_dir)
            minutiae = _load_minutiae(minutiae_csv)
            image_shape = _read_gray(image_path).shape[:2]
            overlay = _overlay_minutiae(image_path, minutiae, Path(spec["support_mask"]) if spec["support_mask"] else None)
            overlay_path = case_dir / "minutiae_overlay.png"
            cv2.imwrite(str(overlay_path), overlay)

            summary = _summarize_minutiae(
                minutiae,
                image_shape,
                Path(spec["support_mask"]) if spec["support_mask"] else None,
                Path(spec["quality_mask"]) if spec["quality_mask"] else None,
            )
            summary.update(
                {
                    "image": str(image_path),
                    "support_mask": str(spec["support_mask"]) if spec["support_mask"] else None,
                    "quality_mask": str(spec["quality_mask"]) if spec["quality_mask"] else None,
                    "minutiae_json": str(minutiae_json),
                    "minutiae_csv": str(minutiae_csv),
                    "core_csv": str(core_csv),
                    "core_count": int(core_count),
                    "overlay": str(overlay_path),
                    "shape_hw": [int(image_shape[0]), int(image_shape[1])],
                }
            )
            role_report[kind] = summary
            panels.append(
                _draw_label(
                    _fit_panel(overlay, panel_size),
                    f"{role} {kind}: n={summary['minutiae_count']} score={summary['median_score']:.2f}",
                )
            )
        report["roles"][role] = role_report
        contact_rows.append(np.hstack(panels))

    contact_sheet = np.vstack(contact_rows)
    contact_path = output_dir / "fingerflow_side_unwrap_contact_sheet.png"
    cv2.imwrite(str(contact_path), contact_sheet)
    report["contact_sheet"] = str(contact_path)
    report_path = output_dir / "fingerflow_side_unwrap_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, default=REPO_ROOT / "ground_truth_smoke_dense_fixed_limit6")
    parser.add_argument(
        "--unwrap-dir",
        type=Path,
        default=REPO_ROOT / "tmp" / "algorithm1_surface_unwrap_fixed" / "s01_f01_a01",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "tmp" / "fingerflow_side_unwrap_compare" / "s01_f01_a01",
    )
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / ".fingerflow_models")
    args = parser.parse_args()

    report = run(args.gt_root, args.unwrap_dir, args.output_dir, args.model_dir)
    print(
        json.dumps(
            {
                "output_dir": report["output_dir"],
                "contact_sheet": report["contact_sheet"],
                "report_path": report["report_path"],
                "left": {
                    kind: {
                        "count": report["roles"]["left"][kind]["minutiae_count"],
                        "median_score": report["roles"]["left"][kind]["median_score"],
                        "inside_quality": report["roles"]["left"][kind]["inside_scale_quality_mask_count"],
                    }
                    for kind in INPUT_KINDS
                },
                "right": {
                    kind: {
                        "count": report["roles"]["right"][kind]["minutiae_count"],
                        "median_score": report["roles"]["right"][kind]["median_score"],
                        "inside_quality": report["roles"]["right"][kind]["inside_scale_quality_mask_count"],
                    }
                    for kind in INPUT_KINDS
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

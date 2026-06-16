from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import diagnose_featurenet_minutia_instability as diag

import cv2
import numpy as np
import pandas as pd


ANGLE_TOLERANCE_RAD = math.pi / 6.0
NEAR_RADIUS = 16.0
FAR_RADIUS = 32.0
LOW_DESCRIPTOR_SCORE = 0.45


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "angle": float(row["angle"]),
                "score": float(row["score"]),
            }
            for row in csv.DictReader(handle)
        ]


def _load_meta(case: dict[str, Any]) -> dict[str, Any]:
    return json.loads((Path(case["preprocess_dir"]) / "meta.json").read_text(encoding="utf-8"))


def _mask_stats(mask_path: str | Path) -> dict[str, Any]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return {"area": 0, "bbox_xywh": None}
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return {"area": 0, "bbox_xywh": None}
    x, y, w, h = cv2.boundingRect((mask > 0).astype(np.uint8))
    return {"area": int(xs.size), "bbox_xywh": [int(x), int(y), int(w), int(h)]}


def _score_map(case: dict[str, Any]) -> np.ndarray:
    return np.load(Path(case["case_dir"]) / "minutia_score_sigmoid.npy").astype(np.float32)


def _angle_delta(a: float, b: float) -> float:
    delta = (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi
    delta = abs(delta)
    return min(delta, abs(math.pi - delta))


def _nearest(point: tuple[float, float], rows: list[dict[str, Any]], x_key: str = "x", y_key: str = "y") -> tuple[int | None, dict[str, Any] | None, float]:
    if not rows:
        return None, None, float("inf")
    xy = np.asarray([[float(row[x_key]), float(row[y_key])] for row in rows], dtype=np.float32)
    p = np.asarray(point, dtype=np.float32)
    distances = np.sqrt(((xy - p) ** 2).sum(axis=1))
    index = int(np.argmin(distances))
    return index, rows[index], float(distances[index])


def _sample_score(score_map: np.ndarray, input_shape_hw: list[int], x: float, y: float) -> float:
    in_h, in_w = int(input_shape_hw[0]), int(input_shape_hw[1])
    out_h, out_w = score_map.shape
    col = int(round(float(x) / max(float(in_w) / max(out_w, 1), 1e-6)))
    row = int(round(float(y) / max(float(in_h) / max(out_h, 1), 1e-6)))
    if row < 0 or col < 0 or row >= out_h or col >= out_w:
        return 0.0
    return float(score_map[row, col])


def _score_response_rows(score_map: np.ndarray, input_shape_hw: list[int], threshold: float) -> list[dict[str, float]]:
    out_h, out_w = score_map.shape
    in_h, in_w = int(input_shape_hw[0]), int(input_shape_hw[1])
    scale_x = float(in_w) / max(float(out_w), 1.0)
    scale_y = float(in_h) / max(float(out_h), 1.0)
    active = np.argwhere(score_map >= float(threshold))
    return [
        {
            "x": (float(col) + 0.5) * scale_x,
            "y": (float(row) + 0.5) * scale_y,
            "score": float(score_map[row, col]),
        }
        for row, col in active
    ]


def _apply_centroid_transform_to_b_rows(
    rows_b: list[dict[str, float]],
    details: dict[str, Any],
) -> list[dict[str, Any]]:
    transform = details.get("transform") or {}
    query_centroid = np.asarray(transform.get("query_centroid", [0.0, 0.0]), dtype=np.float32)
    template_centroid = np.asarray(transform.get("template_centroid", [0.0, 0.0]), dtype=np.float32)
    rotation = float(transform.get("rotation", 0.0))
    scale = float(transform.get("scale", 1.0))
    cos_t = math.cos(rotation)
    sin_t = math.sin(rotation)
    mapped: list[dict[str, Any]] = []
    for index, row in enumerate(rows_b):
        xy = np.asarray([row["x"], row["y"]], dtype=np.float32) - query_centroid
        x = (scale * ((xy[0] * cos_t) - (xy[1] * sin_t))) + template_centroid[0]
        y = (scale * ((xy[0] * sin_t) + (xy[1] * cos_t))) + template_centroid[1]
        item = dict(row)
        item["source_index"] = index
        item["x_aligned"] = float(x)
        item["y_aligned"] = float(y)
        item["angle_aligned"] = float((row["angle"] + rotation + math.pi) % (2.0 * math.pi) - math.pi)
        mapped.append(item)
    return mapped


def _counterpart_stats(rows_a: list[dict[str, float]], aligned_b: list[dict[str, Any]]) -> dict[str, Any]:
    distances: list[float] = []
    angle_errors: list[float] = []
    within_16 = 0
    within_32 = 0
    for row in rows_a:
        _idx, b, dist = _nearest((row["x"], row["y"]), aligned_b, "x_aligned", "y_aligned")
        distances.append(dist)
        if dist <= NEAR_RADIUS:
            within_16 += 1
        if dist <= FAR_RADIUS:
            within_32 += 1
        if b is not None:
            angle_errors.append(_angle_delta(row["angle"], b.get("angle_aligned", b["angle"])))
    return {
        "counterparts_within_16px": int(within_16),
        "counterparts_within_32px": int(within_32),
        "median_location_error": float(np.median(distances)) if distances else None,
        "median_angle_error_degrees": float(math.degrees(np.median(angle_errors))) if angle_errors else None,
        "unmatched_a_minutiae_32px": int(len(rows_a) - within_32),
    }


def _run_mcc_details(case_a: dict[str, Any], case_b: dict[str, Any], method: str, threshold: float) -> tuple[float, np.ndarray, dict[str, Any]]:
    import main as mcc_main

    if method.upper() in {"LSA-CENTROID", "LSA-R-CENTROID"}:
        return mcc_main.match_minutiae_csv_centroid_details(
            Path(case_a["case_dir"]) / f"decoded_threshold_{threshold:.2f}_nms_on.csv",
            Path(case_b["case_dir"]) / f"decoded_threshold_{threshold:.2f}_nms_on.csv",
            method=method,
            orientation_path_a=case_a["orientation_npy"],
            orientation_path_b=case_b["orientation_npy"],
            ridge_period_path_a=case_a["ridge_period_npy"],
            ridge_period_path_b=case_b["ridge_period_npy"],
        )
    score, sim_matrix = mcc_main.match_minutiae_csv(
        Path(case_a["case_dir"]) / f"decoded_threshold_{threshold:.2f}_nms_on.csv",
        Path(case_b["case_dir"]) / f"decoded_threshold_{threshold:.2f}_nms_on.csv",
        method=method,
        mask_path_a=Path(case_a["mask_png"]),
        mask_path_b=Path(case_b["mask_png"]),
        overlap_mode="auto",
    )
    return float(score), np.asarray(sim_matrix), {
        "method": method,
        "selected_pairs": [],
        "selected_pair_count": 0,
        "left_descriptor_count_after": int(np.asarray(sim_matrix).shape[0]) if np.asarray(sim_matrix).ndim == 2 else 0,
        "right_descriptor_count_after": int(np.asarray(sim_matrix).shape[1]) if np.asarray(sim_matrix).ndim == 2 else 0,
        "warnings": ["details_unavailable_for_non_centroid_method"],
    }


def _descriptor_index_maps(case_a: dict[str, Any], case_b: dict[str, Any], method: str, threshold: float) -> tuple[dict[int, int], dict[int, int]]:
    import main as mcc_main

    if method.upper() not in {"LSA-CENTROID", "LSA-R-CENTROID"}:
        return {}, {}
    path_a = Path(case_a["case_dir"]) / f"decoded_threshold_{threshold:.2f}_nms_on.csv"
    path_b = Path(case_b["case_dir"]) / f"decoded_threshold_{threshold:.2f}_nms_on.csv"
    raw_a = mcc_main._load_minutiae_frame_for_centroid(path_a)
    raw_b = mcc_main._load_minutiae_frame_for_centroid(path_b)
    minutiae_a = mcc_main.pose_norm.coerce_minutiae(raw_a.to_dict(orient="records"))
    minutiae_b = mcc_main.pose_norm.coerce_minutiae(raw_b.to_dict(orient="records"))
    centroid_a = mcc_main.pose_norm.compute_minutiae_centroid(minutiae_a, use_quality_weights=True)
    centroid_b = mcc_main.pose_norm.compute_minutiae_centroid(minutiae_b, use_quality_weights=True)
    score, _sim, details = _run_mcc_details(case_a, case_b, method, threshold)
    transform_details = details.get("transform") or {}
    transform = mcc_main.pose_norm.SimilarityTransform(
        query_centroid=centroid_b,
        template_centroid=centroid_a,
        rotation=float(transform_details.get("rotation", 0.0)),
        scale=float(transform_details.get("scale", 1.0)),
        rotation_source=str(transform_details.get("rotation_source", "diagnostic")),
        scale_source=str(transform_details.get("scale_source", "diagnostic")),
    )
    normalized_a, _ = mcc_main.pose_norm.translate_minutiae_to_centroid(minutiae_a, centroid_a)
    normalized_b = mcc_main.pose_norm.apply_similarity_transform_to_minutiae(minutiae_b, transform)
    frame_a = mcc_main._frame_from_minutiae(normalized_a)
    frame_b = mcc_main._frame_from_minutiae(normalized_b)
    descriptors_a = mcc_main.build_descriptors(frame_a, validity_mask_path=None, validity_mode="auto")
    descriptors_b = mcc_main.build_descriptors(frame_b, validity_mask_path=None, validity_mode="auto")
    return (
        {int(desc.center_index): idx for idx, desc in enumerate(descriptors_a)},
        {int(desc.center_index): idx for idx, desc in enumerate(descriptors_b)},
    )


def _classify_rows(
    *,
    rows_a: list[dict[str, float]],
    rows_b: list[dict[str, float]],
    raw_b: list[dict[str, float]],
    aligned_b: list[dict[str, Any]],
    aligned_raw_b: list[dict[str, Any]],
    score_b: np.ndarray,
    input_shape_b: list[int],
    threshold: float,
    sim_matrix: np.ndarray,
    details: dict[str, Any],
    desc_a_by_raw: dict[int, int],
    desc_b_by_raw: dict[int, int],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    selected = {(int(pair["row"]), int(pair["col"])) for pair in details.get("selected_pairs", [])}
    rows: list[dict[str, Any]] = []
    counter: Counter[str] = Counter()
    for index, a in enumerate(rows_a):
        raw_idx, raw_near, raw_dist = _nearest((a["x"], a["y"]), aligned_raw_b, "x_aligned", "y_aligned")
        dec_idx, dec_near, dec_dist = _nearest((a["x"], a["y"]), aligned_b, "x_aligned", "y_aligned")
        nearest_score = float(raw_near["score"]) if raw_near is not None else 0.0
        corresponding_score = _sample_score(score_b, input_shape_b, a["x"], a["y"])
        passes_threshold = nearest_score >= float(threshold) and raw_dist <= FAR_RADIUS
        survives_nms = dec_near is not None and dec_dist <= FAR_RADIUS
        angle_error = _angle_delta(a["angle"], dec_near.get("angle_aligned", dec_near["angle"])) if dec_near is not None else float("inf")
        desc_a = desc_a_by_raw.get(index)
        desc_b = desc_b_by_raw.get(int(dec_near["source_index"])) if dec_near is not None else None
        best_pair_score = 0.0
        selected_by_lsa = False
        if desc_a is not None and sim_matrix.ndim == 2 and sim_matrix.shape[0] > desc_a:
            if desc_b is not None and sim_matrix.shape[1] > desc_b:
                best_pair_score = float(sim_matrix[desc_a, desc_b])
                selected_by_lsa = (desc_a, desc_b) in selected
            elif sim_matrix.shape[1] > 0:
                best_pair_score = float(np.max(sim_matrix[desc_a]))

        if selected_by_lsa:
            category = "matched"
        elif raw_near is None or raw_dist > FAR_RADIUS:
            category = "model_score_missing"
        elif not passes_threshold:
            category = "threshold_drop"
        elif dec_near is None or dec_dist > FAR_RADIUS:
            category = "nms_suppressed" if raw_dist <= FAR_RADIUS else "offset_moved"
        elif dec_dist > NEAR_RADIUS:
            category = "offset_moved"
        elif angle_error > ANGLE_TOLERANCE_RAD:
            category = "orientation_changed"
        elif desc_a is None or desc_b is None:
            category = "mcc_descriptor_dropped"
        elif best_pair_score < LOW_DESCRIPTOR_SCORE:
            category = "descriptor_score_low"
        elif not selected_by_lsa:
            category = "assignment_rejected"
        else:
            category = "unknown"

        counter[category] += 1
        rows.append(
            {
                "id": index,
                "x_a": a["x"],
                "y_a": a["y"],
                "theta_a": a["angle"],
                "conf_a": a["score"],
                "nearest_b_score_response": max(nearest_score, corresponding_score),
                "nearest_b_pre_nms_distance": raw_dist,
                "nearest_b_decoded_distance": dec_dist,
                "nearest_b_angle_error": angle_error,
                "has_nearby_score_response": bool(raw_dist <= FAR_RADIUS),
                "has_nearby_pre_nms_peak": bool(raw_dist <= FAR_RADIUS and nearest_score >= threshold),
                "passes_threshold_in_b": bool(passes_threshold),
                "survives_nms_in_b": bool(survives_nms),
                "offset_within_16px": bool(dec_dist <= NEAR_RADIUS),
                "offset_within_32px": bool(dec_dist <= FAR_RADIUS),
                "orientation_within_tolerance": bool(angle_error <= ANGLE_TOLERANCE_RAD),
                "mcc_descriptor_a_exists": bool(desc_a is not None),
                "mcc_descriptor_b_exists": bool(desc_b is not None),
                "best_mcc_pair_score": best_pair_score,
                "selected_by_lsa": bool(selected_by_lsa),
                "failure_category": category,
            }
        )
    return rows, counter


def _save_heatmap(score: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    norm = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(path), cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO))


def _draw_minutiae(image_path: Path, rows: list[dict[str, Any]], output_path: Path, color: tuple[int, int, int]) -> None:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for row in rows:
        x, y = int(round(row["x"])), int(round(row["y"]))
        cv2.circle(canvas, (x, y), 5, color, 1, lineType=cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def _make_conclusion(summary: dict[str, Any], categories: dict[str, int]) -> str:
    if categories.get("model_score_missing", 0) >= max(4, summary["total_a_minutiae"] * 0.25):
        return "FeatureNet extraction/image-quality robustness is the likely failure: many A minutiae have no nearby score response in B after diagnostic alignment."
    if categories.get("threshold_drop", 0) + categories.get("nms_suppressed", 0) >= max(4, summary["total_a_minutiae"] * 0.25):
        return "Thresholding/NMS is the likely failure: score responses exist but many do not survive final decoding."
    if categories.get("offset_moved", 0) + categories.get("orientation_changed", 0) >= max(4, summary["total_a_minutiae"] * 0.25):
        return "Localization/orientation inconsistency is the likely failure: decoded counterparts exist but are displaced or orientation-inconsistent."
    if categories.get("mcc_descriptor_dropped", 0) >= max(3, summary["total_a_minutiae"] * 0.15):
        return "MCC descriptor construction is the likely failure: decoded counterparts exist but descriptors are dropped."
    if categories.get("descriptor_score_low", 0) + categories.get("assignment_rejected", 0) >= max(4, summary["total_a_minutiae"] * 0.25):
        return "MCC descriptor scoring or LSA assignment is the likely failure: decoded counterparts exist but are not selected as strong MCC matches."
    return "No single dominant failure stage was isolated; failures are distributed across extraction, localization, and MCC assignment."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose a low-scoring real same-finger FeatureNet/MCC pair.")
    parser.add_argument("--image-a", type=Path, required=True)
    parser.add_argument("--image-b", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--method", type=str, default="LSA-CENTROID")
    parser.add_argument("--nms", dest="nms", action="store_true", default=True)
    parser.add_argument("--no-nms", dest="nms", action="store_false")
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = diag._resolve_device(args.device)
    model = diag.load_checkpoint_model(args.weights, device)
    thresholds = tuple(sorted(set((*diag.DEFAULT_THRESHOLDS, float(args.threshold)))))

    case_a = diag._run_case(
        name="a",
        image_path=args.image_a,
        output_dir=output_dir,
        model=model,
        device=device,
        score_threshold=float(args.threshold),
        thresholds=thresholds,
    )
    case_b = diag._run_case(
        name="b",
        image_path=args.image_b,
        output_dir=output_dir,
        model=model,
        device=device,
        score_threshold=float(args.threshold),
        thresholds=thresholds,
    )

    key = f"{args.threshold:.2f}"
    mode = "nms_on" if args.nms else "nms_off"
    rows_a = case_a["decoded"][mode][key]
    rows_b = case_b["decoded"][mode][key]
    raw_b = case_b["decoded"]["nms_off"][key]
    score_a = _score_map(case_a)
    score_b = _score_map(case_b)
    mcc_score, sim_matrix, details = _run_mcc_details(case_a, case_b, args.method, float(args.threshold))
    aligned_b = _apply_centroid_transform_to_b_rows(rows_b, details)
    aligned_raw_b = _apply_centroid_transform_to_b_rows(raw_b, details)
    desc_a_by_raw, desc_b_by_raw = _descriptor_index_maps(case_a, case_b, args.method, float(args.threshold))
    per_rows, category_counts = _classify_rows(
        rows_a=rows_a,
        rows_b=rows_b,
        raw_b=raw_b,
        aligned_b=aligned_b,
        aligned_raw_b=aligned_raw_b,
        score_b=score_b,
        input_shape_b=case_b["input_shape_hw"],
        threshold=float(args.threshold),
        sim_matrix=np.asarray(sim_matrix),
        details=details,
        desc_a_by_raw=desc_a_by_raw,
        desc_b_by_raw=desc_b_by_raw,
    )

    counterpart = _counterpart_stats(rows_a, aligned_b)
    pre_meta_a = _load_meta(case_a)
    pre_meta_b = _load_meta(case_b)
    threshold_stats_a = diag._threshold_stats(score_a, float(args.threshold))
    threshold_stats_b = diag._threshold_stats(score_b, float(args.threshold))
    descriptor_pairs_available = int(np.count_nonzero(np.asarray(sim_matrix) > 0.0)) if np.asarray(sim_matrix).ndim == 2 else 0
    summary = {
        "total_a_minutiae": int(len(rows_a)),
        "total_b_minutiae": int(len(rows_b)),
        "a_mcc_descriptor_count": int(details.get("left_descriptor_count_after", 0)),
        "b_mcc_descriptor_count": int(details.get("right_descriptor_count_after", 0)),
        **counterpart,
        "score_map_responses_found": int(sum(1 for row in per_rows if row["has_nearby_score_response"])),
        "pre_nms_candidates_found": int(sum(1 for row in per_rows if row["has_nearby_pre_nms_peak"])),
        "survived_nms": int(sum(1 for row in per_rows if row["survives_nms_in_b"])),
        "descriptor_pairs_available": descriptor_pairs_available,
        "lsa_selected_pairs": int(details.get("selected_pair_count", 0)),
        "final_mcc_score": float(mcc_score),
        "failure_categories": dict(category_counts),
    }

    summary_rows = [
        {
            "image": "A",
            "ridge_period": pre_meta_a.get("ridge_period"),
            "scale": pre_meta_a.get("scale"),
            "yaw": pre_meta_a.get("yaw_angle"),
            "output_shape": case_a["input_shape_hw"],
            "mask_area": _mask_stats(case_a["mask_png"])["area"],
            "bbox": _mask_stats(case_a["mask_png"])["bbox_xywh"],
            "decoded_minutiae": len(rows_a),
            "mcc_descriptors": details.get("left_descriptor_count_after", 0),
            "raw_active_cells": threshold_stats_a["raw_active_cells"],
            "local_maxima_cells": threshold_stats_a["local_maxima_cells"],
        },
        {
            "image": "B",
            "ridge_period": pre_meta_b.get("ridge_period"),
            "scale": pre_meta_b.get("scale"),
            "yaw": pre_meta_b.get("yaw_angle"),
            "output_shape": case_b["input_shape_hw"],
            "mask_area": _mask_stats(case_b["mask_png"])["area"],
            "bbox": _mask_stats(case_b["mask_png"])["bbox_xywh"],
            "decoded_minutiae": len(rows_b),
            "mcc_descriptors": details.get("right_descriptor_count_after", 0),
            "raw_active_cells": threshold_stats_b["raw_active_cells"],
            "local_maxima_cells": threshold_stats_b["local_maxima_cells"],
        },
    ]
    _write_csv(
        output_dir / "summary_table.csv",
        summary_rows,
        ["image", "ridge_period", "scale", "yaw", "output_shape", "mask_area", "bbox", "decoded_minutiae", "mcc_descriptors", "raw_active_cells", "local_maxima_cells"],
    )
    _write_csv(
        output_dir / "per_minutia_failure_table.csv",
        per_rows,
        [
            "id",
            "x_a",
            "y_a",
            "theta_a",
            "conf_a",
            "nearest_b_score_response",
            "nearest_b_pre_nms_distance",
            "nearest_b_decoded_distance",
            "nearest_b_angle_error",
            "has_nearby_score_response",
            "has_nearby_pre_nms_peak",
            "passes_threshold_in_b",
            "survives_nms_in_b",
            "offset_within_16px",
            "offset_within_32px",
            "orientation_within_tolerance",
            "mcc_descriptor_a_exists",
            "mcc_descriptor_b_exists",
            "best_mcc_pair_score",
            "selected_by_lsa",
            "failure_category",
        ],
    )

    visuals: dict[str, str] = {}
    if args.save_visuals:
        visuals_dir = output_dir / "visuals"
        a_img = Path(case_a["preprocess_dir"]) / "masked_image.png"
        b_img = Path(case_b["preprocess_dir"]) / "masked_image.png"
        _draw_minutiae(a_img, rows_a, visuals_dir / "a_minutiae.png", (0, 255, 255))
        _draw_minutiae(b_img, rows_b, visuals_dir / "b_minutiae.png", (0, 255, 255))
        _save_heatmap(score_a, visuals_dir / "heatmap_a.png")
        _save_heatmap(score_b, visuals_dir / "heatmap_b.png")
        if score_a.shape == score_b.shape:
            _save_heatmap(np.abs(score_a - score_b), visuals_dir / "heatmap_absdiff.png")
        visuals = {path.stem: str(path) for path in visuals_dir.glob("*.png")}

    report = {
        "inputs": {
            "image_a": str(args.image_a),
            "image_b": str(args.image_b),
            "weights": str(args.weights),
            "threshold": float(args.threshold),
            "method": str(args.method),
            "nms": bool(args.nms),
        },
        "preprocessing": {
            "a": {**pre_meta_a, **_mask_stats(case_a["mask_png"]), "input_shape_hw": case_a["input_shape_hw"]},
            "b": {**pre_meta_b, **_mask_stats(case_b["mask_png"]), "input_shape_hw": case_b["input_shape_hw"]},
        },
        "featurenet": {
            "a": {"score_map_shape": list(score_a.shape), **threshold_stats_a},
            "b": {"score_map_shape": list(score_b.shape), **threshold_stats_b},
            "score_map_correlation_raw": float(np.corrcoef(score_a.reshape(-1), score_b.reshape(-1))[0, 1]) if score_a.shape == score_b.shape else None,
        },
        "alignment": {
            "source": "MCC centroid diagnostic transform",
            "transform": details.get("transform"),
            **counterpart,
        },
        "mcc": {
            "score": float(mcc_score),
            "similarity_matrix_shape": list(np.asarray(sim_matrix).shape),
            "details": details,
        },
        "summary": summary,
        "conclusion": _make_conclusion(summary, dict(category_counts)),
        "artifacts": {
            "report_json": str(output_dir / "report.json"),
            "summary_table_csv": str(output_dir / "summary_table.csv"),
            "per_minutia_failure_table_csv": str(output_dir / "per_minutia_failure_table.csv"),
            "case_a_dir": case_a["case_dir"],
            "case_b_dir": case_b["case_dir"],
            "visuals": visuals,
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"score": mcc_score, "summary": summary, "conclusion": report["conclusion"]}, indent=2))


if __name__ == "__main__":
    main()

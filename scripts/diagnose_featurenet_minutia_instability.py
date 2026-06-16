from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import sysconfig
from collections import Counter
from pathlib import Path
from typing import Any


def _ensure_stdlib_copy_module() -> None:
    stdlib_copy = Path(sysconfig.get_paths()["stdlib"]) / "copy.py"
    spec = importlib.util.spec_from_file_location("copy", stdlib_copy)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not resolve stdlib copy module from {stdlib_copy}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = module
    spec.loader.exec_module(module)


_ensure_stdlib_copy_module()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import preprocess as preprocess_module
from generate_ground_truth import load_bgr_image
from featurenet.models.finger_pad_segmentation import save_finger_pad_debug, segment_finger_pad
from featurenet.models.infer import (
    _orientation_vectors_to_radians,
    _resolve_device,
    decode_minutiae_rows,
    load_checkpoint_model,
    preprocess_input_bgr,
    run_inference,
    save_minutiae_csv,
    save_pose_sidecars,
    serialize_outputs,
)


DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
DEFAULT_FIXED_RIDGE_PERIOD = 13.0846
MATCH_RADII = (8.0, 16.0, 24.0, 32.0)
ANGLE_LIMIT = math.pi / 6.0


def _u8_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _save_mask_png(mask_tensor: torch.Tensor, path: Path) -> None:
    mask = mask_tensor.detach().cpu().numpy()
    if mask.ndim == 4:
        mask = mask[0, 0]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), _u8_mask(mask))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fixed_period_patch(fixed_period: float, target_period: float) -> None:
    def fixed_scale_to_paper_ridge_period(
        enhanced: np.ndarray,
        full_mask: np.ndarray,
        *,
        target_period: float = target_period,
        orientation: float | np.ndarray | None = None,
        center_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        scale = float(target_period) / float(fixed_period)
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
        return scaled_image.astype(np.uint8), _u8_mask(scaled_mask), float(fixed_period), float(scale)

    preprocess_module.scale_to_paper_ridge_period = fixed_scale_to_paper_ridge_period


def _score_map(outputs: dict[str, torch.Tensor]) -> np.ndarray:
    score = torch.sigmoid(outputs["minutia_score"].detach().float())[0, 0]
    return score.cpu().numpy().astype(np.float32)


def _offset_maps(outputs: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    x_map = torch.sigmoid(outputs["minutia_x"].detach().float())[0, 0]
    y_map = torch.sigmoid(outputs["minutia_y"].detach().float())[0, 0]
    return x_map.cpu().numpy().astype(np.float32), y_map.cpu().numpy().astype(np.float32)


def _angle_map(outputs: dict[str, torch.Tensor]) -> np.ndarray:
    angle = _orientation_vectors_to_radians(outputs["minutia_orientation"].detach())[0]
    return angle.cpu().numpy().astype(np.float32)


def _local_maxima_mask(score_map: np.ndarray) -> np.ndarray:
    score = torch.from_numpy(score_map).unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(score, kernel_size=3, stride=1, padding=1)[0, 0].numpy()
    return score_map >= (pooled - 1e-8)


def _threshold_stats(score_map: np.ndarray, threshold: float) -> dict[str, int]:
    active = score_map >= float(threshold)
    local_maxima = active & _local_maxima_mask(score_map)
    return {
        "raw_active_cells": int(np.count_nonzero(active)),
        "local_maxima_cells": int(np.count_nonzero(local_maxima)),
    }


def _mask_frame(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.where(mask > 0)
    if xs.size < 8:
        raise RuntimeError("cannot estimate frame from an empty/degenerate mask")
    points = np.column_stack([xs, ys]).astype(np.float32)
    center = points.mean(axis=0)
    centered = points - center
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis_u = eigenvectors[:, int(np.argmax(eigenvalues))].astype(np.float32)
    axis_u /= max(float(np.linalg.norm(axis_u)), 1e-6)
    if axis_u[1] < 0:
        axis_u = -axis_u
    axis_v = np.array([-axis_u[1], axis_u[0]], dtype=np.float32)
    long_coord = centered @ axis_u
    cross_coord = centered @ axis_v
    return {
        "center": center,
        "axis_u": axis_u,
        "axis_v": axis_v,
        "long_span": max(float(np.percentile(long_coord, 95) - np.percentile(long_coord, 5)), 1.0),
        "cross_span": max(float(np.percentile(cross_coord, 95) - np.percentile(cross_coord, 5)), 1.0),
    }


def _affine_to_base(src_mask: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
    src = _mask_frame(src_mask)
    base = _mask_frame(base_mask)
    scale_long = float(base["long_span"] / src["long_span"])
    scale_cross = float(base["cross_span"] / src["cross_span"])
    src_basis = np.column_stack([src["axis_u"], src["axis_v"]]).astype(np.float32)
    base_basis = np.column_stack([base["axis_u"] * scale_long, base["axis_v"] * scale_cross]).astype(np.float32)
    linear = base_basis @ np.linalg.inv(src_basis)
    translation = base["center"] - (linear @ src["center"])
    affine = np.eye(3, dtype=np.float32)
    affine[:2, :2] = linear
    affine[:2, 2] = translation
    return affine


def _transform_xy(points_xy: np.ndarray, affine: np.ndarray) -> np.ndarray:
    if points_xy.size == 0:
        return points_xy.reshape(0, 2)
    hom = np.column_stack([points_xy[:, 0], points_xy[:, 1], np.ones(points_xy.shape[0], dtype=np.float32)])
    mapped = hom @ affine.T
    return mapped[:, :2]


def _transform_rows(rows: list[dict[str, float]], affine: np.ndarray) -> list[dict[str, float]]:
    if not rows:
        return []
    xy = np.asarray([[row["x"], row["y"]] for row in rows], dtype=np.float32)
    mapped = _transform_xy(xy, affine)
    transformed: list[dict[str, float]] = []
    for row, (x, y) in zip(rows, mapped):
        item = dict(row)
        item["x_mapped"] = float(x)
        item["y_mapped"] = float(y)
        transformed.append(item)
    return transformed


def _nearest(row: dict[str, float], candidates: list[dict[str, float]]) -> tuple[dict[str, float] | None, float]:
    if not candidates:
        return None, float("inf")
    xy = np.asarray([[cand.get("x_mapped", cand["x"]), cand.get("y_mapped", cand["y"])] for cand in candidates])
    point = np.asarray([row["x"], row["y"]], dtype=np.float32)
    distances = np.sqrt(((xy - point) ** 2).sum(axis=1))
    index = int(np.argmin(distances))
    return candidates[index], float(distances[index])


def _angle_delta(a: float, b: float) -> float:
    delta = (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi
    delta = abs(delta)
    return min(delta, abs(math.pi - delta))


def _inverse_map_point(point_xy: tuple[float, float], affine_to_base: np.ndarray) -> tuple[float, float]:
    inverse = np.linalg.inv(affine_to_base)
    point = np.asarray([point_xy[0], point_xy[1], 1.0], dtype=np.float32)
    mapped = inverse @ point
    return float(mapped[0]), float(mapped[1])


def _sample_score_at_input_xy(score_map: np.ndarray, input_shape_hw: tuple[int, int], point_xy: tuple[float, float]) -> float:
    in_h, in_w = input_shape_hw
    out_h, out_w = score_map.shape
    scale_x = float(in_w) / max(float(out_w), 1.0)
    scale_y = float(in_h) / max(float(out_h), 1.0)
    col = int(round(point_xy[0] / scale_x))
    row = int(round(point_xy[1] / scale_y))
    if row < 0 or col < 0 or row >= out_h or col >= out_w:
        return 0.0
    return float(score_map[row, col])


def _match_counts(base_rows: list[dict[str, float]], transformed_rows: list[dict[str, float]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for radius in MATCH_RADII:
        matched = 0
        for row in base_rows:
            _cand, dist = _nearest(row, transformed_rows)
            if dist <= radius:
                matched += 1
        counts[f"within_{int(radius)}px"] = matched
    return counts


def _classify_baseline_rows(
    *,
    base_rows: list[dict[str, float]],
    src_raw_rows_mapped: list[dict[str, float]],
    src_nms_rows_mapped: list[dict[str, float]],
    src_score_map: np.ndarray,
    src_input_shape_hw: tuple[int, int],
    affine_to_base: np.ndarray,
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counter: Counter[str] = Counter()
    for index, base in enumerate(base_rows):
        raw, raw_dist = _nearest(base, src_raw_rows_mapped)
        nms, nms_dist = _nearest(base, src_nms_rows_mapped)
        src_xy = _inverse_map_point((base["x"], base["y"]), affine_to_base)
        corresponding_score = _sample_score_at_input_xy(src_score_map, src_input_shape_hw, src_xy)
        raw_score = float(raw["score"]) if raw is not None else 0.0
        nms_score = float(nms["score"]) if nms is not None else 0.0
        angle_delta = _angle_delta(base["angle"], nms["angle"]) if nms is not None else float("nan")

        if nms is not None and nms_dist <= 16.0:
            if angle_delta > ANGLE_LIMIT:
                reason = "orientation_changed"
            else:
                reason = "matched"
        elif raw is not None and raw_dist <= 16.0:
            reason = "nms_suppressed"
        elif nms is not None and nms_dist <= 32.0:
            reason = "offset_moved"
        elif corresponding_score < float(threshold) and corresponding_score >= 0.3:
            reason = "threshold_drop"
        elif raw is not None and raw_dist <= 32.0:
            reason = "offset_moved"
        else:
            reason = "model_score_changed"
        counter[reason] += 1
        rows.append(
            {
                "base_index": index,
                "base_x": float(base["x"]),
                "base_y": float(base["y"]),
                "base_angle": float(base["angle"]),
                "base_score": float(base["score"]),
                "corresponding_src_score": corresponding_score,
                "nearest_raw_distance": raw_dist,
                "nearest_raw_score": raw_score,
                "nearest_nms_distance": nms_dist,
                "nearest_nms_score": nms_score,
                "nearest_nms_angle_delta_rad": angle_delta,
                "classification": reason,
            }
        )
    return rows, dict(counter)


def _score_map_correlation(
    base_score: np.ndarray,
    src_score: np.ndarray,
    base_mask: np.ndarray,
    src_mask: np.ndarray,
    affine_to_base: np.ndarray,
    input_shape_hw: tuple[int, int],
) -> dict[str, float]:
    in_h, in_w = input_shape_hw
    out_h, out_w = base_score.shape
    scale_x = float(in_w) / max(float(out_w), 1.0)
    scale_y = float(in_h) / max(float(out_h), 1.0)
    to_input = np.asarray([[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    to_grid = np.asarray([[1.0 / scale_x, 0.0, 0.0], [0.0, 1.0 / scale_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    grid_affine = to_grid @ affine_to_base @ to_input
    warped_score = cv2.warpAffine(
        src_score,
        grid_affine[:2],
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    mask_small = cv2.resize(_u8_mask(base_mask), (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0
    base_values = base_score[mask_small].reshape(-1)
    warped_values = warped_score[mask_small].reshape(-1)
    if base_values.size < 2 or float(np.std(base_values)) <= 1e-8 or float(np.std(warped_values)) <= 1e-8:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(base_values, warped_values)[0, 1])
    return {
        "score_map_correlation_in_base_mask": correlation,
        "mean_abs_score_delta_in_base_mask": float(np.mean(np.abs(base_values - warped_values))) if base_values.size else 0.0,
        "max_abs_score_delta_in_base_mask": float(np.max(np.abs(base_values - warped_values))) if base_values.size else 0.0,
    }


def _run_case(
    *,
    name: str,
    image_path: Path,
    output_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    score_threshold: float,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    case_dir = output_dir / name
    preprocess_dir = case_dir / "preprocess"
    finger_dir = case_dir / "finger_pad_auto"
    case_dir.mkdir(parents=True, exist_ok=True)

    full_bgr = load_bgr_image(image_path)
    segmentation = segment_finger_pad(full_bgr)
    finger_debug = save_finger_pad_debug(output_dir=finger_dir, bgr=full_bgr, segmentation=segmentation)
    image_tensor, mask_tensor, input_shape_hw = preprocess_input_bgr(
        full_bgr,
        save_preprocess_dir=preprocess_dir,
        solov2_score_thr=0.15,
        fallback_mask=segmentation.distal_mask,
        fallback_mask_source="finger_pad_auto",
    )
    outputs = run_inference(model=model, image_tensor=image_tensor, mask_tensor=mask_tensor, device=device)
    serialize_outputs(outputs, case_dir / "outputs.npz")
    orientation_npy, ridge_period_npy = save_pose_sidecars(outputs, case_dir)
    mask_png = case_dir / "mask.png"
    _save_mask_png(mask_tensor, mask_png)

    score_map = _score_map(outputs)
    x_offsets, y_offsets = _offset_maps(outputs)
    angle_map = _angle_map(outputs)
    np.save(case_dir / "minutia_score_sigmoid.npy", score_map)
    np.save(case_dir / "minutia_x_offset_sigmoid.npy", x_offsets)
    np.save(case_dir / "minutia_y_offset_sigmoid.npy", y_offsets)
    np.save(case_dir / "minutia_orientation_radians.npy", angle_map)

    threshold_rows: list[dict[str, Any]] = []
    decoded: dict[str, dict[str, list[dict[str, float]]]] = {"nms_on": {}, "nms_off": {}}
    for threshold in thresholds:
        stats = _threshold_stats(score_map, threshold)
        for apply_nms, mode_name in ((True, "nms_on"), (False, "nms_off")):
            rows = decode_minutiae_rows(
                outputs=outputs,
                input_shape_hw=input_shape_hw,
                score_threshold=float(threshold),
                apply_nms=apply_nms,
            )
            decoded[mode_name][f"{threshold:.2f}"] = rows
            csv_path = case_dir / f"decoded_threshold_{threshold:.2f}_{mode_name}.csv"
            save_minutiae_csv(rows, csv_path)
            threshold_rows.append(
                {
                    "case": name,
                    "threshold": f"{threshold:.2f}",
                    "nms": mode_name,
                    "decoded_count": len(rows),
                    **stats,
                }
            )
    _write_csv(
        case_dir / "threshold_sweep.csv",
        threshold_rows,
        ["case", "threshold", "nms", "decoded_count", "raw_active_cells", "local_maxima_cells"],
    )

    return {
        "name": name,
        "image_path": str(image_path),
        "case_dir": str(case_dir),
        "input_shape_hw": [int(input_shape_hw[0]), int(input_shape_hw[1])],
        "score_map_shape_hw": list(score_map.shape),
        "score_map": score_map,
        "mask": cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE),
        "decoded": decoded,
        "threshold_sweep": threshold_rows,
        "finger_pad": segmentation.diagnostics,
        "finger_pad_debug": finger_debug,
        "orientation_npy": str(orientation_npy),
        "ridge_period_npy": str(ridge_period_npy),
        "mask_png": str(mask_png),
        "preprocess_dir": str(preprocess_dir),
    }


def _summarize_case(case: dict[str, Any], threshold: float) -> dict[str, Any]:
    key = f"{threshold:.2f}"
    stats = _threshold_stats(case["score_map"], threshold)
    return {
        "input_shape_hw": case["input_shape_hw"],
        "score_map_shape_hw": case["score_map_shape_hw"],
        "raw_active_cells_at_threshold": stats["raw_active_cells"],
        "local_maxima_cells_at_threshold": stats["local_maxima_cells"],
        "decoded_count_nms_on": len(case["decoded"]["nms_on"][key]),
        "decoded_count_nms_off": len(case["decoded"]["nms_off"][key]),
        "finger_pad_candidate": case["finger_pad"].get("candidate"),
        "repair_attempted": case["finger_pad"].get("repair_attempted"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose FeatureNet minutia extraction vs NMS instability.")
    parser.add_argument("--base-image", type=Path, default=Path("amit_right_ind1.jpg"))
    parser.add_argument("--shift-image", type=Path, default=Path("tmp/cb_match_diag/variants/amit_right_ind1_shift_x5.jpg"))
    parser.add_argument("--rot-image", type=Path, default=Path("tmp/cb_match_diag/variants/amit_right_ind1_rot_p5.jpg"))
    parser.add_argument("--weights-path", type=Path, default=Path("weights") / "best.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("tmp") / "cb_minutia_instability_diag")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--fixed-ridge-period", type=float, default=DEFAULT_FIXED_RIDGE_PERIOD)
    parser.add_argument("--target-period", type=float, default=10.0)
    parser.add_argument("--score-threshold", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _fixed_period_patch(float(args.fixed_ridge_period), float(args.target_period))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    model = load_checkpoint_model(args.weights_path, device)
    thresholds = tuple(DEFAULT_THRESHOLDS)

    cases = {
        "base": args.base_image,
        "same_copy": args.base_image,
        "shift_x5": args.shift_image,
        "rot_p5": args.rot_image,
    }
    results = {
        name: _run_case(
            name=name,
            image_path=path,
            output_dir=output_dir,
            model=model,
            device=device,
            score_threshold=float(args.score_threshold),
            thresholds=thresholds,
        )
        for name, path in cases.items()
    }

    base = results["base"]
    base_mask = base["mask"]
    base_rows_nms = base["decoded"]["nms_on"][f"{args.score_threshold:.2f}"]
    base_rows_raw = base["decoded"]["nms_off"][f"{args.score_threshold:.2f}"]
    comparisons: dict[str, Any] = {}
    classification_csv_rows: list[dict[str, Any]] = []

    for name in ("same_copy", "shift_x5", "rot_p5"):
        src = results[name]
        affine = _affine_to_base(src["mask"], base_mask)
        src_raw_mapped = _transform_rows(src["decoded"]["nms_off"][f"{args.score_threshold:.2f}"], affine)
        src_nms_mapped = _transform_rows(src["decoded"]["nms_on"][f"{args.score_threshold:.2f}"], affine)
        per_row, category_counts = _classify_baseline_rows(
            base_rows=base_rows_nms,
            src_raw_rows_mapped=src_raw_mapped,
            src_nms_rows_mapped=src_nms_mapped,
            src_score_map=src["score_map"],
            src_input_shape_hw=tuple(src["input_shape_hw"]),
            affine_to_base=affine,
            threshold=float(args.score_threshold),
        )
        for row in per_row:
            row["comparison"] = name
        classification_csv_rows.extend(per_row)
        _write_csv(
            output_dir / f"{name}_baseline_minutia_classification.csv",
            per_row,
            [
                "comparison",
                "base_index",
                "base_x",
                "base_y",
                "base_angle",
                "base_score",
                "corresponding_src_score",
                "nearest_raw_distance",
                "nearest_raw_score",
                "nearest_nms_distance",
                "nearest_nms_score",
                "nearest_nms_angle_delta_rad",
                "classification",
            ],
        )
        comparisons[name] = {
            "affine_to_base": affine.tolist(),
            "nms_on_match_counts": _match_counts(base_rows_nms, src_nms_mapped),
            "nms_off_match_counts": _match_counts(base_rows_nms, src_raw_mapped),
            "lost_peak_categories": category_counts,
            **_score_map_correlation(
                base["score_map"],
                src["score_map"],
                base["mask"],
                src["mask"],
                affine,
                tuple(base["input_shape_hw"]),
            ),
        }

    _write_csv(
        output_dir / "all_baseline_minutia_classifications.csv",
        classification_csv_rows,
        [
            "comparison",
            "base_index",
            "base_x",
            "base_y",
            "base_angle",
            "base_score",
            "corresponding_src_score",
            "nearest_raw_distance",
            "nearest_raw_score",
            "nearest_nms_distance",
            "nearest_nms_score",
            "nearest_nms_angle_delta_rad",
            "classification",
        ],
    )

    summary_rows = []
    summary = {
        "fixed_ridge_period": float(args.fixed_ridge_period),
        "fixed_scale": float(args.target_period) / float(args.fixed_ridge_period),
        "score_threshold": float(args.score_threshold),
        "cases": {},
        "comparisons": comparisons,
    }
    for name, case in results.items():
        case_summary = _summarize_case(case, float(args.score_threshold))
        summary["cases"][name] = {
            **case_summary,
            "case_dir": case["case_dir"],
            "mask_png": case["mask_png"],
            "preprocess_dir": case["preprocess_dir"],
        }
        summary_rows.append({"case": name, **case_summary})
    _write_csv(
        output_dir / "summary_table.csv",
        summary_rows,
        [
            "case",
            "input_shape_hw",
            "score_map_shape_hw",
            "raw_active_cells_at_threshold",
            "local_maxima_cells_at_threshold",
            "decoded_count_nms_on",
            "decoded_count_nms_off",
            "finger_pad_candidate",
            "repair_attempted",
        ],
    )
    (output_dir / "diagnosis_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

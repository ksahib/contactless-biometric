from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import diagnose_featurenet_minutia_instability as diag

import cv2
import numpy as np


def _score_pair(
    *,
    case_a: dict[str, Any],
    case_b: dict[str, Any],
    method: str,
    threshold: float,
) -> tuple[float, list[int]]:
    import main as mcc_main

    score, sim_matrix = mcc_main.match_minutiae_csv(
        path_a=Path(case_a["case_dir"]) / f"decoded_threshold_{threshold:.2f}_nms_on.csv",
        path_b=Path(case_b["case_dir"]) / f"decoded_threshold_{threshold:.2f}_nms_on.csv",
        method=method,
        mask_path_a=Path(case_a["mask_png"]),
        mask_path_b=Path(case_b["mask_png"]),
        orientation_path_a=case_a["orientation_npy"],
        orientation_path_b=case_b["orientation_npy"],
        ridge_period_path_a=case_a["ridge_period_npy"],
        ridge_period_path_b=case_b["ridge_period_npy"],
        overlap_mode="auto",
    )
    return float(score), list(np.asarray(sim_matrix).shape)


def _preprocess_meta(case: dict[str, Any]) -> dict[str, Any]:
    meta_path = Path(case["preprocess_dir"]) / "meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _patch_preprocess_with_reference(reference_meta: dict[str, Any]) -> tuple[Any, Any]:
    original_scale = diag.preprocess_module.scale_to_paper_ridge_period
    original_rotate = diag.preprocess_module.rotate_to_vertical_centerline
    forced_ridge_period = float(reference_meta["ridge_period"])
    forced_scale = float(reference_meta["scale"])
    forced_yaw = float(reference_meta["yaw_angle"])

    def forced_scale_to_paper_ridge_period(
        enhanced: np.ndarray,
        full_mask: np.ndarray,
        *,
        target_period: float = 10.0,
        orientation: float | np.ndarray | None = None,
        center_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        scaled_image = cv2.resize(
            enhanced,
            None,
            fx=forced_scale,
            fy=forced_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        scaled_mask = cv2.resize(
            full_mask,
            None,
            fx=forced_scale,
            fy=forced_scale,
            interpolation=cv2.INTER_NEAREST,
        )
        return (
            scaled_image.astype(np.uint8),
            np.where(scaled_mask > 0, 255, 0).astype(np.uint8),
            forced_ridge_period,
            forced_scale,
        )

    def forced_rotate_to_vertical_centerline(
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        image_u8 = np.asarray(image).astype(np.uint8)
        mask_u8 = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
        rotated_image = diag.preprocess_module._rotate_same_canvas(
            image_u8,
            forced_yaw,
            interpolation=cv2.INTER_LINEAR,
            border_value=0,
        )
        rotated_mask = diag.preprocess_module._rotate_same_canvas(
            mask_u8,
            forced_yaw,
            interpolation=cv2.INTER_NEAREST,
            border_value=0,
        )
        rotated_mask = np.where(rotated_mask > 0, 255, 0).astype(np.uint8)
        rotated_image[rotated_mask <= 0] = 0
        return rotated_image.astype(np.uint8), rotated_mask, forced_yaw

    diag.preprocess_module.scale_to_paper_ridge_period = forced_scale_to_paper_ridge_period
    diag.preprocess_module.rotate_to_vertical_centerline = forced_rotate_to_vertical_centerline
    return original_scale, original_rotate


def _restore_preprocess(original_scale: Any, original_rotate: Any) -> None:
    diag.preprocess_module.scale_to_paper_ridge_period = original_scale
    diag.preprocess_module.rotate_to_vertical_centerline = original_rotate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ind1/ind2 normal preprocessing against ind2 forced to use ind1 scale/yaw."
    )
    parser.add_argument("--image-a", type=Path, default=Path("amit_right_ind1.jpg"))
    parser.add_argument("--image-b", type=Path, default=Path("amit_right_ind2.jpg"))
    parser.add_argument("--weights-path", type=Path, default=Path("weights") / "best.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("tmp") / "cb_ind1_ind2_forced_preprocess")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--method", type=str, default="LSA-CENTROID")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = diag._resolve_device(args.device)
    model = diag.load_checkpoint_model(args.weights_path, device)
    thresholds = tuple(diag.DEFAULT_THRESHOLDS)

    normal_a = diag._run_case(
        name="ind1_normal",
        image_path=args.image_a,
        output_dir=output_dir,
        model=model,
        device=device,
        score_threshold=float(args.score_threshold),
        thresholds=thresholds,
    )
    normal_b = diag._run_case(
        name="ind2_normal",
        image_path=args.image_b,
        output_dir=output_dir,
        model=model,
        device=device,
        score_threshold=float(args.score_threshold),
        thresholds=thresholds,
    )
    normal_score, normal_shape = _score_pair(
        case_a=normal_a,
        case_b=normal_b,
        method=str(args.method),
        threshold=float(args.score_threshold),
    )

    ref_meta = _preprocess_meta(normal_a)
    original_scale, original_rotate = _patch_preprocess_with_reference(ref_meta)
    try:
        forced_b = diag._run_case(
            name="ind2_forced_ind1_preprocess",
            image_path=args.image_b,
            output_dir=output_dir,
            model=model,
            device=device,
            score_threshold=float(args.score_threshold),
            thresholds=thresholds,
        )
    finally:
        _restore_preprocess(original_scale, original_rotate)

    forced_score, forced_shape = _score_pair(
        case_a=normal_a,
        case_b=forced_b,
        method=str(args.method),
        threshold=float(args.score_threshold),
    )

    report = {
        "image_a": str(args.image_a),
        "image_b": str(args.image_b),
        "weights_path": str(args.weights_path),
        "method": str(args.method),
        "score_threshold": float(args.score_threshold),
        "minutia_nms_enabled": True,
        "normal": {
            "score": normal_score,
            "similarity_matrix_shape": normal_shape,
            "a_minutiae": len(normal_a["decoded"]["nms_on"][f"{args.score_threshold:.2f}"]),
            "b_minutiae": len(normal_b["decoded"]["nms_on"][f"{args.score_threshold:.2f}"]),
            "a_preprocess": _preprocess_meta(normal_a),
            "b_preprocess": _preprocess_meta(normal_b),
        },
        "forced_b_uses_a_preprocess_scalars": {
            "score": forced_score,
            "similarity_matrix_shape": forced_shape,
            "a_minutiae": len(normal_a["decoded"]["nms_on"][f"{args.score_threshold:.2f}"]),
            "b_minutiae": len(forced_b["decoded"]["nms_on"][f"{args.score_threshold:.2f}"]),
            "forced_values_from_a": {
                "ridge_period": float(ref_meta["ridge_period"]),
                "scale": float(ref_meta["scale"]),
                "yaw_angle": float(ref_meta["yaw_angle"]),
            },
            "b_forced_preprocess": _preprocess_meta(forced_b),
        },
        "artifacts": {
            "run_dir": str(output_dir),
            "report_json": str(output_dir / "report.json"),
            "ind1_normal_dir": normal_a["case_dir"],
            "ind2_normal_dir": normal_b["case_dir"],
            "ind2_forced_dir": forced_b["case_dir"],
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

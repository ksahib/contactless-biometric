from __future__ import annotations

import argparse
import csv
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


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_number_list(value: str, *, as_int: bool = False) -> list[int | float]:
    items: list[int | float] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        items.append(int(float(raw)) if as_int else float(raw))
    if not items:
        raise ValueError("number list must contain at least one value")
    return items


def _shift_image(bgr: np.ndarray, dx: int, dy: int = 0) -> np.ndarray:
    height, width = bgr.shape[:2]
    matrix = np.asarray([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    return cv2.warpAffine(
        bgr,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _rotate_same_canvas(bgr: np.ndarray, angle_degrees: float) -> np.ndarray:
    height, width = bgr.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_degrees), 1.0)
    return cv2.warpAffine(
        bgr,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _rotate_bound(bgr: np.ndarray, angle_degrees: float) -> np.ndarray:
    height, width = bgr.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_degrees), 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    matrix[0, 2] += (new_width / 2.0) - center[0]
    matrix[1, 2] += (new_height / 2.0) - center[1]
    return cv2.warpAffine(
        bgr,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _variant_paths(
    *,
    base_image: Path,
    variant_image: Path,
    output_dir: Path,
    shifts: list[int],
    rotations: list[float],
    rotation_canvas: str,
    combined: bool,
) -> dict[str, Path]:
    bgr = diag.load_bgr_image(variant_image)
    variant_dir = output_dir / "variants"
    variant_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if combined:
        for px in shifts:
            shifted = _shift_image(bgr, int(px), 0)
            for degrees in rotations:
                label = str(degrees).replace("-", "m").replace(".", "p")
                if not label.startswith("m"):
                    label = f"p{label}"
                name = f"shift_x{px}_rot_{label}"
                path = variant_dir / f"{variant_image.stem}_{name}.jpg"
                rotated = (
                    _rotate_bound(shifted, float(degrees))
                    if rotation_canvas == "bound"
                    else _rotate_same_canvas(shifted, float(degrees))
                )
                if not cv2.imwrite(str(path), rotated):
                    raise RuntimeError(f"failed to write combined transform image: {path}")
                paths[name] = path
    else:
        for px in shifts:
            name = f"shift_x{px}"
            path = variant_dir / f"{variant_image.stem}_{name}.jpg"
            if not cv2.imwrite(str(path), _shift_image(bgr, int(px), 0)):
                raise RuntimeError(f"failed to write shifted image: {path}")
            paths[name] = path

        for degrees in rotations:
            label = str(degrees).replace("-", "m").replace(".", "p")
            if not label.startswith("m"):
                label = f"p{label}"
            name = f"rot_{label}"
            path = variant_dir / f"{variant_image.stem}_{name}.jpg"
            rotated = _rotate_bound(bgr, float(degrees)) if rotation_canvas == "bound" else _rotate_same_canvas(bgr, float(degrees))
            if not cv2.imwrite(str(path), rotated):
                raise RuntimeError(f"failed to write rotated image: {path}")
            paths[name] = path

    return paths


def _decoded_csv(case_dir: Path, threshold: float, *, nms_on: bool = True) -> Path:
    mode = "nms_on" if nms_on else "nms_off"
    return case_dir / f"decoded_threshold_{threshold:.2f}_{mode}.csv"


def _score_pair(
    *,
    base_case: dict[str, Any],
    variant_case: dict[str, Any],
    method: str,
    threshold: float,
) -> tuple[float, list[int]]:
    import main as mcc_main

    base_dir = Path(base_case["case_dir"])
    variant_dir = Path(variant_case["case_dir"])
    score, sim_matrix = mcc_main.match_minutiae_csv(
        path_a=_decoded_csv(base_dir, threshold, nms_on=True),
        path_b=_decoded_csv(variant_dir, threshold, nms_on=True),
        method=method,
        mask_path_a=Path(base_case["mask_png"]),
        mask_path_b=Path(variant_case["mask_png"]),
        orientation_path_a=base_case["orientation_npy"],
        orientation_path_b=variant_case["orientation_npy"],
        ridge_period_path_a=base_case["ridge_period_npy"],
        ridge_period_path_b=variant_case["ridge_period_npy"],
        overlap_mode="auto",
    )
    return float(score), list(np.asarray(sim_matrix).shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep fixed-ridge-period FeatureNet transform variants and report MCC matching scores."
    )
    parser.add_argument("--base-image", type=Path, default=Path("amit_right_ind1.jpg"))
    parser.add_argument(
        "--variant-image",
        type=Path,
        default=None,
        help="Image to transform before matching against --base-image. Defaults to --base-image.",
    )
    parser.add_argument("--weights-path", type=Path, default=Path("weights") / "best.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("tmp") / "cb_fixed_period_transform_scores")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--fixed-ridge-period", type=float, default=diag.DEFAULT_FIXED_RIDGE_PERIOD)
    parser.add_argument("--target-period", type=float, default=10.0)
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--method", type=str, default="LSA-CENTROID")
    parser.add_argument("--shifts", type=str, default="10,15,20")
    parser.add_argument("--rotations", type=str, default="10,15,20")
    parser.add_argument("--rotation-canvas", choices=("same", "bound"), default="same")
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Generate shift-then-rotate variants for every shift/rotation pair instead of separate shift and rotation variants.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    shifts = [int(value) for value in _parse_number_list(args.shifts, as_int=True)]
    rotations = [float(value) for value in _parse_number_list(args.rotations)]
    variant_image = args.variant_image if args.variant_image is not None else args.base_image

    diag._fixed_period_patch(float(args.fixed_ridge_period), float(args.target_period))
    device = diag._resolve_device(args.device)
    model = diag.load_checkpoint_model(args.weights_path, device)
    thresholds = tuple(diag.DEFAULT_THRESHOLDS)
    variants = _variant_paths(
        base_image=args.base_image,
        variant_image=variant_image,
        output_dir=output_dir,
        shifts=shifts,
        rotations=rotations,
        rotation_canvas=str(args.rotation_canvas),
        combined=bool(args.combined),
    )

    same_label = "same_copy" if variant_image == args.base_image else "variant_original"
    case_paths = {"base": args.base_image, same_label: variant_image, **variants}
    cases: dict[str, dict[str, Any]] = {}
    for name, path in case_paths.items():
        cases[name] = diag._run_case(
            name=name,
            image_path=path,
            output_dir=output_dir,
            model=model,
            device=device,
            score_threshold=float(args.score_threshold),
            thresholds=thresholds,
        )

    rows: list[dict[str, Any]] = []
    for name, case in cases.items():
        if name == "base":
            continue
        score, sim_shape = _score_pair(
            base_case=cases["base"],
            variant_case=case,
            method=str(args.method),
            threshold=float(args.score_threshold),
        )
        variant_kind = (
            "same" if name == "same_copy" else "variant_original" if name == "variant_original" else
            ("combined" if "_rot_" in name and name.startswith("shift_") else ("shift" if name.startswith("shift_") else "rotation"))
        )
        rows.append(
            {
                "case": name,
                "kind": variant_kind,
                "method": str(args.method),
                "fixed_ridge_period": float(args.fixed_ridge_period),
                "score_threshold": float(args.score_threshold),
                "nms": "on",
                "mcc_score": score,
                "base_minutiae": len(cases["base"]["decoded"]["nms_on"][f"{args.score_threshold:.2f}"]),
                "variant_minutiae": len(case["decoded"]["nms_on"][f"{args.score_threshold:.2f}"]),
                "base_input_shape_hw": cases["base"]["input_shape_hw"],
                "variant_input_shape_hw": case["input_shape_hw"],
                "similarity_matrix_shape": sim_shape,
                "variant_image": str(case_paths[name]),
                "case_dir": case["case_dir"],
            }
        )

    fieldnames = [
        "case",
        "kind",
        "method",
        "fixed_ridge_period",
        "score_threshold",
        "nms",
        "mcc_score",
        "base_minutiae",
        "variant_minutiae",
        "base_input_shape_hw",
        "variant_input_shape_hw",
        "similarity_matrix_shape",
        "variant_image",
        "case_dir",
    ]
    _write_csv(output_dir / "mcc_transform_scores.csv", rows, fieldnames)
    report = {
        "base_image": str(args.base_image),
        "variant_image": str(variant_image),
        "weights_path": str(args.weights_path),
        "method": str(args.method),
        "fixed_ridge_period": float(args.fixed_ridge_period),
        "fixed_scale": float(args.target_period) / float(args.fixed_ridge_period),
        "score_threshold": float(args.score_threshold),
        "minutia_nms_enabled": True,
        "shifts_px": shifts,
        "rotations_degrees": rotations,
        "rotation_canvas": str(args.rotation_canvas),
        "combined_shift_then_rotate": bool(args.combined),
        "scores": rows,
        "outputs": {
            "scores_csv": str(output_dir / "mcc_transform_scores.csv"),
            "run_dir": str(output_dir),
        },
    }
    (output_dir / "mcc_transform_scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

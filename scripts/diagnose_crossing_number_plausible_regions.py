#!/usr/bin/env python
"""Smoke-test crossing-number plausible minutia regions.

This diagnostic inspects one generated bundle, skeletonizes the center unwarped
image, computes crossing-number candidates, expands them into plausible regions,
and overlays those regions on both the unwarped image and the model-aligned
training input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))

import generate_ground_truth as gt  # noqa: E402


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"unable to load grayscale image: {path}")
    return image


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.shape[:2] != canvas.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return canvas

    tint = np.zeros_like(canvas)
    tint[:, :] = np.asarray(color, dtype=np.uint8)
    tinted = cv2.addWeighted(canvas, 1.0, tint, float(alpha), 0.0)
    canvas[mask_bool] = tinted[mask_bool]
    return canvas


def _load_sample_context(sample_dir: Path) -> tuple[dict[str, Any], Path, dict[str, Any], np.ndarray, np.ndarray]:
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing bundle metadata: {meta_path}")
    meta = _read_json(meta_path)
    reconstruction_info = meta.get("multiview_reconstruction")
    if not isinstance(reconstruction_info, dict) or not reconstruction_info.get("reconstruction_dir"):
        raise RuntimeError("sample metadata does not contain a reconstruction_dir")

    reconstruction_dir = Path(str(reconstruction_info["reconstruction_dir"])).expanduser().resolve()
    if not reconstruction_dir.exists():
        raise FileNotFoundError(f"missing reconstruction directory: {reconstruction_dir}")

    raw_view_index = int(meta.get("raw_view_index", -1))
    role = {0: "front", 1: "left", 2: "right"}.get(raw_view_index)
    if role is None:
        raise RuntimeError(f"unsupported raw_view_index for smoke test: {raw_view_index}")
    if role == "front":
        label_frame = {
            "role": role,
            "label_frame": "center_unwrapped",
            "image_path": reconstruction_dir / "center_unwarped.png",
            "mask_path": reconstruction_dir / "center_unwarped_mask.png",
            "maps_path": reconstruction_dir / "center_unwarp_maps.npz",
        }
    else:
        side_dir = reconstruction_dir / "side_depth_unwrap_v4" / role
        label_frame = {
            "role": role,
            "label_frame": "algorithm1_v4_side_unwrapped",
            "image_path": side_dir / f"{role}_depth_unwrapped.png",
            "mask_path": side_dir / f"{role}_depth_unwrapped_mask.png",
            "maps_path": side_dir / f"{role}_depth_unwarp_maps.npz",
        }
    source_image = Path(label_frame["image_path"])
    source_mask = Path(label_frame["mask_path"])
    if not source_image.exists() or not source_mask.exists():
        raise FileNotFoundError(f"missing label-frame reconstruction artifacts for role={role}")

    return meta, reconstruction_dir, label_frame, _read_gray(source_image), _read_gray(source_mask)


def _run(sample_dir: Path, output_dir: Path, polarity: str, dilation_radius: int) -> dict[str, Any]:
    meta, reconstruction_dir, label_frame, unwarped_gray, unwarped_mask = _load_sample_context(sample_dir)
    masked_image = _read_gray(sample_dir / "masked_image.png")
    raw_input_path = sample_dir / "raw_input.png"
    raw_image = _read_gray(raw_input_path) if raw_input_path.exists() else masked_image.copy()

    plausible_unwarped_mask, cn_details = gt._build_crossing_number_plausible_mask(
        unwarped_gray,
        unwarped_mask,
        polarity=polarity,
        dilation_radius=dilation_radius,
    )

    reconstruction = meta.get("multiview_reconstruction")
    if not isinstance(reconstruction, dict):
        raise RuntimeError("missing multiview reconstruction metadata")
    role = label_frame["role"]
    prepared = gt.PreparedBundleArtifacts(
        sample=gt.RawViewSample(
            sample_id=str(meta["sample_id"]),
            subject_id=int(meta["subject_id"]),
            subject_index=int(meta.get("subject_index", 0)),
            finger_id=int(meta["finger_id"]),
            acquisition_id=int(meta["acquisition_id"]),
            finger_class_id=int(meta["finger_class_id"]),
            raw_image_path=str(meta.get("raw_image_path", raw_input_path)),
            raw_view_index=int(meta["raw_view_index"]),
            sire_path=meta.get("sire_path"),
            raw_view_paths=[str(p) for p in meta.get("raw_view_paths", [])],
            variant_paths=dict(meta.get("variant_paths", {})),
            is_extra_acquisition=bool(meta.get("is_extra_acquisition", False)),
        ),
        bundle_dir=sample_dir,
        image_path=raw_input_path,
        preprocessed=gt.PreprocessedContactlessImage(
            raw_gray=raw_image,
            normalized_gray=raw_image,
            pose_normalized_gray=raw_image,
            pose_normalized_mask=np.where(unwarped_mask > 0, 255, 0).astype(np.uint8),
            preprocessed_gray=masked_image,
            final_mask=np.where(masked_image > 0, 255, 0).astype(np.uint8),
            mask_source="bundle",
            pose_rotation_degrees=0.0,
            ridge_scale_factor=1.0,
        ),
        gray_image=masked_image,
        mask=np.where(masked_image > 0, 255, 0).astype(np.uint8),
        orientation=np.zeros_like(masked_image, dtype=np.float32),
        ridge_period=np.zeros_like(masked_image, dtype=np.float32),
        visualization_gradient=np.zeros((*masked_image.shape, 2), dtype=np.float32),
        reconstruction_gradient=None,
        masked_image=masked_image,
        enhanced_image=masked_image,
        visualize=False,
        reconstruction=gt.AcquisitionReconstructionResult(
            acquisition_id=str(reconstruction.get("acquisition_id") or reconstruction_dir.name),
            reconstruction_dir=str(reconstruction_dir),
            depth_front_path=str(reconstruction.get("depth_front_path", "")),
            depth_left_path=str(reconstruction.get("depth_left_path", "")),
            depth_right_path=str(reconstruction.get("depth_right_path", "")),
            depth_gradient_labels_path=str(reconstruction.get("depth_gradient_labels_path", "")),
            reconstruction_maps_path=str(reconstruction.get("reconstruction_maps_path", "")),
            support_mask_path=str(reconstruction.get("support_mask_path", "")),
            row_measurements_path=str(reconstruction.get("row_measurements_path", "")),
            meta_path=str(reconstruction.get("meta_path", "")),
            preview_path=str(reconstruction.get("preview_path", "")),
            center_unwarp_maps_path=str(reconstruction.get("center_unwarp_maps_path", "")),
            center_unwarped_image_path=str(reconstruction_dir / "center_unwarped.png"),
            center_unwarped_mask_path=str(reconstruction_dir / "center_unwarped_mask.png"),
            surface_front_3d_html_path=str(reconstruction.get("surface_front_3d_html_path", "")),
            surface_front_3d_png_path=str(reconstruction.get("surface_front_3d_png_path", "")),
            surface_all_branches_3d_html_path=str(reconstruction.get("surface_all_branches_3d_html_path", "")),
            surface_all_branches_3d_png_path=str(reconstruction.get("surface_all_branches_3d_png_path", "")),
            reprojection_report_path=str(reconstruction.get("reprojection_report_path", "")),
            reprojection_preview_path=str(reconstruction.get("reprojection_preview_path", "")),
            valid_row_count=int(reconstruction.get("valid_row_count", 1)),
            support_pixel_count=int(reconstruction.get("support_pixel_count", 1)),
            input_view_paths={},
            debug_view_paths={},
        ),
    )
    aligned_mask, reprojection_details = gt._build_crossing_number_plausible_reprojected_mask(
        prepared,
        polarity=polarity,
        dilation_radius=dilation_radius,
    )
    if aligned_mask is None:
        raise RuntimeError("crossing-number reprojection did not produce an aligned mask")

    aligned_mask_u8 = np.where(aligned_mask > 0, 255, 0).astype(np.uint8)
    if aligned_mask_u8.ndim == 3 and aligned_mask_u8.shape[0] == 1:
        aligned_mask_u8 = aligned_mask_u8[0]
    aligned_overlay = _overlay_mask(masked_image, aligned_mask_u8, (0, 255, 0))
    raw_overlay = _overlay_mask(
        raw_image,
        cv2.resize(aligned_mask_u8, (raw_image.shape[1], raw_image.shape[0]), interpolation=cv2.INTER_NEAREST),
        (0, 255, 0),
    )
    unwarped_overlay = _overlay_mask(unwarped_gray, plausible_unwarped_mask, (0, 255, 0))

    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "unwarped_plausible_mask.png"), plausible_unwarped_mask)
    cv2.imwrite(str(output_dir / "unwarped_plausible_overlay.png"), unwarped_overlay)
    cv2.imwrite(str(output_dir / "aligned_plausible_mask.png"), aligned_mask_u8)
    cv2.imwrite(str(output_dir / "aligned_plausible_overlay.png"), aligned_overlay)
    cv2.imwrite(str(output_dir / "raw_plausible_overlay.png"), raw_overlay)

    report = {
        "sample_id": meta.get("sample_id"),
        "bundle_dir": str(sample_dir.resolve()),
        "reconstruction_dir": str(reconstruction_dir),
        "view_role": role,
        "source_label_frame": str(label_frame.get("label_frame")),
        "source_image_path": str(Path(label_frame["image_path"]).resolve()),
        "source_mask_path": str(Path(label_frame["mask_path"]).resolve()),
        "polarity": cn_details.get("polarity"),
        "threshold": cn_details.get("threshold"),
        "foreground_ratio": cn_details.get("foreground_ratio"),
        "skeleton_pixels": cn_details.get("skeleton_pixels"),
        "candidate_pixels": cn_details.get("candidate_pixels"),
        "plausible_pixels_unwarped": cn_details.get("plausible_pixels"),
        "plausible_pixels_aligned": reprojection_details.get("reprojected_plausible_pixels"),
        "cn_endpoint_pixels": cn_details.get("cn_endpoint_pixels"),
        "cn_bifurcation_pixels": cn_details.get("cn_bifurcation_pixels"),
        "dilation_radius": int(dilation_radius),
        "outputs": {
            "unwarped_plausible_mask": str((output_dir / "unwarped_plausible_mask.png").resolve()),
            "unwarped_plausible_overlay": str((output_dir / "unwarped_plausible_overlay.png").resolve()),
            "aligned_plausible_mask": str((output_dir / "aligned_plausible_mask.png").resolve()),
            "aligned_plausible_overlay": str((output_dir / "aligned_plausible_overlay.png").resolve()),
            "raw_plausible_overlay": str((output_dir / "raw_plausible_overlay.png").resolve()),
        },
        "reprojection": reprojection_details,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True, help="Path to one generated bundle sample directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write smoke-test overlays and report.")
    parser.add_argument(
        "--polarity",
        type=str,
        default="dark",
        choices=("dark", "light", "auto"),
        help="Crossing-number binarization polarity.",
    )
    parser.add_argument("--dilation-radius", type=int, default=0, help="Radius used to expand CN candidate points.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample_dir = args.sample_dir.resolve()
    if args.output_dir is None:
        output_dir = Path("tmp") / "crossing_number_plausible_regions" / sample_dir.name
    else:
        output_dir = args.output_dir.resolve()

    report = _run(sample_dir, output_dir, args.polarity, int(args.dilation_radius))
    print(
        json.dumps(
            {
                "report": report["outputs"],
                "summary": {k: report[k] for k in ("polarity", "threshold", "skeleton_pixels", "candidate_pixels", "plausible_pixels_unwarped", "plausible_pixels_aligned")},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

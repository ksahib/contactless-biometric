#!/usr/bin/env python
"""Smoke-test plausible minutia regions on the aligned image.

This diagnostic inspects one generated bundle, skeletonizes the aligned
training image directly, validates ridge-ending and bifurcation candidates via
skeleton branch tracing and border suppression, and writes overlays for
inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))

from featurenet.models import plausible_minutiae as pm  # noqa: E402


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"unable to load grayscale image: {path}")
    return image


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    return pm.overlay_mask(image, mask, color=color, alpha=alpha)


def _build_bifurcation_plausible_mask(
    aligned_gray: np.ndarray,
    roi_mask: np.ndarray,
    *,
    polarity: str,
    border_margin: int,
    min_branch_length: int,
    suppression_radius: int,
    dilation_radius: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    return pm.build_plausible_minutia_mask(
        aligned_gray,
        roi_mask,
        polarity=polarity,
        border_margin=border_margin,
        min_branch_length=min_branch_length,
        suppression_radius=suppression_radius,
        dilation_radius=dilation_radius,
        include_endings=True,
        include_bifurcations=True,
    )


def _label_frame_for_role(reconstruction_dir: Path, role: str) -> dict[str, Any]:
    return pm.label_frame_for_role(reconstruction_dir, role)


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
    role = pm.role_for_raw_view_index(raw_view_index)
    if role is None:
        raise RuntimeError(f"unsupported raw_view_index for smoke test: {raw_view_index}")

    label_frame = _label_frame_for_role(reconstruction_dir, role)
    source_image = Path(label_frame["image_path"])
    source_mask = Path(label_frame["mask_path"])
    if not source_image.exists() or not source_mask.exists():
        raise FileNotFoundError(f"missing label-frame reconstruction artifacts for role={role}")

    return meta, reconstruction_dir, label_frame, _read_gray(source_image), _read_gray(source_mask)


def _load_aligned_context(sample_dir: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing bundle metadata: {meta_path}")
    meta = _read_json(meta_path)

    aligned_gray = _read_gray(sample_dir / "masked_image.png")
    roi_path = sample_dir / "mask.png"
    roi_mask = _read_gray(roi_path) if roi_path.exists() else np.where(aligned_gray > 0, 255, 0).astype(np.uint8)
    raw_input_path = sample_dir / "raw_input.png"
    raw_image = _read_gray(raw_input_path) if raw_input_path.exists() else aligned_gray.copy()
    return meta, aligned_gray, roi_mask, raw_image


def _run(
    sample_dir: Path,
    output_dir: Path,
    polarity: str,
    min_branch_length: int,
    border_margin: int,
    suppression_radius: int,
    dilation_radius: int,
) -> dict[str, Any]:
    meta, aligned_gray, roi_mask, raw_image = _load_aligned_context(sample_dir)
    plausible_mask, validation_details = _build_bifurcation_plausible_mask(
        aligned_gray,
        roi_mask,
        polarity=polarity,
        border_margin=int(border_margin),
        min_branch_length=int(min_branch_length),
        suppression_radius=int(suppression_radius),
        dilation_radius=int(dilation_radius),
    )

    aligned_mask_u8 = np.where(plausible_mask > 0, 255, 0).astype(np.uint8)
    aligned_overlay = _overlay_mask(aligned_gray, aligned_mask_u8, (0, 255, 0))
    raw_overlay = _overlay_mask(
        raw_image,
        cv2.resize(aligned_mask_u8, (raw_image.shape[1], raw_image.shape[0]), interpolation=cv2.INTER_NEAREST),
        (0, 255, 0),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "source_plausible_mask.png"), aligned_mask_u8)
    cv2.imwrite(str(output_dir / "source_plausible_overlay.png"), aligned_overlay)
    cv2.imwrite(str(output_dir / "aligned_plausible_mask.png"), aligned_mask_u8)
    cv2.imwrite(str(output_dir / "aligned_plausible_overlay.png"), aligned_overlay)
    cv2.imwrite(str(output_dir / "raw_plausible_overlay.png"), raw_overlay)

    report = {
        "sample_id": meta.get("sample_id"),
        "bundle_dir": str(sample_dir.resolve()),
        "mode": validation_details.get("mode"),
        "source_image_path": str((sample_dir / "masked_image.png").resolve()),
        "source_mask_path": str((sample_dir / "mask.png").resolve()),
        "polarity": validation_details.get("polarity"),
        "threshold": validation_details.get("threshold"),
        "foreground_ratio": validation_details.get("foreground_ratio"),
        "skeleton_pixels": validation_details.get("skeleton_pixels"),
        "candidate_pixels": validation_details.get("candidate_pixels"),
        "candidate_endings_pixels": validation_details.get("cn_endpoint_pixels"),
        "candidate_bifurcation_pixels": validation_details.get("cn_bifurcation_pixels"),
        "plausible_pixels_source": validation_details.get("plausible_pixels"),
        "plausible_pixels_aligned": validation_details.get("plausible_pixels"),
        "cn_endpoint_pixels": validation_details.get("cn_endpoint_pixels"),
        "cn_bifurcation_pixels": validation_details.get("cn_bifurcation_pixels"),
        "border_margin": int(border_margin),
        "min_branch_length": int(min_branch_length),
        "suppression_radius": int(suppression_radius),
        "dilation_radius": int(dilation_radius),
        "validated_candidate_count": validation_details.get("validated_candidate_count"),
        "accepted_candidate_count": validation_details.get("accepted_candidate_count"),
        "rejected_candidate_count": validation_details.get("rejected_candidate_count"),
        "suppressed_candidate_count": validation_details.get("suppressed_candidate_count"),
        "rejection_counts": validation_details.get("rejection_counts"),
        "candidate_results": validation_details.get("candidate_results"),
        "outputs": {
            "source_plausible_mask": str((output_dir / "source_plausible_mask.png").resolve()),
            "source_plausible_overlay": str((output_dir / "source_plausible_overlay.png").resolve()),
            "aligned_plausible_mask": str((output_dir / "aligned_plausible_mask.png").resolve()),
            "aligned_plausible_overlay": str((output_dir / "aligned_plausible_overlay.png").resolve()),
            "raw_plausible_overlay": str((output_dir / "raw_plausible_overlay.png").resolve()),
        },
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
    parser.add_argument("--min-branch-length", type=int, default=1, help="Minimum traced branch length required to keep a bifurcation.")
    parser.add_argument("--border-margin", type=int, default=10, help="Minimum distance in pixels from the ROI boundary.")
    parser.add_argument("--suppression-radius", type=int, default=8, help="Radius used to suppress nearby lower-confidence candidates.")
    parser.add_argument("--dilation-radius", type=int, default=0, help="Radius used to expand accepted candidates for the overlay mask.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample_dir = args.sample_dir.resolve()
    if args.output_dir is None:
        output_dir = Path("tmp") / "crossing_number_plausible_regions_dedup" / sample_dir.name
    else:
        output_dir = args.output_dir.resolve()

    report = _run(
        sample_dir,
        output_dir,
        args.polarity,
        int(args.min_branch_length),
        int(args.border_margin),
        int(args.suppression_radius),
        int(args.dilation_radius),
    )
    print(
        json.dumps(
            {
                "report": report["outputs"],
                "summary": {
                    k: report[k]
                    for k in (
                        "mode",
                        "polarity",
                        "threshold",
                        "skeleton_pixels",
                        "candidate_pixels",
                        "plausible_pixels_source",
                        "plausible_pixels_aligned",
                        "validated_candidate_count",
                        "accepted_candidate_count",
                        "suppressed_candidate_count",
                    )
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

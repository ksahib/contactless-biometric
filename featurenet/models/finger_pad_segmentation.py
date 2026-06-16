from __future__ import annotations

import importlib.util
import json
import sys
import sysconfig
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

from dataclasses import dataclass

import cv2
import numpy as np


class FingerPadSegmentationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        diagnostics: dict[str, Any] | None = None,
        full_mask: np.ndarray | None = None,
        distal_mask: np.ndarray | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.diagnostics = diagnostics or {}
        self.full_mask = full_mask
        self.distal_mask = distal_mask


@dataclass(slots=True)
class FingerPadSegmentation:
    full_mask: np.ndarray
    distal_mask: np.ndarray
    diagnostics: dict[str, Any]
    candidate_full_mask: np.ndarray | None = None
    repair_attempted: bool = False


def _as_u8_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise FingerPadSegmentationError(f"expected 2D mask, got {mask.shape}")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _mask_stats(mask: np.ndarray) -> dict[str, Any]:
    mask_u8 = _as_u8_mask(mask)
    ys, xs = np.where(mask_u8 > 0)
    area = int(xs.size)
    if area == 0:
        return {
            "area": 0,
            "area_ratio": 0.0,
            "bbox_xyxy": None,
            "bbox_aspect_long_over_short": 0.0,
            "touches_border_count": 0,
            "border_touch_flags": {
                "left": False,
                "top": False,
                "right": False,
                "bottom": False,
            },
            "bbox_extent": 0.0,
        }
    h, w = mask_u8.shape
    x0, y0, x1, y1 = _bbox(mask_u8) or [0, 0, 0, 0]
    bbox_w = max(1, int(x1 - x0 + 1))
    bbox_h = max(1, int(y1 - y0 + 1))
    border_margin = max(2, int(round(0.01 * max(h, w))))
    touch_flags = {
        "left": bool(x0 <= border_margin),
        "top": bool(y0 <= border_margin),
        "right": bool(x1 >= w - 1 - border_margin),
        "bottom": bool(y1 >= h - 1 - border_margin),
    }
    bbox_area = max(1, bbox_w * bbox_h)
    return {
        "area": area,
        "area_ratio": float(area / max(1, mask_u8.size)),
        "bbox_xyxy": [x0, y0, x1, y1],
        "bbox_aspect_long_over_short": float(max(bbox_w, bbox_h) / max(1, min(bbox_w, bbox_h))),
        "touches_border_count": int(sum(touch_flags.values())),
        "border_touch_flags": touch_flags,
        "bbox_extent": float(area / bbox_area),
    }


def _largest_component(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(_as_u8_mask(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise FingerPadSegmentationError("no finger-like component was found")
    largest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(out, [largest], -1, 255, thickness=cv2.FILLED)
    return out


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = _as_u8_mask(mask)
    h, w = mask_u8.shape
    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask_u8, holes)


def _postprocess_color_mask(mask: np.ndarray) -> np.ndarray:
    mask = _as_u8_mask(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), dtype=np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), dtype=np.uint8), iterations=1)
    appendage_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, appendage_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), dtype=np.uint8), iterations=1)
    mask = _fill_holes(mask)
    return _largest_component(mask)


def _broad_finger_color_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    h, s, v = cv2.split(hsv)
    y, cr, cb = cv2.split(ycrcb)

    skin_ycrcb = (
        (y > 80)
        & (cr > 134)
        & (cr < 188)
        & (cb > 96)
        & (cb < 132)
    )
    skin_hsv = (
        (v > 45)
        & (s > 24)
        & (((h < 18) | (h > 152)))
        & (cr > 132)
    )

    mask = (skin_ycrcb | skin_hsv).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((19, 19), dtype=np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), dtype=np.uint8), iterations=1)
    mask = _fill_holes(mask)
    return _largest_component(mask)


def _strict_finger_color_masks(bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    _h, _s, v = cv2.split(hsv)
    y, cr, cb = cv2.split(ycrcb)
    rules = [
        ("strict_skin_balanced", (y > 100) & (cr > 136) & (cb < 124) & (v > 105)),
        ("strict_skin_bright", (y > 120) & (cr > 136) & (cb < 124) & (v > 120)),
        ("strict_skin_red", (y > 90) & (cr > 138) & (cb < 124) & (v > 95)),
    ]
    candidates: list[tuple[str, np.ndarray]] = []
    for name, raw in rules:
        try:
            candidates.append((name, _postprocess_color_mask(raw.astype(np.uint8) * 255)))
        except FingerPadSegmentationError:
            continue
    return candidates


def _finger_color_mask_candidates(bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    candidates: list[tuple[str, np.ndarray]] = [("broad_skin", _broad_finger_color_mask(bgr))]
    seen: set[tuple[int, tuple[int, ...] | None]] = set()
    unique: list[tuple[str, np.ndarray]] = []
    for name, mask in candidates + _strict_finger_color_masks(bgr):
        stats = _mask_stats(mask)
        key = (int(stats["area"]), tuple(stats["bbox_xyxy"]) if stats["bbox_xyxy"] is not None else None)
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, mask))
    return unique


def _finger_color_mask(bgr: np.ndarray) -> np.ndarray:
    return _broad_finger_color_mask(bgr)


def _contour_for_mask(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(_as_u8_mask(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise FingerPadSegmentationError("mask has no contour")
    return max(contours, key=cv2.contourArea)


def _pca_axes(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    if xs.size < 32:
        raise FingerPadSegmentationError("finger mask is too small for axis estimation")
    points = np.column_stack([xs, ys]).astype(np.float32)
    center = points.mean(axis=0)
    centered = points - center
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis_u = eigenvectors[:, int(np.argmax(eigenvalues))].astype(np.float32)
    axis_u /= max(float(np.linalg.norm(axis_u)), 1e-6)
    projections = centered @ axis_u
    return points, center.astype(np.float32), axis_u, projections


def _choose_tip_direction(points: np.ndarray, axis_u: np.ndarray, projections: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, float]:
    h, w = shape
    p_min = float(np.min(projections))
    p_max = float(np.max(projections))
    length = max(1.0, p_max - p_min)

    def end_score(is_min: bool) -> tuple[float, np.ndarray]:
        end_p = p_min if is_min else p_max
        band = points[np.abs(projections - end_p) <= max(8.0, 0.04 * length)]
        if band.size == 0:
            band = points[np.argmin(projections) if is_min else np.argmax(projections)].reshape(1, 2)
        endpoint = band.mean(axis=0).astype(np.float32)
        x, y = float(endpoint[0]), float(endpoint[1])
        border_distance = min(x, y, float(w - 1) - x, float(h - 1) - y) / max(1.0, min(h, w))
        return border_distance, endpoint

    min_score, min_endpoint = end_score(True)
    max_score, max_endpoint = end_score(False)
    if max_score > min_score:
        tip_center = max_endpoint
        tip_to_base = -axis_u
    else:
        tip_center = min_endpoint
        tip_to_base = axis_u
    tip_to_base = tip_to_base.astype(np.float32)
    tip_to_base /= max(float(np.linalg.norm(tip_to_base)), 1e-6)
    return tip_center, tip_to_base, float(length)


def _distal_mask_from_full_mask(full_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    points, _center, axis_u, projections = _pca_axes(full_mask)
    tip_center, tip_to_base, length = _choose_tip_direction(points, axis_u, projections, full_mask.shape)

    ys, xs = np.where(full_mask > 0)
    coords = np.column_stack([xs, ys]).astype(np.float32)
    rel = coords - tip_center
    long_coord = rel @ tip_to_base

    axis_v = np.array([-tip_to_base[1], tip_to_base[0]], dtype=np.float32)
    cross_coord = rel @ axis_v
    valid_long = long_coord[(long_coord >= 0.0) & (long_coord <= length)]
    if valid_long.size < 32:
        raise FingerPadSegmentationError("finger axis did not intersect enough foreground pixels")

    bins = np.linspace(0.0, length, 28, dtype=np.float32)
    widths: list[float] = []
    for left, right in zip(bins[:-1], bins[1:]):
        in_bin = (long_coord >= left) & (long_coord < right)
        if np.count_nonzero(in_bin) < 8:
            continue
        widths.append(float(np.percentile(cross_coord[in_bin], 95) - np.percentile(cross_coord[in_bin], 5)))
    if not widths:
        raise FingerPadSegmentationError("could not estimate finger width profile")

    width_values = np.asarray(widths, dtype=np.float32)
    stable_width = float(np.percentile(width_values, 70))
    long_min = -0.035 * length
    long_max = min(0.58 * length, max(1.65 * stable_width, 0.32 * length))
    if long_max <= 0.12 * length:
        raise FingerPadSegmentationError("computed distal pad band is implausibly short")

    keep = (long_coord >= long_min) & (long_coord <= long_max)
    distal = np.zeros_like(full_mask, dtype=np.uint8)
    distal[ys[keep], xs[keep]] = 255
    distal = cv2.morphologyEx(distal, cv2.MORPH_CLOSE, np.ones((11, 11), dtype=np.uint8), iterations=1)
    distal = cv2.morphologyEx(distal, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8), iterations=1)
    distal = cv2.bitwise_and(_largest_component(distal), full_mask)

    diagnostics = {
        "tip_center_xy": [float(tip_center[0]), float(tip_center[1])],
        "tip_to_base": [float(tip_to_base[0]), float(tip_to_base[1])],
        "finger_axis_length": float(length),
        "stable_width": stable_width,
        "long_min": float(long_min),
        "long_max": float(long_max),
    }
    return distal, diagnostics


def _ridge_energy(bgr: np.ndarray, mask: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    values = mag[mask > 0]
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, 70))


def _mask_shape_stats(mask: np.ndarray, *, name: str, min_aspect: float = 1.18) -> dict[str, Any]:
    stats = _mask_stats(mask)
    if stats["area"] <= 0:
        raise FingerPadSegmentationError(f"{name} mask is empty", stage=name, diagnostics=stats)
    area_ratio = float(stats["area_ratio"])
    aspect = float(stats["bbox_aspect_long_over_short"])
    if area_ratio < 0.015 or area_ratio > 0.62:
        raise FingerPadSegmentationError(
            f"{name} mask area ratio is implausible: {area_ratio:.3f}",
            stage=name,
            diagnostics=stats,
            full_mask=_as_u8_mask(mask),
        )
    if aspect < float(min_aspect):
        raise FingerPadSegmentationError(
            f"{name} mask is not finger-like enough: aspect={aspect:.3f}",
            stage=name,
            diagnostics=stats,
            full_mask=_as_u8_mask(mask),
        )

    contour = _contour_for_mask(mask)
    hull = cv2.convexHull(contour)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    contour_area = float(cv2.contourArea(contour))
    solidity = contour_area / hull_area
    stats["solidity"] = float(solidity)
    if solidity < 0.55:
        raise FingerPadSegmentationError(
            f"{name} mask is too fragmented/concave: solidity={solidity:.3f}",
            stage=name,
            diagnostics=stats,
            full_mask=_as_u8_mask(mask),
        )
    return stats


def _is_rectangular_background_like(stats: dict[str, Any]) -> bool:
    touches = int(stats["touches_border_count"])
    extent = float(stats["bbox_extent"])
    area_ratio = float(stats["area_ratio"])
    aspect = float(stats["bbox_aspect_long_over_short"])
    return touches >= 3 and extent >= 0.92 and (area_ratio >= 0.45 or aspect < 1.35)


def validate_finger_mask(mask: np.ndarray, *, image_shape: tuple[int, int], name: str) -> dict[str, Any]:
    stats = _mask_shape_stats(mask, name=name)
    touches = int(stats["touches_border_count"])
    if touches >= 3:
        raise FingerPadSegmentationError(
            f"{name} mask touches too many borders: {touches}",
            stage=name,
            diagnostics=stats,
            full_mask=_as_u8_mask(mask),
        )
    if _is_rectangular_background_like(stats):
        raise FingerPadSegmentationError(
            f"{name} mask is too rectangular/background-like: extent={float(stats['bbox_extent']):.3f}",
            stage=name,
            diagnostics=stats,
            full_mask=_as_u8_mask(mask),
        )
    return stats


def validate_full_finger_mask(mask: np.ndarray, *, image_shape: tuple[int, int]) -> dict[str, Any]:
    stats = _mask_shape_stats(mask, name="full finger")
    touches = int(stats["touches_border_count"])
    if touches >= 4:
        raise FingerPadSegmentationError(
            f"full finger mask touches all image borders: {touches}",
            stage="full finger",
            diagnostics=stats,
            full_mask=_as_u8_mask(mask),
        )
    if _is_rectangular_background_like(stats):
        raise FingerPadSegmentationError(
            f"full finger mask is too rectangular/background-like: extent={float(stats['bbox_extent']):.3f}",
            stage="full finger",
            diagnostics=stats,
            full_mask=_as_u8_mask(mask),
        )
    if touches == 3:
        stats["warning"] = "full finger mask touches three borders; accepted pending distal-pad validation"
    return stats


def validate_distal_mask(bgr: np.ndarray, full_mask: np.ndarray, distal_mask: np.ndarray) -> dict[str, Any]:
    try:
        stats = _mask_shape_stats(distal_mask, name="distal", min_aspect=1.0)
    except FingerPadSegmentationError as exc:
        exc.full_mask = _as_u8_mask(full_mask)
        exc.distal_mask = _as_u8_mask(distal_mask)
        raise
    touches = int(stats["touches_border_count"])
    if touches >= 2:
        raise FingerPadSegmentationError(
            f"distal mask touches too many borders: {touches}",
            stage="distal",
            diagnostics=stats,
            full_mask=_as_u8_mask(full_mask),
            distal_mask=_as_u8_mask(distal_mask),
        )
    if float(stats["bbox_extent"]) < 0.45:
        raise FingerPadSegmentationError(
            f"distal mask is too sparse/background-like: extent={float(stats['bbox_extent']):.3f}",
            stage="distal",
            diagnostics=stats,
            full_mask=_as_u8_mask(full_mask),
            distal_mask=_as_u8_mask(distal_mask),
        )
    full_area = max(int(np.count_nonzero(full_mask)), 1)
    distal_area = int(np.count_nonzero(distal_mask))
    ratio = float(distal_area / full_area)
    stats["distal_to_full_ratio"] = ratio
    if ratio < 0.16 or ratio > 0.72:
        raise FingerPadSegmentationError(
            f"distal mask covers an implausible amount of the finger: {ratio:.3f}",
            stage="distal",
            diagnostics=stats,
            full_mask=_as_u8_mask(full_mask),
            distal_mask=_as_u8_mask(distal_mask),
        )
    outside = int(np.count_nonzero((distal_mask > 0) & (full_mask <= 0)))
    if outside > max(32, int(0.01 * distal_area)):
        raise FingerPadSegmentationError(
            "distal mask leaks outside the full finger mask",
            stage="distal",
            diagnostics=stats,
            full_mask=_as_u8_mask(full_mask),
            distal_mask=_as_u8_mask(distal_mask),
        )
    energy = _ridge_energy(bgr, distal_mask)
    stats["ridge_energy_p70"] = energy
    if energy < 5.0:
        raise FingerPadSegmentationError(
            f"distal mask has too little ridge/edge energy: {energy:.3f}",
            stage="distal",
            diagnostics=stats,
            full_mask=_as_u8_mask(full_mask),
            distal_mask=_as_u8_mask(distal_mask),
        )
    return stats


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


def _overlay(bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = bgr.copy()
    mask_u8 = _as_u8_mask(mask)
    color_img = np.zeros_like(out)
    color_img[:, :] = np.asarray(color, dtype=np.uint8)
    out = np.where(mask_u8[:, :, None] > 0, cv2.addWeighted(out, 0.62, color_img, 0.38, 0), out)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness=3)
    return out


def save_finger_pad_debug(
    *,
    output_dir: Path,
    bgr: np.ndarray,
    segmentation: FingerPadSegmentation,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_mask_path = output_dir / "full_finger_mask.png"
    distal_mask_path = output_dir / "distal_pad_mask.png"
    full_overlay_path = output_dir / "full_finger_overlay.png"
    distal_overlay_path = output_dir / "distal_pad_overlay.png"
    masked_distal_path = output_dir / "distal_pad_black_gray.png"
    diagnostics_path = output_dir / "diagnostics.json"

    _write_image(full_mask_path, segmentation.full_mask)
    _write_image(distal_mask_path, segmentation.distal_mask)
    _write_image(full_overlay_path, _overlay(bgr, segmentation.full_mask, (0, 255, 0)))
    _write_image(distal_overlay_path, _overlay(bgr, segmentation.distal_mask, (0, 0, 255)))
    extra_paths: dict[str, str] = {}
    if segmentation.repair_attempted and segmentation.candidate_full_mask is not None:
        candidate_mask_path = output_dir / "candidate_full_mask.png"
        candidate_overlay_path = output_dir / "candidate_full_overlay.png"
        repaired_full_path = output_dir / "repaired_full_mask.png"
        repaired_distal_path = output_dir / "repaired_distal_mask.png"
        _write_image(candidate_mask_path, segmentation.candidate_full_mask)
        _write_image(candidate_overlay_path, _overlay(bgr, segmentation.candidate_full_mask, (0, 255, 255)))
        _write_image(repaired_full_path, segmentation.full_mask)
        _write_image(repaired_distal_path, segmentation.distal_mask)
        extra_paths.update(
            {
                "candidate_full_mask_png": str(candidate_mask_path),
                "candidate_full_overlay_png": str(candidate_overlay_path),
                "repaired_full_mask_png": str(repaired_full_path),
                "repaired_distal_mask_png": str(repaired_distal_path),
            }
        )
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    masked = gray.copy()
    masked[segmentation.distal_mask <= 0] = 0
    _write_image(masked_distal_path, cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR))
    diagnostics_path.write_text(json.dumps(segmentation.diagnostics, indent=2), encoding="utf-8")
    return {
        "full_mask_png": str(full_mask_path),
        "distal_mask_png": str(distal_mask_path),
        "full_overlay_png": str(full_overlay_path),
        "distal_overlay_png": str(distal_overlay_path),
        "distal_black_gray_png": str(masked_distal_path),
        "diagnostics_json": str(diagnostics_path),
        **extra_paths,
    }


def save_finger_pad_rejection(
    *,
    output_dir: Path,
    bgr: np.ndarray,
    reason: str,
    image_path: str | Path | None = None,
    error: FingerPadSegmentationError | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rejection_path = output_dir / "rejection.json"
    payload = {
        "mask_source": "finger_pad_auto",
        "accepted": False,
        "reason": reason,
        "image_shape_hwc": [int(v) for v in bgr.shape],
    }
    if image_path is not None:
        payload["image_path"] = str(image_path)
    if error is not None:
        payload["stage"] = error.stage
        payload["diagnostics"] = error.diagnostics
    paths = {"rejection_json": str(rejection_path)}
    if error is not None and error.full_mask is not None:
        full_mask_path = output_dir / "rejected_full_finger_mask.png"
        full_overlay_path = output_dir / "rejected_full_finger_overlay.png"
        _write_image(full_mask_path, error.full_mask)
        _write_image(full_overlay_path, _overlay(bgr, error.full_mask, (0, 255, 255)))
        paths["rejected_full_mask_png"] = str(full_mask_path)
        paths["rejected_full_overlay_png"] = str(full_overlay_path)
    if error is not None and error.distal_mask is not None:
        distal_mask_path = output_dir / "rejected_distal_pad_mask.png"
        distal_overlay_path = output_dir / "rejected_distal_pad_overlay.png"
        _write_image(distal_mask_path, error.distal_mask)
        _write_image(distal_overlay_path, _overlay(bgr, error.distal_mask, (0, 0, 255)))
        paths["rejected_distal_mask_png"] = str(distal_mask_path)
        paths["rejected_distal_overlay_png"] = str(distal_overlay_path)
    rejection_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return paths


def segment_finger_pad(bgr: np.ndarray) -> FingerPadSegmentation:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise FingerPadSegmentationError(f"expected BGR image with shape [H,W,3], got {bgr.shape}")
    candidates = _finger_color_mask_candidates(bgr)
    first_candidate_mask = candidates[0][1] if candidates else None
    failures: list[dict[str, Any]] = []
    last_error: FingerPadSegmentationError | None = None
    for index, (candidate_name, full_mask) in enumerate(candidates):
        try:
            full_stats = validate_full_finger_mask(full_mask, image_shape=full_mask.shape)
            distal_mask, geometry = _distal_mask_from_full_mask(full_mask)
            distal_stats = validate_distal_mask(bgr, full_mask, distal_mask)
        except FingerPadSegmentationError as exc:
            exc.full_mask = _as_u8_mask(full_mask)
            failures.append(
                {
                    "candidate": candidate_name,
                    "stage": exc.stage,
                    "reason": str(exc),
                    "diagnostics": exc.diagnostics,
                }
            )
            last_error = exc
            continue

        repair_attempted = index > 0
        diagnostics = {
            "mask_source": "finger_pad_auto",
            "candidate": candidate_name,
            "repair_attempted": repair_attempted,
            "repair_reason": failures[0]["reason"] if repair_attempted and failures else None,
            "failed_candidates": failures,
            "full_mask": full_stats,
            "distal_mask": distal_stats,
            "geometry": geometry,
        }
        return FingerPadSegmentation(
            full_mask=_as_u8_mask(full_mask),
            distal_mask=_as_u8_mask(distal_mask),
            diagnostics=diagnostics,
            candidate_full_mask=_as_u8_mask(first_candidate_mask) if repair_attempted and first_candidate_mask is not None else None,
            repair_attempted=repair_attempted,
        )

    if last_error is not None:
        last_error.diagnostics = {
            **last_error.diagnostics,
            "candidate_failures": failures,
        }
        raise last_error
    raise FingerPadSegmentationError("no finger-like color candidates were found", stage="candidate_generation")

from __future__ import annotations

import importlib.util
import json
import math
import sys
import sysconfig
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def ensure_stdlib_copy_module() -> None:
    stdlib_copy = Path(sysconfig.get_paths()["stdlib"]) / "copy.py"
    spec = importlib.util.spec_from_file_location("copy", stdlib_copy)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not resolve stdlib copy module from {stdlib_copy}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = module
    spec.loader.exec_module(module)


ensure_stdlib_copy_module()

import cv2  # noqa: E402


ROLE_BY_RAW_VIEW_INDEX = {0: "front", 1: "left", 2: "right"}


def role_for_raw_view_index(raw_view_index: int) -> str | None:
    return ROLE_BY_RAW_VIEW_INDEX.get(int(raw_view_index))


def label_frame_for_role(reconstruction_dir: Path, role: str) -> dict[str, Any]:
    reconstruction_dir = Path(reconstruction_dir)
    if role == "front":
        return {
            "role": role,
            "label_frame": "center_unwrapped",
            "image_path": reconstruction_dir / "center_unwarped.png",
            "mask_path": reconstruction_dir / "center_unwarped_mask.png",
            "maps_path": reconstruction_dir / "center_unwarp_maps.npz",
        }
    side_dir = reconstruction_dir / "side_depth_unwrap_v4" / role
    return {
        "role": role,
        "label_frame": "algorithm1_v4_side_unwrapped",
        "image_path": side_dir / f"{role}_depth_unwrapped.png",
        "mask_path": side_dir / f"{role}_depth_unwrapped_mask.png",
        "maps_path": side_dir / f"{role}_depth_unwarp_maps.npz",
    }


def load_gray_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"unable to load grayscale image: {path}")
    return image


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
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


def _zhang_suen_thinning(binary_mask: np.ndarray) -> np.ndarray:
    if binary_mask.ndim != 2:
        raise ValueError(f"expected 2D binary mask, got shape {binary_mask.shape}")

    image = (binary_mask > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            padded = np.pad(image, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]

            neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
            neighbor_sum = sum(neighbors)
            transitions = np.zeros_like(image, dtype=np.uint8)
            for idx in range(8):
                transitions += ((neighbors[idx] == 0) & (neighbors[(idx + 1) % 8] == 1)).astype(np.uint8)

            if step == 0:
                c1 = (p2 * p4 * p6) == 0
                c2 = (p4 * p6 * p8) == 0
            else:
                c1 = (p2 * p4 * p8) == 0
                c2 = (p2 * p6 * p8) == 0

            removable = (
                (image == 1)
                & (neighbor_sum >= 2)
                & (neighbor_sum <= 6)
                & (transitions == 1)
                & c1
                & c2
            )
            if np.any(removable):
                image[removable] = 0
                changed = True

    return (image * 255).astype(np.uint8)


def _skeletonize_binary_mask(binary_mask: np.ndarray) -> np.ndarray:
    ximgproc = getattr(cv2, "ximgproc", None)
    thinning = getattr(ximgproc, "thinning", None) if ximgproc is not None else None
    if callable(thinning):
        try:
            return thinning(binary_mask.astype(np.uint8), thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except Exception:
            pass
    return _zhang_suen_thinning(binary_mask)


def _crossing_number_map(skeleton_mask: np.ndarray) -> np.ndarray:
    if skeleton_mask.ndim != 2:
        raise ValueError(f"expected 2D skeleton mask, got shape {skeleton_mask.shape}")

    skeleton = (skeleton_mask > 0).astype(np.uint8)
    padded = np.pad(skeleton, 1, mode="constant")
    p2 = padded[:-2, 1:-1]
    p3 = padded[:-2, 2:]
    p4 = padded[1:-1, 2:]
    p5 = padded[2:, 2:]
    p6 = padded[2:, 1:-1]
    p7 = padded[2:, :-2]
    p8 = padded[1:-1, :-2]
    p9 = padded[:-2, :-2]
    neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)

    cn = np.zeros_like(skeleton, dtype=np.float32)
    for idx in range(8):
        cn += np.abs(neighbors[idx].astype(np.int16) - neighbors[(idx + 1) % 8].astype(np.int16)).astype(np.float32)
    cn *= 0.5
    cn[skeleton == 0] = 0.0
    return cn


def _select_crossing_number_binarization(
    gray_image: np.ndarray,
    mask: np.ndarray,
    polarity: str = "dark",
    auto_target_ratio: float = 0.35,
) -> tuple[np.ndarray, dict[str, Any]]:
    if gray_image.ndim != 2:
        raise ValueError(f"expected 2D grayscale image, got shape {gray_image.shape}")
    if mask.ndim != 2:
        raise ValueError(f"expected 2D mask, got shape {mask.shape}")

    mask_bool = mask > 0
    if not np.any(mask_bool):
        raise ValueError("cannot estimate crossing-number polarity on an empty mask")

    gray_u8 = gray_image.astype(np.uint8)
    masked_pixels = gray_u8[mask_bool].reshape(-1, 1)
    threshold_value, _ = cv2.threshold(masked_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_value = int(threshold_value)

    candidates: list[tuple[str, np.ndarray, float]] = []
    for mode in ("dark", "light"):
        if mode == "dark":
            binary = np.where(gray_u8 <= threshold_value, 255, 0).astype(np.uint8)
        else:
            binary = np.where(gray_u8 > threshold_value, 255, 0).astype(np.uint8)
        binary[~mask_bool] = 0
        foreground_ratio = float(np.count_nonzero(binary)) / float(np.count_nonzero(mask_bool))
        candidates.append((mode, binary, foreground_ratio))

    normalized = polarity.strip().lower()
    if normalized == "auto":
        chosen_mode, chosen_binary, chosen_ratio = min(
            candidates,
            key=lambda item: (abs(item[2] - auto_target_ratio), 0 if item[0] == "dark" else 1),
        )
    elif normalized in {"dark", "light"}:
        chosen_mode, chosen_binary, chosen_ratio = next(item for item in candidates if item[0] == normalized)
    else:
        raise ValueError("polarity must be one of {'dark', 'light', 'auto'}")

    details = {
        "polarity": chosen_mode,
        "threshold": threshold_value,
        "foreground_ratio": float(chosen_ratio),
        "candidate_foreground_ratio_dark": float(candidates[0][2]),
        "candidate_foreground_ratio_light": float(candidates[1][2]),
    }
    return chosen_binary, details


def _skeleton_degree_map(skeleton: np.ndarray) -> np.ndarray:
    skeleton_bool = skeleton > 0
    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(
        skeleton_bool.astype(np.uint8),
        ddepth=-1,
        kernel=neighbor_kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    degree = neighbor_count - skeleton_bool.astype(np.uint8)
    degree[~skeleton_bool] = 0
    return degree.astype(np.uint8)


def _trace_branch_path(
    skeleton: np.ndarray,
    degree_map: np.ndarray,
    roi_distance: np.ndarray,
    start_y: int,
    start_x: int,
    prev_y: int,
    prev_x: int,
    *,
    border_margin: int,
) -> tuple[bool, dict[str, Any]]:
    height, width = skeleton.shape
    path: list[tuple[int, int]] = [(int(start_y), int(start_x))]
    visited = {(int(prev_y), int(prev_x)), (int(start_y), int(start_x))}
    curr_y, curr_x = int(start_y), int(start_x)
    prev_y, prev_x = int(prev_y), int(prev_x)

    while True:
        if roi_distance[curr_y, curr_x] < float(border_margin):
            return False, {
                "rejection_reason": "branch_hits_roi_boundary",
                "branch_length": int(len(path)),
                "endpoint": (int(curr_y), int(curr_x)),
            }

        degree = int(degree_map[curr_y, curr_x])
        if degree == 1:
            return True, {
                "branch_length": int(len(path)),
                "endpoint": (int(curr_y), int(curr_x)),
                "path": [(int(y), int(x)) for y, x in path],
                "termination_reason": "ridge_end",
            }
        if degree > 2:
            return True, {
                "branch_length": int(len(path)),
                "endpoint": (int(curr_y), int(curr_x)),
                "termination_reason": "branch_rebranches",
            }

        next_pixels: list[tuple[int, int]] = []
        for ny in range(max(0, curr_y - 1), min(height, curr_y + 2)):
            for nx in range(max(0, curr_x - 1), min(width, curr_x + 2)):
                if ny == curr_y and nx == curr_x:
                    continue
                if (ny, nx) == (prev_y, prev_x):
                    continue
                if skeleton[ny, nx] > 0:
                    next_pixels.append((int(ny), int(nx)))

        if len(next_pixels) != 1:
            return True, {
                "branch_length": int(len(path)),
                "endpoint": (int(curr_y), int(curr_x)),
                "next_count": int(len(next_pixels)),
                "termination_reason": "branch_loops_or_is_open",
            }

        next_y, next_x = next_pixels[0]
        if (next_y, next_x) in visited:
            return False, {
                "rejection_reason": "branch_loops_or_is_open",
                "branch_length": int(len(path)),
                "endpoint": (int(curr_y), int(curr_x)),
                "loop_detected": True,
            }

        visited.add((next_y, next_x))
        path.append((next_y, next_x))
        prev_y, prev_x = curr_y, curr_x
        curr_y, curr_x = next_y, next_x


def _validate_ending_candidate(
    skeleton: np.ndarray,
    roi_distance: np.ndarray,
    y: int,
    x: int,
    *,
    border_margin: int,
    min_branch_length: int,
) -> tuple[bool, dict[str, Any]]:
    if roi_distance[y, x] < float(border_margin):
        return False, {
            "x": int(x),
            "y": int(y),
            "candidate_type": "ending",
            "status": "rejected",
            "rejection_reason": "near_roi_boundary",
        }

    degree_map = _skeleton_degree_map(skeleton)
    start_pixels = [
        (int(ny), int(nx))
        for ny in range(max(0, y - 1), min(skeleton.shape[0], y + 2))
        for nx in range(max(0, x - 1), min(skeleton.shape[1], x + 2))
        if not (ny == y and nx == x) and skeleton[ny, nx] > 0
    ]
    if len(start_pixels) != 1:
        return False, {
            "x": int(x),
            "y": int(y),
            "candidate_type": "ending",
            "status": "rejected",
            "rejection_reason": "not_single_branch",
            "branch_label_count": int(len(start_pixels)),
        }

    accepted, branch_detail = _trace_branch_path(
        skeleton,
        degree_map,
        roi_distance,
        start_pixels[0][0],
        start_pixels[0][1],
        y,
        x,
        border_margin=border_margin,
    )
    if not accepted:
        return False, {
            "x": int(x),
            "y": int(y),
            "candidate_type": "ending",
            "status": "rejected",
            "rejection_reason": str(branch_detail.get("rejection_reason", "branch_trace_failed")),
            "branch_length": int(branch_detail.get("branch_length", 0)),
        }

    branch_length = int(branch_detail.get("branch_length", 0))
    if branch_length < int(min_branch_length):
        return False, {
            "x": int(x),
            "y": int(y),
            "candidate_type": "ending",
            "status": "rejected",
            "rejection_reason": "branch_too_short",
            "branch_length": branch_length,
        }

    confidence = float(branch_length)
    return True, {
        "x": int(x),
        "y": int(y),
        "candidate_type": "ending",
        "status": "validated",
        "confidence": confidence,
        "score": confidence,
        "theta": 0.0,
        "branch_lengths": [branch_length],
        "branches": [
            {
                "endpoint": [int(branch_detail["endpoint"][0]), int(branch_detail["endpoint"][1])],
                "branch_length": branch_length,
            }
        ],
    }


def _validate_bifurcation_candidate(
    skeleton: np.ndarray,
    roi_mask: np.ndarray,
    roi_distance: np.ndarray,
    y: int,
    x: int,
    *,
    border_margin: int,
    min_branch_length: int,
) -> tuple[bool, dict[str, Any]]:
    if roi_distance[y, x] < float(border_margin):
        return False, {
            "x": int(x),
            "y": int(y),
            "candidate_type": "bifurcation",
            "status": "rejected",
            "rejection_reason": "near_roi_boundary",
        }

    degree_map = _skeleton_degree_map(skeleton)
    start_pixels = []
    for ny in range(max(0, y - 1), min(skeleton.shape[0], y + 2)):
        for nx in range(max(0, x - 1), min(skeleton.shape[1], x + 2)):
            if ny == y and nx == x:
                continue
            if skeleton[ny, nx] > 0:
                start_pixels.append((int(ny), int(nx)))
    if not start_pixels:
        return False, {
            "x": int(x),
            "y": int(y),
            "candidate_type": "bifurcation",
            "status": "rejected",
            "rejection_reason": "no_branch_components",
        }

    branch_paths: dict[tuple[int, int], dict[str, Any]] = {}
    for start_y, start_x in start_pixels:
        accepted, branch_detail = _trace_branch_path(
            skeleton,
            degree_map,
            roi_distance,
            start_y,
            start_x,
            y,
            x,
            border_margin=border_margin,
        )
        if not accepted:
            return False, {
                "x": int(x),
                "y": int(y),
                "candidate_type": "bifurcation",
                "status": "rejected",
                "rejection_reason": str(branch_detail.get("rejection_reason", "branch_trace_failed")),
                "branch_length": int(branch_detail.get("branch_length", 0)),
            }
        endpoint = tuple(branch_detail["endpoint"])
        if endpoint not in branch_paths or int(branch_detail["branch_length"]) > int(branch_paths[endpoint]["branch_length"]):
            branch_paths[endpoint] = branch_detail

    if len(branch_paths) < 2:
        return False, {
            "x": int(x),
            "y": int(y),
            "candidate_type": "bifurcation",
            "status": "rejected",
            "rejection_reason": "too_few_branches",
            "branch_label_count": int(len(branch_paths)),
        }
    if len(branch_paths) > 3:
        ranked_branches = sorted(branch_paths.items(), key=lambda item: int(item[1]["branch_length"]), reverse=True)[:3]
        branch_paths = {endpoint: detail for endpoint, detail in ranked_branches}

    branch_lengths: list[int] = []
    branch_details: list[dict[str, Any]] = []
    for endpoint, branch_detail in branch_paths.items():
        branch_length = int(branch_detail["branch_length"])
        if branch_length < int(min_branch_length):
            return False, {
                "x": int(x),
                "y": int(y),
                "candidate_type": "bifurcation",
                "status": "rejected",
                "rejection_reason": "branch_too_short",
                "branch_length": branch_length,
            }
        branch_lengths.append(branch_length)
        branch_details.append(
            {
                "endpoint": [int(endpoint[0]), int(endpoint[1])],
                "branch_length": branch_length,
            }
        )

    confidence = float(np.min(branch_lengths) + 0.35 * np.mean(branch_lengths) - np.std(branch_lengths))
    return True, {
        "x": int(x),
        "y": int(y),
        "candidate_type": "bifurcation",
        "status": "validated",
        "confidence": confidence,
        "score": confidence,
        "theta": 0.0,
        "branch_lengths": branch_lengths,
        "branches": branch_details,
    }


def _suppress_candidates_by_confidence(
    candidates: list[dict[str, Any]],
    *,
    suppression_radius: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    radius_sq = float(suppression_radius) * float(suppression_radius)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: (-float(item["confidence"]), int(item["y"]), int(item["x"]))):
        if any((float(candidate["x"]) - float(kept_item["x"])) ** 2 + (float(candidate["y"]) - float(kept_item["y"])) ** 2 <= radius_sq for kept_item in kept):
            suppressed = dict(candidate)
            suppressed["status"] = "rejected"
            suppressed["rejection_reason"] = "suppressed_by_higher_confidence"
            rejected.append(suppressed)
            continue
        kept.append(dict(candidate))
    return kept, rejected


def extract_plausible_minutiae(
    gray_image: np.ndarray,
    roi_mask: np.ndarray,
    *,
    polarity: str = "dark",
    border_margin: int = 10,
    min_branch_length: int = 1,
    suppression_radius: int = 8,
    include_endings: bool = True,
    include_bifurcations: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    binary, details = _select_crossing_number_binarization(gray_image, roi_mask, polarity=polarity)
    skeleton = _skeletonize_binary_mask(binary)
    skeleton = np.where(roi_mask > 0, skeleton, 0).astype(np.uint8)
    cn = _crossing_number_map(skeleton)

    candidate_mask = np.zeros_like(skeleton, dtype=bool)
    if include_endings:
        candidate_mask |= (cn == 1.0)
    if include_bifurcations:
        candidate_mask |= (cn == 3.0)
    candidate_mask &= (skeleton > 0) & (roi_mask > 0)

    candidate_mask_u8 = np.where(candidate_mask, 255, 0).astype(np.uint8)
    labels_count, labels = cv2.connectedComponents(candidate_mask_u8, connectivity=8)
    roi_distance = cv2.distanceTransform((roi_mask > 0).astype(np.uint8), cv2.DIST_L2, 3)

    candidate_results: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    component_winners: list[dict[str, Any]] = []

    for label in range(1, labels_count):
        component_coords = np.argwhere(labels == label)
        if component_coords.size == 0:
            continue

        component_valid: list[dict[str, Any]] = []
        for y, x in component_coords.tolist():
            candidate_type = "bifurcation" if cn[y, x] == 3.0 else "ending"
            if candidate_type == "ending":
                accepted, result = _validate_ending_candidate(
                    skeleton,
                    roi_distance,
                    int(y),
                    int(x),
                    border_margin=border_margin,
                    min_branch_length=min_branch_length,
                )
            else:
                accepted, result = _validate_bifurcation_candidate(
                    skeleton,
                    roi_mask,
                    roi_distance,
                    int(y),
                    int(x),
                    border_margin=border_margin,
                    min_branch_length=min_branch_length,
                )
            result["cn_value"] = float(cn[y, x])
            result["component_label"] = int(label)
            candidate_results.append(result)
            if accepted:
                component_valid.append(result)
            else:
                reason = str(result.get("rejection_reason", "unknown_rejection"))
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        if not component_valid:
            continue

        winner = max(component_valid, key=lambda item: float(item["confidence"]))
        component_winners.append(winner)
        for candidate in component_valid:
            if candidate is winner:
                candidate["status"] = "accepted"
                continue
            candidate["status"] = "rejected"
            candidate["rejection_reason"] = "suppressed_by_higher_confidence"
            rejection_counts["suppressed_by_higher_confidence"] = rejection_counts.get("suppressed_by_higher_confidence", 0) + 1

    accepted_candidates, suppressed_candidates = _suppress_candidates_by_confidence(
        component_winners,
        suppression_radius=suppression_radius,
    )
    accepted_keys = {(int(candidate["y"]), int(candidate["x"])) for candidate in accepted_candidates}
    for candidate in component_winners:
        key = (int(candidate["y"]), int(candidate["x"]))
        if key in accepted_keys:
            candidate["status"] = "accepted"
            continue
        candidate["status"] = "rejected"
        candidate["rejection_reason"] = "suppressed_by_higher_confidence"
        rejection_counts["suppressed_by_higher_confidence"] = rejection_counts.get("suppressed_by_higher_confidence", 0) + 1
    for candidate in suppressed_candidates:
        candidate["status"] = "rejected"
        rejection_counts[candidate["rejection_reason"]] = rejection_counts.get(candidate["rejection_reason"], 0) + 1

    details.update(
        {
            "mode": "direct_aligned_plausible_validation",
            "skeleton_pixels": int(np.count_nonzero(skeleton)),
            "cn_endpoint_pixels": int(np.count_nonzero(cn == 1.0)),
            "cn_bifurcation_pixels": int(np.count_nonzero(cn == 3.0)),
            "candidate_pixels_raw": int(np.count_nonzero(candidate_mask)),
            "candidate_components": int(max(0, labels_count - 1)),
            "candidate_pixels": int(np.count_nonzero(candidate_mask)),
            "validated_candidate_count": int(sum(1 for candidate in candidate_results if candidate.get("status") in {"accepted", "rejected"} and "confidence" in candidate)),
            "accepted_candidate_count": int(len(accepted_candidates)),
            "rejected_candidate_count": int(len(candidate_results) - len(accepted_candidates)),
            "suppressed_candidate_count": int(len(suppressed_candidates)),
            "border_margin": int(border_margin),
            "min_branch_length": int(min_branch_length),
            "suppression_radius": int(suppression_radius),
            "rejection_counts": rejection_counts,
            "candidate_results": candidate_results,
        }
    )
    return accepted_candidates, details


def build_plausible_minutia_mask(
    gray_image: np.ndarray,
    roi_mask: np.ndarray,
    *,
    polarity: str = "dark",
    border_margin: int = 10,
    min_branch_length: int = 1,
    suppression_radius: int = 8,
    dilation_radius: int = 0,
    include_endings: bool = True,
    include_bifurcations: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    accepted_candidates, details = extract_plausible_minutiae(
        gray_image,
        roi_mask,
        polarity=polarity,
        border_margin=border_margin,
        min_branch_length=min_branch_length,
        suppression_radius=suppression_radius,
        include_endings=include_endings,
        include_bifurcations=include_bifurcations,
    )

    plausible_mask = np.zeros_like(roi_mask, dtype=np.uint8)
    for candidate in accepted_candidates:
        plausible_mask[int(candidate["y"]), int(candidate["x"])] = 255
    if int(dilation_radius) > 0 and np.count_nonzero(plausible_mask) > 0:
        kernel_size = int(dilation_radius) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        plausible_mask = cv2.dilate(plausible_mask, kernel, iterations=1)
    plausible_mask = np.where(roi_mask > 0, plausible_mask, 0).astype(np.uint8)

    details.update(
        {
            "plausible_pixels": int(np.count_nonzero(plausible_mask)),
            "dilation_radius": int(dilation_radius),
        }
    )
    return plausible_mask, details


def encode_binary_mask_rle(mask: np.ndarray) -> list[list[int]]:
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"expected 2D mask or [1,H,W] mask, got shape {array.shape}")
    flat = (array > 0).astype(np.uint8).reshape(-1)
    rle: list[list[int]] = []
    start = None
    length = 0
    for index, value in enumerate(flat.tolist()):
        if value:
            if start is None:
                start = index
                length = 1
            else:
                length += 1
        elif start is not None:
            rle.append([int(start), int(length)])
            start = None
            length = 0
    if start is not None:
        rle.append([int(start), int(length)])
    return rle


def decode_binary_mask_rle(mask_rle: Any, output_shape: tuple[int, int] | list[int]) -> np.ndarray:
    if not isinstance(output_shape, (list, tuple)) or len(output_shape) < 2:
        raise ValueError(f"expected output_shape [H, W], got {output_shape!r}")
    height = int(output_shape[0])
    width = int(output_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid output_shape: {output_shape!r}")

    mask = np.zeros((height * width,), dtype=np.float32)
    if mask_rle is None:
        return mask.reshape(1, height, width)
    for run in mask_rle:
        if not isinstance(run, (list, tuple)) or len(run) < 2:
            continue
        try:
            start = int(run[0])
            length = int(run[1])
        except (TypeError, ValueError):
            continue
        if start < 0 or length <= 0:
            continue
        end = min(start + length, mask.size)
        if start >= mask.size or end <= start:
            continue
        mask[start:end] = 1.0
    return mask.reshape(1, height, width)


def sidecar_row_to_mask(row: Mapping[str, Any], default_shape: tuple[int, int] | None = None) -> np.ndarray | None:
    shape = row.get("output_shape", default_shape)
    if shape is None:
        return None
    try:
        return decode_binary_mask_rle(row.get("mask_rle"), shape)
    except Exception:
        return None

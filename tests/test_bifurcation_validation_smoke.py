from __future__ import annotations

import importlib.util
import sys
import sysconfig
from pathlib import Path

import numpy as np


def _ensure_stdlib_copy_module() -> None:
    stdlib_copy = Path(sysconfig.get_paths()["stdlib"]) / "copy.py"
    spec = importlib.util.spec_from_file_location("copy", stdlib_copy)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not resolve stdlib copy module from {stdlib_copy}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = module
    spec.loader.exec_module(module)


_ensure_stdlib_copy_module()

import cv2


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_crossing_number_plausible_regions_dedup.py"
SPEC = importlib.util.spec_from_file_location("bifurcation_smoke", MODULE_PATH)
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
assert SPEC.loader is not None
SPEC.loader.exec_module(smoke)


def _full_mask(shape: tuple[int, int]) -> np.ndarray:
    return np.full(shape, 255, dtype=np.uint8)


def _draw_y_bifurcation(shape: tuple[int, int], *, short_branch: bool = False, branch_again: bool = False) -> np.ndarray:
    image = np.full(shape, 255, dtype=np.uint8)
    center = (shape[1] // 2, shape[0] // 2)
    cv2.line(image, center, (center[0], max(1, center[1] - 6)), 0, 1)
    left_end = (max(1, center[0] - 5), min(shape[0] - 2, center[1] + 5))
    right_end = (min(shape[1] - 2, center[0] + 5), min(shape[0] - 2, center[1] + 5))
    if short_branch:
        left_end = (center[0] - 2, center[1] + 2)
    cv2.line(image, center, left_end, 0, 1)
    cv2.line(image, center, right_end, 0, 1)
    if branch_again:
        spur_start = (center[0] + 2, center[1] + 2)
        spur_end = (min(shape[1] - 2, spur_start[0] + 3), max(1, spur_start[1] - 2))
        cv2.line(image, spur_start, spur_end, 0, 1)
    return image


def _draw_ending(shape: tuple[int, int], *, short: bool = False) -> np.ndarray:
    image = np.full(shape, 255, dtype=np.uint8)
    center = (shape[1] // 2, shape[0] // 2)
    length = 3 if short else 8
    cv2.line(image, center, (center[0] + length, center[1]), 0, 1)
    return image


def test_clean_ending_is_accepted() -> None:
    gray = _draw_ending((21, 21))
    mask = _full_mask((21, 21))

    plausible_mask, details = smoke._build_bifurcation_plausible_mask(
        gray,
        mask,
        polarity="dark",
        border_margin=2,
        min_branch_length=4,
        suppression_radius=0,
    )

    assert details["accepted_candidate_count"] >= 1
    assert int(np.count_nonzero(plausible_mask)) >= 1


def test_short_ending_is_rejected() -> None:
    gray = _draw_ending((21, 21), short=True)
    mask = _full_mask((21, 21))

    plausible_mask, details = smoke._build_bifurcation_plausible_mask(
        gray,
        mask,
        polarity="dark",
        border_margin=2,
        min_branch_length=6,
        suppression_radius=0,
    )

    assert details["accepted_candidate_count"] == 0
    assert details["rejected_candidate_count"] >= 1
    assert details["rejection_counts"]["branch_too_short"] >= 1
    assert int(np.count_nonzero(plausible_mask)) == 0


def test_clean_bifurcation_is_accepted() -> None:
    gray = _draw_y_bifurcation((21, 21))
    mask = _full_mask((21, 21))

    plausible_mask, details = smoke._build_bifurcation_plausible_mask(
        gray,
        mask,
        polarity="dark",
        border_margin=2,
        min_branch_length=4,
        suppression_radius=0,
    )

    assert details["accepted_candidate_count"] >= 1
    assert int(np.count_nonzero(plausible_mask)) >= 1


def test_short_branch_is_rejected() -> None:
    gray = _draw_y_bifurcation((21, 21), short_branch=True)
    mask = _full_mask((21, 21))

    plausible_mask, details = smoke._build_bifurcation_plausible_mask(
        gray,
        mask,
        polarity="dark",
        border_margin=2,
        min_branch_length=4,
        suppression_radius=0,
    )

    assert details["accepted_candidate_count"] >= 1
    assert details["rejection_counts"]["branch_too_short"] >= 1
    assert int(np.count_nonzero(plausible_mask)) >= 1


def test_branching_again_is_rejected() -> None:
    gray = _draw_y_bifurcation((21, 21), branch_again=True)
    mask = _full_mask((21, 21))

    plausible_mask, details = smoke._build_bifurcation_plausible_mask(
        gray,
        mask,
        polarity="dark",
        border_margin=2,
        min_branch_length=4,
        suppression_radius=0,
    )

    assert details["accepted_candidate_count"] >= 1
    assert details["rejected_candidate_count"] >= 1
    assert details["rejection_counts"].get("branch_too_short", 0) >= 1
    assert int(np.count_nonzero(plausible_mask)) >= 1


def test_near_border_is_rejected() -> None:
    gray = np.full((21, 21), 255, dtype=np.uint8)
    cv2.line(gray, (3, 3), (3, 8), 0, 1)
    cv2.line(gray, (3, 3), (8, 3), 0, 1)
    cv2.line(gray, (3, 3), (8, 8), 0, 1)
    mask = _full_mask((21, 21))

    plausible_mask, details = smoke._build_bifurcation_plausible_mask(
        gray,
        mask,
        polarity="dark",
        border_margin=4,
        min_branch_length=3,
        suppression_radius=0,
    )

    assert details["accepted_candidate_count"] >= 1
    assert any(
        reason in details["rejection_counts"]
        for reason in ("near_roi_boundary", "branch_hits_roi_boundary", "branch_rebranches", "branch_too_short")
    )
    assert int(np.count_nonzero(plausible_mask)) >= 1

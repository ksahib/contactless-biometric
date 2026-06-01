from __future__ import annotations

import math

import numpy as np

import main as mcc
import pandas as pd


def _cylinder(index: int, x: float, y: float, theta: float) -> mcc.MCCCylinder:
    return mcc.MCCCylinder(
        center_index=index,
        center_x=float(x),
        center_y=float(y),
        center_theta=float(theta),
        radius=mcc.MCC_RADIUS,
        angle_height=2.0 * math.pi,
        ns=mcc.MCC_NS,
        nd=mcc.MCC_ND,
        delta_s=(2.0 * mcc.MCC_RADIUS) / mcc.MCC_NS,
        delta_d=(2.0 * math.pi) / mcc.MCC_ND,
        sigma_s=mcc.MCC_SIGMA_S,
        sigma_d=mcc.MCC_SIGMA_D,
        cells=[],
    )


def _apply_transform(points: np.ndarray, scale: float, rotation: float, tx: float, ty: float) -> np.ndarray:
    matrix = np.array(
        [
            [math.cos(rotation), -math.sin(rotation)],
            [math.sin(rotation), math.cos(rotation)],
        ],
        dtype=np.float64,
    )
    return (scale * (points @ matrix.T)) + np.array([tx, ty], dtype=np.float64)


def test_similarity_transform_estimation_recovers_known_transform() -> None:
    query = np.array([[0.0, 0.0], [40.0, 0.0], [15.0, 30.0], [55.0, 35.0]], dtype=np.float64)
    expected_scale = 1.25
    expected_rotation = 0.35
    expected_tx = 18.0
    expected_ty = -9.0
    template = _apply_transform(query, expected_scale, expected_rotation, expected_tx, expected_ty)

    transform = mcc._estimate_similarity_transform_from_points(template, query)

    assert transform is not None
    assert transform["scale"] == pytest_approx(expected_scale)
    assert mcc.wrap_angle(transform["rotation"] - expected_rotation) == pytest_approx(0.0)
    assert transform["translation_x"] == pytest_approx(expected_tx)
    assert transform["translation_y"] == pytest_approx(expected_ty)


def test_ransac_recovers_transform_with_outlier_correspondences() -> None:
    query = np.array(
        [
            [0.0, 0.0],
            [40.0, 0.0],
            [15.0, 30.0],
            [55.0, 35.0],
            [20.0, 70.0],
            [85.0, 50.0],
        ],
        dtype=np.float64,
    )
    scale = 1.1
    rotation = -0.28
    tx = 12.0
    ty = 6.5
    template = _apply_transform(query, scale, rotation, tx, ty)
    descriptors_a = [_cylinder(i, x, y, 0.4 + rotation) for i, (x, y) in enumerate(template)]
    descriptors_b = [_cylinder(i, x, y, 0.4) for i, (x, y) in enumerate(query)]
    candidates = [(i, i, 1.0) for i in range(len(query))]
    candidates.extend([(0, 5, 0.9), (4, 1, 0.8), (2, 5, 0.7)])

    transform, inliers, diagnostics = mcc._estimate_ransac_similarity_transform(
        descriptors_a,
        descriptors_b,
        candidates,
        iterations=120,
        min_inliers=4,
        spatial_threshold=3.0,
        angle_threshold=0.2,
    )

    assert transform is not None
    assert len(inliers) >= 6
    assert diagnostics["inlier_count"] >= 6
    assert transform["scale"] == pytest_approx(scale)
    assert mcc.wrap_angle(transform["rotation"] - rotation) == pytest_approx(0.0)
    assert transform["translation_x"] == pytest_approx(tx)
    assert transform["translation_y"] == pytest_approx(ty)


def test_ransac_dispatch_accepts_methods_and_falls_back_with_too_few_points() -> None:
    frame_a = pd.DataFrame(
        [
            {"x": 0.0, "y": 0.0, "angle": 0.0, "score": 1.0},
            {"x": 20.0, "y": 0.0, "angle": 0.0, "score": 1.0},
            {"x": 0.0, "y": 20.0, "angle": 0.0, "score": 1.0},
        ]
    )
    frame_b = frame_a.copy()

    for method, base_method in (("LSA-RANSAC", "LSA"), ("LSA-R-RANSAC", "LSA-R")):
        score, sim_matrix, details = mcc.match_minutiae_csv_ransac_details(frame_a, frame_b, method=method)
        assert isinstance(score, float)
        assert sim_matrix.ndim == 2
        assert details["method"] == method
        assert details["base_method"] == base_method
        assert details["fallback_to_base"] is True
        assert details["fallback_reason"] == "too_few_minutiae_for_ransac"


def pytest_approx(value: float, abs: float = 1e-6) -> object:
    import pytest

    return pytest.approx(value, abs=abs)

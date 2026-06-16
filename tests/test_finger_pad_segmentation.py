from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from featurenet.models.finger_pad_segmentation import (
    FingerPadSegmentationError,
    save_finger_pad_rejection,
    segment_finger_pad,
    validate_distal_mask,
    validate_finger_mask,
    validate_full_finger_mask,
)
import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename", ["amit_right_ind1.jpg", "amit_right_ind2.jpg"])
def test_finger_pad_auto_segments_amit_regression_images(filename: str) -> None:
    image_path = REPO_ROOT / filename
    if not image_path.exists():
        pytest.skip(f"regression image is not present: {image_path}")

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert bgr is not None

    result = segment_finger_pad(bgr)
    full = result.diagnostics["full_mask"]
    distal = result.diagnostics["distal_mask"]

    assert full["area_ratio"] == pytest.approx(full["area_ratio"], abs=0.0)
    assert 0.20 < full["area_ratio"] < 0.50
    assert 0.12 < distal["area_ratio"] < 0.30
    assert distal["distal_to_full_ratio"] < 0.70
    assert distal["ridge_energy_p70"] > 5.0
    assert distal["touches_border_count"] <= 1


@pytest.mark.parametrize("filename", ["tonmoy1.jpg", "tony.jpg"])
def test_finger_pad_auto_repairs_background_connected_regression_images(filename: str) -> None:
    image_path = REPO_ROOT / filename
    if not image_path.exists():
        pytest.skip(f"regression image is not present: {image_path}")

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert bgr is not None

    result = segment_finger_pad(bgr)
    full = result.diagnostics["full_mask"]
    distal = result.diagnostics["distal_mask"]

    assert result.diagnostics["candidate"].startswith("strict_skin")
    assert result.diagnostics["repair_attempted"] is True
    assert full["touches_border_count"] <= 1
    assert distal["touches_border_count"] <= 1
    assert distal["bbox_extent"] > 0.45
    assert distal["ridge_energy_p70"] > 5.0
    assert 0.10 < distal["area_ratio"] < 0.30


def test_quality_gate_rejects_large_background_rectangle() -> None:
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[:, :240] = 255

    with pytest.raises(FingerPadSegmentationError, match="area ratio|too many borders"):
        validate_finger_mask(mask, image_shape=mask.shape, name="bad")


def test_full_finger_gate_accepts_plausible_three_border_mask_with_valid_distal_pad() -> None:
    full = np.zeros((160, 128), dtype=np.uint8)
    cv2.ellipse(full, (28, 80), (55, 94), 0, 0, 360, 255, thickness=cv2.FILLED)
    distal = np.zeros_like(full)
    yy, xx = np.indices(full.shape)
    distal[(full > 0) & (yy >= 8) & (yy <= 110) & (xx >= 8) & (xx <= 75)] = 255

    x = np.arange(full.shape[1], dtype=np.float32)
    stripe = ((np.sin(x / 3.0) + 1.0) * 45.0 + 85.0).astype(np.uint8)
    gray = np.repeat(stripe[None, :], full.shape[0], axis=0)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    full_stats = validate_full_finger_mask(full, image_shape=full.shape)
    distal_stats = validate_distal_mask(bgr, full, distal)

    assert full_stats["touches_border_count"] == 3
    assert "warning" in full_stats
    assert full_stats["bbox_extent"] < 0.92
    assert distal_stats["ridge_energy_p70"] > 5.0


def test_full_finger_gate_rejects_three_border_rectangular_background() -> None:
    mask = np.zeros((160, 128), dtype=np.uint8)
    mask[:, :75] = 255

    with pytest.raises(FingerPadSegmentationError, match="rectangular|background"):
        validate_full_finger_mask(mask, image_shape=mask.shape)


def test_finger_pad_auto_repairs_synthetic_connected_horizontal_ribbon() -> None:
    bgr = np.zeros((220, 260, 3), dtype=np.uint8)
    bgr[:, :, :] = (70, 80, 75)
    cv2.ellipse(bgr, (130, 86), (45, 72), 0, 0, 360, (175, 185, 225), thickness=cv2.FILLED)
    cv2.rectangle(bgr, (0, 140), (259, 190), (105, 116, 130), thickness=cv2.FILLED)
    for x in range(90, 171, 6):
        cv2.line(bgr, (x, 25), (x + 8, 130), (145, 155, 190), thickness=1)

    result = segment_finger_pad(bgr)
    full = result.diagnostics["full_mask"]
    distal = result.diagnostics["distal_mask"]

    assert result.diagnostics["repair_attempted"] is True
    assert full["bbox_xyxy"][0] > 20
    assert full["bbox_xyxy"][2] < 220
    assert distal["touches_border_count"] == 0
    assert distal["bbox_extent"] > 0.45


def test_quality_gate_rejects_distal_mask_outside_full_finger() -> None:
    bgr = np.zeros((128, 128, 3), dtype=np.uint8)
    bgr[:, :, :] = (120, 120, 120)
    full = np.zeros((128, 128), dtype=np.uint8)
    full[20:120, 45:95] = 255
    distal = np.zeros_like(full)
    distal[10:90, 30:70] = 255

    with pytest.raises(FingerPadSegmentationError, match="leaks outside"):
        validate_distal_mask(bgr, full, distal)


def test_quality_gate_rejects_distal_mask_without_ridge_energy() -> None:
    bgr = np.zeros((128, 128, 3), dtype=np.uint8)
    bgr[:, :, :] = (120, 120, 120)
    full = np.zeros((128, 128), dtype=np.uint8)
    full[15:115, 50:90] = 255
    distal = np.zeros_like(full)
    distal[20:90, 53:87] = 255

    with pytest.raises(FingerPadSegmentationError, match="too little ridge"):
        validate_distal_mask(bgr, full, distal)


def test_rejection_artifact_records_failure_reason(tmp_path: Path) -> None:
    bgr = np.zeros((16, 20, 3), dtype=np.uint8)
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[:, :18] = 255
    error = FingerPadSegmentationError(
        "full finger mask is too rectangular/background-like: extent=1.000",
        stage="full finger",
        diagnostics={"touches_border_count": 3, "bbox_extent": 1.0},
        full_mask=mask,
    )
    paths = save_finger_pad_rejection(
        output_dir=tmp_path,
        bgr=bgr,
        reason=str(error),
        image_path="example.jpg",
        error=error,
    )

    payload = (tmp_path / "rejection.json").read_text(encoding="utf-8")
    assert paths["rejection_json"] == str(tmp_path / "rejection.json")
    assert paths["rejected_full_mask_png"] == str(tmp_path / "rejected_full_finger_mask.png")
    assert "rectangular/background-like" in payload
    assert "example.jpg" in payload
    assert '"stage": "full finger"' in payload
    assert '"accepted": false' in payload

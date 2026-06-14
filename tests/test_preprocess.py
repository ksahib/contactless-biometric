from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "preprocess.py"
SPEC = importlib.util.spec_from_file_location("contactless_preprocess", MODULE_PATH)
preprocess = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preprocess
assert SPEC.loader is not None
SPEC.loader.exec_module(preprocess)


def _result_with_mask(mask: np.ndarray, score: float = 0.9) -> SimpleNamespace:
    pred_instances = SimpleNamespace(
        scores=np.array([score], dtype=np.float32),
        masks=np.expand_dims(mask.astype(np.uint8), axis=0),
    )
    return SimpleNamespace(pred_instances=pred_instances)


def _synthetic_ridge_image(
    period: float,
    *,
    orientation: float = np.pi / 2.0,
    shape: tuple[int, int] = (128, 128),
) -> np.ndarray:
    y_grid, x_grid = np.indices(shape, dtype=np.float32)
    normal_x = -float(np.sin(orientation))
    normal_y = float(np.cos(orientation))
    phase = x_grid * normal_x + y_grid * normal_y
    image = 127.5 + 100.0 * np.cos(2.0 * np.pi * phase / float(period))
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def _principal_axis_angle(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    coords = np.column_stack([xs, ys]).astype(np.float32)
    coords -= np.mean(coords, axis=0, keepdims=True)
    covariance = np.cov(coords, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    return (angle + 180.0) % 360.0 - 180.0


def _distance_to_vertical(angle: float) -> float:
    return min(abs(angle - 90.0), abs(angle + 90.0))


def _tapered_finger_mask(
    *,
    shape: tuple[int, int] = (160, 160),
    tip_at_top: bool = True,
) -> np.ndarray:
    height, width = shape
    cx = width // 2
    y_tip = int(round(0.12 * height))
    y_base = int(round(0.88 * height))
    tip_half_width = int(round(0.06 * width))
    base_half_width = int(round(0.20 * width))
    points = np.array(
        [
            [cx - tip_half_width, y_tip],
            [cx + tip_half_width, y_tip],
            [cx + base_half_width, y_base],
            [cx - base_half_width, y_base],
        ],
        dtype=np.int32,
    )
    mask = np.zeros(shape, dtype=np.uint8)
    preprocess.cv2.fillConvexPoly(mask, points, 255)
    if not tip_at_top:
        mask = np.ascontiguousarray(np.flipud(np.fliplr(mask)))
    return mask


def _top_bottom_widths(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask > 0)
    y_min = int(ys.min())
    y_max = int(ys.max())
    height = y_max - y_min + 1
    band_height = max(2, int(round(0.2 * height)))

    def band_width(row_start: int, row_stop: int) -> float:
        widths: list[float] = []
        for y in range(row_start, row_stop + 1):
            cols = xs[ys == y]
            if cols.size > 0:
                widths.append(float(cols.max() - cols.min() + 1))
        return float(np.median(np.asarray(widths, dtype=np.float32)))

    top_width = band_width(y_min, min(y_min + band_height - 1, y_max))
    bottom_width = band_width(max(y_min, y_max - band_height + 1), y_max)
    return top_width, bottom_width


def _assert_tip_at_top(mask: np.ndarray) -> None:
    top_width, bottom_width = _top_bottom_widths(mask)
    assert top_width < bottom_width


def test_estimate_ridge_frequency_returns_scalar_for_known_period() -> None:
    image = _synthetic_ridge_image(10.0, orientation=np.pi / 2.0)
    mask = np.ones_like(image, dtype=np.uint8)

    frequency = preprocess.estimate_ridge_frequency(
        image,
        mask=mask,
        orientation=np.pi / 2.0,
    )

    assert isinstance(frequency, float)
    assert frequency == pytest.approx(1.0 / 10.0, rel=0.08)


def test_estimate_ridge_frequency_auto_orientation() -> None:
    image = _synthetic_ridge_image(8.0, orientation=np.pi / 2.0)
    mask = np.ones_like(image, dtype=np.uint8)

    frequency = preprocess.estimate_ridge_frequency(image, mask=mask)

    assert frequency == pytest.approx(1.0 / 8.0, rel=0.1)


def test_estimate_ridge_frequency_uses_masked_central_roi() -> None:
    image = _synthetic_ridge_image(14.0, orientation=np.pi / 2.0)
    central = _synthetic_ridge_image(8.0, orientation=np.pi / 2.0)
    image[32:96, 32:96] = central[32:96, 32:96]

    roi = np.zeros_like(image, dtype=np.uint8)
    roi[24:104, 24:104] = 1
    mask = preprocess.circular_mask(roi)

    frequency = preprocess.estimate_ridge_frequency(
        image,
        mask=mask,
        orientation=np.pi / 2.0,
    )

    assert frequency == pytest.approx(1.0 / 8.0, rel=0.1)


def test_estimate_ridge_frequency_rejects_uniform_roi() -> None:
    image = np.full((96, 96), 120, dtype=np.uint8)
    mask = np.ones_like(image, dtype=np.uint8)

    with pytest.raises(RuntimeError, match="ridge period"):
        preprocess.estimate_ridge_frequency(image, mask=mask, orientation=np.pi / 2.0)


def test_rotate_to_vertical_centerline_keeps_canvas_and_verticalizes_mask() -> None:
    mask = np.zeros((120, 120), dtype=np.uint8)
    box = preprocess.cv2.boxPoints(((60.0, 60.0), (18.0, 78.0), 35.0))
    preprocess.cv2.drawContours(mask, [np.round(box).astype(np.int32)], -1, 255, thickness=-1)
    image = np.where(mask > 0, 180, 0).astype(np.uint8)

    rotated_image, rotated_mask, yaw_angle = preprocess.rotate_to_vertical_centerline(image, mask)

    assert rotated_image.shape == image.shape
    assert rotated_mask.shape == mask.shape
    assert rotated_image.dtype == np.uint8
    assert rotated_mask.dtype == np.uint8
    assert abs(yaw_angle) > 1.0
    assert _distance_to_vertical(_principal_axis_angle(rotated_mask)) < 8.0


def test_rotate_to_vertical_centerline_keeps_tip_at_top_when_already_correct() -> None:
    mask = _tapered_finger_mask(tip_at_top=True)
    image = np.where(mask > 0, 180, 0).astype(np.uint8)

    rotated_image, rotated_mask, yaw_angle = preprocess.rotate_to_vertical_centerline(image, mask)

    assert rotated_image.shape == image.shape
    assert rotated_mask.shape == mask.shape
    assert yaw_angle == pytest.approx(0.0, abs=1.0)
    assert _distance_to_vertical(_principal_axis_angle(rotated_mask)) < 8.0
    _assert_tip_at_top(rotated_mask)


def test_rotate_to_vertical_centerline_flips_upside_down_tapered_finger() -> None:
    mask = _tapered_finger_mask(tip_at_top=False)
    image = np.where(mask > 0, 180, 0).astype(np.uint8)

    rotated_image, rotated_mask, yaw_angle = preprocess.rotate_to_vertical_centerline(image, mask)

    assert rotated_image.shape == image.shape
    assert rotated_mask.shape == mask.shape
    assert abs(abs(yaw_angle) - 180.0) < 1.0
    assert _distance_to_vertical(_principal_axis_angle(rotated_mask)) < 8.0
    _assert_tip_at_top(rotated_mask)


def test_rotate_to_vertical_centerline_verticalizes_angled_tapered_finger_tip_up() -> None:
    mask = _tapered_finger_mask(tip_at_top=True)
    image = np.where(mask > 0, 180, 0).astype(np.uint8)
    angled_mask = preprocess._rotate_same_canvas(mask, -35.0, interpolation=preprocess.cv2.INTER_NEAREST)
    angled_image = preprocess._rotate_same_canvas(image, -35.0, interpolation=preprocess.cv2.INTER_LINEAR)
    angled_image[angled_mask <= 0] = 0

    rotated_image, rotated_mask, yaw_angle = preprocess.rotate_to_vertical_centerline(angled_image, angled_mask)

    assert rotated_image.shape == image.shape
    assert rotated_mask.shape == mask.shape
    assert abs(yaw_angle) > 1.0
    assert _distance_to_vertical(_principal_axis_angle(rotated_mask)) < 8.0
    _assert_tip_at_top(rotated_mask)


def test_run_preprocess_pipeline_uses_requested_stage_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    image = np.full((16, 16, 3), 20, dtype=np.uint8)
    enhanced = np.full((16, 16), 80, dtype=np.uint8)
    full_mask = np.ones((16, 16), dtype=np.uint8) * 255
    center_mask = np.eye(16, dtype=np.uint8)
    scaled_image = np.full((20, 20), 90, dtype=np.uint8)
    scaled_mask = np.ones((20, 20), dtype=np.uint8) * 255
    rotated_image = np.full((20, 20), 100, dtype=np.uint8)
    rotated_mask = scaled_mask.copy()

    def fake_segment_then_clahe(*args: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        calls.append("segment")
        assert args[0] is image
        return enhanced, full_mask

    def fake_circular_mask(mask: np.ndarray) -> np.ndarray:
        calls.append("circle")
        assert mask is full_mask
        return center_mask

    def fake_scale_to_paper_ridge_period(
        enhanced_arg: np.ndarray,
        mask_arg: np.ndarray,
        **kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        calls.append("scale")
        assert enhanced_arg is enhanced
        assert mask_arg is full_mask
        assert kwargs["center_mask"] is center_mask
        return scaled_image, scaled_mask, 12.5, 0.8

    def fake_rotate_to_vertical_centerline(
        image_arg: np.ndarray,
        mask_arg: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        calls.append("rotate")
        assert image_arg is scaled_image
        assert mask_arg is scaled_mask
        return rotated_image, rotated_mask, -7.0

    monkeypatch.setattr(preprocess, "segment_then_clahe", fake_segment_then_clahe)
    monkeypatch.setattr(preprocess, "circular_mask", fake_circular_mask)
    monkeypatch.setattr(preprocess, "scale_to_paper_ridge_period", fake_scale_to_paper_ridge_period)
    monkeypatch.setattr(preprocess, "rotate_to_vertical_centerline", fake_rotate_to_vertical_centerline)

    result = preprocess.run_preprocess_pipeline(image, device="cpu")

    assert calls == ["segment", "circle", "scale", "rotate"]
    assert result.enhanced is enhanced
    assert result.scaled_image is scaled_image
    assert result.rotated_image is rotated_image
    assert result.ridge_period == pytest.approx(12.5)
    assert result.scale == pytest.approx(0.8)
    assert result.yaw_angle == pytest.approx(-7.0)


def test_segment_then_clahe_returns_masked_full_size_grayscale(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[:, :, 0] = np.tile(np.arange(24, dtype=np.uint8), (24, 1))
    image[:, :, 1] = 80
    image[:, :, 2] = 160

    raw_mask = np.zeros((24, 24), dtype=np.uint8)
    raw_mask[5:19, 7:17] = 1

    monkeypatch.setattr(preprocess, "_get_detector", lambda **_: object())
    monkeypatch.setattr(
        preprocess,
        "_load_mmdet_apis",
        lambda: (
            object(),
            lambda detector, inference_bgr: _result_with_mask(raw_mask),
        ),
    )

    enhanced, mask = preprocess.segment_then_clahe(image)

    assert enhanced.shape == (24, 24)
    assert mask.shape == (24, 24)
    assert enhanced.dtype == np.uint8
    assert mask.dtype == np.uint8
    assert np.all(enhanced[mask == 0] == 0)
    assert np.count_nonzero(mask) > 0
    assert np.count_nonzero(enhanced) > 0


def test_get_detector_reuses_cached_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "model.py"
    checkpoint_path = tmp_path / "weights.pth"
    config_path.write_text("model = dict(type='SOLOv2')\n", encoding="utf-8")
    checkpoint_path.write_bytes(b"checkpoint")

    init_calls: list[tuple[str, str, str]] = []

    def fake_init_detector(config: str, checkpoint: str, device: str) -> object:
        init_calls.append((config, checkpoint, device))
        return {"config": config, "checkpoint": checkpoint, "device": device}

    monkeypatch.setattr(preprocess, "_MODEL_CACHE", {})
    monkeypatch.setattr(preprocess, "_resolve_device", lambda device: "cpu")
    monkeypatch.setattr(
        preprocess,
        "_load_mmdet_apis",
        lambda: (fake_init_detector, object()),
    )

    first = preprocess._get_detector(
        model_config=config_path,
        checkpoint=checkpoint_path,
        device=None,
    )
    second = preprocess._get_detector(
        model_config=config_path,
        checkpoint=checkpoint_path,
        device=None,
    )

    assert first is second
    assert len(init_calls) == 1


def test_get_detector_raises_clear_error_for_missing_assets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_config = tmp_path / "missing_config.py"
    missing_checkpoint = tmp_path / "missing_checkpoint.pth"

    monkeypatch.delenv(preprocess.SOLOV2_CONFIG_ENV, raising=False)
    monkeypatch.delenv(preprocess.SOLOV2_CHECKPOINT_ENV, raising=False)
    monkeypatch.setattr(preprocess, "DEFAULT_SOLOV2_CONFIG", missing_config)
    monkeypatch.setattr(preprocess, "DEFAULT_SOLOV2_CHECKPOINT", missing_checkpoint)

    with pytest.raises(FileNotFoundError, match="FINGER_SOLOV2_CONFIG"):
        preprocess._get_detector()


def test_segment_then_clahe_raises_when_detection_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.full((16, 16), 120, dtype=np.uint8)

    empty_result = SimpleNamespace(
        pred_instances=SimpleNamespace(
            scores=np.zeros((0,), dtype=np.float32),
            masks=np.zeros((0, 16, 16), dtype=np.uint8),
        )
    )

    monkeypatch.setattr(preprocess, "_get_detector", lambda **_: object())
    monkeypatch.setattr(
        preprocess,
        "_load_mmdet_apis",
        lambda: (object(), lambda detector, inference_bgr: empty_result),
    )

    with pytest.raises(RuntimeError, match="no distal phalanx instance was detected"):
        preprocess.segment_then_clahe(image)


@pytest.mark.skipif(
    not (
        Path(os.environ.get(preprocess.SOLOV2_CONFIG_ENV, preprocess.DEFAULT_SOLOV2_CONFIG)).exists()
        and Path(os.environ.get(preprocess.SOLOV2_CHECKPOINT_ENV, preprocess.DEFAULT_SOLOV2_CHECKPOINT)).exists()
        and importlib.util.find_spec("mmdet") is not None
        and os.name != "nt"
    ),
    reason="SOLOv2 integration smoke test requires repo-local assets and a Linux/WSL MMDetection runtime",
)
def test_segment_then_clahe_smoke_with_real_assets() -> None:
    sample_image = (
        Path(__file__).resolve().parents[1]
        / "dataset"
        / "DS3"
        / "106"
        / "raw"
        / "106_3_2_0.jpg"
    )
    if not sample_image.exists():
        pytest.skip("sample image for smoke test is not available")

    import cv2

    image = cv2.imread(str(sample_image), cv2.IMREAD_COLOR)
    assert image is not None

    enhanced, mask = preprocess.segment_then_clahe(image)

    assert enhanced.shape == image.shape[:2]
    assert mask.shape == image.shape[:2]
    assert np.count_nonzero(mask) > 0

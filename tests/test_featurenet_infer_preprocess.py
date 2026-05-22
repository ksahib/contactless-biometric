from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import importlib
import sys
import types

import numpy as np
import pytest

try:
    from featurenet.models import infer
except ImportError as exc:
    if "FeatureExtractor" not in str(exc):
        raise
    stub = types.ModuleType("featurenet.models.feature_extractor")

    class FeatureExtractor:  # pragma: no cover - only used when local model edits are incomplete.
        pass

    stub.FeatureExtractor = FeatureExtractor
    sys.modules["featurenet.models.feature_extractor"] = stub
    sys.modules.pop("featurenet.models.infer", None)
    infer = importlib.import_module("featurenet.models.infer")


def test_preprocess_input_bgr_uses_canonical_preprocess_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    full_bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    rotated_image = np.arange(24, dtype=np.uint8).reshape(4, 6)
    rotated_mask = np.zeros((4, 6), dtype=np.uint8)
    rotated_mask[:, 1:5] = 255

    calls: list[dict[str, object]] = []

    def fake_run_preprocess_pipeline(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace(
            enhanced=np.full((8, 8), 30, dtype=np.uint8),
            full_mask=np.ones((8, 8), dtype=np.uint8) * 255,
            center_mask=np.eye(8, dtype=np.uint8) * 255,
            scaled_image=np.full((4, 6), 40, dtype=np.uint8),
            scaled_mask=rotated_mask.copy(),
            ridge_period=12.5,
            scale=0.8,
            rotated_image=rotated_image,
            rotated_mask=rotated_mask,
            yaw_angle=-5.0,
        )

    monkeypatch.setattr(infer.solov2_preprocess, "run_preprocess_pipeline", fake_run_preprocess_pipeline)

    image_tensor, mask_tensor, input_shape = infer.preprocess_input_bgr(
        full_bgr,
        save_preprocess_dir=tmp_path,
        solov2_config=Path("model.py"),
        solov2_checkpoint=Path("weights.pth"),
        solov2_device="cpu",
        solov2_score_thr=0.05,
    )

    assert len(calls) == 1
    assert calls[0]["args"] == (full_bgr,)
    assert calls[0]["kwargs"]["score_thr"] == pytest.approx(0.05)
    assert calls[0]["kwargs"]["device"] == "cpu"
    assert input_shape == (4, 6)
    assert image_tensor.shape == (1, 1, 4, 6)
    assert mask_tensor.shape == (1, 1, 4, 6)
    assert np.allclose(mask_tensor.numpy()[0, 0], (rotated_mask > 0).astype(np.float32))
    assert np.all(image_tensor.numpy()[0, 0][rotated_mask <= 0] == 0.0)
    assert (tmp_path / "meta.json").exists()
    assert not hasattr(infer, "normalise_brightness_array")
    assert not hasattr(infer, "_normalize_ridge_frequency")

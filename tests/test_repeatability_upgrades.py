from __future__ import annotations

import importlib.util
import sys
import sysconfig
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _ensure_stdlib_copy_module() -> None:
    stdlib_copy = Path(sysconfig.get_paths()["stdlib"]) / "copy.py"
    spec = importlib.util.spec_from_file_location("copy", stdlib_copy)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not resolve stdlib copy module from {stdlib_copy}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["copy"] = module
    spec.loader.exec_module(module)


_ensure_stdlib_copy_module()
sys.path.append(str(Path(__file__).resolve().parents[1]))

from featurenet.models.augmentation import (  # noqa: E402
    AugmentationParams,
    apply_photometric,
    render_2d_augmented,
)
from featurenet.models.blocks import BlurPool2d, MaxBlurPool2d  # noqa: E402
from featurenet.models.feature_extractor import FeatureExtractor  # noqa: E402
from featurenet.models.losses import (  # noqa: E402
    ConsistencyConfig,
    apply_gpu_photometric,
    make_affine_theta,
    masked_mse,
    warp_tensor,
)
from featurenet.models.train import _augment_losses_with_consistency  # noqa: E402


def _neutral_params(**overrides) -> AugmentationParams:
    base = dict(
        translate=False,
        yaw=False,
        pitch=False,
        roll=False,
        dx=0.0,
        dy=0.0,
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
    )
    base.update(overrides)
    return AugmentationParams(**base)


class PhotometricAugmentationTests(unittest.TestCase):
    def test_neutral_params_are_a_no_op(self) -> None:
        image = np.zeros((16, 16), dtype=np.uint8)
        image[4:12, 4:12] = 150
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[3:13, 3:13] = 255
        out = apply_photometric(image, mask, _neutral_params())
        self.assertTrue(np.array_equal(out, image))

    def test_brightness_changes_foreground_and_keeps_background_zero(self) -> None:
        image = np.full((16, 16), 100, dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:12, 4:12] = 255
        image[mask == 0] = 0
        params = _neutral_params(brightness=40.0)
        out = apply_photometric(image, mask, params)
        foreground = mask > 0
        self.assertGreater(float(out[foreground].mean()), float(image[foreground].mean()))
        self.assertEqual(int(out[~foreground].sum()), 0)


class ScaleJitterTests(unittest.TestCase):
    def test_uniform_scale_scales_ridge_and_minutiae(self) -> None:
        size = 32
        image = np.full((size, size), 120, dtype=np.uint8)
        mask = np.full((size, size), 255, dtype=np.uint8)
        orientation = np.zeros((size, size), dtype=np.float32)
        ridge = np.full((size, size), 10.0, dtype=np.float32)
        gradient = np.zeros((size, size, 2), dtype=np.float32)
        minutiae = [{"x": 8.0, "y": 8.0, "theta": 0.0, "score": 1.0, "type": "ending"}]
        scale = 1.5
        params = _neutral_params(scale=scale)

        rendered = render_2d_augmented(
            image=image,
            mask=mask,
            orientation=orientation,
            ridge_period=ridge,
            gradient=gradient,
            minutiae=minutiae,
            params=params,
        )

        foreground = rendered["mask"] > 0
        self.assertAlmostEqual(float(rendered["ridge_period"][foreground].mean()), 10.0 * scale, delta=0.3)

        center = (size - 1.0) * 0.5
        expected_x = center + scale * (8.0 - center)
        self.assertEqual(len(rendered["minutiae"]), 1)
        self.assertAlmostEqual(rendered["minutiae"][0]["x"], expected_x, delta=0.5)
        self.assertAlmostEqual(rendered["minutiae"][0]["y"], expected_x, delta=0.5)


class ConsistencyHelperTests(unittest.TestCase):
    def test_identity_warp_and_masked_mse_zero_when_equal(self) -> None:
        x = torch.rand(2, 1, 8, 8)
        theta = make_affine_theta(2, 0.0, 0.0, 0.0, torch.device("cpu"))
        warped = warp_tensor(x, theta, mode="bilinear")
        torch.testing.assert_close(warped, x, atol=1e-5, rtol=1e-4)
        mask = torch.ones(2, 1, 8, 8)
        self.assertLess(float(masked_mse(x, warped, mask)), 1e-8)

    def test_consistency_gradient_only_flows_to_partner_branch(self) -> None:
        s1 = torch.randn(2, 1, 8, 8, requires_grad=True)
        s2 = torch.randn(2, 1, 8, 8, requires_grad=True)
        theta = make_affine_theta(2, 10.0, 0.05, 0.05, torch.device("cpu"))
        prob1 = torch.sigmoid(s1)
        prob2 = torch.sigmoid(s2)
        reference = warp_tensor(prob1, theta, mode="bilinear").detach()
        mask = torch.ones(2, 1, 8, 8)
        loss = masked_mse(prob2, reference, mask)
        loss.backward()
        self.assertIsNone(s1.grad)
        self.assertIsNotNone(s2.grad)
        self.assertGreater(float(s2.grad.abs().sum()), 0.0)

    def test_gpu_photometric_preserves_shape_and_masks_background(self) -> None:
        image = torch.rand(2, 1, 8, 8)
        mask = torch.zeros(2, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1.0
        out = apply_gpu_photometric(image, mask, ConsistencyConfig())
        self.assertEqual(out.shape, image.shape)
        self.assertEqual(float(out[mask == 0].abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(out).all().item())


class BlurPoolTests(unittest.TestCase):
    def test_output_shape_matches_maxpool(self) -> None:
        reference = nn.MaxPool2d(kernel_size=2, stride=2)
        for height, width in [(8, 8), (7, 9), (16, 15), (33, 32)]:
            x = torch.randn(1, 4, height, width)
            blur_out = MaxBlurPool2d(4)(x)
            max_out = reference(x)
            self.assertEqual(blur_out.shape[-2:], max_out.shape[-2:], f"{height}x{width}")

    def test_blur_kernel_is_non_persistent(self) -> None:
        self.assertNotIn("kernel", BlurPool2d(4).state_dict())
        self.assertEqual(len(MaxBlurPool2d(4).state_dict()), 0)


class FeatureExtractorBlurPoolTests(unittest.TestCase):
    def test_state_dict_has_no_blur_kernel_keys(self) -> None:
        model = FeatureExtractor()
        self.assertFalse(any(key.endswith("kernel") for key in model.state_dict()))

    def test_forward_produces_expected_heads(self) -> None:
        model = FeatureExtractor()
        model.eval()
        image = torch.rand(1, 1, 64, 64)
        mask = torch.ones(1, 1, 64, 64)
        with torch.no_grad():
            outputs = model(image, mask=mask)
        for key in ("minutia_score", "minutia_x", "minutia_y", "orientation", "ridge_period", "gradient"):
            self.assertIn(key, outputs)
        self.assertEqual(outputs["minutia_score"].shape[-2:], (8, 8))


class ConsistencyTrainStepTests(unittest.TestCase):
    @staticmethod
    def _make_targets(batch: int, out_h: int, out_w: int) -> dict[str, torch.Tensor]:
        targets = {
            "minutia_score": torch.zeros(batch, out_h, out_w),
            "minutia_score_weight_map": torch.ones(batch, out_h, out_w),
        }
        targets["minutia_score"][:, out_h // 2, out_w // 2] = 1.0
        return targets

    def _run_mode(self, mode: str) -> None:
        torch.manual_seed(0)
        model = FeatureExtractor()
        model.train()
        image = torch.rand(2, 1, 64, 64)
        mask = torch.ones(2, 1, 64, 64)
        outputs = model(image, mask=mask)
        targets = self._make_targets(2, 8, 8)
        losses = {"total": outputs["minutia_score"].mean()}
        cfg = ConsistencyConfig(weight=10.0, mode=mode)

        _augment_losses_with_consistency(
            model, image, mask, outputs, targets, losses, cfg, amp=False, amp_dtype=torch.float16
        )

        self.assertIn("consistency", losses)
        self.assertGreaterEqual(float(losses["consistency"]), 0.0)
        self.assertTrue(torch.isfinite(losses["total"]).item())

        losses["total"].backward()
        grads = [param.grad for param in model.parameters() if param.grad is not None]
        self.assertGreater(len(grads), 0)

    def test_consistency_term_added_and_backpropagates(self) -> None:
        self._run_mode("supervised")

    def test_legacy_self_mode_still_works(self) -> None:
        self._run_mode("self")

    def test_supervised_mode_penalizes_score_collapse(self) -> None:
        """Suppressing every score must cost far more than predicting the
        warped GT — the collapse shortcut of the legacy self mode is gone.
        Uses a zero-magnitude warp so the warped target is deterministic."""
        torch.manual_seed(0)
        inner = FeatureExtractor()
        inner.train()
        image = torch.rand(1, 1, 64, 64)
        mask = torch.ones(1, 1, 64, 64)
        outputs = inner(image, mask=mask)
        targets = self._make_targets(1, 8, 8)
        cfg = ConsistencyConfig(
            weight=1.0,
            self_weight=0.0,
            orientation_weight=0.0,
            max_rot_deg=0.0,
            max_shift_frac=0.0,
            max_scale_delta=0.0,
        )

        target_logits = torch.where(
            targets["minutia_score"].unsqueeze(1) > 0.5,
            torch.full((1, 1, 8, 8), 12.0),
            torch.full((1, 1, 8, 8), -12.0),
        )

        class _FixedScoreModel(torch.nn.Module):
            def __init__(self, wrapped: torch.nn.Module, score_logits: torch.Tensor):
                super().__init__()
                self.wrapped = wrapped
                self.score_logits = score_logits

            def forward(self, x, mask=None):
                out = dict(self.wrapped(x, mask=mask))
                out["minutia_score"] = self.score_logits.clone().requires_grad_(True)
                return out

        losses_ideal = {"total": torch.zeros(())}
        _augment_losses_with_consistency(
            _FixedScoreModel(inner, target_logits), image, mask, outputs, targets, losses_ideal, cfg,
            amp=False, amp_dtype=torch.float16,
        )
        losses_collapsed = {"total": torch.zeros(())}
        _augment_losses_with_consistency(
            _FixedScoreModel(inner, torch.full((1, 1, 8, 8), -12.0)), image, mask, outputs, targets,
            losses_collapsed, cfg, amp=False, amp_dtype=torch.float16,
        )

        ideal = float(losses_ideal["consistency"].detach())
        collapsed = float(losses_collapsed["consistency"].detach())
        self.assertLess(ideal, 0.01)
        self.assertGreater(collapsed, 0.5)
        self.assertGreater(collapsed, ideal * 100)


if __name__ == "__main__":
    unittest.main()

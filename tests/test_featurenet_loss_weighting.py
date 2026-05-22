from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from featurenet.models.losses import FeatureNetLoss, soft_bce_logits_loss


def _dummy_step():
    batch, height, width = 1, 2, 2
    outputs = {
        "orientation": torch.zeros(batch, 180, height, width),
        "ridge_period": torch.full((batch, 1, height, width), 0.25),
        "gradient": torch.full((batch, 2, height, width), 0.25),
        "minutia_score": torch.zeros(batch, 1, height, width),
        "minutia_x": torch.zeros(batch, 1, height, width),
        "minutia_y": torch.zeros(batch, 1, height, width),
        "minutia_orientation": torch.ones(batch, 2, height, width),
    }
    target_orientation = torch.zeros(batch, 180, height, width)
    target_orientation[:, 0, :, :] = 1.0
    targets = {
        "mask": torch.ones(batch, 1, height, width),
        "orientation": target_orientation,
        "ridge_period": torch.zeros(batch, 1, height, width),
        "gradient": torch.zeros(batch, 2, height, width),
        "minutia_score": torch.ones(batch, 1, height, width),
        "minutia_valid_mask": torch.ones(batch, 1, height, width),
        "minutia_x_offset": torch.full((batch, 1, height, width), 0.25),
        "minutia_y_offset": torch.full((batch, 1, height, width), 0.75),
        "minutia_orientation_vec": torch.stack(
            [
                torch.ones(batch, height, width),
                torch.zeros(batch, height, width),
            ],
            dim=1,
        ),
    }
    return outputs, targets


def test_zero_dense_weights_isolate_score_loss():
    criterion = FeatureNetLoss(
        orientation_weight=0.0,
        ridge_weight=0.0,
        gradient_weight=0.0,
        mu_score=1.0,
        mu_x=0.0,
        mu_y=0.0,
        mu_ori=0.0,
    )
    losses = criterion(*_dummy_step())

    assert torch.allclose(losses["total"], losses["m1"])


def test_zero_minutia_weights_isolate_dense_orientation_loss():
    criterion = FeatureNetLoss(
        orientation_weight=1.0,
        ridge_weight=0.0,
        gradient_weight=0.0,
        mu_score=0.0,
        mu_x=0.0,
        mu_y=0.0,
        mu_ori=0.0,
    )
    losses = criterion(*_dummy_step())

    assert torch.allclose(losses["total"], losses["orientation"])


def test_minutia_subweights_accept_zero_for_orientation_only():
    criterion = FeatureNetLoss(
        orientation_weight=0.0,
        ridge_weight=0.0,
        gradient_weight=0.0,
        mu_score=0.0,
        mu_x=0.0,
        mu_y=0.0,
        mu_ori=1.0,
    )
    losses = criterion(*_dummy_step())

    assert torch.allclose(losses["total"], losses["m4"])


def test_default_loss_runs_xy_offset_heads():
    criterion = FeatureNetLoss(
        orientation_weight=0.0,
        ridge_weight=0.0,
        gradient_weight=0.0,
        mu_score=0.0,
        mu_x=1.0,
        mu_y=1.0,
        mu_ori=0.0,
    )
    losses = criterion(*_dummy_step())

    assert torch.isfinite(losses["m2"])
    assert torch.isfinite(losses["m3"])
    assert torch.allclose(losses["total"], losses["m2"] + losses["m3"])


def test_soft_bce_logits_loss_accepts_soft_targets_and_backward():
    logits = torch.zeros((2, 1, 4, 4), requires_grad=True)
    target = torch.full((2, 1, 4, 4), 0.25)
    weight = torch.ones((2, 1, 4, 4))

    loss = soft_bce_logits_loss(logits, target, weight)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_soft_bce_logits_loss_zero_weights_ignores_cells():
    logits = torch.zeros((1, 1, 2, 2), requires_grad=True)
    target = torch.tensor([[[[0.0, 1.0], [0.25, 0.75]]]])
    weight = torch.zeros_like(target)

    loss = soft_bce_logits_loss(logits, target, weight)
    loss.backward()

    assert torch.allclose(loss, torch.tensor(0.0))
    assert logits.grad is not None
    assert torch.allclose(logits.grad, torch.zeros_like(logits.grad))


def test_feature_loss_uses_score_weight_map():
    outputs, targets = _dummy_step()
    targets["minutia_score"] = torch.full_like(targets["minutia_score"], 0.25)
    targets["minutia_score_weight_map"] = torch.zeros_like(targets["minutia_score"])
    criterion = FeatureNetLoss(
        orientation_weight=0.0,
        ridge_weight=0.0,
        gradient_weight=0.0,
        mu_score=1.0,
        mu_x=0.0,
        mu_y=0.0,
        mu_ori=0.0,
    )

    losses = criterion(outputs, targets)

    assert torch.allclose(losses["m1"], torch.tensor(0.0))
    assert torch.allclose(losses["total"], torch.tensor(0.0))

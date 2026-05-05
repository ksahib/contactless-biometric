import torch

from featurenet.models.losses import FeatureNetLoss


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

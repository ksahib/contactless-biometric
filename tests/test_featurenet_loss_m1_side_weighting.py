import math

import torch
import torch.nn.functional as F

from featurenet.models.losses import FeatureNetLoss


def _criterion(side_weight=3.0, neg_weight=2.0):
    return FeatureNetLoss(
        m1_focal_gamma=0.0,
        m1_hard_neg_ratio=0.0,
        m1_hard_neg_min=1,
        m1_neg_weight=neg_weight,
        m1_side_pos_weight=side_weight,
    )


def _single_sample_inputs():
    logits = torch.tensor([[[[0.0, 2.0, -2.0]]]], dtype=torch.float32)
    target = torch.tensor([[[[1.0, 0.0, 0.0]]]], dtype=torch.float32)
    mask = torch.ones_like(target)
    return logits, target, mask


def _positive_bce_for_inputs(logits, target):
    positive_count = (target > 0.5).float().sum()
    negative_count = (target <= 0.0).float().sum()
    pos_weight = torch.sqrt(negative_count / (positive_count + 1e-8)).clamp(min=1.0, max=100.0)
    bce_map = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=pos_weight,
    )
    return bce_map[target > 0.5].mean()


def test_m1_missing_raw_view_index_matches_front():
    criterion = _criterion()
    logits, target, mask = _single_sample_inputs()

    missing_loss = criterion._compute_m1_score_loss(logits, target, mask)
    front_loss = criterion._compute_m1_score_loss(
        logits,
        target,
        mask,
        raw_view_index=torch.tensor([0]),
    )

    assert torch.allclose(missing_loss, front_loss)


def test_m1_side_view_weights_positive_loss_only():
    side_weight = 3.0
    criterion = _criterion(side_weight=side_weight)
    logits, target, mask = _single_sample_inputs()

    front_loss = criterion._compute_m1_score_loss(
        logits,
        target,
        mask,
        raw_view_index=torch.tensor([0]),
    )
    side_loss = criterion._compute_m1_score_loss(
        logits,
        target,
        mask,
        raw_view_index=torch.tensor([1]),
    )
    pos_loss = _positive_bce_for_inputs(logits, target)

    expected_delta = (side_weight - 1.0) * pos_loss
    assert torch.allclose(side_loss - front_loss, expected_delta, atol=1e-6)


def test_m1_right_side_view_uses_same_positive_weight():
    side_weight = 2.5
    criterion = _criterion(side_weight=side_weight)
    logits, target, mask = _single_sample_inputs()

    left_loss = criterion._compute_m1_score_loss(
        logits,
        target,
        mask,
        raw_view_index=torch.tensor([1]),
    )
    right_loss = criterion._compute_m1_score_loss(
        logits,
        target,
        mask,
        raw_view_index=torch.tensor([2]),
    )

    assert torch.allclose(left_loss, right_loss)


def test_m1_mixed_batch_weights_only_side_sample_positives():
    side_weight = 4.0
    criterion = _criterion(side_weight=side_weight)
    logits_one, target_one, mask_one = _single_sample_inputs()
    logits = logits_one.repeat(2, 1, 1, 1)
    target = target_one.repeat(2, 1, 1, 1)
    mask = mask_one.repeat(2, 1, 1, 1)

    front_batch_loss = criterion._compute_m1_score_loss(
        logits,
        target,
        mask,
        raw_view_index=torch.tensor([0, 0]),
    )
    mixed_batch_loss = criterion._compute_m1_score_loss(
        logits,
        target,
        mask,
        raw_view_index=torch.tensor([0, 1]),
    )
    pos_loss = _positive_bce_for_inputs(logits, target)

    expected_delta = ((side_weight - 1.0) / 2.0) * pos_loss
    assert torch.allclose(mixed_batch_loss - front_batch_loss, expected_delta, atol=1e-6)


def test_m1_side_weight_does_not_change_negative_only_loss():
    criterion = _criterion(side_weight=10.0)
    logits = torch.tensor([[[[0.0, 3.0, -3.0]]]], dtype=torch.float32)
    target = torch.zeros_like(logits)
    mask = torch.ones_like(logits)

    front_loss = criterion._compute_m1_score_loss(
        logits,
        target,
        mask,
        raw_view_index=torch.tensor([0]),
    )
    side_loss = criterion._compute_m1_score_loss(
        logits,
        target,
        mask,
        raw_view_index=torch.tensor([1]),
    )

    assert torch.allclose(side_loss, front_loss)
    assert math.isfinite(float(side_loss.item()))


def test_m1_hard_negative_selection_excludes_positives():
    criterion = _criterion()
    logits = torch.tensor([[[[0.0, 4.0, -4.0, 2.0]]]], dtype=torch.float32)
    target = torch.tensor([[[[1.0, 0.0, 0.25, 0.0]]]], dtype=torch.float32)
    valid = torch.ones_like(target, dtype=torch.bool)
    positive_mask = (target > 0.5) & valid
    negative_mask = (target <= 0.0) & valid

    selected_negative_mask = criterion._select_m1_hard_negative_mask(
        logits,
        positive_mask,
        negative_mask,
    )

    assert not bool((selected_negative_mask & positive_mask).any().item())
    assert not bool((selected_negative_mask & ((target > 0.0) & (target <= 0.5))).any().item())
    assert bool(selected_negative_mask.any().item())

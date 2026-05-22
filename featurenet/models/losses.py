import torch
import torch.nn as nn
import torch.nn.functional as F

MINUTIA_ORIENTATION_BINS = 360


def soft_bce_logits_loss(
    score_logits: torch.Tensor,
    score_target: torch.Tensor,
    score_weight: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Weighted BCEWithLogitsLoss for soft Gaussian minutiae targets."""
    if score_logits.ndim == 4 and score_logits.shape[1] == 1 and score_target.ndim == 3:
        score_target = score_target.unsqueeze(1)
    if score_logits.ndim == 4 and score_logits.shape[1] == 1 and score_weight is not None and score_weight.ndim == 3:
        score_weight = score_weight.unsqueeze(1)

    if score_logits.shape != score_target.shape:
        raise ValueError(
            "score_logits and score_target shape mismatch: "
            f"{tuple(score_logits.shape)} vs {tuple(score_target.shape)}"
        )

    score_target = score_target.to(device=score_logits.device, dtype=score_logits.dtype)
    score_target = torch.clamp(score_target, 0.0, 1.0)

    if score_weight is None:
        score_weight = torch.ones_like(score_target)
    else:
        if score_weight.shape != score_target.shape:
            raise ValueError(
                "score_weight and score_target shape mismatch: "
                f"{tuple(score_weight.shape)} vs {tuple(score_target.shape)}"
            )
        score_weight = score_weight.to(device=score_logits.device, dtype=score_logits.dtype)
        score_weight = torch.clamp(score_weight, min=0.0)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    per_cell = criterion(score_logits, score_target)
    weighted = per_cell * score_weight
    denom = torch.clamp(score_weight.sum(), min=eps)
    return weighted.sum() / denom


class OrientationLoss(nn.Module):
    def __init__(self, num_bins=180, alpha=1.0, eps=1e-8):
        super().__init__()
        self.num_bins = int(num_bins)
        self.alpha = float(alpha)
        self.eps = float(eps)

        kernel = torch.ones(1, 1, 3, 3) / 9.0
        self.register_buffer("smooth_kernel", kernel)

    def forward(self, pred, target, mask):
        pred = pred.float()
        target = target.float()

        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.float()

        if pred.dim() != 4:
            raise ValueError(f"expected pred with shape [B,C,H,W], got {tuple(pred.shape)}")

        if pred.shape[1] != self.num_bins:
            raise ValueError(
                f"pred has {pred.shape[1]} orientation bins, "
                f"but OrientationLoss was initialized with num_bins={self.num_bins}"
            )

        if target.shape != pred.shape:
            raise ValueError(
                f"target shape {tuple(target.shape)} must match pred shape {tuple(pred.shape)}"
            )

        # Paper-style BCE over the 180 orientation bins.
        class_loss_map = F.binary_cross_entropy_with_logits(
            pred,
            target,
            reduction="none",
        ).sum(dim=1, keepdim=True)

        L_class = (class_loss_map * mask).sum() / (mask.sum() + self.eps)

        # Paper-style probability estimate.
        pred_prob = torch.sigmoid(pred)

        # The paper uses N = 180 orientation bins.
        # These bins represent 0..180 degrees.
        angles = torch.arange(
            self.num_bins,
            device=pred.device,
            dtype=pred.dtype,
        ) * (180.0 / self.num_bins)

        rads = angles * torch.pi / 180.0

        # Paper formula: cos(360 * i / N), sin(360 * i / N).
        # With bins over 0..180, this is exactly doubled-angle encoding.
        cosine = torch.cos(2.0 * rads).view(1, self.num_bins, 1, 1)
        sine = torch.sin(2.0 * rads).view(1, self.num_bins, 1, 1)

        # With sigmoid probabilities, /num_bins is a real average.
        dcos = (pred_prob * cosine).sum(dim=1, keepdim=True) / self.num_bins
        dsin = (pred_prob * sine).sum(dim=1, keepdim=True) / self.num_bins

        dcos_s = F.conv2d(dcos, self.smooth_kernel, padding=1)
        dsin_s = F.conv2d(dsin, self.smooth_kernel, padding=1)

        magnitude = torch.sqrt(dcos_s.pow(2) + dsin_s.pow(2) + self.eps)

        # Correct sign: coherent field -> magnitude high -> loss low.
        L_coh_map = (1.0 - magnitude).clamp_min(0.0)
        L_coh = (L_coh_map * mask).sum() / (mask.sum() + self.eps)

        return L_class + self.alpha * L_coh

class RidgePeriodLoss(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target, mask):
        pred = pred.float()
        target = target.float()
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.float()
        mse = ((pred - target) ** 2)
        loss_mse = (mse * mask).sum() / (mask.sum() + 1e-8)
        dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        grad_loss = (dx**2).mean() + (dy**2).mean()
        return loss_mse + self.beta * grad_loss
    
class GradientLoss(nn.Module):
    def __init__(self, gamma, sigma):
        super().__init__()
        self.gamma = gamma
        self.sigma = sigma

    def forward(self, pred, target, mask):
        pred = pred.float()
        target = target.float()
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.float()
        mag = torch.sqrt((target ** 2).sum(dim=1, keepdim=True)+ 1e-8)
        w = torch.exp(-mag / self.sigma)
        w = w * mask
        mse = ((pred - target) ** 2).sum(dim=1, keepdim=True)
        dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        mask_dx = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        mask_dy = mask[:, :, :, 1:] * mask[:, :, :, :-1]

        dx_loss = ((dx ** 2).sum(dim=1, keepdim=True) * mask_dx).sum() / (mask_dx.sum() + 1e-8)
        dy_loss = ((dy ** 2).sum(dim=1, keepdim=True) * mask_dy).sum() / (mask_dy.sum() + 1e-8)

        grad_loss = dx_loss + dy_loss
        loss = (w * mse).sum()/(w.sum() + 1e-8) + self.gamma * grad_loss
        return loss

class FeatureNetLoss(nn.Module):
    def __init__(
        self,
        alpha=1.0,
        beta=60.0,
        gamma=300.0,
        sigma=0.5,
        orientation_weight=1.0,
        ridge_weight=1.0,
        gradient_weight=1.0,
        mu_score=120.0,
        mu_x=20.0,
        mu_y=20.0,
        mu_ori=5.0,
        m1_focal_gamma=2.0,
        m1_pos_weight_max=100.0,
        m1_hard_neg_enable=True,
        m1_hard_neg_ratio=20.0,
        m1_hard_neg_min=2000,
        m1_hard_neg_fraction=0.00,
        m1_neg_weight=2.0,
        m1_side_pos_weight=2.0,
        m1_fp_margin=0.60,
        m1_fp_penalty_weight=10.0,
        m1_ignore_radius=1,
        m1_candidate_loss_weight=1.0,
        m1_candidate_topk=512,
        m1_candidate_match_radius=1,
        m1_candidate_focal_gamma=2.0,
        xy_beta=0.05,
        xy_center_margin=0.02,
        xy_offcenter_weight=2.0,
        xy_anti_center_weight=1.0,
        minutia_ori_tolerance_deg=10.0,
        minutia_ori_angle_penalty_weight=1.0,
    ):
        super().__init__()
        if m1_side_pos_weight <= 0.0:
            raise ValueError("m1_side_pos_weight must be positive")
        if orientation_weight < 0.0 or ridge_weight < 0.0 or gradient_weight < 0.0:
            raise ValueError("orientation_weight, ridge_weight, and gradient_weight must be non-negative")
        if mu_score < 0.0 or mu_x < 0.0 or mu_y < 0.0 or mu_ori < 0.0:
            raise ValueError("mu_score, mu_x, mu_y, and mu_ori must be non-negative")
        if xy_beta <= 0.0:
            raise ValueError("xy_beta must be positive")
        if xy_center_margin < 0.0 or xy_offcenter_weight < 0.0 or xy_anti_center_weight < 0.0:
            raise ValueError(
                "xy_center_margin, xy_offcenter_weight, and xy_anti_center_weight must be non-negative"
            )

        # sub-losses
        self.orientation_loss = OrientationLoss(alpha=alpha)
        self.ridge_loss = RidgePeriodLoss(beta=beta)
        self.gradient_loss = GradientLoss(gamma=gamma, sigma=sigma)

        # weights
        self.orientation_weight = float(orientation_weight)
        self.ridge_weight = float(ridge_weight)
        self.gradient_weight = float(gradient_weight)
        self.mu_score = mu_score
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.mu_ori = mu_ori
        self.m1_focal_gamma = m1_focal_gamma
        self.m1_pos_weight_max = m1_pos_weight_max
        self.m1_hard_neg_enable = bool(m1_hard_neg_enable)
        self.m1_hard_neg_ratio = float(m1_hard_neg_ratio)
        self.m1_hard_neg_min = int(m1_hard_neg_min)
        self.m1_hard_neg_fraction = float(m1_hard_neg_fraction)
        self.m1_neg_weight = float(m1_neg_weight)
        self.m1_side_pos_weight = float(m1_side_pos_weight)
        self.m1_neg_weight = float(m1_neg_weight)
        self.m1_fp_margin = float(m1_fp_margin)
        self.m1_fp_penalty_weight = float(m1_fp_penalty_weight)
        self.m1_ignore_radius = int(m1_ignore_radius)
        self.m1_candidate_loss_weight = float(m1_candidate_loss_weight)
        self.m1_candidate_topk = int(m1_candidate_topk)
        self.m1_candidate_match_radius = int(m1_candidate_match_radius)
        self.m1_candidate_focal_gamma = float(m1_candidate_focal_gamma)
        self.xy_beta = float(xy_beta)
        self.xy_center_margin = float(xy_center_margin)
        self.xy_offcenter_weight = float(xy_offcenter_weight)
        self.xy_anti_center_weight = float(xy_anti_center_weight)
        self.minutia_ori_tolerance_deg = float(minutia_ori_tolerance_deg)
        self.minutia_ori_angle_penalty_weight = float(minutia_ori_angle_penalty_weight)


    def _resolve_center_minutia_mask(self, targets, fallback_minutia_mask):
        """
        Use only true positive / center minutia cells for x/y regression.

        Prefer the explicit diagnostic center map for Gaussian bundles.
        Fall back to old binary score/valid masks for older bundles.
        """
        center_source = targets.get("minutia_score_center_map")
        if center_source is None:
            center_source = targets.get("minutia_score")
        if center_source is None:
            return fallback_minutia_mask.float()

        center_mask = center_source
        if center_mask.dim() == 3:
            center_mask = center_mask.unsqueeze(1)

        center_mask = center_mask.float().to(fallback_minutia_mask.device)

        # Only keep cells that are both true score positives and valid minutia cells.
        return ((center_mask > 0.5) & (fallback_minutia_mask > 0.5)).float()

    def _orientation_bins_to_unit_vectors(self, bins: torch.Tensor) -> torch.Tensor:
        if bins.dim() == 4 and bins.shape[1] == 1:
            bins = bins.squeeze(1)
        angle = (bins.float().clamp(0, MINUTIA_ORIENTATION_BINS - 1) + 0.5) * (
            2.0 * torch.pi / float(MINUTIA_ORIENTATION_BINS)
        )
        cos_gt = torch.cos(angle)
        sin_gt = torch.sin(angle)
        return torch.stack([cos_gt, sin_gt], dim=1)

    def _resolve_minutia_orientation_target_vectors(self, targets: dict[str, torch.Tensor]) -> torch.Tensor:
        if "minutia_orientation_vec" in targets:
            target_vec = targets["minutia_orientation_vec"]
            if target_vec.dim() == 3:
                target_vec = target_vec.unsqueeze(0)
            if target_vec.dim() == 4 and target_vec.shape[-1] == 2 and target_vec.shape[1] != 2:
                target_vec = target_vec.permute(0, 3, 1, 2)
            if target_vec.dim() != 4 or target_vec.shape[1] != 2:
                raise ValueError(
                    f"expected minutia_orientation_vec with shape [B,2,H,W], got {tuple(target_vec.shape)}"
                )
            return F.normalize(target_vec.float(), dim=1, eps=1e-8)
        if "minutia_orientation" in targets:
            return self._orientation_bins_to_unit_vectors(targets["minutia_orientation"])
        raise KeyError("missing minutia orientation targets: expected minutia_orientation_vec or minutia_orientation")

    def _resolve_score_mask(self, mask):
        score_mask = mask
        if score_mask.dim() == 3:
            score_mask = score_mask.unsqueeze(1)
        score_mask = score_mask.float()
        if float(score_mask.sum().item()) <= 0.0:
            return torch.ones_like(score_mask, dtype=torch.float32)
        return score_mask

    def _resolve_minutia_mask(self, targets, score_mask):
        if "minutia_valid_mask" not in targets:
            return score_mask.float()
        minutia_mask = targets["minutia_valid_mask"]
        if minutia_mask.dim() == 3:
            minutia_mask = minutia_mask.unsqueeze(1)
        return minutia_mask.float()

    def _compute_m1_candidate_score_loss(
        self,
        logits: torch.Tensor,
        target_score: torch.Tensor,
        score_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Candidate-level score loss.

        Dense M1 loss asks:
            Is every cell positive/negative?

        Candidate M1 loss asks:
            Among the cells the model itself scores highest,
            are those cells actually near GT minutiae?

        This directly attacks high-score false positives.
        """
        eps = 1e-8

        if target_score.dim() == 3:
            target_score = target_score.unsqueeze(1)
        if score_mask.dim() == 3:
            score_mask = score_mask.unsqueeze(1)

        logits = logits.float()
        target_score = target_score.float().to(logits.device)
        score_mask = score_mask.float().to(logits.device)

        valid = score_mask > 0.5
        positive_mask = (target_score > 0.5) & valid

        batch_size = logits.shape[0]
        losses = []

        # A candidate is considered positive if it falls within this radius
        # of a GT positive cell. Radius is in output-grid cells, not pixels.
        match_radius = getattr(self, "m1_candidate_match_radius", 1)

        if match_radius > 0:
            kernel_size = 2 * match_radius + 1
            positive_region = F.max_pool2d(
                positive_mask.float(),
                kernel_size=kernel_size,
                stride=1,
                padding=match_radius,
            ) > 0.5
        else:
            positive_region = positive_mask

        topk = getattr(self, "m1_candidate_topk", 512)
        gamma = getattr(self, "m1_candidate_focal_gamma", 2.0)

        for b in range(batch_size):
            valid_b = valid[b, 0]
            if not bool(valid_b.any().item()):
                continue

            logits_b = logits[b, 0]
            labels_b = positive_region[b, 0].float()

            valid_logits = logits_b[valid_b]
            valid_labels = labels_b[valid_b]

            num_valid = valid_logits.numel()
            if num_valid <= 0:
                continue

            k = min(topk, num_valid)

            # Decode candidates by model confidence.
            # This is the important part: train on the model's own top predictions.
            topk_indices = torch.topk(
                valid_logits,
                k=k,
                sorted=False,
            ).indices

            cand_logits = valid_logits[topk_indices]
            cand_labels = valid_labels[topk_indices]

            # BCE on decoded candidates.
            bce = F.binary_cross_entropy_with_logits(
                cand_logits,
                cand_labels,
                reduction="none",
            )

            # Focal weighting so confident wrong candidates matter more.
            pt = torch.exp(-bce)
            focal = torch.pow((1.0 - pt).clamp(min=0.0), gamma) * bce

            # Balance positives/negatives inside candidate set.
            pos = cand_labels > 0.5
            neg = ~pos

            if bool(pos.any().item()) and bool(neg.any().item()):
                pos_loss = focal[pos].mean()
                neg_loss = focal[neg].mean()
                loss_b = pos_loss + neg_loss
            else:
                loss_b = focal.mean()

            losses.append(loss_b)

        if len(losses) == 0:
            return logits.new_tensor(0.0)

        return torch.stack(losses).mean()

    def _select_m1_hard_negative_mask(self, focal_map, positive_mask, negative_mask):
        selected_negative_mask = torch.zeros_like(negative_mask, dtype=torch.bool)
        if not self.m1_hard_neg_enable:
            return negative_mask

        batch_size = focal_map.shape[0]

        for batch_index in range(batch_size):
            focal_b = focal_map[batch_index, 0]
            positive_b = positive_mask[batch_index, 0]
            negative_b = negative_mask[batch_index, 0]

            negative_count = int(negative_b.sum().item())
            positive_count = int(positive_b.sum().item())

            if negative_count <= 0:
                continue

            ratio_k = int(positive_count * self.m1_hard_neg_ratio)
            hard_k = max(ratio_k, self.m1_hard_neg_min)
            hard_k = min(hard_k, negative_count)

            if hard_k <= 0:
                continue

            negative_scores = focal_b[negative_b]
            negative_indices = torch.nonzero(negative_b, as_tuple=False)

            if hard_k < negative_count:
                topk_indices = torch.topk(
                    negative_scores,
                    k=hard_k,
                    sorted=False,
                ).indices
                chosen_indices = negative_indices[topk_indices]
            else:
                chosen_indices = negative_indices

            selected_negative_mask[batch_index, 0, chosen_indices[:, 0], chosen_indices[:, 1]] = True

        return selected_negative_mask

    def _m1_side_positive_weight_map(self, focal_map, raw_view_index=None):
        side_weight_map = torch.ones_like(focal_map)
        if raw_view_index is None:
            return side_weight_map

        raw_view_tensor = torch.as_tensor(raw_view_index, device=focal_map.device).flatten()
        batch_size = focal_map.shape[0]
        if raw_view_tensor.numel() == 1 and batch_size > 1:
            raw_view_tensor = raw_view_tensor.expand(batch_size)
        if raw_view_tensor.numel() != batch_size:
            raise ValueError(
                f"expected raw_view_index with {batch_size} values, got {raw_view_tensor.numel()}"
            )

        side = (raw_view_tensor == 1) | (raw_view_tensor == 2)
        if bool(side.any().item()):
            side_weight_map[side, :, :, :] = self.m1_side_pos_weight
        return side_weight_map

    def _compute_m1_score_loss(self, logits, target_score, score_mask, raw_view_index=None, score_weight=None):
        if target_score.dim() == 3:
            target_score = target_score.unsqueeze(1)
        if score_mask.dim() == 3:
            score_mask = score_mask.unsqueeze(1)

        effective_weight = score_mask.float().to(logits.device)
        if score_weight is not None:
            if score_weight.dim() == 3:
                score_weight = score_weight.unsqueeze(1)
            effective_weight = effective_weight * score_weight.float().to(logits.device).clamp_min(0.0)

        return soft_bce_logits_loss(
            score_logits=logits,
            score_target=target_score,
            score_weight=effective_weight,
        )

    def _resolve_offset_target(
        self,
        target_offsets: torch.Tensor,
        output_like: torch.Tensor,
        head_name: str,
    ) -> torch.Tensor:
        if target_offsets.dim() == 3:
            target_offsets = target_offsets.unsqueeze(1)
        if target_offsets.dim() != 4 or target_offsets.shape[1] != 1:
            raise ValueError(
                f"expected {head_name} targets with shape [B,1,H,W] or [B,H,W], got {tuple(target_offsets.shape)}"
            )
        if target_offsets.shape[-2:] != output_like.shape[-2:]:
            raise ValueError(
                f"{head_name} target shape {tuple(target_offsets.shape[-2:])} does not match output shape "
                f"{tuple(output_like.shape[-2:])}"
            )
        return target_offsets.float().to(output_like.device)

    def _compute_xy_offset_loss(
        self,
        logits: torch.Tensor,
        target_offsets: torch.Tensor,
        minutia_mask: torch.Tensor,
        head_name: str,
    ) -> torch.Tensor:
        if logits.dim() != 4 or logits.shape[1] != 1:
            raise ValueError(
                f"expected {head_name} logits with shape [B,1,H,W], got {tuple(logits.shape)}"
            )

        target_offsets = self._resolve_offset_target(
            target_offsets,
            output_like=logits,
            head_name=head_name,
        )

        if minutia_mask.dim() == 3:
            minutia_mask = minutia_mask.unsqueeze(1)
        minutia_mask = minutia_mask.float().to(logits.device)

        active_mask = minutia_mask > 0.5

        invalid = ((target_offsets < 0.0) | (target_offsets > 1.0)) & active_mask
        if bool(invalid.any().item()):
            invalid_values = target_offsets[invalid]
            min_invalid = float(invalid_values.min().item())
            max_invalid = float(invalid_values.max().item())
            raise ValueError(
                f"{head_name} offsets contain out-of-range values on active cells; expected [0,1], "
                f"got min={min_invalid:.6f}, max={max_invalid:.6f}"
            )

        # Keep continuous offset semantics unchanged:
        # raw logit -> sigmoid -> offset in [0, 1]
        pred_offsets = torch.sigmoid(logits)

        # 1. Sharper continuous localization loss.
        # Default SmoothL1 is too forgiving for 0.20-0.25 cell errors.
        base_loss = F.smooth_l1_loss(
            pred_offsets,
            target_offsets,
            reduction="none",
            beta=self.xy_beta,
        )

        # 2. Explicit anti-center term.
        # If pred is not better than the center baseline, penalize it.
        pred_err = torch.abs(pred_offsets - target_offsets)
        center_err = torch.abs(target_offsets - 0.5)

        anti_center_loss = F.relu(
            pred_err - center_err + self.xy_center_margin
        )

        # 3. Weight off-center targets more.
        # Center prediction is especially bad for targets far from 0.5.
        offcenter_weight = 1.0 + self.xy_offcenter_weight * center_err.detach()

        loss_map = (
            base_loss
            + self.xy_anti_center_weight * anti_center_loss
        ) * offcenter_weight

        denom = minutia_mask.sum()
        if float(denom.item()) <= 0.0:
            return logits.new_tensor(0.0)

        return (loss_map * minutia_mask).sum() / (denom + 1e-8)

    def forward(self, outputs, targets):
        mask = targets["mask"]
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        score_mask = self._resolve_score_mask(mask)
        minutia_mask = self._resolve_minutia_mask(targets, score_mask)
        minutia_score_mask = torch.maximum(score_mask.float(), minutia_mask.float())
        minutia_center_mask = self._resolve_center_minutia_mask(
            targets,
            fallback_minutia_mask=minutia_mask,
        )

        # ---------------------------
        # 1. Orientation loss
        # ---------------------------
        L_ori = self.orientation_loss(
            outputs["orientation"],
            targets["orientation"],
            score_mask
        )

        # ---------------------------
        # 2. Ridge period loss
        # ---------------------------
        L_ridge = self.ridge_loss(
            outputs["ridge_period"],
            targets["ridge_period"],
            score_mask
        )

        # ---------------------------
        # 3. Gradient loss
        # ---------------------------
        L_grad = self.gradient_loss(
            outputs["gradient"],
            targets["gradient"],
            score_mask
        )

        # ---------------------------
        # 4. Minutiae losses
        # ---------------------------

        # --- M1: score (soft BCE over logits)
        L_m1 = self._compute_m1_score_loss(
            outputs["minutia_score"],
            targets["minutia_score"],
            minutia_score_mask,
            raw_view_index=targets.get("raw_view_index"),
            score_weight=targets.get("minutia_score_weight_map"),
        )

        L_m1_candidate = outputs["minutia_score"].new_tensor(0.0)

        # --- M2: x offset regression (continuous within-cell target)
        L_m2 = self._compute_xy_offset_loss(
            outputs["minutia_x"],
            targets["minutia_x_offset"],
            minutia_mask=minutia_center_mask,
            head_name="minutia_x",
        )

        # --- M3: y offset regression (continuous within-cell target)
        L_m3 = self._compute_xy_offset_loss(
            outputs["minutia_y"],
            targets["minutia_y_offset"],
            minutia_mask=minutia_center_mask,
            head_name="minutia_y",
        )

        # --- M4: minutia orientation circular loss
        pred_ori = outputs["minutia_orientation"]
        if pred_ori.dim() != 4 or pred_ori.shape[1] != 2:
            raise ValueError(
                f"expected minutia_orientation output with shape [B,2,H,W], got {tuple(pred_ori.shape)}"
            )

        pred_ori = F.normalize(pred_ori.float(), dim=1, eps=1e-8)
        target_ori = self._resolve_minutia_orientation_target_vectors(targets).to(pred_ori.device)

        if target_ori.shape != pred_ori.shape:
            raise ValueError(
                f"target minutia orientation vector shape {tuple(target_ori.shape)} "
                f"does not match prediction shape {tuple(pred_ori.shape)}"
            )

        target_ori = F.normalize(target_ori.float(), dim=1, eps=1e-8)

        # dot = cos(theta_pred - theta_gt), safely clamped
        dot = (pred_ori * target_ori).sum(dim=1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

        # Main circular loss.
        # 0 when aligned, 1 when 90 deg wrong, 2 when 180 deg wrong.
        cosine_loss = 1.0 - dot

        # Optional: explicitly punish angular error above tolerance.
        # This makes "kind of close" not enough if you want high orientation accuracy.
        angle_err = torch.acos(dot)  # radians, range [0, pi]

        ori_tolerance = self.minutia_ori_tolerance_deg * torch.pi / 180.0

        angle_penalty = F.relu(angle_err - ori_tolerance)

        # Combine.
        # Start with 1.0; increase to 2.0 if orientation MAE refuses to move.
        angle_penalty_weight = self.minutia_ori_angle_penalty_weight

        ori_loss_map = cosine_loss + angle_penalty_weight * angle_penalty

        L_m4 = (ori_loss_map * minutia_center_mask).sum() / (minutia_center_mask.sum() + 1e-8)

        # combine minutiae
        L_minu = (
            self.mu_score * L_m1 +
            self.mu_x * L_m2 +
            self.mu_y * L_m3 +
            self.mu_ori * L_m4
        )

        # ---------------------------
        # 5. Final total loss
        # ---------------------------
        total_loss = (
            self.orientation_weight * L_ori +
            self.ridge_weight * L_ridge +
            self.gradient_weight * L_grad +
            L_minu
        )

        return {
            "total": total_loss,
            "orientation": L_ori,
            "ridge": L_ridge,
            "gradient": L_grad,
            "minutia": L_minu,
            "m1": L_m1,
            "m1_candidate": L_m1_candidate,
            "m2": L_m2,
            "m3": L_m3,
            "m4": L_m4,
        }
        

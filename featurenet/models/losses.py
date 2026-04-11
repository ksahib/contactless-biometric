import torch
import torch.nn as nn
import torch.nn.functional as F

MINUTIA_ORIENTATION_BINS = 360
MINUTIA_XY_BINS = 8

class OrientationLoss(nn.Module):
    def __init__(self, num_bins=180, alpha=1.0, eps=1e-8):
        super().__init__()
        self.num_bins = num_bins
        self.alpha = alpha
        self.eps = eps

        kernel = torch.ones(1, 1, 3, 3) / 9.0
        self.register_buffer("smooth_kernel", kernel)

    def forward(self, pred, target, mask):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.float()

        pred_prob = torch.softmax(pred, dim=1)

        ce = -(
            target * torch.log(pred_prob + self.eps) +
            (1 - target) * torch.log(1 - pred_prob + self.eps)
        ).sum(dim=1, keepdim=True)

        ce = (ce * mask).sum() / (mask.sum() + self.eps)

        # angles
        angles = torch.arange(self.num_bins, device=pred.device, dtype=pred.dtype) * (360.0 / self.num_bins)
        rads = angles * torch.pi / 180.0

        cosine = torch.cos(2 * rads).view(1, self.num_bins, 1, 1)
        sine   = torch.sin(2 * rads).view(1, self.num_bins, 1, 1)

        dcos = (pred_prob * cosine).sum(dim=1, keepdim=True) / self.num_bins
        dsin = (pred_prob * sine).sum(dim=1, keepdim=True) / self.num_bins

        dcos_s = F.conv2d(dcos, self.smooth_kernel, padding=1)
        dsin_s = F.conv2d(dsin, self.smooth_kernel, padding=1)

        magnitude = torch.sqrt(dcos_s.pow(2) + dsin_s.pow(2) + self.eps)

        coh = ((magnitude - 1.0) * mask).sum() / (mask.sum() + self.eps)

        loss = ce + self.alpha * coh
        return loss

class RidgePeriodLoss(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target, mask):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.float()
        mse = ((pred - target) ** 2)
        loss_mse = (mse * mask).sum() / mask.sum()
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
        mu_score=120.0,
        mu_x=20.0,
        mu_y=20.0,
        mu_ori=5.0,
        m1_focal_gamma=2.0,
        m1_pos_weight_max=100.0,
        m1_hard_neg_enable=True,
        m1_hard_neg_ratio=20.0,
        m1_hard_neg_min=2000,
        m1_hard_neg_fraction=0.05,
        xy_soft_target_sigma=1.0,
    ):
        super().__init__()

        # sub-losses
        self.orientation_loss = OrientationLoss(alpha=alpha)
        self.ridge_loss = RidgePeriodLoss(beta=beta)
        self.gradient_loss = GradientLoss(gamma=gamma, sigma=sigma)

        # weights
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
        self.xy_soft_target_sigma = float(xy_soft_target_sigma)
        if self.xy_soft_target_sigma <= 0.0:
            raise ValueError("xy_soft_target_sigma must be positive")

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
        minutia_mask = targets.get("minutia_valid_mask", score_mask)
        if minutia_mask.dim() == 3:
            minutia_mask = minutia_mask.unsqueeze(1)
        minutia_mask = minutia_mask.float() * score_mask.float()
        if float(minutia_mask.sum().item()) <= 0.0:
            return score_mask.float()
        return minutia_mask

    def _select_hard_negative_mask(self, focal_map, target_score, valid_mask):
        if not self.m1_hard_neg_enable:
            return valid_mask

        selected = torch.zeros_like(valid_mask, dtype=torch.bool)
        batch_size = focal_map.shape[0]
        for batch_index in range(batch_size):
            valid_b = valid_mask[batch_index, 0]
            if not bool(valid_b.any().item()):
                continue
            target_b = target_score[batch_index, 0]
            focal_b = focal_map[batch_index, 0]
            positive_b = (target_b > 0.5) & valid_b
            negative_b = (~positive_b) & valid_b

            selected_b = positive_b.clone()
            negative_count = int(negative_b.sum().item())
            positive_count = int(positive_b.sum().item())
            if negative_count > 0:
                ratio_k = int(positive_count * self.m1_hard_neg_ratio)
                fraction_k = int(negative_count * self.m1_hard_neg_fraction)
                hard_k = max(ratio_k, self.m1_hard_neg_min, fraction_k)
                hard_k = min(hard_k, negative_count)
                if hard_k > 0:
                    negative_scores = focal_b[negative_b]
                    negative_indices = torch.nonzero(negative_b, as_tuple=False)
                    if hard_k < negative_count:
                        topk_indices = torch.topk(negative_scores, k=hard_k, sorted=False).indices
                        chosen_indices = negative_indices[topk_indices]
                    else:
                        chosen_indices = negative_indices
                    selected_b[chosen_indices[:, 0], chosen_indices[:, 1]] = True
            selected[batch_index, 0] = selected_b

        if float(selected.float().sum().item()) <= 0.0:
            return valid_mask
        return selected

    def _compute_m1_score_loss(self, logits, target_score, score_mask):
        eps = 1e-8
        if target_score.dim() == 3:
            target_score = target_score.unsqueeze(1)
        valid = score_mask > 0.5
        positive = ((target_score > 0.5) & valid).float().sum()
        negative = ((target_score <= 0.5) & valid).float().sum()
        pos_weight = torch.sqrt(negative / (positive + eps)).clamp(min=1.0, max=self.m1_pos_weight_max)
        bce_map = F.binary_cross_entropy_with_logits(
            logits,
            target_score,
            reduction="none",
            pos_weight=pos_weight,
        )
        pt = torch.exp(-bce_map)
        focal_map = torch.pow((1.0 - pt).clamp(min=0.0), self.m1_focal_gamma) * bce_map
        selected_mask = self._select_hard_negative_mask(
            focal_map=focal_map,
            target_score=target_score,
            valid_mask=valid,
        )
        selected_mask_float = selected_mask.float()
        return (focal_map * selected_mask_float).sum() / (selected_mask_float.sum() + eps)

    def _ensure_xy_target_bins(self, target_bins: torch.Tensor, head_name: str) -> torch.Tensor:
        if target_bins.dim() == 4 and target_bins.shape[1] == 1:
            target_bins = target_bins.squeeze(1)
        if target_bins.dim() != 3:
            raise ValueError(
                f"expected {head_name} targets with shape [B,H,W] or [B,1,H,W], got {tuple(target_bins.shape)}"
            )
        return target_bins

    def _validate_xy_bins_on_active_cells(
        self,
        target_bins: torch.Tensor,
        active_mask: torch.Tensor,
        head_name: str,
    ) -> None:
        invalid = ((target_bins < 0) | (target_bins >= MINUTIA_XY_BINS)) & active_mask
        if not bool(invalid.any().item()):
            return
        invalid_values = target_bins[invalid]
        min_invalid = int(invalid_values.min().item())
        max_invalid = int(invalid_values.max().item())
        raise ValueError(
            f"{head_name} targets contain out-of-range labels on active cells; expected [0, {MINUTIA_XY_BINS - 1}], "
            f"got min={min_invalid}, max={max_invalid}"
        )

    def _gaussian_soft_targets_from_bins(self, target_bins: torch.Tensor) -> torch.Tensor:
        # Ordered (non-circular) bin smoothing over 8 x/y bins.
        bin_positions = torch.arange(
            MINUTIA_XY_BINS,
            device=target_bins.device,
            dtype=torch.float32,
        ).view(1, MINUTIA_XY_BINS, 1, 1)
        centers = target_bins.float().unsqueeze(1)
        weights = torch.exp(-0.5 * ((bin_positions - centers) / self.xy_soft_target_sigma).pow(2))
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def _compute_xy_soft_ce_loss(
        self,
        logits: torch.Tensor,
        target_bins: torch.Tensor,
        minutia_mask: torch.Tensor,
        head_name: str,
    ) -> torch.Tensor:
        if logits.dim() != 4 or logits.shape[1] != MINUTIA_XY_BINS:
            raise ValueError(
                f"expected {head_name} logits with shape [B,{MINUTIA_XY_BINS},H,W], got {tuple(logits.shape)}"
            )
        target_bins = self._ensure_xy_target_bins(target_bins, head_name=head_name)
        active_mask = (minutia_mask > 0.5).squeeze(1)
        self._validate_xy_bins_on_active_cells(target_bins, active_mask, head_name=head_name)

        soft_targets = self._gaussian_soft_targets_from_bins(target_bins).to(dtype=logits.dtype)
        log_probs = F.log_softmax(logits, dim=1)
        soft_ce_map = -(soft_targets * log_probs).sum(dim=1, keepdim=True)
        return (soft_ce_map * minutia_mask).sum() / (minutia_mask.sum() + 1e-8)

    def forward(self, outputs, targets):
        mask = targets["mask"]
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        score_mask = self._resolve_score_mask(mask)
        minutia_mask = self._resolve_minutia_mask(targets, score_mask)

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

        # --- M1: score (binary)
        L_m1 = self._compute_m1_score_loss(
            outputs["minutia_score"],
            targets["minutia_score"],
            score_mask,
        )

        # --- M2: x (Gaussian soft-target CE over ordered 8 bins)
        L_m2 = self._compute_xy_soft_ce_loss(
            outputs["minutia_x"],
            targets["minutia_x"],
            minutia_mask=minutia_mask,
            head_name="minutia_x",
        )

        # --- M3: y (Gaussian soft-target CE over ordered 8 bins)
        L_m3 = self._compute_xy_soft_ce_loss(
            outputs["minutia_y"],
            targets["minutia_y"],
            minutia_mask=minutia_mask,
            head_name="minutia_y",
        )

        # --- M4: orientation (continuous cos/sin)
        pred_ori = outputs["minutia_orientation"]
        if pred_ori.dim() != 4 or pred_ori.shape[1] != 2:
            raise ValueError(
                f"expected minutia_orientation output with shape [B,2,H,W], got {tuple(pred_ori.shape)}"
            )
        pred_ori = F.normalize(pred_ori, dim=1, eps=1e-8)
        target_ori = self._resolve_minutia_orientation_target_vectors(targets).to(pred_ori.device)
        if target_ori.shape != pred_ori.shape:
            raise ValueError(
                f"target minutia orientation vector shape {tuple(target_ori.shape)} "
                f"does not match prediction shape {tuple(pred_ori.shape)}"
            )
        ori_loss_map = (pred_ori - target_ori).pow(2).sum(dim=1, keepdim=True)
        L_m4 = (ori_loss_map * minutia_mask).sum() / (minutia_mask.sum() + 1e-8)

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
        total_loss = L_ori + L_ridge + L_grad + L_minu

        return {
            "total": total_loss,
            "orientation": L_ori,
            "ridge": L_ridge,
            "gradient": L_grad,
            "minutia": L_minu,
            "m1": L_m1,
            "m2": L_m2,
            "m3": L_m3,
            "m4": L_m4
        }
        

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        mu_ori=20.0,
        m1_focal_gamma=2.0,
        m1_pos_weight_max=30.0,
        m1_hard_neg_enable=True,
        m1_hard_neg_ratio=3.0,
        m1_hard_neg_min=128,
    ):
        super().__init__()

        # sub-losses
        self.orientation_loss = OrientationLoss(alpha=alpha)
        self.ridge_loss = RidgePeriodLoss(beta=beta)
        self.gradient_loss = GradientLoss(gamma=gamma, sigma=sigma)

        # minutiae losses
        self.ce = nn.CrossEntropyLoss(reduction='none')

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
                if positive_count > 0:
                    hard_k = max(int(positive_count * self.m1_hard_neg_ratio), self.m1_hard_neg_min)
                else:
                    hard_k = self.m1_hard_neg_min
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
        pos_weight = (negative / (positive + eps)).clamp(min=1.0, max=self.m1_pos_weight_max)
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

        # --- M2: x
        x_loss_map = self.ce(outputs["minutia_x"], targets["minutia_x"])
        x_loss_map = x_loss_map.unsqueeze(1)
        L_m2 = (x_loss_map * minutia_mask).sum() / (minutia_mask.sum() + 1e-8)

        # --- M3: y
        y_loss_map = self.ce(outputs["minutia_y"], targets["minutia_y"])
        y_loss_map = y_loss_map.unsqueeze(1)
        L_m3 = (y_loss_map * minutia_mask).sum() / (minutia_mask.sum() + 1e-8)

        # --- M4: orientation
        ori_loss_map = self.ce(
            outputs["minutia_orientation"],
            targets["minutia_orientation"]
        )
        ori_loss_map = ori_loss_map.unsqueeze(1)
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
        

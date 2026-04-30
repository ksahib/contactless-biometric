import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # branch 1
        self.branch1 = nn.Sequential(
            ConvBlock(2, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Ridge branch stems
        self.branch_stem_ridge = nn.Sequential(
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
        )

        self.branch_stem_orient = nn.Sequential(
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
        )

        # branch 2 (minutiae branch), exposed stages for /4 and /8 localization features
        self.branch2_conv1 = ConvBlock(2, 64, kernel_size=9, stride=1, padding=4)
        self.branch2_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.branch2_conv2 = ConvBlock(64, 128, kernel_size=5, stride=1, padding=2)
        self.branch2_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.branch2_conv3 = ConvBlock(128, 256, kernel_size=3, stride=1, padding=1)
        self.branch2_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ridge_conv = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        self.gradient_conv = nn.Sequential(
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0),
        )
        self.orientation_conv = nn.Conv2d(256, 180, kernel_size=1, stride=1, padding=0)

        self.minuiae_orient_head = nn.Sequential(
            ConvBlock(512, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0),
        )

        # score head semantics remain unchanged (deep /8 feature map)
        self.minutiae_score_head = nn.Sequential(
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0),
        )

        # x/y localization fusion path: /4 shallow + upsampled /8 deep -> /4 refine
        self.xy_fuse_conv1 = ConvBlock(384, 256, kernel_size=3, stride=1, padding=1)
        self.xy_fuse_conv2 = ConvBlock(256, 256, kernel_size=3, stride=1, padding=1)

        # Patch-aware /8 descriptor: preserve each 2x2 /4 neighborhood per /8 cell via pixel_unshuffle.
        self.xy_patch_refine1 = ConvBlock(1024, 256, kernel_size=1, stride=1, padding=0)
        self.xy_patch_refine2 = ConvBlock(256, 256, kernel_size=3, stride=1, padding=1)

        # continuous x/y offsets: one raw logit per /8 score cell
        self.minutia_head_x = nn.Sequential(
            ConvBlock(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0),
        )
        self.minutia_head_y = nn.Sequential(
            ConvBlock(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0),
        )

    @staticmethod
    def _pad_bottom_right_to_even(x: torch.Tensor) -> torch.Tensor:
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h))

    @staticmethod
    def _crop_to_spatial_shape(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        target_h, target_w = int(shape[0]), int(shape[1])
        if x.shape[-2] < target_h or x.shape[-1] < target_w:
            raise ValueError(
                f"cannot crop tensor with shape {tuple(x.shape[-2:])} to larger target {(target_h, target_w)}"
            )
        return x[:, :, :target_h, :target_w]

    def forward(self, x, mask=None):
        if mask is not None:
            image = x * mask
            x = torch.cat([image, mask], dim=1)
        else:
            raise ValueError("mask is required for feature extractor")

        if x.dim() != 4:
            raise ValueError(f"expected input with shape [B, 2, H, W], got {tuple(x.shape)}")
        if x.shape[1] != 2:
            raise ValueError(f"expected 2 input channels (masked image + mask), got {x.shape[1]}")

        # branch 2 staged features
        branch2_stage1 = self.branch2_conv1(x)
        branch2_stage1_pooled = self.branch2_pool1(branch2_stage1)
        branch2_stage2 = self.branch2_conv2(branch2_stage1_pooled)
        branch2_feat_4x = self.branch2_pool2(branch2_stage2)
        branch2_stage3 = self.branch2_conv3(branch2_feat_4x)
        branch2_feat_8x = self.branch2_pool3(branch2_stage3)

        # branch 1 features for orientation/ridge/gradient and minutia orientation
        x = self.branch1(x)
        ridge_interim = self.branch_stem_ridge(x)
        orient_interim = self.branch_stem_orient(x)

        ridge_period = self.ridge_conv(ridge_interim)
        grad = self.gradient_conv(ridge_interim)
        orient = self.orientation_conv(orient_interim)

        minu_orient_input = torch.cat([branch2_feat_8x, orient_interim], dim=1)
        minu_orient = self.minuiae_orient_head(minu_orient_input)

        # score remains on deep /8 features
        score_input = torch.cat([branch2_feat_8x, orient_interim, ridge_interim], dim=1)
        minu_score = self.minutiae_score_head(score_input)

        # x/y localization from fused /4 + /8 context.
        # We keep explicit 2x2 /4 geometry for each /8 cell using pixel_unshuffle.
        branch2_feat_8x_up = F.interpolate(
            branch2_feat_8x,
            size=branch2_feat_4x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        xy_fuse_input = torch.cat([branch2_feat_4x, branch2_feat_8x_up], dim=1)
        xy_feat_4x = self.xy_fuse_conv2(self.xy_fuse_conv1(xy_fuse_input))

        # Pad bottom/right if /4 spatial dims are odd, then regroup 2x2 /4 patches into channels.
        xy_feat_4x_padded = self._pad_bottom_right_to_even(xy_feat_4x)
        xy_patch_feat_8x = F.pixel_unshuffle(xy_feat_4x_padded, downscale_factor=2)
        xy_feat_8x = self.xy_patch_refine2(self.xy_patch_refine1(xy_patch_feat_8x))
        # Align x/y logits to the exact score-grid resolution.
        xy_feat_8x = self._crop_to_spatial_shape(xy_feat_8x, branch2_feat_8x.shape[-2:])

        minu_x = self.minutia_head_x(xy_feat_8x)
        minu_y = self.minutia_head_y(xy_feat_8x)

        return {
            "orientation": orient,
            "ridge_period": ridge_period,
            "gradient": grad,
            "minutia_orientation": minu_orient,
            "minutia_score": minu_score,
            "minutia_x": minu_x,
            "minutia_y": minu_y,
        }

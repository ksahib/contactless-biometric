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

        # In forward(), replace the minu_orient section:
        orient_at_4x = F.interpolate(
            orient_interim, 
            size=branch2_feat_4x.shape[-2:],
            mode="bilinear", 
            align_corners=False,
        )
        minu_orient_input = torch.cat([branch2_feat_4x, orient_at_4x], dim=1)  # 128+256=384ch
        minu_orient = self.minuiae_orient_head(minu_orient_input)
        # minu_orient is now at /4 resolution; downsample to /8 to match score head
        minu_orient = F.avg_pool2d(minu_orient, kernel_size=2, stride=2)
        minu_orient = self._crop_to_spatial_shape(minu_orient, branch2_feat_8x.shape[-2:])

        # score remains on deep /8 features
        score_4x = F.avg_pool2d(branch2_feat_4x, kernel_size=2, stride=2)  # /4 → /8
        score_4x = self._crop_to_spatial_shape(score_4x, branch2_feat_8x.shape[-2:])

        score_input = torch.cat([branch2_feat_8x, score_4x], dim=1)  # 256+128=384ch
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
        xy_context = torch.cat([
            xy_feat_8x,
            branch2_feat_8x,
            orient_interim,
            ridge_interim,
        ], dim=1)

        xy_feat_8x = self.xy_context_refine(xy_context)
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
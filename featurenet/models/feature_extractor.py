import torch
import torch.nn as nn
from .blocks import ConvBlock

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        #branch 1
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

        #Ridge branch stems
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


        #branch 2 (minuitae branch)
        self.branch2 = nn.Sequential(
            ConvBlock(2, 64, kernel_size=9, stride=1, padding=4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.ridge_conv = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        #self.gradient_conv = nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0)
        self.gradient_conv = nn.Sequential(
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0),
        )
        self.orientation_conv = nn.Conv2d(256, 180, kernel_size=1, stride=1, padding=0)

        self.minuiae_orient_head = nn.Sequential(
            ConvBlock(512, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 360, kernel_size=1, stride=1, padding=0),
        )

        self.minutiae_score_head = nn.Sequential(
            ConvBlock(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0),
        )

        self.minutia_head_x = nn.Sequential(
            ConvBlock(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 8, kernel_size=1, stride=1, padding=0),
        )

        self.minutia_head_y = nn.Sequential(
            ConvBlock(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 8, kernel_size=1, stride=1, padding=0),
        )

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

        x1 = self.branch2(x)
        x = self.branch1(x)
        ridge_interim = self.branch_stem_ridge(x)
        orient_interim = self.branch_stem_orient(x)
        ridge_period = self.ridge_conv(ridge_interim)
        grad = self.gradient_conv(ridge_interim)
        orient = self.orientation_conv(orient_interim)
        minu_orient_input = torch.cat([x1, orient_interim], dim=1)
        minu_orient = self.minuiae_orient_head(minu_orient_input)
        minu_score = self.minutiae_score_head(x1)
        minu_x = self.minutia_head_x(x1)
        minu_y = self.minutia_head_y(x1)

        return {
            "orientation": orient,
            "ridge_period": ridge_period,
            "gradient": grad,
            "minutia_orientation": minu_orient,
            "minutia_score": minu_score,
            "minutia_x": minu_x,
            "minutia_y": minu_y,
        }
        

import torch
import torch.nn as nn


def _resolve_group_count(num_channels: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(_resolve_group_count(out_channels), out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Project residual if channels or stride change
        self.residual_proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            if (in_channels != out_channels or stride != 1)
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual_proj(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x + residual)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_group_count(num_channels: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


def _binomial_kernel_1d(kernel_size: int) -> torch.Tensor:
    coeffs = [1.0]
    for _ in range(kernel_size - 1):
        coeffs = [a + b for a, b in zip([0.0] + coeffs, coeffs + [0.0])]
    return torch.tensor(coeffs, dtype=torch.float32)


class BlurPool2d(nn.Module):
    """Anti-aliased downsampling via a fixed depthwise binomial low-pass filter.

    The blur kernel is a non-persistent buffer so it is excluded from the
    ``state_dict`` (keeps existing checkpoints loadable, no new learnable params).
    """

    def __init__(self, channels: int, stride: int = 2, kernel_size: int = 3):
        super().__init__()
        self.channels = int(channels)
        self.stride = int(stride)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        kernel_1d = _binomial_kernel_1d(self.kernel_size)
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel = kernel_2d.view(1, 1, self.kernel_size, self.kernel_size).repeat(self.channels, 1, 1, 1)
        self.register_buffer("kernel", kernel, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.kernel.to(dtype=x.dtype),
            stride=self.stride,
            padding=self.padding,
            groups=self.channels,
        )


class MaxBlurPool2d(nn.Module):
    """Max-pool densely (stride 1) then anti-aliased subsample (blur, stride 2).

    Output spatial size matches ``nn.MaxPool2d(kernel_size=2, stride=2)`` for both
    even and odd inputs, so downstream crop/fusion logic is unaffected.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.blur = BlurPool2d(channels, stride=2, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blur(self.pool(x))


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
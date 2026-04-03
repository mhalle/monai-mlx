"""
Building blocks for MONAI networks in MLX.

All modules use channels-last layout: (B, D, H, W, C).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .layers import get_activation, get_norm


# ---------------------------------------------------------------------------
# SegResNet blocks
# ---------------------------------------------------------------------------

class ConvOnly(nn.Module):
    """Plain Conv3d without norm or activation (MONAI's conv_only mode)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def __call__(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    """Pre-activation residual block for SegResNet.

    Architecture: norm1 -> act -> conv1 -> norm2 -> act -> conv2 -> + identity
    """

    def __init__(
        self,
        in_channels: int,
        norm: str | tuple = ("GROUP", {"num_groups": 8}),
        kernel_size: int = 3,
        act: str | tuple = ("RELU", {"inplace": True}),
    ):
        super().__init__()
        self.norm1 = get_norm(norm, in_channels)
        self.norm2 = get_norm(norm, in_channels)
        self.act = get_activation(act)
        self.conv1 = ConvOnly(in_channels, in_channels, kernel_size=kernel_size)
        self.conv2 = ConvOnly(in_channels, in_channels, kernel_size=kernel_size)

    def __call__(self, x):
        identity = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + identity


class Upsample3D(nn.Module):
    """Trilinear upsampling (non-trainable) for channels-last 3D data."""

    def __init__(self, scale_factor: int = 2, align_corners: bool = False):
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=(scale_factor, scale_factor, scale_factor),
            mode="linear",
            align_corners=align_corners,
        )

    def __call__(self, x):
        return self.up(x)

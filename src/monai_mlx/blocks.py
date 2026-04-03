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

def _same_padding(kernel_size):
    """Compute 'same' padding for scalar or tuple kernel_size."""
    if isinstance(kernel_size, (list, tuple)):
        return tuple((k - 1) // 2 for k in kernel_size)
    return (kernel_size - 1) // 2


class ConvOnly(nn.Module):
    """Plain Conv3d without norm or activation (MONAI's conv_only mode)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple = 3,
        stride: int | tuple = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=_same_padding(kernel_size),
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


# ---------------------------------------------------------------------------
# BasicUNet blocks
# ---------------------------------------------------------------------------

class ConvNormAct(nn.Module):
    """Conv3d -> Norm -> Activation (MONAI's Convolution with ADN)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
        norm: str | tuple = ("instance", {"affine": True}),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1}),
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,
        )
        self.norm = get_norm(norm, out_channels)
        self.act = get_activation(act)

    def __call__(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class TwoConv(nn.Module):
    """Two ConvNormAct blocks in sequence."""

    def __init__(self, in_channels: int, out_channels: int,
                 act: str | tuple, norm: str | tuple, bias: bool):
        super().__init__()
        self.conv_0 = ConvNormAct(in_channels, out_channels, bias=bias, norm=norm, act=act)
        self.conv_1 = ConvNormAct(out_channels, out_channels, bias=bias, norm=norm, act=act)

    def __call__(self, x):
        return self.conv_1(self.conv_0(x))


class Down(nn.Module):
    """MaxPool3d downsampling + TwoConv."""

    def __init__(self, in_channels: int, out_channels: int,
                 act: str | tuple, norm: str | tuple, bias: bool):
        super().__init__()
        self.convs = TwoConv(in_channels, out_channels, act, norm, bias)

    def __call__(self, x):
        # MaxPool3d 2x2x2: pad if odd, then pool via reshape
        B, D, H, W, C = x.shape
        # Pad to even dimensions if needed
        pd = D % 2
        ph = H % 2
        pw = W % 2
        if pd or ph or pw:
            x = mx.pad(x, [(0, 0), (0, pd), (0, ph), (0, pw), (0, 0)])
            B, D, H, W, C = x.shape
        # 2x2x2 max pooling via reshape
        x = x.reshape(B, D // 2, 2, H // 2, 2, W // 2, 2, C)
        x = mx.max(x, axis=(2, 4, 6))
        return self.convs(x)


class UpCat(nn.Module):
    """Upsample (deconv) + concatenate skip + TwoConv."""

    def __init__(self, in_channels: int, cat_channels: int, out_channels: int,
                 act: str | tuple, norm: str | tuple, bias: bool,
                 halves: bool = True):
        super().__init__()
        up_channels = in_channels // 2 if halves else in_channels
        self.deconv = nn.ConvTranspose3d(
            in_channels, up_channels, kernel_size=2, stride=2, padding=0, bias=True,
        )
        self.convs = TwoConv(cat_channels + up_channels, out_channels, act, norm, bias)

    def __call__(self, x, x_enc):
        x = self.deconv(x)
        # Pad if spatial dims don't match (odd pooling)
        if x_enc is not None:
            for i in range(1, 4):  # spatial dims (D, H, W)
                if x_enc.shape[i] != x.shape[i]:
                    pad_width = [(0, 0)] * 5
                    pad_width[i] = (0, 1)
                    x = mx.pad(x, pad_width, mode="edge")
            x = mx.concatenate([x_enc, x], axis=-1)
        return self.convs(x)


# ---------------------------------------------------------------------------
# Shared upsampling
# ---------------------------------------------------------------------------

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

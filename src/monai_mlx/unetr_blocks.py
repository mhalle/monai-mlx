"""
UNETR decoder blocks in MLX.

UnetResBlock, UnetrBasicBlock, UnetrUpBlock, UnetrPrUpBlock, UnetOutBlock.
All use channels-last (B, D, H, W, C) layout.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .blocks import ConvOnly
from .layers import get_activation, get_norm


class UnetResBlock(nn.Module):
    """Post-activation residual block: conv→norm→act→conv→norm + skip→act.

    Unlike SegResNet's pre-activation pattern, this uses post-activation
    (matching MONAI's dynunet_block.UnetResBlock).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm_name: str | tuple = "instance",
    ):
        super().__init__()
        self.conv1 = ConvOnly(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.conv2 = ConvOnly(out_channels, out_channels, kernel_size=kernel_size, stride=1)
        self.norm1 = get_norm(norm_name, out_channels)
        self.norm2 = get_norm(norm_name, out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)

        # Residual projection if channels or stride change
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = ConvOnly(in_channels, out_channels, kernel_size=1, stride=stride)
            self.norm3 = get_norm(norm_name, out_channels)

    def __call__(self, x):
        residual = x
        x = self.lrelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        if self.downsample is not None:
            residual = self.norm3(self.downsample(residual))
        return self.lrelu(x + residual)


class UnetBasicBlock(nn.Module):
    """Basic block without residual: conv→norm→act→conv→norm→act."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm_name: str | tuple = "instance",
    ):
        super().__init__()
        self.conv1 = ConvOnly(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.conv2 = ConvOnly(out_channels, out_channels, kernel_size=kernel_size, stride=1)
        self.norm1 = get_norm(norm_name, out_channels)
        self.norm2 = get_norm(norm_name, out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)

    def __call__(self, x):
        x = self.lrelu(self.norm1(self.conv1(x)))
        x = self.lrelu(self.norm2(self.conv2(x)))
        return x


class UnetrBasicBlock(nn.Module):
    """Wrapper that selects ResBlock or BasicBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm_name: str | tuple = "instance",
        res_block: bool = True,
    ):
        super().__init__()
        if res_block:
            self.layer = UnetResBlock(in_channels, out_channels, kernel_size, stride, norm_name)
        else:
            self.layer = UnetBasicBlock(in_channels, out_channels, kernel_size, stride, norm_name)

    def __call__(self, x):
        return self.layer(x)


class UnetrUpBlock(nn.Module):
    """ConvTranspose upsample → concat skip → ResBlock/BasicBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        upsample_kernel_size: int = 2,
        norm_name: str | tuple = "instance",
        res_block: bool = True,
    ):
        super().__init__()
        self.transp_conv = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=upsample_kernel_size, stride=upsample_kernel_size,
            padding=0, bias=False,
        )
        Block = UnetResBlock if res_block else UnetBasicBlock
        self.conv_block = Block(
            out_channels + out_channels, out_channels,
            kernel_size=kernel_size, stride=1, norm_name=norm_name,
        )

    def __call__(self, x, skip):
        x = self.transp_conv(x)
        x = mx.concatenate([x, skip], axis=-1)  # channels-last concat
        return self.conv_block(x)


class UnetrPrUpBlock(nn.Module):
    """Progressive upsampling: initial ConvTranspose + N more upsample stages."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layer: int,
        kernel_size: int = 3,
        stride: int = 1,
        upsample_kernel_size: int = 2,
        norm_name: str | tuple = "instance",
        res_block: bool = True,
    ):
        super().__init__()
        self.transp_conv_init = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=upsample_kernel_size, stride=upsample_kernel_size,
            padding=0, bias=False,
        )
        Block = UnetResBlock if res_block else UnetBasicBlock
        self.blocks = []
        for _ in range(num_layer):
            self.blocks.append([
                nn.ConvTranspose3d(
                    out_channels, out_channels,
                    kernel_size=upsample_kernel_size, stride=upsample_kernel_size,
                    padding=0, bias=False,
                ),
                Block(out_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, norm_name=norm_name),
            ])

    def __call__(self, x):
        x = self.transp_conv_init(x)
        for layers in self.blocks:
            for layer in layers:
                x = layer(x)
        return x


class UnetOutBlock(nn.Module):
    """Final 1x1 convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def __call__(self, x):
        return self.conv(x)

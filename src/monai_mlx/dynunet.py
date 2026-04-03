"""
DynUNet — Dynamic U-Net with configurable depth and kernel sizes.

Port of MONAI's DynUNet to MLX. Uses UnetBasicBlock/UnetResBlock
(same blocks as UNETR decoder). Flat encoder-decoder with
concatenation skip connections.

Similar to nnU-Net's PlainConvUNet but with MONAI's block structure.
All operations use channels-last (B, D, H, W, C) layout.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn

from .unetr_blocks import UnetBasicBlock, UnetResBlock, UnetOutBlock


def _to_tuple(val, ndim=3):
    if isinstance(val, (list, tuple)):
        return tuple(val)
    return (val,) * ndim


class DynUNetUpBlock(nn.Module):
    """ConvTranspose upsample → concat skip → UnetBasicBlock."""

    def __init__(self, in_ch, out_ch, kernel_size, upsample_kernel_size,
                 norm_name="instance", res_block=False):
        super().__init__()
        uks = _to_tuple(upsample_kernel_size)
        # kernel=stride, no padding needed for exact 2x upsample
        self.transp_conv = nn.ConvTranspose3d(
            in_ch, out_ch, kernel_size=uks, stride=uks, padding=0, bias=False,
        )
        Block = UnetResBlock if res_block else UnetBasicBlock
        self.conv_block = Block(
            out_ch * 2, out_ch, kernel_size=kernel_size,
            stride=1, norm_name=norm_name,
        )

    def __call__(self, x, skip):
        x = self.transp_conv(x)
        x = mx.concatenate([x, skip], axis=-1)
        return self.conv_block(x)


class DynUNet(nn.Module):
    """Dynamic U-Net matching MONAI's DynUNet architecture.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : list of list/tuple
        Convolution kernel sizes per level.
    strides : list of list/tuple
        Convolution strides per level.
    upsample_kernel_size : list of list/tuple
        Transposed convolution kernel sizes for decoder.
    filters : list of int, optional
        Feature channels per level. If None, auto-computed.
    norm_name : str or tuple
        Normalization. Defaults to "instance".
    res_block : bool
        Use residual blocks. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence,
        strides: Sequence,
        upsample_kernel_size: Sequence,
        filters: Sequence[int] | None = None,
        norm_name: str | tuple = ("instance", {"affine": True}),
        res_block: bool = False,
    ):
        super().__init__()
        n_levels = len(kernel_size)

        if filters is None:
            filters = [min(2 ** (5 + i), 320) for i in range(n_levels)]

        Block = UnetResBlock if res_block else UnetBasicBlock

        # Input block uses strides[0] (often [1,1,1] but not always)
        self.input_block = Block(
            in_channels, filters[0],
            kernel_size=_to_tuple(kernel_size[0]),
            stride=_to_tuple(strides[0]),
            norm_name=norm_name,
        )

        # Encoder
        self.downsamples = []
        for i in range(1, n_levels - 1):
            self.downsamples.append(
                Block(filters[i - 1], filters[i],
                      kernel_size=_to_tuple(kernel_size[i]),
                      stride=_to_tuple(strides[i]),
                      norm_name=norm_name)
            )

        # Bottleneck
        self.bottleneck = Block(
            filters[-2], filters[-1],
            kernel_size=_to_tuple(kernel_size[-1]),
            stride=_to_tuple(strides[-1]),
            norm_name=norm_name,
        )

        # Decoder: MONAI uses kernel_size[1:][::-1] and upsample_kernel_size[::-1]
        up_kernels = list(kernel_size[1:])[::-1]
        up_uks = list(upsample_kernel_size)[::-1]
        up_filters_in = list(filters[1:])[::-1]
        up_filters_out = list(filters[:-1])[::-1]

        self.upsamples = []
        for i in range(len(up_kernels)):
            self.upsamples.append(
                DynUNetUpBlock(up_filters_in[i], up_filters_out[i],
                               kernel_size=_to_tuple(up_kernels[i]),
                               upsample_kernel_size=up_uks[i],
                               norm_name=norm_name, res_block=res_block)
            )

        # Output
        self.output_block = UnetOutBlock(filters[0], out_channels)

    def __call__(self, x):
        # Input
        x = self.input_block(x)
        skips = [x]

        # Encode
        for down in self.downsamples:
            x = down(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode
        for i, up in enumerate(self.upsamples):
            skip = skips[-(i + 1)]
            x = up(x, skip)

        return self.output_block(x)

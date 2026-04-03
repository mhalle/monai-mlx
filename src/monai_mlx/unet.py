"""
MONAI UNet — Enhanced U-Net with ResidualUnit blocks.

Port of MONAI's UNet to MLX. Mirrors the PyTorch module hierarchy exactly
to enable direct weight loading without complex key remapping.

The PyTorch UNet uses recursive SkipConnection nesting. We flatten this
into explicit encoder/decoder paths but keep the internal ResidualUnit
structure matching PyTorch's key names (conv.unit0, conv.unit1, residual).

All operations use channels-last (B, D, H, W, C) layout.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn

from .layers import get_activation, get_norm


class _ConvADN(nn.Module):
    """Single Conv + Norm + Act (matches MONAI's Convolution with ADN)."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, bias=True,
                 norm="instance", act="prelu", conv_only=False, is_transposed=False):
        super().__init__()
        padding = (kernel_size - 1) // 2
        if is_transposed:
            self.conv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size,
                                            stride=stride, padding=padding, bias=bias)
        else:
            self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size,
                                   stride=stride, padding=padding, bias=bias)
        if conv_only:
            self.adn = None
        else:
            self.adn = _ADN(out_ch, norm=norm, act=act)

    def __call__(self, x):
        x = self.conv(x)
        if self.adn is not None:
            x = self.adn(x)
        return x


class _ADN(nn.Module):
    """Norm + Act (matches MONAI's ADN with NDA ordering)."""

    def __init__(self, channels, norm="instance", act="prelu"):
        super().__init__()
        self.N = get_norm(norm, channels)
        self.A = get_activation(act)

    def __call__(self, x):
        x = self.N(x)
        if self.A is not None:
            x = self.A(x)
        return x


class _ResidualUnit(nn.Module):
    """ResidualUnit matching MONAI's key structure exactly.

    Keys: conv.unit0.conv.weight, conv.unit0.adn.N.weight, conv.unit0.adn.A.weight,
          conv.unit1.conv.weight, ..., residual.weight
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, subunits=2,
                 act="prelu", norm="instance", bias=True, last_conv_only=False):
        super().__init__()
        # Build units as named dict to match PyTorch's Sequential naming
        self.conv = {}
        ch = in_ch
        for i in range(subunits):
            is_last = (i == subunits - 1)
            s = stride if i == 0 else 1
            conv_only = is_last and last_conv_only
            self.conv[f"unit{i}"] = _ConvADN(ch, out_ch, kernel_size, s, bias,
                                              norm, act, conv_only=conv_only)
            ch = out_ch

        # Residual projection
        if in_ch != out_ch or stride != 1:
            padding = (kernel_size - 1) // 2
            self.residual = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=bias)
        else:
            self.residual = None

    def __call__(self, x):
        res = x
        if self.residual is not None:
            res = self.residual(res)
        for unit in self.conv.values():
            x = unit(x)
        return x + res


class UNet(nn.Module):
    """MONAI UNet with ResidualUnit blocks.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    channels : tuple
        Feature channels per level.
    strides : tuple
        Stride per level (length = len(channels) - 1).
    kernel_size : int
        Convolution kernel size. Defaults to 3.
    up_kernel_size : int
        Transposed convolution kernel size. Defaults to 3.
    num_res_units : int
        Number of residual sub-units per block. 0 = plain convolutions.
    act : str or tuple
        Activation. Defaults to PReLU.
    norm : str or tuple
        Normalization. Defaults to InstanceNorm.
    bias : bool
        Whether conv layers have bias. Defaults to True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: int = 3,
        up_kernel_size: int = 3,
        num_res_units: int = 0,
        act: str | tuple = "prelu",
        norm: str | tuple = "instance",
        bias: bool = True,
    ):
        super().__init__()
        self.n_levels = len(channels)

        # Encoder
        self.down_blocks = []
        ch_in = in_channels
        for i in range(self.n_levels - 1):
            if num_res_units > 0:
                block = _ResidualUnit(ch_in, channels[i], kernel_size,
                                       stride=strides[i], subunits=num_res_units,
                                       act=act, norm=norm, bias=bias)
            else:
                block = _ConvADN(ch_in, channels[i], kernel_size,
                                  stride=strides[i], bias=bias, norm=norm, act=act)
            self.down_blocks.append(block)
            ch_in = channels[i]

        # Bottom
        if num_res_units > 0:
            self.bottom = _ResidualUnit(channels[-2], channels[-1], kernel_size,
                                         stride=1, subunits=num_res_units,
                                         act=act, norm=norm, bias=bias)
        else:
            self.bottom = _ConvADN(channels[-2], channels[-1], kernel_size,
                                    stride=1, bias=bias, norm=norm, act=act)

        # Decoder
        self.up_paths = []
        for i in range(self.n_levels - 2, -1, -1):
            is_top = (i == 0)
            up_out = out_channels if is_top else channels[i]

            if i == self.n_levels - 2:
                up_in_ch = channels[-1] + channels[i]  # bottom + skip
            else:
                up_in_ch = channels[i + 1] + channels[i]  # prev decoder + skip

            # Transposed conv to upsample (takes just the decoded features, not concatenated)
            transp_in = channels[-1] if i == self.n_levels - 2 else channels[i + 1]

            conv_only = is_top and num_res_units == 0
            transp = _ConvADN(transp_in, transp_in, up_kernel_size,
                               stride=strides[i], bias=bias, norm=norm, act=act,
                               conv_only=conv_only, is_transposed=True)

            if num_res_units > 0:
                ru = _ResidualUnit(transp_in, up_out, kernel_size, stride=1,
                                    subunits=1, act=act, norm=norm, bias=bias,
                                    last_conv_only=is_top)
                self.up_paths.append({"transp": transp, "ru": ru})
            else:
                self.up_paths.append({"transp": transp, "ru": None})

    def __call__(self, x):
        # Encode
        skips = []
        for block in self.down_blocks:
            x = block(x)
            skips.append(x)

        # Bottom
        x = self.bottom(x)

        # Decode
        for i, up_path in enumerate(self.up_paths):
            skip = skips[-(i + 1)]
            x = up_path["transp"](x)
            x = mx.concatenate([x, skip], axis=-1)
            if up_path["ru"] is not None:
                x = up_path["ru"](x)

        return x

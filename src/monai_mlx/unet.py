"""
MONAI UNet — Enhanced U-Net with ResidualUnit blocks.

Port of MONAI's UNet to MLX. The key architectural difference from
standard UNet: concatenation happens BEFORE the transposed convolution,
so the up path handles both channel reduction and upsampling.

Flow: down → cat(skip, deeper) → transposed_conv_upsample → ResUnit

All operations use channels-last (B, D, H, W, C) layout.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn

from .layers import get_activation, get_norm


class _ConvADN(nn.Module):
    """Conv + optional (Norm + Act). Matches MONAI Convolution with ADN."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, bias=True,
                 norm="instance", act="prelu", conv_only=False, is_transposed=False):
        super().__init__()
        padding = (kernel_size - 1) // 2
        if is_transposed:
            # output_padding ensures exact 2x upsampling (matches MONAI)
            out_padding = tuple(s - 1 for s in (stride if isinstance(stride, tuple) else (stride,) * 3))
            self.conv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size,
                                            stride=stride, padding=padding,
                                            output_padding=out_padding, bias=bias)
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
    """Norm + Act (NDA ordering)."""

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
    """ResidualUnit with named sub-units matching MONAI's key structure.

    Keys: conv.unit0.conv.weight, conv.unit0.adn.N.weight, conv.unit0.adn.A.weight,
          conv.unit1.{...}, residual.weight
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, subunits=2,
                 act="prelu", norm="instance", bias=True, last_conv_only=False):
        super().__init__()
        self.conv = {}
        ch = in_ch
        for i in range(subunits):
            is_last = (i == subunits - 1)
            s = stride if i == 0 else 1
            self.conv[f"unit{i}"] = _ConvADN(
                ch, out_ch, kernel_size, s, bias, norm, act,
                conv_only=(is_last and last_conv_only),
            )
            ch = out_ch

        if in_ch != out_ch or stride != 1:
            # MONAI uses kernel_size=kernel_size when stride>1, kernel_size=1 when stride=1
            res_ks = kernel_size if stride != 1 else 1
            res_pad = (res_ks - 1) // 2
            self.residual = nn.Conv3d(in_ch, out_ch, kernel_size=res_ks,
                                       stride=stride, padding=res_pad, bias=bias)
        else:
            self.residual = None

    def __call__(self, x):
        res = self.residual(x) if self.residual is not None else x
        for unit in self.conv.values():
            x = unit(x)
        return x + res


class UNet(nn.Module):
    """MONAI UNet with ResidualUnit blocks.

    Parameters
    ----------
    in_channels, out_channels, channels, strides, kernel_size, up_kernel_size,
    num_res_units, act, norm, bias: see MONAI UNet documentation.
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
        n = self.n_levels

        # Encoder: down_blocks[i] takes stride[i] to downsample
        self.down_blocks = []
        ch_in = in_channels
        for i in range(n - 1):
            if num_res_units > 0:
                self.down_blocks.append(
                    _ResidualUnit(ch_in, channels[i], kernel_size, stride=strides[i],
                                   subunits=num_res_units, act=act, norm=norm, bias=bias))
            else:
                self.down_blocks.append(
                    _ConvADN(ch_in, channels[i], kernel_size, stride=strides[i],
                              bias=bias, norm=norm, act=act))
            ch_in = channels[i]

        # Bottom: no stride
        if num_res_units > 0:
            self.bottom = _ResidualUnit(channels[-2], channels[-1], kernel_size, stride=1,
                                         subunits=num_res_units, act=act, norm=norm, bias=bias)
        else:
            self.bottom = _ConvADN(channels[-2], channels[-1], kernel_size, stride=1,
                                    bias=bias, norm=norm, act=act)

        # Decoder: up_paths[i] processes concatenated (skip + deeper) features
        # Concatenation happens BEFORE the transposed conv in MONAI UNet
        self.up_paths = []
        for i in range(n - 2, -1, -1):
            is_top = (i == 0)
            # Output channels: match the level above (i-1) so SkipConnection can cat
            up_out = out_channels if is_top else channels[i - 1]

            # Input channels after concatenation
            if i == n - 2:
                upc = channels[i] + channels[-1]  # skip + bottom output
            else:
                upc = channels[i] + channels[i]  # skip + decoded from below (both channels[i])

            # Transposed conv: upc → up_out, stride to upsample
            transp = _ConvADN(upc, up_out, up_kernel_size, stride=strides[i],
                               bias=bias, norm=norm, act=act,
                               conv_only=(is_top and num_res_units == 0),
                               is_transposed=True)

            if num_res_units > 0:
                ru = _ResidualUnit(up_out, up_out, kernel_size, stride=1,
                                    subunits=1, act=act, norm=norm, bias=bias,
                                    last_conv_only=is_top)
                self.up_paths.append([transp, ru])
            else:
                self.up_paths.append([transp])

    def __call__(self, x):
        # Encode
        skips = []
        for block in self.down_blocks:
            x = block(x)
            skips.append(x)

        # Bottom
        x = self.bottom(x)

        # Decode: cat(skip, x) → up_path
        for i, up_layers in enumerate(self.up_paths):
            skip = skips[-(i + 1)]
            x = mx.concatenate([skip, x], axis=-1)  # cat along channels
            for layer in up_layers:
                x = layer(x)

        return x

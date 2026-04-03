"""
SegResNet — residual encoder-decoder for 3D medical image segmentation.

Port of MONAI's SegResNet to MLX. Uses pre-activation residual blocks
with GroupNorm, additive skip connections, and trilinear upsampling.

All operations use channels-last (B, D, H, W, C) layout.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .blocks import ConvOnly, ResBlock, Upsample3D
from .layers import get_activation, get_norm


class SegResNet(nn.Module):
    """SegResNet based on '3D MRI brain tumor segmentation using
    autoencoder regularization' (https://arxiv.org/pdf/1810.11654.pdf).

    Parameters
    ----------
    init_filters : int
        Number of output channels for initial convolution. Defaults to 8.
    in_channels : int
        Number of input channels. Defaults to 1.
    out_channels : int
        Number of output channels. Defaults to 2.
    act : str or tuple
        Activation type. Defaults to ("RELU", {}).
    norm : str or tuple
        Normalization type. Defaults to ("GROUP", {"num_groups": 8}).
    use_conv_final : bool
        Whether to add final conv block. Defaults to True.
    blocks_down : tuple
        Number of ResBlocks per encoder stage. Defaults to (1, 2, 2, 4).
    blocks_up : tuple
        Number of ResBlocks per decoder stage. Defaults to (1, 1, 1).
    """

    def __init__(
        self,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        act: str | tuple = ("RELU", {"inplace": True}),
        norm: str | tuple = ("GROUP", {"num_groups": 8}),
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
    ):
        super().__init__()
        self.use_conv_final = use_conv_final
        self.act_spec = act
        self.norm_spec = norm

        # Initial conv
        self.convInit = ConvOnly(in_channels, init_filters, kernel_size=3)

        # Encoder: each stage has optional stride-2 downconv + N ResBlocks
        self.down_layers = []
        for i, n_blocks in enumerate(blocks_down):
            ch = init_filters * 2 ** i
            stage = []
            if i > 0:
                # Stride-2 conv for downsampling
                stage.append(ConvOnly(ch // 2, ch, kernel_size=3, stride=2))
            for _ in range(n_blocks):
                stage.append(ResBlock(ch, norm=norm, act=act))
            self.down_layers.append(stage)

        # Decoder: upsample + 1x1 conv + additive skip + ResBlocks
        n_up = len(blocks_up)
        self.up_samples = []
        self.up_layers = []
        for i in range(n_up):
            ch_in = init_filters * 2 ** (n_up - i)
            ch_out = ch_in // 2
            # 1x1 conv to reduce channels, then upsample
            self.up_samples.append([
                ConvOnly(ch_in, ch_out, kernel_size=1),
                Upsample3D(scale_factor=2),
            ])
            stage = []
            for _ in range(blocks_up[i]):
                stage.append(ResBlock(ch_out, norm=norm, act=act))
            self.up_layers.append(stage)

        # Final conv: norm -> act -> 1x1 conv
        if use_conv_final:
            self.final_norm = get_norm(norm, init_filters)
            self.final_act = get_activation(act)
            self.final_conv = ConvOnly(init_filters, out_channels, kernel_size=1, bias=True)

    def __call__(self, x):
        # Encode
        x = self.convInit(x)

        down_x = []
        for stage in self.down_layers:
            for layer in stage:
                x = layer(x)
            down_x.append(x)

        # Decode (reversed skip connections)
        down_x.reverse()
        for i, (sample_layers, up_stage) in enumerate(zip(self.up_samples, self.up_layers)):
            # Upsample current resolution
            for layer in sample_layers:
                x = layer(x)
            # Add skip connection (additive, not concat)
            x = x + down_x[i + 1]
            # Process
            for layer in up_stage:
                x = layer(x)

        # Final
        if self.use_conv_final:
            x = self.final_norm(x)
            x = self.final_act(x)
            x = self.final_conv(x)

        return x

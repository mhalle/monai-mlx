"""
UNETR — Transformer encoder + CNN decoder for 3D segmentation.

Port of MONAI's UNETR to MLX. Uses ViT encoder with skip connections
from intermediate transformer layers feeding into a CNN decoder.

All operations use channels-last (B, D, H, W, C) layout.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .transformer import ViT
from .unetr_blocks import UnetrBasicBlock, UnetrUpBlock, UnetrPrUpBlock, UnetOutBlock


class UNETR(nn.Module):
    """UNETR based on 'Hatamizadeh et al., UNETR: Transformers for 3D
    Medical Image Segmentation'.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    img_size : tuple
        Input image dimensions (D, H, W).
    feature_size : int
        Base feature size for decoder. Defaults to 16.
    hidden_size : int
        ViT hidden dimension. Defaults to 768.
    mlp_dim : int
        ViT MLP dimension. Defaults to 3072.
    num_heads : int
        Number of attention heads. Defaults to 12.
    norm_name : str or tuple
        Normalization for decoder blocks. Defaults to "instance".
    res_block : bool
        Use residual blocks in decoder. Defaults to True.
    qkv_bias : bool
        Bias in QKV projection. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: tuple[int, ...],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        norm_name: str | tuple = "instance",
        res_block: bool = True,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.num_layers = 12
        patch_size = (16, 16, 16)
        self.feat_size = tuple(i // p for i, p in zip(img_size, patch_size))
        self.hidden_size = hidden_size

        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        # Encoder skip connections from raw input and ViT hidden states
        self.encoder1 = UnetrBasicBlock(
            in_channels, feature_size, kernel_size=3, stride=1,
            norm_name=norm_name, res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            hidden_size, feature_size * 2, num_layer=2,
            kernel_size=3, stride=1, upsample_kernel_size=2,
            norm_name=norm_name, res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            hidden_size, feature_size * 4, num_layer=1,
            kernel_size=3, stride=1, upsample_kernel_size=2,
            norm_name=norm_name, res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            hidden_size, feature_size * 8, num_layer=0,
            kernel_size=3, stride=1, upsample_kernel_size=2,
            norm_name=norm_name, res_block=res_block,
        )

        # Decoder
        self.decoder5 = UnetrUpBlock(
            hidden_size, feature_size * 8, kernel_size=3,
            upsample_kernel_size=2, norm_name=norm_name, res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            feature_size * 8, feature_size * 4, kernel_size=3,
            upsample_kernel_size=2, norm_name=norm_name, res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            feature_size * 4, feature_size * 2, kernel_size=3,
            upsample_kernel_size=2, norm_name=norm_name, res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            feature_size * 2, feature_size, kernel_size=3,
            upsample_kernel_size=2, norm_name=norm_name, res_block=res_block,
        )
        self.out = UnetOutBlock(feature_size, out_channels)

    def proj_feat(self, x):
        """Reshape (B, N, hidden) -> (B, D', H', W', hidden) channels-last."""
        B = x.shape[0]
        return x.reshape(B, *self.feat_size, self.hidden_size)

    def __call__(self, x_in):
        x, hidden_states = self.vit(x_in)

        # Encoder skip connections
        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(self.proj_feat(hidden_states[3]))
        enc3 = self.encoder3(self.proj_feat(hidden_states[6]))
        enc4 = self.encoder4(self.proj_feat(hidden_states[9]))

        # Decoder
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)

        return self.out(out)

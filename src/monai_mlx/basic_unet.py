"""
BasicUNet — classic 5-level U-Net for 3D medical image segmentation.

Port of MONAI's BasicUNet to MLX. Uses MaxPool downsampling,
ConvTranspose3d upsampling, concatenation skip connections,
and InstanceNorm + LeakyReLU.

All operations use channels-last (B, D, H, W, C) layout.
"""

from __future__ import annotations

import mlx.nn as nn

from .blocks import TwoConv, Down, UpCat, ConvOnly


class BasicUNet(nn.Module):
    """BasicUNet based on MONAI's implementation.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Defaults to 1.
    out_channels : int
        Number of output channels. Defaults to 2.
    features : tuple
        Six integers: 5 encoder feature sizes + 1 final decoder feature size.
        Defaults to (32, 32, 64, 128, 256, 32).
    act : str or tuple
        Activation. Defaults to LeakyReLU(0.1).
    norm : str or tuple
        Normalization. Defaults to InstanceNorm(affine=True).
    bias : bool
        Whether conv layers have bias. Defaults to True.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        features: tuple = (32, 32, 64, 128, 256, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
    ):
        super().__init__()
        f = features

        self.conv_0 = TwoConv(in_channels, f[0], act, norm, bias)
        self.down_1 = Down(f[0], f[1], act, norm, bias)
        self.down_2 = Down(f[1], f[2], act, norm, bias)
        self.down_3 = Down(f[2], f[3], act, norm, bias)
        self.down_4 = Down(f[3], f[4], act, norm, bias)

        self.upcat_4 = UpCat(f[4], f[3], f[3], act, norm, bias)
        self.upcat_3 = UpCat(f[3], f[2], f[2], act, norm, bias)
        self.upcat_2 = UpCat(f[2], f[1], f[1], act, norm, bias)
        self.upcat_1 = UpCat(f[1], f[0], f[5], act, norm, bias, halves=False)

        self.final_conv = nn.Conv3d(f[5], out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def __call__(self, x):
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        return self.final_conv(u1)

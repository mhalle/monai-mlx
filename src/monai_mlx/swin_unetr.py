"""
SwinUNETR — Swin Transformer + U-Net decoder for 3D segmentation.

Port of MONAI's SwinUNETR to MLX. Uses window-based multi-head self
attention with shifted windows, relative position bias, and patch merging.

All operations use channels-last (B, D, H, W, C) layout.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .unetr_blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock


# ---------------------------------------------------------------------------
# Window utilities
# ---------------------------------------------------------------------------

def window_partition(x, window_size):
    """Partition (B, D, H, W, C) into windows of (B*nw, wd*wh*ww, C)."""
    b, d, h, w, c = x.shape
    wd, wh, ww = window_size
    x = x.reshape(b, d // wd, wd, h // wh, wh, w // ww, ww, c)
    x = x.transpose(0, 1, 3, 5, 2, 4, 6, 7)  # (B, nd, nh, nw, wd, wh, ww, C)
    x = x.reshape(-1, wd * wh * ww, c)
    return x


def window_reverse(windows, window_size, dims):
    """Reverse window partition back to (B, D, H, W, C)."""
    b, d, h, w = dims
    wd, wh, ww = window_size
    c = windows.shape[-1]
    x = windows.reshape(b, d // wd, h // wh, w // ww, wd, wh, ww, c)
    x = x.transpose(0, 1, 4, 2, 5, 3, 6, 7)  # (B, nd, wd, nh, wh, nw, ww, C)
    x = x.reshape(b, d, h, w, c)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Adjust window/shift size if input is smaller than window."""
    use_window = list(window_size)
    use_shift = list(shift_size) if shift_size is not None else None
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window[i] = x_size[i]
            if use_shift is not None:
                use_shift[i] = 0
    if use_shift is None:
        return tuple(use_window)
    return tuple(use_window), tuple(use_shift)


def compute_mask(dims, window_size, shift_size):
    """Compute attention mask for shifted windows."""
    d, h, w = dims
    img_mask = np.zeros((1, d, h, w, 1), dtype=np.float32)
    cnt = 0
    for ds in (slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)):
        for hs in (slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)):
            for ws in (slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None)):
                img_mask[:, ds, hs, ws, :] = cnt
                cnt += 1

    img_mask_mx = mx.array(img_mask)
    mask_windows = window_partition(img_mask_mx, window_size)  # (nw, win_vol, 1)
    mask_windows = mask_windows.squeeze(-1)  # (nw, win_vol)
    attn_mask = mask_windows[:, :, None] - mask_windows[:, None, :]  # (nw, win_vol, win_vol)
    attn_mask = mx.where(attn_mask != 0, mx.array(-100.0), mx.array(0.0))
    return attn_mask


def roll_3d(x, shifts, axes):
    """Roll tensor along multiple axes (equivalent to torch.roll for 3D)."""
    for shift, axis in zip(shifts, axes):
        if shift == 0:
            continue
        n = x.shape[axis]
        shift = shift % n
        idx1 = [slice(None)] * x.ndim
        idx2 = [slice(None)] * x.ndim
        idx1[axis] = slice(n - shift, None)
        idx2[axis] = slice(0, n - shift)
        x = mx.concatenate([x[tuple(idx1)], x[tuple(idx2)]], axis=axis)
    return x


# ---------------------------------------------------------------------------
# Swin blocks
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    """Window-based multi-head self attention with relative position bias."""

    def __init__(self, dim: int, num_heads: int, window_size: tuple[int, ...],
                 qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        n_rel = np.prod([2 * w - 1 for w in window_size])
        self.relative_position_bias_table = mx.zeros((int(n_rel), num_heads))

        # Precompute relative position index
        coords = np.stack(np.meshgrid(
            *[np.arange(w) for w in window_size], indexing="ij"
        ))  # (3, wd, wh, ww)
        coords_flat = coords.reshape(len(window_size), -1)  # (3, N)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (3, N, N)
        rel = rel.transpose(1, 2, 0)  # (N, N, 3)
        rel[:, :, 0] += window_size[0] - 1
        rel[:, :, 1] += window_size[1] - 1
        rel[:, :, 2] += window_size[2] - 1
        rel[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        rel[:, :, 1] *= (2 * window_size[2] - 1)
        self._rel_pos_index = rel.sum(-1).astype(np.int32)  # (N, N)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x, mask=None):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, heads, N, dim_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(0, 1, 3, 2)  # (B, heads, N, N)

        # Relative position bias
        idx = mx.array(self._rel_pos_index[:n, :n].reshape(-1))
        bias = self.relative_position_bias_table[idx].reshape(n, n, -1)
        bias = bias.transpose(2, 0, 1)  # (heads, N, N)
        attn = attn + bias[None]

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.reshape(b // nw, nw, self.num_heads, n, n) + mask[None, :, None, :, :]
            attn = attn.reshape(-1, self.num_heads, n, n)

        attn = mx.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(b, n, c)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with optional shifted window attention."""

    def __init__(self, dim: int, num_heads: int, window_size: tuple[int, ...],
                 shift_size: tuple[int, ...], mlp_ratio: float = 4.0,
                 qkv_bias: bool = True):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp_linear1 = nn.Linear(dim, mlp_hidden)
        self.mlp_linear2 = nn.Linear(mlp_hidden, dim)
        self.mlp_act = nn.GELU()

    def __call__(self, x, mask_matrix):
        shortcut = x
        x = self.norm1(x)
        b, d, h, w, c = x.shape

        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)

        # Pad to multiples of window size
        pad_d = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_h = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_w = (window_size[2] - w % window_size[2]) % window_size[2]
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = mx.pad(x, [(0, 0), (0, pad_d), (0, pad_h), (0, pad_w), (0, 0)])
        _, dp, hp, wp, _ = x.shape

        # Shift
        if any(s > 0 for s in shift_size):
            shifted_x = roll_3d(x, [-shift_size[0], -shift_size[1], -shift_size[2]], [1, 2, 3])
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # Window partition → attention → reverse
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.reshape(-1, *window_size, c)
        shifted_x = window_reverse(attn_windows, window_size, (b, dp, hp, wp))

        # Reverse shift
        if any(s > 0 for s in shift_size):
            x = roll_3d(shifted_x, [shift_size[0], shift_size[1], shift_size[2]], [1, 2, 3])
        else:
            x = shifted_x

        # Remove padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :d, :h, :w, :]

        # Residual + MLP
        x = shortcut + x
        x = x + self.mlp_linear2(self.mlp_act(self.mlp_linear1(self.norm2(x))))
        return x


class PatchMerging(nn.Module):
    """Downsample by 2x via interleaved slicing + linear projection."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(8 * dim)

    def __call__(self, x):
        b, d, h, w, c = x.shape
        # Pad if odd
        if d % 2 == 1 or h % 2 == 1 or w % 2 == 1:
            pd = d % 2
            ph = h % 2
            pw = w % 2
            x = mx.pad(x, [(0, 0), (0, pd), (0, ph), (0, pw), (0, 0)])
        # Interleaved slicing: 8 sub-volumes concatenated along channel dim
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = mx.concatenate([x0, x1, x2, x3, x4, x5, x6, x7], axis=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    """Conv-based patch embedding for Swin Transformer."""

    def __init__(self, patch_size: tuple[int, ...], in_chans: int, embed_dim: int,
                 norm: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size, padding=0, bias=True)
        self.norm = nn.LayerNorm(embed_dim) if norm else None

    def __call__(self, x):
        # Pad to multiples of patch size
        b, d, h, w, c = x.shape
        pd = (self.patch_size[0] - d % self.patch_size[0]) % self.patch_size[0]
        ph = (self.patch_size[1] - h % self.patch_size[1]) % self.patch_size[1]
        pw = (self.patch_size[2] - w % self.patch_size[2]) % self.patch_size[2]
        if pd > 0 or ph > 0 or pw > 0:
            x = mx.pad(x, [(0, 0), (0, pd), (0, ph), (0, pw), (0, 0)])
        x = self.proj(x)  # (B, D', H', W', embed_dim)
        if self.norm is not None:
            # Flatten, norm, reshape back
            b2, d2, h2, w2, c2 = x.shape
            x = x.reshape(b2, -1, c2)
            x = self.norm(x)
            x = x.reshape(b2, d2, h2, w2, c2)
        return x


class BasicLayer(nn.Module):
    """One stage of Swin Transformer: N blocks + optional PatchMerging."""

    def __init__(self, dim: int, depth: int, num_heads: int,
                 window_size: tuple[int, ...], mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, downsample: bool = True):
        super().__init__()
        self.window_size = window_size
        shift_size = tuple(w // 2 for w in window_size)
        no_shift = tuple(0 for _ in window_size)

        self.blocks = [
            SwinTransformerBlock(
                dim, num_heads, window_size,
                shift_size=no_shift if (i % 2 == 0) else shift_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            )
            for i in range(depth)
        ]
        self.downsample = PatchMerging(dim) if downsample else None
        self._shift_size = shift_size

    def __call__(self, x):
        # x: (B, C, D, H, W) channels-first from SwinTransformer
        # Convert to channels-last for window attention
        b, c, d, h, w = x.shape
        x = x.transpose(0, 2, 3, 4, 1)  # (B, D, H, W, C)

        window_size, shift_size = get_window_size((d, h, w), self.window_size, self._shift_size)
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        attn_mask = compute_mask((dp, hp, wp), window_size, shift_size)

        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = x.reshape(b, d, h, w, -1)
        if self.downsample is not None:
            x = self.downsample(x)

        # Back to channels-first
        x = x.transpose(0, 4, 1, 2, 3)  # (B, C, D, H, W)
        return x


class SwinTransformer(nn.Module):
    """Swin Transformer backbone producing multi-scale features."""

    def __init__(self, in_chans: int, embed_dim: int, window_size: tuple[int, ...],
                 patch_size: tuple[int, ...], depths: Sequence[int],
                 num_heads: Sequence[int], mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, patch_norm: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim, norm=patch_norm)

        # Build 4 stages
        self.layers1 = [BasicLayer(
            embed_dim, depths[0], num_heads[0], window_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, downsample=True,
        )]
        self.layers2 = [BasicLayer(
            embed_dim * 2, depths[1], num_heads[1], window_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, downsample=True,
        )]
        self.layers3 = [BasicLayer(
            embed_dim * 4, depths[2], num_heads[2], window_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, downsample=True,
        )]
        self.layers4 = [BasicLayer(
            embed_dim * 8, depths[3], num_heads[3], window_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, downsample=True,
        )]

    def proj_out(self, x, normalize=False):
        """Optional layer norm on channels-first tensor."""
        if normalize:
            b, c = x.shape[:2]
            spatial = x.shape[2:]
            x = x.reshape(b, c, -1).transpose(0, 2, 1)  # (B, N, C)
            x = mx.fast.layer_norm(x, None, None, 1e-5) if hasattr(mx.fast, 'layer_norm') else x
            # Manual layer norm
            mean = mx.mean(x, axis=-1, keepdims=True)
            var = mx.var(x, axis=-1, keepdims=True)
            x = (x - mean) / mx.sqrt(var + 1e-5)
            x = x.transpose(0, 2, 1).reshape(b, c, *spatial)
        return x

    def __call__(self, x, normalize=True):
        # x: (B, D, H, W, C) channels-last input
        # PatchEmbed outputs channels-last, convert to channels-first for BasicLayer
        x0 = self.patch_embed(x)
        x0 = x0.transpose(0, 4, 1, 2, 3)  # (B, C, D, H, W)
        x0_out = self.proj_out(x0, normalize)

        x1 = self.layers1[0](x0)
        x1_out = self.proj_out(x1, normalize)
        x2 = self.layers2[0](x1)
        x2_out = self.proj_out(x2, normalize)
        x3 = self.layers3[0](x2)
        x3_out = self.proj_out(x3, normalize)
        x4 = self.layers4[0](x3)
        x4_out = self.proj_out(x4, normalize)

        # Convert all outputs to channels-last
        return [out.transpose(0, 2, 3, 4, 1) for out in [x0_out, x1_out, x2_out, x3_out, x4_out]]


# ---------------------------------------------------------------------------
# SwinUNETR
# ---------------------------------------------------------------------------

class SwinUNETR(nn.Module):
    """Swin UNETR: Swin Transformer encoder + UNETR-style CNN decoder.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    feature_size : int
        Base feature dimension (must be divisible by 12). Defaults to 24.
    depths : tuple
        Number of Swin blocks per stage. Defaults to (2, 2, 2, 2).
    num_heads : tuple
        Attention heads per stage. Defaults to (3, 6, 12, 24).
    window_size : tuple
        Local window size. Defaults to (7, 7, 7).
    norm_name : str or tuple
        Normalization for decoder. Defaults to "instance".
    patch_size : int
        Patch embedding size. Defaults to 2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_size: int = 24,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: tuple[int, ...] = (7, 7, 7),
        norm_name: str | tuple = "instance",
        patch_size: int = 2,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.swinViT = SwinTransformer(
            in_chans=in_channels, embed_dim=feature_size,
            window_size=window_size, patch_size=(patch_size,) * 3,
            depths=depths, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        )

        self.encoder1 = UnetrBasicBlock(in_channels, feature_size, 3, 1, norm_name, res_block=True)
        self.encoder2 = UnetrBasicBlock(feature_size, feature_size, 3, 1, norm_name, res_block=True)
        self.encoder3 = UnetrBasicBlock(2 * feature_size, 2 * feature_size, 3, 1, norm_name, res_block=True)
        self.encoder4 = UnetrBasicBlock(4 * feature_size, 4 * feature_size, 3, 1, norm_name, res_block=True)
        self.encoder10 = UnetrBasicBlock(16 * feature_size, 16 * feature_size, 3, 1, norm_name, res_block=True)

        self.decoder5 = UnetrUpBlock(16 * feature_size, 8 * feature_size, 3, 2, norm_name, res_block=True)
        self.decoder4 = UnetrUpBlock(8 * feature_size, 4 * feature_size, 3, 2, norm_name, res_block=True)
        self.decoder3 = UnetrUpBlock(4 * feature_size, 2 * feature_size, 3, 2, norm_name, res_block=True)
        self.decoder2 = UnetrUpBlock(2 * feature_size, feature_size, 3, 2, norm_name, res_block=True)
        self.decoder1 = UnetrUpBlock(feature_size, feature_size, 3, 2, norm_name, res_block=True)

        self.out = UnetOutBlock(feature_size, out_channels)

    def __call__(self, x_in):
        hidden = self.swinViT(x_in, normalize=True)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden[0])
        enc2 = self.encoder3(hidden[1])
        enc3 = self.encoder4(hidden[2])
        dec4 = self.encoder10(hidden[4])
        dec3 = self.decoder5(dec4, hidden[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        return self.out(out)

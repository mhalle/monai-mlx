"""
Transformer building blocks for MONAI in MLX.

PatchEmbedding, self-attention, MLP, TransformerBlock, and ViT.
All sequence operations work on (B, N, C) tensors.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class PatchEmbeddingBlock(nn.Module):
    """Conv-based patch embedding with learnable position embeddings.

    Input: (B, D, H, W, C_in) channels-last
    Output: (B, N, hidden) where N = (D/p)*(H/p)*(W/p)
    """

    def __init__(
        self,
        in_channels: int,
        img_size: tuple[int, ...],
        patch_size: tuple[int, ...],
        hidden_size: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        n_patches = int(np.prod([i // p for i, p in zip(img_size, patch_size)]))

        # Conv projection: kernel=stride=patch_size
        self.proj = nn.Conv3d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size, padding=0, bias=True,
        )
        # Learnable position embeddings
        self.position_embeddings = mx.zeros((1, n_patches, hidden_size))

    def __call__(self, x):
        # x: (B, D, H, W, C)
        x = self.proj(x)  # (B, D', H', W', hidden)
        B = x.shape[0]
        # Flatten spatial dims: (B, D', H', W', hidden) -> (B, N, hidden)
        x = x.reshape(B, -1, self.hidden_size)
        x = x + self.position_embeddings
        return x


class SABlock(nn.Module):
    """Self-attention block with combined QKV projection.

    Input/Output: (B, N, hidden)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = hidden_size // num_heads
        self.scale = self.dim_head ** -0.5
        self.inner_dim = hidden_size

        # Combined QKV projection
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x):
        B, N, C = x.shape
        # QKV projection
        qkv = self.qkv(x)  # (B, N, 3*hidden)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.dim_head)
        qkv = qkv.transpose(0, 2, 3, 1, 4)  # (B, 3, heads, N, dim_head)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each (B, heads, N, dim_head)

        # Scaled dot-product attention
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B, heads, N, N)
        attn = mx.softmax(attn, axis=-1)

        # Apply attention to values
        x = (attn @ v)  # (B, heads, N, dim_head)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, self.inner_dim)  # (B, N, hidden)

        x = self.out_proj(x)
        return x


class MLPBlock(nn.Module):
    """MLP with GELU activation.

    Input/Output: (B, N, hidden)
    """

    def __init__(self, hidden_size: int, mlp_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.fn = nn.GELU()

    def __call__(self, x):
        return self.linear2(self.fn(self.linear1(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> SA -> residual -> LN -> MLP -> residual."""

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(hidden_size, num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = MLPBlock(hidden_size, mlp_dim)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer encoder.

    Returns (final_output, list_of_hidden_states) for use by UNETR decoder.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: tuple[int, ...],
        patch_size: tuple[int, ...],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels, img_size, patch_size, hidden_size,
        )
        self.blocks = [
            TransformerBlock(hidden_size, mlp_dim, num_heads, qkv_bias=qkv_bias)
            for _ in range(num_layers)
        ]
        self.norm = nn.LayerNorm(hidden_size)

    def __call__(self, x):
        x = self.patch_embedding(x)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        return x, hidden_states_out

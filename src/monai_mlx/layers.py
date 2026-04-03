"""
Activation and normalization dispatch for MONAI compatibility.

Maps MONAI's string/tuple-based layer specification to MLX equivalents.
"""

from __future__ import annotations

import mlx.nn as nn


def get_activation(act: str | tuple | None) -> nn.Module | None:
    """Create an MLX activation from MONAI's specification format.

    Supports: string ("relu"), or tuple ("LeakyReLU", {"negative_slope": 0.1}).
    """
    if act is None:
        return None

    if isinstance(act, tuple):
        name = act[0].lower()
        kwargs = {k: v for k, v in act[1].items() if k != "inplace"} if len(act) > 1 else {}
    else:
        name = act.lower()
        kwargs = {}

    if name in ("relu",):
        return nn.ReLU()
    elif name in ("leakyrelu", "leaky_relu"):
        return nn.LeakyReLU(negative_slope=kwargs.get("negative_slope", 0.01))
    elif name in ("prelu",):
        return nn.PReLU()
    elif name in ("gelu",):
        return nn.GELU()
    elif name in ("elu",):
        return nn.ELU()
    elif name in ("selu",):
        return nn.SELU()
    elif name in ("sigmoid",):
        return nn.Sigmoid()
    elif name in ("tanh",):
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation: {act}")


def get_norm(norm: str | tuple, channels: int) -> nn.Module:
    """Create an MLX normalization from MONAI's specification format.

    Supports: string ("group"), or tuple ("GROUP", {"num_groups": 8}).
    """
    if isinstance(norm, tuple):
        name = norm[0].lower()
        kwargs = norm[1] if len(norm) > 1 else {}
    else:
        name = norm.lower()
        kwargs = {}

    if name in ("group",):
        num_groups = kwargs.get("num_groups", 8)
        return nn.GroupNorm(num_groups=num_groups, dims=channels, pytorch_compatible=True)
    elif name in ("instance",):
        affine = kwargs.get("affine", True)
        return nn.InstanceNorm(dims=channels, affine=affine)
    elif name in ("layer",):
        return nn.LayerNorm(dims=channels)
    elif name in ("batch",):
        # MLX doesn't have BatchNorm with running stats for inference.
        # Fall back to InstanceNorm which is equivalent at batch_size=1.
        return nn.InstanceNorm(dims=channels)
    else:
        raise ValueError(f"Unsupported norm: {norm}")

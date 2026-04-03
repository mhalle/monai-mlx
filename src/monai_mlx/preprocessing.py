"""
Preprocessing transforms for MONAI MLX inference.

Adapted from nnunet-mlx with MONAI-style normalization support.
"""

from __future__ import annotations

import numpy as np


def normalize_intensity(
    data: np.ndarray,
    subtrahend: float | None = None,
    divisor: float | None = None,
    nonzero: bool = False,
) -> np.ndarray:
    """Z-score normalization matching MONAI's NormalizeIntensity.

    Parameters
    ----------
    data : np.ndarray
        Input volume.
    subtrahend : float, optional
        Value to subtract. If None, uses mean of data (or nonzero voxels).
    divisor : float, optional
        Value to divide by. If None, uses std of data (or nonzero voxels).
    nonzero : bool
        If True, compute stats only from non-zero voxels.
    """
    data = data.astype(np.float32)
    if nonzero:
        mask = data != 0
        if mask.any():
            mean = data[mask].mean() if subtrahend is None else subtrahend
            std = data[mask].std() if divisor is None else divisor
        else:
            mean = subtrahend or 0.0
            std = divisor or 1.0
    else:
        mean = data.mean() if subtrahend is None else subtrahend
        std = data.std() if divisor is None else divisor

    return (data - mean) / max(std, 1e-8)


def scale_intensity_range(
    data: np.ndarray,
    a_min: float,
    a_max: float,
    b_min: float = 0.0,
    b_max: float = 1.0,
    clip: bool = True,
) -> np.ndarray:
    """Scale intensity to [b_min, b_max] matching MONAI's ScaleIntensityRange."""
    data = data.astype(np.float32)
    if clip:
        data = np.clip(data, a_min, a_max)
    data = (data - a_min) / max(a_max - a_min, 1e-8)
    data = data * (b_max - b_min) + b_min
    return data

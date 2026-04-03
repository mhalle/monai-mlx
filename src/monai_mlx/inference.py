"""
Sliding window prediction with Gaussian weighting for 3D volumes.

Adapted from nnunet-mlx with parameterized defaults for MONAI compatibility.
MONAI defaults to 25% overlap; nnU-Net defaults to 50%.
"""

from __future__ import annotations

import itertools
import time

import mlx.core as mx
import numpy as np


def compute_gaussian(
    tile_size: tuple[int, ...],
    sigma_scale: float = 1.0 / 8,
    value_scaling_factor: float = 1.0,
    dtype=np.float32,
) -> np.ndarray:
    """Compute Gaussian importance map for sliding window aggregation."""
    from scipy.ndimage import gaussian_filter

    tmp = np.zeros(tile_size)
    center = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center)] = 1
    gaussian_map = gaussian_filter(tmp, sigmas, 0, mode="constant", cval=0)
    gaussian_map = gaussian_map / (gaussian_map.max() / value_scaling_factor)
    mask = gaussian_map == 0
    gaussian_map[mask] = gaussian_map[~mask].min()
    return gaussian_map.astype(dtype)


def compute_sliding_window_steps(
    image_size: tuple[int, ...],
    tile_size: tuple[int, ...],
    tile_step_size: float,
) -> list[list[int]]:
    """Compute start positions for each dimension of the sliding window."""
    target_steps = [int(i * tile_step_size) for i in tile_size]
    num_steps = [
        int(np.ceil((i - k) / j)) + 1
        for i, j, k in zip(image_size, target_steps, tile_size)
    ]

    steps = []
    for dim in range(len(tile_size)):
        max_step = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step = max_step / (num_steps[dim] - 1)
        else:
            actual_step = 99999999
        steps.append(
            [int(np.round(actual_step * i)) for i in range(num_steps[dim])]
        )
    return steps


def predict_sliding_window(
    network,
    input_image: np.ndarray,
    patch_size: tuple[int, ...],
    num_classes: int,
    tile_step_size: float = 0.25,
    use_gaussian: bool = True,
    use_mirroring: bool = False,
    mirror_axes: tuple[int, ...] | None = None,
    batch_size: int = 1,
    use_fp16: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """Run batched sliding window inference over a 3D volume.

    Parameters
    ----------
    network : callable
        MLX network. Takes (B, D, H, W, C), returns (B, D, H, W, K).
    input_image : np.ndarray
        Shape (C, D, H, W) in float32.
    patch_size : tuple
        Spatial patch size (D, H, W).
    num_classes : int
        Number of output classes.
    tile_step_size : float
        Overlap fraction (0.25 = 75% overlap, MONAI default).
    use_gaussian : bool
        Weight predictions by Gaussian importance map.
    batch_size : int
        Number of patches to process in parallel.
    use_fp16 : bool
        Run inference in float16.
    verbose : bool
        Print progress info.

    Returns
    -------
    np.ndarray
        Predicted logits, shape (num_classes, D, H, W), float32.
    """
    spatial_shape = input_image.shape[1:]

    # Symmetric padding to match MONAI/nnU-Net
    pad_widths = []
    for s, t in zip(spatial_shape, patch_size):
        total_pad = max(0, t - s)
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_widths.append((pad_before, pad_after))
    needs_padding = any(p[0] > 0 or p[1] > 0 for p in pad_widths)
    if needs_padding:
        full_pad = [(0, 0)] + pad_widths
        input_image = np.pad(input_image, full_pad, mode="constant", constant_values=0)
        spatial_shape = input_image.shape[1:]

    # Convert to channels-last: (C, D, H, W) -> (D, H, W, C)
    data = input_image.transpose(1, 2, 3, 0)

    steps = compute_sliding_window_steps(spatial_shape, patch_size, tile_step_size)
    slicers = [
        (sx, sy, sz) for sx in steps[0] for sy in steps[1] for sz in steps[2]
    ]

    gaussian_np = (
        compute_gaussian(patch_size, sigma_scale=1.0 / 8, value_scaling_factor=10)
        if use_gaussian
        else np.ones(patch_size, dtype=np.float32)
    )

    accum_dtype = np.float16 if num_classes > 20 else np.float32
    predicted_logits = np.zeros((num_classes, *spatial_shape), dtype=accum_dtype)
    n_predictions = np.zeros(spatial_shape, dtype=np.float32)

    if use_mirroring and mirror_axes:
        axes_combos = [
            tuple(m + 1 for m in c)
            for k in range(len(mirror_axes))
            for c in itertools.combinations(mirror_axes, k + 1)
        ]
        n_tta = len(axes_combos) + 1
    else:
        axes_combos = []
        n_tta = 1

    total_batches = int(np.ceil(len(slicers) / batch_size))
    if verbose:
        total_fwd = total_batches * n_tta
        print(
            f"Sliding window: {len(slicers)} patches, batch_size={batch_size}, "
            f"tta={n_tta}x, total_fwd={total_fwd}, "
            f"image={spatial_shape}, patch={patch_size}"
        )

    _t0 = time.perf_counter()
    for batch_idx, batch_start in enumerate(range(0, len(slicers), batch_size)):
        if verbose and batch_idx > 0:
            elapsed = time.perf_counter() - _t0
            eta = elapsed / batch_idx * (total_batches - batch_idx)
            print(f"\r  {batch_idx}/{total_batches} "
                  f"({elapsed:.0f}s, ~{eta:.0f}s left)", end="", flush=True)
        batch_slicers = slicers[batch_start:batch_start + batch_size]

        patches_np = np.stack([
            data[sx:sx + patch_size[0], sy:sy + patch_size[1], sz:sz + patch_size[2], :]
            for sx, sy, sz in batch_slicers
        ])

        patches = mx.array(patches_np)
        if use_fp16:
            patches = patches.astype(mx.float16)

        pred = network(patches)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        if axes_combos:
            pred_sum = pred.astype(mx.float32)
            for axes in axes_combos:
                flipped_in = mx.array(
                    np.flip(patches_np, axis=list(axes)).copy()
                )
                if use_fp16:
                    flipped_in = flipped_in.astype(mx.float16)
                fp = network(flipped_in)
                if isinstance(fp, (list, tuple)):
                    fp = fp[0]
                fp = mx.array(
                    np.flip(np.array(fp.astype(mx.float32)), axis=list(axes)).copy()
                )
                pred_sum = pred_sum + fp
            pred = pred_sum * (1.0 / n_tta)

        mx.eval(pred)
        pred_np = np.array(pred.astype(mx.float32))

        for i, (sx, sy, sz) in enumerate(batch_slicers):
            p = pred_np[i].transpose(3, 0, 1, 2)
            if use_gaussian:
                p *= gaussian_np[None]
            predicted_logits[
                :,
                sx:sx + patch_size[0],
                sy:sy + patch_size[1],
                sz:sz + patch_size[2],
            ] += p
            n_predictions[
                sx:sx + patch_size[0],
                sy:sy + patch_size[1],
                sz:sz + patch_size[2],
            ] += gaussian_np

    if verbose:
        print()

    predicted_logits = predicted_logits.astype(np.float32) / n_predictions[None]

    if needs_padding:
        crop = tuple(
            slice(pb, s - pa) if (pb > 0 or pa > 0) else slice(None)
            for s, (pb, pa) in zip(predicted_logits.shape[1:], pad_widths)
        )
        predicted_logits = predicted_logits[(slice(None), *crop)]

    return predicted_logits


def choose_batch_size(
    patch_size: tuple[int, ...],
    num_classes: int = 14,
    dtype_bytes: int = 4,
) -> int:
    """Choose batch size that fits within Metal buffer limits."""
    max_buffer_bytes = _get_metal_max_buffer_bytes()
    from functools import reduce
    import operator
    act_bytes = _estimate_activation_bytes(patch_size, bytes_per_element=dtype_bytes)
    real_peak_bytes = act_bytes * 3
    output_bytes = reduce(operator.mul, patch_size) * num_classes * dtype_bytes
    per_patch_bytes = real_peak_bytes + output_bytes
    usable_bytes = max_buffer_bytes * 0.85
    batch = max(1, int(usable_bytes / per_patch_bytes))
    return min(batch, 8)


def _estimate_activation_bytes(
    patch_size: tuple[int, ...],
    features: list[int] = (32, 64, 128, 256, 320, 320),
    bytes_per_element: int = 2,
) -> int:
    """Estimate peak activation memory for one patch through a UNet."""
    total = 0
    for i, f in enumerate(features):
        spatial = [p // (2 ** i) if i > 0 else p for p in patch_size]
        total += f * int(np.prod(spatial)) * bytes_per_element * 2
    total *= 2
    return total


def _get_metal_max_buffer_bytes() -> int:
    """Get the Metal max buffer allocation size."""
    try:
        info = mx.device_info()
        return info["max_buffer_length"]
    except Exception:
        return 8 * 1024**3

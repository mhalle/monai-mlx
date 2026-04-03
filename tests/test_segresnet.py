"""
Test SegResNet MLX against PyTorch MONAI reference.

Requires: torch, monai (test dependencies only).
Run: pytest tests/test_segresnet.py -v -s
"""

import numpy as np
import pytest


def test_segresnet_forward_shape():
    """Verify MLX SegResNet produces correct output shape."""
    import mlx.core as mx
    from monai_mlx import SegResNet

    model = SegResNet(
        init_filters=8,
        in_channels=1,
        out_channels=4,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
    )

    x = mx.random.normal((1, 32, 32, 32, 1))  # (B, D, H, W, C)
    y = model(x)
    mx.eval(y)

    assert y.shape == (1, 32, 32, 32, 4), f"Expected (1,32,32,32,4), got {y.shape}"


def test_segresnet_matches_pytorch():
    """Forward pass should match PyTorch MONAI SegResNet within tolerance."""
    import torch
    import mlx.core as mx
    from monai.networks.nets import SegResNet as MonaiSegResNet
    from monai_mlx import SegResNet
    from monai_mlx.weights import convert_pytorch_weights, remap_segresnet_keys

    # Build PyTorch model
    pt_model = MonaiSegResNet(
        spatial_dims=3,
        init_filters=8,
        in_channels=1,
        out_channels=4,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        norm=("GROUP", {"num_groups": 8}),
    )
    pt_model.eval()

    # Build MLX model
    mlx_model = SegResNet(
        init_filters=8,
        in_channels=1,
        out_channels=4,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        norm=("GROUP", {"num_groups": 8}),
    )

    # Convert weights
    pt_weights = pt_model.state_dict()
    mlx_weights = convert_pytorch_weights(pt_weights)
    mlx_weights = remap_segresnet_keys(mlx_weights)
    mlx_model.load_weights(list(mlx_weights.items()))

    # Random input
    np.random.seed(42)
    x_np = np.random.randn(1, 1, 32, 32, 32).astype(np.float32)

    # PyTorch forward
    with torch.no_grad():
        y_pt = pt_model(torch.from_numpy(x_np)).numpy()

    # MLX forward: NCDHW -> NDHWC
    x_mlx = mx.array(x_np[0].transpose(1, 2, 3, 0)[None])
    y_mlx = mlx_model(x_mlx)
    mx.eval(y_mlx)
    # NDHWC -> NCDHW for comparison
    y_mlx_np = np.array(y_mlx)[0].transpose(3, 0, 1, 2)[None]

    # Compare
    diff = np.abs(y_pt - y_mlx_np)
    print(f"\nOutput shape: PT={y_pt.shape} MLX={y_mlx_np.shape}")
    print(f"Max abs diff: {diff.max():.6f}")
    print(f"Mean abs diff: {diff.mean():.6f}")

    assert y_pt.shape == y_mlx_np.shape, f"Shape mismatch: {y_pt.shape} vs {y_mlx_np.shape}"
    assert diff.max() < 1e-3, f"Output differs: max_diff={diff.max():.6f}"

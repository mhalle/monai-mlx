"""
Test BasicUNet MLX against PyTorch MONAI reference.

Run: pytest tests/test_basic_unet.py -v -s
"""

import numpy as np


def test_basic_unet_forward_shape():
    """Verify MLX BasicUNet produces correct output shape."""
    import mlx.core as mx
    from monai_mlx.basic_unet import BasicUNet

    model = BasicUNet(
        in_channels=1, out_channels=4,
        features=(8, 8, 16, 32, 64, 8),
    )
    x = mx.random.normal((1, 32, 32, 32, 1))
    y = model(x)
    mx.eval(y)
    assert y.shape == (1, 32, 32, 32, 4), f"Expected (1,32,32,32,4), got {y.shape}"


def test_basic_unet_matches_pytorch():
    """Forward pass should match PyTorch MONAI BasicUNet within tolerance."""
    import torch
    import mlx.core as mx
    from monai.networks.nets import BasicUNet as MonaiBasicUNet
    from monai_mlx.basic_unet import BasicUNet
    from monai_mlx.weights import convert_pytorch_weights, remap_basic_unet_keys

    pt_model = MonaiBasicUNet(
        spatial_dims=3, in_channels=1, out_channels=4,
        features=(8, 8, 16, 32, 64, 8),
        norm=("instance", {"affine": True}),
    )
    pt_model.eval()

    mlx_model = BasicUNet(
        in_channels=1, out_channels=4,
        features=(8, 8, 16, 32, 64, 8),
        norm=("instance", {"affine": True}),
    )

    pt_weights = pt_model.state_dict()
    mlx_weights = convert_pytorch_weights(pt_weights)
    mlx_weights = remap_basic_unet_keys(mlx_weights)
    mlx_model.load_weights(list(mlx_weights.items()))

    np.random.seed(42)
    x_np = np.random.randn(1, 1, 32, 32, 32).astype(np.float32)

    with torch.no_grad():
        y_pt = pt_model(torch.from_numpy(x_np)).numpy()

    x_mlx = mx.array(x_np[0].transpose(1, 2, 3, 0)[None])
    y_mlx = mlx_model(x_mlx)
    mx.eval(y_mlx)
    y_mlx_np = np.array(y_mlx)[0].transpose(3, 0, 1, 2)[None]

    diff = np.abs(y_pt - y_mlx_np)
    print(f"\nOutput shape: PT={y_pt.shape} MLX={y_mlx_np.shape}")
    print(f"Max abs diff: {diff.max():.6f}")
    print(f"Mean abs diff: {diff.mean():.6f}")

    assert y_pt.shape == y_mlx_np.shape
    assert diff.max() < 1e-3, f"Output differs: max_diff={diff.max():.6f}"

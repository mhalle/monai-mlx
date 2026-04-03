"""
Test MONAI UNet MLX against PyTorch reference.

Run: pytest tests/test_unet.py -v -s
"""

import numpy as np


def test_unet_forward_shape():
    """Verify MLX UNet produces correct output shape."""
    import mlx.core as mx
    from monai_mlx.unet import UNet

    model = UNet(
        in_channels=1, out_channels=2,
        channels=(8, 16, 32), strides=(2, 2),
        num_res_units=2, act="prelu", norm="instance",
    )
    x = mx.random.normal((1, 32, 32, 32, 1))
    y = model(x)
    mx.eval(y)
    assert y.shape == (1, 32, 32, 32, 2), f"Expected (1,32,32,32,2), got {y.shape}"


def test_unet_matches_pytorch():
    """Forward pass should match PyTorch MONAI UNet within tolerance."""
    import torch
    import mlx.core as mx
    from monai.networks.nets import UNet as MonaiUNet
    from monai_mlx.unet import UNet
    from monai_mlx.weights import convert_pytorch_weights, remap_unet_keys

    pt_model = MonaiUNet(
        spatial_dims=3, in_channels=1, out_channels=2,
        channels=(8, 16, 32), strides=(2, 2),
        num_res_units=2, norm="instance",
    )
    pt_model.eval()

    n_levels = 3
    mlx_model = UNet(
        in_channels=1, out_channels=2,
        channels=(8, 16, 32), strides=(2, 2),
        num_res_units=2, norm="instance",
    )

    pt_weights = pt_model.state_dict()
    mlx_weights = convert_pytorch_weights(pt_weights)
    mlx_weights = remap_unet_keys(mlx_weights, n_levels)

    try:
        mlx_model.load_weights(list(mlx_weights.items()))
    except ValueError as e:
        import mlx.nn as nn
        model_keys = sorted(k for k, _ in nn.utils.tree_flatten(mlx_model.parameters()))
        weight_keys = sorted(mlx_weights.keys())
        print(f"\nModel keys ({len(model_keys)}):")
        for k in model_keys:
            print(f"  {k}")
        print(f"\nWeight keys ({len(weight_keys)}):")
        for k in weight_keys:
            print(f"  {k}")
        raise

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
    assert diff.max() < 1e-2, f"Output differs: max_diff={diff.max():.6f}"

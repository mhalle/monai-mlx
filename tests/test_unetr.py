"""
Test UNETR MLX against PyTorch MONAI reference.

Run: pytest tests/test_unetr.py -v -s
"""

import numpy as np


def test_unetr_forward_shape():
    """Verify MLX UNETR produces correct output shape."""
    import mlx.core as mx
    from monai_mlx.unetr import UNETR

    model = UNETR(
        in_channels=1, out_channels=4,
        img_size=(32, 32, 32),
        feature_size=8, hidden_size=48, mlp_dim=96, num_heads=4,
    )

    x = mx.random.normal((1, 32, 32, 32, 1))
    y = model(x)
    mx.eval(y)
    assert y.shape == (1, 32, 32, 32, 4), f"Expected (1,32,32,32,4), got {y.shape}"


def test_unetr_matches_pytorch():
    """Forward pass should match PyTorch MONAI UNETR within tolerance."""
    import torch
    import mlx.core as mx
    from monai.networks.nets import UNETR as MonaiUNETR
    from monai_mlx.unetr import UNETR
    from monai_mlx.weights import convert_pytorch_weights, remap_unetr_keys

    # Small config for fast test
    pt_model = MonaiUNETR(
        in_channels=1, out_channels=4,
        img_size=(32, 32, 32),
        feature_size=8, hidden_size=48, mlp_dim=96,
        num_heads=4, norm_name="instance", res_block=True,
        spatial_dims=3,
    )
    pt_model.eval()

    mlx_model = UNETR(
        in_channels=1, out_channels=4,
        img_size=(32, 32, 32),
        feature_size=8, hidden_size=48, mlp_dim=96,
        num_heads=4, norm_name="instance", res_block=True,
    )

    pt_weights = pt_model.state_dict()
    mlx_weights = convert_pytorch_weights(pt_weights)
    mlx_weights = remap_unetr_keys(mlx_weights)

    try:
        mlx_model.load_weights(list(mlx_weights.items()))
    except ValueError as e:
        # Debug: show mismatched keys
        import mlx.nn as nn
        model_keys = set(k for k, _ in nn.utils.tree_flatten(mlx_model.parameters()))
        weight_keys = set(mlx_weights.keys())
        missing = model_keys - weight_keys
        extra = weight_keys - model_keys
        if missing:
            print(f"Missing ({len(missing)}): {sorted(missing)[:10]}")
        if extra:
            print(f"Extra ({len(extra)}): {sorted(extra)[:10]}")
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

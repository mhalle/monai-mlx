"""
Test SwinUNETR MLX against PyTorch MONAI reference.

Run: pytest tests/test_swin_unetr.py -v -s
"""

import numpy as np


def test_swin_unetr_forward_shape():
    """Verify MLX SwinUNETR produces correct output shape."""
    import mlx.core as mx
    from monai_mlx.swin_unetr import SwinUNETR

    model = SwinUNETR(
        in_channels=1, out_channels=4, feature_size=12,
        depths=(2, 2, 2, 2), num_heads=(3, 3, 3, 3),
        window_size=(4, 4, 4), patch_size=2,
    )
    # Input must be divisible by patch_size^5 = 32
    x = mx.random.normal((1, 64, 64, 64, 1))
    y = model(x)
    mx.eval(y)
    assert y.shape == (1, 64, 64, 64, 4), f"Expected (1,64,64,64,4), got {y.shape}"


def test_swin_unetr_matches_pytorch():
    """Forward pass should match PyTorch MONAI SwinUNETR within tolerance."""
    import torch
    import mlx.core as mx
    from monai.networks.nets import SwinUNETR as MonaiSwinUNETR
    from monai_mlx.swin_unetr import SwinUNETR
    from monai_mlx.weights import convert_pytorch_weights, remap_swin_unetr_keys

    pt_model = MonaiSwinUNETR(
        in_channels=1, out_channels=4, feature_size=12,
        depths=(2, 2, 2, 2), num_heads=(3, 3, 3, 3),
        window_size=(4, 4, 4), spatial_dims=3, patch_size=2,
        norm_name="instance",
    )
    pt_model.eval()

    mlx_model = SwinUNETR(
        in_channels=1, out_channels=4, feature_size=12,
        depths=(2, 2, 2, 2), num_heads=(3, 3, 3, 3),
        window_size=(4, 4, 4), patch_size=2,
        norm_name="instance",
    )

    pt_weights = pt_model.state_dict()
    mlx_weights = convert_pytorch_weights(pt_weights)
    mlx_weights = remap_swin_unetr_keys(mlx_weights)

    try:
        mlx_model.load_weights(list(mlx_weights.items()))
    except ValueError as e:
        import mlx.nn as nn
        model_keys = set(k for k, _ in nn.utils.tree_flatten(mlx_model.parameters()))
        weight_keys = set(mlx_weights.keys())
        missing = sorted(model_keys - weight_keys)
        extra = sorted(weight_keys - model_keys)
        if missing:
            print(f"\nMissing in weights ({len(missing)}):")
            for m in missing[:15]:
                print(f"  {m}")
        if extra:
            print(f"\nExtra in weights ({len(extra)}):")
            for e in extra[:15]:
                print(f"  {e}")
        raise

    np.random.seed(42)
    x_np = np.random.randn(1, 1, 64, 64, 64).astype(np.float32)

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

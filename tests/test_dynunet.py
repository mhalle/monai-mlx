"""
Test DynUNet MLX against PyTorch MONAI reference.

Run: pytest tests/test_dynunet.py -v -s
"""

import numpy as np


def test_dynunet_forward_shape():
    """Verify MLX DynUNet produces correct output shape."""
    import mlx.core as mx
    from monai_mlx.dynunet import DynUNet

    model = DynUNet(
        in_channels=1, out_channels=4,
        kernel_size=[[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
        strides=[[1,1,1],[2,2,2],[2,2,2],[2,2,2]],
        upsample_kernel_size=[[2,2,2],[2,2,2],[2,2,2]],
        filters=[16, 32, 64, 128],
    )
    x = mx.random.normal((1, 32, 32, 32, 1))
    y = model(x)
    mx.eval(y)
    assert y.shape == (1, 32, 32, 32, 4), f"Expected (1,32,32,32,4), got {y.shape}"


def test_dynunet_matches_pytorch():
    """Forward pass should match PyTorch MONAI DynUNet within tolerance."""
    import torch
    import mlx.core as mx
    from monai.networks.nets import DynUNet as MonaiDynUNet
    from monai_mlx.dynunet import DynUNet
    from monai_mlx.weights import convert_pytorch_weights, remap_dynunet_keys

    pt_model = MonaiDynUNet(
        spatial_dims=3, in_channels=1, out_channels=4,
        kernel_size=[[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
        strides=[[1,1,1],[2,2,2],[2,2,2],[2,2,2]],
        upsample_kernel_size=[[2,2,2],[2,2,2],[2,2,2]],
        filters=[16, 32, 64, 128],
    )
    pt_model.eval()

    mlx_model = DynUNet(
        in_channels=1, out_channels=4,
        kernel_size=[[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
        strides=[[1,1,1],[2,2,2],[2,2,2],[2,2,2]],
        upsample_kernel_size=[[2,2,2],[2,2,2],[2,2,2]],
        filters=[16, 32, 64, 128],
    )

    pt_weights = pt_model.state_dict()
    mlx_weights = convert_pytorch_weights(pt_weights)
    mlx_weights = remap_dynunet_keys(mlx_weights)

    try:
        mlx_model.load_weights(list(mlx_weights.items()))
    except ValueError as e:
        import mlx.nn as nn
        model_keys = sorted(k for k, _ in nn.utils.tree_flatten(mlx_model.parameters()))
        weight_keys = sorted(mlx_weights.keys())
        missing = sorted(set(model_keys) - set(weight_keys))
        extra = sorted(set(weight_keys) - set(model_keys))
        if missing:
            print(f"\nMissing ({len(missing)}):")
            for m in missing[:10]: print(f"  {m}")
        if extra:
            print(f"\nExtra ({len(extra)}):")
            for e in extra[:10]: print(f"  {e}")
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
    assert diff.max() < 1e-3, f"Output differs: max_diff={diff.max():.6f}"


def test_dynunet_anisotropic():
    """DynUNet with anisotropic kernels and strides should match PyTorch."""
    import torch
    import mlx.core as mx
    from monai.networks.nets import DynUNet as MonaiDynUNet
    from monai_mlx.dynunet import DynUNet
    from monai_mlx.weights import convert_pytorch_weights, remap_dynunet_keys

    # Anisotropic: different kernel/stride per axis
    cfg = dict(
        kernel_size=[[3, 3, 3], [3, 3, 1], [3, 3, 3], [3, 3, 3]],
        strides=[[1, 1, 1], [2, 2, 1], [2, 2, 2], [2, 2, 2]],
        upsample_kernel_size=[[2, 2, 1], [2, 2, 2], [2, 2, 2]],
        filters=[16, 32, 64, 128],
    )

    pt_model = MonaiDynUNet(spatial_dims=3, in_channels=1, out_channels=4, **cfg)
    pt_model.eval()

    mlx_model = DynUNet(in_channels=1, out_channels=4, **cfg)

    mlx_weights = remap_dynunet_keys(convert_pytorch_weights(pt_model.state_dict()))
    mlx_model.load_weights(list(mlx_weights.items()))

    np.random.seed(42)
    x_np = np.random.randn(1, 1, 32, 32, 16).astype(np.float32)

    with torch.no_grad():
        y_pt = pt_model(torch.from_numpy(x_np)).numpy()

    x_mlx = mx.array(x_np[0].transpose(1, 2, 3, 0)[None])
    y_mlx = mlx_model(x_mlx)
    mx.eval(y_mlx)
    y_mlx_np = np.array(y_mlx)[0].transpose(3, 0, 1, 2)[None]

    diff = np.abs(y_pt - y_mlx_np)
    print(f"\nAnisotropic: shape={y_mlx_np.shape}, max_diff={diff.max():.6f}")

    assert y_pt.shape == y_mlx_np.shape
    assert diff.max() < 1e-3, f"Anisotropic output differs: max_diff={diff.max():.6f}"

"""
MONAI Bundle loader for MLX.

Downloads pretrained MONAI bundles, parses their configs, builds MLX
models, and converts weights. After conversion, no PyTorch needed.

Usage:
    # CLI: download and convert a bundle
    monai-mlx-convert --bundle spleen_ct_segmentation

    # Python: load a converted bundle
    from monai_mlx.bundle import load_bundle
    model = load_bundle("~/.monai_mlx/spleen_ct_segmentation")
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import mlx.core as mx
import numpy as np

from .weights import (
    convert_pytorch_weights,
    load_weights_safetensors,
    save_weights_safetensors,
)

# Registry mapping MONAI _target_ strings to our MLX classes
MODEL_REGISTRY = {}


def _ensure_registry():
    """Lazy-populate the model registry."""
    if MODEL_REGISTRY:
        return
    from .segresnet import SegResNet
    from .basic_unet import BasicUNet
    from .unetr import UNETR
    from .swin_unetr import SwinUNETR

    MODEL_REGISTRY.update({
        "SegResNet": SegResNet,
        "monai.networks.nets.SegResNet": SegResNet,
        "monai.networks.nets.segresnet.SegResNet": SegResNet,
        "BasicUNet": BasicUNet,
        "BasicUnet": BasicUNet,
        "monai.networks.nets.BasicUNet": BasicUNet,
        "UNETR": UNETR,
        "monai.networks.nets.UNETR": UNETR,
        "monai.networks.nets.unetr.UNETR": UNETR,
        "SwinUNETR": SwinUNETR,
        "monai.networks.nets.SwinUNETR": SwinUNETR,
        "monai.networks.nets.swin_unetr.SwinUNETR": SwinUNETR,
    })


def _get_bundle_dir() -> Path:
    """Get the default bundle storage directory."""
    if "MONAI_MLX_HOME" in os.environ:
        return Path(os.environ["MONAI_MLX_HOME"])
    return Path.home() / ".monai_mlx"


def _strip_monai_kwargs(kwargs: dict) -> dict:
    """Remove PyTorch/MONAI-specific kwargs that don't apply to MLX."""
    skip = {"spatial_dims", "dropout_rate", "dropout_prob", "drop_rate",
            "attn_drop_rate", "dropout_path_rate", "use_checkpoint",
            "downsample", "use_v2", "proj_type", "pos_embed_type",
            "classification", "post_activation", "save_attn",
            "normalize", "norm_layer", "patch_norm",
            "img_size"  # UNETR-specific, handled separately
            }
    cleaned = {}
    for k, v in kwargs.items():
        if k.startswith("_") or k in skip:
            continue
        # Strip inplace from activation tuples
        if k == "act" and isinstance(v, (list, tuple)):
            v = list(v)
            if len(v) > 1 and isinstance(v[1], dict):
                v[1] = {kk: vv for kk, vv in v[1].items() if kk != "inplace"}
            v = tuple(v)
        cleaned[k] = v
    return cleaned


def parse_bundle_config(bundle_path: str | Path) -> dict:
    """Parse a MONAI bundle's config to extract network definition.

    Returns dict with keys: model_class, model_kwargs, and raw config.
    """
    bundle_path = Path(bundle_path)

    # Try various config locations
    for config_name in ("inference.json", "inference.yaml",
                         "train.json", "train.yaml"):
        config_file = bundle_path / "configs" / config_name
        if config_file.exists():
            break
    else:
        raise FileNotFoundError(f"No config found in {bundle_path / 'configs'}")

    if config_file.suffix == ".json":
        with open(config_file) as f:
            config = json.load(f)
    else:
        try:
            import yaml
            with open(config_file) as f:
                config = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for .yaml configs: pip install pyyaml")

    # Extract network definition
    net_def = config.get("network_def", config.get("network", {}))
    if isinstance(net_def, dict) and "_target_" in net_def:
        target = net_def["_target_"]
        kwargs = {k: v for k, v in net_def.items() if k != "_target_"}
    else:
        target = None
        kwargs = {}

    return {
        "target": target,
        "kwargs": kwargs,
        "config": config,
        "config_file": str(config_file),
    }


def build_model_from_config(parsed_config: dict):
    """Build an MLX model from parsed bundle config."""
    _ensure_registry()

    target = parsed_config["target"]
    if target is None:
        raise ValueError("No _target_ found in config")

    # Look up in registry — try exact match first, then endswith
    model_cls = MODEL_REGISTRY.get(target)
    if model_cls is None:
        # Try short name (last component)
        short = target.rsplit(".", 1)[-1]
        model_cls = MODEL_REGISTRY.get(short)

    if model_cls is None:
        raise ValueError(
            f"Unsupported model: {target}. "
            f"Supported: {list(set(MODEL_REGISTRY.values()))}"
        )

    kwargs = _strip_monai_kwargs(parsed_config["kwargs"])
    return model_cls(**kwargs)


def load_bundle(
    bundle_path: str | Path,
    weights_name: str = "model_mlx.safetensors",
) -> object:
    """Load a converted MONAI bundle.

    Parameters
    ----------
    bundle_path : str or Path
        Path to the bundle directory.
    weights_name : str
        Name of the safetensors weights file.

    Returns
    -------
    MLX model with loaded weights.
    """
    bundle_path = Path(bundle_path)
    parsed = parse_bundle_config(bundle_path)
    model = build_model_from_config(parsed)

    weights_path = bundle_path / "models" / weights_name
    if weights_path.exists():
        weights = load_weights_safetensors(weights_path)
        model.load_weights(list(weights.items()), strict=False)
    else:
        # Try loading from .pt
        pt_path = bundle_path / "models" / "model.pt"
        if pt_path.exists():
            import torch
            state_dict = torch.load(str(pt_path), map_location="cpu", weights_only=True)
            mlx_weights = convert_pytorch_weights(state_dict)
            model.load_weights(list(mlx_weights.items()), strict=False)
        else:
            raise FileNotFoundError(f"No weights found in {bundle_path / 'models'}")

    return model


def download_bundle(
    name: str,
    version: str | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Download a MONAI bundle from the model zoo.

    Returns path to the downloaded bundle directory.
    """
    import requests

    if output_dir is None:
        output_dir = _get_bundle_dir()
    output_dir = Path(output_dir)
    bundle_dir = output_dir / name

    if bundle_dir.exists():
        print(f"Bundle already exists: {bundle_dir}")
        return bundle_dir

    # Get download URL from MONAI
    try:
        from monai.bundle import scripts
        from monai.bundle.scripts import get_bundle_info
        if version is None:
            bundles = scripts.get_all_bundles_list()
            for bname, bver in bundles:
                if bname == name:
                    version = bver
                    break
            if version is None:
                raise ValueError(f"Bundle '{name}' not found in model zoo")
        info = get_bundle_info(name, version=version)
        url = info["browser_download_url"]
    except ImportError:
        # Fallback: construct URL directly
        if version is None:
            raise ValueError("version required when monai is not installed")
        url = (f"https://github.com/Project-MONAI/model-zoo/releases/download/"
               f"hosting_storage_v1/{name}_v{version}.zip")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use MONAI's download which handles HuggingFace, NGC, etc.
    try:
        from monai.bundle.scripts import download
        print(f"Downloading {name} v{version}...")
        download(name=name, version=version, bundle_dir=str(output_dir))
    except ImportError:
        raise ImportError(
            "monai is required for downloading bundles: "
            "uv add monai --dev"
        )

    if not bundle_dir.exists():
        candidates = list(output_dir.glob(f"{name}*"))
        if candidates:
            candidates[0].rename(bundle_dir)

    print(f"Saved to {bundle_dir}")
    return bundle_dir


def convert_bundle(
    bundle_path: str | Path,
    weights_name: str = "model.pt",
    output_name: str = "model_mlx.safetensors",
):
    """Convert a bundle's PyTorch weights to safetensors.

    One-time operation that requires torch.
    """
    import torch

    bundle_path = Path(bundle_path)
    pt_path = bundle_path / "models" / weights_name
    sf_path = bundle_path / "models" / output_name

    if sf_path.exists():
        print(f"Already converted: {sf_path}")
        return sf_path

    if not pt_path.exists():
        raise FileNotFoundError(f"No weights at {pt_path}")

    print(f"Loading {pt_path}...")
    state_dict = torch.load(str(pt_path), map_location="cpu", weights_only=True)
    mlx_weights = convert_pytorch_weights(state_dict)

    print(f"Saving {sf_path}...")
    save_weights_safetensors(mlx_weights, sf_path)
    print(f"Converted {len(mlx_weights)} tensors")
    return sf_path


def to_fp16(model, safe: bool = True):
    """Cast model weights to float16 to reduce memory usage.

    Parameters
    ----------
    model : nn.Module
        MLX model.
    safe : bool
        If True (default), only cast convolution weights (5D tensors)
        and leave normalization/linear weights in float32. This avoids
        overflow in attention and normalization layers.
        If False, cast everything to fp16.

    Returns
    -------
    The model with fp16 weights. Input data should also be cast to fp16.

    Note
    ----
    fp16 works reliably for convolution-only models (SegResNet, BasicUNet).
    For transformer models (UNETR, SwinUNETR), use safe=True or stay in fp32.
    """
    import mlx.nn as nn

    weights = list(nn.utils.tree_flatten(model.parameters()))
    if safe:
        fp16_weights = [
            (k, v.astype(mx.float16) if v.ndim >= 3 else v)
            for k, v in weights
        ]
    else:
        fp16_weights = [
            (k, v.astype(mx.float16) if v.dtype == mx.float32 else v)
            for k, v in weights
        ]
    model.load_weights(fp16_weights)
    return model


def convert_cli():
    """CLI entry point for bundle download and conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and convert MONAI bundles for MLX inference"
    )
    parser.add_argument("--bundle", help="Bundle name to download and convert")
    parser.add_argument("--version", help="Bundle version (default: latest)")
    parser.add_argument("--path", help="Path to existing bundle to convert")
    parser.add_argument("--output-dir", help="Output directory for downloaded bundles")
    parser.add_argument("--list", action="store_true", help="List available bundles")
    args = parser.parse_args()

    if args.list:
        try:
            from monai.bundle.scripts import get_all_bundles_list
            bundles = get_all_bundles_list()
            print(f"Available bundles ({len(bundles)}):")
            for name, ver in bundles:
                print(f"  {name} (v{ver})")
        except Exception as e:
            print(f"Error listing bundles: {e}")
            print("Install monai and requests: pip install monai requests")
        return

    if args.path:
        convert_bundle(args.path)
        return

    if args.bundle:
        bundle_dir = download_bundle(args.bundle, args.version, args.output_dir)
        convert_bundle(bundle_dir)
        print(f"\nReady! Load with:")
        print(f"  from monai_mlx.bundle import load_bundle")
        print(f"  model = load_bundle('{bundle_dir}')")
        return

    parser.print_help()

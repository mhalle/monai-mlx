"""
Weight conversion from PyTorch MONAI checkpoints to MLX format.

Handles NCDHW -> NDHWC transposition and key remapping for MONAI models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mlx.core as mx


def convert_pytorch_weights(pt_state_dict: dict) -> dict[str, mx.array]:
    """Convert a PyTorch MONAI state dict to MLX format.

    - Conv3d weights: (out, in, D, H, W) -> (out, D, H, W, in)
    - ConvTranspose3d weights: (in, out, D, H, W) -> (out, D, H, W, in)
    - 1D/2D tensors: no change
    """
    mlx_weights = {}

    for key, tensor in pt_state_dict.items():
        if hasattr(tensor, "numpy"):
            arr = tensor.cpu().numpy()
        else:
            arr = np.asarray(tensor)

        # Transpose 5D conv weights
        if arr.ndim == 5:
            # Detect ConvTranspose3d: key contains deconv/transp/upsample,
            # OR shape indicates transposed conv (in_ch > out_ch typically,
            # but more reliably: check if kernel matches stride pattern in UNETR)
            is_transpose = ("deconv" in key or "transp" in key.lower()
                           or "upsample" in key.lower()
                           # UNETR PrUpBlock: blocks.{i}.0.conv.weight is ConvTranspose
                           or (("blocks" in key and ".0.conv.weight" in key
                                and "conv_block" not in key and "conv1" not in key
                                and "conv2" not in key)))
            if is_transpose:
                # PyTorch ConvTranspose3d: (in_ch, out_ch, D, H, W)
                # MLX ConvTranspose3d:     (out_ch, D, H, W, in_ch)
                arr = arr.transpose(1, 2, 3, 4, 0)
            else:
                # PyTorch Conv3d: (out_ch, in_ch, D, H, W)
                # MLX Conv3d:     (out_ch, D, H, W, in_ch)
                arr = arr.transpose(0, 2, 3, 4, 1)

        mlx_weights[key] = mx.array(arr)

    return mlx_weights


def remap_segresnet_keys(mlx_weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap PyTorch SegResNet keys to MLX module hierarchy.

    PyTorch uses nn.Sequential and nn.ModuleList which add numeric indices.
    Our MLX model uses plain lists, so the key structure differs.

    PyTorch: down_layers.0.0.weight         (Identity at stage 0)
             down_layers.0.1.norm1.weight    (first ResBlock)
             down_layers.1.0.conv.weight     (stride-2 conv)
             down_layers.1.1.norm1.weight    (first ResBlock)
    MLX:     down_layers.0.0.norm1.weight    (first ResBlock, no Identity)
             down_layers.1.0.conv.weight     (stride-2 conv)
             down_layers.1.1.norm1.weight    (first ResBlock)
    """
    remapped = {}
    for key, val in mlx_weights.items():
        new_key = key
        # Handle down_layers: PyTorch wraps in nn.Sequential
        # Stage 0 has nn.Identity() as first element, skip it
        if key.startswith("down_layers.0."):
            parts = key.split(".")
            # down_layers.0.{idx}.{rest}
            idx = int(parts[2])
            if idx == 0:
                # This is nn.Identity() — skip it (no parameters)
                continue
            # Shift index down by 1 to account for removed Identity
            parts[2] = str(idx - 1)
            new_key = ".".join(parts)

        # Handle up_samples: PyTorch wraps [conv, upsample] in nn.Sequential
        # up_samples.{i}.0.conv.weight -> up_samples.{i}.0.conv.weight
        # (No remapping needed — our structure matches)

        # Handle up_layers: PyTorch wraps ResBlocks in nn.Sequential
        # up_layers.{i}.{j}.norm1.weight -> up_layers.{i}.{j}.norm1.weight
        # (No remapping needed — our structure matches)

        # Handle conv_final: PyTorch wraps [norm, act, conv] in nn.Sequential
        # conv_final.0.weight -> final_norm.weight
        # conv_final.2.weight -> final_conv.conv.weight
        if key.startswith("conv_final."):
            parts = key.split(".")
            idx = int(parts[1])
            rest = ".".join(parts[2:])
            if idx == 0:
                new_key = f"final_norm.{rest}"
            elif idx == 2:
                new_key = f"final_conv.{rest}"
            else:
                continue  # Skip activation (no parameters)

        remapped[new_key] = val

    return remapped


def remap_basic_unet_keys(mlx_weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap PyTorch BasicUNet keys to MLX module hierarchy.

    PyTorch: conv_0.conv_0.adn.N.weight -> MLX: conv_0.conv_0.norm.weight
             conv_0.conv_0.conv.weight   -> MLX: conv_0.conv_0.conv.weight (same)
             upcat_4.upsample.deconv.*   -> MLX: upcat_4.deconv.*
    """
    remapped = {}
    for key, val in mlx_weights.items():
        new_key = key
        # adn.N -> norm
        new_key = new_key.replace(".adn.N.", ".norm.")
        # upsample.deconv -> deconv
        new_key = new_key.replace(".upsample.deconv.", ".deconv.")
        remapped[new_key] = val
    return remapped


def remap_unetr_keys(mlx_weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap PyTorch UNETR/ViT keys to MLX module hierarchy."""
    remapped = {}
    for key, val in mlx_weights.items():
        new_key = key

        # Skip cross-attention weights (not used in inference)
        if "cross_attn" in key or "norm_cross_attn" in key:
            continue

        # patch_embedding.patch_embeddings -> patch_embedding.proj
        new_key = new_key.replace("patch_embedding.patch_embeddings.", "patch_embedding.proj.")

        # decoder*.transp_conv.conv -> decoder*.transp_conv
        # but NOT decoder*.conv_block.conv1.conv etc
        if ".transp_conv.conv." in new_key and ".conv_block." not in new_key:
            new_key = new_key.replace(".transp_conv.conv.", ".transp_conv.")

        # transp_conv_init.conv -> transp_conv_init
        new_key = new_key.replace(".transp_conv_init.conv.", ".transp_conv_init.")

        # encoder PrUpBlock: blocks.{i}.0.conv -> blocks.{i}.0 (ConvTranspose unwrap)
        import re
        m = re.match(r'(encoder\d+\.blocks\.\d+)\.0\.conv\.(.*)', new_key)
        if m:
            new_key = f"{m.group(1)}.0.{m.group(2)}"

        # encoder1.layer.conv3 -> encoder1.layer.downsample (residual projection)
        new_key = new_key.replace(".layer.conv3.", ".layer.downsample.")
        new_key = new_key.replace(".layer.norm3.", ".layer.norm3.")

        # decoder conv_block.conv3 -> conv_block.downsample
        new_key = new_key.replace(".conv_block.conv3.", ".conv_block.downsample.")

        # out.conv.conv -> out.conv
        if new_key.startswith("out.conv.conv."):
            new_key = new_key.replace("out.conv.conv.", "out.conv.")

        # Convolution.conv wrapper removal for dynunet blocks
        # conv1.conv.weight -> conv1.conv.weight (our ConvOnly also has .conv)
        # This should already match

        remapped[new_key] = val
    return remapped


def remap_swin_unetr_keys(mlx_weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap PyTorch SwinUNETR keys to MLX module hierarchy."""
    # First apply UNETR decoder/encoder remapping
    remapped = remap_unetr_keys(mlx_weights)

    final = {}
    for key, val in remapped.items():
        new_key = key

        # Skip relative_position_index (buffer, not parameter)
        if "relative_position_index" in key:
            continue

        # SwinTransformerBlock: mlp.linear1 -> mlp_linear1
        new_key = new_key.replace(".mlp.linear1.", ".mlp_linear1.")
        new_key = new_key.replace(".mlp.linear2.", ".mlp_linear2.")

        final[new_key] = val
    return final


def load_weights_safetensors(path: str | Path) -> dict[str, mx.array]:
    """Load MLX weights from safetensors format."""
    from safetensors.numpy import load_file
    np_weights = load_file(str(path))
    return {k: mx.array(v) for k, v in np_weights.items()}


def save_weights_safetensors(mlx_weights: dict[str, mx.array], path: str | Path):
    """Save MLX weights to safetensors format."""
    from safetensors.numpy import save_file
    np_weights = {k: np.array(v) for k, v in mlx_weights.items()}
    save_file(np_weights, str(path))

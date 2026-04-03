# monai-mlx

MLX inference backend for [MONAI](https://monai.io/) on Apple Silicon. Runs MONAI pretrained models natively on Metal without PyTorch.

[MONAI](https://github.com/Project-MONAI/MONAI) (Medical Open Network for Artificial Intelligence) is the leading open-source framework for deep learning in medical imaging, providing state-of-the-art pretrained models for segmentation, classification, and detection across CT, MRI, pathology, and more. This package brings MONAI's pretrained model zoo to Apple Silicon with native Metal acceleration.

> **Note:** This is an alpha-level port. It is not our intent to maintain a fork of MONAI. The goal is to demonstrate the potential performance improvement of using MLX to accelerate MONAI inference for Mac users. We hope that these changes can be incorporated back into the official [MONAI](https://github.com/Project-MONAI/MONAI) project in the future.

## Features

- 5 architectures: UNet, SegResNet, BasicUNet, UNETR, SwinUNETR
- 2.7x faster than PyTorch MPS, 12x faster than CPU on SwinUNETR
- Load pretrained models from the MONAI Model Zoo (35+ bundles)
- Sliding window inference with Gaussian weighting for full-volume segmentation
- Torch-free runtime after one-time weight conversion to safetensors
- All models verified against PyTorch MONAI within fp32 precision

## Benchmarks

Tested on an M2 Mac with 16GB RAM. SwinUNETR BTCV segmentation bundle (62M params, 96^3 patch, 14 classes):

| Backend | Time per patch | Speedup |
|---------|---------------|---------|
| **MLX (compiled)** | **0.96s** | **12x vs CPU** |
| MPS | 2.58s | 4.5x vs CPU |
| CPU | 11.6s | baseline |

## Installation

We recommend [uv](https://docs.astral.sh/uv/) for managing Python environments:

```bash
uv add monai-mlx
```

Or with pip:

```bash
pip install monai-mlx
```

## Quick start

```bash
# Download and convert a pretrained MONAI bundle (one-time, requires torch + monai)
uv run --with torch --with monai --with requests --with huggingface-hub \
    monai-mlx-convert --bundle swin_unetr_btcv_segmentation

# List all available bundles
uv run --with monai --with requests monai-mlx-convert --list
```

Then in Python (no torch needed):

```python
import mlx.core as mx
from monai_mlx.bundle import load_bundle

model = load_bundle("~/.monai_mlx/swin_unetr_btcv_segmentation")
compiled = mx.compile(model)

# Run inference on a 96x96x96 patch (channels-last)
x = mx.random.normal((1, 96, 96, 96, 1))
output = compiled(x)  # (1, 96, 96, 96, 14)
```

## Sliding window inference

For volumes larger than the model's patch size, use the sliding window inferer with Gaussian-weighted aggregation:

```python
import numpy as np
import mlx.core as mx
from monai_mlx.bundle import load_bundle
from monai_mlx.inference import predict_sliding_window

model = load_bundle("~/.monai_mlx/swin_unetr_btcv_segmentation")
compiled = mx.compile(model)

# Load a CT volume (C, D, H, W) in float32, preprocessed
volume = np.random.randn(1, 256, 256, 200).astype(np.float32)

logits = predict_sliding_window(
    network=compiled,
    input_image=volume,
    patch_size=(96, 96, 96),
    num_classes=14,
    tile_step_size=0.5,      # 50% overlap (MONAI default: 0.25 for 75%)
    use_gaussian=True,        # Gaussian-weighted blending at patch borders
    batch_size=1,
)

segmentation = np.argmax(logits, axis=0)  # (D, H, W)
```

The sliding window automatically handles padding, overlap, and stitching. Batch size is auto-tuned based on Metal buffer limits if not specified.

## Weight conversion

MONAI bundles include PyTorch `.pt` weight files. monai-mlx converts these to safetensors for fast, torch-free loading at runtime.

**Convert a downloaded bundle:**

```bash
# Download and convert in one step
uv run --with torch --with monai --with requests --with huggingface-hub \
    monai-mlx-convert --bundle swin_unetr_btcv_segmentation

# Or convert an existing bundle directory
uv run --with torch monai-mlx-convert --path ~/.monai_mlx/swin_unetr_btcv_segmentation
```

After conversion, the bundle directory contains both formats:

```
~/.monai_mlx/swin_unetr_btcv_segmentation/
  configs/inference.json        # model architecture definition
  models/model.pt               # original PyTorch weights
  models/model_mlx.safetensors  # converted MLX weights (used at runtime)
```

At runtime, `load_bundle()` prefers safetensors if available, falls back to `.pt` (which requires torch). After conversion, the runtime dependencies are just `mlx`, `numpy`, `nibabel`, `scipy`, and `safetensors`.

## Supported models

| Model | Architecture | Params | Verified | Use case |
|-------|-------------|--------|----------|----------|
| UNet | ResidualUnit + recursive SkipConnection | varies | max diff 0.000008 | Spleen, multi-organ (most common MONAI model) |
| SegResNet | Pre-activation residual + GroupNorm | 4.7M | max diff 0.000002 | Brain tumor, organ segmentation |
| BasicUNet | MaxPool + ConvTranspose + concat skips | varies | max diff 0.000003 | General segmentation |
| UNETR | ViT encoder + CNN decoder | varies | max diff 0.000004 | Multi-organ segmentation |
| SwinUNETR | Swin Transformer + shifted window attention | 62M | max diff 0.000080 | BTCV, brain tumor, whole body |

Not yet supported: DynUNet, diffusion models, detection models.

## Memory optimization (fp16)

For memory-constrained machines, convolution-only models (SegResNet, BasicUNet) can run in fp16:

```python
from monai_mlx import load_bundle, to_fp16

model = load_bundle("~/.monai_mlx/my_segresnet_bundle")
model = to_fp16(model)          # halves weight memory
compiled = mx.compile(model)
output = compiled(x.astype(mx.float16))
```

For transformer models (UNETR, SwinUNETR), use `to_fp16(model, safe=True)` which only casts convolution weights and keeps attention/normalization in fp32 to avoid overflow.

| Model | Full fp16 | Safe fp16 (conv only) |
|-------|----------|----------------------|
| SegResNet | Works (max diff 0.002) | Works |
| BasicUNet | Works | Works |
| UNETR | Risk of overflow | Works |
| SwinUNETR | NaN (attention overflow) | Works (saves ~44% memory) |

## Relationship to nnunet-mlx

This package is a companion to [nnunet-mlx](https://github.com/mhalle/nnunet-mlx), which ports nnU-Net inference to MLX for use with [TotalSegmentator](https://github.com/wasserth/TotalSegmentator). Both packages share the same approach:

- Standalone MLX reimplementation (not a fork)
- PyTorch weight conversion to safetensors
- Channels-last layout with `mx.compile` optimization
- Verified against PyTorch within fp32 precision

The packages are independent -- neither depends on the other. They share ~350 lines of infrastructure (sliding window, weight I/O, Metal buffer detection) copied with attribution.

## How it works

Pure MLX reimplementation of MONAI's inference path:

- **blocks.py** -- ConvNormAct, ResBlock (pre-activation), TwoConv, Down, UpCat
- **segresnet.py** -- SegResNet with trilinear upsampling and additive skips
- **basic_unet.py** -- Classic 5-level U-Net with MaxPool and ConvTranspose
- **transformer.py** -- PatchEmbedding, self-attention, MLP, ViT
- **swin_unetr.py** -- Window attention, shifted windows, relative position bias, PatchMerging, SwinTransformer
- **unet.py** -- MONAI UNet with ResidualUnit blocks and recursive-to-flat architecture
- **unetr_blocks.py** -- UnetResBlock, UnetrUpBlock, UnetrPrUpBlock
- **bundle.py** -- MONAI bundle config parser, model registry, download + convert CLI
- **inference.py** -- Sliding window prediction with Gaussian weighting
- **weights.py** -- PyTorch-to-MLX weight conversion with safetensors I/O
- **layers.py** -- Activation/normalization dispatch (MONAI string format to MLX)
- **preprocessing.py** -- Intensity normalization (z-score, scale range)

Key implementation details:
- `mx.compile` for fused operations (~2-3x speedup on transformer models)
- Channels-last (NDHWC) layout throughout (native MLX format)
- `GroupNorm(pytorch_compatible=True)` for exact numerical match with PyTorch
- Conv weight transposition: NCDHW to NDHWC, with ConvTranspose detection
- Window attention uses numpy for mask computation, MLX for the forward pass

## Adding new models

To port a new MONAI architecture:

1. Implement the model in a new `.py` file using MLX layers
2. Add a weight key remapper in `weights.py` if the PyTorch key hierarchy differs
3. Register the model in `bundle.py`'s `MODEL_REGISTRY`
4. Write a test that builds both PyTorch and MLX versions, converts weights, and compares outputs

The test pattern:

```python
def test_mymodel_matches_pytorch():
    pt_model = MonaiMyModel(...)
    pt_model.eval()
    mlx_model = MyModel(...)

    mlx_weights = convert_pytorch_weights(pt_model.state_dict())
    mlx_weights = remap_mymodel_keys(mlx_weights)
    mlx_model.load_weights(list(mlx_weights.items()))

    # Compare outputs on same random input
    # NCDHW for PyTorch, NDHWC for MLX
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python >= 3.10
- MLX >= 0.22
- For weight conversion: torch, monai (one-time only)
- For bundle download: monai, requests, huggingface-hub (one-time only)

## Citations

If you use this package, please cite the original MONAI and model papers:

**MONAI:**
MONAI Consortium. MONAI: Medical Open Network for AI. 2020. https://monai.io/

**SwinUNETR:**
Hatamizadeh, A. et al. Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. *BrainLes 2021*. https://arxiv.org/abs/2201.01266

**SegResNet:**
Myronenko, A. 3D MRI brain tumor segmentation using autoencoder regularization. *BrainLes 2018*. https://arxiv.org/abs/1810.11654

**UNETR:**
Hatamizadeh, A. et al. UNETR: Transformers for 3D Medical Image Segmentation. *WACV 2022*. https://arxiv.org/abs/2103.10504

## License

Apache 2.0 (same as MONAI).

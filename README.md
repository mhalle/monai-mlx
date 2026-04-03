# monai-mlx

MLX inference backend for [MONAI](https://monai.io/) on Apple Silicon. Runs MONAI pretrained models natively on Metal without PyTorch.

[MONAI](https://github.com/Project-MONAI/MONAI) (Medical Open Network for Artificial Intelligence) is the leading open-source framework for deep learning in medical imaging, providing state-of-the-art pretrained models for segmentation, classification, and detection across CT, MRI, pathology, and more. This package brings MONAI's pretrained model zoo to Apple Silicon with native Metal acceleration.

> **Note:** This is an alpha-level port. It is not our intent to maintain a fork of MONAI. The goal is to demonstrate the potential performance improvement of using MLX to accelerate MONAI inference for Mac users. We hope that these changes can be incorporated back into the official [MONAI](https://github.com/Project-MONAI/MONAI) project in the future.

## Features

- 4 architectures: SegResNet, BasicUNet, UNETR, SwinUNETR
- 2.7x faster than PyTorch MPS, 12x faster than CPU on SwinUNETR
- Load pretrained models from the MONAI Model Zoo (35+ bundles)
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

## Supported models

| Model | Architecture | Params | Verified |
|-------|-------------|--------|----------|
| SegResNet | Pre-activation residual + GroupNorm | 4.7M | max diff 0.000002 |
| BasicUNet | MaxPool + ConvTranspose + concat skips | varies | max diff 0.000003 |
| UNETR | ViT encoder + CNN decoder | varies | max diff 0.000004 |
| SwinUNETR | Swin Transformer + shifted window attention | 62M | max diff 0.000080 |

## How it works

Pure MLX reimplementation of MONAI's inference path:

- **blocks.py** -- ConvNormAct, ResBlock (pre-activation), TwoConv, Down, UpCat
- **segresnet.py** -- SegResNet with trilinear upsampling and additive skips
- **basic_unet.py** -- Classic 5-level U-Net with MaxPool and ConvTranspose
- **transformer.py** -- PatchEmbedding, self-attention, MLP, ViT
- **swin_unetr.py** -- Window attention, shifted windows, relative position bias, PatchMerging, SwinTransformer
- **unetr_blocks.py** -- UnetResBlock, UnetrUpBlock, UnetrPrUpBlock
- **bundle.py** -- MONAI bundle config parser, model registry, download + convert CLI
- **inference.py** -- Sliding window prediction with Gaussian weighting
- **weights.py** -- PyTorch-to-MLX weight conversion with safetensors I/O

Key optimizations:
- `mx.compile` for fused operations (~2-3x speedup on transformer models)
- Channels-last layout throughout (native MLX format)
- `GroupNorm(pytorch_compatible=True)` for exact numerical match

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python >= 3.10
- MLX >= 0.22

## Citations

If you use this package, please cite the original MONAI paper:

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

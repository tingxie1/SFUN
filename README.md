# SFUN
Spatial-Frequency Unfolding Network with Mamba for Dual-Camera Compressive Hyperspectral Imaging
SFUN is a nine-stage, parameter-sharing spectral reconstruction model that retains both row-wise and column-wise Mamba modules. The model entry point is `csi.architectures.sfun`, and the configuration node is `MODEL.SFUN`.

## Installation

First, install PyTorch and torchvision compatible with your CUDA environment.

Mamba and causal-conv1d require a compatible CUDA/PyTorch build environment. Running the model requires CUDA; it will not fall back to a convolutional model if Mamba is not installed.

## Data and Training

Run commands from this directory. Modify the data paths in `configs/sfun.yaml`. The default layout is:

```text
datasets/
  train/                 Training scene .mat files
  test/                  Independent test scene .mat files
  mask_3d_shift.mat       Mask file containing mask_3d_shift
```

The default input has 28 spectral bands, training uses 256 × 256 crops, and PAN guidance is enabled. See `csi/data/data.py` for the supported HSI fields and normalization rules.


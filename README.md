# AJSDM-Mamba
Adaptive Jump Spatial-Spectral Mamba for Hyperspectral Image Classification
This repository contains the official implementation of AJSDM-Mamba (Adaptive Jump Spatial-Spectral Dual Mamba), a novel framework for hyperspectral image (HSI) classification. The model integrates spatial and spectral branches with adaptive step-size scanning and dynamic mutation fusion, achieving state-of-the-art performance on multiple HSI benchmarks.

# Method Overview
The model consists of three main components:
﻿
Spatial feature extraction – uses the proposed AJSS-Mamba block with adaptive scanning strides that depend on local edge complexity.
﻿
Spectral feature extraction – uses AJBS-Mamba to adaptively skip redundant bands based on spectral derivative complexity.
﻿
Dynamic Mutation Fusion – generates multiple mutated feature pairs and fuses them via inverse‑variance weighting.
﻿
Both Mamba‑based branches are built upon the selective scan mechanism (mamba_ssm) and are extended with learnable step‑size control.

# Dependencies
Python >= 3.8

PyTorch >= 1.12

mamba_ssm (requires CUDA and selective scan kernel compilation)

einops, timm

scipy, numpy, scikit-learn, matplotlib

torchsummary, torch_optimizer (optional)

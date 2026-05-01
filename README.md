# AJSDP-Mamba
Adaptive Jump Scanning Mamba with Dynamic Perturbation Fusion for Hyperspectral Image Classification
This repository contains the official implementation of AJSDP-Mamba, a novel framework for hyperspectral image (HSI) classification. The model integrates adaptive spatial‑spectral scanning with a noise‑based perturbation fusion strategy, achieving state‑of‑the‑art performance on multiple HSI benchmarks

# Method Overview
The model consists of three main components:
﻿
Spatial feature extraction – uses the proposed AJSS-Mamba block with adaptive scanning strides that depend on local edge complexity.
﻿
Spectral feature extraction – uses AJBS-Mamba to adaptively skip redundant bands based on spectral derivative complexity.
﻿
Noise‑based Perturbation Ensemble with Variance‑weighted Fusion (NPF) – Diversifies the extracted spatial and spectral features by applying multiple rounds of random masked noise. The perturbed representations are then fused using an inverse‑variance weighting scheme: candidates with higher stability (lower variance) receive larger weights. This mechanism improves feature diversity, suppresses noise, and enhances robustness, especially in small‑sample scenarios.
Both Mamba‑based branches are built upon the selective scan mechanism (mamba_ssm) and are extended with learnable step‑size control.

# Dependencies
Python >= 3.8

PyTorch >= 1.12

mamba_ssm (requires CUDA and selective scan kernel compilation)

einops, timm

scipy, numpy, scikit-learn, matplotlib

torchsummary, torch_optimizer (optional)
# Data Preparation
Place your HSI datasets in the ./data/ folder with the following structure:
```text
./data/
├── IndianPines/
│   ├── Indian_pines_corrected.mat
│   └── Indian_pines_gt.mat
├── PaviaU/
│   ├── PaviaU.mat
│   └── PaviaU_gt.mat
├── Salinas/
│   ├── Salinas_corrected.mat
│   └── Salinas_gt.mat
└── ... (other datasets as named in dataloader.py)
```
The code automatically applies PCA reduction (dimension set per dataset) before training.

# Training
```text
python training_test.py -d PU -b 64 -e 60 --is_PCA True
```
Note: The code assumes GPU availability for training. CPU training is possible but will be very slow.

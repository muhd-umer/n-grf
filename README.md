# Neural Gaussian Radio Fields for Channel Estimation

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0%2B-ee4c2c)]()

**nGRF** (neural Gaussian Radio fields) is a novel framework for accurate and efficient wireless channel estimation using explicit 3D Gaussian primitives. Unlike previous approaches like NeRF-based methods that rely on slow implicit representations, or existing Gaussian splatting methods that use non-physical 2D projections, nGRF performs direct 3D electromagnetic field aggregation with each Gaussian acting as a localized radio modulator.

## Features

- **Physics-informed 3D model** **::** Direct 3D aggregation of radio field contributions instead of 2D splatting or volumetric rendering
- **Explicit representation** **::** Uses 3D Gaussian primitives for fast training and inference
- **MIMO support** **::** Models complex-valued MIMO channel matrices, supporting both SISO and MIMO configurations
- **GPU acceleration** **::** CUDA-optimized kernels for real-time channel rendering
- **Highly efficient** **::** Significantly reduces data requirements, training time, and inference latency compared to NeRF-based approaches

## Installation 

### Prerequisites

- Python 3.12+
- PyTorch 2.7.0+
- CUDA-capable GPU (for accelerated rendering)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/muhd-umer/n-grf.git
   git submodule update --init --recursive
   cd n-grf
   ```

2. Create a virtual environment using `uv`:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   uv sync --inexact
   ```

4. Install the submodules:
   ```bash
   uv pip install "simple-knn @ ./submodules/simple-knn"
   uv pip install "render @ ./render"
   ```

## Usage

### Training

To train the nGRF model, use the `train.py` script:
```bash
python train.py \
  --data_path /path/to/dataset.mat \
  --config configs/default_mimo.yaml
```

For SISO configurations, use the SISO config:
```bash
python train.py \
  --data_path /path/to/dataset.mat \
  --config configs/default_siso.yaml
```

### Evaluation

To evaluate a trained model:
```bash
python eval.py \
  --data_path /path/to/dataset.mat \
  --checkpoint /path/to/model.pt
```

### Hyperparameter Tuning

For automated hyperparameter optimization:
```bash
python tune.py \
  --data_path /path/to/dataset.mat \
  --config configs/default_mimo.yaml \
  --num_trials 50
```

## Project Layout

Following is the directory structure of the nGRF repository:
```python
nGRF/
├── pyproject.toml
├── configs/                # Configuration files for MIMO and SISO runs
├── datasets/               # Dataset generation, loading, and processing
├── models/                 # Core model implementation
│   ├── gaussian_model.py   # nGRF model implementation
│   ├── networks.py         # Neural network architectures
├── render/                 # Efficient rendering implementation
│   ├── _cuda_impl/         # CUDA kernels for rendering
│   ├── _torch_impl/        # PyTorch fallback implementation
├── submodules/             # Submodules for external dependencies
├── train.py                # Training script
├── tune.py                 # Hyperparameter tuning script
├── eval.py                 # Evaluation script
└── utils/                  # Utility functions for training, logging, etc.
```

## Overview

nGRF represents wireless environments as collections of 3D Gaussian primitives that act as radio modulators. Each Gaussian is parameterized by:
- `Position` **::** $\mu$
- `Rotation` **::** quaternion $q$
- `Scaling` **::** anisotropic $s$
- Dynamically computed attributes via neural networks

For channel estimation, nGRF:
1. Computes spatial weights based on 3D Gaussian distributions
2. Determines complex channel contributions from each Gaussian
3. Aggregates contributions to form the complete channel matrix

This approach offers significant advantages in computational efficiency, data requirements, and inference latency compared to previous methods.

## Citation

If you use nGRF in your research, please consider citing our work:

```bibtex
TODO: Add citation information here
```
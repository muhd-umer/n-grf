This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: utils, models, engine/, datasets/*m, datasets/*.py, train.py
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

## Additional Info

# Directory Structure
```
datasets/
  __init__.py
  create_users.m
  dataloader.py
  fspl.m
  generate_csi.m
  generate_pc.m
  get_ray_chan.m
  indoor.m
  outdoor.m
  ray_marching.m
  wireless_dataset.py
engine/
  __init__.py
  render_channel.py
models/
  __init__.py
  gaussian_model.py
  networks.py
utils/
  __init__.py
  general_utils.py
  loss.py
  pos_encoder.py
  train_utils.py
  transform_utils.py
train.py
```

# Files

## File: datasets/__init__.py
```python
# datasets/__init__.py

from .dataloader import get_dataloaders, get_wireless_dataloader
from .wireless_dataset import WirelessDataset
```

## File: engine/render_channel.py
```python
# engine/render_channel.py

import torch
import torch.nn.functional as F

from models.networks import ContributionDecoderNetwork


@torch.jit.script
def compute_spatial_weight(
    d_vec_n: torch.Tensor,
    inv_covariance_n: torch.Tensor,
    base_activation_n: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Computes the spatial weight using the Gaussian PDF.
    w_n = o(a_n) * exp(-0.5 * d_vec_n^T * Σ_n^(-1) * d_vec_n)
    """
    d_vec_n_unsqueezed = d_vec_n.unsqueeze(1)

    exponent_term = torch.bmm(d_vec_n_unsqueezed, inv_covariance_n)
    exponent_term = torch.bmm(
        exponent_term, d_vec_n_unsqueezed.transpose(1, 2)
    ).squeeze()

    pdf_weight = torch.exp(-0.5 * torch.clamp(exponent_term, max=50.0, min=-50.0))

    activation_weight = torch.sigmoid(base_activation_n).squeeze(-1)

    weight = F.relu(activation_weight * pdf_weight)
    weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)

    return weight + eps


def render_channel(
    rx_positions: torch.Tensor,
    gauss_means: torch.Tensor,
    gauss_inv_covs: torch.Tensor,
    gauss_latents: torch.Tensor,
    gauss_activations: torch.Tensor,
    decoder_network: ContributionDecoderNetwork,
    wavelength: float,
    nt: int,
    nr: int,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Renders the complex channel matrix H for a batch of receiver positions.

    Args:
        rx_positions: Batch of receiver positions (B, 3).
        gauss_means: Gaussian means (N, 3).
        gauss_inv_covs: Gaussian inverse covariances (N, 3, 3).
        gauss_latents: Gaussian latent features (N, F).
        gauss_activations: Gaussian base activations (N, 1).
        decoder_network: Network mapping latents to complex contributions.
        wavelength: Wavelength of the signal.
        nt: Number of Tx antennas.
        nr: Number of Rx antennas.
        eps: Small value for numerical stability.

    Returns:
        Batch of predicted channel matrices (B, Nt, Nr) complex.
    """
    batch_size = rx_positions.shape[0]
    num_gaussians = gauss_means.shape[0]
    device = rx_positions.device

    rx_pos_expanded = rx_positions.unsqueeze(1)
    gauss_means_expanded = gauss_means.unsqueeze(0)

    d_vec = rx_pos_expanded - gauss_means_expanded
    d_norm = torch.norm(d_vec, dim=-1, p=2).clamp(min=eps)

    phase_shift_rad = (2.0 * torch.pi / wavelength) * d_norm

    phase_real = torch.cos(-phase_shift_rad)
    phase_imag = torch.sin(-phase_shift_rad)

    phase_real = torch.nan_to_num(phase_real, nan=0.0, posinf=0.0, neginf=0.0)
    phase_imag = torch.nan_to_num(phase_imag, nan=0.0, posinf=0.0, neginf=0.0)
    phase_shift_factor = torch.complex(phase_real, phase_imag)

    d_vec_flat = d_vec.view(-1, 3)
    inv_covs_expanded = gauss_inv_covs.repeat(batch_size, 1, 1)
    activations_expanded = gauss_activations.repeat(batch_size, 1)

    spatial_weights = compute_spatial_weight(
        d_vec_flat, inv_covs_expanded, activations_expanded, eps
    ).view(batch_size, num_gaussians)

    h_contrib_flat = decoder_network(gauss_latents)

    h_contrib_real_flat = h_contrib_flat[..., : (nt * nr)]
    h_contrib_imag_flat = h_contrib_flat[..., (nt * nr) :]

    h_contrib_real_flat = torch.nan_to_num(
        h_contrib_real_flat, nan=0.0, posinf=0.0, neginf=0.0
    )
    h_contrib_imag_flat = torch.nan_to_num(
        h_contrib_imag_flat, nan=0.0, posinf=0.0, neginf=0.0
    )

    h_contrib = torch.complex(
        h_contrib_real_flat.view(num_gaussians, nt, nr),
        h_contrib_imag_flat.view(num_gaussians, nt, nr),
    )

    h_contrib_expanded = h_contrib.unsqueeze(0)
    combined_weight = spatial_weights.unsqueeze(-1).unsqueeze(
        -1
    ) * phase_shift_factor.unsqueeze(-1).unsqueeze(-1)

    weighted_contributions = combined_weight * h_contrib_expanded
    H_pred = torch.sum(weighted_contributions, dim=1)

    return H_pred
```

## File: models/__init__.py
```python
# models/__init__.py

from .gaussian_model import GaussianChannelFieldModel
from .networks import AttributeNetwork, ContributionDecoderNetwork
```

## File: models/networks.py
```python
# models/networks.py

from typing import Tuple

import torch
import torch.nn as nn

from utils.pos_encoder import PositionalEncoder


class SimpleMLP(nn.Module):
    """A simple Multi-Layer Perceptron with ReLU activations."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ContributionDecoderNetwork(SimpleMLP):
    """
    Decodes latent features into complex channel contributions.
    Output dimension is 2 * Nt * Nr (real and imaginary parts).
    """

    def __init__(
        self, latent_dim: int, output_dim: int, hidden_dim: int, num_layers: int
    ):
        super().__init__(latent_dim, output_dim, hidden_dim, num_layers)


class AttributeNetwork(nn.Module):
    """
    Predicts latent features (f_n) and base activations (a_n) from
    Gaussian position (μ_n) and fixed Tx position (P_TX).
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        pos_encoding_freqs: int = 10,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.pos_encoder = PositionalEncoder(3, pos_encoding_freqs)
        encoded_dim = self.pos_encoder.output_dims
        input_mlp_dim = 2 * encoded_dim

        self.network = SimpleMLP(
            input_dim=input_mlp_dim,
            output_dim=latent_dim + 1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(
        self, mu_n: torch.Tensor, p_tx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mu_n: Gaussian means (N, 3)
            p_tx: Transmitter position (1, 3) or (3,) - expanded to (N, 3)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Latent features (N, latent_dim), Base activations (N, 1)
        """
        num_gaussians = mu_n.shape[0]
        if p_tx.dim() == 1:
            p_tx = p_tx.unsqueeze(0)
        p_tx_expanded = p_tx.expand(num_gaussians, -1)

        encoded_mu = self.pos_encoder(mu_n)
        encoded_ptx = self.pos_encoder(p_tx_expanded)

        mlp_input = torch.cat([encoded_mu, encoded_ptx], dim=-1)
        output = self.network(mlp_input)

        latent_features = output[:, : self.latent_dim]
        base_activations = output[:, self.latent_dim :]

        return latent_features, base_activations
```

## File: utils/__init__.py
```python
# utils/__init__.py

from .general_utils import *
from .loss import *
from .pos_encoder import *
from .train_utils import *
from .transform_utils import *
```

## File: utils/general_utils.py
```python
# utils/general_utils.py

import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


def set_random_seed(seed: Optional[int] = None) -> None:
    """Set random seeds for reproducibility."""
    if seed is not None:
        print(f"Setting random seed to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
```

## File: utils/loss.py
```python
# utils/loss.py

import numpy as np
import torch
import torch.nn as nn


def calculate_nmse(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    """
    Calculate Normalized Mean Squared Error (NMSE) for complex tensors.
    Handles batch dimension and potential NaN/Inf values robustly for autograd.

    Args:
        pred: Predicted complex tensor (B, ..., Nt, Nr)
        target: Ground truth complex tensor (B, ..., Nt, Nr)
        eps: Small value for numerical stability

    Returns:
        NMSE loss value (scalar tensor).
    """
    if not torch.is_complex(pred):
        raise TypeError("Prediction tensor must be complex.")
    if not torch.is_complex(target):
        raise TypeError("Target tensor must be complex.")

    target = target.to(device=pred.device, dtype=pred.dtype)

    pred_real_clean = torch.nan_to_num(pred.real, nan=0.0, posinf=0.0, neginf=0.0)
    pred_imag_clean = torch.nan_to_num(pred.imag, nan=0.0, posinf=0.0, neginf=0.0)
    target_real_clean = torch.nan_to_num(target.real, nan=0.0, posinf=0.0, neginf=0.0)
    target_imag_clean = torch.nan_to_num(target.imag, nan=0.0, posinf=0.0, neginf=0.0)

    pred_clean = torch.complex(pred_real_clean, pred_imag_clean)
    target_clean = torch.complex(target_real_clean, target_imag_clean)

    error_sq = torch.abs(pred_clean - target_clean) ** 2
    diff_sq_sum = torch.sum(error_sq, dim=tuple(range(1, pred_clean.dim())))

    target_sq = torch.abs(target_clean) ** 2
    target_sq_sum = torch.sum(target_sq, dim=tuple(range(1, target_clean.dim()))).clamp(
        min=eps
    )

    nmse_per_item = diff_sq_sum / target_sq_sum

    if torch.isnan(nmse_per_item).any() or torch.isinf(nmse_per_item).any():
        print(
            "Warning: NaN or Inf detected in NMSE per item after cleaning. Replacing with 0."
        )
        nmse_per_item = torch.nan_to_num(nmse_per_item, nan=0.0, posinf=0.0, neginf=0.0)

    nmse = torch.mean(nmse_per_item)

    if torch.isnan(nmse) or torch.isinf(nmse):
        print("Warning: Final NMSE is NaN or Inf. Returning 1.0")
        return torch.tensor(1.0, device=pred.device)

    return nmse


class NormalizedMSELoss(nn.Module):
    """Computes the Normalized Mean Squared Error (NMSE) loss for complex tensors."""

    def __init__(self, eps: float = 1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted complex tensor (B, ..., Nt, Nr).
            target: Ground truth complex tensor (B, ..., Nt, Nr).

        Returns:
            Scalar NMSE loss.
        """
        return calculate_nmse(pred, target, self.eps)


def calculate_snr(nmse: torch.Tensor) -> torch.Tensor:
    """Calculate SNR in dB from NMSE loss value."""
    if torch.isnan(nmse) or torch.isinf(nmse):
        print("Warning: Cannot calculate SNR from NaN/Inf NMSE. Returning -Inf.")
        return torch.tensor(float("-inf"), device=nmse.device)
    nmse_clamped = torch.clamp(nmse, min=1e-10)
    snr = -10.0 * torch.log10(nmse_clamped)
    return snr
```

## File: utils/pos_encoder.py
```python
# utils/pos_encoder.py

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    """Sine-cosine positional encoder for input points."""

    def __init__(
        self,
        input_dims: int,
        num_freqs: int,
        include_input: bool = True,
        log_sampling: bool = True,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.output_dims = 0
        self.embedding_fns = []
        self._create_embedding_fn()

    def _create_embedding_fn(self):
        """Create the embedding functions."""
        if self.include_input:
            self.embedding_fns.append(lambda x: x)
            self.output_dims += self.input_dims

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(
                0.0, self.num_freqs - 1, steps=self.num_freqs
            )
        else:
            freq_bands = torch.linspace(
                1.0, 2.0 ** (self.num_freqs - 1), steps=self.num_freqs
            )

        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                self.embedding_fns.append(
                    lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq)
                )
                self.output_dims += self.input_dims

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding.
        Args:
            inputs: Input tensor (..., input_dims)
        Returns:
            Encoded tensor (..., output_dims)
        """
        encoded = torch.cat([fn(inputs) for fn in self.embedding_fns], dim=-1)
        return encoded
```

## File: utils/train_utils.py
```python
# utils/train_utils.py

import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch


def get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1000000,
) -> Callable[[int], float]:
    """Computes learning rate following exponential decay."""

    def helper(step: int) -> float:
        if step < 0 or max_steps <= 0:
            return 0.0
        if lr_init == 0.0 and lr_final == 0.0:
            return 0.0

        if lr_delay_steps > 0:
            delay_factor = lr_delay_mult + (1.0 - lr_delay_mult) * np.sin(
                0.5 * np.pi * min(step / lr_delay_steps, 1.0)
            )
        else:
            delay_factor = 1.0

        progress = min(step / max_steps, 1.0)

        if lr_init <= 0:
            log_lr_init = -np.inf
        else:
            log_lr_init = np.log(lr_init)

        if lr_final <= 0:
            log_lr_final = -np.inf
        else:
            log_lr_final = np.log(lr_final)

        log_lerped_lr = log_lr_init * (1.0 - progress) + log_lr_final * progress
        lerped_lr = np.exp(log_lerped_lr)

        return delay_factor * lerped_lr

    return helper


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def compute_grad_stats(
    model: torch.nn.Module, norm_type: float = 2.0
) -> Dict[str, float]:
    """Computes statistics about gradients for model parameters."""
    total_norm = 0.0
    total_abs_sum = 0.0
    total_elements = 0
    min_grad = float("inf")
    max_grad = float("-inf")
    param_count_with_grad = 0

    for param in model.parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(
                    f"Warning: NaN or Inf detected in gradients for parameter. Skipping stats for this param."
                )
                continue

            param_norm = param.grad.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
            total_abs_sum += param.grad.abs().sum().item()
            total_elements += param.numel()
            param_count_with_grad += 1

            current_min = param.grad.min().item()
            current_max = param.grad.max().item()
            min_grad = min(min_grad, current_min)
            max_grad = max(max_grad, current_max)

    total_norm = total_norm ** (1.0 / norm_type) if total_norm > 0 else 0.0
    mean_abs_grad = total_abs_sum / total_elements if total_elements > 0 else 0.0

    if min_grad == float("inf"):
        min_grad = 0.0
    if max_grad == float("-inf"):
        max_grad = 0.0

    return {
        "grad_norm": total_norm,
        "mean_abs_grad": mean_abs_grad,
        "min_grad": min_grad,
        "max_grad": max_grad,
        "param_count_with_grad": param_count_with_grad,
    }
```

## File: utils/transform_utils.py
```python
# utils/transform_utils.py

from typing import Tuple, Union

import torch


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Convert sigmoid output back to logits."""
    eps = torch.finfo(x.dtype).eps
    x = torch.clamp(x, min=eps, max=1.0 - eps)
    return torch.log(x / (1 - x))


@torch.jit.script
def build_rotation(r: torch.Tensor) -> torch.Tensor:
    """Build rotation matrices from quaternions (ensure normalization)."""
    norm = torch.sqrt(torch.sum(r * r, dim=1, keepdim=True)).clamp(min=1e-10)
    q = r / norm

    R = torch.zeros((q.size(0), 3, 3), device=r.device, dtype=r.dtype)

    qw = q[:, 0]
    qx = q[:, 1]
    qy = q[:, 2]
    qz = q[:, 3]

    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz
    qxqy = qx * qy
    qxqz = qx * qz
    qyqz = qy * qz
    qwqx = qw * qx
    qwqy = qw * qy
    qwqz = qw * qz

    R[:, 0, 0] = 1.0 - 2.0 * (qy2 + qz2)
    R[:, 0, 1] = 2.0 * (qxqy - qwqz)
    R[:, 0, 2] = 2.0 * (qxqz + qwqy)
    R[:, 1, 0] = 2.0 * (qxqy + qwqz)
    R[:, 1, 1] = 1.0 - 2.0 * (qx2 + qz2)
    R[:, 1, 2] = 2.0 * (qyqz - qwqx)
    R[:, 2, 0] = 2.0 * (qxqz - qwqy)
    R[:, 2, 1] = 2.0 * (qyqz + qwqx)
    R[:, 2, 2] = 1.0 - 2.0 * (qx2 + qy2)
    return R


@torch.jit.script
def build_scaling_rotation(s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Build combined scaling and rotation matrix L = R * S."""
    L = torch.zeros((s.shape[0], 3, 3), dtype=s.dtype, device=s.device)
    R = build_rotation(r)
    S_diag = torch.diag_embed(s)
    L = R @ S_diag
    return L


@torch.jit.script
def build_covariance_from_scaling_rotation(
    scaling: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Build covariance matrix Σ = R * S^2 * R^T."""
    R = build_rotation(rotation)
    S_sq_diag = torch.diag_embed(scaling * scaling)
    covariance = R @ S_sq_diag @ R.transpose(1, 2)
    return covariance


@torch.jit.script
def build_covariance_inverse(
    R: torch.Tensor, scaling: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Builds the inverse covariance matrix Σ^-1 = R * S^-2 * R^T.
    Handles potential division by zero in scaling.

    Args:
        R: Rotation matrices (N, 3, 3).
        scaling: Activated scaling factors (N, 3).
        eps: Small value to prevent division by zero.

    Returns:
        Inverse covariance matrices (N, 3, 3).
    """
    scaling_clamped = torch.clamp(scaling, min=eps)
    inv_scaling_sq = 1.0 / (scaling_clamped * scaling_clamped)
    S_inv_sq_diag = torch.diag_embed(inv_scaling_sq)

    inv_covariance = R @ S_inv_sq_diag @ R.transpose(1, 2)
    return inv_covariance
```

## File: datasets/create_users.m
```
function [Users, num_created] = create_users(env_dims, num_users, rx_array, user_params, stl_data)
    % CREATE_USERS Generate user positions in a 3D environment
    %
    % Description:
    %   Creates user receiver sites within specified environmental dimensions
    %   with optional collision detection for buildings and other users.
    %
    % Inputs:
    %   env_dims    - [3x2] Matrix defining environment bounds [xmin xmax; ymin ymax; zmin zmax]
    %   num_users   - Number of users to generate
    %   rx_array    - Antenna array configuration for receivers
    %   user_params - Structure with the following optional fields:
    %                   .check_building_collision - Enable building collision (default: false)
    %                   .check_user_collision    - Enable user separation (default: true)
    %                   .separation_distance     - Minimum distance between users (auto)
    %   stl_data    - STL mesh data for building collision detection (optional)
    %
    % Outputs:
    %   Users       - Array of rxsite objects representing user positions
    %   num_created - Actual number of users created (may be less than requested)
    %
    % Example:
    %   env_dims = [-10 10; -10 10; 0 20];
    %   num_users = 10;
    %   rx_array = phased.ULA('NumElements', 4);
    %   params.check_building_collision = true;
    %   params.separation_distance = 2;
    %   [users, created] = create_users(env_dims, num_users, rx_array, params);

    % default parameters if not provided
    if ~isfield(user_params, 'check_building_collision')
        user_params.check_building_collision = false;
    end

    if ~isfield(user_params, 'check_user_collision')
        user_params.check_user_collision = true;
    end

    if ~isfield(user_params, 'separation_distance')
        % calculate default separation based on environment size and user count
        env_area = (env_dims(1, 2) - env_dims(1, 1)) * (env_dims(2, 2) - env_dims(2, 1));
        user_params.separation_distance = sqrt(env_area / (num_users * pi)) / 2;
    end

    Users(num_users) = rxsite;
    valid_users = 0;
    max_attempts = num_users * 50;
    attempts = 0;

    % store valid positions for faster collision checking
    valid_positions = zeros(3, num_users);

    % preprocess stl data if building collision check is enabled
    if user_params.check_building_collision && ~isempty(stl_data)
        vertices = stl_data.Points;
        faces = stl_data.ConnectivityList;
    end

    while valid_users < num_users && attempts < max_attempts
        pos = [ ...
                   unifrnd(env_dims(1, 1), env_dims(1, 2)); ...
                   unifrnd(env_dims(2, 1), env_dims(2, 2)); ...
                   unifrnd(0.3, 1.8) ...
               ];

        is_valid = true;

        % check building collision if enabled
        if user_params.check_building_collision && ~isempty(stl_data)
            is_valid = ~is_point_in_building(pos', vertices, faces);
        end

        % check user collision if enabled and passed building check
        if is_valid && user_params.check_user_collision && valid_users > 0

            for i = 1:valid_users

                if norm(pos(1:2) - valid_positions(1:2, i)) < user_params.separation_distance
                    is_valid = false;
                    break;
                end

            end

        end

        if is_valid
            valid_users = valid_users + 1;
            valid_positions(:, valid_users) = pos;
            Users(valid_users) = rxsite("cartesian", ...
                "Antenna", rx_array, ...
                "AntennaPosition", pos);
        end

        attempts = attempts + 1;
    end

    if valid_users < num_users
        warning('could not place all users after %d attempts', max_attempts);
        Users = Users(1:valid_users);
    end

    num_created = valid_users;
end

function inside = is_point_in_building(point, vertices, faces)
    % ray casting algorithm for point-in-mesh detection
    ray_direction = [1, 0, 0];
    intersections = 0;

    for i = 1:size(faces, 1)
        triangle = vertices(faces(i, :), :);

        if does_ray_intersect_triangle(point, ray_direction, triangle)
            intersections = intersections + 1;
        end

    end

    inside = mod(intersections, 2) == 1;
end

function intersects = does_ray_intersect_triangle(origin, direction, triangle)
    % möller-trumbore ray-triangle intersection algorithm
    epsilon = 1e-7;

    edge1 = triangle(2, :) - triangle(1, :);
    edge2 = triangle(3, :) - triangle(1, :);
    h = cross(direction, edge2);
    a = dot(edge1, h);

    if abs(a) < epsilon
        intersects = false;
        return;
    end

    f = 1 / a;
    s = origin - triangle(1, :);
    u = f * dot(s, h);

    if u < 0.0 || u > 1.0
        intersects = false;
        return;
    end

    q = cross(s, edge1);
    v = f * dot(direction, q);

    if v < 0.0 || u + v > 1.0
        intersects = false;
        return;
    end

    t = f * dot(edge2, q);
    intersects = t > epsilon;
end
```

## File: datasets/fspl.m
```
function pl = fspl(d, f)
    % FSPL Compute free-space path loss (FSPL)
    %
    % Description:
    %   Calculates the free-space path loss in dB for a given distance and frequency.
    %
    % Inputs:
    %   d - Distance between transmitter and receiver (meters)
    %   f - Frequency (Hz)
    %
    % Output:
    %   pl - Path loss in dB
    %
    % Example:
    %   pl = fspl(100, 2.4e9);

    pl = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/physconst("lightspeed"));
end
```

## File: datasets/generate_pc.m
```
function all_points = generate_pc(vertices, faces, params, env_dims, visualize)
    % GENERATE_PC Generate point clouds for a given environment
    %
    % Description:
    %   Generates a point cloud representation of an environment using multiple
    %   sampling strategies including edge, surface, volume, and boundary points.
    %   Supports optional DBSCAN clustering and visualization.
    %
    % Inputs:
    %   vertices   - [Nx3] Matrix of vertex coordinates from STL
    %   faces      - [Mx3] Matrix of face indices from STL
    %   params     - Structure with the following optional fields:
    %                  .edge_density      - Points per edge unit length (default: 0)
    %                  .surface_density   - Points per triangle unit area (default: 0)
    %                  .volume_density    - Points per unit volume (default: 0)
    %                  .boundary_density  - Points per boundary surface area (default: 0)
    %                  .random_points     - Additional random points in volume (default: 0)
    %                  .noise_std         - Standard deviation for perturbation (default: 0)
    %                  .edge_reduction    - Factor to reduce edge points (default: 1)
    %                  .surface_reduction - Factor to reduce surface points (default: 1)
    %                  .use_dbscan        - Enable DBSCAN clustering (default: false)
    %                  .dbscan_epsilon    - DBSCAN epsilon parameter (default: 0.2)
    %                  .dbscan_minpts     - DBSCAN minimum points (default: 5)
    %   env_dims   - [3x2] Matrix of environment bounds [min_x max_x; min_y max_y; min_z max_z]
    %   visualize  - (Optional) Boolean to enable visualization (default: false)
    %
    % Output:
    %   all_points - [Px3] Matrix of generated point cloud coordinates
    %
    % Example:
    %   [v, f] = stlread('building.stl');
    %   params.edge_density = 1;
    %   params.surface_density = 0.5;
    %   env_dims = [-10 10; -10 10; 0 20];
    %   points = generate_pc(v, f, params, env_dims, true);

    % set default values if not provided
    default_params = struct('edge_density', 0, ...
        'surface_density', 0, ...
        'volume_density', 0, ...
        'boundary_density', 0, ...
        'random_points', 0, ...
        'noise_std', 0, ...
        'edge_reduction', 1, ...
        'surface_reduction', 1, ...
        'use_dbscan', false, ...
        'dbscan_epsilon', 0.2, ...
        'dbscan_minpts', 5);

    % merge provided params with defaults
    if ~isfield(params, 'edge_density'), params.edge_density = default_params.edge_density; end
    if ~isfield(params, 'surface_density'), params.surface_density = default_params.surface_density; end
    if ~isfield(params, 'volume_density'), params.volume_density = default_params.volume_density; end
    if ~isfield(params, 'boundary_density'), params.boundary_density = default_params.boundary_density; end
    if ~isfield(params, 'random_points'), params.random_points = default_params.random_points; end
    if ~isfield(params, 'noise_std'), params.noise_std = default_params.noise_std; end
    if ~isfield(params, 'edge_reduction'), params.edge_reduction = default_params.edge_reduction; end
    if ~isfield(params, 'surface_reduction'), params.surface_reduction = default_params.surface_reduction; end
    if ~isfield(params, 'use_dbscan'), params.use_dbscan = default_params.use_dbscan; end
    if ~isfield(params, 'dbscan_epsilon'), params.dbscan_epsilon = default_params.dbscan_epsilon; end
    if ~isfield(params, 'dbscan_minpts'), params.dbscan_minpts = default_params.dbscan_minpts; end

    if nargin < 5
        visualize = false;
    end

    % edge points generation
    edges = [
             faces(:, [1, 2]);
             faces(:, [2, 3]);
             faces(:, [3, 1])
             ];
    edges = sort(edges, 2);
    edges = unique(edges, 'rows');

    if isempty(gcp('nocreate'))
        parpool('threads');
    end

    edge_indices = randperm(size(edges, 1));
    edge_indices = edge_indices(1:floor(size(edges, 1) / params.edge_reduction));
    edge_points = cell(length(edge_indices), 1);

    parfor idx = 1:length(edge_indices)
        i = edge_indices(idx);
        v1 = vertices(edges(i, 1), :);
        v2 = vertices(edges(i, 2), :);

        edge_length = norm(v2 - v1);
        num_points = max(2, ceil(edge_length * params.edge_density));

        % generate points with slight randomization
        t = linspace(0, 1, num_points)';
        base_points = v1 + t .* (v2 - v1);

        % add small random perturbations
        noise = randn(num_points, 3) * params.noise_std;
        edge_points{idx} = base_points + noise;
    end

    edge_points = cell2mat(edge_points);

    % surface points generation
    face_indices = randperm(size(faces, 1));
    face_indices = face_indices(1:floor(size(faces, 1) / params.surface_reduction));
    surface_points = cell(length(face_indices), 1);

    parfor idx = 1:length(face_indices)
        i = face_indices(idx);
        v1 = vertices(faces(i, 1), :);
        v2 = vertices(faces(i, 2), :);
        v3 = vertices(faces(i, 3), :);

        % calculate triangle area and number of points
        edge1 = v2 - v1;
        edge2 = v3 - v1;
        area = 0.5 * norm(cross(edge1, edge2));
        num_points = max(1, ceil(area * params.surface_density));

        % generate random barycentric coordinates
        r1 = rand(num_points, 1);
        r2 = rand(num_points, 1);

        % convert to barycentric coordinates
        u = 1 - sqrt(r1);
        v = sqrt(r1) .* (1 - r2);
        w = sqrt(r1) .* r2;

        % generate base points
        base_points = u .* v1 + v .* v2 + w .* v3;

        % add small random perturbations
        noise = randn(num_points, 3) * params.noise_std;
        surface_points{idx} = base_points + noise;
    end

    surface_points = cell2mat(surface_points);

    % boundary and volume points generation
    volume_size = diff(env_dims, 1, 2);
    boundary_points = cell(6, 1); % 6 faces of the bounding box

    for i = 1:3

        for j = 1:2
            face_area = prod(volume_size([1:i - 1, i + 1:3]));
            num_points = ceil(face_area * params.boundary_density);

            points = zeros(num_points, 3);
            points(:, i) = env_dims(i, j);

            for k = [1:i - 1, i + 1:3]
                points(:, k) = env_dims(k, 1) + rand(num_points, 1) * volume_size(k);
            end

            % add small inward-facing perturbations
            noise = randn(num_points, 3) * params.noise_std;

            if j == 1
                noise(:, i) = abs(noise(:, i)); % inward for min face
            else
                noise(:, i) = -abs(noise(:, i)); % inward for max face
            end

            idx = (i - 1) * 2 + j;
            boundary_points{idx} = points + noise;
        end

    end

    boundary_points = cell2mat(boundary_points);

    % volume points
    volume = prod(volume_size);
    num_volume_points = ceil(volume * params.volume_density) + params.random_points;

    volume_points = zeros(num_volume_points, 3);

    for i = 1:3
        volume_points(:, i) = env_dims(i, 1) + rand(num_volume_points, 1) * volume_size(i);
    end

    % combine all points
    all_points = [edge_points; surface_points; boundary_points; volume_points];

    if params.use_dbscan

        try
            idx = dbscan(all_points, params.dbscan_epsilon, params.dbscan_minpts);
            valid_points = idx ~= -1;
            all_points = all_points(valid_points, :);
            clusters = unique(idx(idx ~= -1));

            if visualize
                fprintf('DBSCAN Statistics:\n');
                fprintf('Original points: %d\n', size(idx, 1));
                fprintf('Points after clustering: %d\n', sum(valid_points));
                fprintf('Number of clusters: %d\n', length(clusters));
                fprintf('Noise points removed: %d\n', sum(idx == -1));
            end

        catch e
            warning('DBSCAN clustering failed: %s\nProceeding with unclustered points.', e.message);
        end

    end

    if visualize
        figure;

        if params.use_dbscan
            pt_cloud = pointCloud(all_points);
            pcshow(pt_cloud);
            title(sprintf('total points w/ DBSCAN: %d)', size(all_points, 1)));
        else
            pt_cloud = pointCloud(all_points);
            pcshow(pt_cloud);
            title(sprintf('total points %d)', size(all_points, 1)));
        end

        xlabel('x'); ylabel('y'); zlabel('z');
        grid on;
    end

end
```

## File: datasets/get_ray_chan.m
```
function [interaction_points, ray_coeffs] = get_ray_chan(rays, freqs, method)
    % GET_RAY_CHAN Extract channel coefficients from ray tracing results
    %
    % Description:
    %   Computes channel coefficients and interaction points for each ray path
    %   using either Shooting and Bouncing Ray (SBR) method or Free Space
    %   Path Loss (FSPL) calculations.
    %
    % Inputs:
    %   rays   - Array of ray objects from ray tracer
    %   freqs  - Vector of frequencies in Hz for channel computation
    %   method - String specifying computation method: "sbr" or "fspl"
    %
    % Outputs:
    %   interaction_points - [3xN] matrix of final interaction point coordinates for N paths
    %   ray_coeffs         - [FxN] complex matrix of channel coefficients for F frequencies and N paths
    %
    % Example:
    %   rays = rayTrace(tx, rx, environment);
    %   freqs = linspace(2.4e9, 2.5e9, 10);
    %   [interaction_points, ray_coeffs] = get_ray_chan(rays, freqs, "sbr");

    n_paths = length(rays);
    n_freqs = length(freqs);

    interaction_points = zeros(3, n_paths);
    ray_coeffs = zeros(n_freqs, n_paths);

    for i = 1:n_paths

        if rays(i).LineOfSight
            points = [rays(i).TransmitterLocation rays(i).ReceiverLocation];
        else
            points = [rays(i).TransmitterLocation rays(i).Interactions.Location rays(i).ReceiverLocation];
        end

        interaction_points(:, i) = points(:, end - 1);

        for j = 1:n_freqs
            f = freqs(j);

            if strcmp(method, "sbr")
                path_loss = rays(i).PathLoss;
                phase = rays(i).PhaseShift;
            else
                prop_dist = rays(i).PropagationDistance;
                path_loss = fspl(prop_dist, f);
                phase = 2 * pi * f * prop_dist / physconst("lightspeed");
            end

            ray_coeffs(j, i) = 10 ^ (-path_loss / 20) * exp(-1j * phase);
        end
    end
end
```

## File: datasets/ray_marching.m
```
function [step_lengths, ray_points] = ray_marching(rays)
    % RAY_MARCHING Extract ray path information from ray tracing results
    %
    % Description:
    %   Extracts step lengths and interaction points along ray paths from
    %   ray tracing results, handling both line-of-sight and multi-bounce paths.
    %
    % Inputs:
    %   rays - Array of ray objects from ray tracer containing path information
    %
    % Outputs:
    %   step_lengths - Cell array containing lengths of each ray segment
    %   ray_points   - Cell array containing coordinates of ray interaction points
    %
    % Example:
    %   rays = rayTrace(tx, rx, environment);
    %   [step_lengths, ray_points] = ray_marching(rays);

    num_rays = length(rays);
    step_lengths = cell(num_rays, 1);
    ray_points = cell(num_rays, 1);

    for i = 1:num_rays

        if rays(i).LineOfSight
            points = [rays(i).TransmitterLocation rays(i).ReceiverLocation];
        else
            points = [rays(i).TransmitterLocation rays(i).Interactions.Location rays(i).ReceiverLocation];
        end

        ray_points{i} = points;
        point_diffs = diff(points, 1, 2);
        step_lengths{i} = sqrt(sum(point_diffs .^ 2));
    end

end
```

## File: datasets/generate_csi.m
```
function [H, AoD, AoA] = generate_csi(rays, fc, cfg, txArray, rxArray, method, scenario, use_single_sc, sc_idx)
    % GENERATE_CSI Generate a spatially-consistent, frequency-selective MIMO channel
    %
    % Description:
    %   Generates a MIMO channel tensor H from ray tracing data.
    %   Computes the channel response for each subcarrier in the OFDM/NR signal.
    %   The channel tensor is computed using the steering vectors of the transmit
    %   and receive arrays, and the path loss and phase shift of each ray.
    %
    % Inputs:
    %   rays          - Ray objects array from ray tracing
    %   fc            - Center frequency (Hz)
    %   cfg           - OFDM/NR carrier configuration struct
    %   txArray       - Transmit array (phased.URA or phased.IsotropicAntennaElement)
    %   rxArray       - Receive array (phased.ULA or phased.IsotropicAntennaElement)
    %   method        - 'sbr' to use ray.PathLoss/PhaseShift, 'fspl' for pure FSPL
    %   scenario      - "indoor" or "outdoor"
    %   use_single_sc - true to pick one subcarrier via sc_idx, false for all
    %   sc_idx        - Subcarrier index if use_single_sc==true
    %
    % Outputs:
    %   H   - Nt x Nr x Nsc channel tensor (complex)
    %   AoD - 2 x Nrays matrix of departure [az;el] in degrees
    %   AoA - 2 x Nrays matrix of arrival   [az;el] in degrees
    %
    % Example:
    %   [H, AoD, AoA] = generate_csi(rays, fc, cfg, txArray, rxArray, 'sbr', "outdoor", true, 1);

    if nargin < 8, use_single_sc = false; end
    if nargin < 9, sc_idx = []; end

    is_siso = isa(txArray, 'phased.IsotropicAntennaElement') && isa(rxArray, 'phased.IsotropicAntennaElement');

    if scenario == "indoor"
        ofdmInfo   = wlanNonHTOFDMInfo('L-LTF', cfg.ChannelBandwidth);
        actIdx     = ofdmInfo.ActiveFrequencyIndices;
        sc_sp      = wlanSampleRate(cfg.ChannelBandwidth) / ofdmInfo.FFTLength;
        if use_single_sc
            if isempty(sc_idx), sc_idx = ceil(numel(actIdx)/2); end
            freqs = fc + actIdx(sc_idx)*sc_sp;
        else
            freqs = fc + actIdx*sc_sp;
        end
    else
        sc_sp      = cfg.SubcarrierSpacing * 1e3;
        totalSC    = cfg.NSizeGrid * 12;
        actIdx     = -totalSC/2 : totalSC/2-1;
        if use_single_sc
            if isempty(sc_idx), sc_idx = ceil(numel(actIdx)/2); end
            freqs = fc + actIdx(sc_idx)*sc_sp;
        else
            freqs = fc + actIdx*sc_sp;
        end
    end
    Nsc = numel(freqs);

    if is_siso
        Nt = 1;
        Nr = 1;
    else
        Nt = prod(txArray.Size);
        Nr = rxArray.NumElements;
    end
    H  = zeros(Nt, Nr, Nsc);

    if ~is_siso
        svTx = phased.SteeringVector( ...
            'SensorArray',            txArray, ...
            'PropagationSpeed',       physconst('LightSpeed'), ...
            'IncludeElementResponse', false );
        svRx = phased.SteeringVector( ...
            'SensorArray',            rxArray, ...
            'PropagationSpeed',       physconst('LightSpeed'), ...
            'IncludeElementResponse', false );
    end

    numRays = numel(rays);
    AoD     = zeros(2, numRays);
    AoA     = zeros(2, numRays);

    for r = 1:numRays
        ray  = rays(r);
        dist = ray.PropagationDistance;
        tau  = dist / physconst('LightSpeed');

        if ray.LineOfSight
            pts = [ray.TransmitterLocation, ray.ReceiverLocation];
        else
            pts = [ray.TransmitterLocation, ray.Interactions.Location, ray.ReceiverLocation];
        end

        % departure
        vecTx      = pts(:,2) - pts(:,1);
        [azT, elT] = cart2sph(vecTx(1), vecTx(2), vecTx(3));
        azT = rad2deg(azT);  elT = rad2deg(elT);
        AoD(:,r) = [azT; elT];

        % arrival
        vecRx      = pts(:,end) - pts(:,end-1);
        [azR, elR] = cart2sph(vecRx(1), vecRx(2), vecRx(3));
        azR = rad2deg(azR);  elR = rad2deg(elR);
        AoA(:,r) = [azR; elR];

        if ~is_siso
            aTx = svTx(fc, [azT; elT]);
            aRx = svRx(fc, [azR; elR]);
            array_gain = aTx * aRx.';
        else
            array_gain = 1; % isotropic antennas have gain of 1 (0 dBi)
        end

        for k = 1:Nsc
            f = freqs(k);
            if strcmp(method, 'sbr')
                pl0   = ray.PathLoss;
                phi0  = ray.PhaseShift;
                pl    = pl0;
                phase = phi0 + 2*pi*(f - fc)*tau;
            else
                pl    = fspl(dist, f);
                phase = 2*pi*f*tau;
            end
            h = 10^(-pl/20) * exp(-1j*phase);
            H(:,:,k) = H(:,:,k) + array_gain * h;
        end
    end
end
```

## File: datasets/outdoor.m
```
%% environment setup
close all force; clear; clc;
plot_rays = false;
visualize = false;

% load both file formats
stl_file = "models/intersection_and_buildings.stl";
mapFileName = "models/intersection_and_buildings/IntersectionAndBuildings.glb";
[stl_data, ~] = stlread(stl_file);

if visualize
    viewer = siteviewer("SceneModel", mapFileName, "ShowEdges", false, "ShowOrigin", false);
end

% environment dimensions setup from stl
vertices = stl_data.Points;
faces = stl_data.ConnectivityList;

xy_offset = 0.1;
z_offset = 0.1;
min_z = max(0, min(vertices(:, 3)));
env_dims = [
            [min(vertices(:, 1)) + xy_offset, max(vertices(:, 1)) - xy_offset];
            [min(vertices(:, 2)) + xy_offset, max(vertices(:, 2)) - xy_offset];
            [min_z, max(vertices(:, 3)) - z_offset]
            ];

% point cloud generation params
pc_params = struct();
pc_params.edge_density = 0.43;
pc_params.surface_density = 0.06;
pc_params.volume_density = 0;
pc_params.boundary_density = 0;
pc_params.random_points = 0;
pc_params.noise_std = 0;
pc_params.edge_reduction = 1;
pc_params.surface_reduction = 1;

%% generate point cloud
point_cloud = generate_pc(vertices, faces, pc_params, env_dims, visualize);

%% System config
fc = 6e9;
lambda = physconst("lightspeed") / fc;

% OFDM parameters
carrier = nrCarrierConfig;
carrier.SubcarrierSpacing = 15;
carrier.NSizeGrid = 52;
cfg = carrier;

% extra config
use_single_sc = true;
sc_idx = [];
use_siso = false;

if use_siso
    % single-input single-output
    txArray = phased.IsotropicAntennaElement();
    rxArray = phased.IsotropicAntennaElement();

    num_tx_ant = 1;
    num_rx_ant = 1;
else
    % multiple-input multiple-output
    txArray = phased.URA("Size", [8 8], "ElementSpacing", lambda / 2);
    rxArray = phased.ULA("NumElements", 2, "ElementSpacing", lambda / 2);

    num_tx_ant = prod(txArray.Size);
    num_rx_ant = rxArray.NumElements;
end

%% AP setup
AP = txsite("cartesian", ...
    "Antenna", txArray, ...
    "AntennaPosition", [20; 38; 20], ...
    "TransmitterFrequency", fc, ...
    "TransmitterPower", 10);

%% user setup
approx_target_users = 24316;

% seed
S = RandStream("mt19937ar", "Seed", 17);
RandStream.setGlobalStream(S);

user_params.check_building_collision = true;
user_params.check_user_collision = true;
user_params.separation_distance = 1;
[Users, actual_users] = create_users(env_dims, approx_target_users, rxArray, user_params, []);

if actual_users < approx_target_users
    error('Failed to create all requested users. Only created %d out of %d users.', ...
        actual_users, approx_target_users);
end

%% RT simulation
method = "sbr"; % "image" | "sbr"
max_refs = 2;

pm = propagationModel("raytracing", ...
    "Method", method, ...
    "CoordinateSystem", "cartesian", ...
    "MaxNumDiffractions", 1, ...
    "MaxNumReflections", max_refs, ...
    "UseGPU", "on");

rays = raytrace(AP, Users, pm, "Map", mapFileName, "Type", "pathloss");

% filter users to keep only those with valid rays
valid_user_mask = ~cellfun(@isempty, rays);
Users = Users(valid_user_mask);
rays = rays(valid_user_mask);
num_users = sum(valid_user_mask);

if num_users < approx_target_users
    warning('Only %d out of %d users had valid rays, discarding the rest.', ...
        num_users, approx_target_users);
end

%% visualize
if visualize
    show(AP, "ShowAntennaHeight", false)
    show(Users, "ShowAntennaHeight", false)

    if plot_rays
        for userIdx = 1:(num_users / 20) % ~20 % rays
            plot(rays{userIdx}, "Colormap", jet)
            pause(0.05)
        end
    end
end

%% extract positions
rx_positions = zeros(3, num_users);

for userIdx = 1:num_users
    rx_positions(:, userIdx) = Users(userIdx).AntennaPosition;
end

%% CSI collection
if use_single_sc
    numSubcarriers = 1;
    sc_spacing = cfg.SubcarrierSpacing * 1e3;
    total_scs = cfg.NSizeGrid * 12;
    activeFreqIndices = (-total_scs / 2:total_scs / 2 - 1);
    
    if isempty(sc_idx)
        sc_idx = ceil(length(activeFreqIndices) / 2);
    end
    freqs = fc + activeFreqIndices(sc_idx) * sc_spacing;
else
    numSubcarriers = cfg.NSizeGrid * 12;
    sc_spacing = cfg.SubcarrierSpacing * 1e3;
    activeFreqIndices = (-numSubcarriers / 2:numSubcarriers / 2 - 1);
    freqs = fc + activeFreqIndices * sc_spacing;
end

H = zeros(num_users, num_tx_ant, num_rx_ant, numSubcarriers);
AoD_all = cell(num_users, 1);
AoA_all = cell(num_users, 1);
path_loss = zeros(num_users, 1);
path_loss_per_ray = cell(num_users, 1);

for userIdx = 1:num_users
    [H(userIdx, :, :, :), AoD_all{userIdx}, AoA_all{userIdx}] = ...
        generate_csi(rays{userIdx}, fc, cfg, txArray, rxArray, method, 'outdoor', use_single_sc, sc_idx);
    path_loss(userIdx) = mean([rays{userIdx}.PathLoss]);
    path_loss_per_ray{userIdx} = [rays{userIdx}.PathLoss];
end

% check for null values in channel matrix
if any(isnan(H(:)))
    warning('Channel matrix contains NaN values!');
end

%% ray marching and per-ray information (for NeWRF comparison)
ray_steps = cell(num_users, 1);
ray_points = cell(num_users, 1);
ray_interactions = cell(num_users, 1);
ray_coefficients = cell(num_users, 1);

for userIdx = 1:num_users
    [ray_steps{userIdx}, ray_points{userIdx}] = ray_marching(rays{userIdx});
    [ray_interactions{userIdx}, ray_coefficients{userIdx}] = get_ray_chan(rays{userIdx}, freqs, method);
end

% check for null values in ray marching results
if any(cellfun(@(x) any(cellfun(@(y) any(isnan(y(:))), x)), ray_steps)) || ...
        any(cellfun(@(x) any(cellfun(@(y) any(isnan(y(:))), x)), ray_points)) || ...
        any(cellfun(@(x) any(isnan(x(:))), ray_coefficients))
    warning('Ray marching results contain NaN values!');
end

%% save
output_dir = "outputs";
mkdir(output_dir);

[~, mapname] = fileparts(mapFileName);

% create dataset
dataset = struct();

dataset.config.tx_antennas = num_tx_ant;
dataset.config.rx_antennas = num_rx_ant;
dataset.config.frequency = fc;
dataset.config.wavelength = lambda;
dataset.config.num_users = num_users;
dataset.config.use_siso = use_siso;

dataset.environment.dimensions = env_dims;
dataset.environment.point_cloud = point_cloud;
dataset.environment.pc_params = pc_params;

dataset.nodes.ap_position = AP.AntennaPosition';
dataset.nodes.users_positions = rx_positions;

dataset.channel.H = H;
% NOTE: AoD is unneeded as it is primarily related to the txsite
% dataset.channel.AoD = AoD_all;
dataset.channel.path_loss = path_loss;

% truncate data (max 10 paths)
[AoA_trunc, AoD_trunc, path_loss_per_ray_trunc] = truncate_data(AoA_all, AoD_all, path_loss_per_ray, 10);
dataset.channel.AoA = AoA_trunc;
dataset.channel.AoD = AoD_trunc;
dataset.channel.path_loss_per_ray = path_loss_per_ray_trunc;

dataset.channel.ray_steps = ray_steps;
dataset.channel.ray_points = ray_points;
dataset.channel.ray_interactions = ray_interactions;
dataset.channel.ray_coefficients = ray_coefficients;
dataset.channel.frequencies = freqs;

sc_str = '';

if use_single_sc
    sc_str = sprintf('_sc%d', sc_idx);
end

filename = sprintf('%s/iab_%dx%d_%du_%.1fghz_%sRT%s.mat', ...
    output_dir, ...
    num_tx_ant, ...
    num_rx_ant, ...
    num_users, ...
    fc / 1e9, ...
    method, ...
    sc_str);

save(filename, 'dataset', '-v7.3');

function [AoA_trunc, AoD_trunc, path_loss_trunc] = truncate_data(AoA, AoD, path_loss, max_paths)
    num_users = length(AoA);
    AoA_trunc = cell(num_users, 1);
    AoD_trunc = cell(num_users, 1);
    path_loss_trunc = cell(num_users, 1);
    
    for i = 1:num_users
        if ~isempty(AoA{i})
            % sort paths by path loss
            [sorted_pl, sort_idx] = sort(path_loss{i});
            sorted_AoA = AoA{i}(:, sort_idx);
            sorted_AoD = AoD{i}(:, sort_idx);
            
            % take top max_paths with lowest path loss
            num_paths = min(length(sorted_pl), max_paths);
            AoA_trunc{i} = sorted_AoA(:, 1:num_paths);
            AoD_trunc{i} = sorted_AoD(:, 1:num_paths);
            path_loss_trunc{i} = sorted_pl(1:num_paths);
        else
            AoA_trunc{i} = [];
            AoD_trunc{i} = [];
            path_loss_trunc{i} = [];
        end
    end
end
```

## File: engine/__init__.py
```python
# engine/__init__.py

from .render_channel import render_channel
```

## File: datasets/wireless_dataset.py
```python
# datasets/wireless_dataset.py
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from pymatreader import read_mat
from torch.utils.data import Dataset, random_split


class WirelessDataset(Dataset):
    """A dataset class for MIMO/SISO channel data.

    Handles loading and processing of wireless channel data from .mat files,
    including channel matrix, positions, and config. Point cloud and env_dims
    are loaded if present but not used initialization. Infers SISO/MIMO from
    dataset config unless explicitly overridden during loading (not typical).

    Args:
        data_path (str): Path to the .mat dataset file
        train (bool, optional): If True, returns training set, else test set
        train_ratio (float, optional): Ratio of data for training (default: 0.8)
        seed (int, optional): Random seed for train/test split
    """

    def __init__(
        self,
        data_path: str,
        train: bool = True,
        train_ratio: float = 0.8,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.seed = seed if seed is not None else 42
        generator = torch.Generator().manual_seed(self.seed)
        np.random.seed(self.seed)

        print(f"Loading dataset from: {self.data_path}")
        try:
            mat_data = read_mat(str(self.data_path))
            if "dataset" not in mat_data:
                raise KeyError("Loaded .mat file does not contain 'dataset' key.")
            data = mat_data["dataset"]
        except Exception as e:
            print(f"Error loading MAT file: {e}")
            raise

        print("Processing dataset...")
        self._process_data(data)
        print("Dataset processing complete.")

        total_size = self.num_users
        if total_size == 0:
            raise ValueError("Dataset contains no users/samples.")

        train_size = int(total_size * train_ratio)
        test_size = total_size - train_size

        if train_size == 0 or test_size == 0:
            print(
                f"Warning: train_ratio {train_ratio} resulted in zero samples for train/test split. Adjusting."
            )
            if total_size >= 2:
                train_size = max(1, train_size)
                test_size = total_size - train_size
            else:
                train_size = 1 if train else 0
                test_size = 1 - train_size

        self.indices = list(range(total_size))
        train_indices, test_indices = random_split(
            self.indices, [train_size, test_size], generator=generator
        )

        self.active_indices = train_indices if train else test_indices
        print(f"{'Training' if train else 'Test'} set size: {len(self.active_indices)}")

    def _process_data(self, data):
        """Extracts and processes data from the loaded dictionary."""
        self._store_config(data["config"])

        if "environment" in data and "point_cloud" in data["environment"]:
            self.point_cloud_data = torch.from_numpy(
                data["environment"]["point_cloud"]
            ).float()

        else:
            self.point_cloud_data = None

        if "environment" in data and "dimensions" in data["environment"]:
            self.env_dims = torch.from_numpy(data["environment"]["dimensions"]).float()

        else:
            self.env_dims = None

        if "nodes" in data and "ap_position" in data["nodes"]:
            tx_pos_raw = data["nodes"]["ap_position"]
            self.tx_position = torch.from_numpy(np.array(tx_pos_raw)).float().squeeze()
            if self.tx_position.shape != (3,):
                raise ValueError(
                    f"Unexpected transmitter position shape: {tx_pos_raw.shape}"
                )

        else:
            raise ValueError(
                "Transmitter position ('ap_position') not found in dataset."
            )

        if "nodes" in data and "users_positions" in data["nodes"]:
            rx_pos_raw = data["nodes"]["users_positions"]
            if rx_pos_raw.shape[0] != 3:
                raise ValueError(
                    f"Expected receiver positions shape (3, K), got {rx_pos_raw.shape}"
                )
            self.rx_positions = torch.from_numpy(rx_pos_raw.T).float()

            num_users_from_pos = self.rx_positions.shape[0]
            if num_users_from_pos != self.num_users:
                print(
                    f"Warning: num_users mismatch. Config: {self.num_users}, Rx Positions: {num_users_from_pos}. Using {num_users_from_pos}."
                )
                self.num_users = num_users_from_pos
        else:
            raise ValueError(
                "Receiver positions ('users_positions') not found in dataset."
            )

        if "channel" in data and "H" in data["channel"]:
            H_raw = data["channel"]["H"]
            if isinstance(H_raw, dict) and "real" in H_raw and "imag" in H_raw:
                H_real = np.array(H_raw["real"])
                H_imag = np.array(H_raw["imag"])
                if H_real.dtype.kind not in "iufc" or H_imag.dtype.kind not in "iufc":
                    raise TypeError("Real/Imag parts of H are not numeric.")
                H_complex = H_real + 1j * H_imag
                H_tensor = torch.from_numpy(H_complex).to(torch.complex64)
            elif isinstance(H_raw, np.ndarray) and np.iscomplexobj(H_raw):
                H_tensor = torch.from_numpy(H_raw).to(torch.complex64)
            else:
                raise TypeError(
                    f"Unsupported format for H: {type(H_raw)}. Expected complex numpy array or dict with 'real'/'imag'."
                )

            print(f"Raw H tensor shape from MAT: {H_tensor.shape}")
            expected_leading_dim = self.num_users
            target_shape = (expected_leading_dim, self.num_tx_ant, self.num_rx_ant)

            if H_tensor.dim() == 4:
                num_sc = H_tensor.shape[-1]
                sc_idx = num_sc // 2
                print(
                    f"Multiple subcarriers ({num_sc}) detected. Selecting middle subcarrier index {sc_idx}."
                )
                H_selected = H_tensor[:expected_leading_dim, :, :, sc_idx]
                if H_selected.shape == target_shape:
                    self.channel_matrix = H_selected
                elif H_selected.numel() == np.prod(target_shape):
                    print(
                        f"Warning: Attempting flexible reshape of H (from 4D) from {H_selected.shape} to {target_shape}"
                    )
                    self.channel_matrix = H_selected.reshape(target_shape)
                else:
                    raise ValueError(
                        f"Channel matrix shape mismatch (from 4D). Expected {target_shape} or compatible, got {H_selected.shape}."
                    )

            elif H_tensor.dim() == 3:
                H_selected = H_tensor[:expected_leading_dim, :, :]
                if H_selected.shape == target_shape:
                    self.channel_matrix = H_selected
                elif H_selected.numel() == np.prod(target_shape):
                    print(
                        f"Warning: Attempting flexible reshape of H (from 3D) from {H_selected.shape} to {target_shape}"
                    )
                    self.channel_matrix = H_selected.reshape(target_shape)
                else:
                    raise ValueError(
                        f"Channel matrix shape mismatch (from 3D). Expected {target_shape} or compatible, got {H_selected.shape}."
                    )

            elif H_tensor.dim() == 1:
                if self.is_siso:
                    H_selected = H_tensor[:expected_leading_dim]
                    if H_selected.shape[0] == expected_leading_dim:
                        print("Reshaping SISO H from (N_user,) to (N_user, 1, 1)")
                        self.channel_matrix = H_selected.reshape(target_shape)
                    else:
                        raise ValueError(
                            f"Channel matrix shape mismatch (from 1D). Expected ({expected_leading_dim},), got {H_selected.shape}."
                        )
                else:
                    raise ValueError(
                        f"H tensor has dimension 1, but config is not SISO (Nt={self.num_tx_ant}, Nr={self.num_rx_ant})."
                    )

            elif H_tensor.dim() == 2:
                if self.is_siso and H_tensor.shape[1] == 1:
                    H_selected = H_tensor[:expected_leading_dim, :]
                    print("Reshaping SISO H from (N_user, 1) to (N_user, 1, 1)")
                    self.channel_matrix = H_selected.reshape(target_shape)
                elif H_tensor.numel() == np.prod(target_shape):
                    print(
                        f"Warning: H tensor has dimension 2 ({H_tensor.shape}). Attempting flexible reshape to {target_shape}"
                    )

                    self.channel_matrix = H_tensor[:expected_leading_dim, :].reshape(
                        target_shape
                    )
                else:
                    raise ValueError(
                        f"H tensor has dimension 2 ({H_tensor.shape}), but it's not compatible with SISO or MIMO target shape {target_shape}."
                    )

            else:
                raise ValueError(
                    f"Unexpected H tensor dimensions: {H_tensor.dim()}. Expected 1, 2, 3, or 4."
                )

            print(f"Processed channel matrix shape: {self.channel_matrix.shape}")

        else:
            raise ValueError("Channel matrix ('H') not found in dataset.")

    def _store_config(self, config):
        """Stores configuration values from the dataset and infers SISO."""
        try:
            self.num_tx_ant = int(config["tx_antennas"])
            self.num_rx_ant = int(config["rx_antennas"])
            self.frequency = float(config["frequency"])
            self.wavelength = float(config["wavelength"])
            self.num_users = int(config["num_users"])
            self.config_use_siso = config.get("use_siso", False)

            self.is_siso = (
                self.num_tx_ant == 1 and self.num_rx_ant == 1
            ) or self.config_use_siso
            if self.is_siso:

                self.num_tx_ant = 1
                self.num_rx_ant = 1

            print(
                f"Dataset Config: Nt={self.num_tx_ant}, Nr={self.num_rx_ant}, Freq={self.frequency/1e9:.2f}GHz, Lambda={self.wavelength:.4f}m, NumUsers={self.num_users}, IsSISO={self.is_siso}"
            )

        except KeyError as e:
            print(f"Error: Missing key in dataset config: {e}")
            raise
        except ValueError as e:
            print(f"Error: Invalid value type in dataset config: {e}")
            raise

    def get_point_cloud(self) -> Optional[torch.Tensor]:
        """Get the loaded point cloud data if available."""
        return self.point_cloud_data

    def get_env_dims(self) -> Optional[torch.Tensor]:
        """Get environment dimensions if available."""
        return self.env_dims

    def get_tx_position(self) -> torch.Tensor:
        """Get the transmitter position."""
        return self.tx_position

    def get_metadata(self) -> dict:
        """Returns essential metadata."""
        return {
            "num_tx_ant": self.num_tx_ant,
            "num_rx_ant": self.num_rx_ant,
            "frequency": self.frequency,
            "wavelength": self.wavelength,
            "is_siso": self.is_siso,
            "tx_position": self.tx_position,
            "env_dims": self.env_dims,
        }

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, idx):
        """Retrieves a single sample for the active split."""
        original_idx = self.active_indices[idx]
        return {
            "rx_position": self.rx_positions[original_idx],
            "channel_matrix": self.channel_matrix[original_idx],
            "index": original_idx,
        }
```

## File: datasets/dataloader.py
```python
# datasets/dataloader.py

from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .wireless_dataset import WirelessDataset


def collate_wireless_batch(batch: list) -> Dict[str, Any]:
    """Collate function for wireless dataset batches.

    Args:
        batch (list): List of dataset items ({'rx_position': tensor, 'channel_matrix': tensor, 'index': int}).

    Returns:
        Collated batch with stacked tensors.
    """
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], int):
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = [item[key] for item in batch]

    return collated


def get_wireless_dataloader(
    data_path: str,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    train: bool = True,
    train_ratio: float = 0.8,
    seed: Optional[int] = None,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for wireless dataset.

    Args:
        data_path (str): Path to the dataset file.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of workers for data loading.
        shuffle (bool): Whether to shuffle the data (typically True for train).
        train (bool): Whether to load training or test set.
        train_ratio (float): Ratio of data to use for training.
        seed (int, optional): Random seed for train/test split.
        drop_last (bool): Whether to drop the last incomplete batch (typically True for train).
        pin_memory (bool): Whether to use pinned memory for faster GPU transfer.

    Returns:
        The configured data loader.
    """
    dataset = WirelessDataset(
        data_path,
        train=train,
        train_ratio=train_ratio,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_wireless_batch,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def get_dataloaders(
    data_path: str,
    batch_size: int = 1,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    seed: Optional[int] = None,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create training and validation DataLoaders for wireless dataset.

    Args:
        data_path (str): Path to the dataset file.
        batch_size (int): Number of samples per batch for training loader.
        num_workers (int): Number of workers for data loading.
        train_ratio (float): Ratio of data to use for training.
        seed (int, optional): Random seed for train/test split.
        pin_memory (bool): Use pinned memory.

    Returns:
        A tuple of (training loader, validation loader, dataset metadata).
        Validation loader always uses batch_size=1 and shuffle=False.
    """
    train_loader = get_wireless_dataloader(
        data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        train=True,
        train_ratio=train_ratio,
        seed=seed,
        drop_last=True,
        pin_memory=pin_memory,
    )

    val_loader = get_wireless_dataloader(
        data_path,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        train=False,
        train_ratio=train_ratio,
        seed=seed,
        drop_last=False,
        pin_memory=pin_memory,
    )

    metadata = train_loader.dataset.get_metadata()

    return train_loader, val_loader, metadata
```

## File: datasets/indoor.m
```
%% environment setup
close all force; clear; clc;
plot_rays = false;
visualize = false;

mapFileName = "models/conference.stl";
[stl_data, ~] = stlread(mapFileName);

if visualize
    viewer = siteviewer("SceneModel", mapFileName, "Transparency", 0.25);
end

% environment dimensions setup
vertices = stl_data.Points;
faces = stl_data.ConnectivityList;

xy_offset = 0.1;
z_offset = 0.1;
min_z = max(0, min(vertices(:, 3)));
env_dims = [
            [min(vertices(:, 1)) + xy_offset, max(vertices(:, 1)) - xy_offset];
            [min(vertices(:, 2)) + xy_offset, max(vertices(:, 2)) - xy_offset];
            [min_z, max(vertices(:, 3)) - z_offset]
            ];

% point cloud generation params
pc_params = struct();
pc_params.edge_density = 2.1;
pc_params.surface_density = 1.6;
pc_params.volume_density = 0;
pc_params.boundary_density = 87.3;
pc_params.random_points = 0;
pc_params.noise_std = 0;
pc_params.edge_reduction = 1;
pc_params.surface_reduction = 1;

%% generate point cloud
point_cloud = generate_pc(vertices, faces, pc_params, env_dims, visualize);

%% system config
fc = 5e9;
lambda = physconst("lightspeed") / fc;

% OFDM parameters
cfg = wlanNonHTConfig;
cfg.ChannelBandwidth = 'CBW80';

% extra config
use_single_sc = true;
sc_idx = [];
use_siso = true;

if use_siso
    % single-input single-output
    txArray = phased.IsotropicAntennaElement();
    rxArray = phased.IsotropicAntennaElement();

    num_tx_ant = 1;
    num_rx_ant = 1;
else
    % multiple-input multiple-output
    txArray = phased.URA("Size", [4 4], "ElementSpacing", lambda / 2);
    rxArray = phased.ULA("NumElements", 2, "ElementSpacing", lambda / 2);

    num_tx_ant = prod(txArray.Size);
    num_rx_ant = rxArray.NumElements;
end

%% AP setup
AP = txsite("cartesian", ...
    "Antenna", txArray, ...
    "AntennaPosition", [-1.5; 0.0; 2.1], ... % Positioned near ceiling
    "TransmitterFrequency", fc, ...
    "TransmitterPower", 0.05);

%% user setup
approx_target_users = 12518;

% seed
S = RandStream("mt19937ar", "Seed", 17);
RandStream.setGlobalStream(S);

user_params = struct();
user_params.check_building_collision = false;
user_params.check_user_collision = true;
[Users, actual_users] = create_users(env_dims, approx_target_users, rxArray, user_params, []);

if actual_users < approx_target_users
    error('Failed to create all requested users. Only created %d out of %d users.', actual_users, approx_target_users);
end

%% RT simulation
method = "sbr"; % "image" | "sbr"
max_refs = 1;

pm = propagationModel("raytracing", ...
    "Method", method, ...
    "CoordinateSystem", "cartesian", ...
    "SurfaceMaterial", "wood", ...
    "TerrainMaterial", "wood", ...
    "MaxNumReflections", max_refs, ...
    "UseGPU", "auto");

rays = raytrace(AP, Users, pm, "Map", mapFileName);

% filter users to keep only those with valid rays
valid_user_mask = ~cellfun(@isempty, rays);
Users = Users(valid_user_mask);
rays = rays(valid_user_mask);
num_users = sum(valid_user_mask);

if num_users < approx_target_users
    warning('Only %d out of %d users had valid rays, discarding the rest.', ...
        num_users, approx_target_users);
end

%% visualize
if visualize
    show(AP, "ShowAntennaHeight", false)
    show(Users, "ShowAntennaHeight", false)

    if plot_rays

        for userIdx = 1:(num_users / 20) % ~20 % rays
            plot(rays{userIdx}, "Colormap", jet)
            pause(0.05)
        end

    end

end

%% extract positions
rx_positions = zeros(3, num_users);

for userIdx = 1:num_users
    rx_positions(:, userIdx) = Users(userIdx).AntennaPosition;
end

%% CSI collection
ofdmInfo = wlanNonHTOFDMInfo('L-LTF', cfg.ChannelBandwidth);
activeIndices = ofdmInfo.ActiveFrequencyIndices;

if use_single_sc

    if isempty(sc_idx)
        sc_idx = ceil(length(activeIndices) / 2);
    end

    numSubcarriers = 1;
    sc_spacing = wlanSampleRate(cfg.ChannelBandwidth) / ofdmInfo.FFTLength;
    freqs = fc + activeIndices(sc_idx) * sc_spacing;
else
    numSubcarriers = length(activeIndices);
    sc_spacing = wlanSampleRate(cfg.ChannelBandwidth) / ofdmInfo.FFTLength;
    freqs = fc + activeIndices * sc_spacing;
end

H = zeros(num_users, num_tx_ant, num_rx_ant, numSubcarriers);
AoD_all = cell(num_users, 1);
AoA_all = cell(num_users, 1);
path_loss = zeros(num_users, 1);
path_loss_per_ray = cell(num_users, 1);

for userIdx = 1:num_users
    [H(userIdx, :, :, :), AoD_all{userIdx}, AoA_all{userIdx}] = ...
        generate_csi(rays{userIdx}, fc, cfg, txArray, rxArray, method, 'indoor', use_single_sc, sc_idx);
    path_loss(userIdx) = mean([rays{userIdx}.PathLoss]);
    path_loss_per_ray{userIdx} = [rays{userIdx}.PathLoss];
end

% check for null values in channel matrix
if any(isnan(H(:)))
    warning('Channel matrix contains NaN values!');
end

%% ray marching and per-ray information (for NeWRF comparison)
ray_steps = cell(num_users, 1);
ray_points = cell(num_users, 1);
ray_interactions = cell(num_users, 1);
ray_coefficients = cell(num_users, 1);

for userIdx = 1:num_users
    [ray_steps{userIdx}, ray_points{userIdx}] = ray_marching(rays{userIdx});
    [ray_interactions{userIdx}, ray_coefficients{userIdx}] = get_ray_chan(rays{userIdx}, freqs, method);
end

% check for null values in ray marching results
if any(cellfun(@(x) any(cellfun(@(y) any(isnan(y(:))), x)), ray_steps)) || ...
        any(cellfun(@(x) any(cellfun(@(y) any(isnan(y(:))), x)), ray_points)) || ...
        any(cellfun(@(x) any(isnan(x(:))), ray_coefficients))
    warning('Ray marching results contain NaN values!');
end

%% save
output_dir = "outputs";
mkdir(output_dir);

[~, mapname] = fileparts(mapFileName);

% create dataset
dataset = struct();

dataset.config.tx_antennas = num_tx_ant;
dataset.config.rx_antennas = num_rx_ant;
dataset.config.frequency = fc;
dataset.config.wavelength = lambda;
dataset.config.num_users = num_users;
dataset.config.use_siso = use_siso;

dataset.environment.dimensions = env_dims;
dataset.environment.point_cloud = point_cloud;
dataset.environment.pc_params = pc_params;

dataset.nodes.ap_position = AP.AntennaPosition';
dataset.nodes.users_positions = rx_positions;

dataset.channel.H = H;
% NOTE: AoD is unneeded as it is primarily related to the txsite
% dataset.channel.AoD = AoD_all;
dataset.channel.path_loss = path_loss;

% truncate data (max 10 paths)
[AoA_trunc, AoD_trunc, path_loss_per_ray_trunc] = truncate_data(AoA_all, AoD_all, path_loss_per_ray, 10);
dataset.channel.AoA = AoA_trunc;
dataset.channel.AoD = AoD_trunc;
dataset.channel.path_loss_per_ray = path_loss_per_ray_trunc;

dataset.channel.ray_steps = ray_steps;
dataset.channel.ray_points = ray_points;
dataset.channel.ray_interactions = ray_interactions;
dataset.channel.ray_coefficients = ray_coefficients;
dataset.channel.frequencies = freqs;

sc_str = '';

if use_single_sc
    sc_str = sprintf('_sc%d', sc_idx);
end

filename = sprintf('%s/conf_%dx%d_%du_%.1fghz_%sRT%s.mat', ...
    output_dir, ...
    num_tx_ant, ...
    num_rx_ant, ...
    num_users, ...
    fc / 1e9, ...
    method, ...
    sc_str);

save(filename, 'dataset', '-v7.3');

function [AoA_trunc, AoD_trunc, path_loss_trunc] = truncate_data(AoA, AoD, path_loss, max_paths)
    num_users = length(AoA);
    AoA_trunc = cell(num_users, 1);
    AoD_trunc = cell(num_users, 1);
    path_loss_trunc = cell(num_users, 1);
    
    for i = 1:num_users
        if ~isempty(AoA{i})
            % sort paths by path loss
            [sorted_pl, sort_idx] = sort(path_loss{i});
            sorted_AoA = AoA{i}(:, sort_idx);
            sorted_AoD = AoD{i}(:, sort_idx);
            
            % take top max_paths with lowest path loss
            num_paths = min(length(sorted_pl), max_paths);
            AoA_trunc{i} = sorted_AoA(:, 1:num_paths);
            AoD_trunc{i} = sorted_AoD(:, 1:num_paths);
            path_loss_trunc{i} = sorted_pl(1:num_paths);
        else
            AoA_trunc{i} = [];
            AoD_trunc{i} = [];
            path_loss_trunc{i} = [];
        end
    end
end
```

## File: models/gaussian_model.py
```python
# models/gaussian_model.py

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from utils import (
    build_covariance_inverse,
    build_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
)

from .networks import ContributionDecoderNetwork


class GaussianChannelFieldModel(nn.Module):
    """
    Gaussian Channel Field (GCF) model.

    Represents the wireless environment using 3D Gaussians, each holding
    geometric parameters and latent features encoding channel contributions.
    """

    def __init__(
        self,
        num_tx_ant: int,
        num_rx_ant: int,
        latent_dim: int,
        decoder_hidden_dim: int = 64,
        decoder_num_layers: int = 3,
        initial_gaussians: int = 30000,
        init_opacity_value: float = 0.1,
        init_scale_value: float = 0.02,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__()
        self.num_tx_ant = num_tx_ant
        self.num_rx_ant = num_rx_ant
        self.latent_dim = latent_dim
        self.device = device

        self._xyz = nn.Parameter(torch.empty(0, 3, device=device))
        self._rotation = nn.Parameter(torch.empty(0, 4, device=device))
        self._scaling = nn.Parameter(torch.empty(0, 3, device=device))
        self._latent_features = nn.Parameter(torch.empty(0, latent_dim, device=device))
        self._base_activations = nn.Parameter(torch.empty(0, 1, device=device))

        self.contribution_decoder = ContributionDecoderNetwork(
            latent_dim=latent_dim,
            output_dim=2 * num_tx_ant * num_rx_ant,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
        ).to(device)

        self.initial_gaussians = initial_gaussians
        self.init_opacity_logit = inverse_sigmoid(
            torch.tensor(init_opacity_value, device=device)
        )
        self.init_log_scale = torch.log(torch.tensor(init_scale_value, device=device))

        self.optimizer = None
        self.lr_schedulers = {}

        self.setup_activations()

    def setup_activations(self):
        """Setup activation functions."""
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    @property
    def get_xyz(self):
        """Returns Gaussian positions."""
        return self._xyz

    @property
    def get_scaling(self):
        """Returns activated and clamped Gaussian scales."""
        return self.scaling_activation(self._scaling).clamp(min=1e-8)

    @property
    def get_rotation(self):
        """Returns normalized Gaussian rotations (quaternions)."""
        return self.rotation_activation(self._rotation, dim=-1)

    @property
    def get_latent_features(self):
        """Returns Gaussian latent features."""
        return self._latent_features

    @property
    def get_base_activations(self):
        """Returns raw Gaussian base activation logits."""
        return self._base_activations

    @property
    def get_opacity_activated(self):
        """Returns sigmoid-activated base activations."""
        return self.opacity_activation(self._base_activations)

    def get_covariance(
        self, return_inverse=False, eps=1e-6
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Computes covariance matrix Σ and optionally its inverse Σ^-1."""
        scaling = self.get_scaling
        rotation_q = self.get_rotation

        R = build_rotation(rotation_q)
        S_sq_diag = torch.diag_embed(scaling * scaling)
        covariance = R @ S_sq_diag @ R.transpose(1, 2)

        if return_inverse:
            inv_covariance = build_covariance_inverse(R, scaling, eps)
            if torch.isnan(inv_covariance).any() or torch.isinf(inv_covariance).any():
                print(
                    "Warning: NaN or Inf detected in inverse covariance. Replacing with identity."
                )
                bad_indices = torch.isnan(inv_covariance).any(dim=(1, 2)) | torch.isinf(
                    inv_covariance
                ).any(dim=(1, 2))
                identity = torch.eye(
                    3, device=self.device, dtype=inv_covariance.dtype
                ).expand(bad_indices.sum(), -1, -1)
                inv_covariance[bad_indices] = identity
            return covariance, inv_covariance
        else:
            return covariance

    def init_gaussians(
        self,
        env_dims: Optional[torch.Tensor] = None,
        num_points: Optional[int] = None,
    ):
        """
        Initializes Gaussian parameters randomly within environment dimensions
        or a default range. Does not use point clouds.
        """
        num_to_init = num_points if num_points is not None else self.initial_gaussians

        if env_dims is not None:
            print(
                f"Initializing {num_to_init} random Gaussians within environment dimensions."
            )
            env_min = env_dims[:, 0].to(self.device)
            env_max = env_dims[:, 1].to(self.device)
            if env_min.shape != (3,) or env_max.shape != (3,):
                raise ValueError(
                    f"env_dims should result in shapes (3,), got min: {env_min.shape}, max: {env_max.shape}"
                )
            xyz = (
                torch.rand(num_to_init, 3, device=self.device) * (env_max - env_min)
                + env_min
            )
        else:
            print(
                f"Initializing {num_to_init} random Gaussians (no env_dims provided, using [-1, 1] range)."
            )
            xyz = (torch.rand(num_to_init, 3, device=self.device) * 2 - 1) * 1.0

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        scales = torch.ones(num_to_init, 3, device=self.device) * self.init_log_scale
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        rots = torch.zeros((num_to_init, 4), device=self.device)
        rots[:, 0] = 1.0
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        latents = torch.randn(num_to_init, self.latent_dim, device=self.device) * 0.01
        self._latent_features = nn.Parameter(latents.requires_grad_(True))
        activations = (
            torch.ones(num_to_init, 1, device=self.device) * self.init_opacity_logit
        )
        self._base_activations = nn.Parameter(activations.requires_grad_(True))

        print(f"GCF Model initialized with {self.get_xyz.shape[0]} Gaussians.")

    def get_params(self, lr_dict: Dict[str, float]) -> list:
        """Returns parameter groups for the optimizer."""
        param_groups = [
            {"params": [self._xyz], "lr": lr_dict.get("xyz", 0.0), "name": "xyz"},
            {
                "params": [self._rotation],
                "lr": lr_dict.get("rotation", 0.0),
                "name": "rotation",
            },
            {
                "params": [self._scaling],
                "lr": lr_dict.get("scaling", 0.0),
                "name": "scaling",
            },
            {
                "params": [self._latent_features],
                "lr": lr_dict.get("latent", 0.0),
                "name": "latent",
            },
            {
                "params": [self._base_activations],
                "lr": lr_dict.get("activation", 0.0),
                "name": "activation",
            },
            {
                "params": self.contribution_decoder.parameters(),
                "lr": lr_dict.get("decoder", 0.0),
                "name": "decoder",
            },
        ]
        return param_groups

    def training_setup(self, training_args: Any):
        """Setup optimizer and learning rate schedulers."""
        lr_map = {
            "xyz": training_args.position_lr_init,
            "rotation": training_args.rotation_lr,
            "scaling": training_args.scaling_lr,
            "latent": training_args.latent_lr,
            "activation": training_args.activation_lr,
            "decoder": training_args.decoder_lr,
        }
        params = self.get_params(lr_map)

        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=training_args.adam_eps)

        self.lr_schedulers["xyz"] = get_expon_lr_func(
            lr_init=training_args.position_lr_init,
            lr_final=training_args.position_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.iterations,
        )
        for name, lr_init in lr_map.items():
            if name != "xyz":
                self.lr_schedulers[name] = lambda step, lr=lr_init: lr

    def update_learning_rate(self, iteration: int):
        """Update learning rates for all parameter groups."""
        if not self.optimizer:
            return
        for param_group in self.optimizer.param_groups:
            name = param_group["name"]
            if name in self.lr_schedulers:
                new_lr = self.lr_schedulers[name](iteration)
                param_group["lr"] = new_lr

    def save(self, filepath: Path, iteration: Optional[int] = None):
        """Save model state."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        state_dict = {
            "iteration": iteration,
            "xyz": self._xyz.detach().cpu(),
            "rotation": self._rotation.detach().cpu(),
            "scaling": self._scaling.detach().cpu(),
            "latent_features": self._latent_features.detach().cpu(),
            "base_activations": self._base_activations.detach().cpu(),
            "decoder_state_dict": self.contribution_decoder.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer else None
            ),
            "config": {
                "num_tx_ant": self.num_tx_ant,
                "num_rx_ant": self.num_rx_ant,
                "latent_dim": self.latent_dim,
                "decoder_hidden_dim": self.contribution_decoder.hidden_dim,
                "decoder_num_layers": self.contribution_decoder.num_layers,
            },
        }
        torch.save(state_dict, str(filepath))

    @classmethod
    def load(
        cls, filepath: Path, device: torch.device, training_args: Optional[Any] = None
    ):
        """Load model state."""
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")

        state_dict = torch.load(str(filepath), map_location=device)
        config = state_dict["config"]

        model = cls(
            num_tx_ant=config["num_tx_ant"],
            num_rx_ant=config["num_rx_ant"],
            latent_dim=config["latent_dim"],
            decoder_hidden_dim=config.get("decoder_hidden_dim", 64),
            decoder_num_layers=config.get("decoder_num_layers", 3),
            device=device,
        )

        model._xyz = nn.Parameter(state_dict["xyz"].to(device).requires_grad_(True))
        model._rotation = nn.Parameter(
            state_dict["rotation"].to(device).requires_grad_(True)
        )
        model._scaling = nn.Parameter(
            state_dict["scaling"].to(device).requires_grad_(True)
        )
        model._latent_features = nn.Parameter(
            state_dict["latent_features"].to(device).requires_grad_(True)
        )
        model._base_activations = nn.Parameter(
            state_dict["base_activations"].to(device).requires_grad_(True)
        )
        model.contribution_decoder.load_state_dict(state_dict["decoder_state_dict"])

        iteration = state_dict.get("iteration", 0)

        if training_args is not None:
            model.training_setup(training_args)
            if model.optimizer and state_dict.get("optimizer_state_dict"):
                try:
                    model.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
                    for state in model.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                    print("Optimizer state loaded successfully.")
                except Exception as e:
                    print(
                        f"Warning: Could not load optimizer state: {e}. Optimizer state reset."
                    )
            else:
                print("Optimizer state not found in checkpoint or optimizer not setup.")

        print(f"GCF Model loaded from {filepath} (iteration {iteration}).")
        return model, iteration
```

## File: train.py
```python
# train.py

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.dataloader import get_dataloaders
from engine.render_channel import render_channel
from models.gaussian_model import GaussianChannelFieldModel
from utils.general_utils import set_random_seed
from utils.loss import NormalizedMSELoss, calculate_snr
from utils.train_utils import compute_grad_stats, setup_logging


def parse_args():
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description="Train Gaussian Channel Field Model")

    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset file (.mat)"
    )

    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Ratio of data for training"
    )

    parser.add_argument(
        "--initial_gaussians",
        type=int,
        default=3_000,
        help="Number of Gaussians to initialize randomly",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=32,
        help="Dimension of Gaussian latent features (F)",
    )
    parser.add_argument(
        "--decoder_hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension for decoder MLP",
    )
    parser.add_argument(
        "--decoder_num_layers",
        type=int,
        default=4,
        help="Number of layers for decoder MLP (including output)",
    )

    parser.add_argument(
        "--iterations", type=int, default=30_000, help="Total training iterations"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--position_lr_init",
        type=float,
        default=1e-4,
        help="Initial LR for Gaussian positions",
    )
    parser.add_argument(
        "--position_lr_final",
        type=float,
        default=1e-6,
        help="Final LR for Gaussian positions",
    )
    parser.add_argument(
        "--position_lr_delay_mult",
        type=float,
        default=0.01,
        help="Multiplier for position LR delay phase",
    )
    parser.add_argument(
        "--rotation_lr", type=float, default=0.001, help="LR for Gaussian rotations"
    )
    parser.add_argument(
        "--scaling_lr", type=float, default=0.005, help="LR for Gaussian scaling"
    )
    parser.add_argument(
        "--latent_lr", type=float, default=0.001, help="LR for Gaussian latent features"
    )
    parser.add_argument(
        "--activation_lr",
        type=float,
        default=0.01,
        help="LR for Gaussian base activations",
    )
    parser.add_argument(
        "--decoder_lr",
        type=float,
        default=0.001,
        help="LR for the contribution decoder network",
    )
    parser.add_argument(
        "--adam_eps", type=float, default=1e-12, help="Adam optimizer epsilon"
    )
    parser.add_argument(
        "--loss_eps",
        type=float,
        default=1e-12,
        help="Epsilon for NMSE loss denominator",
    )

    parser.add_argument(
        "--lambda_latent_l1",
        type=float,
        default=0.0,
        help="L1 regularization weight for latent features (0 to disable)",
    )
    parser.add_argument(
        "--lambda_activation_l1",
        type=float,
        default=0.0,
        help="L1 regularization weight for base activations (0 to disable)",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save logs and checkpoints",
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=5,
        help="Log training metrics every N iterations",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=200,
        help="Evaluate on validation set every N iterations",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=200,
        help="Save checkpoint every N iterations",
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="Enable TensorBoard logging"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training",
    )

    args = parser.parse_args()
    return args


def evaluate(
    model: GaussianChannelFieldModel,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    wavelength: float,
    nt: int,
    nr: int,
    loss_eps: float,
) -> Dict[str, float]:
    """Evaluates the model on the validation set."""
    model.eval()
    total_loss = 0.0
    total_snr = 0.0
    count = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            rx_pos_batch = batch["rx_position"].to(device)
            h_gt_batch = batch["channel_matrix"].to(device)
            batch_size = rx_pos_batch.shape[0]

            gauss_means = model.get_xyz
            gauss_latents = model.get_latent_features
            gauss_activations = model.get_base_activations
            _, gauss_inv_covs = model.get_covariance(return_inverse=True)

            h_pred_batch = render_channel(
                rx_positions=rx_pos_batch,
                gauss_means=gauss_means,
                gauss_inv_covs=gauss_inv_covs,
                gauss_latents=gauss_latents,
                gauss_activations=gauss_activations,
                decoder_network=model.contribution_decoder,
                wavelength=wavelength,
                nt=nt,
                nr=nr,
                eps=loss_eps,
            )

            loss = criterion(h_pred_batch, h_gt_batch)
            snr = calculate_snr(loss)

            total_loss += loss.item() * batch_size
            if not torch.isinf(snr):
                total_snr += snr.item() * batch_size
            else:
                print("Warning: Infinite SNR detected during evaluation.")

            count += batch_size

    avg_loss = total_loss / count if count > 0 else 0.0
    avg_snr = total_snr / count if count > 0 else float("-inf")

    return {"val_loss": avg_loss, "val_snr_db": avg_snr}


def train(args):
    """Main training loop."""
    set_random_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    log_dir = Path(args.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = log_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    logger = setup_logging(log_dir)
    writer = SummaryWriter(str(log_dir / "tensorboard")) if args.tensorboard else None

    logger.info(f"Starting training; Log directory: {log_dir}")
    logger.info(f"Arguments: {args}")
    logger.info(f"Using device: {device}")

    logger.info("Loading data...")
    try:
        train_loader, val_loader, metadata = get_dataloaders(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
        nt = metadata["num_tx_ant"]
        nr = metadata["num_rx_ant"]
        wavelength = metadata["wavelength"]
        tx_position = metadata["tx_position"].to(device)
        env_dims = metadata["env_dims"]
        logger.info(
            f"Dataset Metadata: Nt={nt}, Nr={nr}, Freq={metadata['frequency']/1e9:.2f}GHz, Lambda={wavelength:.4f}m, IsSISO={metadata['is_siso']}"
        )
        logger.info(f"Tx Position: {tx_position.tolist()}")
        if env_dims is not None:
            logger.info(f"Environment Dimensions: {env_dims.tolist()}")
    except Exception as e:
        logger.exception(f"Failed to load data: {e}")
        if writer:
            writer.close()
        return

    logger.info("Initializing model...")
    model = GaussianChannelFieldModel(
        num_tx_ant=nt,
        num_rx_ant=nr,
        latent_dim=args.latent_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_num_layers=args.decoder_num_layers,
        initial_gaussians=args.initial_gaussians,
        device=device,
    )

    start_iteration = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        try:
            model, start_iteration = GaussianChannelFieldModel.load(
                Path(args.resume), device, args
            )
            start_iteration += 1
            logger.info(f"Resumed from iteration {start_iteration -1}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}. Starting from scratch.")
            args.resume = None

    if not args.resume:
        model.init_gaussians(
            env_dims=env_dims.to(device) if env_dims is not None else None
        )
        model.training_setup(args)

    model = model.to(device)
    logger.info(f"Model initialized with {model.get_xyz.shape[0]} Gaussians.")
    logger.info(
        f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    criterion = NormalizedMSELoss(eps=args.loss_eps).to(device)

    logger.info("Starting training...")
    progress_bar = tqdm(
        range(start_iteration, args.iterations), desc="Training Gaussians"
    )
    ema_loss = -1.0
    train_iter = iter(train_loader)

    for iteration in progress_bar:
        iter_start_time = time.time()
        model.train()
        model.update_learning_rate(iteration)

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        rx_pos_batch = batch["rx_position"].to(device)
        h_gt_batch = batch["channel_matrix"].to(device)

        gauss_means = model.get_xyz
        gauss_latents = model.get_latent_features
        gauss_activations = model.get_base_activations
        _, gauss_inv_covs = model.get_covariance(return_inverse=True)

        h_pred_batch = render_channel(
            rx_positions=rx_pos_batch,
            gauss_means=gauss_means,
            gauss_inv_covs=gauss_inv_covs,
            gauss_latents=gauss_latents,
            gauss_activations=gauss_activations,
            decoder_network=model.contribution_decoder,
            wavelength=wavelength,
            nt=nt,
            nr=nr,
            eps=args.loss_eps,
        )

        loss = criterion(h_pred_batch, h_gt_batch)
        total_loss = loss
        l1_latent_loss = torch.tensor(0.0, device=device)
        l1_activation_loss = torch.tensor(0.0, device=device)

        if args.lambda_latent_l1 > 0:
            l1_latent_loss = torch.mean(torch.abs(model.get_latent_features))
            total_loss = total_loss + args.lambda_latent_l1 * l1_latent_loss
        if args.lambda_activation_l1 > 0:
            l1_activation_loss = torch.mean(torch.abs(model.get_base_activations))
            total_loss = total_loss + args.lambda_activation_l1 * l1_activation_loss

        model.optimizer.zero_grad()
        total_loss.backward()

        found_nan_grad = False
        for param in model.parameters():
            if param.grad is not None and (
                torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
            ):
                logger.warning(
                    f"NaN or Inf gradient detected at iteration {iteration}. Skipping optimizer step."
                )
                found_nan_grad = True
                break

        if not found_nan_grad:
            grad_stats = compute_grad_stats(model)
            model.optimizer.step()
        else:
            model.optimizer.zero_grad()
            grad_stats = {}

        iter_time = time.time() - iter_start_time
        with torch.no_grad():
            current_loss = loss.item()
            if np.isnan(current_loss) or np.isinf(current_loss):
                logger.warning(
                    f"NaN or Inf loss detected at iteration {iteration}. Resetting EMA loss."
                )
                ema_loss = -1.0
            elif ema_loss < 0:
                ema_loss = current_loss
            else:
                ema_loss = 0.95 * ema_loss + 0.05 * current_loss

            if iteration % args.log_freq == 0:
                snr = calculate_snr(loss).item()
                log_msg = (
                    f"[{iteration}/{args.iterations}] Loss: {current_loss:.3e} | "
                    f"EMA Loss: {ema_loss:.3e} | SNR: {snr:.2f} dB | "
                    f"Time: {iter_time:.2f}s | Gaussians: {gauss_means.shape[0]}"
                )
                logger.info(log_msg)
                if grad_stats:
                    grad_log_msg = (
                        f"Grad Norm: {grad_stats['grad_norm']:.3e} | "
                        f"Mean Abs: {grad_stats['mean_abs_grad']:.3e} | "
                        f"Min: {grad_stats['min_grad']:.3e} | Max: {grad_stats['max_grad']:.3e}"
                    )
                    logger.info(grad_log_msg)

                if writer is not None:
                    writer.add_scalar("train/loss", current_loss, iteration)
                    writer.add_scalar("train/ema_loss", ema_loss, iteration)
                    writer.add_scalar("train/snr_db", snr, iteration)
                    writer.add_scalar("train/iteration_time_sec", iter_time, iteration)
                    writer.add_scalar(
                        "train/num_gaussians", gauss_means.shape[0], iteration
                    )
                    if grad_stats:
                        writer.add_scalar(
                            "grads/norm", grad_stats["grad_norm"], iteration
                        )
                        writer.add_scalar(
                            "grads/mean_abs", grad_stats["mean_abs_grad"], iteration
                        )
                    if args.lambda_latent_l1 > 0:
                        writer.add_scalar(
                            "train/loss_l1_latent", l1_latent_loss.item(), iteration
                        )
                    if args.lambda_activation_l1 > 0:
                        writer.add_scalar(
                            "train/loss_l1_activation",
                            l1_activation_loss.item(),
                            iteration,
                        )
                    if model.optimizer:
                        for i, param_group in enumerate(model.optimizer.param_groups):
                            writer.add_scalar(
                                f"lr/{param_group['name']}",
                                param_group["lr"],
                                iteration,
                            )

        if iteration % args.eval_freq == 0 and iteration > 0:
            logger.info(f"Starting evaluation at iteration {iteration}")
            eval_metrics = evaluate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                wavelength=wavelength,
                nt=nt,
                nr=nr,
                loss_eps=args.loss_eps,
            )
            logger.info(f"Validation Loss: {eval_metrics['val_loss']:.3e}")
            logger.info(f"Validation SNR: {eval_metrics['val_snr_db']:.2f} dB")

            if writer is not None:
                writer.add_scalar(
                    "validation/loss", eval_metrics["val_loss"], iteration
                )
                writer.add_scalar(
                    "validation/snr_db", eval_metrics["val_snr_db"], iteration
                )

        if iteration % args.checkpoint_freq == 0 or iteration == args.iterations - 1:
            checkpoint_path = checkpoints_dir / f"checkpoint_{iteration:07d}.pt"
            model.save(checkpoint_path, iteration=iteration)
            if iteration != args.iterations - 1 and iteration != 0:
                logger.info(f"Checkpoint saved to {checkpoint_path}")

    final_model_path = log_dir / "final_model.pt"
    model.save(final_model_path, iteration=args.iterations - 1)
    logger.info(f"Training completed. Final model saved to {final_model_path}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
```

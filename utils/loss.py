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

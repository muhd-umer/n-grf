# core/loss.py

import torch


def nmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Normalized Mean Square Error loss for complex wireless channel matrices.

    Args:
        pred: Predicted channel tensor
        target: Target (ground truth) channel tensor

    Returns:
        NMSE loss value
    """
    mse = torch.sum(torch.abs(pred - target) ** 2)
    normalization = torch.sum(torch.abs(target) ** 2)
    normalization = torch.clamp(normalization, min=1e-10)

    return mse / normalization


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss for wireless channel matrices.

    Args:
        pred: Predicted channel tensor
        target: Target (ground truth) channel tensor

    Returns:
        L1 loss value
    """
    return torch.abs(pred - target).mean()

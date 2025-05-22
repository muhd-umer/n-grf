# utils/loss.py

import warnings

import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate MSE loss for complex tensors.

    Args:
        pred: Predicted complex tensor
        target: Target complex tensor

    Returns:
        MSE loss as a scalar tensor
    """
    if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("Both pred and target must be torch tensors.")

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    diff = pred - target
    squared_diff = torch.abs(diff) ** 2
    return torch.mean(squared_diff)


def nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Calculate NMSE loss for complex tensors.

    Args:
        pred: Predicted complex tensor
        target: Target complex tensor
        eps: Small value to prevent division by zero

    Returns:
        NMSE loss as a scalar tensor
    """
    if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("Both pred and target must be torch tensors.")

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    diff = pred - target
    numerator = torch.sum(torch.abs(diff) ** 2)
    denominator = torch.sum(torch.abs(target) ** 2)

    if denominator <= eps:
        warnings.warn(
            f"Warning: Target signal power is near zero ({denominator.item():.2e}). "
            f"NMSE calculation may be unstable."
        )
        return torch.tensor(float("inf"), device=pred.device, dtype=torch.float32)

    nmse = numerator / (denominator + eps)
    return nmse


def get_snr_fnmse(nmse: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Calculate SNR in dB from NMSE.

    Args:
        nmse: NMSE value (scalar tensor)
        eps: Small value to prevent log(0)

    Returns:
        SNR in dB as a scalar tensor
    """
    if not isinstance(nmse, torch.Tensor) or nmse.numel() != 1:
        raise TypeError("nmse must be a scalar tensor.")

    nmse_clamped = torch.clamp(nmse, min=eps)
    snr = -10.0 * torch.log10(nmse_clamped)

    return snr.float()


def get_snr_fmse(
    mse_loss: torch.Tensor, target: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    """Calculate SNR in dB from MSE loss and target signal.

    Args:
        mse_loss: MSE loss (scalar tensor)
        target: Target complex tensor
        eps: Small value to prevent division by zero

    Returns:
        SNR in dB as a scalar tensor
    """
    if not isinstance(mse_loss, torch.Tensor) or mse_loss.numel() != 1:
        raise TypeError("mse_loss must be a scalar tensor.")
    if not isinstance(target, torch.Tensor):
        raise TypeError("target must be a tensor.")

    signal_power = torch.mean(torch.abs(target) ** 2)

    if signal_power <= eps:
        warnings.warn(
            f"Warning: Target signal power is near zero ({signal_power.item():.2e}). "
            f"SNR calculation may be unstable."
        )
        return torch.tensor(float("-inf"), device=mse_loss.device, dtype=torch.float32)

    noise_power = torch.clamp(mse_loss, min=eps)
    snr = 10.0 * torch.log10(signal_power / noise_power)

    return snr.float()

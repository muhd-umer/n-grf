# utils/loss.py

import numpy as np
import torch
import torch.nn as nn

# removed calculate_nmse and NormalizedMSELoss as they are replaced by MSE


def calculate_snr(
    mse_loss: torch.Tensor, target: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    """Calculate SNR in dB from MSE loss and target tensor (magnitudes).

    SNR = 10 * log10( Mean(Target^2) / MSE )

    Args:
        mse_loss: The calculated Mean Squared Error loss (scalar tensor).
        target: The ground truth target tensor (e.g., normalized magnitudes).
        eps: Small value for numerical stability, especially for clamping MSE.

    Returns:
        SNR value in dB (scalar tensor). Returns -inf if target power is zero
        or MSE is non-positive after clamping.
    """
    if not isinstance(mse_loss, torch.Tensor) or mse_loss.numel() != 1:
        raise TypeError("mse_loss must be a scalar tensor.")
    if not isinstance(target, torch.Tensor):
        raise TypeError("target must be a tensor.")

    # ensure inputs are on the same device
    target = target.to(device=mse_loss.device)

    # calculate average signal power: Mean(Target^2)
    # target should be normalized magnitude [0, 1]
    signal_power = torch.mean(target**2)

    # handle case where signal power is zero or negative
    if signal_power <= eps:
        print(
            f"Warning: Target signal power is near zero ({signal_power.item():.2e}). SNR calculation may be unstable or -inf."
        )
        # return -inf or a very small number depending on desired behavior
        return torch.tensor(float("-inf"), device=mse_loss.device)

    # noise power is the MSE loss
    noise_power = mse_loss

    # clamp noise power (MSE) to avoid log10(0) or log10(negative)
    # also handles potential NaN/Inf in mse_loss, though they should ideally be handled earlier
    noise_power_clamped = torch.clamp(noise_power, min=eps)

    # calculate SNR
    snr = 10.0 * torch.log10(signal_power / noise_power_clamped)

    # check for final NaN/Inf (e.g., if signal_power / noise_power_clamped is negative or zero)
    if torch.isnan(snr) or torch.isinf(snr):
        print(
            f"Warning: Final SNR is NaN or Inf (Signal Power: {signal_power.item():.2e}, Clamped MSE: {noise_power_clamped.item():.2e}). Returning -Inf."
        )
        return torch.tensor(float("-inf"), device=mse_loss.device)

    return snr

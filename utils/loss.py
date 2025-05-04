# utils/loss.py

import warnings

import torch


def calculate_snr(
    mse_loss: torch.Tensor, target: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    """Calculate SNR in dB from MSE loss and target tensor."""

    if not isinstance(mse_loss, torch.Tensor) or mse_loss.numel() != 1:
        raise TypeError("mse_loss must be a scalar tensor.")
    if not isinstance(target, torch.Tensor):
        raise TypeError("target must be a tensor.")

    target = target.to(device=mse_loss.device)
    signal_power = torch.mean(target**2)

    if signal_power <= eps:
        warnings.warn(
            f"Warning: Target signal power is near zero ({signal_power.item():.2e}). SNR calculation may be unstable or -inf."
        )
        return torch.tensor(float("-inf"), device=mse_loss.device)

    noise_power = mse_loss
    noise_power_clamped = torch.clamp(noise_power, min=eps)
    snr = 10.0 * torch.log10(signal_power / noise_power_clamped)

    if torch.isnan(snr) or torch.isinf(snr):
        warnings.warn(
            f"Warning: Final SNR is NaN or Inf (Signal Power: {signal_power.item():.2e}, Clamped MSE: {noise_power_clamped.item():.2e}). Returning -Inf."
        )
        return torch.tensor(float("-inf"), device=mse_loss.device)

    return snr

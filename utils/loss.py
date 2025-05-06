# utils/loss.py

import warnings

import torch


def calculate_snr(
    mse_loss: torch.Tensor, target_magnitude: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    """Calculate SNR in dB from MSE loss (calculated on magnitudes) and target magnitude tensor."""

    if not isinstance(mse_loss, torch.Tensor) or mse_loss.numel() != 1:
        raise TypeError("mse_loss must be a scalar tensor.")
    if not isinstance(target_magnitude, torch.Tensor):
        raise TypeError("target_magnitude must be a tensor.")
    if torch.is_complex(target_magnitude):
        warnings.warn(
            "Warning: calculate_snr received complex target, expected magnitude."
        )
        target_magnitude = torch.abs(target_magnitude)

    target_magnitude = target_magnitude.to(device=mse_loss.device, dtype=torch.float32)

    signal_power = torch.mean(target_magnitude**2)

    if signal_power <= eps:
        warnings.warn(
            f"Warning: Target signal power (magnitude squared) is near zero ({signal_power.item():.2e}). SNR calculation may be unstable or -inf."
        )

        return torch.tensor(float("-inf"), device=mse_loss.device, dtype=torch.float32)

    noise_power = mse_loss
    noise_power_clamped = torch.clamp(noise_power, min=eps)

    snr = 10.0 * torch.log10(signal_power / noise_power_clamped)

    if torch.isnan(snr) or torch.isinf(snr):

        if noise_power_clamped <= eps and signal_power > eps:

            warnings.warn(
                f"Warning: Clamped MSE is near zero ({noise_power_clamped.item():.2e}) with non-zero signal power ({signal_power.item():.2e}). Returning large positive SNR."
            )

            return torch.tensor(100.0, device=mse_loss.device, dtype=torch.float32)
        else:
            warnings.warn(
                f"Warning: Final SNR is NaN or Inf (Signal Power: {signal_power.item():.2e}, Clamped MSE: {noise_power_clamped.item():.2e}). Returning -Inf."
            )
            return torch.tensor(
                float("-inf"), device=mse_loss.device, dtype=torch.float32
            )

    return snr.float()

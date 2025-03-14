# utils/prop_utils.py

import torch


def compute_path_loss(distances: torch.Tensor, wavelength: float) -> torch.Tensor:
    """Compute free-space path loss based on distances.

    Args:
        distances: Distances from transmitter to points [N]
        wavelength: Signal wavelength in meters

    Returns:
        Path loss values [N, 1]
    """
    PI = 3.14159265358979323846
    distances = torch.clamp(distances, min=1e-6)
    path_loss = wavelength / (4.0 * PI * distances)
    return path_loss.unsqueeze(1)


def compute_phase_rotation(distances: torch.Tensor, wavelength: float) -> torch.Tensor:
    """Compute phase rotation based on propagation distance.

    Args:
        distances: Distances from transmitter to points [N]
        wavelength: Signal wavelength in meters

    Returns:
        Phase rotation values [N, 1] in range [0, 2π)
    """
    PI = 3.14159265358979323846
    phase_raw = -2.0 * PI * distances / wavelength
    phase_sin = torch.sin(phase_raw)
    phase_cos = torch.cos(phase_raw)
    phase = torch.atan2(phase_sin, phase_cos)

    return phase.unsqueeze(1)

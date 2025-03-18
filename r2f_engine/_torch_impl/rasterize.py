# r2f_engine/_torch_impl/rasterize.py

from typing import Tuple

import torch

from utils.transform_utils import inverse_2d_covariance

from .transforms import project_to_channel_space


@torch.jit.script
def compute_gaussian_influence(
    uv: torch.Tensor, cov2d: torch.Tensor, num_tx: int, num_rx: int
) -> torch.Tensor:
    """Compute influence of Gaussians on channel matrix elements using Mahalanobis distance

    Args:
        uv: Channel matrix coordinates of Gaussians [N, 2]
        cov2d: 2D covariance matrices of Gaussians [N, 2, 2]
        num_tx: Number of transmit antennas
        num_rx: Number of receive antennas

    Returns:
        Tensor of shape [N, num_tx, num_rx] with Gaussian influence on each channel element
    """
    N = uv.shape[0]
    inv_cov2d, _ = inverse_2d_covariance(cov2d)

    i_coords = torch.arange(num_tx, device=uv.device) + 0.5
    j_coords = torch.arange(num_rx, device=uv.device) + 0.5

    influences = torch.zeros((N, num_tx, num_rx), device=uv.device)

    for i in range(num_tx):
        for j in range(num_rx):
            antenna_pos_x = i_coords[i]
            antenna_pos_y = j_coords[j]

            d_x = uv[:, 0] - antenna_pos_x
            d_y = uv[:, 1] - antenna_pos_y

            mahalanobis_dist = (
                inv_cov2d[:, 0] * d_x**2
                + inv_cov2d[:, 2] * d_y**2
                + 2 * inv_cov2d[:, 1] * d_x * d_y
            )

            influences[:, i, j] = torch.exp(-0.5 * mahalanobis_dist)

    return influences


@torch.jit.script
def compute_channel(
    attenuation: torch.Tensor,
    phase_rotation: torch.Tensor,
    xyz_rx_distance: torch.Tensor,
    wavelength: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute wireless channel of Gaussians based on wireless physics

    Args:
        attenuation: Learned attenuation amplitude from neural network [N, 1]
        phase_rotation: Learned phase rotation from neural network [N, 1]
        tx_rx_distance: Distance from transmitter to receiver [1]
        xyz_rx_distance: Distance from each Gaussian to receiver [N, 1]
        wavelength: Signal wavelength in meters

    Returns:
        Tuple of tensors (real_part, imag_part) each of shape [N, 1]
    """
    PI: float = 3.14159265358979323846
    path_loss = wavelength / (4.0 * PI * xyz_rx_distance.unsqueeze(1))
    phase_shift = -2.0 * PI * xyz_rx_distance.unsqueeze(1) / wavelength

    total_attenuation = attenuation * path_loss
    total_phase = phase_rotation + phase_shift

    real_part = total_attenuation * torch.cos(total_phase)
    imag_part = total_attenuation * torch.sin(total_phase)

    return real_part, imag_part


@torch.jit.script
def alpha_blending(
    influences: torch.Tensor,
    contributions: torch.Tensor,
    opacity: torch.Tensor,
    sort_indices: torch.Tensor,
    num_tx: int,
    num_rx: int,
) -> torch.Tensor:
    """Perform alpha blending to form the channel matrix

    Args:
        influences: Gaussian influence on each channel element [N, num_tx, num_rx]
        contributions: Wireless contribution of each Gaussian [N, 1]
        opacity: Opacity of each Gaussian [N, 1]
        sort_indices: Indices to sort Gaussians by distance to receiver
        num_tx: Number of transmit antennas
        num_rx: Number of receive antennas

    Returns:
        Complex channel matrix of shape [num_tx, num_rx]
    """
    contributions_real = torch.real(contributions)
    contributions_imag = torch.imag(contributions)

    channel_real = torch.zeros((num_tx, num_rx), device=influences.device)
    channel_imag = torch.zeros((num_tx, num_rx), device=influences.device)

    transmittance = torch.ones((num_tx, num_rx), device=influences.device)

    for idx in sort_indices:
        effective_opacity = opacity[idx] * influences[idx]

        contribution_term_real = (
            transmittance * effective_opacity * contributions_real[idx]
        )
        channel_real = channel_real + contribution_term_real

        contribution_term_imag = (
            transmittance * effective_opacity * contributions_imag[idx]
        )
        channel_imag = channel_imag + contribution_term_imag

        transmittance = transmittance * (1.0 - effective_opacity)

    cat_channel = torch.cat([channel_real, channel_imag], dim=1)

    return cat_channel


def rasterize(
    points: torch.Tensor,
    cov3d: torch.Tensor,
    attenuation: torch.Tensor,
    phase_rotation: torch.Tensor,
    opacity: torch.Tensor,
    receiver: torch.Tensor,
    transmitter: torch.Tensor,
    num_tx: int,
    num_rx: int,
    frequency: float,
) -> torch.Tensor:
    """Rasterize the channel matrix for a specific receiver position

    Args:
        points: Gaussian centers [N, 3]
        cov3d: 3D covariance matrices in compact form [N, 6]
        attenuation: Learned attenuation amplitude from neural network [N, 1]
        phase_rotation: Learned phase rotation from neural network [N, 1]
        opacity: Opacity of each Gaussian [N, 1]
        receiver: Receiver position [3]
        transmitter: Transmitter position [3]
        num_tx: Number of transmit antennas
        num_rx: Number of receive antennas
        frequency: Signal frequency in Hz

    Returns:
        Channel matrix of shape [num_tx, 2*num_rx] with real and imaginary parts
        concatenated
    """

    c = 299792458.0
    wavelength = c / frequency

    xyz_rx_distances, uv, cov2d = project_to_channel_space(
        points=points, cov3d=cov3d, receiver=receiver, num_tx=num_tx, num_rx=num_rx
    )

    sort_indices = torch.argsort(xyz_rx_distances)
    influences = compute_gaussian_influence(uv, cov2d, num_tx, num_rx)

    real_contributions, imag_contributions = compute_channel(
        attenuation,
        phase_rotation,
        xyz_rx_distances,
        wavelength,
    )

    contributions = torch.complex(real_contributions, imag_contributions)

    cat_channel = alpha_blending(
        influences, contributions, opacity, sort_indices, num_tx, num_rx
    )

    return cat_channel

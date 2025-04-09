# engine/_torch_impl/rasterize.py

from typing import Tuple

import torch

from .transforms import project_to_channel_space


@torch.jit.script
def compute_gaussian_influence(
    uv: torch.Tensor, cov2d: torch.Tensor, num_tx: int, num_rx: int
) -> torch.Tensor:
    """Compute influence of Gaussians on channel matrix elements using Mahalanobis distance

    Args:
        uv: Channel matrix coordinates [N, 2]
        cov2d: 2D covariance matrices [N, 2, 2]
        num_tx: Number of transmit antennas
        num_rx: Number of receive antennas

    Returns:
        Tensor of shape [N, num_tx, num_rx] with Gaussian influence on each channel element
    """
    N = uv.shape[0]

    inv_cov2d = torch.inverse(cov2d)
    influences = torch.empty((N, num_tx, num_rx), device=uv.device, dtype=uv.dtype)

    for i in range(num_tx):
        for j in range(num_rx):
            antenna_pos_x = i + 0.5
            antenna_pos_y = j + 0.5

            d = uv - torch.tensor(
                [antenna_pos_x, antenna_pos_y], device=uv.device, dtype=uv.dtype
            )
            d_x = d[:, 0]
            d_y = d[:, 1]

            # compute the Mahalanobis distance
            inv_00 = inv_cov2d[:, 0, 0]
            inv_01 = inv_cov2d[:, 0, 1]
            inv_10 = inv_cov2d[:, 1, 0]
            inv_11 = inv_cov2d[:, 1, 1]
            md = d_x * (inv_00 * d_x + inv_01 * d_y) + d_y * (
                inv_10 * d_x + inv_11 * d_y
            )

            influences[:, i, j] = torch.exp(-0.5 * md)

    return influences


@torch.jit.script
def compute_channel(
    attenuation: torch.Tensor,
    phase_rotation: torch.Tensor,
    distances: torch.Tensor,
    wavelength: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute channel from features of Gaussians based on physics

    Args:
        attenuation: Learned attenuation amplitude from neural network [N, 1]
        phase_rotation: Learned phase rotation from neural network [N, 1]
        distances: Distance from each Gaussian to receiver [N]
        wavelength: Signal wavelength in meters

    Returns:
        Tuple of tensors (real_part, imag_part) each of shape [N, 1]
    """
    PI: float = 3.14159265358979323846

    path_loss = wavelength / (4.0 * PI * distances.unsqueeze(1))
    phase_shift = -2.0 * PI * distances.unsqueeze(1) / wavelength

    total_attenuation = attenuation * path_loss
    total_phase = phase_rotation + phase_shift

    real_part = total_attenuation * torch.cos(total_phase)
    imag_part = total_attenuation * torch.sin(total_phase)

    return real_part, imag_part


@torch.jit.script
def alpha_blending(
    influences: torch.Tensor,
    real_contributions: torch.Tensor,
    imag_contributions: torch.Tensor,
    opacity: torch.Tensor,
    sort_indices: torch.Tensor,
    num_tx: int,
    num_rx: int,
) -> torch.Tensor:
    """Perform alpha blending to form the channel matrix

    Args:
        influences: Gaussian influence on each channel element [N, num_tx, num_rx]
        real_contributions: Real part of wireless contributions [N, 1]
        imag_contributions: Imag part of wireless contributions [N, 1]
        opacity: Opacity of each Gaussian [N, 1]
        sort_indices: Indices to sort Gaussians by distance to receiver
        num_tx: Number of transmit antennas
        num_rx: Number of receive antennas

    Returns:
        Channel matrix of shape [num_tx, 2*num_rx] with real and imaginary parts concatenated
    """
    N = influences.shape[0]
    device = influences.device

    # initialize channel matrices and transmittance
    channel_real = torch.zeros((num_tx, num_rx), device=device, dtype=influences.dtype)
    channel_imag = torch.zeros((num_tx, num_rx), device=device, dtype=influences.dtype)
    transmittance = torch.ones((num_tx, num_rx), device=device, dtype=influences.dtype)

    for idx_pos, idx in enumerate(sort_indices):
        eff_opacity = opacity[idx] * influences[idx]

        channel_real = (
            channel_real + transmittance * eff_opacity * real_contributions[idx]
        )
        channel_imag = (
            channel_imag + transmittance * eff_opacity * imag_contributions[idx]
        )

        transmittance = transmittance * (1.0 - eff_opacity)

    return torch.cat([channel_real, channel_imag], dim=1)


def compute_cov3d(
    scaling: torch.Tensor, rotation: torch.Tensor, scale_modifier: float = 1.0
) -> torch.Tensor:
    """Compute 3D covariance matrices from scaling and rotation parameters.

    Args:
        scaling: Scaling factors (already with exponential activation applied) [N, 3]
        rotation: Quaternion rotations (already normalized) [N, 4]
        scale_modifier: Global scale modifier

    Returns:
        Covariance matrices [N, 3, 3]
    """
    scaled_scaling = scaling * scale_modifier
    scaling_mat = torch.diag_embed(scaled_scaling)
    w, x, y, z = rotation[:, 0], rotation[:, 1], rotation[:, 2], rotation[:, 3]

    # construct rotation matrices from quaternions
    R = torch.stack(
        [
            1 - 2 * y**2 - 2 * z**2,
            2 * x * y - 2 * w * z,
            2 * x * z + 2 * w * y,
            2 * x * y + 2 * w * z,
            1 - 2 * x**2 - 2 * z**2,
            2 * y * z - 2 * w * x,
            2 * x * z - 2 * w * y,
            2 * y * z + 2 * w * x,
            1 - 2 * x**2 - 2 * y**2,
        ],
        dim=1,
    ).reshape(-1, 3, 3)

    RS = torch.bmm(R, scaling_mat)
    cov3d = torch.bmm(RS, RS.transpose(1, 2))

    return cov3d


def rasterize(
    points: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    attenuation: torch.Tensor,
    phase_rotation: torch.Tensor,
    opacity: torch.Tensor,
    receiver: torch.Tensor,
    transmitter: torch.Tensor,  # included but not used
    num_tx: int,
    num_rx: int,
    frequency: float,
    scale_modifier: float = 1.0,
) -> torch.Tensor:
    """Rasterize the channel matrix for a specific receiver position

    Args:
        points: Gaussian centers [N, 3]
        scaling: Scaling factors (already with exponential activation applied) [N, 3]
        rotation: Quaternion rotations (already normalized) [N, 4]
        attenuation: Learned attenuation amplitude from neural network [N, 1]
        phase_rotation: Learned phase rotation from neural network [N, 1]
        opacity: Opacity values (already with sigmoid activation applied) [N, 1]
        receiver: Receiver position [3]
        transmitter: Transmitter position [3] (not used in current implementation)
        num_tx: Number of transmit antennas
        num_rx: Number of receive antennas
        frequency: Signal frequency in Hz
        scale_modifier: Global scaling modifier (default is 1.0)

    Returns:
        Channel matrix of shape [num_tx, 2*num_rx] with real and imaginary parts concatenated
    """
    c = 299792458.0  # speed of light in m/s
    wavelength = c / frequency

    cov3d_full = compute_cov3d(scaling, rotation, scale_modifier)

    from utils.transform_utils import strip_symmetric

    cov3d_compact = strip_symmetric(cov3d_full)

    distances, uv, cov2d = project_to_channel_space(
        points=points,
        cov3d=cov3d_compact,
        receiver=receiver,
        num_tx=num_tx,
        num_rx=num_rx,
    )
    sort_indices = torch.argsort(distances)
    influences = compute_gaussian_influence(uv, cov2d, num_tx, num_rx)
    real_contributions, imag_contributions = compute_channel(
        attenuation,
        phase_rotation,
        distances,
        wavelength,
    )

    cat_channel = alpha_blending(
        influences,
        real_contributions,
        imag_contributions,
        opacity,
        sort_indices,
        num_tx,
        num_rx,
    )

    return cat_channel

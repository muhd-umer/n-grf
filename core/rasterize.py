# core/rasterize.py

import torch

from utils.transform_utils import inverse_2d_covariance


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
    distances: torch.Tensor,
    wavelength: float,
) -> torch.Tensor:
    """Compute wireless channel of Gaussians based on wireless physics

    Args:
        attenuation: Learned attenuation amplitude from neural network [N, 1]
        phase_rotation: Learned phase rotation from neural network [N, 1]
        distances: Distances from Gaussians to receiver [N]
        wavelength: Signal wavelength in meters

    Returns:
        Complex tensor of shape [N, 1] with wireless channel of each
        Gaussian
    """
    PI: float = 3.14159265358979323846
    path_loss = wavelength / (4.0 * PI * distances.unsqueeze(1))
    phase_shift = -2.0 * PI * distances.unsqueeze(1) / wavelength

    total_attenuation = attenuation * path_loss
    total_phase = phase_rotation + phase_shift

    real_part = total_attenuation * torch.cos(total_phase)
    imag_part = total_attenuation * torch.sin(total_phase)

    return torch.complex(real_part, imag_part)


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
    channel = torch.zeros(
        (num_tx, num_rx), dtype=torch.complex64, device=influences.device
    )
    transmittance = torch.ones((num_tx, num_rx), device=influences.device)

    for idx in sort_indices:
        effective_opacity = opacity[idx] * influences[idx]
        contribution_term = transmittance * effective_opacity * contributions[idx]
        channel = channel + contribution_term
        transmittance = transmittance * (1.0 - effective_opacity)

    return channel


def format_channel_matrix(channel: torch.Tensor) -> torch.Tensor:
    """Format complex channel matrix as real and imaginary stacked parts

    Args:
        channel: Complex channel matrix of shape [num_tx, num_rx]

    Returns:
        Formatted matrix of shape [num_tx, 2*num_rx] with real and imaginary
        parts
    """
    real_part = torch.real(channel)
    imag_part = torch.imag(channel)

    formatted_channel = torch.cat([real_part, imag_part], dim=1)

    return formatted_channel


def compute_viewspace_points(
    points: torch.Tensor, receiver: torch.Tensor
) -> torch.Tensor:
    """Compute viewspace points for densification.

    Args:
        points: Gaussian centers [N, 3]
        receiver: Receiver position [3]

    Returns:
        Viewspace points with gradient tracking [N, 3]
    """
    viewspace_points = points - receiver

    distances = torch.norm(viewspace_points, dim=1, keepdim=True)
    viewspace_directions = viewspace_points / torch.clamp(distances, min=1e-10)

    viewspace_points_tensor = torch.cat([viewspace_directions[:, :2], distances], dim=1)

    return viewspace_points_tensor


def compute_2d_radii(cov2d: torch.Tensor) -> torch.Tensor:
    """Compute 2D radii from covariance matrices.

    Args:
        cov2d: 2D covariance matrices [N, 2, 2]

    Returns:
        2D radii [N]
    """
    det = torch.det(cov2d)
    trace = cov2d[:, 0, 0] + cov2d[:, 1, 1]

    discriminant = torch.sqrt(torch.clamp(trace**2 - 4 * det, min=0))
    max_eigenvalue = (trace + discriminant) / 2

    radii = torch.sqrt(max_eigenvalue)

    return radii


def rasterize(
    points: torch.Tensor,
    cov3d: torch.Tensor,
    attenuation: torch.Tensor,
    phase_rotation: torch.Tensor,
    opacity: torch.Tensor,
    receiver: torch.Tensor,
    num_tx: int,
    num_rx: int,
    frequency: float = 5e9,
    format_output: bool = True,
    return_viewspace_info: bool = True,
) -> dict:
    """Rasterize the channel matrix for a specific receiver position

    Args:
        points: Gaussian centers [N, 3]
        cov3d: 3D covariance matrices in compact form [N, 6]
        attenuation: Learned attenuation amplitude from neural network [N, 1]
        phase_rotation: Learned phase rotation from neural network [N, 1]
        opacity: Opacity of each Gaussian [N, 1]
        receiver: Receiver position [3]
        num_tx: Number of transmit antennas
        num_rx: Number of receive antennas
        frequency: Signal frequency in Hz
        format_output: Whether to format output as real/imaginary stacked parts
        return_viewspace_info: Whether to return viewspace information for densification

    Returns:
        Dictionary containing channel matrix and viewspace information (if requested)
    """
    from core.transforms import project_to_channel_space

    c = 299792458.0
    wavelength = c / frequency

    proj_dict = project_to_channel_space(
        points=points, cov3d=cov3d, receiver=receiver, num_tx=num_tx, num_rx=num_rx
    )

    uv = proj_dict["uv"]
    cov2d = proj_dict["cov2d"]
    distances = proj_dict["distances"]

    sort_indices = torch.argsort(distances)

    # compute Gaussian influence on channel elements
    influences = compute_gaussian_influence(uv, cov2d, num_tx, num_rx)
    contributions = compute_channel(attenuation, phase_rotation, distances, wavelength)

    channel = alpha_blending(
        influences, contributions, opacity, sort_indices, num_tx, num_rx
    )

    if format_output:
        channel_output = format_channel_matrix(channel)
    else:
        raise NotImplementedError("Unformatted output not supported.")

    result = {"channel": channel_output}

    if return_viewspace_info:
        visibility_filter = torch.ones(
            points.shape[0], dtype=torch.bool, device=points.device
        )

        radii = compute_2d_radii(cov2d)

        result["viewspace_info"] = {
            "visibility_filter": visibility_filter,
            "radii": radii,
            "distances": distances.squeeze(-1) if distances.dim() > 1 else distances,
        }

    return result

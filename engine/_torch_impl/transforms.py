# engine/_torch_impl/transforms.py

from typing import Dict, Tuple

import torch

from utils.transform_utils import symmetric_matrix


@torch.jit.script
def compute_distances_to_receiver(
    points: torch.Tensor, receiver: torch.Tensor
) -> torch.Tensor:
    """Compute distances between Gaussian means and a receiver

    Args:
        points: Gaussian centers [N, 3]
        receiver: Receiver position [3]

    Returns:
        Distances [N]
    """
    return torch.sqrt(torch.sum((points - receiver) ** 2, dim=1))


@torch.jit.script
def compute_spherical_coords(
    points: torch.Tensor, receiver: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute spherical coordinates for Gaussian means relative to receiver

    Args:
        points: Gaussian centers [N, 3]
        receiver: Receiver position [3]

    Returns:
        Tuple containing displacement vectors [N, 3], longitude (azimuthal
        angle) [N], and latitude (elevation angle) [N]
    """
    d = points - receiver
    r = torch.sqrt(torch.sum(d**2, dim=1))

    longitude = torch.atan2(d[:, 1], d[:, 0])
    latitude = torch.asin(torch.clamp(d[:, 2] / r, -1.0, 1.0))

    return d, longitude, latitude


@torch.jit.script
def transform_to_uniform_coords(
    longitude: torch.Tensor, latitude: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform spherical coordinates to uniform coordinates

    Args:
        longitude: Longitude in radians [N]
        latitude: Latitude in radians [N]

    Returns:
        Tuple containing normalized coordinates (s_x, s_y) in [-1, 1] range
    """
    PI: float = 3.14159265358979323846

    s_x = longitude / PI
    s_y = 2.0 * latitude / PI

    return s_x, s_y


@torch.jit.script
def map_to_channel_matrix(
    s_x: torch.Tensor, s_y: torch.Tensor, num_tx: int, num_rx: int
) -> torch.Tensor:
    """Map uniform coordinates to channel matrix coordinates

    Args:
        s_x: X coordinates in (-1, 1) range [N]
        s_y: Y coordinates in (-1, 1) range [N]
        num_tx: Number of transmit antennas
        num_rx: Number of receive antennas

    Returns:
        Channel matrix coordinates [N, 2]
    """
    u = ((s_x + 1.0) / 2.0) * float(num_tx - 1) + 0.5
    v = ((s_y + 1.0) / 2.0) * float(num_rx - 1) + 0.5

    return torch.stack([u, v], dim=1)


@torch.jit.script
def compute_jacobian(
    d: torch.Tensor, r: torch.Tensor, num_tx: int, num_rx: int
) -> torch.Tensor:
    """Compute Jacobian matrix for projection from 3D to channel matrix space

    Args:
        d: Displacement vectors [N, 3]
        r: Distances [N]
        num_tx: Number of transmit antennas
        num_rx: Number of receive antennas

    Returns:
        Jacobian matrices [N, 2, 3]
    """
    PI: float = 3.14159265358979323846

    x = d[:, 0]
    y = d[:, 1]
    z = d[:, 2]

    xy_squared = x**2 + y**2
    xy_squared = torch.clamp(xy_squared, min=1e-10)

    cos_lat = torch.sqrt(1.0 - (z / r) ** 2)
    cos_lat = torch.clamp(cos_lat, min=1e-10)

    tx_factor = float(num_tx - 1) / (2.0 * PI)
    rx_factor = float(num_rx - 1) / PI

    N = d.shape[0]
    J = torch.zeros((N, 2, 3), device=d.device, dtype=d.dtype)

    J[:, 0, 0] = tx_factor * (-y / xy_squared)
    J[:, 0, 1] = tx_factor * (x / xy_squared)

    r_cos_lat_xy = r * cos_lat * xy_squared
    r_cos_lat_xy = torch.clamp(r_cos_lat_xy, min=1e-10)

    J[:, 1, 0] = rx_factor * (z * x) / r_cos_lat_xy
    J[:, 1, 1] = rx_factor * (z * y) / r_cos_lat_xy
    J[:, 1, 2] = rx_factor / (r * cos_lat)

    return J


@torch.jit.script
def project_cov3d_to_cov2d(
    cov3d_mat: torch.Tensor, jacobian: torch.Tensor
) -> torch.Tensor:
    """Project 3D covariance matrices to 2D using Jacobian

    Args:
        cov3d_mat: Full 3D covariance matrices [N, 3, 3]
        jacobian: Jacobian matrices [N, 2, 3]

    Returns:
        2D covariance matrices [N, 2, 2]
    """
    # project using Jacobian: J * Sigma * J^T
    temp = torch.bmm(cov3d_mat, jacobian.transpose(1, 2))
    cov2d = torch.bmm(jacobian, temp)

    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3

    return cov2d


def project_to_channel_space(
    points: torch.Tensor,
    cov3d: torch.Tensor,
    receiver: torch.Tensor,
    num_tx: int,
    num_rx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project 3D Gaussians to channel matrix space for a single receiver

    Args:
        points: Gaussian centers [N, 3]
        cov3d: 3D covariance matrices in compact form [N, 6]
        receiver: Receiver position [3]
        num_tx: Number of transmit antennas
        num_rx: Number of receive antennas

    Returns:
        Tuple containing projection results, i.e., distances [N], uv
        coordinates [N, 2], and projected 2D covariance matrices [N, 2, 2]
    """
    device = points.device
    receiver = receiver.to(device)

    distances = compute_distances_to_receiver(points, receiver)
    d, longitude, latitude = compute_spherical_coords(points, receiver)
    s_x, s_y = transform_to_uniform_coords(longitude, latitude)
    uv = map_to_channel_matrix(s_x, s_y, num_tx, num_rx)
    jacobian = compute_jacobian(d, distances, num_tx, num_rx)
    cov3d_mat = symmetric_matrix(cov3d)
    cov2d = project_cov3d_to_cov2d(cov3d_mat, jacobian)

    return distances, uv, cov2d

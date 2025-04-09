# _torch_impl/transforms.py

import math
from typing import Tuple

import torch

from utils.transform_utils import symmetric_matrix


@torch.jit.script
def project_to_channel_coords(
    points: torch.Tensor, receiver: torch.Tensor, num_tx: int, num_rx: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project points directly to channel matrix coordinates

    Args:
        points: Gaussian centers [N, 3]
        receiver: Receiver position [3]
        num_tx: Number of transmit antennas
        num_rx: Number of receive antennas

    Returns:
        Tuple containing distances [N], displacement vectors [N, 3], and uv
        coordinates [N, 2]
    """
    d = points - receiver
    r = torch.sqrt(torch.sum(d**2, dim=1))

    longitude = torch.atan2(d[:, 1], d[:, 0])
    latitude = torch.asin(torch.clamp(d[:, 2] / r, -1.0, 1.0))

    # transform to uniform coordinates
    s_x = longitude / math.pi
    s_y = 2.0 * latitude / math.pi

    # map to channel matrix coordinates
    u = ((s_x + 1.0) / 2.0) * (num_tx - 1) + 0.5
    v = ((s_y + 1.0) / 2.0) * (num_rx - 1) + 0.5
    uv = torch.stack([u, v], dim=1)

    return r, d, uv


@torch.jit.script
def compute_jacobian(d: torch.Tensor, num_tx: int, num_rx: int) -> torch.Tensor:
    """Compute Jacobian matrix for projection to channel matrix space.

    Args:
        d: Displacement vectors [N, 3] (i.e. points - receiver)
        num_tx: Number of transmit antennas
        num_rx: Number of receive antennas

    Returns:
        Jacobian matrices [N, 2, 3]
    """
    x = d[:, 0]
    y = d[:, 1]
    z = d[:, 2]

    r = torch.sqrt(torch.sum(d**2, dim=1))

    xy_sq = x**2 + y**2
    xy_sq = torch.clamp(xy_sq, min=1e-10)

    cos_lat = torch.sqrt(torch.clamp(1.0 - (z / r) ** 2, min=1e-10))

    tx_factor = (num_tx - 1) / (2.0 * torch.tensor(math.pi))
    rx_factor = (num_rx - 1) / torch.tensor(math.pi)

    N = d.shape[0]
    J = torch.zeros((N, 2, 3), device=d.device, dtype=d.dtype)

    # first row: u-coordinates derivatives (longitude)
    J[:, 0, 0] = tx_factor * (-y / xy_sq)
    J[:, 0, 1] = tx_factor * (x / xy_sq)
    J[:, 0, 2] = 0.0

    # second row: v-coordinates derivatives (latitude)
    r_cos_lat_xy = r * cos_lat * xy_sq
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

    distances, d, uv = project_to_channel_coords(points, receiver, num_tx, num_rx)
    jacobian = compute_jacobian(d, num_tx, num_rx)
    cov3d_mat = symmetric_matrix(cov3d)
    cov2d = project_cov3d_to_cov2d(cov3d_mat, jacobian)

    return distances, uv, cov2d

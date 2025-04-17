# _torch_impl/transforms.py

import math
from typing import Any, Dict, Tuple

import torch


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
    r = torch.clamp(r, min=1e-6)

    longitude = torch.atan2(d[:, 1], d[:, 0])
    latitude = torch.asin(torch.clamp(d[:, 2], -1.0 + 1e-6, 1.0 - 1e-6))

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
    r = torch.clamp(r, min=1e-6)

    xy_sq = x**2 + y**2
    xy_sq = torch.clamp(xy_sq, min=1e-6)

    cos_lat = torch.sqrt(torch.clamp(1.0 - (z / r) ** 2, min=1e-6))

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
    r_cos_lat_xy = torch.clamp(r_cos_lat_xy, min=1e-6)

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


def compute_path_geometry(
    points: torch.Tensor, tx_pos: torch.Tensor, rx_pos: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute path geometry parameters for wireless propagation.

    Args:
        points: Gaussian centers [N, 3]
        tx_pos: Transmitter position [3]
        rx_pos: Receiver position [3]

    Returns:
        Tuple containing distances from TX [N], distances from RX [N], angles of
        departure [N, 2], and angles of arrival [N, 2]
    """
    vec_tx_gauss = points - tx_pos
    vec_gauss_rx = rx_pos - points

    dist_tx = torch.norm(vec_tx_gauss, dim=1).clamp(min=1e-6)
    dist_rx = torch.norm(vec_gauss_rx, dim=1).clamp(min=1e-6)

    aod_az = torch.atan2(vec_tx_gauss[:, 1], vec_tx_gauss[:, 0])
    aod_el = torch.asin(
        torch.clamp(vec_tx_gauss[:, 2] / dist_tx, -1.0 + 1e-6, 1.0 - 1e-6)
    )
    aod = torch.stack([aod_az, aod_el], dim=1)

    aoa_az = torch.atan2(vec_gauss_rx[:, 1], vec_gauss_rx[:, 0])
    aoa_el = torch.asin(
        torch.clamp(vec_gauss_rx[:, 2] / dist_rx, -1.0 + 1e-6, 1.0 - 1e-6)
    )
    aoa = torch.stack([aoa_az, aoa_el], dim=1)

    return dist_tx, dist_rx, aod, aoa


def compute_steering_vector(
    angles: torch.Tensor, array_params: Dict[str, Any], wavelength: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute steering vectors for antenna arrays.

    Args:
        angles: Angles in radians [N, 2] (azimuth, elevation)
        array_params: Dictionary with array parameters
            - type: 'ura' or 'ula'
            - size: Array dimensions
            - element_spacing: Element spacing
        wavelength: Wavelength in meters

    Returns:
        Tuple containing real and imaginary parts of steering vectors
    """
    N = angles.shape[0]
    az, el = angles[:, 0], angles[:, 1]
    array_type = array_params["type"]
    k = 2 * math.pi / wavelength

    if array_type == "ura":
        rows, cols = array_params["size"]
        spacing_x, spacing_y = array_params["element_spacing"]

        m_indices = (
            torch.arange(rows, device=az.device, dtype=az.dtype) - (rows - 1) / 2.0
        )
        n_indices = (
            torch.arange(cols, device=az.device, dtype=az.dtype) - (cols - 1) / 2.0
        )
        grid_m, grid_n = torch.meshgrid(m_indices, n_indices, indexing="ij")

        x_pos = grid_m.reshape(-1) * spacing_x
        y_pos = grid_n.reshape(-1) * spacing_y

        cos_el = torch.cos(el).unsqueeze(-1)
        cos_az = torch.cos(az).unsqueeze(-1)
        sin_az = torch.sin(az).unsqueeze(-1)

        x_pos_exp = x_pos.unsqueeze(0)
        y_pos_exp = y_pos.unsqueeze(0)

        phase = -k * cos_el * (x_pos_exp * cos_az + y_pos_exp * sin_az)

    elif array_type == "ula":
        num_ant = array_params["size"]
        spacing_d = array_params["element_spacing"]

        indices = (
            torch.arange(num_ant, device=az.device, dtype=az.dtype)
            - (num_ant - 1) / 2.0
        )
        x_pos = indices * spacing_d

        cos_el = torch.cos(el)
        cos_az = torch.cos(az)

        x_pos_exp = x_pos.unsqueeze(0)
        phase = -k * x_pos_exp * (cos_el * cos_az).unsqueeze(-1)

    else:
        raise ValueError(f"Unsupported array type: {array_type}")

    sv_real = torch.cos(phase)
    sv_imag = torch.sin(phase)

    return sv_real, sv_imag


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

    from utils.transform_utils import symmetric_matrix

    cov3d_mat = symmetric_matrix(cov3d)
    cov2d = project_cov3d_to_cov2d(cov3d_mat, jacobian)

    return distances, uv, cov2d

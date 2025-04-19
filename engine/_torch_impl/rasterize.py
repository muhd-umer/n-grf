# engine/_torch_impl/rasterize.py

from typing import Any, Dict, Tuple

import torch
from torch import nn

from models.dir_network import DirectionalNetwork
from utils.transform_utils import strip_symmetric

from .transforms import (
    compute_path_geometry,
    compute_steering_vector,
    project_to_channel_space,
)


@torch.jit.script
def compute_spatial_influence(
    uv: torch.Tensor, cov2d: torch.Tensor, num_tx: int, num_rx: int
) -> torch.Tensor:
    """Compute influence of Gaussians on channel matrix elements using Mahalanobis distance

    Args:
        uv: Channel matrix coordinates [N, 2]
        cov2d: 2D covariance matrices [N, 2, 2]
        num_tx: Number of transmit antennas
        num_rx: Number of receive antennas

    Returns:
        Tensor of shape [N, num_tx, num_rx] with Gaussian influence on each
        channel element
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

            md = torch.clamp(md, max=30.0)
            influences[:, i, j] = torch.exp(-0.5 * md)

    return influences


def compute_scattered_paths(
    gamma_real: torch.Tensor,
    gamma_imag: torch.Tensor,
    dist_tx: torch.Tensor,
    dist_rx: torch.Tensor,
    sv_tx_real: torch.Tensor,
    sv_tx_imag: torch.Tensor,
    sv_rx_real: torch.Tensor,
    sv_rx_imag: torch.Tensor,
    wavelength: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scattered path contributions to the channel.

    Args:
        gamma_real: Real scattering coefficients [N]
        gamma_imag: Imag scattering coefficients [N]
        dist_tx: Distance from TX to Gaussian [N]
        dist_rx: Distance from Gaussian to RX [N]
        sv_tx_real: TX steering vector real part [N, Nt]
        sv_tx_imag: TX steering vector imag part [N, Nt]
        sv_rx_real: RX steering vector real part [N, Nr]
        sv_rx_imag: RX steering vector imag part [N, Nr]
        wavelength: Wavelength in meters

    Returns:
        Tuple of tensors (real_part, imag_part) of scattered paths [N, Nt, Nr]
    """
    N = gamma_real.shape[0]

    dist_path = dist_tx + dist_rx
    dist_path = torch.clamp(dist_path, min=1e-8)

    alpha_amp = wavelength / (4 * torch.pi * dist_path)
    alpha_phase = -2 * torch.pi * dist_path / wavelength

    alpha_real = alpha_amp * torch.cos(alpha_phase)
    alpha_imag = alpha_amp * torch.sin(alpha_phase)

    scatter_coef_real = gamma_real * alpha_real - gamma_imag * alpha_imag
    scatter_coef_imag = gamma_real * alpha_imag + gamma_imag * alpha_real

    steering_product_real = torch.einsum(
        "bi,bj->bji", sv_rx_real, sv_tx_real
    ) + torch.einsum("bi,bj->bji", sv_rx_imag, sv_tx_imag)
    steering_product_imag = torch.einsum(
        "bi,bj->bji", sv_rx_imag, sv_tx_real
    ) - torch.einsum("bi,bj->bji", sv_rx_real, sv_tx_imag)

    scat_chan_real = (
        scatter_coef_real.view(N, 1, 1) * steering_product_real
        - scatter_coef_imag.view(N, 1, 1) * steering_product_imag
    )
    scat_chan_imag = (
        scatter_coef_real.view(N, 1, 1) * steering_product_imag
        + scatter_coef_imag.view(N, 1, 1) * steering_product_real
    )

    return scat_chan_real, scat_chan_imag


def compute_direct_path(
    tx_params: Dict[str, Any], rx_params: Dict[str, Any], wavelength: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute direct path channel component.

    Args:
        tx_params: TX array parameters dictionary (including position)
        rx_params: RX array parameters dictionary (including position)
        wavelength: Wavelength in meters

    Returns:
        Tuple of direct path channel real and imag parts [Nt, Nr]
    """
    with torch.no_grad():
        tx_pos = tx_params["position"]
        rx_pos = rx_params["position"]

        vec_tx_rx = rx_pos - tx_pos
        dist_tx_rx = torch.norm(vec_tx_rx).clamp(min=1e-8)

        alpha_fs_amp = wavelength / (4 * torch.pi * dist_tx_rx)
        alpha_fs_phase = -2 * torch.pi * dist_tx_rx / wavelength

        prop_coef_real = alpha_fs_amp * torch.cos(alpha_fs_phase)
        prop_coef_imag = alpha_fs_amp * torch.sin(alpha_fs_phase)

        aod_az = torch.atan2(vec_tx_rx[1], vec_tx_rx[0]).unsqueeze(0)
        aod_el = torch.asin(
            torch.clamp(vec_tx_rx[2] / dist_tx_rx, -1.0 + 1e-8, 1.0 - 1e-8)
        ).unsqueeze(0)
        aod = torch.cat([aod_az, aod_el], dim=0).unsqueeze(0)

        vec_rx_tx = tx_pos - rx_pos
        aoa_az = torch.atan2(vec_rx_tx[1], vec_rx_tx[0]).unsqueeze(0)
        aoa_el = torch.asin(
            torch.clamp(vec_rx_tx[2] / dist_tx_rx, -1.0 + 1e-8, 1.0 - 1e-8)
        ).unsqueeze(0)
        aoa = torch.cat([aoa_az, aoa_el], dim=0).unsqueeze(0)

        sv_tx_real, sv_tx_imag = compute_steering_vector(aod, tx_params, wavelength)
        sv_rx_real, sv_rx_imag = compute_steering_vector(aoa, rx_params, wavelength)

        steering_product_real = torch.einsum(
            "bi,bj->bji", sv_rx_real, sv_tx_real
        ) + torch.einsum("bi,bj->bji", sv_rx_imag, sv_tx_imag)
        steering_product_imag = torch.einsum(
            "bi,bj->bji", sv_rx_imag, sv_tx_real
        ) - torch.einsum("bi,bj->bji", sv_rx_real, sv_tx_imag)

        # apply coefficients
        direct_chan_real = (
            prop_coef_real * steering_product_real
            - prop_coef_imag * steering_product_imag
        )
        direct_chan_imag = (
            prop_coef_real * steering_product_imag
            + prop_coef_imag * steering_product_real
        )

        direct_chan_real = direct_chan_real.squeeze(0)
        direct_chan_imag = direct_chan_imag.squeeze(0)

    return direct_chan_real.detach(), direct_chan_imag.detach()


def weighted_superposition(
    direct_path_real: torch.Tensor,
    direct_path_imag: torch.Tensor,
    scat_path_real: torch.Tensor,
    scat_path_imag: torch.Tensor,
    opacity: torch.Tensor,
    influence: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform weighted superposition of direct and scattered paths.

    Args:
        direct_path_real: Real part of direct path [Nt, Nr]
        direct_path_imag: Imag part of direct path [Nt, Nr]
        scat_path_real: Real part of scattered paths [N, Nt, Nr]
        scat_path_imag: Imag part of scattered paths [N, Nt, Nr]
        opacity: Opacity values [N, 1]
        influence: Spatial influence values [N, Nt, Nr]

    Returns:
        Tuple of real and imaginary parts of channel [Nt, Nr]
    """
    N = scat_path_real.shape[0]

    weights = opacity.view(N, 1, 1) * influence

    # sum weighted scattered paths
    sum_scatter_real = torch.sum(weights * scat_path_real, dim=0)
    sum_scatter_imag = torch.sum(weights * scat_path_imag, dim=0)

    chan_pred_real = direct_path_real + sum_scatter_real
    chan_pred_imag = direct_path_imag + sum_scatter_imag

    return chan_pred_real, chan_pred_imag


def compute_cov3d(
    scaling: torch.Tensor, rotation: torch.Tensor, scale_modifier: float = 1.0
) -> torch.Tensor:
    """Compute 3D covariance matrices from scaling and rotation parameters.

    Args:
        scaling: Scaling factors [N, 3]
        rotation: Quaternion rotations [N, 4]
        scale_modifier: Global scale modifier

    Returns:
        Covariance matrices [N, 3, 3]
    """
    scaled_scaling = scaling * scale_modifier
    scaled_scaling = torch.clamp(scaled_scaling, min=1e-6)
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
    base_features: torch.Tensor,
    opacity: torch.Tensor,
    tx_params: Dict[str, Any],
    rx_params: Dict[str, Any],
    directional_network: DirectionalNetwork,
    dir_embedder: nn.Module,
    scale_modifier: float = 1.0,
) -> torch.Tensor:
    """Rasterize the channel matrix for a specific receiver position.

    Args:
        points: Gaussian centers [N, 3]
        scaling: Activated scaling factors [N, 3]
        rotation: Quaternion rotations [N, 4]
        base_features: Base feature vectors [N, F]
        opacity: Activated opacity values [N, 1]
        tx_params: Transmitter parameters including position
        rx_params: Receiver parameters including position
        directional_network: The network predicting gamma based on direction
        dir_embedder: Positional embedder for directions
        scale_modifier: Global scaling modifier (default is 1.0)

    Returns:
        Channel matrix of shape [num_tx, 2*num_rx] with real and imaginary parts concatenated
    """
    device = points.device
    num_tx = tx_params["num_antennas"]
    num_rx = rx_params["num_antennas"]
    tx_pos = tx_params["position"].to(device)
    rx_pos = rx_params["position"].to(device)
    frequency = tx_params["frequency"]

    c = 299792458.0
    wavelength = c / frequency

    cov3d_full = compute_cov3d(scaling, rotation, scale_modifier)
    cov3d_compact = strip_symmetric(cov3d_full)

    _, uv, cov2d = project_to_channel_space(
        points=points,
        cov3d=cov3d_compact,
        receiver=rx_pos,
        num_tx=num_tx,
        num_rx=num_rx,
    )

    influence = compute_spatial_influence(uv, cov2d, num_tx, num_rx)
    dist_tx, dist_rx, aod, aoa = compute_path_geometry(points, tx_pos, rx_pos)

    incoming_direction = points - tx_pos
    incoming_direction = torch.nn.functional.normalize(
        incoming_direction, p=2, dim=1, eps=1e-6
    )
    dir_input_in = dir_embedder(incoming_direction)

    outgoing_direction = rx_pos - points
    outgoing_direction = torch.nn.functional.normalize(
        outgoing_direction, p=2, dim=1, eps=1e-6
    )
    dir_input_out = dir_embedder(outgoing_direction)

    gamma_real, gamma_imag = directional_network(
        base_features, dir_input_in, dir_input_out
    )
    gamma_real = gamma_real.squeeze(-1)
    gamma_imag = gamma_imag.squeeze(-1)

    sv_tx_real, sv_tx_imag = compute_steering_vector(aod, tx_params, wavelength)
    sv_rx_real, sv_rx_imag = compute_steering_vector(aoa, rx_params, wavelength)

    scat_chan_real, scat_chan_imag = compute_scattered_paths(
        gamma_real,
        gamma_imag,
        dist_tx,
        dist_rx,
        sv_tx_real,
        sv_tx_imag,
        sv_rx_real,
        sv_rx_imag,
        wavelength,
    )

    direct_chan_real, direct_chan_imag = compute_direct_path(
        tx_params, rx_params, wavelength
    )
    chan_real, chan_imag = weighted_superposition(
        direct_chan_real,
        direct_chan_imag,
        scat_chan_real,
        scat_chan_imag,
        opacity,
        influence,
    )

    chan = torch.cat([chan_real, chan_imag], dim=1)

    return chan

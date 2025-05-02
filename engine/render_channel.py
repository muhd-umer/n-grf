# engine/render_channel.py

import torch
import torch.nn.functional as F

from models.networks import ContributionDecoderNetwork


@torch.jit.script
def compute_spatial_weight(
    d_vec_n: torch.Tensor,
    inv_covariance_n: torch.Tensor,
    base_activation_n: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Computes the spatial weight using the Gaussian PDF.
    w_n = o(a_n) * exp(-0.5 * d_vec_n^T * Σ_n^(-1) * d_vec_n)
    """
    d_vec_n_unsqueezed = d_vec_n.unsqueeze(1)

    exponent_term = torch.bmm(d_vec_n_unsqueezed, inv_covariance_n)
    exponent_term = torch.bmm(
        exponent_term, d_vec_n_unsqueezed.transpose(1, 2)
    ).squeeze()

    pdf_weight = torch.exp(-0.5 * torch.clamp(exponent_term, max=50.0, min=-50.0))

    activation_weight = torch.sigmoid(base_activation_n).squeeze(-1)

    weight = F.relu(activation_weight * pdf_weight)
    weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)

    return weight + eps


def render_channel(
    rx_positions: torch.Tensor,
    gauss_means: torch.Tensor,
    gauss_inv_covs: torch.Tensor,
    gauss_latents: torch.Tensor,
    gauss_activations: torch.Tensor,
    decoder_network: ContributionDecoderNetwork,
    wavelength: float,
    nt: int,
    nr: int,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Renders the complex channel matrix H for a batch of receiver positions.

    Args:
        rx_positions: Batch of receiver positions (B, 3).
        gauss_means: Gaussian means (N, 3).
        gauss_inv_covs: Gaussian inverse covariances (N, 3, 3).
        gauss_latents: Gaussian latent features (N, F).
        gauss_activations: Gaussian base activations (N, 1).
        decoder_network: Network mapping latents to complex contributions.
        wavelength: Wavelength of the signal.
        nt: Number of Tx antennas.
        nr: Number of Rx antennas.
        eps: Small value for numerical stability.

    Returns:
        Batch of predicted channel matrices (B, Nt, Nr) complex.
    """
    batch_size = rx_positions.shape[0]
    num_gaussians = gauss_means.shape[0]
    device = rx_positions.device

    rx_pos_expanded = rx_positions.unsqueeze(1)
    gauss_means_expanded = gauss_means.unsqueeze(0)

    d_vec = rx_pos_expanded - gauss_means_expanded
    d_norm = torch.norm(d_vec, dim=-1, p=2).clamp(min=eps)

    phase_shift_rad = (2.0 * torch.pi / wavelength) * d_norm

    phase_real = torch.cos(-phase_shift_rad)
    phase_imag = torch.sin(-phase_shift_rad)

    phase_real = torch.nan_to_num(phase_real, nan=0.0, posinf=0.0, neginf=0.0)
    phase_imag = torch.nan_to_num(phase_imag, nan=0.0, posinf=0.0, neginf=0.0)
    phase_shift_factor = torch.complex(phase_real, phase_imag)

    d_vec_flat = d_vec.view(-1, 3)
    inv_covs_expanded = gauss_inv_covs.repeat(batch_size, 1, 1)
    activations_expanded = gauss_activations.repeat(batch_size, 1)

    spatial_weights = compute_spatial_weight(
        d_vec_flat, inv_covs_expanded, activations_expanded, eps
    ).view(batch_size, num_gaussians)

    h_contrib_flat = decoder_network(gauss_latents)

    h_contrib_real_flat = h_contrib_flat[..., : (nt * nr)]
    h_contrib_imag_flat = h_contrib_flat[..., (nt * nr) :]

    h_contrib_real_flat = torch.nan_to_num(
        h_contrib_real_flat, nan=0.0, posinf=0.0, neginf=0.0
    )
    h_contrib_imag_flat = torch.nan_to_num(
        h_contrib_imag_flat, nan=0.0, posinf=0.0, neginf=0.0
    )

    h_contrib = torch.complex(
        h_contrib_real_flat.view(num_gaussians, nt, nr),
        h_contrib_imag_flat.view(num_gaussians, nt, nr),
    )

    h_contrib_expanded = h_contrib.unsqueeze(0)
    combined_weight = spatial_weights.unsqueeze(-1).unsqueeze(
        -1
    ) * phase_shift_factor.unsqueeze(-1).unsqueeze(-1)

    weighted_contributions = combined_weight * h_contrib_expanded
    H_pred = torch.sum(weighted_contributions, dim=1)

    return H_pred

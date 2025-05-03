# engine/render_channel.py

import torch
import torch.nn.functional as F

from models.gaussian_model import GaussianChannelFieldModel


@torch.jit.script
def compute_spatial_weight(
    d_vec_n: torch.Tensor,
    inv_covariance_n: torch.Tensor,
    base_activation_n: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Computes the spatial weight using the Gaussian PDF."""

    d_vec_n_unsqueezed = d_vec_n.unsqueeze(1)

    exponent_term = torch.bmm(d_vec_n_unsqueezed, inv_covariance_n)
    exponent_term = torch.bmm(
        exponent_term, d_vec_n_unsqueezed.transpose(1, 2)
    ).squeeze()

    pdf_weight = torch.exp(-0.5 * torch.clamp(exponent_term, max=50.0))

    activation_weight = torch.sigmoid(base_activation_n).squeeze(-1)

    weight = activation_weight * pdf_weight

    return weight + eps


def render_channel(
    rx_positions: torch.Tensor,
    model: GaussianChannelFieldModel,
    tx_position: torch.Tensor,
    wavelength: float,
    nt: int,
    nr: int,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Renders the complex channel matrix H for a batch of receiver positions.
    Uses dynamic attributes computed by the model's AttributeNetwork.

    Args:
        rx_positions: Batch of receiver positions (B, 3)
        model: The GaussianChannelFieldModel instance
        tx_position: The fixed transmitter position (3,)
        wavelength: Wavelength of the signal
        nt: Number of Tx antennas
        nr: Number of Rx antennas
        eps: Small value for numerical stability

    Returns:
        Batch of predicted channel matrices (B, Nt, Nr) complex
    """
    batch_size = rx_positions.shape[0]
    device = rx_positions.device
    tx_position = tx_position.to(device)

    gauss_means = model.get_xyz
    num_gaussians = gauss_means.shape[0]

    if num_gaussians == 0:

        return torch.zeros(batch_size, nt, nr, dtype=torch.complex64, device=device)

    gauss_latents, gauss_activations = model.get_attributes(tx_position)

    _, gauss_inv_covs = model.get_covariance(return_inverse=True, eps=eps)

    rx_pos_expanded = rx_positions.unsqueeze(1)
    gauss_means_expanded = gauss_means.unsqueeze(0)
    d_vec = rx_pos_expanded - gauss_means_expanded
    d_norm = torch.norm(d_vec, dim=-1, p=2).clamp(min=eps)

    phase_shift_rad = (2.0 * torch.pi / wavelength) * d_norm
    phase_shift_factor = torch.exp(-1j * phase_shift_rad)

    d_vec_flat = d_vec.view(-1, 3)
    inv_covs_expanded = gauss_inv_covs.repeat(batch_size, 1, 1)
    activations_expanded = gauss_activations.repeat(batch_size, 1)

    spatial_weights_flat = compute_spatial_weight(
        d_vec_flat, inv_covs_expanded, activations_expanded, eps
    )

    spatial_weights = spatial_weights_flat.view(batch_size, num_gaussians)

    h_contrib_flat = model.contribution_decoder(gauss_latents)

    h_contrib_real_flat = h_contrib_flat[..., : (nt * nr)]
    h_contrib_imag_flat = h_contrib_flat[..., (nt * nr) :]

    h_contrib = torch.complex(
        h_contrib_real_flat.view(num_gaussians, nt, nr),
        h_contrib_imag_flat.view(num_gaussians, nt, nr),
    )

    h_contrib_expanded = h_contrib.unsqueeze(0)
    spatial_weights_expanded = spatial_weights.unsqueeze(-1).unsqueeze(-1)
    phase_shift_factor_expanded = phase_shift_factor.unsqueeze(-1).unsqueeze(-1)
    combined_weight = spatial_weights_expanded * phase_shift_factor_expanded
    weighted_contributions = combined_weight * h_contrib_expanded
    H_pred = torch.sum(weighted_contributions, dim=1)

    return H_pred

# engine/render.py

import torch

from models.gaussian_model import GaussianChannelFieldModel


@torch.jit.script
def compute_spatial_weight(
    d_vec_n: torch.Tensor,
    inv_covariance_n: torch.Tensor,
    base_activation_n: torch.Tensor,
) -> torch.Tensor:
    """Computes the spatial weight for each Gaussian based on the distance vector,
    inverse covariance, and base activation.

    The spatial weight is computed as the product of the Gaussian PDF and the
    activated base opacity. The Gaussian PDF is computed using the Mahalanobis
    distance, which is derived from the inverse covariance matrix.

    Args:
        d_vec_n: Difference vectors from Rx pos to Gaussian means (N, 3)
        inv_covariance_n: Inverse covariance matrices for Gaussians (N, 3, 3)
        base_activation_n: Activated base opacities/activations (N, 1)
        eps: Small epsilon for numerical stability (already used in covariance)

    Returns:
        Spatial weights for each Gaussian (N,)
    """

    d_vec_n_unsqueezed = d_vec_n.unsqueeze(1)

    exp_val = torch.bmm(d_vec_n_unsqueezed, inv_covariance_n)
    exp_val = torch.bmm(exp_val, d_vec_n_unsqueezed.transpose(1, 2))
    exp_val = exp_val.squeeze()

    pdf_weight = torch.exp(-0.5 * torch.clamp(exp_val, max=50.0))
    activation_weight = base_activation_n.squeeze(-1)
    weight = activation_weight * pdf_weight

    return weight


def render(
    rx_positions: torch.Tensor,
    model: GaussianChannelFieldModel,
    tx_position: torch.Tensor,
    nt: int,
    nr: int,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Renders the channel magnitude response for a batch of receiver positions.

    Args:
        rx_positions: Batch of receiver positions (B, 3)
        model: The GaussianChannelFieldModel instance
        tx_position: The fixed transmitter position (3,)
        nt: Number of Tx antennas
        nr: Number of Rx antennas
        eps: Small value for numerical stability, especially for inverse covariance

    Returns:
        Batch of predicted channel magnitude responses (B, Nt, Nr)
    """
    batch_size = rx_positions.shape[0]
    device = rx_positions.device
    tx_position = tx_position.to(device)

    gauss_means = model.get_xyz
    num_gaussians = gauss_means.shape[0]

    if num_gaussians == 0:
        print("Warning: Rendering channel with zero Gaussians.")
        return torch.zeros(batch_size, nt, nr, dtype=torch.float32, device=device)

    gauss_latents, gauss_activations_activated = model.get_attributes_and_activation(
        tx_position
    )

    _, gauss_inv_covs = model.get_covariance(return_inverse=True, eps=eps)
    rx_pos_expanded = rx_positions.unsqueeze(1)
    gauss_means_expanded = gauss_means.unsqueeze(0)

    d_vec = rx_pos_expanded - gauss_means_expanded
    d_vec_flt = d_vec.view(-1, 3)

    inv_covs_expanded = gauss_inv_covs.repeat(batch_size, 1, 1)
    activations_expanded = gauss_activations_activated.repeat(batch_size, 1)

    spatial_weights_flt = compute_spatial_weight(
        d_vec_flt, inv_covs_expanded, activations_expanded
    )
    spatial_weights = spatial_weights_flt.view(batch_size, num_gaussians)

    cmr_contrib_flt = model.contribution_decoder(gauss_latents)
    cmr_contrib = cmr_contrib_flt.view(num_gaussians, nt, nr)
    cmr_contrib_expanded = cmr_contrib.unsqueeze(0)

    spatial_weights_expanded = spatial_weights.unsqueeze(-1).unsqueeze(-1)
    weighted_cmr_contributions = spatial_weights_expanded * cmr_contrib_expanded

    cmr_pred = torch.sum(weighted_cmr_contributions, dim=1)

    return cmr_pred.float()

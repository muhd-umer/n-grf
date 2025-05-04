# engine/render_magnitude.py # renamed from render_channel.py

import torch
import torch.nn.functional as F

from models.gaussian_model import GaussianChannelFieldModel


@torch.jit.script  # keep jit for potential performance boost
def compute_spatial_weight(
    d_vec_n: torch.Tensor,  # difference vector (Rx - Gaussian Mean) - shape (N, 3)
    inv_covariance_n: torch.Tensor,  # inverse covariance matrix - shape (N, 3, 3)
    base_activation_n: torch.Tensor,  # base activation (after sigmoid) - shape (N, 1)
    eps: float = 1e-10,
) -> torch.Tensor:
    """Computes the spatial weight alpha_n * GaussianPDF(d_vec_n).

    Args:
        d_vec_n: Difference vectors from Rx pos to Gaussian means (N, 3).
        inv_covariance_n: Inverse covariance matrices for Gaussians (N, 3, 3).
        base_activation_n: Activated base opacities/activations (N, 1).
        eps: Small epsilon for numerical stability (already used in covariance).

    Returns:
        Spatial weights for each Gaussian (N,).
    """
    # add batch dimension for bmm: (N, 1, 3)
    d_vec_n_unsqueezed = d_vec_n.unsqueeze(1)

    # calculate exponent term: -0.5 * (d^T * Sigma^-1 * d)
    # (N, 1, 3) @ (N, 3, 3) -> (N, 1, 3)
    exponent_term = torch.bmm(d_vec_n_unsqueezed, inv_covariance_n)
    # (N, 1, 3) @ (N, 3, 1) -> (N, 1, 1)
    exponent_term = torch.bmm(exponent_term, d_vec_n_unsqueezed.transpose(1, 2))
    # remove extra dims: (N,)
    exponent_term = exponent_term.squeeze()

    # calculate gaussian pdf component (unnormalized - determinant factor ignored as it's constant per gaussian)
    # clamp exponent to avoid exp overflow
    pdf_weight = torch.exp(
        -0.5 * torch.clamp(exponent_term, max=50.0)
    )  # clamp max value

    # multiply by the base activation (opacity)
    # squeeze base_activation_n from (N, 1) to (N,)
    activation_weight = base_activation_n.squeeze(-1)

    # final spatial weight = activation * pdf
    weight = activation_weight * pdf_weight

    # return weight.clamp(min=eps) # clamp minimum weight? maybe not needed if eps handled in cov
    return weight


def render_magnitude(
    rx_positions: torch.Tensor,  # shape (B, 3)
    model: GaussianChannelFieldModel,
    tx_position: torch.Tensor,  # shape (3,) or (1, 3)
    nt: int,
    nr: int,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Renders the normalized channel magnitude matrix H_mag for a batch of receiver positions.

    Args:
        rx_positions: Batch of receiver positions (B, 3).
        model: The GaussianChannelFieldModel instance.
        tx_position: The fixed transmitter position (3,).
        nt: Number of Tx antennas.
        nr: Number of Rx antennas.
        eps: Small value for numerical stability, especially for inverse covariance.

    Returns:
        Batch of predicted normalized channel magnitude matrices (B, Nt, Nr).
    """
    batch_size = rx_positions.shape[0]
    device = rx_positions.device
    tx_position = tx_position.to(device)  # ensure tx pos is on correct device

    # get gaussian parameters
    gauss_means = model.get_xyz  # (N, 3)
    num_gaussians = gauss_means.shape[0]

    # handle case with no gaussians
    if num_gaussians == 0:
        print("Warning: Rendering magnitude with zero Gaussians.")
        return torch.zeros(batch_size, nt, nr, dtype=torch.float32, device=device)

    # get dynamically computed attributes: latent features and base activation *logits*
    gauss_latents, gauss_activation_logits = model.get_attributes(
        tx_position
    )  # (N, latent_dim), (N, 1)
    # activate the base activation logits
    gauss_activations_activated = model.opacity_activation(
        gauss_activation_logits
    )  # (N, 1)

    # get inverse covariance matrices
    _, gauss_inv_covs = model.get_covariance(return_inverse=True, eps=eps)  # (N, 3, 3)

    # --- Prepare for Batch Processing ---
    # expand rx positions and gaussian means for batch calculation
    # rx_pos_expanded: (B, 1, 3)
    rx_pos_expanded = rx_positions.unsqueeze(1)
    # gauss_means_expanded: (1, N, 3)
    gauss_means_expanded = gauss_means.unsqueeze(0)

    # calculate difference vectors d_vec = rx_pos - gauss_mean
    # shape: (B, N, 3)
    d_vec = rx_pos_expanded - gauss_means_expanded

    # flatten d_vec for batch matrix multiplication: (B*N, 3)
    d_vec_flat = d_vec.view(-1, 3)

    # expand inverse covariances and activations for batch processing
    # inv_covs_expanded: (B*N, 3, 3) - repeat each gaussian's cov for every batch item
    inv_covs_expanded = gauss_inv_covs.repeat(batch_size, 1, 1)
    # activations_expanded: (B*N, 1) - repeat each gaussian's activation for every batch item
    activations_expanded = gauss_activations_activated.repeat(batch_size, 1)

    # --- Compute Spatial Weights ---
    # compute weights for all batch-gaussian pairs: (B*N,)
    spatial_weights_flat = compute_spatial_weight(
        d_vec_flat, inv_covs_expanded, activations_expanded, eps
    )
    # reshape back to (B, N)
    spatial_weights = spatial_weights_flat.view(batch_size, num_gaussians)

    # --- Decode Magnitude Contributions ---
    # decode latent features into magnitude contributions (already activated by sigmoid)
    # h_mag_contrib_flat shape: (N, Nt * Nr)
    h_mag_contrib_flat = model.contribution_decoder(gauss_latents)
    # reshape to (N, Nt, Nr)
    h_mag_contrib = h_mag_contrib_flat.view(num_gaussians, nt, nr)

    # --- Combine Contributions ---
    # expand contributions and weights for batch summation
    # h_mag_contrib_expanded: (1, N, Nt, Nr)
    h_mag_contrib_expanded = h_mag_contrib.unsqueeze(0)
    # spatial_weights_expanded: (B, N, 1, 1)
    spatial_weights_expanded = spatial_weights.unsqueeze(-1).unsqueeze(-1)

    # weight the contributions by spatial weights
    # weighted_contributions: (B, N, Nt, Nr)
    weighted_contributions = spatial_weights_expanded * h_mag_contrib_expanded

    # sum contributions over the gaussian dimension (dim=1)
    # H_mag_pred: (B, Nt, Nr)
    H_mag_pred = torch.sum(weighted_contributions, dim=1)

    # ensure output is float32
    return H_mag_pred.float()

# models/dir_network.py

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .embedder import get_embedder


@dataclass
class DirNetworkConfig:
    """Configuration for the Directional Network.

    Args:
        hidden_size: Size of hidden layers.
        num_layers: Number of hidden layers.
        input_dir_multires: Positional encoding resolution for directions.
            Only used if use_positional_encoding is True.
        use_positional_encoding: Whether positional encoding is applied *before*
            the input direction vector is passed to this network. This affects
            the expected input dimension.
        use_layer_norm: Whether to use Layer Normalization.
        dropout_prob: Dropout probability.
    """

    hidden_size: int = 64
    num_layers: int = 4
    input_dir_multires: int = 4
    use_positional_encoding: bool = True
    use_layer_norm: bool = False
    dropout_prob: float = 0.0


class DirectionalNetwork(nn.Module):
    """Network to predict view-dependent scattering coefficients.

    Takes base features and (potentially embedded) direction vectors as input
    and outputs complex scattering coefficients (gamma).

    Args:
        base_feature_dim: The dimensionality of the base feature vectors
        dir_input_dim: The dimensionality of the direction input vector. This
            will be 3 if positional encoding is off, or the embedding dimension
            if positional encoding is on
        config: Configuration object for the network architecture
    """

    def __init__(
        self, base_feature_dim: int, dir_input_dim: int, config: DirNetworkConfig
    ):
        super().__init__()
        self.config = config

        input_dim = base_feature_dim + 2 * dir_input_dim

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, config.hidden_size))
        if config.use_layer_norm:
            self.layers.append(nn.LayerNorm(config.hidden_size))
        self.layers.append(nn.ReLU())
        if config.dropout_prob > 0:
            self.layers.append(nn.Dropout(config.dropout_prob))

        for _ in range(config.num_layers - 1):
            self.layers.append(nn.Linear(config.hidden_size, config.hidden_size))
            if config.use_layer_norm:
                self.layers.append(nn.LayerNorm(config.hidden_size))
            self.layers.append(nn.ReLU())
            if config.dropout_prob > 0:
                self.layers.append(nn.Dropout(config.dropout_prob))

        self.gamma_amp = nn.Linear(config.hidden_size, 1)
        self.gamma_phase = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        base_features: torch.Tensor,
        directions_in: torch.Tensor,
        directions_out: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to compute direction-dependent scattering coefficients.

        Args:
            base_features: Base feature vectors from the FeatureEncoder [N, F]
            directions_in: Incoming direction vectors
            directions_out: Outgoing direction vectors

        Returns:
            Tuple of tensors (gamma_real, gamma_imag) each of shape [N, 1]
        """
        x = torch.cat([base_features, directions_in, directions_out], dim=-1)

        hidden_state = x
        for layer in self.layers:
            hidden_state = layer(hidden_state)

        raw_gamma_A = self.gamma_amp(hidden_state)
        raw_gamma_psi = self.gamma_phase(hidden_state)

        gamma_A = F.softplus(raw_gamma_A)
        gamma_psi = torch.sigmoid(raw_gamma_psi) * 2 * torch.pi
        gamma_real = gamma_A * torch.cos(gamma_psi)
        gamma_imag = gamma_A * torch.sin(gamma_psi)

        return gamma_real, gamma_imag

# models/networks.py

from typing import Tuple

import torch
import torch.nn as nn

from utils.pos_encoder import PositionalEncoder


class SimpleMLP(nn.Module):
    """A simple Multi-Layer Perceptron with ReLU activations."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ContributionDecoderNetwork(SimpleMLP):
    """
    Decodes latent features into complex channel contributions.
    Output dimension is 2 * Nt * Nr (real and imaginary parts).
    """

    def __init__(
        self, latent_dim: int, output_dim: int, hidden_dim: int, num_layers: int
    ):
        super().__init__(latent_dim, output_dim, hidden_dim, num_layers)


class AttributeNetwork(nn.Module):
    """
    Predicts latent features (f_n) and base activations (a_n) from
    Gaussian position (μ_n) and fixed Tx position (P_TX).
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        pos_encoding_freqs: int = 10,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.pos_encoder = PositionalEncoder(3, pos_encoding_freqs)
        encoded_dim = self.pos_encoder.output_dims
        input_mlp_dim = 2 * encoded_dim

        self.network = SimpleMLP(
            input_dim=input_mlp_dim,
            output_dim=latent_dim + 1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(
        self, mu_n: torch.Tensor, p_tx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mu_n: Gaussian means (N, 3)
            p_tx: Transmitter position (1, 3) or (3,) - expanded to (N, 3)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Latent features (N, latent_dim), Base activations (N, 1)
        """
        num_gaussians = mu_n.shape[0]
        if p_tx.dim() == 1:
            p_tx = p_tx.unsqueeze(0)
        p_tx_expanded = p_tx.expand(num_gaussians, -1)

        encoded_mu = self.pos_encoder(mu_n)
        encoded_ptx = self.pos_encoder(p_tx_expanded)

        mlp_input = torch.cat([encoded_mu, encoded_ptx], dim=-1)
        output = self.network(mlp_input)

        latent_features = output[:, : self.latent_dim]
        base_activations = output[:, self.latent_dim :]

        return latent_features, base_activations

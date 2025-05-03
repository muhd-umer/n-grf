# models/networks.py

from typing import Tuple

import torch
import torch.nn as nn

from utils.pos_encoder import PositionalEncoder


class SimpleMLP(nn.Module):
    """A simple Multi-Layer Perceptron"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        use_leaky_relu: bool = True,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        current_dim = input_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))

            layers.append(nn.LeakyReLU() if use_leaky_relu else nn.GELU())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
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

        super().__init__(
            latent_dim, output_dim, hidden_dim, num_layers, use_leaky_relu=True
        )


class AttributeNetwork(nn.Module):
    """
    Predicts latent features and base activations from
    Gaussian position and fixed Tx position.
    """

    def __init__(
        self,
        latent_dim: int,
        mlp_hidden_dim: int,
        mlp_num_layers: int,
        pos_encoding_freqs: int = 10,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.pos_encoder_mean = PositionalEncoder(3, pos_encoding_freqs)

        self.pos_encoder_tx = PositionalEncoder(3, pos_encoding_freqs)

        encoded_dim_mean = self.pos_encoder_mean.output_dims
        encoded_dim_tx = self.pos_encoder_tx.output_dims
        input_dim = encoded_dim_mean + encoded_dim_tx

        output_dim = latent_dim + 1

        self.network = SimpleMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            use_leaky_relu=False,
            dropout_p=0.0,
        )

    def forward(
        self, mu_n: torch.Tensor, p_tx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mu_n: Gaussian means (N, 3)
            p_tx: Transmitter position (1, 3) or (3,)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Latent features (N, latent_dim),
            Base activations (N, 1)
        """
        if p_tx.shape[0] != mu_n.shape[0]:
            if p_tx.dim() == 1:
                p_tx = p_tx.unsqueeze(0)
            p_tx_expanded = p_tx.expand(mu_n.shape[0], -1)
        else:
            p_tx_expanded = p_tx

        encoded_mu = self.pos_encoder_mean(mu_n)
        encoded_ptx = self.pos_encoder_tx(p_tx_expanded)

        mlp_input = torch.cat([encoded_mu, encoded_ptx], dim=-1)
        output = self.network(mlp_input)

        latent_features = output[:, : self.latent_dim]
        base_activations = output[:, self.latent_dim :]

        return latent_features, base_activations

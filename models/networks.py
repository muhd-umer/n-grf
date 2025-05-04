# models/networks.py

from typing import Optional, Tuple

import torch
import torch.nn as nn

from utils.pos_encoder import PositionalEncoder


class SimpleMLP(nn.Module):
    """A simple multi-layer perceptron"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        use_leaky_relu: bool = True,
        dropout_p: float = 0.1,
        final_activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if num_layers < 2:
            raise ValueError("MLP must have at least 2 layers (input -> output)")

        layers = []
        current_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))

            layers.append(nn.LeakyReLU(0.1) if use_leaky_relu else nn.GELU())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        if final_activation is not None:
            layers.append(final_activation)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ContributionDecoderNetwork(SimpleMLP):
    """Decodes latent features into channel contributions."""

    def __init__(
        self, latent_dim: int, output_dim: int, hidden_dim: int, num_layers: int
    ):
        super().__init__(
            input_dim=latent_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_leaky_relu=True,
            dropout_p=0.1,
            final_activation=nn.Sigmoid(),
        )


class AttributeNetwork(nn.Module):
    """
    Predicts latent features and base activations logits from Gaussian position
    and fixed Tx position using positional encoding.
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

        self.pos_encoder_mean = PositionalEncoder(
            input_dims=3, num_freqs=pos_encoding_freqs, include_input=True
        )
        self.pos_encoder_tx = PositionalEncoder(
            input_dims=3, num_freqs=pos_encoding_freqs, include_input=True
        )

        encoded_dim_mean = self.pos_encoder_mean.output_dims
        encoded_dim_tx = self.pos_encoder_tx.output_dims

        input_dim = encoded_dim_mean + encoded_dim_tx
        output_dim = latent_dim + 1

        self.network = SimpleMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            use_leaky_relu=True,
            dropout_p=0.0,
            final_activation=None,
        )

    def forward(
        self, mu_n: torch.Tensor, p_tx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts latent features and base activation logits.

        Args:
            mu_n: Gaussian means (N, 3)
            p_tx: Transmitter position (N, 3) - MUST be expanded to match N

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Latent features (N, latent_dim)
                - Base activation logits (N, 1)
        """
        num_gaussians = mu_n.shape[0]
        if num_gaussians == 0:
            return torch.empty(0, self.latent_dim, device=mu_n.device), torch.empty(
                0, 1, device=mu_n.device
            )

        encoded_mu = self.pos_encoder_mean(mu_n)
        encoded_ptx = self.pos_encoder_tx(p_tx)

        mlp_input = torch.cat([encoded_mu, encoded_ptx], dim=-1)
        output = self.network(mlp_input)

        latent_features = output[:, : self.latent_dim]
        base_activations_logits = output[:, self.latent_dim :]

        return latent_features, base_activations_logits

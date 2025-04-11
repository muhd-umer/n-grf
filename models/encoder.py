# models/encoder.py

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .embedder import get_embedder


@dataclass
class EncoderConfig:
    """Configuration for encoder

    Args:
        hidden_size: Size of hidden layers
        num_layers: Number of hidden layers
        skip_layers: List of layer indices to add skip connections
        input_pos_multires: Positional encoding resolution for positions
        use_positional_encoding: Whether to use positional encoding
        use_layer_norm: Whether to use Layer Normalization
    """

    hidden_size: int = 128
    num_layers: int = 8
    skip_layers: Tuple[int, ...] = (4,)
    input_pos_multires: int = 10
    use_positional_encoding: bool = True
    use_layer_norm: bool = True


class FeatureEncoder(nn.Module):
    """Encoder network for channel reconstruction.

    Maps environment geometry and wireless properties to features for Gaussian
    splatting, specifically attenuation and phase rotation.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.use_positional_encoding = config.use_positional_encoding

        if self.use_positional_encoding:
            self.pos_embedder, pos_embed_dim = get_embedder(
                config.input_pos_multires, input_dims=3
            )
            point_feat_dim = pos_embed_dim
            tx_feat_dim = pos_embed_dim
            rx_feat_dim = pos_embed_dim
        else:
            point_feat_dim = 3
            tx_feat_dim = 3
            rx_feat_dim = 3

        input_dim = point_feat_dim + tx_feat_dim + rx_feat_dim

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, config.hidden_size))
        if config.use_layer_norm:
            self.layers.append(nn.LayerNorm(config.hidden_size))
        self.layers.append(nn.ReLU())

        for i in range(config.num_layers - 1):
            layer_input_dim = config.hidden_size
            if i + 1 in config.skip_layers:
                layer_input_dim += input_dim

            self.layers.append(nn.Linear(layer_input_dim, config.hidden_size))
            if config.use_layer_norm:
                self.layers.append(nn.LayerNorm(config.hidden_size))
            self.layers.append(nn.ReLU())

        self.attenuation_head = nn.Linear(config.hidden_size, 1)
        self.phase_rotation_head = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        points: torch.Tensor,
        tx_pos: torch.Tensor,
        rx_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to generate SH features for Gaussian virtual transmitters.

        Args:
            points: Point positions (N, 3)
            tx_pos: Transmitter position (3,)
            rx_pos: Receiver position (3,)

        Returns:
            Tuple of tensors (attenuation, phase_rotation) each of shape (N, 1)
        """
        if self.use_positional_encoding:
            points_embed = self.pos_embedder(points)
            tx_embed = self.pos_embedder(tx_pos.expand(points.shape[0], -1))
            rx_embed = self.pos_embedder(rx_pos.expand(points.shape[0], -1))
            x = torch.cat([points_embed, tx_embed, rx_embed], dim=-1)
        else:
            x = torch.cat(
                [
                    points,
                    tx_pos.expand(points.shape[0], -1),
                    rx_pos.expand(points.shape[0], -1),
                ],
                dim=-1,
            )

        input_features = x
        hidden_state = x

        layer_idx = 0
        linear_layer_count = 0
        while layer_idx < len(self.layers):
            layer = self.layers[layer_idx]

            if isinstance(layer, nn.Linear) and linear_layer_count > 0:
                current_conceptual_layer = linear_layer_count
                if current_conceptual_layer in self.config.skip_layers:
                    hidden_state = torch.cat([hidden_state, input_features], dim=-1)

            hidden_state = layer(hidden_state)

            if isinstance(layer, nn.Linear):
                linear_layer_count += 1

            layer_idx += 1

        final_hidden_state = hidden_state
        raw_attenuation = self.attenuation_head(final_hidden_state)
        raw_phase_rotation = self.phase_rotation_head(final_hidden_state)

        attenuation = F.softplus(raw_attenuation)
        phase_rotation = torch.sigmoid(raw_phase_rotation) * 2 * torch.pi

        return attenuation, phase_rotation

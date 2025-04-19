# models/encoder.py

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .embedder import get_embedder


@dataclass
class EncoderConfig:
    """Configuration for the base feature encoder.

    Args:
        hidden_size: Size of hidden layers
        num_layers: Number of hidden layers
        skip_layers: List of layer indices to add skip connections
        input_pos_multires: Positional encoding resolution for positions
        use_positional_encoding: Whether to use positional encoding for inputs
        use_layer_norm: Whether to use Layer Normalization
        dropout_prob: Dropout probability
        base_feature_dim: Output dimension for base features
    """

    hidden_size: int = 128
    num_layers: int = 8
    skip_layers: Tuple[int, ...] = (4,)
    input_pos_multires: int = 10
    use_positional_encoding: bool = True
    use_layer_norm: bool = True
    dropout_prob: float = 0.2
    base_feature_dim: int = 64


class FeatureEncoder(nn.Module):
    """Encoder network for channel reconstruction.

    Maps environment geometry (Gaussian positions) and transmitter position to
    base feature vectors for subsequent directional processing

    Args:
        config: Configuration object for the encoder architecture
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
        else:
            self.pos_embedder = nn.Identity()
            point_feat_dim = 3
            tx_feat_dim = 3

        input_dim = point_feat_dim + tx_feat_dim

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, config.hidden_size))
        if config.use_layer_norm:
            self.layers.append(nn.LayerNorm(config.hidden_size))
        self.layers.append(nn.ReLU())
        if config.dropout_prob > 0:
            self.layers.append(nn.Dropout(config.dropout_prob))

        for i in range(config.num_layers - 1):
            layer_input_dim = config.hidden_size
            if i + 1 in config.skip_layers:
                layer_input_dim += input_dim

            self.layers.append(nn.Linear(layer_input_dim, config.hidden_size))
            if config.use_layer_norm:
                self.layers.append(nn.LayerNorm(config.hidden_size))
            self.layers.append(nn.ReLU())
            if config.dropout_prob > 0:
                self.layers.append(nn.Dropout(config.dropout_prob))

        self.feature_head = nn.Linear(config.hidden_size, config.base_feature_dim)

    def forward(
        self,
        points: torch.Tensor,
        tx_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass to generate base feature vectors for Gaussians.

        Args:
            points: Point positions [N, 3]
            tx_pos: Transmitter position (3,] -> broadcasted to [N, 3]

        Returns:
            Base feature vectors of shape [N, base_feature_dim]
        """
        num_points = points.shape[0]
        if tx_pos.dim() == 1:
            tx_pos_expanded = tx_pos.expand(num_points, -1)
        else:
            tx_pos_expanded = tx_pos

        if self.use_positional_encoding:
            points_embed = self.pos_embedder(points)
            tx_embed = self.pos_embedder(tx_pos_expanded)
            x = torch.cat([points_embed, tx_embed], dim=-1)
        else:
            x = torch.cat(
                [
                    points,
                    tx_pos_expanded,
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

        base_features = self.feature_head(hidden_state)
        return base_features

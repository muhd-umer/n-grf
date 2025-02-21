# models/encoder.py

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .embedder import get_embedder


@dataclass
class EncoderConfig:
    """Configuration for wireless channel encoder.

    Args:
        hidden_size: Size of hidden layers
        num_layers: Number of hidden layers
        skip_layers: List of layer indices to add skip connections
        input_pos_multires: Positional encoding (PE) resolution for positions
        input_dir_multires: PE  resolution for directions
        sh_degree: Degree of spherical harmonics output (0 or 1 supported)
    """

    hidden_size: int = 128
    num_layers: int = 8
    skip_layers: Tuple[int, ...] = (4,)
    input_pos_multires: int = 10
    input_dir_multires: int = 4
    sh_degree: int = 1  # only 0 or 1 supported


class WirelessEncoder(nn.Module):
    """Encoder network for wireless channel reconstruction.

    Maps environment geometry and wireless properties to spherical harmonic
    features for Gaussian splatting. Designed for reconstructing complex MIMO
    channel matrices using 3D Gaussians as virtual transmitters.

    Args:
        config: Configuration object
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # for positional encoding
        self.pos_embedder, pos_embed_dim = get_embedder(
            config.input_pos_multires, input_dims=3
        )
        self.dir_embedder, dir_embed_dim = get_embedder(
            config.input_dir_multires, input_dims=3
        )

    def forward(
        self,
    ) -> dict[str, torch.Tensor]:
        pass

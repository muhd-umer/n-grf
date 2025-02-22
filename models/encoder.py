# models/encoder.py

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .embedder import get_embedder


@dataclass
class EncoderConfig:
    """Configuration for wireless channel encoder"""

    hidden_size: int = 128
    num_layers: int = 8
    skip_layers: Tuple[int, ...] = (4,)
    input_pos_multires: int = 10
    path_encode_dim: int = 32

    # only degree 0 or 1 supported for SH
    sh_degree: int = 1

    # scenario type
    is_indoor: bool = True
    is_outdoor: bool = not is_indoor

    @property
    def num_tx_ant(self) -> int:
        return 16 if self.is_indoor else 64

    @property
    def num_rx_ant(self) -> int:
        return 2


class PathFeatureEncoder(nn.Module):
    """Encoder for processing variable number of propagation paths.

    Uses attention mechanism to create a fixed-dimensional encoding
    of path features regardless of number of paths.
    """

    def __init__(self, path_encode_dim: int):
        super().__init__()

        self.azim_elev_dim = 2

        self.path_mlp = nn.Sequential(
            nn.Linear(self.azim_elev_dim, path_encode_dim),
            nn.ReLU(),
            nn.Linear(path_encode_dim, path_encode_dim),
        )

        self.num_heads = 4
        self.attention = nn.MultiheadAttention(
            embed_dim=path_encode_dim, num_heads=self.num_heads, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, path_encode_dim))

    def forward(self, aoa: torch.Tensor) -> torch.Tensor:
        """Process variable number of paths to fixed dimension encoding.

        Args:
            aoa: Angle of arrival tensor [2, num_paths] with azimuth and elevation
                angles for each path

        Returns:
            Fixed dimensional path encoding [path_encode_dim]
        """
        aoa = aoa.t()
        path_features = self.path_mlp(aoa)
        path_features = path_features.unsqueeze(0)
        query = self.query.expand(1, -1, -1)

        attn_output, _ = self.attention(
            query=query, key=path_features, value=path_features
        )

        return attn_output.squeeze(0).squeeze(0)


class WirelessEncoder(nn.Module):
    """Encoder network for wireless channel reconstruction.

    Maps environment geometry and wireless properties to spherical harmonic
    features for Gaussian splatting. Designed for reconstructing complex MIMO
    channel matrices using 3D Gaussians as virtual transmitters.

    The network takes:
    - Point position (3D)
    - Transmitter position (3D)
    - Receiver position (3D)
    - Path loss information (1D)
    - AoA information [2, num_paths] with variable num_paths

    And outputs 4 sets of SH features for:
    - Signal amplitude (3 channels)
    - Signal phase (3 channels)
    - Attenuation (3 channels)
    - Phase rotation (3 channels)

    Each feature has (sh_degree + 1)^2 coefficients per channel.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        self.pos_embedder, pos_embed_dim = get_embedder(
            config.input_pos_multires, input_dims=3
        )
        self.path_encoder = PathFeatureEncoder(config.path_encode_dim)
        point_feat_dim = pos_embed_dim
        tx_feat_dim = pos_embed_dim
        rx_feat_dim = pos_embed_dim
        path_feat_dim = config.path_encode_dim
        pathloss_feat_dim = 1

        input_dim = (
            point_feat_dim
            + tx_feat_dim
            + rx_feat_dim
            + path_feat_dim
            + pathloss_feat_dim
        )

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, config.hidden_size))

        for i in range(config.num_layers - 1):
            if i in config.skip_layers:
                layer_input_dim = config.hidden_size + input_dim
            else:
                layer_input_dim = config.hidden_size
            self.layers.append(nn.Linear(layer_input_dim, config.hidden_size))

        out_dim = 3 * (config.sh_degree + 1) ** 2  # 3 channels, (d+1)^2 SH coeffs
        self.signal_amp_head = nn.Linear(config.hidden_size, out_dim)
        self.signal_phase_head = nn.Linear(config.hidden_size, out_dim)
        self.attenuation_head = nn.Linear(config.hidden_size, out_dim)
        self.phase_rotation_head = nn.Linear(config.hidden_size, out_dim)

    def forward(
        self,
        points: torch.Tensor,
        tx_pos: torch.Tensor,
        rx_pos: torch.Tensor,
        path_loss: torch.Tensor,
        aoa: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass to generate SH features for Gaussian virtual transmitters.

        Args:
            points: Point positions (N, 3)
            tx_pos: Transmitter position (3,)
            rx_pos: Receiver position (3,)
            path_loss: Path loss values (N, 1)
            aoa: Angles of arrival (2, P) with azimuth and elevation for P paths

        Returns:
            Dictionary with SH features for:
            - signal_amplitude: Signal amplitude features (N, 3, (d+1)^2)
            - signal_phase: Signal phase features (N, 3, (d+1)^2)
            - attenuation: Attenuation features (N, 3, (d+1)^2)
            - phase_rotation: Phase rotation features (N, 3, (d+1)^2)
        """
        points_embed = self.pos_embedder(points)
        tx_embed = self.pos_embedder(tx_pos.expand(points.shape[0], -1))
        rx_embed = self.pos_embedder(rx_pos.expand(points.shape[0], -1))

        path_encoding = self.path_encoder(aoa)
        path_encoding = path_encoding.expand(points.shape[0], -1)

        x = torch.cat(
            [
                points_embed,
                tx_embed,
                rx_embed,
                path_encoding,
                path_loss.expand(points.shape[0], -1),
            ],
            dim=-1,
        )

        input_features = x
        for i, layer in enumerate(self.layers):
            if i in self.config.skip_layers:
                x = torch.cat([x, input_features], dim=-1)
            x = layer(x)
            x = torch.relu(x)

        signal_amp = self.signal_amp_head(x)
        signal_phase = self.signal_phase_head(x)
        attenuation = self.attenuation_head(x)
        phase_rotation = self.phase_rotation_head(x)

        # reshape to (N, 3, (d+1)^2)
        sh_dim = (self.config.sh_degree + 1) ** 2
        signal_amp = signal_amp.view(-1, 3, sh_dim)
        signal_phase = signal_phase.view(-1, 3, sh_dim)
        attenuation = attenuation.view(-1, 3, sh_dim)
        phase_rotation = phase_rotation.view(-1, 3, sh_dim)

        return {
            "signal_amplitude": signal_amp,
            "signal_phase": signal_phase,
            "attenuation": attenuation,
            "phase_rotation": phase_rotation,
        }

# models/encoder.py

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .embedder import get_embedder


@dataclass
class EncoderConfig:
    """Configuration for wireless channel encoder

    Args:
        hidden_size: Size of hidden layers
        num_layers: Number of hidden layers
        skip_layers: List of layer indices to add skip connections
        input_pos_multires: Positional encoding resolution for positions
        path_encode_dim: Dimension for path feature encoding
        sh_degree: Degree of spherical harmonics output (0 or 1 supported)
        use_attention: Whether to use attention for path encoding
        max_paths: Maximum number of paths to consider when using masking
        is_indoor: Whether using indoor scenario (defines antenna counts)
    """

    hidden_size: int = 128
    num_layers: int = 8
    skip_layers: Tuple[int, ...] = (4,)
    input_pos_multires: int = 10
    path_encode_dim: int = 32

    # only degree 0 or 1 supported for SH
    sh_degree: int = 1

    use_attention: bool = True
    max_paths: int = 10

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

    Has two modes:
    1. Attention-based: Uses multihead attention to aggregate path features
    2. Masking-based: Selects top-k paths based on minimum path loss

    Args:
        path_encode_dim: Dimension of encoded path features
        use_attention: Whether to use attention mechanism
        max_paths: Maximum number of paths to consider when using masking
    """

    def __init__(
        self, path_encode_dim: int, use_attention: bool = True, max_paths: int = 10
    ):
        super().__init__()
        self.use_attention = use_attention
        self.max_paths = max_paths
        self.path_encode_dim = path_encode_dim
        self.azim_elev_dim = 2

        self.path_mlp = nn.Sequential(
            nn.Linear(self.azim_elev_dim, path_encode_dim),
            nn.ReLU(),
            nn.Linear(path_encode_dim, path_encode_dim),
        )

        if use_attention:
            self.num_heads = 4
            self.attention = nn.MultiheadAttention(
                embed_dim=path_encode_dim, num_heads=self.num_heads, batch_first=True
            )
            self.query = nn.Parameter(torch.randn(1, 1, path_encode_dim))
        else:
            self.aggregation_mlp = nn.Sequential(
                nn.Linear(max_paths * path_encode_dim, path_encode_dim * 2),
                nn.ReLU(),
                nn.Linear(path_encode_dim * 2, path_encode_dim),
            )
            self.fallback_encoder = nn.Linear(path_encode_dim, path_encode_dim)

    def forward(
        self, aoa: torch.Tensor, path_loss_per_ray: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process variable number of paths to fixed dimension encoding.

        Args:
            aoa: Angle of arrival tensor [2, num_paths] with azimuth and elevation
                angles for each path
            path_loss_per_ray: Path loss for each ray [num_paths], used for masking
                approach to select top-k paths

        Returns:
            Fixed dimensional path encoding [path_encode_dim]
        """
        num_paths = aoa.shape[1]

        if self.use_attention:
            aoa = aoa.t()
            path_features = self.path_mlp(aoa)
            path_features = path_features.unsqueeze(
                0
            )  # [1, num_paths, path_encode_dim]
            query = self.query.expand(1, -1, -1)  # [1, 1, path_encode_dim]

            attn_output, _ = self.attention(
                query=query, key=path_features, value=path_features
            )
            return attn_output.squeeze(0).squeeze(0)
        else:
            # select top-k paths based on path loss
            if num_paths == 0:
                return torch.zeros(self.path_encode_dim, device=aoa.device)

            if num_paths <= self.max_paths:
                # lower than max_paths, process all and pad
                aoa_t = aoa.t()
                path_features = self.path_mlp(aoa_t)

                # if only one path
                if num_paths == 1:
                    return self.fallback_encoder(path_features.squeeze(0))

                # otherwise pad to max_paths
                padding = torch.zeros(
                    (self.max_paths - num_paths, self.path_encode_dim),
                    device=aoa.device,
                )
                padded_features = torch.cat([path_features, padding], dim=0)

                flat_features = padded_features.view(-1)
                return self.aggregation_mlp(flat_features)
            else:
                # more than max_paths, select top-k based on path loss
                if path_loss_per_ray is None:
                    # if path loss not provided, just take first max_paths
                    indices = torch.arange(self.max_paths, device=aoa.device)
                else:
                    # sort by lowest path loss (lower = stronger signal)
                    _, indices = torch.sort(path_loss_per_ray, descending=False)
                    indices = indices[: self.max_paths]

                selected_aoa = aoa[:, indices].t()
                path_features = self.path_mlp(selected_aoa)

                flat_features = path_features.view(-1)
                return self.aggregation_mlp(flat_features)


class WirelessEncoder(nn.Module):
    """Encoder network for wireless channel reconstruction.

    Maps environment geometry and wireless properties to spherical harmonic
    features for Gaussian splatting. Designed for reconstructing complex MIMO
    channel matrices using 3D Gaussians as virtual transmitters.

    Each feature has (sh_degree + 1)^2 coefficients per channel.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # for positional encoding
        self.pos_embedder, pos_embed_dim = get_embedder(
            config.input_pos_multires, input_dims=3
        )

        self.path_encoder = PathFeatureEncoder(
            config.path_encode_dim,
            use_attention=config.use_attention,
            max_paths=config.max_paths,
        )

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
            if i + 1 in config.skip_layers:
                layer_input_dim = config.hidden_size + input_dim
            else:
                layer_input_dim = config.hidden_size
            self.layers.append(nn.Linear(layer_input_dim, config.hidden_size))

        # modified output dimension to be 3 (channels) instead of 3 * (d+1)^2
        out_dim = 3
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
        path_loss_per_ray: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass to generate SH features for Gaussian virtual transmitters.

        Args:
            points: Point positions (N, 3)
            tx_pos: Transmitter position (3,)
            rx_pos: Receiver position (3,)
            path_loss: Path loss values (N, 1)
            aoa: Angles of arrival (2, P) with azimuth and elevation for P paths
            path_loss_per_ray: Path loss per ray (P) for selecting important paths

        Returns:
            Dictionary with SH features for:
            > signal_amplitude: Signal amplitude features (N, 3, 1)
            > signal_phase: Signal phase features (N, 3, 1)
            > attenuation: Attenuation features (N, 3, 1)
            > phase_rotation: Phase rotation features (N, 3, 1)
        """
        points_embed = self.pos_embedder(points)
        tx_embed = self.pos_embedder(tx_pos.expand(points.shape[0], -1))
        rx_embed = self.pos_embedder(rx_pos.expand(points.shape[0], -1))

        path_encoding = self.path_encoder(aoa, path_loss_per_ray)
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

        # reshape to (N, 3, 1); one value per channel
        signal_amp = signal_amp.view(-1, 3, 1)
        signal_phase = signal_phase.view(-1, 3, 1)
        attenuation = attenuation.view(-1, 3, 1)
        phase_rotation = phase_rotation.view(-1, 3, 1)

        return {
            "signal_amplitude": signal_amp,
            "signal_phase": signal_phase,
            "attenuation": attenuation,
            "phase_rotation": phase_rotation,
        }

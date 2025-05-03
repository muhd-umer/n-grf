# models/gaussian_model.py

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    build_covariance_inverse,
    build_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
)

from .networks import AttributeNetwork, ContributionDecoderNetwork


class GaussianChannelFieldModel(nn.Module):
    """Gaussian channel field (GCF) model."""

    def __init__(
        self,
        num_tx_ant: int,
        num_rx_ant: int,
        latent_dim: int,
        attribute_mlp_hidden_dim: int = 64,
        attribute_mlp_num_layers: int = 3,
        attribute_pos_enc_freqs: int = 10,
        decoder_hidden_dim: int = 64,
        decoder_num_layers: int = 3,
        initial_gaussians: int = 30000,
        init_opacity_value: float = 0.1,
        init_scale_value: float = 0.02,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__()
        self.num_tx_ant = num_tx_ant
        self.num_rx_ant = num_rx_ant
        self.latent_dim = latent_dim
        self.device = device

        self._xyz = nn.Parameter(torch.empty(0, 3, device=device))
        self._rotation = nn.Parameter(torch.empty(0, 4, device=device))
        self._scaling = nn.Parameter(torch.empty(0, 3, device=device))

        self.attribute_network = AttributeNetwork(
            latent_dim=latent_dim,
            mlp_hidden_dim=attribute_mlp_hidden_dim,
            mlp_num_layers=attribute_mlp_num_layers,
            pos_encoding_freqs=attribute_pos_enc_freqs,
        ).to(device)

        self.contribution_decoder = ContributionDecoderNetwork(
            latent_dim=latent_dim,
            output_dim=2 * num_tx_ant * num_rx_ant,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
        ).to(device)

        self.initial_gaussians = initial_gaussians
        self.init_opacity_logit = inverse_sigmoid(
            torch.tensor(init_opacity_value, device=device)
        )
        self.init_log_scale = torch.log(torch.tensor(init_scale_value, device=device))

        self.optimizer = None
        self.lr_schedulers = {}

        self.setup_activations()

    def setup_activations(self):
        """Setup activation functions."""
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = lambda r: F.normalize(r, p=2, dim=-1)

    @property
    def get_xyz(self):
        """Returns Gaussian positions."""
        return self._xyz

    @property
    def get_scaling(self):
        """Returns activated and clamped Gaussian scales."""

        return self.scaling_activation(self._scaling).clamp(min=1e-8)

    @property
    def get_rotation(self):
        """Returns normalized Gaussian rotations (quaternions)."""
        return self.rotation_activation(self._rotation)

    def get_attributes(
        self, tx_position: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes latent features and base activations dynamically."""
        if self._xyz.shape[0] == 0:

            return torch.empty(0, self.latent_dim, device=self.device), torch.empty(
                0, 1, device=self.device
            )

        if tx_position.dim() == 1:
            tx_position = tx_position.unsqueeze(0)

        tx_pos_expanded = (
            tx_position.expand(self._xyz.shape[0], -1)
            if self._xyz.shape[0] > 0
            else torch.empty(0, 3, device=self.device)
        )

        latent_features, base_activations = self.attribute_network(
            self._xyz, tx_pos_expanded
        )
        return latent_features, base_activations

    def get_opacity_activated(self, tx_position: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid-activated base activations, computed dynamically."""
        _, base_activations = self.get_attributes(tx_position)
        return self.opacity_activation(base_activations)

    def get_covariance(
        self, return_inverse=False, eps=1e-6
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Computes covariance matrix Σ and optionally its inverse Σ^-1."""
        scaling = self.get_scaling
        rotation_q = self.get_rotation

        if scaling.shape[0] == 0:
            empty_cov = torch.empty(0, 3, 3, device=self.device)
            return (empty_cov, empty_cov) if return_inverse else empty_cov

        R = build_rotation(rotation_q)
        S_sq_diag = torch.diag_embed(scaling * scaling)
        covariance = R @ S_sq_diag @ R.transpose(1, 2)

        if return_inverse:
            inv_covariance = build_covariance_inverse(R, scaling, eps)

            if torch.isnan(inv_covariance).any() or torch.isinf(inv_covariance).any():
                print(
                    "Warning: NaN or Inf detected in inverse covariance. Replacing offending matrices with identity."
                )
                bad_indices = torch.isnan(inv_covariance).any(dim=(1, 2)) | torch.isinf(
                    inv_covariance
                ).any(dim=(1, 2))
                identity = torch.eye(
                    3, device=self.device, dtype=inv_covariance.dtype
                ).expand(bad_indices.sum(), -1, -1)
                inv_covariance[bad_indices] = identity
            return covariance, inv_covariance
        else:
            return covariance

    def init_gaussians(
        self,
        env_dims: Optional[torch.Tensor] = None,
        num_points: Optional[int] = None,
        point_cloud: Optional[torch.Tensor] = None,
    ):
        """Initializes Gaussian parameters."""
        num_to_init = num_points if num_points is not None else self.initial_gaussians

        if point_cloud is not None:
            num_available_points = point_cloud.shape[0]
            print(f"Point cloud provided with {num_available_points} points.")
            if num_available_points == 0:
                print(
                    "Warning: Point cloud is empty. Falling back to random initialization."
                )
                point_cloud = None
            else:
                if num_to_init > num_available_points:
                    print(
                        f"Warning: Requested {num_to_init} Gaussians, but point cloud only has {num_available_points}. "
                        f"Using all {num_available_points} points."
                    )
                    num_to_init = num_available_points
                    indices = torch.arange(num_available_points)
                else:
                    print(
                        f"Randomly sampling {num_to_init} points from the point cloud."
                    )

                    indices_np = np.random.choice(
                        num_available_points, num_to_init, replace=False
                    )
                    indices = torch.from_numpy(indices_np).long()

                xyz = point_cloud[indices].to(self.device).float()
                if xyz.shape[1] != 3:
                    raise ValueError(
                        f"Point cloud must have shape (N, 3), got {point_cloud.shape}"
                    )

        if point_cloud is None:
            if env_dims is not None:
                print(
                    f"Initializing {num_to_init} random Gaussians within environment dimensions."
                )
                env_min = env_dims[:, 0].to(self.device)
                env_max = env_dims[:, 1].to(self.device)
                if env_min.shape != (3,) or env_max.shape != (3,):
                    raise ValueError(
                        f"env_dims should result in shapes (3,), got min: {env_min.shape}, max: {env_max.shape}"
                    )
                xyz = (
                    torch.rand(num_to_init, 3, device=self.device) * (env_max - env_min)
                    + env_min
                )
            else:
                print(
                    f"Initializing {num_to_init} random Gaussians (no env_dims/point_cloud provided, using [-1, 1] range)."
                )
                xyz = (torch.rand(num_to_init, 3, device=self.device) * 2 - 1) * 1.0

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        scales = torch.ones(num_to_init, 3, device=self.device) * self.init_log_scale
        self._scaling = nn.Parameter(scales.requires_grad_(True))

        rots = torch.zeros((num_to_init, 4), device=self.device)
        rots[:, 0] = 1.0
        self._rotation = nn.Parameter(rots.requires_grad_(True))

        print(f"GCF Model initialized with {self.get_xyz.shape[0]} Gaussians.")

    def get_params(self, lr_dict: Dict[str, float]) -> list:
        """Returns parameter groups for the optimizer."""
        param_groups = [
            {"params": [self._xyz], "lr": lr_dict.get("xyz", 0.0), "name": "xyz"},
            {
                "params": [self._rotation],
                "lr": lr_dict.get("rotation", 0.0),
                "name": "rotation",
            },
            {
                "params": [self._scaling],
                "lr": lr_dict.get("scaling", 0.0),
                "name": "scaling",
            },
            {
                "params": self.attribute_network.parameters(),
                "lr": lr_dict.get("attribute_net", 0.0),
                "name": "attribute_net",
            },
            {
                "params": self.contribution_decoder.parameters(),
                "lr": lr_dict.get("decoder", 0.0),
                "name": "decoder",
            },
        ]
        return param_groups

    def training_setup(self, training_args: Any):
        """Setup optimizer (Adam) and learning rate schedulers."""

        lr_map = {
            "xyz": training_args.position_lr_init,
            "rotation": training_args.rotation_lr,
            "scaling": training_args.scaling_lr,
            "attribute_net": training_args.attribute_net_lr,
            "decoder": training_args.decoder_lr,
        }
        params = self.get_params(lr_map)

        self.optimizer = torch.optim.Adam(
            params,
            lr=0.0,
            eps=1e-15,
            weight_decay=training_args.weight_decay,
        )

        self.lr_schedulers["xyz"] = get_expon_lr_func(
            lr_init=training_args.position_lr_init,
            lr_final=training_args.position_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.iterations,
        )

        for name, lr_init in lr_map.items():
            if name != "xyz":
                self.lr_schedulers[name] = lambda step, lr=lr_init: lr

    def update_learning_rate(self, iteration: int, training_args: Any):
        """Update learning rates for all parameter groups based on schedulers."""
        if not self.optimizer:
            return
        for param_group in self.optimizer.param_groups:
            name = param_group["name"]
            if name in self.lr_schedulers:
                new_lr = self.lr_schedulers[name](iteration)

                if name == "xyz" and iteration > training_args.stop_xyz_iter:
                    new_lr = 0.0
                param_group["lr"] = new_lr

    def save(self, filepath: Path, iteration: Optional[int] = None):
        """Save model state."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        state_dict = {
            "iteration": iteration,
            "xyz": self._xyz.detach().cpu(),
            "rotation": self._rotation.detach().cpu(),
            "scaling": self._scaling.detach().cpu(),
            "attribute_network_state_dict": self.attribute_network.state_dict(),
            "decoder_state_dict": self.contribution_decoder.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer else None
            ),
            "config": {
                "num_tx_ant": self.num_tx_ant,
                "num_rx_ant": self.num_rx_ant,
                "latent_dim": self.latent_dim,
                "attribute_mlp_hidden_dim": self.attribute_network.network.hidden_dim,
                "attribute_mlp_num_layers": self.attribute_network.network.num_layers,
                "attribute_pos_enc_freqs": self.attribute_network.pos_encoder_mean.num_freqs,
                "decoder_hidden_dim": self.contribution_decoder.hidden_dim,
                "decoder_num_layers": self.contribution_decoder.num_layers,
            },
        }
        torch.save(state_dict, str(filepath))

    @classmethod
    def load(
        cls, filepath: Path, device: torch.device, training_args: Optional[Any] = None
    ):
        """Load model state."""
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")

        state_dict = torch.load(str(filepath), map_location=device)
        config = state_dict["config"]

        model = cls(
            num_tx_ant=config["num_tx_ant"],
            num_rx_ant=config["num_rx_ant"],
            latent_dim=config["latent_dim"],
            attribute_mlp_hidden_dim=config.get("attribute_mlp_hidden_dim", 64),
            attribute_mlp_num_layers=config.get("attribute_mlp_num_layers", 3),
            attribute_pos_enc_freqs=config.get("attribute_pos_enc_freqs", 10),
            decoder_hidden_dim=config.get("decoder_hidden_dim", 64),
            decoder_num_layers=config.get("decoder_num_layers", 3),
            device=device,
        )

        model._xyz = nn.Parameter(state_dict["xyz"].to(device).requires_grad_(True))
        model._rotation = nn.Parameter(
            state_dict["rotation"].to(device).requires_grad_(True)
        )
        model._scaling = nn.Parameter(
            state_dict["scaling"].to(device).requires_grad_(True)
        )

        model.attribute_network.load_state_dict(
            state_dict["attribute_network_state_dict"]
        )
        model.contribution_decoder.load_state_dict(state_dict["decoder_state_dict"])

        iteration = state_dict.get("iteration", 0)

        if training_args is not None:
            model.training_setup(training_args)
            if model.optimizer and state_dict.get("optimizer_state_dict"):
                try:

                    model.optimizer.load_state_dict(state_dict["optimizer_state_dict"])

                    for state in model.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                    print("Optimizer state loaded successfully.")
                except Exception as e:
                    print(
                        f"Warning: Could not load optimizer state: {e}. Optimizer state reset."
                    )

                    model.optimizer.state = {}
            else:
                print("Optimizer state not found in checkpoint or optimizer not setup.")

        print(f"GCF Model loaded from {filepath} (iteration {iteration}).")
        print(f"Loaded model has {model.get_xyz.shape[0]} Gaussians.")

        return model, iteration

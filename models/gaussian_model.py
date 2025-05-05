# models/gaussian_model.py

import math
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, open_dict

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
        attribute_hidden_dim: int = 64,
        attribute_num_layers: int = 3,
        attribute_pos_enc_freqs: int = 10,
        decoder_hidden_dim: int = 64,
        decoder_num_layers: int = 4,
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
            mlp_hidden_dim=attribute_hidden_dim,
            mlp_num_layers=attribute_num_layers,
            pos_encoding_freqs=attribute_pos_enc_freqs,
        ).to(device)

        self.contribution_decoder = ContributionDecoderNetwork(
            latent_dim=latent_dim,
            output_dim=num_tx_ant * num_rx_ant,
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
        """Setup activation functions for Gaussian parameters."""
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = lambda r: F.normalize(r, p=2, dim=-1)

    @property
    def get_xyz(self):
        """Returns Gaussian positions (means)."""
        return self._xyz

    @property
    def get_scaling(self):
        """Returns activated and clamped Gaussian scales."""

        return self.scaling_activation(self._scaling).clamp(min=1e-8)

    @property
    def get_rotation(self):
        """Returns normalized Gaussian rotations (quaternions)."""
        return self.rotation_activation(self._rotation)

    def get_attributes_and_activation(
        self, tx_position: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes latent features and *activated* base activations dynamically."""
        if self._xyz.shape[0] == 0:

            return torch.empty(0, self.latent_dim, device=self.device), torch.empty(
                0, 1, device=self.device
            )

        if tx_position.dim() == 1:
            tx_position_exp = tx_position.unsqueeze(0).expand(self._xyz.shape[0], -1)
        elif tx_position.shape[0] == 1:
            tx_position_exp = tx_position.expand(self._xyz.shape[0], -1)
        elif tx_position.shape[0] == self._xyz.shape[0]:
            tx_position_exp = tx_position
        else:
            raise ValueError(
                f"tx_position shape {tx_position.shape} mismatch with Gaussians {self._xyz.shape[0]}"
            )

        latent_features, base_activations_logits = self.attribute_network(
            self._xyz, tx_position_exp
        )

        base_activations_activated = self.opacity_activation(base_activations_logits)

        return (
            latent_features,
            base_activations_activated,
        )

    def get_base_activation_logits(self, tx_position: torch.Tensor) -> torch.Tensor:
        """Returns the base activation *logits* (before sigmoid), computed dynamically."""
        if self._xyz.shape[0] == 0:
            return torch.empty(0, 1, device=self.device)

        if tx_position.dim() == 1:
            tx_position_exp = tx_position.unsqueeze(0).expand(self._xyz.shape[0], -1)
        elif tx_position.shape[0] == 1:
            tx_position_exp = tx_position.expand(self._xyz.shape[0], -1)
        elif tx_position.shape[0] == self._xyz.shape[0]:
            tx_position_exp = tx_position
        else:
            raise ValueError("tx_position shape mismatch")

        _, base_activations_logits = self.attribute_network(self._xyz, tx_position_exp)
        return base_activations_logits

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
                warnings.warn(
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
        """Initializes Gaussian parameters (position, rotation, scale)."""
        num_to_init = num_points if num_points is not None else 0
        if num_to_init <= 0:
            warnings.warn(
                "Warning: No Gaussians requested for initialization (num_points <= 0)."
            )
            self._xyz = nn.Parameter(
                torch.empty(0, 3, device=self.device).requires_grad_(True)
            )
            self._rotation = nn.Parameter(
                torch.empty(0, 4, device=self.device).requires_grad_(True)
            )
            self._scaling = nn.Parameter(
                torch.empty(0, 3, device=self.device).requires_grad_(True)
            )
            return

        if point_cloud is not None:
            num_available_points = point_cloud.shape[0]
            print(f"Point cloud provided with {num_available_points} points.")
            if num_available_points == 0:
                warnings.warn(
                    "Warning: Point cloud is empty. Falling back to random initialization within env_dims (if provided)."
                )
                point_cloud = None
            else:
                if num_to_init > num_available_points:
                    warnings.warn(
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
            if env_dims is not None and env_dims.shape == (3, 2):
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
                warnings.warn(
                    f"Warning: No point cloud or valid env_dims provided. "
                    f"Initializing {num_to_init} random Gaussians in [-1, 1] range."
                )
                xyz = (torch.rand(num_to_init, 3, device=self.device) * 2 - 1) * 1.0

        self._xyz = nn.Parameter(xyz.requires_grad_(True))

        scales = torch.full(
            (num_to_init, 3), self.init_log_scale.item(), device=self.device
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))

        rots = torch.zeros((num_to_init, 4), device=self.device)
        rots[:, 0] = 1.0
        self._rotation = nn.Parameter(rots.requires_grad_(True))

        print(f"GCF Model initialized with {self.get_xyz.shape[0]} Gaussians.")

    def get_params(self, lr_dict: Dict[str, float]) -> list:
        """Returns parameter groups for the optimizer with specified learning
        rates."""
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

    def training_setup(self, cfg: DictConfig):
        """Setup optimizer and learning rate schedulers based on config."""

        lr_map = {
            "xyz": cfg.training.learning_rate.position_init,
            "rotation": cfg.training.learning_rate.rotation,
            "scaling": cfg.training.learning_rate.scaling,
            "attribute_net": cfg.training.learning_rate.attribute_net,
            "decoder": cfg.training.learning_rate.decoder,
        }
        params = self.get_params(lr_map)

        self.optimizer = torch.optim.AdamW(
            params,
            lr=0.0,
            eps=cfg.training.optimizer.eps,
            weight_decay=cfg.training.optimizer.weight_decay,
        )
        print(
            f"Optimizer AdamW initialized with weight decay: {cfg.training.optimizer.weight_decay}"
        )

        self.lr_schedulers = {}
        self.lr_schedulers["xyz"] = get_expon_lr_func(
            lr_init=cfg.training.learning_rate.position_init,
            lr_final=cfg.training.learning_rate.position_final,
            lr_delay_mult=cfg.training.learning_rate.position_delay_mult,
            max_steps=cfg.training.iterations,
        )

        for name, lr_init in lr_map.items():
            if name != "xyz":
                self.lr_schedulers[name] = lambda step, lr=lr_init: lr

    def update_learning_rate(self, iteration: int, cfg: DictConfig):
        """Update learning rates for all parameter groups based on schedulers and iteration."""
        if not self.optimizer:
            warnings.warn(
                "Warning: Optimizer not initialized, cannot update learning rate."
            )
            return

        stop_xyz_iter = int(cfg.training.iterations * cfg.training.stop_xyz_iter_ratio)
        for param_group in self.optimizer.param_groups:
            name = param_group["name"]
            if name in self.lr_schedulers:
                new_lr = self.lr_schedulers[name](iteration)
                if name == "xyz" and iteration >= stop_xyz_iter:
                    new_lr = 0.0
                param_group["lr"] = new_lr

    def save(
        self,
        filepath: Path,
        iteration: Optional[int] = None,
        cfg: Optional[DictConfig] = None,
    ):
        """Save model state, optimizer state, and configuration."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        config_dict_to_save = None
        if cfg is None:
            warnings.warn(
                "Warning: Saving model without config. Loading might be incomplete."
            )
        else:
            temp_cfg = cfg.copy()
            with open_dict(temp_cfg):
                if "model" not in temp_cfg:
                    temp_cfg.model = OmegaConf.create()
                temp_cfg.model.num_tx_ant = self.num_tx_ant
                temp_cfg.model.num_rx_ant = self.num_rx_ant
            config_dict_to_save = OmegaConf.to_container(temp_cfg, resolve=True)

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
            "config_dict": config_dict_to_save,
        }
        torch.save(state_dict, str(filepath))

    @classmethod
    def load(
        cls,
        filepath: Path,
        device: torch.device,
        resume_cfg: Optional[DictConfig] = None,
    ):
        """Load model state from a checkpoint."""
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")

        state_dict = torch.load(str(filepath), map_location=device, weights_only=False)
        if "config_dict" not in state_dict or state_dict["config_dict"] is None:
            raise ValueError(
                f"Checkpoint {filepath} does not contain 'config_dict'. Cannot load model architecture."
            )
        config_dict = state_dict["config_dict"]
        base_cfg = OmegaConf.load("configs/default.yaml")
        checkpoint_cfg = OmegaConf.merge(base_cfg, OmegaConf.create(config_dict))

        try:
            model = cls(
                num_tx_ant=checkpoint_cfg.model.num_tx_ant,
                num_rx_ant=checkpoint_cfg.model.num_rx_ant,
                latent_dim=checkpoint_cfg.model.latent_dim,
                attribute_hidden_dim=checkpoint_cfg.model.attribute_network.hidden_dim,
                attribute_num_layers=checkpoint_cfg.model.attribute_network.num_layers,
                attribute_pos_enc_freqs=checkpoint_cfg.model.attribute_network.pos_enc_freqs,
                decoder_hidden_dim=checkpoint_cfg.model.contribution_decoder.hidden_dim,
                decoder_num_layers=checkpoint_cfg.model.contribution_decoder.num_layers,
                init_opacity_value=checkpoint_cfg.initialization.opacity_value,
                init_scale_value=checkpoint_cfg.initialization.scale_value,
                device=device,
            )
        except Exception as e:
            print("Error during model instantiation using loaded config:")
            print(OmegaConf.to_yaml(checkpoint_cfg))
            raise e

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

        if resume_cfg is not None:
            model.training_setup(resume_cfg)
            if model.optimizer and state_dict.get("optimizer_state_dict"):
                try:
                    model.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
                    for state in model.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                    print("Optimizer state loaded successfully.")
                except Exception as e:
                    warnings.warn(
                        f"Warning: Could not load optimizer state: {e}. Optimizer state reset."
                    )
                    model.optimizer.state = {}
            else:
                print(
                    "Optimizer state not found in checkpoint or optimizer not setup for loading."
                )
        else:
            print(
                "Not resuming training (resume_cfg=None), optimizer state not loaded."
            )

        print(f"GCF Model loaded from {filepath} (iteration {iteration}).")
        print(f"Loaded model has {model.get_xyz.shape[0]} Gaussians.")

        return model, iteration

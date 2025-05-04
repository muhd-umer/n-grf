# models/gaussian_model.py

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import inverse_sigmoid  # still potentially useful for init opacity logit
from utils import build_covariance_inverse, build_rotation, get_expon_lr_func

# import updated networks
from .networks import AttributeNetwork, ContributionDecoderNetwork


class GaussianChannelFieldModel(nn.Module):
    """Gaussian channel field (GCF) model for predicting channel magnitude."""

    def __init__(
        self,
        num_tx_ant: int,
        num_rx_ant: int,
        latent_dim: int,
        attribute_hidden_dim: int = 64,
        attribute_num_layers: int = 3,
        attribute_pos_enc_freqs: int = 10,
        decoder_hidden_dim: int = 64,
        decoder_num_layers: int = 4,  # increased default slightly
        initial_gaussians: int = 30000,
        init_opacity_value: float = 0.1,  # initial base activation (before sigmoid) related value
        init_scale_value: float = 0.02,  # initial scale (before exp) related value
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__()
        self.num_tx_ant = num_tx_ant
        self.num_rx_ant = num_rx_ant
        self.latent_dim = latent_dim
        self.device = device

        # gaussian parameters (initialized later)
        self._xyz = nn.Parameter(torch.empty(0, 3, device=device))
        self._rotation = nn.Parameter(
            torch.empty(0, 4, device=device)
        )  # quaternion (w, x, y, z)
        self._scaling = nn.Parameter(torch.empty(0, 3, device=device))  # log scale

        # networks
        self.attribute_network = AttributeNetwork(
            latent_dim=latent_dim,
            mlp_hidden_dim=attribute_hidden_dim,
            mlp_num_layers=attribute_num_layers,
            pos_encoding_freqs=attribute_pos_enc_freqs,
        ).to(device)

        # decoder predicts normalized magnitude contributions (Nt * Nr outputs)
        self.contribution_decoder = ContributionDecoderNetwork(
            latent_dim=latent_dim,
            output_dim=num_tx_ant * num_rx_ant,  # direct magnitude prediction
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            # sigmoid activation is now inside ContributionDecoderNetwork
        ).to(device)

        # initialization helpers
        self.initial_gaussians = initial_gaussians
        # store the logit corresponding to the initial opacity value
        self.init_opacity_logit = inverse_sigmoid(
            torch.tensor(init_opacity_value, device=device)
        )
        # store the log scale corresponding to the initial scale value
        self.init_log_scale = torch.log(torch.tensor(init_scale_value, device=device))

        self.optimizer = None
        self.lr_schedulers = {}

        self.setup_activations()

    def setup_activations(self):
        """Setup activation functions for Gaussian parameters."""
        self.scaling_activation = torch.exp  # scales are stored in log space
        self.opacity_activation = torch.sigmoid  # activates the base activation logits
        self.rotation_activation = lambda r: F.normalize(
            r, p=2, dim=-1
        )  # normalize quaternions

    @property
    def get_xyz(self):
        """Returns Gaussian positions (means)."""
        return self._xyz

    @property
    def get_scaling(self):
        """Returns activated and clamped Gaussian scales."""
        # clamp to prevent scales from becoming too small or zero
        return self.scaling_activation(self._scaling).clamp(min=1e-8)

    @property
    def get_rotation(self):
        """Returns normalized Gaussian rotations (quaternions)."""
        return self.rotation_activation(self._rotation)

    def get_attributes(
        self, tx_position: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes latent features and base activation logits dynamically."""
        if self._xyz.shape[0] == 0:
            # handle case with no gaussians
            return torch.empty(0, self.latent_dim, device=self.device), torch.empty(
                0, 1, device=self.device
            )

        # ensure tx_position is expanded correctly
        if tx_position.dim() == 1:
            tx_position_exp = tx_position.unsqueeze(0).expand(self._xyz.shape[0], -1)
        elif tx_position.shape[0] == 1:
            tx_position_exp = tx_position.expand(self._xyz.shape[0], -1)
        elif tx_position.shape[0] == self._xyz.shape[0]:
            tx_position_exp = tx_position
        else:
            raise ValueError("tx_position shape mismatch")

        # pass means and expanded tx position to attribute network
        latent_features, base_activations_logits = self.attribute_network(
            self._xyz, tx_position_exp
        )
        return latent_features, base_activations_logits  # return logits directly

    def get_opacity_activated(self, tx_position: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid-activated base activations, computed dynamically."""
        _, base_activations_logits = self.get_attributes(tx_position)
        # apply sigmoid activation to the logits
        return self.opacity_activation(base_activations_logits)

    def get_covariance(
        self, return_inverse=False, eps=1e-6
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Computes covariance matrix Σ and optionally its inverse Σ^-1."""
        scaling = self.get_scaling  # activated scales
        rotation_q = self.get_rotation  # normalized quaternions

        if scaling.shape[0] == 0:
            # handle no gaussians case
            empty_cov = torch.empty(0, 3, 3, device=self.device)
            return (empty_cov, empty_cov) if return_inverse else empty_cov

        # build rotation matrix from quaternion
        R = build_rotation(rotation_q)  # (N, 3, 3)
        # create diagonal matrix for squared scales
        S_sq_diag = torch.diag_embed(scaling * scaling)  # (N, 3, 3)
        # compute covariance: Sigma = R * S^2 * R^T
        covariance = R @ S_sq_diag @ R.transpose(1, 2)

        if return_inverse:
            # compute inverse covariance: Sigma^-1 = R * S^-2 * R^T
            inv_covariance = build_covariance_inverse(R, scaling, eps)

            # check for NaNs/Infs in inverse covariance (can happen if scale is near zero)
            if torch.isnan(inv_covariance).any() or torch.isinf(inv_covariance).any():
                print(
                    "Warning: NaN or Inf detected in inverse covariance. Replacing offending matrices with identity."
                )
                # identify gaussians with bad inverse covariance
                bad_indices = torch.isnan(inv_covariance).any(dim=(1, 2)) | torch.isinf(
                    inv_covariance
                ).any(dim=(1, 2))
                # create identity matrices for replacement
                identity = torch.eye(
                    3, device=self.device, dtype=inv_covariance.dtype
                ).expand(bad_indices.sum(), -1, -1)
                # replace bad matrices
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
        num_to_init = num_points if num_points is not None else self.initial_gaussians
        if num_to_init <= 0:
            print("Warning: No Gaussians requested for initialization.")
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

        # initialize positions (xyz)
        if point_cloud is not None:
            num_available_points = point_cloud.shape[0]
            print(f"Point cloud provided with {num_available_points} points.")
            if num_available_points == 0:
                print(
                    "Warning: Point cloud is empty. Falling back to random initialization within env_dims (if provided)."
                )
                point_cloud = None  # fallback
            else:
                if num_to_init > num_available_points:
                    print(
                        f"Warning: Requested {num_to_init} Gaussians, but point cloud only has {num_available_points}. "
                        f"Using all {num_available_points} points and potentially adding random points."
                    )
                    # use all points and maybe add more later if needed? or just use available? let's use available.
                    num_to_init = num_available_points
                    indices = torch.arange(num_available_points)

                else:
                    print(
                        f"Randomly sampling {num_to_init} points from the point cloud."
                    )
                    # sample without replacement
                    indices_np = np.random.choice(
                        num_available_points, num_to_init, replace=False
                    )
                    indices = torch.from_numpy(indices_np).long()

                # select points and ensure correct type/device
                xyz = point_cloud[indices].to(self.device).float()
                if xyz.shape[1] != 3:
                    raise ValueError(
                        f"Point cloud must have shape (N, 3), got {point_cloud.shape}"
                    )

        # if no point cloud or fallback needed
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
                # generate random points within bounds
                xyz = (
                    torch.rand(num_to_init, 3, device=self.device) * (env_max - env_min)
                    + env_min
                )
            else:
                # default random initialization if no bounds given
                print(
                    f"Warning: No point cloud or valid env_dims provided. "
                    f"Initializing {num_to_init} random Gaussians in [-1, 1] range."
                )
                xyz = (
                    torch.rand(num_to_init, 3, device=self.device) * 2 - 1
                ) * 1.0  # scale appropriately if needed

        # set parameters
        self._xyz = nn.Parameter(xyz.requires_grad_(True))

        # initialize scales (log scale)
        scales = torch.full(
            (num_to_init, 3), self.init_log_scale.item(), device=self.device
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))

        # initialize rotations (identity quaternion: w=1, x=y=z=0)
        rots = torch.zeros((num_to_init, 4), device=self.device)
        rots[:, 0] = 1.0
        self._rotation = nn.Parameter(rots.requires_grad_(True))

        print(f"GCF Model initialized with {self.get_xyz.shape[0]} Gaussians.")

    def get_params(self, lr_dict: Dict[str, float]) -> list:
        """Returns parameter groups for the optimizer with specified learning rates."""
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
            # include network parameters
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
        # filter out groups with no parameters (can happen if networks are empty/frozen)
        # param_groups = [pg for pg in param_groups if len(pg['params']) > 0]
        return param_groups

    def training_setup(self, training_args: Any):
        """Setup optimizer (AdamW) and learning rate schedulers based on training args."""

        # map training args LRs to parameter group names
        lr_map = {
            "xyz": training_args.position_lr_init,
            "rotation": training_args.rotation_lr,
            "scaling": training_args.scaling_lr,
            "attribute_net": training_args.attribute_net_lr,
            "decoder": training_args.decoder_lr,
        }
        params = self.get_params(lr_map)

        # use AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            params,
            lr=0.0,  # initial LR set by scheduler
            eps=(
                training_args.optimizer_eps
                if hasattr(training_args, "optimizer_eps")
                else 1e-8
            ),  # allow configuring eps
            weight_decay=training_args.weight_decay,
        )
        print(
            f"Optimizer AdamW initialized with weight decay: {training_args.weight_decay}"
        )

        # setup LR schedulers
        self.lr_schedulers = {}
        # exponential decay for positions
        self.lr_schedulers["xyz"] = get_expon_lr_func(
            lr_init=training_args.position_lr_init,
            lr_final=training_args.position_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.iterations,
        )

        # constant LR for other parameters (can be changed to schedulers if needed)
        for name, lr_init in lr_map.items():
            if name != "xyz":
                # use a simple lambda for constant LR
                self.lr_schedulers[name] = lambda step, lr=lr_init: lr
        print("Learning rate schedulers set up.")

    def update_learning_rate(self, iteration: int, training_args: Any):
        """Update learning rates for all parameter groups based on schedulers and iteration."""
        if not self.optimizer:
            print("Warning: Optimizer not initialized, cannot update learning rate.")
            return

        for param_group in self.optimizer.param_groups:
            name = param_group["name"]
            if name in self.lr_schedulers:
                # get the scheduled LR
                new_lr = self.lr_schedulers[name](iteration)

                # apply position freezing logic
                if name == "xyz" and iteration >= training_args.stop_xyz_iter:
                    new_lr = 0.0  # freeze position updates

                # assign the new LR to the parameter group
                param_group["lr"] = new_lr
            # else:
            # print(f"Warning: No LR scheduler found for parameter group '{name}'.")

    def save(self, filepath: Path, iteration: Optional[int] = None):
        """Save model state, optimizer state, and configuration."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # ensure parameters are detached and on CPU for saving
        state_dict = {
            "iteration": iteration,
            "xyz": self._xyz.detach().cpu(),
            "rotation": self._rotation.detach().cpu(),
            "scaling": self._scaling.detach().cpu(),
            "attribute_network_state_dict": self.attribute_network.state_dict(),
            "decoder_state_dict": self.contribution_decoder.state_dict(),
            # save optimizer state if it exists
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer else None
            ),
            # save model configuration
            "config": {
                "num_tx_ant": self.num_tx_ant,
                "num_rx_ant": self.num_rx_ant,
                "latent_dim": self.latent_dim,
                "attribute_hidden_dim": self.attribute_network.network.hidden_dim,
                "attribute_num_layers": self.attribute_network.network.num_layers,
                "attribute_pos_enc_freqs": self.attribute_network.pos_encoder_mean.num_freqs,
                "decoder_hidden_dim": self.contribution_decoder.hidden_dim,
                "decoder_num_layers": self.contribution_decoder.num_layers,
                # store initial values used, might be useful
                # "initial_gaussians": self.initial_gaussians,
                # "init_opacity_value": self.init_opacity_logit, # maybe save original value?
                # "init_scale_value": self.init_log_scale, # maybe save original value?
            },
        }
        torch.save(state_dict, str(filepath))
        # print(f"Model state saved to {filepath} at iteration {iteration}")

    @classmethod
    def load(
        cls, filepath: Path, device: torch.device, training_args: Optional[Any] = None
    ):
        """Load model state from a checkpoint."""
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")

        # load state dict onto the specified device
        state_dict = torch.load(
            str(filepath), map_location=device, weights_only=False
        )  # weights_only=False needed for optimizer
        config = state_dict["config"]

        # create a new model instance with the loaded configuration
        model = cls(
            num_tx_ant=config["num_tx_ant"],
            num_rx_ant=config["num_rx_ant"],
            latent_dim=config["latent_dim"],
            attribute_hidden_dim=config.get(
                "attribute_hidden_dim", 64
            ),  # use get for backward compatibility
            attribute_num_layers=config.get("attribute_num_layers", 3),
            attribute_pos_enc_freqs=config.get("attribute_pos_enc_freqs", 10),
            decoder_hidden_dim=config.get("decoder_hidden_dim", 64),
            decoder_num_layers=config.get("decoder_num_layers", 4),  # updated default
            device=device,
            # initial_gaussians etc. are not needed here as parameters are loaded directly
        )

        # load gaussian parameters
        model._xyz = nn.Parameter(state_dict["xyz"].to(device).requires_grad_(True))
        model._rotation = nn.Parameter(
            state_dict["rotation"].to(device).requires_grad_(True)
        )
        model._scaling = nn.Parameter(
            state_dict["scaling"].to(device).requires_grad_(True)
        )

        # load network states
        model.attribute_network.load_state_dict(
            state_dict["attribute_network_state_dict"]
        )
        model.contribution_decoder.load_state_dict(state_dict["decoder_state_dict"])

        # get iteration number from checkpoint
        iteration = state_dict.get("iteration", 0)  # default to 0 if not found

        # setup training components (optimizer, schedulers) if training_args are provided
        if training_args is not None:
            model.training_setup(
                training_args
            )  # re-initialize optimizer and schedulers
            # load optimizer state if available in checkpoint and optimizer exists
            if model.optimizer and state_dict.get("optimizer_state_dict"):
                try:
                    model.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
                    # ensure optimizer state tensors are on the correct device
                    for state in model.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                    print("Optimizer state loaded successfully.")
                except Exception as e:
                    print(
                        f"Warning: Could not load optimizer state: {e}. Optimizer state reset."
                    )
                    # reset optimizer state if loading fails
                    model.optimizer.state = {}  # or re-initialize?
            else:
                print(
                    "Optimizer state not found in checkpoint or optimizer not setup for loading."
                )
        else:
            print("No training_args provided, optimizer state not loaded.")

        print(f"GCF Model loaded from {filepath} (iteration {iteration}).")
        print(f"Loaded model has {model.get_xyz.shape[0]} Gaussians.")

        return model, iteration

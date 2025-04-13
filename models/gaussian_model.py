# models/gaussian_model.py

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from simple_knn._C import distCUDA2  # type: ignore

from utils.prop_utils import compute_path_loss, compute_phase_rotation
from utils.transform_utils import (
    build_scaling_rotation,
    inverse_sigmoid,
    strip_symmetric,
)

from .encoder import EncoderConfig, FeatureEncoder


class GaussianModel(nn.Module):
    """Gaussian model for channel reconstruction.

    Each Gaussian primitive captures a point in the environment that affects
    wireless signal propagation. The Gaussians are initialized from point clouds
    and optimized to reconstruct wireless channels.

    Initialization details:
    - xyz: Point positions from input point cloud
    - rotation: 4D quaternions initialized as [1,0,0,0] (identity)
    - scaling: Log of point-wise distances to enforce minimum scale
    - opacity: Inverse sigmoid of constant value (0.1)
    - features_dc: Main spherical harmonic features
    - features_rest: Higher order spherical harmonics initialized to zero

    Args:
        encoder_cfg: Configuration for the encoder
    """

    def __init__(
        self,
        encoder_cfg: Optional[EncoderConfig] = None,
    ):
        super().__init__()

        self.encoder_cfg = encoder_cfg or EncoderConfig()
        self.encoder = FeatureEncoder(self.encoder_cfg)

        # initialize empty tensors; will be set in init_from_pc
        self._xyz = torch.empty(0)  # positions
        self._rotation = torch.empty(0)  # rotation quaternions
        self._scaling = torch.empty(0)  # scaling factors
        self._opacity = torch.empty(0)  # opacity values
        self.features = torch.empty(0)  # wireless features

        # training state
        self.optimizer = None

        self.setup_functions()

    def setup_functions(self):
        """Setup activation and transformation functions used by the model."""

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            """Build covariance matrix from scaling and rotation.

            Args:
                scaling: Scale factors per Gaussian
                scaling_modifier: Global scale modifier
                rotation: Rotation quaternions per Gaussian

            Returns:
                Covariance matrices in symmetric form
            """
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        # setup activation functions
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.covariance_activation = build_covariance_from_scaling_rotation

    def init_from_pc(
        self,
        points: torch.Tensor,
        tx_position: torch.Tensor = None,
        frequency: float = None,
        use_physics_init: bool = False,
    ):
        """Initialize Gaussian properties from point cloud.

        Args:
            points: Point cloud tensor of shape [N, 3]
            tx_position: Transmitter position [3], required for physics init
            frequency: Signal frequency in Hz, required for physics init
            use_physics_init: Whether to use physics-based feature initialization
        """
        num_points = points.shape[0]
        device = points.device

        self._xyz = nn.Parameter(points.to(device))

        # compute scales based on point cloud density
        dist2 = torch.clamp_min(distCUDA2(points), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        self._scaling = nn.Parameter(scales.to(device))

        rots = torch.zeros((num_points, 4), device=device)
        rots[:, 0] = 1
        self._rotation = nn.Parameter(rots)

        # features
        if use_physics_init and tx_position is not None and frequency is not None:
            from utils.prop_utils import compute_path_loss, compute_phase_rotation

            tx_distances = torch.sqrt(torch.sum((self._xyz - tx_position) ** 2, dim=1))
            c = 299792458.0
            wavelength = c / frequency

            attenuation = compute_path_loss(tx_distances, wavelength)
            phase_rotation = compute_phase_rotation(tx_distances, wavelength)

            self.features = torch.cat([attenuation, phase_rotation], dim=1).to(device)
        else:
            self.features = torch.zeros(
                (num_points, 2), device=device  # 0: attenuation, 1: phase_rotation
            ).float()

        # initialize opacity
        init_opacity = 0.1 * torch.ones((num_points, 1), device=device)
        self._opacity = nn.Parameter(inverse_sigmoid(init_opacity))

    def init_randomly(
        self,
        num_points: int,
        env_dims: torch.Tensor,
        tx_position: torch.Tensor = None,
        frequency: float = None,
        use_physics_init: bool = False,
    ):
        """Initialize Gaussian properties randomly within environment dimensions.

        Args:
            num_points: Number of random points to initialize
            env_dims: Environment dimensions as [3, 2] tensor with min/max per dimension
            tx_position: Transmitter position [3], required for physics init
            frequency: Signal frequency in Hz, required for physics init
            use_physics_init: Whether to use physics-based feature initialization
        """
        device = env_dims.device

        env_min = env_dims[:, 0]
        env_max = env_dims[:, 1]

        random_points = torch.rand(num_points, 3, device=device)
        random_points = random_points * (env_max - env_min) + env_min

        self.init_from_pc(random_points, tx_position, frequency, use_physics_init)

    @property
    def get_scaling(self):
        """Get scaling factors with activation applied."""
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        """Get rotations with normalization applied."""
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        """Get Gaussian positions."""
        return self._xyz

    @property
    def get_opacity(self):
        """Get opacity values with sigmoid activation."""
        return self.opacity_activation(self._opacity)

    @property
    def get_features(self):
        """Get wireless-related features."""
        return self.features

    def get_covariance(self, scaling_modifier: float = 1.0):
        """Compute covariance matrices for each Gaussian.

        Args:
            scaling_modifier: Global scaling factor modifier

        Returns:
            Covariance matrices in symmetric form
        """
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def save(self, filepath, save_optimizer=True, iteration=None, best_val_loss=None):
        """Save model weights and optionally training state to file.

        Args:
            filepath: Path where to save the model
            save_optimizer: Whether to save optimizer states (for resuming training)
            iteration: Current training iteration for resuming
            best_val_loss: Best validation loss achieved so far
        """
        from pathlib import Path

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_state: Dict[str, Any] = {
            "encoder_config": self.encoder_cfg,
            "xyz": self._xyz,
            "rotation": self._rotation,
            "scaling": self._scaling,
            "opacity": self._opacity,
            "features": self.features,
        }

        if save_optimizer:
            model_state.update(
                {
                    "optimizer_state": (
                        self.optimizer.state_dict() if self.optimizer else None
                    ),
                    "encoder_optimizer_state": (
                        self.encoder_optimizer.state_dict()
                        if hasattr(self, "encoder_optimizer")
                        else None
                    ),
                    "iteration": iteration,
                    "best_val_loss": best_val_loss,
                }
            )

        torch.save(model_state, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath, device=None, training_args=None):
        """Load model from file and return an initialized model instance.

        Args:
            filepath: Path to the saved model file
            device: Device to load the model to (default: current CUDA device)
            training_args: Optional training arguments for resuming training

        Returns:
            An initialized GaussianModel instance with loaded weights
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state = torch.load(filepath, map_location=device, weights_only=False)

        encoder_cfg = state.get("encoder_config", None)

        model = cls(
            encoder_cfg=encoder_cfg,
        )

        model._xyz = state["xyz"].to(device)
        model._rotation = state["rotation"].to(device)
        model._scaling = state["scaling"].to(device)
        model._opacity = state["opacity"].to(device)
        model.features = state["features"].to(device)

        if training_args is not None and "optimizer_state" in state:
            model.training_setup(training_args)

            if state["optimizer_state"] is not None:
                model.optimizer.load_state_dict(state["optimizer_state"])

                for param_group in model.optimizer.param_groups:
                    for param in param_group["params"]:
                        if param.grad is not None:
                            param.grad = param.grad.to(device)

            if (
                "encoder_optimizer_state" in state
                and state["encoder_optimizer_state"] is not None
            ):
                model.encoder_optimizer.load_state_dict(
                    state["encoder_optimizer_state"]
                )

        model.to(device)
        return model

    def embed_features(self, enc_data: Dict[str, Union[torch.Tensor, float]]):
        """Embed iteration of wireless data into Gaussian features.

        The wireless data should contain the following keys:
        - tx_pos: Transmitter position (3,)
        - rx_pos: Receiver position (3,)

        Args:
            enc_data: Dictionary containing wireless data tensors with keys
        """
        # extract data
        tx_pos = enc_data["tx_pos"]
        rx_pos = enc_data.get("rx_pos", None)

        # compute features
        attenuation, phase_rotation = self.encoder(self._xyz, tx_pos, rx_pos)

        # update features
        self.features = torch.cat([attenuation, phase_rotation], dim=-1)

    def to(self, device):
        """Override to() to ensure encoder also moves to the same device."""
        self.encoder = self.encoder.to(device)
        return super().to(device)

    def training_setup(self, training_args):
        """Setup optimizer and training parameters.

        Args:
            training_args: Training arguments including learning rates
        """
        param_groups = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init,
                "name": "xyz",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
        ]

        self.optimizer = torch.optim.Adam(
            param_groups,
            lr=0.0,
            fused=True,
        )
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=training_args.encoder_lr,
            weight_decay=training_args.weight_decay,
            fused=True,
        )

        from utils.train_utils import get_expon_lr_func

        self.position_lr_scheduler = get_expon_lr_func(
            lr_init=training_args.position_lr_init,
            lr_final=training_args.position_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.iterations,
        )

    def update_learning_rate(self, iteration):
        """Update learning rates based on schedulers.

        Args:
            iteration: Current training iteration
        """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group["lr"] = self.position_lr_scheduler(iteration)

    def reset_opacity(self):
        """Reset opacity for Gaussians with very low opacity."""
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )

        for group in self.optimizer.param_groups:
            if group["name"] == "opacity":
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    del self.optimizer.state[group["params"][0]]

                group["params"][0] = nn.Parameter(opacities_new.requires_grad_(True))

                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(opacities_new)
                    stored_state["exp_avg_sq"] = torch.zeros_like(opacities_new)
                    self.optimizer.state[group["params"][0]] = stored_state

                self._opacity = group["params"][0]
                break

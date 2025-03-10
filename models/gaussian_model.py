# models/gaussian_model.py

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from simple_knn._C import distCUDA2  # type: ignore

from utils.transform_utils import (
    build_scaling_rotation,
    inverse_sigmoid,
    strip_symmetric,
)

from .encoder import EncoderConfig, WirelessEncoder


@dataclass
class GaussianModelConfig:
    """Configuration for GaussianModel"""

    num_components: int = 2  # real and imaginary parts
    use_pred_normals: bool = False


class GaussianModel(nn.Module):
    """Gaussian model for wireless channel reconstruction.

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
    - Screen-space max radii tracking for adaptive density
    - Gradients and denominator accumulators for training

    Args:
        model_cfg: Configuration for the model
        encoder_cfg
    """

    def __init__(
        self,
        model_cfg: Optional[GaussianModelConfig] = None,
        encoder_cfg: Optional[EncoderConfig] = None,
    ):
        super().__init__()

        self.model_cfg = model_cfg or GaussianModelConfig()
        self.encoder_cfg = encoder_cfg or EncoderConfig()
        self.encoder = WirelessEncoder(self.encoder_cfg)

        # initialize empty tensors; will be set in init_from_pc
        self._xyz = torch.empty(0)  # positions
        self._rotation = torch.empty(0)  # rotation quaternions
        self._scaling = torch.empty(0)  # scaling factors
        self._opacity = torch.empty(0)  # opacity values
        self.features = torch.empty(0)  # wireless features

        # training state
        self.max_radii2D = torch.empty(0)  # screen-space radii for adaptive density
        self.xyz_gradient_accum = torch.empty(0)  # accumulated position gradients
        self.denom = torch.empty(0)  # gradient step denominator
        self.optimizer = None
        self.percent_dense = 0

        # optional predicted normals
        self._normals = torch.empty(0) if self.model_cfg.use_pred_normals else None

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

    def init_from_pc(self, points: torch.Tensor):
        """Initialize Gaussian properties from point cloud.

        Args:
            points: Point cloud tensor of shape [N, 3]
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

        # wireless features
        self.features = torch.zeros(
            (num_points, 2), device=device  # 0: attenuation, 1: phase_rotation
        ).float()

        # initialize opacity
        init_opacity = 0.1 * torch.ones((num_points, 1), device=device)
        self._opacity = nn.Parameter(inverse_sigmoid(init_opacity))

        # initialize training state tensors
        self.max_radii2D = torch.zeros((num_points,), device=device)
        self.xyz_gradient_accum = torch.zeros((num_points, 1), device=device)
        self.denom = torch.zeros((num_points, 1), device=device)

        # initialize optional normals
        if self.model_cfg.use_pred_normals:
            self._normals = nn.Parameter(torch.randn(num_points, 3, device=device))

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

    @property
    def get_normals(self):
        """Get predicted normals if enabled."""
        if not self.model_cfg.use_pred_normals:
            raise ValueError("Predicted normals not enabled in config")
        return self.rotation_activation(self._normals)

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

    def embed_features(self, wireless_data: dict[str, torch.Tensor]):
        """Embed iteration of wireless data into Gaussian features.

        The wireless data should contain the following keys:
        - tx_pos: Transmitter position (3,)
        - rx_pos: Receiver position (3,)
        - path_loss: Path loss values (N, 1) or (1, 1) for batched processing
        - aoa: Angles of arrival (2, P) with azimuth and elevation for P paths
        - path_loss_per_ray: Path loss per ray (P) for selecting important paths

        Args:
            wireless_data: Dictionary containing wireless data tensors with keys
        """
        # extract data
        tx_pos = wireless_data["tx_pos"]
        rx_pos = wireless_data["rx_pos"]
        path_loss = wireless_data["path_loss"]
        aoa = wireless_data["aoa"]
        path_loss_per_ray = wireless_data["path_loss_per_ray"]

        # compute features
        enc_output = self.encoder(
            self._xyz, tx_pos, rx_pos, path_loss, aoa, path_loss_per_ray
        )

        attenuation = enc_output["attenuation"]
        phase_rotation = enc_output["phase_rotation"]

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
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device=self._xyz.device
        )
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self._xyz.device)

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

        if self.model_cfg.use_pred_normals and self._normals is not None:
            param_groups.append(
                {
                    "params": [self._normals],
                    "lr": training_args.normals_lr,
                    "name": "normals",
                }
            )

        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=training_args.encoder_lr,
            weight_decay=training_args.weight_decay,
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

    def densification_postfix(
        self, new_xyz, new_scaling, new_rotation, new_opacity, new_features
    ):
        """Add new Gaussians to the model after densification.

        Args:
            new_xyz: New position parameters [N, 3]
            new_scaling: New scaling parameters [N, 3]
            new_rotation: New rotation parameters [N, 4]
            new_opacity: New opacity parameters [N, 1]
            new_features: New feature parameters [N, 2]
        """
        d = {
            "xyz": new_xyz,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "opacity": new_opacity,
        }

        optimizable_tensors = self._cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._opacity = optimizable_tensors["opacity"]

        self.features = torch.cat([self.features, new_features], dim=0)

        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device=self._xyz.device
        )
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self._xyz.device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self._xyz.device)

    def _cat_tensors_to_optimizer(self, tensors_dict):
        """Concatenate new tensors to existing ones in optimizer.

        Args:
            tensors_dict: Dictionary of tensor names to new tensors

        Returns:
            Dictionary of updated parameters
        """
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:
            if group["name"] not in tensors_dict:
                continue

            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)

            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )

            optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_points(self, mask):
        """Remove Gaussians based on a mask.

        Args:
            mask: Boolean mask where True indicates points to remove
        """
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._opacity = optimizable_tensors["opacity"]

        self.features = self.features[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.model_cfg.use_pred_normals and self._normals is not None:
            self._normals = nn.Parameter(
                self._normals[valid_points_mask].requires_grad_(True)
            )

    def _prune_optimizer(self, mask):
        """Update optimizer parameters with pruning mask.

        Args:
            mask: Boolean mask where True indicates points to keep

        Returns:
            Dictionary of updated parameters
        """
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)

            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )

            optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

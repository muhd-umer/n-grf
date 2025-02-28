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
    - `xyz`: Point positions from input point cloud
    - `rotation`: 4D quaternions initialized as [1,0,0,0] (identity)
    - `scaling`: Log of point-wise distances to enforce minimum scale
    - `opacity`: Inverse sigmoid of constant value (0.1)
    - `features_dc`: Main spherical harmonic features
    - `features_rest`: Higher order spherical harmonics initialized to zero
    - Colors from RGB are converted to SH coefficients
    - Screen-space max radii tracking for adaptive density
    - Gradients and denominator accumulators for training

    Args:
        config: Configuration for the model
    """

    def __init__(self, config: Optional[GaussianModelConfig] = None):
        super().__init__()

        self.config = config or GaussianModelConfig()

        # initialize empty tensors; will be set in init_from_pc
        self._xyz = torch.empty(0)  # positions
        self._rotation = torch.empty(0)  # rotation quaternions
        self._scaling = torch.empty(0)  # scaling factors
        self._opacity = torch.empty(0)  # opacity values
        self._features_dc = torch.empty(0)  # dc component of SH features
        self._features_rest = torch.empty(0)  # higher order SH features

        # training state
        self.max_radii2D = torch.empty(0)  # screen-space radii for adaptive density
        self.xyz_gradient_accum = torch.empty(0)  # accumulated position gradients
        self.denom = torch.empty(0)  # gradient step denominator
        self.optimizer = None
        self.percent_dense = 0

        # optional predicted normals
        self._normals = torch.empty(0) if self.config.use_pred_normals else None

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
        if self.config.use_pred_normals:
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
        if not self.config.use_pred_normals:
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

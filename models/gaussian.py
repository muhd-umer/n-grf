# models/gaussian.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sh_utils import RGB2SH, SH2RGB
from utils.train_utils import get_expon_lr_func
from utils.transform_utils import (
    build_rotation,
    build_scaling_rotation,
    inverse_sigmoid,
    strip_symmetric,
)

from .base import BaseModel
from .config import GaussianConfig

try:
    from simple_knn._C import distCUDA2
except ImportError:
    print("Warning: KNN module not found")


class GaussianModel(BaseModel):
    def __init__(self, config: GaussianConfig = None):
        super().__init__()

        self.config = config or GaussianConfig()

        # base parameters
        self.active_sh_degree = 0
        self.max_sh_degree = self.config.sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.spatial_lr_scale = 0.0

        self.setup_functions()

    def setup_functions(self):
        """Setup activation functions and transformations"""

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.covariance_activation = build_covariance_from_scaling_rotation

    def init_from_pcd(self, pcd: torch.Tensor):
        """Initialize Gaussians from point cloud

        Args:
            pcd (torch.Tensor): Point cloud tensor [N, 3]
        """
        # if point cloud is larger than num_pts, randomly sample points
        if len(pcd) > self.config.num_pts:
            pcd_indices = np.random.choice(len(pcd), self.config.num_pts, replace=False)
            pcd = pcd[pcd_indices]

        # convert to cuda tensors
        pcd = pcd.float().cuda()

        # initialize random features
        features = torch.rand((pcd.shape[0], self.config.color_dim)).float().cuda()

        # compute distances between points for scaling
        distances = torch.clamp_min(distCUDA2(pcd), 0.0000001)
        scales = (
            torch.log(torch.sqrt(distances))[..., None].repeat(1, 3)
            * self.config.scaling_init
        )

        # initialize rotations as identity quaternions
        rotations = torch.zeros((pcd.shape[0], 4), device="cuda")
        rotations[:, 0] = 1

        # initialize opacity
        opacities = inverse_sigmoid(
            self.config.opacity_init
            * torch.ones((pcd.shape[0], 1), dtype=torch.float, device="cuda")
        )

        # initialize sh features
        features = RGB2SH(features)  # convert to sh basis

        sh_features = (
            torch.zeros((pcd.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        )
        sh_features[:, :, 0] = features
        sh_features[:, :, 1:] = 0.0

        # create optimization parameters
        self._xyz = nn.Parameter(pcd.requires_grad_(True))
        self._features_dc = nn.Parameter(
            sh_features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            sh_features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rotations.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def get_covariance(self, scaling_modifier=1):
        """Get covariance matrices for Gaussians"""
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def get_output(self, rx_position):
        """Generate channel matrix for given receiver position"""
        raise NotImplementedError

    def training_setup(self, training_args):
        """Setup training parameters and optimizers"""
        raise NotImplementedError

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

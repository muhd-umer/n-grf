# models/config.py

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class GaussianConfig:
    """Configuration for Gaussian Splatting model"""

    sh_degree: int = 3
    num_pts: int = 100000

    # feature dimensions
    pos_dim: int = 3
    color_dim: int = 3

    # initialization settings
    opacity_init: float = 0.1
    scaling_init: float = 0.1

    # training settings
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.025
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001

# engine/__init__.py

import _torch_impl

from ._wrapper import CUDA_AVAILABLE, rasterize

__all__ = [
    "rasterize",
    "CUDA_AVAILABLE",
    "_torch_impl",
]

if CUDA_AVAILABLE:
    from ._C import (
        alpha_blending,
        compute_channel,
        compute_distances_to_receiver,
        compute_gaussian_influence,
        compute_jacobian,
        compute_spherical_coords,
        map_to_channel_matrix,
        project_cov3d_to_cov2d,
        project_to_channel_space,
        transform_to_uniform_coords,
    )

    __all__.extend(
        [
            "alpha_blending",
            "compute_channel",
            "compute_distances_to_receiver",
            "compute_gaussian_influence",
            "compute_jacobian",
            "compute_spherical_coords",
            "map_to_channel_matrix",
            "project_cov3d_to_cov2d",
            "project_to_channel_space",
            "transform_to_uniform_coords",
        ]
    )

# engine/__init__.py

import warnings

from . import _torch_impl


def _placeholder_function(*args, **kwargs):
    warnings.warn("CUDA implementation not available. Using PyTorch fallback.")
    return None


try:
    from ._wrapper import CUDA_AVAILABLE, rasterize
except ImportError:
    warnings.warn("CUDA wrapper not found. Using PyTorch implementation directly.")
    CUDA_AVAILABLE = False
    from ._torch_impl.rasterize import rasterize

__all__ = [
    "rasterize",
    "CUDA_AVAILABLE",
    "_torch_impl",
]


if CUDA_AVAILABLE:
    try:
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
            rasterize_backward,
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
                "rasterize_backward",  # testing
            ]
        )
    except (ImportError, AttributeError):
        warnings.warn(
            "CUDA functions not found despite CUDA being available. Check your build."
        )
        CUDA_AVAILABLE = False
else:
    alpha_blending = _placeholder_function
    compute_channel = _placeholder_function
    compute_distances_to_receiver = _placeholder_function
    compute_gaussian_influence = _placeholder_function
    compute_jacobian = _placeholder_function
    compute_spherical_coords = _placeholder_function
    map_to_channel_matrix = _placeholder_function
    project_cov3d_to_cov2d = _placeholder_function
    project_to_channel_space = _placeholder_function
    transform_to_uniform_coords = _placeholder_function

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

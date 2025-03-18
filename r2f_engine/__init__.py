# r2f_engine/__init__.py

import os
import warnings

try:
    from ._wrapper import rasterize

    CUSTOM_KERNEL = True
except ImportError:
    from _torch_impl.rasterize import rasterize

    warnings.warn(
        "Custom CUDA kernel not found. Using PyTorch implementation instead. Note that this may be significantly slower.",
    )

    CUSTOM_KERNEL = False

import _torch_impl

__all__ = [
    "rasterize",
    "CUSTOM_KERNEL",
    "_torch_impl",
    "_cuda_impl",
]

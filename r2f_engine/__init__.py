# core/__init__.py

import os
import warnings

try:
    from r2f_engine.r2f_engine import rasterize

    CUSTOM_KERNEL = True
except ImportError:
    from r2f_engine._torch_impl.rasterize import rasterize

    warnings.warn(
        "Custom CUDA kernel not found. Using PyTorch implementation instead. Note that this may be significantly slower.",
    )

    CUSTOM_KERNEL = False

import r2f_engine._cuda_impl
import r2f_engine._torch_impl

__all__ = [
    "rasterize",
    "CUSTOM_KERNEL",
    "_torch_impl",
    "_cuda_impl",
]

# engine/__init__.py

import warnings

from . import _torch_impl

try:
    from ._wrapper import CUDA_AVAILABLE, rasterize
except ImportError:
    warnings.warn("CUDA wrapper not found. Using PyTorch implementation directly.")
    CUDA_AVAILABLE = False
    from ._torch_impl.rasterize import rasterize

__all__ = [
    "CUDA_AVAILABLE",
    "_torch_impl",
    "rasterize",
]

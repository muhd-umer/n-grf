# core/__init__.py

try:
    from core._C import project_to_channel_space, rasterize

    CCUDA_AVAILABLE = True
except ImportError:
    from core._torch_impl.rasterize import rasterize
    from core._torch_impl.transforms import project_to_channel_space

    CCUDA_AVAILABLE = False

import core._torch_impl

__all__ = ["rasterize", "project_to_channel_space", "CCUDA_AVAILABLE", "_torch_impl"]

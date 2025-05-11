# render/__init__.py

import warnings

CUDA_AVAILABLE = False
render_channel = None

try:
    from ._wrapper import CUDA_AVAILABLE as _wrapper_cuda_available

    if _wrapper_cuda_available:
        from ._wrapper import render_channel as cuda_render_channel

        render_channel = cuda_render_channel
        CUDA_AVAILABLE = True
    else:
        warnings.warn(
            "nGRF WARNING: CUDA wrapper loaded, but CUDA is not available. Falling back to PyTorch."
        )
        from ._torch_impl import render_channel as torch_render_channel

        render_channel = torch_render_channel

except ImportError:
    warnings.warn(
        "nGRF WARNING: CUDA wrapper not found. Falling back to PyTorch rendering implementation."
    )
    from ._torch_impl import render_channel as torch_render_channel

    render_channel = torch_render_channel

if render_channel is None:
    raise ImportError(
        "Failed to load any rendering implementation (CUDA or PyTorch) for GRF."
    )

__all__ = ["render_channel", "CUDA_AVAILABLE"]

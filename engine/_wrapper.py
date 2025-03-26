# engine/_wrapper.py

import torch
from torch.autograd import Function

try:
    import _C

    CUDA_AVAILABLE = True
except ImportError:
    import warnings

    from _torch_impl.rasterize import rasterize as torch_rasterize

    warnings.warn(
        "CUDA implementation not found. Using PyTorch implementation instead. "
        "Make sure to build the CUDA extension with `pip install -e .` in the engine directory."
    )
    CUDA_AVAILABLE = False


class RasterizeFunction(Function):
    @staticmethod
    def forward(
        ctx,
        points,
        cov3d,
        attenuation,
        phase_rotation,
        opacity,
        receiver,
        transmitter,
        num_tx,
        num_rx,
        frequency,
    ):
        """Forward pass of rasterization"""

        ctx.save_for_backward(
            points,
            cov3d,
            attenuation,
            phase_rotation,
            opacity,
            receiver,
            transmitter,
        )
        ctx.num_tx = num_tx
        ctx.num_rx = num_rx
        ctx.frequency = frequency

        if CUDA_AVAILABLE:
            outputs = _C.rasterize_forward(
                points,
                cov3d,
                attenuation,
                phase_rotation,
                opacity,
                receiver,
                transmitter,
                num_tx,
                num_rx,
                frequency,
            )
            return outputs[0]  # Return channel matrix
        else:
            return torch_rasterize(
                points,
                cov3d,
                attenuation,
                phase_rotation,
                opacity,
                receiver,
                transmitter,
                num_tx,
                num_rx,
                frequency,
            )

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of rasterization"""
        (
            points,
            cov3d,
            attenuation,
            phase_rotation,
            opacity,
            receiver,
            transmitter,
        ) = ctx.saved_tensors

        if CUDA_AVAILABLE:
            (
                grad_points,
                grad_cov3d,
                grad_attenuation,
                grad_phase_rotation,
                grad_opacity,
            ) = _C.rasterize_backward(
                grad_output,
                points,
                cov3d,
                attenuation,
                phase_rotation,
                opacity,
                receiver,
                transmitter,
                ctx.num_tx,
                ctx.num_rx,
                ctx.frequency,
            )
        else:
            grad_points = torch.zeros_like(points)
            grad_cov3d = torch.zeros_like(cov3d)
            grad_attenuation = torch.zeros_like(attenuation)
            grad_phase_rotation = torch.zeros_like(phase_rotation)
            grad_opacity = torch.zeros_like(opacity)

            warnings.warn(
                "Backward pass not implemented in PyTorch fallback. "
                "Returning zero gradients. Build the CUDA extension for proper training."
            )

        return (
            grad_points,  # points
            grad_cov3d,  # cov3d
            grad_attenuation,  # attenuation
            grad_phase_rotation,  # phase_rotation
            grad_opacity,  # opacity
            None,  # receiver
            None,  # transmitter
            None,  # num_tx
            None,  # num_rx
            None,  # frequency
        )


def rasterize(
    points,
    cov3d=None,
    attenuation=None,
    phase_rotation=None,
    opacity=None,
    receiver=None,
    transmitter=None,
    num_tx=None,
    num_rx=None,
    frequency=None,
    scaling=None,
    rotation=None,
    scale_modifier=1.0,
):
    """
    Rasterize the channel matrix for a specific receiver position

    Args:
        points: Gaussian centers [N, 3]
        cov3d: 3D covariance matrices in compact form [N, 6]. If None, computed from scaling and rotation.
        attenuation: Learned attenuation amplitude from neural network [N, 1]
        phase_rotation: Learned phase rotation from neural network [N, 1]
        opacity: Opacity of each Gaussian [N, 1]
        receiver: Receiver position [3]
        transmitter: Transmitter position [3]
        num_tx: Number of transmit antennas
        num_rx: Number of receive antennas
        frequency: Signal frequency in Hz
        scaling: Scaling factors for each Gaussian [N, 3]. Required if cov3d is None.
        rotation: Rotation quaternions for each Gaussian [N, 4]. Required if cov3d is None.
        scale_modifier: Global scaling factor modifier. Used if cov3d is None.

    Returns:
        Channel matrix of shape [num_tx, 2*num_rx] with real and imaginary parts
        concatenated
    """
    if cov3d is None:
        assert (
            CUDA_AVAILABLE
        ), "CUDA implementation required for computing covariance in the forward pass"
        assert (
            scaling is not None and rotation is not None
        ), "Scaling and rotation must be provided when cov3d is None"

        cov3d = _C.compute_cov3d_from_scaling_rotation(
            scaling,
            rotation,
            scale_modifier,
        )

    return RasterizeFunction.apply(
        points,
        cov3d,
        attenuation,
        phase_rotation,
        opacity,
        receiver,
        transmitter,
        num_tx,
        num_rx,
        frequency,
    )

# core/_C/__init__.py

import torch
from torch.autograd import Function

from . import _C


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
        scale_factor=None,
    ):
        """Forward pass of rasterization"""

        # Save for backward
        ctx.save_for_backward(
            points,
            cov3d,
            attenuation,
            phase_rotation,
            opacity,
            receiver,
            transmitter,
            scale_factor,
        )
        ctx.num_tx = num_tx
        ctx.num_rx = num_rx
        ctx.frequency = frequency

        # Call CUDA implementation
        return _C.rasterize_forward(
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
            scale_factor,
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
            scale_factor,
        ) = ctx.saved_tensors

        # Call CUDA implementation
        grad_points, grad_cov3d, grad_attenuation, grad_phase_rotation, grad_opacity = (
            _C.rasterize_backward(
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
                scale_factor,
            )
        )

        # Return gradients for each input (None for parameters that don't require gradients)
        return (
            grad_points,
            grad_cov3d,
            grad_attenuation,
            grad_phase_rotation,
            grad_opacity,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def rasterize(
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
    scale_factor=None,
):
    """
    Rasterize the channel matrix for a specific receiver position

    Args:
        points: Gaussian centers [N, 3]
        cov3d: 3D covariance matrices in compact form [N, 6]
        attenuation: Learned attenuation amplitude from neural network [N, 1]
        phase_rotation: Learned phase rotation from neural network [N, 1]
        opacity: Opacity of each Gaussian [N, 1]
        receiver: Receiver position [3]
        transmitter: Transmitter position [3]
        num_tx: Number of transmit antennas
        num_rx: Number of receive antennas
        frequency: Signal frequency in Hz
        scale_factor: Scale factor for normalization. If None, no scaling is applied.

    Returns:
        Channel matrix of shape [num_tx, 2*num_rx] with real and imaginary parts concatenated
    """
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
        scale_factor,
    )

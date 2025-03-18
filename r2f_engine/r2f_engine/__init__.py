# core/_C/__init__.py

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

        # rasterize_forward returns tuple of (channel_matrix, aux_data1, aux_data2)
        # we only need the channel matrix for the forward pass
        result = _C.rasterize_forward(
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

        return result[0]

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
            )
        )

        # return gradients for each input
        return (
            grad_points,
            grad_cov3d,
            grad_attenuation,
            grad_phase_rotation,
            grad_opacity,
            None,  # receiver
            None,  # transmitter
            None,  # num_tx
            None,  # num_rx
            None,  # frequency
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

    Returns:
        Channel matrix of shape [num_tx, 2*num_rx] with real and imaginary parts
        concatenated
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
    )

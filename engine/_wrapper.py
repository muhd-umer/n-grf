# engine/_wrapper.py

from torch.autograd import Function

try:
    import _C

    CUDA_AVAILABLE = True
except ImportError:
    import warnings

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
        scaling,
        rotation,
        attenuation,
        phase_rotation,
        opacity,
        receiver,
        transmitter,
        num_tx,
        num_rx,
        frequency,
        scale_modifier,
    ):
        """Forward pass of rasterization using CUDA"""
        assert CUDA_AVAILABLE, "CUDA implementation required for rasterization"

        ctx.save_for_backward(
            points,
            scaling,
            rotation,
            attenuation,
            phase_rotation,
            opacity,
            receiver,
            transmitter,
        )
        ctx.num_tx = num_tx
        ctx.num_rx = num_rx
        ctx.frequency = frequency
        ctx.scale_modifier = scale_modifier

        outputs = _C.rasterize_forward(
            points,
            scaling,
            rotation,
            attenuation,
            phase_rotation,
            opacity,
            receiver,
            transmitter,
            num_tx,
            num_rx,
            frequency,
            scale_modifier,
        )
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of rasterization using CUDA"""
        assert CUDA_AVAILABLE, "CUDA implementation required for rasterization backward"

        (
            points,
            scaling,
            rotation,
            attenuation,
            phase_rotation,
            opacity,
            receiver,
            transmitter,
        ) = ctx.saved_tensors

        (
            grad_points,
            grad_attenuation,
            grad_phase_rotation,
            grad_opacity,
            grad_scaling,
            grad_rotation,
        ) = _C.rasterize_backward(
            grad_output,
            points,
            scaling,
            rotation,
            attenuation,
            phase_rotation,
            opacity,
            receiver,
            transmitter,
            ctx.num_tx,
            ctx.num_rx,
            ctx.frequency,
            ctx.scale_modifier,
        )

        return (
            grad_points,  # points
            grad_scaling,  # scaling
            grad_rotation,  # rotation
            grad_attenuation,  # attenuation
            grad_phase_rotation,  # phase_rotation
            grad_opacity,  # opacity
            None,  # receiver
            None,  # transmitter
            None,  # num_tx
            None,  # num_rx
            None,  # frequency
            None,  # scale_modifier
        )


def rasterize(
    points,
    scaling=None,
    rotation=None,
    attenuation=None,
    phase_rotation=None,
    opacity=None,
    receiver=None,
    transmitter=None,
    num_tx=None,
    num_rx=None,
    frequency=None,
    scale_modifier=1.0,
):
    """
    Rasterize the channel matrix for a specific receiver position

    Args:
        points: Gaussian centers [N, 3]
        scaling: Scaling factors for each Gaussian [N, 3]
        rotation: Rotation quaternions for each Gaussian [N, 4]
        attenuation: Learned attenuation amplitude from neural network [N, 1]
        phase_rotation: Learned phase rotation from neural network [N, 1]
        opacity: Opacity of each Gaussian [N, 1]
        receiver: Receiver position [3]
        transmitter: Transmitter position [3]
        num_tx: Number of transmit antennas
        num_rx: Number of receive antennas
        frequency: Signal frequency in Hz
        scale_modifier: Global scaling factor modifier

    Returns:
        Channel matrix of shape [num_tx, 2*num_rx] with real and imaginary parts
        concatenated
    """
    assert CUDA_AVAILABLE, "CUDA implementation required for rasterization"
    assert (
        scaling is not None and rotation is not None
    ), "Scaling and rotation must be provided"

    return RasterizeFunction.apply(
        points,
        scaling,
        rotation,
        attenuation,
        phase_rotation,
        opacity,
        receiver,
        transmitter,
        num_tx,
        num_rx,
        frequency,
        scale_modifier,
    )

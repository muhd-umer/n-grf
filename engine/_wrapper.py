import torch
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


class QuaternionToRotation(Function):
    @staticmethod
    def forward(ctx, quaternion):
        rotation = torch.empty(
            quaternion.shape[0], 3, 3, dtype=quaternion.dtype, device=quaternion.device
        )
        _C.quaternion_to_rotation_cuda(quaternion, rotation)
        ctx.save_for_backward(quaternion)
        return rotation

    @staticmethod
    def backward(ctx, grad_rotation):
        quaternion = ctx.saved_tensors[0]
        grad_quaternion = torch.empty_like(quaternion)
        _C.quaternion_to_rotation_backward_cuda(
            quaternion, grad_rotation, grad_quaternion
        )
        return grad_quaternion


class ComputeScalingMatrix(Function):
    @staticmethod
    def forward(ctx, scaling, scale_modifier=1.0):
        scaling_matrix = torch.empty(
            scaling.shape[0], 3, 3, dtype=scaling.dtype, device=scaling.device
        )
        _C.compute_scaling_matrix_cuda(scaling, scale_modifier, scaling_matrix)
        ctx.save_for_backward(scaling)
        ctx.scale_modifier = scale_modifier
        return scaling_matrix

    @staticmethod
    def backward(ctx, grad_scaling_matrix):
        scaling = ctx.saved_tensors[0]
        grad_scaling = torch.empty_like(scaling)
        _C.compute_scaling_matrix_backward_cuda(
            scaling, grad_scaling_matrix, ctx.scale_modifier, grad_scaling
        )
        return grad_scaling, None


class MatrixMultiply(Function):
    @staticmethod
    def forward(ctx, A, B):
        C = torch.empty(
            A.shape[0], A.shape[1], B.shape[2], dtype=A.dtype, device=A.device
        )
        _C.matrix_multiply_cuda(A, B, C)
        ctx.save_for_backward(A, B)
        return C

    @staticmethod
    def backward(ctx, grad_C):
        A, B = ctx.saved_tensors
        grad_A = torch.empty_like(A)
        grad_B = torch.empty_like(B)
        _C.matrix_multiply_backward_cuda(A, B, grad_C, grad_A, grad_B)
        return grad_A, grad_B


class CovarianceMatrix(Function):
    @staticmethod
    def forward(ctx, RS):
        cov3d = torch.empty_like(RS)
        _C.covariance_matrix_cuda(RS, cov3d)
        ctx.save_for_backward(RS)
        return cov3d

    @staticmethod
    def backward(ctx, grad_cov3d):
        RS = ctx.saved_tensors[0]
        grad_RS = torch.empty_like(RS)
        _C.covariance_matrix_backward_cuda(RS, grad_cov3d, grad_RS)
        return grad_RS


class ProjectToChannelCoords(Function):
    @staticmethod
    def forward(ctx, points, receiver, num_tx, num_rx):
        distances = torch.empty(
            points.shape[0], dtype=points.dtype, device=points.device
        )
        displacement = torch.empty_like(points)
        uv_coords = torch.empty(
            points.shape[0], 2, dtype=points.dtype, device=points.device
        )
        _C.project_to_channel_coords_cuda(
            points, receiver, num_tx, num_rx, distances, displacement, uv_coords
        )
        ctx.save_for_backward(points, receiver, distances, displacement)
        ctx.num_tx = num_tx
        ctx.num_rx = num_rx
        return distances, displacement, uv_coords

    @staticmethod
    def backward(ctx, grad_distances, grad_displacement, grad_uv):
        points, receiver, distances, displacement = ctx.saved_tensors
        grad_points = torch.empty_like(points)
        _C.project_to_channel_coords_backward_cuda(
            points,
            receiver,
            distances,
            displacement,
            grad_distances,
            grad_displacement,
            grad_uv,
            ctx.num_tx,
            ctx.num_rx,
            grad_points,
        )
        return grad_points, None, None, None


class ComputeJacobian(Function):
    @staticmethod
    def forward(ctx, d, num_tx, num_rx):
        J = torch.empty(d.shape[0], 2, 3, dtype=d.dtype, device=d.device)
        _C.compute_jacobian_cuda(d, num_tx, num_rx, J)
        ctx.save_for_backward(d)
        ctx.num_tx = num_tx
        ctx.num_rx = num_rx
        return J

    @staticmethod
    def backward(ctx, grad_J):
        d = ctx.saved_tensors[0]
        grad_d = torch.empty_like(d)
        _C.compute_jacobian_backward_cuda(d, grad_J, ctx.num_tx, ctx.num_rx, grad_d)
        return grad_d, None, None


class ProjectCov3dToCov2d(Function):
    @staticmethod
    def forward(ctx, cov3d, jacobian):
        cov2d = torch.empty(
            cov3d.shape[0], 2, 2, dtype=cov3d.dtype, device=cov3d.device
        )
        _C.project_cov3d_to_cov2d_cuda(cov3d, jacobian, cov2d)
        ctx.save_for_backward(cov3d, jacobian)
        return cov2d

    @staticmethod
    def backward(ctx, grad_cov2d):
        cov3d, jacobian = ctx.saved_tensors
        grad_cov3d = torch.empty_like(cov3d)
        grad_jacobian = torch.empty_like(jacobian)
        _C.project_cov3d_to_cov2d_backward_cuda(
            cov3d, jacobian, grad_cov2d, grad_cov3d, grad_jacobian
        )
        return grad_cov3d, grad_jacobian


class ComputeGaussianInfluence(Function):
    @staticmethod
    def forward(ctx, uv, cov2d, num_tx, num_rx):
        influences = torch.empty(
            uv.shape[0], num_tx, num_rx, dtype=uv.dtype, device=uv.device
        )
        _C.compute_gaussian_influence_cuda(uv, cov2d, num_tx, num_rx, influences)
        ctx.save_for_backward(uv, cov2d, influences)
        ctx.num_tx = num_tx
        ctx.num_rx = num_rx
        return influences

    @staticmethod
    def backward(ctx, grad_influences):
        uv, cov2d, influences = ctx.saved_tensors
        grad_uv = torch.empty_like(uv)
        grad_cov2d = torch.empty_like(cov2d)
        _C.compute_gaussian_influence_backward_cuda(
            uv,
            cov2d,
            influences,
            grad_influences,
            ctx.num_tx,
            ctx.num_rx,
            grad_uv,
            grad_cov2d,
        )
        return grad_uv, grad_cov2d, None, None


class ComputeWirelessChannel(Function):
    @staticmethod
    def forward(ctx, attenuation, phase_rotation, distances, wavelength):
        real_contributions = torch.empty_like(attenuation)
        imag_contributions = torch.empty_like(attenuation)
        _C.compute_wireless_channel_cuda(
            attenuation,
            phase_rotation,
            distances,
            wavelength,
            real_contributions,
            imag_contributions,
        )
        ctx.save_for_backward(attenuation, phase_rotation, distances)
        ctx.wavelength = wavelength
        return real_contributions, imag_contributions

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        attenuation, phase_rotation, distances = ctx.saved_tensors
        grad_attenuation = torch.empty_like(attenuation)
        grad_phase_rotation = torch.empty_like(phase_rotation)
        grad_distances = torch.empty_like(distances)
        _C.compute_wireless_channel_backward_cuda(
            attenuation,
            phase_rotation,
            distances,
            ctx.wavelength,
            grad_real,
            grad_imag,
            grad_attenuation,
            grad_phase_rotation,
            grad_distances,
        )
        return grad_attenuation, grad_phase_rotation, grad_distances, None


class AlphaBlending(Function):
    @staticmethod
    def forward(
        ctx,
        influences,
        real_contributions,
        imag_contributions,
        opacity,
        sort_indices,
        num_tx,
        num_rx,
    ):
        channel_matrix = torch.empty(
            num_tx, 2 * num_rx, dtype=influences.dtype, device=influences.device
        )
        _C.alpha_blending_cuda(
            influences,
            real_contributions,
            imag_contributions,
            opacity,
            sort_indices,
            num_tx,
            num_rx,
            channel_matrix,
        )
        ctx.save_for_backward(
            influences, real_contributions, imag_contributions, opacity, sort_indices
        )
        ctx.num_tx = num_tx
        ctx.num_rx = num_rx
        return channel_matrix

    @staticmethod
    def backward(ctx, grad_channel_matrix):
        influences, real_contributions, imag_contributions, opacity, sort_indices = (
            ctx.saved_tensors
        )
        grad_influences = torch.empty_like(influences)
        grad_real_contributions = torch.empty_like(real_contributions)
        grad_imag_contributions = torch.empty_like(imag_contributions)
        grad_opacity = torch.empty_like(opacity)
        _C.alpha_blending_backward_cuda(
            influences,
            real_contributions,
            imag_contributions,
            opacity,
            sort_indices,
            grad_channel_matrix,
            ctx.num_tx,
            ctx.num_rx,
            grad_influences,
            grad_real_contributions,
            grad_imag_contributions,
            grad_opacity,
        )
        return (
            grad_influences,
            grad_real_contributions,
            grad_imag_contributions,
            grad_opacity,
            None,
            None,
            None,
        )


def rasterize(
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
    scale_modifier=1.0,
):
    """Rasterize the channel matrix for a specific receiver position

    Args:
        points: Gaussian centers [N, 3]
        scaling: Scaling factors (already with exponential activation applied) [N, 3]
        rotation: Quaternion rotations (already normalized) [N, 4]
        attenuation: Learned attenuation amplitude from neural network [N, 1]
        phase_rotation: Learned phase rotation from neural network [N, 1]
        opacity: Opacity values (already with sigmoid activation applied) [N, 1]
        receiver: Receiver position [3]
        transmitter: Transmitter position [3] (not used in current implementation)
        num_tx: Number of transmit antennas
        num_rx: Number of receive antennas
        frequency: Signal frequency in Hz
        scale_modifier: Global scaling modifier (default is 1.0)

    Returns:
        Channel matrix of shape [num_tx, 2*num_rx] with real and imaginary parts concatenated
    """
    c = 299792458.0
    wavelength = c / frequency

    R = QuaternionToRotation.apply(rotation)
    S = ComputeScalingMatrix.apply(scaling, scale_modifier)
    RS = MatrixMultiply.apply(R, S)
    cov3d = CovarianceMatrix.apply(RS)
    distances, d, uv = ProjectToChannelCoords.apply(points, receiver, num_tx, num_rx)
    jacobian = ComputeJacobian.apply(d, num_tx, num_rx)
    cov2d = ProjectCov3dToCov2d.apply(cov3d, jacobian)
    sort_indices = torch.argsort(distances)
    influences = ComputeGaussianInfluence.apply(uv, cov2d, num_tx, num_rx)

    real_contributions, imag_contributions = ComputeWirelessChannel.apply(
        attenuation, phase_rotation, distances, wavelength
    )

    cat_channel = AlphaBlending.apply(
        influences,
        real_contributions,
        imag_contributions,
        opacity,
        sort_indices,
        num_tx,
        num_rx,
    )

    return cat_channel

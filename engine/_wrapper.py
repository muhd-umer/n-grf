# engine/_wrapper.py

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
        q_norm = torch.nn.functional.normalize(quaternion, dim=1)
        rotation = torch.empty(
            quaternion.shape[0], 3, 3, dtype=quaternion.dtype, device=quaternion.device
        )
        _C.quaternion_to_rotation_cuda(q_norm, rotation)
        ctx.save_for_backward(quaternion)
        return rotation

    @staticmethod
    def backward(ctx, grad_rotation):
        quaternion = ctx.saved_tensors[0]
        grad_quaternion = torch.empty_like(quaternion)
        _C.quaternion_to_rotation_backward_cuda(
            quaternion, grad_rotation.contiguous(), grad_quaternion
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
            scaling, grad_scaling_matrix.contiguous(), ctx.scale_modifier, grad_scaling
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
        _C.matrix_multiply_backward_cuda(A, B, grad_C.contiguous(), grad_A, grad_B)
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
        _C.covariance_matrix_backward_cuda(RS, grad_cov3d.contiguous(), grad_RS)
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
        grad_points = torch.zeros_like(points)
        _C.project_to_channel_coords_backward_cuda(
            points,
            receiver,
            distances,
            displacement,
            grad_distances.contiguous(),
            grad_displacement.contiguous(),
            grad_uv.contiguous(),
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
        _C.compute_jacobian_backward_cuda(
            d, grad_J.contiguous(), ctx.num_tx, ctx.num_rx, grad_d
        )
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
        grad_cov3d = torch.zeros_like(cov3d)
        grad_jacobian = torch.zeros_like(jacobian)
        _C.project_cov3d_to_cov2d_backward_cuda(
            cov3d, jacobian, grad_cov2d.contiguous(), grad_cov3d, grad_jacobian
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
            grad_influences.contiguous(),
            ctx.num_tx,
            ctx.num_rx,
            grad_uv,
            grad_cov2d,
        )
        return grad_uv, grad_cov2d, None, None


class ComputeWirelessChannel(Function):
    @staticmethod
    def forward(ctx, attenuation, phase_rotation, distances, wavelength):
        attenuation_cont = attenuation.contiguous()
        phase_rotation_cont = phase_rotation.contiguous()
        distances_cont = distances.contiguous()

        real_contributions = torch.empty_like(attenuation_cont)
        imag_contributions = torch.empty_like(attenuation_cont)
        _C.compute_wireless_channel_cuda(
            attenuation_cont,
            phase_rotation_cont,
            distances_cont,
            wavelength,
            real_contributions,
            imag_contributions,
        )
        ctx.save_for_backward(attenuation_cont, phase_rotation_cont, distances_cont)
        ctx.wavelength = wavelength

        return real_contributions, imag_contributions

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        attenuation, phase_rotation, distances = ctx.saved_tensors
        grad_attenuation = torch.zeros_like(attenuation)
        grad_phase_rotation = torch.zeros_like(phase_rotation)
        grad_distances = torch.zeros_like(distances)

        grad_real_cont = grad_real.contiguous()
        grad_imag_cont = grad_imag.contiguous()

        _C.compute_wireless_channel_backward_cuda(
            attenuation,
            phase_rotation,
            distances,
            ctx.wavelength,
            grad_real_cont,
            grad_imag_cont,
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
        contributions_real,
        contributions_imag,
        opacity,
        sort_indices,
        num_tx,
        num_rx,
    ):
        N = influences.shape[0]
        device = influences.device
        dtype = influences.dtype

        influences_cont = influences.contiguous()
        contrib_real_cont = contributions_real.contiguous().view(N)
        contrib_imag_cont = contributions_imag.contiguous().view(N)
        opacity_cont = opacity.contiguous().view(N)
        sort_indices_cont = sort_indices.contiguous().to(torch.int32)

        channel_matrix = torch.empty((num_tx, 2 * num_rx), device=device, dtype=dtype)
        eff_opacity = torch.empty((N, num_tx, num_rx), device=device, dtype=dtype)
        transmittance = torch.empty((N + 1, num_tx, num_rx), device=device, dtype=dtype)

        _C.alpha_blending_forward_cuda(
            influences_cont,
            contrib_real_cont,
            contrib_imag_cont,
            opacity_cont,
            sort_indices_cont,
            num_tx,
            num_rx,
            channel_matrix,
            eff_opacity,
            transmittance,
        )

        ctx.save_for_backward(
            influences_cont,
            contrib_real_cont,
            contrib_imag_cont,
            opacity_cont,
            eff_opacity,
            transmittance,
            sort_indices_cont,
        )
        ctx.num_tx = num_tx
        ctx.num_rx = num_rx
        ctx.N = N
        ctx.contrib_original_shape = contributions_real.shape
        ctx.opacity_original_shape = opacity.shape

        return channel_matrix

    @staticmethod
    def backward(ctx, grad_cat_channel):
        (
            influences,
            contributions_real,
            contributions_imag,
            opacity,
            eff_opacity,
            transmittance,
            sort_indices,
        ) = ctx.saved_tensors
        num_tx = ctx.num_tx
        num_rx = ctx.num_rx
        N = ctx.N

        grad_cat_channel_cont = grad_cat_channel.contiguous()

        grad_influences = torch.zeros_like(influences)
        grad_contrib_real = torch.zeros_like(contributions_real)
        grad_contrib_imag = torch.zeros_like(contributions_imag)
        grad_opacity = torch.zeros_like(opacity)

        _C.alpha_blending_backward_cuda(
            influences,
            contributions_real,
            contributions_imag,
            opacity,
            eff_opacity,
            transmittance,
            sort_indices,
            grad_cat_channel_cont,
            num_tx,
            num_rx,
            grad_influences,
            grad_contrib_real,
            grad_contrib_imag,
            grad_opacity,
        )

        grad_contrib_real = grad_contrib_real.view(ctx.contrib_original_shape)
        grad_contrib_imag = grad_contrib_imag.view(ctx.contrib_original_shape)
        grad_opacity = grad_opacity.view(ctx.opacity_original_shape)

        return (
            grad_influences,
            grad_contrib_real,
            grad_contrib_imag,
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
    if not CUDA_AVAILABLE:
        raise ImportError(
            "CUDA extension _C is not available. Cannot use CUDA rasterizer."
        )

    c = 299792458.0
    wavelength = c / frequency

    R = QuaternionToRotation.apply(rotation)
    S = ComputeScalingMatrix.apply(scaling, scale_modifier)
    RS = MatrixMultiply.apply(R, S)
    cov3d = CovarianceMatrix.apply(RS)

    distances, d, uv = ProjectToChannelCoords.apply(points, receiver, num_tx, num_rx)
    jacobian = ComputeJacobian.apply(d, num_tx, num_rx)
    cov2d = ProjectCov3dToCov2d.apply(cov3d, jacobian)

    influences = ComputeGaussianInfluence.apply(uv, cov2d, num_tx, num_rx)

    attenuation_cont = attenuation.contiguous()
    phase_rotation_cont = phase_rotation.contiguous()
    distances_cont = distances.contiguous()

    real_contributions, imag_contributions = ComputeWirelessChannel.apply(
        attenuation_cont, phase_rotation_cont, distances_cont, wavelength
    )

    sort_indices = torch.argsort(distances).to(dtype=torch.int32)

    cat_channel = AlphaBlending.apply(
        influences.contiguous(),
        real_contributions,
        imag_contributions,
        opacity.contiguous(),
        sort_indices,
        num_tx,
        num_rx,
    )

    return cat_channel

# engine/_wrapper.py

import warnings
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.autograd import Function

from models.dir_network import DirectionalNetwork

try:
    import _C

    CUDA_AVAILABLE = True
except ImportError:
    warnings.warn(
        "CUDA implementation not found. Using PyTorch implementation instead. "
        "Make sure to build the CUDA extension with `pip install -e .` in the engine directory."
    )
    CUDA_AVAILABLE = False


from _torch_impl.rasterize import compute_direct_path as torch_compute_direct_path


class QuaternionToRotation(Function):
    @staticmethod
    def forward(ctx, quaternion):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for QuaternionToRotation."
            )
        q_norm = torch.nn.functional.normalize(quaternion, dim=1)
        rotation = torch.empty(
            quaternion.shape[0], 3, 3, dtype=quaternion.dtype, device=quaternion.device
        )
        _C.quaternion_to_rotation_cuda(q_norm, rotation)
        ctx.save_for_backward(quaternion)
        return rotation

    @staticmethod
    def backward(ctx, grad_rotation):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for QuaternionToRotation backward."
            )
        quaternion = ctx.saved_tensors[0]
        grad_quaternion = torch.empty_like(quaternion)
        _C.quaternion_to_rotation_backward_cuda(
            quaternion.contiguous(), grad_rotation.contiguous(), grad_quaternion
        )
        return grad_quaternion


class ComputeScalingMatrix(Function):
    @staticmethod
    def forward(ctx, scaling, scale_modifier=1.0):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for ComputeScalingMatrix."
            )
        scaling_matrix = torch.empty(
            scaling.shape[0], 3, 3, dtype=scaling.dtype, device=scaling.device
        )
        _C.compute_scaling_matrix_cuda(
            scaling.contiguous(), scale_modifier, scaling_matrix
        )
        ctx.save_for_backward(scaling)
        ctx.scale_modifier = scale_modifier
        return scaling_matrix

    @staticmethod
    def backward(ctx, grad_scaling_matrix):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for ComputeScalingMatrix backward."
            )
        scaling = ctx.saved_tensors[0]
        grad_scaling = torch.empty_like(scaling)
        _C.compute_scaling_matrix_backward_cuda(
            scaling.contiguous(),
            grad_scaling_matrix.contiguous(),
            ctx.scale_modifier,
            grad_scaling,
        )
        return grad_scaling, None


class MatrixMultiply(Function):
    @staticmethod
    def forward(ctx, A, B):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA implementation not available for MatrixMultiply.")
        C = torch.empty(
            A.shape[0], A.shape[1], B.shape[2], dtype=A.dtype, device=A.device
        )
        _C.matrix_multiply_cuda(A.contiguous(), B.contiguous(), C)
        ctx.save_for_backward(A, B)
        return C

    @staticmethod
    def backward(ctx, grad_C):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for MatrixMultiply backward."
            )
        A, B = ctx.saved_tensors
        grad_A = torch.empty_like(A)
        grad_B = torch.empty_like(B)
        _C.matrix_multiply_backward_cuda(
            A.contiguous(), B.contiguous(), grad_C.contiguous(), grad_A, grad_B
        )
        return grad_A, grad_B


class CovarianceMatrix(Function):
    @staticmethod
    def forward(ctx, RS):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for CovarianceMatrix."
            )
        cov3d = torch.empty_like(RS)
        _C.covariance_matrix_cuda(RS.contiguous(), cov3d)
        ctx.save_for_backward(RS)
        return cov3d

    @staticmethod
    def backward(ctx, grad_cov3d):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for CovarianceMatrix backward."
            )
        RS = ctx.saved_tensors[0]
        grad_RS = torch.empty_like(RS)
        _C.covariance_matrix_backward_cuda(
            RS.contiguous(), grad_cov3d.contiguous(), grad_RS
        )
        return grad_RS


class ProjectToChannelCoordinates(Function):
    @staticmethod
    def forward(ctx, points, receiver, num_tx, num_rx):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for ProjectToChannelCoordinates."
            )
        distances = torch.empty(
            points.shape[0], dtype=points.dtype, device=points.device
        )
        displacement = torch.empty_like(points)
        uv_coords = torch.empty(
            points.shape[0], 2, dtype=points.dtype, device=points.device
        )
        _C.project_to_channel_coords_cuda(
            points.contiguous(),
            receiver.contiguous(),
            num_tx,
            num_rx,
            distances,
            displacement,
            uv_coords,
        )
        ctx.save_for_backward(points, receiver, distances, displacement)
        ctx.num_tx = num_tx
        ctx.num_rx = num_rx
        return distances, displacement, uv_coords

    @staticmethod
    def backward(ctx, grad_r, grad_d, grad_uv):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for ProjectToChannelCoordinates backward."
            )
        points, receiver, distances, displacement = ctx.saved_tensors
        grad_points = torch.zeros_like(points)
        _C.project_to_channel_coords_backward_cuda(
            points.contiguous(),
            receiver.contiguous(),
            distances.contiguous(),
            displacement.contiguous(),
            grad_r.contiguous(),
            grad_d.contiguous(),
            grad_uv.contiguous(),
            ctx.num_tx,
            ctx.num_rx,
            grad_points,
        )
        return grad_points, None, None, None


class ComputeJacobian(Function):
    @staticmethod
    def forward(ctx, d, num_tx, num_rx):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA implementation not available for ComputeJacobian.")
        J = torch.empty(d.shape[0], 2, 3, dtype=d.dtype, device=d.device)
        _C.compute_jacobian_cuda(d.contiguous(), num_tx, num_rx, J)
        ctx.save_for_backward(d)
        ctx.num_tx = num_tx
        ctx.num_rx = num_rx
        return J

    @staticmethod
    def backward(ctx, grad_J):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for ComputeJacobian backward."
            )
        d = ctx.saved_tensors[0]
        grad_d = torch.empty_like(d)
        _C.compute_jacobian_backward_cuda(
            d.contiguous(), grad_J.contiguous(), ctx.num_tx, ctx.num_rx, grad_d
        )
        return grad_d, None, None


class ProjectCov3dToCov2d(Function):
    @staticmethod
    def forward(ctx, cov3d, jacobian):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for ProjectCov3dToCov2d."
            )
        cov2d = torch.empty(
            cov3d.shape[0], 2, 2, dtype=cov3d.dtype, device=cov3d.device
        )
        _C.project_cov3d_to_cov2d_cuda(cov3d.contiguous(), jacobian.contiguous(), cov2d)
        ctx.save_for_backward(cov3d, jacobian)
        return cov2d

    @staticmethod
    def backward(ctx, grad_cov2d):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for ProjectCov3dToCov2d backward."
            )
        cov3d, jacobian = ctx.saved_tensors
        grad_cov3d = torch.zeros_like(cov3d)
        grad_jacobian = torch.zeros_like(jacobian)
        _C.project_cov3d_to_cov2d_backward_cuda(
            cov3d.contiguous(),
            jacobian.contiguous(),
            grad_cov2d.contiguous(),
            grad_cov3d,
            grad_jacobian,
        )
        return grad_cov3d, grad_jacobian


class ComputeSpatialInfluence(Function):
    @staticmethod
    def forward(ctx, uv, cov2d, num_tx, num_rx):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for ComputeSpatialInfluence."
            )
        influences = torch.empty(
            uv.shape[0], num_tx, num_rx, dtype=uv.dtype, device=uv.device
        )
        _C.compute_spatial_influence_cuda(
            uv.contiguous(), cov2d.contiguous(), num_tx, num_rx, influences
        )
        ctx.save_for_backward(uv, cov2d, influences)
        ctx.num_tx = num_tx
        ctx.num_rx = num_rx
        return influences

    @staticmethod
    def backward(ctx, grad_influences):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for ComputeSpatialInfluence backward."
            )
        uv, cov2d, influences = ctx.saved_tensors
        grad_uv = torch.empty_like(uv)
        grad_cov2d = torch.empty_like(cov2d)
        _C.compute_spatial_influence_backward_cuda(
            uv.contiguous(),
            cov2d.contiguous(),
            influences.contiguous(),
            grad_influences.contiguous(),
            ctx.num_tx,
            ctx.num_rx,
            grad_uv,
            grad_cov2d,
        )
        return grad_uv, grad_cov2d, None, None


class ComputePathGeometry(Function):
    @staticmethod
    def forward(ctx, points_xyz, tx_pos, rx_pos):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for ComputePathGeometry."
            )
        N = points_xyz.shape[0]
        dist_tx = torch.empty(N, dtype=points_xyz.dtype, device=points_xyz.device)
        dist_rx = torch.empty(N, dtype=points_xyz.dtype, device=points_xyz.device)
        aod = torch.empty(N, 2, dtype=points_xyz.dtype, device=points_xyz.device)
        aoa = torch.empty(N, 2, dtype=points_xyz.dtype, device=points_xyz.device)
        _C.compute_path_geometry_cuda(
            points_xyz.contiguous(),
            tx_pos.contiguous(),
            rx_pos.contiguous(),
            dist_tx,
            dist_rx,
            aod,
            aoa,
        )
        ctx.save_for_backward(points_xyz, tx_pos, rx_pos, dist_tx, dist_rx, aod, aoa)
        return dist_tx, dist_rx, aod, aoa

    @staticmethod
    def backward(ctx, grad_dist_tx, grad_dist_rx, grad_aod, grad_aoa):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for ComputePathGeometry backward."
            )
        points_xyz, tx_pos, rx_pos, dist_tx, dist_rx, aod, aoa = ctx.saved_tensors
        grad_points = torch.zeros_like(points_xyz)

        grad_tx_pos = torch.zeros_like(tx_pos)
        grad_rx_pos = torch.zeros_like(rx_pos)
        _C.compute_path_geometry_backward_cuda(
            points_xyz.contiguous(),
            tx_pos.contiguous(),
            rx_pos.contiguous(),
            dist_tx.contiguous(),
            dist_rx.contiguous(),
            aod.contiguous(),
            aoa.contiguous(),
            grad_dist_tx.contiguous(),
            grad_dist_rx.contiguous(),
            grad_aod.contiguous(),
            grad_aoa.contiguous(),
            grad_points,
            grad_tx_pos,
            grad_rx_pos,
        )
        return grad_points, None, None


class ComputeSteeringVector(Function):
    @staticmethod
    def forward(ctx, angles_rad, array_params: Dict[str, Any], wavelength: float):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for ComputeSteeringVector."
            )
        N = angles_rad.shape[0]
        array_type_str = array_params["type"]
        if array_type_str == "ura":
            array_type_int = 0
            rows, cols = array_params["size"]
            num_ant = rows * cols
            size_tensor = torch.tensor(
                [rows, cols], dtype=angles_rad.dtype, device=angles_rad.device
            )
            spacing_tensor = torch.tensor(
                array_params["element_spacing"],
                dtype=angles_rad.dtype,
                device=angles_rad.device,
            )
        elif array_type_str == "ula":
            array_type_int = 1
            num_ant = array_params["size"]
            size_tensor = torch.tensor(
                [num_ant, 0], dtype=angles_rad.dtype, device=angles_rad.device
            )
            spacing_tensor = torch.tensor(
                [array_params["element_spacing"], 0],
                dtype=angles_rad.dtype,
                device=angles_rad.device,
            )
        else:
            raise ValueError(f"Unsupported array type: {array_type_str}")

        sv_real = torch.empty(
            N, num_ant, dtype=angles_rad.dtype, device=angles_rad.device
        )
        sv_imag = torch.empty(
            N, num_ant, dtype=angles_rad.dtype, device=angles_rad.device
        )

        _C.compute_steering_vector_cuda(
            angles_rad.contiguous(),
            size_tensor,
            spacing_tensor,
            array_type_int,
            wavelength,
            sv_real,
            sv_imag,
        )
        ctx.save_for_backward(angles_rad, sv_real, sv_imag)
        ctx.array_params = array_params
        ctx.wavelength = wavelength
        ctx.array_type_int = array_type_int
        ctx.size_tensor = size_tensor
        ctx.spacing_tensor = spacing_tensor
        return sv_real, sv_imag

    @staticmethod
    def backward(ctx, grad_sv_real, grad_sv_imag):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for ComputeSteeringVector backward."
            )
        angles_rad, sv_real, sv_imag = ctx.saved_tensors
        grad_angles = torch.zeros_like(angles_rad)
        _C.compute_steering_vector_backward_cuda(
            angles_rad.contiguous(),
            ctx.size_tensor,
            ctx.spacing_tensor,
            ctx.array_type_int,
            ctx.wavelength,
            sv_real.contiguous(),
            sv_imag.contiguous(),
            grad_sv_real.contiguous(),
            grad_sv_imag.contiguous(),
            grad_angles,
        )
        return grad_angles, None, None


class ComputeScatteredPaths(Function):
    @staticmethod
    def forward(
        ctx,
        gamma_real,
        gamma_imag,
        dist_tx,
        dist_rx,
        sv_tx_real,
        sv_tx_imag,
        sv_rx_real,
        sv_rx_imag,
        wavelength,
    ):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for ComputeScatteredPaths."
            )
        N = gamma_real.shape[0]
        Nt = sv_tx_real.shape[1]
        Nr = sv_rx_real.shape[1]
        scat_chan_real = torch.empty(
            N, Nt, Nr, dtype=gamma_real.dtype, device=gamma_real.device
        )
        scat_chan_imag = torch.empty(
            N, Nt, Nr, dtype=gamma_real.dtype, device=gamma_real.device
        )

        _C.compute_scattered_paths_cuda(
            gamma_real.contiguous(),
            gamma_imag.contiguous(),
            dist_tx.contiguous(),
            dist_rx.contiguous(),
            sv_tx_real.contiguous(),
            sv_tx_imag.contiguous(),
            sv_rx_real.contiguous(),
            sv_rx_imag.contiguous(),
            wavelength,
            scat_chan_real,
            scat_chan_imag,
        )
        ctx.save_for_backward(
            gamma_real,
            gamma_imag,
            dist_tx,
            dist_rx,
            sv_tx_real,
            sv_tx_imag,
            sv_rx_real,
            sv_rx_imag,
        )
        ctx.wavelength = wavelength
        return scat_chan_real, scat_chan_imag

    @staticmethod
    def backward(ctx, grad_scat_chan_real, grad_scat_chan_imag):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for ComputeScatteredPaths backward."
            )
        (
            gamma_real,
            gamma_imag,
            dist_tx,
            dist_rx,
            sv_tx_real,
            sv_tx_imag,
            sv_rx_real,
            sv_rx_imag,
        ) = ctx.saved_tensors
        wavelength = ctx.wavelength

        grad_gamma_real = torch.zeros_like(gamma_real)
        grad_gamma_imag = torch.zeros_like(gamma_imag)
        grad_dist_tx = torch.zeros_like(dist_tx)
        grad_dist_rx = torch.zeros_like(dist_rx)
        grad_sv_tx_real = torch.zeros_like(sv_tx_real)
        grad_sv_tx_imag = torch.zeros_like(sv_tx_imag)
        grad_sv_rx_real = torch.zeros_like(sv_rx_real)
        grad_sv_rx_imag = torch.zeros_like(sv_rx_imag)

        _C.compute_scattered_paths_backward_cuda(
            gamma_real.contiguous(),
            gamma_imag.contiguous(),
            dist_tx.contiguous(),
            dist_rx.contiguous(),
            sv_tx_real.contiguous(),
            sv_tx_imag.contiguous(),
            sv_rx_real.contiguous(),
            sv_rx_imag.contiguous(),
            wavelength,
            grad_scat_chan_real.contiguous(),
            grad_scat_chan_imag.contiguous(),
            grad_gamma_real,
            grad_gamma_imag,
            grad_dist_tx,
            grad_dist_rx,
            grad_sv_tx_real,
            grad_sv_tx_imag,
            grad_sv_rx_real,
            grad_sv_rx_imag,
        )
        return (
            grad_gamma_real,
            grad_gamma_imag,
            grad_dist_tx,
            grad_dist_rx,
            grad_sv_tx_real,
            grad_sv_tx_imag,
            grad_sv_rx_real,
            grad_sv_rx_imag,
            None,
        )


class WeightedSuperposition(Function):
    @staticmethod
    def forward(
        ctx,
        direct_path_real,
        direct_path_imag,
        scat_path_real,
        scat_path_imag,
        opacity,
        influence,
    ):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for WeightedSuperposition."
            )
        Nt = direct_path_real.shape[0]
        Nr = direct_path_real.shape[1]
        chan_real = torch.empty(
            Nt, Nr, dtype=direct_path_real.dtype, device=direct_path_real.device
        )
        chan_imag = torch.empty(
            Nt, Nr, dtype=direct_path_real.dtype, device=direct_path_real.device
        )

        _C.weighted_superposition_cuda(
            direct_path_real.contiguous(),
            direct_path_imag.contiguous(),
            scat_path_real.contiguous(),
            scat_path_imag.contiguous(),
            opacity.contiguous(),
            influence.contiguous(),
            chan_real,
            chan_imag,
        )
        ctx.save_for_backward(scat_path_real, scat_path_imag, opacity, influence)
        return chan_real, chan_imag

    @staticmethod
    def backward(ctx, grad_chan_pred_real, grad_chan_pred_imag):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA implementation not available for WeightedSuperposition backward."
            )
        scat_path_real, scat_path_imag, opacity, influence = ctx.saved_tensors
        grad_scat_path_real = torch.zeros_like(scat_path_real)
        grad_scat_path_imag = torch.zeros_like(scat_path_imag)
        grad_opacity = torch.zeros_like(opacity)
        grad_influence = torch.zeros_like(influence)

        _C.weighted_superposition_backward_cuda(
            scat_path_real.contiguous(),
            scat_path_imag.contiguous(),
            opacity.contiguous(),
            influence.contiguous(),
            grad_chan_pred_real.contiguous(),
            grad_chan_pred_imag.contiguous(),
            grad_scat_path_real,
            grad_scat_path_imag,
            grad_opacity,
            grad_influence,
        )

        if opacity.dim() > 1 and opacity.size(1) == 1:
            grad_opacity = grad_opacity.view(opacity.shape)

        return (
            None,
            None,
            grad_scat_path_real,
            grad_scat_path_imag,
            grad_opacity,
            grad_influence,
        )


def compute_direct_path(
    tx_params: Dict[str, Any], rx_params: Dict[str, Any], wavelength: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch_compute_direct_path(tx_params, rx_params, wavelength)


def rasterize(
    points: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    base_features: torch.Tensor,
    opacity: torch.Tensor,
    tx_params: Dict[str, Any],
    rx_params: Dict[str, Any],
    directional_network: DirectionalNetwork,
    dir_embedder: nn.Module,
    scale_modifier: float = 1.0,
) -> torch.Tensor:
    """Rasterize the channel matrix.

    Args:
        points: Gaussian centers [N, 3]
        scaling: Activated scaling factors [N, 3]
        rotation: Quaternion rotations [N, 4] (unnormalized is fine)
        base_features: Base feature vectors [N, F]
        opacity: Activated opacity values [N, 1]
        tx_params: Transmitter parameters
        rx_params: Receiver parameters
        directional_network: Network predicting gamma based on direction
        dir_embedder: Positional embedder for directions
        scale_modifier: Global scaling modifier

    Returns:
        Predicted channel matrix [Nt, 2*Nr] (real/imag stacked)
    """
    tx_pos = tx_params["position"]
    rx_pos = rx_params["position"]
    num_tx = tx_params["num_antennas"]
    num_rx = rx_params["num_antennas"]
    frequency = tx_params["frequency"]
    wavelength = 299792458.0 / frequency

    R = QuaternionToRotation.apply(rotation)
    S = ComputeScalingMatrix.apply(scaling, scale_modifier)
    RS = MatrixMultiply.apply(R, S)
    cov3d = CovarianceMatrix.apply(RS)
    dist_tx, dist_rx, aod, aoa = ComputePathGeometry.apply(points, tx_pos, rx_pos)

    incoming_direction = points - tx_pos
    incoming_direction = torch.nn.functional.normalize(
        incoming_direction, p=2, dim=1, eps=1e-6
    )
    dir_input_in = dir_embedder(incoming_direction)

    outgoing_direction = rx_pos - points
    outgoing_direction = torch.nn.functional.normalize(
        outgoing_direction, p=2, dim=1, eps=1e-6
    )
    dir_input_out = dir_embedder(outgoing_direction)

    gamma_real, gamma_imag = directional_network(
        base_features, dir_input_in, dir_input_out
    )
    gamma_real = gamma_real.squeeze(-1)
    gamma_imag = gamma_imag.squeeze(-1)

    sv_tx_real, sv_tx_imag = ComputeSteeringVector.apply(aod, tx_params, wavelength)
    sv_rx_real, sv_rx_imag = ComputeSteeringVector.apply(aoa, rx_params, wavelength)

    scat_path_real, scat_path_imag = ComputeScatteredPaths.apply(
        gamma_real,
        gamma_imag,
        dist_tx,
        dist_rx,
        sv_tx_real,
        sv_tx_imag,
        sv_rx_real,
        sv_rx_imag,
        wavelength,
    )

    _, d_proj, uv = ProjectToChannelCoordinates.apply(points, rx_pos, num_tx, num_rx)
    jacobian = ComputeJacobian.apply(d_proj, num_tx, num_rx)
    cov2d = ProjectCov3dToCov2d.apply(cov3d, jacobian)
    influence = ComputeSpatialInfluence.apply(uv, cov2d, num_tx, num_rx)

    direct_path_real, direct_path_imag = compute_direct_path(
        tx_params, rx_params, wavelength
    )

    chan_pred_real, chan_pred_imag = WeightedSuperposition.apply(
        direct_path_real,
        direct_path_imag,
        scat_path_real,
        scat_path_imag,
        opacity,
        influence,
    )
    chan_pred = torch.cat([chan_pred_real, chan_pred_imag], dim=1)

    return chan_pred

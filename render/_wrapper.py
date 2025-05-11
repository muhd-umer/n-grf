# render/_wrapper.py

import warnings

import torch
from torch.autograd import Function

try:
    from . import _C as cuda_ngrf

    CUDA_AVAILABLE = True
except ImportError:
    warnings.warn(
        "CUDA implementation for nGRF rendering not found. "
        "Ensure it is compiled (`pip install -e .` in render directory). "
        "Falling back to PyTorch for some operations if _torch_impl is used."
    )
    CUDA_AVAILABLE = False
    cuda_ngrf = None


class NormalizeQuaternion(Function):
    @staticmethod
    def forward(ctx, q_raw: torch.Tensor, eps: float = 1e-8):
        if not CUDA_AVAILABLE or cuda_ngrf is None:
            raise RuntimeError(
                "CUDA nGRF rendering not available for NormalizeQuaternion."
            )
        q_norm = torch.empty_like(q_raw)
        cuda_ngrf.normalize_quaternion_fwd_cuda(q_raw.contiguous(), eps, q_norm)

        norm_orig = torch.norm(q_raw, p=2, dim=1, keepdim=True)
        norm_clamped_orig = norm_orig.clamp(min=eps)

        ctx.save_for_backward(q_raw, q_norm, norm_clamped_orig)
        return q_norm

    @staticmethod
    def backward(ctx, grad_q_norm: torch.Tensor):
        if not CUDA_AVAILABLE or cuda_ngrf is None:
            raise RuntimeError(
                "CUDA nGRF rendering not available for NormalizeQuaternion backward."
            )
        q_raw, q_norm, norm_clamped_orig = ctx.saved_tensors
        grad_q_raw = torch.empty_like(q_raw)
        cuda_ngrf.normalize_quaternion_bwd_cuda(
            q_raw.contiguous(),
            q_norm.contiguous(),
            norm_clamped_orig.contiguous().squeeze(-1),
            grad_q_norm.contiguous(),
            grad_q_raw,
        )
        return grad_q_raw, None


class QuaternionToRotationMatrix(Function):
    @staticmethod
    def forward(ctx, q_norm: torch.Tensor):
        if not CUDA_AVAILABLE or cuda_ngrf is None:
            raise RuntimeError(
                "CUDA nGRF rendering not available for QuaternionToRotationMatrix."
            )
        R_matrices = torch.empty(
            q_norm.shape[0], 3, 3, dtype=q_norm.dtype, device=q_norm.device
        )
        cuda_ngrf.quaternion_to_rotation_matrix_fwd_cuda(
            q_norm.contiguous(), R_matrices
        )
        ctx.save_for_backward(q_norm)
        return R_matrices

    @staticmethod
    def backward(ctx, grad_R_matrices: torch.Tensor):
        if not CUDA_AVAILABLE or cuda_ngrf is None:
            raise RuntimeError(
                "CUDA nGRF rendering not available for QuaternionToRotationMatrix backward."
            )
        (q_norm,) = ctx.saved_tensors
        grad_q_norm = torch.empty_like(q_norm)
        cuda_ngrf.quaternion_to_rotation_matrix_bwd_cuda(
            q_norm.contiguous(), grad_R_matrices.contiguous(), grad_q_norm
        )
        return grad_q_norm


class ExponentialScaling(Function):
    @staticmethod
    def forward(ctx, s_log: torch.Tensor):
        if not CUDA_AVAILABLE or cuda_ngrf is None:
            raise RuntimeError(
                "CUDA nGRF rendering not available for ExponentialScaling."
            )
        s_act = torch.empty_like(s_log)
        cuda_ngrf.exponential_scaling_fwd_cuda(s_log.contiguous(), s_act)
        ctx.save_for_backward(s_act)
        return s_act

    @staticmethod
    def backward(ctx, grad_s_act: torch.Tensor):
        if not CUDA_AVAILABLE or cuda_ngrf is None:
            raise RuntimeError(
                "CUDA nGRF rendering not available for ExponentialScaling backward."
            )
        (s_act,) = ctx.saved_tensors
        grad_s_log = torch.empty_like(s_act)
        cuda_ngrf.exponential_scaling_bwd_cuda(
            s_act.contiguous(), grad_s_act.contiguous(), grad_s_log
        )
        return grad_s_log


class SigmoidActivation(Function):
    @staticmethod
    def forward(ctx, x_logit: torch.Tensor):
        if not CUDA_AVAILABLE or cuda_ngrf is None:
            raise RuntimeError(
                "CUDA nGRF rendering not available for SigmoidActivation."
            )
        x_act = torch.empty_like(x_logit)
        cuda_ngrf.sigmoid_activation_fwd_cuda(x_logit.contiguous(), x_act)
        ctx.save_for_backward(x_act)
        return x_act

    @staticmethod
    def backward(ctx, grad_x_act: torch.Tensor):
        if not CUDA_AVAILABLE or cuda_ngrf is None:
            raise RuntimeError(
                "CUDA nGRF rendering not available for SigmoidActivation backward."
            )
        (x_act,) = ctx.saved_tensors
        grad_x_logit = torch.empty_like(x_act)
        cuda_ngrf.sigmoid_activation_bwd_cuda(
            x_act.contiguous(), grad_x_act.contiguous(), grad_x_logit
        )
        return grad_x_logit


class BuildInverseCovariance(Function):
    @staticmethod
    def forward(ctx, R: torch.Tensor, s_act: torch.Tensor, eps_clamp: float = 1e-8):
        if not CUDA_AVAILABLE or cuda_ngrf is None:
            raise RuntimeError(
                "CUDA nGRF rendering not available for BuildInverseCovariance."
            )
        Sigma_inv = torch.empty(R.shape[0], 3, 3, dtype=R.dtype, device=R.device)
        cuda_ngrf.build_inverse_covariance_fwd_cuda(
            R.contiguous(), s_act.contiguous(), eps_clamp, Sigma_inv
        )

        s_clamped = torch.clamp(s_act, min=eps_clamp)
        s_inv_sq = 1.0 / (s_clamped * s_clamped)
        S_inv_sq_diag_tensor = torch.diag_embed(s_inv_sq)

        ctx.save_for_backward(R, s_act, s_clamped, S_inv_sq_diag_tensor)
        ctx.eps_clamp = eps_clamp
        return Sigma_inv

    @staticmethod
    def backward(ctx, grad_Sigma_inv: torch.Tensor):
        if not CUDA_AVAILABLE or cuda_ngrf is None:
            raise RuntimeError(
                "CUDA nGRF rendering not available for BuildInverseCovariance backward."
            )
        R, s_act, s_clamped, S_inv_sq_diag_tensor = ctx.saved_tensors
        eps_clamp = ctx.eps_clamp

        grad_R_matrices = (
            torch.empty_like(R).contiguous()
            if ctx.needs_input_grad[0]
            else torch.empty(0, device=R.device, dtype=R.dtype)
        )
        grad_s_act_out = (
            torch.empty_like(s_act).contiguous()
            if ctx.needs_input_grad[1]
            else torch.empty(0, device=s_act.device, dtype=s_act.dtype)
        )

        cuda_ngrf.build_inverse_covariance_bwd_cuda(
            R.contiguous(),
            s_act.contiguous(),
            s_clamped.contiguous(),
            S_inv_sq_diag_tensor.contiguous(),
            grad_Sigma_inv.contiguous(),
            eps_clamp,
            grad_R_matrices,
            grad_s_act_out,
        )

        grad_R_out = grad_R_matrices if ctx.needs_input_grad[0] else None
        grad_s_act_out_final = grad_s_act_out if ctx.needs_input_grad[1] else None

        return grad_R_out, grad_s_act_out_final, None


class ComputeSpatialWeight(Function):
    @staticmethod
    def forward(
        ctx,
        d_vec: torch.Tensor,
        Sigma_inv: torch.Tensor,
        alpha: torch.Tensor,
        clamp_max: float = 50.0,
    ):
        if not CUDA_AVAILABLE or cuda_ngrf is None:
            raise RuntimeError(
                "CUDA nGRF rendering not available for ComputeSpatialWeight."
            )

        spatial_weights_flt = torch.empty(
            d_vec.shape[0], dtype=d_vec.dtype, device=d_vec.device
        )
        cuda_ngrf.compute_spatial_weight_fwd_cuda(
            d_vec.contiguous(),
            Sigma_inv.contiguous(),
            alpha.contiguous().squeeze(-1),
            clamp_max,
            spatial_weights_flt,
        )

        alpha_squeezed = alpha.squeeze(-1)
        m_sq_fwd = torch.bmm(d_vec.unsqueeze(1), Sigma_inv)
        m_sq_fwd = torch.bmm(m_sq_fwd, d_vec.unsqueeze(2)).squeeze(-1).squeeze(-1)
        m_sq_clamped_fwd = torch.clamp(m_sq_fwd, max=clamp_max)
        pdf_weight_fwd = torch.exp(-0.5 * m_sq_clamped_fwd)

        ctx.save_for_backward(
            d_vec, Sigma_inv, alpha_squeezed, pdf_weight_fwd, m_sq_fwd, m_sq_clamped_fwd
        )
        ctx.clamp_max = clamp_max
        return spatial_weights_flt

    @staticmethod
    def backward(ctx, grad_weights_flt: torch.Tensor):
        if not CUDA_AVAILABLE or cuda_ngrf is None:
            raise RuntimeError(
                "CUDA nGRF rendering not available for ComputeSpatialWeight backward."
            )
        d_vec, Sigma_inv, alpha_squeezed, pdf_weight, m_sq, m_sq_clamped = (
            ctx.saved_tensors
        )
        clamp_max = ctx.clamp_max

        grad_weights_flt_contiguous = grad_weights_flt.contiguous()

        grad_d_vec_flt = (
            torch.empty_like(d_vec).contiguous()
            if ctx.needs_input_grad[0]
            else torch.empty(0, device=d_vec.device, dtype=d_vec.dtype)
        )
        grad_Sigma_inv_expanded = (
            torch.empty_like(Sigma_inv).contiguous()
            if ctx.needs_input_grad[1]
            else torch.empty(0, device=Sigma_inv.device, dtype=Sigma_inv.dtype)
        )
        grad_alpha_squeezed_flt = (
            torch.empty_like(alpha_squeezed).contiguous()
            if ctx.needs_input_grad[2]
            else torch.empty(
                0, device=alpha_squeezed.device, dtype=alpha_squeezed.dtype
            )
        )

        cuda_ngrf.compute_spatial_weight_bwd_cuda(
            d_vec.contiguous(),
            Sigma_inv.contiguous(),
            alpha_squeezed.contiguous(),
            pdf_weight.contiguous(),
            m_sq.contiguous(),
            m_sq_clamped.contiguous(),
            grad_weights_flt_contiguous,
            clamp_max,
            grad_d_vec_flt,
            grad_Sigma_inv_expanded,
            grad_alpha_squeezed_flt,
        )

        grad_d_out = grad_d_vec_flt if ctx.needs_input_grad[0] else None
        grad_Sigma_out = grad_Sigma_inv_expanded if ctx.needs_input_grad[1] else None
        grad_alpha_out = (
            grad_alpha_squeezed_flt.unsqueeze(-1) if ctx.needs_input_grad[2] else None
        )

        return grad_d_out, grad_Sigma_out, grad_alpha_out, None


class WeightedComplexSum(Function):
    @staticmethod
    def forward(ctx, weights: torch.Tensor, contributions_complex: torch.Tensor):
        if not CUDA_AVAILABLE or cuda_ngrf is None:
            raise RuntimeError(
                "CUDA nGRF rendering not available for WeightedComplexSum."
            )

        B, N_gauss = weights.shape
        Nt = contributions_complex.size(1)
        Nr = contributions_complex.size(2)

        H_pred_complex = torch.empty(
            B, Nt, Nr, dtype=contributions_complex.dtype, device=weights.device
        ).contiguous()
        cuda_ngrf.weighted_complex_sum_fwd_cuda(
            weights.contiguous(), contributions_complex.contiguous(), H_pred_complex
        )

        ctx.save_for_backward(weights, contributions_complex)
        return H_pred_complex

    @staticmethod
    def backward(ctx, grad_H_pred_complex: torch.Tensor):
        if not CUDA_AVAILABLE or cuda_ngrf is None:
            raise RuntimeError(
                "CUDA nGRF rendering not available for WeightedComplexSum backward."
            )
        weights, contributions_complex = ctx.saved_tensors

        grad_weights = (
            torch.empty_like(weights).contiguous()
            if ctx.needs_input_grad[0]
            else torch.empty(0, device=weights.device, dtype=weights.dtype)
        )
        grad_contributions_complex_out = (
            torch.empty_like(contributions_complex).contiguous()
            if ctx.needs_input_grad[1]
            else torch.empty(
                0,
                device=contributions_complex.device,
                dtype=contributions_complex.dtype,
            )
        )

        cuda_ngrf.weighted_complex_sum_bwd_cuda(
            weights.contiguous(),
            contributions_complex.contiguous(),
            grad_H_pred_complex.contiguous(),
            grad_weights,
            grad_contributions_complex_out,
        )

        grad_weights_out = grad_weights if ctx.needs_input_grad[0] else None
        grad_contrib_out = (
            grad_contributions_complex_out if ctx.needs_input_grad[1] else None
        )

        return grad_weights_out, grad_contrib_out


def render_channel(
    rx_positions: torch.Tensor,
    model: "GaussianRadioFieldModel",
    tx_position: torch.Tensor,
    nt: int,
    nr: int,
    eps: float = 1e-10,
) -> torch.Tensor:
    if not CUDA_AVAILABLE:
        raise RuntimeError(
            "CUDA nGRF rendering is not available. This function should only be called when CUDA is available."
        )

    batch_size = rx_positions.shape[0]
    device = rx_positions.device
    tx_position = tx_position.to(device)

    gauss_means_raw = model._xyz
    gauss_rot_raw = model._rotation
    gauss_scale_log = model._scaling

    num_gaussians = gauss_means_raw.shape[0]

    if num_gaussians == 0:
        warnings.warn("Warning: Rendering channel with zero Gaussians (CUDA path).")
        return torch.zeros(batch_size, nt, nr, dtype=torch.complex64, device=device)

    tx_pos_expanded_attr = tx_position
    if tx_position.dim() == 1:
        tx_pos_expanded_attr = tx_position.unsqueeze(0).expand(num_gaussians, -1)
    elif tx_position.shape[0] == 1 and num_gaussians > 1:
        tx_pos_expanded_attr = tx_position.expand(num_gaussians, -1)
    elif tx_position.shape[0] != num_gaussians and num_gaussians > 0:
        raise ValueError(
            f"tx_position shape {tx_position.shape} mismatch with num_gaussians {num_gaussians} for attribute_network input."
        )

    gauss_latents_torch, base_activations_logits_torch = model.attribute_network(
        gauss_means_raw, tx_pos_expanded_attr
    )
    gauss_activations_activated = SigmoidActivation.apply(base_activations_logits_torch)

    q_norm = NormalizeQuaternion.apply(gauss_rot_raw, eps)
    R_matrices = QuaternionToRotationMatrix.apply(q_norm)
    s_activated = ExponentialScaling.apply(gauss_scale_log)
    gauss_inv_covs = BuildInverseCovariance.apply(R_matrices, s_activated, eps)

    rx_pos_expanded = rx_positions.unsqueeze(1)
    gauss_means_expanded_for_dvec = gauss_means_raw.unsqueeze(0)
    d_vec = rx_pos_expanded - gauss_means_expanded_for_dvec
    d_vec_flt = d_vec.reshape(-1, 3)

    inv_covs_expanded = (
        gauss_inv_covs.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(-1, 3, 3)
    )
    activations_expanded_flt = (
        gauss_activations_activated.unsqueeze(0)
        .expand(batch_size, -1, -1)
        .reshape(-1, 1)
    )

    spatial_weights_flt = ComputeSpatialWeight.apply(
        d_vec_flt, inv_covs_expanded, activations_expanded_flt
    )
    spatial_weights = spatial_weights_flt.view(batch_size, num_gaussians)

    channel_contrib_ri_flat = model.contribution_decoder(gauss_latents_torch)
    channel_contrib_ri = channel_contrib_ri_flat.view(num_gaussians, 2, nt, nr)
    channel_contrib_cplx = torch.complex(
        channel_contrib_ri[:, 0, :, :],
        channel_contrib_ri[:, 1, :, :],
    )

    channel_pred = WeightedComplexSum.apply(spatial_weights, channel_contrib_cplx)

    return channel_pred.to(torch.complex64)

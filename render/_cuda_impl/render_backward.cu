// render/_cuda_impl/render_backward.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>

#include "checks.cuh"
#include "matrix.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600

__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

template <typename T>
__global__ void normalize_quaternion_bwd_kernel(
    const T* __restrict__ q_raw, const T* __restrict__ q_norm,
    const T* __restrict__ norm_clamped_flat, const T* __restrict__ grad_q_norm,
    const int N, T* __restrict__ grad_q_raw) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const T* q_raw_i = q_raw + i * 4;
    const T* q_norm_i = q_norm + i * 4;
    const T* grad_q_norm_i = grad_q_norm + i * 4;
    T* grad_q_raw_i = grad_q_raw + i * 4;
    const T norm_val = norm_clamped_flat[i];

    T q_norm_dot_grad = T(0.0);
    for (int k = 0; k < 4; ++k) {
        q_norm_dot_grad += q_norm_i[k] * grad_q_norm_i[k];
    }

    T inv_norm_val = T(1.0) / norm_val;
    for (int k = 0; k < 4; ++k) {
        grad_q_raw_i[k] =
            (grad_q_norm_i[k] - q_norm_dot_grad * q_norm_i[k]) * inv_norm_val;
    }
}

void normalize_quaternion_bwd_cuda(torch::Tensor q_raw, torch::Tensor q_norm,
                                   torch::Tensor norm_clamped,
                                   torch::Tensor grad_q_norm,
                                   torch::Tensor grad_q_raw) {
    CHECK_VALID_INPUT(q_raw);
    CHECK_VALID_INPUT(q_norm);
    CHECK_VALID_INPUT(norm_clamped);
    CHECK_VALID_INPUT(grad_q_norm);
    CHECK_VALID_INPUT(grad_q_raw);

    const int N = q_raw.size(0);
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    auto norm_clamped_squeezed = norm_clamped.squeeze(-1).contiguous();

    AT_DISPATCH_FLOATING_TYPES(
        q_raw.scalar_type(), "normalize_quaternion_bwd_kernel", ([&] {
            normalize_quaternion_bwd_kernel<scalar_t><<<blocks, threads>>>(
                q_raw.data_ptr<scalar_t>(), q_norm.data_ptr<scalar_t>(),
                norm_clamped_squeezed.data_ptr<scalar_t>(),
                grad_q_norm.data_ptr<scalar_t>(), N,
                grad_q_raw.data_ptr<scalar_t>());
        }));
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void quaternion_to_rotation_matrix_bwd_kernel(
    const T* __restrict__ q_norm, const T* __restrict__ grad_R_matrices,
    const int N, T* __restrict__ grad_q_norm) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const T* q = q_norm + i * 4;
    const T* grad_R = grad_R_matrices + i * 9;
    T* grad_q = grad_q_norm + i * 4;

    T w = q[0], x = q[1], y = q[2], z = q[3];
    T dR00 = grad_R[0], dR01 = grad_R[1], dR02 = grad_R[2];
    T dR10 = grad_R[3], dR11 = grad_R[4], dR12 = grad_R[5];
    T dR20 = grad_R[6], dR21 = grad_R[7], dR22 = grad_R[8];

    grad_q[0] = T(2.0) * (-z * dR01 + y * dR02 + z * dR10 - x * dR12 -
                          y * dR20 + x * dR21);
    grad_q[1] = T(2.0) * (y * dR01 + z * dR02 + y * dR10 - T(2.0) * x * dR11 -
                          w * dR12 + z * dR20 + w * dR21 - T(2.0) * x * dR22);
    grad_q[2] = T(2.0) * (-T(2.0) * y * dR00 + x * dR01 + w * dR02 + x * dR10 +
                          z * dR12 - w * dR20 + z * dR21 - T(2.0) * y * dR22);
    grad_q[3] = T(2.0) * (-T(2.0) * z * dR00 - w * dR01 + x * dR02 + w * dR10 -
                          T(2.0) * z * dR11 + y * dR12 + x * dR20 + y * dR21);
}

void quaternion_to_rotation_matrix_bwd_cuda(torch::Tensor q_norm,
                                            torch::Tensor grad_R_matrices,
                                            torch::Tensor grad_q_norm) {
    CHECK_VALID_INPUT(q_norm);
    CHECK_VALID_INPUT(grad_R_matrices);
    CHECK_VALID_INPUT(grad_q_norm);

    const int N = q_norm.size(0);
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(
        q_norm.scalar_type(), "quaternion_to_rotation_matrix_bwd_kernel", ([&] {
            quaternion_to_rotation_matrix_bwd_kernel<scalar_t>
                <<<blocks, threads>>>(q_norm.data_ptr<scalar_t>(),
                                      grad_R_matrices.data_ptr<scalar_t>(), N,
                                      grad_q_norm.data_ptr<scalar_t>());
        }));
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void exponential_scaling_bwd_kernel(const T* __restrict__ s_act,
                                               const T* __restrict__ grad_s_act,
                                               const int num_elements,
                                               T* __restrict__ grad_s_log) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;
    grad_s_log[i] = grad_s_act[i] * s_act[i];
}

void exponential_scaling_bwd_cuda(torch::Tensor s_act, torch::Tensor grad_s_act,
                                  torch::Tensor grad_s_log) {
    CHECK_VALID_INPUT(s_act);
    CHECK_VALID_INPUT(grad_s_act);
    CHECK_VALID_INPUT(grad_s_log);

    const int num_elements = s_act.numel();
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(
        s_act.scalar_type(), "exponential_scaling_bwd_kernel", ([&] {
            exponential_scaling_bwd_kernel<scalar_t><<<blocks, threads>>>(
                s_act.data_ptr<scalar_t>(), grad_s_act.data_ptr<scalar_t>(),
                num_elements, grad_s_log.data_ptr<scalar_t>());
        }));
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void sigmoid_activation_bwd_kernel(const T* __restrict__ x_act,
                                              const T* __restrict__ grad_x_act,
                                              const int num_elements,
                                              T* __restrict__ grad_x_logit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;
    grad_x_logit[i] = grad_x_act[i] * x_act[i] * (T(1.0) - x_act[i]);
}

void sigmoid_activation_bwd_cuda(torch::Tensor x_act, torch::Tensor grad_x_act,
                                 torch::Tensor grad_x_logit) {
    CHECK_VALID_INPUT(x_act);
    CHECK_VALID_INPUT(grad_x_act);
    CHECK_VALID_INPUT(grad_x_logit);

    const int num_elements = x_act.numel();
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(
        x_act.scalar_type(), "sigmoid_activation_bwd_kernel", ([&] {
            sigmoid_activation_bwd_kernel<scalar_t><<<blocks, threads>>>(
                x_act.data_ptr<scalar_t>(), grad_x_act.data_ptr<scalar_t>(),
                num_elements, grad_x_logit.data_ptr<scalar_t>());
        }));
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void build_inverse_covariance_bwd_kernel(
    const T* __restrict__ R_matrices, const T* __restrict__ s_act,
    const T* __restrict__ s_clamped, const T* __restrict__ S_inv_sq_diag_tensor,
    const T* __restrict__ grad_Sigma_inv, const T eps_clamp, const int N,
    T* __restrict__ grad_R_matrices, T* __restrict__ grad_s_act_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const T* R_i = R_matrices + i * 9;
    const T* s_act_i = s_act + i * 3;
    const T* s_clamped_i = s_clamped + i * 3;
    const T* S_inv_sq_diag_i = S_inv_sq_diag_tensor + i * 9;
    const T* grad_Sigma_inv_i = grad_Sigma_inv + i * 9;

    T* grad_R_i = grad_R_matrices + i * 9;
    T* grad_s_act_i = grad_s_act_out + i * 3;

    T grad_Sigma_inv_sym[9];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            grad_Sigma_inv_sym[r * 3 + c] =
                grad_Sigma_inv_i[r * 3 + c] + grad_Sigma_inv_i[c * 3 + r];
        }
    }

    T temp_R_D[9];
    matmul3x3(R_i, S_inv_sq_diag_i, temp_R_D);
    matmul3x3(grad_Sigma_inv_sym, temp_R_D, grad_R_i);

    T R_T_i[9];
    transpose3x3(R_i, R_T_i);

    T R_T_grad_Sigma_inv[9];
    matmul3x3(R_T_i, grad_Sigma_inv_i, R_T_grad_Sigma_inv);

    T R_T_grad_Sigma_inv_R[9];
    matmul3x3(R_T_grad_Sigma_inv, R_i, R_T_grad_Sigma_inv_R);

    T grad_D_diag[3];
    grad_D_diag[0] = R_T_grad_Sigma_inv_R[0 * 3 + 0];
    grad_D_diag[1] = R_T_grad_Sigma_inv_R[1 * 3 + 1];
    grad_D_diag[2] = R_T_grad_Sigma_inv_R[2 * 3 + 2];

    for (int k = 0; k < 3; ++k) {
        T s_clamped_k_cubed = s_clamped_i[k] * s_clamped_i[k] * s_clamped_i[k];
        T ds_inv_sq_ds_clamped_k =
            T(-2.0) / max(s_clamped_k_cubed, T(NGRF_ROBUST_EPSILON_FLOAT));
        T grad_s_clamped_k = grad_D_diag[k] * ds_inv_sq_ds_clamped_k;
        grad_s_act_i[k] =
            grad_s_clamped_k * (s_act_i[k] >= eps_clamp ? T(1.0) : T(0.0));
    }
}

void build_inverse_covariance_bwd_cuda(
    torch::Tensor R_matrices, torch::Tensor s_act, torch::Tensor s_clamped,
    torch::Tensor S_inv_sq_diag_tensor, torch::Tensor grad_Sigma_inv,
    float eps_clamp, torch::Tensor grad_R_matrices,
    torch::Tensor grad_s_act_out) {
    CHECK_VALID_INPUT(R_matrices);
    CHECK_VALID_INPUT(s_act);
    CHECK_VALID_INPUT(s_clamped);
    CHECK_VALID_INPUT(S_inv_sq_diag_tensor);
    CHECK_VALID_INPUT(grad_Sigma_inv);
    if (grad_R_matrices.numel() > 0) CHECK_VALID_INPUT(grad_R_matrices);
    if (grad_s_act_out.numel() > 0) CHECK_VALID_INPUT(grad_s_act_out);

    const int N = R_matrices.size(0);
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(
        R_matrices.scalar_type(), "build_inverse_covariance_bwd_kernel", ([&] {
            build_inverse_covariance_bwd_kernel<scalar_t><<<blocks, threads>>>(
                R_matrices.data_ptr<scalar_t>(), s_act.data_ptr<scalar_t>(),
                s_clamped.data_ptr<scalar_t>(),
                S_inv_sq_diag_tensor.data_ptr<scalar_t>(),
                grad_Sigma_inv.data_ptr<scalar_t>(),
                static_cast<scalar_t>(eps_clamp), N,
                grad_R_matrices.numel() > 0
                    ? grad_R_matrices.data_ptr<scalar_t>()
                    : nullptr,
                grad_s_act_out.numel() > 0 ? grad_s_act_out.data_ptr<scalar_t>()
                                           : nullptr);
        }));
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void compute_spatial_weight_bwd_kernel(
    const T* __restrict__ d_vec_flt, const T* __restrict__ Sigma_inv_expanded,
    const T* __restrict__ alpha_squeezed_flt,
    const T* __restrict__ pdf_weight_flt, const T* __restrict__ m_sq_flt,
    const T* __restrict__ m_sq_clamped_flt,
    const T* __restrict__ grad_weights_flt, const T clamp_max, const int M,
    T* __restrict__ grad_d_vec_flt, T* __restrict__ grad_Sigma_inv_expanded,
    T* __restrict__ grad_alpha_squeezed_flt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    const T* d_n = d_vec_flt + i * 3;
    const T* Sigma_inv_n = Sigma_inv_expanded + i * 9;
    const T alpha_n = alpha_squeezed_flt[i];
    const T pdf_n = pdf_weight_flt[i];
    const T m_sq_n = m_sq_flt[i];
    const T grad_w_n = grad_weights_flt[i];

    if (grad_alpha_squeezed_flt != nullptr) {
        grad_alpha_squeezed_flt[i] = grad_w_n * pdf_n;
    }

    T grad_pdf_n = grad_w_n * alpha_n;
    T grad_m_sq_clamped_n = grad_pdf_n * pdf_n * T(-0.5);
    T grad_m_sq_n =
        grad_m_sq_clamped_n * (m_sq_n <= clamp_max ? T(1.0) : T(0.0));

    if (grad_d_vec_flt != nullptr) {
        T* grad_d_n = grad_d_vec_flt + i * 3;
        T Sigma_inv_d[3];
        Sigma_inv_d[0] = Sigma_inv_n[0] * d_n[0] + Sigma_inv_n[1] * d_n[1] +
                         Sigma_inv_n[2] * d_n[2];
        Sigma_inv_d[1] = Sigma_inv_n[3] * d_n[0] + Sigma_inv_n[4] * d_n[1] +
                         Sigma_inv_n[5] * d_n[2];
        Sigma_inv_d[2] = Sigma_inv_n[6] * d_n[0] + Sigma_inv_n[7] * d_n[1] +
                         Sigma_inv_n[8] * d_n[2];
        grad_d_n[0] = T(2.0) * grad_m_sq_n * Sigma_inv_d[0];
        grad_d_n[1] = T(2.0) * grad_m_sq_n * Sigma_inv_d[1];
        grad_d_n[2] = T(2.0) * grad_m_sq_n * Sigma_inv_d[2];
    }

    if (grad_Sigma_inv_expanded != nullptr) {
        T* grad_Sigma_inv_n = grad_Sigma_inv_expanded + i * 9;
        T d_outer_d[9];
        d_outer_d[0] = d_n[0] * d_n[0];
        d_outer_d[1] = d_n[0] * d_n[1];
        d_outer_d[2] = d_n[0] * d_n[2];
        d_outer_d[3] = d_n[1] * d_n[0];
        d_outer_d[4] = d_n[1] * d_n[1];
        d_outer_d[5] = d_n[1] * d_n[2];
        d_outer_d[6] = d_n[2] * d_n[0];
        d_outer_d[7] = d_n[2] * d_n[1];
        d_outer_d[8] = d_n[2] * d_n[2];

        for (int k = 0; k < 9; ++k) {
            grad_Sigma_inv_n[k] = grad_m_sq_n * d_outer_d[k];
        }
    }
}

void compute_spatial_weight_bwd_cuda(
    torch::Tensor d_vec_flt, torch::Tensor Sigma_inv_expanded,
    torch::Tensor alpha_squeezed_flt, torch::Tensor pdf_weight_flt,
    torch::Tensor m_sq_flt, torch::Tensor m_sq_clamped_flt,
    torch::Tensor grad_weights_flt, float clamp_max,
    torch::Tensor grad_d_vec_flt, torch::Tensor grad_Sigma_inv_expanded,
    torch::Tensor grad_alpha_squeezed_flt) {
    CHECK_VALID_INPUT(d_vec_flt);
    CHECK_VALID_INPUT(Sigma_inv_expanded);
    CHECK_VALID_INPUT(alpha_squeezed_flt);
    CHECK_VALID_INPUT(pdf_weight_flt);
    CHECK_VALID_INPUT(m_sq_flt);
    CHECK_VALID_INPUT(m_sq_clamped_flt);
    CHECK_VALID_INPUT(grad_weights_flt);
    if (grad_d_vec_flt.numel() > 0) CHECK_VALID_INPUT(grad_d_vec_flt);
    if (grad_Sigma_inv_expanded.numel() > 0)
        CHECK_VALID_INPUT(grad_Sigma_inv_expanded);
    if (grad_alpha_squeezed_flt.numel() > 0)
        CHECK_VALID_INPUT(grad_alpha_squeezed_flt);

    const int M = d_vec_flt.size(0);
    const int threads = 256;
    const int blocks = (M + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(
        d_vec_flt.scalar_type(), "compute_spatial_weight_bwd_kernel", ([&] {
            compute_spatial_weight_bwd_kernel<scalar_t><<<blocks, threads>>>(
                d_vec_flt.data_ptr<scalar_t>(),
                Sigma_inv_expanded.data_ptr<scalar_t>(),
                alpha_squeezed_flt.data_ptr<scalar_t>(),
                pdf_weight_flt.data_ptr<scalar_t>(),
                m_sq_flt.data_ptr<scalar_t>(),
                m_sq_clamped_flt.data_ptr<scalar_t>(),
                grad_weights_flt.data_ptr<scalar_t>(),
                static_cast<scalar_t>(clamp_max), M,
                grad_d_vec_flt.numel() > 0 ? grad_d_vec_flt.data_ptr<scalar_t>()
                                           : nullptr,
                grad_Sigma_inv_expanded.numel() > 0
                    ? grad_Sigma_inv_expanded.data_ptr<scalar_t>()
                    : nullptr,
                grad_alpha_squeezed_flt.numel() > 0
                    ? grad_alpha_squeezed_flt.data_ptr<scalar_t>()
                    : nullptr);
        }));
    CUDA_CHECK(cudaGetLastError());
}

template <typename T_real, typename T_complex>
__global__ void weighted_complex_sum_bwd_kernel_weights(
    const T_complex* __restrict__ contributions_complex,
    const T_complex* __restrict__ grad_H_pred_complex, const int B,
    const int N_gauss, const int Nt, const int Nr,
    T_real* __restrict__ grad_weights) {
    const int b = blockIdx.x;
    const int n_start = threadIdx.x;
    const int n_stride = blockDim.x;

    if (b >= B) return;

    for (int n = n_start; n < N_gauss; n += n_stride) {
        T_real grad_w_bn = T_real(0.0);
        for (int t = 0; t < Nt; ++t) {
            for (int r = 0; r < Nr; ++r) {
                T_complex grad_H_bij =
                    grad_H_pred_complex[(b * Nt + t) * Nr + r];
                T_complex c_nij = contributions_complex[(n * Nt + t) * Nr + r];
                grad_w_bn += grad_H_bij.x * c_nij.x + grad_H_bij.y * c_nij.y;
            }
        }
        grad_weights[b * N_gauss + n] = grad_w_bn;
    }
}

template <typename T_real, typename T_complex>
__global__ void weighted_complex_sum_bwd_kernel_contributions(
    const T_real* __restrict__ weights,
    const T_complex* __restrict__ grad_H_pred_complex, const int B,
    const int N_gauss, const int Nt, const int Nr,
    T_complex* __restrict__ grad_contributions_complex) {
    const int n = blockIdx.x;
    const int t = threadIdx.x;
    const int r = threadIdx.y;

    if (n >= N_gauss || t >= Nt || r >= Nr) return;

    T_complex grad_c_nij = {T_real(0.0), T_real(0.0)};
    for (int b = 0; b < B; ++b) {
        T_real w_bn = weights[b * N_gauss + n];
        T_complex grad_H_bij = grad_H_pred_complex[(b * Nt + t) * Nr + r];
        grad_c_nij.x += w_bn * grad_H_bij.x;
        grad_c_nij.y += w_bn * grad_H_bij.y;
    }
    grad_contributions_complex[(n * Nt + t) * Nr + r] = grad_c_nij;
}

void weighted_complex_sum_bwd_cuda(torch::Tensor weights,
                                   torch::Tensor contributions_complex,
                                   torch::Tensor grad_H_pred_complex,
                                   torch::Tensor grad_weights,
                                   torch::Tensor grad_contributions_complex) {
    CHECK_VALID_INPUT(weights);
    CHECK_VALID_INPUT(contributions_complex);
    CHECK_VALID_INPUT(grad_H_pred_complex);
    if (grad_weights.numel() > 0) CHECK_VALID_INPUT(grad_weights);
    if (grad_contributions_complex.numel() > 0)
        CHECK_VALID_INPUT(grad_contributions_complex);

    const int B = weights.size(0);
    const int N_gauss = weights.size(1);
    const int Nt = contributions_complex.size(1);
    const int Nr = contributions_complex.size(2);

    if (grad_weights.numel() > 0) {
        const int threads_w = min(N_gauss, 1024);
        dim3 blocks_w(B);
        dim3 threads_dim_w(threads_w);

        if (weights.scalar_type() == torch::kFloat32 &&
            grad_H_pred_complex.scalar_type() == torch::kComplexFloat) {
            weighted_complex_sum_bwd_kernel_weights<float, float2>
                <<<blocks_w, threads_dim_w>>>(
                    reinterpret_cast<const float2*>(
                        contributions_complex.data_ptr<c10::complex<float>>()),
                    reinterpret_cast<const float2*>(
                        grad_H_pred_complex.data_ptr<c10::complex<float>>()),
                    B, N_gauss, Nt, Nr, grad_weights.data_ptr<float>());
        } else if (weights.scalar_type() == torch::kFloat64 &&
                   grad_H_pred_complex.scalar_type() == torch::kComplexDouble) {
            weighted_complex_sum_bwd_kernel_weights<double, double2>
                <<<blocks_w, threads_dim_w>>>(
                    reinterpret_cast<const double2*>(
                        contributions_complex.data_ptr<c10::complex<double>>()),
                    reinterpret_cast<const double2*>(
                        grad_H_pred_complex.data_ptr<c10::complex<double>>()),
                    B, N_gauss, Nt, Nr, grad_weights.data_ptr<double>());
        } else {
            AT_ERROR(
                "Unsupported dtype combination for "
                "weighted_complex_sum_bwd_cuda (weights)");
        }
        CUDA_CHECK(cudaGetLastError());
    }

    if (grad_contributions_complex.numel() > 0) {
        dim3 threads_c(Nt, Nr);
        dim3 blocks_c(N_gauss);
        TORCH_CHECK(Nt * Nr <= 1024,
                    "Nt*Nr exceeds max threads per block (1024) for "
                    "contributions gradient");

        if (weights.scalar_type() == torch::kFloat32 &&
            grad_H_pred_complex.scalar_type() == torch::kComplexFloat) {
            weighted_complex_sum_bwd_kernel_contributions<float, float2>
                <<<blocks_c, threads_c>>>(
                    weights.data_ptr<float>(),
                    reinterpret_cast<const float2*>(
                        grad_H_pred_complex.data_ptr<c10::complex<float>>()),
                    B, N_gauss, Nt, Nr,
                    reinterpret_cast<float2*>(
                        grad_contributions_complex
                            .data_ptr<c10::complex<float>>()));
        } else if (weights.scalar_type() == torch::kFloat64 &&
                   grad_H_pred_complex.scalar_type() == torch::kComplexDouble) {
            weighted_complex_sum_bwd_kernel_contributions<double, double2>
                <<<blocks_c, threads_c>>>(
                    weights.data_ptr<double>(),
                    reinterpret_cast<const double2*>(
                        grad_H_pred_complex.data_ptr<c10::complex<double>>()),
                    B, N_gauss, Nt, Nr,
                    reinterpret_cast<double2*>(
                        grad_contributions_complex
                            .data_ptr<c10::complex<double>>()));
        } else {
            AT_ERROR(
                "Unsupported dtype combination for "
                "weighted_complex_sum_bwd_cuda (contributions)");
        }
        CUDA_CHECK(cudaGetLastError());
    }
}
// render/_cuda_impl/render.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>

#include "checks.cuh"
#include "matrix.cuh"

template <typename T>
__global__ void normalize_quaternion_fwd_kernel(const T* __restrict__ q_raw,
                                                const T eps, const int N,
                                                T* __restrict__ q_norm) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const T* q_raw_i = q_raw + i * 4;
    T* q_norm_i = q_norm + i * 4;

    T norm_sq = T(0.0);
    for (int k = 0; k < 4; ++k) {
        norm_sq += q_raw_i[k] * q_raw_i[k];
    }
    T norm = sqrt(norm_sq);
    T norm_clamped = max(norm, eps);
    T inv_norm_clamped = T(1.0) / norm_clamped;

    for (int k = 0; k < 4; ++k) {
        q_norm_i[k] = q_raw_i[k] * inv_norm_clamped;
    }
}

void normalize_quaternion_fwd_cuda(torch::Tensor q_raw, float eps,
                                   torch::Tensor q_norm) {
    CHECK_VALID_INPUT(q_raw);
    CHECK_VALID_INPUT(q_norm);
    TORCH_CHECK(q_raw.dim() == 2 && q_raw.size(1) == 4, "q_raw must be Nx4");
    TORCH_CHECK(q_norm.sizes() == q_raw.sizes(),
                "q_norm must match q_raw size");

    const int N = q_raw.size(0);
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(
        q_raw.scalar_type(), "normalize_quaternion_fwd_kernel", ([&] {
            normalize_quaternion_fwd_kernel<scalar_t><<<blocks, threads>>>(
                q_raw.data_ptr<scalar_t>(), static_cast<scalar_t>(eps), N,
                q_norm.data_ptr<scalar_t>());
        }));
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void quaternion_to_rotation_matrix_fwd_kernel(
    const T* __restrict__ q_norm, const int N, T* __restrict__ R_matrices) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const T* q = q_norm + i * 4;
    T* R = R_matrices + i * 9;

    T w = q[0], x = q[1], y = q[2], z = q[3];

    T x2 = x * x, y2 = y * y, z2 = z * z;
    T xy = x * y, xz = x * z, yz = y * z;
    T wx = w * x, wy = w * y, wz = w * z;

    R[0] = T(1.0) - T(2.0) * (y2 + z2);
    R[1] = T(2.0) * (xy - wz);
    R[2] = T(2.0) * (xz + wy);
    R[3] = T(2.0) * (xy + wz);
    R[4] = T(1.0) - T(2.0) * (x2 + z2);
    R[5] = T(2.0) * (yz - wx);
    R[6] = T(2.0) * (xz - wy);
    R[7] = T(2.0) * (yz + wx);
    R[8] = T(1.0) - T(2.0) * (x2 + y2);
}

void quaternion_to_rotation_matrix_fwd_cuda(torch::Tensor q_norm,
                                            torch::Tensor R_matrices) {
    CHECK_VALID_INPUT(q_norm);
    CHECK_VALID_INPUT(R_matrices);
    TORCH_CHECK(q_norm.dim() == 2 && q_norm.size(1) == 4, "q_norm must be Nx4");
    TORCH_CHECK(R_matrices.dim() == 3 && R_matrices.size(1) == 3 &&
                    R_matrices.size(2) == 3,
                "R_matrices must be Nx3x3");
    TORCH_CHECK(R_matrices.size(0) == q_norm.size(0), "Batch size mismatch");

    const int N = q_norm.size(0);
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(
        q_norm.scalar_type(), "quaternion_to_rotation_matrix_fwd_kernel", ([&] {
            quaternion_to_rotation_matrix_fwd_kernel<scalar_t>
                <<<blocks, threads>>>(q_norm.data_ptr<scalar_t>(), N,
                                      R_matrices.data_ptr<scalar_t>());
        }));
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void exponential_scaling_fwd_kernel(const T* __restrict__ s_log,
                                               const int num_elements,
                                               T* __restrict__ s_act) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;
    s_act[i] = exp(s_log[i]);
}

void exponential_scaling_fwd_cuda(torch::Tensor s_log, torch::Tensor s_act) {
    CHECK_VALID_INPUT(s_log);
    CHECK_VALID_INPUT(s_act);
    TORCH_CHECK(s_act.sizes() == s_log.sizes(), "s_act must match s_log size");

    const int num_elements = s_log.numel();
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(
        s_log.scalar_type(), "exponential_scaling_fwd_kernel", ([&] {
            exponential_scaling_fwd_kernel<scalar_t>
                <<<blocks, threads>>>(s_log.data_ptr<scalar_t>(), num_elements,
                                      s_act.data_ptr<scalar_t>());
        }));
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void sigmoid_activation_fwd_kernel(const T* __restrict__ x_logit,
                                              const int num_elements,
                                              T* __restrict__ x_act) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;
    x_act[i] = T(1.0) / (T(1.0) + exp(-x_logit[i]));
}

void sigmoid_activation_fwd_cuda(torch::Tensor x_logit, torch::Tensor x_act) {
    CHECK_VALID_INPUT(x_logit);
    CHECK_VALID_INPUT(x_act);
    TORCH_CHECK(x_act.sizes() == x_logit.sizes(),
                "x_act must match x_logit size");

    const int num_elements = x_logit.numel();
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(
        x_logit.scalar_type(), "sigmoid_activation_fwd_kernel", ([&] {
            sigmoid_activation_fwd_kernel<scalar_t>
                <<<blocks, threads>>>(x_logit.data_ptr<scalar_t>(),
                                      num_elements, x_act.data_ptr<scalar_t>());
        }));
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void build_inverse_covariance_fwd_kernel(
    const T* __restrict__ R_matrices, const T* __restrict__ s_act,
    const T eps_clamp, const int N, T* __restrict__ Sigma_inv) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const T* R = R_matrices + i * 9;
    const T* s = s_act + i * 3;
    T* Sigma_inv_i = Sigma_inv + i * 9;

    T s_clamped[3];
    s_clamped[0] = max(s[0], eps_clamp);
    s_clamped[1] = max(s[1], eps_clamp);
    s_clamped[2] = max(s[2], eps_clamp);

    T S_inv_sq_diag[9] = {0};
    S_inv_sq_diag[0] = T(1.0) / (s_clamped[0] * s_clamped[0]);
    S_inv_sq_diag[4] = T(1.0) / (s_clamped[1] * s_clamped[1]);
    S_inv_sq_diag[8] = T(1.0) / (s_clamped[2] * s_clamped[2]);

    T R_S_inv_sq[9];
    matmul3x3(R, S_inv_sq_diag, R_S_inv_sq);

    T R_T[9];
    transpose3x3(R, R_T);

    matmul3x3(R_S_inv_sq, R_T, Sigma_inv_i);
}

void build_inverse_covariance_fwd_cuda(torch::Tensor R_matrices,
                                       torch::Tensor s_act, float eps_clamp,
                                       torch::Tensor Sigma_inv) {
    CHECK_VALID_INPUT(R_matrices);
    CHECK_VALID_INPUT(s_act);
    CHECK_VALID_INPUT(Sigma_inv);
    TORCH_CHECK(R_matrices.dim() == 3 && R_matrices.size(1) == 3 &&
                    R_matrices.size(2) == 3,
                "R_matrices must be Nx3x3");
    TORCH_CHECK(s_act.dim() == 2 && s_act.size(1) == 3, "s_act must be Nx3");
    TORCH_CHECK(Sigma_inv.sizes() == R_matrices.sizes(),
                "Sigma_inv must match R_matrices size");
    TORCH_CHECK(R_matrices.size(0) == s_act.size(0),
                "Batch size mismatch for R and s_act");

    const int N = R_matrices.size(0);
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(
        R_matrices.scalar_type(), "build_inverse_covariance_fwd_kernel", ([&] {
            build_inverse_covariance_fwd_kernel<scalar_t><<<blocks, threads>>>(
                R_matrices.data_ptr<scalar_t>(), s_act.data_ptr<scalar_t>(),
                static_cast<scalar_t>(eps_clamp), N,
                Sigma_inv.data_ptr<scalar_t>());
        }));
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void compute_spatial_weight_fwd_kernel(
    const T* __restrict__ d_vec_flt, const T* __restrict__ Sigma_inv_expanded,
    const T* __restrict__ alpha_squeezed_flt, const T clamp_max, const int M,
    T* __restrict__ spatial_weights_flt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    const T* d_n = d_vec_flt + i * 3;
    const T* Sigma_inv_n = Sigma_inv_expanded + i * 9;
    const T alpha_n = alpha_squeezed_flt[i];

    T temp_vec[3];
    temp_vec[0] = Sigma_inv_n[0] * d_n[0] + Sigma_inv_n[1] * d_n[1] +
                  Sigma_inv_n[2] * d_n[2];
    temp_vec[1] = Sigma_inv_n[3] * d_n[0] + Sigma_inv_n[4] * d_n[1] +
                  Sigma_inv_n[5] * d_n[2];
    temp_vec[2] = Sigma_inv_n[6] * d_n[0] + Sigma_inv_n[7] * d_n[1] +
                  Sigma_inv_n[8] * d_n[2];

    T m_sq = d_n[0] * temp_vec[0] + d_n[1] * temp_vec[1] + d_n[2] * temp_vec[2];
    T m_sq_clamped = min(m_sq, clamp_max);
    T pdf_weight = exp(T(-0.5) * m_sq_clamped);

    spatial_weights_flt[i] = alpha_n * pdf_weight;
}

void compute_spatial_weight_fwd_cuda(torch::Tensor d_vec_flt,
                                     torch::Tensor Sigma_inv_expanded,
                                     torch::Tensor alpha_squeezed_flt,
                                     float clamp_max,
                                     torch::Tensor spatial_weights_flt) {
    CHECK_VALID_INPUT(d_vec_flt);
    CHECK_VALID_INPUT(Sigma_inv_expanded);
    CHECK_VALID_INPUT(alpha_squeezed_flt);
    CHECK_VALID_INPUT(spatial_weights_flt);

    const int M = d_vec_flt.size(0);
    TORCH_CHECK(d_vec_flt.dim() == 2 && d_vec_flt.size(1) == 3,
                "d_vec_flt must be (B*N_gauss)x3");
    TORCH_CHECK(
        Sigma_inv_expanded.dim() == 3 && Sigma_inv_expanded.size(0) == M &&
            Sigma_inv_expanded.size(1) == 3 && Sigma_inv_expanded.size(2) == 3,
        "Sigma_inv_expanded must be (B*N_gauss)x3x3");
    TORCH_CHECK(
        alpha_squeezed_flt.dim() == 1 && alpha_squeezed_flt.size(0) == M,
        "alpha_squeezed_flt must be (B*N_gauss)");
    TORCH_CHECK(
        spatial_weights_flt.dim() == 1 && spatial_weights_flt.size(0) == M,
        "spatial_weights_flt must be (B*N_gauss)");

    const int threads = 256;
    const int blocks = (M + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(
        d_vec_flt.scalar_type(), "compute_spatial_weight_fwd_kernel", ([&] {
            compute_spatial_weight_fwd_kernel<scalar_t>
                <<<blocks, threads>>>(d_vec_flt.data_ptr<scalar_t>(),
                                      Sigma_inv_expanded.data_ptr<scalar_t>(),
                                      alpha_squeezed_flt.data_ptr<scalar_t>(),
                                      static_cast<scalar_t>(clamp_max), M,
                                      spatial_weights_flt.data_ptr<scalar_t>());
        }));
    CUDA_CHECK(cudaGetLastError());
}

template <typename T_real, typename T_complex>
__global__ void weighted_complex_sum_fwd_kernel(
    const T_real* __restrict__ weights,
    const T_complex* __restrict__ contributions_complex, const int B,
    const int N_gauss, const int Nt, const int Nr,
    T_complex* __restrict__ H_pred_complex) {
    const int b = blockIdx.x;
    const int t = threadIdx.x;
    const int r = threadIdx.y;

    if (b >= B || t >= Nt || r >= Nr) return;

    T_complex sum_val = {T_real(0.0), T_real(0.0)};

    for (int n = 0; n < N_gauss; ++n) {
        T_real w_bn = weights[b * N_gauss + n];
        T_complex c_nij = contributions_complex[(n * Nt + t) * Nr + r];
        sum_val.x += w_bn * c_nij.x;
        sum_val.y += w_bn * c_nij.y;
    }
    H_pred_complex[(b * Nt + t) * Nr + r] = sum_val;
}

void weighted_complex_sum_fwd_cuda(torch::Tensor weights,
                                   torch::Tensor contributions_complex,
                                   torch::Tensor H_pred_complex) {
    CHECK_VALID_INPUT(weights);
    CHECK_VALID_INPUT(contributions_complex);
    CHECK_VALID_INPUT(H_pred_complex);

    const int B = weights.size(0);
    const int N_gauss = weights.size(1);
    TORCH_CHECK(contributions_complex.dim() == 3,
                "contributions_complex must be N_gauss x Nt x Nr");
    TORCH_CHECK(contributions_complex.size(0) == N_gauss, "N_gauss mismatch");
    const int Nt = contributions_complex.size(1);
    const int Nr = contributions_complex.size(2);

    TORCH_CHECK(H_pred_complex.dim() == 3 && H_pred_complex.size(0) == B &&
                    H_pred_complex.size(1) == Nt &&
                    H_pred_complex.size(2) == Nr,
                "H_pred_complex shape mismatch");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32 ||
                    weights.scalar_type() == torch::kFloat64,
                "weights must be float or double");
    TORCH_CHECK(contributions_complex.is_complex(),
                "contributions_complex must be complex");
    TORCH_CHECK(
        H_pred_complex.scalar_type() == contributions_complex.scalar_type(),
        "H_pred_complex dtype must match contributions_complex");

    dim3 threads(Nt, Nr);
    TORCH_CHECK(Nt * Nr <= 1024, "Nt*Nr exceeds max threads per block (1024)");
    dim3 blocks(B);

    if (weights.scalar_type() == torch::kFloat32 &&
        contributions_complex.scalar_type() == torch::kComplexFloat) {
        weighted_complex_sum_fwd_kernel<float, float2><<<blocks, threads>>>(
            weights.data_ptr<float>(),
            reinterpret_cast<float2*>(
                contributions_complex.data_ptr<c10::complex<float>>()),
            B, N_gauss, Nt, Nr,
            reinterpret_cast<float2*>(
                H_pred_complex.data_ptr<c10::complex<float>>()));
    } else if (weights.scalar_type() == torch::kFloat64 &&
               contributions_complex.scalar_type() == torch::kComplexDouble) {
        weighted_complex_sum_fwd_kernel<double, double2><<<blocks, threads>>>(
            weights.data_ptr<double>(),
            reinterpret_cast<double2*>(
                contributions_complex.data_ptr<c10::complex<double>>()),
            B, N_gauss, Nt, Nr,
            reinterpret_cast<double2*>(
                H_pred_complex.data_ptr<c10::complex<double>>()));
    } else {
        AT_ERROR(
            "Unsupported dtype combination for weighted_complex_sum_fwd_cuda");
    }
    CUDA_CHECK(cudaGetLastError());
}
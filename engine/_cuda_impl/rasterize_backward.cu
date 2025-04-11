// engine/_cuda_impl/rasterize_backward.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

#include "checks.cuh"
#include "matrix.cuh"

template <typename T>
__launch_bounds__(1024) __global__
    void compute_gaussian_influence_backward_kernel(
        const T* __restrict__ uv, const T* __restrict__ cov2d,
        const T* __restrict__ influences, const T* __restrict__ grad_influences,
        const int num_tx, const int num_rx, const int N,
        T* __restrict__ grad_uv, T* __restrict__ grad_cov2d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const T a = cov2d[i * 4 + 0];
    const T b = cov2d[i * 4 + 1];
    const T c = cov2d[i * 4 + 2];
    const T d_cov = cov2d[i * 4 + 3];

    const T det = a * d_cov - b * c;
    if (abs(det) < T(1e-10)) {
        grad_uv[i * 2 + 0] = T(0.0);
        grad_uv[i * 2 + 1] = T(0.0);
        grad_cov2d[i * 4 + 0] = T(0.0);
        grad_cov2d[i * 4 + 1] = T(0.0);
        grad_cov2d[i * 4 + 2] = T(0.0);
        grad_cov2d[i * 4 + 3] = T(0.0);
        return;
    }
    const T inv_det = T(1.0) / det;

    T inv_cov[4];
    inv_cov[0] = d_cov * inv_det;
    inv_cov[1] = -b * inv_det;
    inv_cov[2] = -c * inv_det;
    inv_cov[3] = a * inv_det;

    const T u_i = uv[i * 2 + 0];
    const T v_i = uv[i * 2 + 1];

    T grad_u_sum = T(0.0);
    T grad_v_sum = T(0.0);
    T grad_cov2d_sum[4] = {T(0.0), T(0.0), T(0.0), T(0.0)};

    for (int tx = 0; tx < num_tx; tx++) {
        for (int rx = 0; rx < num_rx; rx++) {
            const int influence_idx = i * num_tx * num_rx + tx * num_rx + rx;
            const T grad_influence = grad_influences[influence_idx];

            if (abs(grad_influence) > T(1e-10)) {
                const T antenna_u = tx + T(0.5);
                const T antenna_v = rx + T(0.5);

                T disp[2];
                disp[0] = u_i - antenna_u;
                disp[1] = v_i - antenna_v;

                const T influence = influences[influence_idx];

                const T grad_md = grad_influence * (-T(0.5) * influence);

                T grad_disp[2];
                grad_disp[0] = inv_cov[0] * disp[0] + inv_cov[1] * disp[1];
                grad_disp[1] = inv_cov[2] * disp[0] + inv_cov[3] * disp[1];

                grad_u_sum += grad_md * T(2.0) * grad_disp[0];
                grad_v_sum += grad_md * T(2.0) * grad_disp[1];

                T d_outer[4];
                d_outer[0] = disp[0] * disp[0];
                d_outer[1] = disp[0] * disp[1];
                d_outer[2] = disp[1] * disp[0];
                d_outer[3] = disp[1] * disp[1];

                T temp_mat[4];
                temp_mat[0] = grad_md * d_outer[0];
                temp_mat[1] = grad_md * d_outer[1];
                temp_mat[2] = grad_md * d_outer[2];
                temp_mat[3] = grad_md * d_outer[3];

                T temp_mat2[4];
                matrix_multiply<T>(temp_mat, inv_cov, temp_mat2, 2, 2, 2);

                T grad_C_term[4];
                matrix_multiply<T>(inv_cov, temp_mat2, grad_C_term, 2, 2, 2);

                grad_cov2d_sum[0] -= grad_C_term[0];
                grad_cov2d_sum[1] -= grad_C_term[1];
                grad_cov2d_sum[2] -= grad_C_term[2];
                grad_cov2d_sum[3] -= grad_C_term[3];
            }
        }
    }

    grad_uv[i * 2 + 0] = grad_u_sum;
    grad_uv[i * 2 + 1] = grad_v_sum;

    grad_cov2d[i * 4 + 0] = grad_cov2d_sum[0];
    grad_cov2d[i * 4 + 1] = grad_cov2d_sum[1];
    grad_cov2d[i * 4 + 2] = grad_cov2d_sum[2];
    grad_cov2d[i * 4 + 3] = grad_cov2d_sum[3];
}

void compute_gaussian_influence_backward_cuda(
    torch::Tensor uv, torch::Tensor cov2d, torch::Tensor influences,
    torch::Tensor grad_influences, int num_tx, int num_rx,
    torch::Tensor grad_uv, torch::Tensor grad_cov2d) {
    CHECK_VALID_INPUT(uv);
    CHECK_VALID_INPUT(cov2d);
    CHECK_VALID_INPUT(influences);
    CHECK_VALID_INPUT(grad_influences);
    CHECK_VALID_INPUT(grad_uv);
    CHECK_VALID_INPUT(grad_cov2d);

    const int N = uv.size(0);
    TORCH_CHECK(uv.size(1) == 2, "uv must have shape Nx2");
    TORCH_CHECK(cov2d.size(0) == N && cov2d.size(1) == 2 && cov2d.size(2) == 2,
                "cov2d must have shape Nx2x2");
    TORCH_CHECK(influences.size(0) == N && influences.size(1) == num_tx &&
                    influences.size(2) == num_rx,
                "influences must have shape Nx" + std::to_string(num_tx) + "x" +
                    std::to_string(num_rx));
    TORCH_CHECK(grad_influences.size(0) == N &&
                    grad_influences.size(1) == num_tx &&
                    grad_influences.size(2) == num_rx,
                "grad_influences must have shape Nx" + std::to_string(num_tx) +
                    "x" + std::to_string(num_rx));
    TORCH_CHECK(grad_uv.size(0) == N && grad_uv.size(1) == 2,
                "grad_uv must have shape Nx2");
    TORCH_CHECK(grad_cov2d.size(0) == N && grad_cov2d.size(1) == 2 &&
                    grad_cov2d.size(2) == 2,
                "grad_cov2d must have shape Nx2x2");

    const int max_threads_per_block = 1024;
    const int num_blocks =
        (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    if (uv.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(cov2d);
        CHECK_FLOAT_TENSOR(influences);
        CHECK_FLOAT_TENSOR(grad_influences);
        CHECK_FLOAT_TENSOR(grad_uv);
        CHECK_FLOAT_TENSOR(grad_cov2d);
        compute_gaussian_influence_backward_kernel<float>
            <<<gridsize, blocksize>>>(
                uv.data_ptr<float>(), cov2d.data_ptr<float>(),
                influences.data_ptr<float>(), grad_influences.data_ptr<float>(),
                num_tx, num_rx, N, grad_uv.data_ptr<float>(),
                grad_cov2d.data_ptr<float>());
    } else if (uv.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(cov2d);
        CHECK_DOUBLE_TENSOR(influences);
        CHECK_DOUBLE_TENSOR(grad_influences);
        CHECK_DOUBLE_TENSOR(grad_uv);
        CHECK_DOUBLE_TENSOR(grad_cov2d);
        compute_gaussian_influence_backward_kernel<double>
            <<<gridsize, blocksize>>>(
                uv.data_ptr<double>(), cov2d.data_ptr<double>(),
                influences.data_ptr<double>(),
                grad_influences.data_ptr<double>(), num_tx, num_rx, N,
                grad_uv.data_ptr<double>(), grad_cov2d.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", uv.dtype());
    }
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void compute_wireless_channel_backward_kernel(
    const T* __restrict__ attenuation, const T* __restrict__ phase_rotation,
    const T* __restrict__ distances, const T wavelength,
    const T* __restrict__ grad_real_contributions,
    const T* __restrict__ grad_imag_contributions, const int N,
    T* __restrict__ grad_attenuation, T* __restrict__ grad_phase_rotation,
    T* __restrict__ grad_distances) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const T PI = T(3.14159265358979323846);

    const T r = distances[i];
    const T r_safe = max(r, T(1e-10));
    const T path_loss = wavelength / (T(4.0) * PI * r_safe);
    const T phase_shift = -T(2.0) * PI * r / wavelength;

    const T A = attenuation[i];
    const T psi = phase_rotation[i];

    const T total_phase = psi + phase_shift;
    const T total_attenuation = A * path_loss;

    const T cos_phase = cos(total_phase);
    const T sin_phase = sin(total_phase);

    const T grad_real = grad_real_contributions[i];
    const T grad_imag = grad_imag_contributions[i];

    grad_attenuation[i] =
        grad_real * path_loss * cos_phase + grad_imag * path_loss * sin_phase;

    grad_phase_rotation[i] = -grad_real * total_attenuation * sin_phase +
                             grad_imag * total_attenuation * cos_phase;

    const T grad_path_loss = -wavelength / (T(4.0) * PI * r_safe * r_safe);
    const T grad_phase_shift = -T(2.0) * PI / wavelength;

    grad_distances[i] =
        grad_real * (grad_path_loss * A * cos_phase -
                     total_attenuation * sin_phase * grad_phase_shift) +
        grad_imag * (grad_path_loss * A * sin_phase +
                     total_attenuation * cos_phase * grad_phase_shift);
}

void compute_wireless_channel_backward_cuda(
    torch::Tensor attenuation, torch::Tensor phase_rotation,
    torch::Tensor distances, float wavelength,
    torch::Tensor grad_real_contributions,
    torch::Tensor grad_imag_contributions, torch::Tensor grad_attenuation,
    torch::Tensor grad_phase_rotation, torch::Tensor grad_distances) {
    CHECK_VALID_INPUT(attenuation);
    CHECK_VALID_INPUT(phase_rotation);
    CHECK_VALID_INPUT(distances);
    CHECK_VALID_INPUT(grad_real_contributions);
    CHECK_VALID_INPUT(grad_imag_contributions);
    CHECK_VALID_INPUT(grad_attenuation);
    CHECK_VALID_INPUT(grad_phase_rotation);
    CHECK_VALID_INPUT(grad_distances);

    const int N = attenuation.size(0);
    TORCH_CHECK(attenuation.size(0) == N &&
                    (attenuation.dim() == 1 || attenuation.size(1) == 1),
                "attenuation must have shape N or Nx1");
    TORCH_CHECK(phase_rotation.size(0) == N &&
                    (phase_rotation.dim() == 1 || phase_rotation.size(1) == 1),
                "phase_rotation must have shape N or Nx1");
    TORCH_CHECK(distances.size(0) == N && distances.dim() == 1,
                "distances must have shape N");
    TORCH_CHECK(grad_real_contributions.size(0) == N &&
                    (grad_real_contributions.dim() == 1 ||
                     grad_real_contributions.size(1) == 1),
                "grad_real_contributions must have shape N or Nx1");
    TORCH_CHECK(grad_imag_contributions.size(0) == N &&
                    (grad_imag_contributions.dim() == 1 ||
                     grad_imag_contributions.size(1) == 1),
                "grad_imag_contributions must have shape N or Nx1");
    TORCH_CHECK(
        grad_attenuation.size(0) == N &&
            (grad_attenuation.dim() == 1 || grad_attenuation.size(1) == 1),
        "grad_attenuation must have shape N or Nx1");
    TORCH_CHECK(
        grad_phase_rotation.size(0) == N && (grad_phase_rotation.dim() == 1 ||
                                             grad_phase_rotation.size(1) == 1),
        "grad_phase_rotation must have shape N or Nx1");
    TORCH_CHECK(grad_distances.size(0) == N && grad_distances.dim() == 1,
                "grad_distances must have shape N");

    const int max_threads_per_block = 1024;
    const int num_blocks =
        (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    auto att_cont = attenuation.contiguous().view({N});
    auto phase_cont = phase_rotation.contiguous().view({N});
    auto dist_cont = distances.contiguous();
    auto grad_real_cont = grad_real_contributions.contiguous().view({N});
    auto grad_imag_cont = grad_imag_contributions.contiguous().view({N});
    auto grad_att_cont = grad_attenuation.contiguous().view({N});
    auto grad_phase_cont = grad_phase_rotation.contiguous().view({N});
    auto grad_dist_cont = grad_distances.contiguous();

    if (attenuation.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(phase_rotation);
        CHECK_FLOAT_TENSOR(distances);
        CHECK_FLOAT_TENSOR(grad_real_contributions);
        CHECK_FLOAT_TENSOR(grad_imag_contributions);
        CHECK_FLOAT_TENSOR(grad_attenuation);
        CHECK_FLOAT_TENSOR(grad_phase_rotation);
        CHECK_FLOAT_TENSOR(grad_distances);
        compute_wireless_channel_backward_kernel<float>
            <<<gridsize, blocksize>>>(att_cont.data_ptr<float>(),
                                      phase_cont.data_ptr<float>(),
                                      dist_cont.data_ptr<float>(), wavelength,
                                      grad_real_cont.data_ptr<float>(),
                                      grad_imag_cont.data_ptr<float>(), N,
                                      grad_att_cont.data_ptr<float>(),
                                      grad_phase_cont.data_ptr<float>(),
                                      grad_dist_cont.data_ptr<float>());
    } else if (attenuation.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(phase_rotation);
        CHECK_DOUBLE_TENSOR(distances);
        CHECK_DOUBLE_TENSOR(grad_real_contributions);
        CHECK_DOUBLE_TENSOR(grad_imag_contributions);
        CHECK_DOUBLE_TENSOR(grad_attenuation);
        CHECK_DOUBLE_TENSOR(grad_phase_rotation);
        CHECK_DOUBLE_TENSOR(grad_distances);
        compute_wireless_channel_backward_kernel<double>
            <<<gridsize, blocksize>>>(
                att_cont.data_ptr<double>(), phase_cont.data_ptr<double>(),
                dist_cont.data_ptr<double>(), (double)wavelength,
                grad_real_cont.data_ptr<double>(),
                grad_imag_cont.data_ptr<double>(), N,
                grad_att_cont.data_ptr<double>(),
                grad_phase_cont.data_ptr<double>(),
                grad_dist_cont.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", attenuation.dtype());
    }
    cudaDeviceSynchronize();

    if (attenuation.dim() > 1 && attenuation.size(1) == 1) {
        grad_attenuation.copy_(grad_att_cont.view({N, 1}));
    }
    if (phase_rotation.dim() > 1 && phase_rotation.size(1) == 1) {
        grad_phase_rotation.copy_(grad_phase_cont.view({N, 1}));
    }
}

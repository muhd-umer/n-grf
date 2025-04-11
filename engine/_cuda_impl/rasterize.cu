// engine/_cuda_impl/rasterize.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>

#include "checks.cuh"
#include "matrix.cuh"

template <typename T>
__global__ void compute_gaussian_influence_kernel(const T* __restrict__ uv,
                                                  const T* __restrict__ cov2d,
                                                  const int num_tx,
                                                  const int num_rx, const int N,
                                                  T* __restrict__ influences) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const T a = cov2d[i * 4 + 0];
    const T b = cov2d[i * 4 + 1];
    const T c = cov2d[i * 4 + 2];
    const T d = cov2d[i * 4 + 3];

    const T det = a * d - b * c;
    const T inv_det = T(1.0) / det;

    const T inv_a = d * inv_det;
    const T inv_b = -b * inv_det;
    const T inv_c = -c * inv_det;
    const T inv_d = a * inv_det;

    const T u_i = uv[i * 2 + 0];
    const T v_i = uv[i * 2 + 1];

    for (int tx = 0; tx < num_tx; tx++) {
        for (int rx = 0; rx < num_rx; rx++) {
            const T antenna_u = tx + T(0.5);
            const T antenna_v = rx + T(0.5);

            const T du = u_i - antenna_u;
            const T dv = v_i - antenna_v;

            const T md =
                du * (inv_a * du + inv_b * dv) + dv * (inv_c * du + inv_d * dv);

            influences[i * num_tx * num_rx + tx * num_rx + rx] =
                exp(-T(0.5) * md);
        }
    }
}

void compute_gaussian_influence_cuda(torch::Tensor uv, torch::Tensor cov2d,
                                     int num_tx, int num_rx,
                                     torch::Tensor influences) {
    CHECK_VALID_INPUT(uv);
    CHECK_VALID_INPUT(cov2d);
    CHECK_VALID_INPUT(influences);

    const int N = uv.size(0);
    TORCH_CHECK(uv.size(1) == 2, "uv must have shape Nx2");
    TORCH_CHECK(cov2d.size(0) == N && cov2d.size(1) == 2 && cov2d.size(2) == 2,
                "cov2d must have shape Nx2x2");
    TORCH_CHECK(influences.size(0) == N && influences.size(1) == num_tx &&
                    influences.size(2) == num_rx,
                "influences must have shape Nx" + std::to_string(num_tx) + "x" +
                    std::to_string(num_rx));

    const int max_threads_per_block = 1024;
    const int num_blocks =
        (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    if (uv.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(cov2d);
        CHECK_FLOAT_TENSOR(influences);
        compute_gaussian_influence_kernel<float><<<gridsize, blocksize>>>(
            uv.data_ptr<float>(), cov2d.data_ptr<float>(), num_tx, num_rx, N,
            influences.data_ptr<float>());
    } else if (uv.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(cov2d);
        CHECK_DOUBLE_TENSOR(influences);
        compute_gaussian_influence_kernel<double><<<gridsize, blocksize>>>(
            uv.data_ptr<double>(), cov2d.data_ptr<double>(), num_tx, num_rx, N,
            influences.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", uv.dtype());
    }
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void compute_wireless_channel_kernel(
    const T* __restrict__ attenuation, const T* __restrict__ phase_rotation,
    const T* __restrict__ distances, const T wavelength, const int N,
    T* __restrict__ real_contributions, T* __restrict__ imag_contributions) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const T PI = T(3.14159265358979323846);

    const T r = distances[i];
    const T path_loss = wavelength / (T(4.0) * PI * r);
    const T phase_shift = -T(2.0) * PI * r / wavelength;

    const T A = attenuation[i];
    const T psi = phase_rotation[i];

    const T total_phase = psi + phase_shift;
    const T total_attenuation = A * path_loss;

    real_contributions[i] = total_attenuation * cos(total_phase);
    imag_contributions[i] = total_attenuation * sin(total_phase);
}

void compute_wireless_channel_cuda(torch::Tensor attenuation,
                                   torch::Tensor phase_rotation,
                                   torch::Tensor distances, float wavelength,
                                   torch::Tensor real_contributions,
                                   torch::Tensor imag_contributions) {
    CHECK_VALID_INPUT(attenuation);
    CHECK_VALID_INPUT(phase_rotation);
    CHECK_VALID_INPUT(distances);
    CHECK_VALID_INPUT(real_contributions);
    CHECK_VALID_INPUT(imag_contributions);

    const int N = attenuation.size(0);
    TORCH_CHECK(phase_rotation.size(0) == N,
                "phase_rotation must have shape N");
    TORCH_CHECK(distances.size(0) == N, "distances must have shape N");
    TORCH_CHECK(real_contributions.size(0) == N,
                "real_contributions must have shape N");
    TORCH_CHECK(imag_contributions.size(0) == N,
                "imag_contributions must have shape N");

    const int max_threads_per_block = 1024;
    const int num_blocks =
        (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    if (attenuation.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(phase_rotation);
        CHECK_FLOAT_TENSOR(distances);
        CHECK_FLOAT_TENSOR(real_contributions);
        CHECK_FLOAT_TENSOR(imag_contributions);
        compute_wireless_channel_kernel<float><<<gridsize, blocksize>>>(
            attenuation.data_ptr<float>(), phase_rotation.data_ptr<float>(),
            distances.data_ptr<float>(), wavelength, N,
            real_contributions.data_ptr<float>(),
            imag_contributions.data_ptr<float>());
    } else if (attenuation.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(phase_rotation);
        CHECK_DOUBLE_TENSOR(distances);
        CHECK_DOUBLE_TENSOR(real_contributions);
        CHECK_DOUBLE_TENSOR(imag_contributions);
        compute_wireless_channel_kernel<double><<<gridsize, blocksize>>>(
            attenuation.data_ptr<double>(), phase_rotation.data_ptr<double>(),
            distances.data_ptr<double>(), wavelength, N,
            real_contributions.data_ptr<double>(),
            imag_contributions.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", attenuation.dtype());
    }
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void alpha_blending_forward_kernel(
    const T* __restrict__ influences, const T* __restrict__ real_contributions,
    const T* __restrict__ imag_contributions, const T* __restrict__ opacity,
    const int* __restrict__ sort_indices, const int num_tx, const int num_rx,
    const int N, T* __restrict__ channel_matrix,
    T* __restrict__ eff_opacity_out, T* __restrict__ transmittance_out) {
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int rx = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= num_tx || rx >= num_rx) {
        return;
    }

    T real_sum = T(0.0);
    T imag_sum = T(0.0);
    T current_transmittance = T(1.0);

    transmittance_out[tx * num_rx + rx] = current_transmittance;

    for (int p = 0; p < N; p++) {
        const int idx = sort_indices[p];
        const int influence_idx = idx * num_tx * num_rx + tx * num_rx + rx;
        const int eff_opacity_idx = p * num_tx * num_rx + tx * num_rx + rx;
        const int transmittance_idx_curr =
            p * num_tx * num_rx + tx * num_rx + rx;
        const int transmittance_idx_next =
            (p + 1) * num_tx * num_rx + tx * num_rx + rx;

        const T eff_op = opacity[idx] * influences[influence_idx];
        eff_opacity_out[eff_opacity_idx] = eff_op;

        real_sum += current_transmittance * eff_op * real_contributions[idx];
        imag_sum += current_transmittance * eff_op * imag_contributions[idx];

        current_transmittance *= (T(1.0) - eff_op);
        transmittance_out[transmittance_idx_next] = current_transmittance;
    }

    channel_matrix[tx * (2 * num_rx) + rx] = real_sum;
    channel_matrix[tx * (2 * num_rx) + num_rx + rx] = imag_sum;
}

void alpha_blending_forward_cuda(torch::Tensor influences,
                                 torch::Tensor real_contributions,
                                 torch::Tensor imag_contributions,
                                 torch::Tensor opacity,
                                 torch::Tensor sort_indices, int num_tx,
                                 int num_rx, torch::Tensor channel_matrix,
                                 torch::Tensor eff_opacity_out,
                                 torch::Tensor transmittance_out) {
    CHECK_VALID_INPUT(influences);
    CHECK_VALID_INPUT(real_contributions);
    CHECK_VALID_INPUT(imag_contributions);
    CHECK_VALID_INPUT(opacity);
    CHECK_VALID_INPUT(sort_indices);
    CHECK_VALID_INPUT(channel_matrix);
    CHECK_VALID_INPUT(eff_opacity_out);
    CHECK_VALID_INPUT(transmittance_out);

    const int N = influences.size(0);
    TORCH_CHECK(influences.size(1) == num_tx && influences.size(2) == num_rx,
                "influences shape mismatch");
    TORCH_CHECK(real_contributions.size(0) == N,
                "real_contributions shape mismatch");
    TORCH_CHECK(imag_contributions.size(0) == N,
                "imag_contributions shape mismatch");
    TORCH_CHECK(opacity.size(0) == N, "opacity shape mismatch");
    TORCH_CHECK(sort_indices.size(0) == N, "sort_indices shape mismatch");
    TORCH_CHECK(channel_matrix.size(0) == num_tx &&
                    channel_matrix.size(1) == 2 * num_rx,
                "channel_matrix shape mismatch");
    TORCH_CHECK(eff_opacity_out.size(0) == N &&
                    eff_opacity_out.size(1) == num_tx &&
                    eff_opacity_out.size(2) == num_rx,
                "eff_opacity_out shape mismatch");
    TORCH_CHECK(transmittance_out.size(0) == N + 1 &&
                    transmittance_out.size(1) == num_tx &&
                    transmittance_out.size(2) == num_rx,
                "transmittance_out shape mismatch");

    dim3 blocksize(16, 16, 1);
    dim3 gridsize((num_tx + blocksize.x - 1) / blocksize.x,
                  (num_rx + blocksize.y - 1) / blocksize.y, 1);

    if (influences.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(real_contributions);
        CHECK_FLOAT_TENSOR(imag_contributions);
        CHECK_FLOAT_TENSOR(opacity);
        CHECK_INT_TENSOR(sort_indices);
        CHECK_FLOAT_TENSOR(channel_matrix);
        CHECK_FLOAT_TENSOR(eff_opacity_out);
        CHECK_FLOAT_TENSOR(transmittance_out);
        alpha_blending_forward_kernel<float><<<gridsize, blocksize>>>(
            influences.data_ptr<float>(), real_contributions.data_ptr<float>(),
            imag_contributions.data_ptr<float>(), opacity.data_ptr<float>(),
            sort_indices.data_ptr<int>(), num_tx, num_rx, N,
            channel_matrix.data_ptr<float>(), eff_opacity_out.data_ptr<float>(),
            transmittance_out.data_ptr<float>());
    } else if (influences.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(real_contributions);
        CHECK_DOUBLE_TENSOR(imag_contributions);
        CHECK_DOUBLE_TENSOR(opacity);
        CHECK_INT_TENSOR(sort_indices);
        CHECK_DOUBLE_TENSOR(channel_matrix);
        CHECK_DOUBLE_TENSOR(eff_opacity_out);
        CHECK_DOUBLE_TENSOR(transmittance_out);
        alpha_blending_forward_kernel<double><<<gridsize, blocksize>>>(
            influences.data_ptr<double>(),
            real_contributions.data_ptr<double>(),
            imag_contributions.data_ptr<double>(), opacity.data_ptr<double>(),
            sort_indices.data_ptr<int>(), num_tx, num_rx, N,
            channel_matrix.data_ptr<double>(),
            eff_opacity_out.data_ptr<double>(),
            transmittance_out.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", influences.dtype());
    }
    cudaDeviceSynchronize();
}
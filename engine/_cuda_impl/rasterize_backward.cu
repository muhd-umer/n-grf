// engine/_cuda_impl/rasterize_backward.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>

#include "checks.cuh"
#include "matrix.cuh"

template <typename T>
__global__ void compute_gaussian_influence_backward_kernel(
    const T* __restrict__ uv, const T* __restrict__ cov2d,
    const T* __restrict__ influences,  // precomputed influence values
    const T* __restrict__ grad_influences, const int num_tx, const int num_rx,
    const int N, T* __restrict__ grad_uv, T* __restrict__ grad_cov2d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    // Extract covariance matrix and compute inverse
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

    // Initialize gradients for this Gaussian
    T grad_u_sum = T(0.0);
    T grad_v_sum = T(0.0);
    T grad_a_sum = T(0.0);
    T grad_b_sum = T(0.0);
    T grad_c_sum = T(0.0);
    T grad_d_sum = T(0.0);

    // For each antenna location, compute gradient contribution
    for (int tx = 0; tx < num_tx; tx++) {
        for (int rx = 0; rx < num_rx; rx++) {
            const int influence_idx = i * num_tx * num_rx + tx * num_rx + rx;
            const T grad_influence = grad_influences[influence_idx];

            // Only compute gradient if there's a non-zero gradient
            if (abs(grad_influence) > T(1e-10)) {
                const T antenna_u = tx + T(0.5);
                const T antenna_v = rx + T(0.5);

                const T du = u_i - antenna_u;
                const T dv = v_i - antenna_v;

                // Influence value already computed
                const T influence = influences[influence_idx];

                // Gradient of influence with respect to Mahalanobis distance
                const T grad_md = -T(0.5) * influence * grad_influence;

                // Gradient of Mahalanobis distance with respect to uv
                // coordinates dmd/du = inv_a * du + inv_b * dv dmd/dv = inv_c *
                // du + inv_d * dv
                grad_u_sum += grad_md * (inv_a * du + inv_b * dv);
                grad_v_sum += grad_md * (inv_c * du + inv_d * dv);

                // Outer product of displacement vector
                const T du_du = du * du;
                const T du_dv = du * dv;
                const T dv_dv = dv * dv;

                // Gradient of Mahalanobis distance with respect to inverse
                // covariance dmd/dinv_a = du * du dmd/dinv_b = du * dv
                // dmd/dinv_c = du * dv
                // dmd/dinv_d = dv * dv
                const T grad_inv_a = grad_md * du_du;
                const T grad_inv_b = grad_md * du_dv;
                const T grad_inv_c = grad_md * du_dv;
                const T grad_inv_d = grad_md * dv_dv;

                // Chain rule for covariance matrix inversion
                // dinv_a/da = -inv_a * inv_a * d + inv_b * inv_c
                // dinv_b/da = -inv_a * inv_b * d + inv_b * inv_d
                // dinv_c/da = -inv_a * inv_c * d + inv_c * inv_d
                // dinv_d/da = -inv_a * inv_d * d + inv_d * inv_d

                // Compute gradients with respect to a
                grad_a_sum +=
                    grad_inv_a * (-inv_a * inv_a * d + inv_b * inv_c) +
                    grad_inv_b * (-inv_a * inv_b * d + inv_b * inv_d) +
                    grad_inv_c * (-inv_a * inv_c * d + inv_c * inv_d) +
                    grad_inv_d * (-inv_a * inv_d * d + inv_d * inv_d);

                // Compute gradients with respect to b
                grad_b_sum += grad_inv_a * (inv_a * inv_a * c) +
                              grad_inv_b * (inv_a * inv_b * c - inv_a * inv_d) +
                              grad_inv_c * (inv_a * inv_c * c - inv_c * inv_c) +
                              grad_inv_d * (inv_a * inv_d * c - inv_c * inv_d);

                // Compute gradients with respect to c
                grad_c_sum += grad_inv_a * (inv_a * inv_a * b) +
                              grad_inv_b * (inv_a * inv_b * b - inv_b * inv_b) +
                              grad_inv_c * (inv_a * inv_c * b - inv_b * inv_c) +
                              grad_inv_d * (inv_a * inv_d * b - inv_b * inv_d);

                // Compute gradients with respect to d
                grad_d_sum +=
                    grad_inv_a * (-inv_a * inv_a * a + inv_a * inv_a) +
                    grad_inv_b * (-inv_a * inv_b * a + inv_a * inv_b) +
                    grad_inv_c * (-inv_a * inv_c * a + inv_a * inv_c) +
                    grad_inv_d * (-inv_a * inv_d * a + inv_a * inv_d);
            }
        }
    }

    // Update gradients
    grad_uv[i * 2 + 0] = grad_u_sum;
    grad_uv[i * 2 + 1] = grad_v_sum;

    grad_cov2d[i * 4 + 0] = grad_a_sum;
    grad_cov2d[i * 4 + 1] = grad_b_sum;
    grad_cov2d[i * 4 + 2] = grad_c_sum;
    grad_cov2d[i * 4 + 3] = grad_d_sum;
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
    const T path_loss = wavelength / (T(4.0) * PI * r);
    const T phase_shift = -T(2.0) * PI * r / wavelength;

    const T A = attenuation[i];
    const T psi = phase_rotation[i];

    const T total_phase = psi + phase_shift;
    const T total_attenuation = A * path_loss;

    const T cos_phase = cos(total_phase);
    const T sin_phase = sin(total_phase);

    const T grad_real = grad_real_contributions[i];
    const T grad_imag = grad_imag_contributions[i];

    // Gradient with respect to attenuation
    grad_attenuation[i] =
        grad_real * path_loss * cos_phase + grad_imag * path_loss * sin_phase;

    // Gradient with respect to phase rotation
    grad_phase_rotation[i] = -grad_real * total_attenuation * sin_phase +
                             grad_imag * total_attenuation * cos_phase;

    // Gradient with respect to distance
    const T grad_path_loss = -wavelength / (T(4.0) * PI * r * r);
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
    TORCH_CHECK(phase_rotation.size(0) == N,
                "phase_rotation must have shape N");
    TORCH_CHECK(distances.size(0) == N, "distances must have shape N");
    TORCH_CHECK(grad_real_contributions.size(0) == N,
                "grad_real_contributions must have shape N");
    TORCH_CHECK(grad_imag_contributions.size(0) == N,
                "grad_imag_contributions must have shape N");
    TORCH_CHECK(grad_attenuation.size(0) == N,
                "grad_attenuation must have shape N");
    TORCH_CHECK(grad_phase_rotation.size(0) == N,
                "grad_phase_rotation must have shape N");
    TORCH_CHECK(grad_distances.size(0) == N,
                "grad_distances must have shape N");

    const int max_threads_per_block = 1024;
    const int num_blocks =
        (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    if (attenuation.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(phase_rotation);
        CHECK_FLOAT_TENSOR(distances);
        CHECK_FLOAT_TENSOR(grad_real_contributions);
        CHECK_FLOAT_TENSOR(grad_imag_contributions);
        CHECK_FLOAT_TENSOR(grad_attenuation);
        CHECK_FLOAT_TENSOR(grad_phase_rotation);
        CHECK_FLOAT_TENSOR(grad_distances);
        compute_wireless_channel_backward_kernel<float>
            <<<gridsize, blocksize>>>(attenuation.data_ptr<float>(),
                                      phase_rotation.data_ptr<float>(),
                                      distances.data_ptr<float>(), wavelength,
                                      grad_real_contributions.data_ptr<float>(),
                                      grad_imag_contributions.data_ptr<float>(),
                                      N, grad_attenuation.data_ptr<float>(),
                                      grad_phase_rotation.data_ptr<float>(),
                                      grad_distances.data_ptr<float>());
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
                attenuation.data_ptr<double>(),
                phase_rotation.data_ptr<double>(), distances.data_ptr<double>(),
                wavelength, grad_real_contributions.data_ptr<double>(),
                grad_imag_contributions.data_ptr<double>(), N,
                grad_attenuation.data_ptr<double>(),
                grad_phase_rotation.data_ptr<double>(),
                grad_distances.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", attenuation.dtype());
    }
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void alpha_blending_backward_kernel(
    const T* __restrict__ influences, const T* __restrict__ real_contributions,
    const T* __restrict__ imag_contributions, const T* __restrict__ opacity,
    const int* __restrict__ sort_indices,
    const T* __restrict__ grad_channel_matrix, const int num_tx,
    const int num_rx, const int N, T* __restrict__ grad_influences,
    T* __restrict__ grad_real_contributions,
    T* __restrict__ grad_imag_contributions, T* __restrict__ grad_opacity) {
    // One thread per antenna pair
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int rx = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= num_tx || rx >= num_rx) {
        return;
    }

    // Extract gradients from channel matrix
    const T grad_real = grad_channel_matrix[tx * (2 * num_rx) + rx];
    const T grad_imag = grad_channel_matrix[tx * (2 * num_rx) + num_rx + rx];

    // Precompute effective opacities, transmittances, and accumulated colors
    T eff_opacity[128];    // Fixed array size, should be large enough for most
                           // cases
    T transmittance[129];  // One more for initialization
    T real_accum[128];
    T imag_accum[128];

    transmittance[0] = T(1.0);

    // Forward pass to compute transmittance and accumulated colors
    for (int p = 0; p < N; p++) {
        const int idx_pos = p;
        const int idx = sort_indices[idx_pos];

        eff_opacity[p] =
            opacity[idx] * influences[idx * num_tx * num_rx + tx * num_rx + rx];

        real_accum[p] =
            eff_opacity[p] * real_contributions[idx] * transmittance[p];
        imag_accum[p] =
            eff_opacity[p] * imag_contributions[idx] * transmittance[p];

        transmittance[p + 1] = transmittance[p] * (T(1.0) - eff_opacity[p]);
    }

    // Backward pass to compute gradients
    T dLdT_next =
        T(0.0);  // Gradient of loss with respect to next transmittance

    for (int p = N - 1; p >= 0; p--) {
        const int idx_pos = p;
        const int idx = sort_indices[idx_pos];

        // Gradient with respect to real and imag contributions
        T grad_real_contrib = eff_opacity[p] * transmittance[p] * grad_real;
        T grad_imag_contrib = eff_opacity[p] * transmittance[p] * grad_imag;

        // Atomic add to handle multiple threads updating the same contribution
        atomicAdd(&grad_real_contributions[idx], grad_real_contrib);
        atomicAdd(&grad_imag_contributions[idx], grad_imag_contrib);

        // Direct effect of effective opacity on output
        T grad_eff_opacity_direct =
            transmittance[p] * (real_contributions[idx] * grad_real +
                                imag_contributions[idx] * grad_imag);

        // Indirect effect through transmittance for subsequent Gaussians
        T grad_eff_opacity_indirect = -transmittance[p] * dLdT_next;

        // Total gradient with respect to effective opacity
        T grad_eff_opacity =
            grad_eff_opacity_direct + grad_eff_opacity_indirect;

        // Split gradient between opacity and influence
        const T influence =
            influences[idx * num_tx * num_rx + tx * num_rx + rx];
        atomicAdd(&grad_opacity[idx], grad_eff_opacity * influence);
        atomicAdd(&grad_influences[idx * num_tx * num_rx + tx * num_rx + rx],
                  grad_eff_opacity * opacity[idx]);

        // Compute gradient with respect to transmittance for previous Gaussian
        T dLdT_curr = grad_real * real_accum[p] + grad_imag * imag_accum[p] +
                      dLdT_next * (T(1.0) - eff_opacity[p]);

        dLdT_next = dLdT_curr;
    }
}

void alpha_blending_backward_cuda(
    torch::Tensor influences, torch::Tensor real_contributions,
    torch::Tensor imag_contributions, torch::Tensor opacity,
    torch::Tensor sort_indices, torch::Tensor grad_channel_matrix, int num_tx,
    int num_rx, torch::Tensor grad_influences,
    torch::Tensor grad_real_contributions,
    torch::Tensor grad_imag_contributions, torch::Tensor grad_opacity) {
    CHECK_VALID_INPUT(influences);
    CHECK_VALID_INPUT(real_contributions);
    CHECK_VALID_INPUT(imag_contributions);
    CHECK_VALID_INPUT(opacity);
    CHECK_VALID_INPUT(sort_indices);
    CHECK_VALID_INPUT(grad_channel_matrix);
    CHECK_VALID_INPUT(grad_influences);
    CHECK_VALID_INPUT(grad_real_contributions);
    CHECK_VALID_INPUT(grad_imag_contributions);
    CHECK_VALID_INPUT(grad_opacity);

    const int N = influences.size(0);
    TORCH_CHECK(
        N <= 128,
        "Current implementation supports at most 128 Gaussians");  // Due to
                                                                   // fixed
                                                                   // array size
                                                                   // in kernel
    TORCH_CHECK(influences.size(1) == num_tx && influences.size(2) == num_rx,
                "influences must have shape Nx" + std::to_string(num_tx) + "x" +
                    std::to_string(num_rx));
    TORCH_CHECK(real_contributions.size(0) == N,
                "real_contributions must have shape N");
    TORCH_CHECK(imag_contributions.size(0) == N,
                "imag_contributions must have shape N");
    TORCH_CHECK(opacity.size(0) == N, "opacity must have shape N");
    TORCH_CHECK(sort_indices.size(0) == N, "sort_indices must have shape N");
    TORCH_CHECK(grad_channel_matrix.size(0) == num_tx &&
                    grad_channel_matrix.size(1) == 2 * num_rx,
                "grad_channel_matrix must have shape " +
                    std::to_string(num_tx) + "x" + std::to_string(2 * num_rx));
    TORCH_CHECK(grad_influences.size(0) == N &&
                    grad_influences.size(1) == num_tx &&
                    grad_influences.size(2) == num_rx,
                "grad_influences must have shape Nx" + std::to_string(num_tx) +
                    "x" + std::to_string(num_rx));
    TORCH_CHECK(grad_real_contributions.size(0) == N,
                "grad_real_contributions must have shape N");
    TORCH_CHECK(grad_imag_contributions.size(0) == N,
                "grad_imag_contributions must have shape N");
    TORCH_CHECK(grad_opacity.size(0) == N, "grad_opacity must have shape N");

    // Use 2D grid to parallelize over the channel matrix elements
    dim3 blocksize(16, 16, 1);
    dim3 gridsize((num_tx + blocksize.x - 1) / blocksize.x,
                  (num_rx + blocksize.y - 1) / blocksize.y, 1);

    if (influences.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(real_contributions);
        CHECK_FLOAT_TENSOR(imag_contributions);
        CHECK_FLOAT_TENSOR(opacity);
        CHECK_INT_TENSOR(sort_indices);
        CHECK_FLOAT_TENSOR(grad_channel_matrix);
        CHECK_FLOAT_TENSOR(grad_influences);
        CHECK_FLOAT_TENSOR(grad_real_contributions);
        CHECK_FLOAT_TENSOR(grad_imag_contributions);
        CHECK_FLOAT_TENSOR(grad_opacity);
        alpha_blending_backward_kernel<float><<<gridsize, blocksize>>>(
            influences.data_ptr<float>(), real_contributions.data_ptr<float>(),
            imag_contributions.data_ptr<float>(), opacity.data_ptr<float>(),
            sort_indices.data_ptr<int>(), grad_channel_matrix.data_ptr<float>(),
            num_tx, num_rx, N, grad_influences.data_ptr<float>(),
            grad_real_contributions.data_ptr<float>(),
            grad_imag_contributions.data_ptr<float>(),
            grad_opacity.data_ptr<float>());
    } else if (influences.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(real_contributions);
        CHECK_DOUBLE_TENSOR(imag_contributions);
        CHECK_DOUBLE_TENSOR(opacity);
        CHECK_INT_TENSOR(sort_indices);
        CHECK_DOUBLE_TENSOR(grad_channel_matrix);
        CHECK_DOUBLE_TENSOR(grad_influences);
        CHECK_DOUBLE_TENSOR(grad_real_contributions);
        CHECK_DOUBLE_TENSOR(grad_imag_contributions);
        CHECK_DOUBLE_TENSOR(grad_opacity);
        alpha_blending_backward_kernel<double><<<gridsize, blocksize>>>(
            influences.data_ptr<double>(),
            real_contributions.data_ptr<double>(),
            imag_contributions.data_ptr<double>(), opacity.data_ptr<double>(),
            sort_indices.data_ptr<int>(),
            grad_channel_matrix.data_ptr<double>(), num_tx, num_rx, N,
            grad_influences.data_ptr<double>(),
            grad_real_contributions.data_ptr<double>(),
            grad_imag_contributions.data_ptr<double>(),
            grad_opacity.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", influences.dtype());
    }
    cudaDeviceSynchronize();
}
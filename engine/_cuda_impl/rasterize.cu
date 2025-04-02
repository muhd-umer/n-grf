/*
 * Implementation file for CUDA-PyTorch bindings
 */

#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "auxiliary.h"
#include "forward.h"
#include "rasterize.h"

// CUDA kernel for computing Gaussian influence
__global__ void compute_gaussian_influence_kernel(const float* uv,
                                                  const float* cov2d,
                                                  float* influences, int num_tx,
                                                  int num_rx, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Extract 2D covariance matrix for this Gaussian
    float cov[4] = {cov2d[idx * 4 + 0], cov2d[idx * 4 + 1], cov2d[idx * 4 + 2],
                    cov2d[idx * 4 + 3]};

    // Invert covariance matrix
    float inv_cov[4];
    float det;
    bool success = invert_2x2(cov, inv_cov, &det);

    if (!success) {
        // Handle singular matrix
        for (int i = 0; i < num_tx; i++) {
            for (int j = 0; j < num_rx; j++) {
                influences[idx * num_tx * num_rx + i * num_rx + j] = 0.0f;
            }
        }
        return;
    }

    // Get Gaussian position
    float u = uv[idx * 2];
    float v = uv[idx * 2 + 1];

    // Compute influence on each channel element
    for (int i = 0; i < num_tx; i++) {
        float antenna_pos_x = i + 0.5f;

        for (int j = 0; j < num_rx; j++) {
            float antenna_pos_y = j + 0.5f;

            float d_x = u - antenna_pos_x;
            float d_y = v - antenna_pos_y;

            float m_dist = mahalanobis_distance(inv_cov, d_x, d_y);

            influences[idx * num_tx * num_rx + i * num_rx + j] =
                expf(-0.5f * m_dist);
        }
    }
}

// CUDA kernel for computing channel contribution
__global__ void compute_channel_kernel(const float* attenuation,
                                       const float* phase_rotation,
                                       const float* xyz_rx_distance,
                                       float* real_part, float* imag_part,
                                       float wavelength, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float distance = xyz_rx_distance[idx];
    float att = attenuation[idx];
    float phase = phase_rotation[idx];

    float path_loss = wavelength / (4.0f * PI * distance);
    float phase_shift = -2.0f * PI * distance / wavelength;

    float total_att = att * path_loss;
    float total_phase = phase + phase_shift;

    real_part[idx] = total_att * cosf(total_phase);
    imag_part[idx] = total_att * sinf(total_phase);
}

// CUDA kernel for alpha blending
__global__ void alpha_blending_kernel(const float* influences,
                                      const float* real_contributions,
                                      const float* imag_contributions,
                                      const float* opacity,
                                      const int64_t* sort_indices,
                                      float* channel_real, float* channel_imag,
                                      int num_tx, int num_rx, int N) {
    // One thread per channel element
    int i = blockIdx.x;   // tx index
    int j = threadIdx.x;  // rx index

    if (i >= num_tx || j >= num_rx) return;

    float transmittance = 1.0f;
    float result_real = 0.0f;
    float result_imag = 0.0f;

    for (int k = 0; k < N; k++) {
        int idx = sort_indices[k];
        float effective_opacity =
            opacity[idx] * influences[idx * num_tx * num_rx + i * num_rx + j];

        result_real +=
            transmittance * effective_opacity * real_contributions[idx];
        result_imag +=
            transmittance * effective_opacity * imag_contributions[idx];

        transmittance *= (1.0f - effective_opacity);

        // Early termination if transmittance is near zero
        if (transmittance < 1e-5f) break;
    }

    channel_real[i * num_rx + j] = result_real;
    channel_imag[i * num_rx + j] = result_imag;
}

// Wrapper function for computing Gaussian influence
torch::Tensor computeGaussianInfluenceCUDA(const torch::Tensor& uv,
                                           const torch::Tensor& cov2d,
                                           int num_tx, int num_rx) {
    const at::cuda::CUDAGuard device_guard(uv.device());
    int N = uv.size(0);

    auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(uv.device());

    torch::Tensor influences = torch::empty({N, num_tx, num_rx}, options);

    int threads = 32;
    int blocks = (N + threads - 1) / threads;

    compute_gaussian_influence_kernel<<<blocks, threads>>>(
        uv.data_ptr<float>(), cov2d.data_ptr<float>(),
        influences.data_ptr<float>(), num_tx, num_rx, N);

    return influences;
}

// Wrapper function for computing channel contribution
std::tuple<torch::Tensor, torch::Tensor> computeChannelCUDA(
    const torch::Tensor& attenuation, const torch::Tensor& phase_rotation,
    const torch::Tensor& xyz_rx_distance, float wavelength) {
    const at::cuda::CUDAGuard device_guard(attenuation.device());
    int N = attenuation.size(0);

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(attenuation.device());

    torch::Tensor real_part = torch::empty({N, 1}, options);
    torch::Tensor imag_part = torch::empty({N, 1}, options);

    torch::Tensor att_contig = attenuation.contiguous();
    torch::Tensor phase_contig = phase_rotation.contiguous();

    int threads = 32;
    int blocks = (N + threads - 1) / threads;

    compute_channel_kernel<<<blocks, threads>>>(
        att_contig.data_ptr<float>(), phase_contig.data_ptr<float>(),
        xyz_rx_distance.data_ptr<float>(), real_part.data_ptr<float>(),
        imag_part.data_ptr<float>(), wavelength, N);

    return std::make_tuple(real_part, imag_part);
}

// Wrapper function for alpha blending
torch::Tensor alphaBlendingCUDA(const torch::Tensor& influences,
                                const torch::Tensor& contributions,
                                const torch::Tensor& opacity,
                                const torch::Tensor& sort_indices, int num_tx,
                                int num_rx) {
    const at::cuda::CUDAGuard device_guard(influences.device());
    int N = influences.size(0);

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(influences.device());

    torch::Tensor channel_real = torch::zeros({num_tx, num_rx}, options);
    torch::Tensor channel_imag = torch::zeros({num_tx, num_rx}, options);

    // Extract real and imaginary parts
    torch::Tensor real_contributions = torch::real(contributions).contiguous();
    torch::Tensor imag_contributions = torch::imag(contributions).contiguous();

    // Configure kernel launch
    int threads = num_rx;  // One thread per rx
    int blocks = num_tx;   // One block per tx

    // Make sure we're not launching too many threads
    if (threads > 1024) {
        threads = 32;
        blocks = (num_tx * num_rx + threads - 1) / threads;
    }

    alpha_blending_kernel<<<blocks, threads>>>(
        influences.data_ptr<float>(), real_contributions.data_ptr<float>(),
        imag_contributions.data_ptr<float>(), opacity.data_ptr<float>(),
        sort_indices.data_ptr<int64_t>(), channel_real.data_ptr<float>(),
        channel_imag.data_ptr<float>(), num_tx, num_rx, N);

    // Concatenate real and imaginary parts
    torch::Tensor cat_channel = torch::cat({channel_real, channel_imag}, 1);

    return cat_channel;
}

// Main function for rasterizing channel matrix
std::vector<torch::Tensor> rasterizeForwardCUDA(
    const torch::Tensor& points, const torch::Tensor& scaling,
    const torch::Tensor& rotation, const torch::Tensor& attenuation,
    const torch::Tensor& phase_rotation, const torch::Tensor& opacity,
    const torch::Tensor& receiver, const torch::Tensor& transmitter, int num_tx,
    int num_rx, float frequency, float scale_modifier) {
    const at::cuda::CUDAGuard device_guard(points.device());

    // Constants
    float c = 299792458.0f;
    float wavelength = c / frequency;

    // Compute covariance matrix from scaling and rotation
    torch::Tensor cov3d =
        computeCov3dFromScalingRotationCUDA(scaling, rotation, scale_modifier);

    // Step 1: Project to channel space
    auto [distances, uv, cov2d] =
        projectToChannelSpaceCUDA(points, cov3d, receiver, num_tx, num_rx);

    // Step 2: Sort points by distance
    auto sort_result = distances.sort(0);
    torch::Tensor sort_indices = std::get<1>(sort_result);

    // Step 3: Compute Gaussian influence
    torch::Tensor influences =
        computeGaussianInfluenceCUDA(uv, cov2d, num_tx, num_rx);

    // Step 4: Compute channel contribution
    auto [real_contributions, imag_contributions] =
        computeChannelCUDA(attenuation, phase_rotation, distances, wavelength);

    // Convert to complex tensor for alpha blending
    torch::Tensor contributions = torch::complex(
        real_contributions.squeeze(-1), imag_contributions.squeeze(-1));

    // Step 5: Perform alpha blending
    torch::Tensor opacity_flat = opacity.squeeze(-1);
    torch::Tensor channel = alphaBlendingCUDA(
        influences, contributions, opacity_flat, sort_indices, num_tx, num_rx);

    return {channel};
}
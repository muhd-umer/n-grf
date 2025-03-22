/*
 * Core header file for channel reconstruction CUDA implementation
 * Contains top-level function declarations exposed to Python
 */

#ifndef ENGINE_CUDA_RASTERIZE_H
#define ENGINE_CUDA_RASTERIZE_H

#include <torch/extension.h>

torch::Tensor computeGaussianInfluenceCUDA(const torch::Tensor &uv,
                                           const torch::Tensor &cov2d,
                                           int num_tx, int num_rx);

std::tuple<torch::Tensor, torch::Tensor> computeChannelCUDA(
    const torch::Tensor &attenuation, const torch::Tensor &phase_rotation,
    const torch::Tensor &xyz_rx_distance, float wavelength);

torch::Tensor alphaBlendingCUDA(const torch::Tensor &influences,
                                const torch::Tensor &contributions,
                                const torch::Tensor &opacity,
                                const torch::Tensor &sort_indices, int num_tx,
                                int num_rx);

std::vector<torch::Tensor> rasterizeForwardCUDA(
    const torch::Tensor &points, const torch::Tensor &cov3d,
    const torch::Tensor &attenuation, const torch::Tensor &phase_rotation,
    const torch::Tensor &opacity, const torch::Tensor &receiver,
    const torch::Tensor &transmitter, int num_tx, int num_rx, float frequency);

#endif  // ENGINE_CUDA_RASTERIZE_H
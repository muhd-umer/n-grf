/*
 * Core header file for channel reconstruction CUDA implementation
 * Contains top-level function declarations exposed to Python
 */

#ifndef ENGINE_CUDA_RASTERIZE_H
#define ENGINE_CUDA_RASTERIZE_H

#include <torch/extension.h>

std::vector<torch::Tensor> rasterizeForwardCUDA(
    const torch::Tensor &points, const torch::Tensor &scaling,
    const torch::Tensor &rotation, const torch::Tensor &attenuation,
    const torch::Tensor &phase_rotation, const torch::Tensor &opacity,
    const torch::Tensor &receiver, const torch::Tensor &transmitter, int num_tx,
    int num_rx, float frequency, float scale_modifier);

#endif  // ENGINE_CUDA_RASTERIZE_H
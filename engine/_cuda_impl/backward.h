/*
 * Header file for backward pass CUDA operations in wireless channel
 * reconstruction
 */

#ifndef ENGINE_CUDA_BACKWARD_H_INCLUDED
#define ENGINE_CUDA_BACKWARD_H_INCLUDED

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
rasterizeBackwardCUDA(
    const torch::Tensor& grad_output, const torch::Tensor& points,
    const torch::Tensor& scaling, const torch::Tensor& rotation,
    const torch::Tensor& attenuation, const torch::Tensor& phase_rotation,
    const torch::Tensor& opacity, const torch::Tensor& receiver,
    const torch::Tensor& transmitter, const int num_tx, const int num_rx,
    const float frequency, const float scale_modifier);

#endif  // ENGINE_CUDA_BACKWARD_H_INCLUDED
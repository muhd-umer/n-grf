/*
 * Forward pass header file for channel reconstruction CUDA kernels
 * Contains function declarations for the forward pass operations
 */

#ifndef ENGINE_CUDA_FORWARD_H
#define ENGINE_CUDA_FORWARD_H

#include <torch/extension.h>

// Transform functions
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
computeSphericalCoordsCUDA(const torch::Tensor &points,
                           const torch::Tensor &receiver);

torch::Tensor computeDistancesToReceiverCUDA(const torch::Tensor &points,
                                             const torch::Tensor &receiver);

std::tuple<torch::Tensor, torch::Tensor> transformToUniformCoordsCUDA(
    const torch::Tensor &longitude, const torch::Tensor &latitude);

torch::Tensor computeCov3dFromScalingRotationCUDA(const torch::Tensor &scaling,
                                                  const torch::Tensor &rotation,
                                                  float scale_modifier);

torch::Tensor mapToChannelMatrixCUDA(const torch::Tensor &s_x,
                                     const torch::Tensor &s_y, int num_tx,
                                     int num_rx);

torch::Tensor computeJacobianCUDA(const torch::Tensor &d,
                                  const torch::Tensor &r, int num_tx,
                                  int num_rx);

torch::Tensor projectCov3dToCov2dCUDA(const torch::Tensor &cov3d_mat,
                                      const torch::Tensor &jacobian);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
projectToChannelSpaceCUDA(const torch::Tensor &points,
                          const torch::Tensor &cov3d,
                          const torch::Tensor &receiver, int num_tx,
                          int num_rx);

#endif  // ENGINE_CUDA_FORWARD_H
/*
 * Forward pass header file for channel reconstruction CUDA kernels
 */

#ifndef ENGINE_CUDA_FORWARD_H
#define ENGINE_CUDA_FORWARD_H

#include <torch/extension.h>

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

#endif  // ENGINE_CUDA_FORWARD_H
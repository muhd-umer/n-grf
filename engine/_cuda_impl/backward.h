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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
alphaBlendingBackwardCUDA(const torch::Tensor& grad_output,
                          const torch::Tensor& influences,
                          const torch::Tensor& contributions,
                          const torch::Tensor& opacity,
                          const torch::Tensor& sort_indices, const int num_tx,
                          const int num_rx);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
computeChannelBackwardCUDA(const torch::Tensor& grad_real,
                           const torch::Tensor& grad_imag,
                           const torch::Tensor& attenuation,
                           const torch::Tensor& phase_rotation,
                           const torch::Tensor& xyz_rx_distance,
                           const float wavelength);

std::tuple<torch::Tensor, torch::Tensor> computeGaussianInfluenceBackwardCUDA(
    const torch::Tensor& grad_output, const torch::Tensor& uv,
    const torch::Tensor& cov2d, const int num_tx, const int num_rx);

std::tuple<torch::Tensor, torch::Tensor> projectCov3dToCov2dBackwardCUDA(
    const torch::Tensor& grad_output, const torch::Tensor& cov3d_mat,
    const torch::Tensor& jacobian);

std::tuple<torch::Tensor, torch::Tensor> computeJacobianBackwardCUDA(
    const torch::Tensor& grad_output, const torch::Tensor& d,
    const torch::Tensor& r, const int num_tx, const int num_rx);

std::tuple<torch::Tensor, torch::Tensor> mapToChannelMatrixBackwardCUDA(
    const torch::Tensor& grad_output, const torch::Tensor& s_x,
    const torch::Tensor& s_y, const int num_tx, const int num_rx);

std::tuple<torch::Tensor, torch::Tensor> transformToUniformCoordsBackwardCUDA(
    const torch::Tensor& grad_s_x, const torch::Tensor& grad_s_y,
    const torch::Tensor& longitude, const torch::Tensor& latitude);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
computeSphericalCoordsBackwardCUDA(const torch::Tensor& grad_longitude,
                                   const torch::Tensor& grad_latitude,
                                   const torch::Tensor& points,
                                   const torch::Tensor& receiver);

torch::Tensor computeDistancesToReceiverBackwardCUDA(
    const torch::Tensor& grad_output, const torch::Tensor& points,
    const torch::Tensor& receiver);

std::tuple<torch::Tensor, torch::Tensor>
computeCov3dFromScalingRotationBackwardCUDA(const torch::Tensor& grad_cov3d,
                                            const torch::Tensor& scaling,
                                            const torch::Tensor& rotation,
                                            const float scale_modifier);

#endif  // ENGINE_CUDA_BACKWARD_H_INCLUDED
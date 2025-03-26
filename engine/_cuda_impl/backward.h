/*
 * Backward pass header file for wireless channel reconstruction CUDA kernels
 */

#ifndef ENGINE_CUDA_BACKWARD_H
#define ENGINE_CUDA_BACKWARD_H

#include <torch/extension.h>

torch::Tensor prepareGradOutputCUDA(const torch::Tensor& grad_output,
                                    int num_tx, int num_rx);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
alphaBlendingBackwardCUDA(const torch::Tensor& grad_channel,
                          const torch::Tensor& influences,
                          const torch::Tensor& contributions,
                          const torch::Tensor& opacity,
                          const torch::Tensor& sort_indices, int num_tx,
                          int num_rx);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
computeChannelBackwardCUDA(const torch::Tensor& grad_real_contributions,
                           const torch::Tensor& grad_imag_contributions,
                           const torch::Tensor& attenuation,
                           const torch::Tensor& phase_rotation,
                           const torch::Tensor& xyz_rx_distance,
                           float wavelength);

std::tuple<torch::Tensor, torch::Tensor> computeGaussianInfluenceBackwardCUDA(
    const torch::Tensor& grad_influences, const torch::Tensor& uv,
    const torch::Tensor& cov2d, int num_tx, int num_rx);

std::tuple<torch::Tensor, torch::Tensor> projectCov3dToCov2dBackwardCUDA(
    const torch::Tensor& grad_cov2d, const torch::Tensor& cov3d_mat,
    const torch::Tensor& jacobian);

std::tuple<torch::Tensor, torch::Tensor> computeJacobianBackwardCUDA(
    const torch::Tensor& grad_jacobian, const torch::Tensor& d,
    const torch::Tensor& r, int num_tx, int num_rx);

torch::Tensor mapToChannelMatrixBackwardCUDA(const torch::Tensor& grad_uv,
                                             int num_tx, int num_rx);

torch::Tensor transformToUniformCoordsBackwardCUDA(
    const torch::Tensor& grad_s_x, const torch::Tensor& grad_s_y);

std::tuple<torch::Tensor, torch::Tensor> computeSphericalCoordsBackwardCUDA(
    const torch::Tensor& grad_longitude, const torch::Tensor& grad_latitude,
    const torch::Tensor& grad_r_spherical, const torch::Tensor& d,
    const torch::Tensor& r);

torch::Tensor computeDistancesToReceiverBackwardCUDA(
    const torch::Tensor& grad_r, const torch::Tensor& d,
    const torch::Tensor& r);

std::tuple<torch::Tensor, torch::Tensor> computeCov3dBackwardCUDA(
    const torch::Tensor& grad_cov3d, const torch::Tensor& scaling,
    const torch::Tensor& rotation, float scale_modifier);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
rasterizeBackwardCUDA(const torch::Tensor& grad_output,
                      const torch::Tensor& points, const torch::Tensor& cov3d,
                      const torch::Tensor& attenuation,
                      const torch::Tensor& phase_rotation,
                      const torch::Tensor& opacity,
                      const torch::Tensor& receiver,
                      const torch::Tensor& transmitter, int num_tx, int num_rx,
                      float frequency);

#endif  // ENGINE_CUDA_BACKWARD_H
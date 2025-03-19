/*
 * Core header file for wireless channel reconstruction CUDA implementation
 * Contains top-level function declarations exposed to Python
 */

#ifndef CUDA_WIRELESS_RASTERIZE_H_INCLUDED
#define CUDA_WIRELESS_RASTERIZE_H_INCLUDED

#include <cuda.h>
#include <device_launch_parameters.h>
#include <torch/extension.h>

#include "cuda_runtime.h"
#include "forward.h"

// Transforms and projection functions
torch::Tensor compute_distances_to_receiver_cuda(
    torch::Tensor points,
    torch::Tensor receiver);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute_spherical_coords_cuda(
    torch::Tensor points,
    torch::Tensor receiver);

std::tuple<torch::Tensor, torch::Tensor> transform_to_uniform_coords_cuda(
    torch::Tensor longitude,
    torch::Tensor latitude);

torch::Tensor map_to_channel_matrix_cuda(
    torch::Tensor s_x,
    torch::Tensor s_y,
    int num_tx,
    int num_rx);

torch::Tensor compute_jacobian_cuda(
    torch::Tensor d,
    torch::Tensor r,
    int num_tx,
    int num_rx);

torch::Tensor project_cov3d_to_cov2d_cuda(
    torch::Tensor cov3d_mat,
    torch::Tensor jacobian);

// Rasterization functions
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> project_to_channel_space_cuda(
    torch::Tensor points,
    torch::Tensor cov3d,
    torch::Tensor receiver,
    int num_tx,
    int num_rx);

torch::Tensor compute_gaussian_influence_cuda(
    torch::Tensor uv,
    torch::Tensor cov2d,
    int num_tx,
    int num_rx);

std::tuple<torch::Tensor, torch::Tensor> compute_channel_cuda(
    torch::Tensor attenuation,
    torch::Tensor phase_rotation,
    torch::Tensor distances,
    float wavelength);

torch::Tensor alpha_blending_cuda(
    torch::Tensor influences,
    torch::Tensor contributions,
    torch::Tensor opacity,
    torch::Tensor sort_indices,
    int num_tx,
    int num_rx);

// Main rasterization function
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rasterize_forward_cuda(
    torch::Tensor points,
    torch::Tensor cov3d,
    torch::Tensor attenuation,
    torch::Tensor phase_rotation,
    torch::Tensor opacity,
    torch::Tensor receiver,
    torch::Tensor transmitter,
    int num_tx,
    int num_rx,
    float frequency);

#endif  // CUDA_WIRELESS_RASTERIZE_H_INCLUDED
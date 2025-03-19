// engine/ext.cpp

#include <torch/extension.h>

#include "_cuda_impl/rasterize.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Main functions used by the autograd wrapper
    m.def("rasterize_forward", &rasterize_forward_cuda, "Forward pass for wireless channel rasterization (CUDA)");
    m.def("rasterize_backward", [](torch::Tensor grad_output, torch::Tensor points, torch::Tensor cov3d, torch::Tensor attenuation, torch::Tensor phase_rotation, torch::Tensor opacity, torch::Tensor receiver, torch::Tensor transmitter, int num_tx, int num_rx, float frequency) {
            // Return placeholder gradients for now (to be properly implemented)
            auto grad_points = torch::zeros_like(points);
            auto grad_cov3d = torch::zeros_like(cov3d);
            auto grad_attenuation = torch::zeros_like(attenuation);
            auto grad_phase_rotation = torch::zeros_like(phase_rotation);
            auto grad_opacity = torch::zeros_like(opacity);
            
            return std::make_tuple(
                grad_points, 
                grad_cov3d, 
                grad_attenuation,
                grad_phase_rotation, 
                grad_opacity); }, "Backward pass for wireless channel rasterization (CUDA)");

    // Individual transform functions (for testing)
    m.def("compute_distances_to_receiver", &compute_distances_to_receiver_cuda, "Compute distances from points to receiver (CUDA)");
    m.def("compute_spherical_coords", &compute_spherical_coords_cuda, "Compute spherical coordinates (CUDA)");
    m.def("transform_to_uniform_coords", &transform_to_uniform_coords_cuda, "Transform spherical to uniform coordinates (CUDA)");
    m.def("map_to_channel_matrix", &map_to_channel_matrix_cuda, "Map uniform coordinates to channel matrix (CUDA)");
    m.def("compute_jacobian", &compute_jacobian_cuda, "Compute Jacobian matrices (CUDA)");
    m.def("project_cov3d_to_cov2d", &project_cov3d_to_cov2d_cuda, "Project 3D covariance to 2D (CUDA)");
    m.def("project_to_channel_space", &project_to_channel_space_cuda, "Project to channel space (CUDA)");

    // Individual rasterization functions (for testing)
    m.def("compute_gaussian_influence", &compute_gaussian_influence_cuda, "Compute Gaussian influence (CUDA)");
    m.def("compute_channel", &compute_channel_cuda, "Compute wireless channel contribution (CUDA)");
    m.def("alpha_blending", &alpha_blending_cuda, "Alpha blending (CUDA)");
}
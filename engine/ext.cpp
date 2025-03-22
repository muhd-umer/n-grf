// engine/ext.cpp

#include <torch/extension.h>

#include "_cuda_impl/forward.h"
#include "_cuda_impl/rasterize.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Main functions used by the autograd wrapper
    m.def("rasterize_forward", &rasterizeForwardCUDA,
          "Forward pass for channel rasterization");
    m.def(
        "rasterize_backward",
        [](torch::Tensor grad_output, torch::Tensor points, torch::Tensor cov3d,
           torch::Tensor attenuation, torch::Tensor phase_rotation,
           torch::Tensor opacity, torch::Tensor receiver,
           torch::Tensor transmitter, int num_tx, int num_rx, float frequency) {
            // Return placeholder gradients for now (to be properly implemented)
            auto grad_points = torch::zeros_like(points);
            auto grad_cov3d = torch::zeros_like(cov3d);
            auto grad_attenuation = torch::zeros_like(attenuation);
            auto grad_phase_rotation = torch::zeros_like(phase_rotation);
            auto grad_opacity = torch::zeros_like(opacity);

            return std::make_tuple(grad_points, grad_cov3d, grad_attenuation,
                                   grad_phase_rotation, grad_opacity);
        },
        "Backward pass for channel rasterization");

    // Individual transform functions (for testing)
    m.def("compute_distances_to_receiver", &computeDistancesToReceiverCUDA,
          "Compute distances from points to receiver");
    m.def("compute_spherical_coords", &computeSphericalCoordsCUDA,
          "Compute spherical coordinates");
    m.def("transform_to_uniform_coords", &transformToUniformCoordsCUDA,
          "Transform spherical to uniform coordinates");
    m.def("compute_cov3d_from_scaling_rotation",
          &computeCov3dFromScalingRotationCUDA,
          "Compute 3D covariance matrices from scaling and rotation");
    m.def("map_to_channel_matrix", &mapToChannelMatrixCUDA,
          "Map uniform coordinates to channel matrix");
    m.def("compute_jacobian", &computeJacobianCUDA,
          "Compute Jacobian matrices");
    m.def("project_cov3d_to_cov2d", &projectCov3dToCov2dCUDA,
          "Project 3D covariance to 2D");
    m.def("project_to_channel_space", &projectToChannelSpaceCUDA,
          "Project to channel space");

    // Individual rasterization functions (for testing)
    m.def("compute_gaussian_influence", &computeGaussianInfluenceCUDA,
          "Compute Gaussian influence");
    m.def("compute_channel", &computeChannelCUDA,
          "Compute channel contribution");
    m.def("alpha_blending", &alphaBlendingCUDA, "Alpha blending");
}
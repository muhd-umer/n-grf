// engine/ext.cpp

#include <torch/extension.h>

#include "_cuda_impl/backward.h"
#include "_cuda_impl/forward.h"
#include "_cuda_impl/rasterize.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Main functions used by the autograd wrapper
    m.def("rasterize_forward", &rasterizeForwardCUDA,
          "Forward pass for channel rasterization");
    m.def("rasterize_backward", &rasterizeBackwardCUDA,
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
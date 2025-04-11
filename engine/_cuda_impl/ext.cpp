// engine/_cuda_impl/ext.cpp

#include <torch/extension.h>

void quaternion_to_rotation_cuda(torch::Tensor quaternion,
                                 torch::Tensor rotation);
void compute_scaling_matrix_cuda(torch::Tensor scaling, float scale_modifier,
                                 torch::Tensor scaling_matrix);
void matrix_multiply_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C);
void covariance_matrix_cuda(torch::Tensor RS, torch::Tensor cov3d);
void project_to_channel_coords_cuda(torch::Tensor points,
                                    torch::Tensor receiver, int num_tx,
                                    int num_rx, torch::Tensor distances,
                                    torch::Tensor displacement,
                                    torch::Tensor uv_coords);
void compute_jacobian_cuda(torch::Tensor d, int num_tx, int num_rx,
                           torch::Tensor J);
void project_cov3d_to_cov2d_cuda(torch::Tensor cov3d, torch::Tensor J,
                                 torch::Tensor cov2d);
void compute_gaussian_influence_cuda(torch::Tensor uv, torch::Tensor cov2d,
                                     int num_tx, int num_rx,
                                     torch::Tensor influences);
void compute_wireless_channel_cuda(torch::Tensor attenuation,
                                   torch::Tensor phase_rotation,
                                   torch::Tensor distances, float wavelength,
                                   torch::Tensor real_contributions,
                                   torch::Tensor imag_contributions);
void alpha_blending_forward_cuda(torch::Tensor influences,
                                 torch::Tensor real_contributions,
                                 torch::Tensor imag_contributions,
                                 torch::Tensor opacity,
                                 torch::Tensor sort_indices, int num_tx,
                                 int num_rx, torch::Tensor channel_matrix,
                                 torch::Tensor eff_opacity_out,
                                 torch::Tensor transmittance_out);

void quaternion_to_rotation_backward_cuda(torch::Tensor quaternion,
                                          torch::Tensor grad_rotation,
                                          torch::Tensor grad_quaternion);
void compute_scaling_matrix_backward_cuda(torch::Tensor scaling,
                                          torch::Tensor grad_scaling_matrix,
                                          float scale_modifier,
                                          torch::Tensor grad_scaling);
void matrix_multiply_backward_cuda(torch::Tensor A, torch::Tensor B,
                                   torch::Tensor grad_C, torch::Tensor grad_A,
                                   torch::Tensor grad_B);
void covariance_matrix_backward_cuda(torch::Tensor RS, torch::Tensor grad_cov3d,
                                     torch::Tensor grad_RS);
void project_to_channel_coords_backward_cuda(
    torch::Tensor points, torch::Tensor receiver, torch::Tensor distances,
    torch::Tensor displacement, torch::Tensor grad_distances,
    torch::Tensor grad_displacement, torch::Tensor grad_uv, int num_tx,
    int num_rx, torch::Tensor grad_points);
void compute_jacobian_backward_cuda(torch::Tensor d, torch::Tensor grad_J,
                                    int num_tx, int num_rx,
                                    torch::Tensor grad_d);
void project_cov3d_to_cov2d_backward_cuda(torch::Tensor cov3d, torch::Tensor J,
                                          torch::Tensor grad_cov2d,
                                          torch::Tensor grad_cov3d,
                                          torch::Tensor grad_J);
void compute_gaussian_influence_backward_cuda(
    torch::Tensor uv, torch::Tensor cov2d, torch::Tensor influences,
    torch::Tensor grad_influences, int num_tx, int num_rx,
    torch::Tensor grad_uv, torch::Tensor grad_cov2d);
void compute_wireless_channel_backward_cuda(
    torch::Tensor attenuation, torch::Tensor phase_rotation,
    torch::Tensor distances, float wavelength,
    torch::Tensor grad_real_contributions,
    torch::Tensor grad_imag_contributions, torch::Tensor grad_attenuation,
    torch::Tensor grad_phase_rotation, torch::Tensor grad_distances);
void alpha_blending_backward_cuda(
    torch::Tensor influences, torch::Tensor real_contributions,
    torch::Tensor imag_contributions, torch::Tensor opacity,
    torch::Tensor eff_opacity, torch::Tensor transmittance,
    torch::Tensor sort_indices, torch::Tensor grad_cat_channel, int num_tx,
    int num_rx, torch::Tensor grad_influences, torch::Tensor grad_contrib_real,
    torch::Tensor grad_contrib_imag, torch::Tensor grad_opacity);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quaternion_to_rotation_cuda", &quaternion_to_rotation_cuda,
          "Quaternion to rotation matrix CUDA");
    m.def("compute_scaling_matrix_cuda", &compute_scaling_matrix_cuda,
          "Compute scaling matrix CUDA");
    m.def("matrix_multiply_cuda", &matrix_multiply_cuda,
          "Matrix multiply CUDA");
    m.def("covariance_matrix_cuda", &covariance_matrix_cuda,
          "Compute covariance matrix CUDA");
    m.def("project_to_channel_coords_cuda", &project_to_channel_coords_cuda,
          "Project to channel coordinates CUDA");
    m.def("compute_jacobian_cuda", &compute_jacobian_cuda,
          "Compute Jacobian CUDA");
    m.def("project_cov3d_to_cov2d_cuda", &project_cov3d_to_cov2d_cuda,
          "Project 3D covariance to 2D CUDA");
    m.def("compute_gaussian_influence_cuda", &compute_gaussian_influence_cuda,
          "Compute Gaussian influence CUDA");
    m.def("compute_wireless_channel_cuda", &compute_wireless_channel_cuda,
          "Compute wireless channel CUDA");
    m.def("alpha_blending_forward_cuda", &alpha_blending_forward_cuda,
          "Alpha blending forward CUDA");

    m.def("quaternion_to_rotation_backward_cuda",
          &quaternion_to_rotation_backward_cuda,
          "Quaternion to rotation matrix backward CUDA");
    m.def("compute_scaling_matrix_backward_cuda",
          &compute_scaling_matrix_backward_cuda,
          "Compute scaling matrix backward CUDA");
    m.def("matrix_multiply_backward_cuda", &matrix_multiply_backward_cuda,
          "Matrix multiply backward CUDA");
    m.def("covariance_matrix_backward_cuda", &covariance_matrix_backward_cuda,
          "Compute covariance matrix backward CUDA");
    m.def("project_to_channel_coords_backward_cuda",
          &project_to_channel_coords_backward_cuda,
          "Project to channel coordinates backward CUDA");
    m.def("compute_jacobian_backward_cuda", &compute_jacobian_backward_cuda,
          "Compute Jacobian backward CUDA");
    m.def("project_cov3d_to_cov2d_backward_cuda",
          &project_cov3d_to_cov2d_backward_cuda,
          "Project 3D covariance to 2D backward CUDA");
    m.def("compute_gaussian_influence_backward_cuda",
          &compute_gaussian_influence_backward_cuda,
          "Compute Gaussian influence backward CUDA");
    m.def("compute_wireless_channel_backward_cuda",
          &compute_wireless_channel_backward_cuda,
          "Compute wireless channel backward CUDA");
    m.def("alpha_blending_backward_cuda", &alpha_blending_backward_cuda,
          "Alpha blending backward CUDA");
}
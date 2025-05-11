// render/_cuda_impl/ext.cpp

#include <torch/extension.h>

void normalize_quaternion_fwd_cuda(torch::Tensor q_raw, float eps,
                                   torch::Tensor q_norm);
void quaternion_to_rotation_matrix_fwd_cuda(torch::Tensor q_norm,
                                            torch::Tensor R_matrices);
void exponential_scaling_fwd_cuda(torch::Tensor s_log, torch::Tensor s_act);
void sigmoid_activation_fwd_cuda(torch::Tensor x_logit, torch::Tensor x_act);
void build_inverse_covariance_fwd_cuda(torch::Tensor R_matrices,
                                       torch::Tensor s_act, float eps_clamp,
                                       torch::Tensor Sigma_inv);
void compute_spatial_weight_fwd_cuda(torch::Tensor d_vec_flt,
                                     torch::Tensor Sigma_inv_expanded,
                                     torch::Tensor alpha_squeezed_flt,
                                     float clamp_max,
                                     torch::Tensor spatial_weights_flt);
void weighted_complex_sum_fwd_cuda(torch::Tensor weights,
                                   torch::Tensor contributions_complex,
                                   torch::Tensor H_pred_complex);

void normalize_quaternion_bwd_cuda(torch::Tensor q_raw, torch::Tensor q_norm,
                                   torch::Tensor norm_clamped_squeezed,
                                   torch::Tensor grad_q_norm,
                                   torch::Tensor grad_q_raw);
void quaternion_to_rotation_matrix_bwd_cuda(torch::Tensor q_norm,
                                            torch::Tensor grad_R_matrices,
                                            torch::Tensor grad_q_norm);
void exponential_scaling_bwd_cuda(torch::Tensor s_act, torch::Tensor grad_s_act,
                                  torch::Tensor grad_s_log);
void sigmoid_activation_bwd_cuda(torch::Tensor x_act, torch::Tensor grad_x_act,
                                 torch::Tensor grad_x_logit);
void build_inverse_covariance_bwd_cuda(
    torch::Tensor R_matrices, torch::Tensor s_act, torch::Tensor s_clamped,
    torch::Tensor S_inv_sq_diag_tensor, torch::Tensor grad_Sigma_inv,
    float eps_clamp, torch::Tensor grad_R_matrices,
    torch::Tensor grad_s_act_out);
void compute_spatial_weight_bwd_cuda(
    torch::Tensor d_vec_flt, torch::Tensor Sigma_inv_expanded,
    torch::Tensor alpha_squeezed_flt, torch::Tensor pdf_weight_flt,
    torch::Tensor m_sq_flt, torch::Tensor m_sq_clamped_flt,
    torch::Tensor grad_weights_flt, float clamp_max,
    torch::Tensor grad_d_vec_flt, torch::Tensor grad_Sigma_inv_expanded,
    torch::Tensor grad_alpha_squeezed_flt);
void weighted_complex_sum_bwd_cuda(torch::Tensor weights,
                                   torch::Tensor contributions_complex,
                                   torch::Tensor grad_H_pred_complex,
                                   torch::Tensor grad_weights,
                                   torch::Tensor grad_contributions_complex);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("normalize_quaternion_fwd_cuda", &normalize_quaternion_fwd_cuda,
          "NormalizeQuaternion Forward (CUDA)");
    m.def("quaternion_to_rotation_matrix_fwd_cuda",
          &quaternion_to_rotation_matrix_fwd_cuda,
          "QuaternionToRotationMatrix Forward (CUDA)");
    m.def("exponential_scaling_fwd_cuda", &exponential_scaling_fwd_cuda,
          "ExponentialScaling Forward (CUDA)");
    m.def("sigmoid_activation_fwd_cuda", &sigmoid_activation_fwd_cuda,
          "SigmoidActivation Forward (CUDA)");
    m.def("build_inverse_covariance_fwd_cuda",
          &build_inverse_covariance_fwd_cuda,
          "BuildInverseCovariance Forward (CUDA)");
    m.def("compute_spatial_weight_fwd_cuda", &compute_spatial_weight_fwd_cuda,
          "ComputeSpatialWeight Forward (CUDA)");
    m.def("weighted_complex_sum_fwd_cuda", &weighted_complex_sum_fwd_cuda,
          "WeightedComplexSum Forward (CUDA)");

    m.def("normalize_quaternion_bwd_cuda", &normalize_quaternion_bwd_cuda,
          "NormalizeQuaternion Backward (CUDA)");
    m.def("quaternion_to_rotation_matrix_bwd_cuda",
          &quaternion_to_rotation_matrix_bwd_cuda,
          "QuaternionToRotationMatrix Backward (CUDA)");
    m.def("exponential_scaling_bwd_cuda", &exponential_scaling_bwd_cuda,
          "ExponentialScaling Backward (CUDA)");
    m.def("sigmoid_activation_bwd_cuda", &sigmoid_activation_bwd_cuda,
          "SigmoidActivation Backward (CUDA)");
    m.def("build_inverse_covariance_bwd_cuda",
          &build_inverse_covariance_bwd_cuda,
          "BuildInverseCovariance Backward (CUDA)");
    m.def("compute_spatial_weight_bwd_cuda", &compute_spatial_weight_bwd_cuda,
          "ComputeSpatialWeight Backward (CUDA)");
    m.def("weighted_complex_sum_bwd_cuda", &weighted_complex_sum_bwd_cuda,
          "WeightedComplexSum Backward (CUDA)");
}
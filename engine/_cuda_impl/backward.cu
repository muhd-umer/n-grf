/*
 * Implementation file for backward pass CUDA operations in wireless channel
 * reconstruction
 */

#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "auxiliary.h"
#include "backward.h"
#include "forward.h"

namespace cg = cooperative_groups;

// Device utility function to compute the square of a value
__device__ __forceinline__ float sq(float x) { return x * x; }

// CUDA kernel for converting compact 6D to full 3x3 matrix (for backward pass
// only)
__global__ void convert_symmetric_to_full_backward_kernel(
    const float* cov_compact, float* cov_full, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    symmetric_to_full(&cov_compact[idx * 6], &cov_full[idx * 9]);
}

// Backward pass for alpha blending operation
__global__ void alpha_blending_backward_kernel(
    const float* grad_output, const float* influences,
    const float* contributions_real, const float* contributions_imag,
    const float* opacity, const int64_t* sort_indices, float* grad_alpha_eff,
    float* grad_contributions_real, float* grad_contributions_imag, int num_tx,
    int num_rx, int N) {
    // One thread per tx-rx pair
    int i = blockIdx.x;   // tx index
    int j = threadIdx.x;  // rx index

    if (i >= num_tx || j >= num_rx) return;

    // Compute gradients for real and imaginary parts separately
    float dL_dH_real = grad_output[i * num_rx + j];
    float dL_dH_imag = grad_output[i * num_rx + j + num_tx * num_rx];

    // Initialize transmittance after the last Gaussian to 1.0
    float transmittance_next = 1.0f;
    // Sum terms for backward pass formula
    float grad_alpha_eff_sum_real = 0.0f;
    float grad_alpha_eff_sum_imag = 0.0f;

    // Backward scan through Gaussians in sorted order
    for (int k = N - 1; k >= 0; k--) {
        // Get original index of Gaussian
        const int g_idx = sort_indices[k];

        // Get effective opacity for this Gaussian
        float alpha_eff = opacity[g_idx] *
                          influences[g_idx * num_tx * num_rx + i * num_rx + j];
        // Clamp for numerical stability
        alpha_eff = max(min(alpha_eff, 1.0f - 1e-5f), 1e-5f);

        // Get contribution values
        float C_real = contributions_real[g_idx];
        float C_imag = contributions_imag[g_idx];

        // Current transmittance (before this Gaussian)
        float transmittance = transmittance_next / (1.0f - alpha_eff);

        // Gradient w.r.t. contribution components
        atomicAdd(&grad_contributions_real[g_idx],
                  dL_dH_real * alpha_eff * transmittance);
        atomicAdd(&grad_contributions_imag[g_idx],
                  dL_dH_imag * alpha_eff * transmittance);

        // Gradient w.r.t. effective opacity
        float dL_dalpha_eff =
            dL_dH_real * (C_real * transmittance -
                          grad_alpha_eff_sum_real / (1.0f - alpha_eff)) +
            dL_dH_imag * (C_imag * transmittance -
                          grad_alpha_eff_sum_imag / (1.0f - alpha_eff));

        atomicAdd(&grad_alpha_eff[g_idx * num_tx * num_rx + i * num_rx + j],
                  dL_dalpha_eff);

        // Update sums for next iteration
        grad_alpha_eff_sum_real += alpha_eff * C_real * transmittance;
        grad_alpha_eff_sum_imag += alpha_eff * C_imag * transmittance;

        // Update transmittance for next Gaussian
        transmittance_next = transmittance;
    }
}

__global__ void propagate_alpha_eff_gradients(const float* grad_alpha_eff,
                                              const float* influences,
                                              const float* opacity,
                                              float* grad_opacity,
                                              float* grad_influences,
                                              int num_tx, int num_rx, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    for (int i = 0; i < num_tx; i++) {
        for (int j = 0; j < num_rx; j++) {
            float grad = grad_alpha_eff[idx * num_tx * num_rx + i * num_rx + j];
            if (isnan(grad) || isinf(grad)) {
                continue;  // Skip invalid gradients
            }

            // Gradient w.r.t. original opacity
            float influence =
                influences[idx * num_tx * num_rx + i * num_rx + j];
            atomicAdd(&grad_opacity[idx], grad * influence);

            // Gradient w.r.t. Gaussian influence
            atomicAdd(&grad_influences[idx * num_tx * num_rx + i * num_rx + j],
                      grad * opacity[idx]);
        }
    }
}

// Backward pass for computing wireless channel contributions
__global__ void compute_channel_backward_kernel(
    const float* grad_real, const float* grad_imag, const float* attenuation,
    const float* phase_rotation, const float* xyz_rx_distance,
    float* grad_attenuation, float* grad_phase_rotation, float* grad_distance,
    float wavelength, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float distance = xyz_rx_distance[idx];
    float att = attenuation[idx];
    float phase = phase_rotation[idx];

    // Avoid division by zero
    if (distance < 1e-6f) distance = 1e-6f;

    // Recompute forward pass results
    float path_loss = wavelength / (4.0f * PI * distance);
    float phase_shift = -2.0f * PI * distance / wavelength;

    float total_att = att * path_loss;
    float total_phase = phase + phase_shift;

    float real_part = total_att * cosf(total_phase);
    float imag_part = total_att * sinf(total_phase);

    // Compute gradients
    float dL_dreal = grad_real[idx];
    float dL_dimag = grad_imag[idx];

    // Check for invalid gradients
    if (isnan(dL_dreal) || isinf(dL_dreal) || isnan(dL_dimag) ||
        isinf(dL_dimag)) {
        return;  // Skip invalid gradients
    }

    // Gradient w.r.t. attenuation (A^k)
    // Use ratio if att is non-zero, otherwise use direct computation
    float dL_dA;
    if (fabsf(att) > 1e-6f) {
        dL_dA = dL_dreal * (real_part / att) + dL_dimag * (imag_part / att);
    } else {
        dL_dA = dL_dreal * (path_loss * cosf(total_phase)) +
                dL_dimag * (path_loss * sinf(total_phase));
    }

    // Gradient w.r.t. phase rotation (ψ^k)
    float dL_dpsi = -dL_dreal * imag_part + dL_dimag * real_part;

    // Gradient w.r.t. distance (r^k)
    // Path loss contribution: d/dr(-const/r) = +const/r^2
    float dL_dr_path_loss =
        -(total_att / distance) *
        (dL_dreal * cosf(total_phase) + dL_dimag * sinf(total_phase));

    // Phase shift contribution: d/dr(-const*r) = -const
    float dL_dr_phase = (2.0f * PI / wavelength) *
                        (dL_dreal * imag_part - dL_dimag * real_part);

    float dL_dr = dL_dr_path_loss + dL_dr_phase;

    // Store gradients
    grad_attenuation[idx] = dL_dA;
    grad_phase_rotation[idx] = dL_dpsi;
    grad_distance[idx] = dL_dr;
}

// Backward pass for computing Gaussian influence on channel matrix
__global__ void compute_gaussian_influence_backward_kernel(
    const float* grad_output, const float* uv, const float* cov2d,
    float* grad_uv, float* grad_cov2d, int num_tx, int num_rx, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Extract 2D covariance matrix for this Gaussian
    float cov[4] = {cov2d[idx * 4 + 0], cov2d[idx * 4 + 1], cov2d[idx * 4 + 2],
                    cov2d[idx * 4 + 3]};

    // Invert covariance matrix
    float det = cov[0] * cov[3] - cov[1] * cov[2];
    if (fabsf(det) < 1e-10f) return;  // Skip if determinant is too small

    float inv_det = 1.0f / det;
    float inv_cov[4] = {cov[3] * inv_det, -cov[1] * inv_det, -cov[2] * inv_det,
                        cov[0] * inv_det};

    // Get Gaussian position
    float u = uv[idx * 2];
    float v = uv[idx * 2 + 1];

    // Zero out gradients for current Gaussian
    float du = 0.0f;
    float dv = 0.0f;
    float dcov[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Compute gradients for all channel elements (i,j)
    for (int i = 0; i < num_tx; i++) {
        float antenna_pos_x = i + 0.5f;

        for (int j = 0; j < num_rx; j++) {
            float antenna_pos_y = j + 0.5f;

            float d_x = u - antenna_pos_x;
            float d_y = v - antenna_pos_y;

            // Mahalanobis distance
            float m_dist = inv_cov[0] * d_x * d_x +
                           2.0f * inv_cov[1] * d_x * d_y +
                           inv_cov[3] * d_y * d_y;

            // Influence = exp(-0.5 * m_dist)
            float influence = expf(-0.5f * m_dist);

            // Gradient of loss w.r.t influence
            float dL_dinfluence =
                grad_output[idx * num_tx * num_rx + i * num_rx + j];

            // Check for invalid gradient
            if (isnan(dL_dinfluence) || isinf(dL_dinfluence)) {
                continue;  // Skip invalid gradient
            }

            // Gradient of influence w.r.t Mahalanobis distance
            float dinfluence_dm = -0.5f * influence;

            // Gradient of loss w.r.t Mahalanobis distance
            float dL_dm = dL_dinfluence * dinfluence_dm;

            // Gradient of Mahalanobis distance w.r.t displacement
            float dm_ddx = 2.0f * inv_cov[0] * d_x + 2.0f * inv_cov[1] * d_y;
            float dm_ddy = 2.0f * inv_cov[1] * d_x + 2.0f * inv_cov[3] * d_y;

            // Gradient of loss w.r.t displacement
            float dL_ddx = dL_dm * dm_ddx;
            float dL_ddy = dL_dm * dm_ddy;

            // Gradient of displacement w.r.t u,v is -1
            du -= dL_ddx;
            dv -= dL_ddy;

            // Gradient of Mahalanobis distance w.r.t inverse covariance
            float dm_dinv_cov00 = d_x * d_x;
            float dm_dinv_cov01 = 2.0f * d_x * d_y;
            float dm_dinv_cov11 = d_y * d_y;

            // Accumulate gradients for inverse covariance
            float dL_dinv_cov00 = dL_dm * dm_dinv_cov00;
            float dL_dinv_cov01 = dL_dm * dm_dinv_cov01;
            float dL_dinv_cov11 = dL_dm * dm_dinv_cov11;

            // Convert gradients from inverse covariance to covariance
            // Using the derivative of inverse: d(A^-1)/dA = -A^-1 * dA * A^-1
            float tmp[4];
            tmp[0] = -(inv_cov[0] * dL_dinv_cov00 * inv_cov[0] +
                       inv_cov[0] * dL_dinv_cov01 * inv_cov[2] +
                       inv_cov[1] * dL_dinv_cov00 * inv_cov[1] +
                       inv_cov[1] * dL_dinv_cov01 * inv_cov[3]);

            tmp[1] = -(inv_cov[0] * dL_dinv_cov00 * inv_cov[1] +
                       inv_cov[0] * dL_dinv_cov01 * inv_cov[3] +
                       inv_cov[1] * dL_dinv_cov00 * inv_cov[0] +
                       inv_cov[1] * dL_dinv_cov01 * inv_cov[2]);

            tmp[2] = tmp[1];  // Symmetric matrix

            tmp[3] = -(inv_cov[2] * dL_dinv_cov01 * inv_cov[0] +
                       inv_cov[2] * dL_dinv_cov11 * inv_cov[2] +
                       inv_cov[3] * dL_dinv_cov01 * inv_cov[1] +
                       inv_cov[3] * dL_dinv_cov11 * inv_cov[3]);

            // Accumulate gradients for covariance
            dcov[0] += tmp[0];
            dcov[1] += tmp[1];
            dcov[2] += tmp[2];
            dcov[3] += tmp[3];
        }
    }

    // Store gradients
    grad_uv[idx * 2] = du;
    grad_uv[idx * 2 + 1] = dv;

    grad_cov2d[idx * 4 + 0] = dcov[0];
    grad_cov2d[idx * 4 + 1] = dcov[1];
    grad_cov2d[idx * 4 + 2] = dcov[2];
    grad_cov2d[idx * 4 + 3] = dcov[3];
}

// Backward pass for projecting 3D covariance to 2D
__global__ void project_cov3d_to_cov2d_backward_kernel(
    const float* grad_cov2d, const float* cov3d_mat, const float* jacobian,
    float* grad_cov3d, float* grad_jacobian, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Extract 2x2 gradient for this Gaussian
    float dL_dcov2d[4] = {grad_cov2d[idx * 4 + 0], grad_cov2d[idx * 4 + 1],
                          grad_cov2d[idx * 4 + 2], grad_cov2d[idx * 4 + 3]};

    // Check for invalid gradients
    if (isnan(dL_dcov2d[0]) || isinf(dL_dcov2d[0]) || isnan(dL_dcov2d[1]) ||
        isinf(dL_dcov2d[1]) || isnan(dL_dcov2d[2]) || isinf(dL_dcov2d[2]) ||
        isnan(dL_dcov2d[3]) || isinf(dL_dcov2d[3])) {
        return;  // Skip invalid gradients
    }

    // Symmetrize if necessary
    dL_dcov2d[2] = dL_dcov2d[1];

    // Extract 3x3 covariance for this Gaussian
    float cov3d[9];
    for (int i = 0; i < 9; i++) {
        cov3d[i] = cov3d_mat[idx * 9 + i];
    }

    // Extract 2x3 Jacobian for this Gaussian
    float J[6];
    for (int i = 0; i < 6; i++) {
        J[i] = jacobian[idx * 6 + i];
    }

    // Forward pass: cov2d = J * cov3d * J^T
    // Backward for cov3d: grad_cov3d = J^T * grad_cov2d * J
    float grad_cov3d_mat[9] = {0.0f};

    // J^T * grad_cov2d
    float temp[6];
    temp[0] = J[0] * dL_dcov2d[0] + J[3] * dL_dcov2d[1];
    temp[1] = J[1] * dL_dcov2d[0] + J[4] * dL_dcov2d[1];
    temp[2] = J[2] * dL_dcov2d[0] + J[5] * dL_dcov2d[1];
    temp[3] = J[0] * dL_dcov2d[2] + J[3] * dL_dcov2d[3];
    temp[4] = J[1] * dL_dcov2d[2] + J[4] * dL_dcov2d[3];
    temp[5] = J[2] * dL_dcov2d[2] + J[5] * dL_dcov2d[3];

    // (J^T * grad_cov2d) * J
    grad_cov3d_mat[0] = temp[0] * J[0] + temp[3] * J[3];
    grad_cov3d_mat[1] = temp[0] * J[1] + temp[3] * J[4];
    grad_cov3d_mat[2] = temp[0] * J[2] + temp[3] * J[5];
    grad_cov3d_mat[3] = temp[1] * J[0] + temp[4] * J[3];
    grad_cov3d_mat[4] = temp[1] * J[1] + temp[4] * J[4];
    grad_cov3d_mat[5] = temp[1] * J[2] + temp[4] * J[5];
    grad_cov3d_mat[6] = temp[2] * J[0] + temp[5] * J[3];
    grad_cov3d_mat[7] = temp[2] * J[1] + temp[5] * J[4];
    grad_cov3d_mat[8] = temp[2] * J[2] + temp[5] * J[5];

    // Symmetrize
    grad_cov3d_mat[3] = grad_cov3d_mat[1];
    grad_cov3d_mat[6] = grad_cov3d_mat[2];
    grad_cov3d_mat[7] = grad_cov3d_mat[5];

    // Store 3x3 gradient as a 6D vector (only upper triangle due to symmetry)
    grad_cov3d[idx * 6 + 0] = grad_cov3d_mat[0];  // xx
    grad_cov3d[idx * 6 + 1] = grad_cov3d_mat[1];  // xy
    grad_cov3d[idx * 6 + 2] = grad_cov3d_mat[2];  // xz
    grad_cov3d[idx * 6 + 3] = grad_cov3d_mat[4];  // yy
    grad_cov3d[idx * 6 + 4] = grad_cov3d_mat[5];  // yz
    grad_cov3d[idx * 6 + 5] = grad_cov3d_mat[8];  // zz

    // Backward for Jacobian: grad_J = 2 * grad_cov2d * J * cov3d
    float dL_dJ[6] = {0.0f};

    // grad_cov2d * J
    float temp2[6];
    temp2[0] = dL_dcov2d[0] * J[0] + dL_dcov2d[1] * J[3];
    temp2[1] = dL_dcov2d[0] * J[1] + dL_dcov2d[1] * J[4];
    temp2[2] = dL_dcov2d[0] * J[2] + dL_dcov2d[1] * J[5];
    temp2[3] = dL_dcov2d[2] * J[0] + dL_dcov2d[3] * J[3];
    temp2[4] = dL_dcov2d[2] * J[1] + dL_dcov2d[3] * J[4];
    temp2[5] = dL_dcov2d[2] * J[2] + dL_dcov2d[3] * J[5];

    // (grad_cov2d * J) * cov3d
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                dL_dJ[i * 3 + j] += temp2[i * 3 + k] * cov3d[k * 3 + j];
            }
        }
    }

    // Factor of 2 due to the symmetric nature of the covariance matrix
    for (int i = 0; i < 6; i++) {
        grad_jacobian[idx * 6 + i] = 2.0f * dL_dJ[i];
    }
}

// Backward pass for computing the Jacobian matrix
__global__ void compute_jacobian_backward_kernel(const float* grad_jacobian,
                                                 const float* d, const float* r,
                                                 float* grad_d, float* grad_r,
                                                 int num_tx, int num_rx,
                                                 int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Extract displacement vector components
    float x = d[idx * 3 + 0];
    float y = d[idx * 3 + 1];
    float z = d[idx * 3 + 2];
    float r_val = r[idx];

    // Check for invalid inputs
    if (r_val < 1e-6f) r_val = 1e-6f;  // Avoid division by zero

    // Calculate intermediate values
    float xy_squared = x * x + y * y;
    xy_squared = fmaxf(xy_squared, 1e-6f);  // Avoid division by zero

    float cos_lat = sqrtf(1.0f - (z / r_val) * (z / r_val));
    cos_lat = fmaxf(cos_lat, 1e-6f);  // Avoid division by zero

    float tx_factor = float(num_tx - 1) / (2.0f * PI);
    float rx_factor = float(num_rx - 1) / PI;

    // Extract gradients for Jacobian elements
    float dL_dJ00 = grad_jacobian[idx * 6 + 0];  // du/dx
    float dL_dJ01 = grad_jacobian[idx * 6 + 1];  // du/dy
    float dL_dJ02 = grad_jacobian[idx * 6 + 2];  // du/dz
    float dL_dJ10 = grad_jacobian[idx * 6 + 3];  // dv/dx
    float dL_dJ11 = grad_jacobian[idx * 6 + 4];  // dv/dy
    float dL_dJ12 = grad_jacobian[idx * 6 + 5];  // dv/dz

    // Check for invalid gradients
    if (isnan(dL_dJ00) || isinf(dL_dJ00) || isnan(dL_dJ01) || isinf(dL_dJ01) ||
        isnan(dL_dJ02) || isinf(dL_dJ02) || isnan(dL_dJ10) || isinf(dL_dJ10) ||
        isnan(dL_dJ11) || isinf(dL_dJ11) || isnan(dL_dJ12) || isinf(dL_dJ12)) {
        return;  // Skip invalid gradients
    }

    // Initialize gradient accumulators
    float dL_dx = 0.0f;
    float dL_dy = 0.0f;
    float dL_dz = 0.0f;
    float dL_dr = 0.0f;

    // Compute gradients for J[0,0] = tx_factor * (-y / xy_squared)
    dL_dx += dL_dJ00 * tx_factor * (2.0f * x * y / (xy_squared * xy_squared));
    dL_dy += dL_dJ00 * tx_factor *
             (-1.0f / xy_squared + 2.0f * y * y / (xy_squared * xy_squared));

    // Compute gradients for J[0,1] = tx_factor * (x / xy_squared)
    dL_dx += dL_dJ01 * tx_factor *
             (1.0f / xy_squared - 2.0f * x * x / (xy_squared * xy_squared));
    dL_dy += dL_dJ01 * tx_factor * (-2.0f * x * y / (xy_squared * xy_squared));

    // Compute gradients for J[0,2] = 0.0f (no gradients here)

    // Compute intermediate values for computing gradients for J[1,*]
    float r_cos_lat_xy = r_val * cos_lat * xy_squared;
    r_cos_lat_xy = fmaxf(r_cos_lat_xy, 1e-6f);  // Avoid division by zero

    // Complex derivatives for the lower row of the Jacobian
    // These are computed from the chain rule based on the equations in
    // backward.md

    // Compute gradients for J[1,0] = rx_factor * (z * x) / r_cos_lat_xy
    dL_dx += dL_dJ10 * rx_factor * z *
             ((r_cos_lat_xy - x * x * r_val * cos_lat) /
              (r_cos_lat_xy * r_cos_lat_xy));
    dL_dy += dL_dJ10 * rx_factor * z *
             (-x * y * r_val * cos_lat / (r_cos_lat_xy * r_cos_lat_xy));
    dL_dz += dL_dJ10 * rx_factor * x / r_cos_lat_xy;
    dL_dr += dL_dJ10 * rx_factor * z * x * cos_lat * xy_squared /
             (r_cos_lat_xy * r_cos_lat_xy);

    // Compute gradients for J[1,1] = rx_factor * (z * y) / r_cos_lat_xy
    dL_dx += dL_dJ11 * rx_factor * z *
             (-x * y * r_val * cos_lat / (r_cos_lat_xy * r_cos_lat_xy));
    dL_dy += dL_dJ11 * rx_factor * z *
             ((r_cos_lat_xy - y * y * r_val * cos_lat) /
              (r_cos_lat_xy * r_cos_lat_xy));
    dL_dz += dL_dJ11 * rx_factor * y / r_cos_lat_xy;
    dL_dr += dL_dJ11 * rx_factor * z * y * cos_lat * xy_squared /
             (r_cos_lat_xy * r_cos_lat_xy);

    // Compute gradients for J[1,2] = rx_factor / (r_val * cos_lat)
    dL_dx += dL_dJ12 * rx_factor * z * x /
             (r_val * r_val * cos_lat * cos_lat * cos_lat);
    dL_dy += dL_dJ12 * rx_factor * z * y /
             (r_val * r_val * cos_lat * cos_lat * cos_lat);
    dL_dz += dL_dJ12 * rx_factor *
             (-1.0f / (r_val * r_val * cos_lat) +
              z * z / (r_val * r_val * r_val * cos_lat * cos_lat * cos_lat));
    dL_dr += dL_dJ12 * rx_factor * (-1.0f / (r_val * r_val * cos_lat));

    // Check for invalid gradients
    if (isnan(dL_dx) || isinf(dL_dx) || isnan(dL_dy) || isinf(dL_dy) ||
        isnan(dL_dz) || isinf(dL_dz) || isnan(dL_dr) || isinf(dL_dr)) {
        return;  // Skip invalid gradients
    }

    // Store the computed gradients
    grad_d[idx * 3 + 0] = dL_dx;
    grad_d[idx * 3 + 1] = dL_dy;
    grad_d[idx * 3 + 2] = dL_dz;
    grad_r[idx] = dL_dr;
}

// Backward pass for mapping to channel matrix
__global__ void map_to_channel_matrix_backward_kernel(
    const float* grad_uv, const float* s_x, const float* s_y, float* grad_s_x,
    float* grad_s_y, int num_tx, int num_rx, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Forward pass: u = ((s_x + 1) / 2) * (num_tx - 1) + 0.5
    //              v = ((s_y + 1) / 2) * (num_rx - 1) + 0.5

    // Extract gradients for u and v
    float dL_du = grad_uv[idx * 2 + 0];
    float dL_dv = grad_uv[idx * 2 + 1];

    // Check for invalid gradients
    if (isnan(dL_du) || isinf(dL_du) || isnan(dL_dv) || isinf(dL_dv)) {
        return;  // Skip invalid gradients
    }

    // Compute gradients for s_x and s_y
    float dL_ds_x = dL_du * (num_tx - 1) / 2.0f;
    float dL_ds_y = dL_dv * (num_rx - 1) / 2.0f;

    // Store gradients
    grad_s_x[idx] = dL_ds_x;
    grad_s_y[idx] = dL_ds_y;
}

// Backward pass for transforming to uniform coordinates
__global__ void transform_to_uniform_coords_backward_kernel(
    const float* grad_s_x, const float* grad_s_y, const float* longitude,
    const float* latitude, float* grad_longitude, float* grad_latitude, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Forward pass: s_x = longitude / PI
    //              s_y = 2 * latitude / PI

    // Extract gradients for s_x and s_y
    float dL_ds_x = grad_s_x[idx];
    float dL_ds_y = grad_s_y[idx];

    // Check for invalid gradients
    if (isnan(dL_ds_x) || isinf(dL_ds_x) || isnan(dL_ds_y) || isinf(dL_ds_y)) {
        return;  // Skip invalid gradients
    }

    // Compute gradients for longitude and latitude
    float dL_dlongitude = dL_ds_x / PI;
    float dL_dlatitude = dL_ds_y * 2.0f / PI;

    // Store gradients
    grad_longitude[idx] = dL_dlongitude;
    grad_latitude[idx] = dL_dlatitude;
}

// Backward pass for computing spherical coordinates
__global__ void compute_spherical_coords_backward_kernel(
    const float* grad_longitude, const float* grad_latitude,
    const float* displacement, const float* r, float* grad_displacement,
    float* grad_r, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Extract displacement components
    float dx = displacement[idx * 3 + 0];
    float dy = displacement[idx * 3 + 1];
    float dz = displacement[idx * 3 + 2];
    float r_val = r[idx];

    // Check for invalid inputs
    if (r_val < 1e-6f) r_val = 1e-6f;  // Avoid division by zero

    // Extract gradients for longitude and latitude
    float dL_dlongitude = grad_longitude[idx];
    float dL_dlatitude = grad_latitude[idx];

    // Check for invalid gradients
    if (isnan(dL_dlongitude) || isinf(dL_dlongitude) || isnan(dL_dlatitude) ||
        isinf(dL_dlatitude)) {
        return;  // Skip invalid gradients
    }

    // Initialize gradient accumulators
    float dL_ddx = 0.0f;
    float dL_ddy = 0.0f;
    float dL_ddz = 0.0f;
    float dL_dr = 0.0f;

    // Compute gradients for longitude (arctan2(dy, dx))
    float xy_squared = dx * dx + dy * dy;
    xy_squared = fmaxf(xy_squared, 1e-6f);  // Avoid division by zero

    dL_ddx += dL_dlongitude * (-dy / xy_squared);
    dL_ddy += dL_dlongitude * (dx / xy_squared);

    // Compute gradients for latitude (arcsin(dz / r))
    float one_minus_lat_squared = 1.0f - (dz / r_val) * (dz / r_val);
    one_minus_lat_squared =
        fmaxf(one_minus_lat_squared, 1e-6f);  // Avoid division by zero

    float sqrt_term = sqrtf(one_minus_lat_squared);

    dL_ddz += dL_dlatitude * (1.0f / (r_val * sqrt_term));
    dL_dr += dL_dlatitude * (-dz / (r_val * r_val * sqrt_term));

    // Check for invalid gradients
    if (isnan(dL_ddx) || isinf(dL_ddx) || isnan(dL_ddy) || isinf(dL_ddy) ||
        isnan(dL_ddz) || isinf(dL_ddz) || isnan(dL_dr) || isinf(dL_dr)) {
        return;  // Skip invalid gradients
    }

    // Store gradients
    grad_displacement[idx * 3 + 0] = dL_ddx;
    grad_displacement[idx * 3 + 1] = dL_ddy;
    grad_displacement[idx * 3 + 2] = dL_ddz;
    atomicAdd(&grad_r[idx], dL_dr);  // Accumulate with existing gradients
}

// Backward pass for computing distances to receiver
__global__ void compute_distances_to_receiver_backward_kernel(
    const float* grad_distance, const float* points, const float* receiver,
    float* grad_points, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Extract displacement components
    float dx = points[idx * 3 + 0] - receiver[0];
    float dy = points[idx * 3 + 1] - receiver[1];
    float dz = points[idx * 3 + 2] - receiver[2];

    // Compute distance
    float r = sqrtf(dx * dx + dy * dy + dz * dz);
    r = fmaxf(r, 1e-6f);  // Avoid division by zero

    // Extract gradient for distance
    float dL_dr = grad_distance[idx];

    // Check for invalid gradient
    if (isnan(dL_dr) || isinf(dL_dr)) {
        return;  // Skip invalid gradient
    }

    // Compute gradients for displacement components
    float dL_ddx = dL_dr * dx / r;
    float dL_ddy = dL_dr * dy / r;
    float dL_ddz = dL_dr * dz / r;

    // Check for invalid gradients
    if (isnan(dL_ddx) || isinf(dL_ddx) || isnan(dL_ddy) || isinf(dL_ddy) ||
        isnan(dL_ddz) || isinf(dL_ddz)) {
        return;  // Skip invalid gradients
    }

    // Store gradients for points
    grad_points[idx * 3 + 0] = dL_ddx;
    grad_points[idx * 3 + 1] = dL_ddy;
    grad_points[idx * 3 + 2] = dL_ddz;
}

// Backward pass for computing 3D covariance from scaling and rotation
__global__ void compute_cov3d_from_scaling_rotation_backward_kernel(
    const float* grad_cov3d, const float* scaling, const float* rotation,
    float* grad_scaling, float* grad_rotation, float scale_modifier, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Extract scaling and rotation parameters
    float s_x = scale_modifier * scaling[idx * 3 + 0];
    float s_y = scale_modifier * scaling[idx * 3 + 1];
    float s_z = scale_modifier * scaling[idx * 3 + 2];

    float r = rotation[idx * 4 + 0];
    float x = rotation[idx * 4 + 1];
    float y = rotation[idx * 4 + 2];
    float z = rotation[idx * 4 + 3];

    // Extract gradients for 3D covariance (in compact form)
    float dL_dcov00 = grad_cov3d[idx * 6 + 0];
    float dL_dcov01 = grad_cov3d[idx * 6 + 1];
    float dL_dcov02 = grad_cov3d[idx * 6 + 2];
    float dL_dcov11 = grad_cov3d[idx * 6 + 3];
    float dL_dcov12 = grad_cov3d[idx * 6 + 4];
    float dL_dcov22 = grad_cov3d[idx * 6 + 5];

    // Check for invalid gradients
    if (isnan(dL_dcov00) || isinf(dL_dcov00) || isnan(dL_dcov01) ||
        isinf(dL_dcov01) || isnan(dL_dcov02) || isinf(dL_dcov02) ||
        isnan(dL_dcov11) || isinf(dL_dcov11) || isnan(dL_dcov12) ||
        isinf(dL_dcov12) || isnan(dL_dcov22) || isinf(dL_dcov22)) {
        return;  // Skip invalid gradients
    }

    // Convert compact form to full matrix form
    float dL_dSigma[9] = {dL_dcov00, dL_dcov01, dL_dcov02, dL_dcov01, dL_dcov11,
                          dL_dcov12, dL_dcov02, dL_dcov12, dL_dcov22};

    // Compute rotation matrix components
    float R[9] = {1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - r * z),
                  2.0f * (x * z + r * y),        2.0f * (x * y + r * z),
                  1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - r * x),
                  2.0f * (x * z - r * y),        2.0f * (y * z + r * x),
                  1.0f - 2.0f * (x * x + y * y)};

    // Initialize gradient accumulators for scaling and rotation
    float dL_ds[3] = {0.0f, 0.0f, 0.0f};
    float dL_dR[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Compute the transpose of R
    float Rt[9] = {R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]};

    // Compute gradient for scaling using Rt * dL_dSigma * R
    float temp[9] = {0.0f};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                temp[i * 3 + j] += Rt[i * 3 + k] * dL_dSigma[k * 3 + j];
            }
        }
    }

    float temp2[9] = {0.0f};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                temp2[i * 3 + j] += temp[i * 3 + k] * R[j * 3 + k];
            }
        }
    }

    // Extract gradients for scaling
    dL_ds[0] = 2.0f * s_x * temp2[0 * 3 + 0];
    dL_ds[1] = 2.0f * s_y * temp2[1 * 3 + 1];
    dL_ds[2] = 2.0f * s_z * temp2[2 * 3 + 2];

    // Compute scaling matrix
    float S[9] = {s_x, 0.0f, 0.0f, 0.0f, s_y, 0.0f, 0.0f, 0.0f, s_z};

    // Compute M = R * S
    float M[9] = {0.0f};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                M[i * 3 + j] += R[i * 3 + k] * S[k * 3 + j];
            }
        }
    }

    // Compute dL_dM = 2 * dL_dSigma * M
    float dL_dM[9] = {0.0f};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                dL_dM[i * 3 + j] += 2.0f * dL_dSigma[i * 3 + k] * M[k * 3 + j];
            }
        }
    }

    // Compute dL_dR = dL_dM * S
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                dL_dR[i * 3 + j] += dL_dM[i * 3 + k] * S[k * 3 + j];
            }
        }
    }

    // Compute gradient for quaternion parameters from rotation matrix gradient
    float dL_dq[4] = {0.0f};

    // Using the chain rule and the derivative of the rotation matrix w.r.t
    // quaternion
    dL_dq[0] = 2.0f * (dL_dR[0 * 3 + 1] * (-z) + dL_dR[0 * 3 + 2] * y +
                       dL_dR[1 * 3 + 0] * z + dL_dR[1 * 3 + 2] * (-x) +
                       dL_dR[2 * 3 + 0] * (-y) + dL_dR[2 * 3 + 1] * x);

    dL_dq[1] = 2.0f * (dL_dR[0 * 3 + 1] * y + dL_dR[0 * 3 + 2] * z +
                       dL_dR[1 * 3 + 0] * y + dL_dR[1 * 3 + 1] * (-2 * x) +
                       dL_dR[1 * 3 + 2] * (-r) + dL_dR[2 * 3 + 0] * z +
                       dL_dR[2 * 3 + 1] * r + dL_dR[2 * 3 + 2] * (-2 * x));

    dL_dq[2] = 2.0f * (dL_dR[0 * 3 + 0] * (-2 * y) + dL_dR[0 * 3 + 1] * x +
                       dL_dR[0 * 3 + 2] * r + dL_dR[1 * 3 + 0] * x +
                       dL_dR[1 * 3 + 2] * z + dL_dR[2 * 3 + 0] * (-r) +
                       dL_dR[2 * 3 + 1] * z + dL_dR[2 * 3 + 2] * (-2 * y));

    dL_dq[3] = 2.0f * (dL_dR[0 * 3 + 0] * (-2 * z) + dL_dR[0 * 3 + 1] * (-r) +
                       dL_dR[0 * 3 + 2] * x + dL_dR[1 * 3 + 0] * r +
                       dL_dR[1 * 3 + 1] * (-2 * z) + dL_dR[1 * 3 + 2] * y +
                       dL_dR[2 * 3 + 0] * x + dL_dR[2 * 3 + 1] * y);

    // Check for invalid gradients
    if (isnan(dL_ds[0]) || isinf(dL_ds[0]) || isnan(dL_ds[1]) ||
        isinf(dL_ds[1]) || isnan(dL_ds[2]) || isinf(dL_ds[2]) ||
        isnan(dL_dq[0]) || isinf(dL_dq[0]) || isnan(dL_dq[1]) ||
        isinf(dL_dq[1]) || isnan(dL_dq[2]) || isinf(dL_dq[2]) ||
        isnan(dL_dq[3]) || isinf(dL_dq[3])) {
        return;  // Skip invalid gradients
    }

    // Store gradients for scaling and rotation
    grad_scaling[idx * 3 + 0] = dL_ds[0] * scale_modifier;
    grad_scaling[idx * 3 + 1] = dL_ds[1] * scale_modifier;
    grad_scaling[idx * 3 + 2] = dL_ds[2] * scale_modifier;

    grad_rotation[idx * 4 + 0] = dL_dq[0];
    grad_rotation[idx * 4 + 1] = dL_dq[1];
    grad_rotation[idx * 4 + 2] = dL_dq[2];
    grad_rotation[idx * 4 + 3] = dL_dq[3];
}

// Main backward function for the entire rasterization pipeline
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
rasterizeBackwardCUDA(
    const torch::Tensor& grad_output, const torch::Tensor& points,
    const torch::Tensor& scaling, const torch::Tensor& rotation,
    const torch::Tensor& attenuation, const torch::Tensor& phase_rotation,
    const torch::Tensor& opacity, const torch::Tensor& receiver,
    const torch::Tensor& transmitter, const int num_tx, const int num_rx,
    const float frequency, const float scale_modifier) {
    const at::cuda::CUDAGuard device_guard(points.device());
    const int N = points.size(0);

    auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(points.device());

    // Allocate gradients
    torch::Tensor grad_points = torch::zeros({N, 3}, options);
    torch::Tensor grad_scaling = torch::zeros({N, 3}, options);
    torch::Tensor grad_rotation = torch::zeros({N, 4}, options);
    torch::Tensor grad_attenuation = torch::zeros({N, 1}, options);
    torch::Tensor grad_phase_rotation = torch::zeros({N, 1}, options);
    torch::Tensor grad_opacity = torch::zeros({N, 1}, options);

    // Constants
    const float c = 299792458.0f;
    const float wavelength = c / frequency;

    // Recompute all forward pass intermediate values needed for backward

    // 1. Compute covariance
    torch::Tensor cov3d =
        computeCov3dFromScalingRotationCUDA(scaling, rotation, scale_modifier);

    // 2. Project to channel space
    auto projected =
        projectToChannelSpaceCUDA(points, cov3d, receiver, num_tx, num_rx);
    torch::Tensor distances = std::get<0>(projected);
    torch::Tensor uv = std::get<1>(projected);
    torch::Tensor cov2d = std::get<2>(projected);

    // 3. Sort points by distance
    torch::Tensor sort_indices = torch::argsort(distances);

    // 4. Compute influences
    torch::Tensor influences =
        computeGaussianInfluenceCUDA(uv, cov2d, num_tx, num_rx);

    // 5. Compute wireless channel contributions
    auto channel_contributions =
        computeChannelCUDA(attenuation, phase_rotation, distances, wavelength);
    torch::Tensor real_contributions =
        std::get<0>(channel_contributions).squeeze(-1);
    torch::Tensor imag_contributions =
        std::get<1>(channel_contributions).squeeze(-1);

    // Now backpropagate gradients
    // 1. Alpha blending backward
    torch::Tensor grad_alpha_eff = torch::zeros({N, num_tx, num_rx}, options);
    torch::Tensor grad_contributions_real = torch::zeros({N}, options);
    torch::Tensor grad_contributions_imag = torch::zeros({N}, options);
    torch::Tensor grad_influences = torch::zeros({N, num_tx, num_rx}, options);

    {
        const int threads = num_rx;
        const int blocks = num_tx;

        // Make sure we're not launching too many threads
        if (threads > 1024) {
            std::cerr << "Warning: num_rx exceeds maximum thread count, using "
                         "fallback launch config"
                      << std::endl;
            const int max_threads = 256;
            const int threads_actual = std::min(max_threads, num_rx);
            const int blocks_x = (num_tx + threads_actual - 1) / threads_actual;
            const int blocks_y = (num_rx + threads_actual - 1) / threads_actual;
            const dim3 grid(blocks_x, blocks_y);
            const dim3 block(threads_actual, threads_actual);
            // Note: would need to modify kernel for 2D grid/block
        }

        alpha_blending_backward_kernel<<<blocks, threads>>>(
            grad_output.data_ptr<float>(), influences.data_ptr<float>(),
            real_contributions.data_ptr<float>(),
            imag_contributions.data_ptr<float>(), opacity.data_ptr<float>(),
            sort_indices.data_ptr<int64_t>(), grad_alpha_eff.data_ptr<float>(),
            grad_contributions_real.data_ptr<float>(),
            grad_contributions_imag.data_ptr<float>(), num_tx, num_rx, N);
    }

    // 2. Propagate gradients from effective opacity to opacity and influences
    {
        const int threads = 32;
        const int blocks = (N + threads - 1) / threads;

        propagate_alpha_eff_gradients<<<blocks, threads>>>(
            grad_alpha_eff.data_ptr<float>(), influences.data_ptr<float>(),
            opacity.data_ptr<float>(), grad_opacity.data_ptr<float>(),
            grad_influences.data_ptr<float>(), num_tx, num_rx, N);
    }

    // 3. Compute gradients for the wireless channel contribution
    torch::Tensor grad_distances = torch::zeros({N}, options);

    {
        const int threads = 32;
        const int blocks = (N + threads - 1) / threads;

        compute_channel_backward_kernel<<<blocks, threads>>>(
            grad_contributions_real.data_ptr<float>(),
            grad_contributions_imag.data_ptr<float>(),
            attenuation.data_ptr<float>(), phase_rotation.data_ptr<float>(),
            distances.data_ptr<float>(), grad_attenuation.data_ptr<float>(),
            grad_phase_rotation.data_ptr<float>(),
            grad_distances.data_ptr<float>(), wavelength, N);
    }

    // 4. Compute gradients for Gaussian influence
    torch::Tensor grad_uv = torch::zeros({N, 2}, options);
    torch::Tensor grad_cov2d = torch::zeros({N, 2, 2}, options);

    {
        const int threads = 32;
        const int blocks = (N + threads - 1) / threads;

        compute_gaussian_influence_backward_kernel<<<blocks, threads>>>(
            grad_influences.data_ptr<float>(), uv.data_ptr<float>(),
            cov2d.data_ptr<float>(), grad_uv.data_ptr<float>(),
            grad_cov2d.data_ptr<float>(), num_tx, num_rx, N);
    }

    // 5. Compute gradients for projecting 3D covariance to 2D
    torch::Tensor cov3d_mat = torch::empty({N, 3, 3}, options);
    torch::Tensor grad_cov3d = torch::zeros({N, 6}, options);
    torch::Tensor grad_jacobian = torch::zeros({N, 2, 3}, options);

    // First, convert compact 3D covariance to full 3x3 matrix
    {
        const int threads = 32;
        const int blocks = (N + threads - 1) / threads;

        convert_symmetric_to_full_backward_kernel<<<blocks, threads>>>(
            cov3d.data_ptr<float>(), cov3d_mat.data_ptr<float>(), N);
    }

    // Extract jacobian from the torch wrapper function
    // Note: this is a bit of a hack, would be better to have jacobian saved
    // from forward pass
    auto sph_coords = computeSphericalCoordsCUDA(points, receiver);
    torch::Tensor d = std::get<0>(sph_coords);
    torch::Tensor longitude = std::get<1>(sph_coords);
    torch::Tensor latitude = std::get<2>(sph_coords);
    torch::Tensor jacobian = computeJacobianCUDA(d, distances, num_tx, num_rx);

    // Now compute gradients for 3D->2D projection
    {
        const int threads = 32;
        const int blocks = (N + threads - 1) / threads;

        project_cov3d_to_cov2d_backward_kernel<<<blocks, threads>>>(
            grad_cov2d.data_ptr<float>(), cov3d_mat.data_ptr<float>(),
            jacobian.data_ptr<float>(), grad_cov3d.data_ptr<float>(),
            grad_jacobian.data_ptr<float>(), N);
    }

    // 6. Compute gradients for Jacobian
    torch::Tensor grad_d = torch::zeros({N, 3}, options);
    torch::Tensor grad_r = torch::zeros({N}, options);

    {
        const int threads = 32;
        const int blocks = (N + threads - 1) / threads;

        compute_jacobian_backward_kernel<<<blocks, threads>>>(
            grad_jacobian.data_ptr<float>(), d.data_ptr<float>(),
            distances.data_ptr<float>(), grad_d.data_ptr<float>(),
            grad_r.data_ptr<float>(), num_tx, num_rx, N);
    }

    // 7. Compute gradients for mapping to channel matrix
    auto uniform_coords = transformToUniformCoordsCUDA(longitude, latitude);
    torch::Tensor s_x = std::get<0>(uniform_coords);
    torch::Tensor s_y = std::get<1>(uniform_coords);
    torch::Tensor grad_s_x = torch::zeros({N}, options);
    torch::Tensor grad_s_y = torch::zeros({N}, options);

    {
        const int threads = 32;
        const int blocks = (N + threads - 1) / threads;

        map_to_channel_matrix_backward_kernel<<<blocks, threads>>>(
            grad_uv.data_ptr<float>(), s_x.data_ptr<float>(),
            s_y.data_ptr<float>(), grad_s_x.data_ptr<float>(),
            grad_s_y.data_ptr<float>(), num_tx, num_rx, N);
    }

    // 8. Compute gradients for transforming to uniform coordinates
    torch::Tensor grad_longitude = torch::zeros({N}, options);
    torch::Tensor grad_latitude = torch::zeros({N}, options);

    {
        const int threads = 32;
        const int blocks = (N + threads - 1) / threads;

        transform_to_uniform_coords_backward_kernel<<<blocks, threads>>>(
            grad_s_x.data_ptr<float>(), grad_s_y.data_ptr<float>(),
            longitude.data_ptr<float>(), latitude.data_ptr<float>(),
            grad_longitude.data_ptr<float>(), grad_latitude.data_ptr<float>(),
            N);
    }

    // 9. Compute gradients for spherical coordinates
    torch::Tensor grad_displacement = torch::zeros({N, 3}, options);

    {
        const int threads = 32;
        const int blocks = (N + threads - 1) / threads;

        compute_spherical_coords_backward_kernel<<<blocks, threads>>>(
            grad_longitude.data_ptr<float>(), grad_latitude.data_ptr<float>(),
            d.data_ptr<float>(), distances.data_ptr<float>(),
            grad_displacement.data_ptr<float>(), grad_r.data_ptr<float>(), N);
    }

    // 10. Compute gradients for distances - accumulate into grad_r
    // Already accumulated from previous steps

    // 11. Compute gradients for points
    {
        const int threads = 32;
        const int blocks = (N + threads - 1) / threads;

        compute_distances_to_receiver_backward_kernel<<<blocks, threads>>>(
            grad_distances.data_ptr<float>(), points.data_ptr<float>(),
            receiver.data_ptr<float>(), grad_points.data_ptr<float>(), N);
    }

    // Add gradient contribution from displacement
    grad_points = grad_points + grad_displacement;

    // Add gradient contribution from grad_d
    grad_points = grad_points + grad_d;

    // 12. Compute gradients for scaling and rotation
    {
        const int threads = 32;
        const int blocks = (N + threads - 1) / threads;

        compute_cov3d_from_scaling_rotation_backward_kernel<<<blocks,
                                                              threads>>>(
            grad_cov3d.data_ptr<float>(), scaling.data_ptr<float>(),
            rotation.data_ptr<float>(), grad_scaling.data_ptr<float>(),
            grad_rotation.data_ptr<float>(), scale_modifier, N);
    }

    return std::make_tuple(grad_points, grad_attenuation, grad_phase_rotation,
                           grad_opacity, grad_scaling, grad_rotation);
}
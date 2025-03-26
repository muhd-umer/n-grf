/*
 * Backward pass implementation for wireless channel reconstruction
 */

#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "auxiliary.h"
#include "backward.h"
#include "forward.h"

// Convert compact symmetric 6D covariance to full 3x3 matrix
__global__ void convertCompactToFullKernel(const float* cov_compact,
                                           float* cov_full, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    symmetric_to_full(&cov_compact[idx * 6], &cov_full[idx * 9]);
}

// Split gradient from output channel matrix
__global__ void prepareGradOutputKernel(const float* grad_output,
                                        float* grad_channel_real,
                                        float* grad_channel_imag, int num_tx,
                                        int num_rx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_tx * num_rx) return;

    int tx = i / num_rx;
    int rx = i % num_rx;

    // Split real and imaginary parts
    grad_channel_real[tx * num_rx + rx] = grad_output[tx * num_rx + rx];
    grad_channel_imag[tx * num_rx + rx] =
        grad_output[num_tx * num_rx + tx * num_rx + rx];
}

// Alpha blending backward propagation
__global__ void alphaBlendingBackwardKernel(
    const float* grad_channel_real, const float* grad_channel_imag,
    const float* influences, const float* real_contributions,
    const float* imag_contributions, const float* opacity,
    const int64_t* sort_indices, float* grad_real_contributions,
    float* grad_imag_contributions, float* grad_opacity, float* grad_influences,
    int num_tx, int num_rx, int N) {
    int i = blockIdx.x;
    int j = threadIdx.x;

    if (i >= num_tx || j >= num_rx) return;

    float grad_real = grad_channel_real[i * num_rx + j];
    float grad_imag = grad_channel_imag[i * num_rx + j];

    float transmittance = 1.0f;
    float grad_alpha_eff_sum_real = 0.0f;
    float grad_alpha_eff_sum_imag = 0.0f;

    // Backward scan through sorted Gaussians
    for (int k = 0; k < N; k++) {
        int idx = sort_indices[k];
        float infl = influences[idx * num_tx * num_rx + i * num_rx + j];
        float effective_opacity = opacity[idx] * infl;

        // Clamp for numerical stability
        effective_opacity =
            fmaxf(fminf(effective_opacity, 1.0f - 1e-5f), 1e-5f);

        // Gradient for contributions
        float grad_contrib_real = transmittance * effective_opacity * grad_real;
        float grad_contrib_imag = transmittance * effective_opacity * grad_imag;

        atomicAdd(&grad_real_contributions[idx], grad_contrib_real);
        atomicAdd(&grad_imag_contributions[idx], grad_contrib_imag);

        // Gradient for effective opacity
        float real_term = transmittance * real_contributions[idx] * grad_real;
        float imag_term = transmittance * imag_contributions[idx] * grad_imag;
        float grad_effective_opacity = real_term + imag_term;

        // Add contribution from gradient accumulation term
        if (effective_opacity < 1.0f - 1e-5f) {
            grad_effective_opacity -=
                (grad_alpha_eff_sum_real + grad_alpha_eff_sum_imag) /
                (1.0f - effective_opacity);
        }

        // Update gradient accumulators for next iteration
        grad_alpha_eff_sum_real +=
            effective_opacity * real_contributions[idx] * grad_real;
        grad_alpha_eff_sum_imag +=
            effective_opacity * imag_contributions[idx] * grad_imag;

        // Split effective opacity gradient to opacity and influence
        float grad_opacity_term = infl * grad_effective_opacity;
        float grad_influence_term = opacity[idx] * grad_effective_opacity;

        // Accumulate gradients
        atomicAdd(&grad_opacity[idx], grad_opacity_term);
        atomicAdd(&grad_influences[idx * num_tx * num_rx + i * num_rx + j],
                  grad_influence_term);

        // Update transmittance for next gaussian
        transmittance *= (1.0f - effective_opacity);

        // Early termination if transmittance is near zero
        if (transmittance < 1e-5f) break;
    }
}

// Compute Gaussian influence backward
__global__ void computeGaussianInfluenceBackwardKernel(
    const float* grad_influences, const float* uv, const float* cov2d,
    float* grad_uv, float* grad_cov2d, int num_tx, int num_rx, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Extract gaussian properties
    float u = uv[idx * 2];
    float v = uv[idx * 2 + 1];
    float cov[4] = {cov2d[idx * 4 + 0], cov2d[idx * 4 + 1], cov2d[idx * 4 + 2],
                    cov2d[idx * 4 + 3]};

    // Invert covariance matrix
    float inv_cov[4];
    float det;
    bool success = invert_2x2(cov, inv_cov, &det);
    if (!success) return;

    // Initialize accumulators for gradients
    float grad_u_accum = 0.0f;
    float grad_v_accum = 0.0f;
    float grad_cov_xx_accum = 0.0f;
    float grad_cov_xy_accum = 0.0f;
    float grad_cov_yx_accum = 0.0f;
    float grad_cov_yy_accum = 0.0f;

    // Process each channel element
    for (int i = 0; i < num_tx; i++) {
        for (int j = 0; j < num_rx; j++) {
            float grad_influence =
                grad_influences[idx * num_tx * num_rx + i * num_rx + j];

            if (fabsf(grad_influence) < 1e-10f) {
                continue;  // Skip insignificant gradients
            }

            // Compute displacement vector d_ij
            float antenna_pos_x = i + 0.5f;
            float antenna_pos_y = j + 0.5f;
            float d_x = u - antenna_pos_x;
            float d_y = v - antenna_pos_y;

            // Compute Mahalanobis distance
            float m_dist = mahalanobis_distance(inv_cov, d_x, d_y);

            // Original influence value
            float influence = expf(-0.5f * m_dist);

            // Gradient of influence w.r.t. Mahalanobis distance
            float grad_mdist = -0.5f * influence * grad_influence;

            // Gradient of Mahalanobis distance w.r.t. displacement vector
            float grad_dx =
                2.0f * (inv_cov[0] * d_x + inv_cov[1] * d_y) * grad_mdist;
            float grad_dy =
                2.0f * (inv_cov[1] * d_x + inv_cov[3] * d_y) * grad_mdist;

            // Gradient of displacement vector w.r.t. u, v
            grad_u_accum -= grad_dx;
            grad_v_accum -= grad_dy;

            // Gradient of Mahalanobis distance w.r.t. inverse covariance
            float outer_prod[4] = {d_x * d_x, d_x * d_y, d_y * d_x, d_y * d_y};

            // Gradient for inverse covariance
            float grad_inv_xx = outer_prod[0] * grad_mdist;
            float grad_inv_xy = outer_prod[1] * grad_mdist;
            float grad_inv_yx = outer_prod[2] * grad_mdist;
            float grad_inv_yy = outer_prod[3] * grad_mdist;

            // Convert to gradient w.r.t covariance
            float temp[4] = {0.0f};
            temp[0] = grad_inv_xx * inv_cov[0] + grad_inv_xy * inv_cov[1];
            temp[1] = grad_inv_xx * inv_cov[2] + grad_inv_xy * inv_cov[3];
            temp[2] = grad_inv_yx * inv_cov[0] + grad_inv_yy * inv_cov[1];
            temp[3] = grad_inv_yx * inv_cov[2] + grad_inv_yy * inv_cov[3];

            grad_cov_xx_accum -= inv_cov[0] * temp[0] + inv_cov[1] * temp[2];
            grad_cov_xy_accum -= inv_cov[0] * temp[1] + inv_cov[1] * temp[3];
            grad_cov_yx_accum -= inv_cov[2] * temp[0] + inv_cov[3] * temp[2];
            grad_cov_yy_accum -= inv_cov[2] * temp[1] + inv_cov[3] * temp[3];
        }
    }

    // Write accumulated gradients
    atomicAdd(grad_uv + idx * 2, grad_u_accum);
    atomicAdd(grad_uv + idx * 2 + 1, grad_v_accum);

    atomicAdd(grad_cov2d + idx * 4 + 0, grad_cov_xx_accum);
    atomicAdd(grad_cov2d + idx * 4 + 1, grad_cov_xy_accum);
    atomicAdd(grad_cov2d + idx * 4 + 2, grad_cov_yx_accum);
    atomicAdd(grad_cov2d + idx * 4 + 3, grad_cov_yy_accum);
}

// Project 3D covariance to 2D (backward)
__global__ void projectCov3dToCov2dBackwardKernel(const float* grad_cov2d,
                                                  const float* cov3d_mat,
                                                  const float* jacobian,
                                                  float* grad_cov3d,
                                                  float* grad_jacobian, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Extract 2x2 gradient matrix
    float g_cov2d[4] = {grad_cov2d[idx * 4 + 0], grad_cov2d[idx * 4 + 1],
                        grad_cov2d[idx * 4 + 2], grad_cov2d[idx * 4 + 3]};

    // Extract 3x3 covariance matrix
    float cov3d[9];
    for (int i = 0; i < 9; i++) {
        cov3d[i] = cov3d_mat[idx * 9 + i];
    }

    // Extract 2x3 Jacobian
    float J[6];
    for (int i = 0; i < 6; i++) {
        J[i] = jacobian[idx * 6 + i];
    }

    // Gradient w.r.t. 3D covariance: ∂L/∂Σ3D = J^T · (∂L/∂Σ2D) · J
    float temp[6];  // 3x2 matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            temp[i * 2 + j] = 0.0f;
            for (int k = 0; k < 2; k++) {
                temp[i * 2 + j] += J[k * 3 + i] * g_cov2d[k * 2 + j];
            }
        }
    }

    float g_cov3d[9];  // 3x3 matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            g_cov3d[i * 3 + j] = 0.0f;
            for (int k = 0; k < 2; k++) {
                g_cov3d[i * 3 + j] += temp[i * 2 + k] * J[k * 3 + j];
            }
        }
    }

    // Write 3D covariance gradient
    for (int i = 0; i < 9; i++) {
        atomicAdd(grad_cov3d + idx * 9 + i, g_cov3d[i]);
    }

    // Gradient w.r.t. Jacobian: ∂L/∂J = 2 · (∂L/∂Σ2D) · J · Σ3D
    float temp2[6];  // 2x3 matrix
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            temp2[i * 3 + j] = 0.0f;
            for (int k = 0; k < 2; k++) {
                temp2[i * 3 + j] += g_cov2d[i * 2 + k] * J[k * 3 + j];
            }
        }
    }

    float g_J[6];  // 2x3 matrix
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            g_J[i * 3 + j] = 0.0f;
            for (int k = 0; k < 3; k++) {
                g_J[i * 3 + j] += temp2[i * 3 + k] * cov3d[k * 3 + j];
            }
        }
    }

    // Scale by 2 and write Jacobian gradient
    for (int i = 0; i < 6; i++) {
        atomicAdd(grad_jacobian + idx * 6 + i, 2.0f * g_J[i]);
    }
}

// Jacobian computation backward
__global__ void computeJacobianBackwardKernel(const float* grad_jacobian,
                                              const float* d, const float* r,
                                              float* grad_d, float* grad_r,
                                              int num_tx, int num_rx, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Extract values needed for gradient computation
    float x = d[idx * 3];
    float y = d[idx * 3 + 1];
    float z = d[idx * 3 + 2];
    float r_val = r[idx];

    // Extract gradients for this Jacobian
    float g_J[6];
    for (int i = 0; i < 6; i++) {
        g_J[i] = grad_jacobian[idx * 6 + i];
    }

    // Compute intermediate values
    float xy_squared = x * x + y * y;
    xy_squared = fmaxf(xy_squared, 1e-10f);

    float cos_lat = sqrtf(1.0f - (z / r_val) * (z / r_val));
    cos_lat = fmaxf(cos_lat, 1e-10f);

    float tx_factor = float(num_tx - 1) / (2.0f * PI);
    float rx_factor = float(num_rx - 1) / PI;

    float r_cos_lat_xy = r_val * cos_lat * xy_squared;
    r_cos_lat_xy = fmaxf(r_cos_lat_xy, 1e-10f);

    // Initialize gradient accumulators
    float grad_x = 0.0f;
    float grad_y = 0.0f;
    float grad_z = 0.0f;
    float grad_r_val = 0.0f;

    // Gradient for first row of Jacobian (du/d*)
    float dJ00_dx = tx_factor * (2.0f * x * y / (xy_squared * xy_squared));
    float dJ00_dy = tx_factor * ((y * y - x * x) / (xy_squared * xy_squared));

    grad_x += g_J[0] * dJ00_dx;
    grad_y += g_J[0] * dJ00_dy;

    float dJ01_dx = tx_factor * ((y * y - x * x) / (xy_squared * xy_squared));
    float dJ01_dy = tx_factor * (-2.0f * x * y / (xy_squared * xy_squared));

    grad_x += g_J[1] * dJ01_dx;
    grad_y += g_J[1] * dJ01_dy;

    // Gradient for second row of Jacobian (dv/d*)
    float z_over_r = z / r_val;
    float factor1 = rx_factor / r_cos_lat_xy;
    float factor2 = rx_factor / (r_val * cos_lat);

    float dJ10_dx = factor1 * z - factor1 * z * (2.0f * x * x) / xy_squared;
    float dJ10_dy = -factor1 * z * (2.0f * x * y) / xy_squared;
    float dJ10_dz = factor1 * x;
    float dJ10_dr = -factor1 * (z * x) / r_val;

    grad_x += g_J[3] * dJ10_dx;
    grad_y += g_J[3] * dJ10_dy;
    grad_z += g_J[3] * dJ10_dz;
    grad_r_val += g_J[3] * dJ10_dr;

    float dJ11_dx = -factor1 * z * (2.0f * x * y) / xy_squared;
    float dJ11_dy = factor1 * z - factor1 * z * (2.0f * y * y) / xy_squared;
    float dJ11_dz = factor1 * y;
    float dJ11_dr = -factor1 * (z * y) / r_val;

    grad_x += g_J[4] * dJ11_dx;
    grad_y += g_J[4] * dJ11_dy;
    grad_z += g_J[4] * dJ11_dz;
    grad_r_val += g_J[4] * dJ11_dr;

    float dJ12_dz = factor2 * z_over_r / (cos_lat * cos_lat);
    float dJ12_dr =
        -factor2 / r_val - factor2 * z_over_r * z_over_r / (cos_lat * cos_lat);

    grad_z += g_J[5] * dJ12_dz;
    grad_r_val += g_J[5] * dJ12_dr;

    // Write gradient outputs
    atomicAdd(grad_d + idx * 3 + 0, grad_x);
    atomicAdd(grad_d + idx * 3 + 1, grad_y);
    atomicAdd(grad_d + idx * 3 + 2, grad_z);
    atomicAdd(grad_r + idx, grad_r_val);
}

// Mapping to channel matrix (backward)
__global__ void mapToChannelMatrixBackwardKernel(const float* grad_uv,
                                                 float* grad_s_x,
                                                 float* grad_s_y, int num_tx,
                                                 int num_rx, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Extract gradients
    float grad_u = grad_uv[idx * 2];
    float grad_v = grad_uv[idx * 2 + 1];

    // Compute gradients for s_x and s_y using chain rule
    // u = ((s_x + 1.0f) / 2.0f) * (num_tx - 1) + 0.5f;
    // v = ((s_y + 1.0f) / 2.0f) * (num_rx - 1) + 0.5f;
    float grad_s_x_val = grad_u * (num_tx - 1) / 2.0f;
    float grad_s_y_val = grad_v * (num_rx - 1) / 2.0f;

    // Write gradients
    atomicAdd(grad_s_x + idx, grad_s_x_val);
    atomicAdd(grad_s_y + idx, grad_s_y_val);
}

// Spherical to uniform coordinates (backward)
__global__ void transformToUniformCoordsBackwardKernel(const float* grad_s_x,
                                                       const float* grad_s_y,
                                                       float* grad_longitude,
                                                       float* grad_latitude,
                                                       int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // s_x = longitude / PI -> ds_x/dlongitude = 1/PI
    // s_y = 2.0f * latitude / PI -> ds_y/dlatitude = 2/PI
    float grad_longitude_val = grad_s_x[idx] / PI;
    float grad_latitude_val = grad_s_y[idx] * 2.0f / PI;

    atomicAdd(grad_longitude + idx, grad_longitude_val);
    atomicAdd(grad_latitude + idx, grad_latitude_val);
}

// Spherical coordinates (backward)
__global__ void computeSphericalCoordsBackwardKernel(
    const float* grad_longitude, const float* grad_latitude,
    const float* grad_r_spherical, const float* d, const float* r,
    float* grad_d, float* grad_r, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Extract displacement and distance
    float x = d[idx * 3];
    float y = d[idx * 3 + 1];
    float z = d[idx * 3 + 2];
    float r_val = r[idx];

    // Compute intermediate values
    float xy_squared = x * x + y * y;
    xy_squared = fmaxf(xy_squared, 1e-10f);
    float r_squared = r_val * r_val;
    float sqrt_r_squared_minus_z_squared =
        sqrtf(fmaxf(r_squared - z * z, 1e-10f));

    // Gradient for longitude: longitude = atan2(y, x)
    float grad_x_from_longitude = -grad_longitude[idx] * y / xy_squared;
    float grad_y_from_longitude = grad_longitude[idx] * x / xy_squared;

    // Gradient for latitude: latitude = asin(z / r)
    float grad_z_from_latitude =
        grad_latitude[idx] / sqrt_r_squared_minus_z_squared;
    float grad_r_from_latitude =
        -grad_latitude[idx] * z / (r_val * sqrt_r_squared_minus_z_squared);

    // Add gradients from coordinate path
    atomicAdd(grad_d + idx * 3 + 0, grad_x_from_longitude);
    atomicAdd(grad_d + idx * 3 + 1, grad_y_from_longitude);
    atomicAdd(grad_d + idx * 3 + 2, grad_z_from_latitude);
    atomicAdd(grad_r + idx, grad_r_from_latitude + grad_r_spherical[idx]);
}

// Receiver distance (backward)
__global__ void computeDistancesToReceiverBackwardKernel(const float* grad_r,
                                                         const float* d,
                                                         const float* r,
                                                         float* grad_d, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Extract displacement and distance
    float x = d[idx * 3];
    float y = d[idx * 3 + 1];
    float z = d[idx * 3 + 2];
    float r_val = r[idx];

    // Compute gradients: r = sqrt(x^2 + y^2 + z^2)
    float grad_factor = grad_r[idx] / fmaxf(r_val, 1e-10f);
    float grad_x = x * grad_factor;
    float grad_y = y * grad_factor;
    float grad_z = z * grad_factor;

    // Add gradients from distance path
    atomicAdd(grad_d + idx * 3 + 0, grad_x);
    atomicAdd(grad_d + idx * 3 + 1, grad_y);
    atomicAdd(grad_d + idx * 3 + 2, grad_z);
}

// Wireless channel contribution backward
__global__ void computeChannelBackwardKernel(
    const float* grad_real_contributions, const float* grad_imag_contributions,
    const float* attenuation, const float* phase_rotation,
    const float* xyz_rx_distance, float* grad_attenuation,
    float* grad_phase_rotation, float* grad_xyz_rx_distance, float wavelength,
    int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float distance = xyz_rx_distance[idx];
    float att = attenuation[idx];
    float phase = phase_rotation[idx];

    // Get incoming gradients
    float grad_real = grad_real_contributions[idx];
    float grad_imag = grad_imag_contributions[idx];

    // Original forward computation
    float path_loss = wavelength / (4.0f * PI * distance);
    float phase_shift = -2.0f * PI * distance / wavelength;
    float total_phase = phase + phase_shift;

    // Compute gradient w.r.t. attenuation
    float real_contrib = path_loss * cosf(total_phase);
    float imag_contrib = path_loss * sinf(total_phase);
    float grad_att = grad_real * real_contrib + grad_imag * imag_contrib;
    atomicAdd(grad_attenuation + idx, grad_att);

    // Compute gradient w.r.t. phase rotation
    float grad_phase = -grad_real * imag_contrib + grad_imag * real_contrib;
    atomicAdd(grad_phase_rotation + idx, grad_phase);

    // Compute gradient w.r.t. distance
    float distance_factor_real =
        -real_contrib / distance + 2.0f * PI * imag_contrib / wavelength;
    float distance_factor_imag =
        -imag_contrib / distance - 2.0f * PI * real_contrib / wavelength;
    float grad_distance =
        grad_real * distance_factor_real + grad_imag * distance_factor_imag;
    atomicAdd(grad_xyz_rx_distance + idx, grad_distance);
}

// 3D covariance matrix backward
__global__ void computeCov3dBackwardKernel(
    const float* grad_cov3d, const float* scaling, const float* rotation,
    float* grad_scaling, float* grad_rotation, float scale_modifier, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Extract rotation quaternion
    float qr = rotation[idx * 4];  // real part
    float qx = rotation[idx * 4 + 1];
    float qy = rotation[idx * 4 + 2];
    float qz = rotation[idx * 4 + 3];

    // Extract scaling
    float sx = scaling[idx * 3] * scale_modifier;
    float sy = scaling[idx * 3 + 1] * scale_modifier;
    float sz = scaling[idx * 3 + 2] * scale_modifier;

    // Compute rotation matrix from quaternion
    float R[9];
    R[0] = 1.0f - 2.0f * (qy * qy + qz * qz);
    R[1] = 2.0f * (qx * qy - qr * qz);
    R[2] = 2.0f * (qx * qz + qr * qy);
    R[3] = 2.0f * (qx * qy + qr * qz);
    R[4] = 1.0f - 2.0f * (qx * qx + qz * qz);
    R[5] = 2.0f * (qy * qz - qr * qx);
    R[6] = 2.0f * (qx * qz - qr * qy);
    R[7] = 2.0f * (qy * qz + qr * qx);
    R[8] = 1.0f - 2.0f * (qx * qx + qy * qy);

    // Extract symmetric 3D covariance gradient
    float g_cov3d[9];
    for (int i = 0; i < 9; i++) {
        g_cov3d[i] = grad_cov3d[idx * 9 + i];
    }

    // Compute scaling gradients: ∂L/∂s_i = 2s_i (R^T · ∂L/∂Σ3D · R)_ii
    float RT_g_cov3d[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            RT_g_cov3d[i * 3 + j] = 0.0f;
            for (int k = 0; k < 3; k++) {
                RT_g_cov3d[i * 3 + j] += R[k * 3 + i] * g_cov3d[k * 3 + j];
            }
        }
    }

    float RT_g_cov3d_R[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            RT_g_cov3d_R[i * 3 + j] = 0.0f;
            for (int k = 0; k < 3; k++) {
                RT_g_cov3d_R[i * 3 + j] += RT_g_cov3d[i * 3 + k] * R[j * 3 + k];
            }
        }
    }

    float grad_sx = 2.0f * sx * RT_g_cov3d_R[0 * 3 + 0] * scale_modifier;
    float grad_sy = 2.0f * sy * RT_g_cov3d_R[1 * 3 + 1] * scale_modifier;
    float grad_sz = 2.0f * sz * RT_g_cov3d_R[2 * 3 + 2] * scale_modifier;

    atomicAdd(grad_scaling + idx * 3 + 0, grad_sx);
    atomicAdd(grad_scaling + idx * 3 + 1, grad_sy);
    atomicAdd(grad_scaling + idx * 3 + 2, grad_sz);

    // Compute rotation gradients: ∂L/∂R = 2 · ∂L/∂Σ3D · R · S^2
    float S2[9] = {0.0f};
    S2[0 * 3 + 0] = sx * sx;
    S2[1 * 3 + 1] = sy * sy;
    S2[2 * 3 + 2] = sz * sz;

    float g_cov3d_R[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            g_cov3d_R[i * 3 + j] = 0.0f;
            for (int k = 0; k < 3; k++) {
                g_cov3d_R[i * 3 + j] += g_cov3d[i * 3 + k] * R[k * 3 + j];
            }
        }
    }

    float g_R[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            g_R[i * 3 + j] = 0.0f;
            for (int k = 0; k < 3; k++) {
                g_R[i * 3 + j] += g_cov3d_R[i * 3 + k] * S2[k * 3 + j];
            }
        }
    }

    for (int i = 0; i < 9; i++) {
        g_R[i] *= 2.0f;
    }

    // Compute quaternion gradients using chain rule
    float grad_qr = 2.0f * (g_R[0 * 3 + 1] * (-qz) + g_R[0 * 3 + 2] * qy +
                            g_R[1 * 3 + 0] * qz + g_R[1 * 3 + 2] * (-qx) +
                            g_R[2 * 3 + 0] * (-qy) + g_R[2 * 3 + 1] * qx);

    float grad_qx =
        2.0f * (g_R[0 * 3 + 1] * qy + g_R[0 * 3 + 2] * qz +
                g_R[1 * 3 + 0] * qy + g_R[1 * 3 + 1] * (-2.0f * qx) +
                g_R[1 * 3 + 2] * (-qr) + g_R[2 * 3 + 0] * qz +
                g_R[2 * 3 + 1] * qr + g_R[2 * 3 + 2] * (-2.0f * qx));

    float grad_qy =
        2.0f * (g_R[0 * 3 + 0] * (-2.0f * qy) + g_R[0 * 3 + 1] * qx +
                g_R[0 * 3 + 2] * qr + g_R[1 * 3 + 0] * qx +
                g_R[1 * 3 + 2] * qz + g_R[2 * 3 + 0] * (-qr) +
                g_R[2 * 3 + 1] * qz + g_R[2 * 3 + 2] * (-2.0f * qy));

    float grad_qz =
        2.0f * (g_R[0 * 3 + 0] * (-2.0f * qz) + g_R[0 * 3 + 1] * (-qr) +
                g_R[0 * 3 + 2] * qx + g_R[1 * 3 + 0] * qr +
                g_R[1 * 3 + 1] * (-2.0f * qz) + g_R[1 * 3 + 2] * qy +
                g_R[2 * 3 + 0] * qx + g_R[2 * 3 + 1] * qy +
                g_R[2 * 3 + 2] * (-2.0f * qz));

    atomicAdd(grad_rotation + idx * 4 + 0, grad_qr);
    atomicAdd(grad_rotation + idx * 4 + 1, grad_qx);
    atomicAdd(grad_rotation + idx * 4 + 2, grad_qy);
    atomicAdd(grad_rotation + idx * 4 + 3, grad_qz);
}

// Wrapper function to prepare gradient from output
torch::Tensor prepareGradOutputCUDA(const torch::Tensor& grad_output,
                                    int num_tx, int num_rx) {
    const at::cuda::CUDAGuard device_guard(grad_output.device());

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(grad_output.device());

    torch::Tensor grad_channel_real = torch::empty({num_tx, num_rx}, options);
    torch::Tensor grad_channel_imag = torch::empty({num_tx, num_rx}, options);

    int threads = 256;
    int total_elements = num_tx * num_rx;
    int blocks = (total_elements + threads - 1) / threads;

    prepareGradOutputKernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(), grad_channel_real.data_ptr<float>(),
        grad_channel_imag.data_ptr<float>(), num_tx, num_rx);

    return torch::cat({grad_channel_real, grad_channel_imag}, 0);
}

// Wrapper function for alpha blending backward
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
alphaBlendingBackwardCUDA(const torch::Tensor& grad_channel,
                          const torch::Tensor& influences,
                          const torch::Tensor& contributions,
                          const torch::Tensor& opacity,
                          const torch::Tensor& sort_indices, int num_tx,
                          int num_rx) {
    const at::cuda::CUDAGuard device_guard(grad_channel.device());

    int N = influences.size(0);

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(grad_channel.device());

    torch::Tensor grad_channel_real = grad_channel.slice(0, 0, num_tx);
    torch::Tensor grad_channel_imag = grad_channel.slice(0, num_tx, 2 * num_tx);

    torch::Tensor real_contributions = torch::real(contributions);
    torch::Tensor imag_contributions = torch::imag(contributions);

    torch::Tensor grad_real_contributions = torch::zeros({N}, options);
    torch::Tensor grad_imag_contributions = torch::zeros({N}, options);
    torch::Tensor grad_opacity = torch::zeros({N}, options);
    torch::Tensor grad_influences = torch::zeros({N, num_tx, num_rx}, options);

    int threads = num_rx > 1024 ? 256 : num_rx;
    int blocks = num_tx;

    alphaBlendingBackwardKernel<<<blocks, threads>>>(
        grad_channel_real.data_ptr<float>(),
        grad_channel_imag.data_ptr<float>(), influences.data_ptr<float>(),
        real_contributions.data_ptr<float>(),
        imag_contributions.data_ptr<float>(), opacity.data_ptr<float>(),
        sort_indices.data_ptr<int64_t>(),
        grad_real_contributions.data_ptr<float>(),
        grad_imag_contributions.data_ptr<float>(),
        grad_opacity.data_ptr<float>(), grad_influences.data_ptr<float>(),
        num_tx, num_rx, N);

    torch::Tensor grad_contributions =
        torch::complex(grad_real_contributions, grad_imag_contributions);

    return std::make_tuple(grad_contributions, grad_opacity, grad_influences,
                           sort_indices);
}

// Wrapper function for channel computation backward
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
computeChannelBackwardCUDA(const torch::Tensor& grad_real_contributions,
                           const torch::Tensor& grad_imag_contributions,
                           const torch::Tensor& attenuation,
                           const torch::Tensor& phase_rotation,
                           const torch::Tensor& xyz_rx_distance,
                           float wavelength) {
    const at::cuda::CUDAGuard device_guard(attenuation.device());

    int N = attenuation.size(0);

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(attenuation.device());

    torch::Tensor grad_attenuation = torch::zeros({N, 1}, options);
    torch::Tensor grad_phase_rotation = torch::zeros({N, 1}, options);
    torch::Tensor grad_xyz_rx_distance = torch::zeros({N}, options);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    computeChannelBackwardKernel<<<blocks, threads>>>(
        grad_real_contributions.data_ptr<float>(),
        grad_imag_contributions.data_ptr<float>(),
        attenuation.data_ptr<float>(), phase_rotation.data_ptr<float>(),
        xyz_rx_distance.data_ptr<float>(), grad_attenuation.data_ptr<float>(),
        grad_phase_rotation.data_ptr<float>(),
        grad_xyz_rx_distance.data_ptr<float>(), wavelength, N);

    return std::make_tuple(grad_attenuation, grad_phase_rotation,
                           grad_xyz_rx_distance);
}

// Wrapper function for Gaussian influence backward
std::tuple<torch::Tensor, torch::Tensor> computeGaussianInfluenceBackwardCUDA(
    const torch::Tensor& grad_influences, const torch::Tensor& uv,
    const torch::Tensor& cov2d, int num_tx, int num_rx) {
    const at::cuda::CUDAGuard device_guard(grad_influences.device());

    int N = uv.size(0);

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(grad_influences.device());

    torch::Tensor grad_uv = torch::zeros({N, 2}, options);
    torch::Tensor grad_cov2d = torch::zeros({N, 2, 2}, options);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    computeGaussianInfluenceBackwardKernel<<<blocks, threads>>>(
        grad_influences.data_ptr<float>(), uv.data_ptr<float>(),
        cov2d.data_ptr<float>(), grad_uv.data_ptr<float>(),
        grad_cov2d.data_ptr<float>(), num_tx, num_rx, N);

    return std::make_tuple(grad_uv, grad_cov2d);
}

// Wrapper function for 3D to 2D covariance projection backward
std::tuple<torch::Tensor, torch::Tensor> projectCov3dToCov2dBackwardCUDA(
    const torch::Tensor& grad_cov2d, const torch::Tensor& cov3d_mat,
    const torch::Tensor& jacobian) {
    const at::cuda::CUDAGuard device_guard(grad_cov2d.device());

    int N = cov3d_mat.size(0);

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(grad_cov2d.device());

    torch::Tensor grad_cov3d = torch::zeros({N, 3, 3}, options);
    torch::Tensor grad_jacobian = torch::zeros({N, 2, 3}, options);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    projectCov3dToCov2dBackwardKernel<<<blocks, threads>>>(
        grad_cov2d.data_ptr<float>(), cov3d_mat.data_ptr<float>(),
        jacobian.data_ptr<float>(), grad_cov3d.data_ptr<float>(),
        grad_jacobian.data_ptr<float>(), N);

    return std::make_tuple(grad_cov3d, grad_jacobian);
}

// Wrapper function for Jacobian computation backward
std::tuple<torch::Tensor, torch::Tensor> computeJacobianBackwardCUDA(
    const torch::Tensor& grad_jacobian, const torch::Tensor& d,
    const torch::Tensor& r, int num_tx, int num_rx) {
    const at::cuda::CUDAGuard device_guard(grad_jacobian.device());

    int N = d.size(0);

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(grad_jacobian.device());

    torch::Tensor grad_d = torch::zeros({N, 3}, options);
    torch::Tensor grad_r = torch::zeros({N}, options);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    computeJacobianBackwardKernel<<<blocks, threads>>>(
        grad_jacobian.data_ptr<float>(), d.data_ptr<float>(),
        r.data_ptr<float>(), grad_d.data_ptr<float>(), grad_r.data_ptr<float>(),
        num_tx, num_rx, N);

    return std::make_tuple(grad_d, grad_r);
}

// Wrapper function for channel matrix mapping backward
torch::Tensor mapToChannelMatrixBackwardCUDA(const torch::Tensor& grad_uv,
                                             int num_tx, int num_rx) {
    const at::cuda::CUDAGuard device_guard(grad_uv.device());

    int N = grad_uv.size(0);

    auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(grad_uv.device());

    torch::Tensor grad_s_x = torch::zeros({N}, options);
    torch::Tensor grad_s_y = torch::zeros({N}, options);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    mapToChannelMatrixBackwardKernel<<<blocks, threads>>>(
        grad_uv.data_ptr<float>(), grad_s_x.data_ptr<float>(),
        grad_s_y.data_ptr<float>(), num_tx, num_rx, N);

    return torch::stack({grad_s_x, grad_s_y}, 1);
}

// Wrapper function for uniform coordinates transformation backward
torch::Tensor transformToUniformCoordsBackwardCUDA(
    const torch::Tensor& grad_s_x, const torch::Tensor& grad_s_y) {
    const at::cuda::CUDAGuard device_guard(grad_s_x.device());

    int N = grad_s_x.size(0);

    auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(grad_s_x.device());

    torch::Tensor grad_longitude = torch::zeros({N}, options);
    torch::Tensor grad_latitude = torch::zeros({N}, options);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    transformToUniformCoordsBackwardKernel<<<blocks, threads>>>(
        grad_s_x.data_ptr<float>(), grad_s_y.data_ptr<float>(),
        grad_longitude.data_ptr<float>(), grad_latitude.data_ptr<float>(), N);

    return torch::stack({grad_longitude, grad_latitude}, 1);
}

// Wrapper function for spherical coordinates backward
std::tuple<torch::Tensor, torch::Tensor> computeSphericalCoordsBackwardCUDA(
    const torch::Tensor& grad_longitude, const torch::Tensor& grad_latitude,
    const torch::Tensor& grad_r_spherical, const torch::Tensor& d,
    const torch::Tensor& r) {
    const at::cuda::CUDAGuard device_guard(grad_longitude.device());

    int N = d.size(0);

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(grad_longitude.device());

    torch::Tensor grad_d = torch::zeros({N, 3}, options);
    torch::Tensor grad_r = torch::zeros({N}, options);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    computeSphericalCoordsBackwardKernel<<<blocks, threads>>>(
        grad_longitude.data_ptr<float>(), grad_latitude.data_ptr<float>(),
        grad_r_spherical.data_ptr<float>(), d.data_ptr<float>(),
        r.data_ptr<float>(), grad_d.data_ptr<float>(), grad_r.data_ptr<float>(),
        N);

    return std::make_tuple(grad_d, grad_r);
}

// Wrapper function for distance to receiver backward
torch::Tensor computeDistancesToReceiverBackwardCUDA(
    const torch::Tensor& grad_r, const torch::Tensor& d,
    const torch::Tensor& r) {
    const at::cuda::CUDAGuard device_guard(grad_r.device());

    int N = d.size(0);

    auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(grad_r.device());

    torch::Tensor grad_d = torch::zeros({N, 3}, options);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    computeDistancesToReceiverBackwardKernel<<<blocks, threads>>>(
        grad_r.data_ptr<float>(), d.data_ptr<float>(), r.data_ptr<float>(),
        grad_d.data_ptr<float>(), N);

    return grad_d;
}

// Wrapper function for 3D covariance matrix backward
std::tuple<torch::Tensor, torch::Tensor> computeCov3dBackwardCUDA(
    const torch::Tensor& grad_cov3d, const torch::Tensor& scaling,
    const torch::Tensor& rotation, float scale_modifier) {
    const at::cuda::CUDAGuard device_guard(grad_cov3d.device());

    int N = scaling.size(0);

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(grad_cov3d.device());

    torch::Tensor grad_scaling = torch::zeros({N, 3}, options);
    torch::Tensor grad_rotation = torch::zeros({N, 4}, options);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    computeCov3dBackwardKernel<<<blocks, threads>>>(
        grad_cov3d.data_ptr<float>(), scaling.data_ptr<float>(),
        rotation.data_ptr<float>(), grad_scaling.data_ptr<float>(),
        grad_rotation.data_ptr<float>(), scale_modifier, N);

    return std::make_tuple(grad_scaling, grad_rotation);
}

// Main backward function that integrates all steps
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
rasterizeBackwardCUDA(const torch::Tensor& grad_output,
                      const torch::Tensor& points, const torch::Tensor& cov3d,
                      const torch::Tensor& attenuation,
                      const torch::Tensor& phase_rotation,
                      const torch::Tensor& opacity,
                      const torch::Tensor& receiver,
                      const torch::Tensor& transmitter, int num_tx, int num_rx,
                      float frequency) {
    const at::cuda::CUDAGuard device_guard(points.device());

    // Constants
    float c = 299792458.0f;
    float wavelength = c / frequency;

    // Prepare gradient output (split real/imag)
    torch::Tensor grad_channel =
        prepareGradOutputCUDA(grad_output, num_tx, num_rx);

    // Recompute intermediate values from forward pass
    auto [distances, uv, cov2d] =
        projectToChannelSpaceCUDA(points, cov3d, receiver, num_tx, num_rx);

    // Sort points by distance
    auto sort_result = distances.sort(0);
    torch::Tensor sort_indices = std::get<1>(sort_result);

    // Compute Gaussian influence
    torch::Tensor influences =
        computeGaussianInfluenceCUDA(uv, cov2d, num_tx, num_rx);

    // Compute channel contribution
    auto [real_contributions, imag_contributions] =
        computeChannelCUDA(attenuation, phase_rotation, distances, wavelength);

    // Convert to complex tensor for backward
    torch::Tensor contributions = torch::complex(
        real_contributions.squeeze(-1), imag_contributions.squeeze(-1));

    // 1. Backward pass for alpha blending
    auto [grad_contributions, grad_opacity, grad_influences, _] =
        alphaBlendingBackwardCUDA(grad_channel, influences, contributions,
                                  opacity.squeeze(-1), sort_indices, num_tx,
                                  num_rx);

    // 2. Backward pass for channel computation
    auto [grad_attenuation, grad_phase_rotation, grad_distances_channel] =
        computeChannelBackwardCUDA(torch::real(grad_contributions),
                                   torch::imag(grad_contributions), attenuation,
                                   phase_rotation, distances, wavelength);

    // 3. Backward pass for Gaussian influence
    auto [grad_uv, grad_cov2d] = computeGaussianInfluenceBackwardCUDA(
        grad_influences, uv, cov2d, num_tx, num_rx);

    // 4. Backward pass for 3D to 2D covariance projection
    torch::Tensor cov3d_mat = torch::empty(
        {points.size(0), 3, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(points.device()));

    // Convert compact -> full for cov3D matrix
    int threads = 256;
    int blocks = (points.size(0) + threads - 1) / threads;

    convertCompactToFullKernel<<<blocks, threads>>>(
        cov3d.data_ptr<float>(), cov3d_mat.data_ptr<float>(), points.size(0));

    // Recompute Jacobian
    torch::Tensor jacobian = computeJacobianCUDA(points - receiver.view({1, 3}),
                                                 distances, num_tx, num_rx);

    auto [grad_cov3d_mat, grad_jacobian] =
        projectCov3dToCov2dBackwardCUDA(grad_cov2d, cov3d_mat, jacobian);

    // 5. Backward pass for Jacobian computation
    auto [grad_d_jacobian, grad_r_jacobian] = computeJacobianBackwardCUDA(
        grad_jacobian, points - receiver.view({1, 3}), distances, num_tx,
        num_rx);

    // 6. Backward pass for mapping to channel matrix
    torch::Tensor grad_uniform_coords =
        mapToChannelMatrixBackwardCUDA(grad_uv, num_tx, num_rx);

    // 7. Backward pass for uniform coordinates transformation
    torch::Tensor grad_spherical_coords = transformToUniformCoordsBackwardCUDA(
        grad_uniform_coords.select(1, 0), grad_uniform_coords.select(1, 1));

    // 8. Backward pass for spherical coordinates
    auto [grad_d_spherical, grad_r_spherical] =
        computeSphericalCoordsBackwardCUDA(
            grad_spherical_coords.select(1, 0),
            grad_spherical_coords.select(1, 1), grad_r_jacobian,
            points - receiver.view({1, 3}), distances);

    // 9. Backward pass for distance to receiver
    torch::Tensor grad_d_distance = computeDistancesToReceiverBackwardCUDA(
        grad_distances_channel + grad_r_spherical,
        points - receiver.view({1, 3}), distances);

    // 10. Combine gradients for displacement vector and convert to position
    // gradient
    torch::Tensor grad_d = grad_d_jacobian + grad_d_spherical + grad_d_distance;
    torch::Tensor grad_points = grad_d;  // d = p - r, so ∂L/∂p = ∂L/∂d

    // 11. Backward pass for 3D covariance matrix
    // Compress full 3x3 grad_cov3d_mat to compact form (6 elements)
    torch::Tensor grad_cov3d_compact = torch::zeros_like(cov3d);

    // Extract upper triangular part (symmetric matrix)
    grad_cov3d_compact.index_put_(
        {torch::arange(points.size(0),
                       torch::TensorOptions().device(points.device())),
         0},
        grad_cov3d_mat.select(1, 0).select(1, 0));  // [0,0]

    grad_cov3d_compact.index_put_(
        {torch::arange(points.size(0),
                       torch::TensorOptions().device(points.device())),
         1},
        grad_cov3d_mat.select(1, 0).select(1, 1));  // [0,1]

    grad_cov3d_compact.index_put_(
        {torch::arange(points.size(0),
                       torch::TensorOptions().device(points.device())),
         2},
        grad_cov3d_mat.select(1, 0).select(1, 2));  // [0,2]

    grad_cov3d_compact.index_put_(
        {torch::arange(points.size(0),
                       torch::TensorOptions().device(points.device())),
         3},
        grad_cov3d_mat.select(1, 1).select(1, 1));  // [1,1]

    grad_cov3d_compact.index_put_(
        {torch::arange(points.size(0),
                       torch::TensorOptions().device(points.device())),
         4},
        grad_cov3d_mat.select(1, 1).select(1, 2));  // [1,2]

    grad_cov3d_compact.index_put_(
        {torch::arange(points.size(0),
                       torch::TensorOptions().device(points.device())),
         5},
        grad_cov3d_mat.select(1, 2).select(1, 2));  // [2,2]

    return std::make_tuple(grad_points, grad_cov3d_compact, grad_attenuation,
                           grad_phase_rotation, grad_opacity);
}
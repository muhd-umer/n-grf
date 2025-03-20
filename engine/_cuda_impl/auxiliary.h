/*
 * Auxiliary header file for channel reconstruction CUDA kernels
 * Contains utility functions and constants used by the CUDA implementation
 */

#ifndef CUDA_WIRELESS_AUXILIARY_H_INCLUDED
#define CUDA_WIRELESS_AUXILIARY_H_INCLUDED

#include <cooperative_groups.h>
#include <cuda.h>
#include <device_launch_parameters.h>

#include "cuda_runtime.h"

namespace cg = cooperative_groups;

// Constants for wireless calculations
#define PI 3.14159265358979323846f
#define SPEED_OF_LIGHT 299792458.0f

// Block size for CUDA kernel launches
// For channel matrices, we adapt tile sizes for wireless channel dimensions
// which are typically of the form (Tx antennas x Rx antennas)
#define MAX_TX_BLOCK_SIZE 32  // Maximum block size for transmit dimension
#define MAX_RX_BLOCK_SIZE 4   // Maximum block size for receive dimension

// Dynamic block size computation based on matrix dimensions
__forceinline__ __host__ __device__ dim3 getOptimalBlockDim(int num_tx, int num_rx) {
    // For small matrices, use smaller blocks
    if (num_tx <= 16) {
        return dim3(4, 2);  // Adjusted for small matrices (e.g., 16x2)
    }
    // For medium-sized matrices
    else if (num_tx <= 32) {
        return dim3(16, 2);
    }
    // For larger matrices
    else {
        return dim3(32, 2);
    }
}

// Helper functions for geometric calculations
__forceinline__ __device__ float3 transformPoint(const float3& p, const float3& receiver) {
    // Compute displacement vector from receiver to point
    return make_float3(p.x - receiver.x, p.y - receiver.y, p.z - receiver.z);
}

__forceinline__ __device__ float computeDistance(const float3& p, const float3& receiver) {
    // Compute Euclidean distance
    float3 d = transformPoint(p, receiver);
    return sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
}

__forceinline__ __device__ void computeSphericalCoords(
    const float3& d,
    float r,
    float& longitude,
    float& latitude) {
    // Compute spherical coordinates from displacement vector
    longitude = atan2f(d.y, d.x);
    latitude = asinf(fminf(fmaxf(d.z / r, -1.0f), 1.0f));
}

__forceinline__ __device__ void transformToUniformCoords(
    float longitude,
    float latitude,
    float& s_x,
    float& s_y) {
    // Transform spherical coordinates to uniform coordinates in [-1,1] range
    s_x = longitude / PI;
    s_y = 2.0f * latitude / PI;
}

__forceinline__ __device__ float2 mapToChannelMatrix(
    float s_x,
    float s_y,
    int num_tx,
    int num_rx) {
    // Map uniform coordinates to channel matrix coordinates
    // Antennas are at half-integer positions
    float u = ((s_x + 1.0f) / 2.0f) * (float)(num_tx - 1) + 0.5f;
    float v = ((s_y + 1.0f) / 2.0f) * (float)(num_rx - 1) + 0.5f;
    return make_float2(u, v);
}

// Jacobian matrix computation for projection from 3D to channel matrix space
__forceinline__ __device__ void computeJacobian(
    const float3& d,
    float r,
    int num_tx,
    int num_rx,
    float* J)  // Output: 2x3 matrix stored as [J00, J01, J02, J10, J11, J12]
{
    float x = d.x;
    float y = d.y;
    float z = d.z;

    // Compute intermediate values with numerical stability
    float xy_squared = x * x + y * y;
    xy_squared = fmaxf(xy_squared, 1e-10f);

    float cos_lat = sqrtf(1.0f - (z / r) * (z / r));
    cos_lat = fmaxf(cos_lat, 1e-10f);

    float tx_factor = (float)(num_tx - 1) / (2.0f * PI);
    float rx_factor = (float)(num_rx - 1) / PI;

    // Jacobian elements for u (channel matrix x-coordinate)
    J[0] = tx_factor * (-y / xy_squared);  // du/dx
    J[1] = tx_factor * (x / xy_squared);   // du/dy
    J[2] = 0.0f;                           // du/dz

    // Jacobian elements for v (channel matrix y-coordinate)
    float r_cos_lat_xy = r * cos_lat * xy_squared;
    r_cos_lat_xy = fmaxf(r_cos_lat_xy, 1e-10f);

    J[3] = rx_factor * (z * x) / r_cos_lat_xy;  // dv/dx
    J[4] = rx_factor * (z * y) / r_cos_lat_xy;  // dv/dy
    J[5] = rx_factor / (r * cos_lat);           // dv/dz
}

// Project 3D covariance to 2D using Jacobian
__forceinline__ __device__ void projectCov3DToCov2D(
    const float* cov3d,  // Input: symmetric 3D covariance (6 values: xx, xy, xz, yy, yz, zz)
    const float* J,      // Input: Jacobian 2x3 matrix (6 values)
    float* cov2d)        // Output: symmetric 2D covariance (3 values: xx, xy, yy)
{
    // Convert compact 3D covariance to full matrix
    float cov3d_full[9];
    cov3d_full[0] = cov3d[0];  // xx
    cov3d_full[1] = cov3d[1];  // xy
    cov3d_full[2] = cov3d[2];  // xz
    cov3d_full[3] = cov3d[1];  // xy (symmetric)
    cov3d_full[4] = cov3d[3];  // yy
    cov3d_full[5] = cov3d[4];  // yz
    cov3d_full[6] = cov3d[2];  // xz (symmetric)
    cov3d_full[7] = cov3d[4];  // yz (symmetric)
    cov3d_full[8] = cov3d[5];  // zz

    // Compute T = Σ * J^T
    float temp[6];  // 3x2 matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            temp[i * 2 + j] = 0.0f;
            for (int k = 0; k < 3; k++) {
                // J is stored as [J00, J01, J02, J10, J11, J12]
                // cov3d_full is stored as [Σ00, Σ01, Σ02, Σ10, Σ11, Σ12, Σ20, Σ21, Σ22]
                // We need to transpose J to get J^T
                temp[i * 2 + j] += cov3d_full[i * 3 + k] * J[j * 3 + k];
            }
        }
    }

    // Compute cov2d = J * T
    float cov2d_full[4];  // 2x2 matrix
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            cov2d_full[i * 2 + j] = 0.0f;
            for (int k = 0; k < 3; k++) {
                cov2d_full[i * 2 + j] += J[i * 3 + k] * temp[k * 2 + j];
            }
        }
    }

    // Store in symmetric form (xx, xy, yy)
    cov2d[0] = cov2d_full[0];  // xx
    cov2d[1] = cov2d_full[1];  // xy (same as cov2d_full[2])
    cov2d[2] = cov2d_full[3];  // yy

    // Add regularization (as in PyTorch implementation)
    cov2d[0] += 0.3f;  // xx
    cov2d[2] += 0.3f;  // yy
}

// Compute inverse of 2D covariance matrix
__forceinline__ __device__ void inverseCov2D(
    const float* cov2d,  // Input: symmetric 2D covariance (3 values: xx, xy, yy)
    float* inv_cov2d)    // Output: inverse 2D covariance (3 values: xx, xy, yy)
{
    float xx = cov2d[0];
    float xy = cov2d[1];
    float yy = cov2d[2];

    float det = xx * yy - xy * xy;
    det = fmaxf(det, 1e-10f);  // Ensure numerical stability
    float inv_det = 1.0f / det;

    inv_cov2d[0] = yy * inv_det;   // xx component of inverse
    inv_cov2d[1] = -xy * inv_det;  // xy component of inverse
    inv_cov2d[2] = xx * inv_det;   // yy component of inverse
}

// Compute wireless channel contribution
__forceinline__ __device__ void computeChannel(
    float attenuation,
    float phase_rotation,
    float distance,
    float wavelength,
    float& real_part,
    float& imag_part) {
    // Compute path loss
    float path_loss = wavelength / (4.0f * PI * distance);

    // Compute phase shift due to propagation
    float phase_shift = -2.0f * PI * distance / wavelength;

    // Combine attenuation and phase
    float total_attenuation = attenuation * path_loss;
    float total_phase = phase_rotation + phase_shift;

    // Convert to real and imaginary parts
    real_part = total_attenuation * cosf(total_phase);
    imag_part = total_attenuation * sinf(total_phase);
}

// Compute Gaussian influence on a specific channel matrix element using Mahalanobis distance
__forceinline__ __device__ float computeGaussianInfluence(
    const float2& uv,
    float antenna_pos_x,
    float antenna_pos_y,
    const float* inv_cov2d) {
    // Compute displacement from Gaussian center to antenna position
    float d_x = uv.x - antenna_pos_x;
    float d_y = uv.y - antenna_pos_y;

    // Compute Mahalanobis distance
    float mahalanobis_dist = inv_cov2d[0] * d_x * d_x +
                             inv_cov2d[2] * d_y * d_y +
                             2.0f * inv_cov2d[1] * d_x * d_y;

    // Apply Gaussian function to get influence
    return expf(-0.5f * mahalanobis_dist);
}

#endif  // CUDA_WIRELESS_AUXILIARY_H_INCLUDED
/*
 * Forward pass implementation for channel reconstruction CUDA kernels
 */

#include <cooperative_groups.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "auxiliary.h"
#include "cuda_runtime.h"
#include "forward.h"

namespace cg = cooperative_groups;

// Compute distances from Gaussians to receiver
__global__ void computeDistancesCUDA(
    int N,                  // Number of Gaussians
    const float* points,    // 3D positions of Gaussians [N, 3]
    const float* receiver,  // Receiver position [3]
    float* distances)       // Output: distances [N]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load point and receiver coordinates
    float3 p = make_float3(points[idx * 3], points[idx * 3 + 1], points[idx * 3 + 2]);
    float3 r = make_float3(receiver[0], receiver[1], receiver[2]);

    // Compute and store distance
    distances[idx] = computeDistance(p, r);
}

// Compute spherical coordinates
__global__ void computeSphericalCoordsCUDA(
    int N,                  // Number of Gaussians
    const float* points,    // 3D positions of Gaussians [N, 3]
    const float* receiver,  // Receiver position [3]
    float* d_out,           // Output: displacement vectors [N, 3]
    float* distances,       // Output: distances [N]
    float* longitude,       // Output: longitude angles [N]
    float* latitude)        // Output: latitude angles [N]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load point and receiver coordinates
    float3 p = make_float3(points[idx * 3], points[idx * 3 + 1], points[idx * 3 + 2]);
    float3 r = make_float3(receiver[0], receiver[1], receiver[2]);

    // Compute displacement vector
    float3 d = transformPoint(p, r);
    float dist = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);

    // Store displacement vector
    d_out[idx * 3] = d.x;
    d_out[idx * 3 + 1] = d.y;
    d_out[idx * 3 + 2] = d.z;

    // Store distance
    distances[idx] = dist;

    // Compute and store spherical coordinates
    float lon, lat;
    computeSphericalCoords(d, dist, lon, lat);

    longitude[idx] = lon;
    latitude[idx] = lat;
}

// Transform spherical coordinates to channel matrix space
__global__ void transformToChannelSpaceCUDA(
    int N,                   // Number of Gaussians
    const float* longitude,  // Longitude angles [N]
    const float* latitude,   // Latitude angles [N]
    int num_tx,              // Number of transmit antennas
    int num_rx,              // Number of receive antennas
    float* uv)               // Output: channel matrix coordinates [N, 2]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Convert spherical to uniform coordinates
    float s_x, s_y;
    transformToUniformCoords(longitude[idx], latitude[idx], s_x, s_y);

    // Map to channel matrix coordinates
    float2 uv_coords = mapToChannelMatrix(s_x, s_y, num_tx, num_rx);

    // Store UV coordinates
    uv[idx * 2] = uv_coords.x;
    uv[idx * 2 + 1] = uv_coords.y;
}

// Compute Jacobian matrices
__global__ void computeJacobiansCUDA(
    int N,                   // Number of Gaussians
    const float* d,          // Displacement vectors [N, 3]
    const float* distances,  // Distances [N]
    int num_tx,              // Number of transmit antennas
    int num_rx,              // Number of receive antennas
    float* jacobians)        // Output: Jacobian matrices [N, 2, 3]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load displacement vector and distance
    float3 disp = make_float3(d[idx * 3], d[idx * 3 + 1], d[idx * 3 + 2]);
    float dist = distances[idx];

    // Compute Jacobian matrix
    float* J = &jacobians[idx * 6];  // 2x3 Jacobian matrix
    computeJacobian(disp, dist, num_tx, num_rx, J);
}

// Project 3D covariance matrices to 2D
__global__ void projectCov3DsCUDA(
    int N,                   // Number of Gaussians
    const float* cov3ds,     // 3D covariance matrices [N, 6]
    const float* jacobians,  // Jacobian matrices [N, 2, 3]
    float* cov2ds)           // Output: 2D covariance matrices [N, 3]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load 3D covariance and Jacobian
    const float* cov3d = &cov3ds[idx * 6];
    const float* J = &jacobians[idx * 6];
    float* cov2d = &cov2ds[idx * 3];

    // Project 3D covariance to 2D
    projectCov3DToCov2D(cov3d, J, cov2d);
}

// Compute inverse 2D covariance matrices
__global__ void computeInvCov2DsCUDA(
    int N,                // Number of Gaussians
    const float* cov2ds,  // 2D covariance matrices [N, 3]
    float* inv_cov2ds)    // Output: inverse 2D covariance matrices [N, 3]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load 2D covariance
    const float* cov2d = &cov2ds[idx * 3];
    float* inv_cov2d = &inv_cov2ds[idx * 3];

    // Compute inverse 2D covariance
    inverseCov2D(cov2d, inv_cov2d);
}

// Compute Gaussian influences on channel matrix elements
__global__ void computeGaussianInfluencesCUDA(
    int N,                    // Number of Gaussians
    const float* uv,          // Channel matrix coordinates [N, 2]
    const float* inv_cov2ds,  // Inverse 2D covariance matrices [N, 3]
    int num_tx,               // Number of transmit antennas
    int num_rx,               // Number of receive antennas
    float* influences)        // Output: Gaussian influences [N, num_tx, num_rx]
{
    // Each Gaussian is processed by a separate 2D block of threads
    // One block handles all antenna pairs for one Gaussian
    int n_idx = blockIdx.x;    // Gaussian index
    int tx_idx = threadIdx.x;  // Thread's transmit antenna index offset
    int rx_idx = threadIdx.y;  // Thread's receive antenna index offset

    if (n_idx >= N) return;

    // Load UV coordinates and inverse covariance for this Gaussian
    float2 uv_n = make_float2(uv[n_idx * 2], uv[n_idx * 2 + 1]);
    const float* inv_cov2d = &inv_cov2ds[n_idx * 3];

    // Each thread processes multiple antenna pairs using a grid-stride loop
    for (int tx = tx_idx; tx < num_tx; tx += blockDim.x) {
        for (int rx = rx_idx; rx < num_rx; rx += blockDim.y) {
            // Antenna positions (half-integer coordinates)
            float antenna_pos_x = tx + 0.5f;
            float antenna_pos_y = rx + 0.5f;

            // Compute influence
            float influence = computeGaussianInfluence(uv_n, antenna_pos_x, antenna_pos_y, inv_cov2d);

            // Store influence
            influences[n_idx * num_tx * num_rx + tx * num_rx + rx] = influence;
        }
    }
}

// Compute wireless channel contributions
__global__ void computeChannelsCUDA(
    int N,                        // Number of Gaussians
    const float* attenuation,     // Attenuation values [N, 1]
    const float* phase_rotation,  // Phase rotation values [N, 1]
    const float* distances,       // Distances [N]
    float wavelength,             // Signal wavelength
    float* real_contrib,          // Output: real part of contributions [N, 1]
    float* imag_contrib)          // Output: imaginary part of contributions [N, 1]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load attenuation, phase, and distance
    float att = attenuation[idx];
    float phase = phase_rotation[idx];
    float dist = distances[idx];

    // Compute channel contribution
    float real, imag;
    computeChannel(att, phase, dist, wavelength, real, imag);

    // Store results
    real_contrib[idx] = real;
    imag_contrib[idx] = imag;
}

// Alpha blending to form the channel matrix
__global__ void alphaBlendingCUDA(
    int N,                      // Number of Gaussians
    const float* influences,    // Gaussian influences [N, num_tx, num_rx]
    const float* real_contrib,  // Real part of contributions [N, 1]
    const float* imag_contrib,  // Imaginary part of contributions [N, 1]
    const float* opacity,       // Opacity values [N, 1]
    const int* sort_indices,    // Sorted indices [N]
    int num_tx,                 // Number of transmit antennas
    int num_rx,                 // Number of receive antennas
    float* channel)             // Output: channel matrix [num_tx, 2*num_rx]
{
    // Each thread handles one or more antenna pairs using grid-stride loop pattern
    // This is more efficient for channel matrices with dimensions like 16x2 or 64x2
    int tx_base = blockIdx.x * blockDim.x + threadIdx.x;
    int rx_base = blockIdx.y * blockDim.y + threadIdx.y;

    // Stride across the entire channel matrix with the current thread block
    for (int tx_idx = tx_base; tx_idx < num_tx; tx_idx += gridDim.x * blockDim.x) {
        for (int rx_idx = rx_base; rx_idx < num_rx; rx_idx += gridDim.y * blockDim.y) {
            // Initialize accumulated channel values
            float accum_real = 0.0f;
            float accum_imag = 0.0f;
            float transmittance = 1.0f;

            // Process Gaussians in sorted order (by distance)
            for (int i = 0; i < N; i++) {
                int idx = sort_indices[i];

                // Get influence for this Gaussian at this antenna pair
                float influence = influences[idx * num_tx * num_rx + tx_idx * num_rx + rx_idx];

                // Effective opacity after influence weighting
                float effective_opacity = opacity[idx] * influence;

                // Contribution from this Gaussian
                float contrib_real = real_contrib[idx];
                float contrib_imag = imag_contrib[idx];

                // Update accumulated values with alpha blending
                accum_real += transmittance * effective_opacity * contrib_real;
                accum_imag += transmittance * effective_opacity * contrib_imag;

                // Update transmittance
                transmittance *= (1.0f - effective_opacity);

                // Early termination if transmittance is very low
                if (transmittance < 1e-5f) {
                    break;
                }
            }

            // Store final channel values (concatenate real and imaginary parts)
            // For a channel matrix with format [num_tx, 2*num_rx] where real
            // and imaginary parts are stacked
            channel[tx_idx * (2 * num_rx) + rx_idx] = accum_real;           // Real part
            channel[tx_idx * (2 * num_rx) + num_rx + rx_idx] = accum_imag;  // Imaginary part
        }
    }
}

namespace FORWARD {

void computeDistances(
    int N,
    const float* points,
    const float* receiver,
    float* distances) {
    int num_threads = 256;
    int num_blocks = (N + num_threads - 1) / num_threads;
    computeDistancesCUDA<<<num_blocks, num_threads>>>(
        N, points, receiver, distances);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("computeDistances CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

void computeSphericalCoords(
    int N,
    const float* points,
    const float* receiver,
    float* d,
    float* distances,
    float* longitude,
    float* latitude) {
    int num_threads = 256;
    int num_blocks = (N + num_threads - 1) / num_threads;
    computeSphericalCoordsCUDA<<<num_blocks, num_threads>>>(
        N, points, receiver, d, distances, longitude, latitude);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("computeSphericalCoords CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

void transformToChannelSpace(
    int N,
    const float* longitude,
    const float* latitude,
    int num_tx,
    int num_rx,
    float* uv) {
    int num_threads = 256;
    int num_blocks = (N + num_threads - 1) / num_threads;
    transformToChannelSpaceCUDA<<<num_blocks, num_threads>>>(
        N, longitude, latitude, num_tx, num_rx, uv);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("transformToChannelSpace CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

void computeJacobians(
    int N,
    const float* d,
    const float* distances,
    int num_tx,
    int num_rx,
    float* jacobians) {
    int num_threads = 256;
    int num_blocks = (N + num_threads - 1) / num_threads;
    computeJacobiansCUDA<<<num_blocks, num_threads>>>(
        N, d, distances, num_tx, num_rx, jacobians);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("computeJacobians CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

void projectCov3Ds(
    int N,
    const float* cov3ds,
    const float* jacobians,
    float* cov2ds) {
    int num_threads = 256;
    int num_blocks = (N + num_threads - 1) / num_threads;
    projectCov3DsCUDA<<<num_blocks, num_threads>>>(
        N, cov3ds, jacobians, cov2ds);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("projectCov3Ds CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

void computeInvCov2Ds(
    int N,
    const float* cov2ds,
    float* inv_cov2ds) {
    int num_threads = 256;
    int num_blocks = (N + num_threads - 1) / num_threads;
    computeInvCov2DsCUDA<<<num_blocks, num_threads>>>(
        N, cov2ds, inv_cov2ds);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("computeInvCov2Ds CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

void computeGaussianInfluences(
    int N,
    const float* uv,
    const float* inv_cov2ds,
    int num_tx,
    int num_rx,
    float* influences) {
    // Get optimal block dimensions based on matrix size
    dim3 block_dim = getOptimalBlockDim(num_tx, num_rx);

    // One block per Gaussian, with each block handling all antenna pairs
    dim3 grid_dim(N);

    computeGaussianInfluencesCUDA<<<grid_dim, block_dim>>>(
        N, uv, inv_cov2ds, num_tx, num_rx, influences);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("computeGaussianInfluences CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

void computeChannels(
    int N,
    const float* attenuation,
    const float* phase_rotation,
    const float* distances,
    float wavelength,
    float* real_contrib,
    float* imag_contrib) {
    int num_threads = 256;
    int num_blocks = (N + num_threads - 1) / num_threads;
    computeChannelsCUDA<<<num_blocks, num_threads>>>(
        N, attenuation, phase_rotation, distances, wavelength,
        real_contrib, imag_contrib);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("computeChannels CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

void alphaBlending(
    int N,
    const float* influences,
    const float* real_contrib,
    const float* imag_contrib,
    const float* opacity,
    const int* sort_indices,
    int num_tx,
    int num_rx,
    float* channel) {
    // Get optimal block dimensions based on matrix size
    dim3 block_dim = getOptimalBlockDim(num_tx, num_rx);

    // Determine grid size based on channel matrix dimensions
    // Use multiple blocks for better load balancing when channel matrices are large
    // For small matrices like 16x2, we might only need a single block
    int tx_blocks = (num_tx > 32) ? 4 : ((num_tx > 16) ? 2 : 1);
    int rx_blocks = (num_rx > 2) ? 2 : 1;

    dim3 grid_dim(tx_blocks, rx_blocks);

    alphaBlendingCUDA<<<grid_dim, block_dim>>>(
        N, influences, real_contrib, imag_contrib, opacity,
        sort_indices, num_tx, num_rx, channel);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("alphaBlending CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

void forward(
    int N,
    const float* points,
    const float* cov3ds,
    const float* attenuation,
    const float* phase_rotation,
    const float* opacity,
    const float* receiver,
    const float* transmitter,  // Not used directly
    int num_tx,
    int num_rx,
    float frequency,
    float* channel) {
    // Compute wavelength
    float wavelength = SPEED_OF_LIGHT / frequency;

    // Allocate device memory for intermediate results
    float *d_distances, *d_d, *d_longitude, *d_latitude, *d_uv;
    float *d_jacobians, *d_cov2ds, *d_inv_cov2ds;
    float *d_influences, *d_real_contrib, *d_imag_contrib;
    int* d_sort_indices;

    cudaMalloc(&d_distances, N * sizeof(float));
    cudaMalloc(&d_d, N * 3 * sizeof(float));
    cudaMalloc(&d_longitude, N * sizeof(float));
    cudaMalloc(&d_latitude, N * sizeof(float));
    cudaMalloc(&d_uv, N * 2 * sizeof(float));
    cudaMalloc(&d_jacobians, N * 6 * sizeof(float));
    cudaMalloc(&d_cov2ds, N * 3 * sizeof(float));
    cudaMalloc(&d_inv_cov2ds, N * 3 * sizeof(float));
    cudaMalloc(&d_influences, N * num_tx * num_rx * sizeof(float));
    cudaMalloc(&d_real_contrib, N * sizeof(float));
    cudaMalloc(&d_imag_contrib, N * sizeof(float));
    cudaMalloc(&d_sort_indices, N * sizeof(int));

    // Step 1-2: Compute spherical coordinates, distances, and displacement vectors
    computeSphericalCoords(
        N, points, receiver, d_d, d_distances, d_longitude, d_latitude);

    // Step 3: Transform coordinates to channel space
    transformToChannelSpace(
        N, d_longitude, d_latitude, num_tx, num_rx, d_uv);

    // Step 4: Compute Jacobians
    computeJacobians(
        N, d_d, d_distances, num_tx, num_rx, d_jacobians);

    // Step 5: Project 3D covariance to 2D
    projectCov3Ds(
        N, cov3ds, d_jacobians, d_cov2ds);

    // Step 6: Compute inverse 2D covariance
    computeInvCov2Ds(
        N, d_cov2ds, d_inv_cov2ds);

    // Step 7: Compute Gaussian influences
    computeGaussianInfluences(
        N, d_uv, d_inv_cov2ds, num_tx, num_rx, d_influences);

    // Step 8: Compute wireless channel contributions
    computeChannels(
        N, attenuation, phase_rotation, d_distances, wavelength,
        d_real_contrib, d_imag_contrib);

    // Step 9: Sort Gaussians by distance
    // Create indices array
    thrust::device_vector<int> indices(N);
    thrust::sequence(indices.begin(), indices.end(), 0);

    // Create device pointer to distances
    thrust::device_ptr<float> dist_ptr(d_distances);

    // Sort indices by distance
    thrust::sort_by_key(
        dist_ptr, dist_ptr + N,
        indices.begin());

    // Get raw pointer to sorted indices
    thrust::copy(indices.begin(), indices.end(), thrust::device_pointer_cast(d_sort_indices));

    // Step 10: Alpha blending
    alphaBlending(
        N, d_influences, d_real_contrib, d_imag_contrib,
        opacity, d_sort_indices, num_tx, num_rx, channel);

    // Free allocated device memory
    cudaFree(d_distances);
    cudaFree(d_d);
    cudaFree(d_longitude);
    cudaFree(d_latitude);
    cudaFree(d_uv);
    cudaFree(d_jacobians);
    cudaFree(d_cov2ds);
    cudaFree(d_inv_cov2ds);
    cudaFree(d_influences);
    cudaFree(d_real_contrib);
    cudaFree(d_imag_contrib);
    cudaFree(d_sort_indices);
}

}  // namespace FORWARD
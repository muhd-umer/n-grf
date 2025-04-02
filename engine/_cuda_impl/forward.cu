/*
 * Forward pass implementation for channel reconstruction CUDA kernels
 */

#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "auxiliary.h"
#include "forward.h"

// CUDA kernel for computing distances from points to receiver
__global__ void compute_distances_kernel(const float *points,
                                         const float *receiver,
                                         float *distances, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float dx = points[idx * 3] - receiver[0];
    float dy = points[idx * 3 + 1] - receiver[1];
    float dz = points[idx * 3 + 2] - receiver[2];

    distances[idx] = sqrtf(dx * dx + dy * dy + dz * dz);
}

// CUDA kernel for computing spherical coordinates
__global__ void compute_spherical_coords_kernel(const float *points,
                                                const float *receiver,
                                                float *displacement,
                                                float *longitude,
                                                float *latitude, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float dx = points[idx * 3] - receiver[0];
    float dy = points[idx * 3 + 1] - receiver[1];
    float dz = points[idx * 3 + 2] - receiver[2];

    displacement[idx * 3] = dx;
    displacement[idx * 3 + 1] = dy;
    displacement[idx * 3 + 2] = dz;

    float r = sqrtf(dx * dx + dy * dy + dz * dz);

    longitude[idx] = atan2f(dy, dx);
    latitude[idx] = asinf(fminf(fmaxf(dz / r, -1.0f), 1.0f));
}

// CUDA kernel for transforming spherical to uniform coordinates
__global__ void transform_to_uniform_coords_kernel(const float *longitude,
                                                   const float *latitude,
                                                   float *s_x, float *s_y,
                                                   int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    s_x[idx] = longitude[idx] / PI;
    s_y[idx] = 2.0f * latitude[idx] / PI;
}

// CUDA kernel for computing 3D covariance matrix from scaling and rotation
__global__ void compute_cov3d_from_scaling_rotation_kernel(
    const float *scaling, const float *rotation, float *cov3d,
    float scale_modifier, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Create scaling matrix (diagonal matrix with scaling values)
    float S[9] = {0};
    S[0] = scale_modifier * scaling[idx * 3];      // S[0][0]
    S[4] = scale_modifier * scaling[idx * 3 + 1];  // S[1][1]
    S[8] = scale_modifier * scaling[idx * 3 + 2];  // S[2][2]

    // Extract quaternion
    float r = rotation[idx * 4];
    float x = rotation[idx * 4 + 1];
    float y = rotation[idx * 4 + 2];
    float z = rotation[idx * 4 + 3];

    // Compute rotation matrix
    float R[9];
    R[0] = 1.0f - 2.0f * (y * y + z * z);
    R[1] = 2.0f * (x * y - r * z);
    R[2] = 2.0f * (x * z + r * y);
    R[3] = 2.0f * (x * y + r * z);
    R[4] = 1.0f - 2.0f * (x * x + z * z);
    R[5] = 2.0f * (y * z - r * x);
    R[6] = 2.0f * (x * z - r * y);
    R[7] = 2.0f * (y * z + r * x);
    R[8] = 1.0f - 2.0f * (x * x + y * y);

    // Compute product M = R * S
    float M[9] = {0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                M[i * 3 + j] += R[i * 3 + k] * S[k * 3 + j];
            }
        }
    }

    // Compute covariance matrix Sigma = transpose(M) * M
    float Sigma[9] = {0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                Sigma[i * 3 + j] += M[k * 3 + i] * M[k * 3 + j];
            }
        }
    }

    // Store only the upper triangular part (since it's symmetric)
    cov3d[idx * 6] = Sigma[0];      // [0, 0]
    cov3d[idx * 6 + 1] = Sigma[1];  // [0, 1]
    cov3d[idx * 6 + 2] = Sigma[2];  // [0, 2]
    cov3d[idx * 6 + 3] = Sigma[4];  // [1, 1]
    cov3d[idx * 6 + 4] = Sigma[5];  // [1, 2]
    cov3d[idx * 6 + 5] = Sigma[8];  // [2, 2]
}

// CUDA kernel for mapping uniform coords to channel matrix
__global__ void map_to_channel_matrix_kernel(const float *s_x, const float *s_y,
                                             float *uv, int num_tx, int num_rx,
                                             int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float u = ((s_x[idx] + 1.0f) / 2.0f) * (num_tx - 1) + 0.5f;
    float v = ((s_y[idx] + 1.0f) / 2.0f) * (num_rx - 1) + 0.5f;

    uv[idx * 2] = u;
    uv[idx * 2 + 1] = v;
}

// CUDA kernel for computing Jacobian matrices
__global__ void compute_jacobian_kernel(const float *d, const float *r,
                                        float *jacobian, int num_tx, int num_rx,
                                        int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = d[idx * 3];
    float y = d[idx * 3 + 1];
    float z = d[idx * 3 + 2];
    float r_val = r[idx];

    float xy_squared = x * x + y * y;
    xy_squared = fmaxf(xy_squared, 1e-10f);

    float cos_lat = sqrtf(1.0f - (z / r_val) * (z / r_val));
    cos_lat = fmaxf(cos_lat, 1e-10f);

    float tx_factor = float(num_tx - 1) / (2.0f * PI);
    float rx_factor = float(num_rx - 1) / PI;

    // J[0,0]: du/dx
    jacobian[idx * 6] = tx_factor * (-y / xy_squared);

    // J[0,1]: du/dy
    jacobian[idx * 6 + 1] = tx_factor * (x / xy_squared);

    // J[0,2]: du/dz
    jacobian[idx * 6 + 2] = 0.0f;

    float r_cos_lat_xy = r_val * cos_lat * xy_squared;
    r_cos_lat_xy = fmaxf(r_cos_lat_xy, 1e-10f);

    // J[1,0]: dv/dx
    jacobian[idx * 6 + 3] = rx_factor * (z * x) / r_cos_lat_xy;

    // J[1,1]: dv/dy
    jacobian[idx * 6 + 4] = rx_factor * (z * y) / r_cos_lat_xy;

    // J[1,2]: dv/dz
    jacobian[idx * 6 + 5] = rx_factor / (r_val * cos_lat);
}

// CUDA kernel for projecting 3D covariance to 2D
__global__ void project_cov3d_to_cov2d_kernel(const float *cov3d_mat,
                                              const float *jacobian,
                                              float *cov2d, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Extract 3x3 covariance matrix for this Gaussian
    float cov3d[9];
    for (int i = 0; i < 9; i++) {
        cov3d[i] = cov3d_mat[idx * 9 + i];
    }

    // Extract 2x3 Jacobian for this Gaussian
    float J[6];
    for (int i = 0; i < 6; i++) {
        J[i] = jacobian[idx * 6 + i];
    }

    // Compute intermediate product: temp = cov3d * J^T
    float temp[6];  // 3x2 matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            temp[i * 2 + j] = 0.0f;
            for (int k = 0; k < 3; k++) {
                temp[i * 2 + j] += cov3d[i * 3 + k] * J[j * 3 + k];
            }
        }
    }

    // Compute final product: cov2d = J * temp
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += J[i * 3 + k] * temp[k * 2 + j];
            }
            cov2d[idx * 4 + i * 2 + j] = sum;
        }
    }

    // Add regularization
    cov2d[idx * 4 + 0] += 0.3f;  // xx
    cov2d[idx * 4 + 3] += 0.3f;  // yy
}

// Wrapper function for computing distances to receiver
torch::Tensor computeDistancesToReceiverCUDA(const torch::Tensor &points,
                                             const torch::Tensor &receiver) {
    int N = points.size(0);

    auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(points.device());

    torch::Tensor distances = torch::empty({N}, options);

    int threads = 32;
    int blocks = (N + threads - 1) / threads;

    compute_distances_kernel<<<blocks, threads>>>(
        points.data_ptr<float>(), receiver.data_ptr<float>(),
        distances.data_ptr<float>(), N);

    return distances;
}

// Wrapper function for computing spherical coordinates
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
computeSphericalCoordsCUDA(const torch::Tensor &points,
                           const torch::Tensor &receiver) {
    int N = points.size(0);

    auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(points.device());

    torch::Tensor displacement = torch::empty({N, 3}, options);
    torch::Tensor longitude = torch::empty({N}, options);
    torch::Tensor latitude = torch::empty({N}, options);

    int threads = 32;
    int blocks = (N + threads - 1) / threads;

    compute_spherical_coords_kernel<<<blocks, threads>>>(
        points.data_ptr<float>(), receiver.data_ptr<float>(),
        displacement.data_ptr<float>(), longitude.data_ptr<float>(),
        latitude.data_ptr<float>(), N);

    return std::make_tuple(displacement, longitude, latitude);
}

// Wrapper function for transforming spherical to uniform coordinates
std::tuple<torch::Tensor, torch::Tensor> transformToUniformCoordsCUDA(
    const torch::Tensor &longitude, const torch::Tensor &latitude) {
    int N = longitude.size(0);

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(longitude.device());

    torch::Tensor s_x = torch::empty({N}, options);
    torch::Tensor s_y = torch::empty({N}, options);

    int threads = 32;
    int blocks = (N + threads - 1) / threads;

    transform_to_uniform_coords_kernel<<<blocks, threads>>>(
        longitude.data_ptr<float>(), latitude.data_ptr<float>(),
        s_x.data_ptr<float>(), s_y.data_ptr<float>(), N);

    return std::make_tuple(s_x, s_y);
}

// Wrapper function for computing 3D covariance matrix from scaling and rotation
torch::Tensor computeCov3dFromScalingRotationCUDA(const torch::Tensor &scaling,
                                                  const torch::Tensor &rotation,
                                                  float scale_modifier) {
    const at::cuda::CUDAGuard device_guard(scaling.device());

    int N = scaling.size(0);

    auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(scaling.device());

    torch::Tensor cov3d = torch::empty({N, 6}, options);

    int threads = 32;
    int blocks = (N + threads - 1) / threads;

    compute_cov3d_from_scaling_rotation_kernel<<<blocks, threads>>>(
        scaling.data_ptr<float>(), rotation.data_ptr<float>(),
        cov3d.data_ptr<float>(), scale_modifier, N);

    return cov3d;
}

// Wrapper function for mapping uniform coords to channel matrix
torch::Tensor mapToChannelMatrixCUDA(const torch::Tensor &s_x,
                                     const torch::Tensor &s_y, int num_tx,
                                     int num_rx) {
    int N = s_x.size(0);

    auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(s_x.device());

    torch::Tensor uv = torch::empty({N, 2}, options);

    int threads = 32;
    int blocks = (N + threads - 1) / threads;

    map_to_channel_matrix_kernel<<<blocks, threads>>>(
        s_x.data_ptr<float>(), s_y.data_ptr<float>(), uv.data_ptr<float>(),
        num_tx, num_rx, N);

    return uv;
}

// Wrapper function for computing Jacobian matrices
torch::Tensor computeJacobianCUDA(const torch::Tensor &d,
                                  const torch::Tensor &r, int num_tx,
                                  int num_rx) {
    int N = d.size(0);

    auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(d.device());

    torch::Tensor jacobian = torch::empty({N, 2, 3}, options);

    int threads = 32;
    int blocks = (N + threads - 1) / threads;

    compute_jacobian_kernel<<<blocks, threads>>>(
        d.data_ptr<float>(), r.data_ptr<float>(), jacobian.data_ptr<float>(),
        num_tx, num_rx, N);

    return jacobian;
}

// Wrapper function for projecting 3D covariance to 2D
torch::Tensor projectCov3dToCov2dCUDA(const torch::Tensor &cov3d_mat,
                                      const torch::Tensor &jacobian) {
    int N = cov3d_mat.size(0);

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(cov3d_mat.device());

    torch::Tensor cov2d = torch::empty({N, 2, 2}, options);

    int threads = 32;
    int blocks = (N + threads - 1) / threads;

    project_cov3d_to_cov2d_kernel<<<blocks, threads>>>(
        cov3d_mat.data_ptr<float>(), jacobian.data_ptr<float>(),
        cov2d.data_ptr<float>(), N);

    return cov2d;
}

// CUDA kernel for converting compact 6D covariance to full 3x3 matrix
__global__ void convert_compact_to_full_kernel(const float *cov_compact,
                                               float *cov_full, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    symmetric_to_full(&cov_compact[idx * 6], &cov_full[idx * 9]);
}

// Wrapper function for projecting to channel space
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
projectToChannelSpaceCUDA(const torch::Tensor &points,
                          const torch::Tensor &cov3d,
                          const torch::Tensor &receiver, int num_tx,
                          int num_rx) {
    int N = points.size(0);

    auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(points.device());

    // Step 1: Compute distances
    torch::Tensor distances = computeDistancesToReceiverCUDA(points, receiver);

    // Step 2: Compute spherical coordinates
    auto [d, longitude, latitude] =
        computeSphericalCoordsCUDA(points, receiver);

    // Step 3: Transform to uniform coordinates
    auto [s_x, s_y] = transformToUniformCoordsCUDA(longitude, latitude);

    // Step 4: Map to channel matrix
    torch::Tensor uv = mapToChannelMatrixCUDA(s_x, s_y, num_tx, num_rx);

    // Step 5: Compute Jacobian
    torch::Tensor jacobian = computeJacobianCUDA(d, distances, num_tx, num_rx);

    // Step 6: Convert compact covariance to full matrix
    torch::Tensor cov3d_mat = torch::empty({N, 3, 3}, options);

    int threads = 32;
    int blocks = (N + threads - 1) / threads;

    convert_compact_to_full_kernel<<<blocks, threads>>>(
        cov3d.data_ptr<float>(), cov3d_mat.data_ptr<float>(), N);

    // Step 7: Project 3D covariance to 2D
    torch::Tensor cov2d = projectCov3dToCov2dCUDA(cov3d_mat, jacobian);

    return std::make_tuple(distances, uv, cov2d);
}
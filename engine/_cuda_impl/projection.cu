// engine/_cuda_impl/projection.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>

#include "checks.cuh"
#include "matrix.cuh"

template <typename T>
__global__ void quaternion_to_rotation_kernel(const T* __restrict__ quaternion,
                                              const int N, T* rotation) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    T w = quaternion[i * 4 + 0];
    T x = quaternion[i * 4 + 1];
    T y = quaternion[i * 4 + 2];
    T z = quaternion[i * 4 + 3];

    rotation[i * 9 + 0] = 1 - 2 * y * y - 2 * z * z;
    rotation[i * 9 + 1] = 2 * x * y - 2 * w * z;
    rotation[i * 9 + 2] = 2 * x * z + 2 * w * y;
    rotation[i * 9 + 3] = 2 * x * y + 2 * w * z;
    rotation[i * 9 + 4] = 1 - 2 * x * x - 2 * z * z;
    rotation[i * 9 + 5] = 2 * y * z - 2 * w * x;
    rotation[i * 9 + 6] = 2 * x * z - 2 * w * y;
    rotation[i * 9 + 7] = 2 * y * z + 2 * w * x;
    rotation[i * 9 + 8] = 1 - 2 * x * x - 2 * y * y;
}

void quaternion_to_rotation_cuda(torch::Tensor quaternion,
                                 torch::Tensor rotation) {
    CHECK_VALID_INPUT(quaternion);
    CHECK_VALID_INPUT(rotation);

    const int N = quaternion.size(0);
    TORCH_CHECK(quaternion.size(1) == 4, "quaternion must have shape Nx4");
    TORCH_CHECK(rotation.size(0) == N, "rotation must have shape Nx3x3");
    TORCH_CHECK(rotation.size(1) == 3, "rotation must have shape Nx3x3");
    TORCH_CHECK(rotation.size(2) == 3, "rotation must have shape Nx3x3");

    const int max_threads_per_block = 1024;
    const int num_blocks =
        (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    if (quaternion.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(rotation);
        quaternion_to_rotation_kernel<float><<<gridsize, blocksize>>>(
            quaternion.data_ptr<float>(), N, rotation.data_ptr<float>());
    } else if (quaternion.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(rotation);
        quaternion_to_rotation_kernel<double><<<gridsize, blocksize>>>(
            quaternion.data_ptr<double>(), N, rotation.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", quaternion.dtype());
    }
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void compute_scaling_matrix_kernel(const T* __restrict__ scaling,
                                              const T scale_modifier,
                                              const int N,
                                              T* __restrict__ scaling_matrix) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    // Create 3x3 diagonal matrix with scaled values
    const T sx = scaling[i * 3 + 0] * scale_modifier;
    const T sy = scaling[i * 3 + 1] * scale_modifier;
    const T sz = scaling[i * 3 + 2] * scale_modifier;

    // Fill the diagonal
    scaling_matrix[i * 9 + 0] = sx;
    scaling_matrix[i * 9 + 1] = 0;
    scaling_matrix[i * 9 + 2] = 0;
    scaling_matrix[i * 9 + 3] = 0;
    scaling_matrix[i * 9 + 4] = sy;
    scaling_matrix[i * 9 + 5] = 0;
    scaling_matrix[i * 9 + 6] = 0;
    scaling_matrix[i * 9 + 7] = 0;
    scaling_matrix[i * 9 + 8] = sz;
}

void compute_scaling_matrix_cuda(torch::Tensor scaling, float scale_modifier,
                                 torch::Tensor scaling_matrix) {
    CHECK_VALID_INPUT(scaling);
    CHECK_VALID_INPUT(scaling_matrix);

    const int N = scaling.size(0);
    TORCH_CHECK(scaling.size(1) == 3, "scaling must have shape Nx3");
    TORCH_CHECK(scaling_matrix.size(0) == N,
                "scaling_matrix must have shape Nx3x3");
    TORCH_CHECK(scaling_matrix.size(1) == 3,
                "scaling_matrix must have shape Nx3x3");
    TORCH_CHECK(scaling_matrix.size(2) == 3,
                "scaling_matrix must have shape Nx3x3");

    const int max_threads_per_block = 1024;
    const int num_blocks =
        (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    if (scaling.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(scaling_matrix);
        compute_scaling_matrix_kernel<float>
            <<<gridsize, blocksize>>>(scaling.data_ptr<float>(), scale_modifier,
                                      N, scaling_matrix.data_ptr<float>());
    } else if (scaling.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(scaling_matrix);
        compute_scaling_matrix_kernel<double><<<gridsize, blocksize>>>(
            scaling.data_ptr<double>(), scale_modifier, N,
            scaling_matrix.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", scaling.dtype());
    }
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void matrix_multiply_kernel(const T* __restrict__ A,
                                       const T* __restrict__ B, const int N,
                                       T* __restrict__ C) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    // Assuming A and B are 3x3 matrices
    matrix_multiply<T>(A + i * 9, B + i * 9, C + i * 9, 3, 3, 3);
}

void matrix_multiply_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    CHECK_VALID_INPUT(A);
    CHECK_VALID_INPUT(B);
    CHECK_VALID_INPUT(C);

    const int N = A.size(0);
    TORCH_CHECK(A.size(1) == 3 && A.size(2) == 3, "A must have shape Nx3x3");
    TORCH_CHECK(B.size(0) == N && B.size(1) == 3 && B.size(2) == 3,
                "B must have shape Nx3x3");
    TORCH_CHECK(C.size(0) == N && C.size(1) == 3 && C.size(2) == 3,
                "C must have shape Nx3x3");

    const int max_threads_per_block = 1024;
    const int num_blocks =
        (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    if (A.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(B);
        CHECK_FLOAT_TENSOR(C);
        matrix_multiply_kernel<float><<<gridsize, blocksize>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), N, C.data_ptr<float>());
    } else if (A.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(B);
        CHECK_DOUBLE_TENSOR(C);
        matrix_multiply_kernel<double><<<gridsize, blocksize>>>(
            A.data_ptr<double>(), B.data_ptr<double>(), N,
            C.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", A.dtype());
    }
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void covariance_matrix_kernel(const T* __restrict__ RS, const int N,
                                         T* __restrict__ cov3d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    // Create RS^T
    T RS_T[9];
    transpose<T>(RS + i * 9, RS_T, 3, 3);

    // Compute covariance matrix: C = RS * RS^T.
    matrix_multiply<T>(RS + i * 9, RS_T, cov3d + i * 9, 3, 3, 3);
}

void covariance_matrix_cuda(torch::Tensor RS, torch::Tensor cov3d) {
    CHECK_VALID_INPUT(RS);
    CHECK_VALID_INPUT(cov3d);

    const int N = RS.size(0);
    TORCH_CHECK(RS.size(1) == 3 && RS.size(2) == 3, "RS must have shape Nx3x3");
    TORCH_CHECK(cov3d.size(0) == N && cov3d.size(1) == 3 && cov3d.size(2) == 3,
                "cov3d must have shape Nx3x3");

    const int max_threads_per_block = 1024;
    const int num_blocks =
        (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    if (RS.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(cov3d);
        covariance_matrix_kernel<float><<<gridsize, blocksize>>>(
            RS.data_ptr<float>(), N, cov3d.data_ptr<float>());
    } else if (RS.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(cov3d);
        covariance_matrix_kernel<double><<<gridsize, blocksize>>>(
            RS.data_ptr<double>(), N, cov3d.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", RS.dtype());
    }
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void project_to_channel_coords_kernel(
    const T* __restrict__ points, const T* __restrict__ receiver,
    const int num_tx, const int num_rx, const int N, T* __restrict__ distances,
    T* __restrict__ displacement, T* __restrict__ uv_coords) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    // Compute displacement and distance
    T d[3];
    d[0] = points[i * 3 + 0] - receiver[0];
    d[1] = points[i * 3 + 1] - receiver[1];
    d[2] = points[i * 3 + 2] - receiver[2];

    // Store displacement
    displacement[i * 3 + 0] = d[0];
    displacement[i * 3 + 1] = d[1];
    displacement[i * 3 + 2] = d[2];

    // Compute distance
    T r = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    distances[i] = r;

    // Compute spherical coordinates
    T longitude = atan2(d[1], d[0]);
    T dz_r = d[2] / r;
    dz_r = min(max(dz_r, T(-1.0)), T(1.0));  // Clamp to [-1, 1]
    T latitude = asin(dz_r);

    // Transform to uniform coordinates
    T PI = T(3.14159265358979323846);
    T s_x = longitude / PI;
    T s_y = T(2.0) * latitude / PI;

    // Map to channel matrix coordinates
    T u = ((s_x + T(1.0)) / T(2.0)) * (num_tx - T(1.0)) + T(0.5);
    T v = ((s_y + T(1.0)) / T(2.0)) * (num_rx - T(1.0)) + T(0.5);

    uv_coords[i * 2 + 0] = u;
    uv_coords[i * 2 + 1] = v;
}

void project_to_channel_coords_cuda(torch::Tensor points,
                                    torch::Tensor receiver, int num_tx,
                                    int num_rx, torch::Tensor distances,
                                    torch::Tensor displacement,
                                    torch::Tensor uv_coords) {
    CHECK_VALID_INPUT(points);
    CHECK_VALID_INPUT(receiver);
    CHECK_VALID_INPUT(distances);
    CHECK_VALID_INPUT(displacement);
    CHECK_VALID_INPUT(uv_coords);

    const int N = points.size(0);
    TORCH_CHECK(points.size(1) == 3, "points must have shape Nx3");
    TORCH_CHECK(receiver.size(0) == 3, "receiver must have shape 3");
    TORCH_CHECK(distances.size(0) == N, "distances must have shape N");
    TORCH_CHECK(displacement.size(0) == N && displacement.size(1) == 3,
                "displacement must have shape Nx3");
    TORCH_CHECK(uv_coords.size(0) == N && uv_coords.size(1) == 2,
                "uv_coords must have shape Nx2");

    const int max_threads_per_block = 1024;
    const int num_blocks =
        (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    if (points.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(receiver);
        CHECK_FLOAT_TENSOR(distances);
        CHECK_FLOAT_TENSOR(displacement);
        CHECK_FLOAT_TENSOR(uv_coords);
        project_to_channel_coords_kernel<float><<<gridsize, blocksize>>>(
            points.data_ptr<float>(), receiver.data_ptr<float>(), num_tx,
            num_rx, N, distances.data_ptr<float>(),
            displacement.data_ptr<float>(), uv_coords.data_ptr<float>());
    } else if (points.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(receiver);
        CHECK_DOUBLE_TENSOR(distances);
        CHECK_DOUBLE_TENSOR(displacement);
        CHECK_DOUBLE_TENSOR(uv_coords);
        project_to_channel_coords_kernel<double><<<gridsize, blocksize>>>(
            points.data_ptr<double>(), receiver.data_ptr<double>(), num_tx,
            num_rx, N, distances.data_ptr<double>(),
            displacement.data_ptr<double>(), uv_coords.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", points.dtype());
    }
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void compute_jacobian_kernel(const T* __restrict__ d,
                                        const int num_tx, const int num_rx,
                                        const int N, T* __restrict__ J) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    // Extract displacement components
    T x = d[i * 3 + 0];
    T y = d[i * 3 + 1];
    T z = d[i * 3 + 2];

    // Compute r and necessary intermediate values
    T r = sqrt(x * x + y * y + z * z);

    // Compute x²+y² and clamp for stability
    T xy_sq = x * x + y * y;
    xy_sq = max(xy_sq, T(1e-10));

    // Compute cos_lat = sqrt(1 - (z/r)²)
    T cos_lat = sqrt(max(T(1.0) - (z / r) * (z / r), T(1e-10)));

    // Factors from antenna grid dimensions
    T PI = T(3.14159265358979323846);
    T tx_factor = (num_tx - T(1.0)) / (T(2.0) * PI);
    T rx_factor = (num_rx - T(1.0)) / PI;

    // First row: u-coordinates derivatives (longitude)
    J[i * 6 + 0] = tx_factor * (-y / xy_sq);
    J[i * 6 + 1] = tx_factor * (x / xy_sq);
    J[i * 6 + 2] = T(0.0);

    // Second row: v-coordinates derivatives (latitude)
    T r_cos_lat_xy = r * cos_lat * xy_sq;
    r_cos_lat_xy = max(r_cos_lat_xy, T(1e-10));

    J[i * 6 + 3] = rx_factor * (z * x) / r_cos_lat_xy;
    J[i * 6 + 4] = rx_factor * (z * y) / r_cos_lat_xy;
    J[i * 6 + 5] = rx_factor / (r * cos_lat);
}

void compute_jacobian_cuda(torch::Tensor d, int num_tx, int num_rx,
                           torch::Tensor J) {
    CHECK_VALID_INPUT(d);
    CHECK_VALID_INPUT(J);

    const int N = d.size(0);
    TORCH_CHECK(d.size(1) == 3, "d must have shape Nx3");
    TORCH_CHECK(J.size(0) == N && J.size(1) == 2 && J.size(2) == 3,
                "J must have shape Nx2x3");

    const int max_threads_per_block = 1024;
    const int num_blocks =
        (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    if (d.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(J);
        compute_jacobian_kernel<float><<<gridsize, blocksize>>>(
            d.data_ptr<float>(), num_tx, num_rx, N, J.data_ptr<float>());
    } else if (d.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(J);
        compute_jacobian_kernel<double><<<gridsize, blocksize>>>(
            d.data_ptr<double>(), num_tx, num_rx, N, J.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", d.dtype());
    }
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void project_cov3d_to_cov2d_kernel(const T* __restrict__ cov3d_mat,
                                              const T* __restrict__ jacobian,
                                              const int N,
                                              T* __restrict__ cov2d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    // Step 1: Compute temp = cov3d * J^T
    T J_T[6];  // 3x2
    transpose<T>(jacobian + i * 6, J_T, 2, 3);

    T temp[6];  // 3x2
    matrix_multiply<T>(cov3d_mat + i * 9, J_T, temp, 3, 3, 2);

    // Step 2: Compute cov2d = J * temp
    matrix_multiply<T>(jacobian + i * 6, temp, cov2d + i * 4, 2, 3, 2);

    // Add a small constant to ensure positive definiteness
    cov2d[i * 4 + 0] += T(0.3);  // (0,0)
    cov2d[i * 4 + 3] += T(0.3);  // (1,1)
}

void project_cov3d_to_cov2d_cuda(torch::Tensor cov3d, torch::Tensor J,
                                 torch::Tensor cov2d) {
    CHECK_VALID_INPUT(cov3d);
    CHECK_VALID_INPUT(J);
    CHECK_VALID_INPUT(cov2d);

    const int N = cov3d.size(0);
    TORCH_CHECK(cov3d.size(1) == 3 && cov3d.size(2) == 3,
                "cov3d must have shape Nx3x3");
    TORCH_CHECK(J.size(0) == N && J.size(1) == 2 && J.size(2) == 3,
                "J must have shape Nx2x3");
    TORCH_CHECK(cov2d.size(0) == N && cov2d.size(1) == 2 && cov2d.size(2) == 2,
                "cov2d must have shape Nx2x2");

    const int max_threads_per_block = 1024;
    const int num_blocks =
        (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    if (cov3d.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(J);
        CHECK_FLOAT_TENSOR(cov2d);
        project_cov3d_to_cov2d_kernel<float><<<gridsize, blocksize>>>(
            cov3d.data_ptr<float>(), J.data_ptr<float>(), N,
            cov2d.data_ptr<float>());
    } else if (cov3d.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(J);
        CHECK_DOUBLE_TENSOR(cov2d);
        project_cov3d_to_cov2d_kernel<double><<<gridsize, blocksize>>>(
            cov3d.data_ptr<double>(), J.data_ptr<double>(), N,
            cov2d.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", cov3d.dtype());
    }
    cudaDeviceSynchronize();
}
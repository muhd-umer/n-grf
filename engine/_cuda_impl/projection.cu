// engine/_cuda_impl/projection.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>

#include "checks.cuh"
#include "matrix.cuh"

constexpr int THREADS_PER_BLOCK_PROJ = 256;

template <typename T>
__global__ void quaternion_to_rotation_kernel(const T* __restrict__ quaternion,
                                              const int N,
                                              T* __restrict__ rotation) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const T w = quaternion[i * 4 + 0];
    const T x = quaternion[i * 4 + 1];
    const T y = quaternion[i * 4 + 2];
    const T z = quaternion[i * 4 + 3];

    const T two_x = T(2.0) * x;
    const T two_y = T(2.0) * y;
    const T two_z = T(2.0) * z;
    const T two_wx = T(2.0) * w * x;
    const T two_wy = T(2.0) * w * y;
    const T two_wz = T(2.0) * w * z;
    const T two_xx = two_x * x;
    const T two_xy = two_x * y;
    const T two_xz = two_x * z;
    const T two_yy = two_y * y;
    const T two_yz = two_y * z;
    const T two_zz = two_z * z;

    rotation[i * 9 + 0] = T(1.0) - two_yy - two_zz;
    rotation[i * 9 + 1] = two_xy - two_wz;
    rotation[i * 9 + 2] = two_xz + two_wy;
    rotation[i * 9 + 3] = two_xy + two_wz;
    rotation[i * 9 + 4] = T(1.0) - two_xx - two_zz;
    rotation[i * 9 + 5] = two_yz - two_wx;
    rotation[i * 9 + 6] = two_xz - two_wy;
    rotation[i * 9 + 7] = two_yz + two_wx;
    rotation[i * 9 + 8] = T(1.0) - two_xx - two_yy;
}

void quaternion_to_rotation_cuda(torch::Tensor quaternion,
                                 torch::Tensor rotation) {
    CHECK_VALID_INPUT(quaternion);
    CHECK_VALID_INPUT(rotation);

    const int N = quaternion.size(0);
    TORCH_CHECK(quaternion.size(1) == 4, "quaternion must have shape Nx4");
    TORCH_CHECK(
        rotation.size(0) == N && rotation.size(1) == 3 && rotation.size(2) == 3,
        "rotation must have shape Nx3x3");

    const int num_blocks =
        (N + THREADS_PER_BLOCK_PROJ - 1) / THREADS_PER_BLOCK_PROJ;
    dim3 gridsize(num_blocks);
    dim3 blocksize(THREADS_PER_BLOCK_PROJ);

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
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after quaternion_to_rotation: ",
                cudaGetErrorString(err));
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

    const T sx = scaling[i * 3 + 0] * scale_modifier;
    const T sy = scaling[i * 3 + 1] * scale_modifier;
    const T sz = scaling[i * 3 + 2] * scale_modifier;

    T* mat_ptr = scaling_matrix + i * 9;
    mat_ptr[0] = sx;
    mat_ptr[1] = T(0.0);
    mat_ptr[2] = T(0.0);
    mat_ptr[3] = T(0.0);
    mat_ptr[4] = sy;
    mat_ptr[5] = T(0.0);
    mat_ptr[6] = T(0.0);
    mat_ptr[7] = T(0.0);
    mat_ptr[8] = sz;
}

void compute_scaling_matrix_cuda(torch::Tensor scaling, float scale_modifier,
                                 torch::Tensor scaling_matrix) {
    CHECK_VALID_INPUT(scaling);
    CHECK_VALID_INPUT(scaling_matrix);

    const int N = scaling.size(0);
    TORCH_CHECK(scaling.size(1) == 3, "scaling must have shape Nx3");
    TORCH_CHECK(scaling_matrix.size(0) == N && scaling_matrix.size(1) == 3 &&
                    scaling_matrix.size(2) == 3,
                "scaling_matrix must have shape Nx3x3");

    const int num_blocks =
        (N + THREADS_PER_BLOCK_PROJ - 1) / THREADS_PER_BLOCK_PROJ;
    dim3 gridsize(num_blocks);
    dim3 blocksize(THREADS_PER_BLOCK_PROJ);

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
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after compute_scaling_matrix: ",
                cudaGetErrorString(err));
}

template <typename T>
__global__ void matrix_multiply_kernel(const T* __restrict__ A,
                                       const T* __restrict__ B, const int N,
                                       T* __restrict__ C) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
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

    const int num_blocks =
        (N + THREADS_PER_BLOCK_PROJ - 1) / THREADS_PER_BLOCK_PROJ;
    dim3 gridsize(num_blocks);
    dim3 blocksize(THREADS_PER_BLOCK_PROJ);

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
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA error after matrix_multiply: ", cudaGetErrorString(err));
}

template <typename T>
__global__ void covariance_matrix_kernel(const T* __restrict__ RS, const int N,
                                         T* __restrict__ cov3d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    T RS_T[9];
    transpose<T>(RS + i * 9, RS_T, 3, 3);
    matrix_multiply<T>(RS + i * 9, RS_T, cov3d + i * 9, 3, 3, 3);
}

void covariance_matrix_cuda(torch::Tensor RS, torch::Tensor cov3d) {
    CHECK_VALID_INPUT(RS);
    CHECK_VALID_INPUT(cov3d);

    const int N = RS.size(0);
    TORCH_CHECK(RS.size(1) == 3 && RS.size(2) == 3, "RS must have shape Nx3x3");
    TORCH_CHECK(cov3d.size(0) == N && cov3d.size(1) == 3 && cov3d.size(2) == 3,
                "cov3d must have shape Nx3x3");

    const int num_blocks =
        (N + THREADS_PER_BLOCK_PROJ - 1) / THREADS_PER_BLOCK_PROJ;
    dim3 gridsize(num_blocks);
    dim3 blocksize(THREADS_PER_BLOCK_PROJ);

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
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after covariance_matrix: ",
                cudaGetErrorString(err));
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

    const T rx = receiver[0];
    const T ry = receiver[1];
    const T rz = receiver[2];

    const T px = points[i * 3 + 0];
    const T py = points[i * 3 + 1];
    const T pz = points[i * 3 + 2];

    const T dx = px - rx;
    const T dy = py - ry;
    const T dz = pz - rz;

    displacement[i * 3 + 0] = dx;
    displacement[i * 3 + 1] = dy;
    displacement[i * 3 + 2] = dz;

    const T r_sq = dx * dx + dy * dy + dz * dz;
    const T r = sqrt(r_sq);
    const T r_safe = max(r, T(ROBUST_EPSILON));
    distances[i] = r;

    const T longitude = atan2(dy, dx);

    T dz_r_arg = dz / r_safe;
    dz_r_arg = min(max(dz_r_arg, T(-1.0)), T(1.0));
    const T latitude = asin(dz_r_arg);

    constexpr T PI = T(M_PI);
    constexpr T INV_PI = T(1.0) / PI;
    constexpr T TWO_OVER_PI = T(2.0) / PI;

    const T s_x = longitude * INV_PI;
    const T s_y = latitude * TWO_OVER_PI;

    const T u = (s_x + T(1.0)) * T(0.5) * (num_tx - T(1.0)) + T(0.5);
    const T v = (s_y + T(1.0)) * T(0.5) * (num_rx - T(1.0)) + T(0.5);

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

    const int num_blocks =
        (N + THREADS_PER_BLOCK_PROJ - 1) / THREADS_PER_BLOCK_PROJ;
    dim3 gridsize(num_blocks);
    dim3 blocksize(THREADS_PER_BLOCK_PROJ);

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
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA error after project_to_channel_coords: ",
                cudaGetErrorString(err));
}

template <typename T>
__global__ void compute_jacobian_kernel(const T* __restrict__ d,
                                        const int num_tx, const int num_rx,
                                        const int N, T* __restrict__ J) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const T x = d[i * 3 + 0];
    const T y = d[i * 3 + 1];
    const T z = d[i * 3 + 2];

    const T r_sq = x * x + y * y + z * z;
    const T r = sqrt(r_sq);
    const T r_safe = max(r, T(ROBUST_EPSILON));

    const T xy_sq = x * x + y * y;
    const T xy_sq_safe = max(xy_sq, T(ROBUST_EPSILON));

    const T dz_r = z / r_safe;
    const T dz_r_clamped = min(max(dz_r, T(-1.0)), T(1.0));
    const T cos_lat_sq = T(1.0) - dz_r_clamped * dz_r_clamped;
    const T cos_lat = sqrt(max(cos_lat_sq, T(ROBUST_EPSILON)));
    const T cos_lat_safe = max(cos_lat, T(ROBUST_EPSILON));

    constexpr T PI = T(M_PI);
    const T tx_factor = (num_tx - T(1.0)) / (T(2.0) * PI);
    const T rx_factor = (num_rx - T(1.0)) / PI;

    T* J_ptr = J + i * 6;

    J_ptr[0] = tx_factor * (-y / xy_sq_safe);
    J_ptr[1] = tx_factor * (x / xy_sq_safe);
    J_ptr[2] = T(0.0);

    const T r_cos_lat = r_safe * cos_lat_safe;
    const T r_cos_lat_safe = max(r_cos_lat, T(ROBUST_EPSILON));
    const T r_cos_lat_xy_sq = r_cos_lat_safe * xy_sq_safe;
    const T r_cos_lat_xy_sq_safe = max(r_cos_lat_xy_sq, T(ROBUST_EPSILON));

    J_ptr[3] = rx_factor * (z * x) / r_cos_lat_xy_sq_safe;
    J_ptr[4] = rx_factor * (z * y) / r_cos_lat_xy_sq_safe;
    J_ptr[5] = rx_factor / r_cos_lat_safe;
}

void compute_jacobian_cuda(torch::Tensor d, int num_tx, int num_rx,
                           torch::Tensor J) {
    CHECK_VALID_INPUT(d);
    CHECK_VALID_INPUT(J);

    const int N = d.size(0);
    TORCH_CHECK(d.size(1) == 3, "d must have shape Nx3");
    TORCH_CHECK(J.size(0) == N && J.size(1) == 2 && J.size(2) == 3,
                "J must have shape Nx2x3");

    const int num_blocks =
        (N + THREADS_PER_BLOCK_PROJ - 1) / THREADS_PER_BLOCK_PROJ;
    dim3 gridsize(num_blocks);
    dim3 blocksize(THREADS_PER_BLOCK_PROJ);

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
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA error after compute_jacobian: ", cudaGetErrorString(err));
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

    const T* J_ptr = jacobian + i * 6;
    const T* cov3d_ptr = cov3d_mat + i * 9;
    T* cov2d_ptr = cov2d + i * 4;

    T J_T[6];
    transpose<T>(J_ptr, J_T, 2, 3);

    T temp[6];
    matrix_multiply<T>(cov3d_ptr, J_T, temp, 3, 3, 2);

    matrix_multiply<T>(J_ptr, temp, cov2d_ptr, 2, 3, 2);

    cov2d_ptr[0] += T(0.3);
    cov2d_ptr[3] += T(0.3);
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

    const int num_blocks =
        (N + THREADS_PER_BLOCK_PROJ - 1) / THREADS_PER_BLOCK_PROJ;
    dim3 gridsize(num_blocks);
    dim3 blocksize(THREADS_PER_BLOCK_PROJ);

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
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after project_cov3d_to_cov2d: ",
                cudaGetErrorString(err));
}
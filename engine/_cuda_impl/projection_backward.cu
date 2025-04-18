// engine/_cuda_impl/projection_backward.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>

#include "checks.cuh"
#include "matrix.cuh"

constexpr int THREADS_PER_BLOCK_PROJ_BW = 256;

template <typename T>
__global__ void quaternion_to_rotation_backward_kernel(
    const T* __restrict__ quaternion, const T* __restrict__ grad_rotation,
    const int N, T* __restrict__ grad_quaternion) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const T w_un = quaternion[i * 4 + 0];
    const T x_un = quaternion[i * 4 + 1];
    const T y_un = quaternion[i * 4 + 2];
    const T z_un = quaternion[i * 4 + 3];

    const T norm_sq = w_un * w_un + x_un * x_un + y_un * y_un + z_un * z_un;
    const T norm = sqrt(max(norm_sq, T(1e-15)));
    const T inv_norm = T(1.0) / norm;

    const T w = w_un * inv_norm;
    const T x = x_un * inv_norm;
    const T y = y_un * inv_norm;
    const T z = z_un * inv_norm;

    const T* grad_R_ptr = grad_rotation + i * 9;
    const T dR00 = grad_R_ptr[0];
    const T dR01 = grad_R_ptr[1];
    const T dR02 = grad_R_ptr[2];
    const T dR10 = grad_R_ptr[3];
    const T dR11 = grad_R_ptr[4];
    const T dR12 = grad_R_ptr[5];
    const T dR20 = grad_R_ptr[6];
    const T dR21 = grad_R_ptr[7];
    const T dR22 = grad_R_ptr[8];

    const T grad_w_norm = T(2.0) * (-z * dR01 + y * dR02 + z * dR10 - x * dR12 -
                                    y * dR20 + x * dR21);
    const T grad_x_norm =
        T(2.0) * (y * dR01 + z * dR02 + y * dR10 - T(2.0) * x * dR11 -
                  w * dR12 + z * dR20 + w * dR21 - T(2.0) * x * dR22);
    const T grad_y_norm =
        T(2.0) * (-T(2.0) * y * dR00 + x * dR01 + w * dR02 + x * dR10 +
                  z * dR12 - w * dR20 + z * dR21 - T(2.0) * y * dR22);
    const T grad_z_norm =
        T(2.0) * (-T(2.0) * z * dR00 - w * dR01 + x * dR02 + w * dR10 -
                  T(2.0) * z * dR11 + y * dR12 + x * dR20 + y * dR21);

    const T q_norm_cubed = norm * norm * norm;
    const T factor = (grad_w_norm * w_un + grad_x_norm * x_un +
                      grad_y_norm * y_un + grad_z_norm * z_un) /
                     max(q_norm_cubed, T(1e-15));

    T* grad_q_ptr = grad_quaternion + i * 4;
    grad_q_ptr[0] = grad_w_norm * inv_norm - w_un * factor;
    grad_q_ptr[1] = grad_x_norm * inv_norm - x_un * factor;
    grad_q_ptr[2] = grad_y_norm * inv_norm - y_un * factor;
    grad_q_ptr[3] = grad_z_norm * inv_norm - z_un * factor;
}

void quaternion_to_rotation_backward_cuda(torch::Tensor quaternion,
                                          torch::Tensor grad_rotation,
                                          torch::Tensor grad_quaternion) {
    CHECK_VALID_INPUT(quaternion);
    CHECK_VALID_INPUT(grad_rotation);
    CHECK_VALID_INPUT(grad_quaternion);

    const int N = quaternion.size(0);
    TORCH_CHECK(quaternion.size(1) == 4, "quaternion must have shape Nx4");
    TORCH_CHECK(grad_rotation.size(0) == N && grad_rotation.size(1) == 3 &&
                    grad_rotation.size(2) == 3,
                "grad_rotation must have shape Nx3x3");
    TORCH_CHECK(grad_quaternion.size(0) == N && grad_quaternion.size(1) == 4,
                "grad_quaternion must have shape Nx4");

    const int num_blocks =
        (N + THREADS_PER_BLOCK_PROJ_BW - 1) / THREADS_PER_BLOCK_PROJ_BW;
    dim3 gridsize(num_blocks);
    dim3 blocksize(THREADS_PER_BLOCK_PROJ_BW);

    if (quaternion.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(grad_rotation);
        CHECK_FLOAT_TENSOR(grad_quaternion);
        quaternion_to_rotation_backward_kernel<float><<<gridsize, blocksize>>>(
            quaternion.data_ptr<float>(), grad_rotation.data_ptr<float>(), N,
            grad_quaternion.data_ptr<float>());
    } else if (quaternion.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(grad_rotation);
        CHECK_DOUBLE_TENSOR(grad_quaternion);
        quaternion_to_rotation_backward_kernel<double><<<gridsize, blocksize>>>(
            quaternion.data_ptr<double>(), grad_rotation.data_ptr<double>(), N,
            grad_quaternion.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", quaternion.dtype());
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA error after quaternion_to_rotation_backward: ",
                cudaGetErrorString(err));
}

template <typename T>
__global__ void compute_scaling_matrix_backward_kernel(
    const T* __restrict__ grad_scaling_matrix, const T scale_modifier,
    const int N, T* __restrict__ grad_scaling) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const T* grad_S_ptr = grad_scaling_matrix + i * 9;
    T* grad_s_ptr = grad_scaling + i * 3;

    grad_s_ptr[0] = grad_S_ptr[0] * scale_modifier;
    grad_s_ptr[1] = grad_S_ptr[4] * scale_modifier;
    grad_s_ptr[2] = grad_S_ptr[8] * scale_modifier;
}

void compute_scaling_matrix_backward_cuda(torch::Tensor scaling,
                                          torch::Tensor grad_scaling_matrix,
                                          float scale_modifier,
                                          torch::Tensor grad_scaling) {
    CHECK_VALID_INPUT(scaling);
    CHECK_VALID_INPUT(grad_scaling_matrix);
    CHECK_VALID_INPUT(grad_scaling);

    const int N = scaling.size(0);
    TORCH_CHECK(scaling.size(1) == 3, "scaling must have shape Nx3");
    TORCH_CHECK(grad_scaling_matrix.size(0) == N &&
                    grad_scaling_matrix.size(1) == 3 &&
                    grad_scaling_matrix.size(2) == 3,
                "grad_scaling_matrix must have shape Nx3x3");
    TORCH_CHECK(grad_scaling.size(0) == N && grad_scaling.size(1) == 3,
                "grad_scaling must have shape Nx3");

    const int num_blocks =
        (N + THREADS_PER_BLOCK_PROJ_BW - 1) / THREADS_PER_BLOCK_PROJ_BW;
    dim3 gridsize(num_blocks);
    dim3 blocksize(THREADS_PER_BLOCK_PROJ_BW);

    if (scaling.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(grad_scaling_matrix);
        CHECK_FLOAT_TENSOR(grad_scaling);
        compute_scaling_matrix_backward_kernel<float><<<gridsize, blocksize>>>(
            grad_scaling_matrix.data_ptr<float>(), scale_modifier, N,
            grad_scaling.data_ptr<float>());
    } else if (scaling.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(grad_scaling_matrix);
        CHECK_DOUBLE_TENSOR(grad_scaling);
        compute_scaling_matrix_backward_kernel<double><<<gridsize, blocksize>>>(
            grad_scaling_matrix.data_ptr<double>(), scale_modifier, N,
            grad_scaling.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", scaling.dtype());
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA error after compute_scaling_matrix_backward: ",
                cudaGetErrorString(err));
}

template <typename T>
__global__ void matrix_multiply_backward_kernel(const T* __restrict__ A,
                                                const T* __restrict__ B,
                                                const T* __restrict__ grad_C,
                                                const int N,
                                                T* __restrict__ grad_A,
                                                T* __restrict__ grad_B) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const T* A_ptr = A + i * 9;
    const T* B_ptr = B + i * 9;
    const T* grad_C_ptr = grad_C + i * 9;
    T* grad_A_ptr = grad_A + i * 9;
    T* grad_B_ptr = grad_B + i * 9;

    T B_T[9];
    transpose<T>(B_ptr, B_T, 3, 3);
    matrix_multiply<T>(grad_C_ptr, B_T, grad_A_ptr, 3, 3, 3);

    T A_T[9];
    transpose<T>(A_ptr, A_T, 3, 3);
    matrix_multiply<T>(A_T, grad_C_ptr, grad_B_ptr, 3, 3, 3);
}

void matrix_multiply_backward_cuda(torch::Tensor A, torch::Tensor B,
                                   torch::Tensor grad_C, torch::Tensor grad_A,
                                   torch::Tensor grad_B) {
    CHECK_VALID_INPUT(A);
    CHECK_VALID_INPUT(B);
    CHECK_VALID_INPUT(grad_C);
    CHECK_VALID_INPUT(grad_A);
    CHECK_VALID_INPUT(grad_B);

    const int N = A.size(0);
    TORCH_CHECK(A.size(1) == 3 && A.size(2) == 3, "A must have shape Nx3x3");
    TORCH_CHECK(B.size(0) == N && B.size(1) == 3 && B.size(2) == 3,
                "B must have shape Nx3x3");
    TORCH_CHECK(
        grad_C.size(0) == N && grad_C.size(1) == 3 && grad_C.size(2) == 3,
        "grad_C must have shape Nx3x3");
    TORCH_CHECK(
        grad_A.size(0) == N && grad_A.size(1) == 3 && grad_A.size(2) == 3,
        "grad_A must have shape Nx3x3");
    TORCH_CHECK(
        grad_B.size(0) == N && grad_B.size(1) == 3 && grad_B.size(2) == 3,
        "grad_B must have shape Nx3x3");

    const int num_blocks =
        (N + THREADS_PER_BLOCK_PROJ_BW - 1) / THREADS_PER_BLOCK_PROJ_BW;
    dim3 gridsize(num_blocks);
    dim3 blocksize(THREADS_PER_BLOCK_PROJ_BW);

    if (A.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(B);
        CHECK_FLOAT_TENSOR(grad_C);
        CHECK_FLOAT_TENSOR(grad_A);
        CHECK_FLOAT_TENSOR(grad_B);
        matrix_multiply_backward_kernel<float><<<gridsize, blocksize>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), grad_C.data_ptr<float>(),
            N, grad_A.data_ptr<float>(), grad_B.data_ptr<float>());
    } else if (A.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(B);
        CHECK_DOUBLE_TENSOR(grad_C);
        CHECK_DOUBLE_TENSOR(grad_A);
        CHECK_DOUBLE_TENSOR(grad_B);
        matrix_multiply_backward_kernel<double><<<gridsize, blocksize>>>(
            A.data_ptr<double>(), B.data_ptr<double>(),
            grad_C.data_ptr<double>(), N, grad_A.data_ptr<double>(),
            grad_B.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", A.dtype());
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "CUDA error after matrix_multiply_backward: ", cudaGetErrorString(err));
}

template <typename T>
__global__ void covariance_matrix_backward_kernel(
    const T* __restrict__ RS, const T* __restrict__ grad_cov3d, const int N,
    T* __restrict__ grad_RS) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const T* RS_ptr = RS + i * 9;
    const T* grad_cov3d_ptr = grad_cov3d + i * 9;
    T* grad_RS_ptr = grad_RS + i * 9;

    T grad_cov3d_T[9];
    transpose<T>(grad_cov3d_ptr, grad_cov3d_T, 3, 3);

    T grad_sum[9];
#pragma unroll
    for (int j = 0; j < 9; ++j) {
        grad_sum[j] = grad_cov3d_ptr[j] + grad_cov3d_T[j];
    }

    matrix_multiply<T>(grad_sum, RS_ptr, grad_RS_ptr, 3, 3, 3);
}

void covariance_matrix_backward_cuda(torch::Tensor RS, torch::Tensor grad_cov3d,
                                     torch::Tensor grad_RS) {
    CHECK_VALID_INPUT(RS);
    CHECK_VALID_INPUT(grad_cov3d);
    CHECK_VALID_INPUT(grad_RS);

    const int N = RS.size(0);
    TORCH_CHECK(RS.size(1) == 3 && RS.size(2) == 3, "RS must have shape Nx3x3");
    TORCH_CHECK(grad_cov3d.size(0) == N && grad_cov3d.size(1) == 3 &&
                    grad_cov3d.size(2) == 3,
                "grad_cov3d must have shape Nx3x3");
    TORCH_CHECK(
        grad_RS.size(0) == N && grad_RS.size(1) == 3 && grad_RS.size(2) == 3,
        "grad_RS must have shape Nx3x3");

    const int num_blocks =
        (N + THREADS_PER_BLOCK_PROJ_BW - 1) / THREADS_PER_BLOCK_PROJ_BW;
    dim3 gridsize(num_blocks);
    dim3 blocksize(THREADS_PER_BLOCK_PROJ_BW);

    if (RS.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(grad_cov3d);
        CHECK_FLOAT_TENSOR(grad_RS);
        covariance_matrix_backward_kernel<float><<<gridsize, blocksize>>>(
            RS.data_ptr<float>(), grad_cov3d.data_ptr<float>(), N,
            grad_RS.data_ptr<float>());
    } else if (RS.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(grad_cov3d);
        CHECK_DOUBLE_TENSOR(grad_RS);
        covariance_matrix_backward_kernel<double><<<gridsize, blocksize>>>(
            RS.data_ptr<double>(), grad_cov3d.data_ptr<double>(), N,
            grad_RS.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", RS.dtype());
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA error after covariance_matrix_backward: ",
                cudaGetErrorString(err));
}

template <typename T>
__global__ void project_to_channel_coords_backward_kernel(
    const T* __restrict__ distances, const T* __restrict__ displacement,
    const T* __restrict__ grad_distances,
    const T* __restrict__ grad_displacement, const T* __restrict__ grad_uv,
    const int num_tx, const int num_rx, const int N,
    T* __restrict__ grad_points) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const T r = distances[i];
    const T r_safe = max(r, T(ROBUST_EPSILON));
    const T inv_r_safe = T(1.0) / r_safe;

    const T x = displacement[i * 3 + 0];
    const T y = displacement[i * 3 + 1];
    const T z = displacement[i * 3 + 2];

    const T grad_r = grad_distances[i];
    const T grad_dx = grad_displacement[i * 3 + 0];
    const T grad_dy = grad_displacement[i * 3 + 1];
    const T grad_dz = grad_displacement[i * 3 + 2];
    const T grad_u = grad_uv[i * 2 + 0];
    const T grad_v = grad_uv[i * 2 + 1];

    T grad_px = grad_r * x * inv_r_safe + grad_dx;
    T grad_py = grad_r * y * inv_r_safe + grad_dy;
    T grad_pz = grad_r * z * inv_r_safe + grad_dz;

    constexpr T PI = T(M_PI);
    constexpr T INV_PI = T(1.0) / PI;
    constexpr T TWO_OVER_PI = T(2.0) / PI;

    const T grad_s_x = grad_u * (num_tx - T(1.0)) * T(0.5);
    const T grad_s_y = grad_v * (num_rx - T(1.0)) * T(0.5);

    const T grad_longitude = grad_s_x * INV_PI;
    const T grad_latitude = grad_s_y * TWO_OVER_PI;

    const T xy_squared = x * x + y * y;
    const T xy_squared_safe = max(xy_squared, T(ROBUST_EPSILON));
    const T inv_xy_squared_safe = T(1.0) / xy_squared_safe;

    grad_px += grad_longitude * (-y * inv_xy_squared_safe);
    grad_py += grad_longitude * (x * inv_xy_squared_safe);

    const T dz_r = z * inv_r_safe;
    const T dz_r_clamped = min(max(dz_r, T(-1.0)), T(1.0));
    const T cos_lat_sq = T(1.0) - dz_r_clamped * dz_r_clamped;
    const T cos_lat = sqrt(max(cos_lat_sq, T(ROBUST_EPSILON)));
    const T cos_lat_safe = max(cos_lat, T(ROBUST_EPSILON));

    const T r_cubed_cos_lat = r_safe * r_safe * r_safe * cos_lat_safe;
    const T inv_r_cubed_cos_lat = T(1.0) / max(r_cubed_cos_lat, T(1e-15));

    grad_px += grad_latitude * (-x * z) * inv_r_cubed_cos_lat;
    grad_py += grad_latitude * (-y * z) * inv_r_cubed_cos_lat;
    grad_pz += grad_latitude * xy_squared * inv_r_cubed_cos_lat;

    grad_points[i * 3 + 0] = grad_px;
    grad_points[i * 3 + 1] = grad_py;
    grad_points[i * 3 + 2] = grad_pz;
}

void project_to_channel_coords_backward_cuda(
    torch::Tensor points, torch::Tensor receiver, torch::Tensor distances,
    torch::Tensor displacement, torch::Tensor grad_distances,
    torch::Tensor grad_displacement, torch::Tensor grad_uv, int num_tx,
    int num_rx, torch::Tensor grad_points) {
    CHECK_VALID_INPUT(points);
    CHECK_VALID_INPUT(receiver);
    CHECK_VALID_INPUT(distances);
    CHECK_VALID_INPUT(displacement);
    CHECK_VALID_INPUT(grad_distances);
    CHECK_VALID_INPUT(grad_displacement);
    CHECK_VALID_INPUT(grad_uv);
    CHECK_VALID_INPUT(grad_points);

    const int N = points.size(0);
    TORCH_CHECK(points.size(1) == 3, "points must have shape Nx3");
    TORCH_CHECK(receiver.size(0) == 3, "receiver must have shape 3");
    TORCH_CHECK(distances.size(0) == N, "distances must have shape N");
    TORCH_CHECK(displacement.size(0) == N && displacement.size(1) == 3,
                "displacement must have shape Nx3");
    TORCH_CHECK(grad_distances.size(0) == N,
                "grad_distances must have shape N");
    TORCH_CHECK(
        grad_displacement.size(0) == N && grad_displacement.size(1) == 3,
        "grad_displacement must have shape Nx3");
    TORCH_CHECK(grad_uv.size(0) == N && grad_uv.size(1) == 2,
                "grad_uv must have shape Nx2");
    TORCH_CHECK(grad_points.size(0) == N && grad_points.size(1) == 3,
                "grad_points must have shape Nx3");

    const int num_blocks =
        (N + THREADS_PER_BLOCK_PROJ_BW - 1) / THREADS_PER_BLOCK_PROJ_BW;
    dim3 gridsize(num_blocks);
    dim3 blocksize(THREADS_PER_BLOCK_PROJ_BW);

    if (points.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(receiver);
        CHECK_FLOAT_TENSOR(distances);
        CHECK_FLOAT_TENSOR(displacement);
        CHECK_FLOAT_TENSOR(grad_distances);
        CHECK_FLOAT_TENSOR(grad_displacement);
        CHECK_FLOAT_TENSOR(grad_uv);
        CHECK_FLOAT_TENSOR(grad_points);
        project_to_channel_coords_backward_kernel<float>
            <<<gridsize, blocksize>>>(
                distances.data_ptr<float>(), displacement.data_ptr<float>(),
                grad_distances.data_ptr<float>(),
                grad_displacement.data_ptr<float>(), grad_uv.data_ptr<float>(),
                num_tx, num_rx, N, grad_points.data_ptr<float>());
    } else if (points.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(receiver);
        CHECK_DOUBLE_TENSOR(distances);
        CHECK_DOUBLE_TENSOR(displacement);
        CHECK_DOUBLE_TENSOR(grad_distances);
        CHECK_DOUBLE_TENSOR(grad_displacement);
        CHECK_DOUBLE_TENSOR(grad_uv);
        CHECK_DOUBLE_TENSOR(grad_points);
        project_to_channel_coords_backward_kernel<double>
            <<<gridsize, blocksize>>>(
                distances.data_ptr<double>(), displacement.data_ptr<double>(),
                grad_distances.data_ptr<double>(),
                grad_displacement.data_ptr<double>(),
                grad_uv.data_ptr<double>(), num_tx, num_rx, N,
                grad_points.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", points.dtype());
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA error after project_to_channel_coords_backward: ",
                cudaGetErrorString(err));
}

template <typename T>
__global__ void compute_jacobian_backward_kernel(const T* __restrict__ d,
                                                 const T* __restrict__ grad_J,
                                                 const int num_tx,
                                                 const int num_rx, const int N,
                                                 T* __restrict__ grad_d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const T x = d[i * 3 + 0];
    const T y = d[i * 3 + 1];
    const T z = d[i * 3 + 2];

    const T* grad_J_ptr = grad_J + i * 6;
    const T L11 = grad_J_ptr[0];
    const T L12 = grad_J_ptr[1];
    const T L21 = grad_J_ptr[3];
    const T L22 = grad_J_ptr[4];
    const T L23 = grad_J_ptr[5];

    constexpr T PI = T(M_PI);
    const T t = (num_tx - T(1.0)) / (T(2.0) * PI);  // tx_factor
    const T r_factor = (num_rx - T(1.0)) / PI;      // rx_factor

    const T x2 = x * x;
    const T y2 = y * y;
    const T xy = x * y;
    const T xy_sq = x2 + y2;
    const T xy_sq_safe = max(xy_sq, T(ROBUST_EPSILON));
    const T sqrt_xy_sq_safe = sqrt(xy_sq_safe);
    const T xy_sq_safe_sq = xy_sq_safe * xy_sq_safe;
    const T denom_xy = max(sqrt_xy_sq_safe * xy_sq_safe_sq, T(ROBUST_EPSILON));
    const T inv_denom_xy = T(1.0) / denom_xy;

    const T num_dx =
        T(2.0) * L11 * t * xy * sqrt_xy_sq_safe -
        L12 * t * x2 * sqrt_xy_sq_safe + L12 * t * y2 * sqrt_xy_sq_safe -
        T(2.0) * L21 * r_factor * x2 * z + L21 * r_factor * y2 * z -
        T(3.0) * L22 * r_factor * xy * z - L23 * r_factor * x * xy_sq_safe;
    const T grad_d_x = num_dx * inv_denom_xy;

    const T num_dy =
        -L11 * t * x2 * sqrt_xy_sq_safe + L11 * t * y2 * sqrt_xy_sq_safe -
        T(2.0) * L12 * t * xy * sqrt_xy_sq_safe -
        T(3.0) * L21 * r_factor * xy * z + L22 * r_factor * x2 * z -
        T(2.0) * L22 * r_factor * y2 * z - L23 * r_factor * y * xy_sq_safe;
    const T grad_d_y = num_dy * inv_denom_xy;

    const T xy_sq_pow_1_5 = max(pow(xy_sq_safe, T(1.5)), T(ROBUST_EPSILON));
    const T inv_xy_sq_pow_1_5 = T(1.0) / xy_sq_pow_1_5;
    const T grad_d_z = r_factor * (L21 * x + L22 * y) * inv_xy_sq_pow_1_5;

    T* grad_d_ptr = grad_d + i * 3;
    grad_d_ptr[0] = grad_d_x;
    grad_d_ptr[1] = grad_d_y;
    grad_d_ptr[2] = grad_d_z;
}

void compute_jacobian_backward_cuda(torch::Tensor d, torch::Tensor grad_J,
                                    int num_tx, int num_rx,
                                    torch::Tensor grad_d) {
    CHECK_VALID_INPUT(d);
    CHECK_VALID_INPUT(grad_J);
    CHECK_VALID_INPUT(grad_d);

    const int N = d.size(0);
    TORCH_CHECK(d.size(1) == 3, "d must have shape Nx3");
    TORCH_CHECK(
        grad_J.size(0) == N && grad_J.size(1) == 2 && grad_J.size(2) == 3,
        "grad_J must have shape Nx2x3");
    TORCH_CHECK(grad_d.size(0) == N && grad_d.size(1) == 3,
                "grad_d must have shape Nx3");

    const int num_blocks =
        (N + THREADS_PER_BLOCK_PROJ_BW - 1) / THREADS_PER_BLOCK_PROJ_BW;
    dim3 gridsize(num_blocks);
    dim3 blocksize(THREADS_PER_BLOCK_PROJ_BW);

    if (d.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(grad_J);
        CHECK_FLOAT_TENSOR(grad_d);
        compute_jacobian_backward_kernel<float><<<gridsize, blocksize>>>(
            d.data_ptr<float>(), grad_J.data_ptr<float>(), num_tx, num_rx, N,
            grad_d.data_ptr<float>());
    } else if (d.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(grad_J);
        CHECK_DOUBLE_TENSOR(grad_d);
        compute_jacobian_backward_kernel<double><<<gridsize, blocksize>>>(
            d.data_ptr<double>(), grad_J.data_ptr<double>(), num_tx, num_rx, N,
            grad_d.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", d.dtype());
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA error after compute_jacobian_backward: ",
                cudaGetErrorString(err));
}

template <typename T>
__global__ void project_cov3d_to_cov2d_backward_kernel(
    const T* __restrict__ cov3d, const T* __restrict__ J,
    const T* __restrict__ grad_cov2d, const int N, T* __restrict__ grad_cov3d,
    T* __restrict__ grad_J) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const T* J_ptr = J + i * 6;
    const T* cov3d_ptr = cov3d + i * 9;
    const T* grad_cov2d_ptr = grad_cov2d + i * 4;
    T* grad_cov3d_ptr = grad_cov3d + i * 9;
    T* grad_J_ptr = grad_J + i * 6;

    T J_T[6];
    transpose<T>(J_ptr, J_T, 2, 3);

    T temp_grad_J[6];
    matrix_multiply<T>(grad_cov2d_ptr, J_ptr, temp_grad_J, 2, 2, 3);
    matrix_multiply<T>(J_T, temp_grad_J, grad_cov3d_ptr, 3, 2, 3);

    T J_cov3d[6];
    matrix_multiply<T>(J_ptr, cov3d_ptr, J_cov3d, 2, 3, 3);

    T term1[6];
    matrix_multiply<T>(grad_cov2d_ptr, J_cov3d, term1, 2, 2, 3);

    T grad_cov2d_T[4];
    transpose<T>(grad_cov2d_ptr, grad_cov2d_T, 2, 2);

    T term2[6];
    matrix_multiply<T>(grad_cov2d_T, J_cov3d, term2, 2, 2, 3);

#pragma unroll
    for (int j = 0; j < 6; j++) {
        grad_J_ptr[j] = term1[j] + term2[j];
    }
}

void project_cov3d_to_cov2d_backward_cuda(torch::Tensor cov3d, torch::Tensor J,
                                          torch::Tensor grad_cov2d,
                                          torch::Tensor grad_cov3d,
                                          torch::Tensor grad_J) {
    CHECK_VALID_INPUT(cov3d);
    CHECK_VALID_INPUT(J);
    CHECK_VALID_INPUT(grad_cov2d);
    CHECK_VALID_INPUT(grad_cov3d);
    CHECK_VALID_INPUT(grad_J);

    const int N = cov3d.size(0);
    TORCH_CHECK(cov3d.size(1) == 3 && cov3d.size(2) == 3,
                "cov3d must have shape Nx3x3");
    TORCH_CHECK(J.size(0) == N && J.size(1) == 2 && J.size(2) == 3,
                "J must have shape Nx2x3");
    TORCH_CHECK(grad_cov2d.size(0) == N && grad_cov2d.size(1) == 2 &&
                    grad_cov2d.size(2) == 2,
                "grad_cov2d must have shape Nx2x2");
    TORCH_CHECK(grad_cov3d.size(0) == N && grad_cov3d.size(1) == 3 &&
                    grad_cov3d.size(2) == 3,
                "grad_cov3d must have shape Nx3x3");
    TORCH_CHECK(
        grad_J.size(0) == N && grad_J.size(1) == 2 && grad_J.size(2) == 3,
        "grad_J must have shape Nx2x3");

    const int num_blocks =
        (N + THREADS_PER_BLOCK_PROJ_BW - 1) / THREADS_PER_BLOCK_PROJ_BW;
    dim3 gridsize(num_blocks);
    dim3 blocksize(THREADS_PER_BLOCK_PROJ_BW);

    if (cov3d.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(J);
        CHECK_FLOAT_TENSOR(grad_cov2d);
        CHECK_FLOAT_TENSOR(grad_cov3d);
        CHECK_FLOAT_TENSOR(grad_J);
        project_cov3d_to_cov2d_backward_kernel<float><<<gridsize, blocksize>>>(
            cov3d.data_ptr<float>(), J.data_ptr<float>(),
            grad_cov2d.data_ptr<float>(), N, grad_cov3d.data_ptr<float>(),
            grad_J.data_ptr<float>());
    } else if (cov3d.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(J);
        CHECK_DOUBLE_TENSOR(grad_cov2d);
        CHECK_DOUBLE_TENSOR(grad_cov3d);
        CHECK_DOUBLE_TENSOR(grad_J);
        project_cov3d_to_cov2d_backward_kernel<double><<<gridsize, blocksize>>>(
            cov3d.data_ptr<double>(), J.data_ptr<double>(),
            grad_cov2d.data_ptr<double>(), N, grad_cov3d.data_ptr<double>(),
            grad_J.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", cov3d.dtype());
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA error after project_cov3d_to_cov2d_backward: ",
                cudaGetErrorString(err));
}
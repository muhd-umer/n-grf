// engine__cuda_impl/projection_backward.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "checks.cuh"
#include "matrix.cuh"

template <typename T>
__global__ void quaternion_to_rotation_backward_kernel(
    const T* __restrict__ quaternion, const T* __restrict__ grad_rotation,
    const int N, T* __restrict__ grad_quaternion) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    // Extract quaternion components
    T w = quaternion[i * 4 + 0];
    T x = quaternion[i * 4 + 1];
    T y = quaternion[i * 4 + 2];
    T z = quaternion[i * 4 + 3];

    // Normalize quaternion (redundant if already normalized)
    T norm = sqrt(w * w + x * x + y * y + z * z);
    w /= norm;
    x /= norm;
    y /= norm;
    z /= norm;

    // Extract gradient components
    T dR00 = grad_rotation[i * 9 + 0];
    T dR01 = grad_rotation[i * 9 + 1];
    T dR02 = grad_rotation[i * 9 + 2];
    T dR10 = grad_rotation[i * 9 + 3];
    T dR11 = grad_rotation[i * 9 + 4];
    T dR12 = grad_rotation[i * 9 + 5];
    T dR20 = grad_rotation[i * 9 + 6];
    T dR21 = grad_rotation[i * 9 + 7];
    T dR22 = grad_rotation[i * 9 + 8];

    // Compute gradients with respect to normalized quaternion
    T grad_w = -2 * z * dR01 + 2 * y * dR02 + 2 * z * dR10 - 2 * x * dR12 -
               2 * y * dR20 + 2 * x * dR21;

    T grad_x = 2 * y * dR01 + 2 * z * dR02 + 2 * y * dR10 - 4 * x * dR11 -
               2 * w * dR12 + 2 * z * dR20 + 2 * w * dR21 - 4 * x * dR22;

    T grad_y = -4 * y * dR00 + 2 * x * dR01 + 2 * w * dR02 + 2 * x * dR10 +
               2 * z * dR12 - 2 * w * dR20 + 2 * z * dR21 - 4 * y * dR22;

    T grad_z = -4 * z * dR00 - 2 * w * dR01 + 2 * x * dR02 + 2 * w * dR10 -
               4 * z * dR11 + 2 * y * dR12 + 2 * x * dR20 + 2 * y * dR21;

    // Apply chain rule for quaternion normalization
    T q_norm_cubed = norm * norm * norm;

    T grad_w_unnorm = (1.0 / norm - w * w / q_norm_cubed) * grad_w -
                      (w * x / q_norm_cubed) * grad_x -
                      (w * y / q_norm_cubed) * grad_y -
                      (w * z / q_norm_cubed) * grad_z;

    T grad_x_unnorm = -(w * x / q_norm_cubed) * grad_w +
                      (1.0 / norm - x * x / q_norm_cubed) * grad_x -
                      (x * y / q_norm_cubed) * grad_y -
                      (x * z / q_norm_cubed) * grad_z;

    T grad_y_unnorm = -(w * y / q_norm_cubed) * grad_w -
                      (x * y / q_norm_cubed) * grad_x +
                      (1.0 / norm - y * y / q_norm_cubed) * grad_y -
                      (y * z / q_norm_cubed) * grad_z;

    T grad_z_unnorm = -(w * z / q_norm_cubed) * grad_w -
                      (x * z / q_norm_cubed) * grad_x -
                      (y * z / q_norm_cubed) * grad_y +
                      (1.0 / norm - z * z / q_norm_cubed) * grad_z;

    // Set output gradients
    grad_quaternion[i * 4 + 0] = grad_w_unnorm;
    grad_quaternion[i * 4 + 1] = grad_x_unnorm;
    grad_quaternion[i * 4 + 2] = grad_y_unnorm;
    grad_quaternion[i * 4 + 3] = grad_z_unnorm;
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

    const int max_threads_per_block = 1024;
    const int num_blocks =
        (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

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
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void compute_scaling_matrix_backward_kernel(
    const T* __restrict__ scaling, const T* __restrict__ grad_scaling_matrix,
    const T scale_modifier, const int N, T* __restrict__ grad_scaling) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    // Only diagonal elements have non-zero gradients
    const T sx = exp(scaling[i * 3 + 0]) * scale_modifier;
    const T sy = exp(scaling[i * 3 + 1]) * scale_modifier;
    const T sz = exp(scaling[i * 3 + 2]) * scale_modifier;

    // Gradient propagation through exponential
    grad_scaling[i * 3 + 0] = grad_scaling_matrix[i * 9 + 0] * sx;
    grad_scaling[i * 3 + 1] = grad_scaling_matrix[i * 9 + 4] * sy;
    grad_scaling[i * 3 + 2] = grad_scaling_matrix[i * 9 + 8] * sz;
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

    const int max_threads_per_block = 1024;
    const int num_blocks =
        (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    if (scaling.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(grad_scaling_matrix);
        CHECK_FLOAT_TENSOR(grad_scaling);
        compute_scaling_matrix_backward_kernel<float><<<gridsize, blocksize>>>(
            scaling.data_ptr<float>(), grad_scaling_matrix.data_ptr<float>(),
            scale_modifier, N, grad_scaling.data_ptr<float>());
    } else if (scaling.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(grad_scaling_matrix);
        CHECK_DOUBLE_TENSOR(grad_scaling);
        compute_scaling_matrix_backward_kernel<double><<<gridsize, blocksize>>>(
            scaling.data_ptr<double>(), grad_scaling_matrix.data_ptr<double>(),
            scale_modifier, N, grad_scaling.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", scaling.dtype());
    }
    cudaDeviceSynchronize();
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

    // Create B^T
    T B_T[9];
    transpose<T>(B + i * 9, B_T, 3, 3);

    // grad_A = grad_C @ B^T
    matrix_multiply<T>(grad_C + i * 9, B_T, grad_A + i * 9, 3, 3, 3);

    // Create A^T
    T A_T[9];
    transpose<T>(A + i * 9, A_T, 3, 3);

    // grad_B = A^T @ grad_C
    matrix_multiply<T>(A_T, grad_C + i * 9, grad_B + i * 9, 3, 3, 3);
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

    const int max_threads_per_block = 1024;
    const int num_blocks =
        (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

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
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void covariance_matrix_backward_kernel(
    const T* __restrict__ RS, const T* __restrict__ grad_cov3d, const int N,
    T* __restrict__ grad_RS) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    // Create grad_cov3d + grad_cov3d^T
    T grad_cov3d_T[9];
    transpose<T>(grad_cov3d + i * 9, grad_cov3d_T, 3, 3);

    T grad_cov3d_sym[9];
    for (int j = 0; j < 9; j++) {
        grad_cov3d_sym[j] = grad_cov3d[i * 9 + j] + grad_cov3d_T[j];
    }

    // grad_RS = (grad_cov3d + grad_cov3d^T) @ RS
    matrix_multiply<T>(grad_cov3d_sym, RS + i * 9, grad_RS + i * 9, 3, 3, 3);
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

    const int max_threads_per_block = 1024;
    const int num_blocks =
        (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

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
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void project_to_channel_coords_backward_kernel(
    const T* __restrict__ points, const T* __restrict__ receiver,
    const T* __restrict__ distances, const T* __restrict__ displacement,
    const T* __restrict__ grad_distances,
    const T* __restrict__ grad_displacement, const T* __restrict__ grad_uv,
    const int num_tx, const int num_rx, const int N,
    T* __restrict__ grad_points) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    // Extract input values
    const T r = distances[i];
    const T x = displacement[i * 3 + 0];
    const T y = displacement[i * 3 + 1];
    const T z = displacement[i * 3 + 2];

    // Extract gradients
    const T grad_r = grad_distances[i];

    const T grad_dx = grad_displacement[i * 3 + 0];
    const T grad_dy = grad_displacement[i * 3 + 1];
    const T grad_dz = grad_displacement[i * 3 + 2];

    const T grad_u = grad_uv[i * 2 + 0];
    const T grad_v = grad_uv[i * 2 + 1];

    // Initialize gradient
    T grad_px = 0, grad_py = 0, grad_pz = 0;

    // Gradient from distances: dr/dpoints = d/r
    grad_px += grad_r * x / r;
    grad_py += grad_r * y / r;
    grad_pz += grad_r * z / r;

    // Gradient from displacement vectors: dd/dpoints = I (identity)
    grad_px += grad_dx;
    grad_py += grad_dy;
    grad_pz += grad_dz;

    // Constants
    const T PI = T(3.14159265358979323846);

    // Gradients from uv coordinates through transformation
    // chain rule through transformations
    // s_x to u, s_y to v
    const T grad_s_x = grad_u * (num_tx - T(1)) / T(2);
    const T grad_s_y = grad_v * (num_rx - T(1)) / T(2);

    // longitude to s_x, latitude to s_y
    const T grad_longitude = grad_s_x / PI;
    const T grad_latitude = grad_s_y * T(2) / PI;

    // positions to longitude and latitude
    T xy_squared = x * x + y * y + T(1e-10);  // add epsilon for stability
    grad_px += grad_longitude * (-y / xy_squared);
    grad_py += grad_longitude * (x / xy_squared);

    T dz_r = z / r;
    dz_r = min(max(dz_r, T(-1.0)), T(1.0));  // Clamp to [-1, 1]
    T cos_lat = sqrt(T(1.0) - dz_r * dz_r) + T(1e-10);

    grad_px += grad_latitude * (-x * z) / (r * r * r * cos_lat);
    grad_py += grad_latitude * (-y * z) / (r * r * r * cos_lat);
    grad_pz += grad_latitude * (xy_squared) / (r * r * r * cos_lat);

    // Set output gradients
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

    const int max_threads_per_block = 1024;
    const int num_blocks =
        (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

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
                points.data_ptr<float>(), receiver.data_ptr<float>(),
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
                points.data_ptr<double>(), receiver.data_ptr<double>(),
                distances.data_ptr<double>(), displacement.data_ptr<double>(),
                grad_distances.data_ptr<double>(),
                grad_displacement.data_ptr<double>(),
                grad_uv.data_ptr<double>(), num_tx, num_rx, N,
                grad_points.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", points.dtype());
    }
    cudaDeviceSynchronize();
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

    // Extract displacement components
    const T x = d[i * 3 + 0];
    const T y = d[i * 3 + 1];
    const T z = d[i * 3 + 2];

    // Extract gradient components
    const T L11 = grad_J[i * 6 + 0];  // dJ[0,0]/d(...)
    const T L12 = grad_J[i * 6 + 1];  // dJ[0,1]/d(...)
    // L13 is ignored (since J[0,2] is 0)
    const T L21 = grad_J[i * 6 + 3];  // dJ[1,0]/d(...)
    const T L22 = grad_J[i * 6 + 4];  // dJ[1,1]/d(...)
    const T L23 = grad_J[i * 6 + 5];  // dJ[1,2]/d(...)

    // Constants
    const T PI = T(3.14159265358979323846);
    const T tx_factor = (num_tx - T(1)) / (T(2.0) * PI);
    const T rx_factor = (num_rx - T(1)) / PI;

    // Intermediate calculations
    const T xy_sq = x * x + y * y;
    const T sqrt_xy = sqrt(max(xy_sq, T(1e-10)));
    const T denom2 = xy_sq * xy_sq + T(1e-10);  // (x²+y²)²

    // Compute gradients
    // compute gradients for x using derived formula
    const T grad_d_x =
        (2 * L11 * tx_factor * x * y * sqrt_xy -
         L12 * tx_factor * x * x * sqrt_xy + L12 * tx_factor * y * y * sqrt_xy -
         2 * L21 * rx_factor * x * x * z + L21 * rx_factor * y * y * z -
         3 * L22 * rx_factor * x * y * z - L23 * rx_factor * x * xy_sq) /
        (sqrt_xy * denom2);

    // compute gradients for y using derived formula
    const T grad_d_y =
        (-L11 * tx_factor * x * x * sqrt_xy +
         L11 * tx_factor * y * y * sqrt_xy -
         2 * L12 * tx_factor * x * y * sqrt_xy -
         3 * L21 * rx_factor * x * y * z + L22 * rx_factor * x * x * z -
         2 * L22 * rx_factor * y * y * z - L23 * rx_factor * y * xy_sq) /
        (sqrt_xy * denom2);

    // compute gradients for z using derived formula
    const T grad_d_z =
        rx_factor * (L21 * x + L22 * y) / (pow(xy_sq, T(1.5)) + T(1e-10));

    // Set output gradients
    grad_d[i * 3 + 0] = grad_d_x;
    grad_d[i * 3 + 1] = grad_d_y;
    grad_d[i * 3 + 2] = grad_d_z;
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

    const int max_threads_per_block = 1024;
    const int num_blocks =
        (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

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
    cudaDeviceSynchronize();
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

    // ----------------
    // Gradient wrt cov3d
    // ----------------
    // Create J^T
    T J_T[6];  // 3x2
    transpose<T>(J + i * 6, J_T, 2, 3);

    // Create temp = J^T @ grad_cov2d
    T temp[6];  // 3x2
    matrix_multiply<T>(J_T, grad_cov2d + i * 4, temp, 3, 2, 2);

    // Create grad_cov3d = temp @ J
    matrix_multiply<T>(temp, J + i * 6, grad_cov3d + i * 9, 3, 2, 3);

    // ----------------
    // Gradient wrt J
    // ----------------
    // Create temp1 = grad_cov2d @ J
    T temp1[6];  // 2x3
    matrix_multiply<T>(grad_cov2d + i * 4, J + i * 6, temp1, 2, 2, 3);

    // Create temp2 = temp1 @ cov3d = grad_cov2d @ J @ cov3d
    T temp2[6];  // 2x3
    matrix_multiply<T>(temp1, cov3d + i * 9, temp2, 2, 3, 3);

    // Create grad_cov2d^T
    T grad_cov2d_T[4];  // 2x2
    transpose<T>(grad_cov2d + i * 4, grad_cov2d_T, 2, 2);

    // Create temp3 = grad_cov2d^T @ J
    T temp3[6];  // 2x3
    matrix_multiply<T>(grad_cov2d_T, J + i * 6, temp3, 2, 2, 3);

    // Create temp4 = temp3 @ cov3d = grad_cov2d^T @ J @ cov3d
    T temp4[6];  // 2x3
    matrix_multiply<T>(temp3, cov3d + i * 9, temp4, 2, 3, 3);

    // grad_J = temp2 + temp4
    for (int j = 0; j < 6; j++) {
        grad_J[i * 6 + j] = temp2[j] + temp4[j];
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

    const int max_threads_per_block = 1024;
    const int num_blocks =
        (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

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
    cudaDeviceSynchronize();
}
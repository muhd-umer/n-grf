// engine/_cuda_impl/rasterize_backward.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

#include "checks.cuh"
#include "matrix.cuh"

constexpr int RAST_BW_BLOCK_DIM_X = 4;
constexpr int RAST_BW_BLOCK_DIM_Y = 8;
constexpr int RAST_BW_BLOCK_DIM_Z = 8;

constexpr int SV_BW_BLOCK_DIM_X = 16;
constexpr int SV_BW_BLOCK_DIM_Y = 16;

constexpr int SCAT_BW_BLOCK_DIM_X = 4;
constexpr int SCAT_BW_BLOCK_DIM_Y = 8;
constexpr int SCAT_BW_BLOCK_DIM_Z = 8;

constexpr int WS_BW_BLOCK_DIM_X = 4;
constexpr int WS_BW_BLOCK_DIM_Y = 8;
constexpr int WS_BW_BLOCK_DIM_Z = 8;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

template <typename T>
__global__ void compute_spatial_influence_backward_kernel(
    const T* __restrict__ uv, const T* __restrict__ cov2d,
    const T* __restrict__ influences, const T* __restrict__ grad_influences,
    const int num_tx, const int num_rx, const int N, T* __restrict__ grad_uv,
    T* __restrict__ grad_cov2d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int tx = blockIdx.y * blockDim.y + threadIdx.y;
    const int rx = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N || tx >= num_tx || rx >= num_rx) {
        return;
    }

    const int influence_idx = i * num_tx * num_rx + tx * num_rx + rx;
    const T grad_influence = grad_influences[influence_idx];

    if (abs(grad_influence) < T(1e-15)) {
        return;
    }

    const T* cov2d_ptr = cov2d + i * 4;
    const T a = cov2d_ptr[0];
    const T b = cov2d_ptr[1];
    const T c = cov2d_ptr[2];
    const T d_cov = cov2d_ptr[3];

    const T det = a * d_cov - b * c;
    const T inv_det = T(1.0) / max(det, T(ROBUST_EPSILON));

    T inv_cov[4];
    inv_cov[0] = d_cov * inv_det;
    inv_cov[1] = -b * inv_det;
    inv_cov[2] = -c * inv_det;
    inv_cov[3] = a * inv_det;

    const T u_i = uv[i * 2 + 0];
    const T v_i = uv[i * 2 + 1];

    const T antenna_u = T(tx) + T(0.5);
    const T antenna_v = T(rx) + T(0.5);

    T disp[2];
    disp[0] = u_i - antenna_u;
    disp[1] = v_i - antenna_v;

    const T influence = influences[influence_idx];
    const T grad_infl_times_infl = grad_influence * influence;

    T grad_component[2];
    grad_component[0] = -(inv_cov[0] * disp[0] + inv_cov[1] * disp[1]);
    grad_component[1] = -(inv_cov[2] * disp[0] + inv_cov[3] * disp[1]);

    atomicAdd(&grad_uv[i * 2 + 0], grad_infl_times_infl * grad_component[0]);
    atomicAdd(&grad_uv[i * 2 + 1], grad_infl_times_infl * grad_component[1]);

    T d_outer[4];
    d_outer[0] = disp[0] * disp[0];
    d_outer[1] = disp[0] * disp[1];
    d_outer[2] = disp[1] * disp[0];
    d_outer[3] = disp[1] * disp[1];

    T temp_mat1[4];
    matrix_multiply<T>(inv_cov, d_outer, temp_mat1, 2, 2, 2);

    T temp_mat2[4];
    matrix_multiply<T>(temp_mat1, inv_cov, temp_mat2, 2, 2, 2);

    T factor = T(0.5) * grad_infl_times_infl;
    atomicAdd(&grad_cov2d[i * 4 + 0], factor * temp_mat2[0]);
    atomicAdd(&grad_cov2d[i * 4 + 1], factor * temp_mat2[1]);
    atomicAdd(&grad_cov2d[i * 4 + 2], factor * temp_mat2[2]);
    atomicAdd(&grad_cov2d[i * 4 + 3], factor * temp_mat2[3]);
}

void compute_spatial_influence_backward_cuda(
    torch::Tensor uv, torch::Tensor cov2d, torch::Tensor influences,
    torch::Tensor grad_influences, int num_tx, int num_rx,
    torch::Tensor grad_uv, torch::Tensor grad_cov2d) {
    CHECK_VALID_INPUT(uv);
    CHECK_VALID_INPUT(cov2d);
    CHECK_VALID_INPUT(influences);
    CHECK_VALID_INPUT(grad_influences);
    CHECK_VALID_INPUT(grad_uv);
    CHECK_VALID_INPUT(grad_cov2d);

    const int N = uv.size(0);
    TORCH_CHECK(uv.size(1) == 2, "uv must have shape Nx2");
    TORCH_CHECK(cov2d.size(0) == N && cov2d.size(1) == 2 && cov2d.size(2) == 2,
                "cov2d must have shape Nx2x2");
    TORCH_CHECK(influences.size(0) == N && influences.size(1) == num_tx &&
                    influences.size(2) == num_rx,
                "influences shape mismatch");
    TORCH_CHECK(grad_influences.size(0) == N &&
                    grad_influences.size(1) == num_tx &&
                    grad_influences.size(2) == num_rx,
                "grad_influences shape mismatch");
    TORCH_CHECK(grad_uv.size(0) == N && grad_uv.size(1) == 2,
                "grad_uv must have shape Nx2");
    TORCH_CHECK(grad_cov2d.size(0) == N && grad_cov2d.size(1) == 2 &&
                    grad_cov2d.size(2) == 2,
                "grad_cov2d must have shape Nx2x2");

    grad_uv.zero_();
    grad_cov2d.zero_();

    dim3 blocksize(RAST_BW_BLOCK_DIM_X, RAST_BW_BLOCK_DIM_Y,
                   RAST_BW_BLOCK_DIM_Z);
    dim3 gridsize((N + blocksize.x - 1) / blocksize.x,
                  (num_tx + blocksize.y - 1) / blocksize.y,
                  (num_rx + blocksize.z - 1) / blocksize.z);

    if (uv.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(cov2d);
        CHECK_FLOAT_TENSOR(influences);
        CHECK_FLOAT_TENSOR(grad_influences);
        CHECK_FLOAT_TENSOR(grad_uv);
        CHECK_FLOAT_TENSOR(grad_cov2d);
        compute_spatial_influence_backward_kernel<float>
            <<<gridsize, blocksize>>>(
                uv.data_ptr<float>(), cov2d.data_ptr<float>(),
                influences.data_ptr<float>(), grad_influences.data_ptr<float>(),
                num_tx, num_rx, N, grad_uv.data_ptr<float>(),
                grad_cov2d.data_ptr<float>());
    } else if (uv.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(cov2d);
        CHECK_DOUBLE_TENSOR(influences);
        CHECK_DOUBLE_TENSOR(grad_influences);
        CHECK_DOUBLE_TENSOR(grad_uv);
        CHECK_DOUBLE_TENSOR(grad_cov2d);
        compute_spatial_influence_backward_kernel<double>
            <<<gridsize, blocksize>>>(
                uv.data_ptr<double>(), cov2d.data_ptr<double>(),
                influences.data_ptr<double>(),
                grad_influences.data_ptr<double>(), num_tx, num_rx, N,
                grad_uv.data_ptr<double>(), grad_cov2d.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", uv.dtype());
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA error after compute_spatial_influence_backward: ",
                cudaGetErrorString(err));
}

template <typename T>
__global__ void compute_path_geometry_backward_kernel(
    const T* __restrict__ points, const T* __restrict__ tx_pos,
    const T* __restrict__ rx_pos, const T* __restrict__ dist_tx,
    const T* __restrict__ dist_rx, const T* __restrict__ grad_dist_tx,
    const T* __restrict__ grad_dist_rx, const T* __restrict__ grad_aod,
    const T* __restrict__ grad_aoa, const int N, T* __restrict__ grad_points) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const T tx_x = tx_pos[0];
    const T tx_y = tx_pos[1];
    const T tx_z = tx_pos[2];
    const T rx_x = rx_pos[0];
    const T rx_y = rx_pos[1];
    const T rx_z = rx_pos[2];

    const T px = points[i * 3 + 0];
    const T py = points[i * 3 + 1];
    const T pz = points[i * 3 + 2];

    const T vec_tx_gauss_x = px - tx_x;
    const T vec_tx_gauss_y = py - tx_y;
    const T vec_tx_gauss_z = pz - tx_z;

    const T vec_gauss_rx_x = rx_x - px;
    const T vec_gauss_rx_y = rx_y - py;
    const T vec_gauss_rx_z = rx_z - pz;

    const T d_tx = dist_tx[i];
    const T d_rx = dist_rx[i];
    const T inv_d_tx = T(1.0) / d_tx;
    const T inv_d_rx = T(1.0) / d_rx;

    const T grad_d_tx_val = grad_dist_tx[i];
    const T grad_d_rx_val = grad_dist_rx[i];
    const T grad_aod_az = grad_aod[i * 2 + 0];
    const T grad_aod_el = grad_aod[i * 2 + 1];
    const T grad_aoa_az = grad_aoa[i * 2 + 0];
    const T grad_aoa_el = grad_aoa[i * 2 + 1];

    T grad_px = T(0.0);
    T grad_py = T(0.0);
    T grad_pz = T(0.0);

    grad_px += grad_d_tx_val * vec_tx_gauss_x * inv_d_tx;
    grad_py += grad_d_tx_val * vec_tx_gauss_y * inv_d_tx;
    grad_pz += grad_d_tx_val * vec_tx_gauss_z * inv_d_tx;

    grad_px += grad_d_rx_val * (-vec_gauss_rx_x) * inv_d_rx;
    grad_py += grad_d_rx_val * (-vec_gauss_rx_y) * inv_d_rx;
    grad_pz += grad_d_rx_val * (-vec_gauss_rx_z) * inv_d_rx;

    const T x_tx = vec_tx_gauss_x;
    const T y_tx = vec_tx_gauss_y;
    const T z_tx = vec_tx_gauss_z;
    const T xy_sq_tx = x_tx * x_tx + y_tx * y_tx;
    const T xy_sq_tx_safe = max(xy_sq_tx, T(ROBUST_EPSILON));
    const T inv_xy_sq_tx_safe = T(1.0) / xy_sq_tx_safe;

    grad_px += grad_aod_az * (-y_tx * inv_xy_sq_tx_safe);
    grad_py += grad_aod_az * (x_tx * inv_xy_sq_tx_safe);

    T aod_el_arg = z_tx * inv_d_tx;
    aod_el_arg = min(max(aod_el_arg, T(-1.0)), T(1.0));
    T cos_aod_el_sq = T(1.0) - aod_el_arg * aod_el_arg;
    T cos_aod_el = sqrt(max(cos_aod_el_sq, T(ROBUST_EPSILON)));
    T r_cubed_cos_el_tx = d_tx * d_tx * d_tx * cos_aod_el;
    T inv_r_cubed_cos_el_tx = T(1.0) / max(r_cubed_cos_el_tx, T(1e-15));

    grad_px += grad_aod_el * (-x_tx * z_tx) * inv_r_cubed_cos_el_tx;
    grad_py += grad_aod_el * (-y_tx * z_tx) * inv_r_cubed_cos_el_tx;
    grad_pz += grad_aod_el * xy_sq_tx * inv_r_cubed_cos_el_tx;

    const T x_rx = vec_gauss_rx_x;
    const T y_rx = vec_gauss_rx_y;
    const T z_rx = vec_gauss_rx_z;
    const T xy_sq_rx = x_rx * x_rx + y_rx * y_rx;
    const T xy_sq_rx_safe = max(xy_sq_rx, T(ROBUST_EPSILON));
    const T inv_xy_sq_rx_safe = T(1.0) / xy_sq_rx_safe;

    grad_px += grad_aoa_az * (y_rx * inv_xy_sq_rx_safe);
    grad_py += grad_aoa_az * (-x_rx * inv_xy_sq_rx_safe);

    T aoa_el_arg = z_rx * inv_d_rx;
    aoa_el_arg = min(max(aoa_el_arg, T(-1.0)), T(1.0));
    T cos_aoa_el_sq = T(1.0) - aoa_el_arg * aoa_el_arg;
    T cos_aoa_el = sqrt(max(cos_aoa_el_sq, T(ROBUST_EPSILON)));
    T r_cubed_cos_el_rx = d_rx * d_rx * d_rx * cos_aoa_el;
    T inv_r_cubed_cos_el_rx = T(1.0) / max(r_cubed_cos_el_rx, T(1e-15));

    grad_px += grad_aoa_el * (x_rx * z_rx) * inv_r_cubed_cos_el_rx;
    grad_py += grad_aoa_el * (y_rx * z_rx) * inv_r_cubed_cos_el_rx;
    grad_pz += grad_aoa_el * (-xy_sq_rx) * inv_r_cubed_cos_el_rx;

    grad_points[i * 3 + 0] = grad_px;
    grad_points[i * 3 + 1] = grad_py;
    grad_points[i * 3 + 2] = grad_pz;
}

void compute_path_geometry_backward_cuda(
    torch::Tensor points, torch::Tensor tx_pos, torch::Tensor rx_pos,
    torch::Tensor dist_tx, torch::Tensor dist_rx, torch::Tensor aod,
    torch::Tensor aoa, torch::Tensor grad_dist_tx, torch::Tensor grad_dist_rx,
    torch::Tensor grad_aod, torch::Tensor grad_aoa, torch::Tensor grad_points,
    torch::Tensor grad_tx_pos, torch::Tensor grad_rx_pos) {
    CHECK_VALID_INPUT(points);
    CHECK_VALID_INPUT(tx_pos);
    CHECK_VALID_INPUT(rx_pos);
    CHECK_VALID_INPUT(dist_tx);
    CHECK_VALID_INPUT(dist_rx);
    CHECK_VALID_INPUT(aod);
    CHECK_VALID_INPUT(aoa);
    CHECK_VALID_INPUT(grad_dist_tx);
    CHECK_VALID_INPUT(grad_dist_rx);
    CHECK_VALID_INPUT(grad_aod);
    CHECK_VALID_INPUT(grad_aoa);
    CHECK_VALID_INPUT(grad_points);

    const int N = points.size(0);
    TORCH_CHECK(points.size(1) == 3, "points must have shape Nx3");
    TORCH_CHECK(tx_pos.size(0) == 3, "tx_pos must have shape 3");
    TORCH_CHECK(rx_pos.size(0) == 3, "rx_pos must have shape 3");
    TORCH_CHECK(dist_tx.size(0) == N, "dist_tx must have shape N");
    TORCH_CHECK(dist_rx.size(0) == N, "dist_rx must have shape N");
    TORCH_CHECK(aod.size(0) == N && aod.size(1) == 2, "aod shape mismatch");
    TORCH_CHECK(aoa.size(0) == N && aoa.size(1) == 2, "aoa shape mismatch");
    TORCH_CHECK(grad_dist_tx.size(0) == N, "grad_dist_tx shape mismatch");
    TORCH_CHECK(grad_dist_rx.size(0) == N, "grad_dist_rx shape mismatch");
    TORCH_CHECK(grad_aod.size(0) == N && grad_aod.size(1) == 2,
                "grad_aod shape mismatch");
    TORCH_CHECK(grad_aoa.size(0) == N && grad_aoa.size(1) == 2,
                "grad_aoa shape mismatch");
    TORCH_CHECK(grad_points.size(0) == N && grad_points.size(1) == 3,
                "grad_points shape mismatch");

    const int threads = 256;
    const int num_blocks = (N + threads - 1) / threads;
    dim3 gridsize(num_blocks);
    dim3 blocksize(threads);

    grad_points.zero_();

    if (points.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(tx_pos);
        CHECK_FLOAT_TENSOR(rx_pos);
        CHECK_FLOAT_TENSOR(dist_tx);
        CHECK_FLOAT_TENSOR(dist_rx);
        CHECK_FLOAT_TENSOR(aod);
        CHECK_FLOAT_TENSOR(aoa);
        CHECK_FLOAT_TENSOR(grad_dist_tx);
        CHECK_FLOAT_TENSOR(grad_dist_rx);
        CHECK_FLOAT_TENSOR(grad_aod);
        CHECK_FLOAT_TENSOR(grad_aoa);
        CHECK_FLOAT_TENSOR(grad_points);
        compute_path_geometry_backward_kernel<float><<<gridsize, blocksize>>>(
            points.data_ptr<float>(), tx_pos.data_ptr<float>(),
            rx_pos.data_ptr<float>(), dist_tx.data_ptr<float>(),
            dist_rx.data_ptr<float>(), grad_dist_tx.data_ptr<float>(),
            grad_dist_rx.data_ptr<float>(), grad_aod.data_ptr<float>(),
            grad_aoa.data_ptr<float>(), N, grad_points.data_ptr<float>());
    } else if (points.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(tx_pos);
        CHECK_DOUBLE_TENSOR(rx_pos);
        CHECK_DOUBLE_TENSOR(dist_tx);
        CHECK_DOUBLE_TENSOR(dist_rx);
        CHECK_DOUBLE_TENSOR(aod);
        CHECK_DOUBLE_TENSOR(aoa);
        CHECK_DOUBLE_TENSOR(grad_dist_tx);
        CHECK_DOUBLE_TENSOR(grad_dist_rx);
        CHECK_DOUBLE_TENSOR(grad_aod);
        CHECK_DOUBLE_TENSOR(grad_aoa);
        CHECK_DOUBLE_TENSOR(grad_points);
        compute_path_geometry_backward_kernel<double><<<gridsize, blocksize>>>(
            points.data_ptr<double>(), tx_pos.data_ptr<double>(),
            rx_pos.data_ptr<double>(), dist_tx.data_ptr<double>(),
            dist_rx.data_ptr<double>(), grad_dist_tx.data_ptr<double>(),
            grad_dist_rx.data_ptr<double>(), grad_aod.data_ptr<double>(),
            grad_aoa.data_ptr<double>(), N, grad_points.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", points.dtype());
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA error after compute_path_geometry_backward: ",
                cudaGetErrorString(err));
}

template <typename T>
__global__ void compute_steering_vector_backward_kernel(
    const T* __restrict__ angles, const T* __restrict__ array_size,
    const T* __restrict__ element_spacing, const int array_type,
    const T wavelength, const T* __restrict__ sv_real,
    const T* __restrict__ sv_imag, const T* __restrict__ grad_sv_real,
    const T* __restrict__ grad_sv_imag, const int N, const int M,
    T* __restrict__ grad_angles) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    constexpr T PI = T(M_PI);
    const T k = T(2.0) * PI / wavelength;

    const T az = angles[i * 2 + 0];
    const T el = angles[i * 2 + 1];
    T cos_az, sin_az, cos_el, sin_el;
    sincos(az, &sin_az, &cos_az);
    sincos(el, &sin_el, &cos_el);

    T grad_az_sum = T(0.0);
    T grad_el_sum = T(0.0);

    for (int m = 0; m < M; ++m) {
        const int sv_idx = i * M + m;
        const T sv_r = sv_real[sv_idx];
        const T sv_i = sv_imag[sv_idx];
        const T grad_sv_r = grad_sv_real[sv_idx];
        const T grad_sv_i = grad_sv_imag[sv_idx];

        const T dLdPhase = -grad_sv_r * sv_i + grad_sv_i * sv_r;

        T dPh_dAz = T(0.0);
        T dPh_dEl = T(0.0);

        if (array_type == 0) {
            const int rows = static_cast<int>(array_size[0]);
            const int cols = static_cast<int>(array_size[1]);
            const T spacing_x = element_spacing[0];
            const T spacing_y = element_spacing[1];
            const int row_idx = m / cols;
            const int col_idx = m % cols;
            const T x_pos = (row_idx - (rows - T(1.0)) * T(0.5)) * spacing_x;
            const T y_pos = (col_idx - (cols - T(1.0)) * T(0.5)) * spacing_y;

            dPh_dAz = -k * cos_el * (-x_pos * sin_az + y_pos * cos_az);
            dPh_dEl = k * sin_el * (x_pos * cos_az + y_pos * sin_az);
        } else if (array_type == 1) {
            const int num_ant = static_cast<int>(array_size[0]);
            const T spacing_d = element_spacing[0];
            const T x_pos = (m - (num_ant - T(1.0)) * T(0.5)) * spacing_d;

            dPh_dAz = k * x_pos * cos_el * sin_az;
            dPh_dEl = k * x_pos * sin_el * cos_az;
        }

        grad_az_sum += dLdPhase * dPh_dAz;
        grad_el_sum += dLdPhase * dPh_dEl;
    }

    grad_angles[i * 2 + 0] = grad_az_sum;
    grad_angles[i * 2 + 1] = grad_el_sum;
}

void compute_steering_vector_backward_cuda(
    torch::Tensor angles, torch::Tensor array_size,
    torch::Tensor element_spacing, int array_type, float wavelength,
    torch::Tensor sv_real, torch::Tensor sv_imag, torch::Tensor grad_sv_real,
    torch::Tensor grad_sv_imag, torch::Tensor grad_angles) {
    CHECK_VALID_INPUT(angles);
    CHECK_VALID_INPUT(array_size);
    CHECK_VALID_INPUT(element_spacing);
    CHECK_VALID_INPUT(sv_real);
    CHECK_VALID_INPUT(sv_imag);
    CHECK_VALID_INPUT(grad_sv_real);
    CHECK_VALID_INPUT(grad_sv_imag);
    CHECK_VALID_INPUT(grad_angles);

    const int N = angles.size(0);
    const int M = sv_real.size(1);
    TORCH_CHECK(angles.size(1) == 2, "angles must have shape Nx2");
    TORCH_CHECK(sv_real.size(0) == N && sv_imag.size(0) == N,
                "sv_real/imag shape mismatch");
    TORCH_CHECK(sv_imag.size(1) == M, "sv_imag shape mismatch");
    TORCH_CHECK(grad_sv_real.size(0) == N && grad_sv_real.size(1) == M,
                "grad_sv_real shape mismatch");
    TORCH_CHECK(grad_sv_imag.size(0) == N && grad_sv_imag.size(1) == M,
                "grad_sv_imag shape mismatch");
    TORCH_CHECK(grad_angles.size(0) == N && grad_angles.size(1) == 2,
                "grad_angles must have shape Nx2");

    const int threads = 256;
    const int num_blocks = (N + threads - 1) / threads;
    dim3 gridsize(num_blocks);
    dim3 blocksize(threads);

    grad_angles.zero_();

    if (angles.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(array_size);
        CHECK_FLOAT_TENSOR(element_spacing);
        CHECK_FLOAT_TENSOR(sv_real);
        CHECK_FLOAT_TENSOR(sv_imag);
        CHECK_FLOAT_TENSOR(grad_sv_real);
        CHECK_FLOAT_TENSOR(grad_sv_imag);
        CHECK_FLOAT_TENSOR(grad_angles);
        compute_steering_vector_backward_kernel<float><<<gridsize, blocksize>>>(
            angles.data_ptr<float>(), array_size.data_ptr<float>(),
            element_spacing.data_ptr<float>(), array_type, wavelength,
            sv_real.data_ptr<float>(), sv_imag.data_ptr<float>(),
            grad_sv_real.data_ptr<float>(), grad_sv_imag.data_ptr<float>(), N,
            M, grad_angles.data_ptr<float>());
    } else if (angles.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(array_size);
        CHECK_DOUBLE_TENSOR(element_spacing);
        CHECK_DOUBLE_TENSOR(sv_real);
        CHECK_DOUBLE_TENSOR(sv_imag);
        CHECK_DOUBLE_TENSOR(grad_sv_real);
        CHECK_DOUBLE_TENSOR(grad_sv_imag);
        CHECK_DOUBLE_TENSOR(grad_angles);
        compute_steering_vector_backward_kernel<double>
            <<<gridsize, blocksize>>>(
                angles.data_ptr<double>(), array_size.data_ptr<double>(),
                element_spacing.data_ptr<double>(), array_type, wavelength,
                sv_real.data_ptr<double>(), sv_imag.data_ptr<double>(),
                grad_sv_real.data_ptr<double>(),
                grad_sv_imag.data_ptr<double>(), N, M,
                grad_angles.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", angles.dtype());
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA error after compute_steering_vector_backward: ",
                cudaGetErrorString(err));
}

template <typename T>
__global__ void compute_scattered_paths_backward_kernel(
    const T* __restrict__ gamma_real, const T* __restrict__ gamma_imag,
    const T* __restrict__ dist_tx, const T* __restrict__ dist_rx,
    const T* __restrict__ sv_tx_real, const T* __restrict__ sv_tx_imag,
    const T* __restrict__ sv_rx_real, const T* __restrict__ sv_rx_imag,
    const T wavelength, const T* __restrict__ grad_scat_chan_real,
    const T* __restrict__ grad_scat_chan_imag, const int N, const int Nt,
    const int Nr, T* __restrict__ grad_gamma_real,
    T* __restrict__ grad_gamma_imag, T* __restrict__ grad_dist_tx,
    T* __restrict__ grad_dist_rx, T* __restrict__ grad_sv_tx_real,
    T* __restrict__ grad_sv_tx_imag, T* __restrict__ grad_sv_rx_real,
    T* __restrict__ grad_sv_rx_imag) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int tx_ant = blockIdx.y * blockDim.y + threadIdx.y;
    const int rx_ant = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N || tx_ant >= Nt || rx_ant >= Nr) {
        return;
    }

    constexpr T PI = T(M_PI);
    constexpr T FOUR_PI = T(4.0) * PI;
    constexpr T TWO_PI = T(2.0) * PI;

    const T d_tx = dist_tx[i];
    const T d_rx = dist_rx[i];
    const T dist_path = d_tx + d_rx;
    const T dist_path_safe = max(dist_path, T(ROBUST_EPSILON));
    const T inv_dist_path_safe = T(1.0) / dist_path_safe;

    const T alpha_amp = wavelength / FOUR_PI * inv_dist_path_safe;
    const T alpha_phase = -TWO_PI * dist_path / wavelength;
    T cos_phase, sin_phase;
    sincos(alpha_phase, &sin_phase, &cos_phase);
    const T alpha_real = alpha_amp * cos_phase;
    const T alpha_imag = alpha_amp * sin_phase;

    const T g_real = gamma_real[i];
    const T g_imag = gamma_imag[i];

    const T scatter_coef_real = g_real * alpha_real - g_imag * alpha_imag;
    const T scatter_coef_imag = g_real * alpha_imag + g_imag * alpha_real;

    const int chan_idx = i * Nt * Nr + tx_ant * Nr + rx_ant;
    const T grad_H_real = grad_scat_chan_real[chan_idx];
    const T grad_H_imag = grad_scat_chan_imag[chan_idx];

    const int sv_tx_idx = i * Nt + tx_ant;
    const int sv_rx_idx = i * Nr + rx_ant;
    const T sv_tx_r = sv_tx_real[sv_tx_idx];
    const T sv_tx_i = sv_tx_imag[sv_tx_idx];
    const T sv_rx_r = sv_rx_real[sv_rx_idx];
    const T sv_rx_i = sv_rx_imag[sv_rx_idx];

    const T P_real = sv_rx_r * sv_tx_r + sv_rx_i * sv_tx_i;
    const T P_imag = sv_rx_i * sv_tx_r - sv_rx_r * sv_tx_i;

    const T grad_scatter_coef_real =
        grad_H_real * P_real + grad_H_imag * P_imag;
    const T grad_scatter_coef_imag =
        -grad_H_real * P_imag + grad_H_imag * P_real;

    atomicAdd(&grad_gamma_real[i], grad_scatter_coef_real * alpha_real +
                                       grad_scatter_coef_imag * alpha_imag);
    atomicAdd(&grad_gamma_imag[i], -grad_scatter_coef_real * alpha_imag +
                                       grad_scatter_coef_imag * alpha_real);

    const T grad_alpha_real =
        grad_scatter_coef_real * g_real + grad_scatter_coef_imag * g_imag;
    const T grad_alpha_imag =
        -grad_scatter_coef_real * g_imag + grad_scatter_coef_imag * g_real;

    const T grad_alpha_amp =
        grad_alpha_real * cos_phase + grad_alpha_imag * sin_phase;
    const T grad_alpha_phase = -grad_alpha_real * alpha_amp * sin_phase +
                               grad_alpha_imag * alpha_amp * cos_phase;

    const T grad_dist_path =
        grad_alpha_amp * (-alpha_amp * inv_dist_path_safe) +
        grad_alpha_phase * (-TWO_PI / wavelength);

    atomicAdd(&grad_dist_tx[i], grad_dist_path);
    atomicAdd(&grad_dist_rx[i], grad_dist_path);

    const T grad_P_real =
        scatter_coef_real * grad_H_real + scatter_coef_imag * grad_H_imag;
    const T grad_P_imag =
        -scatter_coef_imag * grad_H_real + scatter_coef_real * grad_H_imag;

    atomicAdd(&grad_sv_tx_real[sv_tx_idx],
              grad_P_real * sv_rx_r + grad_P_imag * sv_rx_i);
    atomicAdd(&grad_sv_tx_imag[sv_tx_idx],
              grad_P_real * sv_rx_i - grad_P_imag * sv_rx_r);
    atomicAdd(&grad_sv_rx_real[sv_rx_idx],
              grad_P_real * sv_tx_r - grad_P_imag * sv_tx_i);
    atomicAdd(&grad_sv_rx_imag[sv_rx_idx],
              grad_P_real * sv_tx_i + grad_P_imag * sv_tx_r);
}

void compute_scattered_paths_backward_cuda(
    torch::Tensor gamma_real, torch::Tensor gamma_imag, torch::Tensor dist_tx,
    torch::Tensor dist_rx, torch::Tensor sv_tx_real, torch::Tensor sv_tx_imag,
    torch::Tensor sv_rx_real, torch::Tensor sv_rx_imag, float wavelength,
    torch::Tensor grad_scat_chan_real, torch::Tensor grad_scat_chan_imag,
    torch::Tensor grad_gamma_real, torch::Tensor grad_gamma_imag,
    torch::Tensor grad_dist_tx, torch::Tensor grad_dist_rx,
    torch::Tensor grad_sv_tx_real, torch::Tensor grad_sv_tx_imag,
    torch::Tensor grad_sv_rx_real, torch::Tensor grad_sv_rx_imag) {
    CHECK_VALID_INPUT(gamma_real);
    CHECK_VALID_INPUT(gamma_imag);
    CHECK_VALID_INPUT(dist_tx);
    CHECK_VALID_INPUT(dist_rx);
    CHECK_VALID_INPUT(sv_tx_real);
    CHECK_VALID_INPUT(sv_tx_imag);
    CHECK_VALID_INPUT(sv_rx_real);
    CHECK_VALID_INPUT(sv_rx_imag);
    CHECK_VALID_INPUT(grad_scat_chan_real);
    CHECK_VALID_INPUT(grad_scat_chan_imag);
    CHECK_VALID_INPUT(grad_gamma_real);
    CHECK_VALID_INPUT(grad_gamma_imag);
    CHECK_VALID_INPUT(grad_dist_tx);
    CHECK_VALID_INPUT(grad_dist_rx);
    CHECK_VALID_INPUT(grad_sv_tx_real);
    CHECK_VALID_INPUT(grad_sv_tx_imag);
    CHECK_VALID_INPUT(grad_sv_rx_real);
    CHECK_VALID_INPUT(grad_sv_rx_imag);

    const int N = gamma_real.size(0);
    const int Nt = sv_tx_real.size(1);
    const int Nr = sv_rx_real.size(1);

    TORCH_CHECK(gamma_imag.size(0) == N, "gamma_imag size mismatch");
    TORCH_CHECK(dist_tx.size(0) == N, "dist_tx size mismatch");
    TORCH_CHECK(dist_rx.size(0) == N, "dist_rx size mismatch");
    TORCH_CHECK(sv_tx_real.size(0) == N, "sv_tx_real size mismatch");
    TORCH_CHECK(sv_tx_imag.size(0) == N && sv_tx_imag.size(1) == Nt,
                "sv_tx_imag shape mismatch");
    TORCH_CHECK(sv_rx_real.size(0) == N, "sv_rx_real size mismatch");
    TORCH_CHECK(sv_rx_imag.size(0) == N && sv_rx_imag.size(1) == Nr,
                "sv_rx_imag shape mismatch");
    TORCH_CHECK(grad_scat_chan_real.size(0) == N &&
                    grad_scat_chan_real.size(1) == Nt &&
                    grad_scat_chan_real.size(2) == Nr,
                "grad_scat_chan_real shape mismatch");
    TORCH_CHECK(grad_scat_chan_imag.size(0) == N &&
                    grad_scat_chan_imag.size(1) == Nt &&
                    grad_scat_chan_imag.size(2) == Nr,
                "grad_scat_chan_imag shape mismatch");
    TORCH_CHECK(grad_gamma_real.size(0) == N, "grad_gamma_real size mismatch");
    TORCH_CHECK(grad_gamma_imag.size(0) == N, "grad_gamma_imag size mismatch");
    TORCH_CHECK(grad_dist_tx.size(0) == N, "grad_dist_tx size mismatch");
    TORCH_CHECK(grad_dist_rx.size(0) == N, "grad_dist_rx size mismatch");
    TORCH_CHECK(grad_sv_tx_real.size(0) == N && grad_sv_tx_real.size(1) == Nt,
                "grad_sv_tx_real shape mismatch");
    TORCH_CHECK(grad_sv_tx_imag.size(0) == N && grad_sv_tx_imag.size(1) == Nt,
                "grad_sv_tx_imag shape mismatch");
    TORCH_CHECK(grad_sv_rx_real.size(0) == N && grad_sv_rx_real.size(1) == Nr,
                "grad_sv_rx_real shape mismatch");
    TORCH_CHECK(grad_sv_rx_imag.size(0) == N && grad_sv_rx_imag.size(1) == Nr,
                "grad_sv_rx_imag shape mismatch");

    grad_gamma_real.zero_();
    grad_gamma_imag.zero_();
    grad_dist_tx.zero_();
    grad_dist_rx.zero_();
    grad_sv_tx_real.zero_();
    grad_sv_tx_imag.zero_();
    grad_sv_rx_real.zero_();
    grad_sv_rx_imag.zero_();

    dim3 blocksize(SCAT_BW_BLOCK_DIM_X, SCAT_BW_BLOCK_DIM_Y,
                   SCAT_BW_BLOCK_DIM_Z);
    dim3 gridsize((N + blocksize.x - 1) / blocksize.x,
                  (Nt + blocksize.y - 1) / blocksize.y,
                  (Nr + blocksize.z - 1) / blocksize.z);

    if (gamma_real.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(gamma_imag);
        CHECK_FLOAT_TENSOR(dist_tx);
        CHECK_FLOAT_TENSOR(dist_rx);
        CHECK_FLOAT_TENSOR(sv_tx_real);
        CHECK_FLOAT_TENSOR(sv_tx_imag);
        CHECK_FLOAT_TENSOR(sv_rx_real);
        CHECK_FLOAT_TENSOR(sv_rx_imag);
        CHECK_FLOAT_TENSOR(grad_scat_chan_real);
        CHECK_FLOAT_TENSOR(grad_scat_chan_imag);
        CHECK_FLOAT_TENSOR(grad_gamma_real);
        CHECK_FLOAT_TENSOR(grad_gamma_imag);
        CHECK_FLOAT_TENSOR(grad_dist_tx);
        CHECK_FLOAT_TENSOR(grad_dist_rx);
        CHECK_FLOAT_TENSOR(grad_sv_tx_real);
        CHECK_FLOAT_TENSOR(grad_sv_tx_imag);
        CHECK_FLOAT_TENSOR(grad_sv_rx_real);
        CHECK_FLOAT_TENSOR(grad_sv_rx_imag);
        compute_scattered_paths_backward_kernel<float><<<gridsize, blocksize>>>(
            gamma_real.data_ptr<float>(), gamma_imag.data_ptr<float>(),
            dist_tx.data_ptr<float>(), dist_rx.data_ptr<float>(),
            sv_tx_real.data_ptr<float>(), sv_tx_imag.data_ptr<float>(),
            sv_rx_real.data_ptr<float>(), sv_rx_imag.data_ptr<float>(),
            wavelength, grad_scat_chan_real.data_ptr<float>(),
            grad_scat_chan_imag.data_ptr<float>(), N, Nt, Nr,
            grad_gamma_real.data_ptr<float>(),
            grad_gamma_imag.data_ptr<float>(), grad_dist_tx.data_ptr<float>(),
            grad_dist_rx.data_ptr<float>(), grad_sv_tx_real.data_ptr<float>(),
            grad_sv_tx_imag.data_ptr<float>(),
            grad_sv_rx_real.data_ptr<float>(),
            grad_sv_rx_imag.data_ptr<float>());
    } else if (gamma_real.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(gamma_imag);
        CHECK_DOUBLE_TENSOR(dist_tx);
        CHECK_DOUBLE_TENSOR(dist_rx);
        CHECK_DOUBLE_TENSOR(sv_tx_real);
        CHECK_DOUBLE_TENSOR(sv_tx_imag);
        CHECK_DOUBLE_TENSOR(sv_rx_real);
        CHECK_DOUBLE_TENSOR(sv_rx_imag);
        CHECK_DOUBLE_TENSOR(grad_scat_chan_real);
        CHECK_DOUBLE_TENSOR(grad_scat_chan_imag);
        CHECK_DOUBLE_TENSOR(grad_gamma_real);
        CHECK_DOUBLE_TENSOR(grad_gamma_imag);
        CHECK_DOUBLE_TENSOR(grad_dist_tx);
        CHECK_DOUBLE_TENSOR(grad_dist_rx);
        CHECK_DOUBLE_TENSOR(grad_sv_tx_real);
        CHECK_DOUBLE_TENSOR(grad_sv_tx_imag);
        CHECK_DOUBLE_TENSOR(grad_sv_rx_real);
        CHECK_DOUBLE_TENSOR(grad_sv_rx_imag);
        compute_scattered_paths_backward_kernel<double>
            <<<gridsize, blocksize>>>(
                gamma_real.data_ptr<double>(), gamma_imag.data_ptr<double>(),
                dist_tx.data_ptr<double>(), dist_rx.data_ptr<double>(),
                sv_tx_real.data_ptr<double>(), sv_tx_imag.data_ptr<double>(),
                sv_rx_real.data_ptr<double>(), sv_rx_imag.data_ptr<double>(),
                wavelength, grad_scat_chan_real.data_ptr<double>(),
                grad_scat_chan_imag.data_ptr<double>(), N, Nt, Nr,
                grad_gamma_real.data_ptr<double>(),
                grad_gamma_imag.data_ptr<double>(),
                grad_dist_tx.data_ptr<double>(),
                grad_dist_rx.data_ptr<double>(),
                grad_sv_tx_real.data_ptr<double>(),
                grad_sv_tx_imag.data_ptr<double>(),
                grad_sv_rx_real.data_ptr<double>(),
                grad_sv_rx_imag.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", gamma_real.dtype());
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA error after compute_scattered_paths_backward: ",
                cudaGetErrorString(err));
}

template <typename T>
__global__ void weighted_superposition_backward_kernel(
    const T* __restrict__ scat_path_real, const T* __restrict__ scat_path_imag,
    const T* __restrict__ opacity, const T* __restrict__ influence,
    const T* __restrict__ grad_chan_real, const T* __restrict__ grad_chan_imag,
    const int N, const int Nt, const int Nr,
    T* __restrict__ grad_scat_path_real, T* __restrict__ grad_scat_path_imag,
    T* __restrict__ grad_opacity, T* __restrict__ grad_influence) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int tx_ant = blockIdx.y * blockDim.y + threadIdx.y;
    const int rx_ant = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N || tx_ant >= Nt || rx_ant >= Nr) {
        return;
    }

    const int chan_idx = tx_ant * Nr + rx_ant;
    const int scat_idx = i * Nt * Nr + tx_ant * Nr + rx_ant;
    const int infl_idx = i * Nt * Nr + tx_ant * Nr + rx_ant;

    const T grad_real = grad_chan_real[chan_idx];
    const T grad_imag = grad_chan_imag[chan_idx];

    const T opac = opacity[i];
    const T infl = influence[infl_idx];
    const T weight = opac * infl;

    const T scat_real = scat_path_real[scat_idx];
    const T scat_imag = scat_path_imag[scat_idx];

    grad_scat_path_real[scat_idx] = weight * grad_real;
    grad_scat_path_imag[scat_idx] = weight * grad_imag;

    const T grad_weight = grad_real * scat_real + grad_imag * scat_imag;

    atomicAdd(&grad_opacity[i], grad_weight * infl);

    grad_influence[infl_idx] = grad_weight * opac;
}

void weighted_superposition_backward_cuda(
    torch::Tensor scat_path_real, torch::Tensor scat_path_imag,
    torch::Tensor opacity, torch::Tensor influence,
    torch::Tensor grad_chan_real, torch::Tensor grad_chan_imag,
    torch::Tensor grad_scat_path_real, torch::Tensor grad_scat_path_imag,
    torch::Tensor grad_opacity, torch::Tensor grad_influence) {
    CHECK_VALID_INPUT(scat_path_real);
    CHECK_VALID_INPUT(scat_path_imag);
    CHECK_VALID_INPUT(opacity);
    CHECK_VALID_INPUT(influence);
    CHECK_VALID_INPUT(grad_chan_real);
    CHECK_VALID_INPUT(grad_chan_imag);
    CHECK_VALID_INPUT(grad_scat_path_real);
    CHECK_VALID_INPUT(grad_scat_path_imag);
    CHECK_VALID_INPUT(grad_opacity);
    CHECK_VALID_INPUT(grad_influence);

    const int N = scat_path_real.size(0);
    const int Nt = scat_path_real.size(1);
    const int Nr = scat_path_real.size(2);

    TORCH_CHECK(scat_path_imag.size(0) == N && scat_path_imag.size(1) == Nt &&
                    scat_path_imag.size(2) == Nr,
                "scat_path_imag shape mismatch");
    TORCH_CHECK(opacity.size(0) == N, "opacity shape mismatch");
    TORCH_CHECK(influence.size(0) == N && influence.size(1) == Nt &&
                    influence.size(2) == Nr,
                "influence shape mismatch");
    TORCH_CHECK(grad_chan_real.size(0) == Nt && grad_chan_real.size(1) == Nr,
                "grad_chan_real shape mismatch");
    TORCH_CHECK(grad_chan_imag.size(0) == Nt && grad_chan_imag.size(1) == Nr,
                "grad_chan_imag shape mismatch");
    TORCH_CHECK(grad_scat_path_real.size(0) == N &&
                    grad_scat_path_real.size(1) == Nt &&
                    grad_scat_path_real.size(2) == Nr,
                "grad_scat_path_real shape mismatch");
    TORCH_CHECK(grad_scat_path_imag.size(0) == N &&
                    grad_scat_path_imag.size(1) == Nt &&
                    grad_scat_path_imag.size(2) == Nr,
                "grad_scat_path_imag shape mismatch");
    TORCH_CHECK(grad_opacity.size(0) == N, "grad_opacity shape mismatch");
    TORCH_CHECK(grad_influence.size(0) == N && grad_influence.size(1) == Nt &&
                    grad_influence.size(2) == Nr,
                "grad_influence shape mismatch");

    dim3 blocksize(WS_BW_BLOCK_DIM_X, WS_BW_BLOCK_DIM_Y, WS_BW_BLOCK_DIM_Z);
    dim3 gridsize((N + blocksize.x - 1) / blocksize.x,
                  (Nt + blocksize.y - 1) / blocksize.y,
                  (Nr + blocksize.z - 1) / blocksize.z);

    grad_opacity.zero_();

    auto opacity_cont = opacity.contiguous().view({N});
    auto grad_opacity_cont = grad_opacity.contiguous().view({N});

    if (scat_path_real.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(scat_path_imag);
        CHECK_FLOAT_TENSOR(opacity);
        CHECK_FLOAT_TENSOR(influence);
        CHECK_FLOAT_TENSOR(grad_chan_real);
        CHECK_FLOAT_TENSOR(grad_chan_imag);
        CHECK_FLOAT_TENSOR(grad_scat_path_real);
        CHECK_FLOAT_TENSOR(grad_scat_path_imag);
        CHECK_FLOAT_TENSOR(grad_opacity);
        CHECK_FLOAT_TENSOR(grad_influence);
        weighted_superposition_backward_kernel<float><<<gridsize, blocksize>>>(
            scat_path_real.data_ptr<float>(), scat_path_imag.data_ptr<float>(),
            opacity_cont.data_ptr<float>(), influence.data_ptr<float>(),
            grad_chan_real.data_ptr<float>(), grad_chan_imag.data_ptr<float>(),
            N, Nt, Nr, grad_scat_path_real.data_ptr<float>(),
            grad_scat_path_imag.data_ptr<float>(),
            grad_opacity_cont.data_ptr<float>(),
            grad_influence.data_ptr<float>());
    } else if (scat_path_real.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(scat_path_imag);
        CHECK_DOUBLE_TENSOR(opacity);
        CHECK_DOUBLE_TENSOR(influence);
        CHECK_DOUBLE_TENSOR(grad_chan_real);
        CHECK_DOUBLE_TENSOR(grad_chan_imag);
        CHECK_DOUBLE_TENSOR(grad_scat_path_real);
        CHECK_DOUBLE_TENSOR(grad_scat_path_imag);
        CHECK_DOUBLE_TENSOR(grad_opacity);
        CHECK_DOUBLE_TENSOR(grad_influence);
        weighted_superposition_backward_kernel<double><<<gridsize, blocksize>>>(
            scat_path_real.data_ptr<double>(),
            scat_path_imag.data_ptr<double>(), opacity_cont.data_ptr<double>(),
            influence.data_ptr<double>(), grad_chan_real.data_ptr<double>(),
            grad_chan_imag.data_ptr<double>(), N, Nt, Nr,
            grad_scat_path_real.data_ptr<double>(),
            grad_scat_path_imag.data_ptr<double>(),
            grad_opacity_cont.data_ptr<double>(),
            grad_influence.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", scat_path_real.dtype());
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA error after weighted_superposition_backward: ",
                cudaGetErrorString(err));
}
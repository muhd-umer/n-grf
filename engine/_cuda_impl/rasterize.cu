// engine/_cuda_impl/rasterize.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

#include "checks.cuh"
#include "matrix.cuh"

constexpr int RAST_BLOCK_DIM_X = 4;
constexpr int RAST_BLOCK_DIM_Y = 8;
constexpr int RAST_BLOCK_DIM_Z = 8;

constexpr int SV_BLOCK_DIM_X = 16;
constexpr int SV_BLOCK_DIM_Y = 16;

constexpr int WS_REDUCE_BLOCK_SIZE = 256;

template <typename T>
__global__ void compute_spatial_influence_kernel(const T* __restrict__ uv,
                                                 const T* __restrict__ cov2d,
                                                 const int num_tx,
                                                 const int num_rx, const int N,
                                                 T* __restrict__ influences) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int tx = blockIdx.y * blockDim.y + threadIdx.y;
    const int rx = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N || tx >= num_tx || rx >= num_rx) {
        return;
    }

    const T* cov2d_ptr = cov2d + i * 4;
    const T a = cov2d_ptr[0];
    const T b = cov2d_ptr[1];
    const T c = cov2d_ptr[2];
    const T d_cov = cov2d_ptr[3];

    const T det = a * d_cov - b * c;
    const T inv_det = T(1.0) / max(det, T(ROBUST_EPSILON));

    const T inv_a = d_cov * inv_det;
    const T inv_b = -b * inv_det;
    const T inv_c = -c * inv_det;
    const T inv_d = a * inv_det;

    const T u_i = uv[i * 2 + 0];
    const T v_i = uv[i * 2 + 1];

    const T antenna_u = T(tx) + T(0.5);
    const T antenna_v = T(rx) + T(0.5);

    const T du = u_i - antenna_u;
    const T dv = v_i - antenna_v;

    const T md =
        du * (inv_a * du + inv_b * dv) + dv * (inv_c * du + inv_d * dv);
    const T md_clamped = min(md, T(30.0));

    influences[i * num_tx * num_rx + tx * num_rx + rx] =
        exp(-T(0.5) * md_clamped);
}

void compute_spatial_influence_cuda(torch::Tensor uv, torch::Tensor cov2d,
                                    int num_tx, int num_rx,
                                    torch::Tensor influences) {
    CHECK_VALID_INPUT(uv);
    CHECK_VALID_INPUT(cov2d);
    CHECK_VALID_INPUT(influences);

    const int N = uv.size(0);
    TORCH_CHECK(uv.size(1) == 2, "uv must have shape Nx2");
    TORCH_CHECK(cov2d.size(0) == N && cov2d.size(1) == 2 && cov2d.size(2) == 2,
                "cov2d must have shape Nx2x2");
    TORCH_CHECK(influences.size(0) == N && influences.size(1) == num_tx &&
                    influences.size(2) == num_rx,
                "influences must have shape NxTxR");

    dim3 blocksize(RAST_BLOCK_DIM_X, RAST_BLOCK_DIM_Y, RAST_BLOCK_DIM_Z);
    dim3 gridsize((N + blocksize.x - 1) / blocksize.x,
                  (num_tx + blocksize.y - 1) / blocksize.y,
                  (num_rx + blocksize.z - 1) / blocksize.z);

    if (uv.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(cov2d);
        CHECK_FLOAT_TENSOR(influences);
        compute_spatial_influence_kernel<float><<<gridsize, blocksize>>>(
            uv.data_ptr<float>(), cov2d.data_ptr<float>(), num_tx, num_rx, N,
            influences.data_ptr<float>());
    } else if (uv.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(cov2d);
        CHECK_DOUBLE_TENSOR(influences);
        compute_spatial_influence_kernel<double><<<gridsize, blocksize>>>(
            uv.data_ptr<double>(), cov2d.data_ptr<double>(), num_tx, num_rx, N,
            influences.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", uv.dtype());
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA error after compute_spatial_influence: ",
                cudaGetErrorString(err));
}

template <typename T>
__global__ void compute_path_geometry_kernel(
    const T* __restrict__ points, const T* __restrict__ tx_pos,
    const T* __restrict__ rx_pos, const int N, T* __restrict__ dist_tx,
    T* __restrict__ dist_rx, T* __restrict__ aod, T* __restrict__ aoa) {
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

    const T d_tx_sq = vec_tx_gauss_x * vec_tx_gauss_x +
                      vec_tx_gauss_y * vec_tx_gauss_y +
                      vec_tx_gauss_z * vec_tx_gauss_z;
    const T d_tx = sqrt(d_tx_sq);
    const T d_tx_safe = max(d_tx, T(ROBUST_EPSILON));
    dist_tx[i] = d_tx_safe;

    const T d_rx_sq = vec_gauss_rx_x * vec_gauss_rx_x +
                      vec_gauss_rx_y * vec_gauss_rx_y +
                      vec_gauss_rx_z * vec_gauss_rx_z;
    const T d_rx = sqrt(d_rx_sq);
    const T d_rx_safe = max(d_rx, T(ROBUST_EPSILON));
    dist_rx[i] = d_rx_safe;

    aod[i * 2 + 0] = atan2(vec_tx_gauss_y, vec_tx_gauss_x);
    T aod_el_arg = vec_tx_gauss_z / d_tx_safe;
    aod_el_arg = min(max(aod_el_arg, T(-1.0)), T(1.0));
    aod[i * 2 + 1] = asin(aod_el_arg);

    aoa[i * 2 + 0] = atan2(vec_gauss_rx_y, vec_gauss_rx_x);
    T aoa_el_arg = vec_gauss_rx_z / d_rx_safe;
    aoa_el_arg = min(max(aoa_el_arg, T(-1.0)), T(1.0));
    aoa[i * 2 + 1] = asin(aoa_el_arg);
}

void compute_path_geometry_cuda(torch::Tensor points, torch::Tensor tx_pos,
                                torch::Tensor rx_pos, torch::Tensor dist_tx,
                                torch::Tensor dist_rx, torch::Tensor aod,
                                torch::Tensor aoa) {
    CHECK_VALID_INPUT(points);
    CHECK_VALID_INPUT(tx_pos);
    CHECK_VALID_INPUT(rx_pos);
    CHECK_VALID_INPUT(dist_tx);
    CHECK_VALID_INPUT(dist_rx);
    CHECK_VALID_INPUT(aod);
    CHECK_VALID_INPUT(aoa);

    const int N = points.size(0);
    TORCH_CHECK(points.size(1) == 3, "points must have shape Nx3");
    TORCH_CHECK(tx_pos.size(0) == 3, "tx_pos must have shape 3");
    TORCH_CHECK(rx_pos.size(0) == 3, "rx_pos must have shape 3");
    TORCH_CHECK(dist_tx.size(0) == N, "dist_tx must have shape N");
    TORCH_CHECK(dist_rx.size(0) == N, "dist_rx must have shape N");
    TORCH_CHECK(aod.size(0) == N && aod.size(1) == 2,
                "aod must have shape Nx2");
    TORCH_CHECK(aoa.size(0) == N && aoa.size(1) == 2,
                "aoa must have shape Nx2");

    const int threads = 256;
    const int num_blocks = (N + threads - 1) / threads;
    dim3 gridsize(num_blocks);
    dim3 blocksize(threads);

    if (points.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(tx_pos);
        CHECK_FLOAT_TENSOR(rx_pos);
        CHECK_FLOAT_TENSOR(dist_tx);
        CHECK_FLOAT_TENSOR(dist_rx);
        CHECK_FLOAT_TENSOR(aod);
        CHECK_FLOAT_TENSOR(aoa);
        compute_path_geometry_kernel<float><<<gridsize, blocksize>>>(
            points.data_ptr<float>(), tx_pos.data_ptr<float>(),
            rx_pos.data_ptr<float>(), N, dist_tx.data_ptr<float>(),
            dist_rx.data_ptr<float>(), aod.data_ptr<float>(),
            aoa.data_ptr<float>());
    } else if (points.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(tx_pos);
        CHECK_DOUBLE_TENSOR(rx_pos);
        CHECK_DOUBLE_TENSOR(dist_tx);
        CHECK_DOUBLE_TENSOR(dist_rx);
        CHECK_DOUBLE_TENSOR(aod);
        CHECK_DOUBLE_TENSOR(aoa);
        compute_path_geometry_kernel<double><<<gridsize, blocksize>>>(
            points.data_ptr<double>(), tx_pos.data_ptr<double>(),
            rx_pos.data_ptr<double>(), N, dist_tx.data_ptr<double>(),
            dist_rx.data_ptr<double>(), aod.data_ptr<double>(),
            aoa.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", points.dtype());
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after compute_path_geometry: ",
                cudaGetErrorString(err));
}

template <typename T>
__global__ void compute_steering_vector_kernel(
    const T* __restrict__ angles, const T* __restrict__ array_size,
    const T* __restrict__ element_spacing, const int array_type,
    const T wavelength, const int N, const int M, T* __restrict__ sv_real,
    T* __restrict__ sv_imag) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= N || m >= M) {
        return;
    }

    constexpr T PI = T(M_PI);
    const T k = T(2.0) * PI / wavelength;

    const T az = angles[i * 2 + 0];
    const T el = angles[i * 2 + 1];
    const T cos_el = cos(el);
    const T cos_az = cos(az);
    const T sin_az = sin(az);

    T phase = T(0.0);

    if (array_type == 0) {
        const int rows = static_cast<int>(array_size[0]);
        const int cols = static_cast<int>(array_size[1]);
        const T spacing_x = element_spacing[0];
        const T spacing_y = element_spacing[1];

        const int row_idx = m / cols;
        const int col_idx = m % cols;

        const T x_pos = (row_idx - (rows - T(1.0)) * T(0.5)) * spacing_x;
        const T y_pos = (col_idx - (cols - T(1.0)) * T(0.5)) * spacing_y;

        phase = -k * cos_el * (x_pos * cos_az + y_pos * sin_az);
    } else if (array_type == 1) {
        const int num_ant = static_cast<int>(array_size[0]);
        const T spacing_d = element_spacing[0];
        const T x_pos = (m - (num_ant - T(1.0)) * T(0.5)) * spacing_d;
        phase = -k * x_pos * cos_el * cos_az;
    }

    sv_real[i * M + m] = cos(phase);
    sv_imag[i * M + m] = sin(phase);
}

void compute_steering_vector_cuda(torch::Tensor angles,
                                  torch::Tensor array_size,
                                  torch::Tensor element_spacing, int array_type,
                                  float wavelength, torch::Tensor sv_real,
                                  torch::Tensor sv_imag) {
    CHECK_VALID_INPUT(angles);
    CHECK_VALID_INPUT(array_size);
    CHECK_VALID_INPUT(element_spacing);
    CHECK_VALID_INPUT(sv_real);
    CHECK_VALID_INPUT(sv_imag);

    const int N = angles.size(0);
    const int M = sv_real.size(1);
    TORCH_CHECK(angles.size(1) == 2, "angles must have shape Nx2");
    TORCH_CHECK(sv_real.size(0) == N && sv_imag.size(0) == N,
                "sv_real/imag must have shape NxM");
    TORCH_CHECK(sv_imag.size(1) == M, "sv_imag must have shape NxM");

    dim3 blocksize(SV_BLOCK_DIM_X, SV_BLOCK_DIM_Y, 1);
    dim3 gridsize((N + blocksize.x - 1) / blocksize.x,
                  (M + blocksize.y - 1) / blocksize.y, 1);

    if (angles.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(array_size);
        CHECK_FLOAT_TENSOR(element_spacing);
        CHECK_FLOAT_TENSOR(sv_real);
        CHECK_FLOAT_TENSOR(sv_imag);
        compute_steering_vector_kernel<float><<<gridsize, blocksize>>>(
            angles.data_ptr<float>(), array_size.data_ptr<float>(),
            element_spacing.data_ptr<float>(), array_type, wavelength, N, M,
            sv_real.data_ptr<float>(), sv_imag.data_ptr<float>());
    } else if (angles.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(array_size);
        CHECK_DOUBLE_TENSOR(element_spacing);
        CHECK_DOUBLE_TENSOR(sv_real);
        CHECK_DOUBLE_TENSOR(sv_imag);
        compute_steering_vector_kernel<double><<<gridsize, blocksize>>>(
            angles.data_ptr<double>(), array_size.data_ptr<double>(),
            element_spacing.data_ptr<double>(), array_type, wavelength, N, M,
            sv_real.data_ptr<double>(), sv_imag.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", angles.dtype());
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "CUDA error after compute_steering_vector: ", cudaGetErrorString(err));
}

template <typename T>
__global__ void compute_scattered_paths_kernel(
    const T* __restrict__ gamma_real, const T* __restrict__ gamma_imag,
    const T* __restrict__ dist_tx, const T* __restrict__ dist_rx,
    const T* __restrict__ sv_tx_real, const T* __restrict__ sv_tx_imag,
    const T* __restrict__ sv_rx_real, const T* __restrict__ sv_rx_imag,
    const T wavelength, const int N, const int Nt, const int Nr,
    T* __restrict__ scat_chan_real, T* __restrict__ scat_chan_imag) {
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

    const T sv_tx_r = sv_tx_real[i * Nt + tx_ant];
    const T sv_tx_i = sv_tx_imag[i * Nt + tx_ant];
    const T sv_rx_r = sv_rx_real[i * Nr + rx_ant];
    const T sv_rx_i = sv_rx_imag[i * Nr + rx_ant];

    const T P_real = sv_rx_r * sv_tx_r + sv_rx_i * sv_tx_i;
    const T P_imag = sv_rx_i * sv_tx_r - sv_rx_r * sv_tx_i;

    const T H_real = scatter_coef_real * P_real - scatter_coef_imag * P_imag;
    const T H_imag = scatter_coef_real * P_imag + scatter_coef_imag * P_real;

    const int out_idx = i * Nt * Nr + tx_ant * Nr + rx_ant;
    scat_chan_real[out_idx] = H_real;
    scat_chan_imag[out_idx] = H_imag;
}

void compute_scattered_paths_cuda(
    torch::Tensor gamma_real, torch::Tensor gamma_imag, torch::Tensor dist_tx,
    torch::Tensor dist_rx, torch::Tensor sv_tx_real, torch::Tensor sv_tx_imag,
    torch::Tensor sv_rx_real, torch::Tensor sv_rx_imag, float wavelength,
    torch::Tensor scat_chan_real, torch::Tensor scat_chan_imag) {
    CHECK_VALID_INPUT(gamma_real);
    CHECK_VALID_INPUT(gamma_imag);
    CHECK_VALID_INPUT(dist_tx);
    CHECK_VALID_INPUT(dist_rx);
    CHECK_VALID_INPUT(sv_tx_real);
    CHECK_VALID_INPUT(sv_tx_imag);
    CHECK_VALID_INPUT(sv_rx_real);
    CHECK_VALID_INPUT(sv_rx_imag);
    CHECK_VALID_INPUT(scat_chan_real);
    CHECK_VALID_INPUT(scat_chan_imag);

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
    TORCH_CHECK(scat_chan_real.size(0) == N && scat_chan_real.size(1) == Nt &&
                    scat_chan_real.size(2) == Nr,
                "scat_chan_real shape mismatch");
    TORCH_CHECK(scat_chan_imag.size(0) == N && scat_chan_imag.size(1) == Nt &&
                    scat_chan_imag.size(2) == Nr,
                "scat_chan_imag shape mismatch");

    dim3 blocksize(RAST_BLOCK_DIM_X, RAST_BLOCK_DIM_Y, RAST_BLOCK_DIM_Z);
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
        CHECK_FLOAT_TENSOR(scat_chan_real);
        CHECK_FLOAT_TENSOR(scat_chan_imag);
        compute_scattered_paths_kernel<float><<<gridsize, blocksize>>>(
            gamma_real.data_ptr<float>(), gamma_imag.data_ptr<float>(),
            dist_tx.data_ptr<float>(), dist_rx.data_ptr<float>(),
            sv_tx_real.data_ptr<float>(), sv_tx_imag.data_ptr<float>(),
            sv_rx_real.data_ptr<float>(), sv_rx_imag.data_ptr<float>(),
            wavelength, N, Nt, Nr, scat_chan_real.data_ptr<float>(),
            scat_chan_imag.data_ptr<float>());
    } else if (gamma_real.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(gamma_imag);
        CHECK_DOUBLE_TENSOR(dist_tx);
        CHECK_DOUBLE_TENSOR(dist_rx);
        CHECK_DOUBLE_TENSOR(sv_tx_real);
        CHECK_DOUBLE_TENSOR(sv_tx_imag);
        CHECK_DOUBLE_TENSOR(sv_rx_real);
        CHECK_DOUBLE_TENSOR(sv_rx_imag);
        CHECK_DOUBLE_TENSOR(scat_chan_real);
        CHECK_DOUBLE_TENSOR(scat_chan_imag);
        compute_scattered_paths_kernel<double><<<gridsize, blocksize>>>(
            gamma_real.data_ptr<double>(), gamma_imag.data_ptr<double>(),
            dist_tx.data_ptr<double>(), dist_rx.data_ptr<double>(),
            sv_tx_real.data_ptr<double>(), sv_tx_imag.data_ptr<double>(),
            sv_rx_real.data_ptr<double>(), sv_rx_imag.data_ptr<double>(),
            wavelength, N, Nt, Nr, scat_chan_real.data_ptr<double>(),
            scat_chan_imag.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", gamma_real.dtype());
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "CUDA error after compute_scattered_paths: ", cudaGetErrorString(err));
}

template <typename T>
__global__ void weighted_superposition_kernel(
    const T* __restrict__ direct_path_real,
    const T* __restrict__ direct_path_imag,
    const T* __restrict__ scat_path_real, const T* __restrict__ scat_path_imag,
    const T* __restrict__ opacity, const T* __restrict__ influence, const int N,
    const int Nt, const int Nr, T* __restrict__ chan_real,
    T* __restrict__ chan_imag) {
    __shared__ T sdata[WS_REDUCE_BLOCK_SIZE * 2];
    T* s_sum_real = sdata;
    T* s_sum_imag = s_sum_real + WS_REDUCE_BLOCK_SIZE;

    const int tx_ant = blockIdx.x;
    const int rx_ant = blockIdx.y;
    const int tid = threadIdx.x;

    s_sum_real[tid] = T(0.0);
    s_sum_imag[tid] = T(0.0);

    const int chan_offset = tx_ant * Nr + rx_ant;
    const int infl_offset = tx_ant * Nr + rx_ant;
    const int scat_offset = tx_ant * Nr + rx_ant;

    for (int i = tid; i < N; i += WS_REDUCE_BLOCK_SIZE) {
        const T opac = opacity[i];
        const T infl = influence[i * Nt * Nr + infl_offset];
        const T weight = opac * infl;

        const T scat_real = scat_path_real[i * Nt * Nr + scat_offset];
        const T scat_imag = scat_path_imag[i * Nt * Nr + scat_offset];

        s_sum_real[tid] += weight * scat_real;
        s_sum_imag[tid] += weight * scat_imag;
    }
    __syncthreads();

    for (int s = WS_REDUCE_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum_real[tid] += s_sum_real[tid + s];
            s_sum_imag[tid] += s_sum_imag[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        chan_real[chan_offset] = direct_path_real[chan_offset] + s_sum_real[0];
        chan_imag[chan_offset] = direct_path_imag[chan_offset] + s_sum_imag[0];
    }
}

void weighted_superposition_cuda(torch::Tensor direct_path_real,
                                 torch::Tensor direct_path_imag,
                                 torch::Tensor scat_path_real,
                                 torch::Tensor scat_path_imag,
                                 torch::Tensor opacity, torch::Tensor influence,
                                 torch::Tensor chan_real,
                                 torch::Tensor chan_imag) {
    CHECK_VALID_INPUT(direct_path_real);
    CHECK_VALID_INPUT(direct_path_imag);
    CHECK_VALID_INPUT(scat_path_real);
    CHECK_VALID_INPUT(scat_path_imag);
    CHECK_VALID_INPUT(opacity);
    CHECK_VALID_INPUT(influence);
    CHECK_VALID_INPUT(chan_real);
    CHECK_VALID_INPUT(chan_imag);

    const int N = scat_path_real.size(0);
    const int Nt = direct_path_real.size(0);
    const int Nr = direct_path_real.size(1);

    TORCH_CHECK(
        direct_path_imag.size(0) == Nt && direct_path_imag.size(1) == Nr,
        "direct_path_imag shape mismatch");
    TORCH_CHECK(scat_path_real.size(1) == Nt && scat_path_real.size(2) == Nr,
                "scat_path_real shape mismatch");
    TORCH_CHECK(scat_path_imag.size(0) == N && scat_path_imag.size(1) == Nt &&
                    scat_path_imag.size(2) == Nr,
                "scat_path_imag shape mismatch");
    TORCH_CHECK(opacity.size(0) == N, "opacity shape mismatch");
    TORCH_CHECK(influence.size(0) == N && influence.size(1) == Nt &&
                    influence.size(2) == Nr,
                "influence shape mismatch");
    TORCH_CHECK(chan_real.size(0) == Nt && chan_real.size(1) == Nr,
                "chan_real shape mismatch");
    TORCH_CHECK(chan_imag.size(0) == Nt && chan_imag.size(1) == Nr,
                "chan_imag shape mismatch");

    dim3 gridsize(Nt, Nr, 1);
    dim3 blocksize(WS_REDUCE_BLOCK_SIZE, 1, 1);
    size_t smem_size =
        WS_REDUCE_BLOCK_SIZE * 2 *
        ((direct_path_real.dtype() == torch::kFloat64) ? sizeof(double)
                                                       : sizeof(float));

    auto opacity_cont = opacity.contiguous().view({N});

    if (direct_path_real.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(direct_path_imag);
        CHECK_FLOAT_TENSOR(scat_path_real);
        CHECK_FLOAT_TENSOR(scat_path_imag);
        CHECK_FLOAT_TENSOR(opacity);
        CHECK_FLOAT_TENSOR(influence);
        CHECK_FLOAT_TENSOR(chan_real);
        CHECK_FLOAT_TENSOR(chan_imag);
        weighted_superposition_kernel<float>
            <<<gridsize, blocksize, smem_size>>>(
                direct_path_real.data_ptr<float>(),
                direct_path_imag.data_ptr<float>(),
                scat_path_real.data_ptr<float>(),
                scat_path_imag.data_ptr<float>(),
                opacity_cont.data_ptr<float>(), influence.data_ptr<float>(), N,
                Nt, Nr, chan_real.data_ptr<float>(),
                chan_imag.data_ptr<float>());
    } else if (direct_path_real.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(direct_path_imag);
        CHECK_DOUBLE_TENSOR(scat_path_real);
        CHECK_DOUBLE_TENSOR(scat_path_imag);
        CHECK_DOUBLE_TENSOR(opacity);
        CHECK_DOUBLE_TENSOR(influence);
        CHECK_DOUBLE_TENSOR(chan_real);
        CHECK_DOUBLE_TENSOR(chan_imag);
        weighted_superposition_kernel<double>
            <<<gridsize, blocksize, smem_size>>>(
                direct_path_real.data_ptr<double>(),
                direct_path_imag.data_ptr<double>(),
                scat_path_real.data_ptr<double>(),
                scat_path_imag.data_ptr<double>(),
                opacity_cont.data_ptr<double>(), influence.data_ptr<double>(),
                N, Nt, Nr, chan_real.data_ptr<double>(),
                chan_imag.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", direct_path_real.dtype());
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after weighted_superposition: ",
                cudaGetErrorString(err));
}
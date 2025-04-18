// engine/_cuda_impl/rasterize.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

#include "checks.cuh"
#include "matrix.cuh"

template <typename T>
__global__ void compute_spatial_influence_kernel(const T* __restrict__ uv,
                                                 const T* __restrict__ cov2d,
                                                 const int num_tx,
                                                 const int num_rx, const int N,
                                                 T* __restrict__ influences) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const T a = cov2d[i * 4 + 0];
    const T b = cov2d[i * 4 + 1];
    const T c = cov2d[i * 4 + 2];
    const T d_cov = cov2d[i * 4 + 3];

    const T det = a * d_cov - b * c;
    const T inv_det = T(1.0) / max(det, T(ROBUST_EPSILON));

    const T inv_a = d_cov * inv_det;
    const T inv_b = -b * inv_det;
    const T inv_c = -c * inv_det;
    const T inv_d = a * inv_det;

    const T u_i = uv[i * 2 + 0];
    const T v_i = uv[i * 2 + 1];

    for (int tx = 0; tx < num_tx; tx++) {
        for (int rx = 0; rx < num_rx; rx++) {
            const T antenna_u = tx + T(0.5);
            const T antenna_v = rx + T(0.5);

            const T du = u_i - antenna_u;
            const T dv = v_i - antenna_v;

            const T md =
                du * (inv_a * du + inv_b * dv) + dv * (inv_c * du + inv_d * dv);

            const T md_clamped = min(md, T(30.0));
            influences[i * num_tx * num_rx + tx * num_rx + rx] =
                exp(-T(0.5) * md_clamped);
        }
    }
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
                "influences must have shape Nx" + std::to_string(num_tx) + "x" +
                    std::to_string(num_rx));

    const int max_threads_per_block = 1024;
    const int num_blocks =
        (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

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
    cudaDeviceSynchronize();
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

    T vec_tx_gauss[3];
    vec_tx_gauss[0] = points[i * 3 + 0] - tx_pos[0];
    vec_tx_gauss[1] = points[i * 3 + 1] - tx_pos[1];
    vec_tx_gauss[2] = points[i * 3 + 2] - tx_pos[2];

    T vec_gauss_rx[3];
    vec_gauss_rx[0] = rx_pos[0] - points[i * 3 + 0];
    vec_gauss_rx[1] = rx_pos[1] - points[i * 3 + 1];
    vec_gauss_rx[2] = rx_pos[2] - points[i * 3 + 2];

    T d_tx_sq = vec_tx_gauss[0] * vec_tx_gauss[0] +
                vec_tx_gauss[1] * vec_tx_gauss[1] +
                vec_tx_gauss[2] * vec_tx_gauss[2];
    T d_tx = sqrt(d_tx_sq);
    dist_tx[i] = max(d_tx, T(ROBUST_EPSILON));

    T d_rx_sq = vec_gauss_rx[0] * vec_gauss_rx[0] +
                vec_gauss_rx[1] * vec_gauss_rx[1] +
                vec_gauss_rx[2] * vec_gauss_rx[2];
    T d_rx = sqrt(d_rx_sq);
    dist_rx[i] = max(d_rx, T(ROBUST_EPSILON));

    aod[i * 2 + 0] = atan2(vec_tx_gauss[1], vec_tx_gauss[0]);
    T aod_el_arg = vec_tx_gauss[2] / dist_tx[i];
    aod_el_arg = min(max(aod_el_arg, T(-1.0)), T(1.0));
    aod[i * 2 + 1] = asin(aod_el_arg);

    aoa[i * 2 + 0] = atan2(vec_gauss_rx[1], vec_gauss_rx[0]);
    T aoa_el_arg = vec_gauss_rx[2] / dist_rx[i];
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

    const int max_threads_per_block = 1024;
    const int num_blocks =
        (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

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
    cudaDeviceSynchronize();
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

    const T PI = T(3.14159265358979323846);
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

        const T x_pos = (row_idx - (rows - T(1.0)) / T(2.0)) * spacing_x;
        const T y_pos = (col_idx - (cols - T(1.0)) / T(2.0)) * spacing_y;

        phase = -k * cos_el * (x_pos * cos_az + y_pos * sin_az);
    } else if (array_type == 1) {
        const int num_ant = static_cast<int>(array_size[0]);
        const T spacing_d = element_spacing[0];
        const T x_pos = (m - (num_ant - T(1.0)) / T(2.0)) * spacing_d;
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

    dim3 blocksize(16, 16, 1);
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
    cudaDeviceSynchronize();
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

    const T PI = T(3.14159265358979323846);

    const T d_tx = dist_tx[i];
    const T d_rx = dist_rx[i];
    const T dist_path = d_tx + d_rx;
    const T dist_path_safe = max(dist_path, T(ROBUST_EPSILON));

    const T alpha_amp = wavelength / (T(4.0) * PI * dist_path_safe);
    const T alpha_phase = -T(2.0) * PI * dist_path / wavelength;
    const T alpha_real = alpha_amp * cos(alpha_phase);
    const T alpha_imag = alpha_amp * sin(alpha_phase);

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

    const T H_T_real = scatter_coef_real * P_real - scatter_coef_imag * P_imag;
    const T H_T_imag = scatter_coef_real * P_imag + scatter_coef_imag * P_real;

    const int out_idx = i * Nt * Nr + tx_ant * Nr + rx_ant;
    scat_chan_real[out_idx] = H_T_real;
    scat_chan_imag[out_idx] = H_T_imag;
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

    dim3 blocksize(4, 4, 4);
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
    cudaDeviceSynchronize();
}
template <typename T>
__global__ void weighted_superposition_kernel(
    const T* __restrict__ direct_path_real,
    const T* __restrict__ direct_path_imag,
    const T* __restrict__ scat_path_real, const T* __restrict__ scat_path_imag,
    const T* __restrict__ opacity, const T* __restrict__ influence, const int N,
    const int Nt, const int Nr, T* __restrict__ chan_real,
    T* __restrict__ chan_imag) {
    const int tx_ant = blockIdx.x * blockDim.x + threadIdx.x;
    const int rx_ant = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx_ant >= Nt || rx_ant >= Nr) {
        return;
    }

    T sum_scatter_real = T(0.0);
    T sum_scatter_imag = T(0.0);

    for (int i = 0; i < N; ++i) {
        const T opac = opacity[i];
        const T infl = influence[i * Nt * Nr + tx_ant * Nr + rx_ant];
        const T weight = opac * infl;

        const T scat_real = scat_path_real[i * Nt * Nr + tx_ant * Nr + rx_ant];
        const T scat_imag = scat_path_imag[i * Nt * Nr + tx_ant * Nr + rx_ant];

        sum_scatter_real += weight * scat_real;
        sum_scatter_imag += weight * scat_imag;
    }

    const int chan_idx = tx_ant * Nr + rx_ant;
    chan_real[chan_idx] = direct_path_real[chan_idx] + sum_scatter_real;
    chan_imag[chan_idx] = direct_path_imag[chan_idx] + sum_scatter_imag;
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

    dim3 blocksize(16, 16, 1);
    dim3 gridsize((Nt + blocksize.x - 1) / blocksize.x,
                  (Nr + blocksize.y - 1) / blocksize.y, 1);

    auto opacity_cont = opacity.contiguous().view({N});

    if (direct_path_real.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(direct_path_imag);
        CHECK_FLOAT_TENSOR(scat_path_real);
        CHECK_FLOAT_TENSOR(scat_path_imag);
        CHECK_FLOAT_TENSOR(opacity);
        CHECK_FLOAT_TENSOR(influence);
        CHECK_FLOAT_TENSOR(chan_real);
        CHECK_FLOAT_TENSOR(chan_imag);
        weighted_superposition_kernel<float><<<gridsize, blocksize>>>(
            direct_path_real.data_ptr<float>(),
            direct_path_imag.data_ptr<float>(),
            scat_path_real.data_ptr<float>(), scat_path_imag.data_ptr<float>(),
            opacity_cont.data_ptr<float>(), influence.data_ptr<float>(), N, Nt,
            Nr, chan_real.data_ptr<float>(), chan_imag.data_ptr<float>());
    } else if (direct_path_real.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(direct_path_imag);
        CHECK_DOUBLE_TENSOR(scat_path_real);
        CHECK_DOUBLE_TENSOR(scat_path_imag);
        CHECK_DOUBLE_TENSOR(opacity);
        CHECK_DOUBLE_TENSOR(influence);
        CHECK_DOUBLE_TENSOR(chan_real);
        CHECK_DOUBLE_TENSOR(chan_imag);
        weighted_superposition_kernel<double><<<gridsize, blocksize>>>(
            direct_path_real.data_ptr<double>(),
            direct_path_imag.data_ptr<double>(),
            scat_path_real.data_ptr<double>(),
            scat_path_imag.data_ptr<double>(), opacity_cont.data_ptr<double>(),
            influence.data_ptr<double>(), N, Nt, Nr,
            chan_real.data_ptr<double>(), chan_imag.data_ptr<double>());
    } else {
        AT_ERROR("Unsupported data type: ", direct_path_real.dtype());
    }
    cudaDeviceSynchronize();
}
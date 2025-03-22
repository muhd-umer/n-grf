/*
 * Auxiliary header file for CUDA kernels
 * Contains utility functions and constants used by the implementation
 */

#ifndef ENGINE_CUDA_AUXILIARY_H
#define ENGINE_CUDA_AUXILIARY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>

#define PI 3.14159265358979323846f

// CUDA kernel helpers
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t status = call;                                          \
        if (status != cudaSuccess) {                                        \
            printf("CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(status));                             \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// Convert symmetric 6D vector to full 3x3 matrix
__host__ __device__ inline void symmetric_to_full(const float* cov_compact,
                                                  float* cov_full) {
    cov_full[0] = cov_compact[0];  // xx
    cov_full[1] = cov_compact[1];  // xy
    cov_full[2] = cov_compact[2];  // xz
    cov_full[3] = cov_compact[1];  // yx = xy
    cov_full[4] = cov_compact[3];  // yy
    cov_full[5] = cov_compact[4];  // yz
    cov_full[6] = cov_compact[2];  // zx = xz
    cov_full[7] = cov_compact[4];  // zy = yz
    cov_full[8] = cov_compact[5];  // zz
}

// Matrix multiplication 3x3 * 3x2 = 3x2
__host__ __device__ inline void mat_mul_3x3_3x2(const float* mat1,
                                                const float* mat2,
                                                float* result) {
    // mat1: 3x3, mat2: 3x2, result: 3x2
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            result[i * 2 + j] = 0.0f;
            for (int k = 0; k < 3; k++) {
                result[i * 2 + j] += mat1[i * 3 + k] * mat2[k * 2 + j];
            }
        }
    }
}

// Matrix multiplication 2x3 * 3x2 = 2x2
__host__ __device__ inline void mat_mul_2x3_3x2(const float* mat1,
                                                const float* mat2,
                                                float* result) {
    // mat1: 2x3, mat2: 3x2, result: 2x2
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            result[i * 2 + j] = 0.0f;
            for (int k = 0; k < 3; k++) {
                result[i * 2 + j] += mat1[i * 3 + k] * mat2[k * 2 + j];
            }
        }
    }
}

// Invert 2x2 matrix
__host__ __device__ inline bool invert_2x2(const float* mat, float* inv,
                                           float* det) {
    // mat: 2x2, inv: 2x2, det: scalar
    *det = mat[0] * mat[3] - mat[1] * mat[2];
    if (fabs(*det) < 1e-10f) {
        return false;
    }
    float inv_det = 1.0f / *det;
    inv[0] = mat[3] * inv_det;
    inv[1] = -mat[1] * inv_det;
    inv[2] = -mat[2] * inv_det;
    inv[3] = mat[0] * inv_det;
    return true;
}

// Compute Mahalanobis distance: d^T * inv_cov * d
__host__ __device__ inline float mahalanobis_distance(const float* inv_cov,
                                                      const float dx,
                                                      const float dy) {
    return inv_cov[0] * dx * dx + 2 * inv_cov[1] * dx * dy +
           inv_cov[3] * dy * dy;
}

#endif  // ENGINE_CUDA_AUXILIARY_H
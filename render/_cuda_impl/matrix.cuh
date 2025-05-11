// render/_cuda_impl/matrix.cuh

#pragma once
#ifndef NGRF_MATRIX_CUH
#define NGRF_MATRIX_CUH

#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
__device__ __forceinline__ void transpose3x3(const T* __restrict__ A,
                                             T* __restrict__ A_T) {
    A_T[0] = A[0];
    A_T[1] = A[3];
    A_T[2] = A[6];
    A_T[3] = A[1];
    A_T[4] = A[4];
    A_T[5] = A[7];
    A_T[6] = A[2];
    A_T[7] = A[5];
    A_T[8] = A[8];
}

template <typename T>
__device__ __forceinline__ void matmul3x3(const T* __restrict__ A,
                                          const T* __restrict__ B,
                                          T* __restrict__ C) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            T sum = T(0.0);
            for (int k = 0; k < 3; ++k) {
                sum += A[r * 3 + k] * B[k * 3 + c];
            }
            C[r * 3 + c] = sum;
        }
    }
}

#endif
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
    C[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
    C[1] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
    C[2] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];

    C[3] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
    C[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
    C[5] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];

    C[6] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];
    C[7] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];
    C[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];
}

#endif
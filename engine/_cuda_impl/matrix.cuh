// engine/_cuda_impl/matrix.cuh

#pragma once
#include <cuda.h>

template <typename T>
__device__ void transpose(const T* A, T* A_T, int num_rows_input,
                          int num_cols_input) {
#pragma unroll
    for (int row = 0; row < num_rows_input; row++) {
#pragma unroll
        for (int col = 0; col < num_cols_input; col++) {
            A_T[col * num_rows_input + row] = A[row * num_cols_input + col];
        }
    }
}

template <typename T>
__device__ void matrix_multiply(const T* A, const T* B, T* C, int num_rows_A,
                                int num_cols_A, int num_cols_B) {
#pragma unroll
    for (int row_a = 0; row_a < num_rows_A; row_a++) {
#pragma unroll
        for (int col_b = 0; col_b < num_cols_B; col_b++) {
            T sum = 0;
#pragma unroll
            for (int cols_A = 0; cols_A < num_cols_A; cols_A++) {
                sum += A[row_a * num_cols_A + cols_A] *
                       B[cols_A * num_cols_B + col_b];
            }
            C[row_a * num_cols_B + col_b] = sum;
        }
    }
}

// Utility function to strip symmetric matrix to compact form [a, b, c, d, e, f]
// (upper triangular)
template <typename T>
__device__ void strip_symmetric(const T* full_matrix, T* compact_form) {
    compact_form[0] = full_matrix[0];  // [0,0]
    compact_form[1] = full_matrix[1];  // [0,1]
    compact_form[2] = full_matrix[2];  // [0,2]
    compact_form[3] = full_matrix[4];  // [1,1]
    compact_form[4] = full_matrix[5];  // [1,2]
    compact_form[5] = full_matrix[8];  // [2,2]
}

// Utility function to expand compact form to full 3x3 matrix
template <typename T>
__device__ void expand_symmetric(const T* compact_form, T* full_matrix) {
    full_matrix[0] = compact_form[0];  // [0,0]
    full_matrix[1] = compact_form[1];  // [0,1]
    full_matrix[2] = compact_form[2];  // [0,2]
    full_matrix[3] = compact_form[1];  // [1,0]
    full_matrix[4] = compact_form[3];  // [1,1]
    full_matrix[5] = compact_form[4];  // [1,2]
    full_matrix[6] = compact_form[2];  // [2,0]
    full_matrix[7] = compact_form[4];  // [2,1]
    full_matrix[8] = compact_form[5];  // [2,2]
}
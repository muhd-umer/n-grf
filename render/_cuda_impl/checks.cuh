// render/_cuda_impl/checks.cuh

#pragma once
#ifndef NGRF_CHECKS_CUH
#define NGRF_CHECKS_CUH

#include <cuda_runtime.h>
#include <torch/extension.h>

#define NGRF_ROBUST_EPSILON_FLOAT 1e-7f
#define NGRF_ROBUST_EPSILON_DOUBLE 1e-15

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_VALID_INPUT(x) \
    CHECK_CUDA(x);           \
    CHECK_CONTIGUOUS(x)

#define CHECK_FLOAT_TENSOR(x) \
    TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be a float tensor")
#define CHECK_DOUBLE_TENSOR(x) \
    TORCH_CHECK(x.dtype() == torch::kFloat64, #x " must be a double tensor")
#define CHECK_COMPLEX_FLOAT_TENSOR(x)              \
    TORCH_CHECK(x.dtype() == torch::kComplexFloat, \
                #x " must be a complex float tensor")
#define CHECK_COMPLEX_DOUBLE_TENSOR(x)              \
    TORCH_CHECK(x.dtype() == torch::kComplexDouble, \
                #x " must be a complex double tensor")

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, \
                    __LINE__, cudaGetErrorString(err));                    \
            TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));   \
        }                                                                  \
    } while (0)

#endif
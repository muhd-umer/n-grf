// ext.cpp

#include <torch/extension.h>

#include "core/cuda/rasterizer.h"

// Forward declarations
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rasterize_forward(
    torch::Tensor points,
    torch::Tensor cov3d,
    torch::Tensor attenuation,
    torch::Tensor phase_rotation,
    torch::Tensor opacity,
    torch::Tensor receiver,
    torch::Tensor transmitter,
    int num_tx,
    int num_rx,
    float frequency,
    torch::Tensor scale_factor);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_backward(
    torch::Tensor grad_output,
    torch::Tensor points,
    torch::Tensor cov3d,
    torch::Tensor attenuation,
    torch::Tensor phase_rotation,
    torch::Tensor opacity,
    torch::Tensor receiver,
    torch::Tensor transmitter,
    int num_tx,
    int num_rx,
    float frequency,
    torch::Tensor scale_factor);

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_forward", &rasterize_forward, "Rasterize forward (CUDA)");
    m.def("rasterize_backward", &rasterize_backward, "Rasterize backward (CUDA)");
}
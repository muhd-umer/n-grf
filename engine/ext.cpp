// ext.cpp

#include <torch/extension.h>

#include "_cuda_impl/rasterize.h"

// forward
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
    float frequency) {
    // placeholder
    auto output = torch::zeros({num_tx, 2 * num_rx}, points.options());
    auto aux1 = torch::zeros({1}, points.options());
    auto aux2 = torch::zeros({1}, points.options());
    return std::make_tuple(output, aux1, aux2);
}

// backward
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
    float frequency) {
    // placeholder
    auto grad_points = torch::zeros_like(points);
    auto grad_cov3d = torch::zeros_like(cov3d);
    auto grad_attenuation = torch::zeros_like(attenuation);
    auto grad_phase_rotation = torch::zeros_like(phase_rotation);
    auto grad_opacity = torch::zeros_like(opacity);

    return std::make_tuple(
        grad_points, grad_cov3d, grad_attenuation,
        grad_phase_rotation, grad_opacity);
}

// bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_forward", &rasterize_forward, "Rasterize forward (CUDA)");
    m.def("rasterize_backward", &rasterize_backward, "Rasterize backward (CUDA)");
}
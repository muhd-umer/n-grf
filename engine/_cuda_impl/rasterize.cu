/*
 * Implementation file for CUDA-PyTorch bindings
 */

#include <torch/extension.h>

#include "auxiliary.h"
#include "forward.h"
#include "rasterize.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

// Compute distances from Gaussians to receiver
torch::Tensor compute_distances_to_receiver_cuda(
    torch::Tensor points,
    torch::Tensor receiver) {
    // Validate inputs
    CHECK_INPUT(points);
    CHECK_INPUT(receiver);

    int P = points.size(0);

    // Create output tensor for distances
    auto distances = torch::empty({P}, points.options());

    // Call CUDA function
    FORWARD::computeDistances(
        P,
        points.data_ptr<float>(),
        receiver.data_ptr<float>(),
        distances.data_ptr<float>());

    return distances;
}

// Compute spherical coordinates
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute_spherical_coords_cuda(
    torch::Tensor points,
    torch::Tensor receiver) {
    // Validate inputs
    CHECK_INPUT(points);
    CHECK_INPUT(receiver);

    int P = points.size(0);

    // Create output tensors
    auto d = torch::empty({P, 3}, points.options());
    auto distances = torch::empty({P}, points.options());
    auto longitude = torch::empty({P}, points.options());
    auto latitude = torch::empty({P}, points.options());

    // Call CUDA function
    FORWARD::computeSphericalCoords(
        P,
        points.data_ptr<float>(),
        receiver.data_ptr<float>(),
        d.data_ptr<float>(),
        distances.data_ptr<float>(),
        longitude.data_ptr<float>(),
        latitude.data_ptr<float>());

    return std::make_tuple(d, longitude, latitude);
}

// Transform spherical coordinates to uniform coordinates
std::tuple<torch::Tensor, torch::Tensor> transform_to_uniform_coords_cuda(
    torch::Tensor longitude,
    torch::Tensor latitude) {
    // Validate inputs
    CHECK_INPUT(longitude);
    CHECK_INPUT(latitude);

    int P = longitude.size(0);

    // Create output tensors
    auto s_x = torch::empty({P}, longitude.options());
    auto s_y = torch::empty({P}, longitude.options());

    // Call CUDA function
    FORWARD::transformToUniformCoords(
        P,
        longitude.data_ptr<float>(),
        latitude.data_ptr<float>(),
        s_x.data_ptr<float>(),
        s_y.data_ptr<float>());

    return std::make_tuple(s_x, s_y);
}

// Map uniform coordinates to channel matrix
torch::Tensor map_to_channel_matrix_cuda(
    torch::Tensor s_x,
    torch::Tensor s_y,
    int num_tx,
    int num_rx) {
    // Validate inputs
    CHECK_INPUT(s_x);
    CHECK_INPUT(s_y);

    int P = s_x.size(0);

    // Create output tensor
    auto uv = torch::empty({P, 2}, s_x.options());

    // Call CUDA function
    FORWARD::mapToChannelMatrix(
        P,
        s_x.data_ptr<float>(),
        s_y.data_ptr<float>(),
        num_tx,
        num_rx,
        uv.data_ptr<float>());

    return uv;
}

// Compute Jacobian matrices
torch::Tensor compute_jacobian_cuda(
    torch::Tensor d,
    torch::Tensor r,
    int num_tx,
    int num_rx) {
    // Validate inputs
    CHECK_INPUT(d);
    CHECK_INPUT(r);

    int P = d.size(0);

    // Create output tensor for jacobian matrices
    auto jacobians = torch::empty({P, 2, 3}, d.options());

    // Call CUDA function
    FORWARD::computeJacobians(
        P,
        d.data_ptr<float>(),
        r.data_ptr<float>(),
        num_tx,
        num_rx,
        jacobians.data_ptr<float>());

    return jacobians;
}

// Project 3D covariance to 2D
torch::Tensor project_cov3d_to_cov2d_cuda(
    torch::Tensor cov3d_mat,
    torch::Tensor jacobian) {
    // Validate inputs
    CHECK_INPUT(cov3d_mat);
    CHECK_INPUT(jacobian);

    int P = cov3d_mat.size(0);

    // Create output tensor for 2D covariance matrices
    auto cov2d = torch::empty({P, 2, 2}, cov3d_mat.options());

    // Convert cov3d_mat from [P, 3, 3] to compact form [P, 6]
    auto cov3d_compact = torch::empty({P, 6}, cov3d_mat.options());

    // Fill in compact form
    cov3d_compact.index({torch::indexing::Slice(), 0}) = cov3d_mat.index({torch::indexing::Slice(), 0, 0});  // xx
    cov3d_compact.index({torch::indexing::Slice(), 1}) = cov3d_mat.index({torch::indexing::Slice(), 0, 1});  // xy
    cov3d_compact.index({torch::indexing::Slice(), 2}) = cov3d_mat.index({torch::indexing::Slice(), 0, 2});  // xz
    cov3d_compact.index({torch::indexing::Slice(), 3}) = cov3d_mat.index({torch::indexing::Slice(), 1, 1});  // yy
    cov3d_compact.index({torch::indexing::Slice(), 4}) = cov3d_mat.index({torch::indexing::Slice(), 1, 2});  // yz
    cov3d_compact.index({torch::indexing::Slice(), 5}) = cov3d_mat.index({torch::indexing::Slice(), 2, 2});  // zz

    // Convert jacobian to flattened form [P, 6]
    auto jacobian_flat = jacobian.reshape({P, 6});

    // Create temporary tensor for compact 2D covariance
    auto cov2d_compact = torch::empty({P, 3}, cov3d_mat.options());

    // Call CUDA function
    FORWARD::projectCov3Ds(
        P,
        cov3d_compact.data_ptr<float>(),
        jacobian_flat.data_ptr<float>(),
        cov2d_compact.data_ptr<float>());

    // Convert compact form back to full matrix form
    cov2d.index({torch::indexing::Slice(), 0, 0}) = cov2d_compact.index({torch::indexing::Slice(), 0});  // xx
    cov2d.index({torch::indexing::Slice(), 0, 1}) = cov2d_compact.index({torch::indexing::Slice(), 1});  // xy
    cov2d.index({torch::indexing::Slice(), 1, 0}) = cov2d_compact.index({torch::indexing::Slice(), 1});  // xy (symmetric)
    cov2d.index({torch::indexing::Slice(), 1, 1}) = cov2d_compact.index({torch::indexing::Slice(), 2});  // yy

    return cov2d;
}

// Project to channel space
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> project_to_channel_space_cuda(
    torch::Tensor points,
    torch::Tensor cov3d,
    torch::Tensor receiver,
    int num_tx,
    int num_rx) {
    // Validate inputs
    CHECK_INPUT(points);
    CHECK_INPUT(cov3d);
    CHECK_INPUT(receiver);

    int P = points.size(0);

    // Create output tensors
    auto distances = torch::empty({P}, points.options());
    auto d = torch::empty({P, 3}, points.options());
    auto longitude = torch::empty({P}, points.options());
    auto latitude = torch::empty({P}, points.options());
    auto s_x = torch::empty({P}, points.options());
    auto s_y = torch::empty({P}, points.options());
    auto uv = torch::empty({P, 2}, points.options());
    auto jacobians = torch::empty({P, 2, 3}, points.options());
    auto cov2d_compact = torch::empty({P, 3}, points.options());
    auto cov2d = torch::empty({P, 2, 2}, points.options());

    // Step 1: Compute spherical coordinates, distances, and displacement vectors
    FORWARD::computeSphericalCoords(
        P,
        points.data_ptr<float>(),
        receiver.data_ptr<float>(),
        d.data_ptr<float>(),
        distances.data_ptr<float>(),
        longitude.data_ptr<float>(),
        latitude.data_ptr<float>());

    // Step 2: Transform to uniform coordinates
    FORWARD::transformToUniformCoords(
        P,
        longitude.data_ptr<float>(),
        latitude.data_ptr<float>(),
        s_x.data_ptr<float>(),
        s_y.data_ptr<float>());

    // Step 3: Map to channel matrix coordinates
    FORWARD::mapToChannelMatrix(
        P,
        s_x.data_ptr<float>(),
        s_y.data_ptr<float>(),
        num_tx,
        num_rx,
        uv.data_ptr<float>());

    // Step 4: Compute Jacobians
    FORWARD::computeJacobians(
        P,
        d.data_ptr<float>(),
        distances.data_ptr<float>(),
        num_tx,
        num_rx,
        jacobians.data_ptr<float>());

    // Step 5: Project 3D covariance to 2D
    FORWARD::projectCov3Ds(
        P,
        cov3d.data_ptr<float>(),
        jacobians.reshape({P, 6}).data_ptr<float>(),
        cov2d_compact.data_ptr<float>());

    // Convert compact form to full matrix form
    cov2d.index({torch::indexing::Slice(), 0, 0}) = cov2d_compact.index({torch::indexing::Slice(), 0});  // xx
    cov2d.index({torch::indexing::Slice(), 0, 1}) = cov2d_compact.index({torch::indexing::Slice(), 1});  // xy
    cov2d.index({torch::indexing::Slice(), 1, 0}) = cov2d_compact.index({torch::indexing::Slice(), 1});  // xy (symmetric)
    cov2d.index({torch::indexing::Slice(), 1, 1}) = cov2d_compact.index({torch::indexing::Slice(), 2});  // yy

    return std::make_tuple(distances, uv, cov2d);
}

// Compute Gaussian influence
torch::Tensor compute_gaussian_influence_cuda(
    torch::Tensor uv,
    torch::Tensor cov2d,
    int num_tx,
    int num_rx) {
    // Validate inputs
    CHECK_INPUT(uv);
    CHECK_INPUT(cov2d);

    int P = uv.size(0);

    // Create output tensor
    auto influences = torch::zeros({P, num_tx, num_rx}, uv.options());

    // Convert cov2d to compact form
    auto cov2d_compact = torch::empty({P, 3}, uv.options());
    cov2d_compact.index({torch::indexing::Slice(), 0}) = cov2d.index({torch::indexing::Slice(), 0, 0});  // xx
    cov2d_compact.index({torch::indexing::Slice(), 1}) = cov2d.index({torch::indexing::Slice(), 0, 1});  // xy
    cov2d_compact.index({torch::indexing::Slice(), 2}) = cov2d.index({torch::indexing::Slice(), 1, 1});  // yy

    // Compute inverse covariance
    auto inv_cov2d = torch::empty({P, 3}, uv.options());
    FORWARD::computeInvCov2Ds(
        P,
        cov2d_compact.data_ptr<float>(),
        inv_cov2d.data_ptr<float>());

    // Compute influence
    FORWARD::computeGaussianInfluences(
        P,
        uv.data_ptr<float>(),
        inv_cov2d.data_ptr<float>(),
        num_tx,
        num_rx,
        influences.data_ptr<float>());

    return influences;
}

// Compute wireless channel
std::tuple<torch::Tensor, torch::Tensor> compute_channel_cuda(
    torch::Tensor attenuation,
    torch::Tensor phase_rotation,
    torch::Tensor distances,
    float wavelength) {
    // Validate inputs
    CHECK_INPUT(attenuation);
    CHECK_INPUT(phase_rotation);
    CHECK_INPUT(distances);

    int P = attenuation.size(0);

    // Create output tensors
    auto real_contrib = torch::empty({P, 1}, attenuation.options());
    auto imag_contrib = torch::empty({P, 1}, attenuation.options());

    // Call CUDA function
    FORWARD::computeChannels(
        P,
        attenuation.data_ptr<float>(),
        phase_rotation.data_ptr<float>(),
        distances.data_ptr<float>(),
        wavelength,
        real_contrib.data_ptr<float>(),
        imag_contrib.data_ptr<float>());

    return std::make_tuple(real_contrib, imag_contrib);
}

// Alpha blending
torch::Tensor alpha_blending_cuda(
    torch::Tensor influences,
    torch::Tensor contributions,
    torch::Tensor opacity,
    torch::Tensor sort_indices,
    int num_tx,
    int num_rx) {
    // Validate inputs
    CHECK_INPUT(influences);
    CHECK_INPUT(contributions);
    CHECK_INPUT(opacity);
    CHECK_INPUT(sort_indices);

    int P = opacity.size(0);

    // Create output tensor
    auto channel = torch::zeros({num_tx, 2 * num_rx}, influences.options());

    // Extract real and imaginary parts
    auto real_contrib = torch::real(contributions).reshape({P, 1});
    auto imag_contrib = torch::imag(contributions).reshape({P, 1});

    // Convert sort_indices to int32 (this fixes the data type issue)
    auto sort_indices_int = sort_indices.to(torch::kInt32);

    // Call CUDA function
    FORWARD::alphaBlending(
        P,
        influences.data_ptr<float>(),
        real_contrib.data_ptr<float>(),
        imag_contrib.data_ptr<float>(),
        opacity.data_ptr<float>(),
        sort_indices_int.data_ptr<int>(),
        num_tx,
        num_rx,
        channel.data_ptr<float>());

    return channel;
}

// Complete forward pass - main function that wrapper calls
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rasterize_forward_cuda(
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
    // Validate inputs
    CHECK_INPUT(points);
    CHECK_INPUT(cov3d);
    CHECK_INPUT(attenuation);
    CHECK_INPUT(phase_rotation);
    CHECK_INPUT(opacity);
    CHECK_INPUT(receiver);
    CHECK_INPUT(transmitter);

    int P = points.size(0);

    // Create output tensor for channel matrix
    auto channel = torch::empty({num_tx, 2 * num_rx}, points.options());

    // Create auxiliary tensors for backward pass (can be used later)
    auto aux1 = torch::empty({P, 1}, points.options());  // For now, just placeholders
    auto aux2 = torch::empty({P, 1}, points.options());

    // Call CUDA function
    FORWARD::forward(
        P,
        points.data_ptr<float>(),
        cov3d.data_ptr<float>(),
        attenuation.data_ptr<float>(),
        phase_rotation.data_ptr<float>(),
        opacity.data_ptr<float>(),
        receiver.data_ptr<float>(),
        transmitter.data_ptr<float>(),
        num_tx,
        num_rx,
        frequency,
        channel.data_ptr<float>());

    return std::make_tuple(channel, aux1, aux2);
}
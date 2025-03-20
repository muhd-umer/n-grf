/*
 * Forward pass header file for channel reconstruction CUDA kernels
 * Contains function declarations for the forward pass operations
 */

#ifndef CUDA_WIRELESS_FORWARD_H_INCLUDED
#define CUDA_WIRELESS_FORWARD_H_INCLUDED

#include <cuda.h>
#include <device_launch_parameters.h>

#include "cuda_runtime.h"

namespace FORWARD {

// Compute distances from Gaussians to receiver
void computeDistances(
    int N,                  // Number of Gaussians
    const float* points,    // 3D positions of Gaussians [N, 3]
    const float* receiver,  // Receiver position [3]
    float* distances);      // Output: distances [N]

// Compute spherical coordinates
void computeSphericalCoords(
    int N,                  // Number of Gaussians
    const float* points,    // 3D positions of Gaussians [N, 3]
    const float* receiver,  // Receiver position [3]
    float* d_out,           // Output: displacement vectors [N, 3]
    float* distances,       // Output: distances [N]
    float* longitude,       // Output: longitude angles [N]
    float* latitude);       // Output: latitude angles [N]

// Transform spherical coordinates to uniform coordinates
void transformToUniformCoords(
    int N,                   // Number of Gaussians
    const float* longitude,  // Longitude angles [N]
    const float* latitude,   // Latitude angles [N]
    float* s_x,              // Output: uniform x coordinates [N]
    float* s_y);             // Output: uniform y coordinates [N]

// Map uniform coordinates to channel matrix coordinates
void mapToChannelMatrix(
    int N,             // Number of Gaussians
    const float* s_x,  // Uniform x coordinates [N]
    const float* s_y,  // Uniform y coordinates [N]
    int num_tx,        // Number of transmit antennas
    int num_rx,        // Number of receive antennas
    float* uv);        // Output: channel matrix coordinates [N, 2]

// Transform spherical coordinates to channel matrix space
void transformToChannelSpace(
    int N,                   // Number of Gaussians
    const float* longitude,  // Longitude angles [N]
    const float* latitude,   // Latitude angles [N]
    int num_tx,              // Number of transmit antennas
    int num_rx,              // Number of receive antennas
    float* uv);              // Output: channel matrix coordinates [N, 2]

// Compute Jacobian matrices for 3D to 2D projection
void computeJacobians(
    int N,                   // Number of Gaussians
    const float* d,          // Displacement vectors [N, 3]
    const float* distances,  // Distances [N]
    int num_tx,              // Number of transmit antennas
    int num_rx,              // Number of receive antennas
    float* jacobians);       // Output: Jacobian matrices [N, 2, 3]

// Project 3D covariance matrices to 2D
void projectCov3Ds(
    int N,                   // Number of Gaussians
    const float* cov3ds,     // 3D covariance matrices [N, 6]
    const float* jacobians,  // Jacobian matrices [N, 2, 3]
    float* cov2ds);          // Output: 2D covariance matrices [N, 3]

// Compute inverse 2D covariance matrices
void computeInvCov2Ds(
    int N,                // Number of Gaussians
    const float* cov2ds,  // 2D covariance matrices [N, 3]
    float* inv_cov2ds);   // Output: inverse 2D covariance matrices [N, 3]

// Compute Gaussian influences on channel matrix elements
void computeGaussianInfluences(
    int N,                    // Number of Gaussians
    const float* uv,          // Channel matrix coordinates [N, 2]
    const float* inv_cov2ds,  // Inverse 2D covariance matrices [N, 3]
    int num_tx,               // Number of transmit antennas
    int num_rx,               // Number of receive antennas
    float* influences);       // Output: Gaussian influences [N, num_tx, num_rx]

// Compute wireless channel contributions
void computeChannels(
    int N,                        // Number of Gaussians
    const float* attenuation,     // Attenuation values [N, 1]
    const float* phase_rotation,  // Phase rotation values [N, 1]
    const float* distances,       // Distances [N]
    float wavelength,             // Signal wavelength
    float* real_contrib,          // Output: real part of contributions [N, 1]
    float* imag_contrib);         // Output: imaginary part of contributions [N, 1]

// Perform alpha blending to form the channel matrix
void alphaBlending(
    int N,                      // Number of Gaussians
    const float* influences,    // Gaussian influences [N, num_tx, num_rx]
    const float* real_contrib,  // Real part of contributions [N, 1]
    const float* imag_contrib,  // Imaginary part of contributions [N, 1]
    const float* opacity,       // Opacity values [N, 1]
    const int* sort_indices,    // Sorted indices [N]
    int num_tx,                 // Number of transmit antennas
    int num_rx,                 // Number of receive antennas
    float* channel);            // Output: channel matrix [num_tx, 2*num_rx]

// Complete forward pass
void forward(
    int N,                        // Number of Gaussians
    const float* points,          // 3D positions of Gaussians [N, 3]
    const float* cov3ds,          // 3D covariance matrices [N, 6]
    const float* attenuation,     // Attenuation values [N, 1]
    const float* phase_rotation,  // Phase rotation values [N, 1]
    const float* opacity,         // Opacity values [N, 1]
    const float* receiver,        // Receiver position [3]
    const float* transmitter,     // Transmitter position [3]
    int num_tx,                   // Number of transmit antennas
    int num_rx,                   // Number of receive antennas
    float frequency,              // Signal frequency
    float* channel);              // Output: channel matrix [num_tx, 2*num_rx]

}  // namespace FORWARD

#endif  // CUDA_WIRELESS_FORWARD_H_INCLUDED
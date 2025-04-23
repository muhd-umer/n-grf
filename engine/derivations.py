# %%
import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function, gradcheck

# %% [markdown]
# ## `Covariance Matrix`
#
# ### **Forward Pass**
#
# The covariance matrix computation follows these steps:
#
# 1. Convert quaternion to rotation matrix:
#    $$R(q) = \begin{bmatrix}
#    1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\
#    2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\
#    2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2)
#    \end{bmatrix}$$
#
# 2. Create scaling matrix with modifier:
#    $$S = \begin{bmatrix}
#    \alpha s_1 & 0 & 0 \\
#    0 & \alpha s_2 & 0 \\
#    0 & 0 & \alpha s_3
#    \end{bmatrix}$$
#
# 3. Compute intermediate matrix:
#    $$RS = R \cdot S$$
#
# 4. Compute covariance matrix:
#    $$\Sigma = RS \cdot (RS)^T = RS \cdot (RS)^T$$
#
# ### **Backward Pass**
#
# 1. Gradient with respect to quaternion components:
#
#    $$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial R} : \begin{bmatrix}
#    0 & -2z & 2y \\
#    2z & 0 & -2x \\
#    -2y & 2x & 0
#    \end{bmatrix}$$
#
#    $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial R} : \begin{bmatrix}
#    0 & 2y & 2z \\
#    2y & -4x & -2w \\
#    2z & 2w & -4x
#    \end{bmatrix}$$
#
#    $$\frac{\partial L}{\partial y} = \frac{\partial L}{\partial R} : \begin{bmatrix}
#    -4y & 2x & 2w \\
#    2x & 0 & 2z \\
#    -2w & 2z & -4y
#    \end{bmatrix}$$
#
#    $$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial R} : \begin{bmatrix}
#    -4z & -2w & 2x \\
#    2w & -4z & 2y \\
#    2x & 2y & 0
#    \end{bmatrix}$$
#
#    Where $:$ denotes the Frobenius inner product.
#
# 2. Gradient with respect to scaling factors:
#    $$\frac{\partial L}{\partial s_i} = \alpha \cdot \frac{\partial L}{\partial S_{ii}}$$
#
# 3. Gradient with respect to RS:
#    $$\frac{\partial L}{\partial RS} = (\frac{\partial L}{\partial \Sigma} + \frac{\partial L}{\partial \Sigma}^T) \cdot RS$$
#
# 4. Gradient for matrix multiplication:
#    $$\frac{\partial L}{\partial R} = \frac{\partial L}{\partial RS} \cdot S^T$$
#    $$\frac{\partial L}{\partial S} = R^T \cdot \frac{\partial L}{\partial RS}$$


# %%
class QuaternionToRotation(Function):
    @staticmethod
    def forward(ctx, q_norm):
        """Convert normalized quaternions to rotation matrices.

        Args:
            q_norm: Normalized quaternion vectors [N, 4]

        Returns:
            Rotation matrices [N, 3, 3]
        """
        w, x, y, z = q_norm[:, 0], q_norm[:, 1], q_norm[:, 2], q_norm[:, 3]

        R = torch.stack(
            [
                1 - 2 * y**2 - 2 * z**2,
                2 * x * y - 2 * w * z,
                2 * x * z + 2 * w * y,
                2 * x * y + 2 * w * z,
                1 - 2 * x**2 - 2 * z**2,
                2 * y * z - 2 * w * x,
                2 * x * z - 2 * w * y,
                2 * y * z + 2 * w * x,
                1 - 2 * x**2 - 2 * y**2,
            ],
            dim=1,
        ).reshape(-1, 3, 3)

        ctx.save_for_backward(q_norm)
        return R

    @staticmethod
    def backward(ctx, grad_R):
        """Backward pass for quaternion to rotation conversion."""
        q_norm = ctx.saved_tensors[0]
        w, x, y, z = q_norm[:, 0], q_norm[:, 1], q_norm[:, 2], q_norm[:, 3]

        grad_w = (
            -2 * z * grad_R[:, 0, 1]
            + 2 * y * grad_R[:, 0, 2]
            + 2 * z * grad_R[:, 1, 0]
            - 2 * x * grad_R[:, 1, 2]
            - 2 * y * grad_R[:, 2, 0]
            + 2 * x * grad_R[:, 2, 1]
        )
        grad_x = (
            2 * y * grad_R[:, 0, 1]
            + 2 * z * grad_R[:, 0, 2]
            + 2 * y * grad_R[:, 1, 0]
            - 4 * x * grad_R[:, 1, 1]
            - 2 * w * grad_R[:, 1, 2]
            + 2 * z * grad_R[:, 2, 0]
            + 2 * w * grad_R[:, 2, 1]
            - 4 * x * grad_R[:, 2, 2]
        )
        grad_y = (
            -4 * y * grad_R[:, 0, 0]
            + 2 * x * grad_R[:, 0, 1]
            + 2 * w * grad_R[:, 0, 2]
            + 2 * x * grad_R[:, 1, 0]
            + 2 * z * grad_R[:, 1, 2]
            - 2 * w * grad_R[:, 2, 0]
            + 2 * z * grad_R[:, 2, 1]
            - 4 * y * grad_R[:, 2, 2]
        )
        grad_z = (
            -4 * z * grad_R[:, 0, 0]
            - 2 * w * grad_R[:, 0, 1]
            + 2 * x * grad_R[:, 0, 2]
            + 2 * w * grad_R[:, 1, 0]
            - 4 * z * grad_R[:, 1, 1]
            + 2 * y * grad_R[:, 1, 2]
            + 2 * x * grad_R[:, 2, 0]
            + 2 * y * grad_R[:, 2, 1]
        )

        grad_q_norm = torch.stack([grad_w, grad_x, grad_y, grad_z], dim=1)
        return grad_q_norm


class ComputeScalingMatrix(Function):
    @staticmethod
    def forward(ctx, scaling, scale_modifier=1.0):
        """Create diagonal scaling matrix from scaling factors."""
        scaled_scaling = scale_modifier * scaling
        S = torch.diag_embed(scaled_scaling)
        ctx.save_for_backward(scaling)
        ctx.scale_modifier = scale_modifier
        return S

    @staticmethod
    def backward(ctx, grad_S):
        """Backward pass for scaling matrix computation."""
        scaling = ctx.saved_tensors[0]
        scale_modifier = ctx.scale_modifier

        grad_S_diag = torch.diagonal(grad_S, dim1=1, dim2=2)
        grad_scaling = grad_S_diag * scale_modifier

        return grad_scaling, None


class MatrixMultiply(Function):
    @staticmethod
    def forward(ctx, A, B):
        """Perform batched matrix multiplication."""
        C = torch.bmm(A, B)
        ctx.save_for_backward(A, B)
        return C

    @staticmethod
    def backward(ctx, grad_C):
        """Backward pass for matrix multiplication."""
        A, B = ctx.saved_tensors
        grad_A = torch.bmm(grad_C, B.transpose(1, 2))
        grad_B = torch.bmm(A.transpose(1, 2), grad_C)
        return grad_A, grad_B


class CovarianceMatrix(Function):
    @staticmethod
    def forward(ctx, RS):
        """Compute covariance matrix from RS matrix."""
        RSRSt = torch.bmm(RS, RS.transpose(1, 2))
        ctx.save_for_backward(RS)
        return RSRSt

    @staticmethod
    def backward(ctx, grad_RSRSt):
        """Backward pass for covariance matrix computation."""
        (RS,) = ctx.saved_tensors
        grad_RS = torch.bmm(grad_RSRSt + grad_RSRSt.transpose(1, 2), RS)
        return grad_RS


# %%
print("Testing covariance computation...")
scaling = torch.rand(100, 3, requires_grad=True, dtype=torch.float64)
rotation = torch.rand(100, 4, requires_grad=True, dtype=torch.float64)
scale_modifier = 1.0
norm = torch.norm(rotation, dim=1, keepdim=True)
q_norm = rotation / norm


test_scaling = gradcheck(
    lambda s: ComputeScalingMatrix.apply(s, scale_modifier), (scaling,)
)
print(f"ComputeScalingMatrix gradcheck: {test_scaling}")

R = QuaternionToRotation.apply(q_norm)
S = ComputeScalingMatrix.apply(scaling, scale_modifier)
test_matrix_multiply = gradcheck(MatrixMultiply.apply, (R, S))
print(f"MatrixMultiply gradcheck: {test_matrix_multiply}")

RS = MatrixMultiply.apply(R, S)
test_covariance = gradcheck(CovarianceMatrix.apply, (RS,))
print(f"CovarianceMatrix gradcheck: {test_covariance}")

# %% [markdown]
# ## `ProjectToChannelCoordinates`
#
# ### **Forward Pass**
#
# This function combines several steps of the original implementation into a single operation:
#
# 1. Compute the displacement vector and distance from Gaussian to receiver:
#    $$\mathbf{d}^k_j = \mathbf{p}^k - \mathbf{r}_j$$
#    $$r^k_j = \|\mathbf{d}^k_j\|$$
#
# 2. Compute spherical coordinates:
#    $$\Omega_{lon} = \arctan2(d^k_{j,y}, d^k_{j,x})$$
#    $$\Omega_{lat} = \arcsin\left(\frac{d^k_{j,z}}{r^k_j}\right)$$
#
# 3. Transform to uniform coordinates:
#    $$s_x = \frac{\Omega_{lon}}{\pi}$$
#    $$s_y = \frac{2\Omega_{lat}}{\pi}$$
#
# 4. Map to channel matrix coordinates:
#    $$u^k_i = \left(\frac{s_x + 1}{2}\right) \cdot (N_t - 1) + 0.5$$
#    $$v^k_j = \left(\frac{s_y + 1}{2}\right) \cdot (N_r - 1) + 0.5$$
#
# ### **Backward Pass**
#
# Using the chain rule, we compute the gradient of the loss with respect to point positions:
#
# - For displacement vector $\mathbf{d}$ and distance $r$:
# $$\frac{\partial r}{\partial \mathbf{p}^k} = \frac{\mathbf{d}^k_j}{r^k_j}$$
#
# $$\frac{\partial \mathbf{d}^k_j}{\partial \mathbf{p}^k} = \mathbf{I}$$
# (where $\mathbf{I}$ is the identity matrix)
#
# - For spherical coordinates:
# $$\frac{\partial \Omega_{lon}}{\partial \mathbf{p}^k} = \begin{bmatrix}
# \frac{-d^k_{j,y}}{(d^k_{j,x})^2 + (d^k_{j,y})^2} &
# \frac{d^k_{j,x}}{(d^k_{j,x})^2 + (d^k_{j,y})^2} &
# 0 \end{bmatrix}^T$$
#
# $$\frac{\partial \Omega_{lat}}{\partial \mathbf{p}^k} = \begin{bmatrix}
# \frac{-d^k_{j,z} \cdot d^k_{j,x}}{r^k_j \cdot \sqrt{(r^k_j)^2 - (d^k_{j,z})^2} \cdot \sqrt{(d^k_{j,x})^2 + (d^k_{j,y})^2}} &
# \frac{-d^k_{j,z} \cdot d^k_{j,y}}{r^k_j \cdot \sqrt{(r^k_j)^2 - (d^k_{j,z})^2} \cdot \sqrt{(d^k_{j,x})^2 + (d^k_{j,y})^2}} &
# \frac{\sqrt{(d^k_{j,x})^2 + (d^k_{j,y})^2}}{r^k_j \cdot \sqrt{(r^k_j)^2 - (d^k_{j,z})^2}} \end{bmatrix}^T$$
#
# - For uniform coordinates:
# $$\frac{\partial s_x}{\partial \Omega_{lon}} = \frac{1}{\pi}$$
# $$\frac{\partial s_y}{\partial \Omega_{lat}} = \frac{2}{\pi}$$
#
# - For channel matrix coordinates:
# $$\frac{\partial u^k_i}{\partial s_x} = \frac{N_t - 1}{2}$$
# $$\frac{\partial v^k_j}{\partial s_y} = \frac{N_r - 1}{2}$$


# %%
class ProjectToChannelCoordinates(Function):
    @staticmethod
    def forward(ctx, points, receiver, num_tx, num_rx):
        """Project points to channel matrix coordinates

        Args:
            points: Gaussian centers [N, 3]
            receiver: Receiver position [3]
            num_tx: Number of transmit antennas
            num_rx: Number of receive antennas

        Returns:
            Tuple containing distances [N], displacement vectors [N, 3], and uv coordinates [N, 2]
        """
        d = points - receiver
        r = torch.sqrt(torch.sum(d**2, dim=1))

        longitude = torch.atan2(d[:, 1], d[:, 0])
        latitude = torch.asin(torch.clamp(d[:, 2] / r, -1.0, 1.0))

        s_x = longitude / math.pi
        s_y = 2.0 * latitude / math.pi

        u = ((s_x + 1.0) / 2.0) * (num_tx - 1) + 0.5
        v = ((s_y + 1.0) / 2.0) * (num_rx - 1) + 0.5
        uv = torch.stack([u, v], dim=1)

        ctx.save_for_backward(points, receiver, r, d, longitude, latitude, s_x, s_y)
        ctx.num_tx = num_tx
        ctx.num_rx = num_rx

        return r, d, uv

    @staticmethod
    def backward(ctx, grad_r, grad_d, grad_uv):
        """Backward pass for projection to channel coordinates."""
        points, receiver, r, d, longitude, latitude, s_x, s_y = ctx.saved_tensors
        num_tx = ctx.num_tx
        num_rx = ctx.num_rx

        grad_points = torch.zeros_like(points)

        for i in range(3):
            grad_points[:, i] += grad_r * d[:, i] / (r + 1e-10)

        grad_points += grad_d

        grad_u = grad_uv[:, 0]
        grad_v = grad_uv[:, 1]

        grad_s_x = grad_u * (num_tx - 1) / 2.0
        grad_s_y = grad_v * (num_rx - 1) / 2.0

        grad_longitude = grad_s_x / math.pi
        grad_latitude = grad_s_y * 2.0 / math.pi

        xy_squared = d[:, 0] ** 2 + d[:, 1] ** 2 + 1e-10
        grad_points[:, 0] += grad_longitude * (-d[:, 1] / xy_squared)
        grad_points[:, 1] += grad_longitude * (d[:, 0] / xy_squared)

        cos_lat = torch.cos(latitude) + 1e-10
        grad_points[:, 0] += grad_latitude * (-d[:, 0] * d[:, 2]) / (r**3 * cos_lat)
        grad_points[:, 1] += grad_latitude * (-d[:, 1] * d[:, 2]) / (r**3 * cos_lat)
        grad_points[:, 2] += (
            grad_latitude * ((d[:, 0] ** 2 + d[:, 1] ** 2)) / (r**3 * cos_lat)
        )

        return grad_points, None, None, None


# %%
print("Testing ProjectToChannelCoordinates...")
points = torch.rand(100, 3, requires_grad=True, dtype=torch.float64)
receiver = torch.rand(3, requires_grad=False, dtype=torch.float64)
num_tx = 16
num_rx = 2

test = gradcheck(ProjectToChannelCoordinates.apply, (points, receiver, num_tx, num_rx))
print(f"ProjectToChannelCoordinates gradcheck: {test}")

# %% [markdown]
# ## `ComputeJacobian`
#
# ### **Forward Pass**
#
# The Jacobian matrix represents the linear approximation of the transformation from 3D space to 2D channel matrix space:
#
# $$\mathbf{J}^k_j = \begin{bmatrix}
# \frac{\partial u^k_i}{\partial x} & \frac{\partial u^k_i}{\partial y} & \frac{\partial u^k_i}{\partial z} \\
# \frac{\partial v^k_j}{\partial x} & \frac{\partial v^k_j}{\partial y} & \frac{\partial v^k_j}{\partial z}
# \end{bmatrix}$$
#
# The explicit form of the Jacobian is:
#
# $$\mathbf{J}^k_j = \begin{bmatrix}
# \frac{N_t - 1}{2\pi} \cdot \frac{-d^k_{j,y}}{(d^k_{j,x})^2+(d^k_{j,y})^2} & \frac{N_t - 1}{2\pi} \cdot \frac{d^k_{j,x}}{(d^k_{j,x})^2+(d^k_{j,y})^2} & 0 \\
# \frac{N_r - 1}{\pi} \cdot \frac{d^k_{j,z} \cdot d^k_{j,x}}{r^k_j \cdot \sqrt{1-(d^k_{j,z}/r^k_j)^2} \cdot ((d^k_{j,x})^2+(d^k_{j,y})^2)} & \frac{N_r - 1}{\pi} \cdot \frac{d^k_{j,z} \cdot d^k_{j,y}}{r^k_j \cdot \sqrt{1-(d^k_{j,z}/r^k_j)^2} \cdot ((d^k_{j,x})^2+(d^k_{j,y})^2)} & \frac{N_r - 1}{\pi} \cdot \frac{1}{r^k_j \cdot \sqrt{1-(d^k_{j,z}/r^k_j)^2}}
# \end{bmatrix}$$
#
# ### **Backward Pass**
#
# For compactness, let $x=d^k_{j,x}$, $y=d^k_{j,y}$, $z=d^k_{j,z}$, and $L_{ij}$ be the upstream gradients for each element of the Jacobian matrix. Let $t = tx\_factor = \frac{N_t - 1}{2\pi}$ and $r = rx\_factor = \frac{N_r - 1}{\pi}$.
#
# $$\frac{\partial L}{\partial x} = \frac{2 \cdot L_{11} \cdot t \cdot x \cdot y \cdot \sqrt{x^2+y^2} - L_{12} \cdot t \cdot x^2 \cdot \sqrt{x^2+y^2} + L_{12} \cdot t \cdot y^2 \cdot \sqrt{x^2+y^2} - 2 \cdot L_{21} \cdot r \cdot x^2 \cdot z + L_{21} \cdot r \cdot y^2 \cdot z - 3 \cdot L_{22} \cdot r \cdot x \cdot y \cdot z - L_{23} \cdot r \cdot x \cdot (x^2+y^2)}{\sqrt{x^2+y^2} \cdot (x^2+y^2)^2}$$
#
# $$\frac{\partial L}{\partial y} = \frac{-L_{11} \cdot t \cdot x^2 \cdot \sqrt{x^2+y^2} + L_{11} \cdot t \cdot y^2 \cdot \sqrt{x^2+y^2} - 2 \cdot L_{12} \cdot t \cdot x \cdot y \cdot \sqrt{x^2+y^2} - 3 \cdot L_{21} \cdot r \cdot x \cdot y \cdot z + L_{22} \cdot r \cdot x^2 \cdot z - 2 \cdot L_{22} \cdot r \cdot y^2 \cdot z - L_{23} \cdot r \cdot y \cdot (x^2+y^2)}{\sqrt{x^2+y^2} \cdot (x^2+y^2)^2}$$
#
# $$\frac{\partial L}{\partial z} = \frac{r \cdot (L_{21} \cdot x + L_{22} \cdot y)}{(x^2+y^2)^{3/2}}$$


# %%
class ComputeJacobian(Function):
    @staticmethod
    def forward(ctx, d, num_tx, num_rx):
        """Compute Jacobian matrix for projection to channel matrix space.

        Args:
            d: Displacement vectors [N, 3] (i.e. points - receiver)
            num_tx: Number of transmit antennas
            num_rx: Number of receive antennas

        Returns:
            Jacobian matrices [N, 2, 3]
        """
        x = d[:, 0]
        y = d[:, 1]
        z = d[:, 2]

        r = torch.sqrt(torch.sum(d**2, dim=1))

        xy_sq = x**2 + y**2
        xy_sq = torch.clamp(xy_sq, min=1e-10)
        sqrt_xy = torch.sqrt(xy_sq)

        cos_lat = torch.sqrt(torch.clamp(1.0 - (z / r) ** 2, min=1e-10))

        tx_factor = (num_tx - 1) / (2.0 * torch.tensor(math.pi))
        rx_factor = (num_rx - 1) / torch.tensor(math.pi)

        N = d.shape[0]
        J = torch.zeros((N, 2, 3), device=d.device, dtype=d.dtype)

        J[:, 0, 0] = tx_factor * (-y / xy_sq)
        J[:, 0, 1] = tx_factor * (x / xy_sq)
        J[:, 0, 2] = 0.0

        r_cos_lat_xy = r * cos_lat * xy_sq
        r_cos_lat_xy = torch.clamp(r_cos_lat_xy, min=1e-10)

        J[:, 1, 0] = rx_factor * (z * x) / r_cos_lat_xy
        J[:, 1, 1] = rx_factor * (z * y) / r_cos_lat_xy
        J[:, 1, 2] = rx_factor / (r * cos_lat)

        ctx.save_for_backward(d, r, xy_sq, cos_lat, J)
        ctx.tx_factor = tx_factor
        ctx.rx_factor = rx_factor

        return J

    @staticmethod
    def backward(ctx, grad_J):
        """Backward pass for the Jacobian computation."""
        d, r, xy_sq, cos_lat, J = ctx.saved_tensors
        tx_factor = ctx.tx_factor
        rx_factor = ctx.rx_factor
        x = d[:, 0]
        y = d[:, 1]
        z = d[:, 2]

        sqrt_xy = torch.sqrt(torch.clamp(xy_sq, min=1e-10))
        denom2 = (xy_sq) ** 2

        L11 = grad_J[:, 0, 0]
        L12 = grad_J[:, 0, 1]
        L21 = grad_J[:, 1, 0]
        L22 = grad_J[:, 1, 1]
        L23 = grad_J[:, 1, 2]

        grad_d_x = (
            2 * L11 * tx_factor * x * y * sqrt_xy
            - L12 * tx_factor * x**2 * sqrt_xy
            + L12 * tx_factor * y**2 * sqrt_xy
            - 2 * L21 * rx_factor * x**2 * z
            + L21 * rx_factor * y**2 * z
            - 3 * L22 * rx_factor * x * y * z
            - L23 * rx_factor * x * xy_sq
        ) / (sqrt_xy * denom2 + 1e-10)

        grad_d_y = (
            -L11 * tx_factor * x**2 * sqrt_xy
            + L11 * tx_factor * y**2 * sqrt_xy
            - 2 * L12 * tx_factor * x * y * sqrt_xy
            - 3 * L21 * rx_factor * x * y * z
            + L22 * rx_factor * x**2 * z
            - 2 * L22 * rx_factor * y**2 * z
            - L23 * rx_factor * y * xy_sq
        ) / (sqrt_xy * denom2 + 1e-10)

        grad_d_z = rx_factor * (L21 * x + L22 * y) / (torch.pow(xy_sq, 1.5) + 1e-10)

        grad_d = torch.stack([grad_d_x, grad_d_y, grad_d_z], dim=1)

        return grad_d, None, None


# %%
print("Testing ComputeJacobian...")
N = 10
num_tx = 16
num_rx = 2
d = torch.rand(N, 3, requires_grad=True, dtype=torch.float64)
test = torch.autograd.gradcheck(ComputeJacobian.apply, (d, num_tx, num_rx))
print(f"ComputeJacobian gradcheck: {test}")

# %% [markdown]
# ## `ProjectCov3dToCov2d`
#
# ### **Forward Pass**
#
# The 2D covariance matrix is obtained by projecting the 3D covariance matrix using the Jacobian:
#
# $$\Sigma^k_{2D,j} = \mathbf{J}^k_j \cdot \Sigma^k_{3D} \cdot (\mathbf{J}^k_j)^T$$
#
# A small constant is added to the diagonal elements to ensure positive definiteness.
#
# ### **Backward Pass**
#
# The backward pass computes the gradients with respect to the 3D covariance and the Jacobian:
#
# $$\frac{\partial L}{\partial \Sigma^k_{3D}} = (\mathbf{J}^k_j)^T \cdot \frac{\partial L}{\partial \Sigma^k_{2D,j}} \cdot \mathbf{J}^k_j$$
#
# $$\frac{\partial L}{\partial \mathbf{J}^k_j} = \frac{\partial L}{\partial \Sigma^k_{2D,j}} \cdot \mathbf{J}^k_j \cdot \Sigma^k_{3D} + \left(\frac{\partial L}{\partial \Sigma^k_{2D,j}}\right)^T \cdot \mathbf{J}^k_j \cdot \Sigma^k_{3D}$$


# %%
class ProjectCov3dToCov2d(Function):
    @staticmethod
    def forward(ctx, cov3d, jacobian):
        """Project 3D covariance matrices to 2D using Jacobian

        Args:
            cov3d: 3D covariance matrices [N, 3, 3]
            jacobian: Jacobian matrices [N, 2, 3]

        Returns:
            2D covariance matrices [N, 2, 2]
        """
        temp = torch.bmm(cov3d, jacobian.transpose(1, 2))
        cov2d = torch.bmm(jacobian, temp)

        cov2d[:, 0, 0] += 0.3
        cov2d[:, 1, 1] += 0.3

        ctx.save_for_backward(cov3d, jacobian, cov2d)
        return cov2d

    @staticmethod
    def backward(ctx, grad_cov2d):
        """Backward pass for projection of 3D covariance to 2D."""
        cov3d, jacobian, cov2d = ctx.saved_tensors

        grad_cov3d = torch.bmm(
            jacobian.transpose(1, 2), torch.bmm(grad_cov2d, jacobian)
        )

        grad_J = torch.bmm(grad_cov2d, torch.bmm(jacobian, cov3d)) + torch.bmm(
            grad_cov2d.transpose(1, 2), torch.bmm(jacobian, cov3d)
        )

        return grad_cov3d, grad_J


# %%
print("Testing ProjectCov3dToCov2d...")
cov3d = torch.rand(100, 3, 3, requires_grad=True, dtype=torch.float64)

cov3d = 0.5 * (cov3d + cov3d.transpose(1, 2))
jacobian = torch.rand(100, 2, 3, requires_grad=True, dtype=torch.float64)

test = gradcheck(ProjectCov3dToCov2d.apply, (cov3d, jacobian), eps=1e-6, atol=1e-5)
print(f"ProjectCov3dToCov2d gradcheck: {test}")

# %% [markdown]
# ## `ComputeSpatialInfluence`
#
# ### **Forward Pass**
#
# The influence of each Gaussian on each channel element is computed using the Mahalanobis distance:
#
# $$\sigma^k_{ij} = \exp\left(-\frac{1}{2} (\mathbf{uv}^k - \mathbf{a}_{ij})^T \cdot (\Sigma^k_{2D,j})^{-1} \cdot (\mathbf{uv}^k - \mathbf{a}_{ij})\right)$$
#
# Where:
# - $\mathbf{uv}^k$ is the projected 2D coordinates of Gaussian $k$
# - $\mathbf{a}_{ij} = [i+0.5, j+0.5]^T$ is the position of antenna $(i,j)$ in the channel matrix space
# - $(\Sigma^k_{2D,j})^{-1}$ is the inverse of the 2D covariance matrix
#
# ### **Backward Pass**
#
# The gradient with respect to the projected coordinates:
#
# $$\frac{\partial \sigma^k_{ij}}{\partial \mathbf{uv}^k} = -\sigma^k_{ij} \cdot (\Sigma^k_{2D,j})^{-1} \cdot (\mathbf{uv}^k - \mathbf{a}_{ij})$$
#
# The gradient with respect to the 2D covariance:
#
# $$\frac{\partial \sigma^k_{ij}}{\partial \Sigma^k_{2D,j}} = \frac{1}{2} \cdot \sigma^k_{ij} \cdot (\Sigma^k_{2D,j})^{-1} \cdot (\mathbf{uv}^k - \mathbf{a}_{ij}) \cdot (\mathbf{uv}^k - \mathbf{a}_{ij})^T \cdot (\Sigma^k_{2D,j})^{-1}$$


# %%
class ComputeSpatialInfluence(Function):
    @staticmethod
    def forward(ctx, uv, cov2d, num_tx, num_rx):
        """Compute influence of Gaussians on channel matrix elements

        Args:
            uv: Channel matrix coordinates [N, 2]
            cov2d: 2D covariance matrices [N, 2, 2]
            num_tx: Number of transmit antennas
            num_rx: Number of receive antennas

        Returns:
            Influences [N, num_tx, num_rx]
        """
        N = uv.shape[0]

        inv_cov2d = torch.inverse(cov2d)
        influences = torch.empty((N, num_tx, num_rx), device=uv.device, dtype=uv.dtype)

        for i in range(num_tx):
            for j in range(num_rx):
                antenna_pos_x = i + 0.5
                antenna_pos_y = j + 0.5

                d = uv - torch.tensor(
                    [antenna_pos_x, antenna_pos_y], device=uv.device, dtype=uv.dtype
                )
                d_x = d[:, 0]
                d_y = d[:, 1]

                inv_00 = inv_cov2d[:, 0, 0]
                inv_01 = inv_cov2d[:, 0, 1]
                inv_10 = inv_cov2d[:, 1, 0]
                inv_11 = inv_cov2d[:, 1, 1]
                md = d_x * (inv_00 * d_x + inv_01 * d_y) + d_y * (
                    inv_10 * d_x + inv_11 * d_y
                )

                influences[:, i, j] = torch.exp(-0.5 * md)

        ctx.save_for_backward(uv, cov2d, inv_cov2d, influences)
        ctx.num_tx = num_tx
        ctx.num_rx = num_rx
        return influences

    @staticmethod
    def backward(ctx, grad_influences):
        """Backward pass for spatial influence computation."""
        uv, cov2d, inv_cov2d, influences = ctx.saved_tensors
        num_tx = ctx.num_tx
        num_rx = ctx.num_rx
        N = uv.shape[0]

        grad_uv = torch.zeros_like(uv)
        grad_cov2d = torch.zeros_like(cov2d)

        for i in range(num_tx):
            for j in range(num_rx):
                antenna_pos = torch.tensor(
                    [i + 0.5, j + 0.5], device=uv.device, dtype=uv.dtype
                )
                d = uv - antenna_pos

                grad_component = -(inv_cov2d @ d.unsqueeze(-1)).squeeze(-1)

                grad_uv = grad_uv + (
                    grad_influences[:, i, j].unsqueeze(-1)
                    * influences[:, i, j].unsqueeze(-1)
                    * grad_component
                )

                d_outer = d.unsqueeze(-1) * d.unsqueeze(1)
                grad_cov2d = grad_cov2d + 0.5 * grad_influences[:, i, j].view(
                    N, 1, 1
                ) * influences[:, i, j].view(N, 1, 1) * (
                    inv_cov2d @ d_outer @ inv_cov2d
                )

        return grad_uv, grad_cov2d, None, None


# %%
print("Testing ComputeSpatialInfluence...")

uv = torch.rand(100, 2, requires_grad=True, dtype=torch.float64)
cov2d = torch.rand(100, 2, 2, requires_grad=True, dtype=torch.float64)

new_cov2d = []
for i in range(100):
    new_cov = torch.matmul(cov2d[i], cov2d[i].T) + torch.eye(
        2, dtype=torch.float64, device=cov2d.device
    )
    new_cov2d.append(new_cov)
cov2d = torch.stack(new_cov2d)

num_tx = 16
num_rx = 2

test = gradcheck(ComputeSpatialInfluence.apply, (uv, cov2d, num_tx, num_rx))
print(f"ComputeSpatialInfluence gradcheck: {test}")

# %% [markdown]
# ## `ComputePathGeometry`
#
# ### **Forward Pass**
#
# This function computes the geometric properties of the paths between transmitters, Gaussians, and receivers:
#
# 1. Compute the vectors from TX to Gaussian and from Gaussian to RX:
#    $$\mathbf{v}_{tx\_gauss} = \mathbf{p}^k - \mathbf{p}_{tx}$$
#    $$\mathbf{v}_{gauss\_rx} = \mathbf{p}_{rx} - \mathbf{p}^k$$
#
# 2. Compute the distances:
#    $$d_{tx} = \|\mathbf{v}_{tx\_gauss}\|_2$$
#    $$d_{rx} = \|\mathbf{v}_{gauss\_rx}\|_2$$
#
# 3. Compute angles of departure and arrival:
#    $$\phi_{aod} = \arctan2(v_{tx\_gauss,y}, v_{tx\_gauss,x})$$
#    $$\theta_{aod} = \arcsin\left(\frac{v_{tx\_gauss,z}}{d_{tx}}\right)$$
#    $$\phi_{aoa} = \arctan2(v_{gauss\_rx,y}, v_{gauss\_rx,x})$$
#    $$\theta_{aoa} = \arcsin\left(\frac{v_{gauss\_rx,z}}{d_{rx}}\right)$$
#
# ### **Backward Pass**
#
# Using the chain rule, we compute gradients with respect to point positions:
#
# 1. Gradient from distances:
#    $$\frac{\partial d_{tx}}{\partial \mathbf{p}^k} = \frac{\mathbf{v}_{tx\_gauss}}{d_{tx}}$$
#    $$\frac{\partial d_{rx}}{\partial \mathbf{p}^k} = -\frac{\mathbf{v}_{gauss\_rx}}{d_{rx}}$$
#
# 2. Gradient from angles of departure (AoD):
#    $$\frac{\partial \phi_{aod}}{\partial \mathbf{p}^k} = \begin{bmatrix}
#    \frac{-v_{tx\_gauss,y}}{v_{tx\_gauss,x}^2 + v_{tx\_gauss,y}^2} \\
#    \frac{v_{tx\_gauss,x}}{v_{tx\_gauss,x}^2 + v_{tx\_gauss,y}^2} \\
#    0
#    \end{bmatrix}$$
#
#    $$\frac{\partial \theta_{aod}}{\partial \mathbf{p}^k} = \begin{bmatrix}
#    \frac{-v_{tx\_gauss,x} \cdot v_{tx\_gauss,z}}{d_{tx}^3 \cdot \cos\theta_{aod}} \\
#    \frac{-v_{tx\_gauss,y} \cdot v_{tx\_gauss,z}}{d_{tx}^3 \cdot \cos\theta_{aod}} \\
#    \frac{v_{tx\_gauss,x}^2 + v_{tx\_gauss,y}^2}{d_{tx}^3 \cdot \cos\theta_{aod}}
#    \end{bmatrix}$$
#
# 3. Gradient from angles of arrival (AoA):
#    $$\frac{\partial \phi_{aoa}}{\partial \mathbf{p}^k} = \begin{bmatrix}
#    \frac{v_{gauss\_rx,y}}{v_{gauss\_rx,x}^2 + v_{gauss\_rx,y}^2} \\
#    \frac{-v_{gauss\_rx,x}}{v_{gauss\_rx,x}^2 + v_{gauss\_rx,y}^2} \\
#    0
#    \end{bmatrix}$$
#
#    $$\frac{\partial \theta_{aoa}}{\partial \mathbf{p}^k} = \begin{bmatrix}
#    \frac{v_{gauss\_rx,x} \cdot v_{gauss\_rx,z}}{d_{rx}^3 \cdot \cos\theta_{aoa}} \\
#    \frac{v_{gauss\_rx,y} \cdot v_{gauss\_rx,z}}{d_{rx}^3 \cdot \cos\theta_{aoa}} \\
#    \frac{-(v_{gauss\_rx,x}^2 + v_{gauss\_rx,y}^2)}{d_{rx}^3 \cdot \cos\theta_{aoa}}
#    \end{bmatrix}$$


# %%
class ComputePathGeometry(Function):
    @staticmethod
    def forward(ctx, points_xyz, tx_pos, rx_pos):
        """Computes distances and angles for Tx-Gaussian-Rx paths.

        Args:
            points_xyz: Gaussian centers [N, 3]
            tx_pos: Tx position [3]
            rx_pos: Rx position [3]

        Returns:
            tuple: A tuple containing:
                dist_tx: Distance from Tx to Gaussian centers [N]
                dist_rx: Distance from Gaussian centers to Rx [N]
                aod: Angle of departure [N, 2] (azimuth, elevation)
                aoa: Angle of arrival [N, 2] (azimuth, elevation)
        """
        vec_tx_gauss = points_xyz - tx_pos
        vec_gauss_rx = rx_pos - points_xyz
        dist_tx = torch.norm(vec_tx_gauss, dim=1).clamp(min=1e-7)
        dist_rx = torch.norm(vec_gauss_rx, dim=1).clamp(min=1e-7)
        aod_az = torch.atan2(vec_tx_gauss[:, 1], vec_tx_gauss[:, 0])
        aod_el = torch.asin((vec_tx_gauss[:, 2] / dist_tx))
        aod = torch.stack([aod_az, aod_el], dim=1)
        aoa_az = torch.atan2(vec_gauss_rx[:, 1], vec_gauss_rx[:, 0])
        aoa_el = torch.asin((vec_gauss_rx[:, 2] / dist_rx))
        aoa = torch.stack([aoa_az, aoa_el], dim=1)
        ctx.save_for_backward(points_xyz, vec_tx_gauss, vec_gauss_rx, dist_tx, dist_rx)
        return dist_tx, dist_rx, aod, aoa

    @staticmethod
    def backward(ctx, grad_dist_tx, grad_dist_rx, grad_aod, grad_aoa):
        """Backward pass for path geometry computation."""
        points_xyz, vec_tx_gauss, vec_gauss_rx, dist_tx, dist_rx = ctx.saved_tensors

        grad_points_xyz = torch.zeros_like(points_xyz)

        # grad from dist_tx
        grad_points_xyz += grad_dist_tx.unsqueeze(-1) * (
            vec_tx_gauss / dist_tx.unsqueeze(-1)
        )

        # grad from dist_rx
        grad_points_xyz += grad_dist_rx.unsqueeze(-1) * (
            -vec_gauss_rx / dist_rx.unsqueeze(-1)
        )

        # grad from AoD
        grad_aod_az, grad_aod_el = grad_aod[:, 0], grad_aod[:, 1]
        vec_x, vec_y, vec_z = vec_tx_gauss[:, 0], vec_tx_gauss[:, 1], vec_tx_gauss[:, 2]
        xy_sq_tx = vec_x**2 + vec_y**2
        grad_points_xyz[:, 0] += grad_aod_az * (-vec_y / xy_sq_tx)
        grad_points_xyz[:, 1] += grad_aod_az * (vec_x / xy_sq_tx)
        cos_aod_el_sq = 1.0 - (vec_z / dist_tx) ** 2
        cos_aod_el = torch.sqrt(cos_aod_el_sq)
        r_cubed_cos_el_tx = dist_tx**3 * cos_aod_el
        grad_points_xyz[:, 0] += grad_aod_el * (-vec_x * vec_z) / r_cubed_cos_el_tx
        grad_points_xyz[:, 1] += grad_aod_el * (-vec_y * vec_z) / r_cubed_cos_el_tx
        grad_points_xyz[:, 2] += grad_aod_el * xy_sq_tx / r_cubed_cos_el_tx

        # grad from AoA
        grad_aoa_az, grad_aoa_el = grad_aoa[:, 0], grad_aoa[:, 1]
        vec_x_a, vec_y_a, vec_z_a = (
            vec_gauss_rx[:, 0],
            vec_gauss_rx[:, 1],
            vec_gauss_rx[:, 2],
        )
        xy_sq_rx = vec_x_a**2 + vec_y_a**2
        grad_points_xyz[:, 0] += grad_aoa_az * (vec_y_a / xy_sq_rx)
        grad_points_xyz[:, 1] += grad_aoa_az * (-vec_x_a / xy_sq_rx)
        cos_aoa_el_sq = 1.0 - (vec_z_a / dist_rx) ** 2
        cos_aoa_el = torch.sqrt(cos_aoa_el_sq)
        r_cubed_cos_el_rx = dist_rx**3 * cos_aoa_el
        grad_points_xyz[:, 0] += grad_aoa_el * (vec_x_a * vec_z_a) / r_cubed_cos_el_rx
        grad_points_xyz[:, 1] += grad_aoa_el * (vec_y_a * vec_z_a) / r_cubed_cos_el_rx
        grad_points_xyz[:, 2] += grad_aoa_el * (-xy_sq_rx) / r_cubed_cos_el_rx

        return grad_points_xyz, None, None


# %%
print("Testing ComputePathGeometry...")
points_xyz = torch.rand(100, 3, requires_grad=True, dtype=torch.float64)
tx_pos = torch.rand(3, dtype=torch.float64)
rx_pos = torch.rand(3, dtype=torch.float64)
test_geom = gradcheck(
    ComputePathGeometry.apply, (points_xyz, tx_pos, rx_pos), eps=1e-6, atol=1e-5
)
print(f"ComputePathGeometry gradcheck: {test_geom}")

# %% [markdown]
# ## `ComputeSteeringVector`
#
# ### **Forward Pass**
#
# The steering vector represents the relative phase shifts across antenna elements due to the impinging wave direction:
#
# 1. For Uniform Rectangular Array (URA):
#    $$\mathbf{a}_{ura}(θ, φ) = \exp\left(-jk \cos(θ) \cdot [x_m \cos(φ) + y_n \sin(φ)]\right)$$
#    where $(x_m, y_n)$ are the coordinates of each antenna element, $k = 2π/λ$ is the wavenumber,
#    and $(θ, φ)$ are elevation and azimuth angles respectively.
#
# 2. For Uniform Linear Array (ULA):
#    $$\mathbf{a}_{ula}(θ, φ) = \exp\left(-jk \cdot x_i \cdot \cos(θ)\cos(φ)\right)$$
#    where $x_i$ is the position of antenna element $i$.
#
# ### **Backward Pass**
#
# 1. For URA, the gradient with respect to angles:
#    $$\frac{\partial \mathbf{a}_{ura,real}}{\partial φ} = k\cos(θ) \cdot [-x_m\sin(φ) + y_n\cos(φ)] \cdot \mathbf{a}_{ura,imag}$$
#    $$\frac{\partial \mathbf{a}_{ura,imag}}{\partial φ} = -k\cos(θ) \cdot [-x_m\sin(φ) + y_n\cos(φ)] \cdot \mathbf{a}_{ura,real}$$
#    $$\frac{\partial \mathbf{a}_{ura,real}}{\partial θ} = k\sin(θ) \cdot [x_m\cos(φ) + y_n\sin(φ)] \cdot \mathbf{a}_{ura,imag}$$
#    $$\frac{\partial \mathbf{a}_{ura,imag}}{\partial θ} = -k\sin(θ) \cdot [x_m\cos(φ) + y_n\sin(φ)] \cdot \mathbf{a}_{ura,real}$$
#
# 2. For ULA, the gradient with respect to angles:
#    $$\frac{\partial \mathbf{a}_{ula,real}}{\partial φ} = kx_i\cos(θ)\sin(φ) \cdot \mathbf{a}_{ula,imag}$$
#    $$\frac{\partial \mathbf{a}_{ula,imag}}{\partial φ} = -kx_i\cos(θ)\sin(φ) \cdot \mathbf{a}_{ula,real}$$
#    $$\frac{\partial \mathbf{a}_{ula,real}}{\partial θ} = kx_i\sin(θ)\cos(φ) \cdot \mathbf{a}_{ula,imag}$$
#    $$\frac{\partial \mathbf{a}_{ula,imag}}{\partial θ} = -kx_i\sin(θ)\cos(φ) \cdot \mathbf{a}_{ula,real}$$


# %%
class ComputeSteeringVector(Function):
    @staticmethod
    def forward(ctx, angles_rad, array_params: Dict[str, Any], wavelength: float):
        """Computes steering vector for URA or ULA.

        Args:
            angles_rad: Angles in radians [N, 2] (azimuth, elevation)
            array_params: Dictionary with array parameters
                - type: 'ura' or 'ula'
                - size: Number of elements in the array (rows, cols for URA)
                - element_spacing: Element spacing in meters
            wavelength: Wavelength in meters

        Returns:
            sv_real: Real part of the steering vector [N, M]
            sv_imag: Imaginary part of the steering vector [N, M]
        """
        N = angles_rad.shape[0]
        az, el = angles_rad[:, 0], angles_rad[:, 1]
        array_type = array_params["type"]
        spacing = array_params["element_spacing"]
        k = 2 * math.pi / wavelength
        if array_type == "ura":
            rows, cols = array_params["size"]
            spacing_x, spacing_y = spacing[0], spacing[1]
            m_indices = (
                torch.arange(rows, device=az.device, dtype=az.dtype) - (rows - 1) / 2.0
            )
            n_indices = (
                torch.arange(cols, device=az.device, dtype=az.dtype) - (cols - 1) / 2.0
            )
            grid_m, grid_n = torch.meshgrid(m_indices, n_indices, indexing="ij")
            x_pos = grid_m.reshape(-1) * spacing_x
            y_pos = grid_n.reshape(-1) * spacing_y
            cos_el = torch.cos(el).unsqueeze(-1)
            cos_az = torch.cos(az).unsqueeze(-1)
            sin_az = torch.sin(az).unsqueeze(-1)
            x_pos_exp = x_pos.unsqueeze(0)
            y_pos_exp = y_pos.unsqueeze(0)
            phase = -k * cos_el * (x_pos_exp * cos_az + y_pos_exp * sin_az)
        elif array_type == "ula":
            num_ant = array_params["size"]
            spacing_d = spacing
            indices = (
                torch.arange(num_ant, device=az.device, dtype=az.dtype)
                - (num_ant - 1) / 2.0
            )
            x_pos = indices * spacing_d
            cos_el = torch.cos(el)
            cos_az = torch.cos(az)
            x_pos_exp = x_pos.unsqueeze(0)
            phase = -k * x_pos_exp * (cos_el * cos_az).unsqueeze(-1)
        else:
            raise ValueError("Unsupported array type")
        sv_real = torch.cos(phase)
        sv_imag = torch.sin(phase)
        ctx.save_for_backward(angles_rad, phase)
        ctx.array_params = array_params
        ctx.wavelength = wavelength
        ctx.k = k
        ctx.array_type = array_type
        if array_type == "ura":
            ctx.x_pos = x_pos
            ctx.y_pos = y_pos
        else:
            ctx.x_pos = x_pos
        return sv_real, sv_imag

    @staticmethod
    def backward(ctx, grad_sv_real, grad_sv_imag):
        """Backward pass for steering vector computation."""
        angles_rad, phase = ctx.saved_tensors
        array_params = ctx.array_params
        k = ctx.k
        array_type = ctx.array_type
        dLdPhase = grad_sv_real * (-torch.sin(phase)) + grad_sv_imag * torch.cos(phase)
        az, el = angles_rad[:, 0], angles_rad[:, 1]
        cos_az, sin_az = torch.cos(az), torch.sin(az)
        cos_el, sin_el = torch.cos(el), torch.sin(el)
        grad_angles = torch.zeros_like(angles_rad)
        if array_type == "ura":
            x_pos = ctx.x_pos.unsqueeze(0)
            y_pos = ctx.y_pos.unsqueeze(0)
            dPh_dAz = (
                -k
                * cos_el.unsqueeze(-1)
                * (-x_pos * sin_az.unsqueeze(-1) + y_pos * cos_az.unsqueeze(-1))
            )
            dPh_dEl = (
                k
                * sin_el.unsqueeze(-1)
                * (x_pos * cos_az.unsqueeze(-1) + y_pos * sin_az.unsqueeze(-1))
            )
        elif array_type == "ula":
            x_pos = ctx.x_pos.unsqueeze(0)
            dPh_dAz = k * x_pos * (cos_el * sin_az).unsqueeze(-1)
            dPh_dEl = k * x_pos * (sin_el * cos_az).unsqueeze(-1)
        else:
            raise ValueError("Unsupported array type")
        grad_angles[:, 0] = torch.sum(dLdPhase * dPh_dAz, dim=1)
        grad_angles[:, 1] = torch.sum(dLdPhase * dPh_dEl, dim=1)
        return grad_angles, None, None


# %%
print("Testing ComputeSteeringVector...")
angles_rad = (
    torch.rand(100, 2, requires_grad=True, dtype=torch.float64) * math.pi - math.pi / 2
)
angles_rad[:, 1] = angles_rad[:, 1].clamp(-math.pi / 2 + 1e-7, math.pi / 2 - 1e-7)
wavelength = 0.1
array_params_ura = {"type": "ura", "size": [4, 4], "element_spacing": [0.05, 0.05]}
test_ura = gradcheck(
    lambda a: ComputeSteeringVector.apply(a, array_params_ura, wavelength),
    (angles_rad,),
)
print(f"ComputeSteeringVector (URA) gradcheck: {test_ura}")
array_params_ula = {"type": "ula", "size": 8, "element_spacing": 0.05}
test_ula = gradcheck(
    lambda a: ComputeSteeringVector.apply(a, array_params_ula, wavelength),
    (angles_rad,),
)
print(f"ComputeSteeringVector (ULA) gradcheck: {test_ula}")

# %% [markdown]
# ## `ComputeScatteredPaths`
#
# ### **Forward Pass**
#
# This function computes the channel contribution from scattered paths:
#
# 1. Calculate path loss and phase shift for propagation:
#    $$\alpha_{amp} = \frac{\lambda}{4\pi d_{path}}$$
#    $$\alpha_{phase} = -\frac{2\pi d_{path}}{\lambda}$$
#    $$\alpha_{real} = \alpha_{amp} \cos(\alpha_{phase})$$
#    $$\alpha_{imag} = \alpha_{amp} \sin(\alpha_{phase})$$
#
# 2. Apply scattering coefficients (where $\beta$ represents the scatter coefficient):
#    $$\beta_{real} = \gamma_{real} \cdot \alpha_{real} - \gamma_{imag} \cdot \alpha_{imag}$$
#    $$\beta_{imag} = \gamma_{real} \cdot \alpha_{imag} + \gamma_{imag} \cdot \alpha_{real}$$
#
# 3. Compute outer product of steering vectors (where $\mathbf{P}$ represents the steering product):
#    $$\mathbf{P}_{real} = \mathbf{sv}_{rx,real} \otimes \mathbf{sv}_{tx,real} + \mathbf{sv}_{rx,imag} \otimes \mathbf{sv}_{tx,imag}$$
#    $$\mathbf{P}_{imag} = \mathbf{sv}_{rx,imag} \otimes \mathbf{sv}_{tx,real} - \mathbf{sv}_{rx,real} \otimes \mathbf{sv}_{tx,imag}$$
#
# 4. Compute channel matrices:
#    $$\mathbf{H}_{real} = \beta_{real} \cdot \mathbf{P}_{real} - \beta_{imag} \cdot \mathbf{P}_{imag}$$
#    $$\mathbf{H}_{imag} = \beta_{real} \cdot \mathbf{P}_{imag} + \beta_{imag} \cdot \mathbf{P}_{real}$$
#
# ### **Backward Pass**
#
# Using the chain rule, the gradients are computed with respect to:
#
# 1. Scattering coefficients:
#    $$\frac{\partial L}{\partial \gamma_{real}} = \frac{\partial L}{\partial \beta_{real}} \cdot \alpha_{real} + \frac{\partial L}{\partial \beta_{imag}} \cdot \alpha_{imag}$$
#    $$\frac{\partial L}{\partial \gamma_{imag}} = -\frac{\partial L}{\partial \beta_{real}} \cdot \alpha_{imag} + \frac{\partial L}{\partial \beta_{imag}} \cdot \alpha_{real}$$
#
# 2. Path distances:
#    $$\frac{\partial L}{\partial d_{path}} = \frac{\partial L}{\partial \alpha_{amp}} \cdot \frac{-\alpha_{amp}}{d_{path}} + \frac{\partial L}{\partial \alpha_{phase}} \cdot \frac{-2\pi}{\lambda}$$
#
# 3. Steering vectors:
#    $$\frac{\partial L}{\partial \mathbf{sv}_{tx,real}} = \frac{\partial L}{\partial \mathbf{P}_{real}} \cdot \mathbf{sv}_{rx,real} + \frac{\partial L}{\partial \mathbf{P}_{imag}} \cdot \mathbf{sv}_{rx,imag}$$
#    $$\frac{\partial L}{\partial \mathbf{sv}_{tx,imag}} = \frac{\partial L}{\partial \mathbf{P}_{real}} \cdot \mathbf{sv}_{rx,imag} - \frac{\partial L}{\partial \mathbf{P}_{imag}} \cdot \mathbf{sv}_{rx,real}$$
#    $$\frac{\partial L}{\partial \mathbf{sv}_{rx,real}} = \frac{\partial L}{\partial \mathbf{P}_{real}} \cdot \mathbf{sv}_{tx,real} - \frac{\partial L}{\partial \mathbf{P}_{imag}} \cdot \mathbf{sv}_{tx,imag}$$
#    $$\frac{\partial L}{\partial \mathbf{sv}_{rx,imag}} = \frac{\partial L}{\partial \mathbf{P}_{real}} \cdot \mathbf{sv}_{tx,imag} + \frac{\partial L}{\partial \mathbf{P}_{imag}} \cdot \mathbf{sv}_{tx,real}$$


# %%
class ComputeScatteredPaths(Function):
    @staticmethod
    def forward(
        ctx,
        gamma_real,
        gamma_imag,
        dist_tx,
        dist_rx,
        sv_tx_real,
        sv_tx_imag,
        sv_rx_real,
        sv_rx_imag,
        wavelength,
    ):
        """Forward pass for scattered-paths channel computation.

        Args:
            gamma_real: Real scattering coefficients [N]
            gamma_imag: Imag scattering coefficients [N]
            dist_tx: Transmit distances [N]
            dist_rx: Receive distances [N]
            sv_tx_real: Tx steering vector real part [N, Nt]
            sv_tx_imag: Tx steering vector imag part [N, Nt]
            sv_rx_real: Rx steering vector real part [N, Nr]
            sv_rx_imag: Rx steering vector imag part [N, Nr]
            wavelength: Wavelength in meters

        Returns:
            scat_chan_real: Real part of scattered path channels [N, Nt, Nr]
            scat_chan_imag: Imag part of scattered path channels [N, Nt, Nr]
        """
        N = gamma_real.shape[0]
        dist_path = dist_tx + dist_rx
        alpha_amp = wavelength / (4 * math.pi * dist_path.clamp(min=1e-10))
        alpha_phase = -2 * math.pi * dist_path / wavelength
        alpha_real = alpha_amp * torch.cos(alpha_phase)
        alpha_imag = alpha_amp * torch.sin(alpha_phase)
        scatter_coef_real = gamma_real * alpha_real - gamma_imag * alpha_imag
        scatter_coef_imag = gamma_real * alpha_imag + gamma_imag * alpha_real
        steering_product_real = torch.einsum(
            "bi,bj->bij", sv_rx_real, sv_tx_real
        ) + torch.einsum("bi,bj->bij", sv_rx_imag, sv_tx_imag)
        steering_product_imag = torch.einsum(
            "bi,bj->bij", sv_rx_imag, sv_tx_real
        ) - torch.einsum("bi,bj->bij", sv_rx_real, sv_tx_imag)
        scat_chan_real_T = (
            scatter_coef_real.view(N, 1, 1) * steering_product_real
            - scatter_coef_imag.view(N, 1, 1) * steering_product_imag
        )
        scat_chan_imag_T = (
            scatter_coef_real.view(N, 1, 1) * steering_product_imag
            + scatter_coef_imag.view(N, 1, 1) * steering_product_real
        )
        scat_chan_real = scat_chan_real_T.permute(0, 2, 1)
        scat_chan_imag = scat_chan_imag_T.permute(0, 2, 1)
        ctx.save_for_backward(
            gamma_real,
            gamma_imag,
            dist_path,
            alpha_amp,
            alpha_phase,
            alpha_real,
            alpha_imag,
            scatter_coef_real,
            scatter_coef_imag,
            sv_tx_real,
            sv_tx_imag,
            sv_rx_real,
            sv_rx_imag,
            steering_product_real,
            steering_product_imag,
        )
        ctx.wavelength = wavelength
        return scat_chan_real, scat_chan_imag

    @staticmethod
    def backward(ctx, grad_scat_chan_real, grad_scat_chan_imag):
        """Backward pass for scattered-paths channel computation."""
        (
            gamma_real,
            gamma_imag,
            dist_path,
            alpha_amp,
            alpha_phase,
            alpha_real,
            alpha_imag,
            scatter_coef_real,
            scatter_coef_imag,
            sv_tx_real,
            sv_tx_imag,
            sv_rx_real,
            sv_rx_imag,
            steering_product_real,
            steering_product_imag,
        ) = ctx.saved_tensors
        wavelength = ctx.wavelength
        N = gamma_real.shape[0]

        grad_scat_chan_real_T = grad_scat_chan_real.permute(0, 2, 1)
        grad_scat_chan_imag_T = grad_scat_chan_imag.permute(0, 2, 1)

        grad_scatter_coef_real = (
            grad_scat_chan_real_T * steering_product_real
            + grad_scat_chan_imag_T * steering_product_imag
        ).sum(dim=(1, 2))
        grad_scatter_coef_imag = (
            -grad_scat_chan_real_T * steering_product_imag
            + grad_scat_chan_imag_T * steering_product_real
        ).sum(dim=(1, 2))

        grad_gamma_real = (
            grad_scatter_coef_real * alpha_real + grad_scatter_coef_imag * alpha_imag
        )
        grad_gamma_imag = (
            -grad_scatter_coef_real * alpha_imag + grad_scatter_coef_imag * alpha_real
        )

        grad_alpha_real = (
            grad_scatter_coef_real * gamma_real + grad_scatter_coef_imag * gamma_imag
        )
        grad_alpha_imag = (
            -grad_scatter_coef_real * gamma_imag + grad_scatter_coef_imag * gamma_real
        )

        grad_alpha_amp = grad_alpha_real * torch.cos(
            alpha_phase
        ) + grad_alpha_imag * torch.sin(alpha_phase)
        grad_alpha_phase = -grad_alpha_real * alpha_amp * torch.sin(
            alpha_phase
        ) + grad_alpha_imag * alpha_amp * torch.cos(alpha_phase)

        grad_dist_path = grad_alpha_amp * (
            -alpha_amp / dist_path
        ) + grad_alpha_phase * (-2 * math.pi / wavelength)
        grad_dist_tx = grad_dist_path
        grad_dist_rx = grad_dist_path

        grad_steering_product_real = (
            scatter_coef_real.view(N, 1, 1) * grad_scat_chan_real_T
            + scatter_coef_imag.view(N, 1, 1) * grad_scat_chan_imag_T
        )
        grad_steering_product_imag = (
            -scatter_coef_imag.view(N, 1, 1) * grad_scat_chan_real_T
            + scatter_coef_real.view(N, 1, 1) * grad_scat_chan_imag_T
        )

        grad_sv_tx_real = (
            grad_steering_product_real * sv_rx_real.unsqueeze(2)
            + grad_steering_product_imag * sv_rx_imag.unsqueeze(2)
        ).sum(dim=1)
        grad_sv_tx_imag = (
            grad_steering_product_real * sv_rx_imag.unsqueeze(2)
            - grad_steering_product_imag * sv_rx_real.unsqueeze(2)
        ).sum(dim=1)
        grad_sv_rx_real = (
            grad_steering_product_real * sv_tx_real.unsqueeze(1)
            - grad_steering_product_imag * sv_tx_imag.unsqueeze(1)
        ).sum(dim=2)
        grad_sv_rx_imag = (
            grad_steering_product_real * sv_tx_imag.unsqueeze(1)
            + grad_steering_product_imag * sv_tx_real.unsqueeze(1)
        ).sum(dim=2)

        return (
            grad_gamma_real,
            grad_gamma_imag,
            grad_dist_tx,
            grad_dist_rx,
            grad_sv_tx_real,
            grad_sv_tx_imag,
            grad_sv_rx_real,
            grad_sv_rx_imag,
            None,
        )


# %%
print("Testing ComputeScatteredPaths...")
N = 100
Nt = 4
Nr = 2
gamma_real = torch.randn(N, requires_grad=True, dtype=torch.float64)
gamma_imag = torch.randn(N, requires_grad=True, dtype=torch.float64)
dist_tx = torch.rand(N, requires_grad=True, dtype=torch.float64) + 1.0
dist_rx = torch.rand(N, requires_grad=True, dtype=torch.float64) + 1.0
sv_tx_real = torch.randn(N, Nt, requires_grad=True, dtype=torch.float64)
sv_tx_imag = torch.randn(N, Nt, requires_grad=True, dtype=torch.float64)
sv_rx_real = torch.randn(N, Nr, requires_grad=True, dtype=torch.float64)
sv_rx_imag = torch.randn(N, Nr, requires_grad=True, dtype=torch.float64)
wavelength = 0.1
test_scatter = gradcheck(
    lambda gr, gi, dt, dr, stxr, stxi, srxr, srxi: ComputeScatteredPaths.apply(
        gr, gi, dt, dr, stxr, stxi, srxr, srxi, wavelength
    ),
    (
        gamma_real,
        gamma_imag,
        dist_tx,
        dist_rx,
        sv_tx_real,
        sv_tx_imag,
        sv_rx_real,
        sv_rx_imag,
    ),
)
print(f"ComputeScatteredPaths gradcheck: {test_scatter}")

# %% [markdown]
# ## `compute_direct_path`
# ### **Forward Pass**
#
# This function computes the non-differentiable direct path channel component:
#
# 1. Calculate the vector and distance from transmitter to receiver:
#    $$\mathbf{v}_{tx\_rx} = \mathbf{p}_{rx} - \mathbf{p}_{tx}$$
#    $$dist_{tx\_rx} = \|\mathbf{v}_{tx\_rx}\|_2$$
#
# 2. Compute free space propagation factors:
#    $$\alpha_{fs\_amp} = \frac{\lambda}{4\pi dist_{tx\_rx}}$$
#    $$\alpha_{fs\_phase} = -\frac{2\pi dist_{tx\_rx}}{\lambda}$$
#    $$\rho_{real} = \alpha_{fs\_amp} \cdot \cos(\alpha_{fs\_phase})$$
#    $$\rho_{imag} = \alpha_{fs\_amp} \cdot \sin(\alpha_{fs\_phase})$$
#
#    where $\rho$ represents the propagation coefficient
#
# 3. Compute angles for steering vectors:
#    $$\phi_{aod} = \arctan2(v_{tx\_rx,y}, v_{tx\_rx,x})$$
#    $$\theta_{aod} = \arcsin\left(\frac{v_{tx\_rx,z}}{dist_{tx\_rx}}\right)$$
#
# 4. Compute steering vectors and outer product:
#    $$\mathbf{P}_{real} = \mathbf{sv}_{rx,real} \otimes \mathbf{sv}_{tx,real} + \mathbf{sv}_{rx,imag} \otimes \mathbf{sv}_{tx,imag}$$
#    $$\mathbf{P}_{imag} = \mathbf{sv}_{rx,imag} \otimes \mathbf{sv}_{tx,real} - \mathbf{sv}_{rx,real} \otimes \mathbf{sv}_{tx,imag}$$
#    where $\mathbf{P}$ represents the steering product
#    $$direct\_chan_{real} = \rho_{real} \cdot \mathbf{P}_{real} - \rho_{imag} \cdot \mathbf{P}_{imag}$$
#    $$direct\_chan_{imag} = \rho_{real} \cdot \mathbf{P}_{imag} + \rho_{imag} \cdot \mathbf{P}_{real}$$
#
# This function is not differentiable as it represents a fixed component of the channel.


# %%
def compute_direct_path(tx_params, rx_params, wavelength):
    """Computes the direct path channel matrix as a non-differentiable constant.

    Args:
        tx_params: Transmitter parameters (dict)
        rx_params: Receiver parameters (dict)
        wavelength: Wavelength in meters

    Returns:
        direct_chan_real: Real part of direct path channel [Nt, Nr]
        direct_chan_imag: Imag part of direct path channel [Nt, Nr]
    """
    with torch.no_grad():
        tx_pos = tx_params["position"]
        rx_pos = rx_params["position"]

        vec_tx_rx = rx_pos - tx_pos
        dist_tx_rx = torch.norm(vec_tx_rx).clamp(min=1e-10)

        alpha_fs_amp = wavelength / (4 * math.pi * dist_tx_rx)
        alpha_fs_phase = -2 * math.pi * dist_tx_rx / wavelength
        prop_coef_real = alpha_fs_amp * torch.cos(alpha_fs_phase)
        prop_coef_imag = alpha_fs_amp * torch.sin(alpha_fs_phase)

        aod_az = torch.atan2(vec_tx_rx[1], vec_tx_rx[0]).unsqueeze(0)
        aod_el = torch.asin(
            (vec_tx_rx[2] / dist_tx_rx).clamp(-1 + 1e-7, 1 - 1e-7)
        ).unsqueeze(0)
        aod = torch.cat([aod_az, aod_el], dim=0).unsqueeze(0)
        aoa = aod

        # steering vectors
        sv_tx_real, sv_tx_imag = ComputeSteeringVector.apply(aod, tx_params, wavelength)
        sv_rx_real, sv_rx_imag = ComputeSteeringVector.apply(aoa, rx_params, wavelength)

        steering_product_real = torch.einsum(
            "bi,bj->bij", sv_rx_real, sv_tx_real
        ) + torch.einsum("bi,bj->bij", sv_rx_imag, sv_tx_imag)
        steering_product_imag = torch.einsum(
            "bi,bj->bij", sv_rx_imag, sv_tx_real
        ) - torch.einsum("bi,bj->bij", sv_rx_real, sv_tx_imag)
        direct_chan_real_T = (
            prop_coef_real * steering_product_real
            - prop_coef_imag * steering_product_imag
        )
        direct_chan_imag_T = (
            prop_coef_real * steering_product_imag
            + prop_coef_imag * steering_product_real
        )
        direct_chan_real = direct_chan_real_T.squeeze(0).T
        direct_chan_imag = direct_chan_imag_T.squeeze(0).T

    return direct_chan_real.detach(), direct_chan_imag.detach()


# %%
print("Testing compute_direct_path...")
tx_pos_s = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
rx_pos_s = torch.tensor([10.0, 5.0, 1.0], dtype=torch.float64)
wavelength_s = 0.1
tx_params_s = {
    "type": "ura",
    "size": [4, 4],
    "element_spacing": [0.05, 0.05],
    "position": tx_pos_s,
}
rx_params_s = {"type": "ula", "size": 2, "element_spacing": 0.05, "position": rx_pos_s}
direct_path_real, direct_path_imag = compute_direct_path(
    tx_params_s, rx_params_s, wavelength_s
)
print(
    f"Direct Path channel shape: {direct_path_real.shape}, requires_grad: {direct_path_real.requires_grad}"
)

# %% [markdown]
# ## `WeightedSuperposition`
#
# ### **Forward Pass**
#
# This function computes the weighted superposition of scattered paths and the direct path:
#
# 1. Apply weighting to scattered paths:
#    $w_i = o_i \times \mathcal{I}_{i,t,r}$
#    Where $o_i$ is the opacity of Gaussian $i$ and $\mathcal{I}_{i,t,r}$ is its spatial influence.
#
# 2. Sum the weighted scattered paths:
#    $\sum\_scatter_{real} = \sum_{i=1}^{N} w_i \times H_{i,real}$
#    $\sum\_scatter_{imag} = \sum_{i=1}^{N} w_i \times H_{i,imag}$
#
# 3. Add direct path to get final prediction:
#    $H_{pred,real} = H_{direct,real} + \sum\_scatter_{real}$
#    $H_{pred,imag} = H_{direct,imag} + \sum\_scatter_{imag}$
#
# ### **Backward Pass**
#
# The gradients are computed with respect to:
#
# 1. Scattered path matrices:
#    $\frac{\partial L}{\partial H_{i,real}} = w_i \times \frac{\partial L}{\partial H_{pred,real}}$
#    $\frac{\partial L}{\partial H_{i,imag}} = w_i \times \frac{\partial L}{\partial H_{pred,imag}}$
#
# 2. Weights:
#    $\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial H_{pred,real}} \times H_{i,real} + \frac{\partial L}{\partial H_{pred,imag}} \times H_{i,imag}$
#
# 3. Opacity:
#    $\frac{\partial L}{\partial o_i} = \sum_{t,r} \frac{\partial L}{\partial w_i} \times \mathcal{I}_{i,t,r}$
#
# 4. Influence:
#    $\frac{\partial L}{\partial \mathcal{I}_{i,t,r}} = \frac{\partial L}{\partial w_i} \times o_i$
#
# Note that gradients for the direct path components are not computed as they are treated as constants.


# %%
class WeightedSuperposition(Function):
    @staticmethod
    def forward(
        ctx,
        direct_path_real,
        direct_path_imag,
        scat_path_real,
        scat_path_imag,
        opacity,
        influence,
    ):
        """Performs weighted complex superposition of direct and scattered paths.

        Args:
            direct_path_real: Real part of direct path channel [Nt, Nr]
            direct_path_imag: Imag part of direct path channel [Nt, Nr]
            scat_path_real: Real part of scattered paths [N, Nt, Nr]
            scat_path_imag: Imag part of scattered paths [N, Nt, Nr]
            opacity: Gaussian opacity values [N, 1]
            influence: Spatial influence values [N, Nt, Nr]

        Returns:
            chan_pred_real: Real part of predicted channel [Nt, Nr]
            chan_pred_imag: Imag part of predicted channel [Nt, Nr]
        """
        N = scat_path_real.shape[0]
        opacity_exp = opacity.view(N, 1, 1)
        weights = opacity_exp * influence
        sum_scatter_real = torch.sum(weights * scat_path_real, dim=0)
        sum_scatter_imag = torch.sum(weights * scat_path_imag, dim=0)
        chan_pred_real = direct_path_real + sum_scatter_real
        chan_pred_imag = direct_path_imag + sum_scatter_imag

        ctx.save_for_backward(
            scat_path_real, scat_path_imag, opacity, influence, weights
        )
        return chan_pred_real, chan_pred_imag

    @staticmethod
    def backward(ctx, grad_chan_pred_real, grad_chan_pred_imag):
        """Backward pass for weighted superposition."""
        scat_path_real, scat_path_imag, opacity, influence, weights = ctx.saved_tensors
        N = scat_path_real.shape[0]

        # gradient w.r.t scattered paths
        grad_scat_path_real = weights * grad_chan_pred_real.unsqueeze(0)
        grad_scat_path_imag = weights * grad_chan_pred_imag.unsqueeze(0)

        # gradient w.r.t weights
        grad_weights = (
            grad_chan_pred_real.unsqueeze(0) * scat_path_real
            + grad_chan_pred_imag.unsqueeze(0) * scat_path_imag
        )

        grad_opacity = torch.sum(grad_weights * influence, dim=(1, 2)).unsqueeze(-1)
        grad_influence = grad_weights * opacity.view(N, 1, 1)

        return (
            None,
            None,
            grad_scat_path_real,
            grad_scat_path_imag,
            grad_opacity,
            grad_influence,
        )


# %%
print("Testing WeightedSuperposition...")
N = 100
Nt = 4
Nr = 2
direct_path_real = torch.randn(Nt, Nr, dtype=torch.float64)  # Constant
direct_path_imag = torch.randn(Nt, Nr, dtype=torch.float64)
scat_path_real = torch.randn(N, Nt, Nr, requires_grad=True, dtype=torch.float64)
scat_path_imag = torch.randn(N, Nt, Nr, requires_grad=True, dtype=torch.float64)
opacity_act = torch.rand(N, 1, requires_grad=True, dtype=torch.float64)  # Activated
influence = torch.rand(N, Nt, Nr, requires_grad=True, dtype=torch.float64)
test_super = gradcheck(
    WeightedSuperposition.apply,
    (
        direct_path_real,
        direct_path_imag,
        scat_path_real,
        scat_path_imag,
        opacity_act,
        influence,
    ),
    eps=1e-6,
    atol=1e-5,
)
print(f"WeightedSuperposition gradcheck: {test_super}")

# %% [markdown]
# ## **Tests**
#
# This section tests the forward pass made via `Autograd` funcs against the one in `_torch_impl` directory.

# %%
import sys

from _torch_impl.rasterize import rasterize

sys.path.append("../")
from models.loss import MSECorrLoss


# %%
def rasterize_channel(
    points: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    opacity: torch.Tensor,
    gamma_real: torch.Tensor,
    gamma_imag: torch.Tensor,
    tx_params: Dict[str, Any],
    rx_params: Dict[str, Any],
    scale_modifier: float = 1.0,
):
    """Rasterizes the channel using Gaussian scatterers and superposition.

    Args:
        points: Gaussian centers [N, 3]
        scaling: Activated scaling factors [N, 3]
        rotation: Normalized rotation quaternions [N, 4]
        opacity: Activated opacity values [N, 1]
        gamma_real: Learned scattering real part [N]
        gamma_imag: Learned scattering imag part [N]
        tx_params: Transmitter parameters (position, type, size, spacing, freq, num_antennas)
        rx_params: Receiver parameters (position, type, size, spacing, num_antennas)
        scale_modifier: Global scaling modifier

    Returns:
        Predicted channel matrix [Nt, 2*Nr] (real/imag stacked)
    """
    tx_pos = tx_params["position"]
    rx_pos = rx_params["position"]
    num_tx = tx_params["num_antennas"]
    num_rx = rx_params["num_antennas"]
    frequency = tx_params["frequency"]
    wavelength = 299792458.0 / frequency

    R = QuaternionToRotation.apply(rotation)
    S = ComputeScalingMatrix.apply(scaling, scale_modifier)
    RS = MatrixMultiply.apply(R, S)
    cov3d = CovarianceMatrix.apply(RS)

    dist_tx, dist_rx, aod, aoa = ComputePathGeometry.apply(points, tx_pos, rx_pos)

    sv_tx_real, sv_tx_imag = ComputeSteeringVector.apply(aod, tx_params, wavelength)
    sv_rx_real, sv_rx_imag = ComputeSteeringVector.apply(aoa, rx_params, wavelength)

    scat_path_real, scat_path_imag = ComputeScatteredPaths.apply(
        gamma_real,
        gamma_imag,
        dist_tx,
        dist_rx,
        sv_tx_real,
        sv_tx_imag,
        sv_rx_real,
        sv_rx_imag,
        wavelength,
    )

    direct_path_real, direct_path_imag = compute_direct_path(
        tx_params, rx_params, wavelength
    )

    r_proj, d_proj, uv = ProjectToChannelCoordinates.apply(
        points, rx_pos, num_tx, num_rx
    )
    jacobian = ComputeJacobian.apply(d_proj, num_tx, num_rx)
    cov2d = ProjectCov3dToCov2d.apply(cov3d, jacobian)
    influence = ComputeSpatialInfluence.apply(uv, cov2d, num_tx, num_rx)

    chan_pred_real, chan_pred_imag = WeightedSuperposition.apply(
        direct_path_real,
        direct_path_imag,
        scat_path_real,
        scat_path_imag,
        opacity,
        influence,
    )

    chan_pred = torch.cat([chan_pred_real, chan_pred_imag], dim=1)

    return chan_pred


# %%
def test_agf_rasterize_channel():
    """Tests the end-to-end rasterize_channel function."""
    print("Testing end-to-end rasterize_channel function...")
    torch.manual_seed(4)

    N = 100  # number of gaussians
    frequency = 2.4e9
    scale_modifier = 1.0

    tx_params = {
        "position": torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64),
        "type": "ura",
        "size": [4, 4],
        "element_spacing": [0.06, 0.06],
        "frequency": frequency,
        "num_antennas": 16,
    }
    rx_params = {
        "position": torch.tensor([15.0, 8.0, 1.5], dtype=torch.float64),
        "type": "ula",
        "size": 2,
        "element_spacing": 0.06,
        "num_antennas": 2,
    }

    # raw inputs
    scaling_raw = torch.randn(N, 3, requires_grad=True, dtype=torch.float64)
    rotation_raw = torch.randn(N, 4, requires_grad=True, dtype=torch.float64)
    opacity_raw = torch.randn(N, 1, requires_grad=True, dtype=torch.float64)
    points_xyz = (
        (torch.rand(N, 3, dtype=torch.float64) * 15.0).detach().requires_grad_()
    )

    gamma_real = (torch.randn(N, dtype=torch.float64) * 1e-2).detach().requires_grad_()
    gamma_imag = (torch.randn(N, dtype=torch.float64) * 1e-2).detach().requires_grad_()

    # apply activations
    scaling_act = torch.exp(scaling_raw)
    rotation_norm = F.normalize(rotation_raw, dim=1)
    opacity_act = torch.sigmoid(opacity_raw)

    channel = rasterize_channel(
        points_xyz,
        scaling_act,
        rotation_norm,
        opacity_act,
        gamma_real,
        gamma_imag,
        tx_params,
        rx_params,
        scale_modifier,
    )

    print(f"Channel matrix shape: {channel.shape}")
    print(f"Channel matrix type: {channel.dtype}")
    print("First few values of channel matrix:")
    print(channel[:2, :])

    # check if gradients can be computed
    loss = channel.sum()
    loss.backward()

    print("Gradients computed successfully!")
    if points_xyz.grad is not None:
        print(f"Grad w.r.t points_xyz: {points_xyz.grad[:2]}")
    if scaling_raw.grad is not None:
        print(f"Grad w.r.t scaling_raw: {scaling_raw.grad[:2]}")
    if rotation_raw.grad is not None:
        print(f"Grad w.r.t rotation_raw: {rotation_raw.grad[:2]}")
    if opacity_raw.grad is not None:
        print(f"Grad w.r.t opacity_raw: {opacity_raw.grad[:2]}")
    if gamma_real.grad is not None:
        print(f"Grad w.r.t gamma_real: {gamma_real.grad[:2]}")
    if gamma_imag.grad is not None:
        print(f"Grad w.r.t gamma_imag: {gamma_imag.grad[:2]}")

    return True


test_agf_rasterize_channel()


# %%
def test_rasterize_func():
    print("Testing _torch_impl.rasterize/rasterize function...")
    torch.manual_seed(4)  # for reproducibility

    N = 100  # number of gaussians
    frequency = 2.4e9
    scale_modifier = 1.0

    tx_params = {
        "position": torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64),
        "type": "ura",
        "size": [4, 4],
        "element_spacing": [0.06, 0.06],
        "frequency": frequency,
        "num_antennas": 16,
    }
    rx_params = {
        "position": torch.tensor([15.0, 8.0, 1.5], dtype=torch.float64),
        "type": "ula",
        "size": 2,
        "element_spacing": 0.06,
        "num_antennas": 2,
    }

    # raw inputs
    scaling_raw = torch.randn(N, 3, requires_grad=True, dtype=torch.float64)
    rotation_raw = torch.randn(N, 4, requires_grad=True, dtype=torch.float64)
    opacity_raw = torch.randn(N, 1, requires_grad=True, dtype=torch.float64)
    points_xyz = (
        (torch.rand(N, 3, dtype=torch.float64) * 15.0).detach().requires_grad_()
    )

    gamma_real = (torch.randn(N, dtype=torch.float64) * 1e-2).detach().requires_grad_()
    gamma_imag = (torch.randn(N, dtype=torch.float64) * 1e-2).detach().requires_grad_()
    gamma = torch.stack([gamma_real, gamma_imag], dim=1)

    # apply activations
    scaling_act = torch.exp(scaling_raw)
    rotation_norm = F.normalize(rotation_raw, dim=1)
    opacity_act = torch.sigmoid(opacity_raw)

    channel = rasterize(
        points_xyz,
        scaling_act,
        rotation_norm,
        gamma,
        opacity_act,
        tx_params,
        rx_params,
        scale_modifier,
    )

    print(f"Channel matrix shape: {channel.shape}")
    print(f"Channel matrix type: {channel.dtype}")
    print("First few values of channel matrix:")
    print(channel[:2, :])

    # check if gradients can be computed
    loss = channel.sum()
    loss.backward()

    print("Gradients computed successfully!")
    if points_xyz.grad is not None:
        print(f"Grad w.r.t points_xyz: {points_xyz.grad[:2]}")
    if scaling_raw.grad is not None:
        print(f"Grad w.r.t scaling_raw: {scaling_raw.grad[:2]}")
    if rotation_raw.grad is not None:
        print(f"Grad w.r.t rotation_raw: {rotation_raw.grad[:2]}")
    if opacity_raw.grad is not None:
        print(f"Grad w.r.t opacity_raw: {opacity_raw.grad[:2]}")
    if gamma_real.grad is not None:
        print(f"Grad w.r.t gamma_real: {gamma_real.grad[:2]}")
    if gamma_imag.grad is not None:
        print(f"Grad w.r.t gamma_imag: {gamma_imag.grad[:2]}")

    return True


test_rasterize_func()

# %% [markdown]
# ### Comparison


# %%
def run_comparison_test():
    print("Running comparison test between implementations...")
    torch.manual_seed(0)  # for reproducibility

    N = 1000
    num_tx = 16
    num_rx = 2
    frequency = 2.4e9
    scale_modifier = 1.0  # not trainable

    points = torch.rand(N, 3) * 20
    scaling = torch.exp(torch.rand(N, 3) - 0.5)
    rotation = F.normalize(torch.rand(N, 4), dim=1)
    gamma_real = torch.rand(N) * 1e-3
    gamma_imag = torch.rand(N) * 1e-3
    opacity = torch.sigmoid(torch.rand(N, 1))
    receiver = torch.rand(3)
    transmitter = torch.rand(3)

    points_orig = points.clone().requires_grad_()
    scaling_orig = scaling.clone().requires_grad_()
    rotation_orig = rotation.clone().requires_grad_()
    gamma_orig = torch.stack([gamma_real, gamma_imag], dim=1).clone().requires_grad_()
    opacity_orig = opacity.clone().requires_grad_()

    points_autograd = points.clone().requires_grad_()
    scaling_autograd = scaling.clone().requires_grad_()
    rotation_autograd = rotation.clone().requires_grad_()
    gamma_real_autograd = gamma_real.clone().requires_grad_()
    gamma_imag_autograd = gamma_imag.clone().requires_grad_()
    opacity_autograd = opacity.clone().requires_grad_()

    wavelength = 299792458.0 / frequency
    tx_params = {
        "position": transmitter,
        "type": "ura",
        "size": [4, 4],  # 4x4 URA for 16 antennas
        "element_spacing": [wavelength / 2, wavelength / 2],
        "frequency": frequency,
        "num_antennas": num_tx,
    }

    rx_params = {
        "position": receiver,
        "type": "ula",
        "size": num_rx,
        "element_spacing": wavelength / 2,
        "num_antennas": num_rx,
    }

    channel_orig = rasterize(
        points_orig,
        scaling_orig,
        rotation_orig,
        gamma_orig,
        opacity_orig,
        tx_params,
        rx_params,
        scale_modifier,
    )

    channel_autograd = rasterize_channel(
        points_autograd,
        scaling_autograd,
        rotation_autograd,
        opacity_autograd,
        gamma_real_autograd,
        gamma_imag_autograd,
        tx_params,
        rx_params,
        scale_modifier,
    )

    mse_corr_loss = MSECorrLoss()
    max_diff = torch.max(torch.abs(channel_orig - channel_autograd))
    print(f"Forward Pass - Maximum difference: {max_diff.item()}")

    if max_diff > 1e-6:
        print("❌ Forward outputs don't match!")
        print("Original (first few elements):", channel_orig[0, :4])
        print("Autograd (first few elements):", channel_autograd[0, :4])
    else:
        print("✅ Forward outputs match within tolerance.")

    target_channel = channel_orig.detach() + torch.randn_like(channel_orig) * 0.1

    loss_orig = mse_corr_loss(channel_orig, target_channel)
    loss_autograd = mse_corr_loss(channel_autograd, target_channel)

    loss_orig.backward()
    loss_autograd.backward()

    print(
        f"Loss value - Original: {loss_orig.item()}, Autograd: {loss_autograd.item()}"
    )

    print("\n=== Gradient Comparison ===")

    if points_orig.grad is not None and points_autograd.grad is not None:
        points_grad_diff = torch.max(torch.abs(points_orig.grad - points_autograd.grad))
        print(f"Points gradients - Maximum difference: {points_grad_diff.item()}")
        if points_grad_diff > 1e-4:
            print("❌ Points gradients don't match!")
            print("Original (first 3):")
            print(points_orig.grad[:3])
            print("Autograd (first 3):")
            print(points_autograd.grad[:3])
        else:
            print("✅ Points gradients match within tolerance.")
    else:
        print("⚠️ Points gradients - One or both gradients are None")

    if scaling_orig.grad is not None and scaling_autograd.grad is not None:
        scaling_grad_diff = torch.max(
            torch.abs(scaling_orig.grad - scaling_autograd.grad)
        )
        print(f"Scaling gradients - Maximum difference: {scaling_grad_diff.item()}")
        if scaling_grad_diff > 1e-4:
            print("❌ Scaling gradients don't match!")
            print("Original (first 3):")
            print(scaling_orig.grad[:3])
            print("Autograd (first 3):")
            print(scaling_autograd.grad[:3])
        else:
            print("✅ Scaling gradients match within tolerance.")
    else:
        print("⚠️ Scaling gradients - One or both gradients are None")

    if rotation_orig.grad is not None and rotation_autograd.grad is not None:
        rotation_grad_diff = torch.max(
            torch.abs(rotation_orig.grad - rotation_autograd.grad)
        )
        print(f"Rotation gradients - Maximum difference: {rotation_grad_diff.item()}")
        if rotation_grad_diff > 1e-4:
            print("❌ Rotation gradients don't match!")
            print("Original (first 3):")
            print(rotation_orig.grad[:3])
            print("Autograd (first 3):")
            print(rotation_autograd.grad[:3])
        else:
            print("✅ Rotation gradients match within tolerance.")
    else:
        print("⚠️ Rotation gradients - One or both gradients are None")

    if (
        gamma_orig.grad is not None
        and gamma_real_autograd.grad is not None
        and gamma_imag_autograd.grad is not None
    ):
        gamma_real_grad_diff = torch.max(
            torch.abs(gamma_orig.grad[:, 0] - gamma_real_autograd.grad)
        )
        gamma_imag_grad_diff = torch.max(
            torch.abs(gamma_orig.grad[:, 1] - gamma_imag_autograd.grad)
        )
        print(
            f"Gamma real gradients - Maximum difference: {gamma_real_grad_diff.item()}"
        )
        print(
            f"Gamma imag gradients - Maximum difference: {gamma_imag_grad_diff.item()}"
        )

        if gamma_real_grad_diff > 1e-3 or gamma_imag_grad_diff > 1e-3:
            print("❌ Gamma gradients don't match!")
            print("Original gamma (first 3):")
            print(gamma_orig.grad[:3])
            print("Autograd gamma_real (first 3):")
            print(gamma_real_autograd.grad[:3])
            print("Autograd gamma_imag (first 3):")
            print(gamma_imag_autograd.grad[:3])
        else:
            print("✅ Gamma gradients match within tolerance.")
    else:
        print("⚠️ Gamma gradients - One or both gradients are None")

    if opacity_orig.grad is not None and opacity_autograd.grad is not None:
        opacity_grad_diff = torch.max(
            torch.abs(opacity_orig.grad - opacity_autograd.grad)
        )
        print(f"Opacity gradients - Maximum difference: {opacity_grad_diff.item()}")
        if opacity_grad_diff > 1e-4:
            print("❌ Opacity gradients don't match!")
            print("Original (first 3):")
            print(opacity_orig.grad[:3])
            print("Autograd (first 3):")
            print(opacity_autograd.grad[:3])
        else:
            print("✅ Opacity gradients match within tolerance.")
    else:
        print("⚠️ Opacity gradients - One or both gradients are None")

    all_pass = True
    if points_orig.grad is not None and points_autograd.grad is not None:
        all_pass = all_pass and (
            torch.max(torch.abs(points_orig.grad - points_autograd.grad)) <= 1e-4
        )
    if scaling_orig.grad is not None and scaling_autograd.grad is not None:
        all_pass = all_pass and (
            torch.max(torch.abs(scaling_orig.grad - scaling_autograd.grad)) <= 1e-4
        )
    if rotation_orig.grad is not None and rotation_autograd.grad is not None:
        all_pass = all_pass and (
            torch.max(torch.abs(rotation_orig.grad - rotation_autograd.grad)) <= 1e-4
        )
    if (
        gamma_orig.grad is not None
        and gamma_real_autograd.grad is not None
        and gamma_imag_autograd.grad is not None
    ):
        all_pass = all_pass and (
            torch.max(torch.abs(gamma_orig.grad[:, 0] - gamma_real_autograd.grad))
            <= 1e-3
        )
        all_pass = all_pass and (
            torch.max(torch.abs(gamma_orig.grad[:, 1] - gamma_imag_autograd.grad))
            <= 1e-3
        )
    if opacity_orig.grad is not None and opacity_autograd.grad is not None:
        all_pass = all_pass and (
            torch.max(torch.abs(opacity_orig.grad - opacity_autograd.grad)) <= 1e-4
        )

    print("\n" + "=" * 50)
    if all_pass:
        print("✅ SUCCESS: All gradients match within tolerance!")
    else:
        print("❌ FAILURE: Some gradients don't match. See details above.")
    print("=" * 50)

    return all_pass


run_comparison_test()

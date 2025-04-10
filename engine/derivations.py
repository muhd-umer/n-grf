# %%
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function, gradcheck

# %% [markdown]
# ## `Covariance Matrix`
#
# ### Forward Pass
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
# ### Backward Pass
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
        scaled_scaling = scale_modifier * scaling
        S = torch.diag_embed(scaled_scaling)
        ctx.save_for_backward(scaling)
        ctx.scale_modifier = scale_modifier
        return S

    @staticmethod
    def backward(ctx, grad_S):
        scaling = ctx.saved_tensors[0]
        scale_modifier = ctx.scale_modifier

        grad_S_diag = torch.diagonal(grad_S, dim1=1, dim2=2)
        grad_scaling = grad_S_diag * scale_modifier

        return grad_scaling, None


class MatrixMultiply(Function):
    @staticmethod
    def forward(ctx, A, B):
        C = torch.bmm(A, B)
        ctx.save_for_backward(A, B)
        return C

    @staticmethod
    def backward(ctx, grad_C):
        A, B = ctx.saved_tensors
        grad_A = torch.bmm(grad_C, B.transpose(1, 2))
        grad_B = torch.bmm(A.transpose(1, 2), grad_C)
        return grad_A, grad_B


class CovarianceMatrix(Function):
    @staticmethod
    def forward(ctx, RS):
        # compute covariance matrix: C = RS * RS^T.
        RSRSt = torch.bmm(RS, RS.transpose(1, 2))
        ctx.save_for_backward(RS)
        return RSRSt

    @staticmethod
    def backward(ctx, grad_RSRSt):
        (RS,) = ctx.saved_tensors
        # dL/d(RS) = (grad_RSRSt + grad_RSRSt^T) * RS
        grad_RS = torch.bmm(grad_RSRSt + grad_RSRSt.transpose(1, 2), RS)
        return grad_RS


# %% [markdown]
# ## `ProjectToChannelCoordinates`
#
# ### Forward Pass
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
# ### Backward Pass
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
        # compute displacement and distance
        d = points - receiver
        r = torch.sqrt(torch.sum(d**2, dim=1))

        # compute spherical coordinates
        longitude = torch.atan2(d[:, 1], d[:, 0])
        latitude = torch.asin(torch.clamp(d[:, 2] / r, -1.0, 1.0))

        # transform to uniform coordinates
        s_x = longitude / math.pi
        s_y = 2.0 * latitude / math.pi

        # map to channel matrix coordinates
        u = ((s_x + 1.0) / 2.0) * (num_tx - 1) + 0.5
        v = ((s_y + 1.0) / 2.0) * (num_rx - 1) + 0.5
        uv = torch.stack([u, v], dim=1)

        # save values for backward pass
        ctx.save_for_backward(points, receiver, r, d, longitude, latitude, s_x, s_y)
        ctx.num_tx = num_tx
        ctx.num_rx = num_rx

        return r, d, uv

    @staticmethod
    def backward(ctx, grad_r, grad_d, grad_uv):
        points, receiver, r, d, longitude, latitude, s_x, s_y = ctx.saved_tensors
        num_tx = ctx.num_tx
        num_rx = ctx.num_rx

        grad_points = torch.zeros_like(points)

        # gradient from distances: dr/dpoints = d/r
        for i in range(3):
            grad_points[:, i] += grad_r * d[:, i] / (r + 1e-10)

        # gradient from displacement vectors: dd/dpoints = I (identity)
        grad_points += grad_d

        # gradients from channel matrix coordinates
        grad_u = grad_uv[:, 0]
        grad_v = grad_uv[:, 1]

        # chain rule through transformations
        # s_x to u, s_y to v
        grad_s_x = grad_u * (num_tx - 1) / 2.0
        grad_s_y = grad_v * (num_rx - 1) / 2.0

        # longitude to s_x, latitude to s_y
        grad_longitude = grad_s_x / math.pi
        grad_latitude = grad_s_y * 2.0 / math.pi

        # positions to longitude and latitude
        xy_squared = d[:, 0] ** 2 + d[:, 1] ** 2 + 1e-10  # add epsilon for stability
        grad_points[:, 0] += grad_longitude * (-d[:, 1] / xy_squared)
        grad_points[:, 1] += grad_longitude * (d[:, 0] / xy_squared)

        cos_lat = torch.cos(latitude) + 1e-10
        grad_points[:, 0] += grad_latitude * (-d[:, 0] * d[:, 2]) / (r**3 * cos_lat)
        grad_points[:, 1] += grad_latitude * (-d[:, 1] * d[:, 2]) / (r**3 * cos_lat)
        grad_points[:, 2] += (
            grad_latitude * ((d[:, 0] ** 2 + d[:, 1] ** 2)) / (r**3 * cos_lat)
        )

        return grad_points, None, None, None


# %% [markdown]
# ## `ComputeJacobian`
#
# ### Forward Pass
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
# ### Backward Pass
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
        # d = [x, y, z]
        x = d[:, 0]
        y = d[:, 1]
        z = d[:, 2]

        # compute r internally
        r = torch.sqrt(torch.sum(d**2, dim=1))

        # compute x²+y² and clamp for stability
        xy_sq = x**2 + y**2
        xy_sq = torch.clamp(xy_sq, min=1e-10)
        sqrt_xy = torch.sqrt(xy_sq)

        # compute cos_lat = sqrt(1 - (z/r)²)
        cos_lat = torch.sqrt(torch.clamp(1.0 - (z / r) ** 2, min=1e-10))

        # factors from antenna grid dimensions
        tx_factor = (num_tx - 1) / (2.0 * torch.tensor(math.pi))
        rx_factor = (num_rx - 1) / torch.tensor(math.pi)

        N = d.shape[0]
        J = torch.zeros((N, 2, 3), device=d.device, dtype=d.dtype)

        # first row: u-coordinates derivatives (longitude)
        J[:, 0, 0] = tx_factor * (-y / xy_sq)
        J[:, 0, 1] = tx_factor * (x / xy_sq)
        J[:, 0, 2] = 0.0

        # second row: v-coordinates derivatives (latitude)
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
        """Backward pass for the Jacobian. Computes gradients with respect to d."""
        d, r, xy_sq, cos_lat, J = ctx.saved_tensors
        tx_factor = ctx.tx_factor
        rx_factor = ctx.rx_factor
        x = d[:, 0]
        y = d[:, 1]
        z = d[:, 2]

        # compute square root of xy_sq for stability
        sqrt_xy = torch.sqrt(torch.clamp(xy_sq, min=1e-10))
        denom2 = (xy_sq) ** 2  # (x²+y²)²

        # upstream gradients for the Jacobian entries
        L11 = grad_J[:, 0, 0]  # dJ[0,0]/d(...)
        L12 = grad_J[:, 0, 1]  # dJ[0,1]/d(...)
        # L13 is zero (since J[0,2] is 0)
        L21 = grad_J[:, 1, 0]  # dJ[1,0]/d(...)
        L22 = grad_J[:, 1, 1]  # dJ[1,1]/d(...)
        L23 = grad_J[:, 1, 2]  # dJ[1,2]/d(...)

        # compute gradients for x using derived formula
        grad_d_x = (
            2 * L11 * tx_factor * x * y * sqrt_xy
            - L12 * tx_factor * x**2 * sqrt_xy
            + L12 * tx_factor * y**2 * sqrt_xy
            - 2 * L21 * rx_factor * x**2 * z
            + L21 * rx_factor * y**2 * z
            - 3 * L22 * rx_factor * x * y * z
            - L23 * rx_factor * x * xy_sq
        ) / (sqrt_xy * denom2 + 1e-10)

        # compute gradients for y using derived formula
        grad_d_y = (
            -L11 * tx_factor * x**2 * sqrt_xy
            + L11 * tx_factor * y**2 * sqrt_xy
            - 2 * L12 * tx_factor * x * y * sqrt_xy
            - 3 * L21 * rx_factor * x * y * z
            + L22 * rx_factor * x**2 * z
            - 2 * L22 * rx_factor * y**2 * z
            - L23 * rx_factor * y * xy_sq
        ) / (sqrt_xy * denom2 + 1e-10)

        # compute gradients for z using derived formula
        grad_d_z = rx_factor * (L21 * x + L22 * y) / (torch.pow(xy_sq, 1.5) + 1e-10)

        grad_d = torch.stack([grad_d_x, grad_d_y, grad_d_z], dim=1)

        return grad_d, None, None


# %% [markdown]
# ## `ProjectCov3dToCov2d`
#
# ### Forward Pass
#
# The 2D covariance matrix is obtained by projecting the 3D covariance matrix using the Jacobian:
#
# $$\Sigma^k_{2D,j} = \mathbf{J}^k_j \cdot \Sigma^k_{3D} \cdot (\mathbf{J}^k_j)^T$$
#
# A small constant is added to the diagonal elements to ensure positive definiteness.
#
# ### Backward Pass
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
        # compute J * Σ * J^T
        temp = torch.bmm(cov3d, jacobian.transpose(1, 2))
        cov2d = torch.bmm(jacobian, temp)

        # ensure positive definiteness
        cov2d[:, 0, 0] += 0.3
        cov2d[:, 1, 1] += 0.3

        ctx.save_for_backward(cov3d, jacobian, cov2d)
        return cov2d

    @staticmethod
    def backward(ctx, grad_cov2d):
        cov3d, jacobian, cov2d = ctx.saved_tensors

        # gradient with respect to cov3d: grad_cov3d = J^T * grad_cov2d * J
        grad_cov3d = torch.bmm(
            jacobian.transpose(1, 2), torch.bmm(grad_cov2d, jacobian)
        )

        # gradient with respect to jacobian
        grad_J = torch.bmm(grad_cov2d, torch.bmm(jacobian, cov3d)) + torch.bmm(
            grad_cov2d.transpose(1, 2), torch.bmm(jacobian, cov3d)
        )

        return grad_cov3d, grad_J


# %% [markdown]
# ## `ComputeGaussianInfluence`
#
# ### Forward Pass
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
# ### Backward Pass
#
# The gradient with respect to the projected coordinates:
#
# $$\frac{\partial \sigma^k_{ij}}{\partial \mathbf{uv}^k} = -\sigma^k_{ij} \cdot (\Sigma^k_{2D,j})^{-1} \cdot (\mathbf{uv}^k - \mathbf{a}_{ij})$$
#
# The gradient with respect to the 2D covariance:
#
# $$\frac{\partial \sigma^k_{ij}}{\partial \Sigma^k_{2D,j}} = \frac{1}{2} \cdot \sigma^k_{ij} \cdot (\Sigma^k_{2D,j})^{-1} \cdot (\mathbf{uv}^k - \mathbf{a}_{ij}) \cdot (\mathbf{uv}^k - \mathbf{a}_{ij})^T \cdot (\Sigma^k_{2D,j})^{-1}$$


# %%
class ComputeGaussianInfluence(Function):
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

                # compute the Mahalanobis distance
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
        uv, cov2d, inv_cov2d, influences = ctx.saved_tensors
        num_tx = ctx.num_tx
        num_rx = ctx.num_rx
        N = uv.shape[0]

        grad_uv = torch.zeros_like(uv)
        grad_cov2d = torch.zeros_like(cov2d)

        # compute gradients for each antenna (i,j)
        for i in range(num_tx):
            for j in range(num_rx):
                antenna_pos = torch.tensor(
                    [i + 0.5, j + 0.5], device=uv.device, dtype=uv.dtype
                )
                d = uv - antenna_pos  # [N, 2]

                # gradient with respect to uv: grad_uv = grad_f * (-inv_cov2d @ d)
                grad_component = -(inv_cov2d @ d.unsqueeze(-1)).squeeze(-1)  # [N,2]

                # multiply by upstream gradient and influence value
                grad_uv = grad_uv + (
                    grad_influences[:, i, j].unsqueeze(-1)
                    * influences[:, i, j].unsqueeze(-1)
                    * grad_component
                )

                # gradient with respect to cov2d
                # d f/d cov2d = 0.5 * f * inv_cov2d @ (d d^T) @ inv_cov2d
                d_outer = d.unsqueeze(-1) * d.unsqueeze(1)  # [N, 2, 2]
                grad_cov2d = grad_cov2d + 0.5 * grad_influences[:, i, j].view(
                    N, 1, 1
                ) * influences[:, i, j].view(N, 1, 1) * (
                    inv_cov2d @ d_outer @ inv_cov2d
                )

        return grad_uv, grad_cov2d, None, None


# %% [markdown]
# ## `ComputeWirelessChannel`
#
# ### Forward Pass
#
# The wireless channel contribution of each Gaussian is computed based on physics:
#
# $$C^k_{ij} = A^k e^{j\psi^k} \cdot \frac{\lambda}{4\pi r^k_j} \cdot e^{-j2\pi r^k_j/\lambda}$$
#
# Where:
# - $A^k$ is the learned attenuation amplitude
# - $\psi^k$ is the learned phase rotation
# - $r^k_j$ is the distance from Gaussian to receiver
# - $\lambda$ is the wavelength
#
# This is separated into real and imaginary parts:
#
# $$\text{Re}(C^k_{ij}) = A^k \cdot \frac{\lambda}{4\pi r^k_j} \cdot \cos(\psi^k - 2\pi r^k_j/\lambda)$$
# $$\text{Im}(C^k_{ij}) = A^k \cdot \frac{\lambda}{4\pi r^k_j} \cdot \sin(\psi^k - 2\pi r^k_j/\lambda)$$
#
# ### Backward Pass
#
# The gradients with respect to attenuation, phase rotation, and distance:
#
# $$\frac{\partial \text{Re}(C^k_{ij})}{\partial A^k} = \frac{\lambda}{4\pi r^k_j} \cdot \cos(\psi^k - 2\pi r^k_j/\lambda)$$
# $$\frac{\partial \text{Im}(C^k_{ij})}{\partial A^k} = \frac{\lambda}{4\pi r^k_j} \cdot \sin(\psi^k - 2\pi r^k_j/\lambda)$$
#
# $$\frac{\partial \text{Re}(C^k_{ij})}{\partial \psi^k} = -A^k \cdot \frac{\lambda}{4\pi r^k_j} \cdot \sin(\psi^k - 2\pi r^k_j/\lambda)$$
# $$\frac{\partial \text{Im}(C^k_{ij})}{\partial \psi^k} = A^k \cdot \frac{\lambda}{4\pi r^k_j} \cdot \cos(\psi^k - 2\pi r^k_j/\lambda)$$
#
# $$\frac{\partial \text{Re}(C^k_{ij})}{\partial r^k_j} = -A^k \cdot \frac{\lambda}{4\pi (r^k_j)^2} \cdot \cos(\psi^k - 2\pi r^k_j/\lambda) - A^k \cdot \frac{\lambda}{4\pi r^k_j} \cdot \sin(\psi^k - 2\pi r^k_j/\lambda) \cdot \frac{-2\pi}{\lambda}$$
# $$\frac{\partial \text{Im}(C^k_{ij})}{\partial r^k_j} = -A^k \cdot \frac{\lambda}{4\pi (r^k_j)^2} \cdot \sin(\psi^k - 2\pi r^k_j/\lambda) + A^k \cdot \frac{\lambda}{4\pi r^k_j} \cdot \cos(\psi^k - 2\pi r^k_j/\lambda) \cdot \frac{-2\pi}{\lambda}$$


# %%
class ComputeWirelessChannel(Function):
    @staticmethod
    def forward(ctx, attenuation, phase_rotation, distance, wavelength):
        """Compute wireless channel contribution

        Args:
            attenuation: Learned attenuation amplitude [N, 1]
            phase_rotation: Learned phase rotation [N, 1]
            distance: Distance from Gaussian to receiver [N]
            wavelength: Signal wavelength in meters

        Returns:
            Complex channel contribution in real, imag parts [N, 2]
        """
        PI = 3.14159265358979323846
        path_loss = wavelength / (4.0 * PI * distance.unsqueeze(1))
        phase_shift = -2.0 * PI * distance.unsqueeze(1) / wavelength

        total_attenuation = attenuation * path_loss
        total_phase = phase_rotation + phase_shift

        real_part = total_attenuation * torch.cos(total_phase)
        imag_part = total_attenuation * torch.sin(total_phase)

        cat_channel = torch.cat([real_part, imag_part], dim=1)

        ctx.save_for_backward(
            attenuation,
            phase_rotation,
            distance,
            path_loss,
            total_phase,
            total_attenuation,
        )
        ctx.wavelength = wavelength

        return cat_channel

    @staticmethod
    def backward(ctx, grad_cat_channel):
        (
            attenuation,
            phase_rotation,
            distance,
            path_loss,
            total_phase,
            total_attenuation,
        ) = ctx.saved_tensors
        wavelength = ctx.wavelength
        PI = 3.14159265358979323846

        grad_real = grad_cat_channel[:, 0:1]
        grad_imag = grad_cat_channel[:, 1:2]

        # gradient wrt attenuation
        grad_attenuation = grad_real * path_loss * torch.cos(
            total_phase
        ) + grad_imag * path_loss * torch.sin(total_phase)

        # gradient wrt phase_rotation
        grad_phase_rotation = -grad_real * total_attenuation * torch.sin(
            total_phase
        ) + grad_imag * total_attenuation * torch.cos(total_phase)

        # gradient wrt distance
        grad_path_loss = -wavelength / (4.0 * PI * distance**2)
        grad_phase_shift = -2.0 * PI / wavelength

        grad_distance = grad_real * (
            grad_path_loss.unsqueeze(1) * attenuation * torch.cos(total_phase)
            - total_attenuation * torch.sin(total_phase) * grad_phase_shift
        ) + grad_imag * (
            grad_path_loss.unsqueeze(1) * attenuation * torch.sin(total_phase)
            + total_attenuation * torch.cos(total_phase) * grad_phase_shift
        )

        grad_distance = grad_distance.sum(dim=1)

        return grad_attenuation, grad_phase_rotation, grad_distance, None


# %% [markdown]
# ## `AlphaBlending`
#
# ### Forward Pass
#
# The alpha blending process forms the final channel matrix by accumulating contributions from all Gaussians:
#
# $$H_{ij} = \sum_{k \in \mathcal{K}_{ij}} (\alpha^k \cdot \sigma^k_{ij}) \cdot C^k_{ij} \prod_{l<k, l \in \mathcal{K}_{ij}} (1 - \alpha^l \cdot \sigma^l_{ij})$$
#
# Where:
# - $\mathcal{K}_{ij}$ is the set of Gaussians sorted by distance to receiver
# - $\alpha^k$ is the opacity of Gaussian $k$
# - $\sigma^k_{ij}$ is the influence of Gaussian $k$ on channel element $(i,j)$
# - $C^k_{ij}$ is the wireless channel contribution
# - The product term represents transmittance
#
# ### Backward Pass
#
# The backward pass needs to account for how each Gaussian affects both its own contribution and the transmittance for subsequent Gaussians.
#
# For each position $p$ in the sorted order:
#
# 1. Gradient with respect to contributions:
#    $$\frac{\partial L}{\partial C^p_{ij,real}} = \frac{\partial L}{\partial H_{ij,real}} \cdot T_p \cdot \alpha^p \cdot \sigma^p_{ij}$$
#    $$\frac{\partial L}{\partial C^p_{ij,imag}} = \frac{\partial L}{\partial H_{ij,imag}} \cdot T_p \cdot \alpha^p \cdot \sigma^p_{ij}$$
#
# 2. Gradient with respect to effective opacity:
#    $$\frac{\partial L}{\partial (\alpha^p \cdot \sigma^p_{ij})} = \frac{\partial L}{\partial H_{ij,real}} \cdot T_p \cdot C^p_{ij,real} + \frac{\partial L}{\partial H_{ij,imag}} \cdot T_p \cdot C^p_{ij,imag} - T_p \cdot \frac{\partial L}{\partial T_{p+1}}$$
#
# 3. Gradient with respect to opacity and influence:
#    $$\frac{\partial L}{\partial \alpha^p} = \frac{\partial L}{\partial (\alpha^p \cdot \sigma^p_{ij})} \cdot \sigma^p_{ij}$$
#    $$\frac{\partial L}{\partial \sigma^p_{ij}} = \frac{\partial L}{\partial (\alpha^p \cdot \sigma^p_{ij})} \cdot \alpha^p$$
#
# 4. Gradient with respect to transmittance:
#    $$\frac{\partial L}{\partial T_p} = \frac{\partial L}{\partial H_{ij,real}} \cdot \alpha^p \cdot \sigma^p_{ij} \cdot C^p_{ij,real} + \frac{\partial L}{\partial H_{ij,imag}} \cdot \alpha^p \cdot \sigma^p_{ij} \cdot C^p_{ij,imag} + \frac{\partial L}{\partial T_{p+1}} \cdot (1 - \alpha^p \cdot \sigma^p_{ij})$$


# %%
class AlphaBlending(Function):
    @staticmethod
    def forward(
        ctx,
        influences,
        contributions_real,
        contributions_imag,
        opacity,
        sort_indices,
        num_tx,
        num_rx,
    ):
        """Perform alpha blending to form channel matrix

        Args:
            influences: Gaussian influence on each channel element [N, num_tx, num_rx]
            contributions_real: Real part of wireless contributions [N, 1]
            contributions_imag: Imag part of wireless contributions [N, 1]
            opacity: Opacity of each Gaussian [N, 1]
            sort_indices: Indices to sort Gaussians by distance
            num_tx: Number of transmit antennas
            num_rx: Number of receive antennas

        Returns:
            Complex channel matrix [num_tx, 2*num_rx]
        """
        N = influences.shape[0]
        device = influences.device

        channel_real = torch.zeros(
            (num_tx, num_rx), device=device, dtype=influences.dtype
        )
        channel_imag = torch.zeros(
            (num_tx, num_rx), device=device, dtype=influences.dtype
        )

        # pre-allocate arrays for backward pass
        eff_opacity = torch.empty(
            (N, num_tx, num_rx), device=device, dtype=influences.dtype
        )
        transmittance = torch.empty(
            (N + 1, num_tx, num_rx), device=device, dtype=influences.dtype
        )
        transmittance[0] = torch.ones(
            (num_tx, num_rx), device=device, dtype=influences.dtype
        )

        for pos, idx in enumerate(sort_indices):
            eff_opacity[pos] = opacity[idx] * influences[idx]
            term_r = transmittance[pos] * eff_opacity[pos] * contributions_real[idx]
            term_i = transmittance[pos] * eff_opacity[pos] * contributions_imag[idx]
            channel_real = channel_real + term_r
            channel_imag = channel_imag + term_i
            transmittance[pos + 1] = transmittance[pos] * (1 - eff_opacity[pos])

        ctx.save_for_backward(
            influences,
            contributions_real,
            contributions_imag,
            opacity,
            eff_opacity,
            transmittance,
            sort_indices,
        )
        ctx.num_tx = num_tx
        ctx.num_rx = num_rx
        ctx.N = N

        return torch.cat([channel_real, channel_imag], dim=1)

    @staticmethod
    def backward(ctx, grad_cat_channel):
        (
            influences,
            contributions_real,
            contributions_imag,
            opacity,
            eff_opacity,
            transmittance,
            sort_indices,
        ) = ctx.saved_tensors
        num_tx = ctx.num_tx
        num_rx = ctx.num_rx
        N = ctx.N

        # split grad_cat_channel into real and imaginary parts
        grad_real = grad_cat_channel[:, :num_rx]
        grad_imag = grad_cat_channel[:, num_rx:]

        # initialize gradients
        grad_influences = torch.zeros_like(influences)
        grad_contrib_real = torch.zeros_like(contributions_real)
        grad_contrib_imag = torch.zeros_like(contributions_imag)
        grad_opacity = torch.zeros_like(opacity)

        # initialize transmittance gradients
        # dL/dT[N] = 0 since T[N] doesn't directly affect the output
        dLdT = [None] * (N + 1)
        dLdT[N] = torch.zeros(
            (num_tx, num_rx), device=influences.device, dtype=influences.dtype
        )

        # backward pass through layers in reverse order
        for pos in range(N - 1, -1, -1):
            idx = sort_indices[pos]

            # gradient with respect to contributions
            grad_contrib_real[idx] = torch.sum(
                grad_real * transmittance[pos] * eff_opacity[pos]
            )
            grad_contrib_imag[idx] = torch.sum(
                grad_imag * transmittance[pos] * eff_opacity[pos]
            )

            # gradient with respect to effective opacity
            # direct effect on output plus indirect effect via transmittance
            local_grad = grad_real * (
                transmittance[pos] * contributions_real[idx]
            ) + grad_imag * (transmittance[pos] * contributions_imag[idx])

            # add contribution from future transmittance
            local_grad = local_grad + (-transmittance[pos] * dLdT[pos + 1])

            # chain rule for eff_opacity = influences * opacity
            grad_opacity[idx] = grad_opacity[idx] + torch.sum(
                local_grad * influences[idx]
            )
            grad_influences[idx] = grad_influences[idx] + local_grad * opacity[idx]

            # gradient with respect to transmittance
            dLdT_current = grad_real * (
                eff_opacity[pos] * contributions_real[idx]
            ) + grad_imag * (eff_opacity[pos] * contributions_imag[idx])

            # propagate transmittance gradient backward
            # T[pos+1] = T[pos] * (1 - eff_opacity[pos])
            dLdT[pos] = dLdT_current + dLdT[pos + 1] * (1 - eff_opacity[pos])

        return (
            grad_influences,
            grad_contrib_real,
            grad_contrib_imag,
            grad_opacity,
            None,
            None,
            None,
        )


# %% [markdown]
# ## **Tests**
#
# This section tests the forward pass made via `Autograd` funcs against the one in `_torch_impl` directory.

# %%
import sys

from _torch_impl.rasterize import rasterize

sys.path.append("../")
from models.loss import mse_corr_loss, nmse_loss
from utils.transform_utils import strip_symmetric


# %%
def rasterize_channel(
    points: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    attenuation: torch.Tensor,
    phase_rotation: torch.Tensor,
    opacity: torch.Tensor,
    receiver: torch.Tensor,
    num_tx: int,
    num_rx: int,
    frequency: float,
    scale_modifier: float = 1.0,
) -> torch.Tensor:
    """Rasterize the channel matrix for a specific receiver position

    Args:
        points: Gaussian centers [N, 3]
        scaling: Scaling factors (already with exponential activation applied) [N, 3]
        rotation: Rotation quaternions (already normalized) [N, 4]
        attenuation: Learned attenuation amplitude from neural network [N, 1]
        phase_rotation: Learned phase rotation from neural network [N, 1]
        opacity: Opacity values (already with sigmoid activation applied) [N, 1]
        receiver: Receiver position [3]
        num_tx: Number of transmit antennas
        num_rx: Number of receive antennas
        frequency: Signal frequency in Hz
        scale_modifier: Global scaling modifier (default is 1.0)

    Returns:
        Channel matrix of shape [num_tx, 2*num_rx] with real and imaginary parts
        concatenated
    """
    # physical constants
    c = 299792458.0  # speed of light in m/s
    wavelength = c / frequency

    R = QuaternionToRotation.apply(rotation)
    S = ComputeScalingMatrix.apply(scaling, scale_modifier)
    RS = MatrixMultiply.apply(R, S)
    cov3d = CovarianceMatrix.apply(RS)

    distances, d, uv = ProjectToChannelCoordinates.apply(
        points, receiver, num_tx, num_rx
    )

    jacobian = ComputeJacobian.apply(d, num_tx, num_rx)
    cov2d = ProjectCov3dToCov2d.apply(cov3d, jacobian)
    sort_indices = torch.argsort(distances)
    influences = ComputeGaussianInfluence.apply(uv, cov2d, num_tx, num_rx)
    channel_components = ComputeWirelessChannel.apply(
        attenuation, phase_rotation, distances, wavelength
    )
    contributions_real = channel_components[:, 0:1]
    contributions_imag = channel_components[:, 1:2]

    cat_channel = AlphaBlending.apply(
        influences,
        contributions_real,
        contributions_imag,
        opacity,
        sort_indices,
        num_tx,
        num_rx,
    )

    return cat_channel

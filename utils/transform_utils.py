# utils/transform_utils.py

from typing import Tuple, Union

import torch


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Convert sigmoid output back to logits."""
    eps = torch.finfo(x.dtype).eps
    x = torch.clamp(x, min=eps, max=1.0 - eps)
    return torch.log(x / (1 - x))


@torch.jit.script
def build_rotation(r: torch.Tensor) -> torch.Tensor:
    """Build rotation matrices from quaternions (ensure normalization)."""
    norm = torch.sqrt(torch.sum(r * r, dim=1, keepdim=True)).clamp(min=1e-10)
    q = r / norm

    R = torch.zeros((q.size(0), 3, 3), device=r.device, dtype=r.dtype)

    qw = q[:, 0]
    qx = q[:, 1]
    qy = q[:, 2]
    qz = q[:, 3]

    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz
    qxqy = qx * qy
    qxqz = qx * qz
    qyqz = qy * qz
    qwqx = qw * qx
    qwqy = qw * qy
    qwqz = qw * qz

    R[:, 0, 0] = 1.0 - 2.0 * (qy2 + qz2)
    R[:, 0, 1] = 2.0 * (qxqy - qwqz)
    R[:, 0, 2] = 2.0 * (qxqz + qwqy)
    R[:, 1, 0] = 2.0 * (qxqy + qwqz)
    R[:, 1, 1] = 1.0 - 2.0 * (qx2 + qz2)
    R[:, 1, 2] = 2.0 * (qyqz - qwqx)
    R[:, 2, 0] = 2.0 * (qxqz - qwqy)
    R[:, 2, 1] = 2.0 * (qyqz + qwqx)
    R[:, 2, 2] = 1.0 - 2.0 * (qx2 + qy2)
    return R


@torch.jit.script
def build_scaling_rotation(s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Build combined scaling and rotation matrix L = R * S."""
    L = torch.zeros((s.shape[0], 3, 3), dtype=s.dtype, device=s.device)
    R = build_rotation(r)
    S_diag = torch.diag_embed(s)
    L = R @ S_diag
    return L


@torch.jit.script
def build_covariance_from_scaling_rotation(
    scaling: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Build covariance matrix Σ = R * S^2 * R^T."""
    R = build_rotation(rotation)
    S_sq_diag = torch.diag_embed(scaling * scaling)
    covariance = R @ S_sq_diag @ R.transpose(1, 2)
    return covariance


@torch.jit.script
def build_covariance_inverse(
    R: torch.Tensor, scaling: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Builds the inverse covariance matrix Σ^-1 = R * S^-2 * R^T.
    Handles potential division by zero in scaling.

    Args:
        R: Rotation matrices (N, 3, 3).
        scaling: Activated scaling factors (N, 3).
        eps: Small value to prevent division by zero.

    Returns:
        Inverse covariance matrices (N, 3, 3).
    """
    scaling_clamped = torch.clamp(scaling, min=eps)
    inv_scaling_sq = 1.0 / (scaling_clamped * scaling_clamped)
    S_inv_sq_diag = torch.diag_embed(inv_scaling_sq)

    inv_covariance = R @ S_inv_sq_diag @ R.transpose(1, 2)
    return inv_covariance

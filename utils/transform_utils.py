# utils/transform_utils.py

from typing import Tuple

import torch


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Convert sigmoid output back to logits

    Args:
        x: Sigmoid values in range (0, 1)

    Returns:
        Logits corresponding to the sigmoid values
    """
    return torch.log(x / (1 - x))


def strip_symmetric(sym: torch.Tensor) -> torch.Tensor:
    """Extract elements from symmetric matrix

    Args:
        sym: Symmetric matrices [N, 3, 3]

    Returns:
        Compact representation [N, 6]
    """
    return strip_lowerdiag(sym)


@torch.jit.script
def strip_lowerdiag(L: torch.Tensor) -> torch.Tensor:
    """Convert a batch of lower diagonal matrices into compact form

    Args:
        L: Matrices [N, 3, 3]

    Returns:
        Compact representation [N, 6]
    """
    N = L.shape[0]
    uncertainty = torch.zeros((N, 6), dtype=L.dtype, device=L.device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


@torch.jit.script
def build_rotation(r: torch.Tensor) -> torch.Tensor:
    """Build rotation matrices from quaternions

    Args:
        r: Quaternion tensor of shape [N, 4]

    Returns:
        Rotation matrices of shape [N, 3, 3]
    """
    norm = torch.norm(r, dim=1)

    q = r / norm.unsqueeze(1)
    R = torch.zeros((q.size(0), 3, 3), device=r.device, dtype=r.dtype)

    qw = q[:, 0]  # real part
    qx = q[:, 1]  # i
    qy = q[:, 2]  # j
    qz = q[:, 3]  # k

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
    """Build combined scaling and rotation matrix

    Args:
        s: Scaling tensor of shape [N, 3]
        r: Rotation quaternion tensor of shape [N, 4]

    Returns:
        Combined transformation matrices of shape [N, 3, 3]
    """
    N = s.shape[0]
    L = torch.zeros((N, 3, 3), dtype=s.dtype, device=s.device)
    R = build_rotation(r)

    L.diagonal(dim1=1, dim2=2)[:] = s

    return torch.bmm(R, L)


@torch.jit.script
def symmetric_matrix(compact: torch.Tensor) -> torch.Tensor:
    """Convert a batch of compact symmetric matrices to full matrices

    Args:
        compact: Compact representation of symmetric matrices [N, 6] for 3x3
                or [N, 3] for 2x2 matrices

    Returns:
        Full symmetric matrices [N, 3, 3] or [N, 2, 2]
    """
    N = compact.shape[0]
    dim = 3 if compact.shape[1] == 6 else 2

    matrix = torch.zeros((N, dim, dim), device=compact.device, dtype=compact.dtype)

    if dim == 3:
        # for 3x3 matrices with 6 parameters
        matrix[:, 0, 0] = compact[:, 0]
        matrix[:, 0, 1] = compact[:, 1]
        matrix[:, 1, 0] = compact[:, 1]
        matrix[:, 0, 2] = compact[:, 2]
        matrix[:, 2, 0] = compact[:, 2]
        matrix[:, 1, 1] = compact[:, 3]
        matrix[:, 1, 2] = compact[:, 4]
        matrix[:, 2, 1] = compact[:, 4]
        matrix[:, 2, 2] = compact[:, 5]
    else:
        # for 2x2 matrices with 3 parameters
        matrix[:, 0, 0] = compact[:, 0]
        matrix[:, 0, 1] = compact[:, 1]
        matrix[:, 1, 0] = compact[:, 1]
        matrix[:, 1, 1] = compact[:, 2]

    return matrix


@torch.jit.script
def inverse_2d_covariance(cov2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute inverse of 2D covariance matrices and areas

    Args:
        cov2d: 2D covariance matrices [N, 2, 2]

    Returns:
        Tuple containing inverse covariance matrices [N, 3] (compact) and areas
        [N, 2]
    """
    a = cov2d[:, 0, 0]
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1]

    det = a * c - b * b
    det = torch.clamp(det, min=1e-10)
    det_inv = 1.0 / det

    inv_cov2d = torch.stack(
        [
            c * det_inv,  # (0,0) component
            -b * det_inv,  # (0,1) = (1,0) component
            a * det_inv,  # (1,1) component
        ],
        dim=1,
    )

    areas = 3.0 * torch.sqrt(torch.stack([a, c], dim=1))

    return inv_cov2d, areas

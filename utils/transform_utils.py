# utils/transform_utils.py

import torch


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Convert sigmoid output back to logits"""
    return torch.log(x / (1 - x))


def strip_symmetric(sym: torch.Tensor) -> torch.Tensor:
    """Extract elements from symmetric matrix"""
    return strip_lowerdiag(sym)


def strip_lowerdiag(L: torch.Tensor) -> torch.Tensor:
    """Convert a batch of lower diagonal matrices into compact form"""
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def build_rotation(r: torch.Tensor) -> torch.Tensor:
    """Build rotation matrices from quaternions

    Args:
        r: Quaternion tensor of shape (N, 4)

    Returns:
        Rotation matrices of shape (N, 3, 3)
    """
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]  # real part
    x = q[:, 1]  # i
    y = q[:, 2]  # j
    z = q[:, 3]  # k

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Build combined scaling and rotation matrix

    Args:
        s: Scaling tensor of shape (N, 3)
        r: Rotation quaternion tensor of shape (N, 4)

    Returns:
        Combined transformation matrices of shape (N, 3, 3)
    """
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L

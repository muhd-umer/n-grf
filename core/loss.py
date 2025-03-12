# core/loss.py

import torch
import torch.nn.functional as F


def nmse_loss(
    pred: torch.Tensor, target: torch.Tensor, complex_dim: int = 1
) -> torch.Tensor:
    """Normalized MSE loss for complex wireless channel matrices.

    Args:
        pred: Predicted channel tensor
        target: Target (ground truth) channel tensor

    Returns:
        NMSE loss value
    """
    total_rx = pred.size(1)
    num_rx = total_rx // 2

    pred_reshaped = torch.complex(pred[:, :num_rx], pred[:, num_rx:])
    target_reshaped = torch.complex(target[:, :num_rx], target[:, num_rx:])

    error = pred_reshaped - target_reshaped

    # calculate power (absolute square) of error and target
    error_power = torch.abs(error) ** 2
    target_power = torch.abs(target_reshaped) ** 2

    total_error = torch.sum(error_power)
    total_power = torch.sum(target_power)

    epsilon = 1e-10
    total_power = torch.clamp(total_power, min=epsilon)

    nmse = total_error / total_power

    return nmse


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss for wireless channel matrices.

    Args:
        pred: Predicted channel tensor
        target: Target (ground truth) channel tensor

    Returns:
        L1 loss value
    """
    return torch.abs(pred - target).mean()


def ssim(
    pred: torch.Tensor, target: torch.Tensor, size_average: bool = True
) -> torch.Tensor:
    """Calculate SSIM (structural similarity) for wireless channel matrices.

    Args:
        pred: Predicted channel tensor of shape [num_tx, 2*num_rx]
        target: Target channel tensor of shape [num_tx, 2*num_rx]
        size_average: If True, the result is the mean of all windows
            If False, the result is a per-window average

    Returns:
        SSIM value (higher is better, max is 1.0)
    """
    C1 = 0.01**2
    C2 = 0.03**2

    pred_reshaped = pred.unsqueeze(0).unsqueeze(0)  # [1, 1, num_tx, 2*num_rx]
    target_reshaped = target.unsqueeze(0).unsqueeze(0)  # [1, 1, num_tx, 2*num_rx]

    mu1 = F.avg_pool2d(pred_reshaped, kernel_size=3, stride=1, padding=1)
    mu2 = F.avg_pool2d(target_reshaped, kernel_size=3, stride=1, padding=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.avg_pool2d(pred_reshaped * pred_reshaped, kernel_size=3, stride=1, padding=1)
        - mu1_sq
    )
    sigma2_sq = (
        F.avg_pool2d(
            target_reshaped * target_reshaped, kernel_size=3, stride=1, padding=1
        )
        - mu2_sq
    )
    sigma12 = (
        F.avg_pool2d(
            pred_reshaped * target_reshaped, kernel_size=3, stride=1, padding=1
        )
        - mu1_mu2
    )

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def l1_ssim_loss(
    pred: torch.Tensor, target: torch.Tensor, lambda_dssim: float = 0.2
) -> torch.Tensor:
    """Combined L1 and SSIM loss for wireless channel matrices.

    Args:
        pred: Predicted channel tensor
        target: Target (ground truth) channel tensor
        lambda_dssim: Weight for DSSIM component (0.0 means pure L1 loss)

    Returns:
        Combined loss value
    """
    l1 = l1_loss(pred, target)

    # convert to DSSIM: 1-SSIM, so lower is better
    ssim_value = ssim(pred, target)
    dssim = 1.0 - ssim_value

    return (1.0 - lambda_dssim) * l1 + lambda_dssim * dssim

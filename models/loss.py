# models/loss.py

import torch
import torch.nn.functional as F


def nmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Normalized Mean Square Error loss for complex channel matrices.

    Args:
        pred: Predicted channel tensor
        target: Target (ground truth) channel tensor

    Returns:
        NMSE loss value
    """
    mse = torch.sum(torch.abs(pred - target) ** 2)
    normalization = torch.sum(torch.abs(target) ** 2)
    normalization = torch.clamp(normalization, min=1e-10)

    return mse / normalization


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss for channel matrices.

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
    """Calculate SSIM (structural similarity) for channel matrices.

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
    """Combined L1 and SSIM loss for channel matrices.

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


def complex_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Complex MSE loss for MIMO channels.

    This function converts real-valued tensors to complex representation
    and computes MSE in the complex domain. The input tensors are assumed
    to have their real parts in the first half of the feature dimension
    and imaginary parts in the second half.

    Args:
        pred: Predicted channel tensor with shape [..., 2*num_rx]
        target: Target (ground truth) channel tensor with shape [..., 2*num_rx]

    Returns:
        Complex domain MSE loss value (lower is better)
    """
    num_rx = pred.shape[1] // 2
    pred_complex = torch.complex(pred[:, :num_rx], pred[:, num_rx:])
    target_complex = torch.complex(target[:, :num_rx], target[:, num_rx:])
    return torch.mean(torch.abs(pred_complex - target_complex) ** 2)


def channel_corr_loss(
    pred: torch.Tensor, target: torch.Tensor, eps=1e-8
) -> torch.Tensor:
    """Channel correlation loss for wireless MIMO channels.

    Measures how well the predicted channel correlates with the target channel
    by computing normalized correlation coefficient. The function first
    converts real-valued tensors to complex representation, then computes
    normalized correlation.

    Args:
        pred: Predicted channel tensor with shape [..., 2*num_rx]
        target: Target channel tensor with shape [..., 2*num_rx]
        eps: Small value to prevent division by zero during normalization

    Returns:
        Correlation loss value (1-correlation, so lower is better)
    """
    num_rx = pred.shape[1] // 2
    pred_complex = torch.complex(pred[:, :num_rx], pred[:, num_rx:])
    target_complex = torch.complex(target[:, :num_rx], target[:, num_rx:])

    pred_flat = pred_complex.view(-1)
    target_flat = target_complex.view(-1)
    pred_norm = pred_flat / (torch.norm(pred_flat) + eps)
    target_norm = target_flat / (torch.norm(target_flat) + eps)

    correlation = torch.abs(torch.sum(pred_norm * torch.conj(target_norm)))
    return 1.0 - correlation


def mse_corr_loss(
    pred: torch.Tensor, target: torch.Tensor, lambda_mse=0.5, lambda_corr=0.5
) -> torch.Tensor:
    """Combined channel loss for MIMO channel estimation.

    This loss combines complex MSE and correlation metrics for more
    effective channel estimation. The weighting between MSE and
    correlation components can be adjusted through lambda parameters.

    Args:
        pred: Predicted channel tensor with shape [..., 2*num_rx]
        target: Target channel tensor with shape [..., 2*num_rx]
        lambda_mse: Weight for the MSE component (default: 0.5)
        lambda_corr: Weight for the correlation component (default: 0.5)

    Returns:
        Combined loss value (lower is better)
    """
    mse = complex_mse_loss(pred, target)
    corr = channel_corr_loss(pred, target)
    return lambda_mse * mse + lambda_corr * corr

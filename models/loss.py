# models/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_nmse(pred, target, eps=1e-8):
    """Calculate NMSE between predicted and target complex channel matrices"""
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    target = target.to(pred.dtype)

    N_r = pred.shape[2] // 2 if pred.dim() == 3 else pred.shape[1] // 2

    if pred.dim() == 3:
        pred_real, pred_imag = pred[..., :N_r], pred[..., N_r:]
        target_real, target_imag = target[..., :N_r], target[..., N_r:]
    else:
        pred_real, pred_imag = pred[:, :N_r], pred[:, N_r:]
        target_real, target_imag = target[:, :N_r], target[:, N_r:]

    pred_complex = torch.complex(pred_real, pred_imag)
    target_complex = torch.complex(target_real, target_imag)

    diff_sq = torch.sum(torch.abs(pred_complex - target_complex) ** 2)
    target_sq = torch.sum(torch.abs(target_complex) ** 2).clamp(min=eps)
    nmse = diff_sq / target_sq

    return nmse


class NormalizedMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return calculate_nmse(pred, target, self.eps)


class LogMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        target = target.to(pred.dtype)

        N_r = pred.shape[2] // 2
        pred_real, pred_imag = pred[..., :N_r], pred[..., N_r:]
        targ_real, targ_imag = target[..., :N_r], target[..., N_r:]

        eps = torch.tensor(self.eps, dtype=pred.dtype, device=pred.device)

        pred_mag = torch.sqrt(pred_real**2 + pred_imag**2 + eps)
        targ_mag = torch.sqrt(targ_real**2 + targ_imag**2 + eps)
        pred_log = torch.log(pred_mag + eps)
        targ_log = torch.log(targ_mag + eps)
        return F.mse_loss(pred_log, targ_log)


class PolarMSELoss(nn.Module):
    def __init__(self, phase_weight=1.0, eps=1e-8):
        super().__init__()
        self.phase_weight = phase_weight
        self.eps = eps

    def forward(self, pred, target):
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        target = target.to(pred.dtype)

        N_r = pred.shape[2] // 2
        pr, pj = pred[..., :N_r], pred[..., N_r:]
        tr, tj = target[..., :N_r], target[..., N_r:]

        eps = torch.tensor(self.eps, dtype=pred.dtype, device=pred.device)

        pred_mag = torch.sqrt(pr**2 + pj**2 + eps)
        true_mag = torch.sqrt(tr**2 + tj**2 + eps)
        mag_loss = F.mse_loss(pred_mag, true_mag)

        pred_phase = torch.atan2(pj, pr)
        true_phase = torch.atan2(tj, tr)

        pi = torch.tensor(torch.pi, dtype=pred.dtype, device=pred.device)
        phase_diff = torch.remainder(pred_phase - true_phase + pi, 2 * pi) - pi
        phase_loss = (phase_diff**2).mean()

        return mag_loss + self.phase_weight * phase_loss


class CosineSimilarityLoss(nn.Module):
    def forward(self, pred, target):
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        target = target.to(pred.dtype)

        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        cos = F.cosine_similarity(pred_flat, target_flat, dim=1)
        return (1 - cos).mean()


def complex_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num_rx = pred.shape[1] // 2
    pred_complex = torch.complex(pred[:, :num_rx], pred[:, num_rx:])
    target_complex = torch.complex(target[:, :num_rx], target[:, num_rx:])
    return torch.mean(torch.abs(pred_complex - target_complex) ** 2)


def channel_corr_loss(
    pred: torch.Tensor, target: torch.Tensor, eps=1e-8
) -> torch.Tensor:
    num_rx = pred.shape[1] // 2
    pred_complex = torch.complex(pred[:, :num_rx], pred[:, num_rx:])
    target_complex = torch.complex(target[:, :num_rx], target[:, num_rx:])

    pred_flat = pred_complex.view(-1)
    target_flat = target_complex.view(-1)
    pred_norm = pred_flat / (torch.norm(pred_flat) + eps)
    target_norm = target_flat / (torch.norm(target_flat) + eps)

    correlation = torch.abs(torch.sum(pred_norm * torch.conj(target_norm)))
    return 1.0 - correlation


class MSECorrLoss(nn.Module):
    def __init__(self, lambda_mse=0.5, lambda_corr=0.5):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_corr = lambda_corr

    def forward(self, pred, target):
        mse = complex_mse_loss(pred, target)
        corr = channel_corr_loss(pred, target)
        return self.lambda_mse * mse + self.lambda_corr * corr


class LogMagPhaseLoss(nn.Module):
    def __init__(self, phase_weight=1.0, eps=1e-10):
        super().__init__()
        self.phase_weight = phase_weight
        self.eps = eps
        self.log_mse_loss = LogMSELoss(eps=eps)

    def forward(self, pred, target):
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        target = target.to(pred.dtype)
        eps = torch.tensor(self.eps, dtype=pred.dtype, device=pred.device)
        pi = torch.tensor(torch.pi, dtype=pred.dtype, device=pred.device)

        N_r = pred.shape[2] // 2
        pr, pj = pred[..., :N_r], pred[..., N_r:]
        tr, tj = target[..., :N_r], target[..., N_r:]

        mag_loss = self.log_mse_loss(pred, target)

        pred_phase = torch.atan2(pj, pr + eps)
        true_phase = torch.atan2(tj, tr + eps)

        phase_diff = pred_phase - true_phase
        phase_diff = torch.remainder(phase_diff + pi, 2 * pi) - pi
        phase_loss = (phase_diff**2).mean()

        total_loss = mag_loss + self.phase_weight * phase_loss
        return total_loss


def get_loss_function(loss_type, **kwargs):
    if loss_type == "nmse":
        eps = kwargs.get("eps", 1e-8)
        return NormalizedMSELoss(eps=eps)
    elif loss_type == "log_mse":
        eps = kwargs.get("eps", 1e-8)
        return LogMSELoss(eps=eps)
    elif loss_type == "polar_mse":
        phase_weight = kwargs.get("phase_weight", 1.0)
        eps = kwargs.get("eps", 1e-8)
        return PolarMSELoss(phase_weight=phase_weight, eps=eps)
    elif loss_type == "cosine":
        return CosineSimilarityLoss()
    elif loss_type == "mse_corr":
        lambda_mse = kwargs.get("lambda_mse", 0.5)
        lambda_corr = kwargs.get("lambda_corr", 0.5)
        return MSECorrLoss(lambda_mse=lambda_mse, lambda_corr=lambda_corr)
    elif loss_type == "log_mag_phase":
        phase_weight = kwargs.get("phase_weight", 1.0)
        eps = kwargs.get("eps", 1e-10)
        return LogMagPhaseLoss(phase_weight=phase_weight, eps=eps)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def calculate_snr(nmse):
    """Calculate SNR in dB from NMSE loss value"""
    return -10.0 * torch.log10(nmse)

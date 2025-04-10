import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizedMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        target = target.to(pred.dtype)

        eps = torch.tensor(self.eps, dtype=pred.dtype, device=pred.device)

        diff_sq = ((pred - target) ** 2).sum(dim=(1, 2))
        target_sq = ((target**2).sum(dim=(1, 2))).clamp(min=eps)
        nmse = diff_sq / target_sq
        return nmse.mean()


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        target = target.to(pred.dtype)

        eps = torch.tensor(self.eps, dtype=pred.dtype, device=pred.device)

        diff = pred - target
        errors = torch.sqrt(diff * diff + eps**2)
        return errors.mean()


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


def get_loss_function(loss_type, **kwargs):
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "l1":
        return nn.L1Loss()
    elif loss_type == "nmse":
        eps = kwargs.get("eps", 1e-8)
        return NormalizedMSELoss(eps=eps)
    elif loss_type == "charbonnier":
        eps = kwargs.get("eps", 1e-3)
        return CharbonnierLoss(eps=eps)
    elif loss_type == "log_mse":
        eps = kwargs.get("eps", 1e-8)
        return LogMSELoss(eps=eps)
    elif loss_type == "polar_mse":
        phase_weight = kwargs.get("phase_weight", 1.0)
        eps = kwargs.get("eps", 1e-8)
        return PolarMSELoss(phase_weight=phase_weight, eps=eps)
    elif loss_type == "cosine":
        return CosineSimilarityLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

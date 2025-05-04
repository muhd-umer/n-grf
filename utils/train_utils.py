# utils/train_utils.py

import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch


def get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1000000,
) -> Callable[[int], float]:
    """Computes learning rate following exponential decay."""

    def helper(step: int) -> float:
        if step < 0 or max_steps <= 0:
            return 0.0
        if lr_init == 0.0 and lr_final == 0.0:
            return 0.0

        if lr_delay_steps > 0:
            # apply delay multiplier during delay phase
            delay_factor = lr_delay_mult + (1.0 - lr_delay_mult) * np.sin(
                0.5 * np.pi * min(step / lr_delay_steps, 1.0)
            )
        else:
            delay_factor = 1.0

        # calculate exponential decay progress
        # ensure progress doesn't exceed 1.0 even if step > max_steps
        progress = min(step / max_steps, 1.0)

        if lr_init <= 0:
            log_lr_init = -np.inf
        else:
            log_lr_init = np.log(lr_init)

        if lr_final <= 0:
            log_lr_final = -np.inf
        else:
            log_lr_final = np.log(lr_final)

        # linear interpolation in log space = exponential decay in linear space
        log_lerped_lr = log_lr_init * (1.0 - progress) + log_lr_final * progress
        lerped_lr = np.exp(log_lerped_lr)

        return delay_factor * lerped_lr

    return helper


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    root_logger = logging.getLogger()
    # clear existing handlers to avoid duplicate logs if called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)  # get logger for this module
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def compute_grad_stats(
    model: torch.nn.Module, norm_type: float = 2.0
) -> Dict[str, float]:
    """Computes statistics about gradients for model parameters."""
    total_norm = 0.0
    total_abs_sum = 0.0
    total_elements = 0
    min_grad = float("inf")
    max_grad = float("-inf")
    param_count_with_grad = 0
    params_without_grad = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(
                    f"Warning: NaN or Inf detected in gradients for parameter '{name}'. Skipping stats for this param."
                )
                continue

            param_norm = param.grad.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
            total_abs_sum += param.grad.abs().sum().item()
            total_elements += param.numel()
            param_count_with_grad += 1

            current_min = param.grad.min().item()
            current_max = param.grad.max().item()
            min_grad = min(min_grad, current_min)
            max_grad = max(max_grad, current_max)
        elif param.requires_grad:  # only list params that *should* have grads
            params_without_grad.append(name)

    # Log parameters that should have gradients but don't
    if params_without_grad:
        print(
            f"Warning: Parameters requiring grad but missing gradients: {params_without_grad}"
        )

    total_norm = total_norm ** (1.0 / norm_type) if total_norm > 0 else 0.0
    mean_abs_grad = total_abs_sum / total_elements if total_elements > 0 else 0.0

    if min_grad == float("inf"):
        min_grad = 0.0
    if max_grad == float("-inf"):
        max_grad = 0.0

    return {
        "grad_norm": total_norm,
        "mean_abs_grad": mean_abs_grad,
        "min_grad": min_grad,
        "max_grad": max_grad,
        "param_count_with_grad": param_count_with_grad,
    }

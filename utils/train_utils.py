# utils/train_utils.py

import logging
from pathlib import Path

import numpy as np


def get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1000000,
):
    """Get exponential learning rate decay function

    Args:
        lr_init: Initial learning rate
        lr_final: Final learning rate
        lr_delay_steps: Steps before starting decay
        lr_delay_mult: Multiplier for delay period
        max_steps: Total number of steps

    Returns:
        Callable that computes learning rate for given step
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration

    Args:
        log_dir: Directory to save log file

    Returns:
        Configured logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / "train.log"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)

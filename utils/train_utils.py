# utils/train_utils.py

import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np


@lru_cache(maxsize=128)
def get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1000000,
) -> Callable[[int], float]:
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
    log_lr_init = np.log(lr_init) if lr_init > 0 else -np.inf
    log_lr_final = np.log(lr_final) if lr_final > 0 else -np.inf
    half_pi = 0.5 * np.pi

    def helper(step: int) -> float:
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0

        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1.0 - lr_delay_mult) * np.sin(
                half_pi * min(step / lr_delay_steps, 1.0)
            )
        else:
            delay_rate = 1.0

        t = min(step / max_steps, 1.0)
        log_lerp = np.exp(log_lr_init * (1.0 - t) + log_lr_final * t)

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    logger = logging.getLogger(__name__)

    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    return logger

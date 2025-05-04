# utils/general_utils.py

import random
from typing import Optional

import numpy as np
import torch


def set_random_seed(seed: Optional[int] = None) -> None:
    """Set random seeds for reproducibility."""

    if seed is not None:
        print(f"Setting random seed to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

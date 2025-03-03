# utils/general_utils.py
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


def set_random_seed(seed: Optional[int] = None) -> None:
    """Set random seeds for reproducibility

    Args:
        seed: Random seed to use. If None, seeds are not set.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    state_dict: Dict[str, Any], path: Union[str, Path], iteration: int
) -> None:
    """Save model checkpoint

    Args:
        state_dict: Model state to save
        path: Path to save directory
        iteration: Current iteration number
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    save_path = path / f"checkpoint_{iteration:07d}.pt"
    torch.save(state_dict, save_path)

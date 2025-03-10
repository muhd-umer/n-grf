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
    state_dict: Dict[str, Any],
    path: Union[str, Path],
    filename_or_iter: Union[str, int],
) -> None:
    """Save model checkpoint

    Args:
        state_dict: Model state to save
        path: Path to save directory
        filename_or_iter: Either a filename string or iteration number
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if isinstance(filename_or_iter, int):
        save_path = path / f"checkpoint_{filename_or_iter:07d}.pt"
    else:
        save_path = path / filename_or_iter

    torch.save(state_dict, save_path)

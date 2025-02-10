# models/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Base class for all models"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_output(self, rx_position):
        """Forward pass of the model"""
        pass

    @abstractmethod
    def training_setup(self, training_args):
        """Setup training configurations"""
        pass

    def get_state_dict(self) -> Dict[str, Any]:
        """Get state dict for saving"""
        return {
            "model_state": self.state_dict(),
            "config": self.config if hasattr(self, "config") else None,
        }

    def load_state_dict_from_save(self, state_dict: Dict[str, Any]):
        """Load from saved state dict"""
        if "config" in state_dict and hasattr(self, "config"):
            self.config = state_dict["config"]
        self.load_state_dict(state_dict["model_state"])

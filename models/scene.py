# models/scene.py

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from datasets.dataloader import get_wireless_dataloader

from .gaussian import GaussianModel

logger = logging.getLogger(__name__)


class Scene:
    """Scene management class for wireless channel reconstruction.

    Manages data loading and model interfacing.
    """

    def __init__(
        self,
        gaussians: GaussianModel,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        train_ratio: float = 0.8,
        seed: Optional[int] = None,
    ):
        """Initialize scene with data loading setup

        Args:
            gaussians: Gaussian model instance
            data_path: Path to dataset
            batch_size: Batch size for data loading
            num_workers: Number of workers for data loading
            train_ratio: Train/test split ratio
            seed: Random seed
        """
        self.gaussians = gaussians
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        logger.info(f"Loading dataset from {self.data_path}")

        self.train_dataloader = get_wireless_dataloader(
            str(self.data_path),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            train=True,
            train_ratio=train_ratio,
            seed=seed,
        )

        self.test_dataloader = get_wireless_dataloader(
            str(self.data_path),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            train=False,
            train_ratio=train_ratio,
            seed=seed,
        )

        try:
            batch = next(iter(self.train_dataloader))
            self.tx_position = batch["tx_position"]
            self.env_dims = batch["env_dims"]

            self.init_gaussians(batch["point_cloud"])
            logger.info("Successfully initialized scene and Gaussians")

        except Exception as e:
            logger.error(f"Failed to initialize scene: {str(e)}")
            raise

    def init_gaussians(self, point_cloud: torch.Tensor):
        """Initialize Gaussian model from point cloud"""
        self.gaussians.init_from_pcd(point_cloud)

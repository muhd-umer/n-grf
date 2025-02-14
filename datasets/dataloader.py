# datasets/dataloader.py
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .wireless_dataset import WirelessDataset


def collate_wireless_batch(batch: list) -> Dict[str, Any]:
    """Collate function for wireless dataset batches.

    Args:
        batch (list): List of dataset items to be collated

    Returns:
        Dict[str, Any]: Collated batch with stacked tensors
    """
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        if key == "aoa":
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = torch.stack([item[key] for item in batch])
    return collated


def get_wireless_dataloader(
    data_path: str,
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle: bool = True,
    num_pc: Optional[int] = None,
    train: bool = True,
    train_ratio: float = 0.8,
    seed: Optional[int] = None,
    drop_last: bool = False,
) -> DataLoader:
    """Create a DataLoader for the wireless dataset.

    Args:
        data_path (str): Path to the dataset file
        batch_size (int): Number of samples per batch
        num_workers (int): Number of workers for data loading
        shuffle (bool): Whether to shuffle the data
        num_pc (int, optional): Number of point cloud points to sample
        train (bool): Whether to load training or test set
        train_ratio (float): Ratio of data to use for training
        seed (int, optional): Random seed for train/test split
        drop_last (bool): Whether to drop the last incomplete batch

    Returns:
        DataLoader: The configured data loader
    """
    dataset = WirelessDataset(
        data_path,
        num_pc=num_pc,
        train=train,
        train_ratio=train_ratio,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_wireless_batch,
        pin_memory=True,
        drop_last=drop_last,
    )

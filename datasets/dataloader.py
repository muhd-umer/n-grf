# datasets/dataloader.py

from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .wireless_dataset import WirelessDataset


def collate_wireless_batch(batch: list) -> Dict[str, Any]:
    """Collate function for wireless dataset batches.

    Args:
        batch (list): List of dataset items ({'rx_position': tensor, 'channel_matrix': tensor, 'index': int}).

    Returns:
        Collated batch with stacked tensors.
    """
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], int):
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = [item[key] for item in batch]

    return collated


def get_wireless_dataloader(
    data_path: str,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    train: bool = True,
    train_ratio: float = 0.8,
    seed: Optional[int] = None,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for wireless dataset.

    Args:
        data_path (str): Path to the dataset file.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of workers for data loading.
        shuffle (bool): Whether to shuffle the data (typically True for train).
        train (bool): Whether to load training or test set.
        train_ratio (float): Ratio of data to use for training.
        seed (int, optional): Random seed for train/test split.
        drop_last (bool): Whether to drop the last incomplete batch (typically True for train).
        pin_memory (bool): Whether to use pinned memory for faster GPU transfer.

    Returns:
        The configured data loader.
    """
    dataset = WirelessDataset(
        data_path,
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
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def get_dataloaders(
    data_path: str,
    batch_size: int = 1,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    seed: Optional[int] = None,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create training and validation DataLoaders for wireless dataset.

    Args:
        data_path (str): Path to the dataset file.
        batch_size (int): Number of samples per batch for training loader.
        num_workers (int): Number of workers for data loading.
        train_ratio (float): Ratio of data to use for training.
        seed (int, optional): Random seed for train/test split.
        pin_memory (bool): Use pinned memory.

    Returns:
        A tuple of (training loader, validation loader, dataset metadata).
        Validation loader always uses batch_size=1 and shuffle=False.
    """
    train_loader = get_wireless_dataloader(
        data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        train=True,
        train_ratio=train_ratio,
        seed=seed,
        drop_last=True,
        pin_memory=pin_memory,
    )

    val_loader = get_wireless_dataloader(
        data_path,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        train=False,
        train_ratio=train_ratio,
        seed=seed,
        drop_last=False,
        pin_memory=pin_memory,
    )

    metadata = train_loader.dataset.get_metadata()

    return train_loader, val_loader, metadata

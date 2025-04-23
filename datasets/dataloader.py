# datasets/dataloader.py
import os
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .wireless_dataset import WirelessDataset


def collate_wireless_batch(batch: list) -> Dict[str, Any]:
    """Collate function for wireless dataset batches.

    Args:
        batch (list): List of dataset items to be collated

    Returns:
        Collated batch with stacked tensors
    """
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        if key in ["aoa", "aod", "path_loss_per_ray"]:
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = torch.stack([item[key] for item in batch])
    return collated


def get_wireless_dataloader(
    data_path: str,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    train: bool = True,
    train_ratio: float = 0.9,
    seed: Optional[int] = None,
    drop_last: bool = False,
    subcarrier_idx: Optional[int] = None,
    normalize: bool = True,
    normalize_method: str = "stacked",
    stats_file: Optional[str] = None,
) -> DataLoader:
    """Create a DataLoader for the wireless dataset.

    Args:
        data_path (str): Path to the dataset file
        batch_size (int): Number of samples per batch
        num_workers (int): Number of workers for data loading
        shuffle (bool): Whether to shuffle the data
        train (bool): Whether to load training or test set
        train_ratio (float): Ratio of data to use for training
        seed (int, optional): Random seed for train/test split
        drop_last (bool): Whether to drop the last incomplete batch
        subcarrier_idx (int, optional): Index of subcarrier to use for multi-carrier data.
            If None, uses middle subcarrier or extracts from filename for single-carrier.
        normalize (bool): Whether to normalize channel matrices
        normalize_method (str): Method for normalization ('separate' or 'stacked')
        stats_file (str, optional): Path to save/load normalization statistics

    Returns:
        The configured data loader
    """
    dataset = WirelessDataset(
        data_path,
        train=train,
        train_ratio=train_ratio,
        seed=seed,
        subcarrier_idx=subcarrier_idx,
        normalize=normalize,
        normalize_method=normalize_method,
        stats_file=stats_file,
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


def get_dataloaders(
    data_path: str,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    train_ratio: float = 0.9,
    seed: Optional[int] = None,
    drop_last: bool = False,
    subcarrier_idx: Optional[int] = None,
    normalize: bool = True,
    normalize_method: str = "stacked",
    stats_file: Optional[str] = None,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders for the wireless dataset.

    Args:
        data_path (str): Path to the dataset file
        batch_size (int): Number of samples per batch
        num_workers (int): Number of workers for data loading
        shuffle (bool): Whether to shuffle the data
        train_ratio (float): Ratio of data to use for training
        seed (int, optional): Random seed for train/test split
        drop_last (bool): Whether to drop the last incomplete batch
        subcarrier_idx (int, optional): Index of subcarrier to use for multi-carrier data.
            If None, uses middle subcarrier or extracts from filename for single-carrier.
        normalize (bool): Whether to normalize channel matrices
        normalize_method (str): Method for normalization ('separate' or 'stacked')
        stats_file (str, optional): Path to save/load normalization statistics

    Returns:
        A tuple of training and validation DataLoaders
    """
    if normalize and stats_file is None:
        data_dir = os.path.dirname(data_path)
        data_name = os.path.basename(data_path).split(".")[0]
        stats_file = os.path.join(data_dir, f"{data_name}_norm_stats.json")
        os.remove(stats_file) if os.path.exists(stats_file) else None

    train_loader = get_wireless_dataloader(
        data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        train=True,
        train_ratio=train_ratio,
        seed=seed,
        drop_last=drop_last,
        subcarrier_idx=subcarrier_idx,
        normalize=normalize,
        normalize_method=normalize_method,
        stats_file=stats_file,
    )

    val_loader = get_wireless_dataloader(
        data_path,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        train=False,
        train_ratio=train_ratio,
        seed=seed,
        drop_last=drop_last,
        subcarrier_idx=subcarrier_idx,
        normalize=normalize,
        normalize_method=normalize_method,
        stats_file=stats_file,
    )

    return train_loader, val_loader

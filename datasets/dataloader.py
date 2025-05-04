# datasets/dataloader.py

from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .channel_dataset import ChannelDataset


def collate_batch(batch: list) -> Dict[str, Any]:
    """Collate function for wireless dataset batches.

    Args:
        batch (list): List of dataset items ({'rx_position': tensor,
        'channel': tensor, 'index': int})

    Returns:
        Collated batch with stacked tensors and list of indices
    """
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], (int, float, str)):
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
    norm_eps: float = 1e-8,
) -> DataLoader:
    """Create a DataLoader for the wireless magnitude dataset.

    Args:
        data_path (str): Path to the dataset file
        batch_size (int): Number of samples per batch
        num_workers (int): Number of workers for data loading
        shuffle (bool): Whether to shuffle the data
        train (bool): Whether to load training or test set
        train_ratio (float): Ratio of data to use for training
        seed (int, optional): Random seed for train/test split
        drop_last (bool): Whether to drop the last incomplete batch
        pin_memory (bool): Whether to use pinned memory for faster GPU transfer
        norm_eps (float): Epsilon for dataset normalization stability

    Returns:
        Configured data loader
    """
    dataset = ChannelDataset(
        data_path,
        train=train,
        train_ratio=train_ratio,
        seed=seed,
        norm_eps=norm_eps,
    )

    if len(dataset) == 0:
        print(
            f"Warning: DataLoader created for an empty dataset ({'train' if train else 'test'} split)."
        )

        return DataLoader(dataset, batch_size=batch_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
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
    norm_eps: float = 1e-8,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create training and validation DataLoaders for the channel dataset.

    Args:
        data_path (str): Path to the dataset file
        batch_size (int): Number of samples per batch for training loader
        num_workers (int): Number of workers for data loading
        train_ratio (float): Ratio of data to use for training
        seed (int, optional): Random seed for train/test split
        pin_memory (bool): Use pinned memory
        norm_eps (float): Epsilon for dataset normalization stability

    Returns:
        A tuple of (training loader, validation loader, dataset metadata)
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
        norm_eps=norm_eps,
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
        norm_eps=norm_eps,
    )

    metadata = {}
    if (
        hasattr(train_loader, "dataset")
        and train_loader.dataset is not None
        and len(train_loader.dataset) > 0
    ):
        metadata = train_loader.dataset.get_metadata()
    elif (
        hasattr(val_loader, "dataset")
        and val_loader.dataset is not None
        and len(val_loader.dataset) > 0
    ):
        metadata = val_loader.dataset.get_metadata()
        print(
            "Warning: Using metadata from validation dataset as training dataset might be empty."
        )
    else:
        print(
            "Warning: Could not retrieve metadata as both train and val datasets seem unavailable or empty."
        )

        metadata = {
            "num_tx_ant": 0,
            "num_rx_ant": 0,
            "frequency": 0,
            "wavelength": 0,
            "is_siso": False,
            "tx_position": torch.zeros(3),
            "env_dims": None,
            "point_cloud": None,
            "min_magnitude": 0.0,
            "max_magnitude": 1.0,
            "norm_eps": norm_eps,
        }

    return train_loader, val_loader, metadata

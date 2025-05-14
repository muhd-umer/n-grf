# datasets/dataloader.py

import warnings
from typing import Any, Dict, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .channel_dataset import ChannelDataset


def collate_batch(batch: list) -> Dict[str, Any]:
    """Collate function for wireless dataset batches.

    Args:
        batch (list): List of dataset items ({'rx_position': tensor, 'channel':
        tensor, 'index': int})

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


def _get_channel_loader(
    cfg: DictConfig,
    train: bool,
    shuffle: bool,
    batch_size: int,
    drop_last: bool,
    data_path: Optional[str] = None,
) -> DataLoader:
    """Create a DataLoader for the wireless complex channel dataset using config.

    Args:
        cfg (DictConfig): Configuration object.
        train (bool): Whether to load training or test set.
        shuffle (bool): Whether to shuffle the data.
        batch_size (int): Batch size (can differ from cfg.training.batch_size for eval).
        drop_last (bool): Whether to drop the last incomplete batch.
        data_path (Optional[str]): Override for data path if needed.

    Returns:
        Configured data loader
    """
    path = data_path if data_path is not None else cfg.data.path
    if path is None:
        raise ValueError("Data path must be specified in config or passed as argument.")

    dataset = ChannelDataset(
        data_path=path,
        train=train,
        train_ratio=cfg.data.train_ratio,
        seed=cfg.experiment.seed,
        norm_eps=cfg.data.norm_eps,
        normalize=cfg.data.normalize,
    )
    if len(dataset) == 0:
        warnings.warn(
            f"Warning: DataLoader created for an empty dataset ({'train' if train else 'test'} split)."
        )
        return DataLoader(dataset, batch_size=batch_size)

    pin_memory = cfg.experiment.device == "cuda" and torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_batch,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def get_dataloaders(
    cfg: DictConfig,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create training and validation DataLoaders for the channel dataset using config.

    Args:
        cfg (DictConfig): Configuration object containing data and training settings.

    Returns:
        A tuple of (training loader, validation loader, dataset metadata)
    """
    train_loader = _get_channel_loader(
        cfg=cfg,
        train=True,
        shuffle=True,
        batch_size=cfg.training.batch_size,
        drop_last=True,
    )
    val_loader = _get_channel_loader(
        cfg=cfg,
        train=False,
        shuffle=False,
        batch_size=cfg.evaluation.batch_size,
        drop_last=False,
    )

    metadata = {}

    if (
        hasattr(train_loader, "dataset")
        and train_loader.dataset is not None
        and hasattr(train_loader.dataset, "get_metadata")
        and len(train_loader.dataset) > 0
    ):
        metadata = train_loader.dataset.get_metadata()
    elif (
        hasattr(val_loader, "dataset")
        and val_loader.dataset is not None
        and hasattr(val_loader.dataset, "get_metadata")
        and len(val_loader.dataset) > 0
    ):
        metadata = val_loader.dataset.get_metadata()
        warnings.warn(
            "Warning: Using metadata from validation dataset as training dataset might be empty or lack get_metadata."
        )
    else:
        warnings.warn(
            "Warning: Could not retrieve metadata as both train and val datasets seem unavailable, empty, or lack get_metadata."
        )

        metadata = {
            "num_tx_ant": cfg.model.get("num_tx_ant", 0),
            "num_rx_ant": cfg.model.get("num_rx_ant", 0),
            "frequency": 0,
            "wavelength": 0,
            "is_siso": False,
            "tx_position": torch.zeros(3),
            "env_dims": None,
            "point_cloud": None,
            "min_real": 0.0,
            "max_real": 1.0,
            "min_imag": 0.0,
            "max_imag": 1.0,
            "norm_eps": cfg.data.norm_eps,
            "normalize": cfg.data.normalize,
        }

        if "model" in cfg and "num_tx_ant" in cfg.model and "num_rx_ant" in cfg.model:
            metadata["num_tx_ant"] = cfg.model.num_tx_ant
            metadata["num_rx_ant"] = cfg.model.num_rx_ant
            metadata["is_siso"] = (
                cfg.model.num_tx_ant == 1 and cfg.model.num_rx_ant == 1
            )

    if "num_tx_ant" not in metadata or metadata["num_tx_ant"] == 0:
        if "model" in cfg and "num_tx_ant" in cfg.model:
            metadata["num_tx_ant"] = cfg.model.num_tx_ant
    if "num_rx_ant" not in metadata or metadata["num_rx_ant"] == 0:
        if "model" in cfg and "num_rx_ant" in cfg.model:
            metadata["num_rx_ant"] = cfg.model.num_rx_ant

    if "normalize" not in metadata:
        metadata["normalize"] = cfg.data.normalize

    return train_loader, val_loader, metadata

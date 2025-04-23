# datasets/wireless_dataset.py

import json
import os
import re
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from pymatreader import read_mat
from torch.utils.data import Dataset, random_split


class WirelessDataset(Dataset):
    """A dataset class for MIMO channel data.

    This dataset handles loading and processing of wireless channel data from
    .mat files, including point cloud, channel matrix, path loss, angles of
    arrival, receiver positions, and other relevant information.

    Args:
        data_path (str): Path to the .mat dataset file
        num_pc (int, optional): Number of point cloud points to randomly sample
        train (bool, optional): If True, returns training set, else test set
        train_ratio (float, optional): Ratio of data to use for training (default: 0.8)
        seed (int, optional): Random seed for train/test split and point cloud sampling
        subcarrier_idx (int, optional): Index of the subcarrier to use (default: None)
        normalize (bool, optional): Whether to normalize channel matrices (default: True)
        normalize_method (str, optional): How to normalize complex values ('separate' or 'stacked')
        stats_file (str, optional): Path to save/load normalization statistics (default: None)
    """

    def __init__(
        self,
        data_path: str,
        train: bool = True,
        train_ratio: float = 0.8,
        seed: Optional[int] = None,
        subcarrier_idx: Optional[int] = None,
        normalize: bool = True,
        normalize_method: str = "stacked",
        stats_file: Optional[str] = None,
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.seed = seed if seed is not None else 42
        self.normalize = normalize
        self.normalize_method = normalize_method
        self.train = train
        self.stats_file = stats_file

        if self.normalize_method not in ["separate", "stacked"]:
            raise ValueError(
                f"Invalid normalize_method: {normalize_method}. "
                f"Choose from 'separate' or 'stacked'."
            )

        if subcarrier_idx is None:
            filename = self.data_path.stem
            sc_match = re.search(r"_sc(\d+)", filename)
            if sc_match:
                self.subcarrier_idx = int(sc_match.group(1))
            else:
                self.subcarrier_idx = None
        else:
            self.subcarrier_idx = subcarrier_idx

        # set seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # load mat73 file
        data = read_mat(str(self.data_path))["dataset"]
        self._process_data(data)

        # perform train/test split
        total_size = len(self.rx_positions)
        train_size = int(total_size * train_ratio)
        test_size = total_size - train_size

        generator = torch.Generator().manual_seed(self.seed)
        train_dataset, test_dataset = random_split(
            range(total_size), [train_size, test_size], generator=generator
        )

        self.train_indices = list(train_dataset)
        self.test_indices = list(test_dataset)
        self.indices = self.train_indices if train else self.test_indices

        if self.normalize:
            if self.stats_file and os.path.exists(self.stats_file):
                self._load_normalization_stats()
            else:
                self._compute_normalization_stats()
                if self.stats_file:
                    self._save_normalization_stats()

    def _process_data(self, data):
        self.point_cloud = torch.from_numpy(data["environment"]["point_cloud"]).float()

        # handle complex channel matrix
        H = torch.from_numpy(
            data["channel"]["H"]
        )  # shape: [num_users, tx_ant, rx_ant, num_sc] if multiple subcarriers
        # [num_users, tx_ant, rx_ant] if single subcarrier

        if len(H.shape) == 3:  # single subcarrier
            self.channel_matrix = H
        elif len(H.shape) == 4:  # multiple subcarriers case
            if self.subcarrier_idx is None:
                self.subcarrier_idx = H.shape[-1] // 2
            elif self.subcarrier_idx >= H.shape[-1]:
                raise ValueError(
                    f"Subcarrier index {self.subcarrier_idx} out of range [0, {H.shape[-1]-1}]"
                )

            H_selected = H[..., self.subcarrier_idx]
            self.channel_matrix = H_selected
        else:
            raise ValueError(
                f"Invalid channel matrix shape. Expected 3 or 4, got {len(H.shape)}"
            )

        self.tx_position = torch.from_numpy(data["nodes"]["ap_position"]).float()
        self.rx_positions = torch.from_numpy(data["nodes"]["users_positions"].T).float()
        self.path_loss = torch.from_numpy(data["channel"]["path_loss"]).float()

        def process_pl_per_ray(pl_list):
            return [torch.from_numpy(np.array(d)).float() for d in pl_list]

        self.path_loss_per_ray = process_pl_per_ray(
            data["channel"]["path_loss_per_ray"]
        )

        def process_angle_data(angle_list):
            return [torch.from_numpy(np.array(d)).float() for d in angle_list]

        self.aod = process_angle_data(data["channel"]["AoD"])
        self.aoa = process_angle_data(data["channel"]["AoA"])
        self.env_dims = torch.from_numpy(data["environment"]["dimensions"]).float()

        self._store_config(data["config"])
        self._store_channel_info(data["channel"])

    def _store_config(self, config):
        self.num_tx_ant = int(config["tx_antennas"])
        self.num_rx_ant = int(config["rx_antennas"])
        self.frequency = float(config["frequency"])
        self.wavelength = float(config["wavelength"])
        self.num_users = int(config["num_users"])

    def _store_channel_info(self, channel):
        self.frequencies = channel["frequencies"]
        self.ray_steps = channel["ray_steps"]
        self.ray_points = channel["ray_points"]
        self.ray_interactions = channel["ray_interactions"]
        self.ray_coefficients = channel["ray_coefficients"]

    def _compute_normalization_stats(self):
        """Compute normalization statistics from training set."""
        train_channels = [self.channel_matrix[idx] for idx in self.train_indices]
        stkd_channels = torch.stack(train_channels)  # stacked channels

        if self.normalize_method == "separate":
            self.real_mean = stkd_channels.real.mean().item()
            self.imag_mean = stkd_channels.imag.mean().item()
            self.real_std = stkd_channels.real.std().item()
            self.imag_std = stkd_channels.imag.std().item()

            self.real_std = self.real_std
            self.imag_std = self.imag_std

        else:
            real_imag_stacked = torch.cat(
                [stkd_channels.real, stkd_channels.imag], dim=-1
            )
            self.mean = real_imag_stacked.mean().item()
            self.std = real_imag_stacked.std().item()

            self.std = self.std

    def _save_normalization_stats(self):
        """Save normalization statistics to a file."""
        stats_dir = os.path.dirname(self.stats_file)
        if stats_dir and not os.path.exists(stats_dir):
            os.makedirs(stats_dir)

        if self.normalize_method == "separate":
            stats = {
                "normalization_method": "separate",
                "real_mean": self.real_mean,
                "real_std": self.real_std,
                "imag_mean": self.imag_mean,
                "imag_std": self.imag_std,
            }
        else:
            stats = {
                "normalization_method": "stacked",
                "mean": self.mean,
                "std": self.std,
            }

        with open(self.stats_file, "w") as f:
            json.dump(stats, f)

    def _load_normalization_stats(self):
        """Load normalization statistics from a file."""
        with open(self.stats_file, "r") as f:
            stats = json.load(f)

        if stats["normalization_method"] != self.normalize_method:
            warnings.warn(
                f"Adapting normalization method: Loaded stats use {stats['normalization_method']} normalization, "
                f"but {self.normalize_method} was requested. Using {stats['normalization_method']} instead."
            )
            self.normalize_method = stats["normalization_method"]

        if self.normalize_method == "separate":
            self.real_mean = stats["real_mean"]
            self.real_std = stats["real_std"]
            self.imag_mean = stats["imag_mean"]
            self.imag_std = stats["imag_std"]
        else:
            self.mean = stats["mean"]
            self.std = stats["std"]

    def normalize_channel(self, channel):
        """Normalize a channel matrix using the precomputed statistics."""
        if not self.normalize:
            return channel

        if self.normalize_method == "separate":
            real_part = (channel.real - self.real_mean) / self.real_std
            imag_part = (channel.imag - self.imag_mean) / self.imag_std
            return torch.complex(real_part, imag_part)
        else:
            return channel

    def get_normalization_stats(self) -> Dict[str, float]:
        """Get the normalization statistics as a dictionary."""
        if not self.normalize:
            return {}

        if self.normalize_method == "separate":
            return {
                "real_mean": self.real_mean,
                "real_std": self.real_std,
                "imag_mean": self.imag_mean,
                "imag_std": self.imag_std,
                "method": "separate",
            }
        else:
            return {"mean": self.mean, "std": self.std, "method": "stacked"}

    def get_tx_position(self) -> torch.Tensor:
        """Get transmitter position."""
        if not hasattr(self, "tx_position"):
            raise RuntimeError("Dataset not initialized - tx_position not available")
        return self.tx_position

    def get_env_dims(self) -> torch.Tensor:
        """Get environment dimensions."""
        if not hasattr(self, "env_dims"):
            raise RuntimeError("Dataset not initialized - env_dims not available")
        return self.env_dims

    def get_point_cloud(
        self, num_points: Optional[int] = None, seed: Optional[int] = None
    ) -> torch.Tensor:
        """Get point cloud, optionally sampled.

        Args:
            num_points: Number of points to sample. If None, returns full point cloud
            seed: Random seed for sampling. If None, uses dataset seed
        """
        if not hasattr(self, "point_cloud"):
            raise RuntimeError("Dataset not initialized - point cloud not available")

        if num_points is None or num_points >= len(self.point_cloud):
            return self.point_cloud

        rng = np.random.RandomState(seed if seed is not None else self.seed)
        pc_indices = rng.choice(len(self.point_cloud), num_points, replace=False)
        return self.point_cloud[pc_indices]

    @property
    def num_subcarriers(self) -> Optional[int]:
        """Return the number of subcarriers in the dataset if multi-carrier, None otherwise."""
        if hasattr(self, "frequencies"):
            return len(self.frequencies) if self.frequencies is not None else None
        return None

    @property
    def current_subcarrier_idx(self) -> Optional[int]:
        """Return the currently selected subcarrier index."""
        return self.subcarrier_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        current_idx = self.indices[idx]

        channel_matrix = self.channel_matrix[current_idx]

        if self.normalize:
            if self.normalize_method == "separate":
                channel_matrix = self.normalize_channel(channel_matrix)
                channel_matrix_stkd = torch.hstack(
                    (channel_matrix.real, channel_matrix.imag)
                )
            else:
                channel_matrix_stkd = torch.hstack(
                    (channel_matrix.real, channel_matrix.imag)
                )
                channel_matrix_stkd = (channel_matrix_stkd - self.mean) / self.std
        else:
            channel_matrix_stkd = torch.hstack(
                (channel_matrix.real, channel_matrix.imag)
            )

        aoa = self.aoa[current_idx]
        aod = self.aod[current_idx]

        if aoa.dim() < 2:
            aoa = (
                aoa.view(2, -1)
                if aoa.numel() >= 2
                else torch.zeros(2, 0, device=aoa.device)
            )

        if aod.dim() < 2:
            aod = (
                aod.view(2, -1)
                if aod.numel() >= 2
                else torch.zeros(2, 0, device=aod.device)
            )

        path_loss_per_ray = self.path_loss_per_ray[current_idx]

        if path_loss_per_ray.dim() == 0:
            path_loss_per_ray = torch.zeros(
                aoa.shape[1], device=path_loss_per_ray.device
            )

        return {
            "rx_position": self.rx_positions[current_idx],
            "channel_matrix": channel_matrix_stkd,
            "channel_matrix_clx": channel_matrix,
            "aod": aod,
            "aoa": aoa,
            "path_loss_per_ray": path_loss_per_ray,
            "path_loss": self.path_loss[current_idx],
        }

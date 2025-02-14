# datasets/wireless_dataset.py

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from pymatreader import read_mat
from torch.utils.data import Dataset, random_split


class WirelessDataset(Dataset):
    """A dataset class for wireless channel data.

    This dataset handles loading and processing of wireless channel data
    including point clouds, transmitter/receiver positions, and channel
    matrices.

    Args:
        data_path (str): Path to the .mat dataset file
        num_pc (int, optional): Number of point cloud points to randomly sample
        train (bool, optional): If True, returns training set, else test set
        train_ratio (float, optional): Ratio of data to use for training (default: 0.8)
        seed (int, optional): Random seed for train/test split and point cloud sampling
    """

    def __init__(
        self,
        data_path: str,
        num_pc: Optional[int] = None,
        train: bool = True,
        train_ratio: float = 0.8,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.num_pc = num_pc
        self.seed = seed if seed is not None else 42

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

        self.indices = train_dataset if train else test_dataset

    def _process_data(self, data):
        self.point_cloud = torch.from_numpy(data["environment"]["point_cloud"]).float()

        # handle complex channel matrix
        H = torch.from_numpy(
            data["channel"]["H"]
        )  # shape: [num_users, tx_ant, rx_ant, num_sc] if multiple subcarriers
        # [num_users, tx_ant, rx_ant] if single subcarrier

        if len(H.shape) == 3:  # single subcarrier
            self.channel_matrix = torch.stack([H.real, H.imag], dim=-1).float()
        elif len(H.shape) == 4:  # multiple subcarriers case
            mid_subcarrier = H.shape[-1] // 2
            H_mid = H[..., mid_subcarrier]
            self.channel_matrix = torch.stack([H_mid.real, H_mid.imag], dim=-1).float()
        else:
            raise ValueError(
                f"Invalid channel matrix shape. Expected 3 or 4, got {len(H.shape)}"
            )

        self.tx_position = torch.from_numpy(data["nodes"]["ap_position"]).float()
        self.rx_positions = torch.from_numpy(data["nodes"]["users_positions"].T).float()

        def process_angle_data(angle_list):
            return [torch.from_numpy(np.array(d)).float() for d in angle_list]

        # authors: AoD is unneeded as it is primarily related to the txsite
        # self.aod = process_angle_data(data["channel"]["AoD"])
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

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]

        return {
            "rx_position": self.rx_positions[actual_idx],
            "channel_matrix": self.channel_matrix[actual_idx],
            # "aod": self.aod[actual_idx],
            "aoa": self.aoa[actual_idx],
        }

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

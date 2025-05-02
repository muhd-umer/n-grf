# datasets/wireless_dataset.py
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from pymatreader import read_mat
from torch.utils.data import Dataset, random_split


class WirelessDataset(Dataset):
    """A dataset class for MIMO/SISO channel data.

    Handles loading and processing of wireless channel data from .mat files,
    including channel matrix, positions, and config. Point cloud and env_dims
    are loaded if present but not used initialization. Infers SISO/MIMO from
    dataset config unless explicitly overridden during loading (not typical).

    Args:
        data_path (str): Path to the .mat dataset file
        train (bool, optional): If True, returns training set, else test set
        train_ratio (float, optional): Ratio of data for training (default: 0.8)
        seed (int, optional): Random seed for train/test split
    """

    def __init__(
        self,
        data_path: str,
        train: bool = True,
        train_ratio: float = 0.8,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.seed = seed if seed is not None else 42
        generator = torch.Generator().manual_seed(self.seed)
        np.random.seed(self.seed)

        print(f"Loading dataset from: {self.data_path}")
        try:
            mat_data = read_mat(str(self.data_path))
            if "dataset" not in mat_data:
                raise KeyError("Loaded .mat file does not contain 'dataset' key.")
            data = mat_data["dataset"]
        except Exception as e:
            print(f"Error loading MAT file: {e}")
            raise

        print("Processing dataset...")
        self._process_data(data)
        print("Dataset processing complete.")

        total_size = self.num_users
        if total_size == 0:
            raise ValueError("Dataset contains no users/samples.")

        train_size = int(total_size * train_ratio)
        test_size = total_size - train_size

        if train_size == 0 or test_size == 0:
            print(
                f"Warning: train_ratio {train_ratio} resulted in zero samples for train/test split. Adjusting."
            )
            if total_size >= 2:
                train_size = max(1, train_size)
                test_size = total_size - train_size
            else:
                train_size = 1 if train else 0
                test_size = 1 - train_size

        self.indices = list(range(total_size))
        train_indices, test_indices = random_split(
            self.indices, [train_size, test_size], generator=generator
        )

        self.active_indices = train_indices if train else test_indices
        print(f"{'Training' if train else 'Test'} set size: {len(self.active_indices)}")

    def _process_data(self, data):
        """Extracts and processes data from the loaded dictionary."""
        self._store_config(data["config"])

        if "environment" in data and "point_cloud" in data["environment"]:
            self.point_cloud_data = torch.from_numpy(
                data["environment"]["point_cloud"]
            ).float()

        else:
            self.point_cloud_data = None

        if "environment" in data and "dimensions" in data["environment"]:
            self.env_dims = torch.from_numpy(data["environment"]["dimensions"]).float()

        else:
            self.env_dims = None

        if "nodes" in data and "ap_position" in data["nodes"]:
            tx_pos_raw = data["nodes"]["ap_position"]
            self.tx_position = torch.from_numpy(np.array(tx_pos_raw)).float().squeeze()
            if self.tx_position.shape != (3,):
                raise ValueError(
                    f"Unexpected transmitter position shape: {tx_pos_raw.shape}"
                )

        else:
            raise ValueError(
                "Transmitter position ('ap_position') not found in dataset."
            )

        if "nodes" in data and "users_positions" in data["nodes"]:
            rx_pos_raw = data["nodes"]["users_positions"]
            if rx_pos_raw.shape[0] != 3:
                raise ValueError(
                    f"Expected receiver positions shape (3, K), got {rx_pos_raw.shape}"
                )
            self.rx_positions = torch.from_numpy(rx_pos_raw.T).float()

            num_users_from_pos = self.rx_positions.shape[0]
            if num_users_from_pos != self.num_users:
                print(
                    f"Warning: num_users mismatch. Config: {self.num_users}, Rx Positions: {num_users_from_pos}. Using {num_users_from_pos}."
                )
                self.num_users = num_users_from_pos
        else:
            raise ValueError(
                "Receiver positions ('users_positions') not found in dataset."
            )

        if "channel" in data and "H" in data["channel"]:
            H_raw = data["channel"]["H"]
            if isinstance(H_raw, dict) and "real" in H_raw and "imag" in H_raw:
                H_real = np.array(H_raw["real"])
                H_imag = np.array(H_raw["imag"])
                if H_real.dtype.kind not in "iufc" or H_imag.dtype.kind not in "iufc":
                    raise TypeError("Real/Imag parts of H are not numeric.")
                H_complex = H_real + 1j * H_imag
                H_tensor = torch.from_numpy(H_complex).to(torch.complex64)
            elif isinstance(H_raw, np.ndarray) and np.iscomplexobj(H_raw):
                H_tensor = torch.from_numpy(H_raw).to(torch.complex64)
            else:
                raise TypeError(
                    f"Unsupported format for H: {type(H_raw)}. Expected complex numpy array or dict with 'real'/'imag'."
                )

            print(f"Raw H tensor shape from MAT: {H_tensor.shape}")
            expected_leading_dim = self.num_users
            target_shape = (expected_leading_dim, self.num_tx_ant, self.num_rx_ant)

            if H_tensor.dim() == 4:
                num_sc = H_tensor.shape[-1]
                sc_idx = num_sc // 2
                print(
                    f"Multiple subcarriers ({num_sc}) detected. Selecting middle subcarrier index {sc_idx}."
                )
                H_selected = H_tensor[:expected_leading_dim, :, :, sc_idx]
                if H_selected.shape == target_shape:
                    self.channel_matrix = H_selected
                elif H_selected.numel() == np.prod(target_shape):
                    print(
                        f"Warning: Attempting flexible reshape of H (from 4D) from {H_selected.shape} to {target_shape}"
                    )
                    self.channel_matrix = H_selected.reshape(target_shape)
                else:
                    raise ValueError(
                        f"Channel matrix shape mismatch (from 4D). Expected {target_shape} or compatible, got {H_selected.shape}."
                    )

            elif H_tensor.dim() == 3:
                H_selected = H_tensor[:expected_leading_dim, :, :]
                if H_selected.shape == target_shape:
                    self.channel_matrix = H_selected
                elif H_selected.numel() == np.prod(target_shape):
                    print(
                        f"Warning: Attempting flexible reshape of H (from 3D) from {H_selected.shape} to {target_shape}"
                    )
                    self.channel_matrix = H_selected.reshape(target_shape)
                else:
                    raise ValueError(
                        f"Channel matrix shape mismatch (from 3D). Expected {target_shape} or compatible, got {H_selected.shape}."
                    )

            elif H_tensor.dim() == 1:
                if self.is_siso:
                    H_selected = H_tensor[:expected_leading_dim]
                    if H_selected.shape[0] == expected_leading_dim:
                        print("Reshaping SISO H from (N_user,) to (N_user, 1, 1)")
                        self.channel_matrix = H_selected.reshape(target_shape)
                    else:
                        raise ValueError(
                            f"Channel matrix shape mismatch (from 1D). Expected ({expected_leading_dim},), got {H_selected.shape}."
                        )
                else:
                    raise ValueError(
                        f"H tensor has dimension 1, but config is not SISO (Nt={self.num_tx_ant}, Nr={self.num_rx_ant})."
                    )

            elif H_tensor.dim() == 2:
                if self.is_siso and H_tensor.shape[1] == 1:
                    H_selected = H_tensor[:expected_leading_dim, :]
                    print("Reshaping SISO H from (N_user, 1) to (N_user, 1, 1)")
                    self.channel_matrix = H_selected.reshape(target_shape)
                elif H_tensor.numel() == np.prod(target_shape):
                    print(
                        f"Warning: H tensor has dimension 2 ({H_tensor.shape}). Attempting flexible reshape to {target_shape}"
                    )

                    self.channel_matrix = H_tensor[:expected_leading_dim, :].reshape(
                        target_shape
                    )
                else:
                    raise ValueError(
                        f"H tensor has dimension 2 ({H_tensor.shape}), but it's not compatible with SISO or MIMO target shape {target_shape}."
                    )

            else:
                raise ValueError(
                    f"Unexpected H tensor dimensions: {H_tensor.dim()}. Expected 1, 2, 3, or 4."
                )

            print(f"Processed channel matrix shape: {self.channel_matrix.shape}")

        else:
            raise ValueError("Channel matrix ('H') not found in dataset.")

    def _store_config(self, config):
        """Stores configuration values from the dataset and infers SISO."""
        try:
            self.num_tx_ant = int(config["tx_antennas"])
            self.num_rx_ant = int(config["rx_antennas"])
            self.frequency = float(config["frequency"])
            self.wavelength = float(config["wavelength"])
            self.num_users = int(config["num_users"])
            self.config_use_siso = config.get("use_siso", False)

            self.is_siso = (
                self.num_tx_ant == 1 and self.num_rx_ant == 1
            ) or self.config_use_siso
            if self.is_siso:

                self.num_tx_ant = 1
                self.num_rx_ant = 1

            print(
                f"Dataset Config: Nt={self.num_tx_ant}, Nr={self.num_rx_ant}, Freq={self.frequency/1e9:.2f}GHz, Lambda={self.wavelength:.4f}m, NumUsers={self.num_users}, IsSISO={self.is_siso}"
            )

        except KeyError as e:
            print(f"Error: Missing key in dataset config: {e}")
            raise
        except ValueError as e:
            print(f"Error: Invalid value type in dataset config: {e}")
            raise

    def get_point_cloud(self) -> Optional[torch.Tensor]:
        """Get the loaded point cloud data if available."""
        return self.point_cloud_data

    def get_env_dims(self) -> Optional[torch.Tensor]:
        """Get environment dimensions if available."""
        return self.env_dims

    def get_tx_position(self) -> torch.Tensor:
        """Get the transmitter position."""
        return self.tx_position

    def get_metadata(self) -> dict:
        """Returns essential metadata."""
        return {
            "num_tx_ant": self.num_tx_ant,
            "num_rx_ant": self.num_rx_ant,
            "frequency": self.frequency,
            "wavelength": self.wavelength,
            "is_siso": self.is_siso,
            "tx_position": self.tx_position,
            "env_dims": self.env_dims,
        }

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, idx):
        """Retrieves a single sample for the active split."""
        original_idx = self.active_indices[idx]
        return {
            "rx_position": self.rx_positions[original_idx],
            "channel_matrix": self.channel_matrix[original_idx],
            "index": original_idx,
        }

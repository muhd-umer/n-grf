# datasets/wireless_dataset.py

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pymatreader import read_mat
from torch.utils.data import Dataset, random_split


class ChannelDataset(Dataset):
    """A dataset class for MIMO/SISO channel data.

    Handles loading and processing of wireless channel data from .mat files,
    extracting channel magnitude (CMR), and applying min-max normalization across
    the entire dataset. Point cloud and env_dims are loaded if present.

    Args:
        data_path (str): Path to the .mat dataset file
        train (bool, optional): If True, returns training set, else test set
        train_ratio (float, optional): Ratio of data for training (default: 0.8)
        seed (int, optional): Random seed for train/test split
        norm_eps (float, optional): Epsilon for normalization denominator stability
    """

    def __init__(
        self,
        data_path: str,
        train: bool = True,
        train_ratio: float = 0.8,
        seed: Optional[int] = None,
        norm_eps: float = 1e-8,
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.seed = seed if seed is not None else 42
        self.norm_eps = norm_eps
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
            warnings.warn(
                f"Warning: train_ratio {train_ratio} resulted in zero samples for train/test split. Adjusting."
            )
            if total_size >= 2:
                train_size = max(1, train_size)
                test_size = total_size - train_size
            else:
                train_size = 1 if train else 0
                test_size = 1 - train_size
            print(f"Adjusted split: Train={train_size}, Test={test_size}")

        self.indices = list(range(total_size))
        train_indices_dataset, test_indices_dataset = random_split(
            range(total_size), [train_size, test_size], generator=generator
        )
        self.train_indices = train_indices_dataset.indices
        self.test_indices = test_indices_dataset.indices

        self.active_indices = self.train_indices if train else self.test_indices
        print(f"{'Training' if train else 'Test'} set size: {len(self.active_indices)}")
        if not self.active_indices:
            warnings.warn(f"Warning: {'Training' if train else 'Test'} set is empty!")

    def _process_data(self, data):
        """Extracts, processes, and normalizes data from the loaded dictionary."""
        self._store_config(data["config"])
        self.point_cloud_data = None
        self.env_dims = None
        if "environment" in data:
            if "point_cloud" in data["environment"] and isinstance(
                data["environment"]["point_cloud"], np.ndarray
            ):
                self.point_cloud_data = torch.from_numpy(
                    data["environment"]["point_cloud"]
                ).float()
            if "dimensions" in data["environment"] and isinstance(
                data["environment"]["dimensions"], np.ndarray
            ):
                self.env_dims = torch.from_numpy(
                    data["environment"]["dimensions"]
                ).float()

        if "nodes" in data and "ap_position" in data["nodes"]:
            tx_pos_raw = data["nodes"]["ap_position"]
            self.tx_position = torch.from_numpy(np.array(tx_pos_raw)).float().squeeze()
            if self.tx_position.shape != (3,):
                raise ValueError(
                    f"Unexpected transmitter position shape: {self.tx_position.shape}, expected (3,)"
                )
        else:
            raise ValueError(
                "Transmitter position ('ap_position') not found in dataset."
            )

        if "nodes" in data and "users_positions" in data["nodes"]:
            rx_pos_raw = data["nodes"]["users_positions"]
            if (
                isinstance(rx_pos_raw, np.ndarray)
                and rx_pos_raw.ndim == 2
                and rx_pos_raw.shape[0] == 3
            ):
                self.rx_positions = torch.from_numpy(rx_pos_raw.T).float()
            else:
                raise ValueError(
                    f"Expected receiver positions shape (3, K), got {rx_pos_raw.shape if isinstance(rx_pos_raw, np.ndarray) else type(rx_pos_raw)}"
                )

            num_users_from_pos = self.rx_positions.shape[0]
            if num_users_from_pos != self.num_users:
                warnings.warn(
                    f"Warning: num_users mismatch. Config: {self.num_users}, Rx Positions: {num_users_from_pos}. Using {num_users_from_pos}."
                )
                self.num_users = num_users_from_pos
        else:
            raise ValueError(
                "Receiver positions ('users_positions') not found in dataset."
            )

        if "channel" in data and "H" in data["channel"]:
            channel_raw_data = data["channel"]["H"]
            if (
                isinstance(channel_raw_data, dict)
                and "real" in channel_raw_data
                and "imag" in channel_raw_data
            ):
                channel_real = np.array(channel_raw_data["real"])
                channel_imag = np.array(channel_raw_data["imag"])
                if (
                    channel_real.dtype.kind not in "iufc"
                    or channel_imag.dtype.kind not in "iufc"
                ):
                    raise TypeError("Real/Imag parts of H are not numeric.")

                channel_complex = channel_real.astype(
                    np.float32
                ) + 1j * channel_imag.astype(np.float32)
                channel_tensor = torch.from_numpy(channel_complex).to(torch.complex64)
            elif isinstance(channel_raw_data, np.ndarray) and np.iscomplexobj(
                channel_raw_data
            ):
                channel_tensor = torch.from_numpy(channel_raw_data).to(torch.complex64)
            elif (
                isinstance(channel_raw_data, np.ndarray)
                and channel_raw_data.dtype.kind in "iuf"
            ):
                warnings.warn(
                    "Warning: Loaded H is real-valued. Assuming it represents magnitude."
                )
                channel_tensor = torch.from_numpy(channel_raw_data).float()
            else:
                raise TypeError(
                    f"Unsupported format for H: {type(channel_raw_data)}. Expected complex numpy array, real numpy array, or dict with 'real'/'imag'."
                )

            print(f"Raw channel tensor shape from MAT: {channel_tensor.shape}")

            expected_leading_dim = self.num_users
            target_shape = (expected_leading_dim, self.num_tx_ant, self.num_rx_ant)

            if channel_tensor.dim() == 4:
                num_sc = channel_tensor.shape[-1]

                sc_idx = num_sc // 2
                print(
                    f"Multiple subcarriers ({num_sc}) detected. Selecting middle subcarrier index {sc_idx}."
                )
                channel_selected = channel_tensor[:expected_leading_dim, :, :, sc_idx]

            elif channel_tensor.dim() == 3:
                channel_selected = channel_tensor[:expected_leading_dim, :, :]

            elif channel_tensor.dim() == 1:
                if self.is_siso:
                    channel_selected = (
                        channel_tensor[:expected_leading_dim]
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                else:
                    raise ValueError(
                        f"Channel tensor has dim 1, but config is MIMO (Nt={self.num_tx_ant}, Nr={self.num_rx_ant})."
                    )

            elif channel_tensor.dim() == 2:
                if self.is_siso and channel_tensor.shape[1] == 1:
                    channel_selected = channel_tensor[
                        :expected_leading_dim, :
                    ].unsqueeze(-1)
                elif (
                    channel_tensor.shape[0] == expected_leading_dim
                    and channel_tensor.shape[1] == self.num_tx_ant * self.num_rx_ant
                ):
                    warnings.warn(
                        f"Warning: Channel tensor has shape {channel_tensor.shape}. Assuming flattened MIMO and reshaping."
                    )
                    channel_selected = channel_tensor[:expected_leading_dim, :].view(
                        expected_leading_dim, self.num_tx_ant, self.num_rx_ant
                    )
                else:
                    raise ValueError(
                        f"Ambiguous channel tensor shape {channel_tensor.shape} for MIMO/SISO config."
                    )
            else:
                raise ValueError(
                    f"Unexpected channel tensor dimensions: {channel_tensor.dim()}. Expected 1, 2, 3, or 4."
                )

            if channel_selected.shape != target_shape:
                raise ValueError(
                    f"Processed channel tensor shape {channel_selected.shape} does not match target shape {target_shape}."
                )

            print(f"Processed channel tensor shape: {channel_selected.shape}")

            if torch.is_complex(channel_selected):
                self.cmr_raw = torch.abs(channel_selected).float()
            else:
                self.cmr_raw = channel_selected.float()

            print(f"Raw magnitude tensor shape: {self.cmr_raw.shape}")

            self.min_magnitude = torch.min(self.cmr_raw)
            self.max_magnitude = torch.max(self.cmr_raw)
            print(
                f"Magnitude range (min/max): {self.min_magnitude:.4e} / {self.max_magnitude:.4e}"
            )

            magnitude_range = self.max_magnitude - self.min_magnitude
            if magnitude_range < self.norm_eps:
                warnings.warn(
                    f"Warning: Magnitude range is very small ({magnitude_range:.2e}). Setting normalized magnitude to 0.5."
                )
                self.cmr_normalized = torch.full_like(self.cmr_raw, 0.5)
            else:
                self.cmr_normalized = (self.cmr_raw - self.min_magnitude) / (
                    magnitude_range + self.norm_eps
                )

                self.cmr_normalized = torch.clamp(self.cmr_normalized, 0.0, 1.0)

            print(f"Normalized magnitude tensor shape: {self.cmr_normalized.shape}")
            print(
                f"Normalized magnitude range (min/max): {torch.min(self.cmr_normalized):.4f} / {torch.max(self.cmr_normalized):.4f}"
            )

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
                f"Dataset Config: Nt={self.num_tx_ant}, Nr={self.num_rx_ant}, Freq={self.frequency/1e9:.2f}GHz, Lambda={self.wavelength:.3f}m, NumUsers={self.num_users}, IsSISO={self.is_siso}"
            )

        except KeyError as e:
            print(f"Error: Missing key in dataset config: {e}")
            raise
        except (ValueError, TypeError) as e:
            print(f"Error: Invalid value or type in dataset config: {e}")
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
        """Returns essential metadata including normalization parameters."""
        return {
            "num_tx_ant": self.num_tx_ant,
            "num_rx_ant": self.num_rx_ant,
            "frequency": self.frequency,
            "wavelength": self.wavelength,
            "is_siso": self.is_siso,
            "tx_position": self.tx_position,
            "env_dims": self.env_dims,
            "point_cloud": self.point_cloud_data,
            "min_magnitude": (
                self.min_magnitude.item()
                if isinstance(self.min_magnitude, torch.Tensor)
                else self.min_magnitude
            ),
            "max_magnitude": (
                self.max_magnitude.item()
                if isinstance(self.max_magnitude, torch.Tensor)
                else self.max_magnitude
            ),
            "norm_eps": self.norm_eps,
        }

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, idx):
        """Retrieves a single sample (normalized CMR) for the active split."""
        original_idx = self.active_indices[idx]
        return {
            "rx_position": self.rx_positions[original_idx],
            "cmr": self.cmr_normalized[original_idx],
            "index": original_idx,
        }

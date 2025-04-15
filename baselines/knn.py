# baselines/knn.py

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from datasets.dataloader import get_wireless_dataloader
from datasets.wireless_dataset import WirelessDataset


def nmse_loss_np(H_true, H_pred):
    """
    Computes the Normalized Mean Squared Error (NMSE):
        NMSE = ||H_true - H_pred||^2 / ||H_true||^2
    Evaluated over samples in a NumPy array.
    """
    diff = H_true - H_pred
    error = np.linalg.norm(diff, axis=1) ** 2
    norm = np.linalg.norm(H_true, axis=1) ** 2 + 1e-8
    nmse = error / norm
    return np.mean(nmse)


def main():
    data_path = "/Users/muahmed/Desktop/Gaussian Splatting/Neurips 2025/ard-r2f/datasets/output/conf_16x2_414u_5.0ghz_sbrRT_sc104.mat"
    batch_size = 16
    num_epochs = 50
    n_neighbors = 5

    train_loader = get_wireless_dataloader(
        data_path, batch_size=batch_size, shuffle=True, train=True, train_ratio=0.9
    )

    dataset = WirelessDataset(data_path, train=True, train_ratio=0.9)
    tx_position = dataset.get_tx_position()
    d_tx = tx_position.shape[0]

    sample_batch = next(iter(train_loader))
    rx_position = sample_batch["rx_position"]
    path_loss = sample_batch["path_loss"]
    if path_loss.dim() == 1:
        path_loss = path_loss.unsqueeze(1)
    d_rx = rx_position.shape[1]
    input_dim = d_rx + d_tx + 1
    H_sample = sample_batch["channel_matrix"]
    tx_ant = H_sample.shape[1]
    rx_ant = H_sample.shape[2]
    output_dim = 2 * (tx_ant * rx_ant)

    X_train_list = []
    Y_train_list = []

    for batch in train_loader:
        rx_pos_batch = batch["rx_position"]
        path_loss_batch = batch["path_loss"]
        if path_loss_batch.dim() == 1:
            path_loss_batch = path_loss_batch.unsqueeze(1)
        H_batch = batch["channel_matrix"]
        if torch.is_complex(H_batch):
            H_real = H_batch.real
            H_imag = H_batch.imag
        else:
            H_real = H_batch
            H_imag = torch.zeros_like(H_batch)
        H_target = torch.cat(
            [H_real.view(H_real.size(0), -1), H_imag.view(H_imag.size(0), -1)], dim=1
        )
        tx_pos_batch = tx_position.unsqueeze(0).repeat(rx_pos_batch.size(0), 1)
        inputs = torch.cat([rx_pos_batch, tx_pos_batch, path_loss_batch], dim=1)
        X_train_list.append(inputs.cpu().numpy())
        Y_train_list.append(H_target.cpu().numpy())

    X_train = np.vstack(X_train_list)
    Y_train = np.vstack(Y_train_list)

    print(f"Fitting KNN regressor on {X_train.shape[0]} samples...")
    knn_model = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
    knn_model.fit(X_train, Y_train)

    for epoch in range(num_epochs):
        Y_pred = knn_model.predict(X_train)
        nmse = nmse_loss_np(Y_train, Y_pred)
        snr = -10 * np.log10(nmse + 1e-8)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {nmse:.6f}, SNR: {snr:.2f} dB")


if __name__ == "__main__":
    main()

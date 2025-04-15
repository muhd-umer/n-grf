# baselines/mdn.py

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.dataloader import get_wireless_dataloader
from datasets.wireless_dataset import WirelessDataset


def mdn_loss(pi, mu, sigma, target):
    """
    Computes the MDN loss (negative log likelihood) for a Gaussian mixture model.

    Args:
        pi: Tensor of shape (batch_size, n_mixtures) of mixture weights.
        mu: Tensor of shape (batch_size, n_mixtures, D) of means.
        sigma: Tensor of shape (batch_size, n_mixtures, D) of standard deviations.
        target: Tensor of shape (batch_size, D) with ground-truth target.

    Returns:
        Mean negative log-likelihood over the batch.
    """
    D = target.size(1)
    target = target.unsqueeze(1)
    exp_term = -0.5 * torch.sum(((target - mu) / sigma) ** 2, dim=2)
    log_coeff = -0.5 * D * torch.log(
        torch.tensor(2 * np.pi, device=target.device)
    ) - torch.sum(torch.log(sigma), dim=2)
    log_component_probs = torch.log(pi + 1e-8) + log_coeff + exp_term
    log_prob = torch.logsumexp(log_component_probs, dim=1)
    return -torch.mean(log_prob)


class MDNBaseline(nn.Module):
    """
    A Mixture Density Network that takes as input concatenated features
    (receiver position, transmitter position, and path loss) and outputs
    the parameters of a Gaussian mixture over the flattened channel matrix.

    The network uses an MLP architecture to produce the parameters:
        - Mixture weights (pi)
        - Means (mu)
        - Standard deviations (sigma)

    The final layer outputs n_mixtures * (1 + 2 * output_dim) numbers that are
    then partitioned into the three sets.
    """

    def __init__(self, input_dim, hidden_dims, output_dim, n_mixtures=5):
        super(MDNBaseline, self).__init__()
        self.n_mixtures = n_mixtures
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.hidden = nn.Sequential(*layers)

        self.mdn_linear = nn.Linear(prev_dim, n_mixtures * (1 + 2 * output_dim))
        self.softplus = nn.Softplus()

    def forward(self, x):
        batch_size = x.size(0)
        hidden_out = self.hidden(x)
        mdn_params = self.mdn_linear(hidden_out)
        split_size = self.n_mixtures
        pi = mdn_params[:, :split_size]
        mu = mdn_params[:, split_size : split_size + self.n_mixtures * self.output_dim]
        sigma = mdn_params[:, split_size + self.n_mixtures * self.output_dim :]
        mu = mu.view(batch_size, self.n_mixtures, self.output_dim)
        sigma = sigma.view(batch_size, self.n_mixtures, self.output_dim)
        pi = nn.functional.softmax(pi, dim=1)
        sigma = self.softplus(sigma) + 1e-8
        return pi, mu, sigma


def nmse_loss(H_true, H_pred):
    """
    Computes the Normalized Mean Squared Error (NMSE) defined as:
      NMSE = ||H_true - H_pred||^2 / ||H_true||^2
    This loss is averaged over the batch.
    """
    error = torch.norm(H_true - H_pred, dim=1) ** 2
    norm = torch.norm(H_true, dim=1) ** 2
    nmse = error / norm
    return nmse.mean()


def main():
    data_path = "/Users/muahmed/Desktop/Gaussian Splatting/Neurips 2025/ard-r2f/datasets/output/conf_16x2_414u_5.0ghz_sbrRT_sc104.mat"
    batch_size = 16
    learning_rate = 1e-3
    num_epochs = 50

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
    H = sample_batch["channel_matrix"]
    tx_ant = H.shape[1]
    rx_ant = H.shape[2]
    output_dim = 2 * (tx_ant * rx_ant)
    hidden_dims = [128, 256, 128]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MDNBaseline(input_dim, hidden_dims, output_dim, n_mixtures=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_snr = 0.0
        num_batches = 0

        for batch in train_loader:
            rx_pos_batch = batch["rx_position"].to(device)
            path_loss_batch = batch["path_loss"].to(device)
            if path_loss_batch.dim() == 1:
                path_loss_batch = path_loss_batch.unsqueeze(1)
            H_batch = batch["channel_matrix"].to(device)
            if torch.is_complex(H_batch):
                H_real = H_batch.real
                H_imag = H_batch.imag
            else:
                H_real = H_batch
                H_imag = torch.zeros_like(H_batch)
            H_target = torch.cat(
                [H_real.view(H_real.size(0), -1), H_imag.view(H_imag.size(0), -1)],
                dim=1,
            )
            tx_pos_batch = (
                tx_position.to(device).unsqueeze(0).repeat(rx_pos_batch.size(0), 1)
            )
            inputs = torch.cat([rx_pos_batch, tx_pos_batch, path_loss_batch], dim=1)
            pi, mu, sigma = model(inputs)
            loss_mdn = mdn_loss(pi, mu, sigma, H_target)
            loss = loss_mdn
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            weighted_mu = torch.sum(pi.unsqueeze(2) * mu, dim=1)
            snr = -10 * torch.log10(nmse_loss(H_target, weighted_mu) + 1e-8)
            epoch_loss += loss.item()
            epoch_snr += snr.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_snr = epoch_snr / num_batches
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, SNR: {avg_snr:.2f} dB"
        )


if __name__ == "__main__":
    main()

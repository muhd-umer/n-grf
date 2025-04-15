# baselines/transformer.py

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.dataloader import get_wireless_dataloader
from datasets.wireless_dataset import WirelessDataset


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


class TransformerBaseline(nn.Module):
    """
    A transformer-based network that takes as input concatenated features
    (receiver position, transmitter position, and path loss) and outputs a
    flattened channel matrix prediction where the real and imaginary parts are concatenated.

    The network first projects the input features into a sequence of tokens,
    adds positional encoding, and processes the tokens with a transformer encoder.
    The encoder output is then flattened and projected to the desired output dimension.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_tokens=4,
        d_model=128,
        num_layers=2,
        nhead=4,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super(TransformerBaseline, self).__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, num_tokens * d_model)
        self.pos_embedding = nn.Parameter(torch.randn(num_tokens, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_proj = nn.Linear(num_tokens * d_model, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        tokens = self.input_proj(x)
        tokens = tokens.view(batch_size, self.num_tokens, self.d_model)
        tokens = tokens.permute(1, 0, 2)
        tokens = tokens + self.pos_embedding.unsqueeze(1)
        encoded_tokens = self.transformer_encoder(tokens)
        encoded_tokens = (
            encoded_tokens.permute(1, 0, 2).contiguous().view(batch_size, -1)
        )
        output = self.output_proj(encoded_tokens)
        return output


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerBaseline(
        input_dim, output_dim, num_tokens=4, d_model=128, num_layers=2, nhead=4
    ).to(device)
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
            outputs = model(inputs)
            loss_nmse = nmse_loss(H_target, outputs)
            loss = loss_nmse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            snr = -10 * torch.log10(loss_nmse + 1e-8)
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

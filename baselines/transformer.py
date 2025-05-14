# baselines/transformer.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer.
    Note that this positional encoding is different from the one in
    utils/pos_encoder.py, which is the one nGRF uses.

    Args:
        d_model (int): Embedding dimension
        max_len (int): Maximum sequence length
        dropout_p (float): Dropout probability
    """

    def __init__(self, d_model, max_len=5000, dropout_p=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add positional encoding to input sequence

        Args:
            x (torch.Tensor): Input tensor of shape [seq_len, batch_size, d_model]

        Returns:
            torch.Tensor: Output tensor with positional encoding added
        """

        x = x + self.pe[: x.size(0), :].unsqueeze(1)
        return self.dropout(x)


class Transformer(nn.Module):
    """
    Implementation of the Transformer architecture.

    Args:
        input_dim (int): Dimension of input features
        output_dim (int): Dimension of output (flattened channel matrix)
        num_tokens (int): Number of tokens to generate from input
        d_model (int): Transformer model dimension
        nhead (int): Number of attention heads
        num_encoder_layers (int): Number of encoder layers
        num_decoder_layers (int): Number of decoder layers
        dim_feedforward (int): Dimension of feedforward network
        dropout_p (float): Dropout probability
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_tokens=8,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout_p=0.1,
    ):
        super(Transformer, self).__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, num_tokens * d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout_p=dropout_p)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_p,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.query_embed = nn.Parameter(torch.randn(1, d_model))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_p,
            batch_first=False,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        Forward pass through the transformer

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim]
        """
        batch_size = x.size(0)

        tokens = self.input_proj(x)
        tokens = tokens.view(batch_size, self.num_tokens, self.d_model)
        tokens = tokens.transpose(0, 1)
        tokens = self.pos_encoder(tokens)
        memory = self.transformer_encoder(tokens)

        query = self.query_embed.unsqueeze(1).expand(1, batch_size, self.d_model)

        output = self.transformer_decoder(query, memory)
        output = self.output_proj(output.squeeze(0))

        return output


class TransformerBaseline:
    """
    Transformer baseline for channel estimation.

    Args:
        input_dim (int): Input dimension (e.g., 3 for receiver position)
        num_tx_ant (int): Number of transmit antennas
        num_rx_ant (int): Number of receive antennas
        num_tokens (int): Number of tokens for transformer
        d_model (int): Model dimension for transformer
        nhead (int): Number of attention heads
        num_encoder_layers (int): Number of encoder layers
        num_decoder_layers (int): Number of decoder layers
        dim_feedforward (int): Dimension of feedforward network
        dropout_p (float): Dropout probability
        learning_rate (float): Learning rate for optimizer
        weight_decay (float): Weight decay for optimizer
        device (str): Device to use ('cpu' or 'cuda')
    """

    def __init__(
        self,
        input_dim=3,
        num_tx_ant=1,
        num_rx_ant=1,
        num_tokens=8,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout_p=0.1,
        learning_rate=0.001,
        weight_decay=1e-5,
        device="cuda",
    ):
        self.num_tx_ant = num_tx_ant
        self.num_rx_ant = num_rx_ant
        self.output_dim = 2 * num_tx_ant * num_rx_ant
        self.device = device

        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout_p

        self.model = Transformer(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_tokens=self.num_tokens,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout_p=self.dropout_p,
        ).to(device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.criterion = nn.MSELoss()

    def train(self):
        """Set model to training mode"""
        self.model.train()

    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()

    def to(self, device):
        """Move model to device"""
        self.device = device
        self.model.to(device)
        return self

    def fit(self, rx_positions, channels, batch_size=32, epochs=100, verbose=True):
        """
        Train the transformer on the provided data

        Args:
            rx_positions (torch.Tensor): Receiver positions of shape [N, 3]
            channels (torch.Tensor): Complex channel matrices of shape [N, Nt, Nr]
            batch_size (int): Batch size for training
            epochs (int): Number of epochs to train
            verbose (bool): Whether to print progress

        Returns:
            list: Training losses
        """
        self.train()
        dataset_size = rx_positions.shape[0]

        channels_flat = channels.view(channels.shape[0], -1)
        channels_real = torch.cat([channels_flat.real, channels_flat.imag], dim=1).to(
            self.device
        )

        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            indices = torch.randperm(dataset_size)

            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i : min(i + batch_size, dataset_size)]

                rx_batch = rx_positions[batch_indices].to(self.device)
                channels_batch = channels_real[batch_indices]

                outputs = self.model(rx_batch)

                loss = self.criterion(outputs, channels_batch)

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item() * len(batch_indices)

            epoch_loss /= dataset_size
            losses.append(epoch_loss)

            self.scheduler.step(epoch_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")

        return losses

    def predict(self, rx_positions):
        """
        Predict complex channel matrices for new receiver positions

        Args:
            rx_positions (torch.Tensor): Receiver positions of shape [N, 3]

        Returns:
            torch.Tensor: Predicted complex channel matrices of shape [N, Nt, Nr]
        """
        self.eval()
        with torch.no_grad():
            rx_positions = rx_positions.to(self.device)

            predictions = self.model(rx_positions)

            half_dim = self.output_dim // 2
            real_part = predictions[:, :half_dim]
            imag_part = predictions[:, half_dim:]

            complex_preds = torch.complex(real_part, imag_part)
            return complex_preds.view(-1, self.num_tx_ant, self.num_rx_ant)

    def save(self, path):
        """Save model to path"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "num_tx_ant": self.num_tx_ant,
                "num_rx_ant": self.num_rx_ant,
                "output_dim": self.output_dim,
                "input_dim": self.input_dim,
                "num_tokens": self.num_tokens,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_encoder_layers": self.num_encoder_layers,
                "num_decoder_layers": self.num_decoder_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout_p": self.dropout_p,
            },
            path,
        )

    @classmethod
    def load(cls, path, device="cuda"):
        """Load model from path"""
        checkpoint = torch.load(path, map_location=device)

        input_dim = checkpoint.get("input_dim", 3)
        num_tokens = checkpoint.get("num_tokens", 8)
        d_model = checkpoint.get("d_model", 128)
        nhead = checkpoint.get("nhead", 4)
        num_encoder_layers = checkpoint.get("num_encoder_layers", 3)
        num_decoder_layers = checkpoint.get("num_decoder_layers", 3)
        dim_feedforward = checkpoint.get("dim_feedforward", 512)
        dropout_p = checkpoint.get("dropout_p", 0.1)

        instance = cls(
            input_dim=input_dim,
            num_tx_ant=checkpoint["num_tx_ant"],
            num_rx_ant=checkpoint["num_rx_ant"],
            num_tokens=num_tokens,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout_p=dropout_p,
            device=device,
        )

        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        instance.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return instance

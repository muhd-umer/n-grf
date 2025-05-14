# baselines/mlp.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.

    Args:
        dim (int): Feature dimension
        hidden_dim (int): Hidden dimension for the block
        dropout_p (float): Dropout probability
        use_layer_norm (bool): Whether to use layer normalization
    """

    def __init__(self, dim, hidden_dim=None, dropout_p=0.1, use_layer_norm=True):
        super(ResidualBlock, self).__init__()
        if hidden_dim is None:
            hidden_dim = dim

        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout_p)
        self.use_layer_norm = use_layer_norm

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        identity = x

        out = self.linear1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)

        out = out + identity

        if self.use_layer_norm:
            out = self.layer_norm(out)

        return F.relu(out)


class MLP(nn.Module):
    """
    MLP with skip connections.

    Args:
        input_dim (int): Input dimension (e.g., receiver position)
        hidden_dims (list): List of hidden dimensions for each stage of the MLP.
                            Each stage consists of a ResidualBlock.
                            Linear projections are added if dimensions change between stages.
        output_dim (int): Output dimension (flattened channel)
        dropout_p (float): Dropout probability
        use_layer_norm (bool): Whether to use layer normalization in ResidualBlocks
    """

    def __init__(
        self, input_dim, hidden_dims, output_dim, dropout_p=0.1, use_layer_norm=True
    ):
        super(MLP, self).__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])

        self.layers = nn.ModuleList()
        current_dim = hidden_dims[0]

        for i in range(len(hidden_dims)):
            block_target_dim = hidden_dims[i]

            if current_dim != block_target_dim:
                self.layers.append(nn.Linear(current_dim, block_target_dim))
                self.layers.append(nn.ReLU())
                current_dim = block_target_dim

            self.layers.append(
                ResidualBlock(
                    dim=current_dim,
                    hidden_dim=current_dim,
                    dropout_p=dropout_p,
                    use_layer_norm=use_layer_norm,
                )
            )

        self.output_layer = nn.Linear(current_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the MLP

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim]
        """
        x = F.relu(self.input_layer(x))

        for layer_module in self.layers:
            x = layer_module(x)

        return self.output_layer(x)


class MLPBaseline:
    """
    Baseline MLP model for channel estimation.

    Args:
        input_dim (int): Input dimension (e.g., 3 for receiver position)
        hidden_dims (list): List of hidden dimensions for the MLP
        num_tx_ant (int): Number of transmit antennas
        num_rx_ant (int): Number of receive antennas
        dropout_p (float): Dropout probability
        use_layer_norm (bool): Whether to use layer normalization
        learning_rate (float): Learning rate for optimizer
        weight_decay (float): Weight decay for optimizer
        device (str): Device to use ('cpu' or 'cuda')
    """

    def __init__(
        self,
        input_dim=3,
        hidden_dims=[256, 256, 128],
        num_tx_ant=1,
        num_rx_ant=1,
        dropout_p=0.1,
        use_layer_norm=True,
        learning_rate=0.001,
        weight_decay=1e-5,
        device="cuda",
    ):
        self.num_tx_ant = num_tx_ant
        self.num_rx_ant = num_rx_ant
        self.output_dim = 2 * num_tx_ant * num_rx_ant
        self.device = device

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_p = dropout_p
        self.use_layer_norm = use_layer_norm

        self.model = MLP(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            dropout_p=self.dropout_p,
            use_layer_norm=self.use_layer_norm,
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
        Train the MLP on the provided data

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
                "hidden_dims": self.hidden_dims,
                "dropout_p": self.dropout_p,
                "use_layer_norm": self.use_layer_norm,
            },
            path,
        )

    @classmethod
    def load(cls, path, device="cuda"):
        """Load model from path"""
        checkpoint = torch.load(path, map_location=device)

        input_dim = checkpoint.get("input_dim", 3)
        hidden_dims = checkpoint.get("hidden_dims", [256, 256, 128])
        dropout_p = checkpoint.get("dropout_p", 0.1)
        use_layer_norm = checkpoint.get("use_layer_norm", True)

        instance = cls(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_tx_ant=checkpoint["num_tx_ant"],
            num_rx_ant=checkpoint["num_rx_ant"],
            dropout_p=dropout_p,
            use_layer_norm=use_layer_norm,
            device=device,
        )

        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        instance.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return instance

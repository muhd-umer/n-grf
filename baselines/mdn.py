# baselines/mdn.py

import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MixtureDensityNetwork(nn.Module):
    """
    Mixture Density Network (MDN) module.

    Args:
        input_dim (int): Dimension of input features
        hidden_dims (list): List of hidden dimensions for the MLP backbone
        output_dim (int): Dimension of the output (flattened channel matrix)
        n_mixtures (int): Number of Gaussian mixtures
        dropout_p (float): Dropout probability
    """

    def __init__(self, input_dim, hidden_dims, output_dim, n_mixtures=5, dropout_p=0.1):
        super(MixtureDensityNetwork, self).__init__()
        self.n_mixtures = n_mixtures
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            prev_dim = h
        self.hidden = nn.Sequential(*layers)

        self.pi_layer = nn.Linear(prev_dim, n_mixtures)
        self.mu_layer = nn.Linear(prev_dim, n_mixtures * output_dim)
        self.sigma_layer = nn.Linear(prev_dim, n_mixtures * output_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize MDN layers with suitable starting values"""

        nn.init.constant_(self.pi_layer.bias, 0.0)

        nn.init.normal_(self.mu_layer.weight, 0, 0.01)
        nn.init.constant_(self.mu_layer.bias, 0.0)

        nn.init.constant_(self.sigma_layer.weight, 0.0)
        nn.init.constant_(self.sigma_layer.bias, 0.0)

    def forward(self, x):
        """
        Forward pass through the MDN

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor):
                pi: mixture weights [batch_size, n_mixtures]
                mu: mixture means [batch_size, n_mixtures, output_dim]
                sigma: mixture standard deviations [batch_size, n_mixtures, output_dim]
        """
        batch_size = x.size(0)
        hidden_out = self.hidden(x)

        pi = torch.softmax(self.pi_layer(hidden_out), dim=1)

        mu = self.mu_layer(hidden_out)
        mu = mu.view(batch_size, self.n_mixtures, self.output_dim)

        sigma = self.sigma_layer(hidden_out)
        sigma = sigma.view(batch_size, self.n_mixtures, self.output_dim)
        sigma = torch.nn.functional.softplus(sigma) + 1e-6

        return pi, mu, sigma

    def loss_function(self, y, pi, mu, sigma):
        """
        Negative log likelihood loss for a Gaussian mixture model

        Args:
            y (torch.Tensor): Target tensor of shape [batch_size, output_dim]
            pi (torch.Tensor): Mixture weights [batch_size, n_mixtures]
            mu (torch.Tensor): Mixture means [batch_size, n_mixtures, output_dim]
            sigma (torch.Tensor): Mixture standard deviations [batch_size, n_mixtures, output_dim]

        Returns:
            torch.Tensor: Negative log likelihood loss (scalar)
        """
        batch_size = y.size(0)

        y = y.unsqueeze(1).expand(-1, self.n_mixtures, -1)

        exponent = -0.5 * ((y - mu) / sigma) ** 2
        log_probs = -0.5 * (torch.log(2 * math.pi * sigma**2) + ((y - mu) / sigma) ** 2)
        log_probs = torch.sum(log_probs, dim=2)

        weighted_log_probs = torch.log(pi) + log_probs

        max_log_probs = torch.max(weighted_log_probs, dim=1, keepdim=True)[0]
        log_sum = max_log_probs + torch.log(
            torch.sum(
                torch.exp(weighted_log_probs - max_log_probs), dim=1, keepdim=True
            )
        )

        return -torch.mean(log_sum)

    def sample(self, x, num_samples=1):
        """
        Generate samples from the mixture model

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
            num_samples (int): Number of samples to generate per input

        Returns:
            torch.Tensor: Generated samples of shape [batch_size, num_samples, output_dim]
        """
        with torch.no_grad():
            batch_size = x.size(0)
            pi, mu, sigma = self.forward(x)

            samples = torch.zeros(
                batch_size, num_samples, self.output_dim, device=x.device
            )

            for s in range(num_samples):

                mixture_idx = torch.multinomial(pi, 1).squeeze(1)

                batch_indices = torch.arange(batch_size, device=x.device)
                selected_mu = mu[batch_indices, mixture_idx]
                selected_sigma = sigma[batch_indices, mixture_idx]

                epsilon = torch.randn_like(selected_mu)
                samples[:, s, :] = selected_mu + selected_sigma * epsilon

            return samples

    def predict(self, x):
        """
        Generate a single prediction by taking the mean of the most probable mixture component

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]

        Returns:
            torch.Tensor: Predictions of shape [batch_size, output_dim]
        """
        with torch.no_grad():
            batch_size = x.size(0)
            pi, mu, sigma = self.forward(x)

            max_pi_idx = torch.argmax(pi, dim=1)

            batch_indices = torch.arange(batch_size, device=x.device)
            predictions = mu[batch_indices, max_pi_idx]

            return predictions


class MDNBaseline:
    """
    Baseline MDN model for channel estimation.

    Args:
        input_dim (int): Dimension of input features
        hidden_dims (list): List of hidden dimensions for the MLP backbone
        num_tx_ant (int): Number of transmit antennas
        num_rx_ant (int): Number of receive antennas
        n_mixtures (int): Number of Gaussian mixtures
        dropout_p (float): Dropout probability
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
        n_mixtures=5,
        dropout_p=0.1,
        learning_rate=0.001,
        weight_decay=1e-6,
        device="cuda",
    ):
        self.num_tx_ant = num_tx_ant
        self.num_rx_ant = num_rx_ant
        self.output_dim = 2 * num_tx_ant * num_rx_ant
        self.device = device

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_mixtures = n_mixtures
        self.dropout_p = dropout_p

        self.model = MixtureDensityNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            n_mixtures=self.n_mixtures,
            dropout_p=self.dropout_p,
        ).to(device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

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
        Train the MDN on the provided data

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

                pi, mu, sigma = self.model(rx_batch)

                loss = self.model.loss_function(channels_batch, pi, mu, sigma)

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

            predictions = self.model.predict(rx_positions)

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
                "n_mixtures": self.n_mixtures,
                "dropout_p": self.dropout_p,
            },
            path,
        )

    @classmethod
    def load(cls, path, device="cuda"):
        """Load model from path"""
        checkpoint = torch.load(path, map_location=device)

        input_dim = checkpoint.get("input_dim", 3)
        hidden_dims = checkpoint.get("hidden_dims", [256, 256, 128])
        n_mixtures = checkpoint.get("n_mixtures", 5)
        dropout_p = checkpoint.get("dropout_p", 0.1)

        instance = cls(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_tx_ant=checkpoint["num_tx_ant"],
            num_rx_ant=checkpoint["num_rx_ant"],
            n_mixtures=n_mixtures,
            dropout_p=dropout_p,
            device=device,
        )

        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        instance.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return instance

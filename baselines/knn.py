# baselines/knn.py

import torch
from sklearn.neighbors import KNeighborsRegressor


class KNNBaseline:
    """
    Baseline KNN model for channel estimation.

    Args:
        k (int): Number of nearest neighbors to use
        weights (str): Weight function used in prediction. Possible values:
            'uniform' : uniform weights (all points in each neighborhood have equal weight)
            'distance' : weight points by the inverse of their distance
    """

    def __init__(self, k=5, weights="distance"):
        self.k = k
        self.weights = weights
        self.real_regressor = None
        self.imag_regressor = None
        self.is_fitted = False

        self.num_tx_ant = None
        self.num_rx_ant = None
        self.channel_shape = None
        self.device = "cpu"

    def fit(self, rx_positions, channels):
        """
        Fit the KNN model to the training data.

        Args:
            rx_positions (torch.Tensor): Receiver positions of shape [N, 3]
            channels (torch.Tensor): Complex channel matrices of shape [N, Nt, Nr]
                where Nt is number of transmit antennas and Nr is number of receive antennas
        """

        self.device = channels.device
        self.num_tx_ant = channels.shape[1]
        self.num_rx_ant = channels.shape[2]
        self.channel_shape = channels.shape[1:]

        rx_positions_np = rx_positions.cpu().detach().numpy()

        channels_flat = channels.view(channels.shape[0], -1)
        real_parts = channels_flat.real.cpu().detach().numpy()
        imag_parts = channels_flat.imag.cpu().detach().numpy()

        self.real_regressor = KNeighborsRegressor(
            n_neighbors=self.k, weights=self.weights
        )
        self.imag_regressor = KNeighborsRegressor(
            n_neighbors=self.k, weights=self.weights
        )

        self.real_regressor.fit(rx_positions_np, real_parts)
        self.imag_regressor.fit(rx_positions_np, imag_parts)

        self.is_fitted = True
        return self

    def predict(self, rx_positions):
        """
        Predict complex channel matrices for new receiver positions.

        Args:
            rx_positions (torch.Tensor): Receiver positions of shape [N, 3]

        Returns:
            torch.Tensor: Predicted complex channel matrices of shape [N, Nt, Nr]
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        output_device = rx_positions.device
        rx_positions_np = rx_positions.cpu().detach().numpy()

        real_preds = self.real_regressor.predict(rx_positions_np)
        imag_preds = self.imag_regressor.predict(rx_positions_np)

        real_tensor = torch.tensor(
            real_preds, device=output_device, dtype=torch.float32
        )
        imag_tensor = torch.tensor(
            imag_preds, device=output_device, dtype=torch.float32
        )

        complex_preds = torch.complex(real_tensor, imag_tensor)
        return complex_preds.view(-1, self.num_tx_ant, self.num_rx_ant)

    def save(self, path):
        """Save the model to a file"""
        import joblib

        model_data = {
            "k": self.k,
            "weights": self.weights,
            "real_regressor": self.real_regressor,
            "imag_regressor": self.imag_regressor,
            "num_tx_ant": self.num_tx_ant,
            "num_rx_ant": self.num_rx_ant,
            "channel_shape": self.channel_shape,
            "is_fitted": self.is_fitted,
            "device": str(self.device),
        }

        joblib.dump(model_data, path)

    @classmethod
    def load(cls, path, device="cpu"):
        """Load the model from a file"""
        import joblib

        target_device = torch.device(device)

        model_data = joblib.load(path)
        model = cls(k=model_data["k"], weights=model_data["weights"])
        model.real_regressor = model_data["real_regressor"]
        model.imag_regressor = model_data["imag_regressor"]
        model.num_tx_ant = model_data["num_tx_ant"]
        model.num_rx_ant = model_data["num_rx_ant"]
        model.channel_shape = model_data["channel_shape"]
        model.is_fitted = model_data["is_fitted"]

        model.device = target_device
        return model

    def to(self, device):
        """
        Set the target device for output tensors.
        Sklearn models are CPU-based; this affects output tensor placement.
        """
        self.device = torch.device(device)
        return self

    def train(self):
        """Set model to training mode (no-op for KNN)"""
        pass

    def eval(self):
        """Set model to evaluation mode (no-op for KNN)"""
        pass

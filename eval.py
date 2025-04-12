# eval.py

import argparse
import logging
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataloader import get_dataloaders
from engine import _torch_impl as torch_impl
from engine import rasterize
from models.gaussian_model import GaussianModel
from models.loss import get_loss_function
from utils.general_utils import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Gaussian model for channel reconstruction"
    )

    # dataset params
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset file"
    )

    # model params
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results",
    )

    # evaluation params
    parser.add_argument(
        "--num_samples", type=int, default=4, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse_corr",
        choices=["nmse", "log_mse", "mse_corr", "polar_mse", "cosine", "log_mag_phase"],
        help="Loss function to use for evaluation",
    )
    parser.add_argument(
        "--scale_modifier",
        type=float,
        default=1.0,
        help="Scale modifier for Gaussian scaling",
    )
    parser.add_argument(
        "--disable_cuda",
        action="store_true",
        help="Disable CUDA implementation and use PyTorch fallback for rasterization",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    return parser.parse_args()


def setup_logging(output_dir):
    """Set up logging to file and console"""
    log_file = output_dir / "eval.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def calculate_snr(pred_channel, target_channel):
    """Calculate SNR between predicted and target channel matrices"""

    n_rx = pred_channel.shape[1] // 2
    pred_complex = torch.complex(pred_channel[:, :n_rx], pred_channel[:, n_rx:])
    target_complex = torch.complex(target_channel[:, :n_rx], target_channel[:, n_rx:])

    error = pred_complex - target_complex

    signal_power = torch.sum(torch.abs(target_complex) ** 2).item()
    noise_power = torch.sum(torch.abs(error) ** 2).item()

    if noise_power < 1e-10:
        return float("inf")

    snr_db = 10 * np.log10(signal_power / noise_power)

    return snr_db


def rasterize_channel(
    model, rx_position, tx_position, num_tx_ant, num_rx_ant, frequency, args
):
    """Helper function to rasterize the channel based on command-line arguments"""
    if args.disable_cuda:
        return torch_impl.rasterize(
            points=model.get_xyz,
            scaling=model.get_scaling,
            rotation=model.get_rotation,
            attenuation=model.get_features[:, 0:1].contiguous(),
            phase_rotation=model.get_features[:, 1:2].contiguous(),
            opacity=model.get_opacity,
            receiver=rx_position,
            transmitter=tx_position,
            num_tx=num_tx_ant,
            num_rx=num_rx_ant,
            frequency=frequency,
            scale_modifier=args.scale_modifier,
        )
    else:
        return rasterize(
            points=model.get_xyz,
            scaling=model.get_scaling,
            rotation=model.get_rotation,
            attenuation=model.get_features[:, 0:1].contiguous(),
            phase_rotation=model.get_features[:, 1:2].contiguous(),
            opacity=model.get_opacity,
            receiver=rx_position,
            transmitter=tx_position,
            num_tx=num_tx_ant,
            num_rx=num_rx_ant,
            frequency=frequency,
            scale_modifier=args.scale_modifier,
        )


def visualize_channel_matrices(
    pred_channels, target_channels, rx_positions, output_dir, num_tx_ant, num_rx_ant
):
    """Visualize channel matrices with 4 samples in a figure, each with predicted and target side by side"""
    num_samples = min(len(pred_channels), 4)

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, 2)

    for i in range(num_samples):
        pred = pred_channels[i]
        target = target_channels[i]
        rx_pos = rx_positions[i]

        num_tx = pred.shape[0]
        num_rx = pred.shape[1] // 2

        pred_complex = np.complex128(pred[:, :num_rx]) + 1j * np.complex128(
            pred[:, num_rx:]
        )
        target_complex = np.complex128(target[:, :num_rx]) + 1j * np.complex128(
            target[:, num_rx:]
        )

        pred_mag_db = 20 * np.log10(np.abs(pred_complex) + 1e-10)
        target_mag_db = 20 * np.log10(np.abs(target_complex) + 1e-10)

        vmin = min(np.min(pred_mag_db), np.min(target_mag_db))
        vmax = max(np.max(pred_mag_db), np.max(target_mag_db))

        extent = [0, num_rx_ant, num_tx_ant, 0]

        im1 = axes[i, 0].imshow(
            target_mag_db,
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )
        axes[i, 0].set_title(
            f"Target Channel Magnitude (dB)\nRX at ({rx_pos[0]:.2f}, {rx_pos[1]:.2f}, {rx_pos[2]:.2f})"
        )
        axes[i, 0].set_ylabel("TX Antenna")
        axes[i, 0].set_xlabel("RX Antenna")
        axes[i, 0].set_xlim(0, num_rx_ant)
        axes[i, 0].set_ylim(num_tx_ant, 0)
        axes[i, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[i, 0].yaxis.set_major_locator(MaxNLocator(integer=True))

        im2 = axes[i, 1].imshow(
            pred_mag_db,
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )
        axes[i, 1].set_title(f"Predicted Channel Magnitude (dB)")
        axes[i, 1].set_ylabel("TX Antenna")
        axes[i, 1].set_xlabel("RX Antenna")
        axes[i, 1].set_xlim(0, num_rx_ant)
        axes[i, 1].set_ylim(num_tx_ant, 0)
        axes[i, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[i, 1].yaxis.set_major_locator(MaxNLocator(integer=True))

        fig.colorbar(im2, ax=axes[i, 1], label="Magnitude (dB)")

    plt.tight_layout()
    plt.savefig(
        output_dir / "channel_magnitude_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, 2)

    for i in range(num_samples):
        pred = pred_channels[i]
        target = target_channels[i]
        rx_pos = rx_positions[i]

        num_tx = pred.shape[0]
        num_rx = pred.shape[1] // 2

        pred_complex = np.complex128(pred[:, :num_rx]) + 1j * np.complex128(
            pred[:, num_rx:]
        )
        target_complex = np.complex128(target[:, :num_rx]) + 1j * np.complex128(
            target[:, num_rx:]
        )

        pred_phase = np.angle(pred_complex)
        target_phase = np.angle(target_complex)

        vmin = -np.pi
        vmax = np.pi

        extent = [0, num_rx_ant, num_tx_ant, 0]

        im1 = axes[i, 0].imshow(
            target_phase, aspect="auto", cmap="hsv", vmin=vmin, vmax=vmax, extent=extent
        )
        axes[i, 0].set_title(
            f"Target Channel Phase\nRX at ({rx_pos[0]:.2f}, {rx_pos[1]:.2f}, {rx_pos[2]:.2f})"
        )
        axes[i, 0].set_ylabel("TX Antenna")
        axes[i, 0].set_xlabel("RX Antenna")
        axes[i, 0].set_xlim(0, num_rx_ant)
        axes[i, 0].set_ylim(num_tx_ant, 0)
        axes[i, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[i, 0].yaxis.set_major_locator(MaxNLocator(integer=True))

        im2 = axes[i, 1].imshow(
            pred_phase, aspect="auto", cmap="hsv", vmin=vmin, vmax=vmax, extent=extent
        )
        axes[i, 1].set_title(f"Predicted Channel Phase")
        axes[i, 1].set_ylabel("TX Antenna")
        axes[i, 1].set_xlabel("RX Antenna")
        axes[i, 1].set_xlim(0, num_rx_ant)
        axes[i, 1].set_ylim(num_tx_ant, 0)
        axes[i, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[i, 1].yaxis.set_major_locator(MaxNLocator(integer=True))

        fig.colorbar(im2, ax=axes[i, 1], label="Phase (rad)")

    plt.tight_layout()
    plt.savefig(
        output_dir / "channel_phase_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"Arguments: {args}")

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    logger.info("Loading dataset...")
    _, test_dataloader = get_dataloaders(
        args.data_path,
        batch_size=1,
        num_workers=2,
        drop_last=True,
    )

    tx_position = test_dataloader.dataset.get_tx_position().to(device)
    frequency = test_dataloader.dataset.frequency
    num_tx_ant = test_dataloader.dataset.num_tx_ant
    num_rx_ant = test_dataloader.dataset.num_rx_ant

    logger.info(f"Number of TX antennas: {num_tx_ant}")
    logger.info(f"Number of RX antennas: {num_rx_ant}")
    logger.info(f"Operating frequency: {frequency/1e9:.2f} GHz")

    logger.info(f"Loading model from {args.model_path}...")
    model = GaussianModel.load(args.model_path, device=device)
    model.eval()

    logger.info(f"Using loss function: {args.loss_type}")
    loss_fn = get_loss_function(args.loss_type)

    logger.info("Starting evaluation...")

    total_loss = 0.0
    total_snr = 0.0
    num_samples = 0

    vis_pred_channels = []
    vis_target_channels = []
    vis_rx_positions = []

    channel_log_file = output_dir / "channel_values.log"
    with open(channel_log_file, "w") as f:
        f.write("Sample,SNR(dB),Loss\n")

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dataloader, total=20)):
            rx_position = data["rx_position"].to(device).squeeze()
            gt_channel = data["channel_matrix"].to(device).squeeze()
            gt_channel = torch.hstack((gt_channel.real, gt_channel.imag))

            enc_data = {
                "tx_pos": tx_position,
                "rx_pos": rx_position,
                "frequency": frequency,
            }
            model.embed_features(enc_data)

            pred_channel = rasterize_channel(
                model, rx_position, tx_position, num_tx_ant, num_rx_ant, frequency, args
            )

            loss = loss_fn(pred_channel, gt_channel).item()
            snr = calculate_snr(pred_channel, gt_channel)

            total_loss += loss
            total_snr += snr
            num_samples += 1

            with open(channel_log_file, "a") as f:
                f.write(f"{i},{snr:.4f},{loss:.8f}\n")

                if len(vis_pred_channels) < args.num_samples:
                    f.write("\nPredicted Channel (Real | Imaginary):\n")
                    f.write(f"{pred_channel.cpu().numpy()}\n\n")
                    f.write("Target Channel (Real | Imaginary):\n")
                    f.write(f"{gt_channel.cpu().numpy()}\n\n")
                    f.write("-" * 50 + "\n\n")

            if len(vis_pred_channels) < args.num_samples:
                vis_pred_channels.append(pred_channel.cpu().numpy())
                vis_target_channels.append(gt_channel.cpu().numpy())
                vis_rx_positions.append(rx_position.cpu().numpy())

            if len(vis_pred_channels) >= args.num_samples and i >= 20:

                break

    avg_loss = total_loss / num_samples
    avg_snr = total_snr / num_samples

    logger.info(
        f"Evaluation complete. Average loss: {avg_loss:.6f}, Average SNR: {avg_snr:.2f} dB"
    )

    logger.info("Generating visualizations...")
    visualize_channel_matrices(
        vis_pred_channels,
        vis_target_channels,
        vis_rx_positions,
        output_dir,
        num_tx_ant,
        num_rx_ant,
    )

    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Dataset: {args.data_path}\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"Loss type: {args.loss_type}\n")
        f.write(f"Average loss: {avg_loss:.6f}\n")
        f.write(f"Average SNR: {avg_snr:.2f} dB\n")

    logger.info(f"Evaluation results saved to {output_dir}")


if __name__ == "__main__":
    main()

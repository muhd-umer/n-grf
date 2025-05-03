# eval.py

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from datasets.dataloader import get_wireless_dataloader
from engine.render_channel import render_channel
from models.gaussian_model import GaussianChannelFieldModel
from utils.general_utils import set_random_seed
from utils.loss import NormalizedMSELoss, calculate_snr


def setup_eval_logging(log_dir: Path, checkpoint_name: str) -> logging.Logger:
    """Setup logging configuration for evaluation."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"eval_{checkpoint_name}_{timestamp}.log"

    logger = logging.getLogger("EvaluationLogger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def parse_args():
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Gaussian Channel Field Model"
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset file (.mat)"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/eval",
        help="Directory to save evaluation logs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num_samples_to_log",
        type=int,
        default=5,
        help="Number of sample predictions to log in detail",
    )
    parser.add_argument(
        "--loss_eps",
        type=float,
        default=1e-12,
        help="Epsilon for NMSE loss denominator",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train ratio used during training (to split test set correctly)",
    )

    args = parser.parse_args()
    return args


def format_complex_tensor(tensor: torch.Tensor) -> str:
    """Formats a complex tensor for logging."""
    if not torch.is_complex(tensor):
        return str(tensor.cpu().numpy())

    formatted = np.array2string(
        tensor.cpu().numpy(),
        formatter={"complex_kind": lambda x: f"{x.real:.3f}{x.imag:+.3f}j"},
        separator=", ",
    )
    return formatted


def evaluate(args):
    """Main evaluation function."""
    set_random_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    checkpoint_path = Path(args.checkpoint)
    log_dir = Path(args.log_dir)
    checkpoint_name = checkpoint_path.stem
    logger = setup_eval_logging(log_dir, checkpoint_name)

    logger.info("================ Evaluation Start ================")
    logger.info(f"Arguments: {args}")
    logger.info(f"Using device: {device}")
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    try:
        model_state = torch.load(checkpoint_path, map_location="cpu")
        model_config = model_state["config"]
        model = GaussianChannelFieldModel.load(checkpoint_path, device=device)[0]
        model.eval()
        logger.info("Model loaded successfully.")
        logger.info(f"Model Configuration: {model_config}")
        logger.info(f"Total Gaussians in loaded model: {model.get_xyz.shape[0]:,}")
    except Exception as e:
        logger.exception(f"Failed to load checkpoint: {e}")
        return

    logger.info(f"Loading data from: {args.data_path}")

    try:
        val_loader = get_wireless_dataloader(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            train=False,
            train_ratio=args.train_ratio,
            seed=args.seed,
            drop_last=False,
            pin_memory=True,
        )
        metadata = val_loader.dataset.get_metadata()
        nt = metadata["num_tx_ant"]
        nr = metadata["num_rx_ant"]
        wavelength = metadata["wavelength"]
        tx_position = metadata["tx_position"].to(device)
        logger.info(
            f"Dataset Metadata: Nt={nt}, Nr={nr}, Freq={metadata['frequency']/1e9:.2f}GHz, Lambda={wavelength:.4f}m"
        )
        logger.info(f"Validation/Test set size: {len(val_loader.dataset)}")
    except Exception as e:
        logger.exception(f"Failed to load data: {e}")
        return

    criterion = NormalizedMSELoss(eps=args.loss_eps).to(device)
    all_losses = []
    all_snrs = []
    sample_details: List[Dict] = []

    start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            rx_pos_batch = batch["rx_position"].to(device)
            h_gt_batch = batch["channel_matrix"].to(device)
            current_batch_size = rx_pos_batch.shape[0]

            h_pred_batch = render_channel(
                rx_positions=rx_pos_batch,
                model=model,
                tx_position=tx_position,
                wavelength=wavelength,
                nt=nt,
                nr=nr,
                eps=args.loss_eps,
            )

            for j in range(current_batch_size):
                h_pred = h_pred_batch[j].unsqueeze(0)
                h_gt = h_gt_batch[j].unsqueeze(0)

                loss = criterion(h_pred, h_gt).item()
                snr = calculate_snr(torch.tensor(loss, device=device)).item()

                if not np.isnan(loss) and not np.isinf(loss):
                    all_losses.append(loss)
                if not np.isnan(snr) and not np.isinf(snr):
                    all_snrs.append(snr)

                if len(sample_details) < args.num_samples_to_log:
                    sample_details.append(
                        {
                            "index": i * args.batch_size + j,
                            "rx_pos": rx_pos_batch[j].cpu().numpy(),
                            "h_gt": h_gt_batch[j],
                            "h_pred": h_pred_batch[j],
                            "loss": loss,
                            "snr": snr,
                        }
                    )

    end_time = time.time()
    eval_duration = end_time - start_time

    losses_np = np.array(all_losses)
    snrs_np = np.array(all_snrs)

    stats = {}
    if len(losses_np) > 0:
        stats["loss"] = {
            "mean": np.mean(losses_np),
            "std": np.std(losses_np),
            "min": np.min(losses_np),
            "max": np.max(losses_np),
            "count": len(losses_np),
        }
    else:
        stats["loss"] = {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "count": 0,
        }

    if len(snrs_np) > 0:
        stats["snr"] = {
            "mean": np.mean(snrs_np),
            "std": np.std(snrs_np),
            "min": np.min(snrs_np),
            "max": np.max(snrs_np),
            "count": len(snrs_np),
        }
    else:
        stats["snr"] = {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "count": 0,
        }

    logger.info("================ Evaluation Results ================")
    logger.info(f"Evaluation completed in {eval_duration:.2f} seconds.")
    logger.info(f"Total samples evaluated: {stats['loss']['count']}")

    logger.info("\n----- Overall Metrics -----")
    logger.info(f"Average NMSE Loss: {stats['loss']['mean']:.6e}")
    logger.info(f"Average SNR (dB):  {stats['snr']['mean']:.3f}")

    logger.info("\n----- Statistics -----")
    logger.info(f"NMSE Loss:")
    logger.info(f"  Mean: {stats['loss']['mean']:.6e}")
    logger.info(f"  Std Dev: {stats['loss']['std']:.6e}")
    logger.info(f"  Min:  {stats['loss']['min']:.6e}")
    logger.info(f"  Max:  {stats['loss']['max']:.6e}")
    logger.info(f"SNR (dB):")
    logger.info(f"  Mean: {stats['snr']['mean']:.3f}")
    logger.info(f"  Std Dev: {stats['snr']['std']:.3f}")
    logger.info(f"  Min:  {stats['snr']['min']:.3f}")
    logger.info(f"  Max:  {stats['snr']['max']:.3f}")

    if args.num_samples_to_log > 0 and sample_details:
        logger.info(f"\n----- Sample Predictions (Top {len(sample_details)}) -----")
        for sample in sample_details:
            logger.info(f"\nSample Index: {sample['index']}")
            logger.info(f"  Rx Position: {sample['rx_pos']}")
            logger.info(f"  NMSE Loss: {sample['loss']:.6e}")
            logger.info(f"  SNR (dB):  {sample['snr']:.3f}")

            h_gt_str = format_complex_tensor(sample["h_gt"])
            h_pred_str = format_complex_tensor(sample["h_pred"])
            logger.info(f"  H_gt:   {h_gt_str}")
            logger.info(f"  H_pred: {h_pred_str}")
    elif args.num_samples_to_log > 0:
        logger.info("\n----- Sample Predictions -----")
        logger.info("No samples were collected (evaluation might have failed early).")

    logger.info("================ Evaluation End ================")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)

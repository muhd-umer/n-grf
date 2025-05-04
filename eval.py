# eval.py

import argparse
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from datasets.dataloader import get_wireless_dataloader
from engine.render import render
from models.gaussian_model import GaussianChannelFieldModel
from utils.general_utils import set_random_seed
from utils.loss import calculate_snr


def eval_logger(log_dir: Path, checkpoint_name: str) -> Tuple[logging.Logger, Console]:
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

    console = Console(log_path=False)
    console_handler = RichHandler(
        console=console, rich_tracebacks=True, markup=True, show_path=False
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    logger.info(f"Evaluation logging initialized. Log file: {log_file}")
    return logger, console


def parse_args():
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Gaussian channel field model"
    )

    # common arguments
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
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of sample predictions to log in detail",
    )

    # eval-specific arguments
    parser.add_argument(
        "--snr_eps",
        type=float,
        default=1e-10,
        help="Epsilon for SNR calculation stability",
    )
    parser.add_argument(
        "--norm_eps",
        type=float,
        default=1e-8,
        help="Epsilon used for dataset normalization (must match training)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train ratio used during training (to split test set correctly)",
    )
    parser.add_argument(
        "--unnormalize_samples",
        action="store_true",
        help="Un-normalize sample predictions/targets before logging",
    )

    args = parser.parse_args()
    return args


def format_magnitude_tensor(tensor: torch.Tensor) -> str:
    """Formats a magnitude tensor (real numbers) for logging."""
    if torch.is_complex(tensor):

        warnings.warn("Warning: format_magnitude_tensor received a complex tensor")
        tensor = torch.abs(tensor)

    formatted = np.array2string(
        tensor.cpu().numpy(),
        formatter={"float_kind": lambda x: f"{x:.6f}"},
        separator=", ",
    )
    return formatted


def disp_stats(console: Console, stats: Dict) -> None:
    """Display statistics in a rich table."""
    table = Table(title="Evaluation Statistics")

    table.add_column("Metric", style="cyan")
    table.add_column("Mean", justify="right", style="green")
    table.add_column("Std Dev", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Count", justify="right")

    table.add_row(
        "MSE Loss (Normalized)",
        f"{stats['loss']['mean']:.6e}",
        f"{stats['loss']['std']:.6e}",
        f"{stats['loss']['min']:.6e}",
        f"{stats['loss']['max']:.6e}",
        f"{stats['loss']['count']}",
    )
    table.add_row(
        "SNR (dB)",
        f"{stats['snr']['mean']:.6f}",
        f"{stats['snr']['std']:.6f}",
        f"{stats['snr']['min']:.6f}",
        f"{stats['snr']['max']:.6f}",
        f"{stats['snr']['count']}",
    )

    console.print(table)


def disp_samples(
    console: Console,
    sample_details: List[Dict],
    unnormalize: bool,
    min_mag: float,
    max_mag: float,
    norm_eps: float,
) -> None:
    """Display sample details in a rich format, optionally un-normalizing."""
    if not sample_details:
        console.print(
            Panel(
                "[yellow]No samples were collected (evaluation might have failed early or num_samples=0).[/yellow]",
                title="Sample Predictions",
            )
        )
        return

    console.print(
        f"\n[bold cyan]Sample Predictions (Top {len(sample_details)})[/bold cyan]"
    )

    for sample in sample_details:
        table = Table(box=None, show_header=False)
        table.add_column("Property", style="blue", width=20)
        table.add_column("Value")

        table.add_row("Rx Position", str(sample["rx_pos"]))
        table.add_row("MSE Loss (Norm)", f"{sample['loss']:.6e}")
        table.add_row("SNR (dB)", f"{sample['snr']:.6f}")

        panel_content = table
        panel = Panel(
            panel_content,
            title=f"[bold]Sample Index {sample['index']}[/bold]",
            border_style="green",
        )
        console.print(panel)

        h_gt_norm = sample["h_mag_gt"]
        h_pred_norm = sample["h_mag_pred"]

        if unnormalize:
            scale = max_mag - min_mag

            if scale < norm_eps:
                h_gt_unnorm = torch.full_like(h_gt_norm, (max_mag + min_mag) / 2)
                h_pred_unnorm = torch.full_like(h_pred_norm, (max_mag + min_mag) / 2)
                unnorm_label = "(Constant)"
            else:
                h_gt_unnorm = h_gt_norm * scale + min_mag
                h_pred_unnorm = h_pred_norm * scale + min_mag
                unnorm_label = "(Un-normalized)"

            gt_title = Text(f"Ground Truth Magnitude {unnorm_label}", style="cyan")
            pred_title = Text(f"Predicted Magnitude {unnorm_label}", style="cyan")
            console.print(gt_title)
            console.print(format_magnitude_tensor(h_gt_unnorm))
            console.print(pred_title)
            console.print(format_magnitude_tensor(h_pred_unnorm))

        else:
            gt_title = Text("Ground Truth Magnitude (Normalized)", style="cyan")
            pred_title = Text("Predicted Magnitude (Normalized)", style="cyan")
            console.print(gt_title)
            console.print(format_magnitude_tensor(h_gt_norm))
            console.print(pred_title)
            console.print(format_magnitude_tensor(h_pred_norm))

        console.print("")


def evaluate(args):
    """Main evaluation function for magnitude prediction."""
    set_random_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    checkpoint_path = Path(args.checkpoint)
    log_dir = Path(args.log_dir)
    checkpoint_name = checkpoint_path.stem
    logger, console = eval_logger(log_dir, checkpoint_name)

    console.rule("[bold blue]Magnitude Prediction Evaluation Start[/bold blue]")
    logger.info(f"Evaluation Arguments: {vars(args)}")
    logger.info(f"Using device: {device}")
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    try:

        model_state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model_config = model_state["config"]

        model, load_iter = GaussianChannelFieldModel.load(
            checkpoint_path, device=device, training_args=None
        )
        model.eval()
        logger.info(
            f"[green]Model loaded successfully from iteration {load_iter}.[/green]"
        )
        logger.info(f"Model Configuration: {model_config}")
        logger.info(f"Total Gaussians in loaded model: {model.get_xyz.shape[0]:,}")
    except Exception as e:
        logger.exception(f"[bold red]Failed to load checkpoint:[/bold red] {e}")
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
            norm_eps=args.norm_eps,
        )

        metadata = val_loader.dataset.get_metadata()
        nt = metadata["num_tx_ant"]
        nr = metadata["num_rx_ant"]

        tx_position = metadata["tx_position"].to(device)
        min_mag = metadata.get("min_magnitude", 0.0)
        max_mag = metadata.get("max_magnitude", 1.0)

        metadata_table = Table(title="Dataset Metadata")
        metadata_table.add_column("Property", style="cyan")
        metadata_table.add_column("Value")
        metadata_table.add_row("Dataset Path", args.data_path)
        metadata_table.add_row("Transmit Antennas (Nt)", str(nt))
        metadata_table.add_row("Receive Antennas (Nr)", str(nr))
        metadata_table.add_row("Frequency", f"{metadata['frequency']/1e9:.2f} GHz")

        metadata_table.add_row("Is SISO", str(metadata["is_siso"]))
        metadata_table.add_row("Tx Position", str(tx_position.cpu().numpy()))
        metadata_table.add_row("Min Magnitude (Raw)", f"{min_mag:.4e}")
        metadata_table.add_row("Max Magnitude (Raw)", f"{max_mag:.4e}")
        metadata_table.add_row("Test Set Size", f"{len(val_loader.dataset)}")
        console.print(metadata_table)

    except Exception as e:
        logger.exception(f"[bold red]Failed to load data:[/bold red] {e}")
        return

    criterion = nn.MSELoss().to(device)
    all_losses = []
    all_snrs = []
    sample_details: List[Dict] = []

    start_time = time.time()
    with torch.no_grad():
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            eval_task = progress.add_task("[cyan]Evaluating...", total=len(val_loader))
            for i, batch in enumerate(val_loader):
                rx_pos_batch = batch["rx_position"].to(device)

                h_mag_gt_batch = batch["channel"].to(device)
                original_indices = batch["index"]
                current_batch_size = rx_pos_batch.shape[0]

                h_mag_pred_batch = render(
                    rx_positions=rx_pos_batch,
                    model=model,
                    tx_position=tx_position,
                    nt=nt,
                    nr=nr,
                    eps=args.snr_eps,
                )

                for j in range(current_batch_size):
                    h_mag_pred = h_mag_pred_batch[j].unsqueeze(0)
                    h_mag_gt = h_mag_gt_batch[j].unsqueeze(0)

                    loss = criterion(h_mag_pred, h_mag_gt).item()

                    snr_tensor = calculate_snr(
                        torch.tensor(loss, device=device),
                        h_mag_gt,
                        eps=args.snr_eps,
                    )
                    snr = snr_tensor.item()

                    if not np.isnan(loss) and not np.isinf(loss):
                        all_losses.append(loss)
                    if not np.isnan(snr) and not np.isinf(snr):
                        all_snrs.append(snr)
                    if len(sample_details) < args.num_samples:
                        sample_details.append(
                            {
                                "index": original_indices[j],
                                "rx_pos": rx_pos_batch[j].cpu().numpy(),
                                "h_mag_gt": h_mag_gt_batch[j].cpu(),
                                "h_mag_pred": h_mag_pred_batch[j].cpu(),
                                "loss": loss,
                                "snr": snr,
                            }
                        )

                progress.update(eval_task, advance=1)

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

    console.rule("[bold blue]Evaluation Results[/bold blue]")
    console.print(
        f"[green]Evaluation completed in {eval_duration:.2f} seconds.[/green]"
    )
    console.print(f"Total samples evaluated: {stats['loss']['count']}")

    overall_table = Table(box=None, show_header=False)
    overall_table.add_column("Metric", style="cyan", width=25)
    overall_table.add_column("Value", style="green")
    overall_table.add_row("Average MSE Loss (Norm)", f"{stats['loss']['mean']:.6e}")
    overall_table.add_row("Average SNR (dB)", f"{stats['snr']['mean']:.6f}")
    console.print(Panel(overall_table, title="[bold]Overall Metrics[/bold]"))

    disp_stats(console, stats)

    if args.num_samples > 0:
        disp_samples(
            console,
            sample_details,
            args.unnormalize_samples,
            min_mag,
            max_mag,
            args.norm_eps,
        )

    console.rule("[bold blue]Evaluation End[/bold blue]")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)

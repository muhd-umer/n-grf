# eval.py

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from datasets.dataloader import get_wireless_dataloader
from engine.render_channel import render_channel
from models.gaussian_model import GaussianChannelFieldModel
from utils.general_utils import set_random_seed
from utils.loss import NormalizedMSELoss, calculate_snr


def setup_eval_logging(
    log_dir: Path, checkpoint_name: str
) -> Tuple[logging.Logger, Console]:
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

    console = Console()
    console_handler = RichHandler(console=console, rich_tracebacks=True, markup=True)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger, console


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
        default=3,
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
        formatter={"complex_kind": lambda x: f"{x.real:.6f}{x.imag:+.6f}j"},
        separator=", ",
    )
    return formatted


def display_stats_table(console: Console, stats: Dict) -> None:
    """Display statistics in a rich table."""
    table = Table(title="Evaluation Statistics")

    table.add_column("Metric", style="cyan")
    table.add_column("Mean", justify="right", style="green")
    table.add_column("Std Dev", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Count", justify="right")

    table.add_row(
        "NMSE Loss",
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


def display_sample_details(console: Console, sample_details: List[Dict]) -> None:
    """Display sample details in a rich format."""
    if not sample_details:
        console.print(
            Panel(
                "[yellow]No samples were collected (evaluation might have failed early).[/yellow]",
                title="Sample Predictions",
            )
        )
        return

    console.print(
        f"\n[bold cyan]Sample Predictions (Top {len(sample_details)})[/bold cyan]"
    )

    for sample in sample_details:
        table = Table(box=None)
        table.add_column("Property", style="blue")
        table.add_column("Value")

        table.add_row("Rx Position", str(sample["rx_pos"]))
        table.add_row("NMSE Loss", f"{sample['loss']:.6e}")
        table.add_row("SNR (dB)", f"{sample['snr']:.6f}")

        panel_content = table
        panel = Panel(
            panel_content,
            title=f"[bold]Sample {sample['index']}[/bold]",
            border_style="green",
        )
        console.print(panel)

        h_gt_title = Text("Ground Truth Channel", style="cyan")
        h_pred_title = Text("Predicted Channel", style="cyan")

        console.print(h_gt_title)
        console.print(format_complex_tensor(sample["h_gt"]))
        console.print(h_pred_title)
        console.print(format_complex_tensor(sample["h_pred"]))
        console.print("")


def evaluate(args):
    """Main evaluation function."""
    set_random_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    checkpoint_path = Path(args.checkpoint)
    log_dir = Path(args.log_dir)
    checkpoint_name = checkpoint_path.stem
    logger, console = setup_eval_logging(log_dir, checkpoint_name)

    console.rule("[bold blue]Evaluation Start[/bold blue]")
    logger.info(f"Arguments: {args}")
    logger.info(f"Using device: {device}")
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    try:
        model_state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model_config = model_state["config"]
        model = GaussianChannelFieldModel.load(checkpoint_path, device=device)[0]
        model.eval()
        logger.info("[green]Model loaded successfully.[/green]")
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
        )
        metadata = val_loader.dataset.get_metadata()
        nt = metadata["num_tx_ant"]
        nr = metadata["num_rx_ant"]
        wavelength = metadata["wavelength"]
        tx_position = metadata["tx_position"].to(device)

        metadata_table = Table(title="Dataset Metadata")
        metadata_table.add_column("Property", style="cyan")
        metadata_table.add_column("Value")
        metadata_table.add_row("Transmit Antennas", str(nt))
        metadata_table.add_row("Receive Antennas", str(nr))
        metadata_table.add_row("Frequency", f"{metadata['frequency']/1e9:.2f} GHz")
        metadata_table.add_row("Wavelength", f"{wavelength:.4f} m")
        metadata_table.add_row("Dataset Size", f"{len(val_loader.dataset)}")
        console.print(metadata_table)

    except Exception as e:
        logger.exception(f"[bold red]Failed to load data:[/bold red] {e}")
        return

    criterion = NormalizedMSELoss(eps=args.loss_eps).to(device)
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
        ) as progress:
            eval_task = progress.add_task("[cyan]Evaluating...", total=len(val_loader))

            for i, batch in enumerate(val_loader):
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

    overall_table = Table(box=None)
    overall_table.add_column("Metric", style="cyan")
    overall_table.add_column("Value", style="green")
    overall_table.add_row("Average NMSE Loss", f"{stats['loss']['mean']:.6e}")
    overall_table.add_row("Average SNR (dB)", f"{stats['snr']['mean']:.6f}")
    console.print(Panel(overall_table, title="[bold]Overall Metrics[/bold]"))

    display_stats_table(console, stats)

    if args.num_samples_to_log > 0:
        display_sample_details(console, sample_details)

    console.rule("[bold blue]Evaluation End[/bold blue]")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)

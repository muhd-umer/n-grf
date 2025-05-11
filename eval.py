# eval.py

import argparse
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from datasets.dataloader import get_dataloaders
from models.gaussian_model import GaussianRadioFieldModel
from render._torch_impl import render_channel as torch_render_channel
from render._wrapper import CUDA_AVAILABLE as _WRAPPER_CUDA_COMPILED_AND_AVAILABLE
from render._wrapper import render_channel as cuda_render_channel
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


def load_cfg() -> DictConfig:
    """Loads configuration for evaluation using OmegaConf."""
    parser = argparse.ArgumentParser(description="Evaluate Gaussian radio field model")

    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset file (.mat)"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "--disable_cuda",
        action="store_true",
        help="Disable custom CUDA kernels and use PyTorch implementation for rendering.",
    )

    args, unknown_args = parser.parse_known_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config_dict" not in model_state or model_state["config_dict"] is None:

        print(f"Warning: Checkpoint {checkpoint_path} does not contain 'config_dict'.")
        print(
            "Attempting to use a default config structure. Please provide CLI overrides if needed."
        )

        train_cfg_dict = {}
    else:
        train_cfg_dict = model_state["config_dict"]

    train_cfg = OmegaConf.create(train_cfg_dict)
    cli_cfg = OmegaConf.from_cli(unknown_args)

    cfg = OmegaConf.merge(train_cfg, cli_cfg)

    cfg.checkpoint_path = args.checkpoint
    cfg.data.path = args.data_path

    if cfg.data.path is None:
        raise ValueError(
            "data.path must be provided either in config or via --data_path"
        )

    if not OmegaConf.select(cfg, "experiment"):
        with open_dict(cfg):
            cfg.experiment = OmegaConf.create()
    with open_dict(cfg.experiment):
        cfg.experiment.use_custom_cuda = not args.disable_cuda
        if not hasattr(cfg.experiment, "device"):
            cfg.experiment.device = "cuda"

    return cfg


def format_complex_tensor(tensor: torch.Tensor) -> str:
    """Formats a complex tensor for logging."""
    if not torch.is_complex(tensor):
        warnings.warn("Warning: format_complex_tensor received a non-complex tensor")

        return np.array2string(
            tensor.cpu().numpy(),
            formatter={"float_kind": lambda x: f"{x:.4f}"},
            separator=", ",
        )

    formatted = np.array2string(
        tensor.cpu().numpy(),
        formatter={"complex_kind": lambda x: f"{x.real:.4f}{x.imag:+.4f}j"},
        separator=", ",
    )
    return formatted


def unnormalize_complex(
    normalized_tensor: torch.Tensor,
    min_real: float,
    max_real: float,
    min_imag: float,
    max_imag: float,
    eps: float,
) -> torch.Tensor:
    """Un-normalizes a complex tensor using independent real/imag min-max."""
    if not torch.is_complex(normalized_tensor):
        warnings.warn("Trying to unnormalize a non-complex tensor.")
        return normalized_tensor

    norm_real = normalized_tensor.real
    norm_imag = normalized_tensor.imag

    real_range = max_real - min_real
    imag_range = max_imag - min_imag

    if abs(real_range) < eps:
        unnorm_real = torch.full_like(norm_real, (max_real + min_real) / 2)
    else:
        unnorm_real = norm_real * real_range + min_real

    if abs(imag_range) < eps:
        unnorm_imag = torch.full_like(norm_imag, (max_imag + min_imag) / 2)
    else:
        unnorm_imag = norm_imag * imag_range + min_imag

    return torch.complex(unnorm_real, unnorm_imag)


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
        "MSE Loss (Magnitude)",
        f"{stats['loss']['mean']:.6e}",
        f"{stats['loss']['std']:.6e}",
        f"{stats['loss']['min']:.6e}",
        f"{stats['loss']['max']:.6e}",
        f"{stats['loss']['count']}",
    )
    table.add_row(
        "SNR (Magnitude, dB)",
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
    cfg: DictConfig,
) -> None:
    """Display sample details in a rich format, optionally un-normalizing."""
    if not sample_details:
        console.print(
            Panel(
                "[yellow]No samples were collected (evaluation might have failed early or num_samples=0).[/yellow]",
                title="Sample predictions",
            )
        )
        return

    unnormalize = cfg.evaluation.unnormalize_samples

    min_real = sample_details[0].get("min_real", 0.0)
    max_real = sample_details[0].get("max_real", 1.0)
    min_imag = sample_details[0].get("min_imag", 0.0)
    max_imag = sample_details[0].get("max_imag", 1.0)
    norm_eps = cfg.data.norm_eps

    console.print(
        f"\n[bold cyan]Sample predictions (Top {len(sample_details)})[/bold cyan]"
    )

    for sample in sample_details:
        table = Table(box=None, show_header=False)
        table.add_column("Property", style="blue", width=20)
        table.add_column("Value")

        table.add_row("Rx Position", str(sample["rx_pos"]))
        table.add_row("MSE Loss (Mag)", f"{sample['loss']:.6e}")
        table.add_row("SNR", f"{sample['snr']:.6f}")

        panel_content = table
        panel = Panel(
            panel_content,
            title=f"[bold]Sample (Index {sample['index']})[/bold]",
            border_style="green",
        )
        console.print(panel)

        channel_gt_norm = sample["channel_gt"]
        channel_pred_norm = sample["channel_pred"]

        if unnormalize:
            channel_gt_unnorm = unnormalize_complex(
                channel_gt_norm, min_real, max_real, min_imag, max_imag, norm_eps
            )
            channel_pred_unnorm = unnormalize_complex(
                channel_pred_norm, min_real, max_real, min_imag, max_imag, norm_eps
            )
            unnorm_label = "(Un-normalized)"

            gt_title = Text(f"True Channel {unnorm_label}", style="cyan")
            pred_title = Text(f"Predicted Channel {unnorm_label}", style="cyan")
            console.print(gt_title)
            console.print(format_complex_tensor(channel_gt_unnorm))
            console.print(pred_title)
            console.print(format_complex_tensor(channel_pred_unnorm))

        else:
            gt_title = Text("True Channel (Normalized)", style="cyan")
            pred_title = Text("Predicted Channel (Normalized)", style="cyan")
            console.print(gt_title)
            console.print(format_complex_tensor(channel_gt_norm))
            console.print(pred_title)
            console.print(format_complex_tensor(channel_pred_norm))

        console.print("")


def evaluate(cfg: DictConfig):
    """Main evaluation function."""

    set_random_seed(cfg.experiment.seed)
    device = torch.device(
        cfg.experiment.device
        if torch.cuda.is_available() and cfg.experiment.device == "cuda"
        else "cpu"
    )
    checkpoint_path = Path(cfg.checkpoint_path)

    log_dir_base = cfg.experiment.get("log_dir", "logs_eval")
    log_dir = Path(log_dir_base) / "eval_runs" / checkpoint_path.stem
    checkpoint_name = checkpoint_path.stem
    logger, console = eval_logger(log_dir, checkpoint_name)

    console.rule("[bold blue]Evaluating[/bold blue]")

    actual_cuda_usage_message = ""
    if cfg.experiment.use_custom_cuda:
        if not _WRAPPER_CUDA_COMPILED_AND_AVAILABLE:
            logger.warning(
                "Custom CUDA kernels requested, but they are not compiled/available in _wrapper. Falling back to PyTorch implementation."
            )
            console.print(
                "[yellow]Warning: Custom CUDA kernels requested, but not compiled/available. Falling back to PyTorch.[/yellow]"
            )
            render_channel_fn = torch_render_channel
            actual_cuda_usage_message = (
                "PyTorch (_torch_impl, fallback from uncompiled custom CUDA)"
            )
        elif device.type != "cuda":
            logger.warning(
                "Custom CUDA kernels requested, but selected device is CPU. Falling back to PyTorch implementation."
            )
            console.print(
                "[yellow]Warning: Custom CUDA kernels requested, but device is CPU. Falling back to PyTorch.[/yellow]"
            )
            render_channel_fn = torch_render_channel
            actual_cuda_usage_message = (
                "PyTorch (_torch_impl, fallback due to CPU device)"
            )
        else:
            render_channel_fn = cuda_render_channel
            actual_cuda_usage_message = "Custom CUDA (_wrapper)"
    else:
        render_channel_fn = torch_render_channel
        actual_cuda_usage_message = "PyTorch (_torch_impl, user-disabled custom CUDA)"

    logger.info(f"Using render_channel implementation: {actual_cuda_usage_message}")
    console.print(
        f"Using render_channel implementation: [bold]{actual_cuda_usage_message}[/bold]"
    )

    config_str = OmegaConf.to_yaml(cfg)
    syntax = Syntax(config_str, "yaml", background_color="default", line_numbers=True)
    console.print(Panel(syntax, title="Configuration", expand=False))

    logger.info(f"Using device: {device}")
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    try:

        model, load_iter = GaussianRadioFieldModel.load(
            checkpoint_path,
            device=device,
            resume_cfg=None,
        )
        model.eval()
        logger.info(
            f"[green]Model loaded successfully from iteration {load_iter}.[/green]"
        )
        logger.info(f"Total Gaussians in loaded model: {model.get_xyz.shape[0]:,}")
    except Exception as e:
        logger.exception(f"[bold red]Failed to load checkpoint:[/bold red] {e}")
        return

    logger.info(f"Loading data from: {cfg.data.path}")
    try:

        _, val_loader, metadata = get_dataloaders(cfg=cfg)

        if len(val_loader) == 0:
            logger.error(
                "Validation dataloader is empty. Cannot proceed with evaluation."
            )
            return

        nt = metadata["num_tx_ant"]
        nr = metadata["num_rx_ant"]

        if nt != model.num_tx_ant or nr != model.num_rx_ant:
            logger.error(
                f"Mismatch between dataset metadata (Nt={nt}, Nr={nr}) and loaded model (Nt={model.num_tx_ant}, Nr={model.num_rx_ant})."
            )

            logger.warning(
                f"Using Nt={model.num_tx_ant}, Nr={model.num_rx_ant} from loaded model."
            )
            nt = model.num_tx_ant
            nr = model.num_rx_ant

        tx_position = metadata["tx_position"].to(device)

        min_real = metadata.get("min_real", 0.0)
        max_real = metadata.get("max_real", 1.0)
        min_imag = metadata.get("min_imag", 0.0)
        max_imag = metadata.get("max_imag", 1.0)

        metadata_table = Table(title="Dataset Metadata")
        metadata_table.add_column("Property", style="cyan")
        metadata_table.add_column("Value")
        metadata_table.add_row("Dataset Path", cfg.data.path)
        metadata_table.add_row("Transmit Antennas (Nt)", str(nt))
        metadata_table.add_row("Receive Antennas (Nr)", str(nr))
        metadata_table.add_row("Frequency", f"{metadata['frequency']/1e9:.2f} GHz")
        metadata_table.add_row("Is SISO", str(metadata["is_siso"]))
        metadata_table.add_row("Tx Position", str(tx_position.cpu().numpy()))
        metadata_table.add_row("Norm Real Min/Max", f"{min_real:.4e} / {max_real:.4e}")
        metadata_table.add_row("Norm Imag Min/Max", f"{min_imag:.4e} / {max_imag:.4e}")
        metadata_table.add_row("Test Set Size", f"{len(val_loader.dataset)}")
        console.print(metadata_table)

    except Exception as e:
        logger.exception(f"[bold red]Failed to load data or metadata:[/bold red] {e}")
        return

    criterion = nn.MSELoss().to(device)
    all_losses = []
    all_snrs = []
    sample_details: List[Dict] = []
    snr_eps = cfg.evaluation.snr_eps
    sample_idx_counter = 0

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
            for _, batch in enumerate(val_loader):
                rx_pos_batch = batch["rx_position"].to(device)

                channel_gt_batch = batch["channel"].to(device)
                current_batch_size = rx_pos_batch.shape[0]

                channel_pred_batch = render_channel_fn(
                    rx_positions=rx_pos_batch,
                    model=model,
                    tx_position=tx_position,
                    nt=nt,
                    nr=nr,
                    eps=snr_eps,
                )

                for j in range(current_batch_size):
                    channel_pred = channel_pred_batch[j].unsqueeze(0)
                    channel_gt = channel_gt_batch[j].unsqueeze(0)

                    pred_mag = torch.abs(channel_pred)
                    gt_mag = torch.abs(channel_gt)
                    loss_tensor = criterion(pred_mag, gt_mag)
                    loss = loss_tensor.item()

                    snr_tensor = calculate_snr(loss_tensor, gt_mag, eps=snr_eps)
                    snr = snr_tensor.item()

                    if not np.isnan(loss) and not np.isinf(loss):
                        all_losses.append(loss)
                    if not np.isnan(snr) and not np.isinf(snr):
                        all_snrs.append(snr)

                    if len(sample_details) < cfg.evaluation.num_samples:
                        sample_details.append(
                            {
                                "index": sample_idx_counter,
                                "rx_pos": rx_pos_batch[j].cpu().numpy(),
                                "channel_gt": channel_gt_batch[j].cpu(),
                                "channel_pred": channel_pred_batch[j].cpu(),
                                "loss": loss,
                                "snr": snr,
                                "min_real": min_real,
                                "max_real": max_real,
                                "min_imag": min_imag,
                                "max_imag": max_imag,
                            }
                        )
                    sample_idx_counter += 1

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
    overall_table.add_row("Average MSE Loss", f"{stats['loss']['mean']:.6e}")
    overall_table.add_row("Average SNR", f"{stats['snr']['mean']:.6f}")
    console.print(Panel(overall_table, title="[bold]Metrics[/bold]"))

    disp_stats(console, stats)
    if cfg.evaluation.num_samples > 0:
        disp_samples(console, sample_details, cfg)

    console.rule("[bold blue]Evaluation end[/bold blue]")


if __name__ == "__main__":
    cfg = load_cfg()
    evaluate(cfg)

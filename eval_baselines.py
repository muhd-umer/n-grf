# eval_baselines.py

import argparse
import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.nn.modules.transformer"
)

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from baselines.knn import KNNBaseline
from baselines.mdn import MDNBaseline
from baselines.mlp import MLPBaseline
from baselines.transformer import TransformerBaseline
from datasets.dataloader import get_dataloaders
from utils.general_utils import set_random_seed
from utils.loss import get_snr_fnmse, nmse


def setup_logging(run_name, log_dir):
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{run_name}.log"

    logger = logging.getLogger("BaselineLogger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console = Console()
    console_handler = RichHandler(
        console=console, rich_tracebacks=True, markup=True, show_path=False
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger, console


def evaluate_model(model, val_loader, criterion, device, snr_eps=1e-10):
    """
    Evaluate the model on the validation set

    Args:
        model: The model to evaluate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        snr_eps: Epsilon for SNR calculation

    Returns:
        Dictionary with evaluation metrics
    """
    if hasattr(model, "eval"):
        model.eval()

    total_loss = 0.0
    total_snr = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            rx_positions = batch["rx_position"].to(device)
            channel_gt = batch["channel"].to(device)
            batch_size = rx_positions.shape[0]

            channel_pred = model.predict(rx_positions)

            pred_mag = torch.abs(channel_pred)
            gt_mag = torch.abs(channel_gt)

            loss = criterion(channel_pred, channel_gt)
            snr = get_snr_fnmse(loss, channel_gt, eps=snr_eps)

            total_loss += loss.item() * batch_size
            if not torch.isinf(snr) and not torch.isnan(snr):
                total_snr += snr.item() * batch_size
            count += batch_size

    avg_loss = total_loss / count if count > 0 else 0.0
    avg_snr = total_snr / count if count > 0 else float("-inf")

    return {"val_mse_loss": avg_loss, "val_snr_db": avg_snr}


def create_model(model_type, metadata, device, **model_args):
    """
    Create a model instance based on type

    Args:
        model_type: Type of model to create ('knn', 'mdn', 'mlp', 'transformer')
        metadata: Dataset metadata
        device: Device to use
        **model_args: Additional arguments for model initialization

    Returns:
        Initialized model
    """
    num_tx_ant = metadata["num_tx_ant"]
    num_rx_ant = metadata["num_rx_ant"]

    if model_type == "knn":
        k = model_args.get("k", 5)
        weights = model_args.get("weights", "distance")
        return KNNBaseline(k=k, weights=weights)

    elif model_type == "mdn":
        input_dim = model_args.get("input_dim", 3)
        hidden_dims = model_args.get("hidden_dims", [256, 256, 128])
        n_mixtures = model_args.get("n_mixtures", 5)
        dropout_p = model_args.get("dropout_p", 0.1)
        learning_rate = model_args.get("learning_rate", 0.001)
        weight_decay = model_args.get("weight_decay", 1e-6)

        return MDNBaseline(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_tx_ant=num_tx_ant,
            num_rx_ant=num_rx_ant,
            n_mixtures=n_mixtures,
            dropout_p=dropout_p,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
        )

    elif model_type == "mlp":
        input_dim = model_args.get("input_dim", 3)
        hidden_dims = model_args.get("hidden_dims", [256, 256, 256, 128])
        dropout_p = model_args.get("dropout_p", 0.1)
        use_layer_norm = model_args.get("use_layer_norm", True)
        learning_rate = model_args.get("learning_rate", 0.001)
        weight_decay = model_args.get("weight_decay", 1e-5)

        return MLPBaseline(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_tx_ant=num_tx_ant,
            num_rx_ant=num_rx_ant,
            dropout_p=dropout_p,
            use_layer_norm=use_layer_norm,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
        )

    elif model_type == "transformer":
        input_dim = model_args.get("input_dim", 3)
        num_tokens = model_args.get("num_tokens", 8)
        d_model = model_args.get("d_model", 128)
        nhead = model_args.get("nhead", 4)
        num_encoder_layers = model_args.get("num_encoder_layers", 3)
        num_decoder_layers = model_args.get("num_decoder_layers", 3)
        dim_feedforward = model_args.get("dim_feedforward", 512)
        dropout_p = model_args.get("dropout_p", 0.1)
        learning_rate = model_args.get("learning_rate", 0.001)
        weight_decay = model_args.get("weight_decay", 1e-5)

        return TransformerBaseline(
            input_dim=input_dim,
            num_tx_ant=num_tx_ant,
            num_rx_ant=num_rx_ant,
            num_tokens=num_tokens,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout_p=dropout_p,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train and evaluate baseline models")

    # dataset arguments
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset file (.mat)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of data for training vs validation",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Whether to normalize the channel data",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name of the run for logging purposes",
    )

    # model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["knn", "mdn", "mlp", "transformer"],
        help="Type of baseline model to use",
    )

    # training arguments
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=5,
        help="Frequency of evaluation during training (epochs)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory for saving logs and models",
    )

    # sys arguments
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )

    # model specific arguments
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default=None,
        help="Comma-separated list of hidden dimensions",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer"
    )

    # KNN arguments
    parser.add_argument(
        "--k", type=int, default=5, help="Number of nearest neighbors for KNN"
    )

    # MDN arguments
    parser.add_argument(
        "--n_mixtures", type=int, default=5, help="Number of mixtures for MDN"
    )

    # Transformer arguments
    parser.add_argument(
        "--num_tokens", type=int, default=8, help="Number of tokens for transformer"
    )
    parser.add_argument(
        "--d_model", type=int, default=128, help="Model dimension for transformer"
    )
    parser.add_argument(
        "--nhead", type=int, default=4, help="Number of attention heads for transformer"
    )

    args = parser.parse_args()

    if args.hidden_dims:
        args.hidden_dims = [int(dim) for dim in args.hidden_dims.split(",")]

    return args


def create_config(args):
    """
    Create configuration object compatible with get_dataloaders

    Args:
        args: Command line arguments

    Returns:
        OmegaConf configuration object
    """
    cfg_dict = {
        "data": {
            "path": args.data_path,
            "train_ratio": args.train_ratio,
            "norm_eps": 1e-8,
            "num_workers": args.num_workers,
            "normalize": args.normalize,
        },
        "training": {"batch_size": args.batch_size, "snr_eps": 1e-10},
        "evaluation": {"batch_size": args.batch_size, "snr_eps": 1e-10},
        "experiment": {
            "seed": args.seed,
            "device": args.device,
            "name": f"baseline_{args.model_type}",
        },
    }

    return OmegaConf.create(cfg_dict)


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{args.model_type}_{timestamp}" if args.run_name is None else args.run_name
    )
    log_dir = Path(args.log_dir) / run_name
    logger, console = setup_logging(run_name, log_dir)

    model_dir = log_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold blue]Starting baseline run: {run_name}[/bold blue]")
    console.print(
        Panel(
            f"Model: {args.model_type}\n"
            f"Data path: {args.data_path}\n"
            f"Log directory: {log_dir}\n"
            f"Device: {device}\n"
            f"Normalize data: {args.normalize}",
            title="Setup",
            expand=False,
        )
    )

    logger.info("Loading data...")
    try:
        cfg = create_config(args)
        train_loader, val_loader, metadata = get_dataloaders(cfg)

        console.print(f"Dataset loaded: {args.data_path}")
        console.print(f"Training samples: {len(train_loader.dataset)}")
        console.print(f"Validation samples: {len(val_loader.dataset)}")
        console.print(
            f"Tx antennas: {metadata['num_tx_ant']}, Rx antennas: {metadata['num_rx_ant']}"
        )
        console.print(f"SISO model: {metadata['is_siso']}")

    except Exception as e:
        logger.exception(f"Failed to load data: {e}")
        return

    logger.info(f"Creating {args.model_type} model...")
    try:

        model_args = {
            "input_dim": 3,
        }

        if args.hidden_dims:
            model_args["hidden_dims"] = args.hidden_dims

        model_args["learning_rate"] = args.learning_rate
        model_args["weight_decay"] = args.weight_decay

        if args.model_type == "knn":
            model_args["k"] = args.k

        elif args.model_type == "mdn":
            model_args["n_mixtures"] = args.n_mixtures

        elif args.model_type == "transformer":
            model_args["num_tokens"] = args.num_tokens
            model_args["d_model"] = args.d_model
            model_args["nhead"] = args.nhead

        model = create_model(args.model_type, metadata, device, **model_args)

        if hasattr(model, "to"):
            model.to(device)

        console.print(f"[green]Model created successfully![/green]")

    except Exception as e:
        logger.exception(f"Failed to create model: {e}")
        return

    criterion = nmse

    logger.info("Starting training...")
    best_val_snr = float("-inf")
    best_val_loss = float("inf")
    best_epoch = -1
    training_losses = []
    validation_results = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Training {args.model_type}...", total=args.epochs
        )

        for epoch in range(args.epochs):

            if args.model_type != "knn":
                epoch_start_time = time.time()

                if hasattr(model, "train"):
                    model.train()

                rx_positions = []
                channels = []
                for batch in train_loader:
                    rx_positions.append(batch["rx_position"])
                    channels.append(batch["channel"])

                rx_positions = torch.cat(rx_positions, dim=0)
                channels = torch.cat(channels, dim=0)

                epoch_loss = model.fit(
                    rx_positions,
                    channels,
                    batch_size=args.batch_size,
                    epochs=1,
                    verbose=False,
                )

                training_losses.append(
                    epoch_loss[0] if isinstance(epoch_loss, list) else epoch_loss
                )

                epoch_time = time.time() - epoch_start_time
                logger.info(
                    f"Epoch {epoch+1}/{args.epochs} - Loss: {training_losses[-1]:.6f} - Time: {epoch_time:.2f}s"
                )

            else:
                if epoch == 0:
                    epoch_start_time = time.time()

                    rx_positions = []
                    channels = []
                    for batch in train_loader:
                        rx_positions.append(batch["rx_position"])
                        channels.append(batch["channel"])

                    rx_positions = torch.cat(rx_positions, dim=0)
                    channels = torch.cat(channels, dim=0)

                    model.fit(rx_positions, channels)

                    epoch_time = time.time() - epoch_start_time
                    logger.info(f"KNN model fitted in {epoch_time:.2f}s")

            progress.update(
                task, advance=1, description=f"[cyan]Training {args.model_type}..."
            )

        logger.info("Training completed. Running final evaluation...")
        eval_start_time = time.time()
        eval_metrics = evaluate_model(model, val_loader, criterion, device)
        eval_time = time.time() - eval_start_time

        val_loss = eval_metrics["val_mse_loss"]
        val_snr = eval_metrics["val_snr_db"]

        eval_metrics["epoch"] = args.epochs
        eval_metrics["time_sec"] = eval_time
        validation_results.append(eval_metrics)

        logger.info(
            f"Final Validation | Loss={val_loss:.6f} | SNR={val_snr:.2f} dB | Time={eval_time:.2f}s"
        )

        best_val_snr = val_snr
        best_val_loss = val_loss
        best_epoch = args.epochs

        best_model_path = model_dir / "best_model.pt"
        model.save(best_model_path)
        logger.info(f"Final model saved with SNR: {best_val_snr:.2f} dB")

    logger.info(f"Training and evaluation completed after {args.epochs} epochs")

    final_model_path = model_dir / "final_model.pt"
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    if best_epoch != -1:
        console.print(f"[bold green]Best validation results:[/bold green]")
        console.print(f"Epoch: {best_epoch}/{args.epochs}")
        console.print(f"Loss: {best_val_loss:.6f}")
        console.print(f"SNR: {best_val_snr:.2f} dB")

        results = {
            "model_type": args.model_type,
            "data_path": args.data_path,
            "normalize": args.normalize,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_snr": best_val_snr,
            "training_losses": training_losses if args.model_type != "knn" else None,
            "validation_results": validation_results,
            "model_args": model_args,
            "metadata": {
                "num_tx_ant": metadata["num_tx_ant"],
                "num_rx_ant": metadata["num_rx_ant"],
                "is_siso": metadata["is_siso"],
                "frequency": metadata["frequency"],
                "normalize": metadata["normalize"],
            },
        }

        with open(log_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    if validation_results:
        val_table = Table(title="Validation Results")
        val_table.add_column("Epoch", style="dim")
        val_table.add_column("Loss", style="magenta")
        val_table.add_column("SNR (dB)", style="green")
        val_table.add_column("Time (s)")

        for res in validation_results:
            val_table.add_row(
                str(res["epoch"]),
                f"{res['val_mse_loss']:.6f}",
                f"{res['val_snr_db']:.2f}",
                f"{res['time_sec']:.2f}",
            )

        console.print(val_table)

    console.rule("[bold blue]Baseline evaluation complete[/bold blue]")


if __name__ == "__main__":
    main()

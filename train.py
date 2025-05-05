# train.py

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.syntax import Syntax
from rich.table import Table
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.dataloader import get_dataloaders
from engine.render import render
from models.gaussian_model import GaussianChannelFieldModel
from utils.general_utils import set_random_seed
from utils.loss import calculate_snr
from utils.train_utils import compute_grad_stats, setup_logging


def load_config() -> DictConfig:
    """Loads configuration using OmegaConf."""
    parser = argparse.ArgumentParser(description="Train Gaussian channel field model")

    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset file (.mat)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the configuration file",
    )

    args, unknown_args = parser.parse_known_args()
    default_cfg = OmegaConf.load("configs/default.yaml")
    user_cfg = (
        OmegaConf.load(args.config)
        if Path(args.config).exists()
        else OmegaConf.create()
    )

    cli_cfg = OmegaConf.from_cli(unknown_args)
    cfg = OmegaConf.merge(default_cfg, user_cfg, cli_cfg)
    cfg.data.path = args.data_path

    if cfg.data.path is None:
        raise ValueError(
            "data.path must be provided either in config or via --data_path"
        )

    return cfg


def evaluate(
    model: GaussianChannelFieldModel,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tx_position: torch.Tensor,
    nt: int,
    nr: int,
    cfg: DictConfig,
) -> Dict[str, float]:
    """Evaluates the model on the validation set."""
    model.eval()
    total_loss = 0.0
    total_snr = 0.0
    count = 0
    snr_eps = cfg.training.snr_eps

    with torch.no_grad():
        for batch in val_loader:
            rx_pos_batch = batch["rx_position"].to(device)

            cmr_gt_batch = batch["cmr"].to(device)
            batch_size = rx_pos_batch.shape[0]

            cmr_pred_batch = render(
                rx_positions=rx_pos_batch,
                model=model,
                tx_position=tx_position,
                nt=nt,
                nr=nr,
                eps=snr_eps,
            )

            loss = criterion(cmr_pred_batch, cmr_gt_batch)
            snr = calculate_snr(loss, cmr_gt_batch, eps=snr_eps)
            total_loss += loss.item() * batch_size

            if not torch.isinf(snr) and not torch.isnan(snr):
                total_snr += snr.item() * batch_size

            count += batch_size

    avg_loss = total_loss / count if count > 0 else 0.0
    avg_snr = total_snr / count if count > 0 else float("-inf")

    return {"val_mse_loss": avg_loss, "val_snr_db": avg_snr}


def train(cfg: DictConfig):
    """Main training loop."""
    set_random_seed(cfg.experiment.seed)
    device = torch.device(
        cfg.experiment.device
        if torch.cuda.is_available() and cfg.experiment.device == "cuda"
        else "cpu"
    )

    run_name = cfg.experiment.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(cfg.experiment.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = log_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    config_save_path = log_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        OmegaConf.save(cfg, f)

    logger, console = setup_logging(log_dir, use_rich=True)
    if console is None:
        raise RuntimeError("Failed to initialize Rich Console.")
    writer = (
        SummaryWriter(str(log_dir / "tensorboard"))
        if cfg.experiment.tensorboard
        else None
    )

    console.rule(f"[bold blue]Starting training run: {run_name}[/bold blue]")
    console.print(
        Panel(
            f"Log directory: {log_dir}\nDevice: {device}\nConfig: {config_save_path}",
            title="Setup",
            expand=False,
        )
    )

    config_str = OmegaConf.to_yaml(cfg)
    syntax = Syntax(config_str, "yaml", background_color="default", line_numbers=True)
    console.print(Panel(syntax, title="Configuration", expand=False))

    logger.info("Loading data...")
    try:
        train_loader, val_loader, metadata = get_dataloaders(cfg=cfg)

        nt = metadata["num_tx_ant"]
        nr = metadata["num_rx_ant"]
        tx_position = metadata["tx_position"].to(device)
        env_dims = metadata.get("env_dims")
        point_cloud = metadata.get("point_cloud")
        min_mag = metadata.get("min_magnitude", 0.0)
        max_mag = metadata.get("max_magnitude", 1.0)

        metadata_table = Table(title="Dataset Metadata", show_header=False, box=None)
        metadata_table.add_column("Property", style="cyan")
        metadata_table.add_column("Value")
        metadata_table.add_row("Path", cfg.data.path)
        metadata_table.add_row("Nt", str(nt))
        metadata_table.add_row("Nr", str(nr))
        metadata_table.add_row("Frequency", f"{metadata['frequency']/1e9:.2f} GHz")
        metadata_table.add_row("Is SISO", str(metadata["is_siso"]))
        metadata_table.add_row("Norm Min/Max", f"{min_mag:.4e} / {max_mag:.4e}")
        console.print(metadata_table)

        logger.info(f"Transmitter position: {tx_position.cpu().numpy()}")
        if point_cloud is not None:
            logger.info(f"Point cloud loaded with shape: {point_cloud.shape}")
            point_cloud = point_cloud.to(device)
        else:
            logger.info("No point cloud data found or used for initialization")
        if env_dims is not None:
            logger.info(f"Environment dimensions loaded: {env_dims.numpy().tolist()}")
            env_dims = env_dims.to(device)
        else:
            logger.warning(
                "No environment dimensions found. Random init will use default range [-1, 1]."
            )

        if len(train_loader) == 0:
            logger.error(
                "Training dataloader is empty! Check dataset path and train_ratio."
            )
            if writer:
                writer.close()
            return
        if len(val_loader) == 0:
            logger.warning(
                "Validation dataloader is empty! Evaluation will be skipped."
            )

    except Exception as e:
        logger.exception(f"Failed to load data: {e}")
        if writer:
            writer.close()
        return

    logger.info("Initializing model...")
    model = GaussianChannelFieldModel(
        num_tx_ant=nt,
        num_rx_ant=nr,
        latent_dim=cfg.model.latent_dim,
        attribute_hidden_dim=cfg.model.attribute_network.hidden_dim,
        attribute_num_layers=cfg.model.attribute_network.num_layers,
        attribute_pos_enc_freqs=cfg.model.attribute_network.pos_enc_freqs,
        decoder_hidden_dim=cfg.model.contribution_decoder.hidden_dim,
        decoder_num_layers=cfg.model.contribution_decoder.num_layers,
        initial_gaussians=cfg.initialization.num_gaussians,
        init_opacity_value=cfg.initialization.opacity_value,
        init_scale_value=cfg.initialization.scale_value,
        device=device,
    )

    start_iteration = 0

    resume_path_str = cfg.get("resume", None)
    resume_path = Path(resume_path_str) if resume_path_str else None

    if resume_path and resume_path.exists():
        logger.info(f"Resuming from checkpoint: {resume_path}")
        try:
            model, start_iteration = GaussianChannelFieldModel.load(
                resume_path,
                device,
                resume_cfg=cfg,
            )
            start_iteration += 1
            logger.info(f"Resumed from iteration {start_iteration -1}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}. Starting from scratch")
            resume_path = None
    else:
        if resume_path:
            logger.warning(
                f"Resume checkpoint not found at {resume_path}. Starting from scratch."
            )
        resume_path = None

    if not resume_path:
        init_pc_arg = None
        if cfg.initialization.method == "point_cloud":
            if point_cloud is not None:
                logger.info("Using point cloud for Gaussian initialization")
                init_pc_arg = point_cloud
            else:
                logger.warning(
                    "Point cloud initialization requested but no point cloud data found. Falling back to random initialization."
                )
                cfg.initialization.method = "random"

        if cfg.initialization.method == "random":
            logger.info("Using random initialization for Gaussians")
            model.init_gaussians(
                env_dims=env_dims if env_dims is not None else None,
                point_cloud=None,
                num_points=cfg.initialization.num_gaussians,
            )
        elif cfg.initialization.method == "point_cloud":
            model.init_gaussians(
                env_dims=None,
                point_cloud=init_pc_arg,
                num_points=cfg.initialization.num_gaussians,
            )
        model.training_setup(cfg)

    model = model.to(device)
    logger.info(f"Model initialized/loaded with {model.get_xyz.shape[0]} Gaussians")
    logger.info(
        f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    criterion = nn.MSELoss().to(device)
    logger.info("Using MSE Loss for training")

    validation_results: List[Dict] = []
    max_val_disp = 4
    best_val_snr = float("-inf")
    best_val_loss = float("inf")
    best_iteration = -1

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("Loss={task.fields[loss]:.3e}, "),
        TextColumn("SNR={task.fields[snr]:.2f} dB"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    ema_loss = -1.0
    train_iter = iter(train_loader)

    stop_xyz_iter = int(cfg.training.iterations * cfg.training.stop_xyz_iter_ratio)
    logger.info(
        f"Will stop updating Gaussian XYZ positions after iteration {stop_xyz_iter}"
    )

    logger.info(f"Starting training loop from iteration {start_iteration}...")
    with progress:
        task = progress.add_task(
            "[cyan]Training...",
            total=cfg.training.iterations,
            completed=start_iteration,
            loss=float("nan"),
            snr=float("nan"),
        )

        for iteration in range(start_iteration, cfg.training.iterations):
            iter_start_time = time.time()
            model.train()

            model.update_learning_rate(iteration, cfg)

            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            rx_pos_batch = batch["rx_position"].to(device)
            cmr_gt_batch = batch["cmr"].to(device)

            if cfg.training.rx_noise_std > 0:
                noise = torch.randn_like(rx_pos_batch) * cfg.training.rx_noise_std
                rx_pos_batch = rx_pos_batch + noise

            cmr_pred_batch = render(
                rx_positions=rx_pos_batch,
                model=model,
                tx_position=tx_position,
                nt=nt,
                nr=nr,
                eps=cfg.training.snr_eps,
            )

            mse_loss = criterion(cmr_pred_batch, cmr_gt_batch)
            total_loss = mse_loss
            l1_activation_loss = torch.tensor(0.0, device=device)

            if cfg.training.lambda_activation_l1 > 0 and model.get_xyz.shape[0] > 0:
                base_activations_logits = model.get_base_activation_logits(tx_position)
                l1_activation_loss = torch.mean(torch.abs(base_activations_logits))
                total_loss = (
                    total_loss + cfg.training.lambda_activation_l1 * l1_activation_loss
                )

            model.optimizer.zero_grad()
            total_loss.backward()

            found_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                ):
                    logger.warning(
                        f"NaN or Inf gradient detected at iteration {iteration} for parameter '{name}'. Skipping optimizer step."
                    )
                    found_nan_grad = True
                    break

            grad_stats = {}
            if not found_nan_grad:
                grad_stats = compute_grad_stats(model)
                model.optimizer.step()
            else:
                model.optimizer.zero_grad()
                logger.warning(
                    f"NaN/Inf gradient detected at iter {iteration}. Skipping optimizer step."
                )

            iter_time = time.time() - iter_start_time
            with torch.no_grad():
                current_loss = mse_loss.item()
                if np.isnan(current_loss) or np.isinf(current_loss):
                    logger.warning(
                        f"NaN or Inf loss detected at iteration {iteration}. Resetting EMA loss."
                    )
                    ema_loss = -1.0
                elif ema_loss < 0:
                    ema_loss = current_loss
                else:
                    ema_loss = 0.95 * ema_loss + 0.05 * current_loss

                snr = calculate_snr(
                    mse_loss, cmr_gt_batch, eps=cfg.training.snr_eps
                ).item()
                num_gaussians = model.get_xyz.shape[0]

                progress.update(task, advance=1, loss=current_loss, snr=snr)
                if iteration % cfg.experiment.log_freq == 0:
                    log_msg_file = (
                        f"[{iteration}/{cfg.training.iterations}] <<< "
                        f"Loss={current_loss:.4e} | EMA={ema_loss:.4e} | "
                        f"SNR={snr:.2f} dB | Time={iter_time:.3f}s >>>"
                    )
                    logger.info(log_msg_file)
                    if cfg.experiment.log_grad_stats and grad_stats:
                        grad_table = Table(
                            title=f"Gradient stats",
                            box=None,
                            show_header=False,
                        )
                        grad_table.add_column("Metric", style="dim cyan")
                        grad_table.add_column("Value", justify="right")
                        grad_table.add_row(
                            "Norm (L2)", f"{grad_stats['grad_norm']:.3e}"
                        )
                        grad_table.add_row(
                            "Mean Abs", f"{grad_stats['mean_abs_grad']:.3e}"
                        )
                        grad_table.add_row("Min", f"{grad_stats['min_grad']:.3e}")
                        grad_table.add_row("Max", f"{grad_stats['max_grad']:.3e}")
                        grad_table.add_row(
                            "Param Count", f"{grad_stats['param_count_with_grad']}"
                        )
                        console.print(Padding(grad_table, (0, 0, 0, 2)))

                    if writer is not None:
                        writer.add_scalar("train/mse_loss", current_loss, iteration)
                        writer.add_scalar("train/ema_loss", ema_loss, iteration)
                        writer.add_scalar("train/snr_db", snr, iteration)
                        writer.add_scalar(
                            "train/iteration_time_sec", iter_time, iteration
                        )
                        writer.add_scalar(
                            "train/num_gaussians", num_gaussians, iteration
                        )
                        if cfg.training.lambda_activation_l1 > 0:
                            writer.add_scalar(
                                "train/l1_activation_loss",
                                l1_activation_loss.item(),
                                iteration,
                            )
                        if model.optimizer:
                            for i, param_group in enumerate(
                                model.optimizer.param_groups
                            ):
                                writer.add_scalar(
                                    f"lr/{param_group['name']}",
                                    param_group["lr"],
                                    iteration,
                                )
                        if grad_stats:
                            writer.add_scalar(
                                "grads/norm", grad_stats["grad_norm"], iteration
                            )
                            writer.add_scalar(
                                "grads/mean_abs", grad_stats["mean_abs_grad"], iteration
                            )
                            writer.add_scalar(
                                "grads/min", grad_stats["min_grad"], iteration
                            )
                            writer.add_scalar(
                                "grads/max", grad_stats["max_grad"], iteration
                            )

                if (
                    iteration % cfg.experiment.eval_freq == 0 and iteration > 0
                ) or iteration == cfg.training.iterations - 1:
                    if len(val_loader) > 0:
                        progress.update(task, description="[yellow]Evaluating...")
                        eval_start_time = time.time()
                        eval_metrics = evaluate(
                            model=model,
                            val_loader=val_loader,
                            criterion=criterion,
                            device=device,
                            tx_position=tx_position,
                            nt=nt,
                            nr=nr,
                            cfg=cfg,
                        )
                        eval_time = time.time() - eval_start_time
                        progress.update(task, description="[cyan]Training...")

                        eval_metrics["iteration"] = iteration
                        eval_metrics["time_sec"] = eval_time
                        validation_results.append(eval_metrics)
                        if len(validation_results) > max_val_disp:
                            validation_results = validation_results[-max_val_disp:]

                        logger.info(
                            f"Validation @ {iteration} | Loss={eval_metrics['val_mse_loss']:.4e} | SNR={eval_metrics['val_snr_db']:.2f} dB | Time={eval_time:.2f}s"
                        )
                        current_val_snr = eval_metrics["val_snr_db"]
                        if current_val_snr > best_val_snr:
                            best_val_snr = current_val_snr
                            best_val_loss = eval_metrics["val_mse_loss"]
                            best_iteration = iteration
                            best_model_path = log_dir / "best_model.pt"
                            model.save(best_model_path, iteration=iteration, cfg=cfg)
                            logger.info(
                                f"New best validation SNR: {best_val_snr:.2f} dB at iter {best_iteration}. Saved model to {best_model_path}"
                            )

                        val_table = Table(
                            title=f"Validation Results (Last {len(validation_results)})",
                            expand=False,
                        )
                        val_table.add_column("Iter", style="dim", justify="right")
                        val_table.add_column(
                            "MSE Loss", style="magenta", justify="right"
                        )
                        val_table.add_column("SNR (dB)", style="green", justify="right")
                        val_table.add_column("Time (s)", justify="right")
                        for res in validation_results:
                            val_table.add_row(
                                f"{res['iteration']}",
                                f"{res['val_mse_loss']:.4e}",
                                f"{res['val_snr_db']:.2f}",
                                f"{res['time_sec']:.2f}",
                            )

                        if best_iteration != -1:
                            val_table.add_section()
                            val_table.add_row(
                                f"[bold cyan]Best @ {best_iteration}[/bold cyan]",
                                f"[bold magenta]{best_val_loss:.4e}[/bold magenta]",
                                f"[bold green]{best_val_snr:.2f}[/bold green]",
                                "---",
                            )
                        console.print(val_table)

                        if writer is not None:
                            writer.add_scalar(
                                "validation/mse_loss",
                                eval_metrics["val_mse_loss"],
                                iteration,
                            )
                            writer.add_scalar(
                                "validation/snr_db",
                                eval_metrics["val_snr_db"],
                                iteration,
                            )
                    else:
                        if iteration % (cfg.experiment.eval_freq * 5) == 0:
                            logger.info(
                                f"Skipping evaluation at iteration {iteration} (empty val loader)"
                            )
                if (
                    iteration % cfg.experiment.checkpoint_freq == 0 and iteration > 0
                ) or iteration == cfg.training.iterations - 1:
                    checkpoint_path = checkpoints_dir / f"checkpoint_{iteration:07d}.pt"

                    model.save(checkpoint_path, iteration=iteration, cfg=cfg)
                    logger.info(f"Checkpoint saved to {checkpoint_path}")

    progress.stop()
    final_model_path = log_dir / "final_model.pt"

    model.save(final_model_path, iteration=cfg.training.iterations - 1, cfg=cfg)
    logger.info(f"Training completed after {cfg.training.iterations} iterations")
    logger.info(f"Final model saved to {final_model_path}")

    console.rule("[bold blue]Training end[/bold blue]")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    cfg = load_config()
    train(cfg)

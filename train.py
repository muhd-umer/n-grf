# train.py

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.dataloader import get_dataloaders
from engine.render_channel import render_channel
from models.gaussian_model import GaussianChannelFieldModel
from utils.general_utils import set_random_seed
from utils.loss import NormalizedMSELoss, calculate_snr
from utils.train_utils import compute_grad_stats, setup_logging


def parse_args():
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description="Train Gaussian Channel Field Model")

    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset file (.mat)"
    )

    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Ratio of data for training"
    )

    parser.add_argument(
        "--initial_gaussians",
        type=int,
        default=3_000,
        help="Number of Gaussians to initialize randomly",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=32,
        help="Dimension of Gaussian latent features (F)",
    )
    parser.add_argument(
        "--decoder_hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension for decoder MLP",
    )
    parser.add_argument(
        "--decoder_num_layers",
        type=int,
        default=4,
        help="Number of layers for decoder MLP (including output)",
    )

    parser.add_argument(
        "--iterations", type=int, default=30_000, help="Total training iterations"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--position_lr_init",
        type=float,
        default=1e-4,
        help="Initial LR for Gaussian positions",
    )
    parser.add_argument(
        "--position_lr_final",
        type=float,
        default=1e-6,
        help="Final LR for Gaussian positions",
    )
    parser.add_argument(
        "--position_lr_delay_mult",
        type=float,
        default=0.01,
        help="Multiplier for position LR delay phase",
    )
    parser.add_argument(
        "--rotation_lr", type=float, default=0.001, help="LR for Gaussian rotations"
    )
    parser.add_argument(
        "--scaling_lr", type=float, default=0.005, help="LR for Gaussian scaling"
    )
    parser.add_argument(
        "--latent_lr", type=float, default=0.001, help="LR for Gaussian latent features"
    )
    parser.add_argument(
        "--activation_lr",
        type=float,
        default=0.01,
        help="LR for Gaussian base activations",
    )
    parser.add_argument(
        "--decoder_lr",
        type=float,
        default=0.001,
        help="LR for the contribution decoder network",
    )
    parser.add_argument(
        "--adam_eps", type=float, default=1e-12, help="Adam optimizer epsilon"
    )
    parser.add_argument(
        "--loss_eps",
        type=float,
        default=1e-12,
        help="Epsilon for NMSE loss denominator",
    )

    parser.add_argument(
        "--lambda_latent_l1",
        type=float,
        default=0.0,
        help="L1 regularization weight for latent features (0 to disable)",
    )
    parser.add_argument(
        "--lambda_activation_l1",
        type=float,
        default=0.0,
        help="L1 regularization weight for base activations (0 to disable)",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save logs and checkpoints",
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=5,
        help="Log training metrics every N iterations",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=200,
        help="Evaluate on validation set every N iterations",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=200,
        help="Save checkpoint every N iterations",
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="Enable TensorBoard logging"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training",
    )

    args = parser.parse_args()
    return args


def evaluate(
    model: GaussianChannelFieldModel,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    wavelength: float,
    nt: int,
    nr: int,
    loss_eps: float,
) -> Dict[str, float]:
    """Evaluates the model on the validation set."""
    model.eval()
    total_loss = 0.0
    total_snr = 0.0
    count = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            rx_pos_batch = batch["rx_position"].to(device)
            h_gt_batch = batch["channel_matrix"].to(device)
            batch_size = rx_pos_batch.shape[0]

            gauss_means = model.get_xyz
            gauss_latents = model.get_latent_features
            gauss_activations = model.get_base_activations
            _, gauss_inv_covs = model.get_covariance(return_inverse=True)

            h_pred_batch = render_channel(
                rx_positions=rx_pos_batch,
                gauss_means=gauss_means,
                gauss_inv_covs=gauss_inv_covs,
                gauss_latents=gauss_latents,
                gauss_activations=gauss_activations,
                decoder_network=model.contribution_decoder,
                wavelength=wavelength,
                nt=nt,
                nr=nr,
                eps=loss_eps,
            )

            loss = criterion(h_pred_batch, h_gt_batch)
            snr = calculate_snr(loss)

            total_loss += loss.item() * batch_size
            if not torch.isinf(snr):
                total_snr += snr.item() * batch_size
            else:
                print("Warning: Infinite SNR detected during evaluation.")

            count += batch_size

    avg_loss = total_loss / count if count > 0 else 0.0
    avg_snr = total_snr / count if count > 0 else float("-inf")

    return {"val_loss": avg_loss, "val_snr_db": avg_snr}


def train(args):
    """Main training loop."""
    set_random_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    log_dir = Path(args.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = log_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    logger = setup_logging(log_dir)
    writer = SummaryWriter(str(log_dir / "tensorboard")) if args.tensorboard else None

    logger.info(f"Starting training; Log directory: {log_dir}")
    logger.info(f"Arguments: {args}")
    logger.info(f"Using device: {device}")

    logger.info("Loading data...")
    try:
        train_loader, val_loader, metadata = get_dataloaders(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
        nt = metadata["num_tx_ant"]
        nr = metadata["num_rx_ant"]
        wavelength = metadata["wavelength"]
        tx_position = metadata["tx_position"].to(device)
        env_dims = metadata["env_dims"]
        logger.info(
            f"Dataset Metadata: Nt={nt}, Nr={nr}, Freq={metadata['frequency']/1e9:.2f}GHz, Lambda={wavelength:.4f}m, IsSISO={metadata['is_siso']}"
        )
        logger.info(f"Tx Position: {tx_position.tolist()}")
        if env_dims is not None:
            logger.info(f"Environment Dimensions: {env_dims.tolist()}")
    except Exception as e:
        logger.exception(f"Failed to load data: {e}")
        if writer:
            writer.close()
        return

    logger.info("Initializing model...")
    model = GaussianChannelFieldModel(
        num_tx_ant=nt,
        num_rx_ant=nr,
        latent_dim=args.latent_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_num_layers=args.decoder_num_layers,
        initial_gaussians=args.initial_gaussians,
        device=device,
    )

    start_iteration = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        try:
            model, start_iteration = GaussianChannelFieldModel.load(
                Path(args.resume), device, args
            )
            start_iteration += 1
            logger.info(f"Resumed from iteration {start_iteration -1}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}. Starting from scratch.")
            args.resume = None

    if not args.resume:
        model.init_gaussians(
            env_dims=env_dims.to(device) if env_dims is not None else None
        )
        model.training_setup(args)

    model = model.to(device)
    logger.info(f"Model initialized with {model.get_xyz.shape[0]} Gaussians.")
    logger.info(
        f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    criterion = NormalizedMSELoss(eps=args.loss_eps).to(device)

    logger.info("Starting training...")
    progress_bar = tqdm(
        range(start_iteration, args.iterations), desc="Training Gaussians"
    )
    ema_loss = -1.0
    train_iter = iter(train_loader)

    for iteration in progress_bar:
        iter_start_time = time.time()
        model.train()
        model.update_learning_rate(iteration)

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        rx_pos_batch = batch["rx_position"].to(device)
        h_gt_batch = batch["channel_matrix"].to(device)

        gauss_means = model.get_xyz
        gauss_latents = model.get_latent_features
        gauss_activations = model.get_base_activations
        _, gauss_inv_covs = model.get_covariance(return_inverse=True)

        h_pred_batch = render_channel(
            rx_positions=rx_pos_batch,
            gauss_means=gauss_means,
            gauss_inv_covs=gauss_inv_covs,
            gauss_latents=gauss_latents,
            gauss_activations=gauss_activations,
            decoder_network=model.contribution_decoder,
            wavelength=wavelength,
            nt=nt,
            nr=nr,
            eps=args.loss_eps,
        )

        loss = criterion(h_pred_batch, h_gt_batch)
        total_loss = loss
        l1_latent_loss = torch.tensor(0.0, device=device)
        l1_activation_loss = torch.tensor(0.0, device=device)

        if args.lambda_latent_l1 > 0:
            l1_latent_loss = torch.mean(torch.abs(model.get_latent_features))
            total_loss = total_loss + args.lambda_latent_l1 * l1_latent_loss
        if args.lambda_activation_l1 > 0:
            l1_activation_loss = torch.mean(torch.abs(model.get_base_activations))
            total_loss = total_loss + args.lambda_activation_l1 * l1_activation_loss

        model.optimizer.zero_grad()
        total_loss.backward()

        found_nan_grad = False
        for param in model.parameters():
            if param.grad is not None and (
                torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
            ):
                logger.warning(
                    f"NaN or Inf gradient detected at iteration {iteration}. Skipping optimizer step."
                )
                found_nan_grad = True
                break

        if not found_nan_grad:
            grad_stats = compute_grad_stats(model)
            model.optimizer.step()
        else:
            model.optimizer.zero_grad()
            grad_stats = {}

        iter_time = time.time() - iter_start_time
        with torch.no_grad():
            current_loss = loss.item()
            if np.isnan(current_loss) or np.isinf(current_loss):
                logger.warning(
                    f"NaN or Inf loss detected at iteration {iteration}. Resetting EMA loss."
                )
                ema_loss = -1.0
            elif ema_loss < 0:
                ema_loss = current_loss
            else:
                ema_loss = 0.95 * ema_loss + 0.05 * current_loss

            if iteration % args.log_freq == 0:
                snr = calculate_snr(loss).item()
                log_msg = (
                    f"[{iteration}/{args.iterations}] Loss: {current_loss:.3e} | "
                    f"EMA Loss: {ema_loss:.3e} | SNR: {snr:.2f} dB | "
                    f"Time: {iter_time:.2f}s | Gaussians: {gauss_means.shape[0]}"
                )
                logger.info(log_msg)
                if grad_stats:
                    grad_log_msg = (
                        f"Grad Norm: {grad_stats['grad_norm']:.3e} | "
                        f"Mean Abs: {grad_stats['mean_abs_grad']:.3e} | "
                        f"Min: {grad_stats['min_grad']:.3e} | Max: {grad_stats['max_grad']:.3e}"
                    )
                    logger.info(grad_log_msg)

                if writer is not None:
                    writer.add_scalar("train/loss", current_loss, iteration)
                    writer.add_scalar("train/ema_loss", ema_loss, iteration)
                    writer.add_scalar("train/snr_db", snr, iteration)
                    writer.add_scalar("train/iteration_time_sec", iter_time, iteration)
                    writer.add_scalar(
                        "train/num_gaussians", gauss_means.shape[0], iteration
                    )
                    if grad_stats:
                        writer.add_scalar(
                            "grads/norm", grad_stats["grad_norm"], iteration
                        )
                        writer.add_scalar(
                            "grads/mean_abs", grad_stats["mean_abs_grad"], iteration
                        )
                    if args.lambda_latent_l1 > 0:
                        writer.add_scalar(
                            "train/loss_l1_latent", l1_latent_loss.item(), iteration
                        )
                    if args.lambda_activation_l1 > 0:
                        writer.add_scalar(
                            "train/loss_l1_activation",
                            l1_activation_loss.item(),
                            iteration,
                        )
                    if model.optimizer:
                        for i, param_group in enumerate(model.optimizer.param_groups):
                            writer.add_scalar(
                                f"lr/{param_group['name']}",
                                param_group["lr"],
                                iteration,
                            )

        if iteration % args.eval_freq == 0 and iteration > 0:
            logger.info(f"Starting evaluation at iteration {iteration}")
            eval_metrics = evaluate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                wavelength=wavelength,
                nt=nt,
                nr=nr,
                loss_eps=args.loss_eps,
            )
            logger.info(f"Validation Loss: {eval_metrics['val_loss']:.3e}")
            logger.info(f"Validation SNR: {eval_metrics['val_snr_db']:.2f} dB")

            if writer is not None:
                writer.add_scalar(
                    "validation/loss", eval_metrics["val_loss"], iteration
                )
                writer.add_scalar(
                    "validation/snr_db", eval_metrics["val_snr_db"], iteration
                )

        if iteration % args.checkpoint_freq == 0 or iteration == args.iterations - 1:
            checkpoint_path = checkpoints_dir / f"checkpoint_{iteration:07d}.pt"
            model.save(checkpoint_path, iteration=iteration)
            if iteration != args.iterations - 1 and iteration != 0:
                logger.info(f"Checkpoint saved to {checkpoint_path}")

    final_model_path = log_dir / "final_model.pt"
    model.save(final_model_path, iteration=args.iterations - 1)
    logger.info(f"Training completed. Final model saved to {final_model_path}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)

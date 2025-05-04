# train.py

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.dataloader import get_dataloaders
from engine.render import render
from models.gaussian_model import GaussianChannelFieldModel
from utils.general_utils import set_random_seed
from utils.loss import calculate_snr
from utils.train_utils import compute_grad_stats, setup_logging


def parse_args():
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description="Train Gaussian channel field model")

    # data and init params
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset file (.mat)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of data for training",
    )
    parser.add_argument(
        "--initial_gaussians",
        type=int,
        default=1_000,
        help="Number of Gaussians to initialize",
    )
    parser.add_argument(
        "--init_method",
        type=str,
        default="random",
        choices=["random", "point_cloud"],
        help="Initialization method ('random', 'point_cloud')",
    )
    parser.add_argument(
        "--norm_eps",
        type=float,
        default=1e-8,
        help="Epsilon for dataset normalization",
    )

    # model params
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=64,
        help="Dimension of Gaussian latent features (F)",
    )
    parser.add_argument(
        "--attribute_hidden_dim",
        type=int,
        default=128,
        help="Hidden dim for Attribute network",
    )
    parser.add_argument(
        "--attribute_num_layers",
        type=int,
        default=6,
        help="Number of layers for Attribute network",
    )
    parser.add_argument(
        "--attribute_pos_enc_freqs",
        type=int,
        default=57,
        help="Num frequencies for positional encoding in Attribute network",
    )
    parser.add_argument(
        "--decoder_hidden_dim",
        type=int,
        default=64,
        help="Hidden dim for ContributionDecoder",
    )
    parser.add_argument(
        "--decoder_num_layers",
        type=int,
        default=2,
        help="Number of layers for ContributionDecoder",
    )

    # training params
    parser.add_argument(
        "--iterations",
        type=int,
        default=30_000,
        help="Total training iterations",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--optimizer_eps",
        type=float,
        default=1e-8,
        help="AdamW optimizer epsilon",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-6,
        help="Weight decay for AdamW optimizer",
    )
    parser.add_argument(
        "--snr_eps",
        type=float,
        default=1e-10,
        help="Epsilon for SNR calculation",
    )
    parser.add_argument(
        "--rx_noise_std",
        type=float,
        default=0.03,
        help="Std dev of Gaussian noise added to Rx positions during training",
    )
    parser.add_argument(
        "--lambda_activation_l1",
        type=float,
        default=1e-2,
        help="L1 regularization weight for base activations logits",
    )

    # lr params
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
        "--attribute_net_lr", type=float, default=0.001, help="LR for Attribute network"
    )
    parser.add_argument(
        "--decoder_lr",
        type=float,
        default=0.0015,
        help="LR for the contribution decoder network",
    )

    # experiment params
    parser.add_argument(
        "--stop_xyz_iter",
        type=int,
        default=int(0.75 * 30_000),
        help="Stop updating Gaussian positions after this iteration",
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
        default=20,
        help="Log training metrics every N iterations",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=250,
        help="Evaluate on validation set every N iterations",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=2000,
        help="Save checkpoint every N iterations",
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="Enable TensorBoard logging"
    )

    # misc params
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
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

    if hasattr(args, "iterations"):
        args.stop_xyz_iter = int(0.5 * args.iterations)
    else:
        args.stop_xyz_iter = float("inf")

    return args


def evaluate(
    model: GaussianChannelFieldModel,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tx_position: torch.Tensor,
    nt: int,
    nr: int,
    snr_eps: float,
) -> Dict[str, float]:
    """Evaluates the model on the validation set."""
    model.eval()
    total_loss = 0.0
    total_snr = 0.0
    count = 0

    with torch.no_grad():

        for batch in tqdm(
            val_loader, desc="Evaluating", leave=False, dynamic_ncols=True
        ):
            rx_pos_batch = batch["rx_position"].to(device)

            chan_gt_batch = batch["channel"].to(device)
            batch_size = rx_pos_batch.shape[0]

            chan_pred_batch = render(
                rx_positions=rx_pos_batch,
                model=model,
                tx_position=tx_position,
                nt=nt,
                nr=nr,
                eps=snr_eps,
            )

            loss = criterion(chan_pred_batch, chan_gt_batch)

            snr = calculate_snr(loss, chan_gt_batch, eps=snr_eps)

            total_loss += loss.item() * batch_size

            if not torch.isinf(snr) and not torch.isnan(snr):
                total_snr += snr.item() * batch_size

            count += batch_size

    avg_loss = total_loss / count if count > 0 else 0.0
    avg_snr = total_snr / count if count > 0 else float("-inf")

    return {"val_mse_loss": avg_loss, "val_snr_db": avg_snr}


def train(args):
    """Main training loop."""
    set_random_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

    run_name = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_mag_L{args.latent_dim}_N{args.initial_gaussians}"
    )
    log_dir = Path(args.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = log_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    logger = setup_logging(log_dir)
    writer = SummaryWriter(str(log_dir / "tensorboard")) if args.tensorboard else None

    logger.info(f"Starting training run: {run_name}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Using device: {device}")

    logger.info("Loading data...")
    try:
        train_loader, val_loader, metadata = get_dataloaders(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_ratio=args.train_ratio,
            seed=args.seed,
            norm_eps=args.norm_eps,
        )
        nt = metadata["num_tx_ant"]
        nr = metadata["num_rx_ant"]
        tx_position = metadata["tx_position"].to(device)
        env_dims = metadata.get("env_dims")
        point_cloud = metadata.get("point_cloud")
        min_mag = metadata.get("min_magnitude", 0.0)
        max_mag = metadata.get("max_magnitude", 1.0)

        logger.info(
            f"Dataset metadata: Nt={nt}, Nr={nr}, Freq={metadata['frequency']/1e9:.2f}GHz, IsSISO={metadata['is_siso']}"
        )
        logger.info(f"Normalization min/max: Min={min_mag:.4e}, Max={max_mag:.4e}")
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
        latent_dim=args.latent_dim,
        attribute_hidden_dim=args.attribute_hidden_dim,
        attribute_num_layers=args.attribute_num_layers,
        attribute_pos_enc_freqs=args.attribute_pos_enc_freqs,
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
                Path(args.resume),
                device,
                args,
            )
            start_iteration += 1
            logger.info(f"Resumed from iteration {start_iteration -1}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}. Starting from scratch")
            args.resume = None

    if not args.resume:
        init_pc_arg = None

        if args.init_method == "point_cloud":
            if point_cloud is not None:
                logger.info("Using point cloud for Gaussian initialization")
                init_pc_arg = point_cloud
            else:
                logger.warning(
                    "Point cloud initialization requested but no point cloud data found. Falling back to random initialization."
                )
                args.init_method = "random"

        if args.init_method == "random":
            logger.info("Using random initialization for Gaussians")
            model.init_gaussians(
                env_dims=env_dims if env_dims is not None else None,
                point_cloud=None,
                num_points=args.initial_gaussians,
            )
        elif args.init_method == "point_cloud":
            model.init_gaussians(
                env_dims=None,
                point_cloud=init_pc_arg,
                num_points=args.initial_gaussians,
            )

        model.training_setup(args)

    model = model.to(device)
    logger.info(f"Model initialized/loaded with {model.get_xyz.shape[0]} Gaussians")
    logger.info(
        f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    criterion = nn.MSELoss().to(device)
    logger.info("Using MSE Loss for training")

    logger.info(f"Starting training from iteration {start_iteration}...")
    progress_bar = tqdm(
        range(start_iteration, args.iterations), desc="Training GCF", dynamic_ncols=True
    )
    ema_loss = -1.0
    train_iter = iter(train_loader)

    for iteration in progress_bar:
        iter_start_time = time.time()
        model.train()
        model.update_learning_rate(iteration, args)

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        rx_pos_batch = batch["rx_position"].to(device)
        chan_gt_batch = batch["channel"].to(device)

        if args.rx_noise_std > 0:
            noise = torch.randn_like(rx_pos_batch) * args.rx_noise_std
            rx_pos_batch = rx_pos_batch + noise

        chan_pred_batch = render(
            rx_positions=rx_pos_batch,
            model=model,
            tx_position=tx_position,
            nt=nt,
            nr=nr,
            eps=args.snr_eps,
        )

        mse_loss = criterion(chan_pred_batch, chan_gt_batch)
        total_loss = mse_loss
        l1_activation_loss = torch.tensor(0.0, device=device)

        if args.lambda_activation_l1 > 0 and model.get_xyz.shape[0] > 0:
            base_activations_logits = model.get_base_activation_logits(tx_position)
            l1_activation_loss = torch.mean(torch.abs(base_activations_logits))
            total_loss = total_loss + args.lambda_activation_l1 * l1_activation_loss

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
            if iteration % args.log_freq == 0:

                snr = calculate_snr(mse_loss, chan_gt_batch, eps=args.snr_eps).item()
                num_gaussians = model.get_xyz.shape[0]

                log_msg_train = (
                    f"[{iteration}/{args.iterations}] | "
                    f"Loss(MSE): {current_loss:.4e} | EMA Loss: {ema_loss:.4e} | "
                    f"SNR: {snr:.2f} dB | Time: {iter_time:.3f}s | Gauss: {num_gaussians}"
                )
                if args.lambda_activation_l1 > 0:
                    log_msg_train += f" | L1 Act Loss: {l1_activation_loss.item():.4e}"

                logger.info(log_msg_train)

                if grad_stats:
                    grad_log_msg = (
                        f" Grads - Norm: {grad_stats['grad_norm']:.3e} | Mean Abs: {grad_stats['mean_abs_grad']:.3e} | "
                        f"Min: {grad_stats['min_grad']:.3e} | Max: {grad_stats['max_grad']:.3e} | "
                        f"Count: {grad_stats['param_count_with_grad']}"
                    )
                    logger.info(grad_log_msg)

                if writer is not None:
                    writer.add_scalar("train/mse_loss", current_loss, iteration)
                    writer.add_scalar("train/ema_loss", ema_loss, iteration)
                    writer.add_scalar("train/snr_db", snr, iteration)
                    writer.add_scalar("train/iteration_time_sec", iter_time, iteration)
                    writer.add_scalar("train/num_gaussians", num_gaussians, iteration)
                    if args.lambda_activation_l1 > 0:
                        writer.add_scalar(
                            "train/l1_activation_loss",
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

        if iteration % args.eval_freq == 0 and iteration > 0:
            if len(val_loader) > 0:
                logger.info(f"Starting evaluation at iteration {iteration}")
                eval_start_time = time.time()
                eval_metrics = evaluate(
                    model=model,
                    val_loader=val_loader,
                    criterion=criterion,
                    device=device,
                    tx_position=tx_position,
                    nt=nt,
                    nr=nr,
                    snr_eps=args.snr_eps,
                )
                eval_time = time.time() - eval_start_time
                logger.info(
                    f"Validation | Loss: {eval_metrics['val_mse_loss']:.4e} | SNR: {eval_metrics['val_snr_db']:.2f} dB | Time: {eval_time:.2f}s"
                )

                if writer is not None:
                    writer.add_scalar(
                        "validation/mse_loss", eval_metrics["val_mse_loss"], iteration
                    )
                    writer.add_scalar(
                        "validation/snr_db", eval_metrics["val_snr_db"], iteration
                    )
            else:
                logger.info(f"Skipping evaluation at iteration {iteration}")

        if (
            iteration % args.checkpoint_freq == 0 and iteration > 0
        ) or iteration == args.iterations - 1:
            checkpoint_path = checkpoints_dir / f"checkpoint_{iteration:07d}.pt"
            model.save(checkpoint_path, iteration=iteration)
            logger.info(f"Checkpoint saved to {checkpoint_path}")

    progress_bar.close()
    final_model_path = log_dir / "final_model.pt"
    model.save(final_model_path, iteration=args.iterations - 1)
    logger.info(f"Training completed after {args.iterations} iterations")
    logger.info(f"Final model saved to {final_model_path}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)

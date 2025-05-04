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

# import the updated rendering engine
from engine.render_magnitude import render_magnitude
from models.gaussian_model import GaussianChannelFieldModel
from utils.general_utils import set_random_seed

# import updated loss and snr calculation
from utils.loss import calculate_snr  # MSELoss is used directly from nn
from utils.train_utils import compute_grad_stats, setup_logging


def parse_args():
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train Gaussian Channel Field Model for Magnitude Prediction"
    )

    # --- Data and Initialization ---
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset file (.mat)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--initial_gaussians",
        type=int,
        default=32_000,
        help="Number of Gaussians to initialize (default: 32k)",
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
        help="Epsilon for dataset magnitude normalization stability",
    )

    # --- Model Architecture ---
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=32,
        help="Dimension of Gaussian latent features (F) (default: 32)",
    )
    parser.add_argument(
        "--attribute_hidden_dim",
        type=int,
        default=64,
        help="Hidden dim for Attribute Network MLP (default: 64)",
    )
    parser.add_argument(
        "--attribute_num_layers",
        type=int,
        default=3,
        help="Number of layers for Attribute Network MLP (default: 3)",
    )
    parser.add_argument(
        "--attribute_pos_enc_freqs",
        type=int,
        default=10,
        help="Num frequencies for positional encoding in Attribute Net (default: 10)",
    )
    parser.add_argument(
        "--decoder_hidden_dim",
        type=int,
        default=64,
        help="Hidden dim for Contribution Decoder MLP (default: 64)",
    )
    parser.add_argument(
        "--decoder_num_layers",
        type=int,
        default=4,
        help="Number of layers for Contribution Decoder MLP (default: 4)",
    )

    # --- Training ---
    parser.add_argument(
        "--iterations",
        type=int,
        default=50_000,
        help="Total training iterations (increased default)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (increased default)",
    )
    parser.add_argument(
        "--optimizer_eps",
        type=float,
        default=1e-8,
        help="AdamW optimizer epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-6,
        help="Weight decay for AdamW optimizer (small default)",
    )  # added small default wd
    # parser.add_argument("--loss_eps", type=float, default=1e-10, help="Epsilon for SNR calculation stability (default: 1e-10)") # Renamed from loss_eps
    parser.add_argument(
        "--snr_calc_eps",
        type=float,
        default=1e-10,
        help="Epsilon for SNR calculation stability (default: 1e-10)",
    )
    parser.add_argument(
        "--rx_noise_std",
        type=float,
        default=0.01,
        help="Std dev of Gaussian noise added to Rx positions during training (small default)",
    )  # added small default noise
    parser.add_argument(
        "--lambda_activation_l1",
        type=float,
        default=1e-4,
        help="L1 regularization weight for base activations logits (small default)",
    )  # added small default reg

    # --- Learning Rates ---
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
        "--attribute_net_lr", type=float, default=0.001, help="LR for Attribute Network"
    )
    parser.add_argument(
        "--decoder_lr",
        type=float,
        default=0.0015,
        help="LR for the contribution decoder network (slightly increased)",
    )  # slightly increased decoder LR
    parser.add_argument(
        "--stop_xyz_iter",
        type=int,
        default=int(0.75 * 50_000),
        help="Stop updating Gaussian positions after this iteration (default: 75% of total)",
    )  # adjusted default

    # --- Logging and Saving ---
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/magnitude_prediction",
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
    )  # less frequent eval
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=2000,
        help="Save checkpoint every N iterations",
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="Enable TensorBoard logging"
    )

    # --- System ---
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)",
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

    # update stop_xyz_iter based on potentially changed iterations
    if hasattr(args, "iterations"):
        args.stop_xyz_iter = int(0.75 * args.iterations)
    else:
        # fallback if iterations isn't parsed correctly (shouldn't happen)
        args.stop_xyz_iter = float("inf")

    return args


def evaluate(
    model: GaussianChannelFieldModel,
    val_loader: DataLoader,
    criterion: nn.Module,  # expects MSELoss instance
    device: torch.device,
    tx_position: torch.Tensor,
    # wavelength: float, # wavelength not directly needed for magnitude rendering
    nt: int,
    nr: int,
    snr_calc_eps: float,
) -> Dict[str, float]:
    """Evaluates the model on the validation set for magnitude prediction."""
    model.eval()  # set model to evaluation mode
    total_loss = 0.0
    total_snr = 0.0
    count = 0

    with torch.no_grad():  # disable gradient calculations
        # wrap val_loader with tqdm for progress bar
        for batch in tqdm(
            val_loader, desc="Evaluating", leave=False, dynamic_ncols=True
        ):
            rx_pos_batch = batch["rx_position"].to(device)
            # target is now normalized magnitude
            h_mag_gt_batch = batch["channel_magnitude"].to(device)
            batch_size = rx_pos_batch.shape[0]

            # render predicted magnitude
            h_mag_pred_batch = render_magnitude(
                rx_positions=rx_pos_batch,
                model=model,
                tx_position=tx_position,
                # wavelength=wavelength, # not needed
                nt=nt,
                nr=nr,
                eps=snr_calc_eps,  # use snr_calc_eps for rendering stability too
            )

            # calculate MSE loss
            loss = criterion(h_mag_pred_batch, h_mag_gt_batch)
            # calculate SNR based on MSE and target magnitude
            snr = calculate_snr(loss, h_mag_gt_batch, eps=snr_calc_eps)

            total_loss += loss.item() * batch_size
            # accumulate SNR only if it's finite
            if not torch.isinf(snr) and not torch.isnan(snr):
                total_snr += snr.item() * batch_size
            # else:
            # print(f"Warning: Skipping Inf/NaN SNR value ({snr.item()}) in validation accumulation.")

            count += batch_size

    # calculate average loss and SNR
    avg_loss = total_loss / count if count > 0 else 0.0
    avg_snr = total_snr / count if count > 0 else float("-inf")  # or float('nan')?

    return {"val_mse_loss": avg_loss, "val_snr_db": avg_snr}


def train(args):
    """Main training loop for magnitude prediction."""
    set_random_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

    # setup logging directory
    run_name = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_mag_L{args.latent_dim}_N{args.initial_gaussians}"
    )
    log_dir = Path(args.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = log_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # setup logging
    logger = setup_logging(log_dir)
    writer = SummaryWriter(str(log_dir / "tensorboard")) if args.tensorboard else None

    logger.info(f"Starting training run: {run_name}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Arguments: {vars(args)}")  # log all arguments
    logger.info(f"Using device: {device}")

    # --- Data Loading ---
    logger.info("Loading data...")
    try:
        train_loader, val_loader, metadata = get_dataloaders(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_ratio=args.train_ratio,
            seed=args.seed,
            norm_eps=args.norm_eps,  # pass norm eps
        )
        nt = metadata["num_tx_ant"]
        nr = metadata["num_rx_ant"]
        # wavelength = metadata["wavelength"] # not directly used in magnitude rendering
        tx_position = metadata["tx_position"].to(device)
        env_dims = metadata.get("env_dims")  # use get for safety
        point_cloud = metadata.get("point_cloud")
        min_mag = metadata.get("min_magnitude", 0.0)  # get normalization params
        max_mag = metadata.get("max_magnitude", 1.0)

        logger.info(
            f"Dataset Metadata: Nt={nt}, Nr={nr}, Freq={metadata['frequency']/1e9:.2f}GHz, IsSISO={metadata['is_siso']}"
        )
        logger.info(f"Magnitude Normalization: Min={min_mag:.4e}, Max={max_mag:.4e}")
        logger.info(f"Transmitter Position: {tx_position.cpu().numpy()}")
        if point_cloud is not None:
            logger.info(f"Point cloud loaded with shape: {point_cloud.shape}")
            point_cloud = point_cloud.to(
                device
            )  # move to device if using point cloud init
        else:
            logger.info("No point cloud data found or used for initialization.")
        if env_dims is not None:
            logger.info(f"Environment dimensions loaded: {env_dims.numpy().tolist()}")
            env_dims = env_dims.to(device)  # move to device if using random init
        else:
            logger.warning(
                "No environment dimensions found. Random init will use default range [-1, 1]."
            )

        # check if dataloaders are empty
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

    # --- Model Initialization ---
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
        initial_gaussians=args.initial_gaussians,  # pass this for potential use if not resuming
        device=device,
    )

    start_iteration = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        try:
            # load model and potentially optimizer state
            model, start_iteration = GaussianChannelFieldModel.load(
                Path(args.resume),
                device,
                args,  # pass training args to load optimizer state
            )
            start_iteration += 1  # start from the next iteration
            logger.info(f"Resumed from iteration {start_iteration -1}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}. Starting from scratch.")
            args.resume = None  # ensure we don't try to resume again

    # initialize gaussians and training setup only if not resuming
    if not args.resume:
        init_pc_arg = None
        if args.init_method == "point_cloud":
            if point_cloud is not None:
                logger.info("Using point cloud for Gaussian initialization.")
                init_pc_arg = point_cloud
            else:
                logger.warning(
                    "Point cloud initialization requested but no point cloud data found. Falling back to random initialization."
                )
                args.init_method = "random"  # update arg to reflect fallback

        if args.init_method == "random":
            logger.info("Using random initialization for Gaussians.")
            # ensure env_dims is passed if available
            model.init_gaussians(
                env_dims=env_dims if env_dims is not None else None,
                point_cloud=None,  # explicitly None for random
                num_points=args.initial_gaussians,
            )
        elif args.init_method == "point_cloud":
            model.init_gaussians(
                env_dims=None,  # not needed if using point cloud
                point_cloud=init_pc_arg,
                num_points=args.initial_gaussians,  # num_points acts as max sample size here
            )

        # setup optimizer and LR schedulers
        model.training_setup(args)

    # ensure model is on the correct device
    model = model.to(device)
    logger.info(f"Model initialized/loaded with {model.get_xyz.shape[0]} Gaussians.")
    logger.info(
        f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # --- Loss Function ---
    criterion = nn.MSELoss().to(device)  # use standard MSE loss
    logger.info("Using MSE Loss for training.")

    # --- Training Loop ---
    logger.info(f"Starting training from iteration {start_iteration}...")
    progress_bar = tqdm(
        range(start_iteration, args.iterations), desc="Training GCF", dynamic_ncols=True
    )
    ema_loss = -1.0  # use -1 to indicate not yet initialized
    train_iter = iter(train_loader)  # create iterator for training data

    for iteration in progress_bar:
        iter_start_time = time.time()
        model.train()  # set model to training mode
        model.update_learning_rate(
            iteration, args
        )  # update LRs based on current iteration

        # --- Get Batch ---
        try:
            batch = next(train_iter)
        except StopIteration:
            # epoch finished, reset iterator
            train_iter = iter(train_loader)
            batch = next(train_iter)
            logger.info(
                f"Epoch finished, restarting data loader at iteration {iteration}"
            )

        rx_pos_batch = batch["rx_position"].to(device)
        # target is normalized magnitude
        h_mag_gt_batch = batch["channel_magnitude"].to(device)

        # --- Add Noise to Rx Positions (Data Augmentation) ---
        if args.rx_noise_std > 0:
            noise = torch.randn_like(rx_pos_batch) * args.rx_noise_std
            rx_pos_batch = rx_pos_batch + noise

        # --- Forward Pass ---
        h_mag_pred_batch = render_magnitude(
            rx_positions=rx_pos_batch,
            model=model,
            tx_position=tx_position,
            # wavelength=wavelength, # not needed
            nt=nt,
            nr=nr,
            eps=args.snr_calc_eps,  # use snr eps for stability here too
        )

        # --- Calculate Loss ---
        mse_loss = criterion(h_mag_pred_batch, h_mag_gt_batch)
        total_loss = mse_loss
        l1_activation_loss = torch.tensor(0.0, device=device)  # initialize

        # add L1 regularization on base activation *logits* if enabled
        if args.lambda_activation_l1 > 0 and model.get_xyz.shape[0] > 0:
            # get the logits from the attribute network
            _, base_activations_logits = model.get_attributes(tx_position)
            # calculate L1 loss on the logits
            l1_activation_loss = torch.mean(torch.abs(base_activations_logits))
            total_loss = total_loss + args.lambda_activation_l1 * l1_activation_loss

        # --- Backward Pass and Optimization ---
        model.optimizer.zero_grad()  # clear previous gradients
        total_loss.backward()  # compute gradients

        # check for NaN/Inf gradients before optimizer step
        found_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and (
                torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
            ):
                logger.warning(
                    f"NaN or Inf gradient detected at iteration {iteration} for parameter '{name}'. Skipping optimizer step."
                )
                found_nan_grad = True
                break  # skip step if any param has bad grad

        grad_stats = {}  # initialize grad stats dict
        if not found_nan_grad:
            # compute gradient statistics (optional but useful)
            grad_stats = compute_grad_stats(model)
            # clip gradients if needed (optional, can help stability)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            model.optimizer.step()  # update model parameters
        else:
            model.optimizer.zero_grad()  # clear the bad gradients if step was skipped

        # --- Logging ---
        iter_time = time.time() - iter_start_time
        with torch.no_grad():  # log metrics without tracking gradients
            current_loss = mse_loss.item()  # get scalar loss value
            # update EMA loss robustly
            if np.isnan(current_loss) or np.isinf(current_loss):
                logger.warning(
                    f"NaN or Inf loss detected at iteration {iteration}. Resetting EMA loss."
                )
                # consider stopping training or reducing LR if this happens often
                ema_loss = -1.0  # reset EMA
            elif ema_loss < 0:  # first valid loss
                ema_loss = current_loss
            else:  # update EMA
                ema_loss = 0.95 * ema_loss + 0.05 * current_loss

            # log periodically
            if iteration % args.log_freq == 0:
                # calculate SNR for logging
                snr = calculate_snr(
                    mse_loss, h_mag_gt_batch, eps=args.snr_calc_eps
                ).item()
                num_gaussians = model.get_xyz.shape[0]

                # format log message
                log_msg_train = (
                    f"[{iteration}/{args.iterations}] | "
                    f"Loss(MSE): {current_loss:.4e} | EMA Loss: {ema_loss:.4e} | "
                    f"SNR: {snr:.2f} dB | Time: {iter_time:.3f}s | Gauss: {num_gaussians}"
                )
                if args.lambda_activation_l1 > 0:
                    log_msg_train += f" | L1 Act Loss: {l1_activation_loss.item():.4e}"

                logger.info(log_msg_train)

                # log gradient stats if computed
                if grad_stats:
                    grad_log_msg = (
                        f"    Grads - Norm: {grad_stats['grad_norm']:.3e} | Mean Abs: {grad_stats['mean_abs_grad']:.3e} | "
                        f"Min: {grad_stats['min_grad']:.3e} | Max: {grad_stats['max_grad']:.3e} | "
                        f"Count: {grad_stats['param_count_with_grad']}"
                    )
                    logger.info(grad_log_msg)

                # log to tensorboard if enabled
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
                    # log learning rates
                    if model.optimizer:
                        for i, param_group in enumerate(model.optimizer.param_groups):
                            writer.add_scalar(
                                f"lr/{param_group['name']}",
                                param_group["lr"],
                                iteration,
                            )
                    # log gradient stats
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

        # --- Evaluation ---
        if iteration % args.eval_freq == 0 and iteration > 0:
            if len(val_loader) > 0:  # only evaluate if val set exists
                logger.info(f"--- Starting evaluation at iteration {iteration} ---")
                eval_start_time = time.time()
                eval_metrics = evaluate(
                    model=model,
                    val_loader=val_loader,
                    criterion=criterion,
                    device=device,
                    tx_position=tx_position,
                    # wavelength=wavelength, # not needed
                    nt=nt,
                    nr=nr,
                    snr_calc_eps=args.snr_calc_eps,
                )
                eval_time = time.time() - eval_start_time
                logger.info(
                    f"Validation | Loss(MSE): {eval_metrics['val_mse_loss']:.4e} | SNR: {eval_metrics['val_snr_db']:.2f} dB | Time: {eval_time:.2f}s"
                )
                logger.info(f"--- Evaluation finished ---")

                # log validation metrics to tensorboard
                if writer is not None:
                    writer.add_scalar(
                        "validation/mse_loss", eval_metrics["val_mse_loss"], iteration
                    )
                    writer.add_scalar(
                        "validation/snr_db", eval_metrics["val_snr_db"], iteration
                    )
            else:
                logger.info(
                    f"Skipping evaluation at iteration {iteration} (validation loader is empty)."
                )

        # --- Checkpointing ---
        if (
            iteration % args.checkpoint_freq == 0 and iteration > 0
        ) or iteration == args.iterations - 1:
            checkpoint_path = checkpoints_dir / f"checkpoint_{iteration:07d}.pt"
            model.save(checkpoint_path, iteration=iteration)
            logger.info(f"Checkpoint saved to {checkpoint_path}")

    # --- Training Finished ---
    progress_bar.close()  # close the tqdm bar
    final_model_path = log_dir / "final_model.pt"
    model.save(final_model_path, iteration=args.iterations - 1)
    logger.info(f"Training completed after {args.iterations} iterations.")
    logger.info(f"Final model saved to {final_model_path}")

    if writer is not None:
        writer.close()  # close tensorboard writer


if __name__ == "__main__":
    args = parse_args()
    train(args)

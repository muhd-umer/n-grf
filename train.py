# train.py

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from core.adc import adaptive_density_control, add_densification_stats
from core.loss import nmse_loss
from core.rasterize import rasterize
from datasets.dataloader import get_wireless_dataloader
from models.encoder import EncoderConfig
from models.gaussian_model import GaussianModel, GaussianModelConfig
from utils.general_utils import save_checkpoint, set_random_seed
from utils.train_utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train wireless channel Gaussian model"
    )

    # dataset params
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset file"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=16378,
        help="Number of points to sample from point cloud",
    )

    # model params
    parser.add_argument(
        "--use_pred_normals",
        action="store_true",
        help="Whether to predict surface normals",
    )
    parser.add_argument(
        "--use_attention",
        action="store_true",
        help="Whether to use attention for path encoding",
    )
    parser.add_argument(
        "--max_paths",
        type=int,
        default=10,
        help="Maximum number of paths to consider when using masking",
    )

    # optimization params
    parser.add_argument(
        "--position_lr_init",
        type=float,
        default=1e-4,
        help="Initial position learning rate",
    )
    parser.add_argument(
        "--position_lr_final",
        type=float,
        default=1e-6,
        help="Final position learning rate",
    )
    parser.add_argument(
        "--position_lr_delay_mult", type=float, default=0.01, help="LR delay multiplier"
    )
    parser.add_argument(
        "--rotation_lr", type=float, default=1e-4, help="Rotation learning rate"
    )
    parser.add_argument(
        "--scaling_lr", type=float, default=1e-4, help="Scaling learning rate"
    )
    parser.add_argument(
        "--opacity_lr", type=float, default=1e-3, help="Opacity learning rate"
    )
    parser.add_argument(
        "--encoder_lr", type=float, default=1e-4, help="Encoder learning rate"
    )
    parser.add_argument(
        "--normals_lr", type=float, default=1e-4, help="Normals learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Weight decay for encoder"
    )
    parser.add_argument(
        "--percent_dense", type=float, default=0.01, help="Density control parameter"
    )

    # adaptive density control params
    parser.add_argument(
        "--densify_from_iter",
        type=int,
        default=500,
        help="Start densification from iteration",
    )
    parser.add_argument(
        "--densify_until_iter",
        type=int,
        default=15000,
        help="Stop densification at iteration",
    )
    parser.add_argument(
        "--densification_interval",
        type=int,
        default=100,
        help="Apply densification every N iterations",
    )
    parser.add_argument(
        "--opacity_reset_interval",
        type=int,
        default=3000,
        help="Reset opacity every N iterations",
    )
    parser.add_argument(
        "--densify_grad_threshold",
        type=float,
        default=0.0002,
        help="Gradient threshold for densification",
    )

    # training params
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Directory to save outputs"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=300000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=5000,
        help="Save checkpoint every N iterations",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=5000,
        help="Evaluate every N iterations",
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=100,
        help="Log metrics every N iterations",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training",
    )

    # visualization params
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging",
    )

    args = parser.parse_args()
    return args


def setup_experiment(args):
    """Setup experiment directory and logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = log_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    logger = setup_logging(log_dir)
    logger.info(f"Arguments: {args}")

    # initialize TensorBoard writer if requested
    writer = None
    if args.tensorboard:
        logger.info("Initializing TensorBoard writer...")
        writer = SummaryWriter(log_dir / "tensorboard")

    return logger, writer, log_dir


def evaluate(
    model,
    dataloader,
    tx_position,
    frequency,
    num_tx_ant,
    num_rx_ant,
    device,
    logger,
    writer,
    iteration,
):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    num_samples = 0

    logger.info(f"Evaluating at iteration {iteration}...")

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            rx_position = data["rx_position"].to(device).squeeze()
            aoa = data["aoa"][0].to(device)
            path_loss = data["path_loss"].to(device)
            path_loss_per_ray = data["path_loss_per_ray"][0].to(device)
            gt_channel = data["channel_matrix"].to(device).squeeze()
            gt_channel = torch.hstack((gt_channel.real, gt_channel.imag))

            # embed wireless features
            wireless_data = {
                "tx_pos": tx_position,
                "rx_pos": rx_position,
                "path_loss": path_loss,
                "aoa": aoa,
                "path_loss_per_ray": path_loss_per_ray,
            }
            model.embed_features(wireless_data)

            pred_result = rasterize(
                points=model.get_xyz,
                cov3d=model.get_covariance(),
                attenuation=model.get_features[:, 0:1],
                phase_rotation=model.get_features[:, 1:2],
                opacity=model.get_opacity,
                receiver=rx_position,
                num_tx=num_tx_ant,
                num_rx=num_rx_ant,
                frequency=frequency,
                return_viewspace_info=False,
            )
            pred_channel = pred_result["channel"]

            loss = nmse_loss(pred_channel, gt_channel)
            total_loss += loss.item()
            num_samples += 1

    avg_loss = total_loss / max(num_samples, 1)
    logger.info(f"Evaluation NMSE Loss: {avg_loss:.6f}")

    if writer is not None:
        writer.add_scalar("eval/nmse_loss", avg_loss, iteration)

    model.train()
    return avg_loss


def train(args, logger, writer, log_dir):
    """Main training loop"""
    torch.autograd.set_detect_anomaly(True)

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # set up dataloaders
    logger.info("Initializing dataloaders...")
    train_dataloader = get_wireless_dataloader(
        args.data_path,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_dataloader = get_wireless_dataloader(
        args.data_path,
        train=False,
        batch_size=1,
        shuffle=False,
    )

    # get static environment data
    point_cloud = train_dataloader.dataset.get_point_cloud(args.num_points)
    tx_position = train_dataloader.dataset.get_tx_position().to(device)
    env_dims = train_dataloader.dataset.get_env_dims()
    frequency = train_dataloader.dataset.frequency
    num_tx_ant = train_dataloader.dataset.num_tx_ant
    num_rx_ant = train_dataloader.dataset.num_rx_ant
    scene_extent = (env_dims[:, 1] - env_dims[:, 0]).max().item()

    logger.info(f"Number of TX antennas: {num_tx_ant}")
    logger.info(f"Number of RX antennas: {num_rx_ant}")
    logger.info(f"Environment extent: {scene_extent:.2f}")
    logger.info(f"Operating frequency: {frequency/1e9:.2f} GHz")

    # initialize model
    logger.info("Initializing model...")
    model_cfg = GaussianModelConfig(
        use_pred_normals=args.use_pred_normals,
    )
    encoder_cfg = EncoderConfig(
        use_attention=args.use_attention,
        max_paths=args.max_paths,
    )
    model = GaussianModel(model_cfg=model_cfg, encoder_cfg=encoder_cfg).to(device)

    model.init_from_pc(point_cloud.to(device))
    logger.info(f"Initialized model with {len(point_cloud)} Gaussians")
    model.training_setup(args)

    # resume from checkpoint if specified
    start_iteration = 0
    best_val_loss = float("inf")
    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.encoder_optimizer.load_state_dict(
            checkpoint["encoder_optimizer_state_dict"]
        )
        start_iteration = checkpoint["iteration"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"Resuming from iteration {start_iteration}")

    # training loop
    logger.info("Starting training...")
    train_iter = iter(train_dataloader)

    progress_bar = tqdm(range(start_iteration, args.iterations))
    for iteration in progress_bar:
        iter_start_time = time.time()

        try:
            data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            data = next(train_iter)

        # extract data
        rx_position = data["rx_position"].to(device).squeeze()
        aoa = data["aoa"][0].to(device)
        path_loss = data["path_loss"].to(device)
        path_loss_per_ray = data["path_loss_per_ray"][0].to(device)
        gt_channel = data["channel_matrix"].to(device).squeeze()
        gt_channel = torch.hstack((gt_channel.real, gt_channel.imag))

        # embed wireless features
        wireless_data = {
            "tx_pos": tx_position,
            "rx_pos": rx_position,
            "path_loss": path_loss,
            "aoa": aoa,
            "path_loss_per_ray": path_loss_per_ray,
        }
        model.embed_features(wireless_data)

        render_result = rasterize(
            points=model.get_xyz,
            cov3d=model.get_covariance(),
            attenuation=model.get_features[:, 0:1],
            phase_rotation=model.get_features[:, 1:2],
            opacity=model.get_opacity,
            receiver=rx_position,
            num_tx=num_tx_ant,
            num_rx=num_rx_ant,
            frequency=frequency,
            return_viewspace_info=True,
        )

        pred_channel = render_result["channel"]
        viewspace_info = render_result["viewspace_info"]

        # get viewspace information for densification
        visibility_filter = viewspace_info["visibility_filter"]
        radii = viewspace_info["radii"]

        loss = nmse_loss(pred_channel, gt_channel)

        model.optimizer.zero_grad()
        model.encoder_optimizer.zero_grad()
        loss.backward()

        if iteration < args.densify_until_iter:
            model.max_radii2D[visibility_filter] = torch.max(
                model.max_radii2D[visibility_filter], radii[visibility_filter]
            )

            model.xyz_gradient_accum, model.denom = add_densification_stats(
                model.xyz_gradient_accum, model.denom, visibility_filter
            )

        model.update_learning_rate(iteration)
        model.optimizer.step()
        model.encoder_optimizer.step()

        if (
            iteration >= args.densify_from_iter
            and iteration < args.densify_until_iter
            and iteration % args.densification_interval == 0
        ):
            (
                model._xyz.data,
                model._scaling.data,
                model._rotation.data,
                model._opacity.data,
                model.features,
                model.xyz_gradient_accum,
                model.denom,
                model.max_radii2D,
            ) = adaptive_density_control(
                model.xyz_gradient_accum,
                model.denom,
                model.get_xyz,
                model.get_scaling,
                model._rotation,
                model._opacity,
                model.features,
                model.max_radii2D,
                args.densify_grad_threshold,
                min_opacity=0.005,
                extent=scene_extent,
                max_screen_size=num_tx_ant / 4.0,
                percent_dense=args.percent_dense,
            )

            if writer is not None:
                writer.add_scalar(
                    "train/num_gaussians", model.get_xyz.shape[0], iteration
                )

        if iteration % args.opacity_reset_interval == 0:
            model.reset_opacity()

        # ;og progress
        iter_time = time.time() - iter_start_time
        if iteration % args.log_freq == 0:
            logger.info(
                f"[{iteration}/{args.iterations}] "
                f"Loss: {loss.item():.6f}, "
                f"Time: {iter_time:.2f}s, "
                f"Gaussians: {model.get_xyz.shape[0]}"
            )

            if writer is not None:
                writer.add_scalar("train/nmse_loss", loss.item(), iteration)
                writer.add_scalar("train/iteration_time", iter_time, iteration)
                writer.add_scalar(
                    "train/num_gaussians", model.get_xyz.shape[0], iteration
                )

                for i, param_group in enumerate(model.optimizer.param_groups):
                    writer.add_scalar(
                        f"lr/{param_group['name']}", param_group["lr"], iteration
                    )

            progress_bar.set_description(
                f"Loss: {loss.item():.6f}, Gaussians: {model.get_xyz.shape[0]}"
            )

        # evaluate on validation set
        if iteration % args.eval_freq == 0:
            val_loss = evaluate(
                model,
                val_dataloader,
                tx_position,
                frequency,
                num_tx_ant,
                num_rx_ant,
                device,
                logger,
                writer,
                iteration,
            )

            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"New best validation loss: {best_val_loss:.6f}")
                save_checkpoint(
                    {
                        "iteration": iteration,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": model.optimizer.state_dict(),
                        "encoder_optimizer_state_dict": model.encoder_optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                    },
                    log_dir / "checkpoints",
                    "best_model.pt",
                )

        # save checkpoint
        if iteration % args.checkpoint_freq == 0:
            save_checkpoint(
                {
                    "iteration": iteration,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": model.optimizer.state_dict(),
                    "encoder_optimizer_state_dict": model.encoder_optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                log_dir / "checkpoints",
                f"checkpoint_{iteration:06d}.pt",
            )

    # save final model
    save_checkpoint(
        {
            "iteration": args.iterations - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": model.optimizer.state_dict(),
            "encoder_optimizer_state_dict": model.encoder_optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        log_dir / "checkpoints",
        "final_model.pt",
    )

    logger.info("Training completed!")
    return model


def main():
    args = parse_args()
    set_random_seed(args.seed)

    logger, writer, log_dir = setup_experiment(args)

    try:
        model = train(args, logger, writer, log_dir)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.exception("Training failed")
        raise
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()

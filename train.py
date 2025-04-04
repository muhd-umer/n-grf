# train.py

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.dataloader import get_dataloaders
from engine import _torch_impl as torch_impl
from engine import rasterize
from models.encoder import EncoderConfig
from models.gaussian_model import GaussianModel
from models.loss import (
    channel_corr_loss,
    complex_mse_loss,
    l1_ssim_loss,
    mse_corr_loss,
    nmse_loss,
)
from utils.general_utils import set_random_seed
from utils.train_utils import setup_logging

torch.set_float32_matmul_precision("highest")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Gaussian model for channel reconstruction"
    )

    # dataset params
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset file"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=12_000,
        help="Number of points to sample from point cloud",
    )

    # model params
    parser.add_argument(
        "--use_pred_normals",
        action="store_true",
        help="Whether to predict surface normals",
    )

    # initialization params
    parser.add_argument(
        "--init_method",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "random"],
        help="Method to initialize Gaussian points (point_cloud or random)",
    )
    parser.add_argument(
        "--no_physics_init",
        action="store_false",
        dest="physics_init",
        help="Disable physics-based initialization",
    )
    parser.add_argument(
        "--no_positional_encoding",
        action="store_false",
        dest="use_positional_encoding",
        help="Disable positional encoding in the encoder",
    )

    # optimization params
    parser.add_argument(
        "--position_lr_init",
        type=float,
        default=0.00016,
        help="Initial position learning rate",
    )
    parser.add_argument(
        "--position_lr_final",
        type=float,
        default=0.0000016,
        help="Final position learning rate",
    )
    parser.add_argument(
        "--position_lr_delay_mult", type=float, default=0.01, help="LR delay multiplier"
    )
    parser.add_argument(
        "--rotation_lr", type=float, default=0.001, help="Rotation learning rate"
    )
    parser.add_argument(
        "--scaling_lr", type=float, default=0.005, help="Scaling learning rate"
    )
    parser.add_argument(
        "--opacity_lr", type=float, default=0.025, help="Opacity learning rate"
    )
    parser.add_argument(
        "--encoder_lr", type=float, default=0.0075, help="Encoder learning rate"
    )
    parser.add_argument(
        "--normals_lr", type=float, default=0.0025, help="Normals learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-7, help="Weight decay for encoder"
    )
    parser.add_argument(
        "--percent_dense", type=float, default=0.01, help="Density control parameter"
    )

    # training params
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Directory to save outputs"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse_corr",
        choices=[
            "mse_corr",
            "complex_mse",
            "channel_corr",
            "nmse",
            "l1_ssim",
        ],
        help="Loss function to use for training",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=7_000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=700,
        help="Save checkpoint every N iterations",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=700,
        help="Evaluate every N iterations",
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=50,
        help="Log metrics every N iterations",
    )
    parser.add_argument(
        "--opacity_reset_interval",
        type=int,
        default=700,
        help="Reset opacity every N iterations",
    )
    parser.add_argument(
        "--scale_modifier",
        type=float,
        default=1.0,
        help="Scale modifier for Gaussian scaling",
    )
    parser.add_argument(
        "--disable_cuda",
        action="store_true",
        help="Disable CUDA implementation and use PyTorch fallback for rasterization",
    )
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
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


def rasterize_channel(
    model, rx_position, tx_position, num_tx_ant, num_rx_ant, frequency, args
):
    """Helper function to rasterize the channel based on command-line arguments"""
    if args.disable_cuda:
        return torch_impl.rasterize(
            points=model.get_xyz,
            cov3d=model.get_covariance(),
            attenuation=model.get_features[:, 0:1],
            phase_rotation=model.get_features[:, 1:2],
            opacity=model.get_opacity,
            receiver=rx_position,
            transmitter=tx_position,
            num_tx=num_tx_ant,
            num_rx=num_rx_ant,
            frequency=frequency,
        )
    else:
        scaling = model.get_scaling
        rotation = model.get_rotation
        scale_modifier = args.scale_modifier

        return rasterize(
            points=model.get_xyz,
            attenuation=model.get_features[:, 0:1],
            phase_rotation=model.get_features[:, 1:2],
            opacity=model.get_opacity,
            receiver=rx_position,
            transmitter=tx_position,
            num_tx=num_tx_ant,
            num_rx=num_rx_ant,
            frequency=frequency,
            scaling=scaling,
            rotation=rotation,
            scale_modifier=scale_modifier,
        )


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
    loss_type,
    args,
):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    num_samples = 0

    logger.info(f"Evaluating at iteration {iteration}...")

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            rx_position = data["rx_position"].to(device).squeeze()
            gt_channel = data["channel_matrix"].to(device).squeeze()
            gt_channel = torch.hstack((gt_channel.real, gt_channel.imag))

            # embed wireless features
            enc_data = {
                "tx_pos": tx_position,
                "rx_pos": rx_position,
                "frequency": frequency,
            }
            model.embed_features(enc_data)

            pred_channel = rasterize_channel(
                model, rx_position, tx_position, num_tx_ant, num_rx_ant, frequency, args
            )

            if loss_type == "nmse":
                loss = nmse_loss(pred_channel, gt_channel)
            elif loss_type == "l1_ssim":
                loss = l1_ssim_loss(pred_channel, gt_channel)
            elif loss_type == "complex_mse":
                loss = complex_mse_loss(pred_channel, gt_channel)
            elif loss_type == "channel_corr":
                loss = channel_corr_loss(pred_channel, gt_channel)
            elif loss_type == "mse_corr":
                loss = mse_corr_loss(pred_channel, gt_channel)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

            total_loss += loss.item()
            num_samples += 1

    avg_loss = total_loss / max(num_samples, 1)

    logger.info(f"Evaluation Loss ({loss_type}): {avg_loss:.6f}")

    if writer is not None:
        writer.add_scalar("eval/loss", avg_loss, iteration)

    model.train()
    return avg_loss


def compute_grad_stats(model):
    """Compute statistics about gradients for model parameters."""
    grad_stats = {
        "mean_abs": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
        "mean_norm": 0.0,
        "param_count": 0,
    }

    total_params = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_stats["mean_abs"] += param.grad.abs().mean().item() * param.numel()
            grad_stats["min"] = min(grad_stats["min"], param.grad.min().item())
            grad_stats["max"] = max(grad_stats["max"], param.grad.max().item())
            grad_stats["mean_norm"] += param.grad.norm().item()
            grad_stats["param_count"] += 1
            total_params += param.numel()

    if total_params > 0:
        grad_stats["mean_abs"] /= total_params

    if grad_stats["param_count"] > 0:
        grad_stats["mean_norm"] /= grad_stats["param_count"]

    return grad_stats


def train(args, logger, writer, log_dir):
    """Main training loop"""
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # set up dataloaders
    logger.info("Initializing dataloaders...")
    train_dataloader, val_dataloader = get_dataloaders(
        args.data_path,
        num_workers=2,
        drop_last=True,
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
    encoder_cfg = EncoderConfig(
        hidden_size=128,
        num_layers=8,
        skip_layers=(4,),
        input_pos_multires=10,
        use_positional_encoding=args.use_positional_encoding,
    )
    model = GaussianModel(
        encoder_cfg=encoder_cfg, use_pred_normals=args.use_pred_normals
    ).to(device)

    if args.init_method == "point_cloud":
        point_cloud = train_dataloader.dataset.get_point_cloud(args.num_points)
        model.init_from_pc(
            point_cloud.to(device),
            tx_position=tx_position if args.physics_init else None,
            frequency=frequency if args.physics_init else None,
            use_physics_init=args.physics_init,
        )
        logger.info(
            f"Initialized model with {len(point_cloud)} Gaussians from point cloud"
        )
    else:  # random initialization
        model.init_randomly(
            args.num_points,
            env_dims.to(device),
            tx_position=tx_position if args.physics_init else None,
            frequency=frequency if args.physics_init else None,
            use_physics_init=args.physics_init,
        )
        logger.info(f"Initialized model with {args.num_points} random Gaussians")

    model.training_setup(args)

    # resume from checkpoint if specified
    start_iteration = 0
    best_val_loss = float("inf")
    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model = GaussianModel.load(args.resume, device=device, training_args=args)

        # extract training state information
        checkpoint = torch.load(args.resume, map_location=device)
        start_iteration = checkpoint.get("iteration", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"Resuming from iteration {start_iteration}")

    # training loop
    logger.info("Starting training...")
    logger.info(f"Using {args.loss_type} loss function")
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
        gt_channel = data["channel_matrix"].to(device).squeeze()
        gt_channel = torch.hstack((gt_channel.real, gt_channel.imag))

        # embed wireless features
        enc_data = {
            "tx_pos": tx_position,
            "rx_pos": rx_position,
            "frequency": frequency,
        }
        model.embed_features(enc_data)

        pred_channel = rasterize_channel(
            model, rx_position, tx_position, num_tx_ant, num_rx_ant, frequency, args
        )

        if args.loss_type == "nmse":
            loss = nmse_loss(pred_channel, gt_channel)
        elif args.loss_type == "l1_ssim":
            loss = l1_ssim_loss(pred_channel, gt_channel)
        elif args.loss_type == "complex_mse":
            loss = complex_mse_loss(pred_channel, gt_channel)
        elif args.loss_type == "channel_corr":
            loss = channel_corr_loss(pred_channel, gt_channel)
        elif args.loss_type == "mse_corr":
            loss = mse_corr_loss(pred_channel, gt_channel)
        else:
            raise ValueError(f"Unknown loss type: {args.loss_type}")

        model.optimizer.zero_grad()
        model.encoder_optimizer.zero_grad()

        loss.backward()

        grad_stats = compute_grad_stats(model)

        model.update_learning_rate(iteration)
        model.optimizer.step()
        model.encoder_optimizer.step()

        if iteration % args.opacity_reset_interval == 0:
            model.reset_opacity()

        # log progress
        iter_time = time.time() - iter_start_time
        if iteration % args.log_freq == 0:
            logger.info(
                f"[{iteration}/{args.iterations}] "
                f"Loss: {loss.item():.6f}, "
                f"Time: {iter_time:.2f}s, "
                f"Gaussians: {model.get_xyz.shape[0]}"
            )
            logger.info(
                f"Grad stats: Mean abs: {grad_stats['mean_abs']:.6e}, "
                f"Min: {grad_stats['min']:.6e}, "
                f"Max: {grad_stats['max']:.6e}, "
                f"Mean norm: {grad_stats['mean_norm']:.6e}"
            )
            print("pred_channel: ", pred_channel)

            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), iteration)
                writer.add_scalar("train/iteration_time", iter_time, iteration)
                writer.add_scalar(
                    "train/num_gaussians", model.get_xyz.shape[0], iteration
                )

                writer.add_scalar("grad/mean_abs", grad_stats["mean_abs"], iteration)
                writer.add_scalar("grad/min", grad_stats["min"], iteration)
                writer.add_scalar("grad/max", grad_stats["max"], iteration)
                writer.add_scalar("grad/mean_norm", grad_stats["mean_norm"], iteration)

            progress_bar.set_description(
                f"Loss: {loss.item():.6f}, Gaussians: {model.get_xyz.shape[0]}"
            )

        if (
            iteration > 0 and iteration % args.eval_freq == 0
        ) or iteration == args.iterations - 1:
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
                loss_type=args.loss_type,
                args=args,
            )

            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"New best validation loss: {best_val_loss:.6f}")
                model.save(
                    log_dir / "checkpoints" / "best_model.pt",
                    save_optimizer=True,
                    iteration=iteration,
                    best_val_loss=best_val_loss,
                )
                logger.info("Best model saved at iteration {iteration}")

        # save checkpoint
        if iteration > 0 and iteration % args.checkpoint_freq == 0:
            checkpoint_path = log_dir / "checkpoints" / f"checkpoint_{iteration:06d}.pt"
            model.save(
                checkpoint_path,
                save_optimizer=True,
                iteration=iteration,
                best_val_loss=best_val_loss,
            )
            logger.info(f"Checkpoint saved at iteration {iteration}")

    # save final model
    model.save(
        log_dir / "checkpoints" / "final_model.pt",
        save_optimizer=True,
        iteration=args.iterations - 1,
        best_val_loss=best_val_loss,
    )
    model.save(log_dir / "checkpoints" / "eval_model.pt", save_optimizer=False)

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

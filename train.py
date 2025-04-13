# train.py

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.dataloader import get_dataloaders
from engine import _torch_impl as torch_impl
from engine import rasterize
from models.encoder import EncoderConfig
from models.gaussian_model import GaussianModel
from models.loss import calculate_nmse, calculate_snr, get_loss_function
from utils.general_utils import set_random_seed
from utils.train_utils import setup_logging


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
        default=20_000,
        help="Number of points to sample from point cloud",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers for data loading",
    )

    # initialization params
    parser.add_argument(
        "--init_method",
        type=str,
        default="random",
        choices=["point_cloud", "random"],
        help="Method to initialize Gaussian points (point_cloud or random)",
    )
    parser.add_argument(
        "--physics_init",
        action="store_true",
        help="Enable physics-based initialization",
    )
    parser.add_argument(
        "--positional_encoding",
        action="store_true",
        dest="use_positional_encoding",
        help="Enable positional encoding in the encoder",
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
        default=0.000016,
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
        "--weight_decay", type=float, default=1e-8, help="Weight decay for encoder"
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=0.5,
        help="Value to clip gradient norm to (0 to disable)",
    )
    parser.add_argument(
        "--disable_encoder_layernorm",
        action="store_false",
        dest="use_encoder_layernorm",
        help="Disable Layer Normalization in the feature encoder",
    )

    # training params
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Directory to save outputs"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="nmse",
        choices=[
            "nmse",
            "log_mse",
            "mse_corr",
            "polar_mse",
            "cosine",
            "log_mag_phase",
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
        default=70,
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

    # loss-specific arguments
    parser.add_argument(
        "--loss_scale", type=float, default=1e4, help="Scale factor for scaled_mse loss"
    )
    parser.add_argument(
        "--loss_eps", type=float, default=1e-10, help="Epsilon value for loss functions"
    )
    parser.add_argument(
        "--phase_weight",
        type=float,
        default=1.0,
        help="Weight for phase term in polar_mse loss",
    )

    # visualization params
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging",
    )

    args = parser.parse_args()
    if "use_encoder_layernorm" not in args:
        args.use_encoder_layernorm = True

    return args


def rasterize_channel(
    model, rx_position, tx_position, num_tx_ant, num_rx_ant, frequency, args
):
    """Helper function to rasterize the channel based on command-line arguments"""
    if args.disable_cuda:
        return torch_impl.rasterize(
            points=model.get_xyz,
            scaling=model.get_scaling,
            rotation=model.get_rotation,
            attenuation=model.get_features[:, 0:1].contiguous(),
            phase_rotation=model.get_features[:, 1:2].contiguous(),
            opacity=model.get_opacity,
            receiver=rx_position,
            transmitter=tx_position,
            num_tx=num_tx_ant,
            num_rx=num_rx_ant,
            frequency=frequency,
            scale_modifier=args.scale_modifier,
        )
    else:
        return rasterize(
            points=model.get_xyz,
            scaling=model.get_scaling,
            rotation=model.get_rotation,
            attenuation=model.get_features[:, 0:1].contiguous(),
            phase_rotation=model.get_features[:, 1:2].contiguous(),
            opacity=model.get_opacity,
            receiver=rx_position,
            transmitter=tx_position,
            num_tx=num_tx_ant,
            num_rx=num_rx_ant,
            frequency=frequency,
            scale_modifier=args.scale_modifier,
        )


def sequential_fwd(
    model, batch, tx_position, num_tx_ant, num_rx_ant, frequency, args, device
):
    """Process by iterating through each sample in the batch sequentially"""
    batch_size = batch["rx_position"].shape[0]
    pred_channels = []

    for i in range(batch_size):
        rx_position = batch["rx_position"][i].to(device)
        enc_data = {
            "tx_pos": tx_position,
            "rx_pos": rx_position,
            "frequency": frequency,
        }
        model.embed_features(enc_data)

        pred_channel = rasterize_channel(
            model, rx_position, tx_position, num_tx_ant, num_rx_ant, frequency, args
        )
        pred_channels.append(pred_channel)

    return torch.stack(pred_channels)


def get_gt_batch(batch, device):
    """Process ground truth channel matrices from a batch"""
    batch_size = batch["channel_matrix"].shape[0]
    gt_channels = []

    for b in range(batch_size):
        gt_channel = batch["channel_matrix"][b].to(device)
        gt_channel = torch.hstack((gt_channel.real, gt_channel.imag))
        gt_channels.append(gt_channel)

    return torch.stack(gt_channels)


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
    loss_fn,
    args,
):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    total_nmse = 0.0
    num_samples = 0

    logger.info(f"Evaluating at iteration {iteration}...")

    with torch.no_grad():
        for batch in dataloader:
            batch_size = batch["rx_position"].shape[0]
            gt_channels = get_gt_batch(batch, device)
            pred_channels = sequential_fwd(
                model,
                batch,
                tx_position,
                num_tx_ant,
                num_rx_ant,
                frequency,
                args,
                device,
            )

            loss = loss_fn(pred_channels, gt_channels)
            nmse = calculate_nmse(pred_channels, gt_channels)

            total_loss += loss.item() * batch_size
            total_nmse += nmse.item() * batch_size
            num_samples += batch_size

    avg_loss = total_loss / max(num_samples, 1)
    avg_nmse = total_nmse / max(num_samples, 1)
    avg_snr = calculate_snr(torch.tensor(avg_nmse)).item()

    logger.info(
        f"Evaluation Loss ({args.loss_type}): {avg_loss:.6f}, SNR: {avg_snr:.2f} dB"
    )

    if writer is not None:
        writer.add_scalar("eval/loss", avg_loss, iteration)
        writer.add_scalar("eval/nmse", avg_nmse, iteration)
        writer.add_scalar("eval/snr", avg_snr, iteration)

    model.train()
    return avg_loss


def compute_grad_stats(model):
    """Compute statistics about gradients for model parameters."""
    grad_stats = {
        "mean_abs": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
        "param_count": 0,
    }

    total_params = 0
    all_params = []
    for group in model.optimizer.param_groups:
        all_params.extend(group["params"])
    for group in model.encoder_optimizer.param_groups:
        all_params.extend(group["params"])

    for param in all_params:
        if param.grad is not None:
            grad_abs = param.grad.abs()
            grad_stats["mean_abs"] += grad_abs.sum().item()
            grad_stats["min"] = min(grad_stats["min"], param.grad.min().item())
            grad_stats["max"] = max(grad_stats["max"], param.grad.max().item())
            grad_stats["param_count"] += 1
            total_params += param.numel()

    if total_params > 0:
        grad_stats["mean_abs"] /= total_params

    return grad_stats


def train(args, logger, writer, log_dir):
    """Main training loop"""
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # set up dataloaders
    logger.info("Initializing dataloaders...")
    train_dataloader, val_dataloader = get_dataloaders(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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
    logger.info(f"Training with batch size: {args.batch_size}")

    # initialize model
    logger.info("Initializing model...")
    encoder_cfg = EncoderConfig(
        hidden_size=128,
        num_layers=8,
        skip_layers=(4,),
        input_pos_multires=10,
        use_positional_encoding=args.use_positional_encoding,
        use_layer_norm=args.use_encoder_layernorm,
    )
    model = GaussianModel(encoder_cfg=encoder_cfg).to(device)

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

    loss_kwargs = {
        "scale": args.loss_scale,
        "eps": args.loss_eps,
        "phase_weight": args.phase_weight,
    }
    loss_fn = get_loss_function(args.loss_type, **loss_kwargs)
    logger.info(f"Using {args.loss_type} loss function with params: {loss_kwargs}")

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
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        gt_channels = get_gt_batch(batch, device)

        pred_channels = sequential_fwd(
            model,
            batch,
            tx_position,
            num_tx_ant,
            num_rx_ant,
            frequency,
            args,
            device,
        )

        loss = loss_fn(pred_channels, gt_channels)

        with torch.no_grad():
            nmse = calculate_nmse(pred_channels, gt_channels)
            snr = calculate_snr(nmse).item()

        model.encoder_optimizer.zero_grad()
        model.optimizer.zero_grad()

        loss.backward()

        if args.gradient_clip_val > 0:
            all_gaussian_params = []
            for group in model.optimizer.param_groups:
                all_gaussian_params.extend(group["params"])
            if all_gaussian_params:
                clip_grad_norm_(all_gaussian_params, args.gradient_clip_val)

            if model.encoder_optimizer.param_groups[0]["params"]:
                clip_grad_norm_(
                    model.encoder_optimizer.param_groups[0]["params"],
                    args.gradient_clip_val,
                )

        grad_stats = compute_grad_stats(model)

        model.update_learning_rate(iteration)
        model.encoder_optimizer.step()
        model.optimizer.step()

        if iteration % args.opacity_reset_interval == 0:
            model.reset_opacity()

        # log progress
        iter_time = time.time() - iter_start_time
        if iteration % args.log_freq == 0:
            logger.info(
                f"[{iteration}/{args.iterations}] "
                f"Loss ({args.loss_type}): {loss.item():.6f}, "
                f"SNR: {snr:.2f} dB, "
                f"Time: {iter_time:.2f}s, "
                f"Batch Size: {args.batch_size}, "
                f"Gaussians: {model.get_xyz.shape[0]}"
            )
            logger.info(
                f"Grad stats: Mean abs: {grad_stats['mean_abs']:.6e}, "
                f"Min: {grad_stats['min']:.6e}, "
                f"Max: {grad_stats['max']:.6e}"
            )

            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), iteration)
                writer.add_scalar("train/nmse", nmse.item(), iteration)
                writer.add_scalar("train/snr", snr, iteration)
                writer.add_scalar("train/iteration_time", iter_time, iteration)
                writer.add_scalar(
                    "train/num_gaussians", model.get_xyz.shape[0], iteration
                )
                writer.add_scalar(
                    "train/samples_per_second", args.batch_size / iter_time, iteration
                )

                writer.add_scalar("grad/mean_abs", grad_stats["mean_abs"], iteration)
                writer.add_scalar("grad/min", grad_stats["min"], iteration)
                writer.add_scalar("grad/max", grad_stats["max"], iteration)

            progress_bar.set_description(f"Loss: {loss.item():.6f}, SNR: {snr:.2f} dB")

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
                loss_fn=loss_fn,
                args=args,
            )

            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                with torch.no_grad():
                    val_batch = next(iter(val_dataloader))
                    gt_channels = get_gt_batch(val_batch, device)
                    pred_channels = sequential_fwd(
                        model,
                        val_batch,
                        tx_position,
                        num_tx_ant,
                        num_rx_ant,
                        frequency,
                        args,
                        device,
                    )
                    best_nmse = calculate_nmse(pred_channels, gt_channels)
                    best_val_snr = calculate_snr(best_nmse).item()

                logger.info(
                    f"New best validation loss ({args.loss_type}): {best_val_loss:.6f}, SNR: {best_val_snr:.2f} dB"
                )
                model.save(
                    log_dir / "checkpoints" / "best_model.pt",
                    save_optimizer=True,
                    iteration=iteration,
                    best_val_loss=best_val_loss,
                )
                logger.info(f"Best model saved at iteration {iteration}")

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

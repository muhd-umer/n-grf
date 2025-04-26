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
from models.dir_network import DirNetworkConfig
from models.encoder import EncoderConfig
from models.gaussian_model import GaussianModel
from models.loss import calculate_nmse, calculate_snr, get_loss_function
from utils.general_utils import set_random_seed
from utils.train_utils import setup_logging


def parse_args():
    """Parse command line arguments for training."""
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
        default=64_000,
        help="Number of points to use for Gaussian initialization",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
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
        "--positional_encoding",
        action="store_true",
        dest="use_positional_encoding",
        help="Enable positional encoding in the encoder and directional network",
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
        "--opacity_lr",
        type=float,
        default=0.025,
        help="Opacity learning rate",
    )
    parser.add_argument(
        "--encoder_lr",
        type=float,
        default=0.001,
        help="Learning rate for base encoder and directional network",
    )
    parser.add_argument(
        "--enc_weight_decay",
        type=float,
        default=1e-8,
        help="Weight decay for base encoder and directional network",
    )
    parser.add_argument(
        "--gaussian_weight_decay",
        type=float,
        default=0,
        help="Weight decay for Gaussian parameters",
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=8,
        help="Value to clip gradient norm to (0 to disable)",
    )
    parser.add_argument(
        "--disable_encoder_layernorm",
        action="store_false",
        dest="use_encoder_layernorm",
        help="Disable Layer Normalization in the base feature encoder",
    )
    parser.add_argument(
        "--disable_dirnet_layernorm",
        action="store_false",
        dest="use_dirnet_layernorm",
        help="Disable Layer Normalization in the directional network",
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
        default=18_000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=600,
        help="Save checkpoint every N iterations",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=30,
        help="Evaluate every N iterations",
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=5,
        help="Log metrics every N iterations",
    )
    parser.add_argument(
        "--opacity_reset_interval",
        type=int,
        default=1_800,
        help="Reset opacity every N iterations",
    )
    parser.add_argument(
        "--opacity_reset",
        action="store_true",
        help="Disable periodic opacity reset",
    )
    parser.add_argument(
        "--scale_modifier",
        type=float,
        default=1.0,
        help="Scale modifier for Gaussian scaling",
    )
    parser.add_argument(
        "--dropout_prob",
        type=float,
        default=0.25,
        help="Dropout probability for encoder layers",
    )
    parser.add_argument(
        "--gaussian_dropout_rate",
        type=float,
        default=0.0,
        help="Dropout rate for Gaussian primitives during training (0 to disable)",
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
        "--loss_eps", type=float, default=1e-10, help="Epsilon value for loss functions"
    )
    parser.add_argument(
        "--phase_weight",
        type=float,
        default=1.0,
        help="Weight for phase term in polar_mse or log_mag_phase loss",
    )
    parser.add_argument(
        "--rx_noise_std",
        type=float,
        default=0.0,
        help="Std dev of noise to add to receiver positions during training (0 to disable)",
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
    if "use_dirnet_layernorm" not in args:
        args.use_dirnet_layernorm = True

    if args.opacity_reset:
        assert (
            args.opacity_reset_interval > 0
        ), "If opacity reset is enabled, opacity_reset_interval must be > 0"

    if args.disable_cuda:
        from engine import _torch_impl

        args.rasterize_fn = _torch_impl.rasterize
        print("Using PyTorch rasterization implementation.")
    else:
        try:
            from engine import rasterize as cuda_rasterize_fn

            args.rasterize_fn = cuda_rasterize_fn
            print("Using CUDA rasterization implementation.")
        except ImportError:
            from engine import _torch_impl

            args.rasterize_fn = _torch_impl.rasterize
            print(
                "CUDA rasterization not found, falling back to PyTorch implementation."
            )

    return args


def rasterize_channel(model, rx_params, tx_params, args, is_training=False):
    """Helper function to rasterize the channel using the selected function."""
    xyz = model.get_xyz
    scaling = model.get_scaling
    rotation = model.get_rotation
    base_features = model.get_base_features
    opacity = model.get_opacity
    num_gaussians = xyz.shape[0]

    if is_training and args.gaussian_dropout_rate > 0.0 and num_gaussians > 0:
        dropout_rate = args.gaussian_dropout_rate
        num_keep = max(1, int(num_gaussians * (1.0 - dropout_rate)))
        keep_indices = torch.randperm(num_gaussians, device=xyz.device)[:num_keep]

        xyz = xyz[keep_indices]
        scaling = scaling[keep_indices]
        rotation = rotation[keep_indices]
        base_features = base_features[keep_indices]
        opacity = opacity[keep_indices]

    return args.rasterize_fn(
        points=xyz,
        scaling=scaling,
        rotation=rotation,
        base_features=base_features,
        opacity=opacity,
        tx_params=tx_params,
        rx_params=rx_params,
        directional_network=model.directional_network,
        dir_embedder=model.dir_embedder,
        scale_modifier=args.scale_modifier,
    )


def sequential_fwd(
    model,
    batch,
    tx_params,
    rx_params,
    args,
    device,
    update_features=False,
    is_training=False,
):
    """Process by iterating through each sample in the batch sequentially."""
    batch_size = batch["rx_position"].shape[0]
    pred_channels = []
    is_training = model.training

    if update_features:
        enc_data = {"tx_pos": tx_params["position"]}
        if any(p.requires_grad for p in model.base_encoder.parameters()):
            with torch.enable_grad():
                model.embed_features(enc_data)
        else:
            with torch.no_grad():
                model.embed_features(enc_data)

    for i in range(batch_size):
        rx_position = batch["rx_position"][i].to(device)

        if is_training and args.rx_noise_std > 0:
            noise = torch.randn_like(rx_position) * args.rx_noise_std
            rx_position = rx_position + noise

        rx_params_i = rx_params.copy()
        rx_params_i["position"] = rx_position

        pred_channel = rasterize_channel(
            model, rx_params_i, tx_params, args, is_training=is_training
        )
        pred_channels.append(pred_channel)

    return torch.stack(pred_channels)


def get_gt_batch(batch, device):
    """Process ground truth channel matrices from a batch"""
    return batch["channel_matrix"].to(device)


def setup_experiment(args):
    """Setup experiment directory and logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = log_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    logger = setup_logging(log_dir)
    logger.info(f"Arguments: {args}")

    writer = None
    if args.tensorboard:
        logger.info("Initializing TensorBoard writer...")
        writer = SummaryWriter(str(log_dir / "tensorboard"))

    return logger, writer, log_dir


def evaluate(
    model,
    dataloader,
    tx_params,
    rx_params,
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

    total_error_sq_sum = 0.0
    total_gt_sq_sum = 0.0
    total_samples = 0

    logger.info(f"Evaluating at iteration {iteration}...")

    with torch.no_grad():
        enc_data = {"tx_pos": tx_params["position"]}
        model.embed_features(enc_data)

        for batch in tqdm(dataloader, desc="Evaluating"):
            gt_channels = get_gt_batch(batch, device)
            batch_size = gt_channels.shape[0]

            pred_channels = sequential_fwd(
                model,
                batch,
                tx_params,
                rx_params,
                args,
                device,
                update_features=False,
                is_training=False,
            )

            loss = loss_fn(pred_channels, gt_channels)

            total_loss += loss.item() * batch_size

            pred_real, pred_imag = pred_channels[..., 0], pred_channels[..., 1]
            target_real, target_imag = gt_channels[..., 0], gt_channels[..., 1]
            pred_complex = torch.complex(pred_real, pred_imag)
            target_complex = torch.complex(target_real, target_imag)

            batch_error_sq_sum = torch.sum(
                torch.abs(pred_complex - target_complex) ** 2
            )
            batch_gt_sq_sum = torch.sum(torch.abs(target_complex) ** 2)

            total_error_sq_sum += batch_error_sq_sum.item()
            total_gt_sq_sum += batch_gt_sq_sum.item()
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)

    eps = 1e-9
    overall_nmse = total_error_sq_sum / max(total_gt_sq_sum, eps)
    overall_snr = calculate_snr(torch.tensor(overall_nmse)).item()

    logger.info(
        f"Evaluation Loss ({args.loss_type}): {avg_loss:.6f}, SNR: {overall_snr:.2f} dB (NMSE: {overall_nmse:.6f})"
    )

    if writer is not None:
        writer.add_scalar("eval/loss", avg_loss, iteration)
        writer.add_scalar("eval/overall_nmse", overall_nmse, iteration)
        writer.add_scalar("eval/overall_snr", overall_snr, iteration)

    model.train()

    return avg_loss


def compute_grad_stats(model):
    """Compute statistics about gradients for model parameters."""
    grad_stats = {
        "mean_abs": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
        "norm": 0.0,
        "param_count": 0,
    }

    all_params_with_grad = []
    if model.optimizer:
        for group in model.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    all_params_with_grad.append(param)

    if model.encoder_optimizer:
        for group in model.encoder_optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    all_params_with_grad.append(param)

    if not all_params_with_grad:
        return grad_stats

    total_grad_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in all_params_with_grad]), 2
    )
    grad_stats["norm"] = total_grad_norm.item()

    total_params_numel = 0
    for param in all_params_with_grad:
        grad_abs = param.grad.abs()
        grad_stats["mean_abs"] += grad_abs.sum().item()
        current_min = param.grad.min().item()
        current_max = param.grad.max().item()
        if not torch.isnan(torch.tensor(current_min)):
            grad_stats["min"] = min(grad_stats["min"], current_min)
        if not torch.isnan(torch.tensor(current_max)):
            grad_stats["max"] = max(grad_stats["max"], current_max)
        grad_stats["param_count"] += 1
        total_params_numel += param.numel()

    if total_params_numel > 0:
        grad_stats["mean_abs"] /= total_params_numel
    else:
        grad_stats["mean_abs"] = 0.0

    if grad_stats["min"] == float("inf"):
        grad_stats["min"] = 0.0
    if grad_stats["max"] == float("-inf"):
        grad_stats["max"] = 0.0

    return grad_stats


def train(args, logger, writer, log_dir):
    """Main training loop"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Initializing dataloaders...")
    train_dataloader, val_dataloader = get_dataloaders(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )

    # get static environment data
    try:
        point_cloud_full = train_dataloader.dataset.get_point_cloud()
        if args.num_points < len(point_cloud_full):
            pc_indices = torch.randperm(len(point_cloud_full))[: args.num_points]
            point_cloud = point_cloud_full[pc_indices]
        else:
            point_cloud = point_cloud_full

        tx_position = train_dataloader.dataset.get_tx_position().to(device)
        env_dims = train_dataloader.dataset.get_env_dims()
        frequency = train_dataloader.dataset.frequency
        num_tx_ant = train_dataloader.dataset.num_tx_ant
        num_rx_ant = train_dataloader.dataset.num_rx_ant
        scene_extent = (env_dims[:, 1] - env_dims[:, 0]).max().item()

        wavelength = 299792458.0 / frequency

        tx_params = {
            "position": tx_position,
            "type": "ura",
            "size": [int(num_tx_ant**0.5), int(num_tx_ant**0.5)],
            "element_spacing": [
                wavelength / 2,
                wavelength / 2,
            ],  # half-wavelength spacing
            "frequency": frequency,
            "num_antennas": num_tx_ant,
        }

        rx_params = {
            "position": None,
            "type": "ula",
            "size": num_rx_ant,
            "element_spacing": wavelength / 2,
            "num_antennas": num_rx_ant,
        }

    except Exception as e:
        logger.error(f"Failed to load dataset properties: {e}")
        raise

    logger.info(f"Number of TX antennas: {num_tx_ant}")
    logger.info(f"Number of RX antennas: {num_rx_ant}")
    logger.info(f"Environment extent: {scene_extent:.2f}")
    logger.info(f"Operating frequency: {frequency/1e9:.2f} GHz")
    logger.info(f"Training with batch size: {args.batch_size}")
    logger.info(f"Number of Gaussians: {args.num_points}")
    if args.rx_noise_std > 0:
        logger.info(f"Using RX position noise with std dev: {args.rx_noise_std}")
    if args.gaussian_dropout_rate > 0:
        logger.info(f"Using Gaussian dropout with rate: {args.gaussian_dropout_rate}")

    logger.info("Initializing model...")
    encoder_cfg = EncoderConfig(
        hidden_size=128,
        num_layers=8,
        skip_layers=(4,),
        input_pos_multires=8,
        use_positional_encoding=args.use_positional_encoding,
        use_layer_norm=args.use_encoder_layernorm,
        dropout_prob=args.dropout_prob,
        base_feature_dim=64,
    )
    dir_net_cfg = DirNetworkConfig(
        hidden_size=64,
        num_layers=4,
        input_dir_multires=4,
        use_positional_encoding=args.use_positional_encoding,
        use_layer_norm=args.use_dirnet_layernorm,
        dropout_prob=0.0,
    )
    model = GaussianModel(encoder_cfg=encoder_cfg, dir_net_cfg=dir_net_cfg)

    start_iteration = 0
    best_val_loss = float("inf")

    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        try:
            model = GaussianModel.load(args.resume, device=device, training_args=args)
            checkpoint = torch.load(args.resume, map_location=device)
            start_iteration = checkpoint.get("iteration", 0) + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            logger.info(
                f"Resuming from iteration {start_iteration}, best val loss: {best_val_loss:.6f}"
            )
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}. Starting from scratch.")
            args.resume = None

    if args.resume is None:
        model = model.to(device)
        if args.init_method == "point_cloud":
            model.init_from_pc(
                point_cloud.to(device),
                tx_position=tx_position,
            )
            logger.info(
                f"Initialized model with {point_cloud.shape[0]} Gaussians from point cloud"
            )
        else:
            model.init_randomly(
                args.num_points,
                env_dims.to(device),
                tx_position=tx_position,
            )
            logger.info(f"Initialized model with {args.num_points} random Gaussians")

        model.training_setup(args)

    model.train()

    loss_kwargs = {
        "eps": args.loss_eps,
        "phase_weight": args.phase_weight,
    }
    loss_fn = get_loss_function(args.loss_type, **loss_kwargs).to(device)
    logger.info(f"Using {args.loss_type} loss function with params: {loss_kwargs}")

    # training loop
    logger.info("Starting training...")
    train_iter = iter(train_dataloader)

    progress_bar = tqdm(range(start_iteration, args.iterations), desc="Training")
    ema_loss = -1.0

    for iteration in progress_bar:
        iter_start_time = time.time()
        model.train()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        pred_channels = sequential_fwd(
            model,
            batch,
            tx_params,
            rx_params,
            args,
            device,
            update_features=True,
            is_training=True,  # Pass is_training=True for training forward pass
        )

        gt_channels = get_gt_batch(batch, device)

        loss = loss_fn(pred_channels, gt_channels)

        with torch.no_grad():
            nmse = calculate_nmse(pred_channels, gt_channels)
            snr = calculate_snr(nmse).item()
            if ema_loss < 0:
                ema_loss = loss.item()
            else:
                ema_loss = 0.9 * ema_loss + 0.1 * loss.item()

        if model.optimizer:
            model.optimizer.zero_grad()
        if model.encoder_optimizer:
            model.encoder_optimizer.zero_grad()

        loss.backward()

        if args.gradient_clip_val > 0:
            if model.optimizer:
                all_gaussian_params = [
                    p
                    for group in model.optimizer.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
                if all_gaussian_params:
                    clip_grad_norm_(all_gaussian_params, args.gradient_clip_val)

            if model.encoder_optimizer:
                encoder_params = [
                    p
                    for group in model.encoder_optimizer.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
                if encoder_params:
                    clip_grad_norm_(encoder_params, args.gradient_clip_val)

        grad_stats = compute_grad_stats(model)

        if model.optimizer:
            model.optimizer.step()
        if model.encoder_optimizer:
            model.encoder_optimizer.step()

        model.update_learning_rate(iteration)

        if (
            args.opacity_reset
            and iteration > 0
            and iteration % args.opacity_reset_interval == 0
        ):
            logger.info(f"Resetting opacity at iteration {iteration}")
            model.reset_opacity()

        # log progress
        iter_time = time.time() - iter_start_time
        if iteration % args.log_freq == 0:
            log_msg = (
                f"[{iteration}/{args.iterations}] "
                f"Loss: {loss.item():.6f} [EMA: {ema_loss:.6f}], "
                f"SNR: {snr:.2f} dB, "
                f"Time: {iter_time:.2f}s, "
                f"Gaussians: {model.get_xyz.shape[0]}"
            )
            logger.info(log_msg)
            grad_log_msg = (
                f"Grad stats: Norm: {grad_stats['norm']:.4e}, Mean abs: {grad_stats['mean_abs']:.4e}, "
                f"Min: {grad_stats['min']:.4e}, Max: {grad_stats['max']:.4e}"
            )
            logger.info(grad_log_msg)

            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), iteration)
                writer.add_scalar("train/ema_loss", ema_loss, iteration)
                writer.add_scalar("train/nmse", nmse.item(), iteration)
                writer.add_scalar("train/snr", snr, iteration)
                writer.add_scalar("train/iteration_time", iter_time, iteration)
                writer.add_scalar(
                    "train/num_gaussians", model.get_xyz.shape[0], iteration
                )
                if iter_time > 0:
                    writer.add_scalar(
                        "train/samples_per_second",
                        args.batch_size / iter_time,
                        iteration,
                    )

                if model.optimizer:
                    for i, param_group in enumerate(model.optimizer.param_groups):
                        writer.add_scalar(
                            f"lr/{param_group['name']}", param_group["lr"], iteration
                        )
                if model.encoder_optimizer:
                    writer.add_scalar(
                        "lr/encoder",
                        model.encoder_optimizer.param_groups[0]["lr"],
                        iteration,
                    )

                writer.add_scalar("grad/norm", grad_stats["norm"], iteration)
                writer.add_scalar("grad/mean_abs", grad_stats["mean_abs"], iteration)
                writer.add_scalar("grad/min", grad_stats["min"], iteration)
                writer.add_scalar("grad/max", grad_stats["max"], iteration)
                writer.add_scalar(
                    "hyperparams/gaussian_dropout_rate",
                    args.gaussian_dropout_rate,
                    iteration,
                )

        if (
            iteration > 0 and iteration % args.eval_freq == 0
        ) or iteration == args.iterations - 1:
            val_loss = evaluate(
                model,
                val_dataloader,
                tx_params,
                rx_params,
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
                logger.info(
                    f"New best validation loss ({args.loss_type}): {best_val_loss:.6f}"
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
            checkpoint_path = log_dir / "checkpoints" / f"checkpoint_{iteration:07d}.pt"
            model.save(
                checkpoint_path,
                save_optimizer=True,
                iteration=iteration,
                best_val_loss=best_val_loss,
            )
            logger.info(f"Checkpoint saved at iteration {iteration}")

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
    """Main entry point for the training script."""
    args = parse_args()
    set_random_seed(args.seed)

    logger, writer, log_dir = setup_experiment(args)

    try:
        train(args, logger, writer, log_dir)
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

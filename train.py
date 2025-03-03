# train.py

import argparse
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from core.transforms import project_to_channel_space
from datasets.dataloader import get_wireless_dataloader
from models import EncoderConfig
from models.encoder import EncoderConfig
from models.gaussian_model import GaussianModel, GaussianModelConfig
from utils.general_utils import set_random_seed
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

    # training params
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Directory to save outputs"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=200000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=5000,
        help="Save checkpoint every N iterations",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

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
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir)

    writer = None
    if args.tensorboard:
        logger.info("Initializing TensorBoard writer...")
        writer = SummaryWriter(log_dir / "tensorboard")

    return logger, writer


def train(args, logger, writer):
    """Main training loop"""
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    logger.info("Initializing dataloader...")
    dataloader = get_wireless_dataloader(
        args.data_path,
        drop_last=True,
    )

    # get static environment data
    point_cloud = dataloader.dataset.get_point_cloud(args.num_points)
    tx_position = dataloader.dataset.get_tx_position().to(device)
    env_dims = dataloader.dataset.get_env_dims()
    num_tx_ant = dataloader.dataset.num_tx_ant
    num_rx_ant = dataloader.dataset.num_rx_ant

    logger.info(f"Number of TX antennas: {num_tx_ant}")
    logger.info(f"Number of RX antennas: {num_rx_ant}")

    logger.info("Initializing model...")
    model_cfg = GaussianModelConfig(
        use_pred_normals=args.use_pred_normals,
    )
    encoder_cfg = EncoderConfig(
        use_attention=args.use_attention, max_paths=args.max_paths
    )
    model = GaussianModel(model_cfg=model_cfg, encoder_cfg=encoder_cfg).to(device)

    # initialize model with point cloud
    model.init_from_pc(point_cloud.to(device))
    logger.info(f"Initialized model with {len(point_cloud)} Gaussians")

    data = next(iter(dataloader))
    rx_position = data["rx_position"].to(device).squeeze()  # [3]
    aoa = data["aoa"][0].to(device)  # [2, num_paths], 2 for azimuth and elevation
    path_loss = data["path_loss"].to(device)  # [1]
    path_loss_per_ray = data["path_loss_per_ray"][0].to(device)  # [num_paths]

    wireless_data = {
        "tx_pos": tx_position,
        "rx_pos": rx_position,
        "path_loss": path_loss,
        "aoa": aoa,
        "path_loss_per_ray": path_loss_per_ray,
    }

    logger.info("Embedding wireless features...")
    model.embed_features(wireless_data)
    logger.info(f"Shape of features: {model.get_features.shape}")

    logger.info("Projecting to channel space...")
    proj_dict = project_to_channel_space(
        points=model.get_xyz,
        cov3d=model.get_covariance(),
        receiver=rx_position,
        num_tx=num_tx_ant,
        num_rx=num_rx_ant,
    )

    # training loop
    logger.info("Starting training...")
    for iteration in range(args.num_iterations):
        # TODO: actual training iteration

        # log metrics
        if iteration % 100 == 0:
            if writer is not None:
                # TODO: log actual metrics with writer
                pass
            # TODO: log to console/file regardless of tensorboard

        # save checkpoint
        if iteration % args.checkpoint_freq == 0:
            # TODO: save checkpoint
            pass


def main():
    args = parse_args()
    set_random_seed(args.seed)

    logger, writer = setup_experiment(args)

    try:
        train(args, logger, writer)
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

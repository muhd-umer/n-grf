# train.py

import argparse
import logging
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from datasets.dataloader import get_wireless_dataloader
from models import EncoderConfig, WirelessEncoder, get_embedder
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
        "--sh_degree",
        type=int,
        default=1,
        help="Maximum degree of spherical harmonics (only 0 or 1 supported)",
    )
    parser.add_argument(
        "--use_pred_normals",
        action="store_true",
        help="Whether to predict surface normals",
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
    parser.add_argument(
        "--stl_path",
        type=str,
        default="datasets/models/conference.stl",
        help="Path to reference STL model file",
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
    logger.info("Initializing dataloader...")
    dataloader = get_wireless_dataloader(
        args.data_path,
        drop_last=True,
    )

    # get static environment data
    point_cloud = dataloader.dataset.get_point_cloud(args.num_points)
    tx_position = dataloader.dataset.get_tx_position()
    env_dims = dataloader.dataset.get_env_dims()

    logger.info("Initializing model...")
    model_config = GaussianModelConfig(
        sh_degree=args.sh_degree,
        max_sh_degree=args.sh_degree,
        use_pred_normals=args.use_pred_normals,
    )
    model = GaussianModel(model_config).to(args.device)

    # initialize model with point cloud
    model.init_from_pc(point_cloud.to(args.device))
    logger.info(f"Initialized model with {len(point_cloud)} Gaussians")

    # log time
    start_time = time.time()
    cov3d = model.get_covariance()
    end_time = time.time()

    logger.info(f"Covariance computation time: {end_time - start_time:.4f} seconds")

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

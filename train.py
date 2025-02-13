# train.py

import argparse
import logging
from pathlib import Path

import torch

from datasets.dataloader import get_wireless_dataloader
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
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=16378,
        help="Number of points to sample from point cloud",
    )

    # model params
    parser.add_argument(
        "--sh_degree", type=int, default=3, help="Maximum degree of spherical harmonics"
    )
    parser.add_argument(
        "--use_pred_normals",
        action="store_true",
        help="Whether to predict surface normals",
    )

    # training params
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Directory to save outputs"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    set_random_seed(args.seed)

    logger.info("Initializing dataloader...")
    dataloader = get_wireless_dataloader(
        args.data_path,
        batch_size=args.batch_size,
        num_pc=args.num_points,
        drop_last=True,
    )

    batch = next(iter(dataloader))

    model_config = GaussianModelConfig(
        sh_degree=args.sh_degree,
        max_sh_degree=args.sh_degree,
        use_pred_normals=args.use_pred_normals,
    )

    logger.info("Initializing Gaussian model...")
    model = GaussianModel(model_config).to(args.device)

    # initialize Gaussians from point cloud
    point_cloud = batch["point_cloud"].to(args.device)
    model.init_from_pc(point_cloud)

    logger.info(f"Initialized {point_cloud.shape[0]} Gaussians from point cloud")


if __name__ == "__main__":
    main()

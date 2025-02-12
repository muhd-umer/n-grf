# train.py

import argparse
import logging
from pathlib import Path

import torch

from models import GaussianConfig, GaussianModel, Scene


def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Gaussian initialization test")

    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--num_pts", type=int, default=100000)
    parser.add_argument("--opacity_init", type=float, default=0.1)
    parser.add_argument("--scaling_init", type=float, default=0.1)

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this implementation")

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    torch.manual_seed(args.seed)

    logger.info("Initializing model and scene...")
    config = GaussianConfig(
        sh_degree=args.sh_degree,
        num_pts=args.num_pts,
        opacity_init=args.opacity_init,
        scaling_init=args.scaling_init,
    )

    try:
        model = GaussianModel(config=config).to(device)
        scene = Scene(
            gaussians=model,
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )

        logger.info(f"Number of Gaussians: {len(model.get_xyz)}")
        logger.info(f"Gaussian positions shape: {model.get_xyz.shape}")
        logger.info(f"Gaussian features shape: {model.get_features.shape}")
        logger.info(f"Gaussian scaling shape: {model.get_scaling.shape}")
        logger.info(f"Gaussian rotation shape: {model.get_rotation.shape}")
        logger.info(f"Gaussian opacity shape: {model.get_opacity.shape}")

        logger.info(f"Environment dimensions: {scene.env_dims}")
        logger.info(f"Transmitter position: {scene.tx_position}")

        logger.info("Initialization test completed!")

    except Exception as e:
        logger.error(f"Failed to initialize: {str(e)}")
        raise


if __name__ == "__main__":
    main()

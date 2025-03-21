# engine/tests/benchmark.py

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

import engine
import engine._torch_impl as torch_impl
from datasets.dataloader import get_dataloaders
from models.encoder import EncoderConfig
from models.gaussian_model import GaussianModel

NUM_RUNS = 10

TEST_DATA = None
LOG_PATH = f"{Path(__file__).parent}/benchmark_res.log"


def load_global_test_data(data_path, batch_size=12_000, device="cuda"):
    global TEST_DATA

    print(f"Loading data from dataset at {data_path}")

    train_dataloader, _ = get_dataloaders(
        data_path,
        batch_size=1,
        num_workers=2,
        drop_last=True,
    )

    point_cloud = train_dataloader.dataset.get_point_cloud(batch_size)
    tx_position = train_dataloader.dataset.get_tx_position().to(device)
    frequency = train_dataloader.dataset.frequency
    num_tx_ant = train_dataloader.dataset.num_tx_ant
    num_rx_ant = train_dataloader.dataset.num_rx_ant

    data_batch = next(iter(train_dataloader))
    rx_position = data_batch["rx_position"].to(device).squeeze()

    encoder_cfg = EncoderConfig(
        hidden_size=128,
        num_layers=8,
        skip_layers=(4,),
        input_pos_multires=10,
        use_positional_encoding=True,
    )
    model = GaussianModel(encoder_cfg=encoder_cfg).to(device)

    model.init_from_pc(
        point_cloud.to(device),
        tx_position=tx_position,
        frequency=frequency,
        use_physics_init=True,
    )

    enc_data = {
        "tx_pos": tx_position,
        "rx_pos": rx_position,
        "frequency": frequency,
    }
    model.embed_features(enc_data)

    TEST_DATA = {
        "points": model.get_xyz,
        "cov3d": model.get_covariance(),
        "attenuation": model.get_features[:, 0:1],
        "phase_rotation": model.get_features[:, 1:2],
        "opacity": model.get_opacity,
        "receiver": rx_position,
        "transmitter": tx_position,
        "num_tx": num_tx_ant,
        "num_rx": num_rx_ant,
        "frequency": frequency,
        "wavelength": 299792458.0 / frequency,
    }

    return TEST_DATA


def benchmark_rasterize():
    """
    Benchmark the rasterize function, comparing PyTorch vs CUDA implementation
    """
    torch.cuda.synchronize()

    torch_impl.rasterize(
        points=TEST_DATA["points"],
        cov3d=TEST_DATA["cov3d"],
        attenuation=TEST_DATA["attenuation"],
        phase_rotation=TEST_DATA["phase_rotation"],
        opacity=TEST_DATA["opacity"],
        receiver=TEST_DATA["receiver"],
        transmitter=TEST_DATA["transmitter"],
        num_tx=TEST_DATA["num_tx"],
        num_rx=TEST_DATA["num_rx"],
        frequency=TEST_DATA["frequency"],
    )
    torch.cuda.synchronize()

    engine.rasterize(
        points=TEST_DATA["points"],
        cov3d=TEST_DATA["cov3d"],
        attenuation=TEST_DATA["attenuation"],
        phase_rotation=TEST_DATA["phase_rotation"],
        opacity=TEST_DATA["opacity"],
        receiver=TEST_DATA["receiver"],
        transmitter=TEST_DATA["transmitter"],
        num_tx=TEST_DATA["num_tx"],
        num_rx=TEST_DATA["num_rx"],
        frequency=TEST_DATA["frequency"],
    )
    torch.cuda.synchronize()

    torch_times = []
    for i in range(NUM_RUNS):
        start = time.time()
        torch_impl.rasterize(
            points=TEST_DATA["points"],
            cov3d=TEST_DATA["cov3d"],
            attenuation=TEST_DATA["attenuation"],
            phase_rotation=TEST_DATA["phase_rotation"],
            opacity=TEST_DATA["opacity"],
            receiver=TEST_DATA["receiver"],
            transmitter=TEST_DATA["transmitter"],
            num_tx=TEST_DATA["num_tx"],
            num_rx=TEST_DATA["num_rx"],
            frequency=TEST_DATA["frequency"],
        )
        torch.cuda.synchronize()
        end = time.time()
        torch_times.append(end - start)
        print(f"PyTorch run {i+1}/{NUM_RUNS} completed: {torch_times[-1]:.6f}s")

    torch_avg = np.mean(torch_times)
    torch_std = np.std(torch_times)

    cuda_times = []
    for i in range(NUM_RUNS):
        start = time.time()
        engine.rasterize(
            points=TEST_DATA["points"],
            cov3d=TEST_DATA["cov3d"],
            attenuation=TEST_DATA["attenuation"],
            phase_rotation=TEST_DATA["phase_rotation"],
            opacity=TEST_DATA["opacity"],
            receiver=TEST_DATA["receiver"],
            transmitter=TEST_DATA["transmitter"],
            num_tx=TEST_DATA["num_tx"],
            num_rx=TEST_DATA["num_rx"],
            frequency=TEST_DATA["frequency"],
        )
        torch.cuda.synchronize()
        end = time.time()
        cuda_times.append(end - start)
        print(f"CUDA run {i+1}/{NUM_RUNS} completed: {cuda_times[-1]:.6f}s")

    cuda_avg = np.mean(cuda_times)
    cuda_std = np.std(cuda_times)

    speedup = torch_avg / cuda_avg if cuda_avg > 0 else float("inf")

    return {
        "torch_avg": torch_avg,
        "torch_std": torch_std,
        "cuda_avg": cuda_avg,
        "cuda_std": cuda_std,
        "speedup": speedup,
    }


def format_results(result):
    """Format benchmark results as a string"""
    lines = []
    lines.append("Rasterize Function Benchmark Results:")
    lines.append("-" * 50)
    lines.append(
        f"PyTorch Implementation: {result['torch_avg']:.6f} +/- {result['torch_std']:.6f} seconds"
    )
    lines.append(
        f"CUDA Implementation:    {result['cuda_avg']:.6f} +/- {result['cuda_std']:.6f} seconds"
    )
    lines.append("-" * 50)
    lines.append(f"Speedup:                {result['speedup']:.2f}x")

    lines.append("\nDetailed Results:")
    lines.append("+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 20 + "+")
    lines.append(
        "|"
        + "Implementation".center(20)
        + "|"
        + "Average Time (s)".center(20)
        + "|"
        + "Speedup".center(20)
        + "|"
    )
    lines.append("+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 20 + "+")
    lines.append(
        "|"
        + "PyTorch".center(20)
        + "|"
        + f"{result['torch_avg']:.6f}".center(20)
        + "|"
        + "-".center(20)
        + "|"
    )
    lines.append(
        "|"
        + "CUDA".center(20)
        + "|"
        + f"{result['cuda_avg']:.6f}".center(20)
        + "|"
        + f"{result['speedup']:.2f}x".center(20)
        + "|"
    )
    lines.append("+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 20 + "+")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch and CUDA rasterize implementations"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets/outputs/conf_16x2_414u_5.0ghz_sbrRT_sc104.mat",
        help="Path to the dataset file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=LOG_PATH,
        help="Path to the output log file",
    )
    args = parser.parse_args()

    if not engine.CUDA_AVAILABLE:
        print("CUDA implementation not available. Cannot run benchmarks.")
        return

    try:

        load_global_test_data(args.data_path)

        print(f"Running rasterize benchmark with {NUM_RUNS} iterations...")
        result = benchmark_rasterize()

        formatted_results = format_results(result)

        header = [
            f"Benchmark Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Number of runs per test: {NUM_RUNS}",
            f"Dataset: {args.data_path}",
            f"Device: {torch.cuda.get_device_name(0)}",
            f"PyTorch version: {torch.__version__}",
            "-" * 80,
            "",
        ]

        complete_output = "\n".join(header) + "\n" + formatted_results

        with open(args.output, "w", encoding="utf-8") as f:
            f.write(complete_output)

        print("\n" + complete_output)
        print(f"\nBenchmark results saved to {args.output}")

    except Exception as e:
        print(f"Error during benchmarking: {e}")
        raise


if __name__ == "__main__":
    main()

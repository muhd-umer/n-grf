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
from utils.transform_utils import build_scaling_rotation, strip_symmetric

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

    points = model.get_xyz
    scaling = model.get_scaling
    rotation = model.get_rotation
    opacity = model.get_opacity
    attenuation = model.get_features[:, 0:1]
    phase_rotation = model.get_features[:, 1:2]

    cov3d_precomp = model.get_covariance()

    TEST_DATA = {
        "points": points,
        "scaling": scaling,
        "rotation": rotation,
        "cov3d_precomp": cov3d_precomp,
        "attenuation": attenuation,
        "phase_rotation": phase_rotation,
        "opacity": opacity,
        "receiver": rx_position,
        "transmitter": tx_position,
        "num_tx": num_tx_ant,
        "num_rx": num_rx_ant,
        "frequency": frequency,
        "wavelength": 299792458.0 / frequency,
        "scale_modifier": 1.0,
    }

    return TEST_DATA


def benchmark_pytorch_impl():
    """
    Benchmark the PyTorch implementation, including covariance computation
    """
    torch.cuda.synchronize()

    scaling = TEST_DATA["scaling"]
    rotation = TEST_DATA["rotation"]
    scale_modifier = TEST_DATA["scale_modifier"]

    times = []
    for i in range(NUM_RUNS):
        start = time.time()

        L = build_scaling_rotation(scale_modifier * scaling, rotation)
        covariance = torch.bmm(L, L.transpose(1, 2))
        cov3d = strip_symmetric(covariance)

        torch_impl.rasterize(
            points=TEST_DATA["points"],
            cov3d=cov3d,
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
        times.append(end - start)
        print(f"PyTorch run {i+1}/{NUM_RUNS} completed: {times[-1]:.6f}s")

    avg_time = np.mean(times)
    std_time = np.std(times)

    return {
        "avg": avg_time,
        "std": std_time,
        "times": times,
    }


def benchmark_cuda_impl():
    """
    Benchmark the CUDA implementation, including covariance computation
    """
    torch.cuda.synchronize()

    scaling = TEST_DATA["scaling"]
    rotation = TEST_DATA["rotation"]
    scale_modifier = TEST_DATA["scale_modifier"]

    times = []
    for i in range(NUM_RUNS):
        start = time.time()

        engine.rasterize(
            points=TEST_DATA["points"],
            cov3d=None,
            attenuation=TEST_DATA["attenuation"],
            phase_rotation=TEST_DATA["phase_rotation"],
            opacity=TEST_DATA["opacity"],
            receiver=TEST_DATA["receiver"],
            transmitter=TEST_DATA["transmitter"],
            num_tx=TEST_DATA["num_tx"],
            num_rx=TEST_DATA["num_rx"],
            frequency=TEST_DATA["frequency"],
            scaling=scaling,
            rotation=rotation,
            scale_modifier=scale_modifier,
        )

        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
        print(f"CUDA run {i+1}/{NUM_RUNS} completed: {times[-1]:.6f}s")

    avg_time = np.mean(times)
    std_time = np.std(times)

    return {
        "avg": avg_time,
        "std": std_time,
        "times": times,
    }


def benchmark_cuda_parts():
    """
    Benchmark individual parts of the CUDA implementation to identify bottlenecks
    """

    torch.cuda.synchronize()

    scaling = TEST_DATA["scaling"]
    rotation = TEST_DATA["rotation"]
    scale_modifier = TEST_DATA["scale_modifier"]

    cov_times = []
    for i in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = time.time()

        cov3d = engine._C.compute_cov3d_from_scaling_rotation(
            scaling, rotation, scale_modifier
        )

        torch.cuda.synchronize()
        end = time.time()
        cov_times.append(end - start)

    cov_avg = np.mean(cov_times)

    rast_times = []
    cov3d = TEST_DATA["cov3d_precomp"]

    for i in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = time.time()

        engine.rasterize(
            points=TEST_DATA["points"],
            cov3d=cov3d,
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
        rast_times.append(end - start)

    rast_avg = np.mean(rast_times)

    return {
        "cov_avg": cov_avg,
        "cov_times": cov_times,
        "rast_avg": rast_avg,
        "rast_times": rast_times,
    }


def benchmark_rasterize():
    """
    Comprehensive benchmark comparing PyTorch vs CUDA implementations
    """
    torch.cuda.synchronize()

    print(f"\nRunning benchmarks with {NUM_RUNS} iterations per test...")

    print("\n=== PyTorch Implementation (including covariance computation) ===")
    pytorch_results = benchmark_pytorch_impl()

    print("\n=== CUDA Implementation (including covariance computation) ===")
    cuda_results = benchmark_cuda_impl()

    print("\n=== CUDA Implementation (detailed breakdown) ===")
    cuda_parts = benchmark_cuda_parts()

    pytorch_avg = pytorch_results["avg"]
    cuda_avg = cuda_results["avg"]
    speedup = pytorch_avg / cuda_avg if cuda_avg > 0 else float("inf")

    return {
        "torch_avg": pytorch_avg,
        "torch_std": pytorch_results["std"],
        "cuda_avg": cuda_avg,
        "cuda_std": cuda_results["std"],
        "speedup": speedup,
        "detailed": {
            "cuda_cov_time": cuda_parts["cov_avg"],
            "cuda_rast_time": cuda_parts["rast_avg"],
            "cuda_cov_percent": (
                (cuda_parts["cov_avg"] / cuda_avg) * 100 if cuda_avg > 0 else 0
            ),
            "cuda_rast_percent": (
                (cuda_parts["rast_avg"] / cuda_avg) * 100 if cuda_avg > 0 else 0
            ),
        },
    }


def format_results(result):
    """Format benchmark results as a string"""
    lines = []
    lines.append("Channel Rasterization Benchmark Results:")
    lines.append("-" * 60)
    lines.append(
        f"PyTorch Implementation: {result['torch_avg']:.6f} ± {result['torch_std']:.6f} seconds"
    )
    lines.append(
        f"CUDA Implementation:    {result['cuda_avg']:.6f} ± {result['cuda_std']:.6f} seconds"
    )
    lines.append("-" * 60)
    lines.append(f"Speedup:                {result['speedup']:.2f}x")

    lines.append("\nCUDA Implementation Breakdown:")
    lines.append(
        f"Covariance Computation: {result['detailed']['cuda_cov_time']:.6f} seconds "
        + f"({result['detailed']['cuda_cov_percent']:.1f}% of total)"
    )
    lines.append(
        f"Rasterization:         {result['detailed']['cuda_rast_time']:.6f} seconds "
        + f"({result['detailed']['cuda_rast_percent']:.1f}% of total)"
    )

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
    global NUM_RUNS

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
    parser.add_argument(
        "--num_runs",
        type=int,
        default=NUM_RUNS,
        help="Number of iterations per benchmark test",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12_000,
        help="Number of Gaussians to use in the benchmark",
    )
    args = parser.parse_args()

    NUM_RUNS = args.num_runs

    if not engine.CUDA_AVAILABLE:
        print("CUDA implementation not available. Cannot run benchmarks.")
        return

    try:
        load_global_test_data(args.data_path, batch_size=args.batch_size)

        print(f"Running rasterize benchmark with {NUM_RUNS} iterations...")
        result = benchmark_rasterize()

        formatted_results = format_results(result)

        header = [
            f"Benchmark Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Number of runs per test: {NUM_RUNS}",
            f"Number of Gaussians: {args.batch_size}",
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

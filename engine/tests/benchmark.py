# engine/tests/benchmark.py

import argparse
import math
import os
import sys
import time
import unittest

import torch
import torch.nn.functional as F
from tabulate import tabulate

try:

    from _wrapper import CUDA_AVAILABLE
    from _wrapper import rasterize as cuda_rasterize

    if not CUDA_AVAILABLE:
        raise ImportError("CUDA extension loaded but CUDA is not available.")
except ImportError:
    CUDA_AVAILABLE = False
    print(
        "CUDA wrapper not found or CUDA not available. CUDA benchmarks will be skipped.",
        file=sys.stderr,
    )

    def cuda_rasterize(*args, **kwargs):

        raise unittest.SkipTest("CUDA implementation not available")


from _torch_impl.rasterize import rasterize as torch_rasterize


def generate_inputs(N, num_tx, num_rx, frequency, scale_modifier, dtype, device):
    """Generates random inputs for the rasterize function."""
    points = torch.rand(N, 3, device=device, dtype=dtype) * 10
    scaling = torch.rand(N, 3, device=device, dtype=dtype)

    rotation = torch.rand(N, 4, device=device, dtype=dtype)
    gamma_real = torch.randn(N, device=device, dtype=dtype) * 1e-2
    gamma_imag = torch.randn(N, device=device, dtype=dtype) * 1e-2
    gamma = torch.stack([gamma_real, gamma_imag], dim=1)
    opacity = torch.rand(N, 1, device=device, dtype=dtype) * 0.9 + 0.05

    points.requires_grad_(True)
    scaling.requires_grad_(True)
    rotation.requires_grad_(True)
    gamma.requires_grad_(True)
    opacity.requires_grad_(True)

    wavelength = 299792458.0 / frequency
    tx_params = {
        "position": torch.rand(3, device=device, dtype=dtype) * 10,
        "type": "ura",
        "size": [int(math.sqrt(num_tx)), int(math.sqrt(num_tx))],
        "element_spacing": [wavelength / 2, wavelength / 2],
        "frequency": frequency,
        "num_antennas": num_tx,
    }
    rx_params = {
        "position": torch.rand(3, device=device, dtype=dtype) * 10 + 5,
        "type": "ula",
        "size": num_rx,
        "element_spacing": wavelength / 2,
        "num_antennas": num_rx,
    }

    inputs_dict = {
        "points": points,
        "scaling": scaling,
        "rotation": rotation,
        "gamma": gamma,
        "opacity": opacity,
        "tx_params": tx_params,
        "rx_params": rx_params,
        "scale_modifier": scale_modifier,
    }

    grad_tensors = (points, scaling, rotation, gamma, opacity)

    return inputs_dict, grad_tensors


def benchmark_function(
    func, inputs_dict, grad_tensors, n_runs, n_warmup, label, device
):
    """Benchmarks forward and backward pass of a given function."""
    print(f"Benchmarking {label}...")

    print(f"  Warmup ({n_warmup} runs)...")
    for _ in range(n_warmup):

        for p in grad_tensors:
            if p.grad is not None:
                p.grad.zero_()
        output = func(**inputs_dict)

        loss = output.sum()
        loss.backward()

    print(f"  Forward Pass ({n_runs} runs)...")
    fwd_times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(n_runs):
        start_event.record()
        output = func(**inputs_dict)
        end_event.record()
        torch.cuda.synchronize()
        fwd_times.append(start_event.elapsed_time(end_event))

    avg_fwd_time_ms = sum(fwd_times) / n_runs

    print(f"  Backward Pass ({n_runs} runs)...")

    for p in grad_tensors:
        if p.grad is not None:
            p.grad.zero_()
    output = func(**inputs_dict)
    loss = output.sum()

    bwd_times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(n_runs):

        for p in grad_tensors:
            if p.grad is not None:
                p.grad.zero_()

        start_event.record()
        loss.backward(retain_graph=True)
        end_event.record()
        torch.cuda.synchronize()
        bwd_times.append(start_event.elapsed_time(end_event))

    avg_bwd_time_ms = sum(bwd_times) / n_runs

    print(
        f"  {label}: Avg Fwd: {avg_fwd_time_ms:.3f} ms, Avg Bwd: {avg_bwd_time_ms:.3f} ms"
    )
    return label, avg_fwd_time_ms, avg_bwd_time_ms


def main(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(args.seed)
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    num_tx = 16
    num_rx = 2
    frequency = 2.4e9
    scale_modifier = 1.0
    dtype = torch.float32

    results = []

    for N in args.num_gaussians:
        print("-" * 60)
        print(f"Benchmarking with N = {N} Gaussians")
        print("-" * 60)

        inputs_dict, grad_tensors = generate_inputs(
            N, num_tx, num_rx, frequency, scale_modifier, dtype, device
        )

        try:
            label_torch = f"PyTorch (N={N})"
            _, fwd_torch, bwd_torch = benchmark_function(
                torch_rasterize,
                inputs_dict,
                grad_tensors,
                args.runs,
                args.warmup,
                label_torch,
                device,
            )
            results.append([label_torch, f"{fwd_torch:.3f}", f"{bwd_torch:.3f}"])
        except Exception as e:
            print(f"Error benchmarking PyTorch version: {e}")
            results.append([label_torch, "Error", "Error"])

        label_cuda = f"CUDA (N={N})"
        if CUDA_AVAILABLE and device.type == "cuda":
            try:
                _, fwd_cuda, bwd_cuda = benchmark_function(
                    cuda_rasterize,
                    inputs_dict,
                    grad_tensors,
                    args.runs,
                    args.warmup,
                    label_cuda,
                    device,
                )
                results.append([label_cuda, f"{fwd_cuda:.3f}", f"{bwd_cuda:.3f}"])
            except Exception as e:
                print(f"Error benchmarking CUDA version: {e}")
                results.append([label_cuda, "Error", "Error"])
        else:
            print("Skipping CUDA benchmark.")
            results.append([label_cuda, "Skipped", "Skipped"])

    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    headers = ["Implementation", "Avg Forward (ms)", "Avg Backward (ms)"]
    print(tabulate(results, headers=headers, tablefmt="grid"))
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Rasterize Functions")
    parser.add_argument(
        "-n",
        "--num_gaussians",
        type=int,
        nargs="+",
        default=[100, 1000, 10000, 32000],
        help="List of numbers of Gaussians to benchmark.",
    )
    parser.add_argument(
        "-r", "--runs", type=int, default=10, help="Number of timed runs for averaging."
    )
    parser.add_argument(
        "-w", "--warmup", type=int, default=2, help="Number of warmup runs."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    main(args)

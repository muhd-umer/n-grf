# bench_fwd.py

import argparse
import os
import sys
import time
import unittest
import warnings

import torch
import torch.nn.functional as F
from tabulate import tabulate

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models.ngrf_model import nGRF
from render._torch_impl import render_channel as torch_render_channel

try:
    from render._wrapper import CUDA_AVAILABLE
    from render._wrapper import render_channel as cuda_render_channel
except ImportError:
    CUDA_AVAILABLE = False
    warnings.warn(
        "CUDA wrapper (render._C) not found or failed to import. CUDA benchmarks will be skipped.",
        ImportWarning,
    )

    def cuda_render_channel(*_args, **_kwargs):
        raise unittest.SkipTest(
            "CUDA render_channel not available due to import error."
        )

except Exception as e:
    CUDA_AVAILABLE = False
    warnings.warn(
        f"An unexpected error occurred while importing CUDA wrapper: {e}. CUDA benchmarks will be skipped.",
        ImportWarning,
    )

    def cuda_render_channel(*_args, **_kwargs):
        raise unittest.SkipTest(f"CUDA render_channel not available due to: {e}")


def generate_model_and_inputs(N_gauss, Nt, Nr, latent_dim, eps, dtype, device):
    """Generates a model instance and random inputs for render_channel with B=1."""
    B = 1

    model = nGRF(
        num_tx_ant=Nt,
        num_rx_ant=Nr,
        latent_dim=latent_dim,
        attribute_hidden_dim=64,
        attribute_num_layers=3,
        attribute_pos_enc_freqs=10,
        decoder_hidden_dim=64,
        decoder_num_layers=3,
        initial_gaussians=N_gauss,
        device=device,
    )

    xyz_raw = torch.randn(N_gauss, 3, device=device, dtype=dtype) * 5.0
    rot_raw = torch.randn(N_gauss, 4, device=device, dtype=dtype)
    rot_raw = F.normalize(rot_raw, p=2, dim=1)
    scl_log_raw = torch.randn(N_gauss, 3, device=device, dtype=dtype) * 0.5 - 2.0

    model._xyz = torch.nn.Parameter(xyz_raw, requires_grad=False)
    model._rotation = torch.nn.Parameter(rot_raw, requires_grad=False)
    model._scaling = torch.nn.Parameter(scl_log_raw, requires_grad=False)

    model.attribute_network = model.attribute_network.to(device, dtype)
    model.contribution_decoder = model.contribution_decoder.to(device, dtype)

    model.eval()

    rx_positions = torch.randn(B, 3, device=device, dtype=dtype) * 10.0
    tx_position = torch.randn(3, device=device, dtype=dtype) * 2.0

    render_args = {
        "rx_positions": rx_positions,
        "model": model,
        "tx_position": tx_position,
        "nt": Nt,
        "nr": Nr,
        "eps": eps,
    }
    return render_args


def benchmark_forward_pass(
    func_to_benchmark, render_args, n_runs, n_warmup, label, device
):
    """Benchmarks the forward pass of a given render_channel function."""
    print(f"Benchmarking {label} on {device} (Forward Pass Only, B=1)...")

    for _ in range(n_warmup):
        with torch.no_grad():
            _ = func_to_benchmark(**render_args)

    if device.type == "cuda":
        torch.cuda.synchronize()

    fwd_times = []
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(n_runs):
        if device.type == "cuda":
            start_event.record()
        else:
            fwd_start_time = time.perf_counter()

        with torch.no_grad():
            _ = func_to_benchmark(**render_args)

        if device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            fwd_times.append(start_event.elapsed_time(end_event))
        else:
            fwd_times.append((time.perf_counter() - fwd_start_time) * 1000)

    avg_fwd_time_ms = sum(fwd_times) / n_runs
    print(f"  {label}: Avg Fwd Time: {avg_fwd_time_ms:.3f} ms")
    return label, avg_fwd_time_ms


def main(args):
    torch.manual_seed(args.seed)
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        if args.device == "cuda":
            print("CUDA specified but not available, falling back to CPU.")
        device = torch.device("cpu")
        print("Using CPU device.")

    B = 1
    Nt = args.nt
    Nr = args.nr
    latent_dim = args.latent_dim
    eps = 1e-7
    dtype = torch.float32 if args.precision == "float32" else torch.double

    results = []

    for N_gauss in args.num_gaussians:
        print("-" * 70)
        print(
            f"Benchmarking with N_gauss = {N_gauss}, Batch = {B}, Precision = {args.precision}"
        )
        print("-" * 70)

        render_args = generate_model_and_inputs(
            N_gauss, Nt, Nr, latent_dim, eps, dtype, device
        )

        label_torch = f"PyTorch (N={N_gauss}, P={args.precision})"
        try:
            _, fwd_torch = benchmark_forward_pass(
                torch_render_channel,
                render_args,
                args.runs,
                args.warmup,
                label_torch,
                device,
            )
            results.append([label_torch, f"{fwd_torch:.3f}"])
        except Exception as e:
            print(f"Error benchmarking PyTorch version for N={N_gauss}: {e}")
            results.append([label_torch, "Error"])

        label_cuda = f"CUDA (N={N_gauss}, P={args.precision})"
        if not args.disable_cuda and CUDA_AVAILABLE and device.type == "cuda":
            try:
                _, fwd_cuda = benchmark_forward_pass(
                    cuda_render_channel,
                    render_args,
                    args.runs,
                    args.warmup,
                    label_cuda,
                    device,
                )
                results.append([label_cuda, f"{fwd_cuda:.3f}"])
            except unittest.SkipTest:
                print(
                    f"Skipping CUDA benchmark for N={N_gauss} as it's not available internally."
                )
                results.append([label_cuda, "Skipped (Not Available)"])
            except RuntimeError as e:
                print(
                    f"Skipping CUDA benchmark for N={N_gauss} due to RuntimeError: {e}"
                )
                results.append([label_cuda, "Skipped (RuntimeError)"])
            except Exception as e:
                print(f"Error benchmarking CUDA version for N={N_gauss}: {e}")
                results.append([label_cuda, "Error"])
        else:
            skip_reason = ""
            if args.disable_cuda:
                skip_reason = "User disabled CUDA"
            elif not CUDA_AVAILABLE:
                skip_reason = "CUDA wrapper not available"
            elif device.type != "cuda":
                skip_reason = "Not on CUDA device"
            else:
                skip_reason = "Unknown reason"
            print(f"Skipping CUDA benchmark for N={N_gauss} ({skip_reason}).")
            results.append([label_cuda, f"Skipped ({skip_reason})"])

    print("\n" + "=" * 70)
    print("Benchmark Results Summary (Forward Pass, Batch Size=1)")
    print("=" * 70)
    headers = ["Implementation Details", "Avg Forward Time (ms)"]
    print(tabulate(results, headers=headers, tablefmt="grid"))
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark forward pass of render_channel"
    )
    parser.add_argument(
        "-n",
        "--num_gaussians",
        type=int,
        nargs="+",
        default=[500, 1000, 3500],
        help="List of numbers of Gaussians to benchmark.",
    )
    parser.add_argument(
        "-r",
        "--runs",
        type=int,
        default=200,
        help="Number of timed runs for averaging.",
    )
    parser.add_argument(
        "-w", "--warmup", type=int, default=10, help="Number of warmup runs."
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Data type precision for benchmarks.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run benchmarks on.",
    )
    parser.add_argument(
        "--disable_cuda",
        action="store_true",
        help="Disable custom CUDA kernels and only run PyTorch implementation, even if CUDA is available.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--nt", type=int, default=16, help="Number of Tx antennas.")
    parser.add_argument("--nr", type=int, default=4, help="Number of Rx antennas.")
    parser.add_argument(
        "--latent_dim", type=int, default=32, help="Latent dimension of the model."
    )

    args = parser.parse_args()
    main(args)

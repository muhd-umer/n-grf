# render/tests/benchmark.py

import argparse
import os
import sys
import time
import unittest
import warnings

import torch
import torch.nn.functional as F
from tabulate import tabulate

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)
RENDER_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RENDER_MODULE_ROOT)

try:
    from render._wrapper import CUDA_AVAILABLE
    from render._wrapper import render_channel as cuda_render_channel

    if not CUDA_AVAILABLE:
        warnings.warn(
            "CUDA_AVAILABLE is False in _wrapper. CUDA benchmarks may be skipped.",
            ImportWarning,
        )
except ImportError:
    CUDA_AVAILABLE = False
    warnings.warn(
        "CUDA wrapper (render_ngrf._C) not found. CUDA benchmarks will be skipped.",
        ImportWarning,
    )

    def cuda_render_channel(*args, **kwargs):
        raise unittest.SkipTest("CUDA render_channel not available for benchmark")


from models.ngrf_model import nGRF
from render._torch_impl import render_channel as torch_render_channel


def generate_model_and_inputs(N_gauss, B, Nt, Nr, latent_dim, eps, dtype, device):
    """Generates a model instance and random inputs for render_channel."""
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

    xyz_raw = torch.randn(N_gauss, 3) * 5.0
    rot_raw = torch.randn(N_gauss, 4)
    rot_raw = F.normalize(rot_raw, p=2, dim=1)
    scl_log_raw = torch.randn(N_gauss, 3) * 0.5 - 2.0

    model._xyz = torch.nn.Parameter(xyz_raw.to(device, dtype).requires_grad_(True))
    model._rotation = torch.nn.Parameter(rot_raw.to(device, dtype).requires_grad_(True))
    model._scaling = torch.nn.Parameter(
        scl_log_raw.to(device, dtype).requires_grad_(True)
    )

    model.attribute_network = model.attribute_network.to(device, dtype)
    model.contribution_decoder = model.contribution_decoder.to(device, dtype)
    for param in model.attribute_network.parameters():
        param.requires_grad_(True)
    for param in model.contribution_decoder.parameters():
        param.requires_grad_(True)

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

    grad_params = [model._xyz, model._rotation, model._scaling]
    grad_params.extend(list(model.attribute_network.parameters()))
    grad_params.extend(list(model.contribution_decoder.parameters()))
    grad_params = [p for p in grad_params if p.requires_grad]

    return render_args, grad_params


def benchmark_render_function(
    func, render_args, grad_params, n_runs, n_warmup, label, device
):
    """Benchmarks forward and backward pass of a given render_channel function."""
    print(f"Benchmarking {label} on {device}...")
    model = render_args["model"]

    print(f"  Warmup ({n_warmup} runs)...")
    for _ in range(n_warmup):
        for p in grad_params:
            if p.grad is not None:
                p.grad.zero_()
        output = func(**render_args)

        loss = (output.real.sum() + output.imag.sum()) * 1e-3
        loss.backward()

    print(f"  Forward Pass ({n_runs} runs)...")
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
            _ = func(**render_args)

        if device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            fwd_times.append(start_event.elapsed_time(end_event))
        else:
            fwd_times.append((time.perf_counter() - fwd_start_time) * 1000)

    avg_fwd_time_ms = sum(fwd_times) / n_runs

    print(f"  Backward Pass ({n_runs} runs)...")

    for p in grad_params:
        if p.grad is not None:
            p.grad.zero_()
    output = func(**render_args)
    loss = (output.real.sum() + output.imag.sum()) * 1e-3

    bwd_times = []
    for _ in range(n_runs):

        for p in grad_params:
            if p.grad is not None:
                p.grad.zero_()

        if device.type == "cuda":
            start_event.record()
        else:
            bwd_start_time = time.perf_counter()

        loss.backward(retain_graph=True)

        if device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            bwd_times.append(start_event.elapsed_time(end_event))
        else:
            bwd_times.append((time.perf_counter() - bwd_start_time) * 1000)

    avg_bwd_time_ms = sum(bwd_times) / n_runs

    print(
        f"  {label}: Avg Fwd: {avg_fwd_time_ms:.3f} ms, Avg Bwd: {avg_bwd_time_ms:.3f} ms"
    )
    return label, avg_fwd_time_ms, avg_bwd_time_ms


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

    B = args.batch_size
    Nt = 16
    Nr = 4
    latent_dim = 32
    eps = 1e-7
    dtype = torch.float32 if args.precision == "float32" else torch.double

    results = []

    for N_gauss in args.num_gaussians:
        print("-" * 70)
        print(
            f"Benchmarking with N_gauss = {N_gauss}, Batch = {B}, Precision = {args.precision}"
        )
        print("-" * 70)

        render_args, grad_params = generate_model_and_inputs(
            N_gauss, B, Nt, Nr, latent_dim, eps, dtype, device
        )

        label_torch = f"PyTorch (N={N_gauss}, B={B}, {args.precision})"
        try:
            _, fwd_torch, bwd_torch = benchmark_render_function(
                torch_render_channel,
                render_args,
                grad_params,
                args.runs,
                args.warmup,
                label_torch,
                device,
            )
            results.append([label_torch, f"{fwd_torch:.3f}", f"{bwd_torch:.3f}"])
        except Exception as e:
            print(f"Error benchmarking PyTorch version for N={N_gauss}: {e}")
            results.append([label_torch, "Error", "Error"])

        label_cuda = f"CUDA (N={N_gauss}, B={B}, {args.precision})"
        if CUDA_AVAILABLE and device.type == "cuda":
            try:
                _, fwd_cuda, bwd_cuda = benchmark_render_function(
                    cuda_render_channel,
                    render_args,
                    grad_params,
                    args.runs,
                    args.warmup,
                    label_cuda,
                    device,
                )
                results.append([label_cuda, f"{fwd_cuda:.3f}", f"{bwd_cuda:.3f}"])
            except unittest.SkipTest:
                print(f"Skipping CUDA benchmark for N={N_gauss} as it's not available.")
                results.append(
                    [label_cuda, "Skipped (Not Available)", "Skipped (Not Available)"]
                )
            except Exception as e:
                print(f"Error benchmarking CUDA version for N={N_gauss}: {e}")
                results.append([label_cuda, "Error", "Error"])
        else:
            print(
                f"Skipping CUDA benchmark for N={N_gauss} (CUDA not enabled or not on CUDA device)."
            )
            results.append([label_cuda, "Skipped", "Skipped"])

    print("\n" + "=" * 70)
    print("Benchmark Results Summary")
    print("=" * 70)
    headers = ["Implementation Details", "Avg Forward (ms)", "Avg Backward (ms)"]
    print(tabulate(results, headers=headers, tablefmt="grid"))
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark render_channel Functions")
    parser.add_argument(
        "-n",
        "--num_gaussians",
        type=int,
        nargs="+",
        default=[1000, 10000, 50000],
        help="List of numbers of Gaussians to benchmark.",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=4, help="Batch size for rendering."
    )
    parser.add_argument(
        "-r", "--runs", type=int, default=20, help="Number of timed runs for averaging."
    )
    parser.add_argument(
        "-w", "--warmup", type=int, default=5, help="Number of warmup runs."
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    main(args)

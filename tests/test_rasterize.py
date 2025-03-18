# r2f_engine/tests/test_rasterize.py

# tests/test_pt_cuda.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time

import numpy as np
import torch

import r2f_engine
import r2f_engine._torch_impl as torch_impl


def generate_test_data(batch_size=10000, device="cuda"):
    """Generate random test data for benchmarking and testing"""
    points = torch.randn(batch_size, 3, device=device)
    cov3d = torch.randn(batch_size, 6, device=device)  # Compact form
    attenuation = torch.rand(batch_size, 1, device=device)
    phase_rotation = torch.rand(batch_size, 1, device=device) * 2 * np.pi
    opacity = torch.rand(batch_size, 1, device=device)
    receiver = torch.randn(3, device=device)
    transmitter = torch.randn(3, device=device)
    num_tx = 16
    num_rx = 2
    frequency = 2.4e9  # 2.4 GHz

    return {
        "points": points,
        "cov3d": cov3d,
        "attenuation": attenuation,
        "phase_rotation": phase_rotation,
        "opacity": opacity,
        "receiver": receiver,
        "transmitter": transmitter,
        "num_tx": num_tx,
        "num_rx": num_rx,
        "frequency": frequency,
    }


def test_correctness():
    """Test that PyTorch and CUDA implementations produce the same results"""
    print("Testing correctness...")

    data = generate_test_data(batch_size=1000)

    # torch implementation
    torch_result = torch_impl.rasterize(
        data["points"],
        data["cov3d"],
        data["attenuation"],
        data["phase_rotation"],
        data["opacity"],
        data["receiver"],
        data["transmitter"],
        data["num_tx"],
        data["num_rx"],
        data["frequency"],
    )

    # CUDA implementation
    cuda_result = r2f_engine.rasterize(
        data["points"],
        data["cov3d"],
        data["attenuation"],
        data["phase_rotation"],
        data["opacity"],
        data["receiver"],
        data["transmitter"],
        data["num_tx"],
        data["num_rx"],
        data["frequency"],
    )

    try:
        torch.testing.assert_close(torch_result, cuda_result, rtol=1e-3, atol=1e-3)
        print("✅ Results match!")
    except AssertionError as e:
        print("❌ Results don't match!")
        print(e)

        # print some statistics
        print(f"PyTorch result: mean={torch_result.mean()}, std={torch_result.std()}")
        print(f"CUDA result: mean={cuda_result.mean()}, std={cuda_result.std()}")
        print(f"Max absolute difference: {(torch_result - cuda_result).abs().max()}")
        assert (
            False
        ), "Rasterize results do not match between PyTorch and CUDA implementations"


def benchmark(batch_sizes=None, num_runs=10):
    """Benchmark PyTorch vs CUDA implementations"""
    print("Running benchmarks...")

    if batch_sizes is None:
        batch_sizes = [1000, 5000, 10000]

    print(f"{'Batch Size':<15} {'PyTorch (ms)':<15} {'CUDA (ms)':<15} {'Speedup':<10}")
    print("-" * 55)

    for batch_size in batch_sizes:
        data = generate_test_data(batch_size=batch_size)

        for _ in range(3):
            torch_impl.rasterize(
                data["points"],
                data["cov3d"],
                data["attenuation"],
                data["phase_rotation"],
                data["opacity"],
                data["receiver"],
                data["transmitter"],
                data["num_tx"],
                data["num_rx"],
                data["frequency"],
            )

            r2f_engine.rasterize(
                data["points"],
                data["cov3d"],
                data["attenuation"],
                data["phase_rotation"],
                data["opacity"],
                data["receiver"],
                data["transmitter"],
                data["num_tx"],
                data["num_rx"],
                data["frequency"],
            )

        # torch timing
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            torch_impl.rasterize(
                data["points"],
                data["cov3d"],
                data["attenuation"],
                data["phase_rotation"],
                data["opacity"],
                data["receiver"],
                data["transmitter"],
                data["num_tx"],
                data["num_rx"],
                data["frequency"],
            )
        torch.cuda.synchronize()
        torch_time = (time.time() - start) * 1000 / num_runs  # ms

        # CUDA timing
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            r2f_engine.rasterize(
                data["points"],
                data["cov3d"],
                data["attenuation"],
                data["phase_rotation"],
                data["opacity"],
                data["receiver"],
                data["transmitter"],
                data["num_tx"],
                data["num_rx"],
                data["frequency"],
            )
        torch.cuda.synchronize()
        cuda_time = (time.time() - start) * 1000 / num_runs  # ms

        speedup = torch_time / cuda_time if cuda_time > 0 else float("inf")

        print(
            f"{batch_size:<15} {torch_time:<15.2f} {cuda_time:<15.2f} {speedup:<10.2f}x"
        )


def main():
    """Main function to run tests and benchmarks"""
    parser = argparse.ArgumentParser(
        description="Test and benchmark rasterization implementations"
    )
    parser.add_argument("--test", action="store_true", help="Run correctness tests")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[1000, 5000, 10000],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--num_runs", type=int, default=10, help="Number of runs for each benchmark"
    )

    args = parser.parse_args()

    print(f"Using {'CUDA' if r2f_engine.CUSTOM_KERNEL else 'PyTorch'} implementation")

    if args.test or not args.benchmark:
        success = test_correctness()
        if not success:
            return

    if args.benchmark or not args.test:
        benchmark(batch_sizes=args.batch_sizes, num_runs=args.num_runs)


if __name__ == "__main__":
    main()

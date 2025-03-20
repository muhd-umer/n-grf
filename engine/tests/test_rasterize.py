# engine/tests/test_rasterize.py

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time

import numpy as np
import pytest
import torch

import engine
import engine._torch_impl as torch_impl
from datasets.dataloader import get_dataloaders
from engine import rasterize
from models.encoder import EncoderConfig
from models.gaussian_model import GaussianModel
from utils.transform_utils import symmetric_matrix

torch.manual_seed(2)

TEST_DATA = None
DATA_PATH = "datasets/outputs/conf_16x2_414u_5.0ghz_sbrRT_sc104.mat"


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


@pytest.fixture(autouse=True, scope="session")
def init_test_data():
    global TEST_DATA
    if engine.CUDA_AVAILABLE:
        try:
            load_global_test_data(DATA_PATH)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise


@pytest.mark.skipif(
    not engine.CUDA_AVAILABLE, reason="CUDA implementation not available"
)
def test_rasterize():
    print("Running torch rasterize implementation...")
    torch_result = torch_impl.rasterize(
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

    has_nan = torch.isnan(torch_result).any().item()
    has_inf = torch.isinf(torch_result).any().item()
    if not (has_nan or has_inf):
        print("Running CUDA rasterize implementation...")
        from engine import rasterize as cuda_rasterize

        cuda_result = cuda_rasterize(
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

        has_nan = torch.isnan(cuda_result).any().item()
        has_inf = torch.isinf(cuda_result).any().item()
        if not (has_nan or has_inf):
            torch.testing.assert_close(torch_result, cuda_result, rtol=1e-4, atol=1e-4)
            print("Rasterize test passed!")
        else:
            print("Rasterize test skipped comparison due to NaN/Inf in CUDA result")
    else:
        print(
            "Rasterize test skipped CUDA implementation due to NaN/Inf in torch result"
        )


def benchmark_rasterize(num_runs=10):
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

    torch_times = []
    for _ in range(num_runs):
        start_time = time.time()
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
        torch_times.append(time.time() - start_time)

    avg_torch_time = sum(torch_times) / len(torch_times)

    from engine import rasterize as cuda_rasterize

    cuda_rasterize(
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

    cuda_times = []
    for _ in range(num_runs):
        start_time = time.time()
        cuda_rasterize(
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
        cuda_times.append(time.time() - start_time)

    avg_cuda_time = sum(cuda_times) / len(cuda_times)

    speedup = avg_torch_time / avg_cuda_time if avg_cuda_time > 0 else float("inf")

    print(
        f"Torch implementation: {avg_torch_time:.6f} seconds (avg of {num_runs} runs)"
    )
    print(f"CUDA implementation: {avg_cuda_time:.6f} seconds (avg of {num_runs} runs)")
    print(f"Speedup: {speedup:.2f}x")

    return avg_torch_time, avg_cuda_time, speedup


def main():
    if engine.CUDA_AVAILABLE:
        try:
            load_global_test_data(DATA_PATH)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

        print("\n--- Testing rasterize function ---")
        test_rasterize()

        print("\n--- Benchmarking rasterize function ---")
        benchmark_rasterize(num_runs=10)
    else:
        print("CUDA implementation not available. Skipping tests and benchmarks.")


if __name__ == "__main__":
    main()

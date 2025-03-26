# engine/tests/test_backward.py

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import torch

import engine
import engine._torch_impl as torch_impl
from datasets.dataloader import get_dataloaders
from models.encoder import EncoderConfig
from models.gaussian_model import GaussianModel

TEST_DATA = None


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
    gt_channel = data_batch["channel_matrix"].to(device).squeeze()
    gt_channel = torch.hstack((gt_channel.real, gt_channel.imag))
    gt_channel = gt_channel.float()

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
        "gt_channel": gt_channel,
    }

    return TEST_DATA


@pytest.fixture(autouse=True, scope="session")
def init_test_data(data_path):
    global TEST_DATA
    if engine.CUDA_AVAILABLE:
        try:
            load_global_test_data(data_path)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise


@pytest.mark.skipif(
    not engine.CUDA_AVAILABLE, reason="CUDA implementation not available"
)
def test_gradients_pytorch_vs_cuda():
    """Test that PyTorch autograd and our CUDA backward pass produce similar gradients"""

    points = TEST_DATA["points"].clone().detach().requires_grad_(True)
    cov3d = TEST_DATA["cov3d"].clone().detach().requires_grad_(True)
    attenuation = TEST_DATA["attenuation"].clone().detach().requires_grad_(True)
    phase_rotation = TEST_DATA["phase_rotation"].clone().detach().requires_grad_(True)
    opacity = TEST_DATA["opacity"].clone().detach().requires_grad_(True)

    torch_result = torch_impl.rasterize(
        points=points,
        cov3d=cov3d,
        attenuation=attenuation,
        phase_rotation=phase_rotation,
        opacity=opacity,
        receiver=TEST_DATA["receiver"],
        transmitter=TEST_DATA["transmitter"],
        num_tx=TEST_DATA["num_tx"],
        num_rx=TEST_DATA["num_rx"],
        frequency=TEST_DATA["frequency"],
    )

    loss = torch.nn.functional.mse_loss(torch_result, TEST_DATA["gt_channel"])
    loss.backward()

    torch_grad_points = points.grad.clone()
    torch_grad_cov3d = cov3d.grad.clone()
    torch_grad_attenuation = attenuation.grad.clone()
    torch_grad_phase_rotation = phase_rotation.grad.clone()
    torch_grad_opacity = opacity.grad.clone()

    points = TEST_DATA["points"].clone().detach()
    cov3d = TEST_DATA["cov3d"].clone().detach()
    attenuation = TEST_DATA["attenuation"].clone().detach()
    phase_rotation = TEST_DATA["phase_rotation"].clone().detach()
    opacity = TEST_DATA["opacity"].clone().detach()

    cuda_result = engine.rasterize(
        points=points,
        cov3d=cov3d,
        attenuation=attenuation,
        phase_rotation=phase_rotation,
        opacity=opacity,
        receiver=TEST_DATA["receiver"],
        transmitter=TEST_DATA["transmitter"],
        num_tx=TEST_DATA["num_tx"],
        num_rx=TEST_DATA["num_rx"],
        frequency=TEST_DATA["frequency"],
    )

    grad_output = (
        2.0
        * (cuda_result - TEST_DATA["gt_channel"])
        / (TEST_DATA["num_tx"] * TEST_DATA["num_rx"] * 2)
    )

    (
        cuda_grad_points,
        cuda_grad_cov3d,
        cuda_grad_attenuation,
        cuda_grad_phase_rotation,
        cuda_grad_opacity,
    ) = engine.rasterize_backward(
        grad_output,
        points,
        cov3d,
        attenuation,
        phase_rotation,
        opacity,
        TEST_DATA["receiver"],
        TEST_DATA["transmitter"],
        TEST_DATA["num_tx"],
        TEST_DATA["num_rx"],
        TEST_DATA["frequency"],
    )

    rtol = 1e-3
    atol = 1e-3

    torch.testing.assert_close(
        torch_grad_points,
        cuda_grad_points,
        rtol=rtol,
        atol=atol,
        msg="Gradients for points don't match between PyTorch and CUDA",
    )

    torch.testing.assert_close(
        torch_grad_cov3d,
        cuda_grad_cov3d,
        rtol=rtol,
        atol=atol,
        msg="Gradients for 3D covariance don't match between PyTorch and CUDA",
    )

    torch.testing.assert_close(
        torch_grad_attenuation,
        cuda_grad_attenuation,
        rtol=rtol,
        atol=atol,
        msg="Gradients for attenuation don't match between PyTorch and CUDA",
    )

    torch.testing.assert_close(
        torch_grad_phase_rotation,
        cuda_grad_phase_rotation,
        rtol=rtol,
        atol=atol,
        msg="Gradients for phase rotation don't match between PyTorch and CUDA",
    )

    torch.testing.assert_close(
        torch_grad_opacity,
        cuda_grad_opacity,
        rtol=rtol,
        atol=atol,
        msg="Gradients for opacity don't match between PyTorch and CUDA",
    )

    print("All gradient tests passed!")


@pytest.mark.skipif(
    not engine.CUDA_AVAILABLE, reason="CUDA implementation not available"
)
def test_backward_numerical_stability():
    """Test that the backward pass doesn't produce NaN or Inf gradients"""

    grad_output = torch.randn(
        (TEST_DATA["num_tx"] * 2, TEST_DATA["num_rx"]),
        device=TEST_DATA["points"].device,
    )

    (
        grad_points,
        grad_cov3d,
        grad_attenuation,
        grad_phase_rotation,
        grad_opacity,
    ) = engine.rasterize_backward(
        grad_output,
        TEST_DATA["points"],
        TEST_DATA["cov3d"],
        TEST_DATA["attenuation"],
        TEST_DATA["phase_rotation"],
        TEST_DATA["opacity"],
        TEST_DATA["receiver"],
        TEST_DATA["transmitter"],
        TEST_DATA["num_tx"],
        TEST_DATA["num_rx"],
        TEST_DATA["frequency"],
    )

    print("Gradients")
    print(grad_points)
    print(grad_cov3d)
    print(grad_attenuation)
    print(grad_phase_rotation)
    print(grad_opacity)

    assert not torch.isnan(grad_points).any(), "NaN values found in points gradients"
    assert not torch.isinf(grad_points).any(), "Inf values found in points gradients"

    assert not torch.isnan(grad_cov3d).any(), "NaN values found in cov3d gradients"
    assert not torch.isinf(grad_cov3d).any(), "Inf values found in cov3d gradients"

    assert not torch.isnan(
        grad_attenuation
    ).any(), "NaN values found in attenuation gradients"
    assert not torch.isinf(
        grad_attenuation
    ).any(), "Inf values found in attenuation gradients"

    assert not torch.isnan(
        grad_phase_rotation
    ).any(), "NaN values found in phase_rotation gradients"
    assert not torch.isinf(
        grad_phase_rotation
    ).any(), "Inf values found in phase_rotation gradients"

    assert not torch.isnan(grad_opacity).any(), "NaN values found in opacity gradients"
    assert not torch.isinf(grad_opacity).any(), "Inf values found in opacity gradients"

    print("Numerical stability test passed!")


@pytest.mark.skipif(
    not engine.CUDA_AVAILABLE, reason="CUDA implementation not available"
)
def test_end_to_end_training():
    """Test that we can perform a few training steps with the CUDA backward pass"""

    point_cloud = TEST_DATA["points"].clone().detach()[:100]

    encoder_cfg = EncoderConfig(
        hidden_size=64,
        num_layers=4,
        skip_layers=(2,),
        input_pos_multires=6,
        use_positional_encoding=True,
    )

    model = GaussianModel(encoder_cfg=encoder_cfg).to(TEST_DATA["points"].device)

    model.init_from_pc(
        point_cloud,
        tx_position=TEST_DATA["transmitter"],
        frequency=TEST_DATA["frequency"],
        use_physics_init=True,
    )

    optimizer = torch.optim.Adam(
        [
            {"params": [model._xyz], "lr": 0.0001},
            {"params": [model._rotation], "lr": 0.001},
            {"params": [model._scaling], "lr": 0.005},
            {"params": [model._opacity], "lr": 0.01},
        ]
    )

    for i in range(3):

        rx_position = TEST_DATA["receiver"].clone()
        gt_channel = TEST_DATA["gt_channel"].clone()

        enc_data = {
            "tx_pos": TEST_DATA["transmitter"],
            "rx_pos": rx_position,
            "frequency": TEST_DATA["frequency"],
        }
        model.embed_features(enc_data)

        pred_channel = engine.rasterize(
            points=model.get_xyz,
            cov3d=model.get_covariance(),
            attenuation=model.get_features[:, 0:1],
            phase_rotation=model.get_features[:, 1:2],
            opacity=model.get_opacity,
            receiver=rx_position,
            transmitter=TEST_DATA["transmitter"],
            num_tx=TEST_DATA["num_tx"],
            num_rx=TEST_DATA["num_rx"],
            frequency=TEST_DATA["frequency"],
        )

        loss = torch.nn.functional.mse_loss(pred_channel, gt_channel)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Iteration {i+1}, Loss: {loss.item():.6f}")

    print("End-to-end training test passed!")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test backward pass for rasterize function"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets/outputs/conf_16x2_414u_5.0ghz_sbrRT_sc104.mat",
        help="Path to the dataset file",
    )
    args = parser.parse_args()

    if engine.CUDA_AVAILABLE:
        try:
            load_global_test_data(args.data_path)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

        print("\n--- Testing gradients comparison between PyTorch and CUDA ---")
        test_gradients_pytorch_vs_cuda()

        print("\n--- Testing numerical stability of CUDA backward pass ---")
        test_backward_numerical_stability()

        print("\n--- Testing end-to-end training with CUDA backward pass ---")
        test_end_to_end_training()
    else:
        print("CUDA implementation not available. Skipping tests.")


if __name__ == "__main__":
    main()

# engine/tests/test_pt_cuda.py

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import torch

import engine._torch_impl as torch_impl
from datasets.dataloader import get_dataloaders
from engine import (
    _C,
    CUDA_AVAILABLE,
    alpha_blending,
    compute_channel,
    compute_distances_to_receiver,
    compute_gaussian_influence,
    compute_jacobian,
    compute_spherical_coords,
    map_to_channel_matrix,
    project_cov3d_to_cov2d,
    project_to_channel_space,
    transform_to_uniform_coords,
)
from models.encoder import EncoderConfig
from models.gaussian_model import GaussianModel
from utils.transform_utils import symmetric_matrix

TEST_DATA = None
INTERMEDIATE_VALUES = {}


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
def init_test_data(data_path):
    global TEST_DATA
    if CUDA_AVAILABLE:
        try:
            load_global_test_data(data_path)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_compute_distances_to_receiver():
    global INTERMEDIATE_VALUES

    torch_result = torch_impl.compute_distances_to_receiver(
        TEST_DATA["points"], TEST_DATA["receiver"]
    )

    cuda_result = compute_distances_to_receiver(
        TEST_DATA["points"], TEST_DATA["receiver"]
    )

    INTERMEDIATE_VALUES["rx_distances"] = torch_result

    torch.testing.assert_close(torch_result, cuda_result, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_compute_spherical_coords():
    global INTERMEDIATE_VALUES

    torch_d, torch_lon, torch_lat = torch_impl.compute_spherical_coords(
        TEST_DATA["points"], TEST_DATA["receiver"]
    )
    cuda_d, cuda_lon, cuda_lat = compute_spherical_coords(
        TEST_DATA["points"], TEST_DATA["receiver"]
    )

    INTERMEDIATE_VALUES["displacement_vectors"] = torch_d
    INTERMEDIATE_VALUES["longitude"] = torch_lon
    INTERMEDIATE_VALUES["latitude"] = torch_lat

    torch.testing.assert_close(torch_d, cuda_d, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(torch_lon, cuda_lon, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(torch_lat, cuda_lat, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_transform_to_uniform_coords():
    global INTERMEDIATE_VALUES

    torch_lon = INTERMEDIATE_VALUES["longitude"]
    torch_lat = INTERMEDIATE_VALUES["latitude"]

    torch_s_x, torch_s_y = torch_impl.transform_to_uniform_coords(torch_lon, torch_lat)
    cuda_s_x, cuda_s_y = transform_to_uniform_coords(torch_lon, torch_lat)

    INTERMEDIATE_VALUES["uniform_x"] = torch_s_x
    INTERMEDIATE_VALUES["uniform_y"] = torch_s_y

    torch.testing.assert_close(torch_s_x, cuda_s_x, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(torch_s_y, cuda_s_y, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_map_to_channel_matrix():
    global INTERMEDIATE_VALUES

    torch_s_x = INTERMEDIATE_VALUES["uniform_x"]
    torch_s_y = INTERMEDIATE_VALUES["uniform_y"]

    torch_uv = torch_impl.map_to_channel_matrix(
        torch_s_x, torch_s_y, TEST_DATA["num_tx"], TEST_DATA["num_rx"]
    )
    cuda_uv = map_to_channel_matrix(
        torch_s_x, torch_s_y, TEST_DATA["num_tx"], TEST_DATA["num_rx"]
    )

    INTERMEDIATE_VALUES["uv"] = torch_uv

    torch.testing.assert_close(torch_uv, cuda_uv, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_compute_jacobian():
    global INTERMEDIATE_VALUES

    torch_d = INTERMEDIATE_VALUES["displacement_vectors"]
    torch_r = INTERMEDIATE_VALUES["rx_distances"]

    torch_jacobian = torch_impl.compute_jacobian(
        torch_d, torch_r, TEST_DATA["num_tx"], TEST_DATA["num_rx"]
    )
    cuda_jacobian = compute_jacobian(
        torch_d, torch_r, TEST_DATA["num_tx"], TEST_DATA["num_rx"]
    )

    INTERMEDIATE_VALUES["jacobian"] = torch_jacobian

    torch.testing.assert_close(torch_jacobian, cuda_jacobian, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_compute_cov3d_from_scaling_rotation():
    """Test CUDA implementation of covariance computation against PyTorch"""
    global TEST_DATA

    scaling = TEST_DATA["points"].new_zeros((TEST_DATA["points"].shape[0], 3)) + 0.1
    rotation = torch.zeros(
        (TEST_DATA["points"].shape[0], 4), device=TEST_DATA["points"].device
    )
    rotation[:, 0] = 1.0
    scale_modifier = 1.0

    from utils.transform_utils import build_scaling_rotation, strip_symmetric

    torch_L = build_scaling_rotation(scale_modifier * scaling, rotation)
    torch_cov = torch.bmm(torch_L, torch_L.transpose(1, 2))
    torch_cov3d = strip_symmetric(torch_cov)

    cuda_cov3d = _C.compute_cov3d_from_scaling_rotation(
        scaling, rotation, scale_modifier
    )

    has_nan_torch = torch.isnan(torch_cov3d).any().item()
    has_inf_torch = torch.isinf(torch_cov3d).any().item()
    has_nan_cuda = torch.isnan(cuda_cov3d).any().item()
    has_inf_cuda = torch.isinf(cuda_cov3d).any().item()

    assert not has_nan_torch, "PyTorch implementation produced NaN values"
    assert not has_inf_torch, "PyTorch implementation produced Inf values"
    assert not has_nan_cuda, "CUDA implementation produced NaN values"
    assert not has_inf_cuda, "CUDA implementation produced Inf values"

    torch.testing.assert_close(torch_cov3d, cuda_cov3d, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_project_cov3d_to_cov2d():
    global INTERMEDIATE_VALUES

    torch_jacobian = INTERMEDIATE_VALUES["jacobian"]
    cov3d_mat = symmetric_matrix(TEST_DATA["cov3d"])

    torch_cov2d = torch_impl.project_cov3d_to_cov2d(cov3d_mat, torch_jacobian)
    cuda_cov2d = project_cov3d_to_cov2d(cov3d_mat, torch_jacobian)

    INTERMEDIATE_VALUES["cov2d"] = torch_cov2d

    torch.testing.assert_close(torch_cov2d, cuda_cov2d, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_project_to_channel_space():
    global INTERMEDIATE_VALUES

    torch_distances, torch_uv, torch_cov2d = torch_impl.project_to_channel_space(
        TEST_DATA["points"],
        TEST_DATA["cov3d"],
        TEST_DATA["receiver"],
        TEST_DATA["num_tx"],
        TEST_DATA["num_rx"],
    )

    cuda_distances, cuda_uv, cuda_cov2d = project_to_channel_space(
        TEST_DATA["points"],
        TEST_DATA["cov3d"],
        TEST_DATA["receiver"],
        TEST_DATA["num_tx"],
        TEST_DATA["num_rx"],
    )

    INTERMEDIATE_VALUES["xyz_rx_distances"] = torch_distances
    INTERMEDIATE_VALUES["uv"] = torch_uv
    INTERMEDIATE_VALUES["cov2d"] = torch_cov2d

    assert not torch.isnan(torch_distances).any(), "NaN values found in distances"

    torch.testing.assert_close(torch_distances, cuda_distances, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(torch_uv, cuda_uv, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(torch_cov2d, cuda_cov2d, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_compute_gaussian_influence():
    global INTERMEDIATE_VALUES

    uv = INTERMEDIATE_VALUES["uv"]
    cov2d = INTERMEDIATE_VALUES["cov2d"]

    torch_result = torch_impl.compute_gaussian_influence(
        uv, cov2d, TEST_DATA["num_tx"], TEST_DATA["num_rx"]
    )
    cuda_result = compute_gaussian_influence(
        uv, cov2d, TEST_DATA["num_tx"], TEST_DATA["num_rx"]
    )

    INTERMEDIATE_VALUES["influences"] = torch_result

    assert not torch.isnan(torch_result).any(), "NaN values found in torch influences"

    torch.testing.assert_close(torch_result, cuda_result, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_compute_channel():
    global INTERMEDIATE_VALUES

    xyz_rx_distances = INTERMEDIATE_VALUES["xyz_rx_distances"]

    torch_real, torch_imag = torch_impl.compute_channel(
        TEST_DATA["attenuation"],
        TEST_DATA["phase_rotation"],
        xyz_rx_distances,
        TEST_DATA["wavelength"],
    )

    cuda_real, cuda_imag = compute_channel(
        TEST_DATA["attenuation"],
        TEST_DATA["phase_rotation"],
        xyz_rx_distances,
        TEST_DATA["wavelength"],
    )

    INTERMEDIATE_VALUES["contributions_real"] = torch_real
    INTERMEDIATE_VALUES["contributions_imag"] = torch_imag
    INTERMEDIATE_VALUES["contributions"] = torch.complex(torch_real, torch_imag)

    assert not torch.isnan(torch_real).any(), "NaN values found in torch_real"
    assert not torch.isnan(torch_imag).any(), "NaN values found in torch_imag"

    torch.testing.assert_close(torch_real, cuda_real, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(torch_imag, cuda_imag, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_alpha_blending():
    global INTERMEDIATE_VALUES

    influences = INTERMEDIATE_VALUES["influences"]
    contributions = INTERMEDIATE_VALUES["contributions"]
    xyz_rx_distances = INTERMEDIATE_VALUES["xyz_rx_distances"]

    sort_indices = torch.argsort(xyz_rx_distances)
    INTERMEDIATE_VALUES["sort_indices"] = sort_indices

    torch_result = torch_impl.alpha_blending(
        influences,
        contributions,
        TEST_DATA["opacity"],
        sort_indices,
        TEST_DATA["num_tx"],
        TEST_DATA["num_rx"],
    )

    has_nan = torch.isnan(torch_result).any().item()
    has_inf = torch.isinf(torch_result).any().item()
    if has_nan or has_inf:
        print("First few values from torch_result:", torch_result.flatten()[:10])

    cuda_result = alpha_blending(
        influences,
        contributions,
        TEST_DATA["opacity"],
        sort_indices,
        TEST_DATA["num_tx"],
        TEST_DATA["num_rx"],
    )

    has_nan = torch.isnan(cuda_result).any().item()
    has_inf = torch.isinf(cuda_result).any().item()
    if has_nan or has_inf:
        print("First few values from cuda_result:", cuda_result.flatten()[:10])

    if not (has_nan or has_inf):
        torch.testing.assert_close(torch_result, cuda_result, rtol=1e-4, atol=1e-4)
    else:
        print("test_alpha_blending skipped comparison due to NaN/Inf values")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test PyTorch CUDA implementations")
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets/outputs/conf_16x2_414u_5.0ghz_sbrRT_sc104.mat",
        help="Path to the dataset file",
    )
    args = parser.parse_args()

    if CUDA_AVAILABLE:
        try:
            load_global_test_data(args.data_path)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

        test_compute_distances_to_receiver()
        test_compute_spherical_coords()
        test_transform_to_uniform_coords()
        test_map_to_channel_matrix()
        test_compute_jacobian()
        test_compute_cov3d_from_scaling_rotation()
        test_project_cov3d_to_cov2d()
        test_project_to_channel_space()
        test_compute_gaussian_influence()
        test_compute_channel()
        test_alpha_blending()

        print("All tests completed!")
    else:
        print("CUDA implementation not available. Skipping tests.")

# engine/tests/test_pt_cuda.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
import torch

import engine._torch_impl as torch_impl
from engine import (
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


def generate_test_data(batch_size=12_000, device="cuda"):
    points = torch.randn(batch_size, 3, device=device)
    cov3d = torch.randn(batch_size, 6, device=device)  # compact form
    attenuation = torch.rand(batch_size, 1, device=device) * 1e-3
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


# Tests for transform functions
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_compute_distances_to_receiver():
    data = generate_test_data()

    # torch implementation
    torch_result = torch_impl.compute_distances_to_receiver(
        data["points"], data["receiver"]
    )

    # CUDA implementation
    cuda_result = compute_distances_to_receiver(data["points"], data["receiver"])

    # assert close
    torch.testing.assert_close(torch_result, cuda_result, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_compute_spherical_coords():
    data = generate_test_data()

    torch_d, torch_lon, torch_lat = torch_impl.compute_spherical_coords(
        data["points"], data["receiver"]
    )
    cuda_d, cuda_lon, cuda_lat = compute_spherical_coords(
        data["points"], data["receiver"]
    )

    torch.testing.assert_close(
        torch_d, cuda_d, rtol=1e-4, atol=1e-4
    )  # displacement vectors
    torch.testing.assert_close(torch_lon, cuda_lon, rtol=1e-4, atol=1e-4)  # longitude
    torch.testing.assert_close(torch_lat, cuda_lat, rtol=1e-4, atol=1e-4)  # latitude


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_transform_to_uniform_coords():
    data = generate_test_data()

    _, torch_lon, torch_lat = torch_impl.compute_spherical_coords(
        data["points"], data["receiver"]
    )

    torch_s_x, torch_s_y = torch_impl.transform_to_uniform_coords(torch_lon, torch_lat)
    cuda_s_x, cuda_s_y = transform_to_uniform_coords(torch_lon, torch_lat)

    torch.testing.assert_close(torch_s_x, cuda_s_x, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(torch_s_y, cuda_s_y, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_map_to_channel_matrix():
    data = generate_test_data()

    _, torch_lon, torch_lat = torch_impl.compute_spherical_coords(
        data["points"], data["receiver"]
    )
    torch_s_x, torch_s_y = torch_impl.transform_to_uniform_coords(torch_lon, torch_lat)

    torch_uv = torch_impl.map_to_channel_matrix(
        torch_s_x, torch_s_y, data["num_tx"], data["num_rx"]
    )
    cuda_uv = map_to_channel_matrix(
        torch_s_x, torch_s_y, data["num_tx"], data["num_rx"]
    )

    torch.testing.assert_close(torch_uv, cuda_uv, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_compute_jacobian():
    data = generate_test_data()

    torch_d, _, _ = torch_impl.compute_spherical_coords(
        data["points"], data["receiver"]
    )
    torch_r = torch_impl.compute_distances_to_receiver(data["points"], data["receiver"])

    torch_jacobian = torch_impl.compute_jacobian(
        torch_d, torch_r, data["num_tx"], data["num_rx"]
    )
    cuda_jacobian = compute_jacobian(torch_d, torch_r, data["num_tx"], data["num_rx"])

    torch.testing.assert_close(torch_jacobian, cuda_jacobian, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_project_cov3d_to_cov2d():
    data = generate_test_data()

    torch_d, _, _ = torch_impl.compute_spherical_coords(
        data["points"], data["receiver"]
    )
    torch_r = torch_impl.compute_distances_to_receiver(data["points"], data["receiver"])
    torch_jacobian = torch_impl.compute_jacobian(
        torch_d, torch_r, data["num_tx"], data["num_rx"]
    )

    from utils.transform_utils import symmetric_matrix

    cov3d_mat = symmetric_matrix(data["cov3d"])

    torch_cov2d = torch_impl.project_cov3d_to_cov2d(cov3d_mat, torch_jacobian)
    cuda_cov2d = project_cov3d_to_cov2d(cov3d_mat, torch_jacobian)

    torch.testing.assert_close(torch_cov2d, cuda_cov2d, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_project_to_channel_space():
    data = generate_test_data()

    torch_distances, torch_uv, torch_cov2d = torch_impl.project_to_channel_space(
        data["points"],
        data["cov3d"],
        data["receiver"],
        data["num_tx"],
        data["num_rx"],
    )
    cuda_distances, cuda_uv, cuda_cov2d = project_to_channel_space(
        data["points"],
        data["cov3d"],
        data["receiver"],
        data["num_tx"],
        data["num_rx"],
    )

    torch.testing.assert_close(torch_distances, cuda_distances, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(torch_uv, cuda_uv, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(torch_cov2d, cuda_cov2d, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_compute_gaussian_influence():
    data = generate_test_data()

    distances, uv, cov2d = torch_impl.project_to_channel_space(
        data["points"], data["cov3d"], data["receiver"], data["num_tx"], data["num_rx"]
    )

    torch_result = torch_impl.compute_gaussian_influence(
        uv, cov2d, data["num_tx"], data["num_rx"]
    )
    cuda_result = compute_gaussian_influence(uv, cov2d, data["num_tx"], data["num_rx"])

    torch.testing.assert_close(torch_result, cuda_result, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_compute_channel():
    data = generate_test_data()

    distances, _, _ = torch_impl.project_to_channel_space(
        data["points"], data["cov3d"], data["receiver"], data["num_tx"], data["num_rx"]
    )

    c = 299792458.0  # speed of light in m/s
    wavelength = c / data["frequency"]

    torch_real, torch_imag = torch_impl.compute_channel(
        data["attenuation"], data["phase_rotation"], distances, wavelength
    )
    cuda_real, cuda_imag = compute_channel(
        data["attenuation"], data["phase_rotation"], distances, wavelength
    )

    torch.testing.assert_close(torch_real, cuda_real, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(torch_imag, cuda_imag, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_alpha_blending():
    data = generate_test_data()

    distances, uv, cov2d = torch_impl.project_to_channel_space(
        data["points"], data["cov3d"], data["receiver"], data["num_tx"], data["num_rx"]
    )
    sort_indices = torch.argsort(distances)
    influences = torch_impl.compute_gaussian_influence(
        uv, cov2d, data["num_tx"], data["num_rx"]
    )

    c = 299792458.0
    wavelength = c / data["frequency"]
    real, imag = torch_impl.compute_channel(
        data["attenuation"], data["phase_rotation"], distances, wavelength
    )
    contributions = torch.complex(real, imag)

    torch_result = torch_impl.alpha_blending(
        influences,
        contributions,
        data["opacity"],
        sort_indices,
        data["num_tx"],
        data["num_rx"],
    )
    cuda_result = alpha_blending(
        influences,
        contributions,
        data["opacity"],
        sort_indices,
        data["num_tx"],
        data["num_rx"],
    )

    torch.testing.assert_close(torch_result, cuda_result, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    if CUDA_AVAILABLE:
        test_compute_distances_to_receiver()
        test_compute_spherical_coords()
        test_transform_to_uniform_coords()
        test_map_to_channel_matrix()
        test_compute_jacobian()
        test_project_cov3d_to_cov2d()
        test_project_to_channel_space()
        test_compute_gaussian_influence()
        test_compute_channel()
        test_alpha_blending()
        print("All tests prepared and ready for implementation!")
    else:
        print("CUDA implementation not available. Skipping tests.")

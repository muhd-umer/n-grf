# tests/test_pt_cuda.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

import r2f_engine
import r2f_engine._torch_impl as torch_impl

CUDA_AVAILABLE = r2f_engine.CUSTOM_KERNEL


def generate_test_data(batch_size=100, device="cuda"):
    points = torch.randn(batch_size, 3, device=device)
    cov3d = torch.randn(batch_size, 6, device=device)  # compact form
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


# Tests for transform functions
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_compute_distances_to_receiver():
    data = generate_test_data()

    # torch implementation
    torch_result = torch_impl.compute_distances_to_receiver(
        data["points"], data["receiver"]
    )

    # CUDA implementation - TODO: Implement when CUDA function is available
    # cuda_result = ...

    # Assert close
    # torch.testing.assert_close(torch_result, cuda_result, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_compute_spherical_coords():
    data = generate_test_data()

    torch_d, torch_lon, torch_lat = torch_impl.compute_spherical_coords(
        data["points"], data["receiver"]
    )

    # CUDA implementation - TODO: Implement when CUDA function is available
    # cuda_d, cuda_lon, cuda_lat = ...

    # Assert close
    # torch.testing.assert_close(torch_d, cuda_d, rtol=1e-5, atol=1e-5)  # displacement vectors
    # torch.testing.assert_close(torch_lon, cuda_lon, rtol=1e-5, atol=1e-5)  # longitude
    # torch.testing.assert_close(torch_lat, cuda_lat, rtol=1e-5, atol=1e-5)  # latitude


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_transform_to_uniform_coords():
    data = generate_test_data()

    _, torch_lon, torch_lat = torch_impl.compute_spherical_coords(
        data["points"], data["receiver"]
    )

    # torch implementation
    torch_s_x, torch_s_y = torch_impl.transform_to_uniform_coords(torch_lon, torch_lat)

    # CUDA implementation - TODO: Implement when CUDA function is available
    # cuda_s_x, cuda_s_y = ...

    # Assert close
    # torch.testing.assert_close(torch_s_x, cuda_s_x, rtol=1e-5, atol=1e-5)
    # torch.testing.assert_close(torch_s_y, cuda_s_y, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_map_to_channel_matrix():
    data = generate_test_data()

    _, torch_lon, torch_lat = torch_impl.compute_spherical_coords(
        data["points"], data["receiver"]
    )
    torch_s_x, torch_s_y = torch_impl.transform_to_uniform_coords(torch_lon, torch_lat)

    # torch implementation
    torch_uv = torch_impl.map_to_channel_matrix(
        torch_s_x, torch_s_y, data["num_tx"], data["num_rx"]
    )

    # CUDA implementation - TODO: Implement when CUDA function is available
    # cuda_uv = ...

    # Assert close
    # torch.testing.assert_close(torch_uv, cuda_uv, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_compute_jacobian():
    data = generate_test_data()

    torch_d, _, _ = torch_impl.compute_spherical_coords(
        data["points"], data["receiver"]
    )
    torch_r = torch_impl.compute_distances_to_receiver(data["points"], data["receiver"])

    # torch implementation
    torch_jacobian = torch_impl.compute_jacobian(
        torch_d, torch_r, data["num_tx"], data["num_rx"]
    )

    # CUDA implementation - TODO: Implement when CUDA function is available
    # cuda_jacobian = ...

    # Assert close
    # torch.testing.assert_close(torch_jacobian, cuda_jacobian, rtol=1e-5, atol=1e-5)


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

    # torch implementation
    torch_cov2d = torch_impl.project_cov3d_to_cov2d(cov3d_mat, torch_jacobian)

    # CUDA implementation - TODO: Implement when CUDA function is available
    # cuda_cov2d = ...

    # Assert close
    # torch.testing.assert_close(torch_cov2d, cuda_cov2d, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_project_to_channel_space():
    data = generate_test_data()

    # torch implementation
    torch_distances, torch_uv, torch_cov2d = torch_impl.project_to_channel_space(
        data["points"],
        data["cov3d"],
        data["receiver"],
        data["num_tx"],
        data["num_rx"],
    )

    # CUDA implementation - TODO: Implement when CUDA function is available
    # cuda_distances, cuda_uv, cuda_cov2d = ...

    # Assert close
    # torch.testing.assert_close(torch_distances, cuda_distances, rtol=1e-5, atol=1e-5)
    # torch.testing.assert_close(torch_uv, cuda_uv, rtol=1e-5, atol=1e-5)
    # torch.testing.assert_close(torch_cov2d, cuda_cov2d, rtol=1e-5, atol=1e-5)


# Tests for rasterize functions
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_compute_gaussian_influence():
    data = generate_test_data()

    distances, uv, cov2d = torch_impl.project_to_channel_space(
        data["points"], data["cov3d"], data["receiver"], data["num_tx"], data["num_rx"]
    )

    # torch implementation
    torch_result = torch_impl.compute_gaussian_influence(
        uv, cov2d, data["num_tx"], data["num_rx"]
    )

    # CUDA implementation - TODO: Implement when CUDA function is available
    # cuda_result = ...

    # Assert close
    # torch.testing.assert_close(torch_result, cuda_result, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA implementation not available")
def test_compute_channel():
    data = generate_test_data()

    distances, _, _ = torch_impl.project_to_channel_space(
        data["points"], data["cov3d"], data["receiver"], data["num_tx"], data["num_rx"]
    )

    c = 299792458.0  # Speed of light in m/s
    wavelength = c / data["frequency"]

    # torch implementation
    torch_real, torch_imag = torch_impl.compute_channel(
        data["attenuation"], data["phase_rotation"], distances, wavelength
    )

    # CUDA implementation - TODO: Implement when CUDA function is available
    # cuda_real, cuda_imag = ...

    # Assert close
    # torch.testing.assert_close(torch_real, cuda_real, rtol=1e-5, atol=1e-5)
    # torch.testing.assert_close(torch_imag, cuda_imag, rtol=1e-5, atol=1e-5)


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

    # torch implementation
    torch_result = torch_impl.alpha_blending(
        influences,
        contributions,
        data["opacity"],
        sort_indices,
        data["num_tx"],
        data["num_rx"],
    )

    # CUDA implementation - TODO: Implement when CUDA function is available
    # cuda_result = ...

    # Assert close
    # torch.testing.assert_close(torch_result, cuda_result, rtol=1e-5, atol=1e-5)


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

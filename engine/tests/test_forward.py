# engine/tests/test_forward.py

import math
import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _torch_impl.rasterize import (
    compute_scattered_paths as torch_compute_scattered_paths,
)
from _torch_impl.rasterize import (
    compute_spatial_influence as torch_compute_spatial_influence,
)
from _torch_impl.rasterize import (
    compute_steering_vector as torch_compute_steering_vector,
)
from _torch_impl.rasterize import rasterize as torch_rasterize
from _torch_impl.rasterize import weighted_superposition as torch_weighted_superposition
from _torch_impl.transforms import compute_jacobian as torch_compute_jacobian
from _torch_impl.transforms import compute_path_geometry as torch_compute_path_geometry
from _torch_impl.transforms import (
    project_cov3d_to_cov2d as torch_project_cov3d_to_cov2d,
)
from _torch_impl.transforms import (
    project_to_channel_coords as torch_project_to_channel_coords,
)

try:
    from _wrapper import CUDA_AVAILABLE
    from _wrapper import rasterize as cuda_rasterize

    if not CUDA_AVAILABLE:
        raise ImportError("CUDA extension loaded but CUDA is not available.")
except ImportError:
    CUDA_AVAILABLE = False
    print(
        "CUDA wrapper not found or CUDA not available. Skipping CUDA-specific tests.",
        file=sys.stderr,
    )

    class DummyFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args, **kwargs):
            raise unittest.SkipTest("CUDA implementation not available")

        @staticmethod
        def backward(ctx, *args, **kwargs):
            raise unittest.SkipTest("CUDA implementation not available")

    QuaternionToRotation = ComputeScalingMatrix = MatrixMultiply = CovarianceMatrix = (
        DummyFunction
    )
    ProjectToChannelCoordinates = ComputeJacobian = ProjectCov3dToCov2d = DummyFunction
    ComputeSpatialInfluence = ComputePathGeometry = ComputeSteeringVector = (
        DummyFunction
    )
    ComputeScatteredPaths = WeightedSuperposition = DummyFunction

    def cuda_rasterize(*args, **kwargs):
        raise unittest.SkipTest("CUDA implementation not available")


class TestForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        if CUDA_AVAILABLE and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Testing on CUDA device.")
        else:
            self.device = torch.device("cpu")
            print("Testing on CPU device (CUDA tests will be skipped).")
            if not CUDA_AVAILABLE:
                print("CUDA extension not built/found.")
            elif not torch.cuda.is_available():
                print("CUDA extension found, but no CUDA device detected by PyTorch.")

        self.num_gaussians = 100
        self.num_tx = 16
        self.num_rx = 2
        self.frequency = 2.4e9
        self.wavelength = 299792458.0 / self.frequency
        self.scale_modifier = 1.0
        self.dtype = torch.double

        self.tx_params = {
            "position": torch.rand(3, device=self.device, dtype=self.dtype) * 10,
            "type": "ura",
            "size": [4, 4],
            "element_spacing": [self.wavelength / 2, self.wavelength / 2],
            "frequency": self.frequency,
            "num_antennas": self.num_tx,
        }
        self.rx_params = {
            "position": torch.rand(3, device=self.device, dtype=self.dtype) * 10,
            "type": "ula",
            "size": self.num_rx,
            "element_spacing": self.wavelength / 2,
            "num_antennas": self.num_rx,
        }

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_quaternion_to_rotation(self):
        from _wrapper import QuaternionToRotation

        quaternions = torch.rand(
            self.num_gaussians, 4, device=self.device, dtype=self.dtype
        )
        quaternions_norm = F.normalize(quaternions, dim=1)

        w, x, y, z = (
            quaternions_norm[:, 0],
            quaternions_norm[:, 1],
            quaternions_norm[:, 2],
            quaternions_norm[:, 3],
        )
        R_torch = torch.stack(
            [
                1 - 2 * y**2 - 2 * z**2,
                2 * x * y - 2 * w * z,
                2 * x * z + 2 * w * y,
                2 * x * y + 2 * w * z,
                1 - 2 * x**2 - 2 * z**2,
                2 * y * z - 2 * w * x,
                2 * x * z - 2 * w * y,
                2 * y * z + 2 * w * x,
                1 - 2 * x**2 - 2 * y**2,
            ],
            dim=1,
        ).reshape(-1, 3, 3)

        R_cuda = QuaternionToRotation.apply(quaternions)

        self.assertTrue(torch.allclose(R_torch, R_cuda, rtol=1e-5, atol=1e-5))

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_compute_scaling_matrix(self):
        from _wrapper import ComputeScalingMatrix

        scaling = torch.exp(
            torch.rand(self.num_gaussians, 3, device=self.device, dtype=self.dtype)
        )

        scaled_scaling = scaling * self.scale_modifier
        S_torch = torch.diag_embed(scaled_scaling)

        S_cuda = ComputeScalingMatrix.apply(scaling, self.scale_modifier)

        self.assertTrue(torch.allclose(S_torch, S_cuda, rtol=1e-5, atol=1e-5))

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_project_to_channel_coords(self):
        from _wrapper import ProjectToChannelCoordinates

        points = (
            torch.rand(self.num_gaussians, 3, device=self.device, dtype=self.dtype) * 10
        )
        receiver = torch.rand(3, device=self.device, dtype=self.dtype) * 10

        r_torch, d_torch, uv_torch = torch_project_to_channel_coords(
            points, receiver, self.num_tx, self.num_rx
        )

        r_cuda, d_cuda, uv_cuda = ProjectToChannelCoordinates.apply(
            points, receiver, self.num_tx, self.num_rx
        )

        self.assertTrue(torch.allclose(r_torch, r_cuda, rtol=1e-5, atol=1e-5))
        self.assertTrue(torch.allclose(d_torch, d_cuda, rtol=1e-5, atol=1e-5))
        self.assertTrue(torch.allclose(uv_torch, uv_cuda, rtol=1e-5, atol=1e-5))

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_compute_jacobian(self):
        from _wrapper import ComputeJacobian

        d = (
            torch.rand(self.num_gaussians, 3, device=self.device, dtype=self.dtype) * 10
            + 0.1
        )

        J_torch = torch_compute_jacobian(d, self.num_tx, self.num_rx)
        J_cuda = ComputeJacobian.apply(d, self.num_tx, self.num_rx)

        self.assertTrue(torch.allclose(J_torch, J_cuda, rtol=1e-5, atol=1e-5))

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_project_cov3d_to_cov2d(self):
        from _wrapper import ProjectCov3dToCov2d

        cov3d_base = torch.rand(
            self.num_gaussians, 3, 3, device=self.device, dtype=self.dtype
        )
        cov3d = (
            0.5 * (cov3d_base + cov3d_base.transpose(1, 2))
            + torch.eye(3, device=self.device, dtype=self.dtype) * 1e-3
        )
        jacobian = torch.rand(
            self.num_gaussians, 2, 3, device=self.device, dtype=self.dtype
        )

        cov2d_torch = torch_project_cov3d_to_cov2d(cov3d, jacobian)
        cov2d_cuda = ProjectCov3dToCov2d.apply(cov3d, jacobian)

        self.assertTrue(torch.allclose(cov2d_torch, cov2d_cuda, rtol=1e-5, atol=1e-5))

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_compute_spatial_influence(self):
        from _wrapper import ComputeSpatialInfluence

        uv = (
            torch.rand(self.num_gaussians, 2, device=self.device, dtype=self.dtype)
            * self.num_tx
        )

        cov2d_base = torch.rand(
            self.num_gaussians, 2, 2, device=self.device, dtype=self.dtype
        )
        cov2d = torch.matmul(cov2d_base, cov2d_base.transpose(1, 2))
        cov2d = cov2d + 0.5 * torch.eye(
            2, device=self.device, dtype=self.dtype
        ).unsqueeze(0)

        influences_torch = torch_compute_spatial_influence(
            uv, cov2d, self.num_tx, self.num_rx
        )
        influences_cuda = ComputeSpatialInfluence.apply(
            uv, cov2d, self.num_tx, self.num_rx
        )

        self.assertTrue(
            torch.allclose(influences_torch, influences_cuda, rtol=1e-5, atol=1e-5)
        )

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_compute_path_geometry(self):
        from _wrapper import ComputePathGeometry

        points = (
            torch.rand(self.num_gaussians, 3, device=self.device, dtype=self.dtype) * 10
        )
        tx_pos = torch.rand(3, device=self.device, dtype=self.dtype) * 10
        rx_pos = torch.rand(3, device=self.device, dtype=self.dtype) * 10

        dist_tx_torch, dist_rx_torch, aod_torch, aoa_torch = (
            torch_compute_path_geometry(points, tx_pos, rx_pos)
        )
        dist_tx_cuda, dist_rx_cuda, aod_cuda, aoa_cuda = ComputePathGeometry.apply(
            points, tx_pos, rx_pos
        )

        self.assertTrue(
            torch.allclose(dist_tx_torch, dist_tx_cuda, rtol=1e-5, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(dist_rx_torch, dist_rx_cuda, rtol=1e-5, atol=1e-5)
        )
        self.assertTrue(torch.allclose(aod_torch, aod_cuda, rtol=1e-5, atol=1e-5))
        self.assertTrue(torch.allclose(aoa_torch, aoa_cuda, rtol=1e-5, atol=1e-5))

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_compute_steering_vector(self):
        from _wrapper import ComputeSteeringVector

        angles = (
            torch.rand(self.num_gaussians, 2, device=self.device, dtype=self.dtype)
            * math.pi
            - math.pi / 2
        )
        angles[:, 1] = angles[:, 1].clamp(-math.pi / 2 + 1e-7, math.pi / 2 - 1e-7)

        sv_real_torch_ura, sv_imag_torch_ura = torch_compute_steering_vector(
            angles, self.tx_params, self.wavelength
        )
        sv_real_cuda_ura, sv_imag_cuda_ura = ComputeSteeringVector.apply(
            angles, self.tx_params, self.wavelength
        )
        self.assertTrue(
            torch.allclose(sv_real_torch_ura, sv_real_cuda_ura, rtol=1e-5, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(sv_imag_torch_ura, sv_imag_cuda_ura, rtol=1e-5, atol=1e-5)
        )

        sv_real_torch_ula, sv_imag_torch_ula = torch_compute_steering_vector(
            angles, self.rx_params, self.wavelength
        )
        sv_real_cuda_ula, sv_imag_cuda_ula = ComputeSteeringVector.apply(
            angles, self.rx_params, self.wavelength
        )
        self.assertTrue(
            torch.allclose(sv_real_torch_ula, sv_real_cuda_ula, rtol=1e-5, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(sv_imag_torch_ula, sv_imag_cuda_ula, rtol=1e-5, atol=1e-5)
        )

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_compute_scattered_paths(self):
        from _wrapper import ComputeScatteredPaths

        gamma_real = torch.randn(
            self.num_gaussians, device=self.device, dtype=self.dtype
        )
        gamma_imag = torch.randn(
            self.num_gaussians, device=self.device, dtype=self.dtype
        )
        dist_tx = (
            torch.rand(self.num_gaussians, device=self.device, dtype=self.dtype) + 0.1
        )
        dist_rx = (
            torch.rand(self.num_gaussians, device=self.device, dtype=self.dtype) + 0.1
        )
        sv_tx_real = torch.randn(
            self.num_gaussians, self.num_tx, device=self.device, dtype=self.dtype
        )
        sv_tx_imag = torch.randn(
            self.num_gaussians, self.num_tx, device=self.device, dtype=self.dtype
        )
        sv_rx_real = torch.randn(
            self.num_gaussians, self.num_rx, device=self.device, dtype=self.dtype
        )
        sv_rx_imag = torch.randn(
            self.num_gaussians, self.num_rx, device=self.device, dtype=self.dtype
        )

        scat_real_torch, scat_imag_torch = torch_compute_scattered_paths(
            gamma_real,
            gamma_imag,
            dist_tx,
            dist_rx,
            sv_tx_real,
            sv_tx_imag,
            sv_rx_real,
            sv_rx_imag,
            self.wavelength,
        )
        scat_real_cuda, scat_imag_cuda = ComputeScatteredPaths.apply(
            gamma_real,
            gamma_imag,
            dist_tx,
            dist_rx,
            sv_tx_real,
            sv_tx_imag,
            sv_rx_real,
            sv_rx_imag,
            self.wavelength,
        )

        self.assertTrue(
            torch.allclose(scat_real_torch, scat_real_cuda, rtol=1e-5, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(scat_imag_torch, scat_imag_cuda, rtol=1e-5, atol=1e-5)
        )

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_weighted_superposition(self):
        from _torch_impl.rasterize import (
            compute_direct_path as torch_compute_direct_path,
        )
        from _wrapper import WeightedSuperposition

        direct_real, direct_imag = torch_compute_direct_path(
            self.tx_params, self.rx_params, self.wavelength
        )
        scat_real = torch.randn(
            self.num_gaussians,
            self.num_tx,
            self.num_rx,
            device=self.device,
            dtype=self.dtype,
        )
        scat_imag = torch.randn(
            self.num_gaussians,
            self.num_tx,
            self.num_rx,
            device=self.device,
            dtype=self.dtype,
        )
        opacity = torch.rand(
            self.num_gaussians, 1, device=self.device, dtype=self.dtype
        )
        influence = torch.rand(
            self.num_gaussians,
            self.num_tx,
            self.num_rx,
            device=self.device,
            dtype=self.dtype,
        )

        chan_real_torch, chan_imag_torch = torch_weighted_superposition(
            direct_real, direct_imag, scat_real, scat_imag, opacity, influence
        )
        chan_real_cuda, chan_imag_cuda = WeightedSuperposition.apply(
            direct_real, direct_imag, scat_real, scat_imag, opacity, influence
        )

        self.assertTrue(
            torch.allclose(chan_real_torch, chan_real_cuda, rtol=1e-5, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(chan_imag_torch, chan_imag_cuda, rtol=1e-5, atol=1e-5)
        )

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_end_to_end_rasterize(self):
        from _wrapper import rasterize as rasterize_cuda

        points = (
            torch.rand(self.num_gaussians, 3, device=self.device, dtype=self.dtype) * 10
        )
        scaling = torch.rand(
            self.num_gaussians, 3, device=self.device, dtype=self.dtype
        )
        rotation = torch.rand(
            self.num_gaussians, 4, device=self.device, dtype=self.dtype
        )
        rotation = F.normalize(rotation, dim=1)
        gamma_real = (
            torch.randn(self.num_gaussians, device=self.device, dtype=self.dtype) * 1e-3
        )
        gamma_imag = (
            torch.randn(self.num_gaussians, device=self.device, dtype=self.dtype) * 1e-3
        )
        gamma = torch.stack([gamma_real, gamma_imag], dim=1)
        opacity = torch.sigmoid(
            torch.rand(self.num_gaussians, 1, device=self.device, dtype=self.dtype)
        )

        cat_channel_torch = torch_rasterize(
            points=points,
            scaling=scaling,
            rotation=rotation,
            gamma=gamma,
            opacity=opacity,
            tx_params=self.tx_params,
            rx_params=self.rx_params,
            scale_modifier=self.scale_modifier,
        )

        cat_channel_cuda = cuda_rasterize(
            points=points,
            scaling=scaling,
            rotation=rotation,
            gamma=gamma,
            opacity=opacity,
            tx_params=self.tx_params,
            rx_params=self.rx_params,
            scale_modifier=self.scale_modifier,
        )

        is_close = torch.allclose(
            cat_channel_torch, cat_channel_cuda, rtol=1e-3, atol=1e-3
        )
        if not is_close:
            max_diff = torch.max(torch.abs(cat_channel_torch - cat_channel_cuda))
            print(f"Maximum difference in end_to_end_rasterize test: {max_diff.item()}")

        self.assertTrue(is_close)


if __name__ == "__main__":
    unittest.main()

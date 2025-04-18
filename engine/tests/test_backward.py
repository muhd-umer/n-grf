# engine/tests/test_backward.py

import math
import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


try:
    from _wrapper import (
        CUDA_AVAILABLE,
        ComputeJacobian,
        ComputePathGeometry,
        ComputeScalingMatrix,
        ComputeScatteredPaths,
        ComputeSpatialInfluence,
        ComputeSteeringVector,
        CovarianceMatrix,
        MatrixMultiply,
        ProjectCov3dToCov2d,
        ProjectToChannelCoordinates,
        QuaternionToRotation,
        WeightedSuperposition,
    )
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


from _torch_impl.rasterize import compute_direct_path as torch_compute_direct_path
from _torch_impl.rasterize import rasterize as torch_rasterize


class TestBackward(unittest.TestCase):
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
        self.num_tx = 4
        self.num_rx = 2
        self.frequency = 2.4e9
        self.wavelength = 299792458.0 / self.frequency
        self.scale_modifier = 1.0
        self.dtype = torch.double
        self.gradcheck_eps = 1e-6
        self.gradcheck_atol = 1e-5
        self.gradcheck_rtol = 1e-3

        self.tx_params = {
            "position": torch.rand(3, device=self.device, dtype=self.dtype) * 10,
            "type": "ura",
            "size": [2, 2],
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
    def test_quaternion_to_rotation_gradcheck(self):
        quaternions = torch.rand(
            self.num_gaussians,
            4,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        self.assertTrue(
            torch.autograd.gradcheck(
                QuaternionToRotation.apply,
                (quaternions,),
                eps=self.gradcheck_eps,
                atol=self.gradcheck_atol,
                rtol=self.gradcheck_rtol,
            )
        )

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_compute_scaling_matrix_gradcheck(self):
        scaling = torch.rand(
            self.num_gaussians,
            3,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        def func_to_check(s):
            return ComputeScalingMatrix.apply(s, self.scale_modifier)

        self.assertTrue(
            torch.autograd.gradcheck(
                func_to_check,
                (scaling,),
                eps=self.gradcheck_eps,
                atol=self.gradcheck_atol,
                rtol=self.gradcheck_rtol,
            )
        )

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_matrix_multiply_gradcheck(self):
        A = torch.rand(
            self.num_gaussians,
            3,
            3,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        B = torch.rand(
            self.num_gaussians,
            3,
            3,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        self.assertTrue(
            torch.autograd.gradcheck(
                MatrixMultiply.apply,
                (A, B),
                eps=self.gradcheck_eps,
                atol=self.gradcheck_atol,
                rtol=self.gradcheck_rtol,
            )
        )

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_covariance_matrix_gradcheck(self):
        RS = torch.rand(
            self.num_gaussians,
            3,
            3,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        self.assertTrue(
            torch.autograd.gradcheck(
                CovarianceMatrix.apply,
                (RS,),
                eps=self.gradcheck_eps,
                atol=self.gradcheck_atol,
                rtol=self.gradcheck_rtol,
            )
        )

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_project_to_channel_coords_gradcheck(self):
        points = (
            torch.rand(
                self.num_gaussians,
                3,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
            * 5
        )
        receiver = torch.rand(3, device=self.device, dtype=self.dtype) * 5

        def func_dist(p):
            return ProjectToChannelCoordinates.apply(
                p, receiver, self.num_tx, self.num_rx
            )[0]

        def func_disp(
            p,
        ):
            return ProjectToChannelCoordinates.apply(
                p, receiver, self.num_tx, self.num_rx
            )[1]

        def func_uv(p):
            return ProjectToChannelCoordinates.apply(
                p, receiver, self.num_tx, self.num_rx
            )[2]

        print("Checking ProjectToChannelCoords grad (distances)...")
        self.assertTrue(
            torch.autograd.gradcheck(
                func_dist,
                (points,),
                eps=self.gradcheck_eps,
                atol=self.gradcheck_atol,
                rtol=self.gradcheck_rtol,
            )
        )

        print("Checking ProjectToChannelCoords grad (displacement)...")
        self.assertTrue(
            torch.autograd.gradcheck(
                func_disp,
                (points,),
                eps=self.gradcheck_eps,
                atol=self.gradcheck_atol,
                rtol=self.gradcheck_rtol,
            )
        )

        print("Checking ProjectToChannelCoords grad (uv)...")
        self.assertTrue(
            torch.autograd.gradcheck(
                func_uv,
                (points,),
                eps=self.gradcheck_eps,
                atol=self.gradcheck_atol,
                rtol=self.gradcheck_rtol,
            )
        )

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_compute_jacobian_gradcheck(self):
        d = (
            torch.rand(
                self.num_gaussians,
                3,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
            * 5
            + 0.1
        )

        d[:, 2] = torch.clamp(d[:, 2], min=0.1)

        def func_to_check(disp):
            return ComputeJacobian.apply(disp, self.num_tx, self.num_rx)

        self.assertTrue(
            torch.autograd.gradcheck(
                func_to_check,
                (d,),
                eps=self.gradcheck_eps,
                atol=1e-4,
                rtol=self.gradcheck_rtol,
            )
        )

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_project_cov3d_to_cov2d_gradcheck(self):
        cov3d = torch.rand(
            self.num_gaussians, 3, 3, device=self.device, dtype=self.dtype
        )
        cov3d = (
            torch.matmul(cov3d, cov3d.transpose(1, 2))
            + torch.eye(3, device=self.device, dtype=self.dtype) * 1e-3
        )
        cov3d.requires_grad_(True)

        jacobian = torch.rand(
            self.num_gaussians,
            2,
            3,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        self.assertTrue(
            torch.autograd.gradcheck(
                ProjectCov3dToCov2d.apply,
                (cov3d, jacobian),
                eps=self.gradcheck_eps,
                atol=self.gradcheck_atol,
                rtol=self.gradcheck_rtol,
            )
        )

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_compute_spatial_influence_gradcheck(self):
        uv = (
            torch.rand(
                self.num_gaussians,
                2,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
            * self.num_tx
        )

        cov2d_base = torch.rand(
            self.num_gaussians, 2, 2, device=self.device, dtype=self.dtype
        )

        cov2d = torch.matmul(cov2d_base, cov2d_base.transpose(1, 2))
        cov2d = (
            cov2d
            + torch.eye(2, device=self.device, dtype=self.dtype).unsqueeze(0) * 0.5
        )
        cov2d.requires_grad_(True)

        def func_to_check(u, c2d):
            return ComputeSpatialInfluence.apply(u, c2d, self.num_tx, self.num_rx)

        self.assertTrue(
            torch.autograd.gradcheck(
                func_to_check,
                (uv, cov2d),
                eps=self.gradcheck_eps,
                atol=self.gradcheck_atol,
                rtol=self.gradcheck_rtol,
            )
        )

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_compute_path_geometry_gradcheck(self):
        points = (
            torch.rand(
                self.num_gaussians,
                3,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
            * 5
            + 0.1
        )
        tx_pos = torch.rand(3, device=self.device, dtype=self.dtype) * 5
        rx_pos = torch.rand(3, device=self.device, dtype=self.dtype) * 5 + 10

        def func_dist_tx(p):
            return ComputePathGeometry.apply(p, tx_pos, rx_pos)[0]

        def func_dist_rx(p):
            return ComputePathGeometry.apply(p, tx_pos, rx_pos)[1]

        def func_aod(p):
            return ComputePathGeometry.apply(p, tx_pos, rx_pos)[2]

        def func_aoa(p):
            return ComputePathGeometry.apply(p, tx_pos, rx_pos)[3]

        print("Checking ComputePathGeometry grad (dist_tx)...")
        self.assertTrue(
            torch.autograd.gradcheck(
                func_dist_tx,
                (points,),
                eps=self.gradcheck_eps,
                atol=self.gradcheck_atol,
                rtol=self.gradcheck_rtol,
            )
        )
        print("Checking ComputePathGeometry grad (dist_rx)...")
        self.assertTrue(
            torch.autograd.gradcheck(
                func_dist_rx,
                (points,),
                eps=self.gradcheck_eps,
                atol=self.gradcheck_atol,
                rtol=self.gradcheck_rtol,
            )
        )
        print("Checking ComputePathGeometry grad (aod)...")
        self.assertTrue(
            torch.autograd.gradcheck(
                func_aod,
                (points,),
                eps=self.gradcheck_eps,
                atol=1e-4,
                rtol=self.gradcheck_rtol,
            )
        )
        print("Checking ComputePathGeometry grad (aoa)...")
        self.assertTrue(
            torch.autograd.gradcheck(
                func_aoa,
                (points,),
                eps=self.gradcheck_eps,
                atol=1e-4,
                rtol=self.gradcheck_rtol,
            )
        )

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_compute_steering_vector_gradcheck(self):
        angles = (
            torch.rand(self.num_gaussians, 2, device=self.device, dtype=self.dtype)
            * math.pi
            - math.pi / 2
        )
        angles[:, 1] = angles[:, 1].clamp(-math.pi / 2 + 1e-6, math.pi / 2 - 1e-6)
        angles.requires_grad_(True)

        def func_ura(a):
            return ComputeSteeringVector.apply(a, self.tx_params, self.wavelength)

        def func_ula(a):
            return ComputeSteeringVector.apply(a, self.rx_params, self.wavelength)

        print("Checking ComputeSteeringVector grad (URA)...")
        self.assertTrue(
            torch.autograd.gradcheck(
                func_ura,
                (angles,),
                eps=self.gradcheck_eps,
                atol=self.gradcheck_atol,
                rtol=self.gradcheck_rtol,
            )
        )
        print("Checking ComputeSteeringVector grad (ULA)...")
        self.assertTrue(
            torch.autograd.gradcheck(
                func_ula,
                (angles,),
                eps=self.gradcheck_eps,
                atol=self.gradcheck_atol,
                rtol=self.gradcheck_rtol,
            )
        )

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_compute_scattered_paths_gradcheck(self):
        gamma_real = torch.randn(
            self.num_gaussians, device=self.device, dtype=self.dtype, requires_grad=True
        )
        gamma_imag = torch.randn(
            self.num_gaussians, device=self.device, dtype=self.dtype, requires_grad=True
        )
        dist_tx = (
            torch.rand(
                self.num_gaussians,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
            + 0.1
        )
        dist_rx = (
            torch.rand(
                self.num_gaussians,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
            + 0.1
        )
        sv_tx_real = torch.randn(
            self.num_gaussians,
            self.num_tx,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        sv_tx_imag = torch.randn(
            self.num_gaussians,
            self.num_tx,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        sv_rx_real = torch.randn(
            self.num_gaussians,
            self.num_rx,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        sv_rx_imag = torch.randn(
            self.num_gaussians,
            self.num_rx,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        inputs = (
            gamma_real,
            gamma_imag,
            dist_tx,
            dist_rx,
            sv_tx_real,
            sv_tx_imag,
            sv_rx_real,
            sv_rx_imag,
        )

        def func_to_check(*args):
            return ComputeScatteredPaths.apply(*args, self.wavelength)

        print("Checking ComputeScatteredPaths grad...")
        self.assertTrue(
            torch.autograd.gradcheck(
                func_to_check,
                inputs,
                eps=self.gradcheck_eps,
                atol=1e-4,
                rtol=self.gradcheck_rtol,
            )
        )

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_weighted_superposition_gradcheck(self):
        direct_real, direct_imag = torch_compute_direct_path(
            self.tx_params, self.rx_params, self.wavelength
        )
        scat_real = torch.randn(
            self.num_gaussians,
            self.num_tx,
            self.num_rx,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        scat_imag = torch.randn(
            self.num_gaussians,
            self.num_tx,
            self.num_rx,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        opacity = torch.rand(
            self.num_gaussians,
            1,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        influence = torch.rand(
            self.num_gaussians,
            self.num_tx,
            self.num_rx,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        inputs = (direct_real, direct_imag, scat_real, scat_imag, opacity, influence)

        print("Checking WeightedSuperposition grad...")
        self.assertTrue(
            torch.autograd.gradcheck(
                WeightedSuperposition.apply,
                inputs,
                eps=self.gradcheck_eps,
                atol=self.gradcheck_atol,
                rtol=self.gradcheck_rtol,
            )
        )

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_rasterize_backward(self):
        points = (
            torch.rand(
                self.num_gaussians,
                3,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
            * 10
        )
        scaling = torch.rand(
            self.num_gaussians,
            3,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        rotation = torch.rand(
            self.num_gaussians,
            4,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        gamma_real = (
            torch.randn(
                self.num_gaussians,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
            * 1e-2
        )
        gamma_imag = (
            torch.randn(
                self.num_gaussians,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
            * 1e-2
        )
        gamma = torch.stack([gamma_real, gamma_imag], dim=1)

        opacity = (
            torch.rand(
                self.num_gaussians,
                1,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
            * 0.9
            + 0.05
        )

        def func_to_check(p, s, r, g, o):

            return cuda_rasterize(
                points=p,
                scaling=s,
                rotation=r,
                gamma=g,
                opacity=o,
                tx_params=self.tx_params,
                rx_params=self.rx_params,
                scale_modifier=self.scale_modifier,
            )

        print("Checking end-to-end rasterize backward...")

        self.assertTrue(
            torch.autograd.gradcheck(
                func_to_check,
                (points, scaling, rotation, gamma, opacity),
                eps=1e-5,
                atol=1e-3,
                rtol=1e-3,
            )
        )


if __name__ == "__main__":
    unittest.main()

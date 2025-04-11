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
        AlphaBlending,
        ComputeGaussianInfluence,
        ComputeJacobian,
        ComputeScalingMatrix,
        ComputeWirelessChannel,
        CovarianceMatrix,
        MatrixMultiply,
        ProjectCov3dToCov2d,
        ProjectToChannelCoords,
        QuaternionToRotation,
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
    ProjectToChannelCoords = ComputeJacobian = ProjectCov3dToCov2d = DummyFunction
    ComputeGaussianInfluence = ComputeWirelessChannel = AlphaBlending = DummyFunction

    def cuda_rasterize(*args, **kwargs):
        raise unittest.SkipTest("CUDA implementation not available")


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

        self.num_gaussians = 10
        self.num_tx = 4
        self.num_rx = 2
        self.frequency = 2.4e9
        self.wavelength = 299792458.0 / self.frequency
        self.scale_modifier = 1.0
        self.dtype = torch.double

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_quaternion_to_rotation_gradcheck(self):
        quaternions = torch.rand(
            self.num_gaussians,
            4,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        def func_to_check(q):
            q_norm = F.normalize(q, dim=1)
            return QuaternionToRotation.apply(q_norm)

        self.assertTrue(
            torch.autograd.gradcheck(
                func_to_check,
                (quaternions,),
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

        self.assertTrue(torch.autograd.gradcheck(func_to_check, (scaling,)))

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

        def func_to_check(a, b):
            return MatrixMultiply.apply(a, b)

        self.assertTrue(torch.autograd.gradcheck(func_to_check, (A, B)))

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

        def func_to_check(rs):
            return CovarianceMatrix.apply(rs)

        self.assertTrue(torch.autograd.gradcheck(func_to_check, (RS,)))

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
            distances, _, _ = ProjectToChannelCoords.apply(
                p, receiver, self.num_tx, self.num_rx
            )
            return distances.sum()

        def func_uv(p):
            _, _, uv = ProjectToChannelCoords.apply(
                p, receiver, self.num_tx, self.num_rx
            )
            return uv.sum()

        print("Checking ProjectToChannelCoords (distances)...")
        self.assertTrue(
            torch.autograd.gradcheck(
                func_dist,
                (points,),
            )
        )
        print("Checking ProjectToChannelCoords (uv)...")
        self.assertTrue(
            torch.autograd.gradcheck(
                func_uv,
                (points,),
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
        d[:, 2] += 0.5

        def func_to_check(disp):
            return ComputeJacobian.apply(disp, self.num_tx, self.num_rx)

        self.assertTrue(
            torch.autograd.gradcheck(
                func_to_check,
                (d,),
                eps=1e-5,
                atol=1e-4,
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

        def func_to_check(c3d, j):
            return ProjectCov3dToCov2d.apply(c3d, j)

        self.assertTrue(
            torch.autograd.gradcheck(
                func_to_check,
                (cov3d, jacobian),
                eps=1e-6,
                atol=1e-5,
            )
        )

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_compute_gaussian_influence_gradcheck(self):
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
            return ComputeGaussianInfluence.apply(u, c2d, self.num_tx, self.num_rx)

        self.assertTrue(
            torch.autograd.gradcheck(
                func_to_check,
                (uv, cov2d),
                eps=1e-6,
                atol=1e-5,
            )
        )

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_compute_wireless_channel_gradcheck(self):
        attenuation = (
            torch.rand(
                self.num_gaussians,
                1,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
            + 0.1
        )
        phase_rotation = (
            torch.rand(
                self.num_gaussians,
                1,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
            * 2
            * math.pi
        )
        distances = (
            torch.rand(
                self.num_gaussians,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
            * 10
            + 0.5
        )

        def func_to_check(a, p, d):
            real, imag = ComputeWirelessChannel.apply(a, p, d, self.wavelength)
            return real.sum() + imag.sum()

        self.assertTrue(
            torch.autograd.gradcheck(
                func_to_check,
                (attenuation, phase_rotation, distances),
            )
        )

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_alpha_blending_gradcheck(self):
        influences = torch.rand(
            self.num_gaussians,
            self.num_tx,
            self.num_rx,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        contributions_real = torch.randn(
            self.num_gaussians,
            1,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        contributions_imag = torch.randn(
            self.num_gaussians,
            1,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        opacity = (
            torch.rand(
                self.num_gaussians,
                1,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
            * 0.9
        )

        sort_indices = torch.randperm(self.num_gaussians, device=self.device).to(
            torch.int32
        )

        def func_to_check(infl, cr, ci, opac):
            return AlphaBlending.apply(
                infl, cr, ci, opac, sort_indices, self.num_tx, self.num_rx
            )

        print("Checking AlphaBlending...")
        self.assertTrue(
            torch.autograd.gradcheck(
                func_to_check,
                (influences, contributions_real, contributions_imag, opacity),
                eps=1e-6,
                atol=1e-4,
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
        attenuation = torch.rand(
            self.num_gaussians,
            1,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        phase_rotation = (
            torch.rand(
                self.num_gaussians,
                1,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
            * 2
            * math.pi
        )
        opacity = (
            torch.rand(
                self.num_gaussians,
                1,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
            * 0.9
        )

        receiver = torch.rand(3, device=self.device, dtype=self.dtype) * 10
        transmitter = torch.rand(3, device=self.device, dtype=self.dtype) * 10

        def func_to_check(p, s, r, a, ph, o):
            r_norm = F.normalize(r, dim=1)
            return cuda_rasterize(
                points=p,
                scaling=s,
                rotation=r_norm,
                attenuation=a,
                phase_rotation=ph,
                opacity=o,
                receiver=receiver,
                transmitter=transmitter,
                num_tx=self.num_tx,
                num_rx=self.num_rx,
                frequency=self.frequency,
                scale_modifier=self.scale_modifier,
            )

        print("Checking end-to-end rasterize backward...")
        self.assertTrue(
            torch.autograd.gradcheck(
                func_to_check,
                (points, scaling, rotation, attenuation, phase_rotation, opacity),
                eps=1e-5,
                atol=1e-3,
                rtol=1e-3,
            )
        )


if __name__ == "__main__":
    unittest.main()

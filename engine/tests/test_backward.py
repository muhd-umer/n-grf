# engine/tests/test_backward.py

import math
import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from _torch_impl.rasterize import rasterize
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


class TestBackward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gaussians = 100
        self.num_tx = 16
        self.num_rx = 2
        self.frequency = 2.4e9
        self.wavelength = 299792458.0 / self.frequency
        self.scale_modifier = 1.0

    def test_quaternion_to_rotation_gradcheck(self):
        if not CUDA_AVAILABLE:
            self.skipTest("CUDA implementation not available")

        quaternions = torch.rand(
            10, 4, device=self.device, dtype=torch.double, requires_grad=True
        )
        quaternions = F.normalize(quaternions, dim=1)

        def quaternion_to_rotation_func(q):
            return QuaternionToRotation.apply(q).sum()

        self.assertTrue(
            torch.autograd.gradcheck(quaternion_to_rotation_func, (quaternions,))
        )

    def test_compute_scaling_matrix_gradcheck(self):
        if not CUDA_AVAILABLE:
            self.skipTest("CUDA implementation not available")

        scaling = torch.rand(
            10, 3, device=self.device, dtype=torch.double, requires_grad=True
        )

        def compute_scaling_matrix_func(s):
            return ComputeScalingMatrix.apply(s, self.scale_modifier).sum()

        self.assertTrue(
            torch.autograd.gradcheck(compute_scaling_matrix_func, (scaling,))
        )

    def test_matrix_multiply_gradcheck(self):
        if not CUDA_AVAILABLE:
            self.skipTest("CUDA implementation not available")

        A = torch.rand(
            10, 3, 3, device=self.device, dtype=torch.double, requires_grad=True
        )
        B = torch.rand(
            10, 3, 3, device=self.device, dtype=torch.double, requires_grad=True
        )

        def matrix_multiply_func(a, b):
            return MatrixMultiply.apply(a, b).sum()

        self.assertTrue(torch.autograd.gradcheck(matrix_multiply_func, (A, B)))

    def test_covariance_matrix_gradcheck(self):
        if not CUDA_AVAILABLE:
            self.skipTest("CUDA implementation not available")

        RS = torch.rand(
            10, 3, 3, device=self.device, dtype=torch.double, requires_grad=True
        )

        def covariance_matrix_func(rs):
            return CovarianceMatrix.apply(rs).sum()

        self.assertTrue(torch.autograd.gradcheck(covariance_matrix_func, (RS,)))

    def test_project_to_channel_coords_gradcheck(self):
        if not CUDA_AVAILABLE:
            self.skipTest("CUDA implementation not available")

        points = torch.rand(
            10, 3, device=self.device, dtype=torch.double, requires_grad=True
        )
        receiver = torch.rand(3, device=self.device, dtype=torch.double)

        def project_to_channel_coords_func(p):
            distances, _, uv = ProjectToChannelCoords.apply(
                p, receiver, self.num_tx, self.num_rx
            )
            return distances.sum() + uv.sum()

        self.assertTrue(
            torch.autograd.gradcheck(project_to_channel_coords_func, (points,))
        )

    def test_compute_jacobian_gradcheck(self):
        if not CUDA_AVAILABLE:
            self.skipTest("CUDA implementation not available")

        d = torch.rand(
            10, 3, device=self.device, dtype=torch.double, requires_grad=True
        )

        def compute_jacobian_func(disp):
            return ComputeJacobian.apply(disp, self.num_tx, self.num_rx).sum()

        self.assertTrue(torch.autograd.gradcheck(compute_jacobian_func, (d,)))

    def test_project_cov3d_to_cov2d_gradcheck(self):
        if not CUDA_AVAILABLE:
            self.skipTest("CUDA implementation not available")

        cov3d = torch.rand(
            10, 3, 3, device=self.device, dtype=torch.double, requires_grad=True
        )

        cov3d = 0.5 * (cov3d + cov3d.transpose(1, 2))
        jacobian = torch.rand(
            10, 2, 3, device=self.device, dtype=torch.double, requires_grad=True
        )

        def project_cov3d_to_cov2d_func(c3d, j):
            return ProjectCov3dToCov2d.apply(c3d, j).sum()

        self.assertTrue(
            torch.autograd.gradcheck(
                project_cov3d_to_cov2d_func, (cov3d, jacobian), eps=1e-6
            )
        )

    def test_compute_gaussian_influence_gradcheck(self):
        if not CUDA_AVAILABLE:
            self.skipTest("CUDA implementation not available")

        uv = torch.rand(
            10, 2, device=self.device, dtype=torch.double, requires_grad=True
        )
        cov2d = torch.rand(
            10, 2, 2, device=self.device, dtype=torch.double, requires_grad=True
        )

        cov2d = torch.matmul(cov2d, cov2d.transpose(1, 2)) + torch.eye(
            2, device=self.device, dtype=torch.double
        ).unsqueeze(0)

        def compute_gaussian_influence_func(u, c2d):
            return ComputeGaussianInfluence.apply(
                u, c2d, self.num_tx, self.num_rx
            ).sum()

        self.assertTrue(
            torch.autograd.gradcheck(compute_gaussian_influence_func, (uv, cov2d))
        )

    def test_compute_wireless_channel_gradcheck(self):
        if not CUDA_AVAILABLE:
            self.skipTest("CUDA implementation not available")

        attenuation = torch.rand(
            10, 1, device=self.device, dtype=torch.double, requires_grad=True
        )
        phase_rotation = (
            torch.rand(
                10, 1, device=self.device, dtype=torch.double, requires_grad=True
            )
            * 2
            * math.pi
        )
        distances = (
            torch.rand(10, device=self.device, dtype=torch.double, requires_grad=True)
            * 10
            + 1.0
        )

        def compute_wireless_channel_func(a, p, d):
            real, imag = ComputeWirelessChannel.apply(a, p, d, self.wavelength)
            return real.sum() + imag.sum()

        self.assertTrue(
            torch.autograd.gradcheck(
                compute_wireless_channel_func, (attenuation, phase_rotation, distances)
            )
        )

    def test_alpha_blending_gradcheck(self):
        self.skipTest("CUDA implementation not available")

    def test_rasterize_backward(self):
        self.skipTest("CUDA implementation not available")


if __name__ == "__main__":
    unittest.main()

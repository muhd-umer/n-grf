# engine/tests/test_forward.py

import math
import os
import sys
import unittest

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _torch_impl.rasterize import (
    alpha_blending,
    compute_channel,
    compute_gaussian_influence,
    rasterize,
)
from _torch_impl.transforms import compute_jacobian, project_cov3d_to_cov2d


class TestForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_gaussians = 100
        self.num_tx = 16
        self.num_rx = 2
        self.frequency = 2.4e9
        self.wavelength = 299792458.0 / self.frequency
        self.scale_modifier = 1.0

    def test_quaternion_to_rotation(self):
        try:
            import _C

            has_cuda = True
        except ImportError:
            has_cuda = False

        if not has_cuda:
            self.skipTest("CUDA implementation not available")

        from _wrapper import QuaternionToRotation

        quaternions = torch.rand(self.num_gaussians, 4, device=self.device)
        quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)

        w, x, y, z = (
            quaternions[:, 0],
            quaternions[:, 1],
            quaternions[:, 2],
            quaternions[:, 3],
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

    def test_compute_scaling_matrix(self):
        try:
            import _C

            has_cuda = True
        except ImportError:
            has_cuda = False

        if not has_cuda:
            self.skipTest("CUDA implementation not available")

        from _wrapper import ComputeScalingMatrix

        scaling = torch.exp(torch.rand(self.num_gaussians, 3, device=self.device))

        scaled_scaling = scaling * self.scale_modifier
        S_torch = torch.diag_embed(scaled_scaling)

        S_cuda = ComputeScalingMatrix.apply(scaling, self.scale_modifier)

        self.assertTrue(torch.allclose(S_torch, S_cuda, rtol=1e-5, atol=1e-5))

    def test_project_to_channel_coords(self):
        try:
            import _C

            has_cuda = True
        except ImportError:
            has_cuda = False

        if not has_cuda:
            self.skipTest("CUDA implementation not available")

        from _wrapper import ProjectToChannelCoords

        points = torch.rand(self.num_gaussians, 3, device=self.device) * 10
        receiver = torch.rand(3, device=self.device) * 10

        d_torch = points - receiver
        r_torch = torch.sqrt(torch.sum(d_torch**2, dim=1))

        longitude = torch.atan2(d_torch[:, 1], d_torch[:, 0])
        latitude = torch.asin(torch.clamp(d_torch[:, 2] / r_torch, -1.0, 1.0))

        s_x = longitude / math.pi
        s_y = 2.0 * latitude / math.pi

        u = ((s_x + 1.0) / 2.0) * (self.num_tx - 1) + 0.5
        v = ((s_y + 1.0) / 2.0) * (self.num_rx - 1) + 0.5
        uv_torch = torch.stack([u, v], dim=1)

        r_cuda, d_cuda, uv_cuda = ProjectToChannelCoords.apply(
            points, receiver, self.num_tx, self.num_rx
        )

        self.assertTrue(torch.allclose(r_torch, r_cuda, rtol=1e-5, atol=1e-5))
        self.assertTrue(torch.allclose(d_torch, d_cuda, rtol=1e-5, atol=1e-5))
        self.assertTrue(torch.allclose(uv_torch, uv_cuda, rtol=1e-5, atol=1e-5))

    def test_compute_jacobian(self):
        try:
            import _C

            has_cuda = True
        except ImportError:
            has_cuda = False

        if not has_cuda:
            self.skipTest("CUDA implementation not available")

        from _wrapper import ComputeJacobian

        d = torch.rand(self.num_gaussians, 3, device=self.device) * 10

        J_torch = compute_jacobian(d, self.num_tx, self.num_rx)

        J_cuda = ComputeJacobian.apply(d, self.num_tx, self.num_rx)

        self.assertTrue(torch.allclose(J_torch, J_cuda, rtol=1e-5, atol=1e-5))

    def test_project_cov3d_to_cov2d(self):
        try:
            import _C

            has_cuda = True
        except ImportError:
            has_cuda = False

        if not has_cuda:
            self.skipTest("CUDA implementation not available")

        from _wrapper import ProjectCov3dToCov2d

        cov3d_base = torch.rand(self.num_gaussians, 3, 3, device=self.device)
        cov3d = 0.5 * (cov3d_base + cov3d_base.transpose(1, 2))
        jacobian = torch.rand(self.num_gaussians, 2, 3, device=self.device)

        cov2d_torch = project_cov3d_to_cov2d(cov3d, jacobian)

        cov2d_cuda = ProjectCov3dToCov2d.apply(cov3d, jacobian)

        self.assertTrue(torch.allclose(cov2d_torch, cov2d_cuda, rtol=1e-5, atol=1e-5))

    def test_compute_gaussian_influence(self):
        try:
            import _C

            has_cuda = True
        except ImportError:
            has_cuda = False

        if not has_cuda:
            self.skipTest("CUDA implementation not available")

        from _wrapper import ComputeGaussianInfluence

        uv = torch.rand(self.num_gaussians, 2, device=self.device) * self.num_tx

        cov2d_base = torch.rand(self.num_gaussians, 2, 2, device=self.device)
        cov2d = torch.matmul(cov2d_base, cov2d_base.transpose(1, 2))
        cov2d = cov2d + 0.5 * torch.eye(2, device=self.device).unsqueeze(0)

        influences_torch = compute_gaussian_influence(
            uv, cov2d, self.num_tx, self.num_rx
        )

        influences_cuda = ComputeGaussianInfluence.apply(
            uv, cov2d, self.num_tx, self.num_rx
        )

        self.assertTrue(
            torch.allclose(influences_torch, influences_cuda, rtol=1e-5, atol=1e-5)
        )

    def test_compute_channel(self):
        try:
            import _C

            has_cuda = True
        except ImportError:
            has_cuda = False

        if not has_cuda:
            self.skipTest("CUDA implementation not available")

        from _wrapper import ComputeWirelessChannel

        attenuation = torch.rand(self.num_gaussians, 1, device=self.device).contiguous()
        phase_rotation = (
            torch.rand(self.num_gaussians, 1, device=self.device) * 2 * math.pi
        ).contiguous()
        distances = (
            torch.rand(self.num_gaussians, device=self.device) * 10 + 1.0
        ).contiguous()

        real_torch, imag_torch = compute_channel(
            attenuation, phase_rotation, distances, self.wavelength
        )

        real_cuda, imag_cuda = ComputeWirelessChannel.apply(
            attenuation, phase_rotation, distances, self.wavelength
        )

        self.assertTrue(torch.allclose(real_torch, real_cuda, rtol=1e-5, atol=1e-5))
        self.assertTrue(torch.allclose(imag_torch, imag_cuda, rtol=1e-5, atol=1e-5))

    def test_alpha_blending(self):
        try:
            import _C

            has_cuda = True
        except ImportError:
            has_cuda = False

        if not has_cuda:
            self.skipTest("CUDA implementation not available")

        from _wrapper import AlphaBlending

        influences = torch.rand(
            self.num_gaussians, self.num_tx, self.num_rx, device=self.device
        )
        real_contributions = torch.rand(self.num_gaussians, 1, device=self.device)
        imag_contributions = torch.rand(self.num_gaussians, 1, device=self.device)
        opacity = torch.rand(self.num_gaussians, 1, device=self.device) * 0.5

        sort_indices = torch.arange(
            self.num_gaussians, device=self.device, dtype=torch.int32
        )

        cat_channel_torch = alpha_blending(
            influences,
            real_contributions,
            imag_contributions,
            opacity,
            sort_indices,
            self.num_tx,
            self.num_rx,
        )

        cat_channel_cuda = AlphaBlending.apply(
            influences,
            real_contributions,
            imag_contributions,
            opacity,
            sort_indices,
            self.num_tx,
            self.num_rx,
        )

        is_close = torch.allclose(
            cat_channel_torch, cat_channel_cuda, rtol=1e-3, atol=1e-3
        )
        if not is_close:
            max_diff = torch.max(torch.abs(cat_channel_torch - cat_channel_cuda))
            print(f"Maximum difference in alpha_blending test: {max_diff.item()}")

        self.assertTrue(is_close)

    def test_end_to_end_rasterize(self):
        try:
            import _C

            has_cuda = True
        except ImportError:
            has_cuda = False

        if not has_cuda:
            self.skipTest("CUDA implementation not available")

        from _wrapper import rasterize as rasterize_cuda

        points = torch.rand(self.num_gaussians, 3, device=self.device) * 10
        scaling = torch.rand(self.num_gaussians, 3, device=self.device)
        rotation = torch.rand(self.num_gaussians, 4, device=self.device)
        rotation = rotation / torch.norm(rotation, dim=1, keepdim=True)
        attenuation = torch.rand(self.num_gaussians, 1, device=self.device).contiguous()
        phase_rotation = (
            torch.rand(self.num_gaussians, 1, device=self.device) * 2 * math.pi
        ).contiguous()
        opacity = torch.sigmoid(torch.rand(self.num_gaussians, 1, device=self.device))
        receiver = torch.rand(3, device=self.device) * 10
        transmitter = torch.rand(3, device=self.device) * 10

        distances = torch.norm(points - receiver.unsqueeze(0), dim=1)
        sort_indices = torch.argsort(distances).to(dtype=torch.int32)

        import builtins
        import types

        original_sorted = builtins.sorted

        def mock_argsort(tensor):
            return sort_indices

        original_argsort = torch.argsort
        torch.argsort = mock_argsort

        try:

            cat_channel_torch = rasterize(
                points=points,
                scaling=scaling,
                rotation=rotation,
                attenuation=attenuation,
                phase_rotation=phase_rotation,
                opacity=opacity,
                receiver=receiver,
                transmitter=transmitter,
                num_tx=self.num_tx,
                num_rx=self.num_rx,
                frequency=self.frequency,
                scale_modifier=self.scale_modifier,
            )

            cat_channel_cuda = rasterize_cuda(
                points=points,
                scaling=scaling,
                rotation=rotation,
                attenuation=attenuation,
                phase_rotation=phase_rotation,
                opacity=opacity,
                receiver=receiver,
                transmitter=transmitter,
                num_tx=self.num_tx,
                num_rx=self.num_rx,
                frequency=self.frequency,
                scale_modifier=self.scale_modifier,
            )

            is_close = torch.allclose(
                cat_channel_torch, cat_channel_cuda, rtol=1e-3, atol=1e-3
            )
            if not is_close:
                max_diff = torch.max(torch.abs(cat_channel_torch - cat_channel_cuda))
                print(
                    f"Maximum difference in end_to_end_rasterize test: {max_diff.item()}"
                )

            self.assertTrue(is_close)
        finally:

            torch.argsort = original_argsort
            builtins.sorted = original_sorted


if __name__ == "__main__":
    unittest.main()

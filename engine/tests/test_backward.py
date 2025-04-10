import math
import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _torch_impl.rasterize import rasterize


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

    def _check_gradients(
        self, param_torch, param_cuda, param_name, rtol=1e-4, atol=1e-4
    ):
        """Helper function to check if gradients match between PyTorch and CUDA implementations"""
        if param_torch.grad is None or param_cuda.grad is None:
            self.fail(f"{param_name} gradients not computed")

        grad_diff = torch.max(torch.abs(param_torch.grad - param_cuda.grad)).item()
        self.assertTrue(
            torch.allclose(param_torch.grad, param_cuda.grad, rtol=rtol, atol=atol),
            f"{param_name} gradients don't match. Max difference: {grad_diff}",
        )
        return grad_diff

    def test_quaternion_to_rotation_backward(self):
        try:
            import _C

            has_cuda = True
        except ImportError:
            has_cuda = False

        if not has_cuda:
            self.skipTest("CUDA implementation not available")

        from _wrapper import QuaternionToRotation

        class PyTorchQuaternionToRotation(torch.autograd.Function):
            @staticmethod
            def forward(ctx, q_norm):
                w, x, y, z = q_norm[:, 0], q_norm[:, 1], q_norm[:, 2], q_norm[:, 3]

                R = torch.stack(
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

                ctx.save_for_backward(q_norm)
                return R

            @staticmethod
            def backward(ctx, grad_R):
                q_norm = ctx.saved_tensors[0]
                w, x, y, z = q_norm[:, 0], q_norm[:, 1], q_norm[:, 2], q_norm[:, 3]

                grad_w = (
                    -2 * z * grad_R[:, 0, 1]
                    + 2 * y * grad_R[:, 0, 2]
                    + 2 * z * grad_R[:, 1, 0]
                    - 2 * x * grad_R[:, 1, 2]
                    - 2 * y * grad_R[:, 2, 0]
                    + 2 * x * grad_R[:, 2, 1]
                )
                grad_x = (
                    2 * y * grad_R[:, 0, 1]
                    + 2 * z * grad_R[:, 0, 2]
                    + 2 * y * grad_R[:, 1, 0]
                    - 4 * x * grad_R[:, 1, 1]
                    - 2 * w * grad_R[:, 1, 2]
                    + 2 * z * grad_R[:, 2, 0]
                    + 2 * w * grad_R[:, 2, 1]
                    - 4 * x * grad_R[:, 2, 2]
                )
                grad_y = (
                    -4 * y * grad_R[:, 0, 0]
                    + 2 * x * grad_R[:, 0, 1]
                    + 2 * w * grad_R[:, 0, 2]
                    + 2 * x * grad_R[:, 1, 0]
                    + 2 * z * grad_R[:, 1, 2]
                    - 2 * w * grad_R[:, 2, 0]
                    + 2 * z * grad_R[:, 2, 1]
                    - 4 * y * grad_R[:, 2, 2]
                )
                grad_z = (
                    -4 * z * grad_R[:, 0, 0]
                    - 2 * w * grad_R[:, 0, 1]
                    + 2 * x * grad_R[:, 0, 2]
                    + 2 * w * grad_R[:, 1, 0]
                    - 4 * z * grad_R[:, 1, 1]
                    + 2 * y * grad_R[:, 1, 2]
                    + 2 * x * grad_R[:, 2, 0]
                    + 2 * y * grad_R[:, 2, 1]
                )

                grad_q_norm = torch.stack([grad_w, grad_x, grad_y, grad_z], dim=1)
                return grad_q_norm

        q_torch = torch.rand(
            self.num_gaussians, 4, device=self.device, requires_grad=True
        )
        q_cuda = q_torch.clone().detach().requires_grad_(True)

        q_torch_norm = F.normalize(q_torch, dim=1)
        q_cuda_norm = F.normalize(q_cuda, dim=1)

        R_torch = PyTorchQuaternionToRotation.apply(q_torch_norm)
        R_cuda = QuaternionToRotation.apply(q_cuda_norm)

        loss_torch = R_torch.sum()
        loss_cuda = R_cuda.sum()

        loss_torch.backward(torch.ones_like(R_torch).contiguous())
        loss_cuda.backward(torch.ones_like(R_cuda).contiguous())

        self._check_gradients(q_torch, q_cuda, "quaternion")

    def test_compute_wireless_channel_backward(self):
        try:
            import _C

            has_cuda = True
        except ImportError:
            has_cuda = False

        if not has_cuda:
            self.skipTest("CUDA implementation not available")

        from _wrapper import ComputeWirelessChannel

        class PyTorchComputeWirelessChannel(torch.autograd.Function):
            @staticmethod
            def forward(ctx, attenuation, phase_rotation, distance, wavelength):
                PI = 3.14159265358979323846
                path_loss = wavelength / (4.0 * PI * distance.unsqueeze(1))
                phase_shift = -2.0 * PI * distance.unsqueeze(1) / wavelength

                total_attenuation = attenuation * path_loss
                total_phase = phase_rotation + phase_shift

                real_part = total_attenuation * torch.cos(total_phase)
                imag_part = total_attenuation * torch.sin(total_phase)

                ctx.save_for_backward(
                    attenuation,
                    phase_rotation,
                    distance,
                    path_loss,
                    total_phase,
                    total_attenuation,
                )
                ctx.wavelength = wavelength

                return real_part, imag_part

            @staticmethod
            def backward(ctx, grad_real, grad_imag):
                (
                    attenuation,
                    phase_rotation,
                    distance,
                    path_loss,
                    total_phase,
                    total_attenuation,
                ) = ctx.saved_tensors
                wavelength = ctx.wavelength
                PI = 3.14159265358979323846

                grad_attenuation = grad_real * path_loss * torch.cos(
                    total_phase
                ) + grad_imag * path_loss * torch.sin(total_phase)

                grad_phase_rotation = -grad_real * total_attenuation * torch.sin(
                    total_phase
                ) + grad_imag * total_attenuation * torch.cos(total_phase)

                grad_path_loss = -wavelength / (4.0 * PI * distance**2)
                grad_phase_shift = -2.0 * PI / wavelength

                grad_distance = grad_real * (
                    grad_path_loss.unsqueeze(1) * attenuation * torch.cos(total_phase)
                    - total_attenuation * torch.sin(total_phase) * grad_phase_shift
                ) + grad_imag * (
                    grad_path_loss.unsqueeze(1) * attenuation * torch.sin(total_phase)
                    + total_attenuation * torch.cos(total_phase) * grad_phase_shift
                )

                grad_distance = grad_distance.sum(dim=1)

                return grad_attenuation, grad_phase_rotation, grad_distance, None

        attenuation_torch = torch.rand(
            self.num_gaussians, 1, device=self.device, requires_grad=True
        ).contiguous()
        attenuation_cuda = attenuation_torch.clone().detach().requires_grad_(True)

        phase_rotation_torch = (
            (torch.rand(self.num_gaussians, 1, device=self.device) * 2 * math.pi)
            .requires_grad_(True)
            .contiguous()
        )
        phase_rotation_cuda = phase_rotation_torch.clone().detach().requires_grad_(True)

        distances_torch = (
            (torch.rand(self.num_gaussians, device=self.device) * 10 + 1.0)
            .requires_grad_(True)
            .contiguous()
        )
        distances_cuda = distances_torch.clone().detach().requires_grad_(True)

        real_torch, imag_torch = PyTorchComputeWirelessChannel.apply(
            attenuation_torch, phase_rotation_torch, distances_torch, self.wavelength
        )
        real_cuda, imag_cuda = ComputeWirelessChannel.apply(
            attenuation_cuda, phase_rotation_cuda, distances_cuda, self.wavelength
        )

        loss_torch = real_torch.sum() + imag_torch.sum()
        loss_cuda = real_cuda.sum() + imag_cuda.sum()

        loss_torch.backward()
        loss_cuda.backward()

        self._check_gradients(attenuation_torch, attenuation_cuda, "attenuation")
        self._check_gradients(
            phase_rotation_torch, phase_rotation_cuda, "phase_rotation"
        )
        self._check_gradients(distances_torch, distances_cuda, "distances")

    def test_end_to_end_backward(self):
        try:
            import _C

            has_cuda = True
        except ImportError:
            has_cuda = False

        if not has_cuda:
            self.skipTest("CUDA implementation not available")

        from _wrapper import rasterize as rasterize_cuda

        points_torch = (
            torch.rand(self.num_gaussians, 3, device=self.device, requires_grad=True)
            * 10
        )
        points_cuda = points_torch.clone().detach().requires_grad_(True)

        scaling_torch = torch.rand(
            self.num_gaussians, 3, device=self.device, requires_grad=True
        )
        scaling_cuda = scaling_torch.clone().detach().requires_grad_(True)

        rotation_torch = F.normalize(
            torch.rand(self.num_gaussians, 4, device=self.device, requires_grad=True),
            dim=1,
        )
        rotation_cuda = rotation_torch.clone().detach().requires_grad_(True)

        attenuation_torch = torch.rand(
            self.num_gaussians, 1, device=self.device, requires_grad=True
        ).contiguous()
        attenuation_cuda = attenuation_torch.clone().detach().requires_grad_(True)

        phase_rotation_torch = (
            (torch.rand(self.num_gaussians, 1, device=self.device) * 2 * math.pi)
            .requires_grad_(True)
            .contiguous()
        )
        phase_rotation_cuda = phase_rotation_torch.clone().detach().requires_grad_(True)

        opacity_torch = torch.sigmoid(
            torch.rand(self.num_gaussians, 1, device=self.device, requires_grad=True)
        )
        opacity_cuda = opacity_torch.clone().detach().requires_grad_(True)

        receiver = torch.rand(3, device=self.device) * 10
        transmitter = torch.rand(3, device=self.device) * 10

        distances = torch.norm(points_torch - receiver.unsqueeze(0), dim=1)
        sort_indices = torch.argsort(distances).to(dtype=torch.int32)

        import builtins
        import types

        original_sorted = builtins.sorted

        def mock_argsort(tensor):
            return sort_indices

        original_argsort = torch.argsort
        torch.argsort = mock_argsort

        try:

            channel_torch = rasterize(
                points=points_torch,
                scaling=scaling_torch,
                rotation=rotation_torch,
                attenuation=attenuation_torch,
                phase_rotation=phase_rotation_torch,
                opacity=opacity_torch,
                receiver=receiver,
                transmitter=transmitter,
                num_tx=self.num_tx,
                num_rx=self.num_rx,
                frequency=self.frequency,
                scale_modifier=self.scale_modifier,
            )

            channel_cuda = rasterize_cuda(
                points=points_cuda,
                scaling=scaling_cuda,
                rotation=rotation_cuda,
                attenuation=attenuation_cuda,
                phase_rotation=phase_rotation_cuda,
                opacity=opacity_cuda,
                receiver=receiver,
                transmitter=transmitter,
                num_tx=self.num_tx,
                num_rx=self.num_rx,
                frequency=self.frequency,
                scale_modifier=self.scale_modifier,
            )

            target_channel = torch.rand_like(channel_torch)

            loss_torch = torch.nn.functional.mse_loss(channel_torch, target_channel)
            loss_cuda = torch.nn.functional.mse_loss(channel_cuda, target_channel)

            loss_torch.backward()
            loss_cuda.backward()

            points_diff = self._check_gradients(
                points_torch, points_cuda, "points", rtol=1e-3, atol=1e-3
            )
            scaling_diff = self._check_gradients(
                scaling_torch, scaling_cuda, "scaling", rtol=1e-3, atol=1e-3
            )
            rotation_diff = self._check_gradients(
                rotation_torch, rotation_cuda, "rotation", rtol=1e-3, atol=1e-3
            )
            attenuation_diff = self._check_gradients(
                attenuation_torch, attenuation_cuda, "attenuation", rtol=1e-3, atol=1e-3
            )
            phase_rotation_diff = self._check_gradients(
                phase_rotation_torch,
                phase_rotation_cuda,
                "phase_rotation",
                rtol=1e-3,
                atol=1e-3,
            )
            opacity_diff = self._check_gradients(
                opacity_torch, opacity_cuda, "opacity", rtol=1e-3, atol=1e-3
            )

            print("\nMaximum gradient differences:")
            print(f"Points: {points_diff}")
            print(f"Scaling: {scaling_diff}")
            print(f"Rotation: {rotation_diff}")
            print(f"Attenuation: {attenuation_diff}")
            print(f"Phase rotation: {phase_rotation_diff}")
            print(f"Opacity: {opacity_diff}")
        finally:

            torch.argsort = original_argsort
            builtins.sorted = original_sorted


if __name__ == "__main__":
    unittest.main()

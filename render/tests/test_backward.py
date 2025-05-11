# render/tests/test_backward.py

import math
import os
import sys
import unittest
import warnings

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)
RENDER_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RENDER_MODULE_ROOT)

try:
    from render._wrapper import (
        CUDA_AVAILABLE,
        BuildInverseCovariance,
        ComputeSpatialWeight,
        ExponentialScaling,
        NormalizeQuaternion,
        QuaternionToRotationMatrix,
        SigmoidActivation,
        WeightedComplexSum,
    )
    from render._wrapper import render_channel as cuda_render_channel

    if not CUDA_AVAILABLE:

        warnings.warn("CUDA_AVAILABLE is False in _wrapper. Skipping CUDA tests.")
        raise ImportError("CUDA not available via _wrapper.CUDA_AVAILABLE")

except ImportError:
    CUDA_AVAILABLE = False
    warnings.warn(
        "CUDA wrapper (render_ngrf._C) not found or CUDA not available. Skipping CUDA-specific tests.",
        ImportWarning,
    )

    class DummyFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args, **kwargs):

            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                return torch.zeros_like(args[0])
            return torch.tensor(0.0)

        @staticmethod
        def backward(ctx, *args, **kwargs):
            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                return tuple(
                    torch.zeros_like(arg) if isinstance(arg, torch.Tensor) else None
                    for arg in ctx.needs_input_grad
                )
            return None

    NormalizeQuaternion = QuaternionToRotationMatrix = ExponentialScaling = (
        DummyFunction
    )
    SigmoidActivation = BuildInverseCovariance = ComputeSpatialWeight = DummyFunction
    WeightedComplexSum = DummyFunction

    def cuda_render_channel(*args, **kwargs):
        raise unittest.SkipTest("CUDA render_channel not available")


from models.gaussian_model import GaussianRadioFieldModel


class TestBackward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        if CUDA_AVAILABLE and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Testing on CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Testing on CPU device (CUDA tests will be skipped).")
            if not CUDA_AVAILABLE:
                print("CUDA custom extension not built/found for 'render_ngrf._C'.")
            elif not torch.cuda.is_available():
                print(
                    "CUDA custom extension found, but no CUDA device detected by PyTorch."
                )

        self.num_items = 2
        self.dtype = torch.double
        self.gradcheck_eps = 1e-6
        self.gradcheck_atol = 1e-5
        self.gradcheck_rtol = 1e-3
        self.nondet_tol = 1e-7

        self.N_gauss_e2e = 100
        self.Nt_e2e = 2
        self.Nr_e2e = 2
        self.B_e2e = 1
        self.latent_dim_e2e = 8
        self.eps_e2e = 1e-7

    def _run_gradcheck(self, func, inputs, nondet_tol=None):
        if not CUDA_AVAILABLE:
            self.skipTest("CUDA implementation not available")
        if self.device.type == "cpu":
            self.skipTest("Gradcheck for CUDA functions requires a CUDA device.")

        processed_inputs = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                processed_inputs.append(
                    inp.to(device=self.device, dtype=self.dtype).requires_grad_(
                        inp.requires_grad
                    )
                )
            else:
                processed_inputs.append(inp)

        def real_valued_func_wrapper(*args_wrapper):
            output = func(*args_wrapper)
            if isinstance(output, tuple):
                return tuple(
                    torch.view_as_real(o) if o.is_complex() else o for o in output
                )
            elif output.is_complex():
                return torch.view_as_real(output)
            return output

        try:
            test_passed = torch.autograd.gradcheck(
                real_valued_func_wrapper,
                tuple(processed_inputs),
                eps=self.gradcheck_eps,
                atol=self.gradcheck_atol,
                rtol=self.gradcheck_rtol,
                nondet_tol=nondet_tol if nondet_tol is not None else self.nondet_tol,
            )
            self.assertTrue(test_passed)
        except RuntimeError as e:
            self.fail(f"Gradcheck failed with RuntimeError: {e}")

    def test_normalize_quaternion_gradcheck(self):
        q_raw = torch.randn(self.num_items, 4, requires_grad=True)
        eps_norm = 1e-7
        self._run_gradcheck(lambda q: NormalizeQuaternion.apply(q, eps_norm), (q_raw,))

    def test_quaternion_to_rotation_matrix_gradcheck(self):
        q_norm = torch.randn(self.num_items, 4)
        q_norm = F.normalize(q_norm, p=2, dim=1).requires_grad_(True)
        self._run_gradcheck(QuaternionToRotationMatrix.apply, (q_norm,))

    def test_exponential_scaling_gradcheck(self):
        s_log = torch.randn(self.num_items, 3, requires_grad=True)
        self._run_gradcheck(ExponentialScaling.apply, (s_log,))

    def test_sigmoid_activation_gradcheck(self):
        x_logit = torch.randn(self.num_items, 1, requires_grad=True)
        self._run_gradcheck(SigmoidActivation.apply, (x_logit,))

    def test_build_inverse_covariance_gradcheck(self):
        q_for_R = torch.randn(self.num_items, 4)
        q_norm_for_R = F.normalize(q_for_R, p=2, dim=1)

        R_matrices_py = torch.zeros(
            self.num_items, 3, 3, dtype=self.dtype, device=self.device
        )
        for i in range(self.num_items):
            w, x, y, z = q_norm_for_R[i]
            R_matrices_py[i, 0, 0] = 1 - 2 * (y * y + z * z)
            R_matrices_py[i, 0, 1] = 2 * (x * y - w * z)
            R_matrices_py[i, 0, 2] = 2 * (x * z + w * y)
            R_matrices_py[i, 1, 0] = 2 * (x * y + w * z)
            R_matrices_py[i, 1, 1] = 1 - 2 * (x * x + z * z)
            R_matrices_py[i, 1, 2] = 2 * (y * z - w * x)
            R_matrices_py[i, 2, 0] = 2 * (x * z - w * y)
            R_matrices_py[i, 2, 1] = 2 * (y * z + w * x)
            R_matrices_py[i, 2, 2] = 1 - 2 * (x * x + y * y)
        R_matrices_py.requires_grad_(True)

        s_act = (torch.rand(self.num_items, 3) * 0.5 + 0.01).requires_grad_(True)
        eps_clamp = 1e-7
        self._run_gradcheck(
            lambda R, s: BuildInverseCovariance.apply(R, s, eps_clamp),
            (R_matrices_py, s_act),
        )

    def test_compute_spatial_weight_gradcheck(self):
        d_vec = torch.randn(self.num_items, 3, requires_grad=True)

        L_temp = torch.randn(self.num_items, 3, 3)
        cov_temp = L_temp @ L_temp.transpose(1, 2) + torch.eye(3).unsqueeze(0) * 1e-2
        Sigma_inv = torch.inverse(cov_temp).requires_grad_(True)

        alpha = (torch.rand(self.num_items, 1) * 0.8 + 0.1).requires_grad_(True)
        clamp_max = 30.0
        self._run_gradcheck(
            lambda d, Si, a: ComputeSpatialWeight.apply(d, Si, a, clamp_max),
            (d_vec, Sigma_inv, alpha),
        )

    def test_weighted_complex_sum_gradcheck(self):
        B, N, Nt, Nr = 1, self.num_items, 2, 2
        weights = torch.rand(B, N, requires_grad=True)

        contributions_real = torch.randn(N, Nt, Nr, 2, requires_grad=True)

        def func_wrapper(w, c_real_view):
            c_complex = torch.view_as_complex(c_real_view)
            output_complex = WeightedComplexSum.apply(w, c_complex)
            return torch.view_as_real(output_complex)

        self._run_gradcheck(func_wrapper, (weights, contributions_real))

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA implementation not available")
    def test_render_channel_end_to_end_gradcheck(self):
        if self.device.type == "cpu":
            self.skipTest("End-to-end render_channel gradcheck requires a CUDA device.")

        model = GaussianRadioFieldModel(
            num_tx_ant=self.Nt_e2e,
            num_rx_ant=self.Nr_e2e,
            latent_dim=self.latent_dim_e2e,
            attribute_hidden_dim=16,
            attribute_num_layers=2,
            attribute_pos_enc_freqs=4,
            decoder_hidden_dim=16,
            decoder_num_layers=2,
            initial_gaussians=self.N_gauss_e2e,
            device=self.device,
        )

        env_dims = torch.tensor([[-2.0, 2.0]] * 3, dtype=self.dtype, device=self.device)
        model.init_gaussians(env_dims=env_dims, num_points=self.N_gauss_e2e)
        model = model.to(dtype=self.dtype, device=self.device)

        model._xyz.requires_grad_(True)
        model._rotation.requires_grad_(True)
        model._scaling.requires_grad_(True)
        for param in model.attribute_network.parameters():
            param.requires_grad_(True)
        for param in model.contribution_decoder.parameters():
            param.requires_grad_(True)

        rx_positions = torch.randn(self.B_e2e, 3, dtype=self.dtype, device=self.device)
        tx_position = torch.randn(3, dtype=self.dtype, device=self.device)

        inputs_for_gradcheck = (
            model._xyz,
            model._rotation,
            model._scaling,
            *model.attribute_network.parameters(),
            *model.contribution_decoder.parameters(),
        )

        inputs_for_gradcheck = tuple(
            p for p in inputs_for_gradcheck if isinstance(p, torch.Tensor)
        )

        def func_to_check(*params_tuple):
            _xyz, _rot, _scl = params_tuple[0], params_tuple[1], params_tuple[2]

            output_complex = cuda_render_channel(
                rx_positions, model, tx_position, self.Nt_e2e, self.Nr_e2e, self.eps_e2e
            )
            return torch.view_as_real(output_complex)

        print("Checking end-to-end cuda_render_channel backward...")

        try:
            test_passed = torch.autograd.gradcheck(
                func_to_check,
                inputs_for_gradcheck,
                eps=1e-4,
                atol=1e-3,
                rtol=1e-2,
                nondet_tol=1e-4,
            )
            self.assertTrue(test_passed)
        except RuntimeError as e:
            self.fail(f"End-to-end gradcheck failed with RuntimeError: {e}")


if __name__ == "__main__":
    unittest.main()

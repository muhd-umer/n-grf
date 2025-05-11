# render/tests/test_forward.py

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
    from render._wrapper import CUDA_AVAILABLE
    from render._wrapper import render_channel as cuda_render_channel

    if not CUDA_AVAILABLE:
        warnings.warn(
            "CUDA_AVAILABLE is False in _wrapper. CUDA render_channel may not work.",
            ImportWarning,
        )

except ImportError:
    CUDA_AVAILABLE = False
    warnings.warn(
        "CUDA wrapper (render_ngrf._C) not found. CUDA render_channel will not be tested.",
        ImportWarning,
    )

    def cuda_render_channel(*args, **kwargs):
        raise unittest.SkipTest("CUDA render_channel not available")


from models.gaussian_model import GaussianRadioFieldModel
from render._torch_impl import render_channel as torch_render_channel


class TestForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        if CUDA_AVAILABLE and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Testing on CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print(
                "Testing on CPU device (CUDA tests will be skipped for CUDA functions)."
            )

        self.N_gauss = 50
        self.Nt = 4
        self.Nr = 2
        self.B = 3
        self.latent_dim = 32
        self.eps = 1e-7
        self.dtype = torch.double

        self.xyz_raw = torch.randn(self.N_gauss, 3) * 5.0
        self.rot_raw = torch.randn(self.N_gauss, 4)
        self.rot_raw = F.normalize(self.rot_raw, p=2, dim=1)
        self.scl_log_raw = torch.randn(self.N_gauss, 3) * 0.5 - 2.0

        self.rx_positions = torch.randn(self.B, 3) * 10.0
        self.tx_position = torch.randn(3) * 2.0

    def _create_model_instance(self):
        model = GaussianRadioFieldModel(
            num_tx_ant=self.Nt,
            num_rx_ant=self.Nr,
            latent_dim=self.latent_dim,
            attribute_hidden_dim=32,
            attribute_num_layers=2,
            attribute_pos_enc_freqs=5,
            decoder_hidden_dim=32,
            decoder_num_layers=2,
            initial_gaussians=self.N_gauss,
            device=self.device,
        )

        model._xyz = torch.nn.Parameter(
            self.xyz_raw.clone().to(self.device, self.dtype)
        )
        model._rotation = torch.nn.Parameter(
            self.rot_raw.clone().to(self.device, self.dtype)
        )
        model._scaling = torch.nn.Parameter(
            self.scl_log_raw.clone().to(self.device, self.dtype)
        )

        model.attribute_network = model.attribute_network.to(self.device, self.dtype)
        model.contribution_decoder = model.contribution_decoder.to(
            self.device, self.dtype
        )
        return model

    def test_render_channel_forward_consistency(self):
        if not CUDA_AVAILABLE or self.device.type == "cpu":
            self.skipTest(
                "CUDA implementation not available or testing on CPU, skipping forward consistency test."
            )

        model_torch = self._create_model_instance()
        model_cuda = self._create_model_instance()

        model_cuda.load_state_dict(model_torch.state_dict())

        model_torch.eval()
        model_cuda.eval()

        rx_pos_dev = self.rx_positions.to(self.device, self.dtype)
        tx_pos_dev = self.tx_position.to(self.device, self.dtype)

        with torch.no_grad():
            output_torch = torch_render_channel(
                rx_positions=rx_pos_dev,
                model=model_torch,
                tx_position=tx_pos_dev,
                nt=self.Nt,
                nr=self.Nr,
                eps=self.eps,
            )

        with torch.no_grad():
            output_cuda = cuda_render_channel(
                rx_positions=rx_pos_dev,
                model=model_cuda,
                tx_position=tx_pos_dev,
                nt=self.Nt,
                nr=self.Nr,
                eps=self.eps,
            )

        self.assertEqual(output_torch.shape, output_cuda.shape)
        self.assertEqual(output_torch.dtype, output_cuda.dtype)

        rtol = 1e-5
        atol = 1e-7

        if not torch.allclose(output_torch, output_cuda, rtol=rtol, atol=atol):
            abs_diff = torch.abs(output_torch - output_cuda)
            max_diff_val = torch.max(abs_diff)
            mean_diff_val = torch.mean(abs_diff)
            print(f"Max absolute difference: {max_diff_val.item()}")
            print(f"Mean absolute difference: {mean_diff_val.item()}")

        self.assertTrue(
            torch.allclose(output_torch, output_cuda, rtol=rtol, atol=atol),
            "Forward pass outputs of torch_render_channel and cuda_render_channel do not match.",
        )
        print("Forward pass consistency test passed.")


if __name__ == "__main__":
    unittest.main()

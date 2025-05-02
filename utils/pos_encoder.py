# utils/pos_encoder.py

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    """Sine-cosine positional encoder for input points."""

    def __init__(
        self,
        input_dims: int,
        num_freqs: int,
        include_input: bool = True,
        log_sampling: bool = True,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.output_dims = 0
        self.embedding_fns = []
        self._create_embedding_fn()

    def _create_embedding_fn(self):
        """Create the embedding functions."""
        if self.include_input:
            self.embedding_fns.append(lambda x: x)
            self.output_dims += self.input_dims

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(
                0.0, self.num_freqs - 1, steps=self.num_freqs
            )
        else:
            freq_bands = torch.linspace(
                1.0, 2.0 ** (self.num_freqs - 1), steps=self.num_freqs
            )

        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                self.embedding_fns.append(
                    lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq)
                )
                self.output_dims += self.input_dims

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding.
        Args:
            inputs: Input tensor (..., input_dims)
        Returns:
            Encoded tensor (..., output_dims)
        """
        encoded = torch.cat([fn(inputs) for fn in self.embedding_fns], dim=-1)
        return encoded

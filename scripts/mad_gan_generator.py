#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mad_gan_generator.py
------------------------------------
MAD-GAN Style Generator for Multivariate Anomaly Detection
(Feature-Vector / Time-Series Friendly)

Author: William K
Affiliation: University of Vienna
------------------------------------

This module implements a PyTorch-based Generator suitable for
Multivariate Anomaly Detection GANs (MAD-GAN-style) in the AI-DAC /
Triple Loop Learning (TLL) pipeline.

It is intentionally generic so you can use it for:

  - Fixed-length feature vectors (e.g. SQL log embeddings)
  - Sliding-window time-series segments (flattened or [B, T, D])

Key design points:
  - Latent input z ~ N(0, I) or U(-1, 1)
  - Fully-connected backbone (MLP) for robustness
  - Optional reshape to (B, seq_len, feature_dim)
  - Weight initialization helper (GAN-friendly)

Typical integration:
  - Use alongside `mad_gan_discriminator.py`
  - Plug into MAD-GAN training loop for anomaly detection
  - Feed generated samples to AI-DAC / TLL evaluation pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


# ---------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------

@dataclass
class MadGanGenConfig:
    """
    Configuration for MAD-GAN Generator.

    Parameters
    ----------
    latent_dim : int
        Dimension of the noise vector z.
    out_dim : int
        Dimension of the output vector.
        If you want (seq_len, feat_dim), then out_dim = seq_len * feat_dim.
    hidden_dim : int
        Hidden layer width.
    num_layers : int
        Number of linear blocks (min 2 → input + output).
    use_batchnorm : bool
        Whether to use BatchNorm1d between layers.
    final_activation : str
        "tanh", "sigmoid" or "none".
    """
    latent_dim: int = 64
    out_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 3
    use_batchnorm: bool = True
    final_activation: str = "tanh"  # "tanh" | "sigmoid" | "none"


# ---------------------------------------------------------------------
# Generator Network
# ---------------------------------------------------------------------

class MADGANGenerator(nn.Module):
    """
    Multilayer Perceptron-style generator for MAD-GAN.

    Input:
      - z : tensor of shape (batch, latent_dim)

    Output:
      - x_hat : tensor of shape (batch, out_dim)

    The output can be reshaped outside to (batch, seq_len, feat_dim)
    if needed by your MAD-GAN setup.
    """

    def __init__(self, cfg: MadGanGenConfig):
        super().__init__()
        self.cfg = cfg

        layers = []
        in_dim = cfg.latent_dim

        # Hidden layers
        for i in range(cfg.num_layers - 1):
            layers.append(nn.Linear(in_dim, cfg.hidden_dim))
            if cfg.use_batchnorm:
                layers.append(nn.BatchNorm1d(cfg.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = cfg.hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, cfg.out_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

        if cfg.final_activation not in ("tanh", "sigmoid", "none"):
            raise ValueError(f"Unsupported final_activation: {cfg.final_activation}")

    def _init_weights(self) -> None:
        """
        Initialize weights with a GAN-friendly scheme (xavier uniform).
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: z → x_hat

        Parameters
        ----------
        z : tensor, shape (batch_size, latent_dim)

        Returns
        -------
        x_hat : tensor, shape (batch_size, out_dim)
        """
        x = self.net(z)

        if self.cfg.final_activation == "tanh":
            x = torch.tanh(x)
        elif self.cfg.final_activation == "sigmoid":
            x = torch.sigmoid(x)
        # "none": leave x as-is

        return x

    @staticmethod
    def sample_noise(
        batch_size: int,
        latent_dim: int,
        device: Optional[str] = None,
        distribution: str = "normal",
    ) -> torch.Tensor:
        """
        Sample latent noise vectors.

        Parameters
        ----------
        batch_size : int
        latent_dim : int
        device     : 'cpu' or 'cuda' (optional)
        distribution : 'normal' | 'uniform'

        Returns
        -------
        z : tensor, shape (batch_size, latent_dim)
        """
        dev = torch.device(device) if device is not None else torch.device("cpu")

        if distribution == "normal":
            z = torch.randn(batch_size, latent_dim, device=dev)
        elif distribution == "uniform":
            z = torch.empty(batch_size, latent_dim, device=dev).uniform_(-1.0, 1.0)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

        return z


# ---------------------------------------------------------------------
# Optional helper: reshape to (B, T, D)
# ---------------------------------------------------------------------

def reshape_to_sequence(
    x: torch.Tensor,
    seq_len: int,
    feat_dim: int,
) -> torch.Tensor:
    """
    Helper to reshape generator output into (batch, seq_len, feat_dim).

    Parameters
    ----------
    x : tensor of shape (batch, out_dim)
    seq_len : int
        Number of time steps.
    feat_dim : int
        Feature dimension.

    Returns
    -------
    x_seq : tensor of shape (batch, seq_len, feat_dim)

    Notes
    -----
    Ensure `out_dim == seq_len * feat_dim`.
    """
    b, out_dim = x.shape
    if out_dim != seq_len * feat_dim:
        raise ValueError(
            f"reshape_to_sequence: out_dim={out_dim} != seq_len*feat_dim={seq_len * feat_dim}"
        )
    return x.view(b, seq_len, feat_dim)


# ---------------------------------------------------------------------
# Demo / Self-Test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal self-test for MADGANGenerator.

    - Instantiates a generator with default config
    - Samples random latent vectors
    - Produces synthetic sequences for a hypothetical setup:
          seq_len = 16
          feat_dim = 8
          out_dim = seq_len * feat_dim = 128
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seq_len = 16
    feat_dim = 8
    out_dim = seq_len * feat_dim

    cfg = MadGanGenConfig(
        latent_dim=64,
        out_dim=out_dim,
        hidden_dim=256,
        num_layers=3,
        use_batchnorm=True,
        final_activation="tanh",
    )

    gen = MADGANGenerator(cfg).to(device)
    gen.eval()

    batch_size = 32
    z = MADGANGenerator.sample_noise(
        batch_size=batch_size,
        latent_dim=cfg.latent_dim,
        device=device,
        distribution="normal",
    )

    with torch.no_grad():
        x_hat = gen(z)
        x_seq = reshape_to_sequence(x_hat, seq_len=seq_len, feat_dim=feat_dim)

    print("=== MAD-GAN Generator Self-Test ===")
    print(f"Device      : {device}")
    print(f"Latent shape: {tuple(z.shape)}")
    print(f"Output vec  : {tuple(x_hat.shape)}")
    print(f"Output seq  : {tuple(x_seq.shape)}  (B, T, D)")
    print("Self-test completed.")

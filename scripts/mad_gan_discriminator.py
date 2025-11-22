#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mad_gan_discriminator.py
------------------------------------
MAD-GAN Style Discriminator for Multivariate Anomaly Detection
(Feature-Vector / Time-Series Friendly)

Author: William K
Affiliation: University of Vienna
------------------------------------

This module implements a PyTorch-based Discriminator suitable for
Multivariate Anomaly Detection GANs (MAD-GAN-style) in the AI-DAC /
Triple Loop Learning (TLL) pipeline.

It is intentionally generic so you can use it for:

  - Fixed-length feature vectors (e.g. SQL log embeddings)
  - Sliding-window time-series segments (flattened or [B, T, D])

Design choices:
  - MLP backbone, stable for tabular / embedding data
  - Optional spectral normalization for GAN stability
  - Output is either:
      * single logit (real/fake)  → out_dim = 1 (default), or
      * multi-class logits        → out_dim > 1  (e.g. normal/anomaly)
  - Helper methods:
      * forward(x)      – raw logits
      * score(x)        – sigmoid probability (for out_dim=1)
      * flatten_sequence(x) – [B, T, D] → [B, T*D]

Typical integration:
  - Paired with MADGANGenerator from `mad_gan_generator.py`
  - Used in MAD-GAN training loop to produce discrimination scores
  - Output scores can be used by AI-DAC / TLL for anomaly scores
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
class MadGanDiscConfig:
    """
    Configuration for MAD-GAN Discriminator.

    Parameters
    ----------
    in_dim : int
        Input dimension. If using sequences (B, T, D) you usually
        set in_dim = T * D after flattening.
    hidden_dim : int
        Hidden layer width.
    num_layers : int
        Total number of linear blocks (≥ 2 → input + output).
    out_dim : int
        Output dimension:
           - 1 → single real/fake logit
           - 2 → binary classifier (normal / anomaly)
           - C → C-way classifier
    use_batchnorm : bool
        Whether to use BatchNorm1d between layers.
    use_spectral_norm : bool
        Wrap Linear layers with spectral normalization (helps GAN training).
    """

    in_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 3
    out_dim: int = 1
    use_batchnorm: bool = True
    use_spectral_norm: bool = True


# ---------------------------------------------------------------------
# Discriminator Network
# ---------------------------------------------------------------------

class MADGANDiscriminator(nn.Module):
    """
    Multilayer Perceptron-style discriminator for MAD-GAN.

    Input:
      - x : tensor of shape (batch, in_dim)  (flattened)
        or (batch, T, D) if flattened externally.

    Output:
      - logits : tensor of shape (batch, out_dim)

    Notes:
      - For standard GAN use out_dim=1 and apply sigmoid to logits.
      - For anomaly classification use out_dim=2 + CrossEntropyLoss.
    """

    def __init__(self, cfg: MadGanDiscConfig):
        super().__init__()
        self.cfg = cfg

        layers = []
        in_dim = cfg.in_dim

        # Hidden layers
        for i in range(cfg.num_layers - 1):
            lin = nn.Linear(in_dim, cfg.hidden_dim)
            if cfg.use_spectral_norm:
                lin = nn.utils.spectral_norm(lin)
            layers.append(lin)

            if cfg.use_batchnorm:
                layers.append(nn.BatchNorm1d(cfg.hidden_dim))

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = cfg.hidden_dim

        # Output layer (no activation here; caller decides)
        out_lin = nn.Linear(in_dim, cfg.out_dim)
        if cfg.use_spectral_norm:
            out_lin = nn.utils.spectral_norm(out_lin)
        layers.append(out_lin)

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize weights (xavier for linear layers).
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : tensor, shape (B, in_dim) or (B, T, D) if pre-flattened.

        Returns
        -------
        logits : tensor, shape (B, out_dim)
        """
        if x.ndim == 3:
            # (B, T, D) → (B, T*D)
            x = x.view(x.size(0), -1)
        return self.net(x)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to probabilities (for out_dim=1 case).

        Returns
        -------
        probs : tensor, shape (B,)
            Values in (0,1); higher ~ more 'real' / less anomalous.
        """
        logits = self.forward(x)
        if self.cfg.out_dim != 1:
            raise ValueError("score() is only meaningful when out_dim == 1.")
        return torch.sigmoid(logits.view(-1))


# ---------------------------------------------------------------------
# Helper: flatten sequences
# ---------------------------------------------------------------------

def flatten_sequence(x: torch.Tensor) -> torch.Tensor:
    """
    Flatten (B, T, D) → (B, T*D) for feeding into MLP discriminator.

    Parameters
    ----------
    x : tensor, shape (B, T, D)

    Returns
    -------
    x_flat : tensor, shape (B, T*D)
    """
    if x.ndim != 3:
        raise ValueError(f"flatten_sequence: expected 3D tensor, got shape {tuple(x.shape)}")
    b, t, d = x.shape
    return x.view(b, t * d)


# ---------------------------------------------------------------------
# Demo / Self-Test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal self-test for MADGANDiscriminator.

    Scenario:
      - Sequence length T=16, feature dim D=8 → in_dim = 128
      - Discriminator with out_dim=1 (GAN-style real/fake logit)
      - Evaluates random real and fake samples.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seq_len = 16
    feat_dim = 8
    in_dim = seq_len * feat_dim

    cfg = MadGanDiscConfig(
        in_dim=in_dim,
        hidden_dim=256,
        num_layers=3,
        out_dim=1,
        use_batchnorm=True,
        use_spectral_norm=True,
    )

    disc = MADGANDiscriminator(cfg).to(device)
    disc.eval()

    batch_size = 32

    # Fake "real" data
    x_real = torch.randn(batch_size, seq_len, feat_dim, device=device)
    # Fake "generated" data
    x_fake = torch.randn(batch_size, seq_len, feat_dim, device=device) + 0.5

    with torch.no_grad():
        logits_real = disc(x_real)
        logits_fake = disc(x_fake)
        scores_real = torch.sigmoid(logits_real.view(-1))
        scores_fake = torch.sigmoid(logits_fake.view(-1))

    print("=== MAD-GAN Discriminator Self-Test ===")
    print(f"Device          : {device}")
    print(f"Real logits     : shape={tuple(logits_real.shape)}")
    print(f"Fake logits     : shape={tuple(logits_fake.shape)}")
    print(f"Mean score(real): {scores_real.mean().item():.4f}")
    print(f"Mean score(fake): {scores_fake.mean().item():.4f}")
    print("Self-test completed.")

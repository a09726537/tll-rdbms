#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mad_gan_train.py
------------------------------------
MAD-GAN Training Script for Multivariate Anomaly Detection
(Feature-Vector / Time-Series Friendly)

Author: William K
Affiliation: University of Vienna
------------------------------------

This script trains a MAD-GAN-style model using:

  - Generator      : MADGANGenerator (from mad_gan_generator.py)
  - Discriminator  : MADGANDiscriminator (from mad_gan_discriminator.py)

Design:
  - Default training data = synthetic "normal" sequences
      * seq_len x feat_dim, drawn from N(0,1)
  - GAN objective:
      * Discriminator: maximize log D(x_real) + log(1 - D(x_fake))
      * Generator    : maximize log D(x_fake)  (i.e. fool discriminator)
  - Anomaly detection is performed downstream using:
      * reconstruction / discrimination scores on real SQL sequences.

Integration with AI-DAC / TLL:
  - This script can be used offline to train the MAD-GAN on "normal" logs.
  - Checkpoints can be loaded by the AI-DAC runtime for:
      * anomaly scoring,
      * robustness tests,
      * RL / meta-learning loops.

Usage (basic):
  python mad_gan_train.py \
        --seq-len 16 \
        --feat-dim 8 \
        --epochs 50 \
        --batch-size 128 \
        --gen-lr 0.0002 \
        --disc-lr 0.0002

Checkpoints:
  - Generator:   madgan_generator.pt
  - Discriminator: madgan_discriminator.pt
  (paths can be changed via CLI)
"""

import argparse
import os
import time
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mad_gan_generator import MADGANGenerator, MadGanGenConfig, reshape_to_sequence
from mad_gan_discriminator import MADGANDiscriminator, MadGanDiscConfig, flatten_sequence


# ---------------------------------------------------------------------
# Synthetic dataset (default)
# ---------------------------------------------------------------------


def make_synthetic_normal_dataset(
    n_samples: int,
    seq_len: int,
    feat_dim: int,
) -> torch.Tensor:
    """
    Create a synthetic "normal" dataset for demo / baseline training.

    Data: multivariate Gaussian sequences (B, T, D), mean 0, std 1.
    In real use, replace this with real SQL / system log windows.
    """
    x = torch.randn(n_samples, seq_len, feat_dim)
    return x


# ---------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------


def build_models(
    seq_len: int,
    feat_dim: int,
    latent_dim: int,
    hidden_dim: int,
    num_layers: int,
    disc_out_dim: int,
    device: str,
) -> Tuple[MADGANGenerator, MADGANDiscriminator]:
    """
    Build generator & discriminator with compatible dimensions.
    """
    out_dim = seq_len * feat_dim

    gen_cfg = MadGanGenConfig(
        latent_dim=latent_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_batchnorm=True,
        final_activation="tanh",
    )
    disc_cfg = MadGanDiscConfig(
        in_dim=out_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        out_dim=disc_out_dim,
        use_batchnorm=True,
        use_spectral_norm=True,
    )

    G = MADGANGenerator(gen_cfg).to(device)
    D = MADGANDiscriminator(disc_cfg).to(device)

    return G, D


def gan_losses(
    D: MADGANDiscriminator,
    real_flat: torch.Tensor,
    fake_flat: torch.Tensor,
    device: str,
):
    """
    Compute standard non-saturating GAN losses for discriminator & generator.

    - D_loss = BCE(D(real), 1) + BCE(D(fake), 0)
    - G_loss = BCE(D(fake), 1)
    """
    bce = nn.BCEWithLogitsLoss()

    # Real labels = 1, fake labels = 0
    real_labels = torch.ones(real_flat.size(0), 1, device=device)
    fake_labels = torch.zeros(fake_flat.size(0), 1, device=device)

    # Forward through D
    logits_real = D(real_flat)
    logits_fake = D(fake_flat)

    # Ensure shape (B,1) for out_dim=1; otherwise broadcast
    if logits_real.ndim == 1:
        logits_real = logits_real.view(-1, 1)
    if logits_fake.ndim == 1:
        logits_fake = logits_fake.view(-1, 1)

    d_loss_real = bce(logits_real, real_labels)
    d_loss_fake = bce(logits_fake, fake_labels)
    d_loss = d_loss_real + d_loss_fake

    # Generator tries to make fake look "real"
    g_loss = bce(logits_fake, real_labels)

    return d_loss, g_loss, logits_real.detach(), logits_fake.detach()


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------


def train_mad_gan(
    seq_len: int,
    feat_dim: int,
    latent_dim: int,
    hidden_dim: int,
    num_layers: int,
    disc_out_dim: int,
    n_samples: int,
    batch_size: int,
    epochs: int,
    gen_lr: float,
    disc_lr: float,
    beta1: float,
    beta2: float,
    device: str,
    out_dir: str,
    save_every: int = 10,
):
    os.makedirs(out_dir, exist_ok=True)

    print("=== MAD-GAN Training Configuration ===")
    print(f"Device       : {device}")
    print(f"Seq length   : {seq_len}")
    print(f"Feat dim     : {feat_dim}")
    print(f"Latent dim   : {latent_dim}")
    print(f"Hidden dim   : {hidden_dim}")
    print(f"Num layers   : {num_layers}")
    print(f"Disc out_dim : {disc_out_dim}")
    print(f"Samples      : {n_samples}")
    print(f"Batch size   : {batch_size}")
    print(f"Epochs       : {epochs}")
    print(f"G lr / D lr  : {gen_lr} / {disc_lr}")
    print(f"Adam betas   : ({beta1}, {beta2})")
    print(f"Output dir   : {out_dir}")
    print("======================================\n")

    # Build models
    G, D = build_models(
        seq_len=seq_len,
        feat_dim=feat_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        disc_out_dim=disc_out_dim,
        device=device,
    )

    # Optimizers
    opt_G = torch.optim.Adam(G.parameters(), lr=gen_lr, betas=(beta1, beta2))
    opt_D = torch.optim.Adam(D.parameters(), lr=disc_lr, betas=(beta1, beta2))

    # Synthetic training data (replace with real logs in production)
    x_train = make_synthetic_normal_dataset(
        n_samples=n_samples,
        seq_len=seq_len,
        feat_dim=feat_dim,
    )

    dataset = TensorDataset(x_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    G.train()
    D.train()

    for epoch in range(1, epochs + 1):
        epoch_d_loss = []
        epoch_g_loss = []
        epoch_real_score = []
        epoch_fake_score = []

        t0 = time.time()

        for (real_seq,) in loader:
            real_seq = real_seq.to(device)

            # Flatten real sequences: (B, T*D)
            real_flat = flatten_sequence(real_seq)

            # -------------------------
            # 1) Train Discriminator
            # -------------------------
            opt_D.zero_grad()

            # Sample noise and generate fake sequences
            z = MADGANGenerator.sample_noise(
                batch_size=real_seq.size(0),
                latent_dim=latent_dim,
                device=device,
                distribution="normal",
            )
            fake_flat = G(z)  # shape (B, T*D)

            d_loss, g_loss_fake, logits_real, logits_fake = gan_losses(
                D, real_flat, fake_flat.detach(), device=device
            )

            d_loss.backward()
            opt_D.step()

            # -------------------------
            # 2) Train Generator
            # -------------------------
            opt_G.zero_grad()

            # Recompute fake with new G params
            z2 = MADGANGenerator.sample_noise(
                batch_size=real_seq.size(0),
                latent_dim=latent_dim,
                device=device,
                distribution="normal",
            )
            fake_flat2 = G(z2)

            # Generator loss (D wants to classify fake as real)
            _, g_loss, _, logits_fake2 = gan_losses(
                D, real_flat, fake_flat2, device=device
            )

            g_loss.backward()
            opt_G.step()

            # Metrics: mean sigmoid score
            with torch.no_grad():
                s_real = torch.sigmoid(logits_real).mean().item()
                s_fake = torch.sigmoid(logits_fake).mean().item()

            epoch_d_loss.append(d_loss.item())
            epoch_g_loss.append(g_loss.item())
            epoch_real_score.append(s_real)
            epoch_fake_score.append(s_fake)

        t1 = time.time()
        d_mean = float(np.mean(epoch_d_loss))
        g_mean = float(np.mean(epoch_g_loss))
        real_mean = float(np.mean(epoch_real_score))
        fake_mean = float(np.mean(epoch_fake_score))

        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"D_loss={d_mean:.4f} G_loss={g_mean:.4f} "
            f"score_real={real_mean:.3f} score_fake={fake_mean:.3f} "
            f"({t1 - t0:.1f}s)"
        )

        # Periodically save checkpoints
        if epoch % save_every == 0 or epoch == epochs:
            gen_path = os.path.join(out_dir, "madgan_generator.pt")
            disc_path = os.path.join(out_dir, "madgan_discriminator.pt")

            torch.save(
                {
                    "model_state_dict": G.state_dict(),
                    "seq_len": seq_len,
                    "feat_dim": feat_dim,
                    "latent_dim": latent_dim,
                    "config": G.cfg.__dict__,
                },
                gen_path,
            )
            torch.save(
                {
                    "model_state_dict": D.state_dict(),
                    "seq_len": seq_len,
                    "feat_dim": feat_dim,
                    "config": D.cfg.__dict__,
                },
                disc_path,
            )

            print(f"  -> Saved generator to   {gen_path}")
            print(f"  -> Saved discriminator to {disc_path}")

    print("\n[MAD-GAN] Training complete.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="MAD-GAN Trainer for Multivariate Anomaly Detection (AI-DAC / TLL)")

    # Data / shape
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length (time steps) T")
    parser.add_argument("--feat-dim", type=int, default=8, help="Feature dimension D")
    parser.add_argument("--n-samples", type=int, default=20000, help="Number of synthetic normal samples")

    # Model
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent noise dimension")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer width")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of linear blocks (>=2)")
    parser.add_argument("--disc-out-dim", type=int, default=1, help="Discriminator output dim (1 for real/fake)")

    # Training
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--gen-lr", type=float, default=2e-4, help="Generator learning rate")
    parser.add_argument("--disc-lr", type=float, default=2e-4, help="Discriminator learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: 'cuda' or 'cpu'",
    )

    # Output
    parser.add_argument(
        "--out-dir",
        type=str,
        default="models",
        help="Directory to save checkpoints (generator / discriminator).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs (and at the end).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train_mad_gan(
        seq_len=args.seq_len,
        feat_dim=args.feat_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        disc_out_dim=args.disc_out_dim,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        gen_lr=args.gen_lr,
        disc_lr=args.disc_lr,
        beta1=args.beta1,
        beta2=args.beta2,
        device=args.device,
        out_dir=args.out_dir,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()

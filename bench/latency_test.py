#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
latency_test.py
------------------------------------
Inference Latency Benchmark Script
for AI-DAC / Triple Loop Learning (TLL)
(Models: AE / Classifier / Discriminator / RL Policy, etc.)

Author: William K
Affiliation: University of Vienna
------------------------------------

This script measures end-to-end inference latency for a given
PyTorch model, with configurable:

 - Batch size
 - Input dimension
 - Number of warmup runs
 - Number of measured runs
 - Device (CPU / CUDA)
 - Mixed-precision (AMP) support

Usage (with the built-in DummyModel):

    python latency_test.py \
        --input-dim 128 \
        --batch-size 256 \
        --runs 200 \
        --warmup 50 \
        --device cuda

To adapt it to your AI-DAC pipeline:
    - Replace `DummyModel` with your actual model
    - Or, load a state_dict / torchscript module
    - Or, wrap your anomaly detection module (AE, classifier, GAN-D)

This script focuses on *pure model inference latency*.
"""

import argparse
import statistics
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Model Definition (replace with your own)
# ---------------------------------------------------------------------

class DummyModel(nn.Module):
    """
    Simple feed-forward network used as a placeholder.
    Replace this with your AI-DAC model component, e.g.:

        - Autoencoder (AE)
        - Classifier head
        - MAD-GAN discriminator
        - RL policy network

    The ONLY requirement is that forward(x) accepts a tensor of shape:
        (batch_size, input_dim)
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------
# Latency Measurement Logic
# ---------------------------------------------------------------------

def measure_latency(
    model: nn.Module,
    input_shape: tuple,
    device: str = "cpu",
    warmup_runs: int = 50,
    measured_runs: int = 200,
    use_amp: bool = False,
    use_eval_mode: bool = True,
) -> dict:
    """
    Measures per-batch inference latency for a given model.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to benchmark.
    input_shape : tuple
        Shape of the input tensor (batch_size, input_dim, ...) as a tuple.
    device : str
        'cpu' or 'cuda'.
    warmup_runs : int
        Number of warmup forward passes (not measured).
    measured_runs : int
        Number of measured forward passes.
    use_amp : bool
        Whether to use autocast (mixed precision) on CUDA.
    use_eval_mode : bool
        If True, uses model.eval() and disables grad.

    Returns
    -------
    stats : dict
        Dictionary with latency statistics (ms).
    """
    device = torch.device(device)
    model.to(device)

    if use_eval_mode:
        model.eval()
        torch.set_grad_enabled(False)

    # Create a fixed random input batch
    example_input = torch.randn(*input_shape, device=device)

    # CUDA-specific: warm up GPU & clear cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
        # Run a few dummy ops
        _ = model(example_input)
        torch.cuda.synchronize()

    # Warmup phase (not timed)
    for _ in range(warmup_runs):
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                _ = model(example_input)
        else:
            _ = model(example_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

    # Measured runs
    times_ms: List[float] = []

    for _ in range(measured_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter_ns()

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                _ = model(example_input)
        else:
            _ = model(example_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        t1 = time.perf_counter_ns()
        elapsed_ms = (t1 - t0) / 1e6  # ns → ms
        times_ms.append(elapsed_ms)

    # Basic stats
    times_ms_sorted = sorted(times_ms)
    stats = {
        "num_runs": measured_runs,
        "batch_size": input_shape[0],
        "mean_ms": float(statistics.mean(times_ms_sorted)),
        "median_ms": float(statistics.median(times_ms_sorted)),
        "min_ms": float(times_ms_sorted[0]),
        "max_ms": float(times_ms_sorted[-1]),
        "std_ms": float(statistics.pstdev(times_ms_sorted)),
        "p90_ms": float(np.percentile(times_ms_sorted, 90)),
        "p95_ms": float(np.percentile(times_ms_sorted, 95)),
        "p99_ms": float(np.percentile(times_ms_sorted, 99)),
        "raw_ms": times_ms_sorted,
    }

    return stats


def print_latency_report(stats: dict, device: str, input_dim: int):
    """
    Nicely formatted latency report.
    """
    print("\n" + "-" * 70)
    print("      AI-DAC / TLL Inference Latency Report")
    print("-" * 70)
    print(f"Device           : {device}")
    print(f"Batch size       : {stats['batch_size']}")
    print(f"Input dim        : {input_dim}")
    print(f"Measured runs    : {stats['num_runs']}")
    print("-" * 70)
    print(f"Mean latency     : {stats['mean_ms']:.4f} ms / batch")
    print(f"Median latency   : {stats['median_ms']:.4f} ms / batch")
    print(f"Std deviation    : {stats['std_ms']:.4f} ms")
    print(f"Min latency      : {stats['min_ms']:.4f} ms")
    print(f"Max latency      : {stats['max_ms']:.4f} ms")
    print(f"p90 latency      : {stats['p90_ms']:.4f} ms")
    print(f"p95 latency      : {stats['p95_ms']:.4f} ms")
    print(f"p99 latency      : {stats['p99_ms']:.4f} ms")
    print("-" * 70)
    if stats['batch_size'] > 0:
        per_sample_mean = stats['mean_ms'] / stats['batch_size']
        print(f"Mean latency per sample ~ {per_sample_mean:.6f} ms")
    print("-" * 70 + "\n")


# ---------------------------------------------------------------------
# CLI / Main
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Latency Benchmark for AI-DAC / TLL Models (PyTorch)"
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=128,
        help="Input feature dimension (e.g., log embedding size)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for latency test",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=200,
        help="Number of measured runs",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup runs (not measured)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension for DummyModel (ignored if you replace model)",
    )
    parser.add_argument(
        "--out-dim",
        type=int,
        default=2,
        help="Output dimension for DummyModel (e.g., anomaly / normal)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: 'cpu' or 'cuda'",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use autocast (mixed precision) on CUDA",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # -----------------------------------------------------------------
    # 1) Instantiate your model here
    #    TODO: Replace DummyModel with your own AI-DAC component
    # -----------------------------------------------------------------
    model = DummyModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
    )

    # Example if loading from checkpoint:
    #
    # model = YourModelClass(...)
    # ckpt = torch.load("path/to/checkpoint.pth", map_location=args.device)
    # model.load_state_dict(ckpt["model_state_dict"])
    #
    # Ensure you keep the same forward signature: forward(x) with
    # x.shape == (batch_size, input_dim, ...)

    # Define input shape: (batch_size, input_dim)
    input_shape = (args.batch_size, args.input_dim)

    # -----------------------------------------------------------------
    # 2) Run latency measurement
    # -----------------------------------------------------------------
    stats = measure_latency(
        model=model,
        input_shape=input_shape,
        device=args.device,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
        use_amp=args.amp,
        use_eval_mode=True,
    )

    # -----------------------------------------------------------------
    # 3) Print a readable report
    # -----------------------------------------------------------------
    print_latency_report(stats, device=args.device, input_dim=args.input_dim)


if __name__ == "__main__":
    main()


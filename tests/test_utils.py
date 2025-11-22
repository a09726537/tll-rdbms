#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_utils.py
------------------------------------
Utility helpers for testing AI-DAC / Triple Loop Learning (TLL)

Author: William K
Affiliation: University of Vienna
------------------------------------

This module provides small, reusable helpers for:
 - Seeding / determinism
 - Synthetic dataset generation
 - Assertions for tensors and arrays
 - Simple timing context manager
 - Lightweight logging helpers
 - Basic model sanity checks

It is designed to be:
 - Framework-friendly (works with unittest / pytest / custom scripts)
 - Lightweight (no heavy dependencies)
 - Usable both locally and in Docker/Kubernetes environments
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional
    torch = None
    nn = None  # type: ignore


# ---------------------------------------------------------------------
# 1. Seeding / Determinism
# ---------------------------------------------------------------------


def set_global_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    """
    Set seeds for Python, NumPy, and optionally PyTorch.

    Parameters
    ----------
    seed : int
        Random seed value.
    deterministic_torch : bool
        If True and torch is installed, tries to set deterministic behavior.

    Notes
    -----
    This is best-effort only; some CUDA / cuDNN paths may still be nondeterministic.
    """
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            try:
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
            except Exception:
                # Older versions may not expose these attributes
                pass


# ---------------------------------------------------------------------
# 2. Timing Helpers
# ---------------------------------------------------------------------


@dataclass
class TimerResult:
    name: str
    seconds: float

    @property
    def ms(self) -> float:
        return self.seconds * 1000.0

    def __str__(self) -> str:
        return f"{self.name}: {self.ms:.3f} ms"


@contextlib.contextmanager
def timed_block(name: str = "block"):
    """
    Context manager for timing blocks of code.

    Example
    -------
    >>> with timed_block("forward pass") as t:
    ...     y = model(x)
    ...
    >>> print(t)
    """
    start = time.perf_counter()
    result = TimerResult(name=name, seconds=0.0)
    try:
        yield result
    finally:
        end = time.perf_counter()
        result.seconds = end - start
        print(f"[TIMER] {result}")


def time_function(fn, *args, repeat: int = 5, **kwargs) -> Dict[str, Any]:
    """
    Time a function over multiple runs and return timing statistics.

    Parameters
    ----------
    fn : callable
        Function to time.
    repeat : int
        Number of runs.

    Returns
    -------
    dict with keys: mean, std, min, max, runs (list)
    """
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times_ms = [t * 1000.0 for t in times]
    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "runs_ms": times_ms,
    }


# ---------------------------------------------------------------------
# 3. Synthetic Data Generators
# ---------------------------------------------------------------------


def make_linear_separable_dataset(
    n_samples: int = 1024,
    input_dim: int = 2,
    margin: float = 0.5,
    noise_std: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a simple linearly separable binary dataset.

    Data is generated such that:
      - points with w·x + b > margin → label 1
      - points with w·x + b < -margin → label 0
      - near the margin, labels may be noisy.

    Returns
    -------
    X : np.ndarray, shape (n_samples, input_dim)
    y : np.ndarray, shape (n_samples,)
    """
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    w = np.random.randn(input_dim).astype(np.float32)
    b = float(np.random.randn())

    scores = X.dot(w) + b
    y = (scores > 0).astype(np.int64)

    # Add margin / noise
    mask = np.abs(scores) < margin
    X[mask] += np.random.randn(*X[mask].shape).astype(np.float32) * noise_std

    return X, y


def make_sum_threshold_dataset(
    n_samples: int = 1024,
    input_dim: int = 32,
    threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dataset where label is 1 if sum(x) > threshold else 0.

    This mirrors the synthetic logic used in several of the
    AI-DAC test scripts.
    """
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    sums = X.sum(axis=1)
    y = (sums > threshold).astype(np.int64)
    return X, y


def to_torch(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> Tuple[Any, Optional[Any]]:
    """
    Convert NumPy arrays to torch tensors (if torch is available).

    Returns original numpy arrays if torch is not installed.
    """
    if torch is None:
        return X, y

    dev = torch.device(device) if device is not None else torch.device("cpu")
    X_t = torch.from_numpy(X).to(dev)
    y_t = torch.from_numpy(y).long().to(dev) if y is not None else None
    return X_t, y_t


# ---------------------------------------------------------------------
# 4. Assertions & Checks
# ---------------------------------------------------------------------


def assert_close(
    a: Union[float, np.ndarray],
    b: Union[float, np.ndarray],
    rtol: float = 1e-4,
    atol: float = 1e-6,
    msg: str = "",
) -> None:
    """
    Assert that two numbers/arrays are close (relative & absolute tolerance).

    Raises AssertionError on failure.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)

    if not np.allclose(a_arr, b_arr, rtol=rtol, atol=atol):
        default_msg = f"Arrays not close.\nA={a_arr}\nB={b_arr}\nrtol={rtol}, atol={atol}"
        raise AssertionError(msg or default_msg)


def assert_shape(x: Any, expected_shape: Tuple[int, ...], msg: str = "") -> None:
    """
    Assert that x has the expected shape.

    Supports:
    - numpy arrays
    - torch tensors
    """
    if torch is not None and isinstance(x, torch.Tensor):
        shape = tuple(int(s) for s in x.shape)
    elif isinstance(x, np.ndarray):
        shape = x.shape
    else:
        raise TypeError(f"Unsupported type for assert_shape: {type(x)}")

    if shape != expected_shape:
        default_msg = f"Shape mismatch: got {shape}, expected {expected_shape}"
        raise AssertionError(msg or default_msg)


def assert_between(
    value: float,
    low: float,
    high: float,
    inclusive: bool = True,
    msg: str = "",
) -> None:
    """
    Assert that value is in (low, high) or [low, high].

    Parameters
    ----------
    inclusive : bool
        If True, checks low <= value <= high. Otherwise low < value < high.
    """
    if inclusive:
        ok = (value >= low) and (value <= high)
    else:
        ok = (value > low) and (value < high)

    if not ok:
        default_msg = f"Value {value} not in range ({low}, {high}), inclusive={inclusive}"
        raise AssertionError(msg or default_msg)


def tensor_to_numpy(x: Any) -> np.ndarray:
    """
    Convert a torch tensor or numpy array to a numpy array.
    """
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    raise TypeError(f"Unsupported type for tensor_to_numpy: {type(x)}")


# ---------------------------------------------------------------------
# 5. Simple Logging Helpers
# ---------------------------------------------------------------------


def log_json(
    data: Dict[str, Any],
    prefix: str = "[TEST]",
    sort_keys: bool = True,
    indent: int = 2,
) -> None:
    """
    Print a dictionary as pretty JSON, prefixed for clarity.
    """
    txt = json.dumps(data, sort_keys=sort_keys, indent=indent, default=str)
    print(f"{prefix} {txt}")


def log_kv(
    kv: Dict[str, Any],
    prefix: str = "[TEST-KV]",
) -> None:
    """
    Log key-value pairs in a compact single-line format.
    """
    items = " ".join(f"{k}={v}" for k, v in kv.items())
    print(f"{prefix} {items}")


# ---------------------------------------------------------------------
# 6. Basic Model Sanity Checks (PyTorch)
# ---------------------------------------------------------------------


def check_model_forward(
    model: Any,
    input_dim: int,
    out_dim: Optional[int] = None,
    batch_size: int = 16,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a tiny forward pass to verify shape and absence of errors.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    input_dim : int
        Input feature size (assumes input shape (B, input_dim)).
    out_dim : int or None
        If set, checks that output has last dim == out_dim.
    batch_size : int
        Number of samples for test batch.

    Returns
    -------
    dict with fields:
      - ok (bool)
      - input_shape
      - output_shape
      - time_ms
    """
    if torch is None:
        raise RuntimeError("PyTorch not available, cannot run check_model_forward.")

    dev = torch.device(device) if device is not None else torch.device("cpu")
    model.to(dev)
    model.eval()

    x = torch.randn(batch_size, input_dim, device=dev)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x)
    t1 = time.perf_counter()

    out_shape = tuple(int(s) for s in out.shape)
    ok = True

    if out_dim is not None and out_shape[-1] != out_dim:
        ok = False

    result = {
        "ok": ok,
        "input_shape": (batch_size, input_dim),
        "output_shape": out_shape,
        "time_ms": (t1 - t0) * 1000.0,
    }

    if not ok:
        print("[WARN] check_model_forward: unexpected output shape:", out_shape)

    return result


# ---------------------------------------------------------------------
# 7. Self-test / Demo
# ---------------------------------------------------------------------


def _demo() -> None:
    """
    Run a small self-test when called as a script.
    This is safe and quick; it doesn't depend on external resources.
    """
    print("=== test_utils.py self-test (demo) ===")

    set_global_seed(123)

    # Synthetic dataset
    X, y = make_sum_threshold_dataset(n_samples=256, input_dim=16)
    assert X.shape == (256, 16)
    assert y.shape == (256,)
    print("- Synthetic dataset (sum-threshold) OK")

    # Assertions
    assert_close(1.0, 1.0 + 1e-7)
    assert_between(0.5, 0.0, 1.0)
    print("- Assertion helpers OK")

    if torch is not None:
        X_t, y_t = to_torch(X, y, device="cpu")
        assert_shape(X_t, (256, 16))
        assert_shape(y_t, (256,))
        print("- Torch conversion & shape checks OK")

        class Tiny(nn.Module):
            def __init__(self, d_in=16, d_out=2):
                super().__init__()
                self.fc = nn.Linear(d_in, d_out)

            def forward(self, x):
                return self.fc(x)

        model = Tiny()
        res = check_model_forward(model, input_dim=16, out_dim=2, batch_size=8)
        log_json({"model_forward_check": res})

        with timed_block("Tiny model forward (8x16)"):
            _ = model(torch.randn(8, 16))

    stats = time_function(lambda: np.dot(np.random.randn(32, 32), np.random.randn(32, 32)))
    log_kv(stats, prefix="[TIME-FUNC]")

    print("=== test_utils.py self-test completed ===")


if __name__ == "__main__":
    _demo()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_pipeline.py
------------------------------------
End-to-End Test Pipeline for AI-DAC / Triple Loop Learning (TLL)

Author: William K
Affiliation: University of Vienna
------------------------------------

This script provides a unified test harness to validate the
main building blocks of the AI-DAC / TLL stack:

 - Model forward path (sanity check)
 - Latency benchmark (via latency_test.py if available)
 - Robustness tests (via robustness.py if available)
 - (Optional) RL training smoke test (via train_rl.py if available)
 - (Optional) Meta-learning wiring sanity check (via meta_adapt.py if available)

It is designed to run both in a local dev environment and inside
the Docker/Kubernetes deployment.

USAGE EXAMPLES
--------------

  # Quick smoke test (fast, minimal)
  python test_pipeline.py --mode smoke

  # More complete test (latency + robustness)
  python test_pipeline.py --mode standard

  # Everything, including RL + meta-learning checks
  python test_pipeline.py --mode full

By default it uses synthetic data and dummy models unless your
real models are wired in.
"""

import argparse
import os
import sys
import time
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Try to import project modules (optional)
# ---------------------------------------------------------------------

HAS_META = False
HAS_RL = False
HAS_LAT = False
HAS_ROB = False

try:
    from meta_adapt import MetaLearner  # type: ignore
    HAS_META = True
except Exception:
    pass

try:
    from train_rl import DQNAgent, create_dummy_env  # type: ignore
    HAS_RL = True
except Exception:
    pass

try:
    from latency_test import measure_latency  # type: ignore
    HAS_LAT = True
except Exception:
    pass

try:
    from robustness import RobustnessEvaluator, NoiseConfig, FeatureDropConfig, FGSMConfig, accuracy  # type: ignore
    HAS_ROB = True
except Exception:
    pass


# ---------------------------------------------------------------------
# Dummy base model 
# ---------------------------------------------------------------------

class DummyAnomalyModel(nn.Module):
    """
    A small feed-forward model used as a stand-in for:
      - Autoencoder bottleneck
      - Classifier head
      - Discriminator stub

    Forward returns logits for binary classification (normal / anomaly).
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def generate_synthetic_data(
    n_samples: int,
    input_dim: int,
    device: str,
) -> (torch.Tensor, torch.Tensor):
    """
    Synthetic dataset for quick tests:
    - Features ~ N(0,1)
    - Labels: 1 if sum(x) > 0 else 0
    """
    x = torch.randn(n_samples, input_dim)
    y = (x.sum(dim=1) > 0).long()
    return x.to(device), y.to(device)


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)


# ---------------------------------------------------------------------
# 1. Forward / Sanity test
# ---------------------------------------------------------------------

def run_forward_test(model: nn.Module, input_dim: int, device: str) -> Dict[str, Any]:
    print_header("1) Forward / Sanity Test")

    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    batch_size = 16
    x = torch.randn(batch_size, input_dim, device=device)

    t0 = time.time()
    out = model(x)
    t1 = time.time()

    ok_shape = out.shape == (batch_size, 2)

    print(f"- Input  shape: {tuple(x.shape)}")
    print(f"- Output shape: {tuple(out.shape)} (expected: {(batch_size, 2)})")
    print(f"- Forward time: {(t1 - t0)*1000:.3f} ms (batch {batch_size})")
    print(f"- Status      : {'OK' if ok_shape else 'FAIL'}")

    return {
        "forward_ok": ok_shape,
        "forward_time_ms": (t1 - t0) * 1000.0,
    }


# ---------------------------------------------------------------------
# 2. Latency test (delegates to latency_test.measure_latency if present)
# ---------------------------------------------------------------------

def run_latency_test(model: nn.Module, input_dim: int, device: str, batch_size: int, runs: int, warmup: int):
    print_header("2) Latency Test")

    if not HAS_LAT:
        print("[WARN] latency_test.measure_latency not found; running internal minimal timing only.")

        model.to(device)
        model.eval()
        torch.set_grad_enabled(False)

        x = torch.randn(batch_size, input_dim, device=device)

        # warmup
        for _ in range(warmup):
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1e6)

        mean_ms = float(np.mean(times))
        p95_ms = float(np.percentile(times, 95))
        print(f"- Mean latency (batch): {mean_ms:.4f} ms")
        print(f"- p95 latency (batch): {p95_ms:.4f} ms")

        return {
            "mean_ms": mean_ms,
            "p95_ms": p95_ms,
            "batch_size": batch_size,
        }

    # Use the shared latency_test helper
    from latency_test import measure_latency, print_latency_report  # type: ignore

    stats = measure_latency(
        model=model,
        input_shape=(batch_size, input_dim),
        device=device,
        warmup_runs=warmup,
        measured_runs=runs,
        use_amp=False,
        use_eval_mode=True,
    )

    print_latency_report(stats, device=device, input_dim=input_dim)

    return stats


# ---------------------------------------------------------------------
# 3. Robustness test (delegates to robustness.py if present)
# ---------------------------------------------------------------------

def run_robustness_tests(model: nn.Module, input_dim: int, device: str):
    print_header("3) Robustness Tests")

    if not HAS_ROB:
        print("[WARN] robustness.py not found. Skipping robustness tests.")
        return None

    # Synthetic data
    n_samples = 512
    x, y = generate_synthetic_data(n_samples, input_dim, device)

    evaluator = RobustnessEvaluator(model=model, device=device, metric_fn=accuracy)

    noise_cfg = NoiseConfig(noise_levels=[0.01, 0.05, 0.1], noise_type="gaussian")
    drop_cfg = FeatureDropConfig(drop_rates=[0.1, 0.3, 0.5], drop_type="zero")
    fgsm_cfg = FGSMConfig(epsilons=[0.01, 0.05, 0.1], targeted=False)

    print("- Noise robustness:")
    noise_res = evaluator.evaluate_noise_robustness(x, y, noise_cfg)
    for lvl, score in sorted(noise_res.items(), key=lambda kv: kv[0]):
        print(f"  σ={lvl:.3f}: metric={score:.4f}")

    print("- Feature drop robustness:")
    drop_res = evaluator.evaluate_feature_drop_robustness(x, y, drop_cfg)
    for rate, score in sorted(drop_res.items(), key=lambda kv: kv[0]):
        print(f"  drop={rate:.2f}: metric={score:.4f}")

    print("- FGSM robustness:")
    fgsm_res = evaluator.evaluate_fgsm_robustness(x, y, fgsm_cfg, loss_fn=nn.CrossEntropyLoss())
    for eps, score in sorted(fgsm_res.items(), key=lambda kv: kv[0]):
        print(f"  ε={eps:.3f}: metric={score:.4f}")

    return {
        "noise": noise_res,
        "feature_drop": drop_res,
        "fgsm": fgsm_res,
    }


# ---------------------------------------------------------------------
# 4. RL Smoke Test (optional)
# ---------------------------------------------------------------------

def run_rl_smoke_test(device: str):
    print_header("4) RL Smoke Test (DQN)")

    if not HAS_RL:
        print("[WARN] train_rl.py (DQNAgent / create_dummy_env) not found. Skipping RL tests.")
        return None

    from train_rl import DQNAgent, create_dummy_env  # type: ignore

    state_dim = 32
    action_dim = 4

    env = create_dummy_env(state_dim, action_dim)
    example_state = env.reset()
    state_dim = example_state.shape[0]

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        batch_size=32,
        buffer_size=2000,
        eps_decay=2000,
        target_update_freq=200,
    )

    # Run only a few episodes for smoke test
    num_episodes = 5
    max_steps = 50

    total_rewards = []
    print(f"- Running {num_episodes} short episodes for RL smoke test...")

    for ep in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)
        ep_reward = 0.0

        for t in range(max_steps):
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, info = env.step(action)
            next_state_t = torch.tensor(next_state, dtype=torch.float32, device=device)

            agent.replay_buffer.push(state, action, float(reward), next_state_t, float(done))

            state = next_state_t
            ep_reward += float(reward)

            # minimal optimization
            loss = agent.optimize_model()
            if loss is not None and t % 10 == 0:
                agent.maybe_update_target()

            if done:
                break

        total_rewards.append(ep_reward)
        print(f"  Episode {ep+1}/{num_episodes}: reward={ep_reward:.3f}")

    avg_reward = float(np.mean(total_rewards))
    print(f"- Mean episode reward (smoke): {avg_reward:.3f}")

    return {
        "num_episodes": num_episodes,
        "avg_reward": avg_reward,
        "rewards": total_rewards,
    }


# ---------------------------------------------------------------------
# 5. Meta-Learning Sanity Test (optional)
# ---------------------------------------------------------------------

def run_meta_sanity_test(input_dim: int, device: str):
    print_header("5) Meta-Learning Sanity Test (MAML)")

    if not HAS_META:
        print("[WARN] meta_adapt.py (MetaLearner) not found. Skipping meta-learning tests.")
        return None

    from meta_adapt import MetaLearner  # type: ignore

    # Use a small model for quick test
    class SmallNet(nn.Module):
        def __init__(self, in_dim=32, hidden=32, out_dim=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_dim)
            )

        def forward(self, x):
            return self.net(x)

    model = SmallNet(in_dim=input_dim, hidden=32, out_dim=2)
    meta = MetaLearner(
        model=model,
        lr_inner=1e-2,
        lr_outer=1e-3,
        inner_steps=1,
        device=device,
    )

    # Synthetic support/query data for a few "meta-steps"
    n_support = 16
    n_query = 16

    support_x, support_y = generate_synthetic_data(n_support, input_dim, device)
    query_x, query_y = generate_synthetic_data(n_query, input_dim, device)

    loss_fn = nn.CrossEntropyLoss()

    print("- Running 5 meta-train steps...")
    losses = []
    for step in range(5):
        loss = meta.meta_train_step(
            support_x, support_y,
            query_x, query_y,
            loss_fn,
        )
        losses.append(loss)
        print(f"  meta-step {step+1}/5: outer loss={loss:.4f}")

    return {
        "meta_steps": len(losses),
        "losses": losses,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="AI-DAC / TLL End-to-End Test Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["smoke", "standard", "full"],
        help="Test mode: 'smoke' (fast), 'standard' (latency+robustness), 'full' (includes RL + meta).",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=128,
        help="Input dimension for dummy test model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: 'cpu' or 'cuda'.",
    )
    parser.add_argument(


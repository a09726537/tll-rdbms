#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robustness.py
------------------------------------
Robustness & Adversarial Evaluation Utilities
for AI-DAC / Triple Loop Learning (TLL)

Author: William K
Affiliation: University of Vienna
------------------------------------

This script provides:
 - Random noise robustness tests (Gaussian, salt-and-pepper style)
 - Feature drop / masking robustness
 - Simple FGSM adversarial attack (for classifiers)
 - Metric hooks (accuracy, F1, custom)

Intended usage:
 - Evaluate how stable your anomaly detector / classifier is
   under realistic perturbations of SQL log features.
 - Integrate into evaluation / CI pipeline alongside latency_test.py.

Models:
 - Any PyTorch nn.Module with forward(x) → logits or scores.
 - For anomaly detection, you can provide a custom `score_fn` or `metric_fn`.
"""

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Helpers: Metrics
# ---------------------------------------------------------------------

def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Simple accuracy for classification.
    preds: logits or probabilities, shape (N, C)
    targets: integer labels, shape (N,)
    """
    if preds.ndim == 1:
        # binary case with single logit
        pred_labels = (preds > 0).long()
    else:
        pred_labels = preds.argmax(dim=1)
    correct = (pred_labels == targets).sum().item()
    return correct / max(1, targets.numel())


def f1_binary(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Simple F1 score for binary classification (class 1 is 'positive').
    preds: logits or probs of shape (N,) or (N,2) or (N,1).
    targets: {0,1} labels of shape (N,).
    """
    if preds.ndim == 2:
        if preds.size(1) == 2:
            probs_pos = F.softmax(preds, dim=1)[:, 1]
        else:
            probs_pos = torch.sigmoid(preds[:, 0])
    else:
        probs_pos = torch.sigmoid(preds)

    pred_labels = (probs_pos >= threshold).long()
    tp = ((pred_labels == 1) & (targets == 1)).sum().item()
    fp = ((pred_labels == 1) & (targets == 0)).sum().item()
    fn = ((pred_labels == 0) & (targets == 1)).sum().item()

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------
# Dataclasses for configs
# ---------------------------------------------------------------------

@dataclass
class NoiseConfig:
    noise_levels: List[float]  # e.g. [0.01, 0.05, 0.1]
    noise_type: str = "gaussian"  # or "uniform"


@dataclass
class FeatureDropConfig:
    drop_rates: List[float]  # e.g. [0.1, 0.3, 0.5]
    drop_type: str = "zero"  # or "noise"


@dataclass
class FGSMConfig:
    epsilons: List[float]  # e.g. [0.01, 0.05, 0.1]
    targeted: bool = False


# ---------------------------------------------------------------------
# Robustness Evaluator
# ---------------------------------------------------------------------

class RobustnessEvaluator:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        metric_fn: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
    ):
        """
        model: PyTorch model under test.
        metric_fn: takes (preds, targets) → float, higher is better.
                   Default: accuracy.
        """
        self.model = model.to(device)
        self.device = torch.device(device)
        self.metric_fn = metric_fn or accuracy
        self.model.eval()
        torch.set_grad_enabled(False)

    # ---------------- Noise robustness ----------------

    def _add_noise(self, x: torch.Tensor, level: float, noise_type: str = "gaussian") -> torch.Tensor:
        if noise_type == "gaussian":
            noise = torch.randn_like(x) * level
            return x + noise
        elif noise_type == "uniform":
            noise = (torch.rand_like(x) - 0.5) * 2 * level
            return x + noise
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")

    def evaluate_noise_robustness(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        cfg: NoiseConfig,
    ) -> Dict[float, float]:
        """
        Evaluate metric under different Gaussian/Uniform noise levels.
        Returns: {noise_level: metric_value}
        """
        self.model.eval()
        x = x.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            # Baseline metric
            base_preds = self.model(x)
            base_metric = self.metric_fn(base_preds, y)

        results = {0.0: base_metric}

        for sigma in cfg.noise_levels:
            x_noisy = self._add_noise(x, sigma, cfg.noise_type)
            with torch.no_grad():
                preds = self.model(x_noisy)
                score = self.metric_fn(preds, y)
            results[sigma] = score

        return results

    # ---------------- Feature-drop robustness ----------------

    def _drop_features(self, x: torch.Tensor, drop_rate: float, drop_type: str = "zero") -> torch.Tensor:
        mask = (torch.rand_like(x) > drop_rate).float()
        if drop_type == "zero":
            return x * mask
        elif drop_type == "noise":
            noise = torch.randn_like(x) * 0.01
            return x * mask + (1 - mask) * noise
        else:
            raise ValueError(f"Unknown drop_type: {drop_type}")

    def evaluate_feature_drop_robustness(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        cfg: FeatureDropConfig,
    ) -> Dict[float, float]:
        """
        Evaluate metric with random feature masking at varying rates.
        Returns: {drop_rate: metric_value}
        """
        self.model.eval()
        x = x.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            base_preds = self.model(x)
            base_metric = self.metric_fn(base_preds, y)

        results = {0.0: base_metric}

        for rate in cfg.drop_rates:
            x_masked = self._drop_features(x, rate, cfg.drop_type)
            with torch.no_grad():
                preds = self.model(x_masked)
                score = self.metric_fn(preds, y)
            results[rate] = score

        return results

    # ---------------- FGSM Adversarial attack ----------------

    def fgsm_attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: float,
        loss_fn: Optional[nn.Module] = None,
        targeted: bool = False,
    ) -> torch.Tensor:
        """
        Classic FGSM attack:
        x_adv = x + epsilon * sign(∂L/∂x)

        For anomaly detection, you may need to define a custom loss_fn.
        """
        self.model.eval()
        loss_fn = loss_fn or nn.CrossEntropyLoss()

        x = x.to(self.device).detach()
        y = y.to(self.device)

        x_adv = x.clone().detach().requires_grad_(True)

        # Enable grads
        torch.set_grad_enabled(True)

        preds = self.model(x_adv)
        loss = loss_fn(preds, y)

        if targeted:
            loss = -loss

        loss.backward()
        grad_sign = x_adv.grad.data.sign()
        x_perturbed = x_adv + epsilon * grad_sign
        x_perturbed = torch.clamp(x_perturbed, -10.0, 10.0)  # loose clamp for numeric stability

        # Disable grads back
        torch.set_grad_enabled(False)

        return x_perturbed.detach()

    def evaluate_fgsm_robustness(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        cfg: FGSMConfig,
        loss_fn: Optional[nn.Module] = None,
    ) -> Dict[float, float]:
        """
        Evaluate metric under FGSM adversarial perturbations.
        Returns: {epsilon: metric_value}
        """
        self.model.eval()
        x = x.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            base_preds = self.model(x)
            base_metric = self.metric_fn(base_preds, y)

        results = {0.0: base_metric}

        for eps in cfg.epsilons:
            x_adv = self.fgsm_attack(x, y, epsilon=eps, loss_fn=loss_fn, targeted=cfg.targeted)
            with torch.no_grad():
                preds = self.model(x_adv)
                score = self.metric_fn(preds, y)
            results[eps] = score

        return results


# ---------------------------------------------------------------------
# Example dummy model & CLI interface
# ---------------------------------------------------------------------

class DummyClassifier(nn.Module):
    """
    Simple feed-forward classifier used for demonstration.
    Replace with your AI-DAC model (e.g. classifier head).
    """

    def __init__(self, in_dim: int = 32, hidden: int = 64, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Robustness evaluation for AI-DAC / TLL models")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: cpu or cuda")
    parser.add_argument("--n-samples", type=int, default=1024, help="Number of synthetic samples (demo mode)")
    parser.add_argument("--input-dim", type=int, default=32, help="Input dimension")
    parser.add_argument("--noise-levels", type=str, default="0.01,0.05,0.1",
                        help="Comma-separated noise std levels")
    parser.add_argument("--drop-rates", type=str, default="0.1,0.3,0.5",
                        help="Comma-separated feature drop rates")
    parser.add_argument("--epsilons", type=str, default="0.01,0.05,0.1",
                        help="Comma-separated FGSM epsilons")
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device

    # Demo: create random dataset
    n = args.n_samples
    d = args.input_dim
    x = torch.randn(n, d)
    # Simple synthetic binary labels based on sign of sum
    y = (x.sum(dim=1) > 0).long()

    model = DummyClassifier(in_dim=d, hidden=64, out_dim=2)
    model.to(device)

    # In a real scenario, you would load your trained model:
    # model = YourModel(...)
    # ckpt = torch.load("path/to/checkpoint.pth", map_location=device)
    # model.load_state_dict(ckpt["model_state_dict"])

    evaluator = RobustnessEvaluator(
        model=model,
        device=device,
        metric_fn=accuracy  # or f1_binary, or your custom metric
    )

    noise_levels = [float(v) for v in args.noise_levels.split(",") if v]
    drop_rates = [float(v) for v in args.drop_rates.split(",") if v]
    epsilons = [float(v) for v in args.epsilons.split(",") if v]

    noise_cfg = NoiseConfig(noise_levels=noise_levels, noise_type="gaussian")
    drop_cfg = FeatureDropConfig(drop_rates=drop_rates, drop_type="zero")
    fgsm_cfg = FGSMConfig(epsilons=epsilons, targeted=False)

    print("\n=== Noise Robustness (Gaussian) ===")
    noise_results = evaluator.evaluate_noise_robustness(x, y, noise_cfg)
    for lvl, score in sorted(noise_results.items(), key=lambda kv: kv[0]):
        print(f"Noise σ={lvl:.3f} → metric={score:.4f}")

    print("\n=== Feature Drop Robustness ===")
    drop_results = evaluator.evaluate_feature_drop_robustness(x, y, drop_cfg)
    for rate, score in sorted(drop_results.items(), key=lambda kv: kv[0]):
        print(f"Drop rate={rate:.2f} → metric={score:.4f}")

    print("\n=== FGSM Robustness ===")
    fgsm_results = evaluator.evaluate_fgsm_robustness(x, y, fgsm_cfg, loss_fn=nn.CrossEntropyLoss())
    for eps, score in sorted(fgsm_results.items(), key=lambda kv: kv[0]):
        print(f"FGSM ε={eps:.3f} → metric={score:.4f}")

    print("\n[INFO] Robustness evaluation (demo) completed.")


if __name__ == "__main__":
    main()


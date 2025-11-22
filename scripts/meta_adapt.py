#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meta_adapt.py
------------------------------------
MAML-based Meta-Learning Adaptation Module
for AI-DAC / Triple Loop Learning (TLL)
(Loop 3: Meta-Adaptation)

Author: William K
Affiliation: University of Vienna
------------------------------------

This script implements:
 - Inner-loop fast adaptation (task-specific)
 - Outer-loop meta-update (across tasks)
 - Gradient-based meta-learning (MAML-style)
 - Generic hooks for anomaly detection models:
       * classifiers (logits)
       * autoencoders (reconstruction loss)
       * discriminators (GAN / MAD-GAN)

Typical usage in AI-DAC / TLL:
 - Loop 1: supervised anomaly detection
 - Loop 2: RL policy adaptation (DQN)
 - Loop 3: meta-learning (this module) to quickly adapt
   model parameters to new database schemas / workloads.
"""

from copy import deepcopy
from typing import Callable, Dict, Any, Tuple, Optional

import torch
from torch import nn


class MetaLearner:
    """
    Generic MAML Meta-Learner for PyTorch models.

    This class is model-agnostic: you can plug in any nn.Module as long as
    you provide a suitable loss function.

    Example (classifier):
    ---------------------
    model = MyClassifier(...)
    meta = MetaLearner(model, lr_inner=1e-3, lr_outer=1e-4, inner_steps=1)

    loss_fn = nn.CrossEntropyLoss()

    outer_loss = meta.meta_train_step(
        x_support, y_support,
        x_query, y_query,
        loss_fn
    )

    Example (autoencoder with reconstruction loss):
    -----------------------------------------------
    def ae_loss_fn(model, x, _unused_y):
        out = model(x)
        return ((out - x) ** 2).mean()

    meta.meta_train_step(
        x_support, None,
        x_query, None,
        loss_fn=lambda m, x, y: ae_loss_fn(m, x, y)
    )
    """

    def __init__(
        self,
        model: nn.Module,
        lr_inner: float = 1e-3,
        lr_outer: float = 1e-4,
        inner_steps: int = 1,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.inner_steps = inner_steps

        self.outer_opt = torch.optim.Adam(self.model.parameters(), lr=self.lr_outer)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def clone_model(self) -> nn.Module:
        """
        Create a differentiable copy of the base model.
        """
        cloned = deepcopy(self.model)
        cloned.to(self.device)
        return cloned

    def inner_adapt(
        self,
        x_support: torch.Tensor,
        y_support: Optional[torch.Tensor],
        loss_fn: Callable[[nn.Module, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    ) -> nn.Module:
        """
        Perform MAML inner-loop adaptation on the support set.

        Parameters
        ----------
        x_support : tensor (N_s, D, ...)
        y_support : tensor or None
        loss_fn   : function(model, x, y) -> scalar loss

        Returns
        -------
        adapted_model : nn.Module
        """
        x_support = x_support.to(self.device)
        y_support = y_support.to(self.device) if y_support is not None else None

        adapted_model = self.clone_model()
        fast_weights = list(adapted_model.parameters())

        for _ in range(self.inner_steps):
            loss = loss_fn(adapted_model, x_support, y_support)
            grads = torch.autograd.grad(
                loss,
                fast_weights,
                create_graph=True,
                retain_graph=True,
            )

            fast_weights = [
                w - self.lr_inner * g
                for w, g in zip(fast_weights, grads)
            ]

            # Update cloned model parameters in-place
            with torch.no_grad():
                for p, w_new in zip(adapted_model.parameters(), fast_weights):
                    p.copy_(w_new)

        return adapted_model

    def outer_update(
        self,
        adapted_model: nn.Module,
        x_query: torch.Tensor,
        y_query: Optional[torch.Tensor],
        loss_fn: Callable[[nn.Module, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    ) -> float:
        """
        Perform outer-loop meta-update using the query set.

        The gradients from this loss will flow back into the *original*
        base model’s parameters (through the cloned model’s graph).
        """
        self.outer_opt.zero_grad()

        x_query = x_query.to(self.device)
        y_query = y_query.to(self.device) if y_query is not None else None

        loss = loss_fn(adapted_model, x_query, y_query)
        loss.backward()
        self.outer_opt.step()

        return float(loss.item())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def meta_train_step(
        self,
        x_support: torch.Tensor,
        y_support: Optional[torch.Tensor],
        x_query: torch.Tensor,
        y_query: Optional[torch.Tensor],
        loss_fn: Callable[[nn.Module, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    ) -> float:
        """
        One full MAML meta-training step:

        1. Inner adaptation on (x_support, y_support)
        2. Meta-update on (x_query, y_query)

        Returns
        -------
        outer_loss : float
        """
        adapted_model = self.inner_adapt(x_support, y_support, loss_fn)
        outer_loss = self.outer_update(adapted_model, x_query, y_query, loss_fn)
        return outer_loss

    def meta_evaluate(
        self,
        x_support: torch.Tensor,
        y_support: Optional[torch.Tensor],
        x_test: torch.Tensor,
        y_test: Optional[torch.Tensor],
        loss_fn: Callable[[nn.Module, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        pred_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[float, torch.Tensor]:
        """
        Evaluate meta-learned model on a new task:

        1. Adapt model on support set
        2. Evaluate on test set

        Parameters
        ----------
        x_support, y_support : support data
        x_test, y_test       : test data
        loss_fn              : function(model, x, y) -> scalar loss
        pred_fn              : optional transform for outputs (e.g. softmax)

        Returns
        -------
        test_loss : float
        preds     : tensor of raw model outputs or transformed via pred_fn
        """
        adapted_model = self.inner_adapt(x_support, y_support, loss_fn)

        x_test = x_test.to(self.device)
        y_test = y_test.to(self.device) if y_test is not None else None

        with torch.no_grad():
            out = adapted_model(x_test)
            loss = loss_fn(adapted_model, x_test, y_test)
            preds = pred_fn(out) if pred_fn is not None else out

        return float(loss.item()), preds

    # ------------------------------------------------------------------
    # Convenience: a default classification loss wrapper
    # ------------------------------------------------------------------

    @staticmethod
    def default_classification_loss(
        model: nn.Module,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
        loss_fn: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Helper for standard classification tasks.

        If y is None, raises.
        """
        if y is None:
            raise ValueError("y must not be None for classification loss.")
        logits = model(x)
        criterio = loss_fn or nn.CrossEntropyLoss()
        return criterio(logits, y)


# ----------------------------------------------------------------------
# Example usage (demo)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal sanity check to verify that the MetaLearner runs.

    This uses a tiny synthetic classification task:
      - inputs: Gaussian features
      - labels: sign of sum(x)
    """

    class SimpleNet(nn.Module):
        def __init__(self, in_dim=20, hidden=32, out_dim=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create base model & meta-learner
    base_model = SimpleNet(in_dim=20, hidden=32, out_dim=2)
    meta = MetaLearner(
        model=base_model,
        lr_inner=1e-2,
        lr_outer=1e-3,
        inner_steps=1,
        device=device,
    )

    # Synthetic "task"
    def make_task(n: int = 32, dim: int = 20):
        x = torch.randn(n, dim)
        y = (x.sum(dim=1) > 0).long()
        return x, y

    x_s, y_s = make_task(32, 20)  # support
    x_q, y_q = make_task(32, 20)  # query

    loss_fn = lambda m, x, y: MetaLearner.default_classification_loss(m, x, y)

    print("[META] Running 5 demo meta-train steps...")
    for step in range(5):
        outer_loss = meta.meta_train_step(x_s, y_s, x_q, y_q, loss_fn)
        print(f"  step {step+1}/5: outer_loss={outer_loss:.4f}")

    print("[META] Demo completed.")

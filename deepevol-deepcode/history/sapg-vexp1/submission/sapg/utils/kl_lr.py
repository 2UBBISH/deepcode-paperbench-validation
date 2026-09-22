"""KL-adaptive learning rate schedule for SAPG / PPO.

The SAPG paper (and the standard PPO implementation it builds on) adapts the
learning rate based on the measured KL divergence between the old and the new
policy.  If the KL is larger than a threshold the learning rate is decreased,
if it is smaller the learning rate is increased.  The default threshold used in
the paper is ``0.016``.

This module provides a small, dependency-free implementation that can be used
both by the SAPG trainer (``sapg/sapg/sapg_trainer.py``) and by the PPO
baseline (``sapg/baselines/ppo.py``).
"""

from __future__ import annotations

from typing import Optional

import torch


class KLAdaptiveLR:
    """Adaptive learning-rate controller driven by the measured KL divergence.

    Parameters
    ----------
    optimizer:
        The torch optimizer whose ``param_groups[*]["lr"]`` will be updated.
    threshold:
        KL threshold.  If the measured KL is above ``threshold`` the learning
        rate is multiplied by ``factor_down``; if it is below ``threshold`` the
        learning rate is multiplied by ``factor_up``.  Defaults to ``0.016``
        (the value used in the SAPG paper).
    min_lr, max_lr:
        Lower/upper bounds for the learning rate.
    factor_up, factor_down:
        Multiplicative factors applied to the learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        threshold: float = 0.016,
        min_lr: float = 1e-5,
        max_lr: float = 1e-2,
        factor_up: float = 1.5,
        factor_down: float = 0.5,
    ) -> None:
        self.optimizer = optimizer
        self.threshold = float(threshold)
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.factor_up = float(factor_up)
        self.factor_down = float(factor_down)

        # Remember the initial learning rate so that ``reset`` can restore it.
        self._initial_lr = self.current_lr()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def current_lr(self) -> float:
        """Return the learning rate of the first parameter group."""
        return float(self.optimizer.param_groups[0]["lr"])

    def update(self, kl: float) -> float:
        """Update the learning rate given the measured KL divergence.

        Returns the new learning rate.
        """
        kl = float(kl)
        lr = self.current_lr()

        if kl > self.threshold:
            lr = max(self.min_lr, lr * self.factor_down)
        elif kl < self.threshold:
            lr = min(self.max_lr, lr * self.factor_up)

        self._set_lr(lr)
        return lr

    def reset(self) -> float:
        """Restore the initial learning rate."""
        self._set_lr(self._initial_lr)
        return self._initial_lr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def state_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
            "factor_up": self.factor_up,
            "factor_down": self.factor_down,
            "initial_lr": self._initial_lr,
        }

    def load_state_dict(self, state: dict) -> None:
        self.threshold = float(state.get("threshold", self.threshold))
        self.min_lr = float(state.get("min_lr", self.min_lr))
        self.max_lr = float(state.get("max_lr", self.max_lr))
        self.factor_up = float(state.get("factor_up", self.factor_up))
        self.factor_down = float(state.get("factor_down", self.factor_down))
        self._initial_lr = float(state.get("initial_lr", self._initial_lr))


def make_kl_lr(
    optimizer: torch.optim.Optimizer,
    threshold: float = 0.016,
    min_lr: float = 1e-5,
    max_lr: float = 1e-2,
) -> KLAdaptiveLR:
    """Convenience factory mirroring the config keys used in the YAML files."""
    return KLAdaptiveLR(
        optimizer,
        threshold=threshold,
        min_lr=min_lr,
        max_lr=max_lr,
    )


__all__ = ["KLAdaptiveLR", "make_kl_lr"]

"""Back-to-source activation shifting (Section 3.2, Eqn. (7)-(9)).

.. math::

    \\mathbf{e}_N^0 \\leftarrow \\mathbf{e}_N^0 + \\gamma \\mathbf{d}, \\qquad
    \\mathbf{d}_t = \\boldsymbol{\\mu}_N^S - \\boldsymbol{\\mu}_N(t), \\qquad
    \\boldsymbol{\\mu}_N(t) = \\alpha \\boldsymbol{\\mu}_N(\\mathcal{X}_t)
        + (1-\\alpha)\\boldsymbol{\\mu}_N(t-1)

with ``gamma = 1.0`` ("aiming to exactly align the overall center of testing and
training features") and ``alpha = 0.1``.  Per the addendum of the paper,
``mu_N(0)`` is initialised with the statistics of the first test batch, i.e. the EMA is
seeded with ``mu_N(X_1)``.

The shifted activation is the input of the final task head only; the model weights and
all other activations stay untouched, and no back-propagation is involved.
"""
from __future__ import annotations

from typing import Optional

import torch


class BackToSourceShifting:
    """Online estimator of the shifting direction ``d_t`` (Eqn. (8)-(9))."""

    def __init__(
        self,
        source_mean: torch.Tensor,
        alpha: float = 0.1,
        gamma: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self.source_mean = source_mean
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.enabled = bool(enabled)
        self.ema: Optional[torch.Tensor] = None
        self.num_updates = 0

    def reset(self) -> None:
        self.ema = None
        self.num_updates = 0

    @torch.no_grad()
    def update(self, batch_mean: torch.Tensor) -> torch.Tensor:
        """Feed ``mu_N(X_t)`` and return the shift ``gamma * d_t`` (Eqn. (7))."""
        batch_mean = batch_mean.detach()
        if self.ema is None:
            # addendum: mu_N(0) <- mu_N(X_1)
            self.ema = batch_mean.clone()
        else:
            self.ema = self.alpha * batch_mean + (1.0 - self.alpha) * self.ema
        self.num_updates += 1
        if not self.enabled:
            return torch.zeros_like(self.source_mean)
        direction = self.source_mean.to(self.ema.device, self.ema.dtype) - self.ema
        return self.gamma * direction

    @property
    def direction(self) -> Optional[torch.Tensor]:
        if self.ema is None:
            return None
        return self.source_mean.to(self.ema.device, self.ema.dtype) - self.ema

"""Diagonal Fisher Information Matrix estimation.

Appendix C.1::

    L_aux(theta) = sum_i F^i (theta_pre^i - theta^i)^2

where ``F`` is the diagonal of the Fisher Information Matrix.  For a policy we
use the *empirical* Fisher estimated with samples from the pre-training
distribution::

    F_i = E_{s ~ D, a ~ pi_*}[ (d log pi_*(a | s) / d theta_i)^2 ]

The NetHack experiments sample 10000 batches from the NLD-AA dataset to compute
the Fisher matrix (addendum), the Montezuma's Revenge experiments use batches
sampled with the pre-trained agent and the Meta-World experiments follow
Wolczyk et al. (2021).
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch import Tensor


def _zero_like_params(params: Dict[str, nn.Parameter]) -> Dict[str, Tensor]:
    return {name: torch.zeros_like(p, device=p.device) for name, p in params.items()}


class DiagonalFisher:
    """Accumulate the diagonal empirical Fisher over a stream of batches.

    Parameters
    ----------
    module:
        The module whose parameters are tracked (usually the policy/actor).
    log_prob_fn:
        Callable mapping a batch ``(observations, **kwargs)`` to a tensor of
        shape ``(batch,)`` holding ``log pi_*(a | s)`` for actions sampled from
        ``pi_*``.  The actions must be sampled *inside* the callable so that
        gradients flow through the log-density of the sampled actions.
    param_filter:
        Optional predicate selecting which named parameters to track.
    """

    def __init__(
        self,
        module: nn.Module,
        log_prob_fn: Callable[..., Tensor],
        param_filter: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.module = module
        self.log_prob_fn = log_prob_fn
        self.param_filter = param_filter or (lambda name: True)
        self._fisher: Dict[str, Tensor] = {}
        self._num_batches = 0

    # ------------------------------------------------------------------
    def tracked_params(self) -> Dict[str, nn.Parameter]:
        return {
            name: p
            for name, p in self.module.named_parameters()
            if p.requires_grad and self.param_filter(name)
        }

    def __call__(self, *args, **kwargs) -> None:
        """Accumulate one batch of Fisher estimates."""

        self.update(*args, **kwargs)

    def update(self, batch_size_scale: int = 1, *args, **kwargs) -> None:
        params = self.tracked_params()
        if not self._fisher:
            self._fisher = _zero_like_params(params)

        self.module.zero_grad(set_to_none=True)
        log_prob = self.log_prob_fn(*args, **kwargs)
        # (1/N) sum_n (d log pi / d theta)^2 over the whole batch.
        if log_prob.dim() == 0:
            log_prob = log_prob.unsqueeze(0)
        loss = log_prob.mean()
        grads = torch.autograd.grad(
            loss, list(params.values()), retain_graph=False, allow_unused=True
        )
        for (name, _), grad in zip(params.items(), grads):
            if grad is None:
                continue
            self._fisher[name] += grad.detach() ** 2 * batch_size_scale
        self._num_batches += batch_size_scale

    # ------------------------------------------------------------------
    @property
    def num_batches(self) -> int:
        return self._num_batches

    def compute(self, normalize: bool = True) -> Dict[str, Tensor]:
        """Return the (optionally normalized) diagonal Fisher."""

        if not self._fisher:
            raise RuntimeError("No Fisher information accumulated yet.")
        if not normalize or self._num_batches == 0:
            return {k: v.clone() for k, v in self._fisher.items()}
        return {k: v / self._num_batches for k, v in self._fisher.items()}

    # ------------------------------------------------------------------
    @classmethod
    def estimate(
        cls,
        module: nn.Module,
        log_prob_fn: Callable[..., Tensor],
        batches: Iterable,
        num_batches: int = 10000,
        normalize: bool = True,
        progress: bool = False,
        param_filter: Optional[Callable[[str], bool]] = None,
    ) -> Dict[str, Tensor]:
        """Estimate the diagonal Fisher from ``num_batches`` batches.

        ``batches`` is any iterable yielding the positional/keyword arguments
        consumed by ``log_prob_fn``.  NetHack samples 10000 batches from NLD-AA.
        """

        estimator = cls(module, log_prob_fn, param_filter=param_filter)
        iterator = iter(batches)
        for _ in range(num_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(batches)
                batch = next(iterator)
            if isinstance(batch, dict):
                estimator.update(**batch)
            elif isinstance(batch, tuple):
                estimator.update(*batch)
            else:
                estimator.update(batch)
        if progress:
            print(f"[fisher] accumulated {estimator.num_batches} batches")
        return estimator.compute(normalize=normalize)

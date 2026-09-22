"""Adaptive and efficient LM tuning (Section 4.3 of the paper).

Two ingredients:

1. **Adapter salience.**  ``I(H_apt) = sum_{i,j} S(W_B[i,j])`` where
   ``S(W) = |W * dL/dW|`` (Eq. 3).  It measures how important each APT adapter
   is for the current task.
2. **Salience-based rank growth.**  Adapters are sorted by ``I(H_apt)`` and the
   ranks of the *top half* salient ones are increased, subject to the tuning
   budget ``Delta_t``::

       r_apt' = floor(r_apt * Delta_t' / Delta_t)

   New entries of ``W_A`` are drawn from ``N(0, sigma^2)`` and ``W_B`` is zero
   padded, so that adding parameters leaves the layer output unchanged (the
   standard LoRA initialisation).
"""

from __future__ import annotations

from typing import List

import torch

from .adapter import APTLinear


def adapter_importances(topo) -> torch.Tensor:
    """``I(H_apt)`` for every adapter-carrying linear, in registry order."""
    names = [n for n, lin in topo.linears.items() if lin.use_lora]
    return torch.tensor([topo.linears[n].adapter_importance() for n in names], dtype=torch.float64)


def salient_adapter_names(topo, top_fraction: float = 0.5) -> List[str]:
    """Names of the adapters whose rank should be grown.

    "we first calculate the salience of each APT adapter to determine their
    importance.  Next, we select the top-half APT adapters after sorting them
    with salience and add their parameters by increasing their ``r_apt``."
    """
    names = [n for n, lin in topo.linears.items() if lin.use_lora]
    if not names:
        return []
    scores = adapter_importances(topo)
    k = max(1, int(round(len(names) * top_fraction)))
    order = torch.argsort(scores, descending=True)
    return [names[int(i)] for i in order[:k].tolist()]


def grow_salient_adapters(topo, new_rank: int, top_fraction: float = 0.5) -> List[str]:
    """Increase the rank of the top salient adapters to ``new_rank``."""
    grown = []
    for name in salient_adapter_names(topo, top_fraction):
        lin: APTLinear = topo.linears[name]
        if new_rank > lin.rank:
            lin.grow_rank(new_rank)
            grown.append(name)
    return grown


def grow_uniform_adapters(topo, new_rank: int) -> List[str]:
    """Ablation of Figure 5a: grow *every* adapter instead of the salient ones."""
    grown = []
    for name, lin in topo.linears.items():
        if lin.use_lora and new_rank > lin.rank:
            lin.grow_rank(new_rank)
            grown.append(name)
    return grown


def tuning_parameter_count(topo) -> int:
    return topo.n_tuning_parameters()


__all__ = [
    "adapter_importances",
    "salient_adapter_names",
    "grow_salient_adapters",
    "grow_uniform_adapters",
    "tuning_parameter_count",
]

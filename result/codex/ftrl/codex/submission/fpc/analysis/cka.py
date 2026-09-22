"""Centered Kernel Alignment (Kornblith et al., 2019).

Appendix B.3::

    CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))

with a linear kernel in both cases.  We use it to measure how the activations of
the actor and critic change during fine-tuning (Figure 20).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor


def _center(k: Tensor) -> Tensor:
    n = k.shape[0]
    identity = torch.eye(n, device=k.device, dtype=k.dtype)
    ones = torch.ones(n, n, device=k.device, dtype=k.dtype)
    return k - ones @ k / n - k @ ones / n + (ones @ k @ ones) / (n * n)


def linear_kernel(x: Tensor) -> Tensor:
    return x @ x.t()


def hsic(k: Tensor, l: Tensor) -> Tensor:
    """Hilbert-Schmidt Independence Criterion with centred kernels."""

    return (_center(k) * _center(l)).sum() / (k.shape[0] - 1) ** 2


def cka(x: Tensor, y: Tensor, kernel: Callable[[Tensor], Tensor] = linear_kernel) -> float:
    """CKA between two activation matrices ``x`` and ``y`` of shape ``(n, p)``."""

    k, l = kernel(x), kernel(y)
    denom = torch.sqrt(hsic(k, k) * hsic(l, l))
    if float(denom) == 0.0:
        return float("nan")
    return float((hsic(k, l) / denom).item())


def layer_activations(model, inputs: Tensor, layer_names: Optional[Sequence[str]] = None) -> Dict[str, Tensor]:
    """Collect the activations of the named layers for a batch of ``inputs``."""

    activations: Dict[str, Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            activation = output[0] if isinstance(output, tuple) else output
            activations[name] = activation.detach()
        return hook

    for name, module in model.named_modules():
        if not name:
            continue
        if layer_names is None or name in layer_names:
            hooks.append(module.register_forward_hook(make_hook(name)))
    model(inputs)
    for hook in hooks:
        hook.remove()
    return activations


def cka_over_training(
    reference_activations: Dict[str, Tensor],
    current_activations: Dict[str, Tensor],
) -> Dict[str, float]:
    """CKA between the pre-training activations and the current activations."""

    out: Dict[str, float] = {}
    for name, ref in reference_activations.items():
        cur = current_activations.get(name)
        if cur is None:
            continue
        out[name] = cka(ref.float(), cur.float())
    return out

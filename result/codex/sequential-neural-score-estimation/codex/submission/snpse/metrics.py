"""Evaluation metrics.

The paper reports the classification-based two-sample test (C2ST) score
(Lopez-Paz & Oquab, 2017) for every benchmark experiment.  Following the
addendum of the reproduction task, C2ST is computed with the ``sbibm`` library
using its default hyperparameters.
"""

from __future__ import annotations

import torch


def c2st(
    samples_a: torch.Tensor,
    samples_b: torch.Tensor,
    seed: int = 1,
    **kwargs,
) -> float:
    """C2ST score using ``sbibm.metrics.c2st`` with default hyperparameters.

    Lower is better; 0.5 corresponds to a perfect posterior approximation (the
    two sample sets are indistinguishable), 1.0 to fully distinguishable.
    """
    try:
        from sbibm.metrics import c2st as _c2st
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sbibm is required for the C2ST metric (see addendum). "
            "Install with `pip install sbibm`."
        ) from exc

    a = torch.as_tensor(samples_a).detach().cpu().float()
    b = torch.as_tensor(samples_b).detach().cpu().float()
    return float(_c2st(a, b, seed=seed, **kwargs).item())


def posterior_mean_error(samples: torch.Tensor, reference: torch.Tensor) -> float:
    return float((samples.mean(0) - reference.mean(0)).norm().item())

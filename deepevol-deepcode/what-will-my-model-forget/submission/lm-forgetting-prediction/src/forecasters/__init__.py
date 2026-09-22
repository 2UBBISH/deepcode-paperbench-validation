"""Forecasting methods for the "What Will My Model Forget?" reproduction.

This sub-package implements every forgetting forecaster ``g`` studied in the
paper, all of them *cheap* at forecast time: once the persistent caches of
``src.modeling.caches`` exist (``f_0(x_j)`` top-k logits, ``h(x_j, y_j)`` and
the frequency prior ``b_j``), forecasting a stream of forgotten examples costs
``O(|D_PT_hat|)`` inner products / table lookups and requires **no** further
PTLM inference over ``D_PT_hat``.

Modules
-------
``threshold``
    Frequency-threshold baseline (Sec. 3.1, Eq. 1): predict that upstream
    example ``x_j`` is forgotten iff the number of past online refinements that
    forgot it is at least the tuned threshold ``gamma``.

``logit_based``
    Partially-interpretable logit-change-transfer forecaster (Sec. 3.2,
    Eq. 2-3).  It transfers the *observed* logit change of the online example
    ``f_i(x_i) - f_0(x_i)`` to an upstream example through the trainable kernel
    ``Theta_tilde(x_j, x_i) = h(x_j, y_j) h(x_i, y_i)^T`` and predicts
    ``f_hat_i(x_j) = Theta_tilde @ delta_xi + f_0(x_j)``.  Also hosts the
    non-trained ``FixedLogitForecaster`` variant of Sec. 4.2 that reuses the
    frozen base-PTLM representation (exact when only LM heads are tuned).

``representation_based``
    Black-box representation-based forecaster (Sec. 3.3, Eq. 4):
    ``z_tilde_ij = sigma(<h(x_j, y_j), h(x_i, y_i)> + b_j)`` with mean-pooled
    representations, trained with a (positive-down-weighted) BCE objective.
    ``use_prior=False`` / ``--no-prior`` reproduces the "w/o Prior" ablation.

``losses``
    Shared objectives: the Eq. 3 margin loss, the Eq. 4 BCE (with prior), the
    top-k/logit prediction helpers used by both forecasters, and the DER-style
    KL/MSE distillation losses reused by the replay refinement pipeline.

Notes
-----
Following Sec. 2 (and not the apparent ``x_i`` typo in Appendix F), the
forgetting label is ``z_ij = 1[f_i(x_j) != y_j]`` — i.e. it is defined on the
*upstream* example.  All modules here assume that convention.

Heavy dependencies (``torch``, ``transformers``, ``peft``) are imported lazily
inside the leaf modules, so this package stays importable (and self-testable)
in CPU-only / library-light environments.
"""

from __future__ import annotations

__all__ = [
    "threshold",
    "logit_based",
    "representation_based",
    "losses",
]

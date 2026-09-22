"""FOA method sub-package.

This package implements the core of *Test-Time Model Adaptation with Only
Forward Passes* (Forward-Optimization Adaptation, FOA): every module here is
strictly **backpropagation-free** and never writes to the model weights.

Modules
-------
``source_stats``
    Offline source in-distribution CLS activation statistics
    ``{mu_i^S, sigma_i^S}_{i=0..N}`` (Eqn. 5 / Eqn. 7 inputs).
``fitness``
    The unsupervised fitness of Eqn. (5): prediction entropy plus a
    lambda-weighted per-layer activation-statistics discrepancy.
``cma_wrapper``
    CMA-ES ask/tell optimizer (Eqn. 6) over the flattened prompt vector.
``activation_shifting``
    Back-to-source activation shifting of the final-layer CLS feature
    (Eqn. 7-9) with the EMA state ``mu_N(t)``.
``foa``
    Algorithm 1: the online FOA loop at batch size 64.
``foa_interval``
    FOA-I (interval adaptation) for single-sample streams (Table 6).

The sub-modules are deliberately *not* imported eagerly so that importing
``src.method`` stays cheap and free of heavy ``torch``/``timm`` side effects
(e.g. CUDA initialisation or checkpoint downloads); submodules are resolved
only when explicitly requested.
"""

from __future__ import annotations

__all__ = [
    "source_stats",
    "fitness",
    "cma_wrapper",
    "activation_shifting",
    "foa",
    "foa_interval",
]

__version__ = "0.1.0"

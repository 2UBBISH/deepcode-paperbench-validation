"""Ground-truth forgetting label construction and frequency-prior estimation.

This sub-package implements the *data-generation* backbone of the paper
"What Will My Model Forget? Forecasting Forgotten Examples in Language Model
Refinement" (Sec. 2, Sec. 3.3, Appendix F Algorithms 1 & 3):

``ground_truth``
    Sample online errors ``(x_i, y_i) ~ D_R``, refine a base PTLM ``f_0`` on a
    single example to obtain ``f_i``, evaluate on every upstream example in
    ``D_PT_hat``, and record the ground-truth forgetting label::

        z_ij = 1[ f_i(x_j) != y_j ]          (Sec. 2, upstream definition)

    In addition to the label it caches the four logit streams needed by the
    forecasters -- ``f_0(x_i)``, ``f_i(x_i)``, ``f_0(x_j)`` and ``f_i(x_j)`` --
    as top-k (``k=100``) token logits, and provides brute-force verification of
    ``z_ij`` against a direct forward pass on small ``D_PT_hat`` subsets.

``frequency_prior``
    Estimate the per-upstream-example log-odds prior used by the
    representation-based forecaster (Eq. 4)::

        b_j = log P(z_ij = 1) - log P(z_ij = 0)

    The prior is cached per ``x_j`` so that forecast-time inference over
    ``D_PT_hat`` costs ``O(|D_PT_hat|)`` lookups with no additional PTLM
    inference.

Notes
-----
* The forgetting label follows the paper's Sec. 2 definition on the *upstream*
  example ``x_j``.  Appendix F's ``1[f_0(x_i) != f_i(x_i)]`` spelling (online
  example) is treated as a typo; the alternate quantity is retained only as a
  diagnostic on the online record.
* Heavy third-party dependencies (``torch``, ``transformers``) are imported
  lazily inside the leaf modules and their functions, so this package stays
  importable in CPU-only / library-light contexts (config parsing, table
  formatting, ``--self-test`` smoke paths).
"""

from __future__ import annotations

__all__ = ["ground_truth", "frequency_prior"]

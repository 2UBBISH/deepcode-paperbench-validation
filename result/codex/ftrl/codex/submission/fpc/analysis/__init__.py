"""Forgetting diagnostics shared by all three experimental domains.

* :mod:`fpc.analysis.cka` -- Centered Kernel Alignment (Figure 20),
* :mod:`fpc.analysis.forward_transfer` -- AUC-based forward transfer (Table 6),
* :mod:`fpc.analysis.forgetting` -- expert-action log-likelihoods and PCA
  projections (Figure 8), level-visitation densities (Figure 4, 16).
"""

from .cka import cka, cka_over_training, linear_kernel, hsic
from .forward_transfer import auc, forward_transfer
from .forgetting import expert_log_likelihood, pca_projection, level_visitation_density

__all__ = [
    "cka",
    "cka_over_training",
    "linear_kernel",
    "hsic",
    "auc",
    "forward_transfer",
    "expert_log_likelihood",
    "pca_projection",
    "level_visitation_density",
]

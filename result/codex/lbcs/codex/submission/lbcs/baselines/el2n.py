"""EL2N (Paul et al., NeurIPS 2021) baseline.

"The method involves the data points with larger norms of the error vector
that is the predicted class probabilities minus one-hot label encoding."
(Appendix D.1)

    EL2N(x, y) = E || p(x) - onehot(y) ||_2

The expectation is taken over several independently trained proxy models, as
in the reference implementation of the baseline (data_diet).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import DatasetBundle
from .common import (ProxyConfig, class_balanced_top_k, forward_logits_features,
                     top_k_indices, train_proxy_models)


def compute_el2n_scores(bundle: DatasetBundle,
                        model_factory: Callable[[], nn.Module],
                        device: torch.device,
                        cfg: Optional[ProxyConfig] = None,
                        seeds: Optional[list] = None) -> torch.Tensor:
    """Average EL2N score of every training example."""
    cfg = cfg or ProxyConfig()
    models = train_proxy_models(bundle, model_factory, device, cfg, seeds)
    total = torch.zeros(bundle.n, dtype=torch.float64)
    y = bundle.train_y
    for model in models:
        logits, _ = forward_logits_features(model, bundle.train_x, device,
                                            cfg.batch_size)
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(y, num_classes=bundle.num_classes).double()
        err = (probs.double() - onehot).norm(dim=1)
        total += err
    return (total / max(len(models), 1)).float()


def el2n_select(bundle: DatasetBundle, k: int,
                model_factory: Callable[[], nn.Module],
                device: torch.device, cfg: Optional[ProxyConfig] = None,
                seeds: Optional[list] = None, class_balanced: bool = True,
                **_) -> torch.Tensor:
    """Select the ``k`` examples with the largest EL2N scores."""
    scores = compute_el2n_scores(bundle, model_factory, device, cfg, seeds)
    if class_balanced:
        return class_balanced_top_k(scores, bundle.train_y, bundle.num_classes, k)
    return top_k_indices(scores, k)

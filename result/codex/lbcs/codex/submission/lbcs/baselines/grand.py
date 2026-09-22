"""GraNd (Paul et al., NeurIPS 2021) baseline.

"The method builds a coreset by involving the data points with larger loss
gradient norms during training." (Appendix D.1)

    GraNd(x, y) = E || grad_theta l(h(x; theta), y) ||_2

The expectation is taken over several independently trained proxy models.  For
the classification heads used in this paper the gradient of the last linear
layer has a closed form, which we use to compute the score exactly and
cheaply: with ``h`` the penultimate feature, ``p = softmax(z)`` and ``y`` the
one-hot label,

    d l / dW = (p - y) h^T,      d l / db = (p - y),
    || d l / d(W, b) ||_2 = ||p - y||_2 * sqrt(1 + ||h||_2^2).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import DatasetBundle
from .common import (ProxyConfig, class_balanced_top_k, forward_logits_features,
                     top_k_indices, train_proxy_models)


def compute_grand_scores(bundle: DatasetBundle,
                         model_factory: Callable[[], nn.Module],
                         device: torch.device,
                         cfg: Optional[ProxyConfig] = None,
                         seeds: Optional[list] = None) -> torch.Tensor:
    cfg = cfg or ProxyConfig()
    models = train_proxy_models(bundle, model_factory, device, cfg, seeds)
    total = torch.zeros(bundle.n, dtype=torch.float64)
    y = bundle.train_y
    for model in models:
        logits, feats = forward_logits_features(model, bundle.train_x, device,
                                                cfg.batch_size)
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(y, num_classes=bundle.num_classes).double()
        err_norm = (probs.double() - onehot).norm(dim=1)
        feat_norm = feats.double().norm(dim=1)
        grad_norm = err_norm * torch.sqrt(1.0 + feat_norm ** 2)
        total += grad_norm
    return (total / max(len(models), 1)).float()


def grand_select(bundle: DatasetBundle, k: int,
                 model_factory: Callable[[], nn.Module],
                 device: torch.device, cfg: Optional[ProxyConfig] = None,
                 seeds: Optional[list] = None, class_balanced: bool = True,
                 **_) -> torch.Tensor:
    scores = compute_grand_scores(bundle, model_factory, device, cfg, seeds)
    if class_balanced:
        return class_balanced_top_k(scores, bundle.train_y, bundle.num_classes, k)
    return top_k_indices(scores, k)

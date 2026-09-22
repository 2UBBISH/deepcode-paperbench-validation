"""Moderate coreset (Xia et al., ICLR 2023) baseline.

"This method chooses the examples with the scores close to the score median in
coreset selection.  The score is about the distance of an example to its class
center." (Appendix D.1)

The distance is measured in the penultimate feature space of a proxy model,
which is the setting of the reference implementation (Moderate-DS).  The
selection is carried out per class so that the class proportions of the
coreset match those of the full training set.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn as nn

from ..data import DatasetBundle
from .common import (ProxyConfig, forward_logits_features, train_proxy_models)


def compute_moderate_scores(bundle: DatasetBundle,
                            model_factory: Callable[[], nn.Module],
                            device: torch.device,
                            cfg: Optional[ProxyConfig] = None,
                            seeds: Optional[list] = None) -> torch.Tensor:
    """Distance of every training example to the center of its class."""
    cfg = cfg or ProxyConfig()
    models = train_proxy_models(bundle, model_factory, device, cfg, seeds)
    feats_sum = torch.zeros(bundle.n, 0)
    for model in models:
        _, feats = forward_logits_features(model, bundle.train_x, device,
                                           cfg.batch_size)
        feats_sum = torch.cat([feats_sum, feats], dim=1)
    feats = feats_sum / max(len(models), 1)
    scores = torch.zeros(bundle.n)
    for c in range(bundle.num_classes):
        idx_c = torch.nonzero(bundle.train_y == c, as_tuple=False).flatten()
        if idx_c.numel() == 0:
            continue
        center = feats[idx_c].mean(dim=0, keepdim=True)
        scores[idx_c] = (feats[idx_c] - center).norm(dim=1)
    return scores


def _moderate_indices_per_class(scores_c: torch.Tensor, idx_c: torch.Tensor,
                                k_c: int) -> torch.Tensor:
    """Take the ``k_c`` examples of a class whose scores surround the median."""
    order = torch.argsort(scores_c)
    sorted_idx = idx_c[order]
    m = sorted_idx.numel()
    k_c = min(k_c, m)
    median = m // 2
    start = max(0, min(median - k_c // 2, m - k_c))
    return sorted_idx[start:start + k_c]


def moderate_select(bundle: DatasetBundle, k: int,
                    model_factory: Callable[[], nn.Module],
                    device: torch.device, cfg: Optional[ProxyConfig] = None,
                    seeds: Optional[list] = None, **_) -> torch.Tensor:
    """Select the ``k`` examples whose class-center distance is near median."""
    scores = compute_moderate_scores(bundle, model_factory, device, cfg, seeds)
    counts = torch.bincount(bundle.train_y, minlength=bundle.num_classes).float()
    fractions = counts / counts.sum()
    selected: List[torch.Tensor] = []
    for c in range(bundle.num_classes):
        idx_c = torch.nonzero(bundle.train_y == c, as_tuple=False).flatten()
        if idx_c.numel() == 0:
            continue
        k_c = max(1, int(round(float(fractions[c]) * k)))
        selected.append(_moderate_indices_per_class(scores[idx_c], idx_c, k_c))
    out = torch.cat(selected)
    if out.numel() > k:                      # trim to the requested size
        out = out[torch.randperm(out.numel())[:k]]
    return out

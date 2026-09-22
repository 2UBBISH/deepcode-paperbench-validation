"""CCS -- Coverage-centric coreset selection (Zheng et al., ICLR 2023).

"The method proposes a novel one-shot coreset selection method that jointly
considers overall data coverage upon a distribution as well as the importance
of each example." (Appendix D.1)

Implementation.  We follow the coverage-centric recipe of the paper:

1. train a proxy model once and take the penultimate features;
2. partition the data into ``k`` clusters (coverage of the feature
   distribution) with k-means;
3. in every cluster keep the most important example, where importance is the
   per-example training loss of the proxy model;
4. clusters that are empty (or larger-than-average clusters) are compensated
   by filling the remaining budget with the next most important unselected
   examples.

``kmeans_iters``, ``feature_batch_size`` and the number of proxy models are
configurable because k-means over the full feature matrix is the dominant cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import DatasetBundle
from .common import (ProxyConfig, forward_logits_features, train_proxy_models)


@dataclass
class CCSConfig(ProxyConfig):
    kmeans_iters: int = 100
    kmeans_batch_size: int = 4096
    normalize_features: bool = True
    seed: int = 0


def _kmeans_torch(x: torch.Tensor, n_clusters: int, iters: int,
                  batch_size: int, seed: int = 0) -> torch.Tensor:
    """Mini-batch k-means returning the cluster assignment of every row."""
    g = torch.Generator().manual_seed(seed)
    n = x.size(0)
    n_clusters = min(n_clusters, n)
    init_idx = torch.randperm(n, generator=g)[:n_clusters]
    centers = x[init_idx].clone()
    counts = torch.zeros(n_clusters)
    assign = torch.zeros(n, dtype=torch.long)
    for it in range(iters):
        batch_idx = torch.randperm(n, generator=g)[:batch_size]
        xb = x[batch_idx]
        d = torch.cdist(xb, centers)
        a = d.argmin(dim=1)
        assign[batch_idx] = a
        lr = 1.0 / (counts[a] + 1.0)
        lr = lr.clamp(max=1.0)
        for c in range(n_clusters):
            m = a == c
            if m.any():
                step = lr[m].unsqueeze(1)
                centers[c] = (1 - step) * centers[c] + step * xb[m]
                counts[c] += float(m.sum())
    d = torch.cdist(x, centers)
    return d.argmin(dim=1)


def ccs_select(bundle: DatasetBundle, k: int,
               model_factory: Callable[[], nn.Module],
               device: torch.device, cfg: Optional[CCSConfig] = None,
               seeds: Optional[list] = None, **_) -> torch.Tensor:
    cfg = cfg or CCSConfig()
    models = train_proxy_models(bundle, model_factory, device,
                                ProxyConfig(epochs=cfg.epochs,
                                            batch_size=cfg.batch_size,
                                            optimizer=cfg.optimizer, lr=cfg.lr,
                                            momentum=cfg.momentum,
                                            scheduler=cfg.scheduler,
                                            num_models=1), seeds)
    model = models[0]
    logits, feats = forward_logits_features(model, bundle.train_x, device,
                                            cfg.batch_size)
    if cfg.normalize_features:
        feats = F.normalize(feats, dim=1)
    importance = F.cross_entropy(logits, bundle.train_y, reduction="none")

    n = bundle.n
    k = min(k, n)
    assign = _kmeans_torch(feats, k, cfg.kmeans_iters, cfg.kmeans_batch_size,
                           cfg.seed)

    selected = torch.zeros(n, dtype=torch.bool)
    for c in range(k):
        members = torch.nonzero(assign == c, as_tuple=False).flatten()
        if members.numel() == 0:
            continue
        best = members[importance[members].argmax()]
        selected[best] = True
    if int(selected.sum()) < k:
        remainder = torch.nonzero(~selected, as_tuple=False).flatten()
        order = remainder[torch.argsort(importance[remainder], descending=True)]
        selected[order[:k - int(selected.sum())]] = True
    return torch.nonzero(selected, as_tuple=False).flatten()

"""Shared proxy-model pipeline for the score-based baselines.

All score-based baselines read their scores off the *same* proxy model, so the
experiments train the proxy once per (dataset, repetition) and reuse it for
every coreset size ``k``.  :class:`BaselineSelector` exposes a uniform
``select(name, k)`` API for the seven baselines of Section 5.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import DatasetBundle
from .common import (ProxyConfig, class_balanced_top_k, forward_logits_features,
                     top_k_indices, train_proxy_models)
from .influential import InfluenceConfig, _last_layer_gradients
from .moderate import _moderate_indices_per_class


@dataclass
class ScoreConfig(ProxyConfig):
    num_models: int = 1
    influence_damping: float = 1e-3
    val_fraction: float = 0.1
    class_balanced: bool = True


class BaselineSelector:
    """Computes every baseline score from one set of proxy models."""

    def __init__(self, bundle: DatasetBundle,
                 model_factory: Callable[[], nn.Module],
                 device: torch.device,
                 cfg: Optional[ScoreConfig] = None, seed: int = 0):
        self.bundle = bundle
        self.device = device
        self.cfg = cfg or ScoreConfig()
        self.seed = seed
        self.models = train_proxy_models(
            bundle, model_factory, device,
            ProxyConfig(epochs=self.cfg.epochs, batch_size=self.cfg.batch_size,
                        optimizer=self.cfg.optimizer, lr=self.cfg.lr,
                        momentum=self.cfg.momentum, scheduler=self.cfg.scheduler,
                        num_models=self.cfg.num_models),
            seeds=list(range(seed, seed + self.cfg.num_models)))
        self._compute_scores()

    # ------------------------------------------------------------------
    def _compute_scores(self) -> None:
        bundle = self.bundle
        y = bundle.train_y
        onehot = F.one_hot(y, num_classes=bundle.num_classes).double()
        el2n = torch.zeros(bundle.n, dtype=torch.float64)
        grand = torch.zeros(bundle.n, dtype=torch.float64)
        feats_all: List[torch.Tensor] = []
        for model in self.models:
            logits, feats = forward_logits_features(
                model, bundle.train_x, self.device, self.cfg.batch_size)
            probs = F.softmax(logits, dim=1).double()
            err = (probs - onehot).norm(dim=1)
            el2n += err
            grand += err * torch.sqrt(1.0 + feats.double().norm(dim=1) ** 2)
            feats_all.append(feats)
        n_models = max(len(self.models), 1)
        self.el2n = (el2n / n_models).float()
        self.grand = (grand / n_models).float()
        self.feats = torch.cat(feats_all, dim=1) / n_models

        # Moderate: distance to the class center in feature space
        moderate = torch.zeros(bundle.n)
        for c in range(bundle.num_classes):
            idx_c = torch.nonzero(y == c, as_tuple=False).flatten()
            if idx_c.numel() == 0:
                continue
            center = self.feats[idx_c].mean(dim=0, keepdim=True)
            moderate[idx_c] = (self.feats[idx_c] - center).norm(dim=1)
        self.moderate = moderate

        # CCS importance: per-example loss of the first proxy model
        logits, _ = forward_logits_features(self.models[0], bundle.train_x,
                                            self.device, self.cfg.batch_size)
        self.importance = F.cross_entropy(logits, y, reduction="none").float()
        self.logits = logits

        # Influence function scores (last-layer Gauss-Newton approximation)
        grads = _last_layer_gradients(self.feats, logits, y, bundle.num_classes)
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(bundle.n, generator=g)
        n_val = max(1, int(round(self.cfg.val_fraction * bundle.n)))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        g_val = grads[val_idx].mean(dim=0)
        H = (grads[train_idx].T @ grads[train_idx]) / max(train_idx.numel(), 1)
        H = H + self.cfg.influence_damping * torch.eye(H.size(0),
                                                       dtype=H.dtype)
        x = torch.linalg.solve(H, g_val)
        self.influence = (-(grads @ x)).float()
        self.influence_magnitude = self.influence.abs()

    # ------------------------------------------------------------------
    def _balanced(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        if self.cfg.class_balanced:
            return class_balanced_top_k(scores, self.bundle.train_y,
                                        self.bundle.num_classes, k)
        return top_k_indices(scores, k)

    def select(self, name: str, k: int) -> torch.Tensor:
        name = name.lower()
        if name == "uniform":
            g = torch.Generator().manual_seed(self.seed)
            return torch.randperm(self.bundle.n, generator=g)[:k]
        if name == "el2n":
            return self._balanced(self.el2n, k)
        if name == "grand":
            return self._balanced(self.grand, k)
        if name == "moderate":
            return self._moderate(k)
        if name == "influential":
            return self._balanced(self.influence_magnitude, k)
        if name == "ccs":
            return self._ccs(k)
        raise KeyError(f"unknown score-based baseline '{name}'")

    # ------------------------------------------------------------------
    def _moderate(self, k: int) -> torch.Tensor:
        bundle = self.bundle
        counts = torch.bincount(bundle.train_y,
                                minlength=bundle.num_classes).float()
        fractions = counts / counts.sum()
        selected: List[torch.Tensor] = []
        for c in range(bundle.num_classes):
            idx_c = torch.nonzero(bundle.train_y == c, as_tuple=False).flatten()
            if idx_c.numel() == 0:
                continue
            k_c = max(1, int(round(float(fractions[c]) * k)))
            selected.append(_moderate_indices_per_class(self.moderate[idx_c],
                                                        idx_c, k_c))
        out = torch.cat(selected)
        if out.numel() > k:
            g = torch.Generator().manual_seed(self.seed)
            out = out[torch.randperm(out.numel(), generator=g)[:k]]
        return out

    # ------------------------------------------------------------------
    def _ccs(self, k: int, iters: int = 100, batch_size: int = 4096
             ) -> torch.Tensor:
        from .ccs import _kmeans_torch

        feats = F.normalize(self.feats, dim=1)
        n = self.bundle.n
        k = min(k, n)
        assign = _kmeans_torch(feats, k, iters, batch_size, self.seed)
        selected = torch.zeros(n, dtype=torch.bool)
        for c in range(k):
            members = torch.nonzero(assign == c, as_tuple=False).flatten()
            if members.numel() == 0:
                continue
            selected[members[self.importance[members].argmax()]] = True
        if int(selected.sum()) < k:
            remainder = torch.nonzero(~selected, as_tuple=False).flatten()
            order = remainder[torch.argsort(self.importance[remainder],
                                            descending=True)]
            selected[order[:k - int(selected.sum())]] = True
        return torch.nonzero(selected, as_tuple=False).flatten()

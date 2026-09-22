"""Shared helpers for the score-based baselines.

All score-based methods (EL2N, GraNd, Moderate, CCS) need a *proxy* model that
is trained once on the full training set; the scores are then read off the
proxy model.  We also provide a helper that captures the input of the last
linear layer (the penultimate "feature" representation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import DatasetBundle
from ..inner_loop import InnerLoopConfig, train_on_coreset
from ..utils import make_loader, set_seed


@dataclass
class ProxyConfig:
    epochs: int = 100
    batch_size: int = 128
    optimizer: str = "adam"
    lr: float = 1e-3
    momentum: float = 0.9
    scheduler: Optional[str] = None
    num_models: int = 1        # EL2N / GraNd average the scores over models


def train_proxy_models(bundle: DatasetBundle,
                       model_factory: Callable[[], nn.Module],
                       device: torch.device, cfg: ProxyConfig,
                       seeds: Optional[List[int]] = None) -> List[nn.Module]:
    """Train ``cfg.num_models`` proxy models on the full training set."""
    if seeds is None:
        seeds = list(range(cfg.num_models))
    inner = InnerLoopConfig(epochs=cfg.epochs, batch_size=cfg.batch_size,
                            optimizer=cfg.optimizer, lr=cfg.lr,
                            momentum=cfg.momentum, scheduler=cfg.scheduler,
                            warm_start=False)
    all_indices = torch.arange(bundle.n)
    models = []
    for s in seeds:
        set_seed(s)
        model = model_factory()
        model = train_on_coreset(bundle, all_indices, model, inner, device,
                                 seed=s)
        models.append(model)
    return models


def last_linear_layer(model: nn.Module) -> nn.Module:
    """Return the final ``nn.Linear`` of a model (the classification head)."""
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise ValueError("model has no nn.Linear layer")
    return last


@torch.no_grad()
def forward_logits_features(model: nn.Module, x: torch.Tensor,
                            device: torch.device, batch_size: int = 512
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(logits, penultimate features)`` for all inputs ``x``."""
    head = last_linear_layer(model)
    captured: List[torch.Tensor] = []

    def hook(_module, inputs, _output):
        captured.append(inputs[0].detach())

    handle = head.register_forward_hook(hook)
    model.eval()
    logits_all, feats_all = [], []
    try:
        for i in range(0, x.size(0), batch_size):
            xb = x[i:i + batch_size].to(device)
            logits = model(xb)
            logits_all.append(logits.detach().cpu())
            feats_all.append(captured[-1].detach().cpu())
    finally:
        handle.remove()
    return torch.cat(logits_all), torch.cat(feats_all)


def top_k_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the ``k`` largest scores."""
    k = min(k, scores.numel())
    return torch.topk(scores, k, largest=True).indices


def bottom_k_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, scores.numel())
    return torch.topk(scores, k, largest=False).indices


def class_balanced_top_k(scores: torch.Tensor, labels: torch.Tensor,
                         num_classes: int, k: int) -> torch.Tensor:
    """Select ``k`` examples keeping the class proportions of the full set."""
    selected = []
    counts = torch.bincount(labels, minlength=num_classes).float()
    fractions = counts / counts.sum()
    for c in range(num_classes):
        idx_c = torch.nonzero(labels == c, as_tuple=False).flatten()
        if idx_c.numel() == 0:
            continue
        k_c = int(round(float(fractions[c]) * k))
        k_c = max(1, min(k_c, idx_c.numel()))
        order = torch.argsort(scores[idx_c], descending=True)
        selected.append(idx_c[order[:k_c]])
    out = torch.cat(selected)
    if out.numel() > k:
        out = out[:k]
    return out

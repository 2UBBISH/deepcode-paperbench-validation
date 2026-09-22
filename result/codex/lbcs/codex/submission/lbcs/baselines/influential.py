"""Influential coreset (Yang et al., ICLR 2023) baseline.

"This algorithm utilizes the influence function (Hampel, 1974).  The examples
that yield strictly constrained generalization gaps are included in the
coreset." (Appendix D.1)

Implementation.  Following the influence-function literature, the influence of
the training example ``z_i`` on the validation loss is

    I(z_i) = - grad_theta L_val^T  H_theta^{-1}  grad_theta l(z_i)

with ``H_theta`` the Hessian of the (mean) training loss.  We use the
Gauss-Newton / Fisher approximation of the Hessian, ``H = 1/n sum_i g_i g_i^T
+ lambda I``, where ``g_i`` is the per-example gradient of the loss with
respect to the parameters of the last linear layer.  Because the classification
head is linear the per-example gradients have a closed form
``g_i = [(p_i - y_i) h_i^T, (p_i - y_i)]`` and the linear system
``(H + lambda I) x = g_val`` can be solved exactly (the head has only a few
thousand parameters for the networks used here).

The reference implementation prunes the examples whose influence is *not*
large enough to change the generalization gap; equivalently it keeps the most
influential examples first.  ``criterion`` selects between keeping the largest
influence magnitude (default) and keeping the examples that are most helpful
for the validation loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import DatasetBundle
from .common import (ProxyConfig, class_balanced_top_k, forward_logits_features,
                     top_k_indices, train_proxy_models)


@dataclass
class InfluenceConfig(ProxyConfig):
    damping: float = 1e-3
    val_fraction: float = 0.1
    criterion: str = "magnitude"      # "magnitude" | "helpful"


def _last_layer_gradients(feats: torch.Tensor, logits: torch.Tensor,
                          labels: torch.Tensor, num_classes: int
                          ) -> torch.Tensor:
    """``g_i`` for a linear head: ``[(p_i - y_i) h_i^T, (p_i - y_i)]``."""
    probs = F.softmax(logits.double(), dim=1)
    onehot = F.one_hot(labels, num_classes=num_classes).double()
    err = probs - onehot                                  # n x C
    n, d = feats.shape
    weight_grad = (err[:, :, None] * feats.double()[:, None, :]).reshape(n, -1)
    return torch.cat([weight_grad, err], dim=1)           # n x ((d+1) * C)


def compute_influence_scores(bundle: DatasetBundle,
                             model_factory: Callable[[], nn.Module],
                             device: torch.device,
                             cfg: Optional[InfluenceConfig] = None,
                             seeds: Optional[list] = None,
                             generator: Optional[torch.Generator] = None
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(scores, proxy_model)`` for every training example."""
    cfg = cfg or InfluenceConfig()
    models = train_proxy_models(bundle, model_factory, device,
                                ProxyConfig(epochs=cfg.epochs,
                                            batch_size=cfg.batch_size,
                                            optimizer=cfg.optimizer, lr=cfg.lr,
                                            momentum=cfg.momentum,
                                            scheduler=cfg.scheduler,
                                            num_models=1),
                                seeds=seeds)
    model = models[0]
    logits, feats = forward_logits_features(model, bundle.train_x, device,
                                            cfg.batch_size)
    grads = _last_layer_gradients(feats, logits, bundle.train_y,
                                  bundle.num_classes)

    if generator is None:
        generator = torch.Generator().manual_seed(0)
    n = bundle.n
    perm = torch.randperm(n, generator=generator)
    n_val = max(1, int(round(cfg.val_fraction * n)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    g_val = grads[val_idx].mean(dim=0)
    H = (grads[train_idx].T @ grads[train_idx]) / max(train_idx.numel(), 1)
    D = H.size(0)
    H = H + cfg.damping * torch.eye(D, dtype=H.dtype)
    x = torch.linalg.solve(H, g_val)
    scores = -(grads @ x)                 # influence of upweighting on L_val
    return scores.float(), model


def influential_select(bundle: DatasetBundle, k: int,
                       model_factory: Callable[[], nn.Module],
                       device: torch.device,
                       cfg: Optional[InfluenceConfig] = None,
                       seeds: Optional[list] = None,
                       class_balanced: bool = True,
                       **_) -> torch.Tensor:
    cfg = cfg or InfluenceConfig()
    scores, _ = compute_influence_scores(bundle, model_factory, device, cfg,
                                         seeds)
    if cfg.criterion == "helpful":
        scores = -scores                 # most helpful for the validation loss
    else:
        scores = scores.abs()
    if class_balanced:
        return class_balanced_top_k(scores, bundle.train_y, bundle.num_classes, k)
    return top_k_indices(scores, k)

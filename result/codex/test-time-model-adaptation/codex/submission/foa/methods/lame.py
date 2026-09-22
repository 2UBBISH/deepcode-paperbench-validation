"""LAME (Boudiaf et al., 2022) - parameter-free online adaptation of the output
probabilities with a kNN affinity graph and kernel label propagation.

Official implementation: https://github.com/fiveai/LAME
Appendix B.2 of the FOA paper: batch size 64 and ``k = 5`` for the kNN affinity matrix
(chosen on ImageNet-C among {1, 5, 10, 20}).

Closed form used by LAME::

    Z* = (I + alpha * L)^{-1} Z,       L = D - W,

where ``W`` is the Gaussian kNN affinity between (L2-normalised) features, ``D`` its
degree matrix and ``Z`` the source logits.  Predictions are ``softmax(Z* / T)``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import TTAMethod, features_and_logits


@torch.no_grad()
def knn_affinity(features: torch.Tensor, k: int = 5) -> torch.Tensor:
    n = features.shape[0]
    feats = F.normalize(features.float(), dim=1)
    dist = torch.cdist(feats, feats)
    k_eff = min(k, n - 1)
    if k_eff <= 0:
        return torch.eye(n, device=features.device)
    topk = dist.topk(k_eff + 1, largest=False, sorted=True)
    knn_dist, knn_idx = topk.values[:, 1:], topk.indices[:, 1:]
    sigma = knn_dist[:, -1].clamp_min(1e-8)[:, None]
    w = torch.exp(-(knn_dist**2) / (2 * sigma**2))
    W = torch.zeros_like(dist)
    W.scatter_(1, knn_idx, w)
    W = 0.5 * (W + W.t())
    return W


@torch.no_grad()
def lame_correct(
    logits: torch.Tensor,
    features: torch.Tensor,
    k: int = 5,
    alpha: float = 0.5,
    temperature: float = 1.0,
) -> torch.Tensor:
    W = knn_affinity(features, k=k)
    d = W.sum(dim=1)
    L = torch.diag(d) - W
    eye = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
    Z = logits.float()
    Z = Z - Z.max(dim=1, keepdim=True).values  # numerical stabilisation
    adjusted = torch.linalg.solve(eye + alpha * L, Z)
    return F.softmax(adjusted / temperature, dim=1)


class LAME(TTAMethod):
    name = "LAME"

    def __init__(
        self,
        model,
        k: int = 5,
        alpha: float = 0.5,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(model, device=device)
        self.model.eval()
        self.k = k
        self.alpha = alpha
        self.temperature = temperature
        self.last_extra = {}

    @torch.no_grad()
    def step(self, images: torch.Tensor) -> torch.Tensor:
        feats, logits = features_and_logits(self.model, images.to(self.device))
        probs = lame_correct(
            logits, feats, k=self.k, alpha=self.alpha, temperature=self.temperature
        )
        # return log-probabilities; ``argmax`` / ``max`` are unaffected by the transform
        return torch.log(probs.clamp_min(1e-12))

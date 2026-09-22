"""T3A (Iwasawa & Matsuo, 2021) - test-time prototype-based classifier adjustment.

Official implementation: https://github.com/matsuolab/T3A
Appendix B.2 of the FOA paper: batch size 64 and ``M = 20`` supports to restore.

The classifier is replaced at test time by the cosine similarity between the query
feature and the supports of every class (the score of a class is the *largest* similarity
among its supports).  Supports are filtered down to ``filter_K`` and then to ``M`` entries
that are most similar to the class prototype.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import TTAMethod, features_and_logits


class T3A(TTAMethod):
    name = "T3A"

    def __init__(
        self,
        model,
        num_classes: int = 1000,
        filter_k: int = 100,
        num_supports: int = 20,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(model, device=device)
        self.model.eval()
        self.num_classes = num_classes
        self.filter_k = filter_k
        self.M = num_supports
        self.last_extra = {}
        self.reset()

    def reset(self) -> None:
        self.supports = None      # [num_classes, num_supports, d] kept as a list of tensors

    # ----------------------------------------------------------------------------------
    @torch.no_grad()
    def _score(self, features: torch.Tensor) -> torch.Tensor:
        assert self.supports is not None
        q = F.normalize(features, dim=1)
        logits = torch.full(
            (features.shape[0], self.num_classes), -1e4, device=features.device
        )
        for c, sup in enumerate(self.supports):
            if sup is None or sup.numel() == 0:
                continue
            s = F.normalize(sup, dim=1)
            logits[:, c] = (q @ s.t()).max(dim=1).values
        return logits

    @torch.no_grad()
    def _update_supports(self, features: torch.Tensor, logits: torch.Tensor) -> None:
        labels = logits.argmax(dim=1)
        if self.supports is None:
            self.supports = [None] * self.num_classes
        for i, label in enumerate(labels.tolist()):
            f = features[i : i + 1]
            if self.supports[label] is None:
                self.supports[label] = f
            else:
                self.supports[label] = torch.cat([self.supports[label], f], dim=0)
        for c in range(self.num_classes):
            sup = self.supports[c]
            if sup is None:
                continue
            for keep in (self.filter_k, self.M):
                if sup.shape[0] > keep:
                    proto = F.normalize(sup.mean(dim=0, keepdim=True), dim=1)
                    sims = (F.normalize(sup, dim=1) @ proto.t()).squeeze(1)
                    idx = sims.topk(keep, largest=True).indices
                    sup = sup[idx]
            self.supports[c] = sup

    # ----------------------------------------------------------------------------------
    @torch.no_grad()
    def step(self, images: torch.Tensor) -> torch.Tensor:
        feats, logits = features_and_logits(self.model, images.to(self.device))
        if self.supports is None:
            # first batch bootstraps the supports with the source classifier
            output = logits
        else:
            output = self._score(feats)
        self._update_supports(feats, output)
        return output

"""Adapter turning the FOA core algorithm into a ``TTAMethod``."""
from __future__ import annotations

from typing import Optional

import torch

from ..config import FOAConfig
from ..core.foa import FOA, FOAInterval
from ..core.statistics import FeatureStatistics
from .base import TTAMethod


class FOAMethod(TTAMethod):
    """FOA (batch-size > 1) or FOA-I (interval update for single samples)."""

    name = "FOA"

    def __init__(
        self,
        model,
        source_stats: FeatureStatistics,
        cfg: Optional[FOAConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(model, device=device)
        self.cfg = cfg or FOAConfig()
        self.last_extra = {}
        if self.cfg.interval and self.cfg.interval > 1:
            self.impl = FOAInterval(self.model, source_stats, cfg=self.cfg, device=self.device)
            self.name = f"FOA-I (I={self.cfg.interval}, {self.cfg.interval_store})"
        else:
            self.impl = FOA(self.model, source_stats, cfg=self.cfg, device=self.device)

    def reset(self) -> None:
        self.impl.reset()
        self.last_extra = {}

    @torch.no_grad()
    def step(self, images: torch.Tensor) -> Optional[torch.Tensor]:
        out = self.impl.step(images)
        if out is None:
            return None
        self.last_extra = dict(out.extra)
        return out.logits

    @torch.no_grad()
    def flush(self) -> Optional[torch.Tensor]:
        out = self.impl.flush()
        if out is None:
            return None
        self.last_extra = dict(out.extra)
        return out.logits

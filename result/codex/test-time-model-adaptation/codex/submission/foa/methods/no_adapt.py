"""NoAdapt: the frozen source model, no adaptation at all."""
from __future__ import annotations

from typing import Optional

import torch

from .base import TTAMethod


class NoAdapt(TTAMethod):
    name = "NoAdapt"

    def __init__(self, model, device: Optional[torch.device] = None) -> None:
        super().__init__(model, device=device)
        self.last_extra = {}
        self.model.eval()

    @torch.no_grad()
    def step(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images.to(self.device))

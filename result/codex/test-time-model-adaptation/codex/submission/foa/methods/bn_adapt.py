"""BN Adapt (Schneider et al., 2020 / Nado et al., 2020) - re-estimate the BatchNorm
statistics on the test batch.  Used as a baseline for the ResNet-50 experiment
(Table 10 of the FOA paper).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base import TTAMethod, get_backbone


class BNAdapt(TTAMethod):
    name = "BN Adapt"

    def __init__(
        self,
        model,
        momentum: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(model, device=device)
        self.momentum = momentum
        self.last_extra = {}
        self.model.eval()
        self.num_bn = 0
        for m in get_backbone(self.model).modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.train()  # use the statistics of the incoming test batch
                if momentum is not None:
                    m.momentum = momentum
                self.num_bn += 1

    @torch.no_grad()
    def step(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images.to(self.device))

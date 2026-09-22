"""SAR (Niu et al., 2023) - sharpness-aware, reliable-sample entropy minimisation.

Official implementation: https://github.com/mr-eggplant/SAR
Appendix B.2 of the FOA paper: SGD with momentum 0.9, batch size 64, learning rate 1e-3,
entropy threshold ``E_0 = 0.4 * ln C`` and trainable parameters = the affine parameters
of the layer normalisation layers of blocks 1..8 of ViT-Base.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..config import sar_entropy_threshold
from ..core.fitness import prediction_entropy
from .base import TTAMethod, get_backbone


class SAR(TTAMethod):
    name = "SAR"

    def __init__(
        self,
        model,
        lr: float = 1e-3,
        momentum: float = 0.9,
        rho: float = 0.05,
        entropy_threshold: Optional[float] = None,
        device: Optional[torch.device] = None,
        num_classes: int = 1000,
        layer_selection: str = "blocks_1_8",
    ) -> None:
        super().__init__(model, device=device)
        self.last_extra = {}
        self.rho = rho
        self.threshold = (
            entropy_threshold if entropy_threshold is not None
            else sar_entropy_threshold(num_classes)
        )
        self.params = self._configure(layer_selection)
        self.optimizer = torch.optim.SGD(self.params, lr=lr, momentum=momentum)

    # ----------------------------------------------------------------------------------
    def _configure(self, selection: str) -> List[nn.Parameter]:
        self.model.requires_grad_(False)
        backbone = get_backbone(self.model)
        blocks = getattr(backbone, "blocks", None)
        params: List[nn.Parameter] = []
        if blocks is not None and selection.startswith("blocks"):
            first, last = 1, 8
            if "_" in selection:
                spec = selection.split("_")[-2:]
                first, last = int(spec[0]), int(spec[1])
            for idx, blk in enumerate(blocks, start=1):
                if first <= idx <= last:
                    for m in blk.modules():
                        if isinstance(m, nn.LayerNorm):
                            m.requires_grad_(True)
                            m.train()
                            params.extend([p for p in m.parameters() if p.requires_grad])
        else:
            for m in backbone.modules():
                if isinstance(m, nn.LayerNorm):
                    m.requires_grad_(True)
                    m.train()
                    params.extend([p for p in m.parameters() if p.requires_grad])
        assert params, "SAR did not find any normalisation parameters"
        return params

    # ----------------------------------------------------------------------------------
    def _entropy(self, logits: torch.Tensor) -> torch.Tensor:
        return prediction_entropy(logits, reduction="none")

    def step(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        with torch.no_grad():
            output = self.model(images)
            entropies = self._entropy(output)
            mask = entropies < self.threshold
            num_reliable = int(mask.sum().item())
        self.last_extra = {"reliable_frac": num_reliable / max(1, images.shape[0])}
        if num_reliable == 0:
            return output.detach()

        x_sel = images[mask]

        # --- first SAM step: perturb the parameters along the gradient direction ------
        self.optimizer.zero_grad()
        loss = self._entropy(self.model(x_sel)).mean()
        loss.backward()
        grad_norm = torch.norm(
            torch.stack([p.grad.norm() for p in self.params if p.grad is not None])
        )
        e_w = [
            (self.rho * p.grad / (grad_norm + 1e-12)) if p.grad is not None else None
            for p in self.params
        ]
        with torch.no_grad():
            for p, e in zip(self.params, e_w):
                if e is not None:
                    p.add_(e)

        # --- second SAM step: gradient at the perturbed point -------------------------
        self.optimizer.zero_grad()
        loss_adv = self._entropy(self.model(x_sel)).mean()
        loss_adv.backward()
        with torch.no_grad():
            for p, e in zip(self.params, e_w):
                if e is not None:
                    p.sub_(e)
        self.optimizer.step()
        self.last_extra["loss"] = float(loss_adv.item())
        return output.detach()

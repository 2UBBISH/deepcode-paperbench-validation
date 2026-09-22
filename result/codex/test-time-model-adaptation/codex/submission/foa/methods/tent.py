"""TENT (Wang et al., 2021) - entropy minimisation of the LayerNorm affine parameters.

Official implementation: https://github.com/DequanWang/tent
Hyper-parameters used here are the ones quoted in Appendix B.2 of the FOA paper:
SGD, momentum 0.9, batch size 64, learning rate 1e-3, trainable parameters = all affine
parameters of the layer normalisation layers.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..core.fitness import prediction_entropy
from .base import TTAMethod, get_backbone


def collect_norm_parameters(model: nn.Module, layer_selection: str = "all"):
    """Enable gradients on LayerNorm/BatchNorm affine parameters (TENT's ``configure_model``)."""
    model.requires_grad_(False)
    params = []
    backbone = get_backbone(model)
    for name, m in backbone.named_modules():
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            if isinstance(m, nn.LayerNorm) or getattr(m, "elementwise_affine", False):
                m.requires_grad_(True)
                params.extend([p for p in m.parameters() if p.requires_grad])
            m.train()  # TENT keeps the normalisation layers in "train" mode
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            m.train()
            params.extend([p for p in m.parameters() if p.requires_grad])
    return params


class TENT(TTAMethod):
    name = "TENT"

    def __init__(
        self,
        model,
        lr: float = 1e-3,
        momentum: float = 0.9,
        device: Optional[torch.device] = None,
        layer_selection: str = "all",
    ) -> None:
        super().__init__(model, device=device)
        self.last_extra = {}
        self.params = collect_norm_parameters(self.model, layer_selection)
        self.optimizer = torch.optim.SGD(self.params, lr=lr, momentum=momentum)

    def step(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        self.optimizer.zero_grad()
        logits = self.model(images)
        # softmax_entropy(output).mean(0).sum() in the official code == mean batch entropy
        loss = prediction_entropy(logits, reduction="mean")
        loss.backward()
        self.optimizer.step()
        self.last_extra = {"entropy": float(loss.item())}
        # the official implementation returns the prediction of the *pre-update* step
        return logits.detach()

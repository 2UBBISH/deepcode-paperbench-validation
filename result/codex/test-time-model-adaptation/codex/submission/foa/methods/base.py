"""Common interface for every test-time adaptation method (FOA and the baselines)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TTAMethod:
    """Online test-time adaptation method.

    ``step(images)`` returns the predictions for the *oldest unassigned* test samples.
    Returning ``None`` means "the prediction is deferred", which is how the interval
    strategies (``FOA-I``) signal that they are still collecting samples.  At the end of
    the stream the runner calls :meth:`flush` to obtain the predictions of the remaining
    samples.
    """

    name: str = "method"
    needs_labels: bool = False

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)

    def reset(self) -> None:
        """Prepare for a new test stream (re-initialise state / model weights)."""

    def step(self, images: torch.Tensor) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def flush(self) -> Optional[torch.Tensor]:
        """Predictions for the samples that are still pending (default: none)."""
        return None

    # -- helpers ---------------------------------------------------------------------
    def param_groups(self):
        return [p for p in self.model.parameters() if p.requires_grad]


def get_backbone(model: nn.Module) -> nn.Module:
    """Return the underlying ``timm`` model when ``model`` is a FOA wrapper."""
    return getattr(model, "vit", getattr(model, "backbone", model))


def set_requires_grad(model: nn.Module, requires_grad: bool = False) -> None:
    for p in model.parameters():
        p.requires_grad_(requires_grad)


@torch.no_grad()
def features_and_logits(model: nn.Module, images: torch.Tensor):
    """Return ``(e_N^0, logits)`` - the pre-logits CLS activation and the prediction.

    T3A and LAME need the penultimate feature, which for a ViT is exactly the CLS
    activation fed to the task head.
    """
    if hasattr(model, "forward_with_prompt"):
        logits, e_n, _ = model.forward_with_prompt(images)
        return e_n, logits
    feats = model.forward_features(images)          # timm ViT / ConvNet in eval mode
    e_n = model.forward_head(feats, pre_logits=True)
    logits = model.head(e_n) if hasattr(model, "head") else e_n
    return e_n, logits

"""Source in-distribution statistics ``{mu_i^S, sigma_i^S}_{i=0..N}``.

Section 3.1 ("Statistics calculation"): a small set of *unlabelled* source images is
forwarded through the frozen model and the mean / standard deviation of the CLS token of
every layer is recorded.  32 samples are enough for ImageNet (Appendix B.2) and the
statistics are computed *without* the newly inserted prompt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from ..models.base import AdaptableModel


@dataclass
class FeatureStatistics:
    """Per-layer mean / std of the CLS tokens."""

    means: List[torch.Tensor]
    stds: List[torch.Tensor]

    def to(self, device, dtype: Optional[torch.dtype] = None) -> "FeatureStatistics":
        self.means = [m.to(device=device, dtype=dtype) for m in self.means]
        self.stds = [s.to(device=device, dtype=dtype) for s in self.stds]
        return self

    @property
    def num_layers(self) -> int:
        return len(self.means)

    def save(self, path: str) -> None:
        torch.save({"means": self.means, "stds": self.stds}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "FeatureStatistics":
        blob = torch.load(path, map_location=map_location)
        return cls(blob["means"], blob["stds"])


@torch.no_grad()
def compute_source_statistics(
    model: AdaptableModel,
    batches: Sequence[torch.Tensor],
    device: Optional[torch.device] = None,
) -> FeatureStatistics:
    """Estimate ``{mu_i^S, sigma_i^S}`` from a (small) set of source images.

    Args:
        model: frozen backbone.
        batches: a few batches of *source in-distribution* images.  The paper uses 32
            unlabelled ImageNet validation images in total.
    Note:
        The paper computes the statistics "without using the newly inserted prompt"
        (Appendix B.2), hence no prompt is injected here.
    """
    if not batches:
        raise ValueError("at least one batch of source images is required")
    device = device or next(model.parameters()).device
    sum_ = None
    sum_sq = None
    count = 0
    for images in batches:
        images = images.to(device)
        _, _, feats = model.forward_with_prompt(images, prompt=None, return_layers=True)
        assert feats is not None
        if sum_ is None:
            sum_ = [torch.zeros(f.shape[1], dtype=torch.float64, device=f.device) for f in feats]
            sum_sq = [torch.zeros(f.shape[1], dtype=torch.float64, device=f.device) for f in feats]
        for i, f in enumerate(feats):
            f64 = f.double()
            sum_[i] += f64.sum(dim=0)
            sum_sq[i] += (f64**2).sum(dim=0)
        count += feats[0].shape[0]
    means = [s / count for s in sum_]
    var = [sum_sq[i] / count - means[i] ** 2 for i in range(len(means))]
    var = [v.clamp_min(0) for v in var]
    stds = [v.sqrt() for v in var]
    return FeatureStatistics([m.float() for m in means], [s.float() for s in stds])


def stacking_collate(images: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack(list(images), dim=0)

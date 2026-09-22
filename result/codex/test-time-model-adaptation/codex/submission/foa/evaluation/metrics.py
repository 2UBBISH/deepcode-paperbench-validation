"""Evaluation metrics quoted in the paper.

* Classification accuracy (%, higher is better)
* Expected Calibration Error, ECE (%, lower is better) - Eqn. of Naeini et al. (2015),
  implemented with the usual 15 equal-width confidence bins.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Percentage of correct predictions."""
    if logits.numel() == 0:
        return float("nan")
    pred = logits.argmax(dim=-1)
    return 100.0 * (pred == targets).float().mean().item()


def expected_calibration_error(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
    return_bins: bool = False,
):
    """Expected Calibration Error in percent.

    ``ECE = sum_b (n_b / N) * |acc(b) - conf(b)|`` computed on the top-1 confidence.
    """
    if logits.numel() == 0:
        return (float("nan"), None) if return_bins else float("nan")
    probs = F.softmax(logits.float(), dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = (pred == targets).float()
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=logits.device)
    ece = torch.zeros((), dtype=torch.float64, device=logits.device)
    n = logits.shape[0]
    stats = []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.any():
            bin_acc = correct[mask].mean().double()
            bin_conf = conf[mask].mean().double()
            frac = mask.double().sum() / n
            ece += frac * (bin_acc - bin_conf).abs()
            stats.append((float(lo), int(mask.sum().item()), float(bin_acc), float(bin_conf)))
    value = float(ece.item() * 100.0)
    if return_bins:
        return value, stats
    return value


class OnlineMetrics:
    """Streaming accuracy / ECE (the paper reports accuracy and ECE over the stream)."""

    def __init__(self, n_bins: int = 15) -> None:
        self.n_bins = n_bins
        self.logits: list = []
        self.targets: list = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        if logits is None or logits.numel() == 0:
            return
        self.logits.append(logits.detach().float().cpu())
        self.targets.append(targets.detach().cpu())

    def _cat(self) -> tuple:
        if not self.logits:
            return torch.empty(0, 1), torch.empty(0, dtype=torch.long)
        return torch.cat(self.logits, 0), torch.cat(self.targets, 0)

    @property
    def num_samples(self) -> int:
        return int(sum(t.numel() for t in self.targets))

    def compute(self) -> dict:
        logits, targets = self._cat()
        return {
            "acc": accuracy(logits, targets),
            "ece": expected_calibration_error(logits, targets, self.n_bins),
            "num_samples": self.num_samples,
        }


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else float("nan")


def summarize(results: dict, key: str = "acc") -> dict:
    """Average a metric over the 15 ImageNet-C corruptions (or any dict of results)."""
    values = [v[key] for v in results.values() if v.get(key) is not None]
    return {
        "mean": float(np.mean(values)) if values else float("nan"),
        "std": float(np.std(values)) if values else float("nan"),
        "num": len(values),
    }

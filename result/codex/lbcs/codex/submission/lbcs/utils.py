"""Small shared helpers: seeding, batching, evaluation and training loops."""

from __future__ import annotations

import contextlib
import os
import random
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def num_workers() -> int:
    """Data-loader workers; kept small because everything is in-memory."""
    return int(os.environ.get("LBCS_NUM_WORKERS", "0"))


def make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int,
                shuffle: bool = True, generator: Optional[torch.Generator] = None,
                drop_last: bool = False) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers(), generator=generator)


def make_loader_from_indices(x: torch.Tensor, y: torch.Tensor,
                             indices: torch.Tensor, batch_size: int,
                             shuffle: bool = True,
                             generator: Optional[torch.Generator] = None,
                             drop_last: bool = False) -> DataLoader:
    return make_loader(x[indices], y[indices], batch_size, shuffle, generator,
                       drop_last)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader,
                      device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += yb.numel()
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader,
                  device: torch.device, reduction: str = "mean") -> float:
    """Cross-entropy loss of ``model`` over ``loader``.

    With ``reduction='mean'`` this is exactly the objective

        f_1(m) = (1 / n) sum_i l(h(x_i; theta(m)), y_i)

    of the paper (the loss of the coreset-trained model on the *full* data).
    """
    model.eval()
    total, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, reduction="sum")
        total += float(loss.item())
        count += yb.numel()
    if reduction == "mean":
        return total / max(count, 1)
    return total


def accuracy_on_tensor(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                       device: torch.device, batch_size: int = 512) -> float:
    loader = make_loader(x, y, batch_size, shuffle=False)
    return evaluate_accuracy(model, loader, device)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer,
                    device: torch.device, max_steps: Optional[int] = None,
                    step_counter: Optional[list] = None,
                    grad_clip: Optional[float] = None) -> None:
    model.train()
    for i, (xb, yb) in enumerate(loader):
        if max_steps is not None and i >= max_steps:
            break
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if step_counter is not None:
            step_counter[0] += 1


def make_optimizer(model: nn.Module, name: str, lr: float,
                   momentum: float = 0.9, weight_decay: float = 0.0):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                               weight_decay=weight_decay)
    raise KeyError(f"unknown optimizer '{name}'")


class CosineScheduler:
    """Cosine annealing over a fixed number of *epochs* (step granularity=1)."""

    def __init__(self, optimizer, total_epochs: int, warmup_epochs: int = 0):
        self.optimizer = optimizer
        self.total_epochs = max(total_epochs, 1)
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / max(self.warmup_epochs, 1)
        else:
            t = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1)
            scale = 0.5 * (1.0 + np.cos(np.pi * min(t, 1.0)))
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base * scale


@contextlib.contextmanager
def evaluating(model: nn.Module):
    """Temporarily switch ``model`` to eval mode."""
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)


def mask_to_indices(mask: torch.Tensor) -> torch.Tensor:
    """Indices of the selected examples of a 0/1 mask."""
    return torch.nonzero(mask.view(-1) > 0.5, as_tuple=False).flatten()


def indices_to_mask(indices: torch.Tensor, n: int,
                    device=None) -> torch.Tensor:
    mask = torch.zeros(n, dtype=torch.float32, device=device)
    mask[indices] = 1.0
    return mask


def discretize(continuous_mask: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Project a continuous mask in ``[-1, 1]`` onto ``{0, 1}``.

    Following Appendix A of the paper: values smaller than -1 are clamped to
    -1 and values larger than 1 are clamped to 1; then values in ``[-1, 0)``
    are projected to 0 and values in ``[0, 1]`` are projected to 1.
    """
    clamped = continuous_mask.clamp(-1.0, 1.0)
    return (clamped >= threshold).to(torch.float32)


def chunks(seq: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

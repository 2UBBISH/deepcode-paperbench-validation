"""Utility functions for SAPG: KL-adaptive learning rate, gradient clipping,
logging, seeding, and small tensor helpers.

These utilities implement the optimization details described in the SAPG paper
(Appendix B / Section 4.6):

    * Adam optimizer with per-task learning rates (1e-4 AllegroKuka, 5e-4 hands).
    * KL-adaptive learning rate with threshold 0.016 and factor 1.5.
    * Global gradient-norm clipping at 1.0.
    * Deterministic seeding for reproducibility.
    * Lightweight running-mean logging helpers.
"""

from __future__ import annotations

import logging
import os
import random
from collections import deque
from typing import Any, Deque, Dict, Iterable, Optional

import numpy as np
import torch

__all__ = [
    "set_seed",
    "get_logger",
    "RunningMeanStd",
    "RunningMean",
    "kl_adaptive_learning_rate",
    "clip_grad_norm_",
    "explained_variance",
    "to_tensor",
    "flatten_dict",
    "AverageMeter",
    "count_parameters",
    "linear_schedule",
    "get_device",
]


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch RNGs for reproducibility.

    Args:
        seed: The integer seed.
        deterministic: If True, request deterministic cuDNN algorithms
            (slower, but reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str = "sapg", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with a stream handler (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Running statistics
# ---------------------------------------------------------------------------
class RunningMeanStd:
    """Welford-style running mean/variance tracker (for observation normalization)."""

    def __init__(self, shape: Iterable[int] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0] if x.ndim > 0 else 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count


class RunningMean:
    """Simple exponential/arithmetic running mean for scalar logging."""

    def __init__(self, window: int = 100):
        self.window = window
        self.values: Deque[float] = deque(maxlen=window)

    def update(self, value: float) -> None:
        self.values.append(float(value))

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else 0.0

    def reset(self) -> None:
        self.values.clear()


class AverageMeter:
    """Tracks the running average of a scalar quantity."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


# ---------------------------------------------------------------------------
# KL-adaptive learning rate
# ---------------------------------------------------------------------------
def kl_adaptive_learning_rate(
    current_lr: float,
    approx_kl: float,
    kl_threshold: float = 0.016,
    factor: float = 1.5,
    min_lr: float = 1e-6,
    max_lr: float = 1e-2,
) -> float:
    """Adjust the learning rate based on the measured approximate KL.

    Following the SAPG paper (and common PPO implementations), if the KL
    divergence exceeds ``kl_threshold`` the learning rate is divided by
    ``factor``; if it falls below half the threshold it is multiplied by
    ``factor``. The result is clamped to ``[min_lr, max_lr]``.

    Args:
        current_lr: The learning rate used for the current iteration.
        approx_kl: Mean approximate KL between old and new policies.
        kl_threshold: KL threshold (0.016 in the paper).
        factor: Multiplicative adjustment factor (1.5 in the paper).
        min_lr: Lower clamp bound.
        max_lr: Upper clamp bound.

    Returns:
        The learning rate to use for the next iteration.
    """
    if approx_kl > kl_threshold:
        new_lr = current_lr / factor
    elif approx_kl < 0.5 * kl_threshold:
        new_lr = current_lr * factor
    else:
        new_lr = current_lr
    return float(min(max(new_lr, min_lr), max_lr))


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------
def clip_grad_norm_(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> float:
    """Clip gradients of an iterable of parameters in-place.

    Thin wrapper around :func:`torch.nn.utils.clip_grad_norm_` that filters
    out parameters with ``None`` gradients and returns the total norm.

    Args:
        parameters: Iterable of parameters (or param groups).
        max_norm: Maximum allowed gradient norm (1.0 in the paper).
        norm_type: Norm type (2.0 = L2).

    Returns:
        The total gradient norm *before* clipping.
    """
    params = [p for p in parameters if p is not None and p.grad is not None]
    if not params:
        return 0.0
    total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=norm_type)
    return float(total_norm)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute the explained variance of ``y_pred`` w.r.t. ``y_true``.

    Returns 1.0 when the prediction is perfect and 0.0 when it is no better
    than predicting the mean.
    """
    y_pred = y_pred.detach().reshape(-1).float()
    y_true = y_true.detach().reshape(-1).float()
    var_y = torch.var(y_true)
    if var_y < 1e-8:
        return float("nan")
    return float(1.0 - torch.var(y_true - y_pred) / var_y)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def to_tensor(x: Any, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Convert numpy arrays / lists / scalars to a torch tensor on ``device``."""
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t


def flatten_dict(d: Dict[str, Any], prefix: str = "", sep: str = "/") -> Dict[str, Any]:
    """Flatten a nested dictionary into a single-level dict with joined keys."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key, sep))
        else:
            out[key] = v
    return out


def count_parameters(module: torch.nn.Module, only_trainable: bool = True) -> int:
    """Count the number of parameters in a module."""
    params = module.parameters()
    if only_trainable:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def linear_schedule(initial: float, final: float, total_steps: int):
    """Return a callable mapping progress in [0, 1] to a linearly interpolated value."""

    def schedule(progress: float) -> float:
        progress = float(min(max(progress, 0.0), 1.0))
        return initial + (final - initial) * progress

    return schedule


def get_device(requested: str = "cuda") -> torch.device:
    """Resolve a device string, falling back to CPU when CUDA is unavailable."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)

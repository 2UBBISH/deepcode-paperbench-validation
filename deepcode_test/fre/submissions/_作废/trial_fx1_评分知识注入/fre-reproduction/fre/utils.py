"""Shared utilities for FRE reproduction.

This module centralizes small helpers used throughout the codebase:
random seeding, state normalization, tensor conversion, logging,
serialization, and device selection.  Keeping these here avoids
duplicating logic in training, evaluation, and baseline scripts.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


ArrayLike = Union[np.ndarray, torch.Tensor, Sequence, Mapping]

# ---------------------------------------------------------------------------
# RNG / reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    Parameters
    ----------
    seed:
        Integer seed.
    deterministic:
        If ``True``, additionally enable PyTorch deterministic algorithms
        and disable benchmark mode.  This can slow down training noticeably,
        so it is disabled by default.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------------


def resolve_device(device_arg: Union[str, torch.device, None] = "auto") -> torch.device:
    """Resolve a user-supplied device string to a :class:`torch.device`.

    ``"auto"`` selects CUDA when available, otherwise CPU.  ``"cuda"`` and
    ``"cpu"`` are also accepted.
    """
    if device_arg is None:
        device_arg = "auto"
    if isinstance(device_arg, torch.device):
        return device_arg
    device_arg = str(device_arg).lower()
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


# ---------------------------------------------------------------------------
# Tensor/array conversion
# ---------------------------------------------------------------------------


def to_numpy(value: Any) -> np.ndarray:
    """Convert a PyTorch tensor, NumPy array, or scalar to a NumPy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def to_torch(
    value: ArrayLike,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    """Convert an array/tensor to a PyTorch tensor on ``device``.

    Dict and mapping inputs are converted with ``torch.as_tensor`` and are
    only safe for simple homogeneous data; callers working with complex
    mappings should convert each field individually.
    """
    if isinstance(value, torch.Tensor):
        tensor = value.to(dtype=dtype) if dtype is not None else value
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value).to(dtype=dtype)
    else:
        tensor = torch.as_tensor(value, dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def ensure_numpy_array(value: ArrayLike, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Convert ``value`` to a contiguous float32 NumPy array.

    Float32 is the convention used by dataset wrappers and environment
    wrappers for storing states, actions, and rewards.
    """
    arr = np.asarray(to_numpy(value), dtype=dtype or np.float32)
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# State normalization
# ---------------------------------------------------------------------------


class RunningMeanStd:
    """Online mean/variance accumulator (Welford's algorithm).

    Useful for computing normalization statistics over large state pools
    without materialising a single huge array.
    """

    def __init__(self, shape: Sequence[int], epsilon: float = 1e-6):
        self.shape = tuple(shape)
        self.epsilon = float(epsilon)
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a batch of observations."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape[1:] != self.shape:
            raise ValueError(f"Expected trailing shape {self.shape}, got {x.shape}")
        batch_count = int(x.shape[0])
        if batch_count == 0:
            return
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0) if batch_count > 1 else np.zeros_like(batch_mean)
        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / max(total_count, 1)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / max(total_count, 1)
        new_var = m2 / max(total_count, 1)
        self.mean, self.var, self.count = new_mean, new_var, total_count

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + self.epsilon)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.mean.astype(np.float32)) / self.std.astype(np.float32)

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32) * self.std.astype(np.float32) + self.mean.astype(np.float32)


class StateNormalizer:
    """Fixed state normalizer that wraps mean/std arrays.

    This is the object form used by dataset wrappers when they expose
    ``normalize``/``unnormalize`` methods.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, epsilon: float = 1e-6):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32) + np.float32(epsilon)
        self.epsilon = float(epsilon)

    @classmethod
    def fit(cls, states: np.ndarray, next_states: Optional[np.ndarray] = None, epsilon: float = 1e-6) -> "StateNormalizer":
        """Compute per-dimension mean/std from states and next_states."""
        states = np.asarray(states, dtype=np.float32)
        if next_states is not None:
            states = np.concatenate([states, np.asarray(next_states, dtype=np.float32)], axis=0)
        if states.shape[0] == 0:
            raise ValueError("Cannot fit normalizer on an empty state array")
        mean = states.mean(axis=0)
        std = states.std(axis=0)
        std = np.maximum(std, 1e-3)  # avoid degenerate std for constant dims
        return cls(mean=mean, std=std, epsilon=epsilon)

    def normalize(self, states: ArrayLike) -> np.ndarray:
        states = ensure_numpy_array(states)
        return (states - self.mean) / self.std

    def unnormalize(self, states: ArrayLike) -> np.ndarray:
        states = ensure_numpy_array(states)
        return states * self.std + self.mean

    @property
    def mean_std_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.mean, self.std


def compute_state_normalization(
    states: np.ndarray,
    next_states: Optional[np.ndarray] = None,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(mean, std)`` arrays for zero-mean/unit-variance normalization."""
    normalizer = StateNormalizer.fit(states, next_states=next_states, epsilon=epsilon)
    return normalizer.mean, normalizer.std


def normalize_states(states: ArrayLike, mean: ArrayLike, std: ArrayLike) -> np.ndarray:
    """Normalize states with explicit mean/std arrays."""
    states = ensure_numpy_array(states)
    mean = ensure_numpy_array(mean)
    std = ensure_numpy_array(std) + np.float32(1e-6)
    return (states - mean) / std


def unnormalize_states(states: ArrayLike, mean: ArrayLike, std: ArrayLike) -> np.ndarray:
    """Reverse explicit mean/std normalization."""
    states = ensure_numpy_array(states)
    mean = ensure_numpy_array(mean)
    std = ensure_numpy_array(std) + np.float32(1e-6)
    return states * std + mean


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: Union[int, str] = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Configure root logging and return the root logger.

    Parameters
    ----------
    level:
        Logging level, e.g. ``logging.INFO`` or ``"INFO"``.
    log_file:
        Optional path for a file handler.  The directory is created
        automatically if it does not exist.
    """
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(console)

    if log_file is not None:
        log_dir = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(file_handler)

    return root


def get_logger(name: str = "fre") -> logging.Logger:
    """Return a child logger with the shared FRE name prefix."""
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def save_json(data: Mapping[str, Any], path: str, indent: int = 2) -> None:
    """Save a JSON-serializable mapping to ``path``, creating directories."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, sort_keys=True)


def load_json(path: str) -> Any:
    """Load JSON data from ``path``."""
    with open(path, "r") as f:
        return json.load(f)


def save_checkpoint(
    checkpoint: Mapping[str, Any],
    path: str,
    create_dirs: bool = True,
) -> None:
    """Save a PyTorch checkpoint dict to ``path``.

    The dict is expected to contain at least ``state_dict`` or be a plain
    state dict itself.  Metadata keys are preserved as-is.
    """
    if create_dirs:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str, map_location: Optional[str] = "cpu") -> Dict[str, Any]:
    """Load a PyTorch checkpoint dict with flexible device mapping."""
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        return {"state_dict": checkpoint}
    return checkpoint


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def freeze_module(module: nn.Module) -> nn.Module:
    """Disable gradient computation for all parameters in ``module``."""
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def unfreeze_module(module: nn.Module) -> nn.Module:
    """Re-enable gradient computation for all parameters in ``module``."""
    module.train()
    for param in module.parameters():
        param.requires_grad_(True)
    return module


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Return the number of (trainable) parameters in a module."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def soft_update_from_dict(target: nn.Module, source: nn.Module, tau: float = 0.005) -> None:
    """Polyak-average ``source`` parameters into ``target`` parameters.

    This mirrors ``fre.iql.soft_update`` but is provided here so utility
    code does not need to import the IQL module.
    """
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * source_param.data)


# ---------------------------------------------------------------------------
# Aggregation / timing helpers
# ---------------------------------------------------------------------------


def average_dicts(dicts: Iterable[Mapping[str, Union[int, float]]]) -> Dict[str, float]:
    """Average a collection of scalar dictionaries by key."""
    keys: List[str] = []
    for d in dicts:
        keys.extend(d.keys())
    keys = list(dict.fromkeys(keys))

    result: Dict[str, float] = {}
    for key in keys:
        values = [float(d[key]) for d in dicts if key in d]
        result[key] = float(np.mean(values)) if values else 0.0
    return result


def std_dicts(dicts: Iterable[Mapping[str, Union[int, float]]]) -> Dict[str, float]:
    """Compute per-key standard deviation over a collection of dicts."""
    keys: List[str] = []
    for d in dicts:
        keys.extend(d.keys())
    keys = list(dict.fromkeys(keys))

    result: Dict[str, float] = {}
    for key in keys:
        values = [float(d[key]) for d in dicts if key in d]
        result[key] = float(np.std(values)) if len(values) > 1 else 0.0
    return result


class Timer:
    """Small wall-clock timer with formatted elapsed output."""

    def __init__(self) -> None:
        self.start_time = time.time()

    def reset(self) -> None:
        self.start_time = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def elapsed_str(self) -> str:
        return format_time(self.elapsed())


def format_time(seconds: float) -> str:
    """Format a duration in seconds as ``H:MM:SS``."""
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# Convenience data helpers
# ---------------------------------------------------------------------------


def sample_indices(n: int, pool_size: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Sample ``n`` integer indices uniformly from ``[0, pool_size)``.

    If ``rng`` is provided it is used; otherwise the global NumPy RNG is
    used.  Returns an int64 array.
    """
    if n > pool_size:
        raise ValueError(f"Cannot sample {n} indices from a pool of size {pool_size}")
    generator: Union[np.random.RandomState, np.random.Generator]
    if rng is not None:
        generator = rng
    else:
        generator = np.random
    return generator.randint(0, pool_size, size=int(n)).astype(np.int64)


def stable_log_prob(mean: torch.Tensor, log_std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Compute log probability of ``action`` under a diagonal Gaussian.

    This is mostly used by policy implementations that do not rely on
    ``torch.distributions``.  It does not apply tanh correction; callers
    requiring squashed policies should use the distribution API.
    """
    var = torch.exp(2.0 * log_std)
    log_prob = -0.5 * ((action - mean) ** 2 / var + 2.0 * log_std + math.log(2.0 * math.pi))
    return log_prob.sum(dim=-1)


__all__ = [
    "ArrayLike",
    "RunningMeanStd",
    "StateNormalizer",
    "Timer",
    "average_dicts",
    "compute_state_normalization",
    "configure_logging",
    "count_parameters",
    "ensure_numpy_array",
    "format_time",
    "freeze_module",
    "get_logger",
    "load_checkpoint",
    "load_json",
    "normalize_states",
    "resolve_device",
    "sample_indices",
    "save_checkpoint",
    "save_json",
    "set_seed",
    "soft_update_from_dict",
    "stable_log_prob",
    "std_dicts",
    "to_numpy",
    "to_torch",
    "unfreeze_module",
    "unnormalize_states",
]

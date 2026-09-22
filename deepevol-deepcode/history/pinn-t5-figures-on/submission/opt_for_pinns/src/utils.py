"""Utility helpers: seeding, logging, checkpointing, timing.

These helpers are shared across the training loops and experiment scripts.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed python, numpy and torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOGGER_NAME = "opt_for_pinns"


def get_logger(name: str = _LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                                datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
class Timer:
    """Simple wall-clock timer.

    Usage:
        t = Timer()
        with t("section"):
            ...
        print(t.times)
    """

    def __init__(self) -> None:
        self.times: Dict[str, float] = {}
        self._start: Optional[float] = None
        self._label: Optional[str] = None

    def start(self, label: str = "default") -> None:
        self._label = label
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer.stop() called before Timer.start()")
        elapsed = time.perf_counter() - self._start
        assert self._label is not None
        self.times[self._label] = self.times.get(self._label, 0.0) + elapsed
        self._start = None
        self._label = None
        return elapsed

    @contextmanager
    def __call__(self, label: str = "default"):
        self.start(label)
        try:
            yield self
        finally:
            self.stop()


@contextmanager
def timeit(label: str = "block", logger: Optional[logging.Logger] = None):
    """Context manager that logs the elapsed wall-clock time of a block."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if logger is not None:
            logger.info("%s took %.4f s", label, dt)


# ---------------------------------------------------------------------------
# Checkpointing / results IO
# ---------------------------------------------------------------------------
def ensure_dir(path: str) -> str:
    """Create directory (and parents) if it does not exist; return path."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, path: str) -> None:
    """Save a JSON-serialisable object to disk (creating parent dirs)."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _json_default(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return str(o)


def save_checkpoint(model: torch.nn.Module, path: str, extra: Optional[Dict] = None) -> None:
    """Save model state_dict (and optional extra metadata)."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model: torch.nn.Module, path: str, map_location: str = "cpu") -> Dict:
    """Load a checkpoint into ``model`` and return the payload."""
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["state_dict"])
    return payload


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def count_parameters(model: torch.nn.Module) -> int:
    """Total number of trainable scalar parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_float(x: Any) -> float:
    """Best-effort conversion of a scalar tensor / numpy value to float."""
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def summarize(values) -> Dict[str, float]:
    """Return min / median / max summary of a sequence of numbers."""
    arr = np.asarray([to_float(v) for v in values], dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


__all__ = [
    "set_seed",
    "get_logger",
    "Timer",
    "timeit",
    "ensure_dir",
    "save_json",
    "load_json",
    "save_checkpoint",
    "load_checkpoint",
    "count_parameters",
    "to_float",
    "summarize",
]

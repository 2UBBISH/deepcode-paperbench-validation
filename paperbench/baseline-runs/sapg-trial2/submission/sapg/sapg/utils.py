"""Utility helpers for SAPG: seeding, logging, checkpointing, and config loading.

These utilities are intentionally lightweight and dependency-optional so that the
core algorithm can run without TensorBoard / W&B installed.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

try:  # optional
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available device."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a dict (empty dict if file missing)."""
    if not os.path.exists(path):
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to load config files.")
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data


def save_yaml(path: str, data: Dict[str, Any]) -> None:
    """Save a dict to a YAML file."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to save config files.")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def to_serializable(obj: Any) -> Any:
    """Recursively convert dataclasses / tensors into JSON-serializable objects."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
class Logger:
    """Simple logger writing to stdout, a JSONL file, and optionally TensorBoard.

    Usage::

        logger = Logger(log_dir="runs/exp1", use_tensorboard=True)
        logger.log({"reward": 1.0}, step=100)
        logger.close()
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb_project: str = "sapg",
        wandb_run_name: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        self.log_dir = log_dir
        self.verbose = verbose
        self._jsonl_path: Optional[str] = None
        self._writer = None
        self._wandb = None

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self._jsonl_path = os.path.join(log_dir, "metrics.jsonl")

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore

                self._writer = SummaryWriter(log_dir=log_dir)
            except Exception as exc:  # pragma: no cover
                if verbose:
                    print(f"[Logger] TensorBoard unavailable: {exc}")

        if use_wandb:
            try:
                import wandb  # type: ignore

                self._wandb = wandb
                wandb.init(project=wandb_project, name=wandb_run_name, dir=log_dir)
            except Exception as exc:  # pragma: no cover
                if verbose:
                    print(f"[Logger] W&B unavailable: {exc}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        metrics = {k: to_serializable(v) for k, v in metrics.items()}
        if self.verbose:
            prefix = f"[step {step}] " if step is not None else ""
            pretty = "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in metrics.items())
            print(prefix + pretty)

        if self._jsonl_path is not None:
            record = {"step": step, "time": time.time(), **metrics}
            with open(self._jsonl_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        if self._writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._writer.add_scalar(k, v, step if step is not None else 0)

        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str,
    algorithm: Any,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save an algorithm state dict (plus optional optimizer/extra) to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    state: Dict[str, Any] = {"step": step}
    if hasattr(algorithm, "state_dict"):
        state["algorithm"] = algorithm.state_dict()
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if extra:
        state["extra"] = to_serializable(extra)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    algorithm: Any,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a checkpoint into an algorithm (and optional optimizer)."""
    state = torch.load(path, map_location=map_location or "cpu")
    if "algorithm" in state and hasattr(algorithm, "load_state_dict"):
        algorithm.load_state_dict(state["algorithm"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
class RunningMeanStd:
    """Running mean/std tracker (Welford-style) for observation normalization."""

    def __init__(self, shape: Any = (), epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0] if x.ndim > 0 else 1

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8)


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute explained variance of value predictions (1.0 is perfect)."""
    y_pred = y_pred.detach().flatten().float()
    y_true = y_true.detach().flatten().float()
    var_y = torch.var(y_true)
    if var_y == 0:
        return float("nan")
    return float(1.0 - torch.var(y_true - y_pred) / var_y)


def count_parameters(module: torch.nn.Module) -> int:
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def linear_schedule(initial: float, final: float, total_steps: int):
    """Return a function mapping progress in [0,1] to a linearly interpolated value."""

    def fn(progress: float) -> float:
        progress = min(max(progress, 0.0), 1.0)
        return initial + progress * (final - initial)

    return fn


__all__ = [
    "set_seed",
    "get_device",
    "load_yaml",
    "save_yaml",
    "to_serializable",
    "Logger",
    "save_checkpoint",
    "load_checkpoint",
    "RunningMeanStd",
    "explained_variance",
    "count_parameters",
    "linear_schedule",
]

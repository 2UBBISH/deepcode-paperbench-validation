"""Utility helpers for the opt_for_pinns codebase.

Provides seeding, logging, checkpointing, and YAML config loading used across
the experiment orchestration scripts and the CLI entry point.
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (CPU + CUDA) RNGs for reproducibility."""
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
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str = "opt_for_pinns", level: int = logging.INFO) -> logging.Logger:
    """Create (or fetch) a configured logger with a stream handler."""
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
# Config loading
# ---------------------------------------------------------------------------
def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    """Load a YAML config file into a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def save_config(cfg: Dict[str, Any], path: str | os.PathLike) -> None:
    """Persist a config dict to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_configs(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(state: Dict[str, Any], path: str | os.PathLike) -> None:
    """Save a checkpoint dict via ``torch.save``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | os.PathLike, map_location: Optional[Any] = None) -> Dict[str, Any]:
    """Load a checkpoint dict saved by :func:`save_checkpoint`."""
    return torch.load(path, map_location=map_location)


def save_json(obj: Any, path: str | os.PathLike) -> None:
    """Save a JSON-serializable object (e.g. results dict) to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | os.PathLike) -> Any:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def _json_default(o: Any):
    """Fallback encoder for numpy/torch scalars."""
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Results directory helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: str | os.PathLike) -> Path:
    """Create a directory (and parents) if it does not exist; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir(root: str | os.PathLike = "results", *sub: str) -> Path:
    """Build (and create) a results subdirectory."""
    return ensure_dir(Path(root).joinpath(*sub))


__all__ = [
    "set_seed",
    "get_device",
    "get_logger",
    "load_config",
    "save_config",
    "merge_configs",
    "save_checkpoint",
    "load_checkpoint",
    "save_json",
    "load_json",
    "ensure_dir",
    "results_dir",
]

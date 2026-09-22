"""Checkpoint save/load utilities for SAPG training runs.

Provides a small, dependency-light helper for persisting and restoring the
full training state: model parameters, optimizer state, KL-adaptive LR
controller state, iteration counters, RNG state, and arbitrary metadata.

The checkpoint format is a single ``torch.save``-produced file containing a
plain dict, so it is portable across processes and easy to inspect.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "CheckpointManager",
    "capture_rng_state",
    "restore_rng_state",
]


# ---------------------------------------------------------------------------
# RNG state helpers
# ---------------------------------------------------------------------------
def capture_rng_state() -> Dict[str, Any]:
    """Capture Python / NumPy / Torch RNG states for exact resumption."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:  # pragma: no cover - defensive
            state["torch_cuda"] = None
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    """Restore RNG states previously captured by :func:`capture_rng_state`."""
    if not state:
        return
    if "python" in state and state["python"] is not None:
        random.setstate(state["python"])
    if "numpy" in state and state["numpy"] is not None:
        np.random.set_state(state["numpy"])
    if "torch" in state and state["torch"] is not None:
        torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        except Exception:  # pragma: no cover - defensive
            pass


# ---------------------------------------------------------------------------
# Core save / load
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    iteration: int = 0,
    transitions: int = 0,
    extra: Optional[Dict[str, Any]] = None,
    include_rng: bool = True,
) -> str:
    """Save a training checkpoint to ``path``.

    Args:
        path: Destination file path (parent dirs are created).
        model: Model whose ``state_dict`` should be stored.
        optimizer: Optional optimizer whose state should be stored.
        scheduler: Optional object exposing ``state_dict()`` (e.g. KLAdaptiveLR).
        iteration: Current training iteration index.
        transitions: Total environment transitions consumed so far.
        extra: Arbitrary additional metadata to persist.
        include_rng: Whether to capture RNG states.

    Returns:
        The path written to.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    payload: Dict[str, Any] = {
        "iteration": int(iteration),
        "transitions": int(transitions),
        "extra": dict(extra or {}),
    }
    if model is not None:
        payload["model"] = model.state_dict()
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        try:
            payload["scheduler"] = scheduler.state_dict()
        except Exception:  # pragma: no cover - defensive
            payload["scheduler"] = None
    if include_rng:
        payload["rng"] = capture_rng_state()

    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[Any] = None,
    restore_rng: bool = False,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint and optionally restore state into live objects.

    Args:
        path: Checkpoint file path.
        model: If given, load ``state_dict`` into it.
        optimizer: If given, load optimizer state.
        scheduler: If given and it exposes ``load_state_dict``, restore it.
        map_location: Passed to ``torch.load`` (defaults to CPU when CUDA absent).
        restore_rng: Whether to restore captured RNG states.
        strict: Passed to ``model.load_state_dict``.

    Returns:
        The raw checkpoint dict (with ``iteration``, ``transitions``, ``extra``).
    """
    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"

    payload = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None and "model" in payload:
        model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        if hasattr(scheduler, "load_state_dict"):
            try:
                scheduler.load_state_dict(payload["scheduler"])
            except Exception:  # pragma: no cover - defensive
                pass
    if restore_rng and payload.get("rng") is not None:
        restore_rng_state(payload["rng"])

    return payload


# ---------------------------------------------------------------------------
# Convenience manager
# ---------------------------------------------------------------------------
class CheckpointManager:
    """Manage periodic checkpointing with a rolling ``latest`` pointer.

    Example::

        ckpt = CheckpointManager("runs/sapg", save_interval=100)
        ...
        ckpt.maybe_save(iteration, model, optimizer, scheduler, transitions)
        ckpt.save(iteration, model, optimizer, scheduler, transitions, tag="best")
    """

    def __init__(self, output_dir: str, save_interval: int = 100, keep_last: int = 3):
        self.output_dir = output_dir
        self.save_interval = int(save_interval)
        self.keep_last = int(keep_last)
        os.makedirs(output_dir, exist_ok=True)
        self._saved: list = []

    # -- paths -------------------------------------------------------------
    def path_for(self, tag: str) -> str:
        return os.path.join(self.output_dir, f"checkpoint_{tag}.pt")

    @property
    def latest_path(self) -> str:
        return os.path.join(self.output_dir, "checkpoint_latest.pt")

    # -- saving ------------------------------------------------------------
    def save(
        self,
        iteration: int,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        transitions: int = 0,
        tag: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a checkpoint; always refreshes the ``latest`` pointer."""
        tag = tag or f"iter{int(iteration):08d}"
        path = self.path_for(tag)
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            iteration=iteration,
            transitions=transitions,
            extra=extra,
        )
        # Refresh latest pointer (copy of the same payload).
        save_checkpoint(
            self.latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            iteration=iteration,
            transitions=transitions,
            extra=extra,
        )
        self._saved.append(path)
        self._prune()
        return path

    def maybe_save(
        self,
        iteration: int,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        transitions: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Save only when ``iteration % save_interval == 0``."""
        if self.save_interval <= 0:
            return None
        if iteration % self.save_interval != 0:
            return None
        return self.save(
            iteration,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            transitions=transitions,
            extra=extra,
        )

    # -- loading -----------------------------------------------------------
    def load(
        self,
        path: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        restore_rng: bool = False,
    ) -> Dict[str, Any]:
        """Load ``path`` (defaults to the latest checkpoint)."""
        path = path or self.latest_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            restore_rng=restore_rng,
        )

    # -- housekeeping ------------------------------------------------------
    def _prune(self) -> None:
        if self.keep_last <= 0:
            return
        while len(self._saved) > self.keep_last:
            old = self._saved.pop(0)
            try:
                if os.path.exists(old):
                    os.remove(old)
            except OSError:  # pragma: no cover - defensive
                pass

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"CheckpointManager(output_dir={self.output_dir!r}, "
            f"save_interval={self.save_interval}, keep_last={self.keep_last})"
        )

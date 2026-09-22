"""Checkpoint input/output utilities for the FOA reproduction.

This module is *glue only*: it centralises saving/loading of the artefacts that
the FOA pipeline persists between runs.  No paper-specified math lives here.

Artefacts handled
-----------------
* Source in-distribution activation statistics ``{mu_i^S, sigma_i^S}_{i=0..N}``
  (see :mod:`src.method.source_stats`).
* Prompt state (the current prompt tensor / flattened CMA mean).
* CMA-ES optimizer state (mean, sigma, covariance, generation).
* Generic ``torch.save`` / ``torch.load`` checkpoints with metadata.

All helpers are defensive: a missing file raises a helpful error, and objects
that expose ``state_dict()``/``load_state_dict()`` are handled transparently.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Mapping, Optional

import torch

__all__ = [
    "CHECKPOINT_SUFFIX",
    "ensure_dir",
    "checkpoint_exists",
    "save_checkpoint",
    "load_checkpoint",
    "save_torch",
    "load_torch",
    "save_source_stats",
    "load_source_stats_checkpoint",
    "save_prompt_state",
    "load_prompt_state",
    "save_optimizer_state",
    "load_optimizer_state",
    "save_foa_state",
    "load_foa_state",
    "make_meta",
    "DEFAULT_SOURCE_STATS_PATH",
]


CHECKPOINT_SUFFIX = ".pt"
DEFAULT_SOURCE_STATS_PATH = os.path.join("checkpoints", "source_stats_vit_base.pt")


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: str) -> str:
    """Create the parent directory of ``path`` if needed.  Returns the dir."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    return directory


def checkpoint_exists(path: str) -> bool:
    """Return ``True`` when ``path`` points at an existing regular file."""
    return bool(path) and os.path.isfile(path)


def make_meta(**extra: Any) -> Dict[str, Any]:
    """Build a small metadata dict stamped with a wall-clock time."""
    meta: Dict[str, Any] = {"saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    meta.update(extra)
    return meta


def _to_cpu(obj: Any) -> Any:
    """Recursively move tensors in a nested container to CPU.

    Keeping checkpoints CPU-only makes them portable and avoids pinning GPU
    memory from a run that has already finished.
    """
    if torch.is_tensor(obj):
        return obj.detach().to("cpu")
    if isinstance(obj, Mapping):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def save_torch(obj: Any, path: str, *, atomic: bool = True) -> str:
    """Save ``obj`` to ``path`` with :func:`torch.save` (CPU tensors)."""
    ensure_dir(path)
    payload = _to_cpu(obj)
    if atomic:
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
    else:
        torch.save(payload, path)
    return path


def load_torch(path: str, map_location: str = "cpu") -> Any:
    """Load a torch checkpoint, raising a helpful error if it is missing."""
    if not checkpoint_exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. "
            "Run the corresponding preparation script first "
            "(e.g. scripts/compute_source_stats.py)."
        )
    return torch.load(path, map_location=map_location)


def save_checkpoint(obj: Any, path: str, **meta: Any) -> str:
    """Save ``obj`` together with metadata under the key ``meta``.

    Recognises objects exposing ``state_dict()`` (moves to a plain dict) and
    dataclasses with ``state_dict``/``__dict__``.
    """
    if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")):
        payload = {"state_dict": obj.state_dict(), "meta": make_meta(**meta)}
    else:
        payload = {"state_dict": obj, "meta": make_meta(**meta)}
    return save_torch(payload, path)


def load_checkpoint(
    path: str,
    target: Any = None,
    *,
    map_location: str = "cpu",
    strict: bool = False,
) -> Any:
    """Load a checkpoint previously written by :func:`save_checkpoint`.

    When ``target`` is provided and exposes ``load_state_dict`` the stored
    ``state_dict`` is applied to it and ``target`` is returned.
    """
    payload = load_torch(path, map_location=map_location)
    state = payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
    if target is not None and hasattr(target, "load_state_dict"):
        try:
            target.load_state_dict(state, strict=strict)
        except TypeError:
            target.load_state_dict(state)
        return target
    return state


# ---------------------------------------------------------------------------
# source statistics
# ---------------------------------------------------------------------------
def save_source_stats(stats: Any, path: Optional[str] = None) -> str:
    """Persist a :class:`~src.method.source_stats.SourceStats` bank.

    Uses the object's own ``state_dict`` when available so the on-disk format
    stays consistent with :func:`src.method.source_stats.save_source_stats`.
    """
    path = path or DEFAULT_SOURCE_STATS_PATH
    if hasattr(stats, "state_dict") and callable(getattr(stats, "state_dict")):
        state = {"state_dict": stats.state_dict(), "meta": make_meta(kind="source_stats")}
    else:
        state = stats
    return save_torch(state, path)


def load_source_stats_checkpoint(path: Optional[str] = None, device: str = "cpu") -> Any:
    """Load a saved source-statistics bank.

    Prefers :func:`src.method.source_stats.load_source_stats` when importable so
    that a fully-hydrated :class:`SourceStats` object is returned; otherwise
    falls back to the raw state dict.
    """
    path = path or DEFAULT_SOURCE_STATS_PATH
    try:  # pragma: no cover - import guard keeps this module dependency-light
        from ..method.source_stats import load_source_stats as _load

        return _load(path, device=device)
    except Exception:  # noqa: BLE001 - fall through to raw load
        payload = load_torch(path, map_location=device)
        if isinstance(payload, Mapping) and "state_dict" in payload:
            return payload["state_dict"]
        return payload


# ---------------------------------------------------------------------------
# prompt / optimizer / full FOA state
# ---------------------------------------------------------------------------
def save_prompt_state(prompt: Any, path: str, **meta: Any) -> str:
    """Save the prompt vector (flattened ``d*N_p``) to ``path``."""
    if hasattr(prompt, "get_prompt"):
        vector = prompt.get_prompt()
    elif torch.is_tensor(prompt):
        vector = prompt.detach().clone()
    else:
        vector = torch.as_tensor(prompt)
    payload = {
        "prompt": vector.detach().to("cpu"),
        "num_prompts": getattr(prompt, "num_prompts", None),
        "embed_dim": getattr(prompt, "embed_dim", None),
        "meta": make_meta(kind="prompt", **meta),
    }
    return save_torch(payload, path)


def load_prompt_state(path: str, target: Any = None, *, map_location: str = "cpu") -> Any:
    """Load a prompt state written by :func:`save_prompt_state`.

    If ``target`` is a prompt object with ``set_prompt`` the vector is injected
    into it and ``target`` is returned; otherwise the raw vector is returned.
    """
    payload = load_torch(path, map_location=map_location)
    vector = payload.get("prompt", payload) if isinstance(payload, Mapping) else payload
    if target is not None and hasattr(target, "set_prompt"):
        try:
            target.set_prompt(vector)
        except Exception:  # noqa: BLE001 - non-fatal, caller may inspect
            pass
        return target
    return vector


def save_optimizer_state(optimizer: Any, path: str, **meta: Any) -> str:
    """Save a CMA-ES optimizer state (mean/sigma/covariance/generation)."""
    if hasattr(optimizer, "state_dict") and callable(getattr(optimizer, "state_dict")):
        state = optimizer.state_dict()
    else:
        state = optimizer
    return save_torch({"optimizer": state, "meta": make_meta(kind="cma", **meta)}, path)


def load_optimizer_state(path: str, target: Any = None, *, map_location: str = "cpu") -> Any:
    """Load a CMA-ES optimizer state; applies it to ``target`` when supported."""
    payload = load_torch(path, map_location=map_location)
    state = payload.get("optimizer", payload) if isinstance(payload, Mapping) else payload
    if target is not None and hasattr(target, "load_state_dict"):
        try:
            target.load_state_dict(state)
        except Exception:  # noqa: BLE001 - state may be raw
            pass
    return state


def save_foa_state(state: Mapping[str, Any], path: str, **meta: Any) -> str:
    """Save a composite FOA state dict (prompt + optimizer + shifter EMA)."""
    payload = dict(state)
    payload.setdefault("meta", make_meta(kind="foa_state", **meta))
    return save_torch(payload, path)


def load_foa_state(path: str, *, map_location: str = "cpu") -> Dict[str, Any]:
    """Load a composite FOA state dict written by :func:`save_foa_state`."""
    payload = load_torch(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Unexpected FOA state payload in {path!r}: {type(payload)}")
    return dict(payload)

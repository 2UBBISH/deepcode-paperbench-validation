"""Checkpoint save/load utilities for SAPG.

Provides a small, dependency-light checkpointing layer that persists:
  * shared network weights (policy ``B_theta`` and value ``C_psi``),
  * optimizer state,
  * per-follower / leader conditioning vectors ``phi_j``,
  * curriculum state,
  * observation-normalization statistics,
  * logger state and arbitrary training metadata (iteration, env steps, ...).

The functions are intentionally tolerant: any component that exposes a
``state_dict`` / ``load_state_dict`` pair (torch modules, optimizers, the
curriculum, the logger, ...) can be passed in and will be handled uniformly.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Iterable, Mapping, Optional

import torch


__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "save_state_dict",
    "load_state_dict",
    "maybe_load_checkpoint",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_state(obj: Any) -> Any:
    """Convert an object into a picklable / serializable state representation.

    Handles torch modules, optimizers, and anything exposing ``state_dict``.
    Plain values (ints, floats, dicts, lists) are returned unchanged.
    """
    if obj is None:
        return None
    if isinstance(obj, (int, float, str, bool, bytes)):
        return obj
    if isinstance(obj, Mapping):
        return {k: _to_state(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_state(v) for v in obj)
    if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")):
        try:
            return obj.state_dict()
        except Exception:  # pragma: no cover - defensive
            return obj
    return obj


def _load_into(obj: Any, state: Any) -> None:
    """Restore ``state`` into ``obj`` when possible (in-place)."""
    if obj is None or state is None:
        return
    if hasattr(obj, "load_state_dict") and callable(getattr(obj, "load_state_dict")):
        try:
            obj.load_state_dict(state)
        except Exception:  # pragma: no cover - defensive
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str,
    *,
    policy: Any = None,
    value: Any = None,
    optimizer: Any = None,
    followers: Optional[Iterable[Any]] = None,
    leader: Any = None,
    curriculum: Any = None,
    obs_normalizer: Any = None,
    logger: Any = None,
    iteration: Optional[int] = None,
    env_steps: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    use_torch_save: bool = True,
) -> str:
    """Save a training checkpoint to ``path``.

    Args:
        path: Destination file path (directories are created as needed).
        policy: Shared actor network (``GaussianPolicy``).
        value: Shared critic network (``ValueNetwork``).
        optimizer: Shared optimizer (Adam).
        followers: Iterable of ``Follower`` objects (their ``phi`` vectors are
            stored; network weights are shared and saved once).
        leader: ``Leader`` object (its ``phi`` vector is stored).
        curriculum: ``SuccessToleranceCurriculum`` instance.
        obs_normalizer: ``NormalizeObsWrapper`` instance.
        logger: ``MetricLogger`` instance.
        iteration: Current training iteration.
        env_steps: Total environment steps consumed.
        extra: Arbitrary additional metadata.
        use_torch_save: If True use ``torch.save`` (default); otherwise pickle.

    Returns:
        The path written to.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    follower_phis = None
    if followers is not None:
        follower_phis = []
        for f in followers:
            phi = getattr(f, "phi", None)
            follower_phis.append(None if phi is None else phi.detach().cpu().clone())

    leader_phi = None
    if leader is not None:
        phi = getattr(leader, "phi", None)
        leader_phi = None if phi is None else phi.detach().cpu().clone()

    state: Dict[str, Any] = {
        "policy": _to_state(policy),
        "value": _to_state(value),
        "optimizer": _to_state(optimizer),
        "follower_phis": follower_phis,
        "leader_phi": leader_phi,
        "curriculum": _to_state(curriculum),
        "obs_normalizer": _to_state(obs_normalizer),
        "logger": _to_state(logger),
        "iteration": iteration,
        "env_steps": env_steps,
        "extra": dict(extra) if extra else {},
    }

    if use_torch_save:
        torch.save(state, path)
    else:  # pragma: no cover - alternative path
        with open(path, "wb") as fh:
            pickle.dump(state, fh)
    return path


def load_checkpoint(
    path: str,
    *,
    policy: Any = None,
    value: Any = None,
    optimizer: Any = None,
    followers: Optional[Iterable[Any]] = None,
    leader: Any = None,
    curriculum: Any = None,
    obs_normalizer: Any = None,
    logger: Any = None,
    map_location: str = "cpu",
    strict: bool = False,
) -> Dict[str, Any]:
    """Load a checkpoint from ``path`` and restore state into the given objects.

    Objects passed in are restored in-place when they expose
    ``load_state_dict``. Returns the raw checkpoint dict so callers can read
    ``iteration`` / ``env_steps`` / ``extra`` metadata.

    Args:
        path: Checkpoint file path.
        policy, value, optimizer, curriculum, obs_normalizer, logger: objects
            to restore in-place.
        followers: Iterable of ``Follower`` objects whose ``phi`` vectors are
            restored from the checkpoint (matched by order).
        leader: ``Leader`` object whose ``phi`` vector is restored.
        map_location: Device mapping for ``torch.load``.
        strict: If True, raise when the checkpoint file is missing.

    Returns:
        The loaded checkpoint dictionary (empty dict if missing and not strict).
    """
    if not os.path.exists(path):
        if strict:
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return {}

    try:
        state = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # older torch without weights_only kwarg
        state = torch.load(path, map_location=map_location)

    _load_into(policy, state.get("policy"))
    _load_into(value, state.get("value"))
    _load_into(optimizer, state.get("optimizer"))
    _load_into(curriculum, state.get("curriculum"))
    _load_into(obs_normalizer, state.get("obs_normalizer"))
    _load_into(logger, state.get("logger"))

    follower_phis = state.get("follower_phis")
    if followers is not None and follower_phis is not None:
        for f, phi in zip(followers, follower_phis):
            if phi is None:
                continue
            cur = getattr(f, "phi", None)
            if cur is not None:
                with torch.no_grad():
                    cur.copy_(phi.to(cur.device))

    leader_phi = state.get("leader_phi")
    if leader is not None and leader_phi is not None:
        cur = getattr(leader, "phi", None)
        if cur is not None:
            with torch.no_grad():
                cur.copy_(leader_phi.to(cur.device))

    return state


def save_state_dict(path: str, state: Dict[str, Any]) -> str:
    """Save an arbitrary state dict (thin wrapper around ``torch.save``)."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(state, path)
    return path


def load_state_dict(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Load an arbitrary state dict (thin wrapper around ``torch.load``)."""
    if not os.path.exists(path):
        return {}
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # older torch without weights_only kwarg
        return torch.load(path, map_location=map_location)


def maybe_load_checkpoint(path: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """Load a checkpoint only if ``path`` is provided and exists."""
    if not path:
        return {}
    return load_checkpoint(path, **kwargs)

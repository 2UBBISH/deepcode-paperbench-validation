"""Checkpoint save/load helpers.

The runners checkpoint every ``25M`` environment steps (NetHack) and on a
configurable interval for the other environments.  Checkpoints are plain
``torch.save`` archives so that they can be inspected with ``torch.load``.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


_STEP_RE = re.compile(r"step_(\d+)\.pt$")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_checkpoint(
    path: str,
    state: Dict[str, Any],
    step: Optional[int] = None,
) -> str:
    """Save ``state`` to ``path`` (appending the step when provided)."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    if step is not None:
        base, ext = os.path.splitext(path)
        if not base.endswith(f"step_{step}"):
            path = f"{base}_step_{step}{ext or '.pt'}"
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for checkpointing")
    torch.save(state, path)
    return path


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for checkpointing")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location=map_location)


def list_checkpoints(directory: str) -> List[str]:
    """Return checkpoint paths sorted by recorded step."""
    if not os.path.isdir(directory):
        return []
    entries = []
    for name in os.listdir(directory):
        match = _STEP_RE.search(name)
        if match:
            entries.append((int(match.group(1)), os.path.join(directory, name)))
    entries.sort()
    return [path for _, path in entries]


def latest_checkpoint(directory: str) -> Optional[str]:
    ckpts = list_checkpoints(directory)
    return ckpts[-1] if ckpts else None


def maybe_resume(runner_state: Dict[str, Any], path: Optional[str]) -> Dict[str, Any]:
    """Merge a checkpoint into ``runner_state`` if ``path`` is provided."""
    if not path:
        return runner_state
    ckpt = load_checkpoint(path)
    for key, value in ckpt.items():
        runner_state[key] = value
    return runner_state

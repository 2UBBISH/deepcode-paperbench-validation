"""Observability seam: task-scoped context the business layer announces.

DeepCode used these to route per-task log files and to attach LLM/MCP
records to the right task directory. The kernel does nothing with them.
Every function is a no-op that keeps the call sites valid; override the
module attributes if you want telemetry.

Call sites (all inside ``try``/``except`` in the business layer):

    workflows/environment.py               ``set_task_dir(task_dir)``,
                                           ``current_session_id()``
    workflows/agent_orchestration_engine.py ``bind_task(task_id)`` on entry,
                                           ``pop_task(token)`` on exit,
                                           ``current_session_id()`` at the end
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def bind_task(task_id: str) -> Any:
    """Announce that ``task_id`` is now the active task. Returns a token."""
    return None


def pop_task(token: Any) -> None:
    """Undo :func:`bind_task`."""
    return None


def set_task_dir(task_id: str, task_dir: str | Path | None = None) -> None:
    """Announce where the active task keeps its files.

    [paper2code C7] signature matches the caller in ``workflows/environment.py``
    (``set_task_dir(chosen_id, task_dir)``); the stub used to take one argument
    and the engine logged a debug failure on every run.
    """
    return None


def current_session_id() -> str | None:
    """The host session the task belongs to, if any. The kernel has none."""
    return None


__all__ = ["bind_task", "current_session_id", "pop_task", "set_task_dir"]

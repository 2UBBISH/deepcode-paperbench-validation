"""Session seam: where DeepCode recorded task links and final status.

Two call sites, both guarded by ``if session_id:`` and wrapped in
``try``/``except``:

    workflows/environment.py                store.attach_task(session_id, ...)
    workflows/agent_orchestration_engine.py store.update_task_status(
                                                session_id, task_id, status,
                                                metadata=...)

Since :func:`seams.observability.current_session_id` returns ``None`` the
store is never reached, but ``get_default_store`` must still import. The
no-op store below documents the two methods for an integrator who wires a
real session id.
"""

from __future__ import annotations

from typing import Any


class NoopSessionStore:
    def attach_task(
        self,
        session_id: str,
        task_id: str,
        *,
        task_dir: str | None = None,
        task_kind: str | None = None,
        input_source: str | None = None,
        **_: Any,
    ) -> None:
        return None

    def update_task_status(
        self,
        session_id: str,
        task_id: str,
        status: str,
        *,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        return None


_STORE = NoopSessionStore()


def get_default_store() -> NoopSessionStore:
    return _STORE


__all__ = ["NoopSessionStore", "get_default_store"]

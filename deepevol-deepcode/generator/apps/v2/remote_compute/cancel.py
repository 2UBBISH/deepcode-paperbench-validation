"""Remote task compatibility adapter over the canonical run-cancel command."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol
from uuid import UUID

from apps.v2.conversation.models import (
    ActorContext,
    CancelRunCommand,
    CancelRunResult,
)
from apps.v2.conversation.service import ConversationService
from apps.v2.remote_compute.postgres import (
    PsycopgRemoteComputeRepository,
    RemoteComputeConflict,
    RemoteComputeNotFound,
)


class RunCancelContextPort(Protocol):
    def get_run_cancel_context(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        rid: UUID,
    ) -> Mapping[str, Any] | None: ...


class RemoteComputeRunCancel:
    """Cancel a task through Conversation; never mutate the binding directly."""

    def __init__(
        self,
        repository: RunCancelContextPort | PsycopgRemoteComputeRepository,
        conversation: ConversationService,
    ) -> None:
        self.repository = repository
        self.conversation = conversation

    def cancel(
        self,
        *,
        actor: ActorContext,
        rid: UUID,
        operation_id: UUID,
        reason: str | None = None,
    ) -> CancelRunResult:
        context = self.repository.get_run_cancel_context(
            resource_tid=actor.resource_tid,
            owner_uid=actor.uid,
            rid=rid,
        )
        if context is None:
            raise RemoteComputeNotFound("remote compute run was not found")
        state = str(context.get("state") or "")
        desired_state = str(context.get("desired_state") or "")
        if desired_state == "CANCELLED" or state in {"CANCELLING", "CANCELLED"}:
            raise RemoteComputeConflict("remote compute run cancellation is already pending")
        return self.conversation.cancel_run(
            CancelRunCommand(
                actor=actor,
                operation_id=operation_id,
                sid=context["sid"],
                rid=rid,
                expected_run_version=int(context["version"]),
                reason=reason,
            )
        )


__all__ = ["RemoteComputeRunCancel", "RunCancelContextPort"]

"""User-facing RSA orchestration.

The benchmark is a client of RSA, not part of RSA's control flow.  This module
is the product boundary: a caller gives the agent a repository and the user's
natural-language instruction, and RSA owns criterion generation, environment
setup, verification and the points at which it needs the user to decide
something.

The lower-level compiler and pipeline remain useful independently for audits,
but applications should use :class:`RSAAgent` so a real user and a simulated
user exercise the same protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

from .asset_gate import AssetReport
from .criterion import Rung
from .escalation import EscalationCard
from .pipeline import CompileOutcome, PipelineConfig, PipelineOutcome, compile_ladder, run_pipeline
from .freezer import Freezer, FrozenLadder


class AgentStatus(str, Enum):
    SUCCESS = "success"
    NEEDS_USER = "needs_user"
    RECOMPILE = "recompile"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True)
class UserInstruction:
    """The only task input RSA needs from an application.

    ``repository`` is deliberately separate from ``instruction``.  A chat UI
    may collect both in one message, while a benchmark may already know the
    repository from its task record.  RSA never receives a benchmark gold recipe.
    """

    repository: str
    instruction: str
    revision: str = ""
    local_path: Path | None = None
    session_id: str = ""


@dataclass(frozen=True)
class InteractionRequest:
    """A request for information or a decision from the user."""

    kind: str  # clarification | asset | approval | criterion_review | escalation
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InteractionResponse:
    """A user response understood by RSA.

    The action vocabulary is intentionally small.  A UI or benchmark can use
    richer controls and translate them here without leaking that policy into
    the setup agent.
    """

    action: str  # answer | retry | drop_assets | continue | approve | stop
    message: str = ""
    assets: tuple[str, ...] = ()
    grant: dict[str, Any] = field(default_factory=dict)


class UserInteraction(Protocol):
    def request(self, event: InteractionRequest) -> InteractionResponse: ...


InteractionHandler = UserInteraction | Callable[[InteractionRequest], InteractionResponse]


@dataclass
class AgentOutcome:
    status: AgentStatus
    request: UserInstruction
    compile: CompileOutcome | None = None
    pipeline: PipelineOutcome | None = None
    frozen: FrozenLadder | None = None
    pending: InteractionRequest | None = None
    history: list[InteractionRequest] = field(default_factory=list)
    pipeline_attempts: list[PipelineOutcome] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status is AgentStatus.SUCCESS


def _ask(handler: InteractionHandler | None, event: InteractionRequest) -> InteractionResponse | None:
    if handler is None:
        return None
    response = handler.request(event) if hasattr(handler, "request") else handler(event)  # type: ignore[misc]
    if isinstance(response, InteractionResponse):
        return response
    if isinstance(response, str):
        return InteractionResponse(action="answer", message=response)
    raise TypeError("user interaction must return InteractionResponse or str")


class RSAAgent:
    """Run RSA as a user-facing environment setup agent.

    ``run`` may be called with no interaction handler.  In that case it returns
    ``NEEDS_USER`` and the exact pending event.  A stateful HTTP/chat service can
    persist that event and invoke RSA again with the user's answer; a CLI or
    benchmark can provide a handler and drive the same loop synchronously.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, request: UserInstruction, *, interaction: InteractionHandler | None = None) -> AgentOutcome:
        """Run one request and release a remote relay connection on every exit."""
        try:
            return self._run(request, interaction=interaction)
        finally:
            backend = getattr(self.config, "remote_backend", None)
            if backend is not None and hasattr(backend, "close"):
                try:
                    backend.close()
                finally:
                    self.config.remote_backend = None

    def _run(self, request: UserInstruction, *, interaction: InteractionHandler | None = None) -> AgentOutcome:
        history: list[InteractionRequest] = []
        pipeline_attempts: list[PipelineOutcome] = []
        clarification = ""

        while True:
            try:
                compiled = compile_ladder(
                    request.repository,
                    request.instruction,
                    self.config,
                    commit=request.revision,
                    clarification=clarification,
                    local_path=request.local_path,
                )
            except Exception as exc:
                return AgentOutcome(
                    AgentStatus.FAILED,
                    request,
                    history=history,
                    pipeline_attempts=pipeline_attempts,
                    error=f"{type(exc).__name__}: {exc}",
                )
            if not compiled.question:
                break
            event = InteractionRequest(
                kind="clarification",
                message=compiled.question,
                data={"instruction": request.instruction},
            )
            history.append(event)
            response = _ask(interaction, event)
            if response is None:
                return AgentOutcome(AgentStatus.NEEDS_USER, request, compile=compiled,
                                    pending=event, history=history,
                                    pipeline_attempts=pipeline_attempts)
            if response.action != "answer" or not response.message.strip():
                return AgentOutcome(AgentStatus.FAILED, request, compile=compiled,
                                    pending=event, history=history,
                                    pipeline_attempts=pipeline_attempts)
            clarification = response.message

        frozen = compiled.frozen
        if frozen is None:
            return AgentOutcome(AgentStatus.FAILED, request, compile=compiled, history=history,
                                pipeline_attempts=pipeline_attempts)

        rejected = [r for r in compiled.falsification.values() if r.must_recompile]
        if rejected:
            event = InteractionRequest(
                kind="criterion_review",
                message="The automatically generated pytest criterion was rejected by falsification.",
                data={"falsification": {k: v.to_dict() for k, v in compiled.falsification.items()}},
            )
            history.append(event)
            response = _ask(interaction, event)
            if response is None:
                return AgentOutcome(AgentStatus.RECOMPILE, request, compile=compiled,
                                    frozen=frozen, pending=event, history=history,
                                    pipeline_attempts=pipeline_attempts)
            return AgentOutcome(AgentStatus.RECOMPILE, request, compile=compiled,
                                frozen=frozen, pending=event, history=history,
                                pipeline_attempts=pipeline_attempts)

        while True:
            try:
                result = run_pipeline(frozen, self.config)
                pipeline_attempts.append(result)
            except Exception as exc:
                return AgentOutcome(
                    AgentStatus.FAILED,
                    request,
                    compile=compiled,
                    frozen=frozen,
                    history=history,
                    pipeline_attempts=pipeline_attempts,
                    error=f"{type(exc).__name__}: {exc}",
                )
            if result.terminal == "success":
                return AgentOutcome(AgentStatus.SUCCESS, request, compile=compiled,
                                    pipeline=result, frozen=frozen, history=history,
                                    pipeline_attempts=pipeline_attempts)

            if result.terminal == "blocked" and result.assets is not None:
                event = self._asset_event(result.assets)
                history.append(event)
                response = _ask(interaction, event)
                if response is None:
                    return AgentOutcome(AgentStatus.NEEDS_USER, request, compile=compiled,
                                        pipeline=result, frozen=frozen, pending=event,
                                        history=history, pipeline_attempts=pipeline_attempts)
                if response.action == "drop_assets":
                    names = set(response.assets) or {
                        r.name for r in result.assets.results if not r.ok
                    }
                    frozen.ladder.assets = [a for a in frozen.ladder.assets if a.name not in names]
                    remote = self.config.remote_backend if self.config.execution_backend != "local" else None
                    frozen = Freezer(self.config.store).freeze(
                        frozen.ladder,
                        overwrite=True,
                        collector=(remote.collect_test_ids if remote is not None else None),
                    )
                    continue
                if response.action == "retry":
                    continue
                return AgentOutcome(AgentStatus.BLOCKED, request, compile=compiled,
                                    pipeline=result, frozen=frozen, pending=event,
                                    history=history, pipeline_attempts=pipeline_attempts)

            if result.terminal == "needs_approval":
                event = InteractionRequest(
                    kind="approval",
                    message=("The next verification rung needs explicit user approval "
                             "because it may consume hours or days."),
                    data={"reached": result.reached,
                          "rungs": [r.rung for r in (result.outcome.per_rung
                                                     if result.outcome else [])]},
                )
                history.append(event)
                response = _ask(interaction, event)
                if response is None:
                    return AgentOutcome(AgentStatus.NEEDS_USER, request, compile=compiled,
                                        pipeline=result, frozen=frozen, pending=event,
                                        history=history, pipeline_attempts=pipeline_attempts)
                if response.action == "approve":
                    self.config.approve.update({Rung.G3, Rung.G4P})
                    continue
                return AgentOutcome(AgentStatus.FAILED, request, compile=compiled,
                                    pipeline=result, frozen=frozen, pending=event,
                                    history=history, pipeline_attempts=pipeline_attempts)

            if result.terminal == "recompile":
                card = result.outcome.card if result.outcome else None
                event = InteractionRequest(
                    kind="criterion_review",
                    message=(card.render() if card else
                             "The generated pytest criterion must be corrected before setup can continue."),
                    data={"card": card.to_dict() if card else None},
                )
                history.append(event)
                _ask(interaction, event)
                return AgentOutcome(AgentStatus.RECOMPILE, request, compile=compiled,
                                    pipeline=result, frozen=frozen, pending=event,
                                    history=history, pipeline_attempts=pipeline_attempts)

            card = result.outcome.card if result.outcome else None
            if card is None:
                return AgentOutcome(AgentStatus.FAILED, request, compile=compiled,
                                    pipeline=result, frozen=frozen, history=history,
                                    pipeline_attempts=pipeline_attempts)
            event = self._escalation_event(
                card,
                result,
                frozen.rungs[Rung(card.rung)].criterion if card.rung else None,
            )
            history.append(event)
            response = _ask(interaction, event)
            if response is None:
                return AgentOutcome(AgentStatus.NEEDS_USER, request, compile=compiled,
                                    pipeline=result, frozen=frozen, pending=event,
                                    history=history, pipeline_attempts=pipeline_attempts)
            if response.action in ("continue", "approve"):
                self._apply_grant(response.grant)
                if response.action == "approve":
                    self.config.approve.update({Rung.G3, Rung.G4P})
                continue
            return AgentOutcome(AgentStatus.FAILED, request, compile=compiled,
                                pipeline=result, frozen=frozen, pending=event,
                                history=history, pipeline_attempts=pipeline_attempts)

    @staticmethod
    def _asset_event(report: AssetReport) -> InteractionRequest:
        return InteractionRequest(
            kind="asset",
            message="Some declared inputs or machine assets are unavailable.",
            data=report.to_dict(),
        )

    @staticmethod
    def _escalation_event(card: EscalationCard, result: PipelineOutcome,
                          criterion: Any = None) -> InteractionRequest:
        return InteractionRequest(
            kind="escalation",
            message=card.render(),
            data={"card": card.to_dict(), "criterion": criterion.to_dict() if criterion else None,
                  "terminal": result.terminal},
        )

    def _apply_grant(self, grant: dict[str, Any]) -> None:
        if "wall_seconds" in grant:
            self.config.wall_seconds = int(grant["wall_seconds"])
        elif "wall_multiplier" in grant:
            self.config.wall_seconds = int(self.config.wall_seconds * float(grant["wall_multiplier"]))
        if "max_rounds" in grant:
            self.config.max_rounds = int(grant["max_rounds"])
        elif "extra_rounds" in grant:
            self.config.max_rounds += int(grant["extra_rounds"])


def run_user_instruction(
    repository: str,
    instruction: str,
    config: PipelineConfig,
    *,
    interaction: InteractionHandler | None = None,
    revision: str = "",
    local_path: Path | None = None,
) -> AgentOutcome:
    """Convenience entry point for real UIs and simulated users."""
    return RSAAgent(config).run(
        UserInstruction(repository=repository, instruction=instruction,
                        revision=revision, local_path=local_path),
        interaction=interaction,
    )

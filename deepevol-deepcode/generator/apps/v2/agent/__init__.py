"""Lazy public exports for the isolated V2 Agent domains.

Importing an execution or control-plane submodule must not load Gateway code or
Provider adapters.  Explicit public imports remain compatible and resolve only
the module that owns the requested symbol.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "AesGcmKnownResultSealer": (
        ".spool",
        "AesGcmKnownResultSealer",
    ),
    "ArtifactPublishedInboxConsumer": (
        ".artifact_receipts",
        "ArtifactPublishedInboxConsumer",
    ),
    "AskUserAnswerInboxConsumer": (
        ".user_interaction",
        "AskUserAnswerInboxConsumer",
    ),
    "AgentRuntimeRoles": (".repository", "AgentRuntimeRoles"),
    "AgentSchemas": (".repository", "AgentSchemas"),
    "AgentOutboxCrashPoint": (".outbox", "AgentOutboxCrashPoint"),
    "AuthorityFunctions": (".repository", "AuthorityFunctions"),
    "CheckpointCommit": (".checkpoint_runtime", "CheckpointCommit"),
    "CheckpointReference": (".checkpoint_runtime", "CheckpointReference"),
    "CheckpointReferenceKind": (
        ".checkpoint_runtime",
        "CheckpointReferenceKind",
    ),
    "CommittedCheckpoint": (".checkpoint_runtime", "CommittedCheckpoint"),
    "EncryptedJsonlKnownResultSpool": (
        ".spool",
        "EncryptedJsonlKnownResultSpool",
    ),
    "FakeProductExposurePort": (".gateway", "FakeProductExposurePort"),
    "FakeProviderTransport": (".gateway", "FakeProviderTransport"),
    "GatewayService": (".gateway", "GatewayService"),
    "GatewayCrashPoint": (".gateway", "GatewayCrashPoint"),
    "GatewayChatBackend": (".gateway_chat_model", "GatewayChatBackend"),
    "GatewayChatModel": (".gateway_chat_model", "GatewayChatModel"),
    "GatewayChatRequest": (".gateway_chat_model", "GatewayChatRequest"),
    "GatewayChatResponse": (".gateway_chat_model", "GatewayChatResponse"),
    "GatewayInvocationContext": (".gateway_chat_model", "GatewayInvocationContext"),
    "KnownResultSpool": (".spool", "KnownResultSpool"),
    "LocalRunAuthorizationVerifier": (
        ".inbox",
        "LocalRunAuthorizationVerifier",
    ),
    "ProductExposurePort": (".gateway", "ProductExposurePort"),
    "ProviderCapabilityRegistry": (
        ".capabilities",
        "ProviderCapabilityRegistry",
    ),
    "ProviderTransport": (".gateway", "ProviderTransport"),
    "PsycopgCheckpointRuntime": (
        ".checkpoint_runtime",
        "PsycopgCheckpointRuntime",
    ),
    "PsycopgExecutionRepository": (
        ".repository",
        "PsycopgExecutionRepository",
    ),
    "PsycopgAskUserRepository": (
        ".user_interaction",
        "PsycopgAskUserRepository",
    ),
    "PsycopgGatewayRepository": (".repository", "PsycopgGatewayRepository"),
    "PsycopgReconciliationRepository": (
        ".repository",
        "PsycopgReconciliationRepository",
    ),
    "RecoveryCandidate": (".checkpoint_runtime", "RecoveryCandidate"),
    "RecoveryLease": (".checkpoint_runtime", "RecoveryLease"),
    "RecoveryRequest": (".checkpoint_runtime", "RecoveryRequest"),
    "RunCancelRequestedInboxConsumer": (
        ".inbox",
        "RunCancelRequestedInboxConsumer",
    ),
    "RunCancelRequestedInboxRepository": (
        ".inbox",
        "RunCancelRequestedInboxRepository",
    ),
    "RunRequestedInboxConsumer": (".inbox", "RunRequestedInboxConsumer"),
    "RunRequestedInboxRepository": (
        ".inbox",
        "RunRequestedInboxRepository",
    ),
    "VerifiedRunAuthorization": (".inbox", "VerifiedRunAuthorization"),
}

__all__ = [
    "AesGcmKnownResultSealer",
    "AgentOutboxCrashPoint",
    "AgentRuntimeRoles",
    "AgentSchemas",
    "ArtifactPublishedInboxConsumer",
    "AskUserAnswerInboxConsumer",
    "AuthorityFunctions",
    "CheckpointCommit",
    "CheckpointReference",
    "CheckpointReferenceKind",
    "CommittedCheckpoint",
    "EncryptedJsonlKnownResultSpool",
    "FakeProductExposurePort",
    "FakeProviderTransport",
    "GatewayChatBackend",
    "GatewayChatModel",
    "GatewayChatRequest",
    "GatewayChatResponse",
    "GatewayCrashPoint",
    "GatewayInvocationContext",
    "GatewayService",
    "KnownResultSpool",
    "LocalRunAuthorizationVerifier",
    "ProductExposurePort",
    "ProviderCapabilityRegistry",
    "ProviderTransport",
    "PsycopgAskUserRepository",
    "PsycopgCheckpointRuntime",
    "PsycopgExecutionRepository",
    "PsycopgGatewayRepository",
    "PsycopgReconciliationRepository",
    "RecoveryCandidate",
    "RecoveryLease",
    "RecoveryRequest",
    "RunCancelRequestedInboxConsumer",
    "RunCancelRequestedInboxRepository",
    "RunRequestedInboxConsumer",
    "RunRequestedInboxRepository",
    "VerifiedRunAuthorization",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})

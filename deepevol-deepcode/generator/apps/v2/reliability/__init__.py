"""Reusable V2 idempotency, retry, lease, and fault-injection primitives."""

from .canonical import canonical_json_bytes, canonical_json_sha256, sha256_hex
from .failpoints import (
    FAILPOINT_ACTION_VARIABLE,
    FAILPOINT_COMPONENT_VARIABLE,
    FAILPOINT_ENABLE_VARIABLE,
    FAILPOINT_LIST_VARIABLE,
    TEST_ENVIRONMENT_VARIABLE,
    FailpointConfigurationError,
    FailpointTriggered,
    configured_failpoints,
    enum_failpoint_injector,
    failpoint_action,
    failpoints_enabled,
    is_failpoint_enabled,
    trigger_failpoint,
)
from .heartbeat import HeartbeatFailed, HeartbeatInvariantError, HeartbeatRunner
from .deadline_transport import http_egress_deadline, http_egress_deadline_limit
from .event_replay import (
    EventReplayError,
    EventReplayPage,
    replay_window_requires_resync,
)
from .failure_classification import (
    EffectCertainty,
    FailureObservation,
    FailureSignal,
    classify_failure,
)
from .models import FailureClass, Lease, OperationIdentity, RetryDecision, RetryDisposition
from .retry import (
    RETRY_POLICY_V1,
    RETRY_POLICY_VERSION,
    RetryAfter,
    RetryBudget,
    decide_retry,
    parse_retry_after,
    policy_for,
)

__all__ = [
    "FAILPOINT_ACTION_VARIABLE",
    "FAILPOINT_COMPONENT_VARIABLE",
    "FAILPOINT_ENABLE_VARIABLE",
    "FAILPOINT_LIST_VARIABLE",
    "RETRY_POLICY_V1",
    "RETRY_POLICY_VERSION",
    "TEST_ENVIRONMENT_VARIABLE",
    "EffectCertainty",
    "EventReplayError",
    "EventReplayPage",
    "FailpointConfigurationError",
    "FailpointTriggered",
    "FailureClass",
    "FailureObservation",
    "FailureSignal",
    "HeartbeatFailed",
    "HeartbeatInvariantError",
    "HeartbeatRunner",
    "Lease",
    "OperationIdentity",
    "RetryAfter",
    "RetryBudget",
    "RetryDecision",
    "RetryDisposition",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "classify_failure",
    "configured_failpoints",
    "decide_retry",
    "enum_failpoint_injector",
    "failpoint_action",
    "failpoints_enabled",
    "http_egress_deadline",
    "http_egress_deadline_limit",
    "is_failpoint_enabled",
    "parse_retry_after",
    "policy_for",
    "replay_window_requires_resync",
    "sha256_hex",
    "trigger_failpoint",
]

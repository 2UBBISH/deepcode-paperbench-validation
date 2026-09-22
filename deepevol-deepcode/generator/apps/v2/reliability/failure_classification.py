"""Conservative, structured failure classification for ``retry-policy-v1``.

Adapters classify facts at their effect boundary and pass the resulting
``FailureClass`` to the durable retry authority.  This module deliberately
does not inspect exception messages: a translated exception/status must carry
explicit effect certainty so an uncertain Provider call can never be turned
into a blind retry by string matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .models import FailureClass


class FailureSignal(StrEnum):
    """Stable signal emitted by a domain or transport adapter."""

    PERSISTENCE = "PERSISTENCE"
    NETWORK = "NETWORK"
    HTTP = "HTTP"
    CAPACITY = "CAPACITY"
    TIMEOUT = "TIMEOUT"
    LEASE = "LEASE"
    EXTERNAL_WAIT = "EXTERNAL_WAIT"
    AUTHORIZATION = "AUTHORIZATION"
    COMPATIBILITY = "COMPATIBILITY"
    VALIDATION = "VALIDATION"
    SECURITY = "SECURITY"
    CODE = "CODE"
    CANCELLED = "CANCELLED"
    PERMANENT = "PERMANENT"


class EffectCertainty(StrEnum):
    """What is durably known about the failed operation's business effect."""

    PRE_EFFECT = "PRE_EFFECT"
    CONFIRMED_NO_EFFECT = "CONFIRMED_NO_EFFECT"
    EFFECT_MAY_HAVE_OCCURRED = "EFFECT_MAY_HAVE_OCCURRED"
    RESULT_KNOWN = "RESULT_KNOWN"


@dataclass(frozen=True, slots=True)
class FailureObservation:
    """A bounded, message-free input to the shared classifier.

    ``RESULT_KNOWN`` means the external result is already durable somewhere
    safe (for example a local result spool) and only its authoritative commit
    remains.  ``EFFECT_MAY_HAVE_OCCURRED`` always wins over the transport
    signal and requires reconciliation.
    """

    signal: FailureSignal
    effect_certainty: EffectCertainty = EffectCertainty.PRE_EFFECT
    status_code: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.signal, FailureSignal):
            raise TypeError("signal must be a FailureSignal")
        if not isinstance(self.effect_certainty, EffectCertainty):
            raise TypeError("effect_certainty must be an EffectCertainty")
        if self.signal is FailureSignal.HTTP:
            if (
                not isinstance(self.status_code, int)
                or isinstance(self.status_code, bool)
                or not 100 <= self.status_code <= 599
            ):
                raise ValueError("HTTP failure observations require a valid status_code")
        elif self.status_code is not None:
            raise ValueError("status_code is only valid for an HTTP failure signal")


def classify_failure(observation: FailureObservation) -> FailureClass:
    """Map structured evidence to the locked retry-policy category.

    The certainty checks intentionally precede every signal-specific rule.
    Once an effect may have happened, even a timeout or 503 cannot be treated
    as a pre-effect retry.  Conversely, a known result is committed repeatedly
    without invoking the Provider again.
    """

    if not isinstance(observation, FailureObservation):
        raise TypeError("observation must be a FailureObservation")
    if observation.effect_certainty is EffectCertainty.RESULT_KNOWN:
        return FailureClass.KNOWN_RESULT_COMMIT_PENDING
    if observation.effect_certainty is EffectCertainty.EFFECT_MAY_HAVE_OCCURRED:
        return FailureClass.POST_EFFECT_UNKNOWN

    signal = observation.signal
    if signal is FailureSignal.PERSISTENCE:
        return FailureClass.INTERNAL_PERSISTENCE_TRANSIENT
    if signal is FailureSignal.NETWORK:
        return FailureClass.NETWORK_PRE_EFFECT
    if signal is FailureSignal.CAPACITY:
        return FailureClass.RESOURCE_CAPACITY
    if signal is FailureSignal.TIMEOUT:
        return FailureClass.SAFE_TIMEOUT
    if signal is FailureSignal.LEASE:
        return FailureClass.LEASE_LOST
    if signal is FailureSignal.EXTERNAL_WAIT:
        return FailureClass.WAIT_EXTERNAL
    if signal is FailureSignal.AUTHORIZATION:
        return FailureClass.AUTHORIZATION_STALE
    if signal is FailureSignal.COMPATIBILITY:
        return FailureClass.COMPATIBILITY_MISMATCH
    if signal is FailureSignal.VALIDATION:
        return FailureClass.VALIDATION
    if signal is FailureSignal.SECURITY:
        return FailureClass.SECURITY
    if signal is FailureSignal.CODE:
        return FailureClass.BUG
    if signal is FailureSignal.CANCELLED:
        return FailureClass.CANCELLED
    if signal is FailureSignal.PERMANENT:
        return FailureClass.PERMANENT

    assert signal is FailureSignal.HTTP
    assert observation.status_code is not None
    status = observation.status_code
    if status == 429:
        return FailureClass.RATE_LIMITED_PRE_EFFECT
    if status in {503, 507}:
        return FailureClass.RESOURCE_CAPACITY
    if status in {408, 504}:
        return FailureClass.SAFE_TIMEOUT
    if status in {401, 403}:
        return FailureClass.SECURITY
    if status in {400, 404, 409, 410, 412, 415, 422}:
        return FailureClass.VALIDATION
    if 500 <= status <= 599:
        return FailureClass.NETWORK_PRE_EFFECT
    return FailureClass.PERMANENT


__all__ = [
    "EffectCertainty",
    "FailureObservation",
    "FailureSignal",
    "classify_failure",
]

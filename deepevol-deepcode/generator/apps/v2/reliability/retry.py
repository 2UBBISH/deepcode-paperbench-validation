"""Versioned, side-effect-aware retry decisions for durable V2 work."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from types import MappingProxyType
from typing import TypeAlias

from .models import FailureClass, RetryDecision, RetryDisposition


RETRY_POLICY_VERSION = "retry-policy-v1"
RetryAfter: TypeAlias = str | int | float | timedelta | datetime | None


@dataclass(frozen=True, slots=True)
class RetryBudget:
    disposition: RetryDisposition
    max_attempts: int | None
    max_elapsed: timedelta | None
    base_delay: timedelta
    max_delay: timedelta
    honor_retry_after: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, RetryDisposition):
            raise TypeError("retry budget disposition must be a RetryDisposition")
        if self.max_attempts is not None and (
            not isinstance(self.max_attempts, int) or isinstance(self.max_attempts, bool) or self.max_attempts < 0
        ):
            raise ValueError("retry budget max_attempts must be non-negative or None")
        if self.max_elapsed is not None and self.max_elapsed <= timedelta(0):
            raise ValueError("retry budget max_elapsed must be positive")
        if self.base_delay < timedelta(0) or self.max_delay < self.base_delay:
            raise ValueError("retry budget delay bounds are invalid")
        scheduled = self.disposition in {RetryDisposition.RETRY, RetryDisposition.WAIT}
        if scheduled and self.max_delay <= timedelta(0):
            raise ValueError("scheduled retry budgets require a positive delay bound")
        if not scheduled and (self.base_delay != timedelta(0) or self.max_delay != timedelta(0)):
            raise ValueError("non-scheduled retry budgets cannot carry delay bounds")
        if self.disposition is RetryDisposition.RETRY and not self.max_attempts:
            raise ValueError("automatic retry budgets require at least one attempt")


def _budget(
    disposition: RetryDisposition,
    max_attempts: int | None = 0,
    max_elapsed: timedelta | None = None,
    base_delay: timedelta = timedelta(0),
    max_delay: timedelta = timedelta(0),
    *,
    honor_retry_after: bool = False,
) -> RetryBudget:
    return RetryBudget(
        disposition,
        max_attempts,
        max_elapsed,
        base_delay,
        max_delay,
        honor_retry_after,
    )


RETRY_POLICY_V1: Mapping[FailureClass, RetryBudget] = MappingProxyType(
    {
        FailureClass.INTERNAL_PERSISTENCE_TRANSIENT: _budget(
            RetryDisposition.RETRY,
            12,
            timedelta(minutes=30),
            timedelta(milliseconds=250),
            timedelta(seconds=30),
        ),
        FailureClass.NETWORK_PRE_EFFECT: _budget(
            RetryDisposition.RETRY,
            8,
            timedelta(minutes=30),
            timedelta(seconds=1),
            timedelta(minutes=2),
            honor_retry_after=True,
        ),
        FailureClass.RATE_LIMITED_PRE_EFFECT: _budget(
            RetryDisposition.RETRY,
            12,
            timedelta(hours=1),
            timedelta(seconds=2),
            timedelta(minutes=5),
            honor_retry_after=True,
        ),
        FailureClass.RESOURCE_CAPACITY: _budget(
            RetryDisposition.RETRY,
            20,
            timedelta(hours=2),
            timedelta(seconds=5),
            timedelta(minutes=5),
            honor_retry_after=True,
        ),
        FailureClass.SAFE_TIMEOUT: _budget(
            RetryDisposition.RETRY,
            3,
            timedelta(minutes=30),
            timedelta(seconds=5),
            timedelta(minutes=1),
        ),
        FailureClass.LEASE_LOST: _budget(
            RetryDisposition.WAIT,
            None,
            None,
            timedelta(seconds=1),
            timedelta(seconds=1),
            honor_retry_after=True,
        ),
        FailureClass.WAIT_EXTERNAL: _budget(
            RetryDisposition.WAIT,
            None,
            None,
            timedelta(seconds=5),
            timedelta(minutes=1),
            honor_retry_after=True,
        ),
        FailureClass.KNOWN_RESULT_COMMIT_PENDING: _budget(
            RetryDisposition.RETRY,
            100,
            timedelta(hours=24),
            timedelta(seconds=1),
            timedelta(minutes=5),
        ),
        FailureClass.POST_EFFECT_UNKNOWN: _budget(RetryDisposition.RECONCILE),
        FailureClass.AUTHORIZATION_STALE: _budget(RetryDisposition.BLOCKED),
        FailureClass.COMPATIBILITY_MISMATCH: _budget(RetryDisposition.BLOCKED),
        FailureClass.VALIDATION: _budget(RetryDisposition.TERMINAL),
        FailureClass.SECURITY: _budget(RetryDisposition.TERMINAL),
        FailureClass.BUG: _budget(RetryDisposition.TERMINAL),
        FailureClass.CANCELLED: _budget(RetryDisposition.TERMINAL),
        FailureClass.PERMANENT: _budget(RetryDisposition.TERMINAL),
    }
)


def _require_aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _safe_add(value: datetime, delay: timedelta) -> datetime:
    try:
        return value + delay
    except OverflowError:
        return datetime.max.replace(tzinfo=UTC)


def parse_retry_after(value: RetryAfter, *, now: datetime) -> datetime | None:
    """Parse HTTP Retry-After delta-seconds/date or an internal time value.

    Malformed external strings and negative delays are ignored. Unsupported
    Python types are programmer errors and are rejected.
    """

    _require_aware(now, "now")
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("Retry-After cannot be boolean")
    if isinstance(value, datetime):
        _require_aware(value, "Retry-After datetime")
        return value
    if isinstance(value, timedelta):
        return None if value < timedelta(0) else _safe_add(now, value)
    if isinstance(value, (int, float)):
        seconds = float(value)
        if not math.isfinite(seconds) or seconds < 0:
            return None
        return _safe_add(now, timedelta(seconds=seconds))
    if not isinstance(value, str):
        raise TypeError("unsupported Retry-After value")
    raw = value.strip()
    if not raw:
        return None
    if raw.isascii() and raw.isdigit():
        try:
            return _safe_add(now, timedelta(seconds=int(raw)))
        except OverflowError:
            return datetime.max.replace(tzinfo=UTC)
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed is None or parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def policy_for(
    failure_class: FailureClass,
    *,
    policy: Mapping[FailureClass, RetryBudget] = RETRY_POLICY_V1,
) -> RetryBudget:
    if not isinstance(failure_class, FailureClass):
        raise TypeError("failure_class must be a FailureClass")
    try:
        return policy[failure_class]
    except KeyError as exc:
        raise ValueError(f"retry policy has no budget for {failure_class.value}") from exc


def _effective_deadline(
    *,
    first_failure_at: datetime,
    budget: RetryBudget,
    explicit_deadline_at: datetime | None,
) -> datetime | None:
    deadlines: list[datetime] = []
    if budget.max_elapsed is not None:
        deadlines.append(_safe_add(first_failure_at, budget.max_elapsed))
    if explicit_deadline_at is not None:
        _require_aware(explicit_deadline_at, "deadline_at")
        deadlines.append(explicit_deadline_at)
    return min(deadlines) if deadlines else None


def _decision(
    failure_class: FailureClass,
    budget: RetryBudget,
    disposition: RetryDisposition,
    *,
    attempt: int,
    first_failure_at: datetime,
    now: datetime,
    deadline_at: datetime | None,
    next_attempt_at: datetime | None = None,
    retry_after_at: datetime | None = None,
    retry_after_honored: bool = False,
    exhausted_reason: str | None = None,
) -> RetryDecision:
    return RetryDecision(
        policy_version=RETRY_POLICY_VERSION,
        failure_class=failure_class,
        disposition=disposition,
        attempt=attempt,
        max_attempts=budget.max_attempts,
        first_failure_at=first_failure_at,
        decided_at=now,
        deadline_at=deadline_at,
        next_attempt_at=next_attempt_at,
        retry_after_at=retry_after_at,
        retry_after_honored=retry_after_honored,
        exhausted_reason=exhausted_reason,
    )


def _resolve_attempt(*, attempt: int | None, attempt_number: int | None) -> int:
    if attempt is None and attempt_number is None:
        raise TypeError("attempt is required")
    if attempt is not None and attempt_number is not None and attempt != attempt_number:
        raise ValueError("attempt and attempt_number must match when both are provided")
    resolved = attempt if attempt is not None else attempt_number
    if not isinstance(resolved, int) or isinstance(resolved, bool) or resolved < 1:
        raise ValueError("attempt must be a positive integer")
    return resolved


def decide_retry(
    failure_class: FailureClass,
    *,
    first_failure_at: datetime,
    now: datetime,
    attempt: int | None = None,
    attempt_number: int | None = None,
    retry_after: RetryAfter = None,
    deadline_at: datetime | None = None,
    random_source: Callable[[], float] | None = None,
    policy: Mapping[FailureClass, RetryBudget] = RETRY_POLICY_V1,
) -> RetryDecision:
    """Evaluate one failure using capped exponential backoff and full jitter.

    ``attempt`` is the one-based retry ordinal. ``attempt_number`` remains a
    compatibility input alias. An eight-attempt budget schedules ordinals 1..8;
    ordinal 9 returns ``EXHAUSTED``. Callers persist the returned decision
    instead of sleeping in a worker process.
    """

    resolved_attempt = _resolve_attempt(attempt=attempt, attempt_number=attempt_number)
    _require_aware(first_failure_at, "first_failure_at")
    _require_aware(now, "now")
    if first_failure_at > now:
        raise ValueError("first_failure_at cannot be after now")

    budget = policy_for(failure_class, policy=policy)
    effective_deadline = _effective_deadline(
        first_failure_at=first_failure_at,
        budget=budget,
        explicit_deadline_at=deadline_at,
    )
    parsed_retry_after = parse_retry_after(retry_after, now=now)

    if budget.disposition not in {RetryDisposition.RETRY, RetryDisposition.WAIT}:
        return _decision(
            failure_class,
            budget,
            budget.disposition,
            attempt=resolved_attempt,
            first_failure_at=first_failure_at,
            now=now,
            deadline_at=effective_deadline,
            retry_after_at=parsed_retry_after,
        )

    if budget.max_attempts is not None and resolved_attempt > budget.max_attempts:
        return _decision(
            failure_class,
            budget,
            RetryDisposition.EXHAUSTED,
            attempt=resolved_attempt,
            first_failure_at=first_failure_at,
            now=now,
            deadline_at=effective_deadline,
            retry_after_at=parsed_retry_after,
            exhausted_reason="ATTEMPT_BUDGET_EXHAUSTED",
        )
    if effective_deadline is not None and now >= effective_deadline:
        return _decision(
            failure_class,
            budget,
            RetryDisposition.EXHAUSTED,
            attempt=resolved_attempt,
            first_failure_at=first_failure_at,
            now=now,
            deadline_at=effective_deadline,
            retry_after_at=parsed_retry_after,
            exhausted_reason="RETRY_DEADLINE_EXHAUSTED",
        )

    exponent = min(resolved_attempt - 1, 62)
    delay_cap_seconds = min(
        budget.max_delay.total_seconds(),
        budget.base_delay.total_seconds() * (2**exponent),
    )
    source = random.random if random_source is None else random_source
    random_unit = source()
    if (
        not isinstance(random_unit, (int, float))
        or isinstance(random_unit, bool)
        or not math.isfinite(float(random_unit))
        or not 0 <= float(random_unit) < 1
    ):
        raise ValueError("random_source must return a finite value in [0, 1)")
    next_attempt_at = _safe_add(now, timedelta(seconds=delay_cap_seconds * float(random_unit)))
    retry_after_honored = False
    if budget.honor_retry_after and parsed_retry_after is not None and parsed_retry_after > next_attempt_at:
        next_attempt_at = parsed_retry_after
        retry_after_honored = True
    if effective_deadline is not None and next_attempt_at >= effective_deadline:
        return _decision(
            failure_class,
            budget,
            RetryDisposition.EXHAUSTED,
            attempt=resolved_attempt,
            first_failure_at=first_failure_at,
            now=now,
            deadline_at=effective_deadline,
            retry_after_at=parsed_retry_after,
            retry_after_honored=retry_after_honored,
            exhausted_reason=("RETRY_AFTER_EXCEEDS_DEADLINE" if retry_after_honored else "RETRY_DEADLINE_EXHAUSTED"),
        )
    return _decision(
        failure_class,
        budget,
        budget.disposition,
        attempt=resolved_attempt,
        first_failure_at=first_failure_at,
        now=now,
        deadline_at=effective_deadline,
        next_attempt_at=next_attempt_at,
        retry_after_at=parsed_retry_after,
        retry_after_honored=retry_after_honored,
    )


__all__ = [
    "RETRY_POLICY_V1",
    "RETRY_POLICY_VERSION",
    "RetryAfter",
    "RetryBudget",
    "decide_retry",
    "parse_retry_after",
    "policy_for",
]

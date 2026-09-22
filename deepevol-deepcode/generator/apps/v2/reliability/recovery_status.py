"""Bounded, non-secret recovery metadata shared by Agent and Product."""
from collections.abc import Mapping
from datetime import datetime

from .models import FailureClass


def validate_recovery_status(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"status", "failure_class", "retry_at"}:
        raise ValueError("invalid recovery status fields")
    status = value["status"]
    if status not in {"WAITING_RETRY", "WAITING_RECONCILIATION", "WAITING_EXTERNAL"}:
        raise ValueError("invalid recovery status")
    failure = FailureClass(value["failure_class"])
    retry_at = value["retry_at"]
    if retry_at is not None:
        if not isinstance(retry_at, str) or len(retry_at) > 64:
            raise ValueError("invalid recovery retry_at")
        parsed = datetime.fromisoformat(retry_at)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("recovery retry_at requires a timezone")
    if (status == "WAITING_RETRY") != (retry_at is not None):
        raise ValueError("only scheduled recovery has a retry_at")
    if (status == "WAITING_RECONCILIATION") != (failure is FailureClass.POST_EFFECT_UNKNOWN):
        raise ValueError("unknown effects require reconciliation")
    return {"status": status, "failure_class": failure.value, "retry_at": retry_at}

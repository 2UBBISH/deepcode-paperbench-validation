"""Small, JSON-safe evidence records shared by component tests.

The record is intentionally a plain dataclass rather than a test DSL.  It
gives failure reports one stable vocabulary while keeping secrets and raw
identity identifiers out of artifacts.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any
from uuid import UUID


def _redact_identifier(value: UUID | str | None) -> str | None:
    if value is None:
        return None
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass(frozen=True, slots=True)
class TestEvidence:
    """Stable, secret-free evidence for one test operation."""

    # Pytest scans imported names beginning with ``Test`` when collecting
    # classes from test modules.  This is a support dataclass, not a test.
    __test__ = False

    module: str
    operation: str
    outcome: str
    payer_tid: UUID | str | None = None
    operation_id: UUID | str | None = None
    catalog_version: str | None = None
    trace_id: str | None = None
    transaction_result: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return an artifact-safe mapping with payer identity redacted."""

        evidence = asdict(self)
        evidence["payer_tid"] = _redact_identifier(self.payer_tid)
        if self.operation_id is not None:
            evidence["operation_id"] = str(self.operation_id)
        return evidence


__all__ = ["TestEvidence"]

"""Fail-closed gate used before any V2 business repository is enabled."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .specs import DatabaseSpec


class ContractGateClosed(RuntimeError):
    """Raised while a database only contains the infrastructure baseline."""


def require_business_schema(connection: Any, spec: DatabaseSpec) -> str:
    """Return the frozen contract hash or reject an incomplete V2 database.

    ``connection`` deliberately follows the small DB-API/psycopg ``execute``
    protocol so runtime packages do not need to import migration machinery.
    """

    row = connection.execute(
        f"""
        SELECT contract_state, business_schema_enabled, contract_sha256
        FROM {spec.control_schema}.schema_release_gate
        WHERE domain = %s
        """,
        (spec.domain,),
    ).fetchone()
    if row is None:
        raise ContractGateClosed(f"{spec.domain} V2 schema release gate is missing")
    if isinstance(row, Mapping):
        state = row["contract_state"]
        enabled = row["business_schema_enabled"]
        contract_sha256 = row["contract_sha256"]
    else:
        state, enabled, contract_sha256 = row
    if state != "BUSINESS_SCHEMA_READY" or enabled is not True or contract_sha256 is None:
        raise ContractGateClosed(
            f"{spec.domain} V2 business schema is not released (state={state!r}, enabled={enabled!r})"
        )
    actual_contract_sha256 = str(contract_sha256)
    expected_contract_sha256 = spec.expected_contract_sha256
    if actual_contract_sha256 != expected_contract_sha256:
        raise ContractGateClosed(
            f"{spec.domain} V2 business schema contract hash is stale "
            f"(database={actual_contract_sha256}, expected={expected_contract_sha256})"
        )
    return actual_contract_sha256

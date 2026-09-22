"""Execute an immutable, hash-pinned SQL resource one statement at a time."""

from __future__ import annotations

import hashlib
from pathlib import Path

from sqlalchemy.engine import Connection

_SQL_MARKER = "\n-- deepevol:statement\n"


def _split_rendered_sql(document: str) -> tuple[str, ...]:
    return tuple(
        statement.strip()
        for statement in document.split(_SQL_MARKER)
        if statement.strip()
    )


def execute_sql_resource(connection: Connection, path: Path, expected_sha256: str) -> None:
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"immutable migration resource hash mismatch for {path.name}: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
    document = payload.decode("utf-8")
    statements = _split_rendered_sql(document)
    if not statements:
        raise RuntimeError(f"immutable migration resource is empty: {path}")
    raw_connection = connection.execution_options(no_parameters=True)
    for statement in statements:
        raw_connection.exec_driver_sql(statement)

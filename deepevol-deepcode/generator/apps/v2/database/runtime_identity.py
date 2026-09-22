"""Fail-closed activation checks for V2 PostgreSQL runtime principals."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from psycopg import sql


_ROLE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class RuntimeIdentityError(RuntimeError):
    """Raised when a connection could bypass its contracted runtime role."""


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    session_user: str
    current_user: str


def _role_name(value: str) -> str:
    if _ROLE_NAME.fullmatch(value) is None:
        raise ValueError(f"invalid PostgreSQL role name: {value!r}")
    return value


def _row_value(row: Any, name: str, position: int) -> Any:
    if isinstance(row, Mapping):
        return row[name]
    return row[position]


def activate_runtime_role(
    connection: Any,
    *,
    expected_role: str,
    allowed_roles: Iterable[str] = (),
    forbidden_roles: Iterable[str] = (),
) -> RuntimeIdentity:
    """SET ROLE and prove that neither login nor group can bypass V2 isolation."""

    expected_role = _role_name(expected_role)
    allowed = {expected_role, *(_role_name(role) for role in allowed_roles)}
    forbidden = tuple(_role_name(role) for role in forbidden_roles)
    connection.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(expected_role)))
    row = connection.execute(
        """
        SELECT
            session_user::text AS session_user,
            current_user::text AS current_user,
            session_role_row.rolcanlogin AS session_can_login,
            session_role_row.rolinherit AS session_inherit,
            session_role_row.rolsuper AS session_super,
            session_role_row.rolcreatedb AS session_createdb,
            session_role_row.rolcreaterole AS session_createrole,
            session_role_row.rolreplication AS session_replication,
            session_role_row.rolbypassrls AS session_bypassrls,
            active_role_row.rolcanlogin AS current_can_login,
            active_role_row.rolinherit AS current_inherit,
            active_role_row.rolsuper AS current_super,
            active_role_row.rolcreatedb AS current_createdb,
            active_role_row.rolcreaterole AS current_createrole,
            active_role_row.rolreplication AS current_replication,
            active_role_row.rolbypassrls AS current_bypassrls,
            pg_has_role(session_user, current_user, 'MEMBER') AS valid_membership
        FROM pg_roles AS session_role_row
        JOIN pg_roles AS active_role_row ON active_role_row.rolname = current_user
        WHERE session_role_row.rolname = session_user
        """
    ).fetchone()
    if row is None:
        raise RuntimeIdentityError("PostgreSQL runtime identity metadata is unavailable")

    session_user = str(_row_value(row, "session_user", 0))
    current_user = str(_row_value(row, "current_user", 1))
    if current_user != expected_role:
        raise RuntimeIdentityError(
            f"runtime role activation mismatch: expected {expected_role}, got {current_user}"
        )
    if session_user == current_user:
        raise RuntimeIdentityError("runtime DSN must use a separate non-privileged LOGIN principal")

    session_flags = tuple(bool(_row_value(row, name, position)) for name, position in (
        ("session_can_login", 2),
        ("session_inherit", 3),
        ("session_super", 4),
        ("session_createdb", 5),
        ("session_createrole", 6),
        ("session_replication", 7),
        ("session_bypassrls", 8),
    ))
    if session_flags != (True, False, False, False, False, False, False):
        raise RuntimeIdentityError("runtime LOGIN principal has forbidden PostgreSQL attributes")

    current_flags = tuple(bool(_row_value(row, name, position)) for name, position in (
        ("current_can_login", 9),
        ("current_inherit", 10),
        ("current_super", 11),
        ("current_createdb", 12),
        ("current_createrole", 13),
        ("current_replication", 14),
        ("current_bypassrls", 15),
    ))
    if current_flags != (False, False, False, False, False, False, False):
        raise RuntimeIdentityError("runtime group role has forbidden PostgreSQL attributes")
    if not bool(_row_value(row, "valid_membership", 16)):
        raise RuntimeIdentityError("runtime LOGIN principal is not a member of the expected role")

    memberships = {
        str(_row_value(item, "rolname", 0))
        for item in connection.execute(
            """
            SELECT member_role_row.rolname
            FROM pg_roles AS member_role_row
            WHERE member_role_row.rolname <> session_user
              AND pg_has_role(session_user, member_role_row.oid, 'MEMBER')
            """
        ).fetchall()
    }
    unexpected_memberships = sorted(memberships - allowed)
    if unexpected_memberships:
        raise RuntimeIdentityError(
            "runtime LOGIN principal has unexpected role memberships: "
            + ", ".join(unexpected_memberships)
        )

    for forbidden_role in forbidden:
        is_member = connection.execute(
            "SELECT pg_has_role(session_user, %s, 'MEMBER')",
            (forbidden_role,),
        ).fetchone()
        if is_member is not None and bool(_row_value(is_member, "pg_has_role", 0)):
            raise RuntimeIdentityError(
                f"runtime LOGIN principal is a member of forbidden role {forbidden_role}"
            )

    return RuntimeIdentity(session_user=session_user, current_user=current_user)


__all__ = ["RuntimeIdentity", "RuntimeIdentityError", "activate_runtime_role"]

"""Provision fail-closed LOGIN principals for contracted V2 runtime roles."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from psycopg import sql

from .contract_roles import DatabaseRoleCatalog, cutover_verifier_role


_ROLE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_GUC_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class RuntimePrincipalError(RuntimeError):
    """Raised when a runtime LOGIN would gain privileges outside its contract."""


@dataclass(frozen=True, slots=True)
class RuntimePrincipalSpec:
    login_role: str
    runtime_roles: tuple[str, ...]
    database: str
    read_only: bool = False


def build_runtime_principal_spec(
    catalog: DatabaseRoleCatalog,
    *,
    login_role: str,
    runtime_roles: Iterable[str],
    read_only: bool = False,
) -> RuntimePrincipalSpec:
    """Validate a LOGIN-to-NOLOGIN role binding against the schema contracts."""

    _require_role_name(login_role)
    roles = tuple(sorted(set(runtime_roles)))
    if not roles:
        raise ValueError("a runtime principal must activate at least one role")
    for role in roles:
        _require_role_name(role)
    unknown = sorted(set(roles) - set(catalog.runtime_roles))
    if unknown:
        raise ValueError(
            f"runtime principal references non-runtime {catalog.database} roles: "
            + ", ".join(unknown)
        )
    if login_role in catalog.all_roles:
        raise ValueError("runtime LOGIN name must be distinct from every contracted group role")
    verifier_role = cutover_verifier_role(catalog.database)
    if verifier_role in roles and (roles != (verifier_role,) or not read_only):
        raise ValueError(
            "cutover verifier LOGIN must be read-only and bind exactly its verifier role"
        )
    if read_only and roles != (verifier_role,):
        raise ValueError("read-only principal is reserved for the cutover verifier role")
    return RuntimePrincipalSpec(
        login_role=login_role,
        runtime_roles=roles,
        database=catalog.database,
        read_only=read_only,
    )


def provision_runtime_principal(
    connection: Any,
    *,
    spec: RuntimePrincipalSpec,
    password: str,
    session_defaults: Iterable[tuple[str, int]] | None = None,
) -> None:
    """Create one NOINHERIT LOGIN and grant only its contracted runtime groups.

    The caller owns the outer transaction and must connect as a role with
    CREATEROLE. Password material is converted to a SCRAM verifier by libpq
    before it is embedded in an ALTER ROLE statement.

    ``session_defaults`` carries the ``(guc, milliseconds)`` query-governance
    pairs the LOGIN must hold as role defaults; it comes from
    ``PoolSettings.role_session_defaults()`` when the caller does not override
    it. These belong on the role rather than in the connection pool because a
    pooled ``SET`` does not survive PgBouncer transaction pooling: it binds to
    whichever backend is assigned at that moment, leaking to unrelated clients
    of the same pool while leaving every other backend unbounded.
    """

    if len(password) < 16:
        raise ValueError("runtime principal password must contain at least 16 characters")
    role_rows = connection.execute(
        """
        SELECT rolname, rolcanlogin, rolsuper, rolcreatedb, rolcreaterole,
               rolinherit, rolreplication, rolbypassrls
        FROM pg_roles
        WHERE rolname = ANY(%s)
        """,
        (list(spec.runtime_roles),),
    ).fetchall()
    by_name = {_value(row, "rolname", 0): row for row in role_rows}
    missing = sorted(set(spec.runtime_roles) - set(by_name))
    if missing:
        raise RuntimePrincipalError(
            "contracted runtime group roles do not exist: " + ", ".join(missing)
        )
    for role in spec.runtime_roles:
        row = by_name[role]
        flags = tuple(
            bool(_value(row, name, position))
            for name, position in (
                ("rolcanlogin", 1),
                ("rolsuper", 2),
                ("rolcreatedb", 3),
                ("rolcreaterole", 4),
                ("rolinherit", 5),
                ("rolreplication", 6),
                ("rolbypassrls", 7),
            )
        )
        if flags != (False, False, False, False, False, False, False):
            raise RuntimePrincipalError(
                f"contracted runtime group {role} is not a NOLOGIN NOINHERIT non-privileged role"
            )

    login_row = connection.execute(
        """
        SELECT rolcanlogin, rolsuper, rolcreatedb, rolcreaterole, rolinherit,
               rolreplication, rolbypassrls
        FROM pg_roles WHERE rolname = %s
        """,
        (spec.login_role,),
    ).fetchone()
    if login_row is not None:
        login_flags = tuple(bool(_value(login_row, name, position)) for name, position in (
            ("rolcanlogin", 0),
            ("rolsuper", 1),
            ("rolcreatedb", 2),
            ("rolcreaterole", 3),
            ("rolinherit", 4),
            ("rolreplication", 5),
            ("rolbypassrls", 6),
        ))
        if login_flags != (True, False, False, False, False, False, False):
            raise RuntimePrincipalError(
                "existing runtime LOGIN has forbidden PostgreSQL attributes"
            )

    memberships = {
        str(_value(row, "rolname", 0))
        for row in connection.execute(
            """
            SELECT granted.rolname
            FROM pg_auth_members AS membership
            JOIN pg_roles AS member ON member.oid = membership.member
            JOIN pg_roles AS granted ON granted.oid = membership.roleid
            WHERE member.rolname = %s
            """,
            (spec.login_role,),
        ).fetchall()
    }
    unexpected = sorted(memberships - set(spec.runtime_roles))
    if unexpected:
        raise RuntimePrincipalError(
            "runtime LOGIN has unexpected role memberships: " + ", ".join(unexpected)
        )

    if login_row is None:
        connection.execute(
            sql.SQL(
                "CREATE ROLE {} LOGIN NOINHERIT NOSUPERUSER NOCREATEDB "
                "NOCREATEROLE NOREPLICATION NOBYPASSRLS"
            ).format(sql.Identifier(spec.login_role))
        )

    verifier = connection.pgconn.encrypt_password(
        password.encode("utf-8"),
        spec.login_role.encode("ascii"),
        b"scram-sha-256",
    ).decode("ascii")
    connection.execute(
        sql.SQL("ALTER ROLE {} PASSWORD {}").format(
            sql.Identifier(spec.login_role),
            sql.Literal(verifier),
        )
    )
    if spec.read_only:
        connection.execute(
            sql.SQL(
                "ALTER ROLE {} SET default_transaction_read_only TO on"
            ).format(sql.Identifier(spec.login_role))
        )
    else:
        connection.execute(
            sql.SQL(
                "ALTER ROLE {} RESET default_transaction_read_only"
            ).format(sql.Identifier(spec.login_role))
        )
    if session_defaults is None:
        # Imported lazily so the provisioning path does not require psycopg_pool
        # to be installed just to build a role specification.
        from .pool import PoolSettings

        session_defaults = PoolSettings.from_env().role_session_defaults()
    for guc, milliseconds in session_defaults:
        _require_guc_name(guc)
        connection.execute(
            sql.SQL("ALTER ROLE {} SET {} TO {}").format(
                sql.Identifier(spec.login_role),
                sql.Identifier(guc),
                sql.Literal(f"{int(milliseconds)}ms"),
            )
        )
    connection.execute(
        sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
            sql.Identifier(str(connection.info.dbname)),
            sql.Identifier(spec.login_role),
        )
    )
    connection.execute(
        sql.SQL("REVOKE CREATE, TEMPORARY ON DATABASE {} FROM {}").format(
            sql.Identifier(str(connection.info.dbname)),
            sql.Identifier(spec.login_role),
        )
    )
    for role in spec.runtime_roles:
        connection.execute(
            sql.SQL("GRANT {} TO {}").format(
                sql.Identifier(role),
                sql.Identifier(spec.login_role),
            )
        )


def _require_role_name(value: str) -> str:
    if len(value) > 63 or _ROLE_NAME.fullmatch(value) is None:
        raise ValueError(f"invalid PostgreSQL role name: {value!r}")
    return value


def _require_guc_name(value: str) -> str:
    if len(value) > 63 or _GUC_NAME.fullmatch(value) is None:
        raise ValueError(f"invalid PostgreSQL setting name: {value!r}")
    return value


def _value(row: Any, name: str, position: int) -> Any:
    if hasattr(row, "keys"):
        return row[name]
    return row[position]


__all__ = [
    "RuntimePrincipalError",
    "RuntimePrincipalSpec",
    "build_runtime_principal_spec",
    "provision_runtime_principal",
]

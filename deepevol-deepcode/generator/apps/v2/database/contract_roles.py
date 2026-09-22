"""Read the role/schema surface from canonical machine contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_SQL_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*\Z")
_CUTOVER_VERIFIER_ROLES = {
    "PRODUCT": "product_cutover_verifier",
    "AGENT": "agent_cutover_verifier",
}


def cutover_verifier_role(database: str) -> str:
    try:
        return _CUTOVER_VERIFIER_ROLES[database.upper()]
    except KeyError as exc:
        raise ValueError(f"unsupported V2 database: {database}") from exc


@dataclass(frozen=True, slots=True)
class SchemaRoleContract:
    schema_name: str
    migration_role: str
    owner_role: str
    writer_roles: tuple[str, ...]
    reader_roles: tuple[str, ...]

    @property
    def runtime_roles(self) -> tuple[str, ...]:
        return tuple(sorted({*self.writer_roles, *self.reader_roles}))


@dataclass(frozen=True, slots=True)
class DatabaseRoleCatalog:
    database: str
    contracts: tuple[SchemaRoleContract, ...]
    snapshot_sha256: str

    @property
    def schemas(self) -> tuple[str, ...]:
        return tuple(contract.schema_name for contract in self.contracts)

    @property
    def migration_roles(self) -> tuple[str, ...]:
        return tuple(sorted({contract.migration_role for contract in self.contracts}))

    @property
    def owner_roles(self) -> tuple[str, ...]:
        return tuple(sorted({contract.owner_role for contract in self.contracts}))

    @property
    def runtime_roles(self) -> tuple[str, ...]:
        return tuple(sorted({role for contract in self.contracts for role in contract.runtime_roles}))

    @property
    def all_roles(self) -> tuple[str, ...]:
        return tuple(sorted({*self.migration_roles, *self.owner_roles, *self.runtime_roles}))


def load_database_role_catalog(contract_root: Path, database: str) -> DatabaseRoleCatalog:
    normalized_database = database.upper()
    directory = contract_root / normalized_database.lower()
    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        raise ValueError(f"no {normalized_database} schema contracts found in {directory}")

    records: list[dict[str, Any]] = []
    contracts: list[SchemaRoleContract] = []
    seen_schemas: set[str] = set()
    for path in paths:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("database") != normalized_database:
            raise ValueError(f"{path} does not declare database={normalized_database}")
        if raw.get("deployment_scope") != "ACTIVE":
            continue
        if not any(
            table.get("deployment_scope") == "ACTIVE"
            for table in raw.get("tables", ())
        ):
            continue
        roles = raw.get("roles")
        schema_name = raw.get("schema_name")
        if not isinstance(schema_name, str) or schema_name in seen_schemas:
            raise ValueError(f"{path} has a missing or duplicate schema_name")
        if not isinstance(roles, dict):
            raise ValueError(f"{path} has no roles mapping")
        migration = _required_identifier(roles, "migration", path)
        owner = _required_identifier(roles, "owner", path)
        writers = _required_identifier_list(roles, "writers", path)
        readers = _required_identifier_list(roles, "readers", path)
        seen_schemas.add(schema_name)
        record = {
            "schema_name": schema_name,
            "migration_role": migration,
            "owner_role": owner,
            "writer_roles": writers,
            "reader_roles": readers,
        }
        records.append(record)
        contracts.append(
            SchemaRoleContract(
                schema_name=schema_name,
                migration_role=migration,
                owner_role=owner,
                writer_roles=writers,
                reader_roles=readers,
            )
        )

    canonical = json.dumps(records, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return DatabaseRoleCatalog(
        database=normalized_database,
        contracts=tuple(contracts),
        snapshot_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def _required_identifier(mapping: dict[str, Any], key: str, path: Path) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or _SQL_IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{path} roles.{key} must be a SQL identifier")
    return value


def _required_identifier_list(mapping: dict[str, Any], key: str, path: Path) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} roles.{key} must be a non-empty list")
    identifiers = tuple(_required_identifier({key: item}, key, path) for item in value)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{path} roles.{key} contains duplicates")
    return identifiers

"""Frozen infrastructure identities for the two V2 PostgreSQL roots."""

from __future__ import annotations

from dataclasses import dataclass

PRODUCT_CONTRACT_SHA256 = "ebe38d017499da12f500c79b7defabee68d6e7853431edab1c38e17e28fe9acf"
AGENT_CONTRACT_SHA256 = "c569cfa0c20d55084ddccef90e23fbfcd00a93be3003f4247ec7716ddf4354b9"


@dataclass(frozen=True, slots=True)
class DatabaseSpec:
    domain: str
    migration_url_env: str
    version_table: str
    control_schema: str
    # This value is generated from contracts/data by
    # freeze_postgres_baselines.py.  Runtime services use it to reject a
    # database whose release gate still advertises an older contract.
    expected_contract_sha256: str


PRODUCT_DATABASE = DatabaseSpec(
    domain="product",
    migration_url_env="DEEPEVOL_V2_PRODUCT_MIGRATION_DATABASE_URL",
    version_table="product_schema_version",
    control_schema="v2_control",
    expected_contract_sha256=PRODUCT_CONTRACT_SHA256,
)

AGENT_DATABASE = DatabaseSpec(
    domain="agent",
    migration_url_env="DEEPEVOL_V2_AGENT_MIGRATION_DATABASE_URL",
    version_table="agent_schema_version",
    control_schema="v2_control",
    expected_contract_sha256=AGENT_CONTRACT_SHA256,
)

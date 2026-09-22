"""Infrastructure-only metadata shared by the independent V2 roots."""

from __future__ import annotations

import sqlalchemy as sa

from .specs import DatabaseSpec


def build_infrastructure_metadata(spec: DatabaseSpec) -> sa.MetaData:
    metadata = sa.MetaData()
    sa.Table(
        "schema_release_gate",
        metadata,
        sa.Column("domain", sa.String(16), primary_key=True),
        sa.Column(
            "contract_state",
            sa.String(32),
            nullable=False,
            server_default=sa.text("'CONTRACTS_PENDING'"),
        ),
        sa.Column(
            "business_schema_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("contract_sha256", sa.CHAR(64), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "domain IN ('product', 'agent')",
            name="ck_schema_release_gate_domain",
        ),
        sa.CheckConstraint(
            "contract_state IN ('CONTRACTS_PENDING', 'CONTRACTS_FROZEN', 'BUSINESS_SCHEMA_READY')",
            name="ck_schema_release_gate_state",
        ),
        sa.CheckConstraint(
            "(business_schema_enabled = false AND activated_at IS NULL) OR "
            "(business_schema_enabled = true AND contract_state = 'BUSINESS_SCHEMA_READY' "
            "AND contract_sha256 IS NOT NULL AND activated_at IS NOT NULL)",
            name="ck_schema_release_gate_activation",
        ),
        sa.CheckConstraint(
            "contract_sha256 IS NULL OR contract_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_schema_release_gate_contract_sha256",
        ),
        schema=spec.control_schema,
    )
    return metadata

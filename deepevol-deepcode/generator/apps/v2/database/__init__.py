"""Shared V2 PostgreSQL migration and release-gate primitives."""

from .release_gate import ContractGateClosed, require_business_schema
from .runtime_identity import RuntimeIdentity, RuntimeIdentityError, activate_runtime_role
from .runtime_principals import (
    RuntimePrincipalError,
    RuntimePrincipalSpec,
    build_runtime_principal_spec,
    provision_runtime_principal,
)
from .specs import AGENT_DATABASE, PRODUCT_DATABASE, DatabaseSpec
from .pool import (
    LeaseInfo,
    PoolBudget,
    PoolRegistry,
    PoolSettings,
    verify_session_governance,
)

__all__ = [
    "AGENT_DATABASE",
    "PRODUCT_DATABASE",
    "ContractGateClosed",
    "DatabaseSpec",
    "LeaseInfo",
    "PoolBudget",
    "PoolRegistry",
    "PoolSettings",
    "RuntimeIdentity",
    "RuntimeIdentityError",
    "RuntimePrincipalError",
    "RuntimePrincipalSpec",
    "activate_runtime_role",
    "build_runtime_principal_spec",
    "provision_runtime_principal",
    "require_business_schema",
    "verify_session_governance",
]

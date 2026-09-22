"""Reusable Billing component fixtures.

The factories in this module are deliberately ordinary Python helpers. They
keep pytest setup small without hiding the HTTP contract or introducing a
custom test DSL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.common.v2_ids import uuid7
from apps.v2.composition.billing import BillingCompositionContext
from apps.v2.http.billing_public import (
    BillingPublicDependencies,
    build_billing_public_router,
)
from apps.v2.http.product_public import AuthenticatedContext
from apps.v2.product.funding_models import PurchaseResult, RedemptionResult
from apps.v2.product.funding_service import RedemptionCodeKeyRing
from apps.v2.product.models import MemberCreditAggregate, OwnerBillingAggregate

from .evidence import TestEvidence
from .faults import FaultInjector
from .identities import make_test_identity


class FakeBillingFunding:
    """Deterministic Funding Port for public HTTP component tests."""

    def __init__(self, *, payer_tid) -> None:
        self.commands: list[object] = []
        self.error: Exception | None = None
        self.faults = FaultInjector()
        self.redemption = RedemptionResult(
            redemption_id=uuid7(),
            payer_tid=payer_tid,
            credited_fen=1_000,
            wallet_balance_fen=2_000,
            replayed=False,
        )
        self.purchase = PurchaseResult(
            order_id=uuid7(),
            payer_tid=payer_tid,
            amount_fen=2_900,
            wallet_balance_fen=7_100,
            grant_schedule_ids=(uuid7(),),
            immediate_release_ids=(uuid7(),),
            replayed=False,
        )

    def redeem_code(self, command: Any) -> RedemptionResult:
        self.commands.append(command)
        self.faults.raise_if_injected("redeem")
        if self.error is not None:
            raise self.error
        return self.redemption

    def purchase_product(self, command: Any) -> PurchaseResult:
        self.commands.append(command)
        self.faults.raise_if_injected("purchase")
        if self.error is not None:
            raise self.error
        return self.purchase


class FakeBillingSummaries:
    """Deterministic BillingSummary Port with captured request arguments."""

    def __init__(self, result: OwnerBillingAggregate) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def read(self, **kwargs: object) -> OwnerBillingAggregate:
        self.calls.append(kwargs)
        return self.result


@dataclass(frozen=True, slots=True)
class BillingHttpEnvironment:
    client: TestClient
    auth: AuthenticatedContext
    funding: FakeBillingFunding
    summaries: FakeBillingSummaries
    evidence: list[TestEvidence] = field(default_factory=list)

    def record_evidence(self, **kwargs: Any) -> TestEvidence:
        """Capture one artifact-safe Billing operation record for the test."""

        item = TestEvidence(module="billing", **kwargs)
        self.evidence.append(item)
        return item


def make_billing_http_environment() -> BillingHttpEnvironment:
    """Create a public Billing router with explicit deterministic fakes."""

    identity = make_test_identity(context_version=7)
    auth = identity.authenticated_context()
    funding = FakeBillingFunding(payer_tid=identity.payer_tid)
    summaries = FakeBillingSummaries(
        OwnerBillingAggregate(
            payer_tid=identity.payer_tid,
            period_start=date(2026, 7, 1),
            period_end=date(2026, 7, 31),
            posted_balance_credits=-250,
            reserved_credits=50,
            available_credits=-300,
            wallet_balance_fen=7_100,
            members=(MemberCreditAggregate(identity.uid, 1_250, 50, 3),),
        )
    )
    app = FastAPI()
    app.include_router(
        build_billing_public_router(
            BillingPublicDependencies(
                authenticate=lambda _request: auth,
                funding=funding,
                summaries=summaries,
            )
        )
    )
    return BillingHttpEnvironment(
        client=TestClient(app),
        auth=auth,
        funding=funding,
        summaries=summaries,
    )


def make_billing_composition_context(
    *,
    command_connect: Any | None = None,
    query_connect: Any | None = None,
    authenticate: Any | None = None,
    provider_authenticator: Any | None = None,
) -> BillingCompositionContext:
    """Return a no-I/O composition context for builder/component tests."""

    def unopened_connection(role: str) -> Any:
        raise AssertionError(f"Billing composition opened {role} connection during construction")

    return BillingCompositionContext(
        command_dsn="postgresql://billing-command",
        command_connect=command_connect or (lambda: unopened_connection("command")),
        query_dsn="postgresql://billing-query",
        query_connect=query_connect or (lambda: unopened_connection("query")),
        plans_path=Path(__file__).resolve().parents[2] / "config/billing/plans.yaml",
        credit_packs_path=Path(__file__).resolve().parents[2] / "config/billing/token_packs.yaml",
        storage_packs_path=Path(__file__).resolve().parents[2] / "config/billing/storage_packs.yaml",
        resource_policy_path=Path(__file__).resolve().parents[2] / "config/resources.yaml",
        redemption_keys=RedemptionCodeKeyRing(
            {"test": b"r" * 32},
            active_key_id="test",
            request_fingerprint_key=b"f" * 32,
        ),
        authenticate=authenticate or (lambda _request: None),  # type: ignore[return-value]
        provider_authenticator=provider_authenticator
        or (lambda _request, _service=None: "LLM_GATEWAY"),
    )


__all__ = [
    "BillingHttpEnvironment",
    "FakeBillingFunding",
    "FakeBillingSummaries",
    "make_billing_composition_context",
    "make_billing_http_environment",
]

"""Reusable, explicit identity contexts for V2 component tests."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from apps.common.v2_ids import uuid7


@dataclass(frozen=True, slots=True)
class TestIdentity:
    """A test actor and its selected payer context.

    The factory keeps the identity relationship explicit while leaving the
    application identity service untouched.  UUIDs remain fresh per factory
    call so two tests cannot accidentally share mutable state.
    """

    __test__ = False

    uid: UUID
    personal_tid: UUID
    payer_tid: UUID
    context_version: int = 1

    @classmethod
    def new(cls, *, context_version: int = 1) -> TestIdentity:
        if context_version < 1:
            raise ValueError("test identity context_version must be positive")
        return cls(
            uid=uuid7(),
            personal_tid=uuid7(),
            payer_tid=uuid7(),
            context_version=context_version,
        )

    def authenticated_context(self):
        """Build the public HTTP auth context without coupling module import time."""

        from apps.v2.http.product_public import AuthenticatedContext

        return AuthenticatedContext(
            uid=self.uid,
            personal_tid=self.personal_tid,
            selected_payer_tid=self.payer_tid,
            context_version=self.context_version,
        )


def make_test_identity(*, context_version: int = 1) -> TestIdentity:
    """Return a fresh actor/payer relationship for a component test."""

    return TestIdentity.new(context_version=context_version)


__all__ = ["TestIdentity", "make_test_identity"]

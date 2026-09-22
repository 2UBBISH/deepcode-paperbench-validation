"""Canonical V2 UUIDv7 and public typed-ID codec.

PostgreSQL stores the returned ``UUID`` values. Type prefixes exist only at API,
event, log, and operator-facing boundaries.
"""

from __future__ import annotations

import re
import secrets
import time
import uuid
from hashlib import sha256
from collections.abc import Callable, Mapping
from types import MappingProxyType


PUBLIC_ID_PREFIXES: Mapping[str, str] = MappingProxyType(
    {
        "account_id": "acct",
        "asset_id": "ast",
        "asset_version_id": "av",
        "attempt_id": "att",
        "auth_challenge_id": "ach",
        "authorization_id": "authz",
        "billing_account_id": "bacc",
        "charge_id": "chg",
        "billing_subject_id": "bsub",
        "capability_id": "cap",
        "checkpoint_id": "chk",
        "cutover_id": "cut",
        "channel_connection_id": "chconn",
        "context_snapshot_id": "ctx",
        "credit_lot_id": "clot",
        "data_subject_manifest_id": "dsm",
        "data_subject_request_id": "dsr",
        "deletion_job_id": "djob",
        "delivery_attempt_id": "dlvatt",
        "delivery_operation_id": "dlvop",
        "device_session_id": "dses",
        "document_id": "doc",
        "document_version_id": "dver",
        "chapter_id": "chap",
        "chapter_version_id": "chver",
        "estimate_id": "est",
        "event_id": "evt",
        "external_principal_id": "xpr",
        "folder_id": "fld",
        "feedback_message_id": "fmsg",
        "feedback_thread_id": "fth",
        "inbound_message_id": "inmsg",
        "input_manifest_id": "iman",
        "invocation_id": "inv",
        "invitation_id": "tinv",
        "ledger_id": "ldg",
        "legacy_exposure_id": "lexp",
        "logical_step_id": "lstep",
        "membership_id": "mbr",
        "migration_batch_id": "mgb",
        "mid": "mid",
        "operation_id": "op",
        "remote_compute_resource_id": "rcres",
        "remote_compute_binding_id": "rcbind",
        "order_id": "ord",
        "owner_resolution_case_id": "orcase",
        "payment_attempt_id": "payatt",
        "pricing_snapshot_id": "psnap",
        "provider_attempt_id": "pat",
        "project_id": "ppt",
        "source_id": "psrc",
        "outline_slide_id": "pslide",
        "generation_request_id": "pgen",
        "presentation_version_id": "pver",
        "publication_id": "pub",
        "reservation_id": "rsv",
        "redemption_id": "red",
        "reconciliation_case_id": "rcase",
        "rid": "rid",
        "runtime_blob_id": "rblob",
        "settlement_id": "stl",
        "grant_schedule_id": "gsch",
        "sid": "sid",
        "skill_id": "skl",
        "skill_version_id": "skver",
        "storage_object_id": "sobj",
        "tid": "tid",
        "trace_stream_id": "tstr",
        "trace_object_id": "tobj",
        "uid": "uid",
        "upload_session_id": "upl",
        "usage_fact_id": "usgf",
        "usage_id": "usg",
        "worker_task_id": "wtask",
    }
)

_PREFIXES = frozenset(PUBLIC_ID_PREFIXES.values())
_TYPED_ID_RE = re.compile(
    r"^(?P<prefix>[a-z][a-z0-9]*)_"
    r"(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})$"
)
_MAX_UNIX_MS = (1 << 48) - 1


class InvalidTypedId(ValueError):
    """Raised when an external V2 ID is not canonical or has the wrong type."""


def uuid7(
    *,
    unix_ms: int | None = None,
    random_bits: Callable[[int], int] = secrets.randbits,
) -> uuid.UUID:
    """Create an RFC 9562 UUIDv7 using 48 timestamp and 74 random bits."""

    timestamp = time.time_ns() // 1_000_000 if unix_ms is None else unix_ms
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or not 0 <= timestamp <= _MAX_UNIX_MS:
        raise ValueError("unix_ms must be an integer in the unsigned 48-bit range")

    entropy = random_bits(74)
    if isinstance(entropy, bool) or not isinstance(entropy, int) or not 0 <= entropy < (1 << 74):
        raise ValueError("random_bits(74) must return an unsigned 74-bit integer")

    rand_a = entropy >> 62
    rand_b = entropy & ((1 << 62) - 1)
    raw = timestamp << 80
    raw |= 0x7 << 76
    raw |= rand_a << 64
    raw |= 0b10 << 62
    raw |= rand_b
    return uuid.UUID(int=raw)


def derive_uuid7(seed: uuid.UUID, label: bytes) -> uuid.UUID:
    """Derive a stable UUIDv7-shaped identifier from a UUID and bounded label."""

    if not isinstance(seed, uuid.UUID):
        raise TypeError("UUIDv7 derivation seed must be a UUID")
    if not isinstance(label, bytes) or not 1 <= len(label) <= 4096:
        raise ValueError("UUIDv7 derivation label must contain 1 to 4096 bytes")
    digest = sha256(seed.bytes + b"\0" + label).digest()
    raw = bytearray(seed.bytes[:6] + digest[:10])
    raw[6] = (raw[6] & 0x0F) | 0x70
    raw[8] = (raw[8] & 0x3F) | 0x80
    return uuid.UUID(bytes=bytes(raw))


def format_typed_id(prefix: str, value: uuid.UUID) -> str:
    """Serialize a UUIDv7 with one registered public type prefix."""

    _validate_prefix(prefix)
    _validate_uuid7(value)
    return f"{prefix}_{value}"


def new_typed_id(prefix: str) -> str:
    """Create a new canonical external ID for a registered type prefix."""

    return format_typed_id(prefix, uuid7())


def parse_typed_id(value: str, *, expected_prefix: str | None = None) -> uuid.UUID:
    """Parse a canonical typed ID and return the UUID suitable for a PG uuid column."""

    if not isinstance(value, str):
        raise InvalidTypedId("typed ID must be a string")
    match = _TYPED_ID_RE.fullmatch(value)
    if match is None:
        raise InvalidTypedId("typed ID is not in canonical <prefix>_<uuidv7> form")

    prefix = match.group("prefix")
    _validate_prefix(prefix)
    if expected_prefix is not None:
        _validate_prefix(expected_prefix)
        if prefix != expected_prefix:
            raise InvalidTypedId(f"expected prefix {expected_prefix!r}, got {prefix!r}")

    parsed = uuid.UUID(match.group("uuid"))
    _validate_uuid7(parsed)
    if str(parsed) != match.group("uuid"):
        raise InvalidTypedId("UUID component is not canonical lowercase text")
    return parsed


def prefix_for_field(field_name: str) -> str:
    """Return the registered public prefix for a canonical field name."""

    try:
        return PUBLIC_ID_PREFIXES[field_name]
    except KeyError as exc:
        raise InvalidTypedId(f"unknown public ID field {field_name!r}") from exc


def _validate_prefix(prefix: str) -> None:
    if prefix not in _PREFIXES:
        raise InvalidTypedId(f"unknown public ID prefix {prefix!r}")


def _validate_uuid7(value: uuid.UUID) -> None:
    if not isinstance(value, uuid.UUID):
        raise InvalidTypedId("ID value must be a uuid.UUID")
    if value.version != 7 or value.variant != uuid.RFC_4122:
        raise InvalidTypedId("ID value must be an RFC 9562 UUIDv7")


__all__ = [
    "PUBLIC_ID_PREFIXES",
    "InvalidTypedId",
    "derive_uuid7",
    "format_typed_id",
    "new_typed_id",
    "parse_typed_id",
    "prefix_for_field",
    "uuid7",
]

"""The one DeepEvol V2 HTTP error envelope, for writers and readers alike.

Three shapes used to reach the wire: ``{"detail": {"code", "message"}}`` from
most public handlers, ``{"error_code": "..."}`` from the maintenance gate and
the internal service routes, and ``{"success", "data", "error"}`` from a handful
of routers that already matched the contract.  A caller therefore had to guess,
and the browser client carried a translation layer to paper over it.

The contract shape is::

    {"success": false, "data": null, "error": {"code": "...", "message": "..."}}

``error`` may carry extra machine-readable hints (``retry_after_seconds`` is the
one the browser renders as a wait).  Hints are never prose and never disclose
whether an account exists.

This module holds no FastAPI import on purpose: the service-to-service transport
clients read the same envelope and must not pull a web framework in to do it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def error_document(code: str, message: str | None = None, **hints: Any) -> dict[str, Any]:
    """Build the contract error body for one failure."""

    error: dict[str, Any] = {"code": code, "message": message if message else code}
    error.update({key: value for key, value in hints.items() if value is not None})
    return {"success": False, "data": None, "error": error}


def success_document(data: Any) -> dict[str, Any]:
    return {"success": True, "data": data, "error": None}


def error_code_of(document: Any) -> str | None:
    """Read the error code out of a response body, or None when absent.

    The legacy branches exist only for the window in which a caller on the new
    build can still reach a peer on the old one: the V2 services deploy as one
    Compose project but do not restart simultaneously.  Delete both branches
    once a release has fully rolled out; nothing writes those shapes any more.
    """

    if not isinstance(document, Mapping):
        return None
    error = document.get("error")
    if isinstance(error, Mapping) and isinstance(error.get("code"), str):
        return str(error["code"])
    legacy_error_code = document.get("error_code")  # transitional
    if isinstance(legacy_error_code, str) and legacy_error_code:
        return legacy_error_code
    legacy_detail = document.get("detail")  # transitional
    if isinstance(legacy_detail, Mapping) and isinstance(legacy_detail.get("code"), str):
        return str(legacy_detail["code"])
    return None


def error_hint_of(document: Any, name: str) -> Any:
    """Read one machine-readable hint out of an error envelope."""

    if not isinstance(document, Mapping):
        return None
    error = document.get("error")
    if isinstance(error, Mapping):
        return error.get(name)
    return None


__all__ = [
    "error_code_of",
    "error_document",
    "error_hint_of",
    "success_document",
]

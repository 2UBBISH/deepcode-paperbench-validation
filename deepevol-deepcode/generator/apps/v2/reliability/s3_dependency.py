"""S3 admission without retries; streaming permits live until body.close()."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Any

from .circuit import CircuitPolicy, CircuitRegistry, Permit


S3_CIRCUITS = CircuitRegistry(CircuitPolicy.from_env())
_OPERATIONS = {
    "head_bucket": "probe",
    "head_object": "metadata",
    "get_object": "download",
    "put_object": "upload",
    "delete_object": "delete",
    "list_objects_v2": "listing",
}


def _outcome(error: BaseException) -> str:
    response = getattr(error, "response", None)
    if isinstance(response, Mapping):
        code = str(response.get("Error", {}).get("Code", ""))
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"NoSuchKey", "NotFound", "PreconditionFailed", "ConditionalRequestConflict", "404", "409", "412"}:
            return "success"  # A healthy server answered an expected application outcome.
        if code in {"SlowDown", "ServiceUnavailable", "InternalError", "RequestTimeout", "AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"}:
            return "failure"
        if isinstance(status, int) and (status in {401, 403, 408, 429} or status >= 500):
            return "failure"
        return "neutral"
    if isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return "failure"
    # boto3 is an optional runtime dependency, and is only loaded for an S3 call.
    try:
        from botocore.exceptions import ConnectionError as BotoConnectionError
        from botocore.exceptions import HTTPClientError, IncompleteReadError
    except ImportError:
        return "neutral"
    return "failure" if isinstance(error, (BotoConnectionError, HTTPClientError, IncompleteReadError)) else "neutral"


class _Body:
    """The narrow read/close interface consumed by Asset storage."""

    def __init__(self, body: Any, permit: Permit) -> None:
        self._body = body
        self._permit = permit
        self._eof = False
        self._closed = False

    def read(self, amount: int | None = None) -> bytes:
        if self._closed:
            raise ValueError("S3 response body is closed")
        try:
            chunk = self._body.read(amount)
            if amount != 0 and (not chunk or amount is None):
                self._eof = True
            return chunk
        except BaseException as exc:
            try:
                self.close(outcome=_outcome(exc))
            except Exception:
                pass  # Preserve the original transport error and effect certainty.
            raise

    def close(self, *, outcome: str | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._body.close()
        finally:
            self._permit.finish(outcome or ("success" if self._eof else "neutral"))


class GuardedS3Client:
    """An explicit allowlist of the S3 operations used by Asset storage.

    Keys contain configured endpoint/account and bucket/operation group, never
    an object key or user id. No failed PUT/DELETE is automatically repeated.
    """

    def __init__(self, client: Any, *, endpoint: str, account: str, circuits: CircuitRegistry = S3_CIRCUITS) -> None:
        self._client = client
        self._scope = hashlib.sha256(f"{endpoint}\0{account}".encode()).hexdigest()
        self.circuits = circuits

    def __getattr__(self, operation: str) -> Callable[..., Mapping[str, Any]]:
        if operation not in _OPERATIONS:
            raise AttributeError(operation)

        def call(**kwargs: Any) -> Mapping[str, Any]:
            key = hashlib.sha256(f"{self._scope}\0{kwargs.get('Bucket', '')}\0{_OPERATIONS[operation]}".encode()).hexdigest()
            permit = self.circuits.acquire(key)
            handed_off = False
            try:
                result = getattr(self._client, operation)(**kwargs)
                if operation == "get_object" and callable(getattr(result.get("Body"), "read", None)):
                    result = {**result, "Body": _Body(result["Body"], permit)}
                    handed_off = True
                else:
                    permit.finish("success")
                return result
            except BaseException as exc:
                permit.finish(_outcome(exc))
                raise
            finally:
                if not handed_off:
                    permit.finish()

        return call

    def close(self) -> None:
        self._client.close()

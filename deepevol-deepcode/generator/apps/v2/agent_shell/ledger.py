"""What one run consumed through its shell.

Model calls are already billed by the Gateway; the ledger mirrors their usage
so the run's report can show it next to the compute it drove.  Compute is
metered per placement generation (a lease upgrade starts a new bucket) because
that is the granularity the machine is billed at.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from .protocol import Placement


@dataclass
class _ComputeBucket:
    kind: str
    label: str
    commands: int = 0
    failed_commands: int = 0
    exec_seconds: float = 0.0
    jobs_spawned: int = 0
    bytes_uploaded: int = 0
    bytes_downloaded: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "commands": self.commands,
            "failed_commands": self.failed_commands,
            "exec_seconds": round(self.exec_seconds, 3),
            "jobs_spawned": self.jobs_spawned,
            "bytes_uploaded": self.bytes_uploaded,
            "bytes_downloaded": self.bytes_downloaded,
        }


@dataclass
class _EgressBucket:
    requests: int = 0
    failed_requests: int = 0
    credentialed_requests: int = 0
    seconds: float = 0.0
    bytes_in: int = 0
    bytes_out: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "failed_requests": self.failed_requests,
            "credentialed_requests": self.credentialed_requests,
            "seconds": round(self.seconds, 3),
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
        }


@dataclass
class ShellUsageLedger:
    llm_calls: int = 0
    llm_failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    compute: dict[int, _ComputeBucket] = field(default_factory=dict)
    egress: dict[str, _EgressBucket] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # --------------------------------------------------------------- egress
    def record_egress(self, provider: str, *, failed: bool = False, seconds: float = 0.0,
                      bytes_in: int = 0, bytes_out: int = 0, credentialed: bool = False) -> None:
        with self._lock:
            bucket = self.egress.get(provider)
            if bucket is None:
                bucket = _EgressBucket()
                self.egress[provider] = bucket
            bucket.requests += 1
            if failed:
                bucket.failed_requests += 1
            if credentialed:
                bucket.credentialed_requests += 1
            bucket.seconds += max(0.0, float(seconds))
            bucket.bytes_in += int(bytes_in)
            bucket.bytes_out += int(bytes_out)

    # ---------------------------------------------------------------- model
    def record_llm(self, usage: dict[str, int] | None, *, failed: bool = False) -> None:
        with self._lock:
            self.llm_calls += 1
            if failed:
                self.llm_failed_calls += 1
                return
            usage = usage or {}
            self.input_tokens += int(usage.get("input_tokens", 0) or 0)
            self.output_tokens += int(usage.get("output_tokens", 0) or 0)
            self.total_tokens += int(usage.get("total_tokens", 0) or 0)

    # -------------------------------------------------------------- compute
    def _bucket(self, placement: Placement) -> _ComputeBucket:
        bucket = self.compute.get(placement.generation)
        if bucket is None:
            bucket = _ComputeBucket(kind=str(placement.kind), label=placement.label)
            self.compute[placement.generation] = bucket
        return bucket

    def record_exec(self, placement: Placement, *, seconds: float, failed: bool) -> None:
        with self._lock:
            bucket = self._bucket(placement)
            bucket.commands += 1
            bucket.exec_seconds += max(0.0, float(seconds))
            if failed:
                bucket.failed_commands += 1

    def record_spawn(self, placement: Placement) -> None:
        with self._lock:
            self._bucket(placement).jobs_spawned += 1

    def record_transfer(self, placement: Placement, *, uploaded: int = 0, downloaded: int = 0) -> None:
        with self._lock:
            bucket = self._bucket(placement)
            bucket.bytes_uploaded += int(uploaded)
            bucket.bytes_downloaded += int(downloaded)

    # ------------------------------------------------------------- reports
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "llm": {
                    "calls": self.llm_calls,
                    "failed_calls": self.llm_failed_calls,
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                    "total_tokens": self.total_tokens,
                },
                "compute": {
                    str(generation): bucket.to_dict()
                    for generation, bucket in sorted(self.compute.items())
                },
                "egress": {name: bucket.to_dict() for name, bucket in sorted(self.egress.items())},
            }

    def llm_usage(self) -> dict[str, int]:
        """The shape the experiment report already consumes for ``llm_usage``."""

        with self._lock:
            return {
                "calls": self.llm_calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
            }


__all__ = ["ShellUsageLedger"]

"""``LeasedExecutionPort``: rent on first use, release on close, hard cap on top.

The first ``run`` acquires the machine (``RunLease.acquire``), opens the
Docker tunnel (``RemoteDaemon``), checks that Docker exists on the host
(installing it with the minimal bootstrap when the image is not
Docker-ready) and builds the executor. ``close`` tears the tunnel down and
deletes the instance; the driver calls it from ``finally`` and the CLI has
a ``release`` command as the backstop. Every job's timeout is clipped to
the time left before the hard cap, so a job running into the cap fails and
the next ``run`` releases the machine instead of starting.
"""

from __future__ import annotations

import asyncio
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from apps.v2.agent.paper2code.execution.aliyun_lease import MINIMAL_BOOTSTRAP, RunLease, job_limits
from apps.v2.agent.paper2code.execution.job_executor import ImagePolicy, RemoteDockerExecutor
from apps.v2.agent.paper2code.execution.port import Job, JobResult
from apps.v2.agent.paper2code.execution.remote_daemon import RemoteDaemon

REMOTE_ROOT = "/root/paper2code"


class SshOnlyDaemon(RemoteDaemon):
    """The vendored daemon without its docker.sock tunnel: key, ssh, tar sync only.

    The tunnel is never opened (``start``) and never re-opened on a sync
    failure (``ensure``) — the vendored ``_sync`` calls ``ensure()`` between
    attempts, which on the sapg rehearsal turned an SSH hiccup into
    ``CANARY_REMOTE_TUNNEL_FAILED`` on a tunnel that was never in use.
    """

    def start(self) -> None:
        return None

    def ensure(self) -> None:
        return None

    def stop(self) -> None:
        return None


class SshDockerHost:
    """The rented machine as the executor's ``DockerHost``.

    Docker commands run *on the machine* over SSH (``RemoteDaemon.run``), with
    the Dockerfile or script on stdin when needed; directories move with the
    daemon's tar-over-ssh sync. The vendored docker.sock tunnel is not used:
    a ``docker pull`` through it hung for five minutes on a machine that
    pulls the same image in seven seconds when asked directly (C8, 2026-09-17).
    """

    def __init__(self, daemon: Any, *, machine: str) -> None:
        self.daemon = daemon
        self.machine = machine

    def host_path(self, local: Path) -> str:
        return str(self.daemon.host_path(local))

    def up(self, local: Path, *, owner: str | None = None, excludes: Any = ()) -> str:
        return str(self.daemon.up(local, owner=owner, excludes=tuple(excludes)))

    def down(self, local: Path) -> None:
        self.daemon.down(local)

    def remove(self, local: Path) -> None:
        self.daemon.remove(local)

    def ensure(self) -> None:
        return None

    def docker(self, args: list[str], *, timeout: float | None = None, input_bytes: bytes | None = None) -> Any:
        command = shlex.join(["docker", *args])
        return self.daemon.run(command, stdin=input_bytes, timeout=timeout or 600)


class LeasedExecutionPort:
    def __init__(
        self,
        *,
        run_id: str,
        run_dir: Path,
        jobs_dir: Path,
        instance_type: str,
        hard_cap_seconds: float,
        events: Callable[..., Any] | None = None,
        lease: RunLease | None = None,
        daemon_factory: Callable[..., Any] = SshOnlyDaemon,
        executor_factory: Callable[..., Any] = RemoteDockerExecutor,
        bootstrap: bool = True,
    ) -> None:
        self.run_id = run_id
        self.run_dir = Path(run_dir)
        self.jobs_dir = Path(jobs_dir)
        self.instance_type = instance_type
        self.hard_cap_seconds = float(hard_cap_seconds)
        self.events = events
        self.lease = lease or RunLease(self.run_dir, events=events)
        self._daemon_factory = daemon_factory
        self._executor_factory = executor_factory
        self._bootstrap = bootstrap
        self._daemon: Any | None = None
        self._executor: Any | None = None
        self._touched_machine = False
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def started(self) -> bool:
        return self._executor is not None

    @property
    def machine(self) -> str:
        record = self.lease.record
        return record.instance_id if record else "unleased"

    def configure(self, *, instance_type: str | None = None, hard_cap_seconds: float | None = None) -> None:
        """Change what the first job will rent; refused once a machine exists (compute review point).

        A *released* or failed lease is history, not a machine (sapg-2, 2026-09-18: a compute rerun after
        the machine had been released was refused on the record alone); only an active lease refuses.
        """
        if self.started or self.lease.active:
            raise RuntimeError("the machine is already rented; the compute decision must come before the first job")
        if instance_type:
            self.instance_type = instance_type
        if hard_cap_seconds is not None:
            self.hard_cap_seconds = float(hard_cap_seconds)

    # -- ExecutionPort -------------------------------------------------------------------

    async def run(self, job: Job) -> JobResult:
        if self._closed:
            return JobResult(exit_code=None, stdout="", stderr="", duration_s=0.0, machine=self.machine,
                             error="execution port is closed")
        async with self._lock:
            if self.lease.cap_reached():
                await self._teardown("hard cap")
                return JobResult(exit_code=None, stdout="", stderr="", duration_s=0.0, machine=self.machine,
                                 error="run hard cap reached; machine released")
            if self._executor is None:
                await self._start()
        assert self._executor is not None
        result = await self._executor.run(job)
        if self.lease.cap_reached():
            async with self._lock:
                await self._teardown("hard cap")
            if result.error is None:
                result.error = "run hard cap reached after this job; machine released"
        return result

    async def close(self) -> None:
        async with self._lock:
            await self._teardown("run finished")
            self._closed = True

    # -- internals ---------------------------------------------------------------------

    async def _start(self) -> None:
        record = self.lease.record
        if record is None or record.state != "running":
            record = await self.lease.acquire(run_id=self.run_id, instance_type=self.instance_type,
                                              hard_cap_seconds=self.hard_cap_seconds)
        daemon = self._daemon_factory(run_dir=self.run_dir, host=record.host, port=record.port, username=record.username,
                                      remote_root=f"{REMOTE_ROOT}/{self.run_id}", local_root=self.run_dir)
        if self._bootstrap:
            await asyncio.to_thread(self._ensure_docker, daemon)
        self._daemon = daemon
        self._touched_machine = True
        host = SshDockerHost(daemon, machine=f"aliyun:{record.instance_id}@{record.host}")
        limits = job_limits(self.instance_type)
        self._executor = self._executor_factory(
            host=host, run_id=self.run_id, jobs_dir=self.jobs_dir, events=self.events,
            policy=ImagePolicy(cpus=limits["cpus"], memory_mib=int(limits["memory_mib"])),
            deadline=self.lease.seconds_to_cap,
        )
        self._emit("port.started", instance_id=record.instance_id, host=record.host, instance_type=self.instance_type)

    def _ensure_docker(self, daemon: Any) -> None:
        probe = daemon.run("docker version --format '{{.Server.Version}}'", timeout=120)
        if probe.returncode == 0 and probe.stdout.strip():
            self.lease.mark_bootstrap(False)
            return
        logger.warning("Docker is not present on {}; running the minimal bootstrap", daemon.host)
        self.lease.mark_bootstrap(True)
        self._emit("port.bootstrap", host=daemon.host)
        completed = daemon.run("bash -s", stdin=MINIMAL_BOOTSTRAP.encode(), timeout=1500)
        if completed.returncode != 0:
            raise RuntimeError(
                "minimal bootstrap (Docker install) failed: " + completed.stderr.decode(errors="replace")[-800:]
            )

    async def _teardown(self, reason: str) -> None:
        if self._daemon is not None:
            try:
                await asyncio.to_thread(self._daemon.stop)
            except Exception as exc:
                logger.warning("daemon stop failed: {}", exc)
            self._daemon = None
        self._executor = None
        # only a machine this port started is this port's to release: since PLAN-3 item 4 the
        # experiment agent's flow may hold the run's machine across a review point (lease.json
        # "running" while the process is gone), and closing the driver must not take it away
        if self._touched_machine and self.lease.record is not None and self.lease.record.state in {"running", "provisioning"}:
            try:
                await self.lease.release(reason)
            except Exception as exc:
                logger.error("machine release failed ({}); run `paper2code_canary.py release`", exc)
                self._emit("lease.release_failed", error=str(exc), level="error")
                raise

    def _emit(self, kind: str, **fields: Any) -> None:
        if self.events is not None:
            try:
                self.events(kind, **fields)
            except Exception as exc:
                logger.debug("event sink failed: {}", exc)


__all__ = ["REMOTE_ROOT", "LeasedExecutionPort"]

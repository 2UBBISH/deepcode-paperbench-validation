"""External access to the environment under test.

The Adjudicator, the Asset Gate and the Falsifier all need to run commands inside
the container the agent configured, without ever handing that container control
over how it is judged. Product calls use the remote implementation; local bridge
classes remain explicit development and unit-test adapters.

`docker exec` is driven through the CLI rather than docker-py on purpose: the CLI
honours the docker *context*, and on this host the correct daemon is the rootless
one selected by `~/.docker/config.json`. docker-py reads `DOCKER_HOST` instead and
would silently attach to the shared system daemon that carries other people's
production containers.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ContainerBackend(Protocol):
    """Fixed command surface used by RSA's setup and grading code.

    Implementations may dispatch through a local Docker CLI (tests/development)
    or through ``remote_relay`` to a Docker daemon on another host. Callers must
    not inspect either transport: every system command goes through ``run``.
    """

    kind: str
    workdir: str

    def run(self, cmd: str, timeout: int = 300, workdir: str | None = None,
            env: dict[str, str] | None = None) -> "ExecResult": ...

    def alive(self) -> bool: ...


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    output: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    @property
    def timed_out(self) -> bool:
        return self.exit_code == 124


class BridgeError(RuntimeError):
    """The bridge could not be used at all (no daemon, container gone)."""


class Bridge:
    """Common surface. Subclasses differ only in how a command is dispatched."""

    kind = "abstract"
    workdir = "/workspace/repo"

    def run(self, cmd: str, timeout: int = 300, workdir: str | None = None,
            env: dict[str, str] | None = None) -> ExecResult:
        raise NotImplementedError

    # -- conveniences shared by every caller -----------------------------

    def exists(self, path: str) -> bool:
        return self.run(f"test -e {shlex.quote(path)}", timeout=60).ok

    def read(self, path: str, limit: int = 2_000_000) -> str:
        r = self.run(f"head -c {limit} {shlex.quote(path)}", timeout=120)
        if not r.ok:
            raise BridgeError(f"cannot read {path}: {r.output[:200]}")
        return r.output

    def sha256(self, path: str) -> str:
        r = self.run(f"sha256sum {shlex.quote(path)}", timeout=600)
        return r.output.strip().split()[0] if r.ok and r.output.strip() else ""

    def prelude_env(self, *, command: str, timeout: int, repeats: int,
                    env: dict[str, str] | None = None,
                    workdir: str | None = None,
                    snapshot: list[str] | None = None) -> dict[str, str]:
        """The RSA_* variables a generated criteria file reads. See rsa/prelude."""
        return {
            "RSA_EXEC": self.kind,
            "RSA_CONTAINER": getattr(self, "container_id", "") or "",
            "RSA_ROOT": str(getattr(self, "root", "") or ""),
            "RSA_WORKDIR": workdir or self.workdir,
            "RSA_ENV_JSON": json.dumps(env or {}, sort_keys=True),
            "RSA_COMMAND": command,
            "RSA_TIMEOUT": str(timeout),
            "RSA_REPEATS": str(repeats),
            "RSA_SNAPSHOT": json.dumps(sorted(snapshot or [])),
        }


class DockerBridge(Bridge):
    kind = "docker"

    def __init__(self, container_id: str, workdir: str = "/workspace/repo"):
        self.container_id = container_id
        self.workdir = workdir

    def run(self, cmd: str, timeout: int = 300, workdir: str | None = None,
            env: dict[str, str] | None = None) -> ExecResult:
        flags: list[str] = []
        for k, v in (env or {}).items():
            flags += ["-e", f"{k}={v}"]
        argv = ["docker", "exec", *flags, "-w", workdir or self.workdir,
                self.container_id, "bash", "-lc", cmd]
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return ExecResult(124, "TIMEOUT")
        except OSError as e:
            raise BridgeError(f"docker exec unavailable: {type(e).__name__}: {e}") from e
        return ExecResult(p.returncode, (p.stdout or "") + (p.stderr or ""))

    def alive(self) -> bool:
        try:
            p = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", self.container_id],
                capture_output=True, text=True, timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return p.returncode == 0 and p.stdout.strip() == "true"


class LocalBridge(Bridge):
    """Runs on the host filesystem. For unit tests and for containerless use."""

    kind = "local"

    def __init__(self, root: str | Path, workdir: str | None = None):
        self.root = Path(root).resolve()
        self.workdir = str(workdir or self.root)

    def run(self, cmd: str, timeout: int = 300, workdir: str | None = None,
            env: dict[str, str] | None = None) -> ExecResult:
        import os
        try:
            p = subprocess.run(
                ["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout,
                cwd=workdir or self.workdir, env={**os.environ, **(env or {})},
            )
        except subprocess.TimeoutExpired:
            return ExecResult(124, "TIMEOUT")
        except OSError as e:
            raise BridgeError(f"local exec failed: {type(e).__name__}: {e}") from e
        return ExecResult(p.returncode, (p.stdout or "") + (p.stderr or ""))

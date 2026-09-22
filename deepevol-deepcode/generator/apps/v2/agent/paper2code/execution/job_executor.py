"""Run one job in a one-shot container on a Docker host, syncing the workspace around it.

The host is whatever answers the local ``docker`` CLI: with
:class:`LocalDockerHost` it is this machine (tests only); with the vendored
``RemoteDaemon`` it is the rented machine's daemon reached through an SSH
tunnel (``DOCKER_HOST`` points at the tunnel socket), and the workspace is
copied up before the run and back after it (tar over ssh, no deletions).

Image policy (PLAN.md C5): ``python:3.11-slim`` with ``pytest`` preinstalled
is the base; the first job of a run whose workspace carries a
``requirements.txt`` installs it (Aliyun pip mirror, CPU torch index when torch is asked for,
``PAPER2CODE_PIP_TIMEOUT_S`` budget of 20 minutes, ``PAPER2CODE_PIP_EXTRA_ARGS`` appended) and
commits the result as ``paper2code-run-<id>:<sha>``; a changed
requirements file rebuilds; a failed install is logged to ``pip.log`` and
the base image is used instead. Jobs run with ``--network none``.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import os
import shlex
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from apps.v2.agent.paper2code.execution.port import Job, JobResult, tail

SYNC_EXCLUDES: tuple[str, ...] = (".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", ".mypy_cache")
BASE_IMAGE = "python:3.11-slim"
BASE_TAG = "paper2code-base:py311"
PIP_INDEX_URL = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"  # mirrors.aliyun.com: 0.25 MB/s over HTTP/1.1 from cn-hongkong (2026-09-18)
PIP_TRUSTED_HOST = "mirrors.tuna.tsinghua.edu.cn"
PIP_TIMEOUT_S = float(os.environ.get("PAPER2CODE_PIP_TIMEOUT_S", str(20 * 60)))
# CPU wheels for the torch family. The default PyPI ``torch`` drags the whole CUDA 13 stack
# (nccl 216 MB, triton 248 MB, cublas 423 MB, …) onto a machine without a GPU — the sapg
# rerun of 2026-09-17 timed out after 20 minutes still downloading it. With the CPU index added,
# ``2.x.y+cpu`` outranks ``2.x.y`` (a local version label sorts higher) and the CUDA
# metapackages are never resolved; ``PAPER2CODE_TORCH_CPU_INDEX=0`` turns this off for a GPU tier.
TORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
TORCH_FAMILY = ("torch", "torchvision", "torchaudio")
PULL_TIMEOUT_S = 5 * 60
DOCKER_GRACE_S = 30
# Where to fetch the base image when Docker Hub is slow or unreachable from the machine
# (mainland and, at times, Hong Kong instances); tried in order after the plain name.
DEFAULT_BASE_MIRRORS = ("docker.1ms.run/library/{image}", "docker.m.daocloud.io/library/{image}", "dockerproxy.net/library/{image}")


class ExecutorError(RuntimeError):
    """Docker itself is unusable (no daemon, base image cannot be built)."""


class DockerHost(Protocol):
    """Where the daemon runs and how local directories reach it.

    A host may also provide ``docker(args, *, timeout, input_bytes)`` to run the
    Docker CLI where the daemon is (the leased machine does, over SSH); without
    it the local CLI is used with whatever ``DOCKER_HOST`` says.
    """

    @property
    def machine(self) -> str: ...

    def host_path(self, local: Path) -> str: ...

    def up(self, local: Path, *, owner: str | None = None, excludes: Iterable[str] = ()) -> Any: ...

    def down(self, local: Path) -> None: ...

    def remove(self, local: Path) -> None: ...

    def ensure(self) -> None: ...


class LocalDockerHost:
    """This machine's daemon: paths map to themselves, nothing is copied. Tests only."""

    machine = "local"

    def host_path(self, local: Path) -> str:
        return str(Path(local).resolve())

    def up(self, local: Path, *, owner: str | None = None, excludes: Iterable[str] = ()) -> str:
        return self.host_path(local)

    def down(self, local: Path) -> None:
        return None

    def remove(self, local: Path) -> None:
        return None

    def ensure(self) -> None:
        return None


def default_base_image() -> str:
    """``PAPER2CODE_BASE_IMAGE`` overrides the base (a mirror-qualified name, a locally present tag)."""
    return os.environ.get("PAPER2CODE_BASE_IMAGE", "").strip() or BASE_IMAGE


def base_tag_for(base_image: str) -> str:
    """One built base per source image, so an override never reuses another base's layers."""
    return f"{BASE_TAG}-{hashlib.sha256(base_image.encode()).hexdigest()[:8]}"


@dataclass(slots=True)
class ImagePolicy:
    base_image: str = ""
    base_tag: str = ""
    pip_index_url: str = PIP_INDEX_URL
    pip_trusted_host: str = PIP_TRUSTED_HOST
    pip_timeout_s: float = PIP_TIMEOUT_S
    pip_extra_args: str = os.environ.get("PAPER2CODE_PIP_EXTRA_ARGS", "")
    torch_cpu_index: bool = os.environ.get("PAPER2CODE_TORCH_CPU_INDEX", "1") != "0"
    cpus: float | None = None
    memory_mib: int | None = None
    build_network: str = "bridge"

    def __post_init__(self) -> None:
        if not self.base_image:
            self.base_image = default_base_image()
        if not self.base_tag:
            self.base_tag = base_tag_for(self.base_image)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def workspace_manifest(root: Path) -> dict[str, str]:
    """``relative path -> sha256`` for every regular file under ``root`` (excluding sync excludes)."""
    out: dict[str, str] = {}
    root = Path(root)
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in SYNC_EXCLUDES for part in rel.parts):
            continue
        try:
            out[rel.as_posix()] = _sha256_bytes(path.read_bytes())
        except OSError:
            continue
    return out


def mentions_torch(requirements_text: str) -> bool:
    """Whether a requirements file asks for the torch family (comments and blank lines ignored)."""
    for raw in requirements_text.splitlines():
        line = raw.split("#", 1)[0].strip().lower()
        if not line:
            continue
        name = line.split("[")[0]
        for sep in ("==", ">=", "<=", "~=", "!=", ">", "<", " ", ";", "@"):
            name = name.split(sep)[0]
        if name in TORCH_FAMILY:
            return True
    return False


def pip_install_command(requirements_text: str, policy: "ImagePolicy") -> str:
    """The ``pip install`` line run inside the build container (pure; unit-tested)."""
    parts = ["pip", "install", "--no-cache-dir"]
    if policy.torch_cpu_index and mentions_torch(requirements_text):
        parts += ["--extra-index-url", TORCH_CPU_INDEX_URL]
    if policy.pip_extra_args.strip():
        parts += shlex.split(policy.pip_extra_args)
    parts += ["-r", "/build/requirements.txt"]
    return " ".join(shlex.quote(p) for p in parts)


def find_requirements(workspace: Path) -> Path | None:
    """``requirements.txt`` at the workspace root or in exactly one child project directory."""
    direct = workspace / "requirements.txt"
    if direct.is_file():
        return direct
    candidates = sorted(p for p in workspace.glob("*/requirements.txt") if p.is_file())
    return candidates[0] if len(candidates) == 1 else None


class RemoteDockerExecutor:
    """The :class:`ExecutionPort` over a :class:`DockerHost`."""

    def __init__(
        self,
        *,
        host: DockerHost,
        run_id: str,
        jobs_dir: Path,
        events: Callable[..., Any] | None = None,
        policy: ImagePolicy | None = None,
        docker_bin: str = "docker",
        deadline: Callable[[], float | None] | None = None,
    ) -> None:
        self.host = host
        self.run_id = run_id
        self.jobs_dir = Path(jobs_dir)
        self.events = events
        self.policy = policy or ImagePolicy()
        self.docker = docker_bin
        self._deadline = deadline
        self._seq = itertools.count(self._next_seq())
        self._base_ready = False

    # -- public --------------------------------------------------------------------

    async def run(self, job: Job) -> JobResult:
        return await asyncio.to_thread(self._run_sync, job)

    async def close(self) -> None:
        return None

    @property
    def machine(self) -> str:
        return str(getattr(self.host, "machine", None) or "unknown")

    # -- docker helpers ------------------------------------------------------------

    def _docker(self, *args: str, timeout: float | None = None, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
        remote = getattr(self.host, "docker", None)
        if callable(remote):
            return remote(list(args), timeout=timeout, input_bytes=input_bytes)
        return subprocess.run(
            [self.docker, *args],
            input=input_bytes,
            stdin=None if input_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=dict(os.environ),
        )

    def _image_exists(self, tag: str) -> bool:
        return self._docker("image", "inspect", tag, timeout=60).returncode == 0

    POLL_INTERVAL_S = 10.0

    def _run_detached(self, name: str, run_args: list[str], *, timeout_s: float) -> tuple[int | None, bytes, bytes, bool]:
        """``docker run -d`` then poll with short calls; returns (exit_code, stdout, stderr, timed_out).

        A long ``docker run`` held open on one ssh session dies with the session (the sapg C9
        run lost a torch install that way, and the daemon's ssh retry then re-ran the same
        ``docker run`` into a name conflict). Detaching keeps the container independent of the
        transport; every later call is short and safe to retry. The container is removed here.
        """
        started = self._docker("run", "-d", "--name", name, *run_args, timeout=120)
        if started.returncode != 0:
            return None, started.stdout, started.stderr, False
        deadline = time.time() + timeout_s
        timed_out = False
        exit_code: int | None = None
        while True:
            state = self._docker("inspect", "-f", "{{.State.Status}} {{.State.ExitCode}}", name, timeout=60)
            parts = state.stdout.decode(errors="replace").split()
            if state.returncode == 0 and parts and parts[0] == "exited":
                exit_code = int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else None
                break
            if time.time() >= deadline:
                timed_out = True
                self._docker("kill", name, timeout=60)
                time.sleep(2)
                break
            time.sleep(self.POLL_INTERVAL_S)
        logs = self._docker("logs", name, timeout=300)
        self._docker("rm", "-f", name, timeout=60)
        if timed_out and exit_code is None:
            exit_code = 124
        return exit_code, logs.stdout, logs.stderr, timed_out

    def _base_candidates(self) -> list[str]:
        image = self.policy.base_image
        if "/" in image.split(":")[0]:
            return [image]  # already registry-qualified: no mirror rewriting
        mirrors = os.environ.get("PAPER2CODE_BASE_IMAGE_MIRRORS")
        patterns = [m.strip() for m in mirrors.split(",") if m.strip()] if mirrors is not None else list(DEFAULT_BASE_MIRRORS)
        return [image, *(pattern.format(image=image) for pattern in patterns)]

    def _pull_base_image(self, log_path: Path) -> None:
        """Make ``policy.base_image`` present on the daemon: local, Docker Hub, then the mirrors."""
        if self._image_exists(self.policy.base_image):
            return
        attempts: list[str] = []
        for candidate in self._base_candidates():
            self._emit("image.pull", image=candidate)
            try:
                completed = self._docker("pull", candidate, timeout=PULL_TIMEOUT_S)
                ok = completed.returncode == 0
                note = completed.stderr.decode(errors="replace")[-400:]
            except subprocess.TimeoutExpired:
                ok, note = False, f"timed out after {PULL_TIMEOUT_S}s"
            attempts.append(f"{candidate}: {'ok' if ok else note}")
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"pull {candidate}: {'ok' if ok else 'failed'} {note}\n")
            if ok:
                if candidate != self.policy.base_image:
                    self._docker("tag", candidate, self.policy.base_image, timeout=60)
                return
        raise ExecutorError("cannot pull the base image from any source: " + "; ".join(attempts))

    def ensure_base_image(self) -> str:
        if self._base_ready:
            return self.policy.base_tag
        self.host.ensure()
        if not self._image_exists(self.policy.base_tag):
            log_path = self.jobs_dir / "image-build.log"
            self.jobs_dir.mkdir(parents=True, exist_ok=True)
            self._pull_base_image(log_path)
            dockerfile = (
                f"FROM {self.policy.base_image}\n"
                f"ENV PIP_INDEX_URL={self.policy.pip_index_url} PIP_TRUSTED_HOST={self.policy.pip_trusted_host} "
                "PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1\n"
                "RUN pip install --no-cache-dir pytest\n"
            )
            self._emit("image.build", tag=self.policy.base_tag, base=self.policy.base_image)
            try:
                completed = self._docker("build", "--pull=false", "-t", self.policy.base_tag, "-", timeout=self.policy.pip_timeout_s, input_bytes=dockerfile.encode())
            except subprocess.TimeoutExpired as exc:
                with log_path.open("ab") as fh:
                    fh.write(b"\n--- build timed out ---\n" + (exc.stdout or b"") + b"\n" + (exc.stderr or b""))
                raise ExecutorError(f"base image build timed out after {self.policy.pip_timeout_s:.0f}s; see {log_path}") from exc
            with log_path.open("ab") as fh:
                fh.write(b"\n--- build stdout ---\n" + completed.stdout + b"\n--- build stderr ---\n" + completed.stderr)
            if completed.returncode != 0:
                raise ExecutorError(
                    f"cannot build base image {self.policy.base_tag}: {completed.stderr.decode(errors='replace')[-800:]}"
                )
        self._base_ready = True
        return self.policy.base_tag

    def ensure_run_image(self, workspace: Path, job_dir: Path) -> str:
        """The run image for ``workspace``'s requirements (or the base image)."""
        base = self.ensure_base_image()
        requirements = find_requirements(Path(workspace))
        if requirements is None:
            return base
        sha = _sha256_bytes(requirements.read_bytes())
        state_path = self.jobs_dir / "image.json"
        state = self._read_json(state_path)
        if state.get("requirements_sha") == sha:
            if state.get("status") == "ok" and state.get("image") and self._image_exists(str(state["image"])):
                return str(state["image"])
            if state.get("status") == "failed":
                return base
        tag = f"paper2code-run-{self.run_id[:12].lower()}:{sha[:12]}"
        build_dir = job_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        (build_dir / "requirements.txt").write_bytes(requirements.read_bytes())
        remote_build = self.host.up(build_dir)
        container = f"p2c-build-{self.run_id[:12].lower()}-{int(time.time())}"
        command = pip_install_command(requirements.read_text(encoding="utf-8", errors="replace"), self.policy)
        self._emit("image.pip", tag=tag, requirements=str(requirements), sha=sha, command=command, timeout_s=self.policy.pip_timeout_s)
        log_path = job_dir / "pip.log"
        log_path.write_text(f"$ {command}\n", encoding="utf-8")
        # detached: the install survives an ssh drop; the container is committed before removal
        started = self._docker(
            "run", "-d", "--name", container, "--network", self.policy.build_network,
            "-v", f"{remote_build}:/build:ro",
            base, "bash", "-lc", command,
            timeout=120,
        )
        ok = started.returncode == 0
        reason = "" if ok else f"docker run failed: {started.stderr.decode(errors='replace')[-300:]}"
        if ok:
            deadline = time.time() + self.policy.pip_timeout_s
            exit_code: int | None = None
            while True:
                state = self._docker("inspect", "-f", "{{.State.Status}} {{.State.ExitCode}}", container, timeout=60)
                parts = state.stdout.decode(errors="replace").split()
                if state.returncode == 0 and parts and parts[0] == "exited":
                    exit_code = int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else None
                    break
                if time.time() >= deadline:
                    self._docker("kill", container, timeout=60)
                    reason = f"timed out after {self.policy.pip_timeout_s:.0f}s"
                    break
                time.sleep(self.POLL_INTERVAL_S)
            logs = self._docker("logs", container, timeout=300)
            with log_path.open("ab") as fh:
                fh.write(logs.stdout + b"\n--- stderr ---\n" + logs.stderr + (b"\n--- timed out ---\n" if reason else b""))
            ok = exit_code == 0 and not reason
            if not ok and not reason:
                reason = f"exit {exit_code}"
        if ok:
            commit = self._docker("commit", container, tag, timeout=600)
            ok = commit.returncode == 0
            reason = "" if ok else f"docker commit failed: {commit.stderr.decode(errors='replace')[-300:]}"
        self._docker("rm", "-f", container, timeout=60)
        self.host.remove(build_dir)
        if ok:
            self._write_json(state_path, {"requirements_sha": sha, "image": tag, "status": "ok", "command": command, "at": time.time()})
            self._emit("image.ready", tag=tag)
            return tag
        self._write_json(state_path, {"requirements_sha": sha, "image": base, "status": "failed", "reason": reason, "command": command, "at": time.time()})
        self._emit("image.pip_failed", tag=tag, reason=reason, log=str(log_path), level="warning")
        logger.warning("requirements install failed ({}); jobs use the base image. See {}", reason, log_path)
        return base

    # -- the job ---------------------------------------------------------------------

    def _run_sync(self, job: Job) -> JobResult:
        seq = next(self._seq)
        job_dir = self.jobs_dir / f"{seq:04d}"
        job_dir.mkdir(parents=True, exist_ok=True)
        workspace = Path(job.workspace).resolve()
        started = time.time()
        self._write_json(
            job_dir / "job.json",
            {"seq": seq, "label": job.label, "command": job.command, "workspace": str(workspace),
             "timeout_s": job.timeout_s, "has_script": job.script is not None, "started": started},
        )
        if job.script is not None:
            (job_dir / "script.py").write_text(job.script, encoding="utf-8")
        try:
            image = self.ensure_run_image(workspace, job_dir)
            before = workspace_manifest(workspace)
            remote_ws = self.host.up(workspace, excludes=SYNC_EXCLUDES)
            remote_job = self.host.up(job_dir)
        except Exception as exc:
            logger.exception("job {} could not start", seq)
            result = JobResult(exit_code=None, stdout="", stderr="", duration_s=time.time() - started,
                               machine=self.machine, job_dir=job_dir, error=f"{type(exc).__name__}: {exc}")
            self._finish(job_dir, seq, job, result)
            return result

        timeout_s = float(job.timeout_s)
        capped = False
        if self._deadline is not None:
            remaining = self._deadline()
            if remaining is not None:
                if remaining <= 0:
                    result = JobResult(exit_code=None, stdout="", stderr="", duration_s=0.0, machine=self.machine,
                                       job_dir=job_dir, error="run hard cap reached before the job started")
                    self._finish(job_dir, seq, job, result)
                    return result
                if remaining < timeout_s:
                    timeout_s, capped = remaining, True

        name = f"p2c-{self.run_id[:12].lower()}-{seq}-{int(started)}"
        argv = ["--network", "none", "-w", "/workspace",
                "-v", f"{remote_ws}:/workspace", "-v", f"{remote_job}:/job:ro",
                "-e", "JOB_SCRIPT=/job/script.py", "-e", "CI=1", "-e", "PYTHONUNBUFFERED=1", "-e", "PYTHONDONTWRITEBYTECODE=1",
                # bytecode (compileall, imports) goes outside /workspace so the sync-back never carries __pycache__
                "-e", "PYTHONPYCACHEPREFIX=/tmp/pycache"]
        if self.policy.cpus:
            argv += ["--cpus", str(self.policy.cpus)]
        if self.policy.memory_mib:
            argv += ["--memory", f"{int(self.policy.memory_mib)}m"]
        for key, value in job.extra_env.items():
            argv += ["-e", f"{key}={value}"]
        # The budget is enforced inside the container by coreutils `timeout` (TERM, then KILL after 5 s):
        # 124 = timed out, 137 = had to be killed. The outer subprocess timeout only covers Docker itself.
        argv += [image, "timeout", "-k", "5", str(int(max(timeout_s, 1))), "bash", "-lc", job.command]
        (job_dir / "docker.txt").write_text(shlex.join([self.docker, "run", "-d", "--name", name, *argv]), encoding="utf-8")

        timed_out = False
        error: str | None = None
        try:
            exit_code, stdout, stderr, killed = self._run_detached(name, argv, timeout_s=timeout_s + DOCKER_GRACE_S)
            if exit_code is None and not killed:
                error = "container could not start: " + stderr.decode("utf-8", errors="replace")[-300:].strip()
            elif killed or (exit_code in (124, 137) and (time.time() - started) >= timeout_s - 1):
                timed_out = True
                error = "run hard cap reached; job killed" if capped else None
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout, stderr, exit_code = exc.stdout or b"", exc.stderr or b"", 124
            self._docker("rm", "-f", name, timeout=60)
            error = "docker call timed out; container removed"
        except OSError as exc:
            stdout, stderr, exit_code, error = b"", b"", None, f"docker unavailable: {exc}"
        duration = time.time() - started
        (job_dir / "stdout.txt").write_bytes(stdout)
        (job_dir / "stderr.txt").write_bytes(stderr)

        synced: list[str] = []
        try:
            self.host.down(workspace)
            after = workspace_manifest(workspace)
            synced = sorted(path for path, digest in after.items() if before.get(path) != digest)
        except Exception as exc:
            error = (error + "; " if error else "") + f"sync back failed: {exc}"
        result = JobResult(
            exit_code=exit_code,
            stdout=tail(stdout.decode("utf-8", errors="replace")),
            stderr=tail(stderr.decode("utf-8", errors="replace")),
            duration_s=duration,
            machine=self.machine,
            job_dir=job_dir,
            timed_out=timed_out,
            error=error,
            synced_back=synced,
        )
        self._finish(job_dir, seq, job, result)
        return result

    # -- bookkeeping -----------------------------------------------------------------

    def _finish(self, job_dir: Path, seq: int, job: Job, result: JobResult) -> None:
        self._write_json(job_dir / "result.json", result.to_record())
        self._emit("job.run", seq=seq, label=job.label, exit_code=result.exit_code, timed_out=result.timed_out,
                   duration_s=round(result.duration_s, 2), machine=result.machine, error=result.error,
                   synced_back=len(result.synced_back))

    def _emit(self, kind: str, **fields: Any) -> None:
        if self.events is not None:
            try:
                self.events(kind, **fields)
            except Exception as exc:
                logger.debug("event sink failed: {}", exc)

    def _next_seq(self) -> int:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        existing = [int(p.name) for p in self.jobs_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        return (max(existing) + 1) if existing else 1

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


__all__ = [
    "BASE_IMAGE",
    "BASE_TAG",
    "SYNC_EXCLUDES",
    "DockerHost",
    "ExecutorError",
    "ImagePolicy",
    "LocalDockerHost",
    "RemoteDockerExecutor",
    "base_tag_for",
    "default_base_image",
    "find_requirements",
    "mentions_torch",
    "pip_install_command",
    "workspace_manifest",
]

"""Remote Docker backend for the RSA product path.

The relay is deliberately treated as a transport, not as another Docker API.
RSA sends Docker CLI commands to the remote machine through one synchronous
facade. SetupX, the criterion runner and the adjudicator therefore share one
small contract and cannot accidentally fall back to a local daemon.

``remote_relay.zip`` is optional at import time so the deterministic/unit-test
parts of RSA remain usable without SSH dependencies. A real remote run fails
with an actionable error if the relay package or its asyncssh dependency is not
available.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import inspect
import os
import posixpath
import queue
import shlex
import shutil
import sys
import tarfile
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from .bridge import BridgeError


def _relay_api() -> tuple[type, type, Callable[..., Any]]:
    """Load RemoteRuntime and SSHTarget from an installed relay or the supplied zip."""
    # DeepEvol: remote_relay is vendored as a sibling package inside
    # apps/v2/agent_engine (the V2 Agent image ships no top-level packages),
    # so the flat name is tried under that prefix first.
    try:
        from apps.v2.agent_engine.remote_relay.remote import RemoteRuntime  # type: ignore
        from apps.v2.agent_engine.remote_relay.transport.target import (  # type: ignore
            SSHTarget, parse_access_url,
        )
        return RemoteRuntime, SSHTarget, parse_access_url
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        from remote_relay.remote import RemoteRuntime  # type: ignore
        from remote_relay.transport.target import SSHTarget, parse_access_url  # type: ignore
        return RemoteRuntime, SSHTarget, parse_access_url
    except (ImportError, ModuleNotFoundError):
        pass

    candidates = [
        os.environ.get("RSA_REMOTE_RELAY_ROOT", ""),
        str(Path(__file__).resolve().parents[2] / "remote_relay.zip"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        # The archive contains a project directory and then the import package:
        # adding the archive itself makes ``remote_relay.remote_relay`` visible.
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        try:
            from remote_relay.remote_relay.remote import RemoteRuntime  # type: ignore
            from remote_relay.remote_relay.transport.target import (  # type: ignore
                SSHTarget, parse_access_url,
            )
            return RemoteRuntime, SSHTarget, parse_access_url
        except (ImportError, ModuleNotFoundError):
            continue
    raise BridgeError(
        "remote-relay is unavailable; install its dependencies or set "
        "RSA_REMOTE_RELAY_ROOT to the relay checkout"
    )


class _RuntimeThread:
    """Keep one async relay connection while exposing a blocking API to RSA."""

    def __init__(self, runtime: Any):
        self.runtime = runtime
        self._ready = threading.Event()
        self._closed = False
        self._queue: queue.Queue[Any] = queue.Queue()

        def runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._ready.set()
            while True:
                item = self._queue.get()
                if item is None:
                    break
                fn, args, kwargs, future = item
                try:
                    value = fn(*args, **kwargs)
                    if inspect.isawaitable(value):
                        value = loop.run_until_complete(value)
                    future.set_result(value)
                except BaseException as exc:
                    future.set_exception(exc)
            loop.close()

        self._thread = threading.Thread(target=runner, name="rsa-remote-relay", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise BridgeError("remote relay is closed")

        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        self._queue.put((fn, args, kwargs, future))
        try:
            return future.result()
        except Exception as exc:
            raise BridgeError(f"remote relay call failed: {type(exc).__name__}: {exc}") from exc

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.call(self.runtime.close)
        except Exception:
            pass
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=5)


@dataclass
class RemoteCommandResult:
    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    truncated: bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    @property
    def ok(self) -> bool:
        return self.success

    @property
    def timed_out(self) -> bool:
        return self.exit_code == 124

    @property
    def output(self) -> str:
        return self.stdout + self.stderr

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "truncated": self.truncated,
        }


class RemoteDockerBackend:
    """Docker-on-remote-host implementation of the RSA container contract."""

    kind = "remote-docker"

    def __init__(
        self,
        target: str | Any | None = None,
        *,
        runtime: Any | None = None,
        runtime_factory: Callable[..., Any] | None = None,
        username: str = "root",
        password: str | None = None,
        private_key: str | None = None,
        base_image: str = "setupx-base:py310-proxy",
        workdir: str = "/workspace/repo",
        docker_binary: str = "docker",
        grader_image: str = "rsa-grader:py311-v1",
    ):
        self.base_image = base_image
        self.workdir = workdir
        self.docker_binary = docker_binary
        self.grader_image = grader_image
        self.container_id = ""
        self._env_vars: dict[str, str] = {}
        self._checkpoint_repo = f"rsa_checkpoint_{uuid.uuid4().hex[:10]}"
        self._snapshots: list[str] = []
        self._orphaned: list[str] = []

        if runtime is None:
            if target is None:
                raise BridgeError("remote_target is required for the remote Docker backend")
            Runtime, Target, parse_access_url = _relay_api()
            if runtime_factory is not None:
                runtime = runtime_factory(target=target, username=username,
                                          password=password, private_key=private_key)
            elif isinstance(target, Target):
                runtime = Runtime(target)
            else:
                target_obj = parse_access_url(
                    str(target), username=username, password=password,
                    private_key_path=private_key,
                )
                runtime = Runtime(target_obj)
        self.runtime = runtime
        self._sync = _RuntimeThread(runtime)
        try:
            if hasattr(runtime, "start"):
                self._sync.call(runtime.start)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self._sync.close()

    def _relay_result(self, result: Any, command: str) -> RemoteCommandResult:
        status = getattr(result, "exit_status", getattr(result, "exit_code", 1))
        stdout = getattr(result, "stdout", "") or ""
        stderr = getattr(result, "stderr", "") or ""
        if not stdout and not stderr and hasattr(result, "output"):
            stdout = getattr(result, "output", "") or ""
        return RemoteCommandResult(
            command=command,
            exit_code=int(status),
            stdout=str(stdout),
            stderr=str(stderr),
            truncated=bool(getattr(result, "truncated", False)),
        )

    def _host(self, argv: list[str]) -> str:
        return " ".join(shlex.quote(str(x)) for x in argv)

    def _exec_host(self, command: str, timeout: int) -> RemoteCommandResult:
        try:
            result = self._sync.call(self.runtime.exec, command, timeout=float(timeout))
        except BridgeError:
            raise
        except Exception as exc:
            raise BridgeError(f"remote host command failed: {type(exc).__name__}: {exc}") from exc
        return self._relay_result(result, command)

    def _ensure_grader_image(self) -> None:
        """Build the isolated pytest/Docker-CLI image on the remote daemon once.

        The SSH host only needs a shell and Docker. RSA never installs Python or
        pytest into the host, and the target container remains untouched by the
        grader.
        """
        probe = self._exec_host(self._host([
            self.docker_binary, "image", "inspect", self.grader_image,
        ]), 120)
        if probe.success:
            return

        dockerfile = (
            "FROM docker:27.5-cli AS docker_cli\n"
            "FROM python:3.11-slim\n"
            "COPY --from=docker_cli /usr/local/bin/docker /usr/local/bin/docker\n"
            "RUN python -m pip install --no-cache-dir pytest\n"
        )
        encoded = base64.b64encode(dockerfile.encode("ascii")).decode("ascii")
        context = f"/tmp/rsa-grader-image-{uuid.uuid4().hex[:12]}"
        command = (
            f"mkdir -p {shlex.quote(context)} && "
            f"printf %s {shlex.quote(encoded)} | base64 -d > "
            f"{shlex.quote(context + '/Dockerfile')} && "
            f"{self.docker_binary} build --pull -t {shlex.quote(self.grader_image)} "
            f"{shlex.quote(context)}; status=$?; "
            f"rm -rf {shlex.quote(context)}; exit $status"
        )
        built = self._exec_host(command, 1800)
        if not built.success:
            raise BridgeError(
                f"remote Docker could not build grader image {self.grader_image}: "
                f"{built.output[-2000:]}"
            )

    def run(self, cmd: str, timeout: int = 300, workdir: str | None = None,
            env: dict[str, str] | None = None) -> RemoteCommandResult:
        if not self.container_id:
            raise BridgeError("remote Docker container is not initialized")
        args = [self.docker_binary, "exec"]
        for key, value in {**self._env_vars, **(env or {})}.items():
            args += ["--env", f"{key}={value}"]
        args += ["--workdir", workdir or self.workdir, self.container_id,
                 "bash", "-lc", cmd]
        return self._exec_host(self._host(args), timeout)

    def prelude_env(self, *, command: str, timeout: int, repeats: int,
                    env: dict[str, str] | None = None,
                    workdir: str | None = None,
                    snapshot: list[str] | None = None) -> dict[str, str]:
        # This is also the public bridge description for callers that do not use
        # the optimized run_criteria path. The criterion runs in the remote
        # grader and observes the target through the remote Docker socket.
        import json
        return {
            "RSA_EXEC": "docker",
            "RSA_CONTAINER": self.container_id,
            "RSA_ROOT": "",
            "RSA_WORKDIR": workdir or self.workdir,
            "RSA_ENV_JSON": json.dumps(env or {}, sort_keys=True),
            "RSA_COMMAND": command,
            "RSA_TIMEOUT": str(timeout),
            "RSA_REPEATS": str(repeats),
            "RSA_SNAPSHOT": json.dumps(sorted(snapshot or [])),
        }

    def alive(self) -> bool:
        if not self.container_id:
            return False
        r = self._exec_host(self._host([self.docker_binary, "inspect", "-f",
                                        "{{.State.Running}}", self.container_id]), 60)
        return r.success and r.stdout.strip() == "true"

    def create_container(self, repo_url: str, revision: str = "") -> str:
        if self.container_id:
            self.destroy()
        name = f"rsa-{uuid.uuid4().hex[:12]}"
        args = [self.docker_binary, "run", "--detach", "--name", name,
                "--workdir", self.workdir, "--network", "host",
                "--label", f"rsa.run={self._checkpoint_repo}"]
        for key, value in self._env_vars.items():
            args += ["--env", f"{key}={value}"]
        args += [self.base_image, "sleep", "infinity"]
        started = self._exec_host(self._host(args), 600)
        if not started.success:
            raise BridgeError(f"remote Docker could not start container: {started.output[-1000:]}")
        self.container_id = started.stdout.strip().splitlines()[-1].strip()
        try:
            self._exec_container(
                "mkdir -p /workspace && "
                "(command -v git >/dev/null || (apt-get update && apt-get install -y git))",
                timeout=1200, allow_failure=False,
            )
            clone = f"git clone {shlex.quote(repo_url)} {shlex.quote(self.workdir)}"
            if revision:
                clone += f" && git -C {shlex.quote(self.workdir)} checkout --detach {shlex.quote(revision)}"
            result = self._exec_container(clone, timeout=1800, allow_failure=False, workdir="/")
            if not result.success:
                raise BridgeError(f"remote repository clone failed: {result.output[-1000:]}")
            # SetupX assumes every new environment has a rollback baseline.
            self.create_checkpoint("initial_clone")
            return self.container_id
        except Exception:
            self.destroy()
            raise

    def _exec_container(self, cmd: str, *, timeout: int = 300,
                        workdir: str | None = None,
                        allow_failure: bool = True) -> RemoteCommandResult:
        result = self.run(cmd, timeout=timeout, workdir=workdir)
        if not allow_failure and not result.success:
            raise BridgeError(result.output[-2000:])
        return result

    def attach(self, container_id: str, workdir: str | None = None) -> None:
        self.container_id = container_id
        if workdir:
            self.workdir = workdir
        if not self.alive():
            raise BridgeError(f"remote Docker container is not running: {container_id}")

    def set_env(self, key: str, value: str) -> None:
        self._env_vars[str(key)] = str(value)

    def get_env(self, key: str) -> str | None:
        return self._env_vars.get(key)

    @property
    def env_vars(self) -> dict[str, str]:
        return dict(self._env_vars)

    def create_checkpoint(self, tag: str) -> str:
        image = f"{self._checkpoint_repo}:{tag}"
        result = self._exec_host(self._host([self.docker_binary, "commit",
                                             self.container_id, image]), 900)
        if not result.success:
            raise BridgeError(f"remote Docker checkpoint failed: {result.output[-1000:]}")
        self._snapshots.append(tag)
        return result.stdout.strip()

    def rollback_to_checkpoint(self, n_frames: int = 1) -> bool:
        if not self._snapshots or not self.container_id:
            return False
        n = min(max(1, int(n_frames)), len(self._snapshots))
        tags = [self._snapshots.pop() for _ in range(n)]
        target = tags[-1]
        self._orphaned.extend(tags)
        old = self.container_id
        self._exec_host(self._host([self.docker_binary, "rm", "--force", old]), 300)
        env_args = [
            arg for key, value in self._env_vars.items()
            for arg in ("--env", f"{key}={value}")
        ]
        started = self._exec_host(self._host([
            self.docker_binary, "run", "--detach", "--workdir", self.workdir,
            "--network", "host", "--label", f"rsa.run={self._checkpoint_repo}",
            *env_args,
            f"{self._checkpoint_repo}:{target}", "sleep", "infinity",
        ]), 600)
        if not started.success:
            self.container_id = ""
            return False
        self.container_id = started.stdout.strip().splitlines()[-1].strip()
        return True

    def cleanup_snapshots(self) -> None:
        for tag in [*self._snapshots, *self._orphaned]:
            try:
                self._exec_host(self._host([self.docker_binary, "rmi", "--force",
                                            f"{self._checkpoint_repo}:{tag}" ]), 300)
            except BridgeError:
                pass
        self._snapshots.clear()
        self._orphaned.clear()

    def destroy(self) -> None:
        if self.container_id:
            self._exec_host(self._host([self.docker_binary, "rm", "--force", self.container_id]), 300)
            self.container_id = ""

    def cleanup(self) -> None:
        try:
            self.cleanup_snapshots()
        finally:
            self.destroy()

    def put_file(self, local_path: Path, container_path: str) -> None:
        """Transfer a frozen criterion through SFTP, then copy it into the container."""
        if not hasattr(self.runtime, "upload"):
            raise BridgeError("remote relay does not provide file upload")
        staging = f"/tmp/rsa-upload-{uuid.uuid4().hex}/{Path(container_path).name}"
        parent = str(Path(staging).parent)
        self._exec_host(self._host(["mkdir", "-p", parent]), 60)
        self._sync.call(self.runtime.upload, str(local_path), staging)
        self._exec_container(f"mkdir -p {shlex.quote(str(Path(container_path).parent))}", workdir="/", allow_failure=False)
        copied = self._exec_host(self._host([self.docker_binary, "cp", staging,
                                              f"{self.container_id}:{container_path}" ]), 300)
        self._exec_host(self._host(["rm", "-rf", parent]), 60)
        if not copied.success:
            raise BridgeError(f"remote criterion upload failed: {copied.output[-500:]}")

    def collect_test_ids(self, path: Path, timeout: int = 120) -> list[str]:
        """Collect a frozen file in an isolated remote Docker grader."""
        staging = f"/tmp/rsa-collect-{uuid.uuid4().hex[:12]}"
        grader_name = f"rsa-collect-{uuid.uuid4().hex[:12]}"
        try:
            self._ensure_grader_image()
            self._sync.call(self.runtime.upload, str(path), f"{staging}/{path.name}")
            result = self._exec_host(self._host([
                self.docker_binary, "run", "--rm", "--name", grader_name,
                "--network", "none", "--volume", f"{staging}:/grader:ro",
                "--workdir", "/grader", self.grader_image,
                "python", "-m", "pytest", "--collect-only", "-q",
                "-p", "no:cacheprovider", path.name,
            ]), timeout)
            ids = [line.strip() for line in result.output.splitlines()
                   if "::" in line and line.strip().startswith(path.name)]
            if not ids and not result.success:
                raise BridgeError(f"pytest could not collect {path.name}: {result.output[-2000:]}")
            return list(dict.fromkeys(ids))
        finally:
            try:
                self._exec_host(self._host([self.docker_binary, "rm", "--force", grader_name]), 120)
            except BridgeError:
                pass
            try:
                self._exec_host(self._host(["rm", "-rf", staging]), 120)
            except BridgeError:
                pass

    def run_criteria(self, criteria_path: Path, *, command: str, timeout: int,
                     repeats: int, env: dict[str, str], workdir: str,
                     snapshot: list[str]) -> tuple[int, str]:
        """Run the frozen grader in a separate remote container.

        The grader gets only a read-only criterion mount and the Docker socket.
        Every observation reaches the target through Docker CLI, while no pytest
        or criterion code is installed into the SSH host or target environment.
        """
        staging = f"/tmp/rsa-grader-{uuid.uuid4().hex[:12]}"
        grader_name = f"rsa-grader-{uuid.uuid4().hex[:12]}"
        target = self.container_id
        if not target:
            raise BridgeError("cannot run criteria without a target container")
        try:
            import json
            self._ensure_grader_image()
            self._sync.call(self.runtime.upload, str(criteria_path),
                            f"{staging}/{criteria_path.name}")
            grader_env = {
                "RSA_EXEC": "docker",
                "RSA_CONTAINER": target,
                "RSA_ROOT": "",
                "RSA_WORKDIR": workdir,
                "RSA_ENV_JSON": json.dumps(env, sort_keys=True),
                "RSA_COMMAND": command,
                "RSA_TIMEOUT": str(timeout),
                "RSA_REPEATS": str(repeats),
                "RSA_SNAPSHOT": json.dumps(sorted(snapshot)),
            }
            args = [
                self.docker_binary, "run", "--rm", "--name", grader_name,
                "--network", "none", "--volume", f"{staging}:/grader:ro",
                "--volume", "/var/run/docker.sock:/var/run/docker.sock",
                "--workdir", "/grader",
            ]
            args += [
                arg for key, value in grader_env.items()
                for arg in ("--env", f"{key}={value}")
            ]
            args += [self.grader_image, "python", "-m", "pytest", "-rA",
                     "--tb=short", "-p", "no:cacheprovider", criteria_path.name]
            result = self._exec_host(self._host(args), timeout * max(1, repeats) + 900)
            return result.exit_code, result.output
        finally:
            try:
                self._exec_host(self._host([self.docker_binary, "rm", "--force", grader_name]), 120)
            except BridgeError:
                pass
            try:
                self._exec_host(self._host(["rm", "-rf", staging]), 120)
            except BridgeError:
                pass

    def sync_down(self, remote_dir: str, local_dir: Path) -> None:
        if not hasattr(self.runtime, "sync_down"):
            raise BridgeError("remote relay does not provide sync_down")
        self._sync.call(self.runtime.sync_down, remote_dir, str(local_dir))

    def checkout_to_local(self, repo_url: str, revision: str, local_dir: Path) -> tuple[str, set[str]]:
        """Use a remote container to materialize compiler input without local git."""
        cid = self.create_container(repo_url, revision)
        archive_remote = f"/tmp/rsa-source-{uuid.uuid4().hex}.tar.gz"
        host_archive = f"/tmp/rsa-source-{uuid.uuid4().hex}.tar.gz"
        try:
            resolved = self._exec_container("git rev-parse HEAD", timeout=120).stdout.strip()
            tracked_result = self._exec_container("git ls-tree -r --name-only HEAD", timeout=120)
            tracked = {line.strip() for line in tracked_result.stdout.splitlines() if line.strip()}
            self._exec_container(
                f"tar --exclude=.git -czf {shlex.quote(archive_remote)} "
                f"-C {shlex.quote(str(Path(self.workdir).parent))} "
                f"{shlex.quote(Path(self.workdir).name)}",
                timeout=600, workdir="/", allow_failure=False,
            )
            copied = self._exec_host(self._host([self.docker_binary, "cp", f"{cid}:{archive_remote}", host_archive]), 600)
            if not copied.success:
                raise BridgeError(copied.output[-1000:])
            if not hasattr(self.runtime, "download"):
                raise BridgeError("remote relay does not provide file download")
            local_dir.parent.mkdir(parents=True, exist_ok=True)
            local_archive = local_dir.parent / (local_dir.name + ".tar.gz")
            self._sync.call(self.runtime.download, host_archive, str(local_archive))
            top = Path(self.workdir).name
            with tempfile.TemporaryDirectory(prefix="rsa-source-", dir=local_dir.parent) as tmp:
                extract_root = Path(tmp) / "extract"
                extract_root.mkdir()
                with tarfile.open(local_archive, "r:gz") as archive:
                    for member in archive.getmembers():
                        self._validate_archive_member(member, top)
                    archive.extractall(extract_root)
                extracted = extract_root / top
                if not extracted.is_dir():
                    raise BridgeError("remote source archive has no expected repository root")
                if local_dir.exists():
                    shutil.rmtree(local_dir)
                shutil.move(str(extracted), str(local_dir))
            return resolved, tracked
        finally:
            self._exec_host(self._host(["rm", "-f", host_archive]), 60)
            try:
                (local_dir.parent / (local_dir.name + ".tar.gz")).unlink()
            except FileNotFoundError:
                pass
            self.cleanup()

    @staticmethod
    def _validate_archive_member(member: tarfile.TarInfo, top: str) -> None:
        """Reject traversal, escaping links, and device nodes before extraction."""
        name = PurePosixPath(member.name)
        parts = name.parts
        if name.is_absolute() or not parts or parts[0] != top or ".." in parts:
            raise BridgeError("remote source archive contains an unsafe path")
        if member.isdev() or member.isfifo():
            raise BridgeError("remote source archive contains a device node")
        if member.issym() or member.islnk():
            link = PurePosixPath(member.linkname)
            if link.is_absolute():
                raise BridgeError("remote source archive contains an escaping link")
            parent = PurePosixPath(*parts[:-1])
            resolved = PurePosixPath(posixpath.normpath(str(parent / link)))
            resolved_parts = resolved.parts
            if not resolved_parts or resolved_parts[0] != top or ".." in resolved_parts:
                raise BridgeError("remote source archive contains an escaping link")


class RemoteEnvironmentManager:
    """SetupX-compatible facade; SetupX's agent never sees Docker SDK objects."""

    def __init__(self, backend: RemoteDockerBackend | None = None):
        if backend is None:
            backend = _ACTIVE_BACKEND
        if backend is None:
            raise BridgeError("RemoteEnvironmentManager has no configured backend")
        self.backend = backend
        self._container_id = ""
        self._env_vars: dict[str, str] = backend.env_vars

    @property
    def container_id(self) -> str | None:
        return self.backend.container_id or None

    @property
    def container(self) -> Any:
        # SetupX only uses this for diagnostics; a string is safer than pretending
        # to expose a local Docker SDK Container object.
        return self.container_id

    def create_container(self, repo_url: str, revision: str | None = None) -> str:
        cid = self.backend.create_container(repo_url, revision or "")
        self._container_id = cid
        return cid

    def attach(self, container_id: str, repo_dir: str | None = None) -> None:
        self.backend.attach(container_id, repo_dir)
        self._container_id = container_id

    def exec_run(self, command: str, timeout: int | None = None,
                 work_dir: str | None = None) -> RemoteCommandResult:
        return self.backend.run(command, timeout=timeout or 300, workdir=work_dir)

    def set_env(self, key: str, value: str) -> None:
        self._env_vars[key] = value
        self.backend.set_env(key, value)

    def get_env(self, key: str) -> str | None:
        return self._env_vars.get(key)

    def get_env_snapshot(self) -> str:
        return self.exec_run("python3 --version; pwd; env | sort | head -50", timeout=30).stdout[:2000]

    def create_checkpoint(self, tag: str) -> str:
        return self.backend.create_checkpoint(tag)

    def rollback_to_checkpoint(self, n_frames: int = 1) -> bool:
        return self.backend.rollback_to_checkpoint(n_frames)

    def list_checkpoints(self) -> list[str]:
        return list(self.backend._snapshots)

    def cleanup_snapshots(self) -> None:
        self.backend.cleanup_snapshots()

    def destroy(self) -> None:
        self.backend.destroy()

    def cleanup(self) -> None:
        self.backend.cleanup()


_ACTIVE_BACKEND: RemoteDockerBackend | None = None


def bind_remote_backend(backend: RemoteDockerBackend | None) -> None:
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = backend

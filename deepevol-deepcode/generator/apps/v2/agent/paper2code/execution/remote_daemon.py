"""The rented machine's Docker daemon, reached as if it were local (ADR-0010, D step 3).

Docker locally and Docker on the rented host are the same executor with a different daemon: an SSH tunnel forwards
the host's `/var/run/docker.sock` to a Unix socket under the run directory and `DOCKER_HOST` points every Docker
client in this process at it — the executor's and the environment agent's `docker` CLI, RSA's `DockerBridge` and
SetupX's docker-py alike. What a bind mount needs on the host's side of the daemon (a job directory, a bare
repository) is copied up before the container starts and the job directory copied back after it exits; the
executor and the agent already name the daemon's path through `host_root`.

Authentication is a keypair generated per run (`<run_dir>/secrets/id_ed25519`, 0600) whose public half the lease
writes into the instance's `authorized_keys` through Cloud Assistant right after the machine is reachable; the
tunnel pins the host key on first contact (`<run_dir>/secrets/known_hosts`). Nothing here is used unless the run
was initialised with `--compute aliyun`.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
import threading
import time
from hashlib import sha256
from pathlib import Path



class DaemonError(RuntimeError):
    """The tunnel or a sync failed; the message starts with a stable code."""


# [paper2code C5] vendored from feature/reproduction-c
# Agent/DeepEvol/paper_reproduction_agent/canary/remote_daemon.py @ 02853e7a.
# Changes: ManifestViolation -> DaemonError; `up()` takes `excludes`.
ManifestViolation = DaemonError

TUNNEL_READY_SECONDS = 60
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=15",
               "-o", "ServerAliveCountMax=8", "-o", "ConnectTimeout=20", "-o", "LogLevel=ERROR")


def generate_keypair(directory):
    """`id_ed25519` + `.pub` under `directory` (0600), made once per run."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    private = directory / "id_ed25519"
    if not private.exists():
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "deepevol-canary", "-f", str(private)],
                       check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        private.chmod(0o600)
    return private, (directory / "id_ed25519.pub").read_text().strip()


class RemoteDaemon:
    """One rented host: SSH key, socket tunnel, path mapping and directory sync."""

    def __init__(self, *, run_dir, host, port, username, remote_root, local_root):
        self.run_dir, self.host, self.port, self.username = Path(run_dir), host, int(port), username
        self.remote_root, self.local_root = str(remote_root).rstrip("/"), Path(local_root).resolve()
        self.secrets = self.run_dir / "secrets"
        self.key, _ = generate_keypair(self.secrets)
        self.known_hosts = self.secrets / "known_hosts"
        # A Unix socket path is limited to ~100 bytes; the run directory may be deeper than that.
        self.socket = Path(tempfile.gettempdir()) / f"deepevol-{sha256(str(self.run_dir.resolve()).encode()).hexdigest()[:12]}.sock"
        self._tunnel = None
        self._previous_docker_host = None
        self._log = None
        self._watchdog = None
        self._stopping = False

    # -- ssh ---------------------------------------------------------------------------

    def _ssh(self, *extra):
        return ["ssh", "-i", str(self.key), "-o", f"UserKnownHostsFile={self.known_hosts}", *SSH_OPTIONS,
                "-p", str(self.port), *extra]

    def _target(self):
        return f"{self.username}@{self.host}"

    def run(self, command, *, timeout=600, stdin=None, attempts=5):
        """One command over ssh. Exit 255 is ssh itself failing (a host still booting after a resume, a dropped
        connection), not the command: those are tried again with a pause; a trial job's sync-up once failed on the
        first ssh after a resume and the stage went down with it."""
        for attempt in range(attempts):
            completed = subprocess.run([*self._ssh("-T"), self._target(), command], input=stdin,
                                       stdin=None if stdin is not None else subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            if completed.returncode != 255 or attempt + 1 == attempts:
                return completed
            time.sleep(5 * (attempt + 1))
        return completed

    def host_path(self, local):
        """The daemon's path for a directory under `local_root`."""
        return Path(self.remote_root) / Path(local).resolve().relative_to(self.local_root)

    # -- tunnel ------------------------------------------------------------------------

    @property
    def docker_host(self):
        return f"unix://{self.socket}"

    def start(self):
        """Forward the host's Docker socket to `docker.sock` and point `DOCKER_HOST` at it until `stop`; a watchdog
        re-opens the tunnel if ssh exits (a real replay lost its `docker commit` when the tunnel dropped after
        twenty minutes of pip installs)."""
        self._open()
        if self._watchdog is None:
            self._watchdog = threading.Thread(target=self._watch, name="remote-daemon-watchdog", daemon=True)
            self._watchdog.start()

    def ensure(self):
        """Re-open the tunnel if it is gone; what Docker clients call after a connection error before retrying."""
        if self._tunnel is None or self._tunnel.poll() is not None or not self._alive():
            self._open()

    def _watch(self):
        while not self._stopping:
            time.sleep(5)
            if self._stopping:
                break
            tunnel = self._tunnel
            if tunnel is not None and tunnel.poll() is not None:
                try:
                    self._open()
                except (ManifestViolation, AttributeError, OSError):
                    pass  # the next command's ensure() tries again; a dead host is reported there

    def _open(self, attempts=8, backoff=10):
        """A machine just started from parked answers SSH late (the first attempts see "closed by remote host" or a
        banner timeout); the tunnel is tried again with a pause between attempts before the host is declared
        unreachable."""
        if self._tunnel is not None and self._tunnel.poll() is None and self._alive():
            return
        last = None
        for attempt in range(attempts):
            try:
                self._open_once()
                return
            except ManifestViolation as exc:
                last = exc
                if attempt + 1 < attempts:
                    time.sleep(backoff)
        raise last

    def _open_once(self):
        self._close_tunnel()
        if self.socket.exists():
            self.socket.unlink()
        # stderr goes to a file, never a pipe nobody drains: a full pipe would block ssh and hang the tunnel.
        self._log = open(self.run_dir / "tunnel.log", "ab")  # lives as long as the tunnel
        self._tunnel = subprocess.Popen([*self._ssh("-N", "-o", "ExitOnForwardFailure=yes",
                                                     "-L", f"{self.socket}:/var/run/docker.sock"), self._target()],
                                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=self._log)
        if os.environ.get("DOCKER_HOST") != self.docker_host:
            self._previous_docker_host = os.environ.get("DOCKER_HOST")
            os.environ["DOCKER_HOST"] = self.docker_host
        deadline = time.monotonic() + TUNNEL_READY_SECONDS
        while time.monotonic() < deadline:
            tunnel = self._tunnel
            if tunnel is None:  # stop() closed it under the watchdog's feet
                raise ManifestViolation("CANARY_REMOTE_TUNNEL_CLOSED")
            if tunnel.poll() is not None:
                error = self._tail_log()
                self._restore_env()
                raise ManifestViolation(f"CANARY_REMOTE_TUNNEL_FAILED: {error}")
            if self.socket.exists() and self._alive():
                return
            time.sleep(1)
        self._close_tunnel()
        raise ManifestViolation("CANARY_REMOTE_TUNNEL_TIMEOUT")

    def _tail_log(self):
        try:
            return (self.run_dir / "tunnel.log").read_bytes()[-800:].decode(errors="replace")
        except OSError:
            return ""

    def _close_tunnel(self):
        if self._tunnel is not None:
            if self._tunnel.poll() is None:
                self._tunnel.terminate()
                try:
                    self._tunnel.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._tunnel.kill()
            self._tunnel = None
        if self._log is not None:
            self._log.close()
            self._log = None
        if self.socket.exists():
            self.socket.unlink()

    def _alive(self):
        completed = subprocess.run(["docker", "-H", self.docker_host, "version", "--format", "{{.Server.Version}}"],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30)
        return completed.returncode == 0 and bool(completed.stdout.strip())

    def _restore_env(self):
        if os.environ.get("DOCKER_HOST") == self.docker_host:
            if self._previous_docker_host is None:
                os.environ.pop("DOCKER_HOST", None)
            else:
                os.environ["DOCKER_HOST"] = self._previous_docker_host

    def stop(self):
        self._stopping = True
        self._close_tunnel()
        self._restore_env()
        self._watchdog = None
        self._stopping = False

    # -- sync --------------------------------------------------------------------------

    SYNC_ATTEMPTS = 3

    def _sync(self, command, *, stdin=None, code):
        """Run one sync command, again after a pause when it fails: both directions are idempotent (`up` removes and
        re-extracts, `down` re-tars), and a transfer cut mid-way returns a non-255 exit with an empty stderr — two
        trial jobs of the SNSE run died that way and took the stage with them."""
        for attempt in range(self.SYNC_ATTEMPTS):
            completed = self.run(command, stdin=stdin, timeout=900)
            if completed.returncode == 0:
                return completed
            if attempt + 1 < self.SYNC_ATTEMPTS:
                time.sleep(10 * (attempt + 1))
                self.ensure()
        raise ManifestViolation(f"{code}: exit {completed.returncode} {completed.stderr.decode(errors='replace')[-800:]}".rstrip())

    def up(self, local, *, owner=None, excludes=()):
        """Copy a directory under `local_root` to the same place under `remote_root`, owned by root there (a container
        running as root refuses a git repository owned by someone else) or chowned for the uid:gid a job runs as.
        `excludes` are tar patterns (e.g. ``.git``, ``__pycache__``) left out of the copy."""
        local = Path(local).resolve()
        remote = self.host_path(local)
        exclude_args = [arg for pattern in excludes for arg in ("--exclude", pattern)]
        tar = subprocess.run(["tar", "-C", str(local.parent), *exclude_args, "-cf", "-", local.name], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, check=True, timeout=600, env={**os.environ, "COPYFILE_DISABLE": "1"})
        chown = f" && chown -R {shlex.quote(owner)} {shlex.quote(str(remote))}" if owner else ""
        command = (f"rm -rf {shlex.quote(str(remote))} && mkdir -p {shlex.quote(str(remote.parent))}"
                   f" && tar --no-same-owner -C {shlex.quote(str(remote.parent))} -xf -{chown}")
        self._sync(command, stdin=tar.stdout, code="CANARY_REMOTE_SYNC_UP_FAILED")
        return remote

    def down(self, local):
        """Copy the remote copy of a directory back over the local one (a job's results)."""
        local = Path(local).resolve()
        remote = self.host_path(local)
        completed = self._sync(f"tar -C {shlex.quote(str(remote.parent))} -cf - {shlex.quote(remote.name)}",
                               code="CANARY_REMOTE_SYNC_DOWN_FAILED")
        subprocess.run(["tar", "-C", str(local.parent), "-xf", "-"], input=completed.stdout, check=True, timeout=600,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def remove(self, local):
        remote = self.host_path(Path(local).resolve())
        self.run(f"rm -rf {shlex.quote(str(remote))}", timeout=120)

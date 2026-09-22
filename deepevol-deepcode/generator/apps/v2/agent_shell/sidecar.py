"""Parent-side handle on a shell running as its own process.

:class:`SidecarShell` presents the surface the workflow and the lines already
use on :class:`AgentShell` — ``base_url`` / ``token`` (the Agent-facing
endpoint, now a separate process), ``gateway_backend`` (still the in-process
metered proxy: durable LLM dispatch stays with the worker), ``bind_model`` /
``bind_target`` / ``unbind_target`` / ``ledger`` / ``close`` — and drives the
sidecar through its ``/control/*`` face.

Model calls from the Agent reach the sidecar's ``/v1`` and are relayed to an
in-process :class:`AgentShell` face that this object keeps for exactly that
purpose (``model_relay``), so Gateway admission, the durable invocation ledger
and the metering proxy are untouched.  Placements are handed to the sidecar as
specs (``PlacementSpec``): the relay password is resolved inside the sidecar
from its secrets file, never in this process.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import subprocess
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from .protocol import Placement, ShellError, ShellErrorCode
from .server import AgentShell

logger = logging.getLogger(__name__)

SIDECAR_MODULE = "apps.v2.runtime.agent_shell_sidecar"


class SidecarLedger:
    """Read-through view of the sidecar's ledger, merged with the relay's."""

    def __init__(self, sidecar: "SidecarShell") -> None:
        self._sidecar = sidecar

    def snapshot(self) -> dict[str, Any]:
        remote = self._sidecar.control("GET", "usage")
        local = self._sidecar.model_relay.ledger.snapshot()
        # The relay books every model call the Agent made through the
        # sidecar (and in-process lines' calls); the sidecar's own llm
        # counters would double it, so llm comes from the relay only.
        return {"llm": local["llm"], "compute": remote.get("compute", {}), "egress": remote.get("egress", {})}

    def llm_usage(self) -> dict[str, int]:
        return self._sidecar.model_relay.ledger.llm_usage()


class SidecarShell:
    def __init__(
        self,
        *,
        gateway_backend: Any | None = None,
        model_name: str = "",
        secrets_path: str | os.PathLike[str] | None = None,
        python: str | None = None,
        repo_root: str | os.PathLike[str] | None = None,
        listen: str = "127.0.0.1:0",
        advertise_host: str | None = None,
        key_pools: bool = True,
        spawn_timeout: float = 60.0,
    ) -> None:
        # The in-process face the sidecar relays model calls to.  It has no
        # placement and no egress of its own; it exists for the metered
        # Gateway backend and the OpenAI-shaped facade the lines bind.
        self.model_relay = AgentShell(gateway_backend=gateway_backend, model_name=model_name)
        self._control_token = secrets.token_urlsafe(32)
        root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        argv = [python or sys.executable, "-m", SIDECAR_MODULE, "--listen", listen]
        if secrets_path:
            argv += ["--secrets-path", str(secrets_path)]
        if advertise_host:
            argv += ["--advertise-host", advertise_host]
        if not key_pools:
            argv.append("--no-key-pools")
        self._process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True, cwd=str(root), env=env)
        assert self._process.stdin is not None and self._process.stdout is not None
        self._process.stdin.write(json.dumps({"control_token": self._control_token}) + "\n")
        self._process.stdin.flush()
        line = self._read_line(spawn_timeout)
        try:
            endpoint = json.loads(line)
        except json.JSONDecodeError as exc:
            self._kill()
            raise RuntimeError(f"sidecar did not announce an endpoint: {line[:200]!r}") from exc
        if "error" in endpoint:
            self._kill()
            raise RuntimeError(f"sidecar refused to start: {endpoint['error']}")
        self._base_url = str(endpoint["base_url"])
        self._token = str(endpoint["token"])
        self._control_url = str(endpoint["control_url"])
        self._client = httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0), trust_env=False,
                                    headers={"Authorization": f"Bearer {self._control_token}"})
        self.ledger = SidecarLedger(self)
        self._model_name = model_name
        self._closed = False
        # Model calls: sidecar → this process' relay face.
        self.control("POST", "model", {"base_url": self.model_relay.base_url, "token": self.model_relay.token, "model_name": model_name})

    # ----------------------------------------------------------- lifecycle
    def _read_line(self, timeout: float) -> str:
        result: list[str] = []

        def reader() -> None:
            assert self._process.stdout is not None
            result.append(self._process.stdout.readline())

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive() or not result:
            self._kill()
            raise RuntimeError("sidecar did not start in time")
        return result[0].strip()

    def _kill(self) -> None:
        if self._process.poll() is None:
            try:
                self._process.kill()
            except Exception:  # pragma: no cover
                pass

    @property
    def pid(self) -> int:
        return self._process.pid

    def control(self, method: str, action: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        response = self._client.request(method, f"{self._control_url}/{action}", json=dict(payload or {}) if method != "GET" else None)
        if response.status_code >= 400:
            try:
                error = response.json().get("error") or {}
            except ValueError:
                error = {}
            raise ShellError(error.get("code", ShellErrorCode.TARGET_FAILED), error.get("message", response.text[:200]), status=response.status_code)
        return response.json()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            try:
                self.control("POST", "shutdown")
            except Exception:
                logger.debug("sidecar shutdown request failed", exc_info=True)
            try:
                self._process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logger.warning("sidecar pid=%s did not exit; killing", self._process.pid)
                self._kill()
        finally:
            self._client.close()
            self.model_relay.close()

    def __enter__(self) -> "SidecarShell":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------ AgentShell surface
    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def token(self) -> str:
        return self._token

    @property
    def openai_base_url(self) -> str:
        return f"{self._base_url}/v1"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def gateway_backend(self) -> Any:
        return self.model_relay.gateway_backend

    def bind_model(self, facade: Any, *, model_name: str) -> None:
        self.model_relay.bind_model(facade, model_name=model_name)
        self._model_name = model_name
        self.control("POST", "model", {"base_url": self.model_relay.base_url, "token": self.model_relay.token, "model_name": model_name})

    @property
    def placement(self) -> Placement:
        return Placement.from_dict(self.control("GET", "health")["placement"])

    def bind_target(self, target: Any, *, start: bool = True, timeout: float = 600.0) -> Placement:
        """Accepts a placement *spec* (Mapping) or an object exposing
        ``placement_spec()``; live ExecutionTarget objects cannot cross the
        process boundary."""

        spec = target if isinstance(target, Mapping) else getattr(target, "placement_spec", None)
        if callable(spec):
            spec = spec()
        if not isinstance(spec, Mapping):
            raise TypeError("SidecarShell.bind_target needs a placement spec")
        return Placement.from_dict(self.control("POST", "placement", {"spec": dict(spec), "start": start, "timeout": timeout}))

    def unbind_target(self) -> None:
        self.control("DELETE", "placement")


__all__ = ["SIDECAR_MODULE", "SidecarLedger", "SidecarShell"]

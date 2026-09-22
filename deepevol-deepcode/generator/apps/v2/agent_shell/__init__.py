"""Agent shell — the per-run middle layer every Agent talks through.

An Agent (the chat graph, the experiment stack's SetupX/rsa, the paper
pipeline) never holds a provider key, an SSH password or a Docker socket.  It
holds one loopback URL and one per-attempt bearer token, and asks the shell to

1. **talk to a model** — ``POST /v1/chat/completions`` (OpenAI wire subset) is
   answered by the run's Gateway model, so admission, pricing and the usage
   ledger apply exactly as for the chat graph, and the provider credential
   stays in the Gateway service;
2. **run a command / touch a file** — ``/exec``, ``/jobs``, ``/fs`` are routed
   to whatever *placement* the backend leased for this run: the run's local
   Docker sandbox, or a rented remote machine reached through the relay.  The
   Agent cannot tell which; the credentials for either live only in the shell.

Everything the shell does is metered (:mod:`ledger`) so a run's LLM and
compute consumption can be reported and billed from one place.

Layout: :mod:`protocol` (wire types), :mod:`targets` (execution placements),
:mod:`server` (the shell itself), :mod:`client` (the Agent-side stubs: a relay
``Runtime`` and a workspace executor over HTTP).  The package imports nothing
from the Agent engine so it can later run as its own sidecar process; the
per-workspace lookup the graph tools use lives in
``apps.v2.agent_engine.shell_registry`` next to the Docker context registry.
"""

from .client import ShellEgress, ShellEgressAsyncTransport, ShellEgressTransport, ShellRuntime, ShellWorkspaceExecutor
from .egress import Credential, KeyPoolPort, handle as credential_handle
from .ledger import ShellUsageLedger
from .metering import MeteredGatewayBackend
from .protocol import Placement, PlacementKind, ShellError
from .server import AgentShell
from .sidecar import SidecarShell
from .targets import ExecutionTarget, LocalDockerTarget, RelayTarget

__all__ = [
    "AgentShell",
    "Credential",
    "KeyPoolPort",
    "ShellEgress",
    "ShellEgressAsyncTransport",
    "ShellEgressTransport",
    "credential_handle",
    "ExecutionTarget",
    "LocalDockerTarget",
    "MeteredGatewayBackend",
    "Placement",
    "PlacementKind",
    "RelayTarget",
    "ShellError",
    "ShellRuntime",
    "ShellUsageLedger",
    "ShellWorkspaceExecutor",
    "SidecarShell",
]

"""Product-owned remote-compute catalog and Agent workspace boundary.

The package deliberately keeps connection secrets out of PostgreSQL.  Product
stores a stable ``secret_ref`` and Agent resolves that reference from its
deployment secret store only after a run binding has been proven locally.
"""

from .commands import PsycopgRemoteComputeCommandConsumer, PsycopgRemoteComputeResultConsumer
from .models import (
    RemoteComputeAction,
    RemoteComputeAdminRecord,
    RemoteComputeCommand,
    RemoteComputeCommandReceipt,
    RemoteComputeCommandState,
    RemoteComputeResource,
    RemoteComputeResultEvent,
    RemoteWorkspaceBinding,
    RemoteWorkspaceDescriptor,
)
from .postgres import (
    PsycopgRemoteComputeAdminRepository,
    PsycopgRemoteComputeRepository,
    RemoteComputeConflict,
    RemoteComputeNotFound,
)
from .providers import StrictRemoteComputeProvider
from .secrets import FileRemoteComputeSecretResolver

__all__ = [
    "FileRemoteComputeSecretResolver",
    "PsycopgRemoteComputeAdminRepository",
    "PsycopgRemoteComputeCommandConsumer",
    "PsycopgRemoteComputeRepository",
    "PsycopgRemoteComputeResultConsumer",
    "RemoteComputeAction",
    "RemoteComputeAdminRecord",
    "RemoteComputeCommand",
    "RemoteComputeCommandReceipt",
    "RemoteComputeCommandState",
    "RemoteComputeConflict",
    "RemoteComputeNotFound",
    "RemoteComputeResource",
    "RemoteComputeResultEvent",
    "RemoteWorkspaceBinding",
    "RemoteWorkspaceDescriptor",
    "StrictRemoteComputeProvider",
]

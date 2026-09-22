"""Offline doubles for the lease: an ECS client, remote_relay's runtime, the Docker daemon tunnel."""

from __future__ import annotations

import itertools
import subprocess
from types import SimpleNamespace


class FakeEcs:
    """Records every call; instances live in ``self.instances`` until deleted."""

    def __init__(self, *, region="cn-hangzhou", image_id="m-fixture", gpu_image_id="m-fixture-gpu", price=1.5, in_stock=True):
        from apps.v2.agent.paper2code.execution.aliyun_lease import is_gpu_type

        def image_for(instance_type):
            if is_gpu_type(instance_type):
                if not gpu_image_id:
                    raise RuntimeError(f"{instance_type} is a GPU type but no GPU image is configured")
                return gpu_image_id
            return image_id

        self.settings = SimpleNamespace(region_id=region, image_id=image_id, gpu_image_id=gpu_image_id, image_for=image_for)
        self.instances: dict[str, dict] = {}
        self.calls: list[tuple] = []
        self.price = price
        self.stock = in_stock
        self._ids = itertools.count(1)

    def hourly_price_cny(self, instance_type):
        self.calls.append(("price", instance_type))
        return self.price

    def in_stock(self, instance_type):
        return self.stock

    def machine_probe(self, instance_type):
        self.calls.append(("probe", instance_type))
        return {"in_stock": self.stock, "hourly_price_cny": self.price}

    def run_instance(self, *, name, instance_type, password, image_id=None):
        n = next(self._ids)
        instance_id = f"i-fake{n}"
        self.instances[instance_id] = {"name": name, "type": instance_type, "password": password, "status": "Running", "ip": f"10.0.0.{n}", "image": image_id or self.settings.image_for(instance_type)}
        self.calls.append(("create", instance_id))
        return instance_id

    def _row(self, instance_id):
        row = self.instances[instance_id]
        return {"InstanceId": instance_id, "Status": row["status"], "PublicIpAddress": {"IpAddress": [row["ip"]]}}

    def describe_instance(self, instance_id):
        return self._row(instance_id) if instance_id in self.instances else {}

    def wait_for_status(self, instance_id, statuses, *, timeout_seconds=180):
        return self._row(instance_id)

    def allocate_public_ip(self, instance_id):
        return self.instances[instance_id]["ip"]

    def run_command(self, *, instance_id, command, timeout_seconds=600, name="paper2code"):
        self.calls.append(("command", instance_id, name))
        self.instances[instance_id].setdefault("commands", []).append(command)
        return f"t-{len(self.calls)}"

    def wait_for_command(self, *, instance_id, invoke_id, timeout_seconds=180):
        return {"InvocationStatus": "Success", "ExitCode": 0, "Output": ""}

    fail_deletes = 0  # how many delete calls raise before succeeding

    def delete_instance(self, instance_id):
        self.calls.append(("delete", instance_id))
        if self.fail_deletes > 0:
            self.fail_deletes -= 1
            raise RuntimeError("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol")
        self.instances.pop(instance_id, None)

    def instance_exists(self, instance_id):
        return instance_id in self.instances


class FakeRuntime:
    """remote_relay's RemoteRuntime as the lease sees it: start / wait_ready / close."""

    ready = True

    def __init__(self, handle):
        self.handle = handle

    async def start(self):
        pass

    async def wait_ready(self, timeout):
        if not FakeRuntime.ready:
            raise ConnectionError("fixture: never ready")

    async def close(self):
        pass


class FakeDaemon:
    """``RemoteDaemon`` with the daemon on this machine: nothing tunnelled, nothing copied."""

    instances: list = []  # noqa: RUF012 - shared on purpose: tests read what the port built
    docker_present = True
    # no `machine` attribute on purpose: the real RemoteDaemon has none either; the port sets it

    def __init__(self, *, run_dir, host, port, username, remote_root, local_root):
        self.run_dir, self.host, self.port, self.username = run_dir, host, port, username
        self.remote_root, self.local_root = remote_root, local_root
        self.calls: list[tuple] = []
        FakeDaemon.instances.append(self)

    def run(self, command, *, timeout=600, stdin=None, attempts=5):
        self.calls.append(("run", command[:40]))
        if command.startswith("docker version"):
            ok = FakeDaemon.docker_present
            return subprocess.CompletedProcess([], 0 if ok else 127, b"29.0.0\n" if ok else b"", b"" if ok else b"docker: not found")
        if command.startswith("bash -s"):
            FakeDaemon.docker_present = True
            return subprocess.CompletedProcess([], 0, b"29.0.0\n", b"")
        return subprocess.CompletedProcess([], 0, b"", b"")

    def host_path(self, local):
        return str(local)

    def start(self):
        self.calls.append(("start", self.host))

    def stop(self):
        self.calls.append(("stop", self.host))

    def ensure(self):
        pass

    def up(self, local, owner=None, excludes=()):
        self.calls.append(("up", str(local)))
        return str(local)

    def down(self, local):
        self.calls.append(("down", str(local)))

    def remove(self, local):
        self.calls.append(("remove", str(local)))


class FakeExecutor:
    """An executor that records jobs and answers with a canned result."""

    instances: list = []  # noqa: RUF012

    def __init__(self, *, host, run_id, jobs_dir, events=None, policy=None, deadline=None):
        self.host, self.run_id, self.jobs_dir, self.policy, self.deadline = host, run_id, jobs_dir, policy, deadline
        self.jobs: list = []
        FakeExecutor.instances.append(self)

    async def run(self, job):
        from apps.v2.agent.paper2code.execution.port import JobResult

        self.jobs.append(job)
        return JobResult(exit_code=0, stdout="ok\n", stderr="", duration_s=0.01, machine=getattr(self.host, "machine", "unset"))

    async def close(self):
        pass

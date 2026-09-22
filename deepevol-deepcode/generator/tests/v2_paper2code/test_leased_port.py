"""C5: the leased port rents on the first job, releases on close, and the hard cap wins."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from apps.v2.agent.paper2code.config import EventLog
from apps.v2.agent.paper2code.execution.aliyun_lease import LeaseError, RunLease
from apps.v2.agent.paper2code.execution.leased_runtime import LeasedExecutionPort
from apps.v2.agent.paper2code.execution.port import Job
from tests.v2_paper2code.fake_aliyun import FakeDaemon, FakeEcs, FakeExecutor, FakeRuntime


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def _port(tmp_path: Path, clock: _Clock, ecs: FakeEcs, *, hours: float = 2.0) -> LeasedExecutionPort:
    events = EventLog(tmp_path / "events.jsonl")
    lease = RunLease(tmp_path, clock=clock, events=events, client_factory=lambda: ecs, runtime_factory=FakeRuntime, bring_up_backoff=0.0)
    return LeasedExecutionPort(run_id="run-abc", run_dir=tmp_path, jobs_dir=tmp_path / "jobs", instance_type="ecs.c7.xlarge",
                               hard_cap_seconds=hours * 3600, events=events, lease=lease,
                               daemon_factory=FakeDaemon, executor_factory=FakeExecutor)


def setup_function(_fn) -> None:
    FakeDaemon.instances.clear()
    FakeExecutor.instances.clear()
    FakeDaemon.docker_present = True
    FakeRuntime.ready = True


def test_first_run_leases_and_close_releases(tmp_path: Path) -> None:
    clock, ecs = _Clock(), FakeEcs()
    port = _port(tmp_path, clock, ecs)
    assert not port.started
    assert not (tmp_path / "lease.json").exists()

    result = asyncio.run(port.run(Job(workspace=tmp_path, command="true", timeout_s=10)))
    assert result.exit_code == 0
    assert result.machine == "aliyun:i-fake1@10.0.0.1"
    record = json.loads((tmp_path / "lease.json").read_text())
    assert record["state"] == "running"
    assert record["instance_id"] == "i-fake1"
    assert record["host"] == "10.0.0.1"
    assert ("command", "i-fake1", "paper2code-authorize-key") in ecs.calls
    assert (tmp_path / "secrets" / "remote-compute.json").exists()
    assert (tmp_path / "secrets" / "id_ed25519.pub").exists()
    daemon = FakeDaemon.instances[-1]
    assert any(call[0] == "run" and call[1].startswith("docker version") for call in daemon.calls)
    assert not any(call[0] == "start" for call in daemon.calls)  # no docker.sock tunnel
    executor = FakeExecutor.instances[-1]
    assert executor.policy.cpus == 4.0
    assert executor.policy.memory_mib == 7168
    host = executor.host
    assert host.machine == "aliyun:i-fake1@10.0.0.1"
    host.docker(["image", "inspect", "x y"], timeout=5)
    assert daemon.calls[-1] == ("run", "docker image inspect 'x y'"[:40])

    asyncio.run(port.run(Job(workspace=tmp_path, command="true", timeout_s=10)))
    assert len(ecs.instances) == 1  # second job reuses the machine
    asyncio.run(port.close())
    record = json.loads((tmp_path / "lease.json").read_text())
    assert record["state"] == "released"
    assert record["released_at"]
    assert ("delete", "i-fake1") in ecs.calls
    assert ecs.instances == {}
    assert ("stop", "10.0.0.1") in daemon.calls  # daemon.stop() is harmless without a tunnel
    kinds = [json.loads(line)["kind"] for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert "lease.running" in kinds
    assert "lease.released" in kinds
    assert "port.started" in kinds


def test_hard_cap_releases_and_fails_the_next_job(tmp_path: Path) -> None:
    clock, ecs = _Clock(), FakeEcs()
    port = _port(tmp_path, clock, ecs, hours=1.0)
    asyncio.run(port.run(Job(workspace=tmp_path, command="true", timeout_s=10)))
    assert FakeExecutor.instances[-1].deadline() == pytest.approx(3600.0)
    clock.advance(3601)
    result = asyncio.run(port.run(Job(workspace=tmp_path, command="true", timeout_s=10)))
    assert result.exit_code is None
    assert "hard cap" in (result.error or "")
    assert ecs.instances == {}
    assert json.loads((tmp_path / "lease.json").read_text())["state"] == "released"


def test_never_ready_machine_is_released_and_acquire_raises(tmp_path: Path) -> None:
    clock, ecs = _Clock(), FakeEcs()
    FakeRuntime.ready = False
    port = _port(tmp_path, clock, ecs)
    with pytest.raises(LeaseError):
        asyncio.run(port.run(Job(workspace=tmp_path, command="true", timeout_s=10)))
    assert ecs.instances == {}
    assert json.loads((tmp_path / "lease.json").read_text())["state"] == "released"


def test_missing_docker_triggers_minimal_bootstrap(tmp_path: Path) -> None:
    clock, ecs = _Clock(), FakeEcs()
    FakeDaemon.docker_present = False
    port = _port(tmp_path, clock, ecs)
    asyncio.run(port.run(Job(workspace=tmp_path, command="true", timeout_s=10)))
    daemon = FakeDaemon.instances[-1]
    assert any(call[0] == "run" and call[1].startswith("bash -s") for call in daemon.calls)
    assert json.loads((tmp_path / "lease.json").read_text())["bootstrap_required"] is True
    asyncio.run(port.close())


def test_failed_release_is_recorded_and_force_release_deletes(tmp_path: Path) -> None:
    clock, ecs = _Clock(), FakeEcs()
    port = _port(tmp_path, clock, ecs)
    asyncio.run(port.run(Job(workspace=tmp_path, command="true", timeout_s=10)))
    ecs.fail_deletes = 1
    with pytest.raises(LeaseError):
        asyncio.run(port.close())
    record = json.loads((tmp_path / "lease.json").read_text())
    assert record["state"] == "release_failed"
    assert record["released_at"] is None
    assert "i-fake1" in ecs.instances  # still billing

    lease = RunLease(tmp_path, clock=clock, client_factory=lambda: ecs, runtime_factory=FakeRuntime)
    outcome = asyncio.run(lease.force_release("backstop"))
    assert outcome == {"instance_id": "i-fake1", "existed": True, "deleted": True, "state": "released"}
    assert ecs.instances == {}
    assert json.loads((tmp_path / "lease.json").read_text())["state"] == "released"
    # a second force release finds nothing and changes nothing
    outcome = asyncio.run(lease.force_release("again"))
    assert outcome["existed"] is False
    assert outcome["deleted"] is False


def test_ecs_api_retries_transport_errors(monkeypatch) -> None:
    import httpx

    from apps.v2.agent.paper2code.execution.aliyun_lease import AliyunSettings, EcsClient

    client = EcsClient(AliyunSettings(access_key_id="k", access_key_secret="s", region_id="r", vswitch_id="v", security_group_id="g", image_id="m"))
    calls = {"n": 0}

    def _once(action, params):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("[SSL: UNEXPECTED_EOF_WHILE_READING]")
        return {"RequestId": "ok"}

    monkeypatch.setattr(client, "_api_once", _once)
    monkeypatch.setattr("apps.v2.agent.paper2code.execution.aliyun_lease.time.sleep", lambda s: None)
    assert client.api("DeleteInstance", {"InstanceId": "i-x"}) == {"RequestId": "ok"}
    assert calls["n"] == 3


def test_configure_refuses_only_an_active_lease(tmp_path: Path) -> None:
    # sapg-2 (2026-09-18): a compute rerun after the machine had been released must be able to re-decide
    clock, ecs = _Clock(), FakeEcs()
    port = _port(tmp_path, clock, ecs)
    port.configure(instance_type="ecs.c7.2xlarge", hard_cap_seconds=3600)
    assert port.instance_type == "ecs.c7.2xlarge"
    asyncio.run(port.run(Job(workspace=tmp_path, command="true", timeout_s=10)))
    with pytest.raises(RuntimeError, match="already rented"):
        port.configure(instance_type="ecs.c7.4xlarge")
    asyncio.run(port.close())
    assert json.loads((tmp_path / "lease.json").read_text())["state"] == "released"
    fresh = _port(tmp_path, clock, ecs)  # a new process over the same run directory: the record says released
    assert fresh.lease.record is not None
    fresh.configure(instance_type="ecs.gn6i-c8g1.2xlarge", hard_cap_seconds=7200)
    assert fresh.instance_type == "ecs.gn6i-c8g1.2xlarge"

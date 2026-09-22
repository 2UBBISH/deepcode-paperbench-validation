"""PLAN-3 item 4a: the line's ECS lease as main's ``LeaseBackend`` — a PLANNED lease for
``run_experiment_on_machine``, the machine bootstrap on acquire, every state in ``lease.json``,
and re-attaching to a held machine."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.v2.agent.paper2code.execution import machine_bootstrap as mb
from apps.v2.agent.paper2code.execution.aliyun_lease import PreparedLease, RunLease
from apps.v2.agent_engine.experiment.lease import ExperimentLease, LeaseError, LeaseState
from tests.v2_paper2code.fake_aliyun import FakeEcs, FakeRuntime


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def _run_lease(tmp_path: Path, ecs: FakeEcs, clock: _Clock) -> RunLease:
    return RunLease(tmp_path, clock=clock, client_factory=lambda: ecs, runtime_factory=FakeRuntime, ready_timeout=1.0, bring_up_backoff=0.0)


def _record(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "lease.json").read_text())


def setup_function(_fn) -> None:
    FakeRuntime.ready = True


# --- the flow's lease --------------------------------------------------------------------------


def test_experiment_lease_is_mains_orchestrator_with_the_line_backend(tmp_path: Path) -> None:
    ecs, clock = FakeEcs(), _Clock()
    run_lease = _run_lease(tmp_path, ecs, clock)
    seen: list = []

    async def fake_bootstrap(runtime):
        seen.append(runtime)
        return mb.BootstrapResult(seconds=3.0, exit_status=0, docker="29.0.0", git="2.43.0", images={t: "built" for t, _n, _p in mb.RSA_IMAGES}, grader_pytest="ok", machine_seconds=180)

    lease = run_lease.experiment_lease(run_id="run-1", hard_cap_seconds=3600, bootstrap=fake_bootstrap)
    assert isinstance(lease, ExperimentLease)
    assert lease.state is LeaseState.PLANNED

    async def scenario():
        runtime = await lease.acquire(RunLease.spec_for("ecs.c7.xlarge", 3600, clock=clock, hourly_price_cny=0.9))
        assert isinstance(runtime, FakeRuntime)
        lease.mark_running()
        assert lease.state is LeaseState.RUNNING
        after_acquire = _record(tmp_path)
        assert after_acquire["state"] == "running"
        assert after_acquire["instance_type"] == "ecs.c7.xlarge"
        assert after_acquire["bootstrap"]["images"] == {t: "built" for t, _n, _p in mb.RSA_IMAGES}
        assert after_acquire["bootstrap"]["machine_seconds"] == 180
        assert lease.handle is not None
        assert lease.handle.password
        assert lease.handle.access_url.startswith("ssh -p 22 root@10.0.0.")
        assert seen == [runtime]
        assert await lease.release(reason="success") is True

    asyncio.run(scenario())
    final = _record(tmp_path)
    assert final["state"] == "released"
    assert final["released_at"]
    assert ecs.instances == {}
    assert not run_lease.active


def test_bootstrap_failure_releases_the_machine_and_raises(tmp_path: Path) -> None:
    ecs, clock = FakeEcs(), _Clock()
    run_lease = _run_lease(tmp_path, ecs, clock)

    async def broken_bootstrap(runtime):
        raise mb.BootstrapError("grader pytest broken")

    lease = run_lease.experiment_lease(run_id="run-2", hard_cap_seconds=600, bootstrap=broken_bootstrap)
    with pytest.raises(LeaseError, match="could not be prepared"):
        asyncio.run(lease.acquire(RunLease.spec_for("ecs.c7.xlarge", 600, clock=clock)))
    assert lease.state is LeaseState.FAILED
    assert ecs.instances == {}
    record = _record(tmp_path)
    assert record["state"] == "released"
    assert any("bootstrap_failed" in e["note"] for e in record["events"])


def test_never_ready_machine_is_released_and_recorded(tmp_path: Path) -> None:
    ecs, clock = FakeEcs(), _Clock()
    run_lease = _run_lease(tmp_path, ecs, clock)
    FakeRuntime.ready = False
    lease = run_lease.experiment_lease(run_id="run-3", hard_cap_seconds=600, bootstrap=False)
    with pytest.raises(LeaseError):
        asyncio.run(lease.acquire(RunLease.spec_for("ecs.c7.xlarge", 600, clock=clock)))
    assert ecs.instances == {}
    assert _record(tmp_path)["state"] == "released"


def test_release_failure_is_recorded_as_release_failed_and_backstop_deletes(tmp_path: Path, monkeypatch) -> None:
    from apps.v2.agent_engine.experiment import lease as lease_module

    monkeypatch.setattr(lease_module, "RELEASE_BACKOFF_SECONDS", 0.0)
    ecs, clock = FakeEcs(), _Clock()
    run_lease = _run_lease(tmp_path, ecs, clock)
    lease = run_lease.experiment_lease(run_id="run-4", hard_cap_seconds=600, bootstrap=False)

    async def scenario():
        await lease.acquire(RunLease.spec_for("ecs.c7.xlarge", 600, clock=clock))
        ecs.fail_deletes = 2  # both attempts of the orchestrator fail; the backstop's delete goes through
        with pytest.raises(RuntimeError):
            await lease.release(reason="done", attempts=2)

    asyncio.run(scenario())
    assert _record(tmp_path)["state"] == "release_failed"
    assert run_lease.active
    result = asyncio.run(run_lease.force_release("backstop"))
    assert result["deleted"] is True
    assert _record(tmp_path)["state"] == "released"


def test_second_lease_refused_while_a_machine_is_held(tmp_path: Path) -> None:
    ecs, clock = FakeEcs(), _Clock()
    run_lease = _run_lease(tmp_path, ecs, clock)
    lease = run_lease.experiment_lease(run_id="run-5", hard_cap_seconds=600, bootstrap=False)
    asyncio.run(lease.acquire(RunLease.spec_for("ecs.c7.xlarge", 600, clock=clock)))
    with pytest.raises(LeaseError, match="already holds"):
        run_lease.experiment_lease(run_id="run-5", hard_cap_seconds=600, bootstrap=False)


def test_delete_refused_while_initialising_is_retried_by_the_backend(tmp_path: Path) -> None:
    from apps.v2.agent.paper2code.execution import aliyun_lease as al

    ecs, clock = FakeEcs(), _Clock()
    run_lease = _run_lease(tmp_path, ecs, clock)
    slept: list[float] = []

    async def no_sleep(seconds: float) -> None:
        slept.append(seconds)

    refusals = {"left": 3}
    real_delete = ecs.delete_instance

    def refusing_delete(instance_id):
        if refusals["left"] > 0:
            refusals["left"] -= 1
            raise al.LeaseError("Aliyun ECS DeleteInstance failed: IncorrectInstanceStatus.Initializing: The specified instance status does not support this operation.")
        real_delete(instance_id)

    ecs.delete_instance = refusing_delete
    lease = run_lease.experiment_lease(run_id="run-8", hard_cap_seconds=600, bootstrap=False)
    run_lease.backend()._sleep = no_sleep

    async def scenario():
        await lease.acquire(RunLease.spec_for("ecs.c7.xlarge", 600, clock=clock))
        assert await lease.release(reason="done") is True

    asyncio.run(scenario())
    assert slept == [al.DELETE_RETRY_SECONDS] * 3
    assert ecs.instances == {}
    assert _record(tmp_path)["state"] == "released"


# --- re-attaching after a review point held the machine -------------------------------------------


def test_adopt_rebuilds_the_handle_from_lease_json_and_can_release(tmp_path: Path) -> None:
    ecs, clock = FakeEcs(), _Clock()
    first = _run_lease(tmp_path, ecs, clock)
    lease = first.experiment_lease(run_id="run-6", hard_cap_seconds=7200, bootstrap=False)
    asyncio.run(lease.acquire(RunLease.spec_for("ecs.c7.2xlarge", 7200, clock=clock)))
    password = lease.handle.password

    # a new process: fresh RunLease over the same run dir
    clock.now += timedelta(seconds=1800)
    second = _run_lease(tmp_path, ecs, clock)
    adopted = second.adopt(run_id="run-6")
    assert isinstance(adopted, PreparedLease)
    assert adopted.state is LeaseState.READY
    assert adopted.handle.usage_id == lease.handle.usage_id
    assert adopted.handle.password == password
    assert adopted.handle.access_url == lease.handle.access_url
    assert 5390 <= adopted.hard_cap_seconds <= 5400  # the remaining cap, not the original
    assert adopted.adopted
    runtime = asyncio.run(adopted.acquire({"instance_type": "ecs.c7.2xlarge"}))  # the flow's acquire only reconnects
    assert isinstance(runtime, FakeRuntime)
    assert ecs.calls.count(("create", "i-fake1")) == 1  # no second machine
    assert asyncio.run(adopted.release(reason="review point answered: abort")) is True
    assert _record(tmp_path)["state"] == "released"
    assert ecs.instances == {}


def test_adopt_refuses_without_a_held_machine(tmp_path: Path) -> None:
    run_lease = _run_lease(tmp_path, FakeEcs(), _Clock())
    with pytest.raises(LeaseError, match="no machine to adopt"):
        run_lease.adopt(run_id="run-7")


# --- the bootstrap script ------------------------------------------------------------------------------


def test_bootstrap_script_embeds_mains_dockerfiles_verbatim_and_is_idempotent() -> None:
    script = mb.render_bootstrap()
    files = mb.dockerfiles()
    for tag, name, pull in mb.RSA_IMAGES:
        assert files[name].rstrip("\n") in script
        assert f"docker image inspect -f '{{{{index .Config.Labels \"{mb.DOCKERFILE_LABEL}\"}}}}' {tag}" in script
        sha = __import__("hashlib").sha256(files[name].encode()).hexdigest()
        assert f'if [ "$have" = "{sha}" ]' in script  # a stale or unlabelled image is rebuilt (drift detection)
        assert (f"docker build --pull --label {mb.DOCKERFILE_LABEL}={sha}" in script) is pull
        assert f"{mb.MARK} image={tag} built=" in script
    assert "python -m pytest --version" in script
    assert script.count("<<'BOOTSTRAP_DOCKERFILE'") == len(mb.RSA_IMAGES)
    # T4: the setupx-base image carries a torch build chosen on the machine; a wrong-variant image is rebuilt
    assert "torch_variant=none" in script  # parked: torch in the image defeats RSA's bare falsification (PITFALLS)
    import os

    os.environ["PAPER2CODE_TORCH_PREINSTALL"] = "1"
    try:
        script = mb.render_bootstrap()
    finally:
        os.environ.pop("PAPER2CODE_TORCH_PREINSTALL", None)
    assert "if nvidia-smi -L 2>/dev/null | grep -q GPU; then torch_variant=cu121; else torch_variant=cpu; fi" in script
    assert f"""have_variant=$(docker image inspect -f '{{{{index .Config.Labels "{mb.TORCH_VARIANT_LABEL}"}}}}' setupx-base:py310-proxy""" in script
    assert '&& [ "$have_variant" = "$torch_variant" ]; then' in script
    assert f'--build-arg TORCH_INDEX_URL="${{torch_index}}" --build-arg TORCH_VERSION={mb.TORCH_PREINSTALLED[0]} --build-arg TORCHVISION_VERSION={mb.TORCH_PREINSTALLED[1]} --label {mb.TORCH_VARIANT_LABEL}=$torch_variant -t setupx-base:py310-proxy' in script
    assert script.count("--build-arg TORCH_INDEX_URL") == 1  # the grader image is not variant-aware
    assert f"ARG TORCH_VERSION={mb.TORCH_PREINSTALLED[0]}" in files["Dockerfile.setupx-base"]
    assert f"ARG TORCHVISION_VERSION={mb.TORCH_PREINSTALLED[1]}" in files["Dockerfile.setupx-base"]
    # the line's bootstrap installs no driver / toolkit (the GPU image carries them, S5); the Dockerfile's comment may name nvidia wheels
    assert "nvidia-driver" not in script.lower()
    assert "nvidia-container-toolkit" not in script.lower()
    assert "cloudmonitor" not in script.lower()


def test_bootstrap_refuses_a_dockerfile_that_would_break_the_heredoc() -> None:
    with pytest.raises(mb.BootstrapError, match="heredoc"):
        mb.render_bootstrap({name: "FROM x\nBOOTSTRAP_DOCKERFILE\n" for _t, name, _p in mb.RSA_IMAGES})


class _RecordingRuntime:
    def __init__(self, output: str, exit_status: int = 0) -> None:
        self.output, self.exit_status = output, exit_status
        self.uploaded: list[tuple[str, str]] = []
        self.commands: list[tuple[str, float]] = []

    async def upload(self, local: str, remote: str) -> None:
        self.uploaded.append((Path(local).read_text(), remote))

    async def exec(self, command: str, *, timeout: float):
        self.commands.append((command, timeout))
        return SimpleNamespace(stdout=self.output, stderr="", exit_status=self.exit_status)


def _markers(*, built: bool = True, grader: str = "ok") -> str:
    lines = [f"{mb.MARK} docker=29.0.0", f"{mb.MARK} git=2.43.0"]
    lines += [f"{mb.MARK} image={tag} built={'yes' if built else 'no'}" for tag, _n, _p in mb.RSA_IMAGES]
    lines += [f"{mb.MARK} grader_pytest={grader}", f"{mb.MARK} done seconds=412"]
    return "\n".join(lines)


def test_bootstrap_machine_uploads_runs_and_parses(tmp_path: Path) -> None:
    runtime = _RecordingRuntime(_markers())
    result = asyncio.run(mb.bootstrap_machine(runtime, timeout=99.0, script="#!/bin/bash\necho hi\n"))
    assert result.ok
    assert result.images == {tag: "built" for tag, _n, _p in mb.RSA_IMAGES}
    assert result.machine_seconds == 412
    assert result.docker == "29.0.0"
    assert runtime.uploaded[0][0].startswith("#!/bin/bash")
    assert runtime.uploaded[0][1] == f"{mb.REMOTE_ROOT}.sh"
    assert runtime.commands == [(f"bash {mb.REMOTE_ROOT}.sh", 99.0)]
    assert result.record()["ok"] is True


def test_bootstrap_machine_on_a_prepared_image_reports_present(tmp_path: Path) -> None:
    result = asyncio.run(mb.bootstrap_machine(_RecordingRuntime(_markers(built=False)), script="x"))
    assert result.ok
    assert set(result.images.values()) == {"present"}


@pytest.mark.parametrize(
    ("output", "status", "reason"),
    [
        (_markers(grader="broken"), 3, "grader_pytest=broken"),
        (_markers().replace(f"{mb.MARK} image=rsa-grader:py311-v1 built=yes\n", ""), 0, "images="),
        ("apt failed", 100, "exit=100"),
    ],
)
def test_bootstrap_machine_raises_unless_every_step_passed(output: str, status: int, reason: str) -> None:
    with pytest.raises(mb.BootstrapError, match=reason):
        asyncio.run(mb.bootstrap_machine(_RecordingRuntime(output, status), script="x"))


def test_bootstrap_upload_failure_is_a_bootstrap_error() -> None:
    class _NoUpload(_RecordingRuntime):
        async def upload(self, local: str, remote: str) -> None:
            raise OSError("scp: broken pipe")

    with pytest.raises(mb.BootstrapError, match="uploading"):
        asyncio.run(mb.bootstrap_machine(_NoUpload(""), script="x"))

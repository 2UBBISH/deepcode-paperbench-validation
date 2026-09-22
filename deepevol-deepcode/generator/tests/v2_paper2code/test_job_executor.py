"""C5: the Docker executor against this machine's daemon (skipped when Docker is not reachable)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from apps.v2.agent.paper2code.config import EventLog
from apps.v2.agent.paper2code.execution.job_executor import (
    LocalDockerHost,
    RemoteDockerExecutor,
    ImagePolicy,
    find_requirements,
    mentions_torch,
    pip_install_command,
    workspace_manifest,
)
from apps.v2.agent.paper2code.execution.port import Job


def _docker_ok() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


docker_required = pytest.mark.skipif(not _docker_ok(), reason="local Docker daemon not reachable")

# Pulling python:3.11-slim from Docker Hub is slow or blocked on some developer machines; the tests use a
# locally present 3.11 image when there is one (PAPER2CODE_BASE_IMAGE also honours an explicit choice).
_LOCAL_BASES = ("python:3.11-slim", "python:3.11-slim-trixie", "python:3.11-slim-bookworm", "docker.1ms.run/library/python:3.11-slim-bookworm")


@pytest.fixture(autouse=True)
def _local_base_image(monkeypatch):
    if os.environ.get("PAPER2CODE_BASE_IMAGE") or not _docker_ok():
        return
    for candidate in _LOCAL_BASES:
        if subprocess.run(["docker", "image", "inspect", candidate], capture_output=True, timeout=30).returncode == 0:
            monkeypatch.setenv("PAPER2CODE_BASE_IMAGE", candidate)
            return
    pytest.skip("no local python:3.11 image; set PAPER2CODE_BASE_IMAGE or pull python:3.11-slim")


@docker_required
def test_pytest_job_runs_in_container_and_output_syncs_back(tmp_path: Path) -> None:
    workspace = tmp_path / "generate_code" / "proj"
    workspace.mkdir(parents=True)
    (workspace / "test_x.py").write_text("def test_ok():\n    open('made_in_container.txt', 'w').write('hi')\n    assert 1 + 1 == 2\n\ndef test_fail():\n    assert False\n")
    events = EventLog(tmp_path / "events.jsonl")
    executor = RemoteDockerExecutor(host=LocalDockerHost(), run_id="testrun", jobs_dir=tmp_path / "jobs", events=events)

    result = asyncio.run(executor.run(Job(workspace=workspace, command="python3 -m pytest -q", timeout_s=300, label="verify:pytest")))

    assert result.exit_code == 1, result
    assert "1 failed, 1 passed" in result.stdout
    assert result.machine == "local"
    assert not result.timed_out
    assert result.error is None
    assert (workspace / "made_in_container.txt").read_text() == "hi"
    assert "made_in_container.txt" in result.synced_back
    compiled = asyncio.run(executor.run(Job(workspace=workspace, command="python -m compileall -q .", timeout_s=120, label="compileall")))
    assert compiled.exit_code == 0
    assert not list(workspace.rglob("__pycache__"))  # bytecode stays outside the workspace
    job_dir = tmp_path / "jobs" / "0001"
    assert (job_dir / "job.json").exists()
    assert (job_dir / "result.json").exists()
    assert (job_dir / "stdout.txt").exists()
    kinds = [json.loads(line)["kind"] for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert "job.run" in kinds


@docker_required
def test_python_script_runs_outside_workspace_and_timeouts_kill(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    executor = RemoteDockerExecutor(host=LocalDockerHost(), run_id="testrun2", jobs_dir=tmp_path / "jobs")
    result = asyncio.run(executor.run(Job(workspace=workspace, command='python "$JOB_SCRIPT"', timeout_s=60, script="import os\nprint(sorted(os.listdir('.')))\nprint('hello from script')\n", label="execute_python")))
    assert result.exit_code == 0
    assert "hello from script" in result.stdout
    assert "[]" in result.stdout  # the script itself is not inside the workspace
    assert not list(workspace.iterdir())

    slow = asyncio.run(executor.run(Job(workspace=workspace, command="sleep 30", timeout_s=2, label="execute_bash")))
    assert slow.timed_out
    assert slow.exit_code == 124


@docker_required
def test_requirements_build_run_image_once(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "requirements.txt").write_text("six==1.16.0\n")
    executor = RemoteDockerExecutor(host=LocalDockerHost(), run_id="testrun3", jobs_dir=tmp_path / "jobs")
    first = asyncio.run(executor.run(Job(workspace=workspace, command="python -c 'import six; print(six.__version__)'", timeout_s=120)))
    assert first.exit_code == 0, first
    assert "1.16.0" in first.stdout, first
    state = json.loads((tmp_path / "jobs" / "image.json").read_text())
    assert state["status"] == "ok"
    assert state["image"].startswith("paper2code-run-testrun3:")
    second = asyncio.run(executor.run(Job(workspace=workspace, command="python -c 'import six'", timeout_s=120)))
    assert second.exit_code == 0
    assert not (tmp_path / "jobs" / "0002" / "pip.log").exists()  # reused, not rebuilt
    subprocess.run(["docker", "rmi", state["image"]], capture_output=True)


def test_manifest_and_requirements_discovery(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.pyc").write_text("y")
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "requirements.txt").write_text("numpy\n")
    assert list(workspace_manifest(tmp_path)) == ["a.py", "proj/requirements.txt"]
    assert find_requirements(tmp_path) == tmp_path / "proj" / "requirements.txt"
    (tmp_path / "requirements.txt").write_text("torch\n")
    assert find_requirements(tmp_path) == tmp_path / "requirements.txt"


def test_pip_install_command_adds_the_cpu_torch_index_only_when_torch_is_asked_for() -> None:
    policy = ImagePolicy(base_image="python:3.11-slim", pip_extra_args="", torch_cpu_index=True)
    assert pip_install_command("numpy>=1.21\npyyaml\n", policy) == "pip install --no-cache-dir -r /build/requirements.txt"
    assert mentions_torch("# torch is optional\nnumpy\n") is False
    assert mentions_torch("torch>=1.13.0  # core\n") is True
    assert mentions_torch("torchvision[extra]==0.20\n") is True
    assert mentions_torch("pytorch-lightning\n") is False
    with_torch = pip_install_command("torch>=1.13.0\nnumpy\n", policy)
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in with_torch
    assert with_torch.endswith("-r /build/requirements.txt")
    policy_gpu = ImagePolicy(base_image="python:3.11-slim", torch_cpu_index=False, pip_extra_args="--index-url https://mirror.example/simple")
    assert pip_install_command("torch\n", policy_gpu) == "pip install --no-cache-dir --index-url https://mirror.example/simple -r /build/requirements.txt"

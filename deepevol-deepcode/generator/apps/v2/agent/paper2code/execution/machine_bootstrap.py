"""Make a rented machine ready for the experiment agent: Docker, git, RSA's two container images.

A fresh Aliyun instance from a public Ubuntu image has none of what RSA needs on the
daemon (``rsa/DEEPEVOL_VENDOR.md`` "外部前置"): the ``setupx-base:py310-proxy`` image
SetupX builds its containers from, and the ``rsa-grader:py311-v1`` image the
adjudicator runs in. main's product machines get both from a pre-baked custom
image (``deploy/experiment-images/bootstrap.sh``, then ``CreateImage``); the CLI line
has no such image yet, so it builds them on the machine right after the lease's
``wait_ready`` (PLAN-3 §0 "机器镜像": build first, measure, bake once the measured
time says so). The script is idempotent — on a pre-baked image every step is a
probe that says "already there" and the whole thing takes seconds — so the same
code path serves both cases.

The two Dockerfiles are read from ``deploy/experiment-images`` at render time and
embedded verbatim: one source for what those images are, the same bytes main's
bootstrap builds. The Docker part is the line's existing ``MINIMAL_BOOTSTRAP`` (Aliyun
mirrors, registry mirror only when Docker Hub is unreachable). The NVIDIA driver
and Container Toolkit sections of main's script are left out on purpose: CPU tiers
only until PLAN-3 item 8, and the driver install needs a reboot the lease does not
model. CloudMonitor is a product-API image gate, irrelevant to a CLI machine.

The grader's pytest is verified by running it, not by the build succeeding — main's
bootstrap records a broken pytest 9.1.1 that imported fine, exited 0 and collected
nothing, surfacing three layers later as "pytest collected no tests".
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apps.v2.agent.paper2code.execution.aliyun_lease import MINIMAL_BOOTSTRAP

REPO_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_IMAGES_DIR = REPO_ROOT / "deploy" / "experiment-images"
#: (tag, Dockerfile under deploy/experiment-images, pass --pull to docker build)
RSA_IMAGES: tuple[tuple[str, str, bool], ...] = (
    ("setupx-base:py310-proxy", "Dockerfile.setupx-base", True),
    ("rsa-grader:py311-v1", "Dockerfile.rsa-grader", False),
)
REMOTE_ROOT = "/opt/paper2code/bootstrap"
BOOTSTRAP_TIMEOUT_S = float(os.environ.get("PAPER2CODE_BOOTSTRAP_TIMEOUT_S", str(25 * 60)))
MARK = "PAPER2CODE_BOOTSTRAP"
DOCKERFILE_LABEL = "paper2code.dockerfile_sha"
#: T4: which torch build the setupx-base image carries; decided on the machine (``nvidia-smi -L``) and recorded as a
#: second label, so a CPU-built image on a GPU machine (main's bootstrap.sh knows no build args) is rebuilt with CUDA
TORCH_VARIANT_LABEL = "paper2code.torch_variant"
TORCH_PREINSTALLED = ("2.1.2", "0.16.2")  # torch, torchvision — the Dockerfile's ARG defaults; the machine facts name them
TORCH_INDEX = {"cpu": "https://download.pytorch.org/whl/cpu", "cu121": "https://download.pytorch.org/whl/cu121"}


class BootstrapError(RuntimeError):
    pass


def dockerfiles(directory: Path = EXPERIMENT_IMAGES_DIR) -> dict[str, str]:
    """``{Dockerfile name: content}`` for RSA's two images, read from main's deploy directory."""
    files: dict[str, str] = {}
    for _tag, name, _pull in RSA_IMAGES:
        path = directory / name
        if not path.is_file():
            raise BootstrapError(f"{path} is missing; the line builds RSA's images from main's Dockerfiles")
        files[name] = path.read_text(encoding="utf-8")
    return files


def render_bootstrap(files: Mapping[str, str] | None = None) -> str:
    """The idempotent shell script: Docker → git → the two images → grader pytest check → summary line."""
    files = dict(files) if files is not None else dockerfiles()
    for name, content in files.items():
        if "\nBOOTSTRAP_DOCKERFILE\n" in content:
            raise BootstrapError(f"{name} contains the heredoc terminator")
    parts = [
        "#!/usr/bin/env bash",
        "# rendered by apps/v2/agent/paper2code/execution/machine_bootstrap.py — idempotent, safe to rerun",
        MINIMAL_BOOTSTRAP.strip("\n"),
        f'echo "{MARK} docker=$(docker version --format \'{{{{.Server.Version}}}}\')"',
        "if ! command -v git >/dev/null 2>&1; then apt_get -qq update && apt_get -qq install -y git; fi",
        f'echo "{MARK} git=$(git --version | awk \'{{print $3}}\')"',
        f"mkdir -p {shlex.quote(REMOTE_ROOT)}",
        # the torch build for this machine: CUDA when the host driver answers, CPU otherwise (the container only
        # sees the GPU when the host has one — run_flow adds --gpus all on the same evidence)
        # parked (2026-09-18 21:50, PITFALLS): torch in the image defeats RSA's bare falsification; opt in with
        # PAPER2CODE_TORCH_PREINSTALL=1 until the wheelhouse variant (T4b) lands
        (
            "if nvidia-smi -L 2>/dev/null | grep -q GPU; then torch_variant=cu121; else torch_variant=cpu; fi"
            if os.environ.get("PAPER2CODE_TORCH_PREINSTALL") == "1" else "torch_variant=none"
        ),
        f'echo "{MARK} torch_variant=$torch_variant"',
    ]
    for tag, name, pull in RSA_IMAGES:
        build_dir = f"{REMOTE_ROOT}/{name}"
        pull_flag = " --pull" if pull else ""
        variant_aware = name == "Dockerfile.setupx-base"
        # drift detection: the image carries the sha256 of the Dockerfile it was built from; a pre-baked image
        # built from an older Dockerfile (or by main's bootstrap.sh, unlabelled) is rebuilt — from the daemon's
        # layer cache, so only the changed layers cost anything (the pip index change of 2026-09-18 is why)
        want = hashlib.sha256(files[name].encode("utf-8")).hexdigest()
        if variant_aware:
            have_variant = f"have_variant=$(docker image inspect -f '{{{{index .Config.Labels \"{TORCH_VARIANT_LABEL}\"}}}}' {shlex.quote(tag)} 2>/dev/null || true)"
            condition = f'if [ "$have" = "{want}" ] && [ "$have_variant" = "$torch_variant" ]; then'
            build_args = (
                f' --build-arg TORCH_INDEX_URL="${{torch_index}}" --build-arg TORCH_VERSION={TORCH_PREINSTALLED[0]}'
                f" --build-arg TORCHVISION_VERSION={TORCH_PREINSTALLED[1]} --label {TORCH_VARIANT_LABEL}=$torch_variant"
            )
            index_line = [f'  if [ "$torch_variant" = cu121 ]; then torch_index={TORCH_INDEX["cu121"]}; elif [ "$torch_variant" = cpu ]; then torch_index={TORCH_INDEX["cpu"]}; else torch_index=; fi']
            built = f'  echo "{MARK} image={tag} built=yes dockerfile_sha={want[:12]} had=${{have:-none}} torch_variant=$torch_variant"'
        else:
            have_variant, condition, build_args, index_line = "", f'if [ "$have" = "{want}" ]; then', "", []
            built = f'  echo "{MARK} image={tag} built=yes dockerfile_sha={want[:12]} had=${{have:-none}}"'
        parts += [
            f"have=$(docker image inspect -f '{{{{index .Config.Labels \"{DOCKERFILE_LABEL}\"}}}}' {shlex.quote(tag)} 2>/dev/null || true)",
            *([have_variant] if have_variant else []),
            condition,
            f'  echo "{MARK} image={tag} built=no"',
            "else",
            f"  mkdir -p {shlex.quote(build_dir)}",
            f"  cat > {shlex.quote(build_dir)}/Dockerfile <<'BOOTSTRAP_DOCKERFILE'",
            files[name].rstrip("\n"),
            "BOOTSTRAP_DOCKERFILE",
            *index_line,
            f"  docker build{pull_flag} --label {DOCKERFILE_LABEL}={want}{build_args} -t {shlex.quote(tag)} -f {shlex.quote(build_dir)}/Dockerfile {shlex.quote(build_dir)}",
            built,
            "fi",
        ]
    grader = RSA_IMAGES[1][0]
    parts += [
        f"docker run --rm {shlex.quote(grader)} python -m pytest --version >/dev/null 2>&1 "
        f'|| {{ echo "{MARK} grader_pytest=broken"; exit 3; }}',
        f'echo "{MARK} grader_pytest=ok"',
        f'echo "{MARK} done seconds=$SECONDS"',
    ]
    return "\n".join(parts) + "\n"


@dataclass(slots=True)
class BootstrapResult:
    seconds: float
    exit_status: int
    docker: str = ""
    git: str = ""
    images: dict[str, str] = field(default_factory=dict)  # tag -> "built" | "present"
    grader_pytest: str = ""
    machine_seconds: int | None = None
    output_tail: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_status == 0 and self.grader_pytest == "ok" and set(self.images) == {tag for tag, _n, _p in RSA_IMAGES}

    def record(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "seconds": round(self.seconds, 1),
            "machine_seconds": self.machine_seconds,
            "exit_status": self.exit_status,
            "docker": self.docker,
            "git": self.git,
            "images": dict(self.images),
            "grader_pytest": self.grader_pytest,
        }


def parse_markers(output: str) -> dict[str, Any]:
    found: dict[str, Any] = {"images": {}}
    for match in re.finditer(rf"^{MARK} (.+)$", output, flags=re.MULTILINE):
        fields = dict(part.split("=", 1) for part in match.group(1).split() if "=" in part)
        if "image" in fields:
            found["images"][fields["image"]] = "built" if fields.get("built") == "yes" else "present"
        elif "seconds" in fields:
            found["machine_seconds"] = int(fields["seconds"]) if fields["seconds"].isdigit() else None
        else:
            found.update(fields)
    return found


async def bootstrap_machine(runtime: Any, *, timeout: float = BOOTSTRAP_TIMEOUT_S, script: str | None = None) -> BootstrapResult:
    """Upload the script over the lease's runtime and run it; raise :class:`BootstrapError` unless every step passed.

    ``runtime`` is main's remote_relay ``RemoteRuntime`` (``upload`` + ``exec``); a command this long
    goes through its durable job path, so an SSH drop mid ``docker build`` does not lose the build.
    """
    script = script if script is not None else render_bootstrap()
    remote_script = f"{REMOTE_ROOT}.sh"
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="p2c-bootstrap-") as tmp:
        local = Path(tmp) / "bootstrap.sh"
        local.write_text(script, encoding="utf-8")
        try:
            await runtime.upload(str(local), remote_script)
        except Exception as exc:
            raise BootstrapError(f"uploading the bootstrap script failed: {exc}") from exc
    result = await runtime.exec(f"bash {shlex.quote(remote_script)}", timeout=timeout)
    output = ((getattr(result, "stdout", "") or "") + "\n" + (getattr(result, "stderr", "") or "")).strip()
    status = getattr(result, "exit_status", None)
    status = 0 if status is None else int(status)
    markers = parse_markers(output)
    outcome = BootstrapResult(
        seconds=time.monotonic() - started,
        exit_status=status,
        docker=str(markers.get("docker", "")),
        git=str(markers.get("git", "")),
        images=dict(markers.get("images", {})),
        grader_pytest=str(markers.get("grader_pytest", "")),
        machine_seconds=markers.get("machine_seconds"),
        output_tail=output[-2000:],
    )
    if not outcome.ok:
        raise BootstrapError(
            f"machine bootstrap failed (exit={status}, images={outcome.images or '{}'}, grader_pytest={outcome.grader_pytest or '?'}): "
            f"{output[-600:] or '(no output)'}"
        )
    return outcome


__all__ = [
    "BOOTSTRAP_TIMEOUT_S",
    "EXPERIMENT_IMAGES_DIR",
    "RSA_IMAGES",
    "BootstrapError",
    "BootstrapResult",
    "bootstrap_machine",
    "dockerfiles",
    "parse_markers",
    "render_bootstrap",
]

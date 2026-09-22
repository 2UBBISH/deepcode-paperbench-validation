#!/usr/bin/env python3
"""Bake the Paper2Code line's GPU machine image (PLAN-3 §2.3 S5, item 8.1). One-off operations step.

    .venv/bin/python scripts/paper2code_bake_gpu_image.py --env-file ~/Documents/env/aliyun.env \\
        [--instance-type ecs.gn6i-c4g1.xlarge] [--base-image m-…] [--keep-source] [--cleanup]

main's ``scripts/build_experiment_image.py`` is the reference flow (open a machine → ``bootstrap.sh`` →
reboot when the driver needs it → second pass → CreateImage → delete the source), but it imports an
``apps.api`` package this tree does not carry, so the line drives the same steps with its own pieces:
``execution.aliyun_lease.EcsClient`` (the run's credentials from ``--env-file``, never printed),
remote_relay's ``RemoteRuntime`` over SSH, and main's ``deploy/experiment-images/bootstrap.sh`` as it is
(it detects NVIDIA hardware by PCI id, installs the driver and the Container Toolkit, builds RSA's two
images, installs CloudMonitor).

The bake starts from the account's existing CPU image (``DEEPEVOL_API_ALIYUN_IMAGE_ID``: Docker,
``setupx-base`` and ``rsa-grader`` already inside), so on a GPU machine the bootstrap only adds the driver
and the toolkit. Verification before CreateImage: ``nvidia-smi`` on the host and inside a container run
with ``--gpus all`` (the passthrough that ``torch.cuda.is_available()`` needs; torch itself is SetupX's
to install per run). The source machine is always deleted; ``--cleanup`` finds and deletes a leftover
build machine from an interrupted run (``p2c-imgbuild-*``).

Budget (owner, 2026-09-17): ≤ 50 CNY. A T4 (``ecs.gn6i-c4g1.xlarge``) is ~8 CNY/h in cn-hongkong; the
compute part of a bake is 20–40 minutes, the CreateImage part runs on a stopped (non-billing) machine.
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
import string
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from apps.v2.agent.paper2code.driver import load_env_files  # noqa: E402
from apps.v2.agent.paper2code.execution.aliyun_lease import (  # noqa: E402
    AliyunSettings,
    EcsClient,
    LeaseError,
    _runtime_factory,
    public_ip_from_instance,
)
from apps.v2.agent_engine.experiment.lease import LeaseHandle  # noqa: E402

DEFAULT_INSTANCE_TYPE = "ecs.gn6i-c4g1.xlarge"  # T4, the cheapest GPU tier in stock in cn-hongkong (8.07 CNY/h on 2026-09-18)
IMAGE_NAME_PREFIX = "deepevol-linux-gpu"
BUILD_NAME_PREFIX = "p2c-imgbuild-"
BOOTSTRAP = REPO / "deploy" / "experiment-images" / "bootstrap.sh"
BOOTSTRAP_ASSETS = (
    REPO / "deploy" / "experiment-images" / "Dockerfile.setupx-base",
    REPO / "deploy" / "experiment-images" / "Dockerfile.rsa-grader",
)
BOOTSTRAP_TIMEOUT_S = 40 * 60
CREATE_IMAGE_TIMEOUT_S = 90 * 60
GPU_CONTAINER_CHECK = "docker run --rm --gpus all setupx-base:py310-proxy nvidia-smi --query-gpu=name --format=csv,noheader"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _password(length: int = 24) -> str:
    alphabet = string.ascii_letters + string.digits
    return "A1!" + "".join(secrets.choice(alphabet) for _ in range(length - 3))


async def connect(host: str, password: str, *, attempts: int = 6, ready_timeout: float = 420.0):
    handle = LeaseHandle(usage_id="bake", resource_id="bake", access_url=f"ssh -p 22 root@{host}", username="root", password=password)
    last: Exception | None = None
    for i in range(attempts):
        runtime = _runtime_factory(handle)
        try:
            await runtime.start()
            await runtime.wait_ready(timeout=ready_timeout)
            return runtime
        except Exception as exc:
            last = exc
            try:
                await runtime.close()
            except Exception:
                pass
            if i + 1 < attempts:
                log(f"  connect attempt {i + 1} failed ({str(exc)[:90]}); the machine may still be booting")
                await asyncio.sleep(15 * (i + 1))
    raise RuntimeError(f"could not reach the build machine: {last}")


async def run_bootstrap(runtime, region: str, *, label: str) -> None:
    log(f"bootstrap.sh {label}…")
    await runtime.upload(str(BOOTSTRAP), "/root/bootstrap.sh")
    for asset in BOOTSTRAP_ASSETS:
        await runtime.upload(str(asset), f"/root/{asset.name}")
    job = await runtime.spawn(
        f"chmod +x /root/bootstrap.sh && DEEPEVOL_ALIYUN_REGION={region} bash /root/bootstrap.sh > /root/bootstrap.log 2>&1"
    )
    result = await runtime.wait(job.job_id, timeout=BOOTSTRAP_TIMEOUT_S)
    tail = await runtime.exec("tail -30 /root/bootstrap.log", timeout=60)
    log(f"  bootstrap exit={result.exit_status}")
    for line in ((tail.stdout or "") + (tail.stderr or "")).strip().splitlines()[-18:]:
        log(f"    {line[:160]}")
    if result.exit_status not in (0, None):
        raise RuntimeError("bootstrap failed; not baking")


async def run_line_bootstrap(runtime, *, label: str) -> str:
    """T4: after main's bootstrap.sh (unlabelled images, CPU torch), run the line's own rendered bootstrap — the one
    every lease runs — so the snapshot carries the labelled setupx-base with the torch build this machine wants
    (cu121 when nvidia-smi answers, CPU otherwise) and no lease rebuilds it. Returns the torch variant it chose."""
    from apps.v2.agent.paper2code.execution.machine_bootstrap import MARK, render_bootstrap

    import tempfile

    log(f"line bootstrap {label}…")
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(render_bootstrap())
        local = fh.name
    await runtime.upload(local, "/root/line_bootstrap.sh")
    job = await runtime.spawn("bash /root/line_bootstrap.sh > /root/line_bootstrap.log 2>&1")
    result = await runtime.wait(job.job_id, timeout=BOOTSTRAP_TIMEOUT_S)
    tail = await runtime.exec(f"grep '{MARK}' /root/line_bootstrap.log | tail -12", timeout=60)
    text = (tail.stdout or "") + (tail.stderr or "")
    for line in text.strip().splitlines():
        log(f"    {line[:160]}")
    if result.exit_status not in (0, None):
        raise RuntimeError("the line's bootstrap failed; not baking")
    variant = ""
    for line in text.splitlines():
        if "torch_variant=" in line:
            variant = line.split("torch_variant=")[-1].split()[0]
    return variant


async def gpu_state(runtime) -> tuple[bool, bool]:
    hw = await runtime.exec("lspci 2>/dev/null | grep -qi nvidia && echo yes || echo no", timeout=60)
    smi = await runtime.exec("nvidia-smi >/dev/null 2>&1 && echo yes || echo no", timeout=120)
    return "yes" in (hw.stdout or ""), "yes" in (smi.stdout or "")


async def verify_gpu(runtime) -> dict[str, str]:
    checks = {
        "nvidia-smi": "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1",
        "docker nvidia runtime": "docker info 2>/dev/null | grep -i runtimes | head -1",
        "container sees the GPU": GPU_CONTAINER_CHECK + " 2>&1 | head -1",
        "setupx-base image": "docker images --format '{{.Repository}}:{{.Tag}}' | grep -c setupx-base",
        "grader pytest": "docker run --rm rsa-grader:py311-v1 python -m pytest --version 2>&1 | head -1",
        "cms unit": "systemctl is-enabled deepevol-cms-ensure.service 2>&1 | head -1",
    }
    out: dict[str, str] = {}
    for label, cmd in checks.items():
        probe = await runtime.exec(cmd, timeout=180)
        lines = ((probe.stdout or "") + (probe.stderr or "")).strip().splitlines()
        out[label] = (lines[0] if lines else "(empty)")[:120]
        log(f"  check {label}: {out[label]}")
    return out


def delete_with_retry(client: EcsClient, instance_id: str) -> bool:
    deadline = time.monotonic() + 600
    while True:
        try:
            client.delete_instance(instance_id)
            return True
        except LeaseError as exc:
            if time.monotonic() >= deadline:
                log(f"  ⚠️ could not delete {instance_id}: {str(exc)[-120:]}")
                return False
            log(f"  delete refused ({str(exc)[-80:]}); retrying in 15s")
            time.sleep(15)


def cleanup(client: EcsClient) -> int:
    data = client.api("DescribeInstances", {"PageSize": 100})
    leftovers = [
        (i.get("InstanceId"), i.get("InstanceName"), i.get("Status"))
        for i in ((data.get("Instances") or {}).get("Instance") or [])
        if str(i.get("InstanceName") or "").startswith(BUILD_NAME_PREFIX)
    ]
    if not leftovers:
        log("no leftover build machine")
        return 0
    for instance_id, name, status in leftovers:
        log(f"deleting leftover {instance_id} ({name}, {status})")
        delete_with_retry(client, str(instance_id))
    return 0


async def bake(args: argparse.Namespace) -> int:
    settings = AliyunSettings.from_env()
    client = EcsClient(settings)
    region = settings.region_id
    base_image = args.base_image or settings.image_id
    if args.cleanup:
        return cleanup(client)
    data = client.api("DescribeInstances", {"PageSize": 100})
    stale = [i.get("InstanceId") for i in ((data.get("Instances") or {}).get("Instance") or []) if str(i.get("InstanceName") or "").startswith(BUILD_NAME_PREFIX)]
    if stale:
        log(f"❌ a build machine from an earlier run is still there and billing: {stale}; run with --cleanup first")
        return 1
    hourly = client.hourly_price_cny(args.instance_type)
    log(f"region {region} zone {settings.zone_id or '-'} type {args.instance_type} ({hourly:.2f} CNY/h) base image {base_image} disk {settings.system_disk_size_gb} GiB")

    password = _password()
    name = f"{BUILD_NAME_PREFIX}{time.strftime('%m%d%H%M')}"
    image_name = f"{'deepevol-linux-cpu' if args.cpu else IMAGE_NAME_PREFIX}-{time.strftime('%Y%m%d-%H%M')}"
    started = time.time()
    instance_id = ""
    runtime = None
    compute_seconds = 0.0
    checks: dict[str, str] = {}
    try:
        log(f"RunInstances {name}…")
        instance_id = client.run_instance(name=name, instance_type=args.instance_type, password=password, image_id=base_image)
        log(f"  {instance_id}")
        instance = client.wait_for_status(instance_id, {"Running"}, timeout_seconds=300)
        if str(instance.get("Status") or "") != "Running":
            raise RuntimeError(f"instance did not reach Running: {instance.get('Status')!r}")
        host = public_ip_from_instance(instance) or client.allocate_public_ip(instance_id)
        if not host:
            host = public_ip_from_instance(client.describe_instance(instance_id))
        if not host:
            raise RuntimeError("no public IP")
        log(f"  running, host {host}")
        runtime = await connect(host, password)
        await run_bootstrap(runtime, region, label="(first pass)")
        has_gpu, driver_ok = await gpu_state(runtime)
        log(f"GPU hardware={'yes' if has_gpu else 'NO'} nvidia-smi={'ok' if driver_ok else 'not yet'}")
        if args.cpu:
            if has_gpu:
                raise RuntimeError(f"--cpu asked but {args.instance_type} has an NVIDIA device; not baking")
            variant = await run_line_bootstrap(runtime, label="(CPU torch layer)")
            if variant != "cpu":
                raise RuntimeError(f"the line's bootstrap chose torch variant {variant!r} on a CPU machine; not baking")
            check = await runtime.exec("docker run --rm setupx-base:py310-proxy python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'", timeout=300)
            log(f"  container torch: {(check.stdout or '').strip()[:80]}")
            if "+cpu" not in (check.stdout or ""):
                raise RuntimeError("the CPU image's container does not report a +cpu torch; not baking")
        elif not has_gpu:
            raise RuntimeError(f"{args.instance_type} shows no NVIDIA device; not baking (use --cpu for the CPU image)")
        if args.cpu:
            pass  # no driver, no toolkit, no second pass: the line's bootstrap above is the whole CPU bake
        elif not driver_ok:
            log("driver installed but not live — rebooting once…")
            await runtime.close()
            runtime = None
            client.api("RebootInstance", {"InstanceId": instance_id, "ForceStop": "false"})
            await asyncio.sleep(30)
            client.wait_for_status(instance_id, {"Running"}, timeout_seconds=300)
            runtime = await connect(host, password, attempts=8)
            await run_bootstrap(runtime, region, label="(after reboot: toolkit + idempotency)")
            has_gpu, driver_ok = await gpu_state(runtime)
            if not driver_ok:
                raise RuntimeError("driver still not usable after the reboot; not baking")
        else:
            # the toolkit section only runs when nvidia-smi works; a second pass also proves idempotency time
            pass_started = time.time()
            await run_bootstrap(runtime, region, label="(second pass: idempotency)")
            log(f"  second pass took {time.time() - pass_started:.0f}s")
        if not args.cpu:
            checks = await verify_gpu(runtime)
            if "nvidia" not in checks["docker nvidia runtime"].lower():
                raise RuntimeError("the nvidia container runtime is not registered with Docker; not baking")
            if "error" in checks["container sees the GPU"].lower() or checks["container sees the GPU"] in {"(empty)"}:
                raise RuntimeError("a container with --gpus all does not see the GPU; not baking")
            variant = await run_line_bootstrap(runtime, label="(CUDA torch layer)")
            if variant != "cu121":
                raise RuntimeError(f"the line's bootstrap chose torch variant {variant!r} on the GPU machine; not baking")
            check = await runtime.exec("docker run --rm --gpus all setupx-base:py310-proxy python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'", timeout=300)
            log(f"  container torch: {(check.stdout or '').strip()[:80]}")
            if "True" not in (check.stdout or ""):
                raise RuntimeError("torch in the container does not see the GPU; not baking")
        await runtime.close()
        runtime = None

        log("StopInstance (stop-charging)…")
        client.stop_instance(instance_id)
        client.wait_for_status(instance_id, {"Stopped"}, timeout_seconds=300)
        compute_seconds = time.time() - started

        log(f"CreateImage {image_name}…")
        description = ("Paper2Code line CPU image: Docker + RSA images with torch CPU pre-installed (T4)" if args.cpu
                       else "Paper2Code line GPU image: CPU base + NVIDIA driver + Container Toolkit + torch cu121 pre-installed (PLAN-3 S5, T4)")
        created = client.api("CreateImage", {"InstanceId": instance_id, "ImageName": image_name, "Description": description})
        image_id = str(created.get("ImageId") or "")
        if not image_id:
            raise RuntimeError(f"CreateImage returned no ImageId: {created}")
        deadline = time.time() + CREATE_IMAGE_TIMEOUT_S
        last = ""
        failures = 0
        while time.time() < deadline:
            try:
                images = client.api("DescribeImages", {"ImageId": image_id, "Status": "Creating,Available,CreateFailed"})
                failures = 0
            except LeaseError as exc:
                failures += 1
                log(f"  DescribeImages failed ({failures}): {str(exc)[:80]}")
                if failures >= 20:
                    raise RuntimeError(f"cannot poll the image; it may still be creating: {image_id}; source {instance_id}") from exc
                time.sleep(30)
                continue
            items = ((images.get("Images") or {}).get("Image") or [])
            status = str(items[0].get("Status") if items else "missing")
            progress = str(items[0].get("Progress") if items else "")
            if status == "Available":
                break
            if status in {"CreateFailed", "missing", "Deprecated"}:
                raise RuntimeError(f"CreateImage ended in {status}")
            if progress != last:
                log(f"  … {status} {progress} ({time.time() - started:.0f}s)")
                last = progress
            time.sleep(20)
        else:
            raise RuntimeError("CreateImage timed out")
        total_min = (time.time() - started) / 60
        cost = hourly * compute_seconds / 3600
        log(f"✅ image {image_id} ({image_name}); {total_min:.1f} min total, compute {compute_seconds / 60:.1f} min ≈ {cost:.2f} CNY at {hourly:.2f}/h")
        print()
        print("=" * 64)
        print(f"  {'CPU' if args.cpu else 'GPU'} image id   {image_id}")
        print(f"  image name     {image_name}")
        print(f"  instance type  {args.instance_type}")
        print(f"  checks         {checks}")
        print(f"  cost           ≈ {cost:.2f} CNY compute (+ image storage)")
        print("  next           " + ("DEEPEVOL_API_ALIYUN_IMAGE_ID=<id> into aliyun.env (the CPU tiers boot it)" if args.cpu else "DEEPEVOL_API_ALIYUN_GPU_IMAGE_ID=<id> into aliyun.env (S6 picks it for GPU tiers)"))
        print("=" * 64)
        return 0
    except Exception as exc:
        log(f"❌ {type(exc).__name__}: {exc}")
        return 1
    finally:
        if runtime is not None:
            try:
                await runtime.close()
            except Exception:
                pass
        if instance_id and not args.keep_source:
            log(f"deleting the source machine {instance_id}…")
            if delete_with_retry(client, instance_id):
                log("  deleted")
            else:
                pending = REPO / "storage" / "PENDING_CLEANUP.txt"
                pending.parent.mkdir(parents=True, exist_ok=True)
                with pending.open("a", encoding="utf-8") as fh:
                    fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {instance_id} (paper2code_bake_gpu_image could not delete; still billing)\n")
                log(f"  ⚠️ recorded in {pending}; delete it by hand")
        elif instance_id:
            log(f"⚠️ source machine {instance_id} kept as asked; it is billing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", action="append", default=[], help="KEY=VALUE file with the Aliyun settings (repeatable)")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--base-image", default=None, help="image to start from (default: the settings' image, the CPU-baked one)")
    parser.add_argument("--keep-source", action="store_true")
    parser.add_argument("--cleanup", action="store_true", help="delete leftover build machines and exit")
    parser.add_argument("--cpu", action="store_true", help="T4: bake the CPU image instead (no driver checks; torch CPU build in setupx-base); pair with --instance-type ecs.c7.xlarge")
    args = parser.parse_args()
    load_env_files(args.env_file)
    return asyncio.run(bake(args))


if __name__ == "__main__":
    raise SystemExit(main())

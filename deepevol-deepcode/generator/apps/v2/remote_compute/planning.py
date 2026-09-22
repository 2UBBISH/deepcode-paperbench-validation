"""Pure Remote Compute intent and preflight projection shared by V2 runtimes."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from apps.common.v2_ids import format_typed_id

from .models import RemoteComputeResource


EXPERIMENT_KEYWORDS = (
    "experiment",
    "benchmark",
    "train",
    "training",
    "fine-tune",
    "finetune",
    "evaluate",
    "ablation",
    "跑实验",
    "实验",
    "训练",
    "微调",
    "基准",
    "验证",
    "消融",
)
GPU_KEYWORDS = (
    "neural",
    "deep learning",
    "pytorch",
    "tensorflow",
    "jax",
    "cuda",
    "gpu",
    "gnn",
    "transformer",
    "llm",
    "diffusion",
    "神经网络",
    "深度学习",
    "图神经网络",
    "大模型",
    "显卡",
)
CPU_KEYWORDS = (
    "cpu",
    "vcpu",
    "multiprocessing",
    "parallel",
    "joblib",
    "ray",
    "simulation",
    "simulator",
    "sweep",
    "并行",
    "多核",
    "上百核",
    "几十核",
    "仿真",
    "参数扫描",
)
_EXECUTION_ARTIFACT = re.compile(
    r"(?:运行|执行|跑|训练|验证|评测|测试|实验|run|execute|train|evaluate|benchmark)"
    r".{0,80}(?:代码|脚本|模型|结果|文件|artifact|code|script|model|result)",
    re.IGNORECASE | re.DOTALL,
)


def requires_compute_selection(prompt: str) -> bool:
    """Match the existing product gate without importing the legacy server."""

    text = str(prompt or "").strip()
    if not text:
        return False
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in EXPERIMENT_KEYWORDS) or bool(
        _EXECUTION_ARTIFACT.search(text)
    )


def estimate_required_resources(
    prompt: str,
    policy: Mapping[str, int] | None,
) -> dict[str, Any]:
    text = str(prompt or "")
    lowered = text.lower()
    gpu_heavy = any(keyword.lower() in lowered for keyword in GPU_KEYWORDS)
    cpu_heavy = any(keyword.lower() in lowered for keyword in CPU_KEYWORDS)
    local_memory_mb = int((policy or {}).get("memory_mb") or 0)
    small_graph_training = bool(
        re.search(
            r"(?:\b5\b|5\s*个).{0,40}(?:300|500).{0,80}(?:训练|train)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
    )
    estimated_minutes = (
        45
        if gpu_heavy and small_graph_training
        else 90
        if gpu_heavy
        else 60
        if cpu_heavy
        else 30
    )
    return {
        "gpu_recommended": gpu_heavy,
        "cpu_recommended": cpu_heavy and not gpu_heavy,
        "min_cpu_cores": 16 if cpu_heavy and not gpu_heavy else 4 if gpu_heavy else 2,
        "min_memory_mb": 16 * 1024 if gpu_heavy or cpu_heavy else 4 * 1024,
        "min_vram_gb": 16 if gpu_heavy else 0,
        "estimated_minutes": estimated_minutes,
        "local_risk": (
            "high" if gpu_heavy or cpu_heavy or local_memory_mb < 4096 else "medium"
        ),
        "reason": (
            "neural_or_gpu_workload"
            if gpu_heavy
            else "cpu_parallel_workload"
            if cpu_heavy
            else "experiment_or_benchmark"
        ),
    }


# Explicit requirement keys a caller may override (experiment Agent: static
# repository analysis instead of the prompt-text guess).  Anything else in the
# request is dropped rather than refused.
_EXPLICIT_REQUIREMENT_KEYS: dict[str, type] = {
    "gpu_recommended": bool,
    "cpu_recommended": bool,
    "min_cpu_cores": int,
    "min_memory_mb": int,
    "min_memory_gb": int,
    "min_vram_gb": int,
    "min_storage_gb": int,
    "estimated_minutes": int,
}


def merge_resource_requirements(
    estimated: Mapping[str, Any],
    explicit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Overlay explicit workload requirements on the prompt-text estimate.

    The estimate is the V1 heuristic (keywords in the task text); a caller
    that has read the code knows better, and its numbers win.  ``min_memory_gb``
    is accepted as the human unit and folded into ``min_memory_mb``.
    """

    merged = dict(estimated)
    if not explicit:
        return merged
    for key, expected in _EXPLICIT_REQUIREMENT_KEYS.items():
        if key not in explicit:
            continue
        value = explicit[key]
        if expected is bool:
            if not isinstance(value, bool):
                continue
            merged[key] = value
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            continue
        if key == "min_memory_gb":
            merged["min_memory_mb"] = value * 1024
            continue
        merged[key] = value
    if "gpu_recommended" in explicit and merged.get("gpu_recommended"):
        merged["cpu_recommended"] = False
        merged["reason"] = "explicit_gpu_requirement"
        merged["min_vram_gb"] = max(int(merged.get("min_vram_gb") or 0), 1)
    elif "gpu_recommended" in explicit and not merged.get("gpu_recommended"):
        merged["min_vram_gb"] = 0
        merged["reason"] = "explicit_cpu_requirement"
    return merged


def catalog_requirements(required: Mapping[str, Any]) -> dict[str, int]:
    """The integer-only shape the Agent Provider catalog accepts."""

    return {
        "min_cpu_cores": int(required.get("min_cpu_cores") or 0),
        "min_memory_mb": int(required.get("min_memory_mb") or 0),
        "min_vram_gb": int(required.get("min_vram_gb") or 0),
        "gpu_recommended": 1 if required.get("gpu_recommended") else 0,
        "cpu_recommended": 1 if required.get("cpu_recommended") else 0,
        "min_storage_gb": int(required.get("min_storage_gb") or 0),
    }


def build_remote_compute_preflight(
    *,
    prompt: str,
    resource_policy: Mapping[str, int] | None,
    resources: Sequence[RemoteComputeResource],
    provider_options: Sequence[Mapping[str, Any]],
    required: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    required = dict(required) if required is not None else estimate_required_resources(prompt, resource_policy)
    minutes = int(required["estimated_minutes"])
    options: list[dict[str, Any]] = []
    for resource in resources:
        if resource.deleted_at is not None or resource.status not in {
            "ACTIVE",
            "IDLE",
            "STOPPED",
        }:
            continue
        prefix = "cloud_cpu" if resource.provider == "aliyun_ecs" else "cloud_gpu"
        target = f"{prefix}:{format_typed_id('rcres', resource.resource_id)}"
        options.append(
            {
                "id": format_typed_id("rcres", resource.resource_id),
                "target": target,
                "provider": resource.provider,
                "name": resource.name,
                "spec": _resource_spec(resource),
                "region": resource.region,
                "instance_type": resource.instance_type,
                "accelerator_type": resource.accelerator_type,
                "accelerator_count": resource.accelerator_count,
                "vram_gb": resource.vram_gb,
                "cpu_cores": resource.cpu_cores,
                "memory_gb": resource.memory_gb,
                "storage_gb": resource.storage_gb,
                "hourly_price_credits": resource.hourly_price_credits,
                "estimated_minutes": minutes,
                "estimated_credits": _estimate_credits(
                    resource.hourly_price_credits,
                    minutes,
                ),
                "status": "available" if resource.status != "STOPPED" else "stopped",
                "status_label": "可立即使用" if resource.status != "STOPPED" else "可开机复用",
                "available_now": True,
                "existing_resource": True,
            }
        )
    seen = {str(option["target"]) for option in options}
    for raw in provider_options:
        option = _provider_option(raw, minutes=minutes)
        target = str(option["target"])
        if target not in seen:
            options.append(option)
            seen.add(target)
    options.sort(
        key=lambda item: (
            0 if bool(item.get("available_now")) else 1,
            int(item.get("estimated_credits") or 0),
            str(item.get("target") or ""),
        )
    )
    recommended_target = _recommended_target(required, options)
    estimates = [int(item.get("estimated_credits") or 0) for item in options]
    return {
        "required": required,
        "local_option": {
            "target": "local_cpu",
            "cpu_cores": int((resource_policy or {}).get("cpu_cores") or 0),
            "threads": int((resource_policy or {}).get("threads") or 0),
            "memory_mb": int((resource_policy or {}).get("memory_mb") or 0),
            "docker_available": True,
            "risk": required["local_risk"],
            "warning": (
                "本地可能跑不起来，只适合 tiny/sanity 降级。"
                if required["gpu_recommended"]
                else "本地可跑，但仍建议先做小样本 sanity check。"
            ),
        },
        "cloud_options": options[:12],
        "recommended_target": recommended_target,
        "estimated_cost": {
            "minutes": minutes,
            "min_credits": min(estimates, default=0),
            "max_credits": max(estimates, default=0),
        },
    }


def _provider_option(raw: Mapping[str, Any], *, minutes: int) -> dict[str, Any]:
    option = dict(raw)
    target = option.get("target")
    provider = option.get("provider")
    name = option.get("name")
    if not all(isinstance(value, str) and value.strip() for value in (target, provider, name)):
        raise ValueError("remote compute provider option is malformed")
    if not str(target).startswith(("cloud_gpu:", "cloud_cpu:")):
        raise ValueError("remote compute provider target is invalid")
    option["estimated_minutes"] = minutes
    option["estimated_credits"] = _estimate_credits(
        int(option.get("hourly_price_credits") or 0),
        minutes,
    )
    option["existing_resource"] = False
    return option


def _recommended_target(required: Mapping[str, Any], options: Sequence[Mapping[str, Any]]) -> str:
    prefix = "cloud_gpu:" if required.get("gpu_recommended") else "cloud_cpu:"
    preferred = next(
        (
            item
            for item in options
            if str(item.get("target") or "").startswith(prefix)
            and bool(item.get("available_now"))
        ),
        None,
    )
    if preferred is None and required.get("gpu_recommended"):
        preferred = next(
            (
                item
                for item in options
                if str(item.get("target") or "").startswith("cloud_gpu:")
            ),
            None,
        )
    return str((preferred or {}).get("target") or "local_cpu")


def _estimate_credits(hourly_price_credits: int, minutes: int) -> int:
    if hourly_price_credits <= 0 or minutes <= 0:
        return 0
    return math.ceil(hourly_price_credits * minutes / 60)


def _resource_spec(resource: RemoteComputeResource) -> str:
    parts = []
    if resource.accelerator_type:
        parts.append(
            f"{resource.accelerator_type} x{max(resource.accelerator_count, 1)}"
        )
    if resource.cpu_cores:
        parts.append(f"{resource.cpu_cores} vCPU")
    if resource.memory_gb:
        parts.append(f"{resource.memory_gb} GB RAM")
    if resource.region:
        parts.append(resource.region)
    return " / ".join(parts) or resource.provider


__all__ = [
    "build_remote_compute_preflight",
    "catalog_requirements",
    "estimate_required_resources",
    "merge_resource_requirements",
    "requires_compute_selection",
]

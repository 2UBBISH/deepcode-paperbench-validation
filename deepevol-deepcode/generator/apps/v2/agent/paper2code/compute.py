"""Compute phase (算力): the compute spec of the generated code, the tier plan, the review point.

PLAN-3 item 3. The estimate reuses main's ``apps.v2.agent_engine.experiment`` package unchanged:
static analysis of the generated repository (every fact with its evidence line) → ``ComputeSpec``
(memory, storage, whether and how much GPU; deliberately no running time) → a tier plan (GPU tiers
by VRAM step, or two CPU tiers). The line adds only what it needs around that:

* the environment spec can force ``needs_gpu`` when the blueprint says a GPU is required and
  the code did not reveal it;
* the machines a tier maps to come from the line's own catalogues (the ECS types the lease knows
  how to rent): CPU tiers on ``CPU_CATALOG``, GPU tiers on ``GPU_CATALOG`` by VRAM and GPU count
  (PLAN-3 item 8.2 / S6). **Two-stage compute (owner, 2026-09-18 afternoon)**: the default is always the
  cheapest CPU tier — step 10 starts there (a smoke run on sapg never touched the T4 it was on) — and the
  decision names the ``escalation_type``, the GPU machine the phase re-rents when the run's own evidence
  says it needs one (``environment_controller.GPU_REQUIRED``); a person may still pick a GPU tier up front;
* the review point: a request in the DeepEvol ``ask_user`` question shape written to
  ``phases/09_compute.request.json``; the decision comes back from
  ``phases/09_compute.decision.json``; without ``--ask`` the default tier is taken and recorded
  as ``auto``. Nothing is rented before the decision — the leased port only rents on its first
  job, and the chosen machine is set on it here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from collections.abc import Callable

from apps.v2.agent.paper2code.execution.aliyun_lease import CPU_TIERS, GPU_CATALOG, GPU_TIER_KEYS, INSTANCE_CAPACITY, SMALLEST_GPU_TYPE, is_gpu_type
from apps.v2.agent_engine.experiment.compute_spec import ComputeSpec, build_compute_spec
from apps.v2.agent_engine.experiment.cpu_tiers import build_cpu_tiers, cpu_tier_requirements
from apps.v2.agent_engine.experiment.repo_facts import analyse_resources
from apps.v2.agent_engine.experiment.tiers import TierPlan, build_tiers

REQUEST_FILE = "09_compute.request.json"
DECISION_FILE = "09_compute.decision.json"
INTERACTION_TYPE = "compute_review"
CANCEL = "cancel"

#: The line's rentable CPU machines, cheapest first. ``ecs.c7.large`` (2 vCPU / 4 GiB) is deliberately
#: absent: a CPU torch install plus compileall was only ever exercised on ``ecs.c7.xlarge`` and up.
CPU_CATALOG: tuple[str, ...] = ("ecs.c7.xlarge", "ecs.c7.2xlarge", "ecs.c7.4xlarge", "ecs.c7.8xlarge")
GPU_KEY_PREFIX = "gpu-"
#: ``probe(instance_type) -> {"in_stock": bool | None, "hourly_price_cny": float | None}`` (the lease's ``EcsClient.machine_probe``)
Probe = Callable[[str], dict[str, Any]]


# ---------------------------------------------------------------------------
# estimate
# ---------------------------------------------------------------------------


def _catalog() -> list[dict[str, Any]]:
    return [{"instance_type": t, "cpu_cores": INSTANCE_CAPACITY[t][0], "memory_gb": INSTANCE_CAPACITY[t][1]} for t in CPU_CATALOG]


def _gpu_catalog() -> list[dict[str, Any]]:
    return [
        {"instance_type": t, "cpu_cores": INSTANCE_CAPACITY[t][0], "memory_gb": INSTANCE_CAPACITY[t][1], "gpu": name, "vram_gib": vram, "gpu_count": count}
        for t, (name, vram, count) in GPU_CATALOG.items()
    ]


def _gpu_match(tier: dict[str, Any], spec: ComputeSpec, *, exclude: list[str] = ()) -> dict[str, Any] | None:
    """Smallest GPU machine with the tier's VRAM and GPU count and the spec's host RAM (catalogue order = price order)."""
    vram = int(tier.get("vram_gib") or 0)
    count = int(tier.get("gpu_count") or 1)
    host_ram = float(getattr(spec, "host_ram_gib", 0) or 0)
    for option in _gpu_catalog():
        if option["instance_type"] in exclude:
            continue
        if option["vram_gib"] >= vram and option["gpu_count"] >= count and option["memory_gb"] >= host_ram:
            return option
    return None


def _with_probe(entry: dict[str, Any], probe: Probe | None) -> dict[str, Any]:
    if probe is None or not entry.get("instance_type"):
        return entry
    try:
        info = probe(entry["instance_type"]) or {}
    except Exception as exc:  # the review point still opens without stock and price
        info = {"probe_error": f"{type(exc).__name__}: {exc}"[:120]}
    return {**entry, **{k: info.get(k) for k in ("in_stock", "hourly_price_cny", "probe_error") if k in info}}


def _cpu_match(tier_requirements: dict[str, Any], *, exclude: list[str] = ()) -> dict[str, Any] | None:
    """Cheapest catalogue machine meeting the cores/memory floor (price is not part of the rule).

    ``exclude`` keeps two tiers from landing on the same machine: the second tier takes the next
    size up so the review point offers a real choice.
    """
    cores = int(tier_requirements.get("min_cpu_cores") or 0)
    memory = int(tier_requirements.get("min_memory_gb") or 0)
    for option in _catalog():
        if option["instance_type"] in exclude:
            continue
        if option["cpu_cores"] >= cores and (not memory or option["memory_gb"] >= memory):
            return option
    return None


def estimate(code_dir: Path, environment_spec: dict[str, Any] | None, *, probe: Probe | None = None, gpu_available: bool = True) -> dict[str, Any]:
    """Compute spec + tier plan + machine per tier for the generated repository.

    ``gpu_available`` = a GPU image is configured, so GPU tiers are rentable; ``probe`` adds stock and
    hourly price per machine (a live API call each — the caller passes it only for a real decision).
    """
    facts = analyse_resources(code_dir)
    spec: ComputeSpec = build_compute_spec(facts)
    notes: list[str] = []
    env_gpu = ((environment_spec or {}).get("gpu") or {}).get("required")
    env_reason = ((environment_spec or {}).get("gpu") or {}).get("reason")
    if env_gpu is True and not spec.needs_gpu:
        spec.needs_gpu = True
        notes.append(f"environment spec says a GPU is required ({env_reason or 'no reason given'}); the code alone did not show it")
    gpu_plan: TierPlan | None = build_tiers(spec) if spec.needs_gpu else None
    cpu_plan: TierPlan = build_cpu_tiers(spec)
    cpu_tiers = []
    taken: list[str] = []
    for tier in cpu_plan.tiers:
        requirements = cpu_tier_requirements(tier, spec)
        match = _cpu_match(requirements, exclude=taken)
        if match:
            taken.append(match["instance_type"])
        cpu_tiers.append({**tier.to_dict(), "requirements": requirements, "instance_type": match["instance_type"] if match else None,
                          "cpu_cores": match["cpu_cores"] if match else None, "memory_gb": match["memory_gb"] if match else None})
    cpu_notes = [n for n in cpu_plan.notes if "不需要 GPU" not in n] if spec.needs_gpu else list(cpu_plan.notes)
    gpu_tiers: list[dict[str, Any]] = []
    if gpu_plan is not None:
        taken_gpu: list[str] = []
        for tier in gpu_plan.tiers:
            match = _gpu_match(tier.to_dict(), spec, exclude=taken_gpu)
            if match:
                taken_gpu.append(match["instance_type"])
            gpu_tiers.append({**tier.to_dict(), "key": GPU_KEY_PREFIX + tier.key, "instance_type": match["instance_type"] if match else None,
                              "cpu_cores": match["cpu_cores"] if match else None, "memory_gb": match["memory_gb"] if match else None,
                              "machine_gpu": f"{match['gpu']} {match['vram_gib']} GiB×{match['gpu_count']}" if match else None})
    gpu_default = GPU_KEY_PREFIX + gpu_plan.default_key if gpu_plan is not None else None
    if gpu_default is not None and not any(t["key"] == gpu_default and t["instance_type"] for t in gpu_tiers):
        gpu_default = next((t["key"] for t in gpu_tiers if t["instance_type"]), None)
    escalation_type = escalation_machine(gpu_plan, spec) if gpu_available else None
    if spec.needs_gpu and not gpu_available:
        notes.append("the code needs a GPU but no GPU image is configured for this run (ALIYUN_GPU_IMAGE_ID); the CPU tiers are all there is — "
                     "a GPU-class failure stops the run instead of escalating")
    elif gpu_available:
        notes.append(f"step 10 starts on the CPU tier; a failure that needs a GPU escalates to {escalation_type} (two-stage compute)"
                     + ("; the spec says the full experiments need a GPU — that is not the smoke run's need" if spec.needs_gpu else ""))
    default_key = cpu_plan.default_key
    return {
        "facts": {
            "uses_gpu": _fact(facts.uses_gpu), "torch_cuda": _fact(facts.torch_cuda), "batch_size": _fact(facts.batch_size),
            "precision": _fact(facts.precision), "resumable": _fact(facts.resumable), "unknowns": list(facts.unknowns), "coherent": facts.coherent,
        },
        "spec": spec.to_dict(),
        "needs_gpu": spec.needs_gpu,
        "gpu_available": bool(gpu_available),
        "gpu_tiers": gpu_plan.to_dict() if gpu_plan else None,
        "gpu_machines": [_with_probe(t, probe) for t in gpu_tiers],
        "cpu_tiers": [_with_probe(t, probe) for t in cpu_tiers],
        "cpu_default": cpu_plan.default_key,
        "default": default_key,
        "escalation_type": escalation_type,
        "notes": notes + cpu_notes,
    }


def escalation_machine(gpu_plan: TierPlan | None, spec: ComputeSpec) -> str:
    """The GPU machine the phase re-rents on a GPU-class failure: the smallest catalogue GPU whose VRAM and GPU
    count carry the plan's smallest tier (owner 2026-09-18: the 4-vCPU T4 for a smoke run; the plan's default
    16 GiB host-RAM guess is not evidence and does not push it to the 8-vCPU machine)."""
    tiers = list(gpu_plan.tiers) if gpu_plan is not None else []
    smallest = min((t.to_dict() for t in tiers), key=lambda t: (int(t.get("vram_gib") or 0), int(t.get("gpu_count") or 1)), default={"vram_gib": 0, "gpu_count": 1})
    vram, count = int(smallest.get("vram_gib") or 0), int(smallest.get("gpu_count") or 1)
    for option in _gpu_catalog():
        if option["vram_gib"] >= vram and option["gpu_count"] >= count:
            return str(option["instance_type"])
    return SMALLEST_GPU_TYPE


def _fact(fact: Any) -> dict[str, Any] | None:
    if fact is None:
        return None
    return {"value": getattr(fact, "value", None), "evidence": getattr(fact, "evidence", "")}


# ---------------------------------------------------------------------------
# review point
# ---------------------------------------------------------------------------


def _choice(tier: dict[str, Any]) -> dict[str, str]:
    machine = tier["instance_type"] or "no catalogue machine fits"
    size = f"{tier['cpu_cores']} vCPU / {tier['memory_gb']} GiB" if tier["instance_type"] else ""
    if tier.get("machine_gpu"):
        size = f"{tier['machine_gpu']}, {size}"
    extras = []
    if tier.get("hourly_price_cny"):
        extras.append(f"{float(tier['hourly_price_cny']):.2f} CNY/h")
    if tier.get("in_stock") is False:
        extras.append("out of stock now")
    label = f"{tier['label']} · {machine}" + (f" ({size})" if size else "") + (f" — {', '.join(extras)}" if extras else "")
    return {"value": tier["key"], "label": label, "description": f"{tier['blurb']}；{tier['rationale']}"}


def build_request(estimate_record: dict[str, Any], *, run_hours: float, default_key: str) -> dict[str, Any]:
    """The review request, in DeepEvol's ``ask_user`` question shape (one multiple-choice question)."""
    spec = estimate_record["spec"]
    vram = spec.get("vram") or {}
    lines = [
        f"Workload: {spec.get('workload')}; GPU needed: {'yes' if estimate_record['needs_gpu'] else 'no'}; confidence: {spec.get('confidence')}.",
        f"Host RAM ≥ {spec.get('host_ram_gib')} GiB; storage ≥ {spec.get('storage_gib')} GiB.",
    ]
    gpu_machines = estimate_record.get("gpu_machines") or []
    rentable_gpu = bool(estimate_record.get("gpu_available")) and any(t["instance_type"] for t in gpu_machines)
    if estimate_record["gpu_tiers"]:
        tiers = ", ".join(f"{t['label']} {t['vram_gib']} GiB×{t['gpu_count']}" for t in estimate_record["gpu_tiers"]["tiers"])
        state = "rentable" if rentable_gpu else "not rentable by this run"
        lines.append(f"GPU tiers by VRAM ({state}): {tiers}; VRAM estimate {vram.get('low_gib')}–{vram.get('high_gib')} GiB.")
    lines.extend(estimate_record["notes"])
    priced = [t for t in [*gpu_machines, *estimate_record["cpu_tiers"]] if t.get("hourly_price_cny")]
    if priced:
        top = max(float(t["hourly_price_cny"]) for t in priced)
        lines.append(f"Hourly prices are live; at {run_hours:g} h the dearest choice here costs up to {top * run_hours:.0f} CNY.")
    lines.append(f"The machine is rented for at most {run_hours:g} h (run.json run_hours). Pick a tier; nothing is rented before this decision.")
    choices = [_choice(t) for t in (gpu_machines if rentable_gpu else [])] + [_choice(t) for t in estimate_record["cpu_tiers"]] + [{"value": CANCEL, "label": "取消", "description": "do not rent; the environment step is skipped"}]
    return {
        "interaction_type": INTERACTION_TYPE,
        "phase": "compute",
        "questions": [{"question": "\n".join(lines), "type": "multiple_choice", "header": "选一档配置", "choices": choices, "required": True, "custom": True}],
        "default": default_key,
        "requested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision_file": DECISION_FILE,
        "instructions": f"Write {DECISION_FILE} next to this file with {{\"action\": \"approve\" | \"cancel\", \"answers\": [\"<tier key or ecs.* type>\"], \"run_hours\": <optional>}}.",
    }


def resolve_decision(decision: dict[str, Any] | None, estimate_record: dict[str, Any], *, default_key: str) -> dict[str, Any]:
    """Turn a decision (or none, meaning auto-approve) into the machine to rent.

    The answer may be a CPU tier key, a GPU tier key (``gpu-…``, only when GPU tiers are rentable), a
    catalogue ``ecs.*`` type, or a legacy ``compute_tier`` name. The result says whether the machine has a
    GPU (``gpu``) — step 10's goal, SetupX's machine facts and the lease's image follow from that.
    """
    rentable_gpu = bool(estimate_record.get("gpu_available"))
    by_key = {t["key"]: t for t in estimate_record["cpu_tiers"]}
    if rentable_gpu:
        by_key.update({t["key"]: t for t in (estimate_record.get("gpu_machines") or []) if t.get("instance_type")})

    def _result(mode: str, tier: str, instance_type: str, run_hours: Any) -> dict[str, Any]:
        entry = by_key.get(tier) or {}
        return {
            "mode": mode, "action": "approve", "tier": tier, "instance_type": instance_type,
            "run_hours": float(run_hours) if run_hours is not None else None,
            "gpu": is_gpu_type(instance_type), "hourly_price_cny": entry.get("hourly_price_cny"), "in_stock": entry.get("in_stock"),
            # two-stage compute: where a CPU start escalates to on a GPU-class failure (None = no GPU image, or already on one)
            "escalation_type": None if is_gpu_type(instance_type) else estimate_record.get("escalation_type"),
        }

    if decision is None:
        tier = by_key[default_key]
        return _result("auto", default_key, tier["instance_type"], None)
    action = str(decision.get("action") or "approve")
    if action == CANCEL:
        return {"mode": "ask", "action": CANCEL, "tier": None, "instance_type": None, "run_hours": None, "gpu": False}
    answers = decision.get("answers") or []
    answer = str(answers[0]).strip() if answers else default_key
    if answer in by_key:
        instance_type = by_key[answer]["instance_type"]
        tier = answer
    elif answer in INSTANCE_CAPACITY:
        if is_gpu_type(answer) and not rentable_gpu:
            raise ValueError(f"{answer} is a GPU type but this run has no GPU image (ALIYUN_GPU_IMAGE_ID)")
        instance_type, tier = answer, "custom"
    elif answer in CPU_TIERS:
        instance_type, tier = CPU_TIERS[answer], answer
    elif answer in GPU_TIER_KEYS:
        raise ValueError(f"GPU tier {answer!r} is not offered for this code (needs_gpu={estimate_record['needs_gpu']}, gpu_available={rentable_gpu}); pick one of {sorted(by_key)} or an ecs.gn* type")
    else:
        raise ValueError(f"decision answer {answer!r} is neither a tier key {sorted(by_key)} nor a known ecs.* type {sorted(INSTANCE_CAPACITY)}")
    return _result("ask", tier, instance_type, decision.get("run_hours"))


def read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


__all__ = ["CANCEL", "CPU_CATALOG", "DECISION_FILE", "GPU_KEY_PREFIX", "INTERACTION_TYPE", "REQUEST_FILE", "Probe", "build_request", "estimate", "read_json", "resolve_decision", "write_json"]

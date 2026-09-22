"""PLAN-3 item 3: the compute estimate, the tier → machine mapping and the decision rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from apps.v2.agent.paper2code.compute import CPU_CATALOG, build_request, estimate, resolve_decision


def _repo(tmp_path: Path, *, gpu: bool) -> Path:
    root = tmp_path / "gen"
    (root / "proj").mkdir(parents=True)
    body = "import torch\nmodel = torch.nn.Linear(4, 4)\n"
    if gpu:
        body += "device = 'cuda' if torch.cuda.is_available() else 'cpu'\nmodel.to(device)\n"
    (root / "proj" / "train.py").write_text(body + "for step in range(3):\n    pass\n")
    return root


def test_cpu_repository_gets_two_cpu_tiers_on_catalogue_machines(tmp_path: Path) -> None:
    record = estimate(_repo(tmp_path, gpu=False), None)
    assert record["needs_gpu"] is False
    assert record["gpu_tiers"] is None
    assert [t["key"] for t in record["cpu_tiers"]] == ["economy", "standard"]
    assert record["cpu_tiers"][0]["instance_type"] == "ecs.c7.xlarge"
    assert record["cpu_tiers"][1]["instance_type"] == "ecs.c7.2xlarge"  # the next size up, never the same machine twice
    assert all(t["instance_type"] in CPU_CATALOG for t in record["cpu_tiers"])
    assert record["cpu_default"] == "economy"


def test_environment_spec_can_force_gpu_and_without_a_gpu_image_cpu_is_the_fallback(tmp_path: Path) -> None:
    spec = {"gpu": {"required": True, "reason": "IsaacGym requires NVIDIA GPU"}}
    record = estimate(_repo(tmp_path, gpu=False), spec, gpu_available=False)
    assert record["needs_gpu"] is True
    assert any("environment spec says a GPU is required" in n for n in record["notes"])
    assert any("no GPU image is configured" in n for n in record["notes"])
    assert not any("不需要 GPU" in n for n in record["notes"])
    assert record["gpu_tiers"] is not None
    assert record["gpu_available"] is False
    assert record["default"] == "economy"  # a CPU tier: nothing else is rentable
    assert [t["key"] for t in record["cpu_tiers"]] == ["economy", "standard"]
    request = build_request(record, run_hours=6, default_key=record["default"])
    question = request["questions"][0]
    assert "GPU tiers by VRAM (not rentable by this run)" in question["question"]
    assert question["header"] == "选一档配置"
    assert [c["value"] for c in question["choices"]] == ["economy", "standard", "cancel"]
    assert request["default"] == "economy"
    with pytest.raises(ValueError, match="no GPU image"):
        resolve_decision({"answers": ["ecs.gn6i-c4g1.xlarge"]}, record, default_key="economy")
    with pytest.raises(ValueError, match="not offered"):
        resolve_decision({"answers": ["gpu-standard"]}, record, default_key="economy")


def test_gpu_code_with_a_gpu_image_starts_on_cpu_and_names_the_escalation_machine(tmp_path: Path) -> None:
    # PLAN-3 S6: needs_gpu + GPU image → GPU tiers map to catalogue machines by VRAM. Two-stage compute (owner
    # 2026-09-18): the default is still the CPU tier; the decision names the machine a GPU-class failure escalates to
    probes: list[str] = []

    def probe(instance_type):
        probes.append(instance_type)
        return {"in_stock": instance_type != "ecs.gn7i-c32g1.16xlarge", "hourly_price_cny": {"ecs.gn6i-c8g1.2xlarge": 9.53}.get(instance_type, 12.5)}

    record = estimate(_repo(tmp_path, gpu=True), None, probe=probe, gpu_available=True)
    assert record["needs_gpu"] is True
    assert record["gpu_available"] is True
    machines = record["gpu_machines"]
    assert machines, record["gpu_tiers"]
    assert all(m["key"].startswith("gpu-") for m in machines)
    assert all(m["instance_type"] is None or m["instance_type"].startswith("ecs.gn") for m in machines)
    first = machines[0]
    # the smallest machine whose VRAM covers the tier AND whose RAM covers the spec's host RAM (16 GiB: the
    # 4-vCPU T4 has 15 GiB, so the 8-vCPU T4 is the first fit)
    assert first["instance_type"] == "ecs.gn6i-c8g1.2xlarge"
    assert first["vram_gib"] <= 16
    assert first["in_stock"] is True
    assert first["hourly_price_cny"] == 9.53
    assert [m["instance_type"] for m in machines if m["vram_gib"] > 24] == [None] * len([m for m in machines if m["vram_gib"] > 24])  # nothing in the catalogue has 32 GiB
    assert record["default"] == "economy"  # CPU first
    assert record["escalation_type"] == "ecs.gn6i-c4g1.xlarge"  # the smallest T4 carries the 8 GiB tier; the plan's host-RAM guess does not count
    assert any("escalates to ecs.gn6i-c4g1.xlarge" in n for n in record["notes"])
    assert set(probes) >= {m["instance_type"] for m in machines if m["instance_type"]} | {t["instance_type"] for t in record["cpu_tiers"]}
    request = build_request(record, run_hours=2, default_key=record["default"])
    question = request["questions"][0]
    assert "GPU tiers by VRAM (rentable)" in question["question"]
    assert "Hourly prices are live" in question["question"]
    values = [c["value"] for c in question["choices"]]
    assert values[0].startswith("gpu-")
    assert values[-3:] == ["economy", "standard", "cancel"]
    assert "9.53 CNY/h" in question["choices"][0]["label"]
    assert "T4 16 GiB×1" in question["choices"][0]["label"]
    auto = resolve_decision(None, record, default_key=record["default"])
    assert auto["gpu"] is False
    assert auto["instance_type"] == "ecs.c7.xlarge"
    assert auto["escalation_type"] == "ecs.gn6i-c4g1.xlarge"
    by_type = resolve_decision({"answers": ["ecs.gn7i-c8g1.2xlarge"]}, record, default_key=record["default"])
    assert by_type == {**by_type, "tier": "custom", "instance_type": "ecs.gn7i-c8g1.2xlarge", "gpu": True, "escalation_type": None}
    gpu_on_purpose = resolve_decision({"answers": ["gpu-economy"]}, record, default_key=record["default"])
    assert gpu_on_purpose["gpu"] is True
    assert gpu_on_purpose["instance_type"] == first["instance_type"]
    assert gpu_on_purpose["escalation_type"] is None  # already on a GPU


def test_cpu_code_never_defaults_to_a_gpu_tier_even_with_a_gpu_image(tmp_path: Path) -> None:
    record = estimate(_repo(tmp_path, gpu=False), None, gpu_available=True)
    assert record["needs_gpu"] is False
    assert record["gpu_machines"] == []
    assert record["default"] == "economy"
    assert record["escalation_type"] == "ecs.gn6i-c4g1.xlarge"  # even CPU-looking code may want CUDA at run time: the smallest T4 waits
    assert resolve_decision(None, record, default_key="economy")["gpu"] is False
    no_image = estimate(_repo(tmp_path / "second", gpu=True), None, gpu_available=False)
    assert no_image["escalation_type"] is None
    assert resolve_decision(None, no_image, default_key="economy")["escalation_type"] is None


def test_code_evidence_alone_marks_gpu(tmp_path: Path) -> None:
    record = estimate(_repo(tmp_path, gpu=True), None)
    assert record["needs_gpu"] is True
    assert record["facts"]["uses_gpu"]["evidence"].startswith("proj/train.py:")


def test_resolve_decision_rules(tmp_path: Path) -> None:
    record = estimate(_repo(tmp_path, gpu=False), None)
    auto = resolve_decision(None, record, default_key="economy")
    assert auto == {"mode": "auto", "action": "approve", "tier": "economy", "instance_type": "ecs.c7.xlarge", "run_hours": None, "gpu": False, "hourly_price_cny": None, "in_stock": None, "escalation_type": "ecs.gn6i-c4g1.xlarge"}
    by_tier = resolve_decision({"action": "approve", "answers": ["standard"]}, record, default_key="economy")
    assert by_tier["instance_type"] == "ecs.c7.2xlarge"
    by_type = resolve_decision({"answers": ["ecs.c7.4xlarge"], "run_hours": 2}, record, default_key="economy")
    assert by_type["tier"] == "custom"
    assert by_type["instance_type"] == "ecs.c7.4xlarge"
    assert by_type["run_hours"] == 2.0
    legacy = resolve_decision({"answers": ["comfortable"]}, record, default_key="economy")
    assert legacy["instance_type"] == "ecs.c7.2xlarge"
    cancelled = resolve_decision({"action": "cancel"}, record, default_key="economy")
    assert cancelled["action"] == "cancel"
    assert cancelled["instance_type"] is None
    with pytest.raises(ValueError, match="neither a tier key"):
        resolve_decision({"answers": ["ecs.g8.xlarge"]}, record, default_key="economy")

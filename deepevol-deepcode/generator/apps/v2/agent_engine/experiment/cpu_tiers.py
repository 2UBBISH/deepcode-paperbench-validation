"""不需要 GPU 的仓库也要有机器可选。

## 为什么单独一个模块，而不是复用显存档位

两者的**约束轴不同**。显存那边是一根硬轴：装不下就是跑不起来，
所以档位按「区间下沿 / 上沿 / 2× 保险」切。CPU 这边没有那样的悬崖——
核数少只是慢，内存不足才会崩，而内存的估计比显存粗得多
（`compute_spec` 在参数量未知时把 `host_ram_gib` 钉在 16 的下限上）。

把 CPU 硬塞进显存那套里，会写出「区间下沿 + 15% 裕度」这种对 CPU
毫无意义的解释文案 —— 注释与代码打架，下一个人就不敢改了。

## 这里为什么只有两档

CPU 规格之间的差别对用户是「快一点 / 慢一点」，不是「跑得起来 / 跑不起来」。
给三档只会让他在两个没有实质区别的选项之间纠结。
够用 + 宽裕，说清楚差在哪，就够了。
"""

from __future__ import annotations

from typing import Any

from .compute_spec import DEFAULT_HOST_RAM_FLOOR_GIB, ComputeSpec
from .tiers import Tier, TierPlan

#: 「够用」档的最低核数。低于这个数，pip 装依赖和编译扩展会慢到让人以为卡死。
MIN_CORES_ECONOMY = 2

#: 「宽裕」档相对够用档的倍数。2× 是个诚实的粗粒度——
#: CPU 任务的加速比取决于它有没有并行，静态估不出来，所以不承诺倍数。
COMFORT_MULTIPLIER = 2


def build_cpu_tiers(spec: ComputeSpec) -> TierPlan:
    """给不需要 GPU 的仓库出两档。

    `Tier.vram_gib` 在这里恒为 0（CPU 机没有显存），选型靠
    `catalog.match_tier` 的 `min_host_ram_gib` 与下面的核数过滤。
    """
    plan = TierPlan(default_key="economy")

    ram = float(spec.host_ram_gib or DEFAULT_HOST_RAM_FLOOR_GIB)
    # ★ 与 GPU 路径同一条判断：等于下限时这个值**不携带任何信息**
    #   （`compute_spec` 里 base_ram = max(16.0, ...)），
    #   拿它当硬门槛会把便宜机器白白刷掉。见 recommend._enforceable_host_ram。
    ram_is_evidence = ram > DEFAULT_HOST_RAM_FLOOR_GIB

    plan.tiers.append(Tier(
        key="economy", label="够用档", vram_gib=0, gpu_count=0,
        blurb="能跑完，只是慢一点",
        rationale=(
            f"至少 {MIN_CORES_ECONOMY} 核"
            + (f"、内存 ≥ {ram:.0f}G（代码里读出来的）" if ram_is_evidence
               else "；内存没读出明确需求，不额外设限")
        ),
    ))
    plan.tiers.append(Tier(
        key="standard", label="宽裕档", vram_gib=0, gpu_count=0,
        blurb="核数和内存都留了余量，装依赖和跑数据处理会快些",
        rationale=f"够用档的 {COMFORT_MULTIPLIER}× 核数",
    ))

    plan.notes.append(
        "这份代码**不需要 GPU** —— 下面是 CPU 规格，比 GPU 机便宜一到两个数量级"
    )
    if not ram_is_evidence:
        plan.notes.append(
            "没从代码里读出明确的内存需求，所以两档都没按内存卡门槛。"
            "如果你知道它吃内存，直接告诉我要多大"
        )
    # 抢占式：与 GPU 路径同一条规矩 —— 不支持断点续跑就别推，被回收要从头再来。
    plan.spot_allowed = spec.resumable
    return plan


def cpu_tier_requirements(tier: Tier, spec: ComputeSpec) -> dict[str, Any]:
    """这一档对机器的硬要求。`match_tier` 只认内存，核数在这里额外过滤。"""
    cores = MIN_CORES_ECONOMY * (COMFORT_MULTIPLIER if tier.key == "standard" else 1)
    ram = float(spec.host_ram_gib or 0)
    return {
        "min_cpu_cores": cores,
        # 同上：等于下限时不当门槛。
        "min_memory_gb": int(ram) if ram > DEFAULT_HOST_RAM_FLOOR_GIB else 0,
    }


def filter_cpu_options(
    options: list[dict[str, Any]], requirements: dict[str, Any]
) -> list[dict[str, Any]]:
    """按核数与内存筛，再按**绝对价格**排。

    ★ 排序规矩与 GPU 路径一致（`catalog.match_tier`）：档内的候选已经都满足
      需求了，这时该选绝对最便宜的那个，而不是「每核单价」最低的——
      后者会把用户推去买一台 64 核的机器，因为大机器每核更便宜。
    """
    min_cores = int(requirements.get("min_cpu_cores") or 0)
    min_memory = int(requirements.get("min_memory_gb") or 0)
    kept = [
        o for o in options
        if int(o.get("cpu_cores") or 0) >= min_cores
        and (not min_memory or int(o.get("memory_gb") or 0) >= min_memory)
        and float(o.get("hourly_price_cny") or 0) > 0
    ]
    kept.sort(key=lambda o: (float(o["hourly_price_cny"]), int(o.get("cpu_cores") or 0)))
    return kept

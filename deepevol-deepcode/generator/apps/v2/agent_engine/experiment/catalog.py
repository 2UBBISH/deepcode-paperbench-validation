"""把档位落到**真的能买到**的阿里云 SKU 上。

数据来源是 `scripts/aliyun_discover.py` 产出的目录（`DescribeInstanceTypes` ∩
`DescribeAvailableResource` ∩ `DescribePrice` + GPU 能力知识表），本模块只做选型。

## 两条会让自动化流程硬失败的规矩

**一、必须查库存。** A100/H800 这类经常无货。只按规格筛会推荐出买不到的配置——
在人工流程里那只是"换一个"，在自动流程里是 `CreateInstance` 直接失败。
所以目录进来时就应当已经与库存取过交集（discover 脚本负责），这里再做一次断言式过滤。

**二、架构代际是硬门槛，不是打分项。** 用了 flash-attn 2 的代码在 T4 上不是慢，
是起不来。`min_compute_capability` 不满足的直接出局，不能靠价格补回来。

## 同一档内按**绝对价格**排，不按每 GiB 单价

每 GiB 单价是用来**跨地域、跨代际比较**的指标（实测 cn-hongkong 的 L20 48G 是
0.386 元/GB/时，cn-hangzhou 的 A10 24G 是 0.401，境外反而便宜）。
但它**不能用来在同一档内排序**：档内的候选已经都满足显存需求了，
这时该选的是绝对最便宜的那个。

拿它当排序键会得到荒谬结果——实测过一次：稳妥档只要 8G，却把 48G 的 L20
（18.52 元/时）和 384G 的 8 卡机（148 元/时）排在 T4 16G（7.96 元/时）前面，
因为大卡的每 GiB 单价一样低。用户要 8G，被推去付 2.3 倍甚至 18 倍的钱。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .tiers import Tier


@dataclass
class SkuOption:
    instance_type: str
    accelerator_type: str
    accelerator_count: int
    vram_gb: int
    cpu_cores: int
    memory_gb: int
    hourly_price_cny: float
    sm: str = ""
    gen: str = ""
    bf16: bool | None = None
    driver_may_lag: bool = False
    #: preflight 里那条候选的 `target`。选中之后要**原样**交回给 usage/start ——
    #: API 侧是靠它在 run metadata 的 preflight 结果里反查 catalog 条目的
    #: （`remote_compute.py:868`），编一个格式对但不存在的 target，
    #: 反查会落空、开机时缺 image_id / cloudmonitor 标志。
    target: str = ""

    @property
    def total_vram_gb(self) -> int:
        return self.vram_gb * max(self.accelerator_count, 1)

    @property
    def price_per_vram_gb(self) -> float:
        total = self.total_vram_gb
        return (self.hourly_price_cny / total) if total and self.hourly_price_cny else float("inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_type": self.instance_type,
            "accelerator_type": self.accelerator_type,
            "accelerator_count": self.accelerator_count,
            "vram_gb": self.vram_gb,
            "total_vram_gb": self.total_vram_gb,
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "hourly_price_cny": self.hourly_price_cny,
            # ★ CPU 档没有显存，price_per_vram_gb 是 inf —— 而 inf 不是合法 JSON。
            #   它会一路带到回调 payload 里，在 httpx（allow_nan=False）那里抛
            #   ValueError，把整条终态事件弄丢。这里就地转成 None：
            #   「没有显存所以这个比值没有意义」本来就该是 null，不是无穷大。
            "price_per_vram_gb": (
                round(self.price_per_vram_gb, 4)
                if math.isfinite(self.price_per_vram_gb) else None
            ),
            "sm": self.sm, "gen": self.gen, "bf16": self.bf16,
            "driver_may_lag": self.driver_may_lag,
            "target": self.target,
        }


def option_from_catalog(entry: dict[str, Any]) -> SkuOption:
    return SkuOption(
        instance_type=str(entry.get("instance_type") or ""),
        accelerator_type=str(entry.get("accelerator_type") or ""),
        accelerator_count=int(entry.get("accelerator_count") or 1),
        vram_gb=int(entry.get("vram_gb") or 0),
        cpu_cores=int(entry.get("cpu_cores") or 0),
        memory_gb=int(entry.get("memory_gb") or 0),
        hourly_price_cny=float(entry.get("hourly_price_cny") or 0),
        sm=str(entry.get("sm") or ""),
        gen=str(entry.get("gen") or ""),
        bf16=entry.get("bf16"),
        driver_may_lag=bool(entry.get("driver_may_lag")),
        target=str(entry.get("target") or ""),
    )


def _sm_ge(have: str, need: str) -> bool:
    """sm 版本比较。拿不到卡的 sm 时**判不满足**——

    宁可少推荐一个能用的，也不要推荐一个起不来的：前者用户会说"怎么这么少"，
    后者用户会在环境配到一半时炸掉，而且看不出原因。
    """
    if not need:
        return True
    if not have:
        return False
    try:
        return tuple(int(x) for x in have.split(".")) >= tuple(int(x) for x in need.split("."))
    except ValueError:
        return False


@dataclass
class TierMatch:
    tier: Tier
    options: list[SkuOption]
    rejected: dict[str, int]     # 原因 → 被刷掉的数量，用于解释"为什么没有更便宜的"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.to_dict(),
            "options": [o.to_dict() for o in self.options],
            "rejected": self.rejected,
        }


def match_tier(
    tier: Tier,
    catalog: list[dict[str, Any]],
    *,
    min_compute_capability: str = "",
    min_host_ram_gib: int = 0,
    limit: int = 3,
) -> TierMatch:
    """给一档挑候选 SKU。返回按每 GiB 单价排序的前几个 + 落选统计。"""
    rejected = {"显存不足": 0, "架构代际不够": 0, "内存不足": 0, "卡数不足": 0, "无价格": 0}
    kept: list[SkuOption] = []

    for entry in catalog:
        opt = option_from_catalog(entry)
        if opt.accelerator_count < tier.gpu_count:
            rejected["卡数不足"] += 1
            continue
        if opt.total_vram_gb < tier.vram_gib:
            rejected["显存不足"] += 1
            continue
        if not _sm_ge(opt.sm, min_compute_capability):
            rejected["架构代际不够"] += 1
            continue
        if min_host_ram_gib and opt.memory_gb < min_host_ram_gib:
            rejected["内存不足"] += 1
            continue
        if not opt.hourly_price_cny:
            rejected["无价格"] += 1
            continue
        kept.append(opt)

    # 档内已都满足需求 → 按绝对价格排；同价取显存冗余小的（别为用不上的显存付钱）。
    # 每 GiB 单价只作为展示指标，见模块 docstring。
    kept.sort(key=lambda o: (o.hourly_price_cny, o.total_vram_gb - tier.vram_gib))
    return TierMatch(tier=tier, options=kept[:limit],
                     rejected={k: v for k, v in rejected.items() if v})


def match_plan(
    tiers: list[Tier],
    catalog: list[dict[str, Any]],
    *,
    min_compute_capability: str = "",
    min_host_ram_gib: int = 0,
) -> list[TierMatch]:
    return [
        match_tier(t, catalog, min_compute_capability=min_compute_capability,
                   min_host_ram_gib=min_host_ram_gib)
        for t in tiers
    ]

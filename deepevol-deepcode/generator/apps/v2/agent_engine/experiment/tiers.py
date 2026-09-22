"""把 ComputeSpec 折成给用户看的几档配置。

## 档位数量本身是置信度信号

不是固定输出三档。标准 HF 微调仓库读得准，出两档就够；硬塞第三档只会让用户
觉得系统在乱猜。反过来，`confidence == low` 或长任务时**必须**出保险档，
因为那时区间是真的宽。

## 两根正交的轴，成本是读数不是轴

风险轴由 `vram` 区间的分位决定，速度轴是另一回事（换更强的卡 / 更多卡）。
把「便宜/够用/快」并成一列会让用户没法比较——他要的是"我要冒多大风险"和
"我要多快"两个独立选择。

## 速度档只给相对倍数

静态估 FLOPs 尚可，估**实际吞吐**要牵扯 kernel 效率、数据管道瓶颈、通信开销。
说"预计快 2–3 倍"是诚实的，说"预计 3.7 小时"是在编。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .compute_spec import ComputeSpec

#: 阿里云 GPU 显存的实际档位。估到 45 还是 55 不重要，**别跨档**才重要。
VRAM_STEPS = (8, 16, 24, 32, 48, 64, 80, 96, 141)


def round_up_to_step(gib: float) -> int:
    for step in VRAM_STEPS:
        if gib <= step:
            return step
    return int(VRAM_STEPS[-1])


@dataclass
class Tier:
    key: str                     # economy | standard | insurance | speed
    label: str
    vram_gib: int
    gpu_count: int = 1
    #: 给用户看的一句话，说明这档意味着什么
    blurb: str = ""
    #: 只有加速档有；相对基准档的倍数区间，如 (2, 3)
    speedup: tuple[float, float] | None = None
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = {
            "key": self.key, "label": self.label, "vram_gib": self.vram_gib,
            "gpu_count": self.gpu_count, "blurb": self.blurb, "rationale": self.rationale,
        }
        if self.speedup:
            out["speedup"] = list(self.speedup)
        return out


@dataclass
class TierPlan:
    tiers: list[Tier] = field(default_factory=list)
    default_key: str = "standard"
    #: 抢占式实例只有代码支持断点续跑时才该出现
    spot_allowed: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tiers": [t.to_dict() for t in self.tiers],
            "default": self.default_key,
            "spot_allowed": self.spot_allowed,
            "notes": self.notes,
        }


def build_tiers(spec: ComputeSpec, *, long_running: bool = False) -> TierPlan:
    plan = TierPlan()
    if spec.vram is None:
        plan.notes.append("没有显存估计，无法生成档位")
        return plan

    low, high = spec.vram.low, spec.vram.high

    # 经济档：区间下沿 + 一点裕度。可能要减 batch 或开梯度累积。
    economy_raw = low + max(4.0, low * 0.15)
    # 稳妥档：区间上沿。按代码里的配置直接跑，不用改。
    standard_raw = high

    economy = round_up_to_step(economy_raw)
    standard = round_up_to_step(standard_raw)

    # ★ 实测之后，任何一档都不许低于实测下界。
    #   我们已经知道那个规格会 OOM——把它作为选项摆出来，等于让用户再炸一次，
    #   而第二次比第一次更让人恼火（环境已经配过、时间已经花过）。
    if spec.vram_measured:
        floor = round_up_to_step(low)
        economy = max(economy, floor)
        standard = max(standard, floor)
        plan.notes.append(
            f"显存需求来自**实测**（一次 CUDA OOM），已知至少要 {low:.1f}G，"
            f"所有档位不低于 {floor}G。注意这是下界——后续阶段可能有更大峰值"
        )

    plan.tiers.append(Tier(
        key="economy", label="经济档", vram_gib=economy,
        blurb="能跑完，但可能要减 batch 或开梯度累积",
        rationale=f"区间下沿 {low:.1f}G + 15% 裕度 → 向上取到 {economy}G",
    ))
    if standard > economy:
        plan.tiers.append(Tier(
            key="standard", label="稳妥档", vram_gib=standard,
            blurb="按代码里的配置直接跑，不用改",
            rationale=f"区间上沿 {high:.1f}G → 向上取到 {standard}G",
        ))
    else:
        # 区间窄到两档落在同一个 SKU 档上——那就只有一档，说清楚为什么。
        plan.tiers[0].key = "standard"
        plan.tiers[0].label = "稳妥档"
        plan.tiers[0].blurb = "按代码里的配置直接跑"
        plan.notes.append(f"估计区间 {low:.1f}–{high:.1f}G 落在同一档，只给一个选项")

    # 保险档**条件出现**：档位数量本身就是置信度信号
    if spec.confidence == "low" or long_running:
        insurance = round_up_to_step(max(high * 2, standard + 1))
        why = "静态估计置信度低" if spec.confidence == "low" else "任务预计运行时间长"
        plan.tiers.append(Tier(
            key="insurance", label="保险档", vram_gib=insurance,
            blurb=f"{why}，这档留足余量",
            rationale=(f"{why}；未确定项 {len(spec.unknowns)} 个 → 按上沿 2× 取 {insurance}G"),
        ))

    # 加速档：另一根轴。只在代码支持多卡时给——不支持还推多卡就是挖坑。
    if spec.multi_gpu:
        plan.tiers.append(Tier(
            key="speed", label="加速档", vram_gib=standard, gpu_count=2,
            blurb="多卡并行，预计快 1.5–2 倍",
            speedup=(1.5, 2.0),
            rationale=f"代码支持 {'/'.join(spec.parallel)}；只给相对倍数，"
                      "绝对耗时受 kernel 效率与数据管道影响，静态估不准",
        ))

    plan.default_key = "standard" if any(t.key == "standard" for t in plan.tiers) else "economy"

    # 抢占式：便宜 60–80%，但会被回收。不支持断点续跑还推它就是挖坑。
    plan.spot_allowed = spec.resumable
    if spec.resumable:
        plan.notes.append("代码支持断点续跑，可考虑抢占式实例（便宜 60–80%，但可能被回收）")
    else:
        plan.notes.append("未检测到断点续跑，不建议抢占式实例——被回收就得从头再来")

    if spec.min_compute_capability:
        plan.notes.append(
            f"架构下限 sm_{spec.min_compute_capability}：这是硬门槛，"
            "低于它的卡不是慢，是跑不起来"
        )
    return plan

"""把「一个仓库」变成「几档能点的配置」。

这是实验 Agent 首轮的产物：用户给一个 GitHub 链接，我们**在租机器之前**
读代码估出算力需求，再在阿里云真的有货的规格里挑几档给他选。

## 为什么静态分析这一步值得做

租机器之前唯一能知道的就是代码本身。P2 的命中率验收给出 5/10 ——
听起来不高，但这一层的目标不是「估准」，是**给一个合理起点**：
估高了多花钱（线性、有界），估低了会 OOM，而 OOM 现在有 P4 的
「实测 → 升级」闭环兜底（`oom.py` + `upgrade_plan.py`）。
两侧代价不对称，所以估计整体上偏保守，剩下的交给实测修正。

## 一条不能省的规矩：档位数量本身是置信度信号

不是固定输出三档。读得准就出两档；`confidence == low` 时**必须**出保险档，
因为那时区间是真的宽。硬凑三档会让用户以为系统很确定。见 `tiers.py`。
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .catalog import SkuOption, TierMatch, match_plan, option_from_catalog
from .compute_spec import DEFAULT_HOST_RAM_FLOOR_GIB, ComputeSpec, build_compute_spec
from .cpu_tiers import build_cpu_tiers, cpu_tier_requirements, filter_cpu_options
from .preflight_catalog import catalog_from_preflight, last_dropped_non_linux
from .repo_facts import analyse_resources
from .tiers import TierPlan, build_tiers


@dataclass
class Recommendation:
    """给用户看的一整份推荐。`blocked_reason` 非空时 `choices` 必然为空。"""

    spec: ComputeSpec
    plan: TierPlan
    matches: list[TierMatch] = field(default_factory=list)
    blocked_reason: str = ""
    # Set when the code wanted a GPU, none was rentable, and CPU tiers are
    # offered as a degraded choice instead.
    gpu_fallback_reason: str = ""

    @property
    def choices(self) -> list[tuple[str, SkuOption]]:
        """(档位 key, 该档最便宜的候选)。这是卡片上真正可点的东西。

        ★ **同一台机器只出现一次。** 档位是按显存需求分的，
          但目录里未必有对应粒度的规格 —— 实测 nanoGPT：
          经济档要 8G、稳妥档要 16G，而有货的最小规格是 T4 16G，
          于是两档落到同一台机器，报告里渲染出两行一模一样的候选。
          用户分不出差别，看起来像 bug。

          去重时保留**更保守**的那一档（后出现的），因为那才是它实际能给的体验：
          最便宜的能满足经济档的机器同时也满足稳妥档，用户是按经济档的价格
          拿到了稳妥档的余量。说成经济档反而低估了它。
        """
        best: dict[tuple[str, str], str] = {}
        options: dict[tuple[str, str], SkuOption] = {}
        order: list[tuple[str, str]] = []
        for m in self.matches:
            if not m.options:
                continue
            option = m.options[0]
            ident = (option.target, option.instance_type)
            if ident not in options:
                order.append(ident)
                options[ident] = option
            best[ident] = m.tier.key      # 后写的覆盖前写的 = 保留更保守的档
        return [(best[i], options[i]) for i in order]

    @property
    def collapsed_tiers(self) -> bool:
        """是否有多档落到了同一台机器。渲染时要说一句，否则用户会以为
        我们只给了一个选择、或者以为系统偷懒。"""
        matched = sum(1 for m in self.matches if m.options)
        return matched > len(self.choices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "plan": self.plan.to_dict(),
            "matches": [m.to_dict() for m in self.matches],
            "blocked_reason": self.blocked_reason,
            "gpu_fallback_reason": self.gpu_fallback_reason,
        }


_CPU_PINNED_PATTERNS = (
    r"--device[=\s]+cpu\b",
    r"\bdevice\s*[=:]\s*['\"]?cpu\b",
    r"\bcpu[\s-]*only\b",
    r"\bcpu\s*版",
    r"在\s*cpu\s*上",
    r"用\s*cpu\s*(跑|训练|运行|推理)",
    r"(不用|不需要|无需|没有)\s*(gpu|显卡)",
)


def instruction_pins_cpu(instruction: str) -> bool:
    """The user said CPU in so many words (``--device=cpu``, 在 CPU 上跑, CPU 版 …).

    Static analysis only sees what the code *can* do; a repository that
    defaults to CUDA still runs on a CPU when the user passes the flag, and
    then a missing GPU catalog must not end the conversation.
    """
    text = str(instruction or "").lower()
    return any(re.search(pattern, text) for pattern in _CPU_PINNED_PATTERNS)


def recommend_from_repo(
    repo_path: str | Path,
    *,
    cloud_options: list[dict[str, Any]],
    entry_first: str = "",
    long_running: bool = False,
    limit: int = 3,
    instruction: str = "",
) -> Recommendation:
    """仓库目录 + preflight 候选 → 推荐。

    `cloud_options` 直接来自 `POST .../remote-compute/preflight` 的返回。
    **不要**在这里自己去查阿里云 —— 见 `preflight_catalog` 模块 docstring：
    target 对不上，开机时会缺 image_id。
    """
    facts = analyse_resources(repo_path, entry_first=entry_first)
    spec = build_compute_spec(facts)
    if spec.needs_gpu and instruction_pins_cpu(instruction):
        spec.needs_gpu = False
        spec.notes.append("任务说明明确指定用 CPU，按 CPU 选型（代码本身能用 GPU）")
    plan = build_tiers(spec, long_running=long_running)

    if not spec.needs_gpu:
        # ★ 不需要 GPU 的仓库**也要有机器可选**。
        #   先前这里只回一句「用 CPU 机器就能跑」然后一个候选都不给 ——
        #   等于告诉用户「你能跑」再把门关上，他无路可走。
        #   CPU 走自己那套档位（约束轴不同，见 cpu_tiers 模块 docstring）。
        return _recommend_cpu(spec, cloud_options=cloud_options, limit=limit)

    catalog = catalog_from_preflight(cloud_options)
    if not catalog:
        return _cpu_fallback(spec, plan, cloud_options=cloud_options, limit=limit,
                             reason=_no_catalog_reason("GPU"))

    matches = match_plan(
        plan.tiers, catalog,
        min_compute_capability=spec.min_compute_capability,
        min_host_ram_gib=_enforceable_host_ram(spec),
    )
    rec = Recommendation(spec=spec, plan=plan, matches=matches)
    if not rec.choices:
        worst = _dominant_rejection(matches)
        return _cpu_fallback(
            spec, plan, cloud_options=cloud_options, limit=limit,
            reason="有货的规格里没有能满足这份代码的"
            + (f"（最主要的落选原因：{worst}）" if worst else ""),
        )
    return rec


def _cpu_fallback(
    spec: ComputeSpec, plan: Any, *, cloud_options: list[dict[str, Any]], limit: int, reason: str
) -> Recommendation:
    """No rentable GPU: offer CPU machines rather than close the door.

    The user decides whether a slow (or impossible) CPU run is worth trying;
    the card says plainly that the code wanted a GPU and why none was offered.
    """
    cpu = _recommend_cpu(spec, cloud_options=cloud_options, limit=limit)
    if cpu.blocked_reason:
        return Recommendation(spec=spec, plan=plan, blocked_reason=reason)
    cpu.gpu_fallback_reason = reason
    return cpu


def _no_catalog_reason(kind: str) -> str:
    """没有候选时说清楚是**哪一关**卡住的。

    「没有可用机器」和「有机器但都是 Windows」对用户意味着完全不同的下一步：
    前者是等或换地域，后者是这条路根本走不通。
    """
    dropped = last_dropped_non_linux()
    if dropped:
        return (
            f"当前地域有 {len(dropped)} 个 {kind} 规格，但**都不是 Linux/SSH** "
            f"（比如 {dropped[0]}）。实验要在远端跑 Docker 容器，Windows 机器上起不来"
        )
    return (
        f"当前地域没有**立刻可用**的 {kind} 规格。"
        "「可用」同时要求有货、有价、且镜像预装了 CloudMonitor —— "
        "缺任何一件都开不起来"
    )


def _recommend_cpu(
    spec: ComputeSpec, *, cloud_options: list[dict[str, Any]], limit: int
) -> Recommendation:
    """不需要 GPU 时的选型。核数与内存说了算，没有显存这根硬轴。"""
    plan = build_cpu_tiers(spec)
    catalog = catalog_from_preflight(cloud_options, kind="cpu")
    rec = Recommendation(spec=spec, plan=plan)
    if not catalog:
        rec.blocked_reason = _no_catalog_reason("CPU")
        return rec

    for tier in plan.tiers:
        requirements = cpu_tier_requirements(tier, spec)
        kept = filter_cpu_options(catalog, requirements)
        rejected = {} if kept else {"核数或内存不足": len(catalog)}
        rec.matches.append(TierMatch(
            tier=tier,
            options=[option_from_catalog(row) for row in kept[:limit]],
            rejected=rejected,
        ))
    if not rec.choices:
        rec.blocked_reason = "有货的 CPU 规格里没有满足要求的（核数或内存不够）"
    return rec


def _enforceable_host_ram(spec: ComputeSpec) -> int:
    """只有**有证据**支撑的主机内存下限才拿来做硬过滤。

    ★ 这条是踩出来的。`compute_spec` 里 `base_ram = max(16.0, ...)` ——
      16 GiB 是恒定下限，参数量未知时它就等于 16，**不携带任何信息**。
      我一开始把它原样当硬门槛传下去，结果一台 15GB 内存的 T4
      （7.96 元/时）被判「内存不足」刷掉，用户被推去买 11.2 元/时的 A10 ——
      每小时多付 40%，理由却只是「我们默认假设要 16GB」。

      内存 15 与 16 的差别不构成「跑不起来」；显存和架构代际才构成。
      所以：高于下限时说明真的有东西把它顶上去了（参数量大、
      或者检测到多个 DataLoader worker），那时才过滤。
    """
    ram = float(spec.host_ram_gib or 0)
    return int(ram) if ram > DEFAULT_HOST_RAM_FLOOR_GIB else 0


def _dominant_rejection(matches: list[TierMatch]) -> str:
    """把各档的落选统计合起来，找出最主要的那条原因。

    「没有合适的」帮不上忙；「都卡在架构代际不够」用户能据此判断
    是换地域还是改代码 —— 这两个动作完全不同。
    """
    totals: dict[str, int] = {}
    for m in matches:
        for reason, count in (m.rejected or {}).items():
            totals[reason] = totals.get(reason, 0) + count
    return max(totals.items(), key=lambda kv: kv[1])[0] if totals else ""

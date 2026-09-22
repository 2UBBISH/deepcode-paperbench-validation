"""OOM 之后：把实测需求变成一个**能给用户点的**升级提案。

## 为什么要问用户，而不是自动升级

钱是用户的。自动升到一台更贵的机器，哪怕技术上对，也是在替用户做花钱的决定。
更要命的是第二个事实：**没有断点续跑机制的仓库，升级后训练要从头开始**。
自动升级会让用户在毫不知情的情况下，为「重头再跑一遍」付两份钱。

所以这一层的产物是一张卡，不是一个动作。

## 三条必须出现在卡上的信息

1. **实测需求是下界，不是精确值**。OOM 只说明「至少还差这么多」——
   报错发生时后续阶段（eval、更长的序列）根本还没跑到。把下界当精确值用，
   会在升级一次之后又 OOM 一次，而第二次比第一次更让人恼火。
2. **差价**，不是新价格。用户脑子里的锚点是现在这台多少钱。
3. **`resumable` 警告**，且只在真的不可续跑时显示。
   永远显示等于没显示——有 checkpoint 的用户会学会忽略它。

## 为什么候选是「实测下界之上最便宜的」而不是「翻倍」

翻倍是个偷懒的规则，在 24G→48G 时碰巧对，在 80G→160G 时会推荐一台
用户根本不需要的 8 卡机。既然已经有实测下界和目录，就按目录选：
满足下界的里面最便宜的那个。这与首次选型用的是同一条排序规则
（`catalog.match_tier`），不另立一套。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .catalog import SkuOption, match_tier
from .compute_spec import ComputeSpec
from .oom import OomFact, required_vram_gib
from .tiers import Tier, round_up_to_step


@dataclass
class UpgradeCandidate:
    option: SkuOption
    hourly_delta_cny: float

    def to_dict(self) -> dict[str, Any]:
        return {**self.option.to_dict(), "hourly_delta_cny": round(self.hourly_delta_cny, 4)}


@dataclass
class UpgradeProposal:
    """给用户看的东西。`candidates` 为空时 `blocked_reason` 必须说明为什么。"""

    measured_floor_gib: float
    current_vram_gib: float
    candidates: list[UpgradeCandidate] = field(default_factory=list)
    resumable: bool = False
    blocked_reason: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def recommended(self) -> UpgradeCandidate | None:
        return self.candidates[0] if self.candidates else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "measured_floor_gib": round(self.measured_floor_gib, 2),
            "current_vram_gib": round(self.current_vram_gib, 2),
            "candidates": [c.to_dict() for c in self.candidates],
            "resumable": self.resumable,
            "blocked_reason": self.blocked_reason,
            "notes": self.notes,
        }


def build_upgrade_proposal(
    oom: OomFact,
    *,
    spec: ComputeSpec,
    current: SkuOption,
    catalog: list[dict[str, Any]],
    limit: int = 3,
    env_saved: bool = False,
) -> UpgradeProposal:
    """按实测下界在目录里挑升级候选。

    ★ 用 `match_tier` 而不是自己写筛选：架构下限（sm）、主机内存、卡数、
      「拿不到 sm 就判不满足」这些判断已经在那里被想过一遍了，
      在这里重写一份意味着两份逻辑会慢慢分叉。
    """
    floor = required_vram_gib(oom)
    current_total = float(current.total_vram_gb or oom.total_capacity_gib or 0.0)

    proposal = UpgradeProposal(
        measured_floor_gib=floor,
        current_vram_gib=current_total,
        resumable=bool(spec.resumable),
    )
    proposal.notes.append(
        f"实测：在 {oom.total_capacity_gib:.1f}G 的卡上分配 "
        f"{oom.tried_to_allocate_gib:.2f}G 时 OOM，已用 {oom.in_use_gib:.1f}G"
    )
    proposal.notes.append(
        f"因此**至少**需要 {floor:.1f}G。这是下界不是精确值——"
        "报错发生时后续阶段还没跑到，真实峰值只会更高"
    )

    # 借一档「就按实测下界」的 Tier 去查目录，复用既有的筛选与排序。
    #
    # ★ `gpu_count` 固定 1，而且下面还要**再按单卡显存过一遍**。
    #   `match_tier` 比的是 `total_vram_gb`（= 单卡 × 卡数），那对首次选型是合理的；
    #   但 OOM 给的是**单卡**缺口 —— 两张 16G 卡的总显存是 32G，
    #   可那个在 16G 上装不下的张量，在任何一张 16G 卡上仍然装不下，
    #   除非代码真的做了张量切分，而那不是换机器能变出来的。
    #   只按总量筛会推荐一台照样会 OOM 的多卡机，
    #   且用户为多出来的卡付了钱 —— 比不升级更糟。
    probe = Tier(
        key="measured", label="实测档", vram_gib=round_up_to_step(floor), gpu_count=1,
    )
    matched = match_tier(
        probe, catalog,
        min_compute_capability=spec.min_compute_capability,
        min_host_ram_gib=int(spec.host_ram_gib or 0),
        limit=limit + 2,
    )

    current_per_card = float(current.vram_gb or oom.total_capacity_gib or 0.0)
    for opt in matched.options:
        # 单卡显存必须过实测下界。见上面 probe 的注释：总量够不代表单卡够。
        if opt.vram_gb < floor:
            continue
        # 单卡显存没变大的不是升级 —— 换一台一样会 OOM 的机器纯属浪费一轮。
        if opt.vram_gb <= current_per_card:
            continue
        proposal.candidates.append(UpgradeCandidate(
            option=opt,
            hourly_delta_cny=opt.hourly_price_cny - current.hourly_price_cny,
        ))
        if len(proposal.candidates) >= limit:
            break

    if not proposal.candidates:
        worst = max(matched.rejected.items(), key=lambda kv: kv[1])[0] if matched.rejected else ""
        proposal.blocked_reason = (
            f"目录里没有**单卡**显存过 {floor:.1f}G 且满足其他约束的规格"
            + (f"（最主要的落选原因：{worst}）" if worst else "")
        )
        proposal.notes.append(
            "多卡不能替代大卡：总显存够、单卡不够时，那个装不下的张量照样装不下"
        )
        return proposal

    if not spec.resumable:
        # ★ 只在真不可续跑时说。永远显示的警告等于没有警告。
        # ★ 「环境不用重配」只有**镜像真打成了**才成立。
        #   真机上见过一次:快照跑到 36% 被一次瞬时 SSL 断连判死,卡片照旧说
        #   「环境不用重配」—— 而用户实际要再花 46 分钟重配。
        #   在花钱的卡片上说错这件事，等于骗他低估成本。
        env_line = (
            "环境已经存成镜像，不用重配"
            if env_saved
            else "**环境也没保存下来**，换机后要重新配一遍"
        )
        proposal.notes.append(
            "⚠️ 没有检测到断点续跑（checkpoint/resume）机制："
            "换机之后训练要**从头开始**，之前跑掉的时间收不回来。"
            f"{env_line}，训练进度也会丢"
        )
    else:
        proposal.notes.append(
            "检测到断点续跑机制，换机后可以从最近的 checkpoint 接着跑"
            + ("" if env_saved else "；但**环境没保存下来**，要重新配一遍")
        )

    return proposal


# -- 换机策略：打镜像 还是 重开重配 ------------------------------------------
#
# P4 Step 0 实测（香港，40G ESSD PL1 系统盘，已用约 5–7GB）：
#
#     停机        13s
#     CreateImage 2038s = 34.0 分钟
#     镜像开机    13s      ← 与公共镜像开机同速，没有惩罚
#
# 34 分钟不是一个可以忽略的开销。而 P0c 实测 makemore 配一次环境只要 6 分钟
# （外加约 4 元 LLM）。也就是说：
#
#   - 简单仓库：重开重配 ~6 分钟 完胜 打镜像 ~35 分钟
#   - 重仓库（conda + CUDA + 下数据集）：配环境可能半小时以上，打镜像才划算
#
# 静态钉死一种策略在两头都会错。而**做决定的那一刻，我们手里正好有
# 本次环境配置的实际耗时** —— 它刚刚发生过。用它选，比拍脑袋准得多。

#: 环境配置耗时超过这个值就打镜像。
#:
#: 取值低于实测的 34 分钟是**故意**的，因为两边的代价不对称：
#: 打镜像是无人值守的等待，重配环境要花 LLM 的钱、而且是**非确定的** ——
#: RSA 第二次可能落到 needs_user，那要打断人，比多等二十分钟贵得多。
IMAGE_WORTH_IT_SECONDS = 900.0

#: CreateImage 的实测样本：(系统盘 GB, 盘上内容, 耗时秒)。
#:
#: ★ 上面那条「容量还是已用量」的疑问已经验掉了（2026-08-28，香港，10 秒粒度）：
#:   **容量是主因，已用量几乎不影响**。同为公共 Ubuntu 最小系统，
#:   20G 用 31.2 分钟、40G 用 50.1 分钟；而同为 40G，装了驱动+Docker 的
#:   自建镜像 60.7 分钟 vs 最小系统 50.1 分钟 —— 那 20% 的差还落在
#:   阿里云自身的波动范围内（同一份 40G 盘实测跨度 34~81 分钟）。
#:
#: ★ 所以「把盘调小」的收益比原先推测的小得多：40G→20G 只省 38%，
#:   而且 20G 装不下我们的基础镜像（CreateInstance 直接报
#:   `InvalidSystemDiskSize.LessThanImageSize`）。不值得为此改盘。
#:
#: ★ 也验掉了「停机变配」这条路：`gn7i` 全族 13 款规格单卡显存清一色 24G，
#:   变配限同族、只能加卡数 —— 而 OOM 要的是一次大额**连续**分配装得下，
#:   八张 24G 的卡一张也放不下它。变配解决算力不足，解决不了单卡显存不足。
_CREATE_IMAGE_SAMPLES: tuple[tuple[float, str, float], ...] = (
    (40.0, "自建基础镜像（P4 Step 0）", 2038.0),
    (40.0, "实验机（含 torch）", 2340.0),
    (40.0, "公共 Ubuntu 最小系统", 3006.0),
    (40.0, "自建基础镜像", 3642.0),
    (40.0, "实验机（含 torch）", 4860.0),
    (20.0, "公共 Ubuntu 最小系统", 1872.0),
)

#: 对容量做一次线性拟合：`秒 = 截距 + 斜率 × 盘GB`。
#: 只用「同镜像、只差容量」的那两个点定斜率 —— 其余样本的镜像内容不同，
#: 混进去拟合等于把噪声当信号。
_FIT_SLOPE_SECONDS_PER_GB = (3006.0 - 1872.0) / (40.0 - 20.0)   # 56.7 秒/GB
_FIT_INTERCEPT_SECONDS = 1872.0 - 20.0 * _FIT_SLOPE_SECONDS_PER_GB   # 738 秒

#: 实测离散度：同一份 40G 盘跨度 34~81 分钟，相对中位数约 0.7×~1.6×。
#: 给区间而不是给点值 —— 一个偏乐观两倍的「约 34 分钟」比不给还糟。
# 同容量下的离散度**比容量本身的影响还大**（40G 实测 34–81 分钟），
# 所以区间不是拍出来的裕度，而是直接从 40G 那组样本量出来的：
# 拍一个好看的 ±X% 只会像这次一样刚好压不住最慢的那次。
#: 右删失观测：知道它**至少**要多久，但没观测到它什么时候完成。
#:
#: 2026-08-29 真机：40G 系统盘 78 分钟时进度 36%，之后我们的轮询被一次瞬时
#: 查询失败掐断（见 lease_api 的重试），但镜像最终 **Available/100%** ——
#: 阿里云不返回完成时间，所以真实总时长永远拿不到了。
#:
#: 这里**不拿当时的速率线性外推**（那正是 eta_from_progress 里被真机打脸的做法，
#: 会得出 217 分钟这种没有依据的数）。取一个能站住的下界：剩下的 64% 即便按
#: 历史见过的**最快**速率（2%/分钟）跑，也还要 32 分钟 —— 所以总时长 ≥ 110 分钟。
#: 这一条足以证伪「上界 81 分钟」，而 `_SNAPSHOT_DEADLINE` 当初正是照那个上界
#: 拍的 4800 秒，于是慢一点的快照注定超时。
_CENSORED_FLOOR_SECONDS: tuple[tuple[float, float], ...] = ((40.0, 6600.0),)

_40G = tuple(sec for gib, _what, sec in _CREATE_IMAGE_SAMPLES if gib == 40.0)
_40G_MID = _FIT_INTERCEPT_SECONDS + _FIT_SLOPE_SECONDS_PER_GB * 40.0
_SPREAD_LOW = min(_40G) / _40G_MID      # 0.68
_SPREAD_HIGH = max(
    max(_40G),
    max(sec for gib, sec in _CENSORED_FLOOR_SECONDS if gib == 40.0),
) / _40G_MID

#: 兼容旧调用方：40G 的中位数估计。
MEASURED_CREATE_IMAGE_SECONDS = 2038.0


def estimate_create_image_seconds(disk_gb: float = 40.0) -> tuple[float, float, float]:
    """按系统盘容量估 CreateImage 耗时，返回 (下界, 中位, 上界) 秒。

    拟合自 2026-08-28 的实测样本（见 `_CREATE_IMAGE_SAMPLES`）。
    区间宽是**如实反映**阿里云侧的波动，不是模型不好 ——
    同一份盘实测就能差一倍，给个精确到分钟的点值是假精确。
    """
    mid = _FIT_INTERCEPT_SECONDS + _FIT_SLOPE_SECONDS_PER_GB * max(0.0, disk_gb)
    return mid * _SPREAD_LOW, mid, mid * _SPREAD_HIGH


def format_wait_estimate(disk_gb: float = 40.0) -> str:
    """给用户看的等待预期，例如「约 35–80 分钟」。"""
    low, _mid, high = estimate_create_image_seconds(disk_gb)
    return f"约 {low/60:.0f}–{high/60:.0f} 分钟"


def eta_from_progress(
    elapsed_seconds: float, progress_percent: float, disk_gb: float = 40.0
) -> float | None:
    """用**已经跑出来的进度**外推剩余秒数。比任何静态估计都准。

    ★ 前 10% 不外推：实测进度曲线前段明显慢于后段
      （40G 那次前 10 分钟只走 13%，后段到了 2%/分钟），
      拿前段速率外推会把总时长高估一倍，反而比静态估计更误导人。
    """
    if progress_percent < 10 or progress_percent >= 100 or elapsed_seconds <= 0:
        return None
    total = elapsed_seconds * 100.0 / progress_percent
    # ★ 用实测上界夹住外推。真机实测过一次 28.5 分钟才走到 10%，朴素外推报出
    #   「还要 257 分钟」—— 比静态区间(34–81)还离谱,比不报更糟。
    #   6 次真实完成里最慢的一次也没超过 high,所以任何超过它的外推都是曲线前段
    #   在骗人,不是这次真的要跑那么久。
    _low, _mid, high = estimate_create_image_seconds(disk_gb)
    total = min(total, high)
    if total <= elapsed_seconds:
        # 已经跑过历史最慢的一次了 —— 老实说不知道,别编一个「还要 0 分钟」。
        return None
    return total - elapsed_seconds


@dataclass
class UpgradeStrategy:
    use_image: bool
    reason: str
    estimated_wait_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_image": self.use_image,
            "reason": self.reason,
            "estimated_wait_seconds": round(self.estimated_wait_seconds),
        }


def choose_upgrade_strategy(
    *,
    env_setup_seconds: float | None,
    threshold_seconds: float = IMAGE_WORTH_IT_SECONDS,
) -> UpgradeStrategy:
    """用**本次**环境配置的实际耗时，决定升级时打镜像还是重开重配。

    `env_setup_seconds` 为 None 表示没测到（比如复用了已有环境）——
    那时打镜像，因为「不知道要多久」的重配是个无界的赌注，
    而打镜像的耗时是已知的。

    ★ 这里**不看 `resumable`**。断点续跑管的是训练进度要不要从头，
      和环境要不要重配是两件独立的事：镜像保住环境但保不住显存里的训练状态，
      checkpoint 保住训练状态但和换不换镜像无关。混在一起想会两头都错。
    """
    if env_setup_seconds is None:
        return UpgradeStrategy(
            use_image=True,
            reason="没有本次环境配置耗时可参考；重配是无界赌注，打镜像至少耗时已知",
            estimated_wait_seconds=estimate_create_image_seconds()[1],
        )
    if env_setup_seconds >= threshold_seconds:
        return UpgradeStrategy(
            use_image=True,
            reason=(f"本次配环境花了 {env_setup_seconds/60:.0f} 分钟，"
                    f"重配一遍不比打镜像（{format_wait_estimate()}）划算，"
                    "且重配要再花一次 LLM 的钱、还可能中途要人介入"),
            estimated_wait_seconds=estimate_create_image_seconds()[1],
        )
    return UpgradeStrategy(
        use_image=False,
        reason=(f"本次配环境只花了 {env_setup_seconds/60:.0f} 分钟，"
                f"比打镜像（{format_wait_estimate()}）快得多，"
                "直接在新机上重配"),
        estimated_wait_seconds=env_setup_seconds,
    )

"""从 PyTorch 的 OOM 报错里读出**实测**显存需求。

## 为什么解析报错而不是先跑探针

原设计是「先在便宜机器上跑 micro-batch 探针」。实测数据推翻了它：环境配置才是
大头（P0c 里 makemore 光装 torch 就 6 分钟），独立探针机意味着**环境配两遍**，
而 OOM 重试的成本同样是一次环境配置——两者几乎打平。

而 OOM 本来就发生在头几分钟：显存峰值在第一次 forward+backward 加上优化器状态
分配时就到了。所以「跑起来等它炸」和「先跑探针」在时间上差不多，
但前者**零 vendored 改动**——不用往 `rsa/pipeline.py` 里插步骤。

## 这个数是下界，不是精确值

OOM 只告诉你「在那一刻还差多少」，不告诉你整个训练的峰值。后面还可能有更大的
分配（eval 阶段的大 batch、更长的序列）。所以：

- 所有对外展示都必须标明是**下界**
- `required_vram_gib()` 会乘一个安全系数覆盖后续峰值，这个系数是**假设不是测量**

把它当精确值用，会在升级一次之后又 OOM 一次——那比第一次更让人恼火。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: OOM 那一刻不一定是全程峰值（eval 阶段的大 batch、更长序列都可能更高）。
#: 这个系数是**假设**，用来覆盖后续峰值，不是测出来的。
PEAK_SAFETY_FACTOR = 1.25

_UNITS = {"kib": 1 / 1024 ** 2, "mib": 1 / 1024, "gib": 1.0, "tib": 1024.0,
          "kb": 1 / 1024 ** 2, "mb": 1 / 1024, "gb": 1.0, "tb": 1024.0,
          "bytes": 1 / 1024 ** 3}


def _to_gib(value: str, unit: str) -> float:
    return float(value) * _UNITS.get(unit.strip().lower(), 0.0)


_NUM = r"(\d+(?:\.\d+)?)\s*(KiB|MiB|GiB|TiB|KB|MB|GB|TB|bytes)"

# PyTorch 的措辞跨版本改过好几次，逐个匹配而不是指望一条正则通吃。
_TRIED_RE = re.compile(rf"Tried to allocate\s+{_NUM}", re.I)
_DEVICE_RE = re.compile(r"GPU\s+(\d+)", re.I)

# 2.x：「GPU 0 has a total capacity of 15.99 GiB of which 1.50 GiB is free」
_CAPACITY_RE = re.compile(rf"total capacity of\s+{_NUM}", re.I)
_FREE_NEW_RE = re.compile(rf"of which\s+{_NUM}\s+is free", re.I)

# 1.x：「(GPU 0; 10.76 GiB total capacity; 9.71 GiB already allocated; 5.56 MiB free; ...)」
_CAPACITY_OLD_RE = re.compile(rf"{_NUM}\s+total capacity", re.I)
_FREE_OLD_RE = re.compile(rf"{_NUM}\s+free", re.I)
_ALLOCATED_OLD_RE = re.compile(rf"{_NUM}\s+already allocated", re.I)

_OOM_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "hip out of memory",          # ROCm
    "out of memory",              # 兜底；配合下面的 torch 语境判断
)


@dataclass(frozen=True)
class OomFact:
    """一次 CUDA OOM 里能读到的事实。单位统一 GiB。"""

    tried_to_allocate_gib: float
    total_capacity_gib: float = 0.0
    free_gib: float = 0.0
    already_allocated_gib: float = 0.0
    device: str = ""
    #: 原始那一行，进事件与卡片时给用户看
    excerpt: str = ""

    @property
    def in_use_gib(self) -> float:
        """OOM 那一刻已被占用的显存。"""
        if self.total_capacity_gib and self.free_gib:
            return max(self.total_capacity_gib - self.free_gib, 0.0)
        return self.already_allocated_gib

    def to_dict(self) -> dict[str, Any]:
        return {
            "tried_to_allocate_gib": round(self.tried_to_allocate_gib, 2),
            "total_capacity_gib": round(self.total_capacity_gib, 2),
            "free_gib": round(self.free_gib, 2),
            "in_use_gib": round(self.in_use_gib, 2),
            "device": self.device,
            "excerpt": self.excerpt,
        }


def _looks_like_cuda_oom(text: str) -> bool:
    lowered = text.lower()
    if "cuda out of memory" in lowered or "outofmemoryerror" in lowered:
        return True
    if "hip out of memory" in lowered:
        return True
    # 裸的 "out of memory" 太宽（Linux OOM killer、Java 堆都会命中），
    # 必须同时有 torch 语境才算。
    return "out of memory" in lowered and "tried to allocate" in lowered


def parse_cuda_oom(text: str | None) -> OomFact | None:
    """从一段日志里抽出 CUDA OOM 事实。不是 OOM 就返回 None。

    只认**带 `Tried to allocate` 的 CUDA/HIP OOM**：主机内存被 OOM killer 杀掉
    是另一回事（要加内存不是加显存），混在一起会推荐错方向的机器。
    """
    if not text or not _looks_like_cuda_oom(text):
        return None
    tried = _TRIED_RE.search(text)
    if not tried:
        return None

    capacity = _CAPACITY_RE.search(text) or _CAPACITY_OLD_RE.search(text)
    free = _FREE_NEW_RE.search(text) or _FREE_OLD_RE.search(text)
    allocated = _ALLOCATED_OLD_RE.search(text)
    device = _DEVICE_RE.search(text)

    excerpt = ""
    for line in text.splitlines():
        if "tried to allocate" in line.lower():
            excerpt = " ".join(line.split())[:300]
            break

    return OomFact(
        tried_to_allocate_gib=_to_gib(*tried.groups()),
        total_capacity_gib=_to_gib(*capacity.groups()) if capacity else 0.0,
        free_gib=_to_gib(*free.groups()) if free else 0.0,
        already_allocated_gib=_to_gib(*allocated.groups()) if allocated else 0.0,
        device=device.group(1) if device else "",
        excerpt=excerpt,
    )


def required_vram_gib(oom: OomFact, *, safety: float = PEAK_SAFETY_FACTOR) -> float:
    """由 OOM 推出**至少**需要多少显存（GiB）。

    那一刻真正需要的是「已占用 + 这次想分配的」。已占用优先用
    `总容量 - 空闲` 算，拿不到就退到 `already allocated`，再退到总容量本身
    （最保守：假设整张卡都被占满了）。

    再乘 `safety` 覆盖后续峰值——**那部分是假设不是测量**，
    所以返回值仍然只是一个「够用的下界」，不是精确需求。
    """
    in_use = oom.in_use_gib or oom.total_capacity_gib
    return max((in_use + oom.tried_to_allocate_gib) * safety, oom.tried_to_allocate_gib)


def detect_oom(outcome: Any) -> OomFact | None:
    """从 `rsa.agent.AgentOutcome` 里找 CUDA OOM。**不改 vendored rsa**。

    失败信息散落在几处：`outcome.error`（异常路径）、`escalation_md`（求助卡）、
    以及每次 pipeline 尝试的 `outcome`（判据执行的日志）。逐个翻，第一个命中就返回。
    """
    if outcome is None:
        return None

    candidates: list[str] = []
    for attr in ("error", "escalation_md"):
        value = getattr(outcome, attr, "")
        if isinstance(value, str) and value:
            candidates.append(value)

    attempts = list(getattr(outcome, "pipeline_attempts", None) or [])
    final = getattr(outcome, "pipeline", None)
    if final is not None and not any(a is final for a in attempts):
        attempts.append(final)
    for attempt in attempts:
        for attr in ("escalation_md", "outcome"):
            value = getattr(attempt, attr, None)
            if isinstance(value, str) and value:
                candidates.append(value)
            elif value is not None:
                # RunOutcome 之类的对象：把它 str 出来扫，比逐字段猜结构稳
                candidates.append(str(value))

    for text in candidates:
        found = parse_cuda_oom(text)
        if found:
            return found
    return None

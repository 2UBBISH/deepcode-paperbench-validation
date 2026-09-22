"""什么时候该把租来的机器还掉。

这是整条链路上**唯一直接决定用户花多少钱**的地方，所以做成纯函数：不碰网络、
不碰数据库、不看时钟（`elapsed_seconds` 由调用方传），好让每一条规则都能被单测钉死。

## 为什么不能按 idle 判

最容易想到的做法是「没有命令在跑就释放」。这个做法在 RSA 上是错的：
它的三个终态里有一个是 **求助（`needs_user`）**——缺凭据、缺数据、判据可能有误、
要加预算都会停下来问人，而人可能几小时后才回。按 idle 判会**在用户去吃饭的时候
把他的机器连同容器一起删掉**，回来发现要从头再来。

所以释放必须读 `AgentStatus`，而不是读机器忙不忙。

## 但 hold 不能无限

`needs_user` 挂起期间机器在**按秒计费**，这是本设计里最大的钱坑（见
`docs/experiment-agent-design.md` §8.5）。所以 hold 有两道闸：
到 `hold_warn_seconds` 发计费告警，到 `hard_cap_seconds` 无条件释放——
**硬上限压过一切**，包括 `needs_user`。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

#: RSA 跑完并给出的终态里，这几个代表「这一轮结束了，机器可以还」。
TERMINAL_RELEASE_STATUSES = frozenset({"success", "blocked", "failed"})

#: 这几个代表「还在正常回环里，别动机器」。
HOLD_STATUSES = frozenset({"needs_user", "recompile"})

#: 传给 `decide` 表示「run 还在跑，没有终态」。
STILL_RUNNING: str | None = None


class Disposition(str, Enum):
    RELEASE = "release"
    HOLD = "hold"


@dataclass(frozen=True)
class ReleaseDecision:
    disposition: Disposition
    #: 机器可读的原因，进事件与台账；不要拿它当给用户看的文案。
    reason: str
    #: 是否该给用户发一条「机器还开着，在花钱」的提醒。
    warn_cost: bool = False

    @property
    def should_release(self) -> bool:
        return self.disposition is Disposition.RELEASE


def decide(
    *,
    agent_status: str | None,
    elapsed_seconds: float,
    hard_cap_seconds: float,
    hold_warn_seconds: float | None = None,
    session_abandoned: bool = False,
) -> ReleaseDecision:
    """决定这台机器现在该不该还。

    `agent_status` 取 `rsa.agent.AgentStatus` 的值，或 `STILL_RUNNING`（None）表示
    「还在跑、没有终态」——周期性 tick 用这个值调进来，只为让硬上限有机会生效。

    空字符串与未知字符串都按**释放**处理：run 结束了却没报出终态，通常意味着 Agent
    崩了或协议变了，而这种时候把机器留着只会持续烧钱。宁可多还一次
    （下一轮重新租，成本有界）也不要漏一台（成本无界）。
    """
    # 1. 硬上限压过一切，包括 needs_user。租期无界是唯一不可接受的失败模式。
    if elapsed_seconds >= hard_cap_seconds:
        return ReleaseDecision(
            Disposition.RELEASE,
            f"hard_cap_exceeded:{elapsed_seconds:.0f}s>={hard_cap_seconds:.0f}s",
            warn_cost=True,
        )

    # 2. 会话被放弃/关闭：没人在等这台机器的结果了。
    if session_abandoned:
        return ReleaseDecision(Disposition.RELEASE, "session_abandoned")

    # 3. 还在跑：什么都不做（硬上限已经在上面判过了）。
    if agent_status is STILL_RUNNING:
        return ReleaseDecision(Disposition.HOLD, "still_running")

    status = str(agent_status).strip().lower()

    if status in TERMINAL_RELEASE_STATUSES:
        return ReleaseDecision(Disposition.RELEASE, f"terminal:{status}")

    if status in HOLD_STATUSES:
        warn = (
            hold_warn_seconds is not None
            and elapsed_seconds >= hold_warn_seconds
        )
        return ReleaseDecision(
            Disposition.HOLD,
            f"awaiting_user:{status}" if status == "needs_user" else f"in_loop:{status}",
            warn_cost=warn,
        )

    # 4. 空/未知终态 —— 见 docstring：往「释放」的方向错，不往「留着」的方向错。
    return ReleaseDecision(
        Disposition.RELEASE,
        f"unknown_status:{status or 'empty'}",
        warn_cost=True,
    )

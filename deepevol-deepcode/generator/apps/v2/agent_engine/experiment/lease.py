"""租赁编排器：把一台刚 CreateInstance 出来的机器变成「一个活的 Runtime」，用完还掉。

职责边界（`docs/experiment-agent-design.md` §2.1）：

    DeepEvol（这里）：开机 → wait_ready → 交付 Runtime 给 RSA → 读 AgentStatus → 释放 + 记账
    RSA：            拿 Runtime 去建容器、配环境、跑判据

ECS 的创建与删除**不在这里实现**——API 侧的
`/internal/engine/runs/{run_id}/remote-compute/usage/{start,finish}` 早就有了，
本模块通过 `LeaseBackend` 协议调它们。这样 ECS 凭据、计费台账、DB 事务都留在 API 侧，
Agent 侧只做编排。测试注入假 backend，不需要真云。

## 两条不变量

**一、任何路径都不能漏机器。** 开机成功之后的每一条出口——wait_ready 超时、RSA 抛异常、
调用方 cancel、进程收到信号——都必须走到 release。所以主流程是 `try/finally`，
且 `release()` 幂等（重复调只记一次）。漏一台机器的成本是无界的。

**二、注入给 RSA 的 backend 是一次性的。** `RSAAgent.run()` 的 finally
（`rsa/agent.py:127-133`）会 `close()` 它并置 `None`。所以：
每次 run 前新建一个注入，DeepEvol 自己**另持一条控制连接**做 readiness 与释放。
这不是洁癖——`needs_user` 挂起时 RSA 那侧已经没有活的 backend 了，
而那正是最需要盯着计费的时候。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from .release_policy import ReleaseDecision, decide

logger = logging.getLogger(__name__)

#: wait_ready 的默认上限。remote_relay 的 wait_ready 本就是为「刚创建、sshd 还没起、
#: cloud-init 还在跑」的云机器写的；超了要**主动删机器**，否则就是一台漏在外面的实例。
DEFAULT_READY_TIMEOUT = 600.0

#: 释放失败时的重试次数与退避。网络抖动、API 5xx 都可能让一次释放失败，
#: 而失败的代价是一台**无人知晓、持续计费**的机器。
RELEASE_ATTEMPTS = 3
RELEASE_BACKOFF_SECONDS = 2.0

#: 刚创建的机器有两段独立的就绪期：sshd 起来、以及**密码被写入**。
#: 后者可能更晚，而 wait_ready 对认证失败不重试 —— 详见 `_bring_up`。
BRING_UP_ATTEMPTS = 3
BRING_UP_BACKOFF_SECONDS = 15.0

#: 打镜像期间多久把攒下的进度冲成一次事件。实测整个过程 ~50 分钟，
#: 进度大约 30 秒涨 1%，所以这个间隔和它同量级就够了。
SNAPSHOT_DRAIN_INTERVAL = 30.0


class LeaseState(str, Enum):
    PLANNED = "planned"
    PROVISIONING = "provisioning"
    READY = "ready"
    RUNNING = "running"
    UPGRADING = "upgrading"
    RELEASING = "releasing"
    RELEASED = "released"
    FAILED = "failed"


@dataclass(frozen=True)
class LeaseHandle:
    """API 侧开机成功后回给我们的东西。"""

    usage_id: str
    resource_id: str
    #: `"ssh -p 22 root@1.2.3.4"` —— remote_relay 的 parse_access_url 原生认这个格式，
    #: 不需要转换（见 apps/api/.../remote_compute.py:_aliyun_access_fields）。
    access_url: str
    username: str = "root"
    password: str | None = None
    private_key_path: str | None = None
    #: 外壳模式：Agent 侧只拿到凭据的**引用**，密码由外壳在绑定机器时按引用解出
    #: （`shell_bridge.shell_runtime_factory(secrets=…)`），不进本对象也不进 rsa 配置。
    secret_ref: str = ""
    secret_version: str = ""
    spec: dict[str, Any] = field(default_factory=dict)

    def redacted(self) -> dict[str, Any]:
        return {
            "usage_id": self.usage_id,
            "resource_id": self.resource_id,
            "access_url": self.access_url,
            "username": self.username,
            "auth": (
                "password" if self.password
                else "secret_ref" if self.secret_ref
                else "key" if self.private_key_path
                else "agent"
            ),
        }


class LeaseBackend(Protocol):
    """API 侧租赁接口。实现见 `lease_api.py`；测试注入假的。"""

    async def provision(self, *, run_id: str, spec: dict[str, Any]) -> LeaseHandle: ...

    async def release(self, *, run_id: str, usage_id: str, reason: str) -> None: ...

    #: 可选：不实现就没有升级能力，`upgrade()` 会明确报错而不是静默降级。
    #: 故意不写进 Protocol 的必需方法 —— 那会让所有既有假 backend 一夜失效。
    # async def snapshot(self, *, run_id: str, name: str = "",
    #                    on_progress: Callable[[str, str], None] | None = None) -> str: ...
    #: `on_progress` 也是可选的：不收这个参数照样能打镜像，只是用户看不到实时进度
    #: （见 `_accepts_on_progress`）。


class LeaseError(RuntimeError):
    pass


@dataclass
class LeaseEvent:
    state: LeaseState
    detail: str = ""


EventSink = Callable[[LeaseEvent], Awaitable[None] | None]
RuntimeFactory = Callable[[LeaseHandle], Any]


def _accepts_on_progress(snapshot: Any) -> bool:
    try:
        params = inspect.signature(snapshot).parameters
    except (TypeError, ValueError):
        return False
    return "on_progress" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


async def _emit(sink: EventSink | None, state: LeaseState, detail: str = "") -> None:
    if sink is None:
        return
    try:
        result = sink(LeaseEvent(state, detail))
        if hasattr(result, "__await__"):
            await result
    except Exception:  # 事件上报失败绝不能影响租赁本身
        logger.warning("lease event sink failed at %s", state.value, exc_info=True)


class ExperimentLease:
    """一次实验的机器租期。**用 `async with` 或确保 `release()` 被调到。**"""

    def __init__(
        self,
        *,
        run_id: str,
        backend: LeaseBackend,
        runtime_factory: RuntimeFactory,
        hard_cap_seconds: float,
        ready_timeout: float = DEFAULT_READY_TIMEOUT,
        hold_warn_seconds: float | None = None,
        event_sink: EventSink | None = None,
        bring_up_attempts: int = BRING_UP_ATTEMPTS,
        bring_up_backoff: float = BRING_UP_BACKOFF_SECONDS,
    ) -> None:
        self.run_id = run_id
        self.state = LeaseState.PLANNED
        self.handle: LeaseHandle | None = None
        self._backend = backend
        self._runtime_factory = runtime_factory
        #: 公开：调用方要按同一个上限判断，别各自留一份 ——
        #: 两处真相迟早会分叉，而分叉的那一侧就是漏机器的那一侧。
        self.hard_cap_seconds = hard_cap_seconds
        self._hard_cap = hard_cap_seconds
        self._ready_timeout = ready_timeout
        self.hold_warn_seconds = hold_warn_seconds
        self._hold_warn = hold_warn_seconds
        self._sink = event_sink
        self._released = False
        self._runtime: Any = None
        self._bring_up_attempts = max(bring_up_attempts, 1)
        self._bring_up_backoff = bring_up_backoff

    # -- 生命周期 ------------------------------------------------------------

    async def acquire(self, spec: dict[str, Any]) -> Any:
        """开机 → wait_ready → 返回一条**归 DeepEvol 自己持有**的控制连接。

        中途任何失败都会把已经开出来的机器还回去再抛——这是不变量一。
        """
        if self.state is not LeaseState.PLANNED:
            raise LeaseError(f"lease 已经是 {self.state.value}，不能重复 acquire")

        self.state = LeaseState.PROVISIONING
        await _emit(self._sink, self.state, f"instance_type={spec.get('instance_type', '?')}")
        try:
            self.handle = await self._backend.provision(run_id=self.run_id, spec=spec)
        except Exception as exc:
            self.state = LeaseState.FAILED
            await _emit(self._sink, self.state, f"provision_failed: {exc}")
            raise LeaseError(f"开机失败: {exc}") from exc

        # 机器已经存在、已经在计费了。从这里开始，任何失败都必须还机器。
        try:
            runtime = await self._bring_up(self.handle)
        except Exception as exc:
            # PROVISIONING 超时不主动删就是漏机器 —— 设计文档 §8.4 明写的那条。
            await self.release(reason=f"ready_failed:{type(exc).__name__}")
            self.state = LeaseState.FAILED
            await _emit(self._sink, self.state, f"wait_ready_failed: {exc}")
            raise LeaseError(f"机器开出来了但连不上（已释放）: {exc}") from exc

        self._runtime = runtime
        self.state = LeaseState.READY
        await _emit(self._sink, self.state, str(self.handle.redacted()))
        return runtime

    async def _bring_up(self, handle: LeaseHandle) -> Any:
        """建连接并等它可用。带**有界重试**，因为刚创建的机器有两段独立的就绪期。

        ★ `wait_ready` 对认证失败刻意不重试（它的注释：「凭据错了，再等一百年也不会好」）。
        那个判断对**已存在**的机器是对的，对**刚 CreateInstance 出来**的机器不对——
        阿里云的 root 密码是在实例初始化时写入的，**可能比 sshd 晚就绪**，
        于是 sshd 已经在听、但密码还没装上，连接被 `PermissionDenied` 拒掉。

        实测踩到过：实例 17:09:08 创建，24 秒后连接被拒；而同样的代码前两次都过了——
        纯粹是 cloud-init 快慢的随机性。不重试就是个会随机让开机失败的 flaky。

        重试**有界**：真的密码错时不能无限试，SSH 的 MaxAuthTries 会把账号拒死。
        """
        last_exc: Exception | None = None
        for attempt in range(self._bring_up_attempts):
            runtime = self._runtime_factory(handle)
            try:
                if hasattr(runtime, "start"):
                    await runtime.start()
                await runtime.wait_ready(timeout=self._ready_timeout)
                return runtime
            except Exception as exc:
                last_exc = exc
                if hasattr(runtime, "close"):
                    try:
                        await runtime.close()
                    except Exception:
                        pass
                if attempt + 1 < self._bring_up_attempts:
                    logger.warning(
                        "lease %s 连接失败（第 %d 次，机器可能还在初始化），退避重试: %s",
                        self.run_id, attempt + 1, exc,
                    )
                    await asyncio.sleep(self._bring_up_backoff * (attempt + 1))
        raise last_exc if last_exc else LeaseError("连接失败")

    async def upgrade(
        self, target_spec: dict[str, Any], *, reason: str, use_image: bool = True
    ) -> Any:
        """换一台更大的机器，**保住已经配好的环境**。

        流程是三步既有路径的组合，不新增计费语义：
          1. `snapshot` 把当前盘固化成镜像（会先停机）
          2. `release` 结算并删掉旧机 —— 走的是已经被测过的那条结算路径
          3. `provision` 用 `snapshot_image_id` 开新机

        ★ 为什么不走「变配」：变配只能同族（实测 sgn7i-vws 只能变到 6 个同族规格），
          而且要新加「中途结算」入口，否则新价会追溯到整段时长。镜像换机跨族可用，
          且每条 usage 各一条账本，计费天然正确。

        ★ **不变量：任一时刻最多一台机器在计费。**
          所以顺序必须是「先还旧、再开新」，不能为了少一次等待而并行。
          代价是中间有一段两台都没有的空窗——那是对的，空窗不烧钱。

        ★ 如果第 3 步失败，旧机已经删了、新机没起来，`self.handle` 会被置空，
          此时 `_released` 为 True，`__aexit__` 不会再去还一台不存在的机器。
          镜像仍然留着（调用方可以拿 `image_id` 重试），这是有意的：
          删掉它等于把用户已经付过的那次环境配置也丢了。
        """
        if self.handle is None:
            raise LeaseError("没有在租的机器，无法升级")

        previous = self.handle
        self.state = LeaseState.UPGRADING

        if not use_image:
            # ★ 不打镜像的路径。实测 CreateImage 要 34 分钟，而 P0c 实测
            #   简单仓库配一次环境只要 6 分钟 —— 那时打镜像纯属让用户多等半小时。
            #   由 `choose_upgrade_strategy` 用**本次**配环境的实际耗时来选，
            #   见 upgrade_plan.py。这条路径下调用方负责在新机上重配环境。
            await _emit(self._sink, self.state, f"rebuild:{reason}")
            await self._close_runtime()
            return await self._swap_machine(target_spec, reason=reason, previous=previous)

        snapshot = getattr(self._backend, "snapshot", None)
        if snapshot is None:
            raise LeaseError("当前 backend 不支持打镜像，无法在保住环境的前提下升级")
        await _emit(self._sink, self.state, f"snapshot:{reason}")

        # 打镜像前必须断开控制连接：接下来这台机器要停机。
        await self._close_runtime()
        # ★ 打镜像实测要 ~50 分钟。50 分钟不出声，用户会以为卡死了 ——
        #   把进度转成事件往上报。`_emit` 自己吞掉 sink 的异常，
        #   但这里是同步回调，仍要挡一层：进度上报绝不能连累打镜像。
        pending: list[tuple[str, str]] = []

        def _on_progress(status: str, progress: str) -> None:
            pending.append((status, progress))

        async def _drain() -> None:
            while pending:
                status, progress = pending.pop(0)
                await _emit(self._sink, LeaseState.UPGRADING,
                            f"snapshot_progress:{status}:{progress}")

        try:
            image_id = await self._snapshot_with_progress(snapshot, _on_progress, _drain)
        except Exception as exc:
            # ★ 打镜像失败时旧机**没有被删**，盘上的环境还在 —— 这是好事。
            #   但它很可能已经**停机**了：服务端是「先 stop 再 CreateImage」，
            #   失败点在 stop 之后的概率更大。所以不能装作什么都没发生地回到 READY，
            #   那会让调用方拿着一个连不上的 handle 去跑命令。
            #   置 FAILED 并保留 handle：`_released` 仍是 False，
            #   兜底释放会把这台机器还掉，不会漏。
            self.state = LeaseState.FAILED
            await _emit(self._sink, self.state, f"snapshot_failed:{exc}")
            logger.error(
                "lease %s 打镜像失败，旧机 %s 未删除但可能已停机: %s",
                self.run_id, previous.resource_id, exc,
            )
            raise
        await _emit(self._sink, self.state, f"image:{image_id}")

        return await self._swap_machine(
            {**target_spec, "snapshot_image_id": image_id},
            reason=reason, previous=previous, image_id=image_id,
        )

    async def _swap_machine(
        self,
        spec: dict[str, Any],
        *,
        reason: str,
        previous: LeaseHandle,
        image_id: str = "",
    ) -> Any:
        """还掉旧机、开出新机。打镜像与不打镜像两条路径共用这一段。

        ★ **不变量：任一时刻最多一台机器在计费。**
          所以必须「先还旧、再开新」，不能为了少等几分钟而并行。
          代价是中间有一段两台都没有的空窗 —— 那是对的，空窗不烧钱。
        """
        # 先还旧机。这一步失败就整个中止 —— 带着一台旧机去开新机，
        # 会出现两条 usage 同时挂账，正是不变量禁止的情况。
        await self.release(reason=f"upgrade:{reason}")

        self._released = False          # 新一段租期重新开始
        self.handle = None
        self.state = LeaseState.PROVISIONING
        await _emit(self._sink, self.state, f"upgrade_provision:{reason}")
        try:
            self.handle = await self._backend.provision(run_id=self.run_id, spec=spec)
        except Exception:
            # 开机失败：没有机器在计费（旧的已删、新的没建成）。
            # 置 _released 免得兜底再去 finish 一段并不存在的租期。
            self._released = True
            self.state = LeaseState.FAILED
            await _emit(self._sink, self.state,
                        f"upgrade_failed:image={image_id or 'none'}")
            logger.error(
                "lease %s 升级开机失败；旧机 %s 已释放，镜像 %s",
                self.run_id, previous.resource_id, image_id or "（本次未打镜像）",
            )
            raise

        try:
            runtime = await self._bring_up(self.handle)
        except Exception as exc:
            await self.release(reason=f"upgrade_ready_failed:{type(exc).__name__}")
            raise LeaseError(f"升级后的机器没能就绪（已释放）：{exc}") from exc

        self._runtime = runtime
        self.state = LeaseState.READY
        await _emit(self._sink, self.state, f"upgraded:{image_id or 'rebuild'}")
        return runtime

    async def _snapshot_with_progress(self, snapshot, on_progress, drain) -> str:
        """跑 backend.snapshot，同时周期性把它攒下的进度冲成事件。

        为什么要这个中间层：backend 的 `on_progress` 是**同步**回调
        （它在自己的轮询循环里调），而 `_emit` 是异步的。
        直接在同步回调里 `await` 不行，攒起来在这边冲是最简单的做法。
        """
        task = asyncio.ensure_future(
            snapshot(run_id=self.run_id, name=f"deepevol-{self.run_id}"[:128],
                     on_progress=on_progress)
        )
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=SNAPSHOT_DRAIN_INTERVAL)
                await drain()
        finally:
            await drain()
        return await task

    @property
    def can_snapshot(self) -> bool:
        """这个 backend 支不支持打镜像。

        暴露成属性是为了让调用方**不必伸手进 `_backend`** —— 隔着模块摸别人的
        私有属性，等 backend 换实现时会在一个完全无关的地方炸掉。
        """
        return getattr(self._backend, "snapshot", None) is not None

    async def snapshot_now(self, *, name: str = "", on_progress=None) -> str:
        """现在就打一张镜像，但**不动租期**。

        `upgrade()` 里那次打镜像是换机流程的一步；这个是给「先收尾再问用户」
        用的：OOM 之后先把环境固化下来，再把机器还掉，让用户在没有计费压力的
        情况下决定要不要升级。
        """
        snapshot = getattr(self._backend, "snapshot", None)
        if snapshot is None:
            raise LeaseError("当前 backend 不支持打镜像")
        # ★ 进度是锦上添花，**不能成为打镜像的前提**：backend 不支持就照常打，
        #   而不是让整次快照失败（那比没有进度糟得多）。
        if on_progress is None or not _accepts_on_progress(snapshot):
            return await snapshot(
                run_id=self.run_id, name=name or f"deepevol-{self.run_id}"[:128]
            )
        # ★ backend 的回调是**同步**的（它在自己的轮询循环里调），而调用方要 await。
        #   攒下来在 `_snapshot_with_progress` 的间隙冲 —— 与 upgrade() 那条路同一套。
        pending: list[tuple[str, str]] = []

        def _collect(status: str, progress: str) -> None:
            pending.append((status, progress))

        async def _drain() -> None:
            while pending:
                status, progress = pending.pop(0)
                await on_progress(status, progress)

        return await self._snapshot_with_progress(snapshot, _collect, _drain)

    def mark_running(self) -> None:
        if self.state is LeaseState.READY:
            self.state = LeaseState.RUNNING

    async def release(self, *, reason: str, attempts: int = RELEASE_ATTEMPTS) -> bool:
        """还机器。返回是否**真的**释放成功了。

        ★ 「幂等」与「已成功释放」是两件事，别用同一个标志。

        初版把 `_released = True` 写在尝试**之前**，于是释放一失败（比如本机网络断了），
        `__aexit__` 的兜底再调进来会立刻返回 False、**不再重试** —— 机器就这么漏了。
        实测漏过一台：本机 DNS 挂掉导致 SSH 与阿里云 API 同时不可达，
        而那恰恰是最需要清理的时刻。

        所以：只有**成功之后**才置位，失败就带退避重试，全失败才抛。
        """
        if self._released or self.handle is None:
            return False
        self.state = LeaseState.RELEASING
        await _emit(self._sink, self.state, reason)

        last_exc: Exception | None = None
        try:
            for attempt in range(max(attempts, 1)):
                try:
                    await self._backend.release(
                        run_id=self.run_id, usage_id=self.handle.usage_id, reason=reason
                    )
                    self._released = True          # 成功之后才置位
                    self.state = LeaseState.RELEASED
                    await _emit(self._sink, self.state, reason)
                    return True
                except Exception as exc:
                    last_exc = exc
                    if attempt + 1 < max(attempts, 1):
                        logger.warning(
                            "lease %s 释放失败（第 %d 次），退避重试: %s",
                            self.run_id, attempt + 1, exc,
                        )
                        await asyncio.sleep(RELEASE_BACKOFF_SECONDS * (attempt + 1))
        finally:
            await self._close_runtime()

        # 重试全失败：这台机器还在烧钱，只能靠 API 侧的孤儿回收兜底。
        # 必须吼出来 —— 静默的泄漏要等到有人看账单才会发现。
        self.state = LeaseState.FAILED
        await _emit(self._sink, self.state, f"release_failed:{last_exc}")
        logger.error(
            "lease %s 释放失败 %d 次，实例 %s 可能仍在计费，等待服务端孤儿回收: %s",
            self.run_id, max(attempts, 1), (self.handle.resource_id if self.handle else "?"), last_exc,
        )
        raise last_exc if last_exc else LeaseError("释放失败")

    async def _close_runtime(self) -> None:
        runtime, self._runtime = self._runtime, None
        if runtime is None or not hasattr(runtime, "close"):
            return
        try:
            await runtime.close()
        except Exception:
            logger.warning("控制连接关闭失败（不影响释放）", exc_info=True)

    # -- 释放决策 ------------------------------------------------------------

    def decide(
        self,
        *,
        agent_status: str | None,
        elapsed_seconds: float,
        session_abandoned: bool = False,
    ) -> ReleaseDecision:
        return decide(
            agent_status=agent_status,
            elapsed_seconds=elapsed_seconds,
            hard_cap_seconds=self._hard_cap,
            hold_warn_seconds=self._hold_warn,
            session_abandoned=session_abandoned,
        )

    async def settle(
        self,
        *,
        agent_status: str | None,
        elapsed_seconds: float,
        session_abandoned: bool = False,
    ) -> ReleaseDecision:
        """按 AgentStatus 决定去留，该释放就释放。返回决策供调用方记录/告警。"""
        decision = self.decide(
            agent_status=agent_status,
            elapsed_seconds=elapsed_seconds,
            session_abandoned=session_abandoned,
        )
        if decision.should_release:
            await self.release(reason=decision.reason)
        return decision

    # -- 上下文管理 ----------------------------------------------------------

    async def __aenter__(self) -> ExperimentLease:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        # 不变量一的最后一道：无论怎么退出，没结算过的租期一律还掉。
        # 正常路径应该已经通过 settle() 释放过了，这里是 crash / cancel 的兜底。
        if not self._released and self.handle is not None:
            try:
                await self.release(
                    reason=f"context_exit:{exc_type.__name__ if exc_type else 'normal'}"
                )
            except Exception:
                # 这里**不能再抛**：正在处理别的异常时抛出去会把原始失败原因盖掉，
                # 而那通常才是用户要看的东西。release() 内部已经 log.error + 发事件，
                # 泄漏不会静默。
                logger.error("lease %s 兜底释放也失败了，靠服务端孤儿回收", self.run_id)
        return False

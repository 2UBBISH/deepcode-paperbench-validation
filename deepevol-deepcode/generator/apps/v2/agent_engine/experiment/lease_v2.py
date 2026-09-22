"""`LeaseBackend` 的 V2 实现：走 Product 的 remote-compute 内部契约。

V1 的 `lease_api.py` 调 `apps/api` 的 `usage/start` / `server-use-targets` /
`usage/finish`；那些路由随 V1 一起退役了。V2 里同一件事拆成三段，
都已经存在、且各自有持久化与幂等保证：

    开机  = `POST /internal/v2/runs/{rid}/remote-compute/selection`
            → PROVISION saga（Product 记账押金 → Agent 控制面开机 → 本地回执）
            → 202 轮询 → 200 带 usage(binding) 与 resource
    连接  = `GET  /internal/v2/runs/{rid}/remote-workspace?lease_generation=…`
            → host / port / username / secret_ref（密码在 worker 本机的密文文件里）
    还机  = `POST …/remote-compute/usage/finish`（结算这段租期）
          + `POST …/remote-compute/release`（RELEASE saga，删实例；只删本 run 自己开的）

三段都是同步 httpx 客户端（`ProductRemoteComputeLifecycleClient`），
这里用 `asyncio.to_thread` 包成 `ExperimentLease` 要的 async 协议。

## 与 V1 backend 的两处语义差异（都是有意的）

1. **同一 run 里第二次开机要换身份。** `select()` 的 operation_id 由
   (run, prompt, target, resource_policy) 决定，用来让重试幂等。但升级换机、
   或者用户在同一 run 里再跑一次同一档，都会撞上「上一段已经结算的 binding」。
   所以每次 `provision` 把递增的 `experiment_lease_ordinal` 写进
   resource_policy —— 它在台账里可见，且不改变 Product 对 policy 的其它解释。
2. **释放是两步、顺序不可换。** Product 的本地 finalizer 拒绝在 Billing 结算之前
   投影一次 Provider 释放（`finalize_provider_release` 会抛
   `RemoteComputeConflict`），所以必须先 `finish` 再 `release`。
   两步各自幂等：`finish` 重放返回同一张账单，`release` 的 operation_id
   由 PROVISION 回执派生。

镜像（snapshot）暂未接入 V2 控制面；`ExperimentLease.can_snapshot` 为 False，
OOM 升级走「重开重配」路径（`run_flow._settle_for_upgrade` 已按此降级）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from apps.v2.remote_compute.lifecycle_client import (
    ProductRemoteComputePending,
    ProductRemoteComputeRejected,
    ProductRemoteComputeUnavailable,
)

from .lease import LeaseError, LeaseHandle

logger = logging.getLogger(__name__)

#: PROVISION saga 从提交到 200 的等待上限。阿里云 CreateInstance + 等 Running +
#: 公网 IP 通常 1–3 分钟；AutoDL 更慢。saga 自己有 max_attempts=5 的退避重试，
#: 所以这里给足 30 分钟——过早放弃会让一台稍后才开出来的机器无人认领。
DEFAULT_PROVISION_DEADLINE = 30 * 60.0
DEFAULT_RELEASE_DEADLINE = 15 * 60.0
_TERMINAL_FAILURE = {"FAILED", "CANCELLED", "COMPENSATED"}

SleepFn = Callable[[float], Awaitable[None]]


@dataclass
class _RunLocator:
    """`ProductRemoteWorkspaceClient.resolve()` 要的最小定位对象。

    `lease_generation` 这里是 **binding 的代数**（Product 的 remote-workspace
    路由拿它和 binding 行比对），不是 run 租约的代数。
    """

    rid: Any
    sid: Any
    resource_tid: Any
    owner_uid: Any
    lease_generation: int


@dataclass
class V2RemoteComputeLeaseBackend:
    lifecycle: Any
    workspace_client: Any
    secrets: Any
    context: Any
    authority: Any
    prompt: str
    resource_policy: Mapping[str, int] = field(default_factory=dict)
    resource_requirements: Mapping[str, int | bool] = field(default_factory=dict)
    poll_interval: float = 3.0
    provision_deadline: float = DEFAULT_PROVISION_DEADLINE
    release_deadline: float = DEFAULT_RELEASE_DEADLINE
    sleep: SleepFn = asyncio.sleep
    monotonic: Callable[[], float] = time.monotonic
    #: 第几段租期。每次 provision 前 +1，见模块 docstring。
    lease_ordinal: int = 0

    # ------------------------------------------------------------ provision
    async def provision(self, *, run_id: str, spec: dict[str, Any]) -> LeaseHandle:
        target = str(spec.get("target") or spec.get("selected_target") or "").strip()
        if not target:
            raise LeaseError("spec 里缺 target（selected_target），无法开机")
        self.lease_ordinal += 1
        policy = {**dict(self.resource_policy), "experiment_lease_ordinal": self.lease_ordinal}
        deadline = self.monotonic() + self.provision_deadline
        data: Mapping[str, Any] | None = None
        while data is None:
            try:
                data = await asyncio.to_thread(
                    self.lifecycle.select,
                    self.context,
                    self.authority,
                    prompt=self.prompt,
                    selected_target=target,
                    resource_policy=policy,
                    resource_requirements=dict(self.resource_requirements) or None,
                )
            except ProductRemoteComputePending as pending:
                await self._await_operation(
                    pending.operation_id,
                    first_wait=float(pending.retry_after),
                    deadline=deadline,
                    what="开机",
                )
            except ProductRemoteComputeRejected as exc:
                raise LeaseError(f"开机被拒绝：{exc.code}") from exc
            except ProductRemoteComputeUnavailable as exc:
                raise LeaseError(f"开机失败：{getattr(exc, 'code', exc)}") from exc
        usage = data.get("usage") if isinstance(data.get("usage"), Mapping) else None
        resource = data.get("resource") if isinstance(data.get("resource"), Mapping) else {}
        if usage is None:
            raise LeaseError(f"开机返回缺 usage：{sorted(data)[:12]}")
        binding_id = str(usage.get("binding_id") or "").strip()
        resource_id = str(usage.get("resource_id") or "").strip()
        generation = int(usage.get("lease_generation") or 0)
        if not binding_id or not resource_id or generation < 1:
            raise LeaseError(f"开机返回的 usage 不完整：{sorted(usage)[:12]}")
        access = await asyncio.to_thread(self._lookup_access, generation)
        return LeaseHandle(
            usage_id=binding_id,
            resource_id=resource_id,
            access_url=access["access_url"],
            username=access["username"],
            password=access["password"],
            secret_ref=access["secret_ref"],
            secret_version=access["secret_version"],
            spec={
                **dict(spec),
                "resource_id": resource_id,
                "binding_id": binding_id,
                "lease_generation": generation,
                "resource": dict(resource),
            },
        )

    def _lookup_access(self, generation: int) -> dict[str, Any]:
        locator = _RunLocator(
            rid=self.context.rid,
            sid=self.context.sid,
            resource_tid=self.context.resource_tid,
            owner_uid=self.context.owner_uid,
            lease_generation=generation,
        )
        descriptor = self.workspace_client.resolve(locator)
        if descriptor is None:
            raise LeaseError("机器开出来了，但 Product 没有给出可连接的工作区描述（remote-workspace 404）")
        host = str(getattr(descriptor, "host", "") or "").strip()
        if not host:
            raise LeaseError("remote-workspace 描述里没有 host")
        port = int(getattr(descriptor, "port", 22) or 22)
        username = str(getattr(descriptor, "username", "root") or "root")
        secret_ref = str(getattr(descriptor, "secret_ref", "") or "")
        secret_version = str(getattr(descriptor, "secret_version", "") or "")
        password: str | None = None
        if self.secrets is not None:
            # Legacy path: the lease itself resolves the password and the
            # handle carries it.  In shell mode ``secrets`` is None here and
            # the shell resolves the reference when it binds the machine.
            try:
                password = self.secrets.resolve(secret_ref, version=secret_version)
            except Exception as exc:
                raise LeaseError(f"拿不到机器的访问凭据（secret_ref={secret_ref}）：{exc}") from exc
        elif not secret_ref:
            raise LeaseError("机器开出来了，但 Product 的工作区描述里没有凭据引用（secret_ref）")
        return {
            "access_url": f"ssh -p {port} {username}@{host}",
            "username": username,
            "password": password or None,
            "secret_ref": secret_ref,
            "secret_version": secret_version,
        }

    # -------------------------------------------------------------- release
    async def release(self, *, run_id: str, usage_id: str, reason: str) -> None:
        outcome = _settlement_outcome(reason)
        try:
            settled = await asyncio.to_thread(
                self.lifecycle.finish,
                self.context,
                self.authority,
                ended_at=datetime.now(UTC),
                outcome=outcome,
            )
        except (ProductRemoteComputeRejected, ProductRemoteComputeUnavailable) as exc:
            raise LeaseError(f"租期结算失败（usage/finish）：{getattr(exc, 'code', exc)}") from exc
        if settled is None:
            # 没有 binding：机器压根没开出来（或已被别的路径结算并释放）。
            logger.warning("release：run=%s 没有可结算的 binding，跳过（reason=%s）", run_id, reason)
        deadline = self.monotonic() + self.release_deadline
        while True:
            try:
                data = await asyncio.to_thread(
                    self.lifecycle.release,
                    self.context,
                    self.authority,
                    reason=reason,
                )
            except ProductRemoteComputePending as pending:
                await self._await_operation(
                    pending.operation_id,
                    first_wait=float(pending.retry_after),
                    deadline=deadline,
                    what="释放",
                )
                continue
            except ProductRemoteComputeRejected as exc:
                if exc.status_code == 404:
                    logger.warning("release：run=%s 无 usage 可释放（reason=%s）", run_id, reason)
                    return
                raise LeaseError(f"释放被拒绝：{exc.code}") from exc
            except ProductRemoteComputeUnavailable as exc:
                raise LeaseError(f"释放失败：{getattr(exc, 'code', exc)}") from exc
            if data.get("released") is True:
                return
            # 不是本 run 开的机器（用户池里的常驻服务器）：不删，也不算失败。
            logger.warning(
                "release：run=%s 的机器不由本 run 开出（%s），只结算不删除",
                run_id, data.get("reason"),
            )
            return

    # ------------------------------------------------------------- polling
    async def _await_operation(
        self, operation_id: Any, *, first_wait: float, deadline: float, what: str
    ) -> None:
        """等一个 saga operation 走到终态，再由调用方重发原请求拿回执。"""
        wait = max(first_wait, 1.0)
        while True:
            await self.sleep(wait)
            try:
                status = await asyncio.to_thread(
                    self.lifecycle.get_operation_status,
                    self.context,
                    self.authority,
                    operation_id=operation_id,
                )
            except ProductRemoteComputeUnavailable as exc:
                if self.monotonic() > deadline:
                    raise LeaseError(f"{what}进度查询持续失败：{getattr(exc, 'code', exc)}") from exc
                logger.warning("%s进度查询失败，稍后重试：%s", what, exc)
                wait = self.poll_interval
                continue
            state = str(status.get("status") or "")
            if state == "SUCCEEDED":
                return
            if state in _TERMINAL_FAILURE:
                code = status.get("error_code") or state
                raise LeaseError(f"{what}失败：{code}")
            if self.monotonic() > deadline:
                raise LeaseError(f"{what}超时：operation 仍在 {state}")
            wait = self.poll_interval


def _settlement_outcome(reason: str) -> str:
    text = str(reason or "").lower()
    if "cancel" in text:
        return "CANCELLED"
    if "success" in text or text.startswith("terminal:success"):
        return "SUCCEEDED"
    return "FAILED"


def make_relay_runtime(handle: LeaseHandle, *, secrets: Any = None) -> Any:
    """`RuntimeFactory`：把 LeaseHandle 变成 remote_relay 的 `RemoteRuntime`。

    这条连接是 **DeepEvol 自己的控制连接**，不交给 RSA——RSA 那边每次 run 另建一个
    一次性的（`RSAAgent.run()` 的 finally 会把注入的 backend 关掉）。

    外壳模式下 handle 只带 `secret_ref`；密码在这里（外壳绑定机器的闭包里）按引用
    解出，只进 relay 的连接目标，不回流到调用方。
    """
    import dataclasses

    from apps.v2.agent_engine.remote_relay import RemoteRuntime
    from apps.v2.agent_engine.remote_relay.transport.target import parse_access_url

    password = handle.password
    if password is None and handle.secret_ref:
        if secrets is None:
            raise LeaseError("机器凭据只有引用，但没有可用的凭据解析器")
        password = secrets.resolve(handle.secret_ref, version=handle.secret_version)
    target = dataclasses.replace(
        parse_access_url(
            handle.access_url,
            username=handle.username,
            password=password,
        ),
        keepalive_count_max=40,
    )
    return RemoteRuntime(target)


def typed_run_id(context: Any) -> str:
    """run 的字符串形态（rid_…），给 ExperimentLease 记日志用。"""
    try:
        from apps.common.v2_ids import format_typed_id

        return format_typed_id("rid", context.rid)
    except Exception:
        return str(getattr(context, "rid", ""))


__all__ = [
    "DEFAULT_PROVISION_DEADLINE",
    "DEFAULT_RELEASE_DEADLINE",
    "V2RemoteComputeLeaseBackend",
    "make_relay_runtime",
    "typed_run_id",
]

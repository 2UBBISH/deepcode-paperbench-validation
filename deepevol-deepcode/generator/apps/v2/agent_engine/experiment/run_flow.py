"""用户选定一档之后：租机器 → 配环境跑 → 还机器。

## 一条压过一切的不变量：机器一定要还

租下去的机器按秒计费，而**没人会盯着**。所以这一层的每一条出口
——成功、失败、异常、取消、上游把任务 cancel 掉——都必须经过释放。
`ExperimentLease` 的 `async with` 负责兜底，这里不再自己写 try/except
去「顺便」释放：两套释放逻辑会有一套先腐坏。

## RSA 是同步的

`RSAAgent.run` 会阻塞几分钟到几十分钟。直接在事件循环里调它，
整个 Agent 服务的所有会话都会跟着卡住。所以放进线程。
代价是不能中途 cancel 那个线程 —— 硬上限由租期的 `hard_cap_seconds`
兜着，那是钱的边界，不是这里的。

## 显存估错的出口

RSA 跑失败时先看是不是 CUDA OOM。是的话不当作普通失败 ——
带上**实测**需求回去问用户要不要升级（`upgrade_plan`），
这是 P4 那条闭环的入口。此时机器**不释放**：升级要用它打镜像，
而且用户可能选择降 batch 重试。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from .evidence import begin_recording, bind_round_recorder, collect_evidence, render_evidence, take_recorded_rounds
from .git_daemon import serve_repo_on_machine
from .lease import ExperimentLease, LeaseHandle
from .oom import OomFact, detect_oom
from .release_policy import decide
from .upgrade_plan import (
    choose_upgrade_strategy,
    eta_from_progress,
    format_wait_estimate,
)

logger = logging.getLogger(__name__)

#: 默认硬上限。租期无界是唯一不可接受的失败模式 —— 见 release_policy。
DEFAULT_HARD_CAP_SECONDS = 6 * 3600.0


@dataclass
class ExperimentRunResult:
    status: str                    # success | needs_user | oom | failed | cancelled
    markdown: str
    oom: OomFact | None = None
    released: bool = False
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "success"


#: 交给 RSA 的 runtime 用这个阈值决定「流式 exec」还是「耐久作业」。
#:
#: ★ remote_relay 默认 300 秒，而 RSA 的 `run()` / `_exec_container()` 默认
#:   `timeout=300` —— 判据是 `>`，**正好差一点**，于是那些几分钟的长命令
#:   （装依赖、跑训练）全部落在流式通道上。一次网络抖动就是
#:   `RelayConnectionError: connection lost while reading`，判据 0/0、整轮报废。
#:   实测第 22、23 轮各跑了 43 分钟后都倒在这里。
#:
#: ★ 取 120 而不是 0：全转耐久会让 `test -e` 这类一秒探测也去落日志文件 + 轮询，
#:   而 prelude 里这种探测很多。120 能覆盖 RSA 那批 timeout=300 的长命令，
#:   又让 timeout=60 的小探测走快路径。
#: 配环境期间多久发一次心跳。服务端判活窗口是 10 分钟、孤儿回收阈值 30 分钟，
#: 取 3 分钟留足余量：漏发两次也还在窗口内。
HEARTBEAT_SECONDS = 180.0

#: 挂起等人多久之后开始提醒计费。挑 15 分钟是因为它明显长于一次正常的往返
#: 确认，又远短于「人已经走开了」的时间尺度。
DEFAULT_HOLD_WARN_SECONDS = 900.0

#: OOM 之后把工作目录暂存到主机的哪里。
#:
#: ★ 为什么放**主机**而不是留在容器里：升级时我们打的是整机镜像
#:   （`CreateImage` 只传 InstanceId、不带 DiskDeviceMapping，40G 系统盘全进去），
#:   所以主机上的文件会跟着镜像走到新机器上；而 RSA 在新机器上会起一个**全新容器**
#:   并重新 clone，旧容器的层数据虽然也在镜像里，却没人会去用它。
#:   落到主机的固定路径，新容器建好后再拷回去 —— 这条路不需要对象存储、
#:   不需要数据盘，完全复用已经在打的那张镜像。
RESUME_HOST_DIR = "/var/lib/deepevol/resume"
RESUME_CONTAINER_WORKDIR = "/workspace/repo"

RSA_DURABLE_THRESHOLD_SECONDS = 120.0


def _durable_runtime_factory(*, target, username="root", password=None, private_key=None):
    """给 RSA 造一条**断线不丢活**的连接。

    `RemoteDockerBackend` 在 `runtime is None` 时会调这个工厂（关键字传参），
    所以不用改 vendored 代码就能换掉它的执行底座。
    """
    from apps.v2.agent_engine.remote_relay import RemoteRuntime
    from apps.v2.agent_engine.remote_relay.transport.target import parse_access_url

    resolved = parse_access_url(
        str(target), username=username, password=password, private_key_path=private_key,
    )
    # SSHTarget 是 frozen dataclass，只能 replace。keepalive 同样放宽：
    # 长跑期间本来就有大段静默（pip 装包、训练），别把它误判成掉线。
    resolved = dataclasses.replace(resolved, keepalive_count_max=40)
    return RemoteRuntime(resolved, durable_threshold=RSA_DURABLE_THRESHOLD_SECONDS)


def build_rsa_config(
    handle: LeaseHandle,
    *,
    store: Path,
    work: Path,
    backend: str = "small",
    shell_endpoint: tuple[str, str] | None = None,
):
    """按租到的机器造一份 RSA 配置。

    ★ `execution_backend="remote"` 是产品默认，`local` 只是开发逃生口
      （rsa/pipeline.py:47 明写）。这里从不选 local ——
      本地跑意味着在**我们自己的服务器**上执行用户仓库里的代码。

    ★ `remote_target` 直接用租期给的 access_url：它就是
      `ssh -p 22 root@1.2.3.4` 这个形状，remote_relay 的 parse_access_url
      原生认，不需要转换。
    """
    from apps.v2.agent_engine.rsa.pipeline import PipelineConfig

    # ★ 这里**不建 backend**。`RemoteDockerBackend.__init__` 会当场建连
    #   （`remote_backend.py:207` 的 `self._sync.call(runtime.start)`），
    #   放在这里会让本函数变成带副作用的构造 —— 注入假 runner 的单测
    #   也会去连一个不存在的主机，5 次重试带退避，整个测试套挂死（实测踩到）。
    #   耐久 backend 在 `_default_rsa_runner` 里建：那是真正要用它的地方，
    #   紧接着就是 `RSAAgent.run()`，其 finally 会把它关掉。
    if shell_endpoint is not None:
        # ★ 外壳模式：rsa 只拿到外壳的 URL + token，机器地址与密码留在外壳里
        #   （见 shell_bridge）。`remote_target` 仍要非空——pipeline 用它判断
        #   "remote" 后端已配置，但它不再是任何可连接的东西。
        base_url, token = shell_endpoint
        config = PipelineConfig(
            store=store,
            work=work,
            backend=backend,
            execution_backend="remote",
            remote_target=f"shell {base_url}",
            remote_username="",
            remote_password=None,
            remote_private_key=None,
        )
        config.deepevol_shell_endpoint = (base_url, token)
        return config
    return PipelineConfig(
        store=store,
        work=work,
        backend=backend,
        execution_backend="remote",
        remote_target=handle.access_url,
        remote_username=handle.username,
        remote_password=handle.password,
        remote_private_key=handle.private_key_path,
    )


async def run_experiment_on_machine(
    *,
    lease: ExperimentLease,
    spec: dict[str, Any],
    repo_url: str = "",
    instruction: str,
    revision: str = "",
    #: 压缩包路径：本地那份代码的目录。给了它就在机器上起 git daemon，
    #: 把 repo_url 换成只有那台机器能访问的 git://127.0.0.1/... 。
    local_repo_dir: str | Path | None = None,
    session_id: str = "",
    store: Path,
    work: Path,
    #: RSA/SetupX 要用的 OpenAI 兼容端点。**不给就起不来** ——
    #: SetupX 进场时读 `.env.small`，那份文件由 `write_setupx_backend_envs`
    #: 现写；以前没人写过它，于是每次都死在 `no backend config at .../.env.small`。
    #: 留 None 只为让注入了 `rsa_runner` 的测试不必造一个假端点。
    llm_target: Any = None,
    hard_cap_seconds: float | None = None,
    rsa_runner=None,
    on_event=None,
    #: Agent 外壳。给了它，rsa 的执行底座就是外壳的 ShellRuntime（机器凭据不进
    #: rsa 配置），并且外壳账本会记下这一跑的命令与传输量。
    shell: Any = None,
) -> ExperimentRunResult:
    """租机器、跑一轮 RSA、按策略决定要不要还机器。

    `rsa_runner(config, request)` 注入是为了让这条流程能在**没有云、
    没有 SSH** 的情况下被测到 —— 它是唯一会花钱的路径，
    只能靠真机验证的话，每改一次都要付一次。
    """
    import time

    async def _emit(stage: str, detail: str = "") -> None:
        if on_event is None:
            return
        try:
            result = on_event(stage, detail)
            if hasattr(result, "__await__"):
                await result
        except Exception:      # 事件上报绝不能连累实验本身
            logger.warning("实验事件上报失败 stage=%s", stage, exc_info=True)

    async def _heartbeat(every: float = HEARTBEAT_SECONDS) -> None:
        """配环境期间定期发事件，让服务端知道这个 run 还活着。

        ★ 真机验收连撞三轮才定位到的坑：RSA 的 setup loop 一跑几十分钟，
          期间一个事件都不发，服务端 `chat_runs.updated_at` 就停在原地。
          超过心跳窗口后 `_run_is_live()` 判定这个 run 已经死了，30 分钟的
          孤儿回收 `reap_stale_remote_compute_usages()` 就把机器
          **从正在干活的 setup loop 脚下**释放掉。
          随后 RSA 报的是 `cannot connect ... after 5 attempts: TimeoutError`
          —— 看起来像网络问题，其实是机器被自己人删了。
          reaper 那边本来就想放过长任务（注释写着「别关机」），
          但它靠心跳判断，而这里从来不发。
        """
        minutes = 0
        while True:
            await asyncio.sleep(every)
            minutes += every / 60
            await _emit("running", f"仍在配环境（已 {minutes:.0f} 分钟）")

    # ★ 硬上限**只有一个来源**：租期自己的。这里不再留一份默认值 ——
    #   两处真相迟早会分叉，而分叉的那一侧就是漏机器的那一侧。
    cap = hard_cap_seconds if hard_cap_seconds is not None else getattr(
        lease, "hard_cap_seconds", DEFAULT_HARD_CAP_SECONDS
    )
    started = time.monotonic()

    # ★ 这里**不用 `async with lease`**。
    #   `async with` 保证退出必还机器，那对「跑完就结束」是对的，
    #   但 needs_user 与 OOM 要求机器**跨轮次留着**（用户还要用它），
    #   一个 with 块没法既保证必还、又允许有意不还。
    #   所以改成显式控制：**任何异常出口一律还**，
    #   只有那两种被明确标成 `released=False` 的情况才留，
    #   而它们由调用方接手（并且有硬上限与服务端孤儿回收兜底）。
    try:
        await _emit("provisioning", "正在开机")
        runtime = await lease.acquire(spec)
        lease.mark_running()
        handle = lease.handle
        assert handle is not None      # acquire 成功就一定有
        await _emit("running", f"机器就绪，开始配环境（{handle.resource_id}）")

        # ★ 压缩包路径：代码只存在于我们本地，而 RSA 的 clone 在**远端容器内**
        #   执行（`rsa/pipeline.py:130` 明确拒绝远端路径下的 local_path）。
        #   所以先把它送上机器、起一个只监听回环的 git daemon，
        #   再把 repo_url 换成那个 URL。见 git_daemon 模块 docstring。
        effective_repo_url = repo_url
        if local_repo_dir:
            await _emit("serving_code", "正在把代码送到机器上")
            effective_repo_url = await serve_repo_on_machine(runtime, local_repo_dir)
            # 上传的代码没有「分支」这个概念，bundle 里就一个 commit。
            revision = ""

        config = build_rsa_config(
            handle,
            store=store,
            work=work,
            shell_endpoint=None if shell is None else (shell.base_url, shell.token),
        )
        if spec.get("max_rounds"):
            config.max_rounds = int(spec["max_rounds"])
        # 判据守卫是否真的跑过 —— 由守卫在 RSA 线程里写，跑完在这边读。
        integrity: dict[str, Any] = {
            "checked": False,
            "reason": "守卫未执行",
            # 注入了假 runner 的测试路径本来就没有守卫，别给它们扣「未校验」的帽子。
            "expected": rsa_runner is None,
        }
        config.deepevol_integrity = integrity
        runner = rsa_runner or _default_rsa_runner
        setupx_ctx = _setupx_env(llm_target) if llm_target is not None else _noop_ctx()
        heartbeat = asyncio.create_task(_heartbeat())
        try:
            # ★ RSA 是同步的，阻塞几分钟到几十分钟。直接在事件循环里调，
            #   整个 Agent 服务的所有会话都会跟着卡住。
            # ★ V2 里硬上限还要能压住一个**卡死的 RSA 线程**：线程本身停不了，
            #   但钱能停——超时就往下走释放路径，线程留在后台，机器一没它自己会挂。
            # ★ 不用 `asyncio.to_thread`：它借的是事件循环的默认线程池，而
            #   `asyncio.run()` 退出时会 `shutdown_default_executor()` **等池里的
            #   线程跑完**——硬上限一到，机器已经还了，调用方却还得陪着那个
            #   卡死的 RSA 线程等它自己死（真机上等了 11 分钟，等到 run 的
            #   租约出事）。单独开一个池，超时后 `shutdown(wait=False)` 放手。
            rsa_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="deepevol-rsa"
            )
            try:
                with setupx_ctx:
                    outcome = await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(
                            rsa_pool, runner, config, effective_repo_url, instruction, revision, session_id
                        ),
                        timeout=max(cap - (time.monotonic() - started), 1.0),
                    )
            finally:
                rsa_pool.shutdown(wait=False)
        except asyncio.TimeoutError:
            logger.error("实验超过硬上限 %.0fs，强制释放 repo=%s", cap, repo_url)
            _kill_rsa_child(config)
            await _emit("failed", f"超过硬上限 {cap / 3600:.1f} 小时，已强制释放机器")
            released = await _release_quietly(lease, "hard_cap_exceeded")
            return ExperimentRunResult(
                "failed",
                f"### 超时\n\n实验跑了超过 {cap / 3600:.1f} 小时还没结束，已按硬上限强制释放机器。",
                released=released,
                data={"hard_cap_seconds": round(cap)},
            )
        except asyncio.CancelledError:
            # 上游 cancel（用户点停止 / 服务重启）。下面的 except 会还机器，
            # 但要先说清楚是被取消而不是失败。
            _kill_rsa_child(config)
            await _emit("cancelled", "任务被取消，正在释放机器")
            raise
        except Exception as exc:
            logger.exception("RSA 执行异常 repo=%s", repo_url)
            await _emit("failed", f"执行异常：{exc}")
            released = await _release_quietly(lease, f"rsa_exception:{type(exc).__name__}")
            return ExperimentRunResult(
                "failed", f"### 实验没跑起来\n\n执行过程中出错：{exc}", released=released
            )
        finally:
            heartbeat.cancel()

        status = str(getattr(getattr(outcome, "status", None), "value", "") or
                     getattr(outcome, "status", "") or "")

        # ★ 显存不够单独处理：不是普通失败，而是**已经拿到实测数字**的失败。
        oom = _oom_of(outcome)
        if oom is not None:
            return await _settle_for_upgrade(
                lease, oom=oom, status=status,
                env_setup_seconds=time.monotonic() - started, emit=_emit,
                resumable=bool(spec.get("resumable")),
                runtime=runtime,
            )

        elapsed = time.monotonic() - started
        decision = decide(
            agent_status=status or None,
            elapsed_seconds=elapsed,
            hard_cap_seconds=cap,
            # ★ 以前这里没传 hold_warn，于是 warn_cost 恒为假、计费告警从未触发过。
            hold_warn_seconds=getattr(lease, "hold_warn_seconds", None)
            or DEFAULT_HOLD_WARN_SECONDS,
        )
        if not decision.should_release:
            # needs_user / recompile：机器留着，用户回话之后接着用。
            hourly = float(spec.get("hourly_price_cny") or 0)
            if decision.warn_cost:
                # ★ warn_cost 以前算了没人用 —— 挂起变贵时得有人被告知。
                await _emit("cost_warning", format_accrued_cost(elapsed, hourly))
            await _emit("needs_user", decision.reason)
            evidence = _evidence_of(outcome)
            return ExperimentRunResult(
                "needs_user",
                _render_needs_user(
                    outcome,
                    elapsed_seconds=elapsed,
                    hourly_price_cny=hourly,
                    hard_cap_seconds=cap,
                )
                + _evidence_section(evidence),
                released=False,
                data={"agent_status": status, "reason": decision.reason, "evidence": evidence},
            )

        await _emit("releasing", decision.reason)
        released = await lease.release(reason=decision.reason)
        result_status = "success" if status == "success" else "failed"
        await _emit(result_status, f"耗时 {elapsed / 60:.0f} 分钟")
        evidence = _evidence_of(outcome)
        return ExperimentRunResult(
            result_status,
            _render_outcome(
                outcome,
                elapsed_seconds=elapsed,
                hourly_price_cny=float(spec.get("hourly_price_cny") or 0),
                integrity=integrity,
            )
            + _evidence_section(evidence),
            released=released,
            data={"agent_status": status, "elapsed_seconds": round(elapsed), "evidence": evidence},
        )
    except BaseException:
        # ★ 任何异常出口（含 CancelledError、开机失败、代码 bug）一律还机器。
        #   失败时没人会想起来去看那台机器，而它按秒计费。
        await _release_quietly(lease, "abnormal_exit")
        raise


async def _save_resume_state(runtime) -> bool:
    """把容器工作目录拷到主机，好让它随整机镜像一起去新机器。

    ★ 失败绝不能连累升级流程：拷不动最多是「换机后从头训」，
      而抛出去会把一次本可以继续的升级变成彻底失败。
    """
    from .git_daemon import _exec

    script = (
        f"set -e; rm -rf {RESUME_HOST_DIR}; mkdir -p {RESUME_HOST_DIR}; "
        # 容器名固定是 rsa-<hex>（rsa/remote_backend.py:318）。取最近建的那个。
        "cid=$(docker ps -aq --filter 'name=^/rsa-' | head -1); "
        '[ -n "$cid" ] || { echo no-container; exit 0; }; '
        f"docker cp \"$cid:{RESUME_CONTAINER_WORKDIR}\" {RESUME_HOST_DIR}/repo "
        "&& echo saved"
    )
    try:
        text = await _exec(runtime, script, timeout=600, allow_failure=True)
    except Exception:
        logger.warning("保存训练进度失败，升级后将从头开始", exc_info=True)
        return False
    if "saved" not in text:
        logger.warning("没有找到可保存的训练进度：%s", text[:200])
        return False
    return True


async def _settle_for_upgrade(
    lease: ExperimentLease,
    *,
    oom: OomFact,
    status: str,
    env_setup_seconds: float,
    emit,
    resumable: bool = False,
    runtime=None,
) -> ExperimentRunResult:
    """OOM 之后**先把机器收尾，再去问用户**。

    ★ 这条顺序是有意的，而且与直觉相反。
      留着机器等回答看起来更快（用户答得快就能接着用原机），
      但机器按秒计费而**人是会走开的** —— 隔夜回来就是一整晚的账单。
      而且现有的孤儿回收大约 30 分钟就会把它收掉，
      答晚了机器已经没了：不可预测，比明确释放更糟。

    ★ 打不打镜像由 `choose_upgrade_strategy` 用**本次配环境的实际耗时**决定
      （它刚刚发生过）。打镜像时机器是**停机**状态（StopCharging），
      不再计算力费，所以那段等待不烧钱 —— 代价是用户在那段时间里
      还看不到问题。
    """
    await emit("oom", "显存不够，正在收尾")
    strategy = choose_upgrade_strategy(env_setup_seconds=env_setup_seconds)

    # ★ 先保存训练进度，**再**打镜像 —— 顺序不能反：镜像是整机快照，
    #   保存动作必须发生在快门按下之前，否则拷过去的是个空目录。
    resumed = False
    if resumable and runtime is not None:
        resumed = await _save_resume_state(runtime)
        if resumed:
            await emit("resume_saved", "已经保存训练进度，升级后可以接着跑")

    image_id = ""
    if strategy.use_image:
        if lease.can_snapshot:
            # ★ 报区间而不是单点。实测同样 40G 的镜像，阿里云侧从 34 分钟到 81 分钟都出现过
            #   （容量决定、内容几乎无关，见 upgrade_plan._CREATE_IMAGE_SAMPLES）。
            #   给一个偏乐观 2 倍的精确数字，比给区间更糟。
            await emit("snapshot", f"正在保存环境（{format_wait_estimate()}）")
            started_at = time.monotonic()

            last_percent: list[float] = []

            async def _on_snapshot_progress(status: str, progress: str) -> None:
                try:
                    percent = float(str(progress).strip().rstrip("%"))
                except ValueError:
                    return
                # 后端每 30 秒轮一次，百分比没动就别重复刷同一句。
                if last_percent and last_percent[-1] == percent:
                    return
                last_percent.append(percent)
                remaining = eta_from_progress(time.monotonic() - started_at, percent)
                if remaining is None:
                    # 10% 以前不外推（前段比后段慢，早报会高估）；
                    # 已经跑过历史最慢那次时也不编 —— 但百分比本身还是要给，
                    # 「没有任何更新」比「只有百分比」更让人不安。
                    await emit("snapshot", f"正在保存环境（{percent:.0f}%）")
                    return
                if remaining < 90:
                    # 「约还要 0 分钟」是句废话,还显得系统在乱讲。
                    await emit("snapshot", f"正在保存环境（{percent:.0f}%，快好了）")
                    return
                await emit(
                    "snapshot",
                    f"正在保存环境（{percent:.0f}%，约还要 {remaining / 60:.0f} 分钟）",
                )

            try:
                image_id = await lease.snapshot_now(on_progress=_on_snapshot_progress)
            except Exception as exc:
                # 镜像没打成不该让整条路断掉 —— 大不了升级时重配一遍环境。
                logger.warning("OOM 后打镜像失败，降级为重配环境: %s", exc)
                await emit("snapshot_failed", str(exc)[:200])

    released = await _release_quietly(lease, "oom_settled_before_asking")
    await emit("released", "机器已释放，接下来问你要不要升级")
    return ExperimentRunResult(
        "oom",
        "### 显存不够\n\n跑起来之后 CUDA 报了 out of memory。"
        "**机器已经释放**，不会在你决定期间继续计费。"
        + ("环境已经保存成镜像，升级后不用重配。" if image_id else ""),
        oom=oom,
        released=released,
        data={
            "agent_status": status,
            "image_id": image_id,
            "env_setup_seconds": round(env_setup_seconds),
            "strategy": strategy.to_dict(),
            "resumable": bool(resumable),
            # 进度真存下来了才算数：存不成就等于不可续跑，卡片得按这个说。
            "resume_saved": bool(resumed),
        },
    )


async def _release_quietly(lease: ExperimentLease, reason: str) -> bool:
    """尽力释放。释放失败**不再往上抛** —— 那会盖掉调用方真正要看的原因。

    `lease.release` 内部已经 log.error 并发事件，服务端还有孤儿回收兜底，
    所以泄漏不会静默。
    """
    try:
        return await lease.release(reason=reason)
    except Exception:
        logger.error("释放失败，等待服务端孤儿回收 reason=%s", reason, exc_info=True)
        return False


@contextmanager
def _noop_ctx():
    yield None


#: 本进程共用的 SetupX 检出。见 `_setupx_env` 的说明 —— 它**必须**跨轮次稳定。
_SETUPX_ROOT: Path | None = None


@contextmanager
def _setupx_env(llm_target):
    """准备 SetupX 的运行根目录，并写好这一轮的 backend 配置。

    ★ **每进程一份，不是每轮一份。** 最初写成每轮一个临时目录、跑完就删，
      结果第二轮必炸 `FileNotFoundError: .../setupx/log`：
      SetupX 的 `config.py` 在 **import 时**就把
      `PROJECT_ROOT = Path(__file__).parent.parent` 固化下来，而 Agent 是长驻
      进程、`sys.modules` 会缓存它 —— 第二轮即便把 `RSA_SETUPX_ROOT` 指向新
      目录，模块里记的仍是第一轮那个**已被删掉**的路径。
      而且 `RSA_SETUPX_ROOT` 本来就是进程级环境变量，
      「每轮隔离」从一开始就是错觉。

    ★ 仍然不写进 vendored 树：`.env.small` 里是明文 api_key。
      每轮重写它，所以换模型照样生效。
      代价是同进程内两轮实验会共用这份配置 —— 与 `RSA_SETUPX_ROOT`
      的进程级语义一致，真要并发得让 rsa 支持按 run 传 root。
    """
    global _SETUPX_ROOT

    import atexit
    import os
    import shutil
    import tempfile

    from .setupx_env import clear_stale_env_local, write_setupx_backend_envs

    if _SETUPX_ROOT is None or not _SETUPX_ROOT.exists():
        vendored = Path(__file__).resolve().parents[1] / "setupx"
        workdir = Path(tempfile.mkdtemp(prefix="deepevol-setupx-"))
        root = workdir / "setupx"
        shutil.copytree(vendored, root)
        # SetupX 自己会建 log/，但只在**父目录已存在**时；这里先备好，
        # 顺带让「目录必须稳定」这件事显式一点。
        (root / "log").mkdir(exist_ok=True)
        atexit.register(shutil.rmtree, str(workdir), True)
        _append_contract_to_checkout(root)
        _SETUPX_ROOT = root
        logger.info("SetupX 进程级检出 %s", root)

    write_setupx_backend_envs(_SETUPX_ROOT, small=llm_target)
    clear_stale_env_local(_SETUPX_ROOT)
    os.environ["RSA_SETUPX_ROOT"] = str(_SETUPX_ROOT)
    _rebind_setup_loop_summary()
    logger.info("SetupX backend 配置已更新（模型 %s）", llm_target.redacted())
    yield _SETUPX_ROOT


#: 配环境 Agent 上报「这台机器不够」的方式。守卫在评分前读它。
INSUFFICIENT_SENTINEL = "/workspace/.rsa-insufficient-resources"

#: 追加到 SetupX 系统提示后面的一段契约。
#:
#: ★ 为什么需要它：SetupX 的动作集（setupx/src/models.py 的 ActionType）只有
#:   SHELL_COMMAND / TRY_XPU_SUGGESTION / SET_ENV / ROLLBACK_ENV / VERIFY / FINISH，
#:   而系统提示规定 FINISH 只能在 VERIFY 通过之后调用 —— **没有任何一个动作能表达
#:   「这台机器的硬件不够，我做不到」**。任务物理上不可能时，它只剩两条路：
#:   耗到步数上限，或者想办法让 VERIFY 通过。
#:
#:   真机验收七轮实证了后者会发生，而且手法逐代升级：装 .pth + meta_path hook 换掉
#:   实验模块 → 保留 __file__ 与常量不变只换 main() → patch
#:   torch.cuda.max_memory_allocated 伪造峰值 → 改 sitecustomize.py 按调用栈定向缩小
#:   torch.zeros → 改 site-packages/torch/__init__.py → 最后把 python 可执行文件
#:   换成包装脚本。它的补丁注释里写着 `without modifying repository files` ——
#:   **它严格遵守了「不改仓库」的契约**。伪造不是它不守规矩，是契约没给它别的出口。
#:
#: ★ 所以这段做两件事：点名禁掉那几种手法（泛泛说「不要作弊」它会自行判断边界），
#:   并给一个正当的上报出口。
_CONTRACT_ADDENDUM = """

## Environment Integrity (hard rules)

The frozen criterion measures **this repository's real behaviour in this environment**.
Anything that makes the criterion pass without the repository's own code actually doing
the work is a failed run reported as a success, which is worse than an honest failure.

**Never do any of the following to make the criterion pass:**
- Install or edit `sitecustomize.py`, `usercustomize.py`, or any `.pth` file that imports code.
- Register a `sys.meta_path` / `sys.path_hooks` entry that intercepts the repository's imports.
- Edit, patch or monkey-patch files of installed packages (e.g. anything under `site-packages/`),
  including the libraries the criterion itself measures with.
- Replace or wrap the `python` / `python3` executable, or any interpreter the criterion invokes.
- Point `PYTHONPATH` / `PYTHONSTARTUP` at a directory that shadows the repository's modules.

Installing dependencies, setting environment variables the project needs, and fixing genuine
configuration problems all remain expected and encouraged. The line is: change the environment
so the repository's own code can run — never change what the repository's code *does*.

## When the machine is not big enough

If the workload genuinely cannot fit this machine's hardware (GPU memory, RAM, disk),
that is **not** a problem for you to work around. It is a finding the user needs, and
reporting it is the correct, successful outcome of your task.

In that case:
1. Run the failing command once more and keep its exact output.
2. Write that output verbatim to `%s` (a single SHELL_COMMAND with a heredoc is fine).
3. Stop calling VERIFY. Do not attempt further fixes.

The orchestrator reads that file, surfaces the measured shortfall to the user, and offers a
bigger machine. Reporting an honest shortfall is a good outcome; a criterion that passes
because the environment was doctored is not.
""" % INSUFFICIENT_SENTINEL


def _summarise_actions_with_detail(history: list, limit: int = 40) -> list:
    """把配环境 Agent 每一步的**命令原文与退出码**带进升级卡。

    ★ vendored 的 `summarise_actions`（rsa/setup_loop.py）读 `a.get("command")`，
      但 `AgentAction.to_dict()`（setupx/src/models.py）把命令放在 `content` 下 ——
      于是每条 SHELL_COMMAND 都只剩「SHELL_COMMAND」五个字，命令原文全丢。
      结果是：实验失败时，用户和我们从产品输出里都看不到它到底做了什么，
      只能去翻容器里的 SetupX 临时日志。排查那七轮伪造时我就卡在这上面。
    ★ 顺带带上 exit_code：`result` 整个被丢掉了，而「跑了什么」和「成没成」
      分开看基本没用。
    """
    out: list = []
    for h in history or []:
        action = h.get("action") if isinstance(h, dict) else None
        action = action if isinstance(action, dict) else {}
        content = action.get("content")
        content = content if isinstance(content, dict) else {}
        kind = str(action.get("action_type") or action.get("type") or "").upper()
        detail = (
            content.get("command")
            or action.get("command")          # 老形状兜底
            or content.get("message")
            or action.get("message")
            or (f"{content.get('key')}={content.get('value')}" if content.get("key") else "")
            or ""
        )
        result = h.get("result") if isinstance(h, dict) else None
        result = result if isinstance(result, dict) else {}
        code = result.get("exit_code")
        suffix = "" if code is None else f" → exit={code}"
        line = f"{kind}: {detail}".strip().rstrip(":") + suffix
        if line.strip() and line not in out:
            out.append(line[:300])
    return out[-limit:]


def _rebind_setup_loop_summary() -> None:
    """让 vendored 的 setup_loop 用上面那个版本。失败绝不连累实验。"""
    try:
        from apps.v2.agent_engine.rsa import setup_loop
    except Exception:  # pragma: no cover
        # ★ 静默 return 会让「没生效」和「没跑」在日志里长得一模一样 ——
        #   这一晚为此白费过两轮真机，不再犯。
        logger.warning("SetupX 行为摘要：拿不到 rsa.setup_loop，本轮不替换", exc_info=True)
        return
    if getattr(setup_loop, "_deepevol_summary_bound", False):
        return
    setup_loop.summarise_actions = _summarise_actions_with_detail
    setup_loop._deepevol_summary_bound = True
    # 一次性的启动事实，用 warning 是为了在默认日志级别下**看得见** ——
    # 看不见的确认等于没有确认。
    logger.warning("SetupX 行为摘要已换成带命令原文与退出码的版本")


#: 追加到**我们自己那份 SetupX 副本**末尾的补丁。导入 llm_engine 时自然生效。
_CONTRACT_PATCH_MARK = "# --- DeepEvol: environment-integrity contract ---"


def _append_contract_to_checkout(root) -> None:
    """把契约补充写进 SetupX 副本的 `src/llm_engine.py` 末尾。

    ★ 为什么不在这里 import 再改类属性：在这个时点 import SetupX 会触发
      `load_config()`，而它要求 `OPENAI_API_KEY` 已在环境里 —— 那是 RSA 后面
      `setupx_configured(...)` 才设的（`resolve_llm_target` 的 docstring 早写过
      这一点）。真机上就是这么静默失败的：日志只留一句「拿不到 llm_engine」，
      整轮跑完才发现契约根本没生效。
    ★ `_SETUPX_ROOT` 是 `copytree` 出来的**每进程副本**，不是 vendored 树，
      改它不违反 DEEPEVOL_VENDOR.md。追加在文件末尾而不是就地替换字符串：
      不依赖模板的具体写法，vendored 升级了也不会悄悄失配。
    """
    target = Path(root) / "src" / "llm_engine.py"
    try:
        body = target.read_text(encoding="utf-8")
    except Exception:
        logger.warning("SetupX 契约补充：读不到 %s，本轮不注入", target, exc_info=True)
        return
    if _CONTRACT_PATCH_MARK in body:
        return
    patch = (
        f"\n\n{_CONTRACT_PATCH_MARK}\n"
        "try:\n"
        "    LLMEngine.SYSTEM_PROMPT_TEMPLATE = (\n"
        "        LLMEngine.SYSTEM_PROMPT_TEMPLATE + "
        f"{_CONTRACT_ADDENDUM!r}\n"
        "    )\n"
        "except NameError:\n"
        "    pass\n"
    )
    try:
        target.write_text(body + patch, encoding="utf-8")
    except Exception:
        logger.warning("SetupX 契约补充：写不进 %s，本轮不注入", target, exc_info=True)
        return
    logger.warning("SetupX 契约补充已写入副本（禁改环境 + 资源不足上报出口）")


def _gpu_aware_backend_class(integrity: dict | None = None):
    """vendored 的 `RemoteDockerBackend` 加上 GPU 透传。

    ★ 它起实验容器时没有 `--gpus`（那条 `docker run --detach` 只带 network/label），
      于是容器内 `torch.cuda.is_available()` 恒为 False。而夹具第一条断言就是
      「torch 看得见 GPU」—— 这个条件在容器里永远满足不了，setup loop 只能一遍遍
      试着装驱动直到超时。真机验收实测：87 分钟里 SetupX 日志反复出现
      `Unable to locate package nvidia-utils` 与 `CUDA: False`，机器全程 1% 负载。

    ★ 不能无条件加：CPU 机器上 `docker run --gpus all` 会让容器根本起不来。
      所以先问一次宿主机 `nvidia-smi -L`，结果缓存。判断依据是**真的能用**，
      而不是规格名说它应该有卡。

    类在首次使用时才构造 —— import 期 rsa 不一定已经在 sys.path 上。
    按 DEEPEVOL_VENDOR.md 的惯例从外部注入，不动 vendored 树。
    """
    import json
    import shlex

    from apps.v2.agent_engine.rsa.remote_backend import BridgeError, RemoteDockerBackend

    class GpuAwareRemoteDockerBackend(RemoteDockerBackend):
        _gpu_probe_result: bool | None = None

        def _machine_has_gpu(self) -> bool:
            if self._gpu_probe_result is None:
                try:
                    probe = self._exec_host("nvidia-smi -L", 120)
                    self._gpu_probe_result = bool(
                        getattr(probe, "success", False) and "GPU" in (probe.stdout or "")
                    )
                except Exception:
                    # 探测失败按「没有卡」处理：缺一块 GPU 会让实验失败得很明白，
                    # 而在 CPU 机器上误加 --gpus all 会让容器压根起不来。
                    self._gpu_probe_result = False
            return self._gpu_probe_result

        def _host(self, argv: list[str]) -> str:
            # 只认实验容器与回滚容器（`docker run --detach`）。
            # grader 那条是 `--rm --network none`，不该也不需要 GPU。
            if (
                len(argv) > 2
                and str(argv[1]) == "run"
                and "--detach" in argv
                and "--gpus" not in argv
                and self._machine_has_gpu()
            ):
                argv = [argv[0], "run", "--gpus", "all", *argv[2:]]
            return " ".join(shlex.quote(str(x)) for x in argv)

        # ---------------- 判据完整性守卫 ----------------
        #
        # ★ 真机验收发现（见 docs/experiment-agent-design.md §18）：判据在
        #   SetupX 完全掌控的容器里执行，而它收到的指令是「让判据通过」。
        #   六代夹具防御全被绕开 —— 它先读懂断言再针对性伪造，最后直接改
        #   site-packages/torch/__init__.py，并注明「Repository files are not
        #   modified」。工作树回滚保护不了这个：影子代码不在仓库里。
        #   后果不是某个测试跑不过，而是**实验结论不可信**。
        #
        #   这里在评分前做两件事：
        #     ① 把配环境阶段**新增**的代码注入点（sitecustomize / usercustomize /
        #        会执行 import 的 .pth）删掉 —— 与建容器时的基线比对，只删新增的，
        #        不碰镜像自带的（distutils-precedence.pth 这类是正常的）。
        #     ② 校验已安装包的文件是否被改过（比对 dist-info/RECORD 里的 sha256）。
        #        装完再改文件是 ① 拦不住的那一手。
        #   ① 让判据跑回真代码；② 让「改库文件」这条路留下痕迹而不是静默通过。

        _HOOK_BASELINE_PATH = "/tmp/.rsa-hook-baseline.json"

        _SNAPSHOT_HOOKS = (
            "python3 - <<'PY'\n"
            "import hashlib, json, pathlib, site, sys\n"
            "dirs = list(site.getsitepackages()) if hasattr(site, 'getsitepackages') else []\n"
            "try:\n"
            "    dirs.append(site.getusersitepackages())\n"
            "except Exception:\n"
            "    pass\n"
            "seen = {}\n"
            "for d in dirs:\n"
            "    root = pathlib.Path(d)\n"
            "    if not root.is_dir():\n"
            "        continue\n"
            "    for name in ('sitecustomize.py', 'usercustomize.py'):\n"
            "        f = root / name\n"
            "        if f.is_file():\n"
            "            seen[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()\n"
            "    for f in root.glob('*.pth'):\n"
            "        seen[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()\n"
            # ★ 解释器自身也要记：真机第七代绕法就是把 python 换成包装脚本，
            #   site-packages 一个字节没动，前面那些检查全看不见。
            "import shutil, sys\n"
            "for name in ('python', 'python3'):\n"
            "    exe = shutil.which(name)\n"
            "    if not exe:\n"
            "        continue\n"
            "    real = str(pathlib.Path(exe).resolve())\n"
            "    blob = pathlib.Path(real).read_bytes()\n"
            "    seen['interp:' + name] = '%%s|%%s|%%s' %% (\n"
            "        real, blob[:4] == b'\\x7fELF', hashlib.sha256(blob).hexdigest())\n"
            "pathlib.Path(%r).write_text(json.dumps(seen))\n"
            "print('baseline', len(seen))\n"
            "PY" % _HOOK_BASELINE_PATH
        )

        _SANITIZE = (
            "python3 - <<'PY'\n"
            "import hashlib, json, pathlib, site\n"
            "base = {}\n"
            "p = pathlib.Path(%r)\n"
            "if p.is_file():\n"
            "    base = json.loads(p.read_text() or '{}')\n"
            "dirs = list(site.getsitepackages()) if hasattr(site, 'getsitepackages') else []\n"
            "try:\n"
            "    dirs.append(site.getusersitepackages())\n"
            "except Exception:\n"
            "    pass\n"
            "removed = []\n"
            "for d in dirs:\n"
            "    root = pathlib.Path(d)\n"
            "    if not root.is_dir():\n"
            "        continue\n"
            "    cands = [root / 'sitecustomize.py', root / 'usercustomize.py']\n"
            "    cands += list(root.glob('*.pth'))\n"
            "    for f in cands:\n"
            "        if not f.is_file():\n"
            "            continue\n"
            "        digest = hashlib.sha256(f.read_bytes()).hexdigest()\n"
            "        if base.get(str(f)) == digest:\n"
            "            continue\n"
            "        if f.suffix == '.pth':\n"
            "            body = f.read_text(errors='replace')\n"
            "            if 'import ' not in body:\n"
            "                continue\n"
            "        f.unlink()\n"
            "        removed.append(str(f))\n"
            "tampered = []\n"
            "for d in dirs:\n"
            "    root = pathlib.Path(d)\n"
            "    if not root.is_dir():\n"
            "        continue\n"
            "    for rec in root.glob('*.dist-info/RECORD'):\n"
            "        for line in rec.read_text(errors='replace').splitlines():\n"
            "            parts = line.rsplit(',', 2)\n"
            "            if len(parts) != 3 or not parts[1].startswith('sha256='):\n"
            "                continue\n"
            "            target = root / parts[0]\n"
            "            if not target.is_file() or target.suffix not in ('.py',):\n"
            "                continue\n"
            "            import base64\n"
            "            want = parts[1][len('sha256='):]\n"
            "            got = base64.urlsafe_b64encode(\n"
            "                hashlib.sha256(target.read_bytes()).digest()).decode().rstrip('=')\n"
            "            if got != want:\n"
            "                tampered.append(str(target))\n"
            # ★ 还有一条不碰任何文件的通路：把 PYTHONPATH 指向一个假模块目录，
            #   靠 sys.path 顺序让影子模块赢。site-packages 里一个字节都没改，
            #   前面两项检查都看不见。这里把现场一并报出来。
            "import os, sys\n"
            "shadow = []\n"
            "for name in ('torch',):\n"
            "    try:\n"
            "        import importlib.util as u\n"
            "        spec = u.find_spec(name)\n"
            "    except Exception:\n"
            "        continue\n"
            "    origin = getattr(spec, 'origin', '') or ''\n"
            "    if origin and not any(origin.startswith(d) for d in dirs):\n"
            "        shadow.append({'module': name, 'origin': origin})\n"
            "import shutil\n"
            "interp = []\n"
            "for name in ('python', 'python3'):\n"
            "    exe = shutil.which(name)\n"
            "    if not exe:\n"
            "        continue\n"
            "    real = str(pathlib.Path(exe).resolve())\n"
            "    blob = pathlib.Path(real).read_bytes()\n"
            "    now = '%%s|%%s|%%s' %% (real, blob[:4] == b'\\x7fELF',\n"
            "                        hashlib.sha256(blob).hexdigest())\n"
            "    was = base.get('interp:' + name)\n"
            "    if was and was != now:\n"
            "        interp.append({'name': name, 'was': was, 'now': now})\n"
            "print(json.dumps({'removed': removed, 'tampered': tampered[:20],\n"
            "                  'interpreter': interp,\n"
            "                  'shadow': shadow,\n"
            "                  'pythonpath': os.environ.get('PYTHONPATH', ''),\n"
            "                  'pythonstartup': os.environ.get('PYTHONSTARTUP', ''),\n"
            "                  'syspath_head': sys.path[:4]}))\n"
            "PY" % _HOOK_BASELINE_PATH
        )

        def create_container(self, repo_url: str, revision: str = "") -> str:
            cid = super().create_container(repo_url, revision)
            try:
                self._exec_container(self._SNAPSHOT_HOOKS, timeout=180)
            except Exception:
                # 基线拿不到就退化成「不删任何东西」，绝不因为守卫本身让实验失败。
                logger.warning("判据守卫：注入点基线快照失败，本轮不做清理", exc_info=True)
            self._restore_resume_state()
            return cid

        def _restore_resume_state(self) -> None:
            """把上一台机器上存下的训练进度铺回新容器。

            ★ 这一步能成立，全靠升级走的是**整机镜像**：OOM 收尾时
              `_save_resume_state` 把工作目录拷到主机 `RESUME_HOST_DIR`，
              那个目录随镜像一起来到新机器。RSA 在这里起的是全新容器、
              还会重新 clone，所以必须显式铺回去 —— 旧容器的层数据虽然也在
              镜像里，但没有人会去用它。
            ★ 目录不存在就是「没存过」，静默跳过；失败也只是退回从头训，
              绝不让它把一次本可以继续的升级变成失败。
            """
            script = (
                f"if [ -d {RESUME_HOST_DIR}/repo ]; then "
                f"{self.docker_binary} cp {RESUME_HOST_DIR}/repo/. "
                f"{self.container_id}:{RESUME_CONTAINER_WORKDIR} && echo restored; "
                "else echo absent; fi"
            )
            try:
                result = self._exec_host(self._host(["sh", "-lc", script]), 600)
            except Exception:
                logger.warning("恢复训练进度失败，本轮从头开始", exc_info=True)
                return
            text = (getattr(result, "output", "") or getattr(result, "stdout", "") or "")
            if "restored" in text:
                # warning 而不是 info：这条决定用户是不是白跑一遍训练，
                # 被日志级别过滤掉就等于没有。
                logger.warning("已把上一台机器的训练进度铺回新容器")
            elif "absent" not in text:
                logger.warning("恢复训练进度：结果无法判断 %s", text[:200])

        def _rollback_to_clean_checkpoint(self) -> bool:
            """沿 checkpoint 栈回退，直到环境重新通过完整性检查。

            ★ 一次评分里只做一轮回退。不设上限的话会变成
              「回滚 → 它再篡改 → 再回滚」的循环，每一圈都在烧 GPU 的钱。
            ★ 回退太多会把合法的依赖也丢掉，那时判据会**诚实地失败** ——
              这正是我们要的：配环境 Agent 此时该走资源不足的上报出口
              （见 `_CONTRACT_ADDENDUM`），而不是再想办法伪造。
            """
            snapshots = list(getattr(self, "_snapshots", []) or [])
            for _ in range(len(snapshots)):
                try:
                    if not self.rollback_to_checkpoint(1):
                        logger.error("判据守卫：回滚失败，容器可能已不可用")
                        return False
                except Exception:
                    logger.error("判据守卫：回滚抛异常", exc_info=True)
                    return False
                try:
                    result = self._exec_container(self._SANITIZE, timeout=300)
                    body = (getattr(result, "output", "") or "").strip().splitlines()
                    check = json.loads(body[-1]) if body else {}
                except Exception:
                    logger.warning("判据守卫：回滚后复查失败，停止回退", exc_info=True)
                    return False
                if not (check.get("tampered") or check.get("interpreter")):
                    logger.warning("判据守卫：已回滚到干净的检查点，重新评分")
                    return True
            logger.error("判据守卫：回退到栈底仍不干净")
            return False

        def _read_insufficiency_report(self) -> str:
            """配环境 Agent 上报的「这台机器不够」。没有就返回空串。

            ★ 只有能被解析成**真实的资源不足事实**才作数。否则它写个随手编的
              字符串就能逃掉正常的排错工作 —— 那会把一个诚实出口变成新的偷懒出口。
            """
            from .oom import parse_cuda_oom

            try:
                probe = self._exec_host(
                    self._host(["docker", "exec", self.container_id, "sh", "-lc",
                                f"cat {INSUFFICIENT_SENTINEL} 2>/dev/null || true"]),
                    120,
                )
            except Exception:
                logger.warning("判据守卫：读资源不足上报失败，忽略", exc_info=True)
                return ""
            body = (getattr(probe, "stdout", "") or "").strip()
            if not body:
                return ""
            if parse_cuda_oom(body) is None:
                logger.warning(
                    "判据守卫：收到资源不足上报，但内容解析不出真实的资源事实，"
                    "按未上报处理：%r", body[:200])
                return ""
            return body

        def run_criteria(self, criteria_path, *, command: str, timeout: int,
                         repeats: int, env: dict, workdir: str, snapshot: list):
            reported = self._read_insufficiency_report()
            if reported:
                # 它诚实地说了「这台机器不够」。把原文交给上层 —— detect_oom 扫
                # AgentOutcome.error，命中后产品会带着实测下界去问用户要不要换更大的卡。
                logger.warning("判据守卫：配环境 Agent 上报资源不足，按 OOM 处理")
                raise BridgeError(reported)
            try:
                result = self._exec_container(self._SANITIZE, timeout=300)
                body = (getattr(result, "output", "") or "").strip().splitlines()
                report = json.loads(body[-1]) if body else {}
                if integrity is not None:
                    integrity["checked"] = True
            except Exception as exc:
                # ★ 守卫跑不起来时**继续评分**（不该因为守卫故障判死用户的实验），
                #   但绝不能让「没查过」和「查过且干净」在产品上长得一样 ——
                #   对手要绕过这套东西，最省事的正是打掉守卫而不是斗赢它。
                logger.warning("判据守卫：环境清理失败，按原样评分", exc_info=True)
                if integrity is not None:
                    integrity["checked"] = False
                    integrity["reason"] = f"{type(exc).__name__}: {exc}"[:200]
                report = {}
            removed = report.get("removed") or []
            tampered = report.get("tampered") or []
            shadow = report.get("shadow") or []
            interpreter = report.get("interpreter") or []
            # ★ 每次都记完整现场。上一版只在删了东西时才吭声，于是「什么都没查到」
            #   和「守卫没跑」在日志里长得一模一样 —— 排查时白费了一轮真机。
            # ★ warning 而不是 info：这是安全相关的审计线索，被日志级别过滤掉
            #   就等于没有 —— 排查时「什么都没查到」和「守卫没跑」长得一模一样，
            #   我为此白费了两轮真机。
            logger.warning(
                "判据守卫：removed=%s tampered=%s shadow=%s PYTHONPATH=%r syspath_head=%s",
                removed, tampered, shadow, report.get("pythonpath", ""),
                report.get("syspath_head"),
            )
            if removed:
                logger.warning("判据守卫：清掉了配环境阶段新增的代码注入点 %s", removed)
            if shadow:
                logger.error("判据守卫：实验依赖被 sys.path 顺序顶掉 %s", shadow)
                raise BridgeError(
                    "环境完整性检查未通过：判据依赖的模块来自 site-packages 之外 —— "
                    f"{shadow}。多半是 PYTHONPATH 指向了影子模块目录，判据的结论不可信。"
                    "（见 docs/experiment-agent-design.md §18）"
                )
            if tampered or interpreter:
                # 装完再改库文件、或把解释器换成包装脚本 —— 这两种删不得
                # （删了环境就崩），但绝不能静默通过：假成功比失败更糟。
                #
                # ★ 我们是基于 docker 的：RSA 每轮都在打 checkpoint（`docker commit`，
                #   容器创建时就有 `initial_clone` 兜底），所以正确处置是**回滚到最近一个
                #   干净的检查点再重评**，而不是直接判死整轮 —— 篡改点之前的配环境成果
                #   是好的，没必要连坐。
                logger.error("判据守卫：环境被改过 tampered=%s interpreter=%s",
                             tampered, interpreter)
                if self._rollback_to_clean_checkpoint():
                    return super().run_criteria(
                        criteria_path, command=command, timeout=timeout,
                        repeats=repeats, env=env, workdir=workdir, snapshot=snapshot,
                    )
                raise BridgeError(
                    "环境完整性检查未通过：已安装的包或解释器在装完之后被修改过 —— "
                    f"{(tampered or [])[:3]}{interpreter[:2]}，且回滚不到干净的检查点。"
                    "判据的结论不可信，本轮按失败处理。"
                    "（见 docs/experiment-agent-design.md §18）"
                )
            return super().run_criteria(
                criteria_path, command=command, timeout=timeout, repeats=repeats,
                env=env, workdir=workdir, snapshot=snapshot,
            )

    return GpuAwareRemoteDockerBackend


#: 外壳模式下 rsa 默认在**子进程**里跑（`rsa_child`）；设为 "0" 回到进程内线程。
RSA_SUBPROCESS_ENV = "DEEPEVOL_EXPERIMENT_RSA_SUBPROCESS"


def _rsa_in_subprocess(config) -> bool:
    return bool(getattr(config, "deepevol_shell_endpoint", None)) and (
        os.environ.get(RSA_SUBPROCESS_ENV, "1").strip().lower() not in {"0", "false", "no"}
    )


#: What the rsa child process may inherit.  The worker's environment carries
#: database DSNs, service tokens, the Ed25519 signing keys and the Provider
#: account credentials; none of that belongs to Agent code.  The child needs
#: the interpreter basics, rsa/SetupX's own knobs (its .env.small carries only
#: the shell endpoint), proxies, and the shell egress variables.
RSA_CHILD_ENV_EXACT = frozenset({
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TZ", "TERM",
    "PYTHONPATH", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE", "VIRTUAL_ENV",
    "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
    "DEEPEVOL_RSA_CHILD_LOG_LEVEL", "DEEPEVOL_AGENT_SHELL_URL", "DEEPEVOL_AGENT_SHELL_TOKEN",
})
RSA_CHILD_ENV_PREFIXES = ("RSA_", "SETUPX_", "XPU_", "REMOTE_RELAY_", "OURSYS_", "DEEPEVOL_EXPERIMENT_")
# Never inherited even when a prefix matches (defence in depth).
RSA_CHILD_ENV_DENY_TOKENS = ("SECRET", "PASSWORD", "TOKEN", "API_KEY", "PRIVATE_KEY", "DSN", "DATABASE_URL")


def rsa_child_environment(source: Mapping[str, str]) -> dict[str, str]:
    """The allowlisted subset of ``source`` for the rsa child process."""

    env: dict[str, str] = {}
    for key, value in source.items():
        if key in RSA_CHILD_ENV_EXACT:
            env[key] = value
            continue
        if key.startswith(RSA_CHILD_ENV_PREFIXES) and not any(tok in key.upper() for tok in RSA_CHILD_ENV_DENY_TOKENS):
            env[key] = value
    return env


def _run_rsa_child(config, repo_url: str, instruction: str, revision: str, session_id: str):
    """Run rsa in ``rsa_child`` and rehydrate its JSON result as a duck-typed outcome.

    The child holds only the shell's URL + token.  ``config.deepevol_rsa_process``
    exposes the Popen so the hard-cap path can kill it (a thread could only be
    abandoned).  The result carries the evidence already collected, so the
    parent never touches rsa's object graph.
    """
    import json
    import subprocess
    from types import SimpleNamespace

    from .oom import OomFact

    base_url, token = config.deepevol_shell_endpoint
    job = {
        "shell_base_url": base_url,
        "shell_token": token,
        "store": str(config.store),
        "work": str(config.work),
        "backend": config.backend,
        "max_rounds": int(getattr(config, "max_rounds", 0) or 0),
        "repo_url": repo_url,
        "instruction": instruction,
        "revision": revision,
        "session_id": session_id,
    }
    root = Path(__file__).resolve().parents[4]  # …/apps/v2/agent_engine/experiment → repo root
    env = rsa_child_environment(os.environ)
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    process = subprocess.Popen(
        [sys.executable, "-m", "apps.v2.agent_engine.experiment.rsa_child"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,  # the child's logs join the worker's stderr
        text=True,
        cwd=str(root),
        env=env,
    )
    config.deepevol_rsa_process = process
    try:
        raw, _ = process.communicate(json.dumps(job, ensure_ascii=False))
    finally:
        config.deepevol_rsa_process = None
    if process.returncode != 0 and not raw.strip():
        raise RuntimeError(f"rsa child exited with {process.returncode} and no result")
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"rsa child returned no JSON result: {raw[-400:]!r}") from exc
    if result.get("crashed"):
        raise RuntimeError(f"rsa child crashed: {result.get('error')}\n{result.get('traceback', '')[-1500:]}")
    integrity = getattr(config, "deepevol_integrity", None)
    if isinstance(integrity, dict) and isinstance(result.get("integrity"), dict):
        integrity.update(result["integrity"])
    oom = result.get("oom")
    return SimpleNamespace(
        status=SimpleNamespace(value=str(result.get("status") or "")),
        error=str(result.get("error") or ""),
        pending=SimpleNamespace(message=str(result.get("pending_message") or "")),
        pipeline=SimpleNamespace(escalation_md=str(result.get("escalation_md") or "")),
        deepevol_setup_rounds=list(result.get("setup_rounds") or []),
        deepevol_evidence=dict(result.get("evidence") or {}),
        deepevol_oom=None if not isinstance(oom, dict) else OomFact(**oom),
        deepevol_oom_checked=True,
    )


def _default_rsa_runner(config, repo_url: str, instruction: str, revision: str, session_id: str):
    if _rsa_in_subprocess(config):
        return _run_rsa_child(config, repo_url, instruction, revision, session_id)

    from apps.v2.agent_engine.rsa.agent import RSAAgent, UserInstruction

    # ★ 在这里建 backend，只为把执行底座换成**耐久作业**
    #   （见 RSA_DURABLE_THRESHOLD_SECONDS 的说明）。
    #   `_container_backend()` 在 remote_backend 非空时直接用它，
    #   compile / setup / adjudication 三段共用同一个；建连发生在此刻，
    #   而紧接着的 `RSAAgent.run()` 其 finally 会关掉它。
    shell_endpoint = getattr(config, "deepevol_shell_endpoint", None)
    if shell_endpoint:
        # 外壳模式（进程内）：底座是 ShellRuntime，工厂忽略 target/密码——它们本来就是空的。
        base_url, token = shell_endpoint

        def runtime_factory(**_ignored: Any) -> Any:
            from apps.v2.agent_shell import ShellRuntime

            return ShellRuntime(base_url, token, name="rsa-shell")
    else:
        runtime_factory = _durable_runtime_factory
    config.remote_backend = _gpu_aware_backend_class(
        getattr(config, "deepevol_integrity", None)
    )(
        target=config.remote_target,
        runtime_factory=runtime_factory,
        username=config.remote_username,
        password=config.remote_password,
        private_key=config.remote_private_key,
    )
    bind_round_recorder()
    begin_recording()
    try:
        outcome = RSAAgent(config).run(
            UserInstruction(
                repository=repo_url, instruction=instruction,
                revision=revision, session_id=session_id,
            )
        )
    finally:
        rounds = take_recorded_rounds()
    # The rounds ride on the outcome object so the caller (another thread)
    # gets them without a side channel.
    try:
        outcome.deepevol_setup_rounds = rounds  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - frozen outcome types
        pass
    return outcome


def with_resume_hint(instruction: str, *, resumed: bool) -> str:
    """升级重跑时，把「上一台机器的进度还在」这件事写进指令。

    ★ 光把文件铺回去不够：RSA 拿到的是原样的任务指令，它没有理由去找
      checkpoint，多半会规规矩矩地从头训一遍 —— 那样铺回去的文件就白费了。
    ★ 没保存成时**一个字都不加**：骗它去找一个不存在的 checkpoint，
      只会让它多花几轮去排查为什么找不到。
    """
    if not resumed or not instruction.strip():
        return instruction
    return (
        instruction.rstrip()
        + "\n\n补充：这台机器上保留着上一次运行的工作目录（含已经写出的 "
        "checkpoint）。如果代码支持从 checkpoint 恢复，请**接着上次的进度跑**，"
        "不要从头开始；找不到可用的 checkpoint 再从头跑。"
    )


def format_accrued_cost(elapsed_seconds: float, hourly_price_cny: float) -> str:
    """把「已经花了多少」写成人话。价格未知时**只说时长**，不编数字。"""
    minutes = max(0.0, elapsed_seconds) / 60
    if hourly_price_cny <= 0:
        return f"已经跑了约 {minutes:.0f} 分钟"
    spent = hourly_price_cny * max(0.0, elapsed_seconds) / 3600
    return (
        f"已经跑了约 {minutes:.0f} 分钟、约 ¥{spent:.1f}"
        f"（¥{hourly_price_cny:.1f}/小时）"
    )


def _render_needs_user(
    outcome: Any,
    *,
    elapsed_seconds: float = 0.0,
    hourly_price_cny: float = 0.0,
    hard_cap_seconds: float = 0.0,
) -> str:
    """★ 「机器还留着」这句话必须带上钱和期限。

    挂起等人是本设计里最大的钱坑：人会走开，而机器按秒计费。
    只说「仍在计费」等于把算账留给用户自己做；给出已花金额、费率和
    强制释放时间，他才判断得了要不要现在处理。
    """
    pending = getattr(outcome, "pending", None)
    message = str(getattr(pending, "message", "") or "").strip()
    body = message or "配环境的过程中有件事需要你决定。"
    cost = format_accrued_cost(elapsed_seconds, hourly_price_cny)
    if hard_cap_seconds > 0:
        remaining = max(0.0, hard_cap_seconds - max(0.0, elapsed_seconds)) / 3600
        deadline = f"若一直没人回话，约 {remaining:.1f} 小时后会自动释放。"
    else:
        deadline = ""
    return (
        f"### 需要你确认一下\n\n{body}\n\n"
        f"**机器还留着，仍在计费** —— {cost}。{deadline}"
        "你回话之后接着往下跑；不想继续就告诉我一声，我立刻释放。"
    )


def _oom_of(outcome: Any) -> OomFact | None:
    """The child process already inspected rsa's objects; trust its verdict."""
    if getattr(outcome, "deepevol_oom_checked", False):
        return getattr(outcome, "deepevol_oom", None)
    return detect_oom(outcome)


def _kill_rsa_child(config: Any) -> None:
    process = getattr(config, "deepevol_rsa_process", None)
    if process is None or process.poll() is not None:
        return
    logger.warning("rsa 子进程 pid=%s 仍在跑，按硬上限/取消强制结束", process.pid)
    try:
        process.kill()
    except Exception:  # pragma: no cover - best effort
        logger.debug("rsa child kill failed", exc_info=True)


def _evidence_of(outcome: Any) -> dict[str, Any]:
    """Never let evidence collection turn a finished experiment into a failure."""
    precomputed = getattr(outcome, "deepevol_evidence", None)
    if isinstance(precomputed, dict) and precomputed:
        return precomputed
    try:
        return collect_evidence(
            outcome,
            setup_rounds=getattr(outcome, "deepevol_setup_rounds", None) or [],
        )
    except Exception:  # pragma: no cover - diagnostic only
        logger.warning("experiment evidence: collection failed", exc_info=True)
        return {}


def _evidence_section(evidence: dict[str, Any]) -> str:
    if not evidence:
        return ""
    try:
        rendered = render_evidence(evidence)
    except Exception:  # pragma: no cover - diagnostic only
        logger.warning("experiment evidence: rendering failed", exc_info=True)
        return ""
    return ("\n\n---\n\n" + rendered) if rendered else ""


def _render_outcome(
    outcome: Any,
    *,
    elapsed_seconds: float,
    hourly_price_cny: float = 0.0,
    integrity: dict | None = None,
) -> str:
    status = str(getattr(getattr(outcome, "status", None), "value", "") or
                 getattr(outcome, "status", "") or "")
    minutes = elapsed_seconds / 60
    if status == "success":
        spent = (
            f"这一趟约 ¥{hourly_price_cny * elapsed_seconds / 3600:.1f}。"
            if hourly_price_cny > 0
            else ""
        )
        # ★ 「跑通了」这句话的可信度取决于环境完整性查过没有。
        #   守卫故障时我们照常评分（不因守卫故障判死实验），但必须说出来 ——
        #   否则「没查过」和「查过且干净」在产品上一模一样，
        #   而对手要绕过这套东西，最省事的正是打掉守卫。
        caveat = ""
        if integrity and integrity.get("expected") and not integrity.get("checked"):
            caveat = (
                "\n\n> ⚠️ 这一轮的**环境完整性检查没能执行**"
                f"（{integrity.get('reason') or '原因未知'}）。"
                "判据是过了，但没验证过运行环境有没有被改动过，"
                "所以这个「跑通了」的可信度低于平常。"
            )
        return (
            f"### 跑通了\n\n环境配好并且跑完了，用时约 {minutes:.0f} 分钟。"
            f"**机器已经释放**，不再计费。{spent}{caveat}"
        )
    error = str(getattr(outcome, "error", "") or "").strip()
    escalation = str(getattr(getattr(outcome, "pipeline", None), "escalation_md", "") or "").strip()
    detail = error or escalation or "没有拿到更具体的原因。"
    return (
        f"### 没跑通\n\n用时约 {minutes:.0f} 分钟。\n\n{detail[:2000]}\n\n"
        "**机器已经释放**，不再计费。"
    )

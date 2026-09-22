"""实验线接 Agent 外壳（apps.v2.agent_shell）的胶水。

租机是后端的事（lease backend / Product saga），**用机**是 Agent 的事——
两边通过外壳解耦：租到的机器在这里包成 :class:`RelayTarget` 绑进外壳，
密码只在这个闭包里；rsa / SetupX / 本流程拿到的都是 :class:`ShellRuntime`
（一个 URL + 一个 token）。换机（升级档位）只是再 bind 一次，Agent 无感。
"""

from __future__ import annotations

from typing import Any

from apps.v2.agent_shell import AgentShell, Placement, RelayTarget, ShellRuntime

from .lease import LeaseError, LeaseHandle, RuntimeFactory


def placement_for(handle: LeaseHandle) -> Placement:
    """机器的**无凭据**描述：规格名用于报告与计费桶，不带主机名或密码。"""

    spec = dict(handle.spec or {})
    label = str(spec.get("instance_type") or spec.get("target") or handle.resource_id or "")
    return Placement(label=label)


def shell_runtime_factory(
    shell: AgentShell,
    *,
    secrets: Any = None,
    relay_factory: RuntimeFactory | None = None,
) -> RuntimeFactory:
    """给 :class:`ExperimentLease` 的 ``runtime_factory``。

    每次调用（首次开机、bring-up 重试、升级换机）都把机器重新绑进外壳，再返回
    一条通过外壳走的 ``ShellRuntime``。``secrets`` 是凭据解析器：租约给出的
    handle 只带 ``secret_ref``，密码在外壳绑定机器的闭包里才解出，Agent 侧的
    流程、rsa 配置与日志都见不到它。``relay_factory`` 默认是
    :func:`lease_v2.make_relay_runtime`；测试可以换成造本地 relay 的工厂。
    """

    def factory(handle: LeaseHandle) -> Any:
        if _binds_specs(shell):
            # Sidecar: the machine is described by reference; the sidecar
            # resolves secret_ref from its own secrets file when it binds.
            if handle.password and not handle.secret_ref:
                raise LeaseError("sidecar placements need a secret_ref, not a password")
            shell.bind_target(
                {
                    "kind": "relay",
                    "access_url": handle.access_url,
                    "username": handle.username,
                    "secret_ref": handle.secret_ref,
                    "secret_version": handle.secret_version,
                    "label": placement_for(handle).label,
                },
                start=False,
            )
            return ShellRuntime(shell.base_url, shell.token, name=f"shell:{handle.resource_id}")
        if relay_factory is None:
            from .lease_v2 import make_relay_runtime

            def make_relay(h: LeaseHandle) -> Any:
                return make_relay_runtime(h, secrets=secrets)
        else:
            make_relay = relay_factory
        shell.bind_target(
            RelayTarget(lambda: make_relay(handle), placement=placement_for(handle)),
            start=False,
        )
        return ShellRuntime(shell.base_url, shell.token, name=f"shell:{handle.resource_id}")

    return factory


def _binds_specs(shell: Any) -> bool:
    """A sidecar shell takes placement specs; an in-process one takes targets."""
    from apps.v2.agent_shell import SidecarShell

    return isinstance(shell, SidecarShell)


def shell_endpoint(shell: AgentShell) -> tuple[str, str]:
    """rsa 的 ``RemoteDockerBackend`` 用的执行入口：(base_url, token)。"""

    return shell.base_url, shell.token


__all__ = ["placement_for", "shell_endpoint", "shell_runtime_factory"]

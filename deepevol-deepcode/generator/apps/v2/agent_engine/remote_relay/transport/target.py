"""SSH 目标机器的描述与 access_url 解析。

解析逻辑移植自 `DeepEvol1.0/Agent/DeepEvol/mcp_servers/server_use/core.py:611`
（`parse_access_target`）——它已经能吃各家云厂商给的三种写法，没必要重写：

    ssh -p 2222 root@1.2.3.4        # AutoDL 控制台直接复制的那种
    root@1.2.3.4:2222 / 1.2.3.4     # 裸 host[:port]
    ssh://root@1.2.3.4:2222         # URL

中转层本身不关心机器从哪来（手填、AutoDL、阿里云 ECS），只认一个 `SSHTarget`。
租赁逻辑是上层的事。
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class SSHTarget:
    """一台远端机器的连接凭据。"""

    host: str
    port: int = 22
    username: str = "root"
    password: str | None = None
    private_key: str | None = None
    """私钥内容或路径二选一，与 password 可并存（asyncssh 会依次尝试）。"""
    private_key_path: str | None = None
    passphrase: str | None = None
    known_hosts: str | None = None
    """None 表示不校验 host key。租来的临时机器每次指纹都不同，默认不校验。"""
    connect_timeout: float = 20.0
    keepalive_interval: float = 15.0
    keepalive_count_max: int = 4
    max_sessions: int = 8
    """单连接并发 channel 上限。Aliyun / AutoDL 的 sshd 默认 MaxSessions=10，留 2 个余量。"""
    label: str = ""
    """人类可读标识，只用于日志。"""

    def __post_init__(self) -> None:
        if not str(self.host or "").strip():
            raise ValueError("SSHTarget.host is required")

    @property
    def display(self) -> str:
        base = f"{self.username}@{self.host}:{self.port}"
        return f"{self.label} ({base})" if self.label else base

    def redacted(self) -> dict[str, object]:
        """给日志用的脱敏视图——密码和私钥绝不出现。"""
        return {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "auth": "key" if (self.private_key or self.private_key_path) else ("password" if self.password else "agent"),
            "label": self.label,
        }


def _host_port(value: str, default_port: int = 22) -> tuple[str, int]:
    host = value
    port = default_port
    if value.startswith("[") and "]" in value:  # IPv6 字面量
        host_part, _, rest = value[1:].partition("]")
        host = host_part
        if rest.startswith(":") and rest[1:].isdigit():
            port = int(rest[1:])
    elif ":" in value and value.rsplit(":", 1)[1].isdigit():
        host, port_s = value.rsplit(":", 1)
        port = int(port_s)
    return host.strip(), port


def parse_access_url(
    access_url: str,
    *,
    username: str = "root",
    password: str | None = None,
    **overrides: object,
) -> SSHTarget:
    """把云厂商给的连接串解析成 `SSHTarget`。

    显式传入的 username 只在连接串里没带用户名时生效。
    """
    raw = (access_url or "").strip()
    if not raw:
        raise ValueError("empty access_url")

    if raw.startswith("ssh "):
        tokens = shlex.split(raw)
        port = 22
        target = ""
        i = 1
        while i < len(tokens):
            token = tokens[i]
            if token == "-p" and i + 1 < len(tokens):
                port = int(tokens[i + 1])
                i += 2
                continue
            if token.startswith("-p") and token[2:].isdigit():
                port = int(token[2:])
                i += 1
                continue
            if not token.startswith("-"):
                target = token
            i += 1
        if not target:
            raise ValueError(f"cannot parse SSH target from access_url: {access_url}")
        user = username
        if "@" in target:
            user, target = target.split("@", 1)
        host, parsed_port = _host_port(target, port)
        return SSHTarget(host=host, port=parsed_port, username=user or username, password=password, **overrides)  # type: ignore[arg-type]

    parsed = urlparse(raw)
    if parsed.scheme and parsed.hostname:
        if parsed.scheme.lower() == "rdp":
            raise ValueError("unsupported access_url scheme for remote-relay: rdp")
        return SSHTarget(
            host=parsed.hostname,
            port=parsed.port or 22,
            username=parsed.username or username,
            password=parsed.password or password,
            **overrides,  # type: ignore[arg-type]
        )

    user = username
    target = raw
    if "@" in raw:
        user, target = raw.split("@", 1)
    host, port = _host_port(target)
    if not host:
        raise ValueError(f"cannot parse host from access_url: {access_url}")
    return SSHTarget(host=host, port=port, username=user or username, password=password, **overrides)  # type: ignore[arg-type]

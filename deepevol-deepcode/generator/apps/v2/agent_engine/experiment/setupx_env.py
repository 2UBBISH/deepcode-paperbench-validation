"""把 DeepEvol 的模型端点写成 vendored SetupX 的 `.env.{small,large}`。

**为什么必须走文件，不能直接 export 环境变量。**
SetupX 的 `src/config.py:17` 在 **import 期**用 `override=True` 依次加载 `.env` 与
`.env.local`——先设 `os.environ` 会被 `.env` 覆盖掉，这是 rsa 自己的 docstring 里
明写过的坑。而 rsa 的 `setupx_interop.setupx_configured()` 是读 `<root>/.env.{backend}`
拼出 `.env.local` 的，且 `rsa/pipeline.py` 的三个调用点**都没有传 `extra=`**，
所以那个注入钩子也用不上。唯一干净的入口就是这两份 `.env.{backend}` 文件本身。

**V2 里写进去的不是厂商凭据。** 严格 V2 运行时里 worker 进程不持有任何 Provider
key：SetupX 与 rsa 的 Compiler 都是裸 `httpx.post(f"{base_url}/chat/completions")`
（`setupx/src/llm_engine.py:134`、`rsa/compiler/llm.py:107`），所以由
本 attempt 的 Agent 外壳（`apps.v2.agent_shell`）在 worker 进程内只听回环地址，
把 OpenAI 形状的请求转到本次 run 的 `GatewayChatModel`（租约围栏、计费、
无凭据）。这里的 `LlmTarget` 就是外壳的 `/v1` 端点 + 每 attempt 的 bearer。文件内容仍是
「明文 token」，所以两份文件不入仓、0600、每轮重写。

生成出来的三个键同时服务两边：SetupX 的 `OpenAIConfig` 和 rsa 自己的
`compiler/llm.py:LLM.from_env()` 读的是同一组 `OPENAI_*`。
"""

from __future__ import annotations

import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BACKENDS = ("small", "large")

# SetupX 只有 ark 与 openai 两条路（`src/config.py:load_config`），没有 Anthropic 协议。
_SETUPX_LLM_PROVIDER = "openai"

#: V2 里 SetupX/rsa 看到的「提供商」永远是网关回环壳。
GATEWAY_PROVIDER = "deepevol_gateway"


class SetupxEnvError(RuntimeError):
    """模型配置无法映射成 SetupX 能用的形态。"""


@dataclass(frozen=True)
class LlmTarget:
    """一个 OpenAI 兼容端点。`api_key` 绝不进日志——用 `redacted()`。"""

    provider: str
    model_id: str
    base_url: str
    api_key: str

    def redacted(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "base_url": self.base_url,
            "api_key": f"***{len(self.api_key)}" if self.api_key else "",
        }


def resolve_llm_target(
    model_name: str,
    *,
    base_url: str,
    api_key: str,
    provider: str = GATEWAY_PROVIDER,
) -> LlmTarget:
    """把本次 run 的回环网关端点包成 SetupX 认的 OpenAI 目标。

    V1 版本按会话 `model_policy` 里的厂商表解析 base_url/api_key；V2 的 worker
    不持有厂商凭据，端点只能是进程内的网关回环壳，所以这里只做形状校验。
    """
    model_id = str(model_name or "").strip()
    if not model_id:
        raise SetupxEnvError("实验 Agent 缺少模型名：会话的 model_policy.model 为空")
    url = str(base_url or "").strip().rstrip("/")
    if not url.startswith(("http://127.0.0.1", "http://localhost", "http://[::1]", "https://")):
        raise SetupxEnvError(
            f"SetupX 端点 base_url 必须是回环网关壳或 https（收到 {url!r}）"
        )
    token = str(api_key or "").strip()
    if not token:
        raise SetupxEnvError("SetupX 端点缺少 bearer token（回环网关壳没有签发）")
    return LlmTarget(provider=str(provider or GATEWAY_PROVIDER), model_id=model_id, base_url=url, api_key=token)


def render_backend_env(target: LlmTarget, *, extra: Mapping[str, str] | None = None) -> str:
    """渲染一份 `.env.{backend}`。

    刻意**不写** `DOCKER_BASE_IMAGE` 与 `XPU_DISABLED`：`setupx_interop` 会在拼
    `.env.local` 时把它们追加在本文件内容之后，后写的赢。这里写了只会造成
    「同一个键出现两次、以谁为准要去读 interop 源码」的困惑。
    """
    lines = [
        "# 由 DeepEvol 生成（apps/v2/agent_engine/experiment/setupx_env.py），请勿手改、请勿入仓。",
        "# 端点是本次 run 的网关回环壳：token 一次性、用量按 Gateway 台账计。",
        f"LLM_PROVIDER={_SETUPX_LLM_PROVIDER}",
        f"OPENAI_API_KEY={target.api_key}",
        f"OPENAI_BASE_URL={target.base_url}",
        f"OPENAI_MODEL={target.model_id}",
        "# XPU（向量召回）整条关掉：它要 PostgreSQL + pgvector，而 rsa 本来也强制禁用。",
        "XPU_ENABLED=false",
        "XPU_VECTOR_ENABLED=false",
    ]
    for key, value in (extra or {}).items():
        lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def write_setupx_backend_envs(
    setupx_root: str | Path,
    *,
    small: LlmTarget,
    large: LlmTarget | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """写出 `.env.small` 与 `.env.large`，返回 {backend: path}。

    `large` 省略时两档用同一个模型。**这不是偷懒**：上游的 small/large 是
    「gemma 免费 / deepseek 计费」的二选一，而 DeepEvol 的模型选择由自己的路由和
    计费策略决定，backend 这个 flag 在这里退化成「rsa 记录里写哪一档」而已。
    真要分档就显式传两个 target。
    """
    root = Path(setupx_root)
    if not (root / "src" / "agent.py").exists():
        raise SetupxEnvError(f"{root} 看起来不是 SetupX 检出（缺 src/agent.py）")

    targets = {"small": small, "large": large or small}
    written: dict[str, Path] = {}
    for backend, target in targets.items():
        path = root / f".env.{backend}"
        path.write_text(render_backend_env(target, extra=extra), encoding="utf-8")
        # 文件里是明文 key，别让同机其它用户读到。
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        written[backend] = path
    return written


def clear_stale_env_local(setupx_root: str | Path) -> bool:
    """删掉崩溃留下的 `.env.local`，返回是否真的删了。

    `setupx_configured` 进入时会把已有的 `.env.local` 备份、退出时删除，但**崩溃
    路径盖不住**。而 `.env.local` 是最后加载、优先级最高的那份，残留下来会让整个
    run 静默连到一个死端点——rsa 自己的 docstring 记过这件事（源 checkout 里至今
    躺着一个 `.env.local.stale` 作为前车之鉴）。所以每次起 run 前先清一次。
    """
    path = Path(setupx_root) / ".env.local"
    if not path.exists():
        return False
    path.unlink()
    return True

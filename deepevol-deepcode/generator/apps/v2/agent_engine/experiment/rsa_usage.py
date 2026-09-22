"""从 rsa 的 `AgentOutcome` 里取回 token 用量，供折入计费台账。

rsa 自己已经在数了，我们不重复计量：
- `CompileOutcome.tokens` 来自 `rsa/compiler/llm.py` 的 `Usage`（Compiler 那几次调用）
- `PipelineOutcome.tokens` 来自 `rsa/meter.py` 的 `TokenMeter`（SetupX 的 ReAct 主循环，
  它 wrap 的是 httpx 的 `post`，即所有 agent LLM 流量的单一出口）

两个 dict 的键是兼容的（`prompt_tokens` / `completion_tokens` / `cached_tokens` / `calls`），
所以可以直接归一相加。

## 两处必须小心的地方

**1. `outcome.pipeline` 和 `pipeline_attempts[-1]` 是同一个对象。**
`rsa/agent.py:201` 把 `result` append 进 `pipeline_attempts`，随后又以 `pipeline=result`
传给 `AgentOutcome`。两个都加就是**重复计费**。这里按 `id()` 去重。

**2. 重编译轮次的 compile 用量会丢。**
`RSAAgent._run` 的外层 `while True` 每轮重新赋值 `compiled`，而 `AgentOutcome.compile`
只留最后一轮。所以对于走了 recompile 的 run，本函数**必然少算**前几轮 Compiler 的用量。
这是 rsa 的暴露面限制，不是这里的 bug；`compile_rounds_visible` 字段把它显式标出来，
别在台账上把它当成"精确值"。真要补齐得改 vendored 源码，P0 不做。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_FIELDS = ("prompt_tokens", "completion_tokens", "cached_tokens", "calls")


@dataclass(frozen=True)
class RsaUsage:
    """一次 `RSAAgent.run()` 的 token 用量合计。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    calls: int = 0
    model: str = ""
    backend: str = ""
    #: 本次 outcome 里能看见几轮 compile 的用量。走了 recompile 时它 < 实际轮数，
    #: 意味着合计是**下界**而非精确值。
    compile_rounds_visible: int = 0
    pipeline_attempts: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def billable_input_tokens(self) -> int:
        """缓存命中的输入另计价，所以「新鲜输入」要把它减掉。

        与 `config/models/llm_models.yaml` 的 `cached_input_price_credits_per_1m`
        对应：该字段为 0 时上游按输入全价计，此时调用方直接用 `prompt_tokens` 即可。
        """
        return max(self.prompt_tokens - self.cached_tokens, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "billable_input_tokens": self.billable_input_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
            "model": self.model,
            "backend": self.backend,
            "compile_rounds_visible": self.compile_rounds_visible,
            "pipeline_attempts": self.pipeline_attempts,
        }

    @property
    def is_lower_bound(self) -> bool:
        """走过 recompile 时合计只是下界——见模块 docstring 第 2 点。"""
        return self.compile_rounds_visible > 0 and self.pipeline_attempts > 1


def _as_ints(block: Any) -> dict[str, int]:
    if not isinstance(block, dict):
        return dict.fromkeys(_FIELDS, 0)
    out: dict[str, int] = {}
    for name in _FIELDS:
        try:
            out[name] = max(int(block.get(name) or 0), 0)
        except (TypeError, ValueError):
            out[name] = 0
    return out


def collect_rsa_usage(outcome: Any) -> RsaUsage:
    """把一个 `AgentOutcome` 里所有可见的 token 用量加起来。

    对 `None`、缺字段、字段类型不对都容忍——计费收口不该因为上游多了一个可选字段
    就抛异常，把整个 run 的记账搞丢。
    """
    totals = dict.fromkeys(_FIELDS, 0)
    model = ""
    backend = ""
    compile_rounds = 0
    seen: set[int] = set()

    compiled = getattr(outcome, "compile", None)
    compile_tokens = getattr(compiled, "tokens", None)
    if isinstance(compile_tokens, dict):
        compile_rounds = 1
        model = str(compile_tokens.get("model") or "")
        backend = str(compile_tokens.get("backend") or "")
        for name, value in _as_ints(compile_tokens).items():
            totals[name] += value

    attempts = list(getattr(outcome, "pipeline_attempts", None) or [])
    # `outcome.pipeline` 通常就是 attempts 的最后一个（agent.py:201 append 的同一对象）。
    # 按 id 去重；万一哪天它不在 attempts 里了，这里也能把它捞进来。
    final = getattr(outcome, "pipeline", None)
    if final is not None:
        attempts.append(final)

    counted_attempts = 0
    for attempt in attempts:
        if attempt is None or id(attempt) in seen:
            continue
        seen.add(id(attempt))
        counted_attempts += 1
        for name, value in _as_ints(getattr(attempt, "tokens", None)).items():
            totals[name] += value

    return RsaUsage(
        **totals,
        model=model,
        backend=backend,
        compile_rounds_visible=compile_rounds,
        pipeline_attempts=counted_attempts,
    )

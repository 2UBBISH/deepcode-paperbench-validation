"""C4: the loop honours hooks, callbacks, injections and in-place message replacement."""

from __future__ import annotations

import asyncio
from typing import Any

from apps.v2.agent.paper2code import runner as runner_mod
from apps.v2.agent.paper2code.runner import PaperAgentRunner, truncate_result
from apps.v2.agent_engine.paper2code.seams.agent_runtime import (
    AgentHook,
    AgentRunner,
    AgentRunSpec,
    Tool,
    ToolRegistry,
)
from apps.v2.agent_engine.paper2code.seams.llm_runtime import LLMProvider, LLMResponse, ToolCallRequest


class _Echo(Tool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "echo the text argument back to the caller"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def execute(self, **kwargs: Any) -> Any:
        return "echo:" + kwargs["text"]


class _Scripted(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.seen: list[list[dict]] = []

    def get_default_model(self) -> str:
        return "fake"

    async def chat_with_retry(self, messages, tools=None, model=None, max_tokens=None, temperature=None, reasoning_effort=None, tool_choice=None, retry_mode="standard", on_retry_wait=None):
        self.seen.append([dict(m) for m in messages])
        if not self.responses:
            return LLMResponse(content="(out of script)")
        return self.responses.pop(0)


class _Hook(AgentHook):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []
        self.replaced = False

    async def before_iteration(self, ctx):
        self.calls.append(f"before_iteration:{ctx.iteration}")

    async def before_model_request(self, ctx):
        self.calls.append("before_model_request")

    async def on_model_response(self, ctx):
        self.calls.append("on_model_response")

    async def before_execute_tools(self, ctx):
        self.calls.append(f"before_execute_tools:{len(ctx.tool_calls)}")

    async def after_iteration(self, ctx):
        self.calls.append("after_iteration")
        if ctx.tool_calls and not self.replaced:
            self.replaced = True
            ctx.messages[:] = [ctx.messages[0], {"role": "user", "content": "MEMORY RESET"}]

    def finalize_content(self, ctx, content):
        self.calls.append("finalize_content")
        return content


def _tool_call(i: int) -> LLMResponse:
    return LLMResponse(content=None, tool_calls=[ToolCallRequest(id=f"c{i}", name="echo", arguments={"text": str(i)})], finish_reason="tool_calls")


def _spec(provider, hook=None, **overrides) -> AgentRunSpec:
    registry = ToolRegistry()
    registry.register(_Echo())
    base = {
        "initial_messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}],
        "tools": registry,
        "model": "fake",
        "max_iterations": 10,
        "max_tool_result_chars": 1000,
        "hook": hook,
    }
    base.update(overrides)
    return AgentRunSpec(**base)


def test_two_tool_rounds_injection_and_stop_callback() -> None:
    hook = _Hook()
    injected = {"count": 0}
    stops = {"n": 0}

    async def inject():
        if injected["count"] == 0:
            injected["count"] += 1
            return [{"role": "user", "content": "keep going"}]
        return []

    async def should_stop():
        stops["n"] += 1
        return "all done" if stops["n"] >= 4 else None

    provider = _Scripted([_tool_call(1), _tool_call(2), LLMResponse(content="draft"), LLMResponse(content="final")])
    spec = _spec(provider, hook, injection_callback=inject, should_stop_callback=should_stop)
    result = asyncio.run(PaperAgentRunner(provider).run(spec))

    assert result.stop_reason == "callback_stop"
    assert result.final_content == "all done"
    assert result.tools_used == ["echo", "echo"]
    assert result.had_injections
    assert len(provider.seen) == 3  # stopped before the fourth model request
    assert hook.calls[:6] == [
        "before_iteration:1", "before_model_request", "on_model_response", "before_execute_tools:1", "after_iteration", "before_iteration:2",
    ]
    # the code-memory style replacement after round 1 is what round 2 was sent
    assert provider.seen[1][1] == {"role": "user", "content": "MEMORY RESET"}
    assert provider.seen[1][0]["role"] == "system"
    # tool result appended as a tool message in round 1 before the reset
    assert spec.initial_messages is result.messages


def test_finish_reason_error_ends_run_with_error_message() -> None:
    provider = _Scripted([LLMResponse(content="boom", finish_reason="error")])
    result = asyncio.run(PaperAgentRunner(provider).run(_spec(provider, error_message="ERR")))
    assert result.stop_reason == "error"
    assert result.error == "boom"
    assert result.final_content == "ERR"


def test_max_iterations_tries_injection_first_then_stops() -> None:
    calls = {"n": 0}

    async def inject():
        calls["n"] += 1
        return [] if calls["n"] > 1 else [{"role": "user", "content": "more"}]

    provider = _Scripted([_tool_call(i) for i in range(10)])
    spec = _spec(provider, max_iterations=2, injection_callback=inject, max_iterations_message="cap {max_iterations}")
    result = asyncio.run(PaperAgentRunner(provider).run(spec))
    assert result.stop_reason == "max_iterations"
    assert result.final_content == "cap 2"
    assert calls["n"] == 2
    assert len(provider.seen) == 4


def test_empty_reply_retried_twice_then_reported() -> None:
    provider = _Scripted([LLMResponse(content=""), LLMResponse(content="  "), LLMResponse(content="")])
    result = asyncio.run(PaperAgentRunner(provider).run(_spec(provider)))
    assert result.stop_reason == "empty_final_response"
    assert len(provider.seen) == 3
    assert provider.seen[2][-1]["content"].startswith("Your last reply was empty")


def test_length_continuation_joins_pieces() -> None:
    provider = _Scripted([LLMResponse(content="part one ", finish_reason="length"), LLMResponse(content="part two")])
    result = asyncio.run(PaperAgentRunner(provider).run(_spec(provider)))
    assert result.stop_reason == "completed"
    assert result.final_content == "part one part two"


def test_permission_checker_and_truncation() -> None:
    def checker(name, args):
        return ("deny", "policy") if args.get("text") == "1" else ("allow", "")

    provider = _Scripted([_tool_call(1), _tool_call(2), LLMResponse(content="done")])
    spec = _spec(provider, permission_checker=checker, max_tool_result_chars=8)
    result = asyncio.run(PaperAgentRunner(provider).run(spec))
    tool_msgs = [m for m in result.messages if m["role"] == "tool"]
    assert tool_msgs[0]["content"].startswith("Error: permission denied: policy")
    assert "truncated" in tool_msgs[1]["content"] or tool_msgs[1]["content"] == "echo:2"
    out = truncate_result("a" * 100, 20)
    assert out.startswith("a" * 12)
    assert out.endswith("a" * 8)
    assert "80 characters omitted" in out


def test_bind_installs_on_seam_class() -> None:
    runner_mod.bind()
    provider = _Scripted([LLMResponse(content="ok")])
    result = asyncio.run(AgentRunner(provider).run(_spec(provider)))
    assert result.final_content == "ok"

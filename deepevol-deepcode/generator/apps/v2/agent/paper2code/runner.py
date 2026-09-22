"""``AgentRunner.run``: the tool-calling loop, per docs/INTEGRATION.md §4.

The engine constructs ``seams.agent_runtime.AgentRunner`` directly, so
:func:`bind` assigns :meth:`PaperAgentRunner.run` onto that class. The
loop is the reference loop with the optional pieces filled in: empty final
replies are retried twice, ``finish_reason == "length"`` continues at most
three times and the pieces are joined, per-tool ``timeout_s`` is enforced,
and tool results are truncated head-and-tail. After ``after_iteration`` the
loop keeps using ``context.messages`` — the code-memory hook replaces its
contents in place.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger

from apps.v2.agent_engine.paper2code.seams.agent_runtime import (
    DEFAULT_MAX_ITERATIONS_MESSAGE,
    AgentHook,
    AgentHookContext,
    AgentRunner,
    AgentRunResult,
    AgentRunSpec,
)
from apps.v2.agent_engine.paper2code.seams.llm_runtime import LLMResponse, ToolCallRequest

EMPTY_RESPONSE_RETRIES = 2
LENGTH_CONTINUATIONS = 3
EMPTY_RETRY_PROMPT = "Your last reply was empty. Please provide your final response now."
CONTINUE_PROMPT = (
    "Your previous reply was cut off by the output limit. Continue exactly where "
    "you left off, without repeating anything you already wrote."
)
TRUNCATION_MARKER = "\n\n... [tool result truncated: {dropped} characters omitted] ...\n\n"


def truncate_result(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head = max(int(limit * 0.6), 1)
    tail = max(limit - head, 0)
    dropped = len(text) - head - tail
    return text[:head] + TRUNCATION_MARKER.format(dropped=dropped) + (text[-tail:] if tail else "")


def _assistant_tool_message(response: LLMResponse) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": response.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
            }
            for call in response.tool_calls
        ],
    }


def _accumulate(total: dict[str, int], usage: dict[str, int] | None) -> None:
    for key, value in (usage or {}).items():
        try:
            total[key] = total.get(key, 0) + int(value)
        except (TypeError, ValueError):
            continue


class PaperAgentRunner(AgentRunner):
    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        messages = spec.initial_messages
        hook = spec.hook or AgentHook()
        remaining = spec.max_iterations
        tools_used: list[str] = []
        tool_events: list[dict[str, str]] = []
        usage: dict[str, int] = {}
        iteration = 0
        had_injections = False
        empty_retries = 0
        continuations = 0
        carry = ""

        def _result(stop_reason: str, final_content: str | None, error: str | None = None) -> AgentRunResult:
            return AgentRunResult(
                final_content=final_content,
                messages=messages,
                tools_used=tools_used,
                usage=usage,
                stop_reason=stop_reason,
                error=error,
                tool_events=tool_events,
                had_injections=had_injections,
            )

        async def _drain() -> bool:
            nonlocal had_injections
            if spec.injection_callback is None:
                return False
            try:
                items = await spec.injection_callback()
            except Exception as exc:
                logger.warning("injection_callback raised: {}", exc)
                return False
            items = list(items or [])
            for item in items:
                if isinstance(item, dict) and item.get("content"):
                    messages.append({"role": item.get("role") or "user", "content": item["content"]})
            if items:
                had_injections = True
            return bool(items)

        while True:
            if remaining is not None and remaining <= 0:
                if await _drain():
                    remaining = spec.max_iterations
                    continue
                template = spec.max_iterations_message or DEFAULT_MAX_ITERATIONS_MESSAGE
                try:
                    text = template.format(max_iterations=spec.max_iterations)
                except (KeyError, IndexError, ValueError):
                    text = template
                return _result("max_iterations", text)

            if spec.should_stop_callback is not None:
                try:
                    reason = await spec.should_stop_callback()
                except Exception as exc:
                    logger.warning("should_stop_callback raised: {}", exc)
                    reason = None
                if reason:
                    return _result("callback_stop", str(reason))

            if remaining is not None:
                remaining -= 1
            iteration += 1

            ctx = AgentHookContext(iteration=iteration, messages=messages, usage=usage, response_ordinal=iteration)
            await hook.before_iteration(ctx)
            await hook.before_model_request(ctx)
            tool_definitions = spec.tool_definitions()
            request = self.provider.chat_with_retry(
                messages,
                tools=tool_definitions or None,
                model=spec.model,
                max_tokens=spec.max_tokens,
                temperature=spec.temperature,
                reasoning_effort=spec.reasoning_effort,
                retry_mode=spec.provider_retry_mode,
                on_retry_wait=spec.retry_wait_callback,
            )
            try:
                if spec.llm_timeout_s:
                    response = await asyncio.wait_for(request, timeout=spec.llm_timeout_s)
                else:
                    response = await request
            except asyncio.TimeoutError:
                return _result("error", spec.error_message, error=f"model call exceeded llm_timeout_s={spec.llm_timeout_s}")
            ctx.response = response
            _accumulate(usage, response.usage)
            await hook.on_model_response(ctx)

            if response.finish_reason == "error":
                return _result("error", spec.error_message, error=response.content or "provider error")

            if response.should_execute_tools:
                messages.append(_assistant_tool_message(response))
                ctx.tool_calls = list(response.tool_calls)
                await hook.before_execute_tools(ctx)
                for call in ctx.tool_calls:
                    text = await self._execute_one(spec, call)
                    tools_used.append(call.name)
                    ctx.tool_results.append(text)
                    event = {"tool": call.name, "status": "error" if str(text).startswith("Error") else "ok"}
                    ctx.tool_events.append(event)
                    tool_events.append(event)
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": str(text)})
                    if spec.fail_on_tool_error and str(text).startswith("Error"):
                        await hook.after_iteration(ctx)
                        return _result("tool_error", spec.error_message, error=str(text))
                await hook.after_iteration(ctx)
                messages = ctx.messages
                continue

            content = response.content or ""
            if response.finish_reason == "length" and continuations < LENGTH_CONTINUATIONS and content.strip():
                carry += content
                continuations += 1
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": CONTINUE_PROMPT})
                continue

            clean = hook.finalize_content(ctx, carry + content) if (carry or content) else hook.finalize_content(ctx, content)
            if not (clean or "").strip():
                if empty_retries < EMPTY_RESPONSE_RETRIES:
                    empty_retries += 1
                    messages.append({"role": "user", "content": EMPTY_RETRY_PROMPT})
                    continue
                return _result("empty_final_response", "")
            carry = ""
            clean = str(clean)
            messages.append({"role": "assistant", "content": clean})
            ctx.final_content = clean
            ctx.stop_reason = "completed"
            await hook.after_iteration(ctx)
            messages = ctx.messages
            if await _drain():
                remaining = spec.max_iterations
                continue
            return _result("completed", clean)

    async def _execute_one(self, spec: AgentRunSpec, call: ToolCallRequest) -> str:
        denial = await self._check_permission(spec, call)
        if denial is not None:
            return denial
        tool = spec.tools.get(call.name)
        timeout_s = getattr(tool, "timeout_s", None) if tool is not None else None
        try:
            if timeout_s:
                result = await asyncio.wait_for(spec.tools.execute(call.name, call.arguments), timeout=timeout_s)
            else:
                result = await spec.tools.execute(call.name, call.arguments)
        except asyncio.TimeoutError:
            result = f"Error executing {call.name}: timed out after {timeout_s:.0f}s"
        except Exception as exc:
            result = f"Error executing {call.name}: {exc}"
        return truncate_result(str(result), spec.max_tool_result_chars)

    @staticmethod
    async def _check_permission(spec: AgentRunSpec, call: ToolCallRequest) -> str | None:
        checker = spec.permission_checker
        if checker is None:
            return None
        try:
            decision, reason = checker(call.name, call.arguments)
        except Exception as exc:
            return f"Error: permission denied: permission checker failed ({exc})"
        value = getattr(decision, "value", decision)
        if value == "allow":
            return None
        if value == "ask":
            if spec.approval_callback is None:
                return f"Error: permission denied: {reason or 'approval required and no approver configured'}"
            try:
                approved = await spec.approval_callback(call.name, call.arguments, reason)
            except Exception as exc:
                return f"Error: permission denied: approver failed ({exc})"
            return None if approved else f"Error: permission denied: {reason or 'rejected by approver'}"
        return f"Error: permission denied: {reason or 'denied by policy'}"


def bind() -> None:
    """Install the loop (and its helpers) onto the seam's ``AgentRunner`` (idempotent)."""
    for name, member in vars(PaperAgentRunner).items():
        if name.startswith("__") or name in {"_abc_impl"}:
            continue
        if callable(member) or isinstance(member, (staticmethod, classmethod)):
            setattr(AgentRunner, name, member)


__all__ = ["PaperAgentRunner", "bind", "truncate_result"]

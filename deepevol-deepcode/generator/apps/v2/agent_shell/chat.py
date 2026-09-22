"""One OpenAI-shaped chat completion, answered by a Gateway facade.

Behind the shell's ``/v1/chat/completions``.  The facade is
duck-typed (``facade.chat.completions.create(model=, messages=, **params)``)
so this module has no import on the Agent package.
"""

from __future__ import annotations

import secrets
from typing import Any, Mapping

_FORWARDED = ("temperature", "max_tokens", "max_completion_tokens", "top_p", "response_format")


def complete_chat(
    facade: Any,
    payload: Mapping[str, Any],
    *,
    default_model: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Answer one OpenAI-shaped chat completion through the Gateway facade.

    Shared by the loopback endpoint and the Agent shell so both speak exactly
    the same wire subset.  Returns the response body and the normalized usage
    (``input_tokens`` / ``output_tokens`` / ``total_tokens``).
    """

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    for message in messages:
        if not isinstance(message, Mapping) or "role" not in message:
            raise ValueError("every message needs a role")
    if payload.get("stream"):
        raise ValueError("streaming is not supported by the loopback endpoint")
    params = {key: payload[key] for key in _FORWARDED if key in payload}
    requested = str(payload.get("model") or default_model)
    response = facade.chat.completions.create(model=requested, messages=messages, **params)
    text = response.choices[0].message.content
    usage = _usage_dict(getattr(response, "usage", None))
    body = {
        "id": f"loop-{secrets.token_hex(8)}",
        "object": "chat.completion",
        "model": requested,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage["input_tokens"],
            "completion_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
        },
    }
    return body, usage


def _usage_dict(value: Any) -> dict[str, int]:
    result = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if isinstance(value, Mapping):
        for key in result:
            try:
                result[key] = int(value.get(key, 0) or 0)
            except (TypeError, ValueError):
                result[key] = 0
    return result



__all__ = ["complete_chat"]

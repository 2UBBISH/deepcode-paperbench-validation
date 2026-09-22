from __future__ import annotations

import functools
import json
import os
from typing import Any, Literal, Unpack

import openai
import structlog
import tiktoken
from openai import NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.completion_usage import CompletionUsage
from preparedness_turn_completer.turn_completer import TurnCompleter
from preparedness_turn_completer.utils import (
    RetryConfig,
    get_model_context_window_length,
    warn_about_non_empty_params,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = structlog.stdlib.get_logger(component=__name__)


def _schema_example(prop: dict) -> object:
    """A placeholder value of a property's type, for the one-line example in json_object mode."""
    t = prop.get("type")
    return {"boolean": True, "integer": 0, "number": 0.0, "string": "...", "array": [], "object": {}}.get(t, "...")


class OpenAICompletionsTurnCompleter(TurnCompleter):
    def __init__(
        self,
        model: str,
        reasoning_effort: Literal["low", "medium", "high"] | None | NotGiven = NOT_GIVEN,
        response_format: type[BaseModel] | NotGiven = NOT_GIVEN,
        temperature: float | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = NOT_GIVEN,
        top_p: float | None | NotGiven = NOT_GIVEN,
        tools: list[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        retry_config: RetryConfig | None = None,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.response_format = response_format
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.tools = tools
        self.tool_choice = tool_choice
        self.encoding_name: str
        self.retry_config = retry_config or RetryConfig()
        try:
            self.encoding_name = tiktoken.encoding_name_for_model(model)
        except KeyError:
            # Fallback to o200k_base
            logger.warning(f"Model {model} not found in tiktoken, using o200k_base")
            self.encoding_name = "o200k_base"
        self.n_ctx: int = get_model_context_window_length(model)

    class Config(TurnCompleter.Config):
        """
        Completion configuration. Non-exhaustive.
        Add more configuration options as needed, in a backwards-compatible way.
        """

        # needed for NotGiven type hint
        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            json_encoders={NotGiven: lambda v: "NOT_GIVEN"},
        )

        model: str
        reasoning_effort: Literal["low", "medium", "high"] | None | NotGiven = NOT_GIVEN
        response_format: type[BaseModel] | NotGiven = NOT_GIVEN
        temperature: float | None | NotGiven = NOT_GIVEN
        max_tokens: int | None | NotGiven = NOT_GIVEN
        top_p: float | None | NotGiven = NOT_GIVEN
        tools: list[ChatCompletionToolParam] | NotGiven = NOT_GIVEN
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN
        retry_config: RetryConfig = Field(default_factory=RetryConfig)

        def build(self) -> OpenAICompletionsTurnCompleter:
            return OpenAICompletionsTurnCompleter(
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                response_format=self.response_format,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                tools=self.tools,
                tool_choice=self.tool_choice,
                retry_config=self.retry_config,
            )

        @field_validator("*", mode="before")
        @classmethod
        def _decode_not_given(cls: type[OpenAICompletionsTurnCompleter.Config], v: Any) -> Any:
            """
            Turn the string "NOT_GIVEN" back into our sentinel before validation.
            """
            if v == "NOT_GIVEN":
                return NOT_GIVEN
            return v

    class Completion(TurnCompleter.Completion):
        usage: CompletionUsage | None = None

    @functools.cached_property
    def _client(self) -> openai.AsyncClient:
        return openai.AsyncClient()

    def completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> OpenAICompletionsTurnCompleter.Completion:
        raise NotImplementedError("Not implemented, use async_completion instead")

    async def async_completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> OpenAICompletionsTurnCompleter.Completion:
        warn_about_non_empty_params(self, **params)

        # [local compat] DeepSeek's official API (api.deepseek.com) refuses the `json_schema` response_format that
        # `chat.completions.parse` sends ("This response_format type is unavailable now") but honours `json_object`.
        # With PB_STRUCTURED_JSON_MODE=json_object the schema goes into a system message and the reply is validated
        # by the caller (the judge runs `model_validate_json` on the content anyway). Default behaviour unchanged.
        json_object_mode = (
            os.environ.get("PB_STRUCTURED_JSON_MODE", "").strip().lower() == "json_object"
            and not isinstance(self.response_format, NotGiven)
            and self.response_format is not None
        )
        # [local compat] PB_COMPLETER_EXTRA_BODY: a JSON object merged into every request body (e.g.
        # {"enable_thinking": false} — SiliconFlow honours it for DeepSeek-V4-Flash, 2026-09-21 probe; PaperBench has no
        # thinking switch of its own). Empty = nothing added, default behaviour unchanged.
        # It applies to the judge's free-text calls only: the structured parser (json_object_mode) keeps the server's
        # default — with thinking off, Flash's parser declared 23 of 178 JudgeEval leaves "no valid score" although the
        # judge text carried "Score: 0" (2026-09-21).
        # `extra_body` is `Body | None`, not NotGiven: passing the sentinel raised "'NotGiven' object is not a mapping"
        # inside the SDK and voided every parser call of the 09-21 13:41 JudgeEval run (178/178 invalid).
        extra_body = (json.loads(os.environ.get("PB_COMPLETER_EXTRA_BODY") or "{}") or None) if not json_object_mode else None
        async for attempt in self.retry_config.build():
            with attempt:
                if json_object_mode:
                    schema = self.response_format.model_json_schema()  # type: ignore[union-attr]
                    guided = [
                        # "an instance, not the schema": DeepSeek-V4-Pro echoed the schema itself back for 2 of 178 JudgeEval
                        # parser calls (2026-09-21) — say so and show the shape of a valid reply
                        {"role": "system", "content": "Reply with ONE JSON object that is an INSTANCE of the following JSON schema (fill in the fields; do not reply with the schema itself, no other keys, no prose):\n" + json.dumps(schema) + "\nExample of the expected shape: " + json.dumps({k: _schema_example(v) for k, v in (schema.get("properties") or {}).items()})},
                        *conversation,
                    ]
                    completion = await self._client.chat.completions.create(
                        model=self.model,
                        messages=guided,  # type: ignore[arg-type]
                        response_format={"type": "json_object"},
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        extra_body=extra_body,
                    )
                else:
                    completion = await self._client.chat.completions.parse(
                        model=self.model,
                        messages=conversation,
                        reasoning_effort=self.reasoning_effort,
                        response_format=self.response_format,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        tools=self.tools,
                        tool_choice=self.tool_choice,
                        extra_body=extra_body,
                    )
        assert isinstance(completion, ChatCompletion)
        return OpenAICompletionsTurnCompleter.Completion(
            input_conversation=conversation,
            output_messages=[completion.choices[0].message],
            usage=completion.usage,
        )

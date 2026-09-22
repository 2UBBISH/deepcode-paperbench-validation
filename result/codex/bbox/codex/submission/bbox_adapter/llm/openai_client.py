"""Azure OpenAI / OpenAI clients used for GPT-3.5-Turbo, GPT-4 and davinci-002.

Credentials are read from environment variables so that no secret ever enters
the repository:

* Azure: ``AZURE_OPENAI_ENDPOINT``, ``AZURE_OPENAI_API_KEY`` and
  ``AZURE_OPENAI_API_VERSION`` (or an explicit ``api_version`` in the config).
* OpenAI: ``OPENAI_API_KEY`` (optionally ``OPENAI_BASE_URL``).

``gpt-3.5-turbo`` is the black-box LLM that BBOX-ADAPTER adapts, ``gpt-4`` is
the rater that provides AI feedback (Appendix G) and ``davinci-002`` is the
OpenAI base model used for the plug-and-play experiments (Table 3).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from .base import BlackBoxLLM, Generation, LLMResult, retry_call
from .cache import ResponseCache


class OpenAIClient(BlackBoxLLM):
    def __init__(
        self,
        name: str = "gpt-3.5-turbo",
        provider: str = "azure",
        deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        endpoint_env: str = "AZURE_OPENAI_ENDPOINT",
        api_key_env: str = "AZURE_OPENAI_API_KEY",
        max_retries: int = 6,
        cache_dir: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> None:
        super().__init__(name=name, max_new_tokens=max_new_tokens)
        self.provider = provider
        self.deployment = deployment or name
        self.api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION")
        self.endpoint_env = endpoint_env
        self.api_key_env = api_key_env
        self.max_retries = max_retries
        self.cache = ResponseCache(cache_dir)
        self._client = None

    # ------------------------------------------------------------- clients
    def _lazy_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import AzureOpenAI, OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install openai>=1.0 to use the OpenAI clients") from exc

        if self.provider == "azure":
            endpoint = os.environ.get(self.endpoint_env)
            api_key = os.environ.get(self.api_key_env)
            if not endpoint or not api_key:
                raise RuntimeError(
                    f"Azure credentials missing: set {self.endpoint_env} and {self.api_key_env}"
                )
            self._client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=self.api_version or "2023-12-01-preview",
            )
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OpenAI credentials missing: set OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")
            self._client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        return self._client

    # ------------------------------------------------------------ helpers
    @property
    def is_chat_model(self) -> bool:
        return not self.name.startswith("davinci")

    def _call(self, prompt: str, n: int, temperature: float, top_p: float,
              max_new_tokens: int, stop: Optional[Sequence[str]],
              system_prompt: Optional[str], seed: Optional[int]) -> List[Generation]:
        client = self._lazy_client()
        stop = list(stop) if stop else None
        params: Dict[str, Any] = {
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
        }
        if stop:
            params["stop"] = stop
        if seed is not None:
            params["seed"] = seed

        if self.is_chat_model:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(model=self.deployment, messages=messages, **params)
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            per_call_completion = completion_tokens / max(1, len(response.choices))
            return [
                Generation(
                    text=choice.message.content or "",
                    prompt_tokens=prompt_tokens if index == 0 else 0,
                    completion_tokens=int(per_call_completion),
                    finish_reason=choice.finish_reason,
                )
                for index, choice in enumerate(response.choices)
            ]

        response = client.completions.create(model=self.deployment, prompt=prompt, **params)
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        per_call_completion = completion_tokens / max(1, len(response.choices))
        return [
            Generation(
                text=choice.text or "",
                prompt_tokens=prompt_tokens if index == 0 else 0,
                completion_tokens=int(per_call_completion),
                finish_reason=getattr(choice, "finish_reason", None),
            )
            for index, choice in enumerate(response.choices)
        ]

    # --------------------------------------------------------------- main
    def _generate(self, prompts, n, temperature, top_p, max_new_tokens, stop,
                  system_prompt, seed) -> List[LLMResult]:
        results: List[LLMResult] = []
        for prompt in prompts:
            params = {
                "n": n,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_new_tokens,
                "stop": list(stop) if stop else None,
                "system": system_prompt,
                "seed": seed,
            }
            cache_key = self.cache.key(self.name, prompt, params)
            cached = self.cache.get(cache_key)
            if cached is not None:
                results.append(
                    LLMResult(
                        prompt=prompt,
                        generations=[Generation(**gen) for gen in cached],
                    )
                )
                continue
            generations = retry_call(
                lambda: self._call(
                    prompt, n, temperature, top_p, max_new_tokens, stop,
                    system_prompt, seed,
                ),
                max_retries=self.max_retries,
            )
            if not generations:  # never return an empty candidate set
                generations = [Generation(text="")]
            self.cache.put(cache_key, [g.__dict__ for g in generations])
            results.append(LLMResult(prompt=prompt, generations=generations))
        return results

"""Black-box LLM clients.

BBOX-ADAPTER only ever reads the *text* produced by the black-box LLM: neither
model parameters nor output token probabilities are used (Section 2, Table 1).
All clients therefore expose a single primitive, ``generate``, returning plain
strings plus token accounting for the cost analysis (Table 4).
"""

from __future__ import annotations

from typing import Any, Optional

from .base import BlackBoxLLM, Generation, LLMResult
from .mock import MockLLM

__all__ = [
    "BlackBoxLLM",
    "Generation",
    "LLMResult",
    "MockLLM",
    "build_llm",
]


def build_llm(config: Any, cache_dir: Optional[str] = None) -> BlackBoxLLM:
    """Instantiate the client described by ``config`` (:class:`LLMConfig`)."""

    provider = (config.provider or "azure").lower()
    if provider == "mock":
        from .mock import MockLLM

        return MockLLM(name=config.name, max_new_tokens=config.max_new_tokens)
    if provider in {"azure", "openai"}:
        from .openai_client import OpenAIClient

        return OpenAIClient(
            name=config.name,
            provider=provider,
            deployment=config.deployment,
            api_version=config.api_version,
            endpoint_env=config.endpoint_env,
            api_key_env=config.api_key_env,
            max_retries=config.max_retries,
            cache_dir=cache_dir,
        )
    if provider in {"huggingface", "hf", "local"}:
        from .hf_client import HuggingFaceClient

        return HuggingFaceClient(
            name=config.name,
            dtype=config.dtype,
            device_map=config.device_map,
            cache_dir=cache_dir,
        )
    raise ValueError(f"Unknown LLM provider {config.provider!r}")

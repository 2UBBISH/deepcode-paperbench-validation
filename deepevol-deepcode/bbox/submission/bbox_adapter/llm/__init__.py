"""Black-box LLM interface for BBox-Adapter.

This package owns *everything* that talks to (or reasons about) the frozen
black-box proposal generator:

* :mod:`bbox_adapter.llm.blackbox_client` -- text-only clients for Azure OpenAI
  ``gpt-3.5-turbo`` / ``davinci-002``, local HuggingFace ``Mixtral-8x7B-v0.1``,
  a deterministic offline mock, and the ``gpt-4`` AI-feedback rater.
* :mod:`bbox_adapter.llm.prompts` -- the Appendix-J generator prompts, the
  Appendix-G rater prompts, and the sentence-level continuation prompt.
* :mod:`bbox_adapter.llm.token_cost` -- token accounting that converts Azure /
  OpenAI usage statistics into the ``$/1k questions`` numbers of Table 4.

The package enforces the paper's black-box contract (§4.1, Appendix C): the
adapter never requests or consumes log-probabilities, hidden states or
gradients of the black-box LLM -- only raw text strings are exchanged, and
``assert_text_only_payload`` validates every outgoing request payload.

The factory :func:`get_llm` mirrors the naming style used by the other
sub-packages (``data.get_dataset``, ``adapter.get_adapter``, ``inference``) so
scripts can build a proposal generator with one call.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bbox_adapter.llm")

__all__: List[str] = []


def _extend(names: Optional[List[str]]) -> None:
    """Append *names* to ``__all__`` without introducing duplicates."""
    if not names:
        return
    for name in names:
        if name not in __all__:
            __all__.append(name)


# --------------------------------------------------------------------------- #
# blackbox_client (always available -- stdlib + optional requests/torch)
# --------------------------------------------------------------------------- #
_BLACKBOX_AVAILABLE = False
try:  # pragma: no cover - import guard
    from .blackbox_client import (  # noqa: F401
        AZURE_API_VERSION_DEFAULT,
        CHAT_MODELS,
        COMPLETION_MODELS,
        DEFAULT_MAX_LEN,
        DEFAULT_TEMPERATURES,
        FORBIDDEN_REQUEST_KEYS,
        RATER_MODEL,
        TEMPERATURE_BBOX,
        TEMPERATURE_SFT,
        TEMPERATURE_TOXIGEN,
        AzureOpenAIClient,
        BlackBoxClient,
        BlackBoxError,
        GenerationConfig,
        GenerationResult,
        GPT4Rater,
        HuggingFaceClient,
        MockLLMClient,
        assert_text_only_payload,
        build_client,
        build_generator,
        build_rater,
        estimate_tokens,
        is_chat_model,
        resolve_temperature,
    )

    _BLACKBOX_AVAILABLE = True
    _extend(
        [
            "BlackBoxClient",
            "AzureOpenAIClient",
            "HuggingFaceClient",
            "MockLLMClient",
            "GPT4Rater",
            "BlackBoxError",
            "GenerationConfig",
            "GenerationResult",
            "build_client",
            "build_generator",
            "build_rater",
            "assert_text_only_payload",
            "resolve_temperature",
            "is_chat_model",
            "estimate_tokens",
            "DEFAULT_MAX_LEN",
            "TEMPERATURE_BBOX",
            "TEMPERATURE_SFT",
            "TEMPERATURE_TOXIGEN",
            "DEFAULT_TEMPERATURES",
            "FORBIDDEN_REQUEST_KEYS",
            "CHAT_MODELS",
            "COMPLETION_MODELS",
            "RATER_MODEL",
            "AZURE_API_VERSION_DEFAULT",
        ]
    )
except Exception as exc:  # pragma: no cover - defensive
    logger.debug("llm.blackbox_client unavailable: %s", exc)


# --------------------------------------------------------------------------- #
# prompts (always available -- pure string templating)
# --------------------------------------------------------------------------- #
_PROMPTS_AVAILABLE = False
try:  # pragma: no cover - import guard
    from .prompts import (  # noqa: F401
        ANSWER_TERMINATOR,
        BEST_ANSWER_OUTPUT_FORMAT,
        GSM8K_EXAMPLES,
        GSM8K_INSTRUCTION,
        NO_INSTRUCTION_MODELS,
        PROMPT_SPECS,
        RANKING_OUTPUT_FORMAT,
        RATER_CRITERIA,
        RATER_SYSTEM_PROMPT,
        SCIENCEQA_EXAMPLES,
        SCIENCEQA_INSTRUCTION,
        STRATEGYQA_EXAMPLES,
        STRATEGYQA_INSTRUCTION,
        TOXIGEN_INSTRUCTION,
        TRUTHFULQA_INSTRUCTION,
        FewShotExample,
        PromptSpec,
        build_ai_feedback_prompt,
        build_continuation_prompt,
        build_gsm8k_prompt,
        build_prompt,
        build_scienceqa_prompt,
        build_stacked_examples_prompt,
        build_strategyqa_prompt,
        build_toxigen_prompt,
        build_truthfulqa_prompt,
        build_truthfulqa_ranking_prompt,
        format_candidate_block,
        format_choices,
        format_example,
        num_shots,
        resolve_prompt_spec,
        uses_instruction,
    )

    _PROMPTS_AVAILABLE = True
    _extend(
        [
            "FewShotExample",
            "PromptSpec",
            "PROMPT_SPECS",
            "resolve_prompt_spec",
            "num_shots",
            "uses_instruction",
            "format_choices",
            "format_example",
            "build_prompt",
            "build_strategyqa_prompt",
            "build_gsm8k_prompt",
            "build_scienceqa_prompt",
            "build_truthfulqa_prompt",
            "build_toxigen_prompt",
            "build_stacked_examples_prompt",
            "build_continuation_prompt",
            "format_candidate_block",
            "build_ai_feedback_prompt",
            "build_truthfulqa_ranking_prompt",
            "ANSWER_TERMINATOR",
            "NO_INSTRUCTION_MODELS",
            "STRATEGYQA_INSTRUCTION",
            "GSM8K_INSTRUCTION",
            "SCIENCEQA_INSTRUCTION",
            "TRUTHFULQA_INSTRUCTION",
            "TOXIGEN_INSTRUCTION",
            "STRATEGYQA_EXAMPLES",
            "GSM8K_EXAMPLES",
            "SCIENCEQA_EXAMPLES",
            "RATER_CRITERIA",
            "RATER_SYSTEM_PROMPT",
            "BEST_ANSWER_OUTPUT_FORMAT",
            "RANKING_OUTPUT_FORMAT",
        ]
    )
except Exception as exc:  # pragma: no cover - defensive
    logger.debug("llm.prompts unavailable: %s", exc)


# --------------------------------------------------------------------------- #
# token_cost (Table 4 cost accounting)
# --------------------------------------------------------------------------- #
_TOKEN_COST_AVAILABLE = False
try:  # pragma: no cover - import guard
    from .token_cost import (  # noqa: F401
        DEFAULT_PRICING_MODEL,
        PRICING,
        TokenCounter,
        TokenLedger,
        TokenStats,
        compute_costs,
        cost_per_1k_questions,
        price_tokens,
        to_dollars,
    )

    _TOKEN_COST_AVAILABLE = True
    _extend(
        [
            "TokenLedger",
            "TokenCounter",
            "TokenStats",
            "PRICING",
            "DEFAULT_PRICING_MODEL",
            "price_tokens",
            "to_dollars",
            "cost_per_1k_questions",
            "compute_costs",
        ]
    )
except Exception as exc:  # pragma: no cover - defensive
    logger.debug("llm.token_cost unavailable: %s", exc)


# --------------------------------------------------------------------------- #
# convenience factory
# --------------------------------------------------------------------------- #
def get_llm(
    model: str = "gpt-3.5-turbo",
    *,
    kind: str = "bbox",
    ledger: Any = None,
    allow_mock: bool = True,
    **kwargs: Any,
) -> Any:
    """Build a black-box proposal generator (or the ``gpt-4`` rater).

    Parameters
    ----------
    model:
        Model identifier, e.g. ``"gpt-3.5-turbo"``, ``"davinci-002"``,
        ``"mistralai/Mixtral-8x7B-v0.1"`` or ``"gpt-4"``.
    kind:
        ``"bbox"`` (paper temperature 1.0), ``"sft"``/``"cot"`` (temperature
        0.0), ``"toxigen"`` (temperature 0.7) or ``"rater"`` (temperature 0.0).
    ledger:
        Optional token ledger that receives ``add(prompt_tokens=...,
        completion_tokens=..., model=..., n=...)`` calls (Table 4).
    allow_mock:
        Fall back to :class:`MockLLMClient` when Azure credentials or the
        HuggingFace stack are unavailable; keeps the whole pipeline runnable
        offline.

    Returns
    -------
    BlackBoxClient
        A client exposing ``generate(prompt, n=..., temperature=..., max_len=...)``
        returning raw text only.
    """
    if not _BLACKBOX_AVAILABLE:  # pragma: no cover - defensive
        raise ImportError(
            "bbox_adapter.llm.blackbox_client is unavailable; "
            "install the package requirements and retry."
        )
    return build_client(
        model, kind=kind, ledger=ledger, allow_mock=allow_mock, **kwargs
    )


def build_ledger(config: Optional[Any] = None, **kwargs: Any) -> Any:
    """Construct a token/cost ledger for Table-4 accounting.

    Prefers :class:`bbox_adapter.eval.cost.CostLedger` (which understands the
    ``cost:`` YAML section and the gpt-3.5-turbo-1106 prices) and falls back to
    the lightweight :class:`TokenLedger` defined in this package.
    """
    try:  # pragma: no cover - optional dependency
        from ..eval.cost import CostLedger  # noqa: WPS433

        if config is not None:
            try:
                return CostLedger.from_config(config)
            except Exception:
                pass
        return CostLedger(**kwargs)
    except Exception:
        pass

    if _TOKEN_COST_AVAILABLE:
        try:
            return TokenLedger.from_config(config) if config is not None else TokenLedger(**kwargs)
        except Exception:
            try:
                return TokenLedger(**kwargs)
            except Exception:
                return None
    return None


def describe() -> Dict[str, Any]:
    """Return metadata about the black-box layer (used in run headers)."""
    info: Dict[str, Any] = {
        "module": "bbox_adapter.llm",
        "paper": "Lightweight Adapting for Black-Box Large Language Models",
        "sections": {
            "blackbox_client": "4.1, 4.3, H.2, Appendix C, Appendix E",
            "prompts": "Appendix J, Appendix G, 4.1, H.2",
            "token_cost": "4.4, Table 4",
        },
        "available": {
            "blackbox_client": _BLACKBOX_AVAILABLE,
            "prompts": _PROMPTS_AVAILABLE,
            "token_cost": _TOKEN_COST_AVAILABLE,
        },
        "contract": (
            "text-only: no logprobs / hidden states / gradients of the "
            "black-box LLM are ever requested (Appendix C)"
        ),
    }
    if _BLACKBOX_AVAILABLE:
        info["temperatures"] = dict(DEFAULT_TEMPERATURES)
        info["default_max_len"] = DEFAULT_MAX_LEN
        info["forbidden_request_keys"] = list(FORBIDDEN_REQUEST_KEYS)
    if _PROMPTS_AVAILABLE:
        info["prompt_keys"] = sorted(PROMPT_SPECS.keys())
        info["shots"] = {key: spec.n_shot for key, spec in PROMPT_SPECS.items()}
    return info


def _self_test() -> Dict[str, Any]:
    """Dependency-free smoke test (``python -m bbox_adapter.llm``)."""
    results: Dict[str, Any] = {"describe": describe()}

    # Payload guard must reject any probability-bearing request.
    if _BLACKBOX_AVAILABLE:
        try:
            assert_text_only_payload({"model": "gpt-3.5-turbo", "logprobs": 5})
            guard_ok = False
        except ValueError:
            guard_ok = True
        except Exception:
            guard_ok = False
        results["payload_guard_works"] = guard_ok

        temp_ok = (
            resolve_temperature("bbox") == TEMPERATURE_BBOX
            and resolve_temperature("sft") == TEMPERATURE_SFT
            and resolve_temperature("toxigen") == TEMPERATURE_TOXIGEN
        )
        results["temperature_routing_ok"] = temp_ok

        # Offline mock must answer in the datasets' ``####`` format.
        mock = MockLLMClient(seed=0)
        text = mock.generate_one(
            "Example 1:\nQ: Do whales have lungs?\nA: #### Yes.\n\n"
            "Q: Is the sky blue?\nA:",
            temperature=TEMPERATURE_BBOX,
            max_len=64,
        )
        results["mock_generation"] = text[:120]
        results["mock_has_terminator"] = "####" in text

    if _PROMPTS_AVAILABLE:
        results["num_shots"] = {
            "strategyqa": num_shots("strategyqa"),
            "gsm8k": num_shots("gsm8k"),
            "scienceqa": num_shots("scienceqa"),
            "truthfulqa": num_shots("truthfulqa"),
        }
        results["mixtral_drops_instruction"] = (
            not uses_instruction("strategyqa", model="mixtral")
            and uses_instruction("truthfulqa", model="mixtral")
        )

    return results


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import json

    print(json.dumps(_self_test(), indent=2, default=str))

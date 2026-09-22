"""Text-only black-box LLM clients for BBox-Adapter.

BBox-Adapter (Section 3.1-3.3, Section 4.1 Implementations) treats the large
model purely as a *proposal generator*: the adapter ``g_theta`` is the only
component that is ever trained, and it only ever reads the black-box model's
**raw text** outputs.  Appendix C explains why: token-level probabilities are
either truncated (``logprobs`` returns only the top-5 tokens per position) or
deprecated (``echo`` in the legacy Completion API was removed on 2023-10-05), so
they are not usable for adaptation.  Consequently this module deliberately:

* sends only plain text request fields (``prompt``/``messages``, ``n``,
  ``temperature``, ``max_tokens``, ...),
* never sends ``logprobs``/``top_logprobs``/``echo``/``logit_bias``, and
* asserts on every request payload that those keys are absent
  (:func:`assert_text_only_payload`) so the "black-box" contract of the paper is
  enforced by construction rather than by convention.

Backends implemented:

``AzureOpenAIClient``
    Azure OpenAI chat (``gpt-3.5-turbo``) and legacy completion
    (``davinci-002``) endpoints, addressed over the REST API.  Both black-box
    LLMs of Section 4.1 are served this way.
``HuggingFaceClient``
    ``mistralai/Mixtral-8x7B-v0.1`` loaded locally with ``transformers``
    (Section 4.1 Implementations, Section H.2).
``MockLLMClient``
    Deterministic offline generator used when credentials are missing or for
    unit tests / dry runs.  It mimics each dataset's answer format so that the
    whole adaptation pipeline can be exercised without network access.

Generation settings follow Section H.2 exactly: maximum generation length 512
and temperature 1.0 for BBox-Adapter proposal generation, temperature 0.0 for
the SFT baselines (to "avoid instability in performance"), and temperature 0.7
for the ToxiGen extension of Appendix E.  A ``gpt-4`` rater client
(:class:`GPT4Rater`) backs the AI-feedback setting of Section 4.1 / Appendix G.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MAX_LEN",
    "TEMPERATURE_BBOX",
    "TEMPERATURE_SFT",
    "TEMPERATURE_TOXIGEN",
    "AZURE_API_VERSION_DEFAULT",
    "RATER_MODEL",
    "DEFAULT_TEMPERATURES",
    "FORBIDDEN_REQUEST_KEYS",
    "CHAT_MODELS",
    "COMPLETION_MODELS",
    "BlackBoxError",
    "GenerationResult",
    "GenerationConfig",
    "BlackBoxClient",
    "AzureOpenAIClient",
    "HuggingFaceClient",
    "MockLLMClient",
    "GPT4Rater",
    "build_client",
    "build_generator",
    "is_chat_model",
    "assert_text_only_payload",
    "estimate_tokens",
    "resolve_temperature",
]

# --------------------------------------------------------------------------- #
# Paper constants (Section 4.1, Section H.2, Appendix E)
# --------------------------------------------------------------------------- #

DEFAULT_MAX_LEN = 512
"""Maximum generated solution length (Section H.2)."""

TEMPERATURE_BBOX = 1.0
"""Temperature used for BBox-Adapter proposal generation (Section H.2)."""

TEMPERATURE_SFT = 0.0
"""Temperature used for SFT/CoT baselines "to avoid instability" (Section H.2)."""

TEMPERATURE_TOXIGEN = 0.7
"""Temperature used for the Mixtral-8x7B ToxiGen extension (Appendix E)."""

AZURE_API_VERSION_DEFAULT = "2024-02-01"
"""Azure OpenAI REST ``api-version`` used when the environment does not set one."""

RATER_MODEL = "gpt-4"
"""LLM used to simulate human preference for the AI Feedback setting (Section 4.1)."""

DEFAULT_TEMPERATURES: Dict[str, float] = {
    "bbox": TEMPERATURE_BBOX,
    "adapter": TEMPERATURE_BBOX,
    "sft": TEMPERATURE_SFT,
    "cot": TEMPERATURE_SFT,
    "baseline": TEMPERATURE_SFT,
    "toxigen": TEMPERATURE_TOXIGEN,
    "rater": 0.0,
}

FORBIDDEN_REQUEST_KEYS = ("logprobs", "top_logprobs", "echo", "logit_bias")
"""Request fields that would leak token-level information (Appendix C)."""

CHAT_MODELS = (
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4o",
    "gpt-35-turbo",
    "gpt-35-turbo-16k",
    "gpt-4-32k",
)
"""Azure deployment names served through the Chat Completions API."""

COMPLETION_MODELS = (
    "davinci-002",
    "babbage-002",
    "gpt-3.5-turbo-instruct",
    "text-davinci-003",
)
"""Azure deployment names served through the legacy Completions API."""


class BlackBoxError(RuntimeError):
    """Raised when a black-box backend cannot serve a request."""


# --------------------------------------------------------------------------- #
# Request / response containers
# --------------------------------------------------------------------------- #


@dataclass
class GenerationConfig:
    """Sampling configuration for a single :class:`BlackBoxClient` call.

    Parameters mirror Section H.2: ``max_len=512`` and ``temperature=1.0`` for
    BBox-Adapter proposals.
    """

    n: int = 1
    temperature: float = TEMPERATURE_BBOX
    max_len: int = DEFAULT_MAX_LEN
    top_p: float = 1.0
    stop: Optional[Sequence[str]] = None
    seed: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    timeout: float = 120.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "temperature": self.temperature,
            "max_len": self.max_len,
            "top_p": self.top_p,
            "stop": list(self.stop) if self.stop else None,
            "seed": self.seed,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "timeout": self.timeout,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "GenerationConfig":
        if not data:
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class GenerationResult:
    """Raw text outputs plus usage statistics for one generation request.

    ``texts`` holds ``n`` independent samples.  Token counts come from the
    provider's usage block when available (or the local tokenizer for the
    HuggingFace backend) and are estimated otherwise; they feed the cost tables
    of Section 4.4 through :mod:`bbox_adapter.llm.token_cost`.
    """

    texts: List[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    temperature: float = TEMPERATURE_BBOX
    n: int = 1
    finish_reasons: List[str] = field(default_factory=list)
    latency: float = 0.0
    mock: bool = False
    raw: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- convenience ------------------------------------------------------ #
    @property
    def text(self) -> str:
        """First (and most often only) generated text."""
        return self.texts[0] if self.texts else ""

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.texts)

    def __iter__(self):  # pragma: no cover - trivial
        return iter(self.texts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "texts": list(self.texts),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "temperature": self.temperature,
            "n": self.n,
            "finish_reasons": list(self.finish_reasons),
            "latency": self.latency,
            "mock": self.mock,
            "meta": dict(self.meta),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def is_chat_model(model: str) -> bool:
    """Return ``True`` if ``model`` must be served by the Chat Completions API."""
    name = (model or "").lower()
    if any(name.startswith(prefix) for prefix in COMPLETION_MODELS):
        return False
    return any(name.startswith(prefix) for prefix in CHAT_MODELS)


def resolve_temperature(
    mode: Optional[str] = None,
    temperature: Optional[float] = None,
) -> float:
    """Resolve a temperature from an explicit value or a named mode.

    ``mode`` accepts ``"bbox"``/``"adapter"`` (1.0), ``"sft"``/``"cot"`` (0.0)
    and ``"toxigen"`` (0.7), matching Section H.2 and Appendix E.
    """
    if temperature is not None:
        return float(temperature)
    if mode is None:
        return TEMPERATURE_BBOX
    key = str(mode).strip().lower()
    if key in DEFAULT_TEMPERATURES:
        return DEFAULT_TEMPERATURES[key]
    raise ValueError(
        f"Unknown temperature mode {mode!r}; known modes: {sorted(DEFAULT_TEMPERATURES)}"
    )


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Dependency-free token estimate used when a provider omits usage stats."""
    if not text:
        return 0
    return max(1, int(round(len(text) / max(1.0, chars_per_token))))


def assert_text_only_payload(payload: Any, _path: str = "$") -> None:
    """Assert that a request payload carries no token-probability fields.

    This is the code-level guarantee that BBox-Adapter never reads ``logprobs``
    or ``echo`` outputs from the black-box LLM (Appendix C).  Raises
    :class:`ValueError` naming the offending path.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in FORBIDDEN_REQUEST_KEYS and value is not None:
                raise ValueError(
                    f"Black-box request must be text-only: found {_path}.{key} "
                    f"(logprobs/echo are unavailable and unused by BBox-Adapter, "
                    f"see Appendix C)."
                )
            assert_text_only_payload(value, f"{_path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for i, value in enumerate(payload):
            assert_text_only_payload(value, f"{_path}[{i}]")


def _coerce_texts(value: Any) -> List[str]:
    """Normalize a backend response field into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [str(value.get("content", ""))]
    out: List[str] = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            out.append(str(item.get("content", item.get("text", ""))))
        elif isinstance(item, (list, tuple)):  # token id lists -> best effort
            out.append(" ".join(str(t) for t in item))
        else:  # pragma: no cover - defensive
            out.append(str(item))
    return out


# --------------------------------------------------------------------------- #
# Base client
# --------------------------------------------------------------------------- #


class BlackBoxClient:
    """Abstract text-only proposal generator.

    Subclasses implement :meth:`generate_result`.  Every public entry point
    returns plain strings or :class:`GenerationResult` objects.
    """

    model: str = "blackbox"

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        max_len: int = DEFAULT_MAX_LEN,
        temperature: float = TEMPERATURE_BBOX,
        ledger: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> None:
        if model:
            self.model = model
        self.max_len = int(max_len)
        self.temperature = float(temperature)
        self.ledger = ledger
        self.seed = seed
        self.n_calls = 0
        self.n_generations = 0

    # -- API -------------------------------------------------------------- #
    def generate_result(
        self,
        prompt: str,
        n: int = 1,
        temperature: Optional[float] = None,
        max_len: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: Optional[float] = None,
        max_len: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Sample ``n`` raw text completions for ``prompt``.

        Returns a list of ``n`` strings; no probability information is ever
        requested or returned (Appendix C).
        """
        return self.generate_result(
            prompt,
            n=n,
            temperature=temperature,
            max_len=max_len,
            stop=stop,
            seed=seed,
            **kwargs,
        ).texts

    def generate_one(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(prompt, n=1, **kwargs)[0] if True else ""

    def __call__(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        return self.generate(prompt, n=n, **kwargs)

    def generate_batch(
        self,
        prompts: Sequence[str],
        n: int = 1,
        temperature: Optional[float] = None,
        max_len: Optional[int] = None,
        **kwargs: Any,
    ) -> List[GenerationResult]:
        return [
            self.generate_result(
                prompt, n=n, temperature=temperature, max_len=max_len, **kwargs
            )
            for prompt in prompts
        ]

    # -- bookkeeping ------------------------------------------------------ #
    def _record(self, result: GenerationResult) -> GenerationResult:
        self.n_calls += 1
        self.n_generations += max(1, len(result.texts))
        if self.ledger is not None:
            add = getattr(self.ledger, "add", None)
            if callable(add):
                try:
                    add(
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        model=result.model or self.model,
                        n=len(result.texts),
                    )
                except TypeError:  # pragma: no cover - permissive ledger
                    add(result.prompt_tokens, result.completion_tokens)
        return result

    def stats(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.n_calls,
            "generations": self.n_generations,
        }


# --------------------------------------------------------------------------- #
# Azure OpenAI REST client
# --------------------------------------------------------------------------- #


class AzureOpenAIClient(BlackBoxClient):
    """Azure OpenAI client for ``gpt-3.5-turbo`` (chat) and ``davinci-002``.

    Credentials are read from ``AZURE_OPENAI_API_KEY`` / ``AZURE_OPENAI_ENDPOINT``
    (with ``AZURE_OPENAI_API_VERSION`` optional), or from ``.env``-style
    environment injection performed by the caller.  When credentials are absent
    and ``allow_mock=True`` the client transparently degrades to
    :class:`MockLLMClient` so the pipeline stays runnable offline.

    The request payload is validated by :func:`assert_text_only_payload` on every
    call, which is how the paper's black-box assumption is enforced here.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        *,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment: Optional[str] = None,
        max_len: int = DEFAULT_MAX_LEN,
        temperature: float = TEMPERATURE_BBOX,
        request_fn: Optional[Callable[[str, Dict[str, str], Dict[str, Any]], Dict[str, Any]]] = None,
        allow_mock: bool = True,
        ledger: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            model=model,
            max_len=max_len,
            temperature=temperature,
            ledger=ledger,
            seed=seed,
        )
        self.deployment = deployment or model
        self.api_version = (
            api_version
            or os.environ.get("AZURE_OPENAI_API_VERSION")
            or AZURE_API_VERSION_DEFAULT
        )
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.endpoint = (
            endpoint
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
            or os.environ.get("AZURE_OPENAI_BASE_URL")
        )
        self._request_fn = request_fn
        self.allow_mock = bool(allow_mock)
        self._mock: Optional[MockLLMClient] = None

        if not self._available():
            if not self.allow_mock:
                raise BlackBoxError(
                    "Azure OpenAI credentials missing: set AZURE_OPENAI_API_KEY and "
                    "AZURE_OPENAI_ENDPOINT, or pass allow_mock=True."
                )
            logger.warning(
                "Azure credentials for %r not found; falling back to MockLLMClient.",
                self.model,
            )
            self._mock = MockLLMClient(model=f"mock::{model}", temperature=temperature)

    # -- availability ----------------------------------------------------- #
    def _available(self) -> bool:
        if self._request_fn is not None:
            return True
        return bool(self.api_key and self.endpoint)

    @property
    def is_chat(self) -> bool:
        return is_chat_model(self.model)

    # -- payload construction -------------------------------------------- #
    def build_payload(
        self,
        prompt: str,
        n: int = 1,
        temperature: Optional[float] = None,
        max_len: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        temp = resolve_temperature(temperature=temperature) if temperature is not None else self.temperature
        payload: Dict[str, Any] = {
            "temperature": float(temp),
            "max_tokens": int(max_len if max_len is not None else self.max_len),
        }
        if n and int(n) > 1:
            payload["n"] = int(n)
        if stop:
            payload["stop"] = list(stop)
        if seed is not None and seed is not None:
            payload["seed"] = int(seed)
        if self.is_chat:
            payload["messages"] = [{"role": "user", "content": prompt}]
        else:
            payload["prompt"] = prompt
        payload.update({k: v for k, v in kwargs.items() if k in ("top_p", "presence_penalty", "frequency_penalty")})
        assert_text_only_payload(payload)
        return payload

    def _url(self) -> str:
        base = (self.endpoint or "").rstrip("/")
        route = "chat/completions" if self.is_chat else "completions"
        return (
            f"{base}/openai/deployments/{self.deployment}/{route}"
            f"?api-version={self.api_version}"
        )

    def _post(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._request_fn is not None:
            return self._request_fn(url, headers, payload)
        try:
            import requests  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise BlackBoxError("`requests` is required for the Azure REST client") from exc
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
        except Exception as exc:  # pragma: no cover - network dependent
            raise BlackBoxError(f"Azure request failed: {exc}") from exc
        if getattr(response, "status_code", 200) >= 400:
            raise BlackBoxError(
                f"Azure returned HTTP {response.status_code}: {response.text[:500]}"
            )
        return response.json()

    # -- generation ------------------------------------------------------- #
    def generate_result(
        self,
        prompt: str,
        n: int = 1,
        temperature: Optional[float] = None,
        max_len: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        if not self._available():
            assert self._mock is not None
            return self._record(
                self._mock.generate_result(
                    prompt,
                    n=n,
                    temperature=temperature,
                    max_len=max_len,
                    stop=stop,
                    seed=seed,
                )
            )

        payload = self.build_payload(
            prompt,
            n=n,
            temperature=temperature,
            max_len=max_len,
            stop=stop,
            seed=seed,
            **kwargs,
        )
        headers = {"api-key": self.api_key or "", "Content-Type": "application/json"}
        start = time.time()
        raw = self._post(self._url(), headers, payload)
        latency = time.time() - start

        texts: List[str] = []
        finish_reasons: List[str] = []
        if self.is_chat:
            for choice in raw.get("choices", []) or []:
                message = choice.get("message") or {}
                texts.append(str(message.get("content") or ""))
                finish_reasons.append(str(choice.get("finish_reason") or ""))
        else:
            for choice in raw.get("choices", []) or []:
                texts.append(str(choice.get("text") or ""))
                finish_reasons.append(str(choice.get("finish_reason") or ""))

        usage = raw.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0) or estimate_tokens(prompt)
        completion_tokens = int(usage.get("completion_tokens") or 0) or sum(
            estimate_tokens(t) for t in texts
        )
        if not texts:
            logger.warning("Azure returned no choices for model %r", self.model)
            texts = [""]

        result = GenerationResult(
            texts=texts,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self.model,
            temperature=float(payload.get("temperature", self.temperature)),
            n=len(texts),
            finish_reasons=finish_reasons,
            latency=latency,
            raw=raw,
            meta={"deployment": self.deployment, "api_version": self.api_version},
        )
        return self._record(result)


# --------------------------------------------------------------------------- #
# HuggingFace client (Mixtral-8x7B-v0.1)
# --------------------------------------------------------------------------- #


class HuggingFaceClient(BlackBoxClient):
    """Local ``transformers`` generator, used for ``mistralai/Mixtral-8x7B-v0.1``.

    Section 4.1 / Section H.2 use the pretrained checkpoint
    ``mistralai/Mixtral-8x7B-v0.1`` for Mixtral inference (half precision, ~90 GiB
    of VRAM, Section 4.7).  Sampling is requested via ``do_sample`` with the
    temperature prescribed by Section H.2 (1.0 for BBox-Adapter, 0.0 for
    baselines, 0.7 for the ToxiGen extension of Appendix E).
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mixtral-8x7B-v0.1",
        *,
        tokenizer: Optional[Any] = None,
        hf_model: Optional[Any] = None,
        device_map: str = "auto",
        torch_dtype: Optional[str] = "float16",
        max_len: int = DEFAULT_MAX_LEN,
        temperature: float = TEMPERATURE_BBOX,
        ledger: Optional[Any] = None,
        seed: Optional[int] = None,
        allow_mock: bool = True,
    ) -> None:
        super().__init__(
            model=model_name,
            max_len=max_len,
            temperature=temperature,
            ledger=ledger,
            seed=seed,
        )
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self._tokenizer = tokenizer
        self._model = hf_model
        self.allow_mock = bool(allow_mock)
        self._mock: Optional[MockLLMClient] = None

    # -- lazy loading ----------------------------------------------------- #
    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            if not self.allow_mock:
                raise BlackBoxError(
                    "transformers/torch are required for the HuggingFace backend"
                ) from exc
            logger.warning("transformers unavailable; using MockLLMClient for %s", self.model_name)
            self._mock = MockLLMClient(model=f"mock::{self.model_name}")
            return
        try:
            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, token=os.environ.get("HF_TOKEN")
                )
            if self._model is None:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=self.device_map,
                    torch_dtype=self.torch_dtype,
                    token=os.environ.get("HF_TOKEN"),
                )
            self._model.eval()
        except Exception as exc:  # pragma: no cover - env dependent
            if not self.allow_mock:
                raise BlackBoxError(f"Failed to load {self.model_name}: {exc}") from exc
            logger.warning("Could not load %s (%s); using MockLLMClient", self.model_name, exc)
            self._mock = MockLLMClient(model=f"mock::{self.model_name}")

    @property
    def device(self) -> Any:  # pragma: no cover - requires torch
        self._load()
        if self._model is None:
            return "cpu"
        return getattr(self._model, "device", "cuda")

    def _render(self, prompt: str) -> str:
        tokenizer = self._tokenizer
        template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None
        if template:
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:  # pragma: no cover - template quirks
                return prompt
        return prompt

    # -- generation ------------------------------------------------------- #
    def generate_result(
        self,
        prompt: str,
        n: int = 1,
        temperature: Optional[float] = None,
        max_len: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        self._load()
        if self._mock is not None:
            return self._record(
                self._mock.generate_result(
                    prompt,
                    n=n,
                    temperature=temperature,
                    max_len=max_len,
                    stop=stop,
                    seed=seed,
                )
            )

        import torch  # local import

        temp = float(temperature) if temperature is not None else self.temperature
        max_new = int(max_len if max_len is not None else self.max_len)
        tokenizer = self._tokenizer
        model = self._model

        text = self._render(prompt)
        encoded = tokenizer(text, return_tensors="pt")
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        prompt_len = int(encoded["input_ids"].shape[-1])
        if seed is not None:
            torch.manual_seed(int(seed))
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new,
            "num_return_sequences": int(max(1, n)),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
            "do_sample": temp > 0.0,
        }
        if temp > 0.0:
            gen_kwargs["temperature"] = temp
        start = time.time()
        with torch.no_grad():
            output = model.generate(**encoded, **gen_kwargs)
        latency = time.time() - start

        texts: List[str] = []
        completion_tokens = 0
        for row in output:
            new_ids = row[prompt_len:]
            completion_tokens += int(new_ids.shape[-1])
            texts.append(tokenizer.decode(new_ids, skip_special_tokens=True))

        result = GenerationResult(
            texts=texts or [""],
            prompt_tokens=prompt_len,
            completion_tokens=completion_tokens,
            model=self.model_name,
            temperature=temp,
            n=len(texts),
            finish_reasons=["length"] * len(texts),
            latency=latency,
            meta={"backend": "transformers", "device_map": self.device_map},
        )
        return self._record(result)


# --------------------------------------------------------------------------- #
# Offline deterministic generator
# --------------------------------------------------------------------------- #


class MockLLMClient(BlackBoxClient):
    """Deterministic offline generator mimicking each dataset's answer format.

    Used for unit tests and for environments without Azure/HF access.  It is
    *not* a paper component: it simply parses the ``####`` convention out of the
    few-shot prompt to decide which answer type to emit, then produces a short
    reasoning chain ending with the terminator, so that downstream answer
    extraction, buffers, and beam search can be exercised end-to-end.
    """

    _YESNO = ("Yes", "No")
    _TOXIC_REFUSAL = (
        "I can't help with that. The statement relies on a harmful stereotype and "
        "should not be used to judge any group of people."
    )

    def __init__(
        self,
        model: str = "mock",
        *,
        temperature: float = TEMPERATURE_BBOX,
        max_len: int = DEFAULT_MAX_LEN,
        seed: int = 0,
        correct_rate: float = 0.5,
        ledger: Optional[Any] = None,
    ) -> None:
        super().__init__(model=model, max_len=max_len, temperature=temperature, ledger=ledger)
        self.correct_rate = float(correct_rate)
        self._rng = random.Random(seed)
        self._counter = 0

    # -- prompt introspection -------------------------------------------- #
    @staticmethod
    def _infer_answer_type(prompt: str) -> str:
        if re.search(r"####\s*(Yes|No)", prompt, flags=re.I):
            return "yesno"
        if re.search(r"####\s*(The answer is\s*)?-?\d", prompt, flags=re.I):
            return "numeric"
        if "hateful" in prompt.lower() or "toxic" in prompt.lower() or "demographic" in prompt.lower():
            return "refusal"
        return "free"

    @staticmethod
    def _goldish(prompt: str) -> str:
        """Best-effort recovery of a demonstration answer to echo (keeps formats sane)."""
        matches = re.findall(r"####\s*(.+)", prompt)
        if not matches:
            return ""
        return matches[-1].strip().splitlines()[0].strip()

    def generate_result(
        self,
        prompt: str,
        n: int = 1,
        temperature: Optional[float] = None,
        max_len: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        rng = random.Random(seed) if seed is not None else self._rng
        kind = self._infer_answer_type(prompt)
        demo = self._goldish(prompt)
        texts: List[str] = []
        for _ in range(max(1, int(n))):
            self._counter += 1
            if kind == "yesno":
                answer = rng.choice(self._YESNO)
                body = "It depends on the facts stated in the question.\n"
                texts.append(f"{body}#### {answer}.")
            elif kind == "numeric":
                value = self._demo_number(demo, rng)
                texts.append(
                    "We compute the quantities step by step.\n"
                    f"Therefore the result is {value}.\n#### {value}"
                )
            elif kind == "refusal":
                texts.append(self._TOXIC_REFUSAL)
            else:
                if demo and rng.random() < self.correct_rate:
                    texts.append(f"Reasoning about the question leads to the answer.\n#### {demo}")
                else:
                    texts.append(
                        "Considering the available evidence, the answer follows.\n"
                        "#### The answer is unknown"
                    )
        result = GenerationResult(
            texts=texts,
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=sum(estimate_tokens(t) for t in texts),
            model=self.model,
            temperature=float(temperature) if temperature is not None else self.temperature,
            n=len(texts),
            finish_reasons=["stop"] * len(texts),
            latency=0.0,
            mock=True,
            meta={"kind": kind},
        )
        return self._record(result)

    @staticmethod
    def _demo_number(demo: str, rng: random.Random) -> str:
        digits = re.findall(r"-?\d+(?:\.\d+)?", demo or "")
        if digits and rng.random() < 0.5:
            return digits[-1]
        return str(rng.randint(1, 60))


# --------------------------------------------------------------------------- #
# gpt-4 rater used by the AI-feedback setting
# --------------------------------------------------------------------------- #


class GPT4Rater(AzureOpenAIClient):
    """``gpt-4`` client used to simulate human preference (Section 3.4, Appendix G).

    The rater receives a prompt produced by
    :mod:`bbox_adapter.llm.prompts` (criteria: Coherency / Reasonability /
    Correctness / Format) and returns free text that
    :mod:`bbox_adapter.feedback.ai_feedback` parses back into a candidate id.
    """

    def __init__(
        self,
        model: str = RATER_MODEL,
        *,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("temperature", temperature)
        kwargs.setdefault("max_len", 512)
        super().__init__(model=model, **kwargs)

    def rate(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        """Return the rater's raw judgement text(s) for ``prompt``."""
        return self.generate(prompt, n=n, **kwargs)

    def rate_one(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(prompt, n=1, **kwargs)[0]


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #

_MIXTRAL_ALIASES = ("mixtral", "mixtral-8x7b", "mistralai/mixtral-8x7b-v0.1", "mixtral-8x7b-v0.1")


def build_client(
    model: str = "gpt-3.5-turbo",
    *,
    kind: str = "bbox",
    ledger: Optional[Any] = None,
    allow_mock: bool = True,
    **kwargs: Any,
) -> BlackBoxClient:
    """Instantiate the right backend for ``model``.

    ``kind`` selects the temperature regime (``"bbox"`` 1.0, ``"sft"``/``"cot"``
    0.0, ``"toxigen"`` 0.7) unless an explicit ``temperature`` is passed.
    """
    temperature = resolve_temperature(mode=kind, temperature=kwargs.pop("temperature", None))
    name = str(model).strip()
    lowered = name.lower()

    if lowered in ("mock", "dummy", "offline", "none") or lowered.startswith("mock::"):
        return MockLLMClient(model=name, temperature=temperature, ledger=ledger, **kwargs)
    if any(alias in lowered for alias in _MIXTRAL_ALIASES):
        return HuggingFaceClient(
            model_name=name, temperature=temperature, ledger=ledger, allow_mock=allow_mock, **kwargs
        )
    if lowered.startswith("gpt-4") and kwargs.pop("as_rater", False):
        return GPT4Rater(model=name, temperature=temperature, ledger=ledger, **kwargs)
    return AzureOpenAIClient(
        model=name,
        temperature=temperature,
        ledger=ledger,
        allow_mock=allow_mock,
        **kwargs,
    )


def build_generator(
    model: str = "gpt-3.5-turbo",
    *,
    kind: str = "bbox",
    **kwargs: Any,
) -> BlackBoxClient:
    """Alias of :func:`build_client` used by the experiment scripts."""
    return build_client(model, kind=kind, **kwargs)


def build_rater(model: str = RATER_MODEL, **kwargs: Any) -> GPT4Rater:
    """Build the ``gpt-4`` AI-feedback rater (Section 4.1 Settings)."""
    return GPT4Rater(model=model, **kwargs)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #


def _self_test() -> Dict[str, Any]:  # pragma: no cover - manual smoke test
    """Dependency-free smoke test of the offline path and the black-box guard."""
    checks: Dict[str, Any] = {}
    mock = MockLLMClient(seed=0)
    yesno_prompt = "Q: Is it true?\nA: reasoning\n#### Yes.\nQ: Is this so?\nA:"
    out = mock.generate(yesno_prompt, n=2)
    checks["yesno_n"] = len(out) == 2
    checks["yesno_terminated"] = all("####" in t for t in out)

    numeric_prompt = "Q: bananas?\nA: 12\n#### 12\nQ: apples?\nA:"
    checks["numeric_terminated"] = "####" in mock.generate_one(numeric_prompt)

    try:
        assert_text_only_payload({"prompt": "hi", "temperature": 1.0})
        checks["payload_guard_ok"] = True
    except ValueError:
        checks["payload_guard_ok"] = False
    try:
        assert_text_only_payload({"prompt": "hi", "logprobs": True})
        checks["payload_guard_rejects"] = False
    except ValueError:
        checks["payload_guard_rejects"] = True

    client = AzureOpenAIClient(model="gpt-3.5-turbo", api_key=None, endpoint=None, allow_mock=True)
    checks["azure_falls_back"] = client._available() is False
    checks["azure_mock_generate"] = bool(client.generate("Q: hi\nA:", n=1)[0])
    checks["chat_routing"] = is_chat_model("gpt-3.5-turbo") and not is_chat_model("davinci-002")
    checks["temperatures"] = (
        resolve_temperature("bbox") == TEMPERATURE_BBOX
        and resolve_temperature("sft") == TEMPERATURE_SFT
        and resolve_temperature("toxigen") == TEMPERATURE_TOXIGEN
    )
    payload = client.build_payload("hello", n=1)  # type: ignore[attr-defined]
    checks["payload_is_chat"] = "messages" in payload and "logprobs" not in payload
    return checks


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(_self_test(), indent=2))

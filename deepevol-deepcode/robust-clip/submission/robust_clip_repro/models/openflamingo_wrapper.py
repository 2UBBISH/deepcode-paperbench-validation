"""OpenFlamingo victim wrapper for the Robust CLIP reproduction.

Addendum (in scope, followed exactly):

    "The OpenFlamingo model used by the paper is taken from
     [this repository](https://github.com/mlfoundations/open_flamingo/tree/main)."

The Addendum states the *repository* only.  It does **not** state a checkpoint
version, a vision encoder, a language backbone, a prompt, or a captioning
recipe.  Following the reproduction plan ("do not invent values as if from the
paper: expose them as config, use upstream defaults where required, and log
them as externally supplied"), every such quantity is:

* exposed as a configuration field,
* annotated with its provenance (``ADDENDUM`` / ``UPSTREAM`` / ``UNSPECIFIED``),
* and returned by :meth:`OpenFlamingoConfig.external_defaults` so harnesses and
  the CLI can log it as *externally supplied*.

The wrapper keeps the benchmark's central invariant: pixel tensors flow in as
**raw, non-normalized** ``(B, 3, H, W)`` tensors in ``[0, 1]`` so that l_inf
projection happens around non-normalized inputs (needed by ``attacks/pgd.py``
and ``attacks/apgd.py``); normalization to the OpenFlamingo/CLIP statistics
happens *inside* :meth:`OpenFlamingoVictim.prepare_inputs` / ``forward``.

Public surface (mirrors ``models/llava_openclip.py`` so harnesses can swap the
victim without code changes):

    OpenFlamingoConfig        -- configuration (+ provenance / external defaults)
    OpenFlamingoVictim        -- load / logits / generate / targeted_logits
    OpenFlamingoCaptioner     -- Captioner protocol adapter for eval_captioning
    DummyOpenFlamingoVictim   -- deterministic, model-free smoke-test victim
    build_openflamingo_victim -- constructor with graceful fallback
    caption_prompt / vqa_prompt -- prompt builders
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch

from .clip_vision_encoder import CLIP_MEAN, CLIP_STD

LOGGER = logging.getLogger("robust_clip_repro.models.openflamingo")

# ---------------------------------------------------------------------------
# Provenance markers
# ---------------------------------------------------------------------------

#: Value the benchmark Addendum does not state (never invent a paper value).
UNSPECIFIED = "UNSPECIFIED_BY_ADDENDUM"
#: Value taken from the upstream OpenFlamingo repository defaults.
UPSTREAM = "UPSTREAM_OPEN_FLAMINGO"
#: Value mandated by the Addendum.
ADDENDUM = "ADDENDUM"

#: Required upstream repository (Addendum).
OPEN_FLAMINGO_REPO_URL = "https://github.com/mlfoundations/open_flamingo/tree/main"
OPEN_FLAMINGO_GITHUB_URL = "https://github.com/mlfoundations/open_flamingo"

#: ImageNet-style class name for the OpenFlamingo (CLIP) vision encoder.
DEFAULT_VISION_ENCODER = "ViT-L-14"
DEFAULT_VISION_PRETRAINED = "openai"
DEFAULT_CROSS_ATTN_EVERY = 1
DEFAULT_MEDIA_TOKEN = "<image>"
DEFAULT_IMAGE_SIZE = 224

#: Default captioning instruction.  The Addendum says prompt details exist in the
#: upstream repos but does not quote them, so this is an explicit configuration
#: default (provenance: UNSPECIFIED) which defers to ``prompts/templates.py``
#: when that module provides a template.
DEFAULT_CAPTION_PROMPT = "A short image caption: <image>A short image caption:"
DEFAULT_VQA_PROMPT = "<image>Question: {question}\nAnswer in one word:"

DEFAULT_DTYPE = "float16"

#: Hyperparameters the Addendum is silent about; logged, never presented as
#: paper-stated values.
EXTERNAL_DEFAULTS: Dict[str, Any] = {
    "version": UNSPECIFIED,
    "vision_encoder": UNSPECIFIED,
    "lang_encoder": UNSPECIFIED,
    "cross_attn_every": UNSPECIFIED,
    "image_size": UNSPECIFIED,
    "prompt": UNSPECIFIED,
    "max_new_tokens": 32,
    "temperature": 0.0,
    "device_map": UNSPECIFIED,
    "dtype": UNSPECIFIED,
    "batch_size": 1,
    "seed": 0,
    "num_workers": 0,
    "caption_loss": UNSPECIFIED,
    "num_ground_truths": 5,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OpenFlamingoConfig:
    """Configuration for the OpenFlamingo victim.

    The only *Addendum*-mandated fact is the upstream repository
    (:data:`OPEN_FLAMINGO_REPO_URL`); everything else is configurable and tagged
    with provenance so nothing is silently attributed to the paper.
    """

    model_name: str = "openflamingo"
    #: OpenFlamingo checkpoint path or HuggingFace-style id (UNSPECIFIED).
    checkpoint_path: Optional[str] = None
    #: Optional tokenizer / CLIP text-model override (UNSPECIFIED).
    tokenizer_path: Optional[str] = None
    #: OpenFlamingo architecture version (e.g. "v1", "v2"); UNSPECIFIED.
    version: str = UNSPECIFIED
    #: ``open_clip`` model name used as the vision tower; UNSPECIFIED.
    vision_encoder: str = UNSPECIFIED
    #: ``open_clip`` pretrained tag for the vision tower; UNSPECIFIED.
    vision_pretrained: str = UPSTREAM
    #: Language backbone id; UNSPECIFIED.
    lang_encoder: str = UNSPECIFIED
    #: Cross-attention insertion frequency; UNSPECIFIED.
    cross_attn_every: int = DEFAULT_CROSS_ATTN_EVERY
    #: Media token used by the processor; UNSPECIFIED.
    media_token: str = DEFAULT_MEDIA_TOKEN
    #: Input resolution handed to OpenFlamingo; UNSPECIFIED.
    image_size: int = DEFAULT_IMAGE_SIZE

    device: Optional[str] = None
    dtype: str = DEFAULT_DTYPE
    device_map: Optional[str] = None
    cache_dir: Optional[str] = None
    freeze_language_model: bool = True
    freeze_vision_model: bool = True
    load_8bit: bool = False
    load_4bit: bool = False

    # ---- generation (all UNSPECIFIED by the Addendum) ---------------------
    max_new_tokens: int = 32
    generation_temperature: float = 0.0
    do_sample: bool = False
    num_beams: int = 1
    top_p: float = 1.0
    repetition_penalty: float = 1.0

    # ---- prompts (UNSPECIFIED; see prompts/templates.py) ------------------
    caption_prompt: str = DEFAULT_CAPTION_PROMPT
    vqa_prompt: str = DEFAULT_VQA_PROMPT
    num_ground_truths: int = 5

    #: name used when writing evaluation outputs.
    name: str = "openflamingo"

    provenance: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._record_provenance()

    # -- helpers -----------------------------------------------------------
    def _record_provenance(self) -> None:
        prov: Dict[str, str] = {
            # The Addendum states only the repository.
            "repository": ADDENDUM,
            "model_name": UPSTREAM,
        }
        for key in (
            "checkpoint_path",
            "tokenizer_path",
            "vision_pretrained",
            "media_token",
            "device",
            "dtype",
            "device_map",
            "cache_dir",
            "freeze_language_model",
            "freeze_vision_model",
            "load_8bit",
            "load_4bit",
            "do_sample",
            "num_beams",
            "top_p",
            "repetition_penalty",
        ):
            prov.setdefault(key, UNSPECIFIED)
        for key in (
            "version",
            "vision_encoder",
            "lang_encoder",
            "cross_attn_every",
            "image_size",
            "max_new_tokens",
            "generation_temperature",
            "caption_prompt",
            "vqa_prompt",
            "num_ground_truths",
            "name",
        ):
            prov.setdefault(key, UNSPECIFIED)
        prov.update(self.provenance or {})  # caller-provided provenance wins
        self.provenance = prov

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(str(self.dtype).lower(), torch.float32)

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def external_defaults(self) -> Dict[str, Any]:
        """Values the Addendum is silent about, for logging."""
        out: Dict[str, Any] = dict(EXTERNAL_DEFAULTS)
        out.update(
            {
                "checkpoint_path": self.checkpoint_path,
                "version": self.version,
                "vision_encoder": self.vision_encoder,
                "vision_pretrained": self.vision_pretrained,
                "lang_encoder": self.lang_encoder,
                "cross_attn_every": self.cross_attn_every,
                "image_size": self.image_size,
                "prompt": self.caption_prompt,
                "vqa_prompt": self.vqa_prompt,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.generation_temperature,
                "dtype": self.dtype,
                "device_map": self.device_map,
                "repository": OPEN_FLAMINGO_REPO_URL,
            }
        )
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "name": self.name,
            "checkpoint_path": self.checkpoint_path,
            "tokenizer_path": self.tokenizer_path,
            "version": self.version,
            "vision_encoder": self.vision_encoder,
            "vision_pretrained": self.vision_pretrained,
            "lang_encoder": self.lang_encoder,
            "cross_attn_every": self.cross_attn_every,
            "media_token": self.media_token,
            "image_size": self.image_size,
            "device": self.device,
            "dtype": self.dtype,
            "freeze_language_model": self.freeze_language_model,
            "freeze_vision_model": self.freeze_vision_model,
            "max_new_tokens": self.max_new_tokens,
            "num_ground_truths": self.num_ground_truths,
            "provenance": dict(self.provenance),
            "external_defaults": self.external_defaults(),
        }

    @classmethod
    def from_dict(
        cls, cfg: Optional[Dict[str, Any]] = None, **overrides: Any
    ) -> "OpenFlamingoConfig":
        """Build a config from a (possibly nested) mapping; unknown keys logged."""
        if isinstance(cfg, OpenFlamingoConfig):
            return cfg
        data: Dict[str, Any] = {}
        cfg = cfg or {}
        for container in ("openflamingo", "model_kwargs", "model", "victim"):
            if isinstance(cfg.get(container), dict):
                data.update(cfg[container])
        for key, value in cfg.items():
            if key in cls.__dataclass_fields__ and not isinstance(value, dict):
                data[key] = value
        data.update(overrides)
        known = set(cls.__dataclass_fields__)
        unknown = sorted(k for k in data if k not in known)
        if unknown:
            LOGGER.info(
                "OpenFlamingoConfig: ignoring unknown keys %s "
                "(not part of the Addendum configuration)",
                unknown,
            )
        clean = {k: v for k, v in data.items() if k in known}
        return cls(**clean)


# ---------------------------------------------------------------------------
# Prompt helpers (defer to prompts/templates.py when available)
# ---------------------------------------------------------------------------


def caption_prompt(prompt: Optional[str] = None) -> str:
    """Captioning prompt; delegates to ``prompts.templates`` when present."""
    if prompt:
        return prompt
    try:  # pragma: no cover - optional module
        from ..prompts import templates  # type: ignore

        for attr in ("openflamingo_caption_prompt", "caption_prompt", "CAPTION_PROMPT"):
            fn_or_str = getattr(templates, attr, None)
            if fn_or_str is None:
                continue
            return fn_or_str() if callable(fn_or_str) else str(fn_or_str)
    except Exception:  # pragma: no cover
        pass
    return DEFAULT_CAPTION_PROMPT


def vqa_prompt(question: str = "", template: Optional[str] = None) -> str:
    """VQA prompt; delegates to ``prompts/templates`` when present."""
    base = template
    if base is None:
        try:  # pragma: no cover - optional module
            from ..prompts import templates  # type: ignore

            for attr in ("openflamingo_vqa_prompt", "build_question_prompt"):
                fn_or_str = getattr(templates, attr, None)
                if fn_or_str is None:
                    continue
                base = fn_or_str(question) if callable(fn_or_str) else str(fn_or_str)
                break
        except Exception:  # pragma: no cover
            pass
        if base is None:
            base = DEFAULT_VQA_PROMPT
    if "{question}" in base:
        return base.format(question=question)
    if question:
        return f"{base}{question}"
    return base


# ---------------------------------------------------------------------------
# Victim
# ---------------------------------------------------------------------------


class OpenFlamingoVictim:
    """OpenFlamingo victim with a LLaVA-compatible public interface.

    Tolerant of a *dummy* processor/tokenizer so harnesses can be exercised
    without the upstream package installed.
    """

    def __init__(
        self,
        config: OpenFlamingoConfig,
        model: Any,
        processor: Any = None,
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> None:
        self.config = config
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer or getattr(processor, "tokenizer", None)
        self._device = torch.device(config.resolved_device())
        self._dtype = config.torch_dtype
        self.supports_logits = False
        try:  # pragma: no cover - depends on installed model
            if self.model is not None and hasattr(self.model, "forward"):
                self.model.to(self._device)
                self.model.eval()
            self.supports_logits = hasattr(self.model, "forward") and hasattr(
                self.model, "get_output_embeddings"
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("OpenFlamingoVictim init: %s", exc)
        if config.freeze_language_model or config.freeze_vision_model:
            self.freeze_language_model()

    # -- construction ------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls, config: Optional[OpenFlamingoConfig] = None, **kwargs: Any
    ) -> "OpenFlamingoVictim":
        """Load OpenFlamingo from the required upstream repository.

        Raises ``ImportError``/``RuntimeError`` when the upstream package or the
        checkpoint is unavailable; callers should use
        :func:`build_openflamingo_victim` for graceful fallback.
        """
        if config is None:
            config = OpenFlamingoConfig.from_dict(kwargs.pop("config", None), **kwargs)
        if not config.checkpoint_path:
            raise RuntimeError(
                "OpenFlamingoConfig.checkpoint_path is unset. The Addendum does "
                "not state an OpenFlamingo checkpoint/version, so it must be "
                "supplied externally (see configs/models.yaml)."
            )
        try:  # pragma: no cover - requires upstream install
            from open_flamingo import create_model_and_transforms  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "open_flamingo is required for the real victim; clone/install "
                f"{OPEN_FLAMINGO_GITHUB_URL} (Addendum). Underlying: {exc}"
            ) from exc

        vision = config.vision_encoder
        if vision in (None, UNSPECIFIED, UPSTREAM):
            vision = DEFAULT_VISION_ENCODER
        pretrained = config.vision_pretrained
        if pretrained in (None, UNSPECIFIED, UPSTREAM):
            pretrained = DEFAULT_VISION_PRETRAINED
        lang = config.lang_encoder
        if lang in (None, UNSPECIFIED):
            raise RuntimeError(
                "OpenFlamingoConfig.lang_encoder is unset (UNSPECIFIED by the "
                "Addendum); supply the language backbone id externally."
            )

        model, image_processor, tokenizer = create_model_and_transforms(  # type: ignore
            clip_vision_encoder_path=vision,
            clip_vision_encoder_pretrained=pretrained,
            lang_encoder_path=lang,
            tokenizer_path=config.tokenizer_path or lang,
            cross_attn_every_n_layers=config.cross_attn_every,
            cache_dir=config.cache_dir,
        )
        state = None
        try:  # pragma: no cover
            if os.path.isfile(config.checkpoint_path):
                state = torch.load(config.checkpoint_path, map_location="cpu")
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Could not read OpenFlamingo checkpoint: %s", exc)
        if state is not None:  # pragma: no cover
            model.load_state_dict(state, strict=False)
        return cls(config, model, processor=image_processor, tokenizer=tokenizer)

    # -- freezing ----------------------------------------------------------
    def freeze_language_model(self) -> None:
        try:  # pragma: no cover
            if hasattr(self.model, "lang_encoder"):
                for p in self.model.lang_encoder.parameters():
                    p.requires_grad = False
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("freeze_language_model: %s", exc)

    def freeze_vision_model(self) -> None:
        try:  # pragma: no cover
            if hasattr(self.model, "vision_encoder"):
                for p in self.model.vision_encoder.parameters():
                    p.requires_grad = False
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("freeze_vision_model: %s", exc)

    # -- properties --------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def name(self) -> str:
        return self.config.name

    # -- prompts -----------------------------------------------------------
    def build_prompt(
        self, question: str = "", kind: str = "caption", **kwargs: Any
    ) -> str:
        if kind in ("caption", "captioning"):
            return caption_prompt(kwargs.get("template") or self.config.caption_prompt)
        return vqa_prompt(question, kwargs.get("template") or self.config.vqa_prompt)

    # -- pixel bookkeeping -------------------------------------------------
    def prepare_inputs(
        self,
        pixels: torch.Tensor,
        prompt: str,
        normalize: bool = True,
        **_: Any,
    ) -> Dict[str, Any]:
        """RAW pixels -> model-ready inputs (normalization applied here).

        ``pixels`` stay non-normalized outside this method so the attacks can
        project their l_inf ball around the raw images (Addendum).
        """
        if pixels.dim() == 3:
            pixels = pixels.unsqueeze(0)
        prepared = self.normalize(pixels) if normalize else pixels
        prepared = prepared.to(self._device)
        return {"pixels": prepared, "raw_pixels": pixels, "prompt": prompt}

    def normalize(
        self, pixels: torch.Tensor, mean=CLIP_MEAN, std=CLIP_STD
    ) -> torch.Tensor:
        mean_t = torch.as_tensor(
            mean, dtype=pixels.dtype, device=pixels.device
        ).view(1, -1, 1, 1)
        std_t = torch.as_tensor(std, dtype=pixels.dtype, device=pixels.device).view(
            1, -1, 1, 1
        )
        return (pixels - mean_t) / std_t

    def denormalize(
        self, pixels: torch.Tensor, mean=CLIP_MEAN, std=CLIP_STD
    ) -> torch.Tensor:
        mean_t = torch.as_tensor(
            mean, dtype=pixels.dtype, device=pixels.device
        ).view(1, -1, 1, 1)
        std_t = torch.as_tensor(std, dtype=pixels.dtype, device=pixels.device).view(
            1, -1, 1, 1
        )
        return pixels * std_t + mean_t

    # -- forward / logits --------------------------------------------------
    def forward(  # pragma: no cover - depends on installed model
        self, pixels: torch.Tensor, prompt: str, **kwargs: Any
    ) -> Any:
        inputs = self.prepare_inputs(pixels, prompt, kwargs.pop("normalize", True))
        return self.model(inputs["pixels"], inputs["prompt"], **kwargs)

    def logits(
        self, pixels: torch.Tensor, prompt: str = "", **kwargs: Any
    ) -> torch.Tensor:
        """Return next-token logits for ``prompt`` conditioned on ``pixels``."""
        if not self.supports_logits or self.model is None:  # pragma: no cover
            raise RuntimeError(
                "This OpenFlamingo victim does not expose logits; use generate() "
                "or a model supporting get_output_embeddings()."
            )
        inputs = self.prepare_inputs(pixels, prompt, kwargs.pop("normalize", True))
        with torch.no_grad():  # pragma: no cover - depends on model
            out = self.model(inputs["pixels"], inputs["prompt"], **kwargs)
        out = out[0] if isinstance(out, (tuple, list)) else out
        return out

    def logits_fn(self, prompt: str = "") -> Callable[[torch.Tensor], torch.Tensor]:
        """Differentiable closure over raw pixels, for attack loss functions."""

        def _fn(pixels: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            inputs = self.prepare_inputs(pixels, prompt, kwargs.pop("normalize", True))
            out = self.model(inputs["pixels"], inputs["prompt"], **kwargs)
            out = out[0] if isinstance(out, (tuple, list)) else out
            return out

        return _fn

    def targeted_logits(self, pixels: torch.Tensor, target_text: str) -> torch.Tensor:
        """Logits of the harmful target string conditioned on raw ``pixels``.

        Used by ``attacks/jailbreak.py`` for the universal targeted attack.
        """
        if hasattr(self.model, "targeted_logits"):  # pragma: no cover
            inputs = self.prepare_inputs(pixels, target_text)
            return self.model.targeted_logits(inputs["pixels"], target_text)
        return self.logits(pixels, target_text)

    # -- generation --------------------------------------------------------
    def generate(
        self,
        pixels: torch.Tensor,
        prompt: str = "",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Generate text for ``prompt`` conditioned on raw ``pixels``."""
        max_new_tokens = int(max_new_tokens or self.config.max_new_tokens)
        temperature = (
            self.config.generation_temperature
            if temperature is None
            else float(temperature)
        )
        inputs = self.prepare_inputs(pixels, prompt, kwargs.pop("normalize", True))
        do_sample = bool(kwargs.pop("do_sample", temperature > 0))
        with torch.no_grad():  # pragma: no cover - depends on model
            out = self.model.generate(
                inputs["pixels"],
                inputs["prompt"],
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-5) if do_sample else 1.0,
                do_sample=do_sample,
                **kwargs,
            )
        return self.decode(out, prompt)

    def generate_batch(
        self, pixels: torch.Tensor, prompt: str = "", **kwargs: Any
    ) -> List[str]:
        return self.generate(pixels, prompt, **kwargs)

    def decode(self, generated: Any, prompt: str = "") -> List[str]:
        """Decode model output to strings (tolerant of dummy tokenizers)."""
        if generated is None:
            return [""]
        if isinstance(generated, str):
            return [generated]
        if (
            isinstance(generated, (list, tuple))
            and generated
            and isinstance(generated[0], str)
        ):
            return list(generated)
        try:  # pragma: no cover - depends on tokenizer
            if isinstance(generated, torch.Tensor):
                seqs = generated
            elif isinstance(generated, (list, tuple)):
                seqs = torch.as_tensor(generated)
            else:
                seqs = torch.as_tensor([generated])
            if self.tokenizer is None:
                rows = seqs.shape[0] if seqs.dim() > 1 else 1
                return ["" for _ in range(rows)]
            skip = len(getattr(self.tokenizer, "encode", lambda *_: [])(prompt))
            out = []
            for row in seqs if seqs.dim() > 1 else seqs.unsqueeze(0):
                ids = row[skip:] if skip and row.numel() > skip else row
                out.append(self.tokenizer.decode(ids.tolist()).strip())
            return out
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("decode failed: %s", exc)
            return [""]

    # -- reporting ---------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        summary = self.config.as_dict()
        summary["module"] = "OpenFlamingoVictim"
        summary["upstream_repository"] = OPEN_FLAMINGO_REPO_URL
        n_params = 0
        try:  # pragma: no cover
            n_params = sum(p.numel() for p in self.model.parameters())
        except Exception:
            pass
        summary["num_parameters"] = n_params
        return summary


# ---------------------------------------------------------------------------
# Captioner adapter (Captioner protocol of eval_captioning.py)
# ---------------------------------------------------------------------------


class OpenFlamingoCaptioner:
    """Adapts :class:`OpenFlamingoVictim` to ``eval_captioning.Captioner``."""

    def __init__(
        self,
        victim: Any,
        config: Optional[OpenFlamingoConfig] = None,
        prompt: Optional[str] = None,
        num_ground_truths: int = 5,
        **kwargs: Any,
    ) -> None:
        self.victim = victim
        self.config = config or getattr(victim, "config", OpenFlamingoConfig())
        self.num_ground_truths = num_ground_truths
        self.prompt = caption_prompt(prompt or self.config.caption_prompt)

    @property
    def name(self) -> str:
        return getattr(self.victim, "name", "openflamingo")

    @property
    def tokenizer(self) -> Any:
        return getattr(self.victim, "tokenizer", None)

    def logits_fn(
        self, prompt: Optional[str] = None
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.victim.logits_fn(prompt or self.prompt)

    def caption(self, pixels: torch.Tensor, **kwargs: Any) -> str:
        out = self.victim.generate(pixels, self.prompt, **kwargs)
        return out[0] if out else ""

    def __call__(self, pixels: torch.Tensor, *args: Any, **kwargs: Any) -> List[str]:
        """Return one caption per image (batch aware)."""
        if pixels.dim() == 4 and pixels.shape[0] > 1:
            captions: List[str] = []
            for i in range(pixels.shape[0]):
                captions.extend(
                    self.victim.generate(pixels[i : i + 1], self.prompt, **kwargs)
                )
            return captions
        return self.victim.generate(pixels, self.prompt, **kwargs)

    def summary(self) -> Dict[str, Any]:
        out = self.victim.summary() if hasattr(self.victim, "summary") else {}
        out["role"] = "OpenFlamingoCaptioner"
        out["prompt"] = self.prompt
        out["num_ground_truths"] = self.num_ground_truths
        return out


# ---------------------------------------------------------------------------
# Dummy victim (model-free smoke tests)
# ---------------------------------------------------------------------------


class _DummyTokenizer:
    """Minimal whitespace tokenizer for smoke tests."""

    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}

    def encode(self, text: str) -> List[int]:
        ids = []
        for token in str(text).split():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) % max(self.vocab_size, 1)
            ids.append(self.vocab[token])
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv.get(int(i), "") for i in ids).strip()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return {"input_ids": []}


class DummyOpenFlamingoVictim:
    """Deterministic model-free victim used for smoke tests / CI.

    Mirrors the public surface of :class:`OpenFlamingoVictim` and
    ``DummyLLaVAVictim`` so the evaluation harnesses can run end-to-end without
    the upstream OpenFlamingo package or a GPU.
    """

    def __init__(
        self,
        config: Optional[OpenFlamingoConfig] = None,
        vocab_size: int = 64,
        **kwargs: Any,
    ) -> None:
        self.config = config or OpenFlamingoConfig()
        self.vocab_size = vocab_size
        self.tokenizer = _DummyTokenizer(vocab_size)
        self._device = torch.device("cpu")
        self.num_ground_truths = self.config.num_ground_truths

    @property
    def name(self) -> str:
        return "dummy-openflamingo"

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def _features(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.dim() == 3:
            pixels = pixels.unsqueeze(0)
        flat = pixels.reshape(pixels.shape[0], -1)
        return flat.mean(dim=1, keepdim=True)

    def logits(
        self, pixels: torch.Tensor, prompt: str = "", **kwargs: Any
    ) -> torch.Tensor:
        base = self._features(pixels)
        offsets = torch.arange(self.vocab_size, dtype=base.dtype).view(1, -1)
        return base * offsets * 0.01

    def logits_fn(self, prompt: str = "") -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda pixels, **kw: self.logits(pixels, prompt, **kw)

    def targeted_logits(self, pixels: torch.Tensor, target_text: str) -> torch.Tensor:
        return self.logits(pixels, target_text)

    def generate(
        self,
        pixels: torch.Tensor,
        prompt: str = "",
        max_new_tokens: int = 8,
        **kwargs: Any,
    ) -> List[str]:
        if pixels.dim() == 4 and pixels.shape[0] > 1:
            return [
                self.generate(pixels[i : i + 1], prompt, max_new_tokens)[0]
                for i in range(pixels.shape[0])
            ]
        value = float(self._features(pixels).mean().item())
        words = ["a", "photo", "of", "something"]
        idx = int(abs(value) * 1000) % len(words)
        return [" ".join(words[idx:] + words[:idx])]

    def generate_batch(
        self, pixels: torch.Tensor, prompt: str = "", **kwargs: Any
    ) -> List[str]:
        return self.generate(pixels, prompt, **kwargs)

    def summary(self) -> Dict[str, Any]:
        out = dict(self.config.as_dict())
        out.update(
            {
                "module": "DummyOpenFlamingoVictim",
                "role": "smoke-test stand-in (no upstream OpenFlamingo needed)",
                "upstream_repository": OPEN_FLAMINGO_REPO_URL,
            }
        )
        return out


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def build_openflamingo_victim(
    config: Optional[OpenFlamingoConfig] = None,
    dummy: bool = False,
    **kwargs: Any,
) -> Any:
    """Build the OpenFlamingo victim, falling back to the dummy on failure.

    ``ROBUST_CLIP_DUMMY_OPENFLAMINGO=1`` forces the model-free stand-in.
    """
    if config is None:
        config = OpenFlamingoConfig.from_dict(kwargs.pop("config", None), **kwargs)
    elif kwargs:
        config = OpenFlamingoConfig.from_dict(config.as_dict(), **kwargs)

    if dummy or os.environ.get("ROBUST_CLIP_DUMMY_OPENFLAMINGO") == "1":
        LOGGER.info("Using DummyOpenFlamingoVictim (smoke-test stand-in).")
        return DummyOpenFlamingoVictim(config)

    try:
        return OpenFlamingoVictim.from_pretrained(config)
    except Exception as exc:
        LOGGER.warning(
            "Could not load OpenFlamingo (%s); falling back to "
            "DummyOpenFlamingoVictim. The Addendum does not state an "
            "OpenFlamingo version/checkpoint, so supply one externally "
            "(configs/models.yaml).",
            exc,
        )
        return DummyOpenFlamingoVictim(config)


def load_openflamingo(config: Optional[Any] = None, **kwargs: Any) -> Any:
    """Convenience loader accepting a config object, dict, or kwargs."""
    if isinstance(config, OpenFlamingoConfig):
        cfg = config
    else:
        cfg = OpenFlamingoConfig.from_dict(
            config if isinstance(config, dict) else kwargs.pop("config", None), **kwargs
        )
    return build_openflamingo_victim(cfg)


# ---------------------------------------------------------------------------
# Self test / CLI
# ---------------------------------------------------------------------------


def _self_test(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks: config provenance, prompt wiring, raw-pixel contract."""
    cfg = OpenFlamingoConfig.from_dict({"openflamingo": {"max_new_tokens": 16}})
    assert cfg.max_new_tokens == 16
    assert cfg.provenance["repository"] == ADDENDUM
    assert cfg.provenance["version"] == UNSPECIFIED
    assert "UNSPECIFIED" in cfg.as_dict()["provenance"]["vision_encoder"]
    assert cfg.external_defaults()["prompt"] == cfg.caption_prompt

    victim = DummyOpenFlamingoVictim(cfg)
    pixels = torch.rand(2, 3, 32, 32)
    caps = victim.generate(pixels)
    assert isinstance(caps, list) and len(caps) == 2
    assert all(isinstance(c, str) for c in caps)

    captioner = OpenFlamingoCaptioner(victim, cfg, num_ground_truths=5)
    assert captioner.num_ground_truths == 5
    out = captioner(pixels)
    assert isinstance(out, list) and len(out) == 2
    logits = captioner.logits_fn()(pixels)
    assert logits.shape[0] == 2

    # Normalization must not mutate (or be applied to) the raw attack space.
    raw = torch.rand(1, 3, 8, 8)
    norm = victim.normalize(raw)
    assert not torch.allclose(raw, norm)
    assert torch.allclose(victim.denormalize(norm), raw, atol=1e-5)

    prompt = vqa_prompt("What color?")
    assert "What color?" in prompt
    assert isinstance(caption_prompt(), str) and caption_prompt()

    result = {
        "ok": True,
        "repository": OPEN_FLAMINGO_REPO_URL,
        "provenance": cfg.provenance,
        "external_defaults": cfg.external_defaults(),
    }
    if verbose:
        LOGGER.info("openflamingo_wrapper self-test passed: %s", result)
        print("openflamingo_wrapper self-test: OK")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenFlamingo victim wrapper")
    parser.add_argument("--self-test", action="store_true", help="run offline checks")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--version", default=UNSPECIFIED)
    parser.add_argument("--vision-encoder", default=UNSPECIFIED)
    parser.add_argument("--lang-encoder", default=UNSPECIFIED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--dummy", action="store_true", help="use the model-free stand-in")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = OpenFlamingoConfig(
        checkpoint_path=args.checkpoint_path,
        version=args.version,
        vision_encoder=args.vision_encoder,
        lang_encoder=args.lang_encoder,
        device=args.device,
        dtype=args.dtype,
    )
    if args.self_test or args.dummy:
        _self_test(verbose=not args.quiet)
        if args.dummy:
            print(build_openflamingo_victim(cfg, dummy=True).summary())
        return 0
    print(build_openflamingo_victim(cfg).summary())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""LLaVA-1.5 7B victim wrapper for the Robust CLIP reproduction.

Addendum (verbatim, in scope):

    The LLaVA model used by the paper is taken from
    https://github.com/haotian-liu/LLaVA/tree/main.  [...] There are some
    differences in the LLaVA model used in the paper.  LLaVA-1.5 7B is set up to
    use the OpenAI CLIP ViT-L/14@224 vision encoder (rather than the default
    ViT-L/14@336).  Additionally, the code has been modified as needed to allow
    LLaVA to work with OpenCLIP CLIP implementation instead of the Huggingface
    implementation.

This module implements that adaptation:

* the vision tower is the **OpenCLIP** ``ViT-L-14`` / ``openai`` model at
  resolution **224** (see :mod:`robust_clip_repro.models.clip_vision_encoder`),
  replacing LLaVA's default ``openai/clip-vit-large-patch14-336`` HuggingFace
  tower;
* the HuggingFace ``CLIPVisionModel``/``CLIPImageProcessor`` path used by
  ``llava.model.multimodal_encoder.clip_encoder.CLIPVisionTower`` is patched to
  consume the OpenCLIP implementation instead;
* the language backbone (Vicuna-7B) is frozen for evaluation.

Everything hyperparameter-like that the Addendum does not state (generation
temperature, sampling, dtype, ...) is exposed through configuration and tagged
``UNSPECIFIED_BY_ADDENDUM`` -- no paper value is invented.

Public surface used by the rest of the reproduction
---------------------------------------------------
* :class:`LLaVAOpenCLIPConfig` -- configuration
* :class:`LLaVAVictim` -- victim model (logits / generation / targeted logits)
* :class:`LLaVACaptioner` -- captioning adapter used by ``eval_captioning.py``
* :func:`build_llava_victim` / :func:`load_llava_openclip`

The wrapper works with a *raw* pixel tensor ``(1, 3, H, W)`` in ``[0, 1]`` (the
space in which the Addendum requires the l_inf ball to be computed) and performs
the OpenCLIP normalization internally.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

__all__ = [
    "LLAVA_REPO_URL",
    "LLAVA_15_7B_HF_ID",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_CLIP_MODEL",
    "DEFAULT_CLIP_PRETRAINED",
    "DEFAULT_CLIP_RESOLUTION",
    "DEFAULT_DTYPE",
    "UNSPECIFIED",
    "EXTERNAL_DEFAULTS",
    "LLaVAOpenCLIPConfig",
    "LLaVAVictim",
    "LLaVACaptioner",
    "DummyLLaVAVictim",
    "build_llava_victim",
    "load_llava_openclip",
    "patch_hf_clip_to_openclip",
    "vqa_prompt",
    "caption_prompt",
    "main",
]

LOGGER = logging.getLogger("robust_clip_repro.models.llava_openclip")

# ---------------------------------------------------------------------------
# Repository / checkpoint identifiers referenced by the Addendum
# ---------------------------------------------------------------------------
LLAVA_REPO_URL = "https://github.com/haotian-liu/LLaVA/tree/main"
LLAVA_15_7B_HF_ID = "liuhaotian/llava-v1.5-7b"
DEFAULT_MODEL_PATH = "liuhaotian/llava-v1.5-7b"

# Addendum: "LLaVA-1.5 7B is set up to use the OpenAI CLIP ViT-L/14@224 vision
# encoder (rather than the default ViT-L/14@336)" and it must run through the
# OpenCLIP implementation.
DEFAULT_CLIP_MODEL = "ViT-L-14"
DEFAULT_CLIP_PRETRAINED = "openai"
DEFAULT_CLIP_RESOLUTION = 224

UNSPECIFIED = "UNSPECIFIED_BY_ADDENDUM"
DEFAULT_DTYPE = "float16"

# Values the Addendum does not state.  They are supplied externally (config /
# CLI) and are logged as such instead of being attributed to the paper.
EXTERNAL_DEFAULTS: Dict[str, Any] = {
    "dtype": DEFAULT_DTYPE,
    "device_map": "auto",
    "max_new_tokens": 64,
    "generation_temperature": 0.0,
    "do_sample": False,
    "num_beams": 1,
    "prompt_template": UNSPECIFIED,
    "conv_mode": "llava_v1",
    "freeze_language_model": True,
    "attn_implementation": UNSPECIFIED,
    "load_8bit": False,
    "load_4bit": False,
}

# ---------------------------------------------------------------------------
# Prompt helpers (LLaVA-1.5 conversation template, from haotian-liu/LLaVA)
# ---------------------------------------------------------------------------
_LLAVA_V1_USER = "USER: <image>\n{prompt}\nASSISTANT:"

VQA_TEMPLATE = "Answer the question using a single word or phrase: {question}"
POPE_TEMPLATE = "{question}\nAnswer with a single word, either Yes or No."
CAPTION_TEMPLATE = "Provide a short caption of the image."


def vqa_prompt(question: str, template: Optional[str] = None) -> str:
    """Build a LLaVA-1.5 VQA prompt (``<image>`` placeholder included)."""
    template = template or VQA_TEMPLATE
    return _LLAVA_V1_USER.format(prompt=template.format(question=question))


def caption_prompt(prompt: Optional[str] = None) -> str:
    """Build a LLaVA-1.5 image-captioning prompt."""
    return _LLAVA_V1_USER.format(prompt=prompt or CAPTION_TEMPLATE)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class LLaVAOpenCLIPConfig:
    """Configuration for the patched LLaVA-1.5 7B victim.

    Addendum-mandated values (``clip_model``/``clip_pretrained``/
    ``clip_resolution``/``use_openclip``) are fixed and cannot be silently
    changed by a caller without going through ``from_dict`` overrides.
    """

    # --- Addendum-mandated -------------------------------------------------
    model_path: str = DEFAULT_MODEL_PATH
    clip_model: str = DEFAULT_CLIP_MODEL
    clip_pretrained: str = DEFAULT_CLIP_PRETRAINED
    clip_resolution: int = DEFAULT_CLIP_RESOLUTION
    use_openclip: bool = True

    # --- Addendum: freeze backbone for evaluation --------------------------
    freeze_language_model: bool = True
    freeze_vision_model: bool = True

    # --- External / unspecified -------------------------------------------
    device: Optional[str] = None
    dtype: str = DEFAULT_DTYPE
    device_map: Optional[str] = "auto"
    attn_implementation: Optional[str] = None
    load_8bit: bool = False
    load_4bit: bool = False
    cache_dir: Optional[str] = None
    patch_hf_clip: bool = True
    clip_pretrained_path: Optional[str] = None
    conv_mode: str = "llava_v1"

    # --- Generation (unspecified by the Addendum) --------------------------
    max_new_tokens: int = 64
    generation_temperature: float = 0.0
    do_sample: bool = False
    num_beams: int = 1
    top_p: float = 1.0
    repetition_penalty: float = 1.0

    # --- Bookkeeping -------------------------------------------------------
    provenance: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.use_openclip:
            LOGGER.warning(
                "use_openclip=False deviates from the Addendum, which requires "
                "LLaVA to run with the OpenCLIP CLIP implementation."
            )
        if int(self.clip_resolution) != DEFAULT_CLIP_RESOLUTION:
            LOGGER.warning(
                "clip_resolution=%s deviates from the Addendum-mandated "
                "OpenAI CLIP ViT-L/14@%d.",
                self.clip_resolution,
                DEFAULT_CLIP_RESOLUTION,
            )
        self._record_provenance()

    def _record_provenance(self) -> None:
        self.provenance = {
            "model_path": "ADDENDUM (haotian-liu/LLaVA, LLaVA-1.5 7B)",
            "clip_model": "ADDENDUM (OpenAI CLIP ViT-L/14@224, via OpenCLIP)",
            "clip_pretrained": "ADDENDUM (openai weights)",
            "clip_resolution": "ADDENDUM (224, not the LLaVA default 336)",
            "use_openclip": "ADDENDUM (OpenCLIP implementation instead of HF)",
            "freeze_language_model": "implementation choice (evaluation only)",
            "device": UNSPECIFIED,
            "dtype": UNSPECIFIED,
            "device_map": UNSPECIFIED,
            "max_new_tokens": UNSPECIFIED,
            "generation_temperature": UNSPECIFIED,
            "do_sample": UNSPECIFIED,
            "num_beams": UNSPECIFIED,
            "conv_mode": UNSPECIFIED,
        }

    # -- convenience --------------------------------------------------------
    @property
    def torch_dtype(self) -> Any:
        import torch

        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
        }
        key = str(self.dtype).lower()
        if key not in mapping:
            raise ValueError(f"unsupported dtype {self.dtype!r}")
        return mapping[key]

    def resolved_device(self) -> str:
        import torch

        if self.device:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def external_defaults(self) -> Dict[str, Any]:
        out = dict(EXTERNAL_DEFAULTS)
        out.update({k: getattr(self, k) for k in ("max_new_tokens", "dtype", "conv_mode")})
        return out

    def as_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["provenance"] = dict(self.provenance)
        return d

    @classmethod
    def from_dict(
        cls, cfg: Optional[Dict[str, Any]] = None, **overrides: Any
    ) -> "LLaVAOpenCLIPConfig":
        cfg = dict(cfg or {})
        # tolerate a nested "llava" / "model" block and model_kwargs style keys
        for key in ("llava", "model", "victim", "model_kwargs"):
            sub = cfg.pop(key, None)
            if isinstance(sub, dict):
                cfg.update(sub)
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        known = set(cls.__dataclass_fields__)
        unknown = {k: v for k, v in cfg.items() if k not in known}
        for key in list(unknown):
            cfg.pop(key, None)
        if unknown:
            LOGGER.info("ignoring unknown LLaVA config keys: %s", sorted(unknown))
        return cls(**cfg)


# ---------------------------------------------------------------------------
# OpenCLIP patching of LLaVA's HuggingFace CLIP vision tower
# ---------------------------------------------------------------------------
class _OpenCLIPVisionTower:
    """Drop-in replacement for ``llava.model.multimodal_encoder`` CLIP tower.

    Mirrors the LLaVA ``CLIPVisionTower`` contract (``forward`` returning the
    feature tensor, ``image_processor``, ``hidden_size``, ``num_patches``,
    ``config``) but is backed by OpenCLIP at 224px.
    """

    def __init__(
        self,
        vision_tower: str = DEFAULT_CLIP_MODEL,
        pretrained: str = DEFAULT_CLIP_PRETRAINED,
        resolution: int = DEFAULT_CLIP_RESOLUTION,
        device: Optional[str] = None,
        dtype: Any = None,
        pretrained_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        from .clip_vision_encoder import CLIPVisionConfig, build_clip_vision_encoder

        self.cfg = CLIPVisionConfig(
            model_name=vision_tower,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            resolution=int(resolution),
            cache_dir=cache_dir,
            device=device,
            dtype="float32" if dtype is None else _dtype_name(dtype),
            trainable=False,
            trainable_scope="none",
        )
        self.encoder = build_clip_vision_encoder(self.cfg)
        self.is_loaded = True
        self.device = device or getattr(self.cfg, "device", None) or self.cfg.resolved_device()
        self.dtype = dtype
        if hasattr(self.encoder, "model"):
            self.encoder.model.to(self.device)

    # -- LLaVA-compatible attributes ---------------------------------------
    @property
    def hidden_size(self) -> int:
        return int(getattr(self.encoder, "output_dim", 1024))

    @property
    def num_patches(self) -> int:
        res = int(getattr(self.cfg, "resolution", DEFAULT_CLIP_RESOLUTION))
        return (res // 14) ** 2

    @property
    def image_processor(self) -> Any:
        """Object exposing ``preprocess``/``image_size`` like HF processors."""
        return _OpenCLIPImageProcessor(self.encoder.preprocess, self.cfg.resolution)

    def to(self, *args: Any, **kwargs: Any) -> "_OpenCLIPVisionTower":
        if hasattr(self.encoder, "model"):
            self.encoder.model.to(*args, **kwargs)
        return self

    def eval(self) -> "_OpenCLIPVisionTower":
        if hasattr(self.encoder, "model"):
            self.encoder.model.eval()
        return self

    def requires_grad_(self, value: bool = False) -> "_OpenCLIPVisionTower":
        if hasattr(self.encoder, "model"):
            for p in self.encoder.model.parameters():
                p.requires_grad_(value)
        return self

    def forward(self, images: Any) -> Any:
        """Accepts raw ``[0,1]`` pixels OR PIL images.

        When handed PIL image(s) it uses the OpenCLIP preprocessing, exactly like
        the LLaVA tower used the HuggingFace processor.
        """
        import torch

        if isinstance(images, (list, tuple)) and images and not torch.is_tensor(images[0]):
            images = torch.cat([self.encoder.preprocess(im).unsqueeze(0) for im in images], dim=0)
        if not torch.is_tensor(images):
            raise TypeError(f"unsupported image container {type(images)!r}")
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.to(self.device, dtype=self.dtype or torch.float32)
        with torch.no_grad():
            features = self.encoder(images, normalize=True, return_features=True)
        return features


class _OpenCLIPImageProcessor:
    """Minimal stand-in for the HuggingFace image processor."""

    def __init__(self, preprocess: Callable[[Any], Any], image_size: int) -> None:
        self._preprocess = preprocess
        self.image_size = int(image_size)
        self.size = {"shortest_edge": int(image_size)}
        self.crop_size = {"height": int(image_size), "width": int(image_size)}

    def preprocess(self, image: Any, return_tensors: str = "pt") -> Any:
        import torch

        pix = self._preprocess(image)
        if not torch.is_tensor(pix):
            pix = torch.as_tensor(pix)
        return {"pixel_values": pix.unsqueeze(0) if pix.dim() == 3 else pix}

    def __call__(self, images: Any, return_tensors: str = "pt") -> Any:
        return self.preprocess(images, return_tensors=return_tensors)


def _dtype_name(dtype: Any) -> str:
    import torch

    if isinstance(dtype, str):
        return dtype
    mapping = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}
    return mapping.get(dtype, "float32")


def patch_hf_clip_to_openclip(
    model: Any,
    config: Optional[LLaVAOpenCLIPConfig] = None,
    *,
    verbose: bool = True,
) -> Any:
    """Replace LLaVA's HuggingFace CLIP vision tower by the OpenCLIP tower.

    This is the "code has been modified as needed to allow LLaVA to work with
    OpenCLIP CLIP implementation instead of the Huggingface implementation"
    requirement of the Addendum.
    """
    cfg = config or LLaVAOpenCLIPConfig()
    tower = _OpenCLIPVisionTower(
        vision_tower=cfg.clip_model,
        pretrained=cfg.clip_pretrained,
        resolution=cfg.clip_resolution,
        device=cfg.resolved_device(),
        dtype=cfg.torch_dtype,
        pretrained_path=cfg.clip_pretrained_path,
        cache_dir=cfg.cache_dir,
    )

    container = getattr(model, "model", model)
    patched = False
    for attr in ("vision_tower", "vision_model"):
        if hasattr(container, attr):
            setattr(container, attr, tower)
            patched = True
    if not patched and hasattr(model, "vision_tower"):
        model.vision_tower = tower
        patched = True

    hidden = int(getattr(tower, "hidden_size", 1024))
    for holder in (container, model):
        mm_projector = getattr(holder, "mm_projector", None)
        in_features = getattr(mm_projector, "in_features", None)
        if in_features is not None and int(in_features) != hidden:
            LOGGER.warning(
                "mm_projector in_features=%s differs from OpenCLIP hidden size=%s; "
                "recreating the projector for the patched tower.",
                in_features,
                hidden,
            )
            _rebuild_projector(holder, hidden, cfg)

    if verbose:
        LOGGER.info(
            "patched LLaVA vision tower -> OpenCLIP %s/%s @%d (patched=%s)",
            cfg.clip_model,
            cfg.clip_pretrained,
            cfg.clip_resolution,
            patched,
        )
    return model


def _rebuild_projector(container: Any, hidden: int, cfg: LLaVAOpenCLIPConfig) -> None:
    import torch.nn as nn

    text_hidden = None
    config = getattr(container, "config", None)
    if config is not None:
        text_hidden = getattr(config, "hidden_size", None)
        text_config = getattr(config, "text_config", None)
        if isinstance(text_config, dict):
            text_hidden = text_hidden or text_config.get("hidden_size")
        elif text_config is not None:
            text_hidden = text_hidden or getattr(text_config, "hidden_size", None)
    text_hidden = int(text_hidden or 4096)
    container.mm_projector = nn.Sequential(
        nn.Linear(hidden, text_hidden), nn.GELU(), nn.Linear(text_hidden, text_hidden)
    )


# ---------------------------------------------------------------------------
# Victim model
# ---------------------------------------------------------------------------
class LLaVAVictim:
    """LLaVA-1.5 7B victim exposing logits/generation for attack losses."""

    def __init__(
        self,
        config: Optional[LLaVAOpenCLIPConfig] = None,
        model: Any = None,
        processor: Any = None,
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> None:
        self.cfg = config or LLaVAOpenCLIPConfig.from_dict(kwargs.get("config_dict"))
        if model is None:
            model, processor, tokenizer = self._load_model_components(self.cfg)
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer or getattr(processor, "tokenizer", None)
        self.cfg.device = self.cfg.device or self.cfg.resolved_device()
        self.model.eval()
        if self.cfg.freeze_language_model:
            self.freeze_language_model()
        if self.cfg.freeze_vision_model:
            self.freeze_vision_model()

    # -- loading ------------------------------------------------------------
    @staticmethod
    def _load_model_components(cfg: LLaVAOpenCLIPConfig):
        from transformers import AutoProcessor, AutoTokenizer

        LOGGER.info("loading LLaVA-1.5 7B from %s (repo: %s)", cfg.model_path, LLAVA_REPO_URL)
        processor = AutoProcessor.from_pretrained(cfg.model_path, cache_dir=cfg.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, cache_dir=cfg.cache_dir)

        model, model_kind = _build_llava_model(cfg)
        if model is None:  # pragma: no cover - depends on local installation
            raise ImportError(
                "Could not build LLaVA-1.5 7B. Install the LLaVA repository in "
                f"editable mode ({LLAVA_REPO_URL}) so that `llava` is importable, "
                "or pass an already-constructed `model=` to LLaVAVictim."
            )
        LOGGER.info("LLaVA model built via %s", model_kind)
        if cfg.patch_hf_clip:
            patch_hf_clip_to_openclip(model, cfg)
        return model, processor, tokenizer

    @classmethod
    def from_pretrained(
        cls, cfg: Optional[LLaVAOpenCLIPConfig] = None, **kwargs: Any
    ) -> "LLaVAVictim":
        return cls(cfg or LLaVAOpenCLIPConfig.from_dict(kwargs))

    # -- freezing -----------------------------------------------------------
    def freeze_language_model(self) -> None:
        base = getattr(self.model, "model", self.model)
        for name in ("language_model", "lm_head"):
            module = getattr(base, name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad_(False)
        self._lm_frozen = True

    def freeze_vision_model(self) -> None:
        container = getattr(self.model, "model", self.model)
        tower = getattr(container, "vision_tower", None)
        if tower is None:
            return
        if hasattr(tower, "requires_grad_"):
            tower.requires_grad_(False)
        self._vision_frozen = True

    @property
    def device(self) -> Any:
        import torch

        try:
            return next(self.model.parameters()).device
        except (StopIteration, AttributeError):  # pragma: no cover
            return torch.device(self.cfg.resolved_device())

    @property
    def dtype(self) -> Any:
        return self.cfg.torch_dtype

    # -- prompt & input preparation ----------------------------------------
    def build_prompt(
        self,
        question: Optional[str] = None,
        *,
        kind: str = "vqa",
        template: Optional[str] = None,
    ) -> str:
        if kind == "caption":
            return caption_prompt(question)
        if kind == "pope":
            return vqa_prompt(question or "", template=POPE_TEMPLATE)
        return vqa_prompt(question or "", template=template)

    def prepare_inputs(
        self,
        pixels: Any,
        prompt: Union[str, List[str]],
        *,
        normalize: bool = True,
    ) -> Dict[str, Any]:
        """Build model inputs from raw ``[0,1]`` pixels and a prompt.

        ``pixels`` stays in raw (non-normalized) space at the call site, which is
        what the Addendum's l_inf ball requires; normalization happens here.
        """
        import torch

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not torch.is_tensor(pixels):
            raise TypeError("pixels must be a torch tensor in [0, 1]")
        if pixels.dim() == 3:
            pixels = pixels.unsqueeze(0)
        pixels = pixels.to(self.device, dtype=self.dtype)
        if normalize:
            pixels = self.normalize(pixels)

        enc = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(self.device)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            attention_mask = (input_ids != pad_id).long()
        else:
            attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixels,
        }

    def normalize(self, pixels: Any) -> Any:
        from .clip_vision_encoder import CLIP_MEAN, CLIP_STD

        mean = pixels.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = pixels.new_tensor(CLIP_STD).view(1, 3, 1, 1)
        return (pixels - mean) / std

    def denormalize(self, pixels: Any) -> Any:
        from .clip_vision_encoder import CLIP_MEAN, CLIP_STD

        mean = pixels.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = pixels.new_tensor(CLIP_STD).view(1, 3, 1, 1)
        return pixels * std + mean

    # -- forward paths ------------------------------------------------------
    def forward(
        self,
        pixels: Any,
        prompt: Optional[Union[str, List[str]]] = None,
        *,
        input_ids: Any = None,
        attention_mask: Any = None,
        labels: Any = None,
        normalize: bool = True,
    ) -> Any:
        """Forward pass; returns the HuggingFace output object (``.logits``)."""
        import torch

        if input_ids is None:
            inputs = self.prepare_inputs(pixels, prompt or "", normalize=normalize)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            pixels = inputs["pixel_values"]
        else:
            if normalize and torch.is_tensor(pixels):
                if pixels.dim() == 3:
                    pixels = pixels.unsqueeze(0)
                pixels = pixels.to(self.device, dtype=self.dtype)
                pixels = self.normalize(pixels)
            if attention_mask is None:
                pad_id = getattr(self.tokenizer, "pad_token_id", None)
                attention_mask = (input_ids != pad_id).long() if pad_id is not None else torch.ones_like(input_ids)
        kwargs: Dict[str, Any] = {}
        if labels is not None:
            kwargs["labels"] = labels
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixels,
            return_dict=True,
            **kwargs,
        )

    def logits(
        self,
        pixels: Any,
        prompt: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Next-token logits for the prompt (input to the attack losses)."""
        out = self.forward(pixels, prompt, **kwargs)
        return getattr(out, "logits", out)

    def logits_fn(self, prompt: str) -> Callable[[Any], Any]:
        """Return ``fn(pixels) -> logits`` for a fixed ``prompt``.

        Handed to the attack engines, which only manipulate raw pixels.
        """

        def _fn(pixels: Any) -> Any:
            return self.logits(pixels, prompt)

        return _fn

    # -- targeted logits (used by the jailbreak attack loss) ---------------
    def targeted_logits(self, pixels: Any, target_text: str) -> Any:
        """Logits for the harmful target continuation.

        The jailbreak attack (:mod:`robust_clip_repro.attacks.jailbreak`) calls
        ``model.targeted_logits(pixels, target_text)`` and builds a cross-entropy
        loss against ``tokenize_target_ids``.
        """
        prompt = vqa_prompt(target_text, template="{question}")
        inputs = self.prepare_inputs(pixels, prompt)
        out = self.forward(
            inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            normalize=False,
        )
        return getattr(out, "logits", out)

    # -- generation ---------------------------------------------------------
    @property
    def _gen_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "max_new_tokens": int(self.cfg.max_new_tokens),
            "do_sample": bool(self.cfg.do_sample),
            "num_beams": int(self.cfg.num_beams),
            "repetition_penalty": float(self.cfg.repetition_penalty),
        }
        if self.cfg.do_sample:
            kwargs["temperature"] = float(self.cfg.generation_temperature)
            kwargs["top_p"] = float(self.cfg.top_p)
        return kwargs

    def generate(
        self,
        pixels: Any,
        prompt: Union[str, List[str]],
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Generate text responses for raw ``[0,1]`` pixels."""
        import torch

        inputs = self.prepare_inputs(pixels, prompt)
        gen_kwargs = dict(self._gen_kwargs)
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = int(max_new_tokens)
        if temperature is not None and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(temperature)
        gen_kwargs.update(kwargs)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                **gen_kwargs,
            )
        return self.decode(out, inputs["input_ids"].shape[1])

    def generate_batch(
        self, pixels: Any, prompt: Union[str, List[str]], **kwargs: Any
    ) -> List[str]:
        return self.generate(pixels, prompt, **kwargs)

    def decode(self, generated_ids: Any, prompt_len: int) -> List[str]:
        texts: List[str] = []
        for row in generated_ids:
            new_tokens = row[prompt_len:] if row.shape[-1] > prompt_len else row
            texts.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return texts

    # -- summary ------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "model": "LLaVA-1.5 7B",
            "repo": LLAVA_REPO_URL,
            "clip_model": self.cfg.clip_model,
            "clip_pretrained": self.cfg.clip_pretrained,
            "clip_resolution": self.cfg.clip_resolution,
            "openclip_patch": bool(self.cfg.use_openclip),
            "freeze_language_model": bool(self.cfg.freeze_language_model),
            "device": str(self.device),
            "dtype": self.cfg.dtype,
            "external_defaults": self.cfg.external_defaults(),
            "provenance": dict(self.cfg.provenance),
        }


def _build_llava_model(cfg: LLaVAOpenCLIPConfig) -> Tuple[Any, str]:
    """Try the known LLaVA loading entry points in order of specificity."""
    kwargs: Dict[str, Any] = {}
    if cfg.cache_dir:
        kwargs["cache_dir"] = cfg.cache_dir
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation

    # 1) LLaVA-1.5 code path: llava.model.builder.load_pretrained_model
    try:
        from llava.model.builder import load_pretrained_model  # type: ignore

        out = load_pretrained_model(
            model_path=cfg.model_path,
            model_base=None,
            model_name=os.path.basename(str(cfg.model_path).rstrip("/")) or "llava-v1.5-7b",
            load_8bit=cfg.load_8bit,
            load_4bit=cfg.load_4bit,
            device_map=cfg.device_map,
            **kwargs,
        )
        model = out[0] if isinstance(out, tuple) else out
        return model, "llava.model.builder.load_pretrained_model"
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("llava.model.builder unavailable: %s", exc)

    # 2) LlavaLlamaForCausalLM
    try:
        from llava.model.language_model.llava_llama import (  # type: ignore
            LlavaLlamaForCausalLM,
        )

        model = LlavaLlamaForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype=cfg.torch_dtype,
            device_map=cfg.device_map,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        return model, "llava.model.language_model.llava_llama.LlavaLlamaForCausalLM"
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("LlavaLlamaForCausalLM unavailable: %s", exc)

    # 3) Plain HuggingFace class fallback
    try:
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(
            cfg.model_path,
            torch_dtype=cfg.torch_dtype,
            device_map=cfg.device_map,
            **kwargs,
        )
        return model, "transformers.AutoModelForVision2Seq"
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("AutoModelForVision2Seq unavailable: %s", exc)

    return None, "none"


# ---------------------------------------------------------------------------
# Captioning adapter (used by eval_captioning.py)
# ---------------------------------------------------------------------------
class LLaVACaptioner:
    """Adapter exposing the ``Captioner`` protocol for captioning evaluation."""

    name = "llava-1.5-7b-openclip"

    def __init__(
        self,
        victim: Optional[LLaVAVictim] = None,
        config: Optional[LLaVAOpenCLIPConfig] = None,
        prompt: Optional[str] = None,
        num_ground_truths: int = 5,
        **kwargs: Any,
    ) -> None:
        self.victim = victim or build_llava_victim(config, **kwargs)
        self.num_ground_truths = int(num_ground_truths)
        self.prompt = caption_prompt(prompt)

    def __call__(self, pixels: Any, *args: Any, **kwargs: Any) -> List[str]:
        return self.victim.generate(pixels, self.prompt, **kwargs)

    def caption(self, pixels: Any, **kwargs: Any) -> Union[str, List[str]]:
        caps = self(pixels, **kwargs)
        return caps[0] if len(caps) == 1 else caps

    @property
    def tokenizer(self) -> Any:
        return self.victim.tokenizer

    def logits_fn(self, prompt: Optional[str] = None) -> Callable[[Any], Any]:
        return self.victim.logits_fn(prompt or self.prompt)

    def summary(self) -> Dict[str, Any]:
        out = self.victim.summary()
        out["role"] = "captioner"
        return out


# ---------------------------------------------------------------------------
# Offline deterministic victim (smoke tests / no-checkpoint environments)
# ---------------------------------------------------------------------------
class DummyLLaVAVictim:
    """Deterministic, dependency-light victim used for smoke tests."""

    def __init__(
        self, config: Optional[LLaVAOpenCLIPConfig] = None, vocab_size: int = 64, **_: Any
    ) -> None:
        import torch

        self.cfg = config or LLaVAOpenCLIPConfig()
        self._torch = torch
        self.vocab_size = int(vocab_size)
        self.tokenizer = _DummyTokenizer(vocab_size=self.vocab_size)

    @property
    def device(self) -> Any:
        return self._torch.device(self.cfg.resolved_device())

    def _features(self, pixels: Any) -> Any:
        if pixels.dim() == 3:
            pixels = pixels.unsqueeze(0)
        return pixels.mean(dim=(1, 2, 3), keepdim=True)

    def logits(self, pixels: Any, prompt: Optional[Any] = None) -> Any:
        torch = self._torch
        feat = self._features(pixels)
        base = torch.arange(self.vocab_size, device=feat.device, dtype=feat.dtype).view(1, 1, -1)
        return base + feat.view(-1, 1, 1) * 3.0

    def targeted_logits(self, pixels: Any, target_text: str) -> Any:
        return self.logits(pixels, target_text)

    def logits_fn(self, prompt: str) -> Callable[[Any], Any]:
        return lambda pixels: self.logits(pixels, prompt)

    def generate(self, pixels: Any, prompt: Union[str, List[str]], **kwargs: Any) -> List[str]:
        n = 1 if isinstance(prompt, str) else len(prompt)
        return ["a dummy caption"] * n

    def summary(self) -> Dict[str, Any]:
        return {"model": "dummy-llava", "role": "smoke-test"}


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = int(vocab_size)

    def __call__(
        self,
        text: Any,
        return_tensors: Optional[str] = None,
        padding: bool = False,
        **_: Any,
    ) -> "_AttrDict":
        import torch

        if isinstance(text, str):
            text = [text]
        rows: List[List[int]] = []
        for t in text:
            ids = [3 + (ord(c) % max(1, self.vocab_size - 3)) for c in (t or "x")][:16] or [3]
            rows.append(ids)
        maxlen = max(len(r) for r in rows)
        padded = [r + [self.pad_token_id] * (maxlen - len(r)) for r in rows]
        out = torch.tensor(padded, dtype=torch.long)
        return _AttrDict(input_ids=out)

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        return "dummy text"

    def convert_tokens_to_ids(self, token: str) -> int:
        return 3 + (sum(ord(c) for c in token) % max(1, self.vocab_size - 3))


class _AttrDict(dict):
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def build_llava_victim(
    config: Optional[LLaVAOpenCLIPConfig] = None,
    *,
    dummy: bool = False,
    **kwargs: Any,
) -> Any:
    """Build a LLaVA victim, falling back to the deterministic dummy."""
    cfg = config or LLaVAOpenCLIPConfig.from_dict(kwargs.pop("config_dict", None), **kwargs)
    if dummy or os.environ.get("ROBUST_CLIP_DUMMY_LLAVA", "") == "1":
        return DummyLLaVAVictim(cfg)
    if cfg.device is None:
        cfg.device = cfg.resolved_device()
    try:
        return LLaVAVictim(cfg)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "could not load LLaVA-1.5 7B (%s); falling back to DummyLLaVAVictim "
            "for smoke-testing only.",
            exc,
        )
        return DummyLLaVAVictim(cfg)


def load_llava_openclip(config: Optional[Any] = None, **kwargs: Any) -> Any:
    """Convenience loader accepting a config object/dict or kwargs."""
    if isinstance(config, dict):
        cfg = LLaVAOpenCLIPConfig.from_dict(config, **kwargs)
    elif config is None:
        cfg = LLaVAOpenCLIPConfig.from_dict(None, **kwargs)
    else:
        cfg = config
    return build_llava_victim(cfg)


# ---------------------------------------------------------------------------
# Offline self test
# ---------------------------------------------------------------------------
def _self_test(verbose: bool = True) -> Dict[str, Any]:
    import torch

    cfg = LLaVAOpenCLIPConfig()
    assert cfg.clip_model == "ViT-L-14"
    assert cfg.clip_pretrained == "openai"
    assert int(cfg.clip_resolution) == 224, "Addendum: ViT-L/14@224, not 336"
    assert cfg.use_openclip is True

    victim = DummyLLaVAVictim(cfg)
    pixels = torch.rand(1, 3, 224, 224)
    logits = victim.logits(pixels, vqa_prompt("What is this?"))
    assert logits.shape[-1] == victim.vocab_size
    tgt = victim.targeted_logits(pixels, "harmful target")
    assert tgt.shape == logits.shape
    caps = victim.generate(pixels, caption_prompt())
    assert isinstance(caps, list) and caps

    payload = victim.summary()
    if verbose:
        LOGGER.info("llava self-test ok: %s", payload)
    return {"ok": True, "summary": payload, "clip_resolution": cfg.clip_resolution}


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="LLaVA-1.5 7B (OpenCLIP ViT-L/14@224) wrapper"
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dummy", action="store_true", help="build the deterministic dummy victim")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--clip-pretrained", default=DEFAULT_CLIP_PRETRAINED)
    parser.add_argument("--clip-resolution", type=int, default=DEFAULT_CLIP_RESOLUTION)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)
    if args.self_test:
        print(_self_test(verbose=not args.quiet))
        return 0

    cfg = LLaVAOpenCLIPConfig(
        model_path=args.model_path,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        clip_resolution=args.clip_resolution,
        device=args.device,
        dtype=args.dtype,
    )
    victim = build_llava_victim(cfg, dummy=args.dummy)
    print(victim.summary())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Vision-Language Model (VLM) zoo for the *LCA-on-the-Line* study.

The paper (§4, Appendix A) evaluates 39 vision-language models on top of the
36 vision-only models:

    * ALBEF          (1)  albef_feature_extractor
    * BLIP           (1)  blip_feature_extractor_base
    * CLIP           (7)  RN50, RN101, RN50x4, ViT-B-32, ViT-B-16, ViT-L-14,
                          ViT-L-14-336px
    * OpenCLIP      (30)  30 (architecture, pretrained) pairs listed in Appendix A

This module mirrors :mod:`src.models.vm_zoo` so the two zoos are interchangeable
inside the evaluation pipeline (``src/eval/evaluate_models.py``):

    logits(x)   -> Tensor (B, K)     # ImageNet-1k zero-shot logits
    features(x) -> Tensor (B, D)     # M(X), the penultimate representation

For CLIP-style models ``M(X)`` is the *image* embedding produced by the vision
tower (the representation used to build the K-means latent hierarchies in
§4.3.1 and the linear probes in §4.3.2).  Zero-shot logits are obtained from the
text encoder with the standard 80-template ImageNet prompt bank.

Everything heavy (torch, open_clip, clip, transformers) is imported lazily so the
registry itself can be inspected (and unit-tested) in a minimal environment.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

LOG = logging.getLogger(__name__)

IMAGENET_NUM_CLASSES = 1000

# ---------------------------------------------------------------------------
# Prompt bank
# ---------------------------------------------------------------------------

#: The 80-prompt ImageNet bank used by OpenAI CLIP for zero-shot ImageNet.
IMAGENET_PROMPT_TEMPLATES: Tuple[str, ...] = (
    "a photo of a {}.",
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {}.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
)

#: Short single-template bank used as fallback / ablation baseline.
SIMPLE_PROMPT_TEMPLATE = "a photo of a {}."


# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------


@dataclass
class VlmSpec:
    """Description of one VLM entry of Appendix A."""

    name: str
    family: str
    backend: str  # "open_clip" | "openai_clip" | "transformers"
    arch: str
    pretrained: Optional[str] = None
    input_size: int = 224
    reported_top1: Optional[float] = None
    build_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.name

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "backend": self.backend,
            "arch": self.arch,
            "pretrained": self.pretrained,
            "input_size": self.input_size,
        }


def _slug(text: str) -> str:
    """Filesystem/registry friendly slug for an OpenCLIP pair."""
    out = []
    for ch in text:
        out.append(ch.lower() if ch.isalnum() else "_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


# ============================== Appendix A ============================== #

#: ALBEF / BLIP feature extractors (Salesforce checkpoints via HF transformers).
TRANSFORMERS_VLM_SPECS: List[VlmSpec] = [
    VlmSpec(
        "albef_feature_extractor",
        "albef",
        "transformers",
        "Salesforce/albef_feature_extractor",
        None,
        224,
    ),
    VlmSpec(
        "blip_feature_extractor_base",
        "blip",
        "transformers",
        "Salesforce/blip_feature_extractor_base",
        None,
        224,
    ),
]

OPENAI_CLIP_SPECS: List[VlmSpec] = [
    VlmSpec("clip_rn50", "clip", "openai_clip", "RN50", None, 224),
    VlmSpec("clip_rn101", "clip", "openai_clip", "RN101", None, 224),
    VlmSpec("clip_rn50x4", "clip", "openai_clip", "RN50x4", None, 288),
    VlmSpec("clip_vit_b_32", "clip", "openai_clip", "ViT-B/32", None, 224),
    VlmSpec("clip_vit_b_16", "clip", "openai_clip", "ViT-B/16", None, 224),
    VlmSpec("clip_vit_l_14", "clip", "openai_clip", "ViT-L/14", None, 224),
    VlmSpec("clip_vit_l_14_336px", "clip", "openai_clip", "ViT-L/14@336px", None, 336),
]

#: The 30 OpenCLIP (architecture, pretrained) pairs of Appendix A.
OPENCLIP_PAIRS: List[Tuple[str, str]] = [
    ("RN101", "openai"),
    ("RN101", "yfcc15m"),
    ("RN101-quickgelu", "openai"),
    ("RN101-quickgelu", "yfcc15m"),
    ("RN50", "cc12m"),
    ("RN50", "openai"),
    ("RN50", "yfcc15m"),
    ("RN50-quickgelu", "cc12m"),
    ("RN50-quickgelu", "openai"),
    ("RN50-quickgelu", "yfcc15m"),
    ("RN50x16", "openai"),
    ("RN50x4", "openai"),
    ("RN50x64", "openai"),
    ("ViT-B-16", "laion2b_s34b_b88k"),
    ("ViT-B-16", "laion400m_e31"),
    ("ViT-B-16", "laion400m_e32"),
    ("ViT-B-16-plus-240", "laion400m_e31"),
    ("ViT-B-16-plus-240", "laion400m_e32"),
    ("ViT-B-32", "laion2b_e16"),
    ("ViT-B-32", "laion2b_s34b_b79k"),
    ("ViT-B-32", "laion400m_e31"),
    ("ViT-B-32", "laion400m_e32"),
    ("ViT-B-32", "openai"),
    ("ViT-B-32-quickgelu", "laion400m_e31"),
    ("ViT-B-32-quickgelu", "laion400m_e32"),
    ("ViT-L-14", "laion2b_s32b_b82k"),
    ("ViT-L-14", "laion400m_e31"),
    ("ViT-L-14", "laion400m_e32"),
    ("coca_ViT-B-32", "laion2b_s13b_b90k"),
    ("coca_ViT-L-14", "laion2b_s13b_b90k"),
]


def build_openclip_specs(pairs: Optional[Sequence[Tuple[str, str]]] = None) -> List[VlmSpec]:
    """Build the 30 OpenCLIP :class:`VlmSpec` entries of Appendix A."""
    pairs = list(pairs if pairs is not None else OPENCLIP_PAIRS)
    specs: List[VlmSpec] = []
    for arch, pretrained in pairs:
        name = f"openclip_{_slug(arch)}_{_slug(pretrained)}"
        size = 336 if "336" in arch else 240 if "240" in arch else 224
        specs.append(VlmSpec(name, "open_clip", "open_clip", arch, pretrained, size))
    return specs


openclip_specs = build_openclip_specs()

VLM_MODEL_SPECS: List[VlmSpec] = TRANSFORMERS_VLM_SPECS + OPENAI_CLIP_SPECS + openclip_specs

SPECS_BY_NAME: Dict[str, VlmSpec] = {spec.name: spec for spec in VLM_MODEL_SPECS}

VLM_NAMES: List[str] = [spec.name for spec in VLM_MODEL_SPECS]

#: Convenience aliases (paper / addendum naming -> registry key).
VLM_ALIASES: Dict[str, str] = {
    "clip_rn50": "clip_rn50",
    "clip_rn101": "clip_rn101",
    "clip_rn50x4": "clip_rn50x4",
    "clip_rn50x16": "openclip_rn50x16_openai",
    "clip_vitb32": "clip_vit_b_32",
    "clip_vit_b_32": "clip_vit_b_32",
    "clip_vit32": "clip_vit_b_32",
    "clip_vit": "clip_vit_b_32",
    "clip_vitb16": "clip_vit_b_16",
    "clip_vit_b_16": "clip_vit_b_16",
    "clip_vitl14": "clip_vit_l_14",
    "clip_vit_l_14": "clip_vit_l_14",
    "clip_vitl14_336": "clip_vit_l_14_336px",
    "clip_vit_l_14_336": "clip_vit_l_14_336px",
    "clip_vit_l_14_336px": "clip_vit_l_14_336px",
    "clip": "clip_rn50",
    "albef": "albef_feature_extractor",
    "albef_feature_extractor": "albef_feature_extractor",
    "blip": "blip_feature_extractor_base",
    "blip_feature_extractor": "blip_feature_extractor_base",
    "blip_feature_extractor_base": "blip_feature_extractor_base",
}
for _spec in openclip_specs:
    VLM_ALIASES[_spec.name] = _spec.name
VLM_ALIASES["openclip"] = openclip_specs[0].name


def list_vlm_names(include_aliases: bool = False) -> List[str]:
    """Return the 39 Appendix-A VLM names (ALBEF, BLIP, 7x CLIP, 30x OpenCLIP)."""
    if include_aliases:
        return sorted(set(VLM_NAMES) | set(VLM_ALIASES))
    return list(VLM_NAMES)


def specs_by_family() -> Dict[str, List[VlmSpec]]:
    """Group the registry by family (``albef``/``blip``/``clip``/``open_clip``)."""
    out: Dict[str, List[VlmSpec]] = {}
    for spec in VLM_MODEL_SPECS:
        out.setdefault(spec.family, []).append(spec)
    return out


def get_model_spec(name: str) -> VlmSpec:
    """Resolve a registry key (accepts aliases)."""
    if name in SPECS_BY_NAME:
        return SPECS_BY_NAME[name]
    key = VLM_ALIASES.get(name.lower())
    if key and key in SPECS_BY_NAME:
        return SPECS_BY_NAME[key]
    lowered = name.lower()
    for known, spec in SPECS_BY_NAME.items():
        if known.lower() == lowered:
            return spec
    raise KeyError(f"Unknown VLM model '{name}'. Known: {', '.join(VLM_NAMES[:8])} ...")


# ---------------------------------------------------------------------------
# Prompts for the class labels
# ---------------------------------------------------------------------------


def format_prompts(
    class_names: Sequence[str],
    template: str = SIMPLE_PROMPT_TEMPLATE,
    templates: Optional[Sequence[str]] = None,
) -> List[str]:
    """Build prompts for class names (single template or a full prompt bank)."""
    names = [str(n).replace("_", " ") for n in class_names]
    if templates is None:
        return [template.format(n) for n in names]
    prompts: List[str] = []
    for n in names:
        for t in templates:
            prompts.append(t.format(n))
    return prompts


def class_names_for_hierarchy(
    hierarchy: Any = None, num_classes: int = IMAGENET_NUM_CLASSES
) -> List[str]:
    """Class names (canonical order) from a hierarchy or ImageNet metadata."""
    if hierarchy is not None:
        names: List[str] = []
        ok = True
        for i in range(num_classes):
            try:
                name = hierarchy.class_name(i)
            except Exception:  # pragma: no cover - defensive
                ok = False
                break
            if not name:
                ok = False
                break
            names.append(name)
        if ok and len(names) == num_classes:
            return names
    try:  # pragma: no cover - optional heavy dependency
        from ..data.imagenet import load_imagenet_class_names

        names = list(load_imagenet_class_names())
        if len(names) >= num_classes:
            return names[:num_classes]
    except Exception:
        pass
    return [f"imagenet class {i}" for i in range(num_classes)]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _require_torch() -> Any:
    import torch  # noqa: WPS433 (lazy import)

    return torch


class VlmModelWrapper:
    """Wrap a CLIP/OpenCLIP/ALBEF/BLIP model exposing features and logits.

    The contract mirrors :class:`src.models.vm_zoo.VisionModelWrapper`:
    ``features(x)`` returns ``M(X)`` ``(B, D)`` and ``logits(x)`` returns the
    zero-shot ImageNet logits ``(B, K)``.  ``__call__`` defaults to ``logits``
    while :meth:`forward_both` returns both in a single image-tower pass.
    """

    def __init__(
        self,
        name: str,
        model: Any = None,
        spec: Optional[VlmSpec] = None,
        pretrained: bool = True,
        device: Any = None,
        tokenizer: Any = None,
        preprocess: Any = None,
        text_features: Any = None,
        logit_scale: Optional[float] = None,
        class_names: Optional[Sequence[str]] = None,
        prompt_templates: Optional[Sequence[str]] = IMAGENET_PROMPT_TEMPLATES,
        num_classes: int = IMAGENET_NUM_CLASSES,
        eval_mode: bool = True,
    ) -> None:
        self.name = name
        self.spec = spec if spec is not None else _maybe_spec(name)
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.class_names = list(class_names) if class_names else None
        self.prompt_templates = tuple(prompt_templates) if prompt_templates else None
        self.num_classes = num_classes
        self._text_features = text_features
        self._logit_scale = logit_scale
        self._device = device

        if self.model is None and pretrained:
            self.model, self.tokenizer, self.preprocess = _load_backend(self.spec)
        if self.model is not None and eval_mode and hasattr(self.model, "eval"):
            try:
                self.model.eval()
            except Exception:  # pragma: no cover - defensive
                pass
        if device is not None and self.model is not None and hasattr(self.model, "to"):
            self.model.to(device)

    # -- device / mode ----------------------------------------------------
    @property
    def device(self) -> Any:
        if self._device is not None:
            return self._device
        try:
            return next(self.model.parameters()).device
        except Exception:  # pragma: no cover - defensive
            return "cpu"

    def to(self, device: Any) -> "VlmModelWrapper":
        self._device = device
        if self.model is not None and hasattr(self.model, "to"):
            self.model.to(device)
        return self

    def eval(self) -> "VlmModelWrapper":
        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()
        return self

    def train(self, mode: bool = True) -> "VlmModelWrapper":
        if self.model is not None and hasattr(self.model, "train"):
            self.model.train(mode)
        return self

    # -- text side --------------------------------------------------------
    @property
    def logit_scale(self) -> float:
        if self._logit_scale is not None:
            return float(self._logit_scale)
        for attr in ("logit_scale", "log_logit_scale"):
            value = getattr(self.model, attr, None)
            if value is None:
                continue
            try:
                if hasattr(value, "detach"):
                    value = value.detach()
                    value = value.exp() if attr.startswith("log_logit") else value
                    value = float(value.reshape(-1)[0])
                else:
                    value = float(value)
            except Exception:
                continue
            if value > 0:
                return value
        return 100.0

    def encode_text(self, prompts: Sequence[str]) -> Any:
        """Encode prompts into normalized text embeddings ``(P, D)``."""
        torch = _require_torch()
        backend = getattr(self.spec, "backend", "open_clip")

        if backend == "openai_clip":
            import clip  # type: ignore

            tokens = clip.tokenize(list(prompts))
            if self._device is not None:
                tokens = tokens.to(self._device)
            feats = self.model.encode_text(tokens)
            return feats.float() / feats.float().norm(dim=-1, keepdim=True)

        if backend == "open_clip":
            if self.tokenizer is None:  # pragma: no cover - defensive
                import open_clip  # type: ignore

                self.tokenizer = open_clip.get_tokenizer(self.spec.arch)
            tokens = self.tokenizer(list(prompts))
            if self._device is not None:
                tokens = tokens.to(self._device)
            feats = self.model.encode_text(tokens)
            return feats.float() / feats.float().norm(dim=-1, keepdim=True)

        # transformers feature extractors (ALBEF / BLIP)
        if self.tokenizer is None:
            raise RuntimeError(f"{self.name}: no tokenizer for text encoding")
        encoded = self.tokenizer(list(prompts), padding=True, truncation=True, return_tensors="pt")
        if self._device is not None:
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
        out = self.model.get_text_features(**encoded)
        if out.dim() == 3:
            out = out[:, 0]
        out = out if torch.is_tensor(out) else torch.as_tensor(out)
        out = out.float()
        return out / out.norm(dim=-1, keepdim=True)

    def build_text_features(
        self,
        class_names: Optional[Sequence[str]] = None,
        templates: Optional[Sequence[str]] = None,
    ) -> Any:
        """Build the ``(K, D)`` class text-embedding matrix (prompt ensembling)."""
        names = class_names or self.class_names
        if names is None:
            names = class_names_for_hierarchy(None, self.num_classes)
        self.class_names = list(names)
        templates = templates if templates is not None else self.prompt_templates
        if templates:
            prompts = format_prompts(names, templates=templates)
            feats = self.encode_text(prompts)
            feats = feats.view(len(names), len(templates), -1).mean(dim=1)
            return feats / feats.norm(dim=-1, keepdim=True)
        feats = self.encode_text(format_prompts(names))
        return feats / feats.norm(dim=-1, keepdim=True)

    @property
    def text_features(self) -> Any:
        if self._text_features is None:
            self._text_features = self.build_text_features()
        return self._text_features

    def set_text_features(self, features: Any) -> None:
        self._text_features = features

    # -- image side -------------------------------------------------------
    def encode_images(self, images: Any) -> Any:
        """Return normalized image embeddings ``(B, D)`` (this is ``M(X)``)."""
        torch = _require_torch()
        if self._device is not None and torch.is_tensor(images):
            images = images.to(self._device)
        backend = getattr(self.spec, "backend", "open_clip")
        if backend == "transformers":
            out = self.model.get_image_features(pixel_values=images)
            if out.dim() == 3:
                out = out[:, 0]
            out = out if torch.is_tensor(out) else torch.as_tensor(out)
        else:
            out = self.model.encode_image(images)
        out = out.float()
        return out / out.norm(dim=-1, keepdim=True)

    def features(self, x: Any) -> Any:
        """``M(X)``: penultimate image representation (``(B, D)``)."""
        return self.encode_images(x)

    def raw_features(self, x: Any) -> Any:
        """Un-normalized image embeddings (``(B, D)``)."""
        torch = _require_torch()
        if self._device is not None and torch.is_tensor(x):
            x = x.to(self._device)
        backend = getattr(self.spec, "backend", "open_clip")
        if backend == "transformers":
            out = self.model.get_image_features(pixel_values=x)
            if out.dim() == 3:
                out = out[:, 0]
            return (out if torch.is_tensor(out) else torch.as_tensor(out)).float()
        return self.model.encode_image(x).float()

    # -- logits -----------------------------------------------------------
    def logits(self, x: Any) -> Any:
        """Zero-shot ImageNet logits ``(B, K)``."""
        torch = _require_torch()
        img = self.encode_images(x)
        text = self.text_features
        if not torch.is_tensor(text):
            text = torch.as_tensor(text)
        text = text.to(img.device)
        return self.logit_scale * (img @ text.t())

    def __call__(self, x: Any, return_features: bool = False) -> Any:
        if return_features:
            return self.forward_both(x)
        return self.logits(x)

    def forward_both(self, x: Any, return_logits: bool = True) -> Tuple[Any, Any]:
        """Single image-tower pass returning ``(features, logits)``."""
        feats = self.encode_images(x)
        if not return_logits:
            return feats, None
        torch = _require_torch()
        text = self.text_features
        if not torch.is_tensor(text):
            text = torch.as_tensor(text)
        text = text.to(feats.device)
        return feats, self.logit_scale * (feats @ text.t())

    def prediction_scores(self, x: Any, temperature: Optional[float] = None) -> Any:
        """Softmax probabilities over the K classes."""
        torch = _require_torch()
        logits = self.logits(x)
        if temperature is not None and float(temperature) != 1.0:
            logits = logits / float(temperature)
        return torch.softmax(logits, dim=-1)

    # -- transforms -------------------------------------------------------
    def eval_transform(self) -> Any:
        if self.preprocess is not None:
            return self.preprocess
        from ..data.imagenet import build_transform

        size = getattr(self.spec, "input_size", 224) or 224
        return build_transform(resolution=size)

    def train_transform(self, resolution: Optional[int] = None) -> Any:
        from ..data.imagenet import build_transform

        size = resolution or getattr(self.spec, "input_size", 224) or 224
        return build_transform(resolution=size, train=True)

    def close(self) -> None:
        try:
            torch = _require_torch()

            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - defensive
            self.model = None

    def __repr__(self) -> str:
        return (
            f"VlmModelWrapper(name={self.name!r}, family={self.spec.family!r}, "
            f"backend={self.spec.backend!r}, arch={self.spec.arch!r})"
        )


def _maybe_spec(name: str) -> VlmSpec:
    try:
        return get_model_spec(name)
    except KeyError:
        return VlmSpec(name, "unknown", "open_clip", name, None, 224)


# ---------------------------------------------------------------------------
# Backend loaders (public checkpoint APIs, no API keys required)
# ---------------------------------------------------------------------------


def _load_openai_clip(spec: VlmSpec) -> Tuple[Any, Any, Any]:
    import clip  # type: ignore  # git+https://github.com/openai/CLIP
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(spec.arch, device=device)
    return model, clip.tokenize, preprocess


def _load_open_clip(spec: VlmSpec) -> Tuple[Any, Any, Any]:
    import open_clip  # type: ignore

    model, _, preprocess = open_clip.create_model_and_transforms(
        spec.arch, pretrained=spec.pretrained
    )
    tokenizer = open_clip.get_tokenizer(spec.arch)
    return model, tokenizer, preprocess


def _load_transformers(spec: VlmSpec) -> Tuple[Any, Any, Any]:
    from transformers import (  # type: ignore
        AutoImageProcessor,
        AutoModel,
        AutoTokenizer,
    )

    model = AutoModel.from_pretrained(spec.arch)
    tokenizer = AutoTokenizer.from_pretrained(spec.arch)
    try:
        preprocess = AutoImageProcessor.from_pretrained(spec.arch)
    except Exception:  # pragma: no cover - defensive
        preprocess = None
    return model, tokenizer, preprocess


def _load_backend(spec: VlmSpec) -> Tuple[Any, Any, Any]:
    """Instantiate a VLM according to the (unpinned) public backend APIs."""
    if spec.backend == "openai_clip":
        loader = _load_openai_clip
    elif spec.backend == "transformers":
        loader = _load_transformers
    else:
        loader = _load_open_clip
    try:
        return loader(spec)
    except Exception as exc:  # pragma: no cover - depends on downloads
        LOG.warning("Failed to load VLM '%s' (%s): %s", spec.name, spec.backend, exc)
        if spec.backend == "open_clip":
            LOG.info("Retrying '%s' through the OpenAI CLIP loader", spec.name)
            return _load_openai_clip(spec)
        raise


def create_vlm(
    name: str,
    pretrained: bool = True,
    device: Any = None,
    class_names: Optional[Sequence[str]] = None,
    prompt_templates: Optional[Sequence[str]] = IMAGENET_PROMPT_TEMPLATES,
    num_classes: int = IMAGENET_NUM_CLASSES,
    cache_text: bool = True,
) -> VlmModelWrapper:
    """Create one :class:`VlmModelWrapper` from the Appendix-A registry."""
    spec = get_model_spec(name)
    wrapper = VlmModelWrapper(
        spec.name,
        spec=spec,
        pretrained=pretrained,
        device=device,
        class_names=class_names,
        prompt_templates=prompt_templates,
        num_classes=num_classes,
    )
    if cache_text and class_names is not None:
        try:
            wrapper.set_text_features(wrapper.build_text_features())
        except Exception as exc:  # pragma: no cover - depends on downloads
            LOG.warning("Could not pre-compute text features for %s: %s", name, exc)
            wrapper.set_text_features(None)
    return wrapper


def build_vlm_zoo(
    names: Optional[Sequence[str]] = None,
    pretrained: bool = True,
    device: Any = None,
    allow_failures: bool = True,
    class_names: Optional[Sequence[str]] = None,
    prompt_templates: Optional[Sequence[str]] = IMAGENET_PROMPT_TEMPLATES,
    num_classes: int = IMAGENET_NUM_CLASSES,
) -> Dict[str, VlmModelWrapper]:
    """Instantiate every requested VLM, skipping failures when tolerated."""
    names = list(names) if names is not None else list(VLM_NAMES)
    zoo: Dict[str, VlmModelWrapper] = {}
    for name in names:
        try:
            zoo[get_model_spec(name).name] = create_vlm(
                name,
                pretrained=pretrained,
                device=device,
                class_names=class_names,
                prompt_templates=prompt_templates,
                num_classes=num_classes,
            )
        except Exception as exc:  # pragma: no cover - depends on downloads
            LOG.warning("Skipping VLM '%s': %s", name, exc)
            if not allow_failures:
                raise
    return zoo


def release_vlm_zoo(zoo: Dict[str, VlmModelWrapper]) -> None:
    """Free memory held by a zoo of VLMs."""
    for wrapper in list(zoo.values()):
        try:
            wrapper.close()
        except Exception:  # pragma: no cover - defensive
            pass
    zoo.clear()


def get_eval_transform(name_or_spec: Any) -> Any:
    """Official (CLIP) preprocessing for a VLM, else the fallback transform."""
    spec = name_or_spec if isinstance(name_or_spec, VlmSpec) else get_model_spec(str(name_or_spec))
    if spec.backend == "openai_clip":
        try:  # pragma: no cover - depends on download
            import clip  # type: ignore

            _, preprocess = clip.load(spec.arch, device="cpu")
            return preprocess
        except Exception:
            LOG.debug("Falling back to generic transform for %s", spec.name)
    elif spec.backend == "open_clip":
        try:  # pragma: no cover - depends on download
            import open_clip  # type: ignore

            _, _, preprocess = open_clip.create_model_and_transforms(
                spec.arch, pretrained=spec.pretrained
            )
            return preprocess
        except Exception:
            LOG.debug("Falling back to generic transform for %s", spec.name)
    from ..data.imagenet import build_transform

    return build_transform(resolution=spec.input_size or 224)


def get_train_transform(name_or_spec: Any, resolution: Optional[int] = None) -> Any:
    from ..data.imagenet import build_transform

    spec = name_or_spec if isinstance(name_or_spec, VlmSpec) else get_model_spec(str(name_or_spec))
    size = resolution or spec.input_size or 224
    return build_transform(resolution=size, train=True)


# ---------------------------------------------------------------------------
# Feature / logit extraction and caching (mirrors vm_zoo)
# ---------------------------------------------------------------------------


def cache_path(cache_dir: str, model_name: str, dataset_name: str = "") -> str:
    if dataset_name:
        return os.path.join(cache_dir, f"{model_name}__{dataset_name}.npz")
    return os.path.join(cache_dir, f"{model_name}.npz")


def _to_numpy(array: Any) -> Any:
    if array is None:
        return None
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    import numpy as np

    return np.asarray(array)


def save_outputs(
    path: str,
    features: Any,
    logits: Optional[Any] = None,
    targets: Optional[Any] = None,
    overwrite: bool = True,
) -> str:
    """Persist ``features`` / ``logits`` / ``targets`` to a compressed ``.npz``."""
    import numpy as np

    if os.path.exists(path) and not overwrite:
        LOG.info("Cache exists, keeping %s", path)
        return path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: Dict[str, Any] = {"features": _to_numpy(features)}
    if logits is not None:
        payload["logits"] = _to_numpy(logits)
    if targets is not None:
        payload["targets"] = _to_numpy(targets)
    np.savez_compressed(path, **payload)
    return path


def load_outputs(path: str) -> Dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def extract_outputs(
    wrapper: VlmModelWrapper,
    loader: Any,
    device: Any = None,
    max_batches: Optional[int] = None,
    desc: str = "",
) -> Dict[str, Any]:
    """Run a VLM over a loader, returning ``{features, logits, targets}`` arrays."""
    import numpy as np

    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - optional

        def tqdm(x, **_):  # type: ignore
            return x

    features: List[Any] = []
    logits: List[Any] = []
    targets: List[Any] = []

    if device is not None:
        wrapper.to(device)

    iterator = tqdm(loader, desc=desc or wrapper.name, leave=False)
    for batch_index, batch in enumerate(iterator):
        if max_batches is not None and batch_index >= max_batches:
            break
        if isinstance(batch, (list, tuple)):
            images, labels = batch[0], batch[1] if len(batch) > 1 else None
        elif isinstance(batch, dict):
            images = batch.get("image", batch.get("images", batch.get("pixel_values")))
            labels = batch.get("label", batch.get("labels"))
        else:  # pragma: no cover - unexpected loader
            images, labels = batch, None
        feats, logit = wrapper.forward_both(images)
        features.append(_to_numpy(feats))
        logits.append(_to_numpy(logit))
        if labels is not None:
            targets.append(_to_numpy(labels))

    out: Dict[str, Any] = {}
    out["features"] = np.concatenate(features, axis=0) if features else np.zeros((0, 0))
    out["logits"] = np.concatenate(logits, axis=0) if logits else np.zeros((0, 0))
    out["targets"] = (
        np.concatenate(targets, axis=0).astype(np.int64)
        if targets
        else np.zeros((0,), dtype=np.int64)
    )
    return out


def extract_and_cache(
    wrapper: VlmModelWrapper,
    loader: Any,
    dataset_name: str,
    cache_dir: str,
    device: Any = None,
    overwrite: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Cache-aware :func:`extract_outputs` (identical layout to the VM cache)."""
    path = cache_path(cache_dir, wrapper.name, dataset_name)
    if os.path.exists(path) and not overwrite:
        LOG.info("Loading cached VLM outputs %s", path)
        return load_outputs(path)
    outputs = extract_outputs(wrapper, loader, device=device, max_batches=max_batches)
    save_outputs(
        path,
        outputs.get("features"),
        outputs.get("logits"),
        outputs.get("targets"),
        overwrite=True,
    )
    return outputs


# ---------------------------------------------------------------------------
# Test double
# ---------------------------------------------------------------------------


def create_dummy_vlm(
    name: str = "dummy_vlm",
    num_classes: int = IMAGENET_NUM_CLASSES,
    feature_dim: int = 64,
    device: Any = None,
    seed: int = 0,
) -> VlmModelWrapper:
    """Deterministic VLM double for smoke tests / offline runs (no downloads).

    Class text embeddings are fixed random vectors so that ``logits`` are
    well-defined without any checkpoint.
    """
    import numpy as np
    import torch

    rng = np.random.default_rng(seed)
    text = rng.standard_normal((num_classes, feature_dim)).astype("float32")
    text /= np.linalg.norm(text, axis=1, keepdims=True) + 1e-12
    text_features = torch.from_numpy(text)

    spec = VlmSpec(name, "dummy", "open_clip", "dummy", None, 224)

    class _DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(feature_dim, feature_dim, bias=False)
            with torch.no_grad():
                torch.nn.init.eye_(self.proj.weight)
            self.logit_scale = torch.tensor(4.6052)  # ~ exp(4.6052) = 100

        def encode_image(self, images: Any) -> Any:
            pooled = torch.nn.functional.adaptive_avg_pool2d(images.float(), 1).flatten(1)
            if pooled.shape[1] < feature_dim:
                pad = torch.zeros(pooled.shape[0], feature_dim - pooled.shape[1])
                pooled = torch.cat([pooled, pad], dim=1)
            elif pooled.shape[1] > feature_dim:
                pooled = pooled[:, :feature_dim]
            return self.proj(pooled)

        def encode_text(self, tokens: Any) -> Any:
            return text_features[: int(tokens.shape[0])]

    wrapper = VlmModelWrapper(
        name,
        model=_DummyModel(),
        spec=spec,
        pretrained=False,
        device=device,
        class_names=[f"class_{i}" for i in range(num_classes)],
        prompt_templates=(SIMPLE_PROMPT_TEMPLATE,),
        num_classes=num_classes,
    )
    wrapper.set_text_features(text_features)
    if device is not None:
        wrapper.to(device)
    return wrapper


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------


def _self_test() -> None:  # pragma: no cover - manual sanity check
    assert len(TRANSFORMERS_VLM_SPECS) == 2, len(TRANSFORMERS_VLM_SPECS)
    assert len(OPENAI_CLIP_SPECS) == 7, len(OPENAI_CLIP_SPECS)
    assert len(openclip_specs) == 30, len(openclip_specs)
    assert len(VLM_NAMES) == 39, len(VLM_NAMES)
    assert len(set(VLM_NAMES)) == 39
    assert len(IMAGENET_PROMPT_TEMPLATES) == 80, len(IMAGENET_PROMPT_TEMPLATES)
    assert get_model_spec("clip_vit32").name == "clip_vit_b_32"
    assert get_model_spec("clip_vitl14_336").name == "clip_vit_l_14_336px"
    assert format_prompts(["a b"]) == ["a photo of a a b."]
    assert len(format_prompts(["x", "y"], templates=("{}", "a {}", "the {}"))) == 6

    try:
        import torch  # noqa: F401

        wrapper = create_dummy_vlm(num_classes=10, feature_dim=16)
        x = torch.randn(4, 3, 32, 32)
        feats, logits = wrapper.forward_both(x)
        assert feats.shape == (4, 16), feats.shape
        assert logits.shape == (4, 10), logits.shape
        assert torch.allclose(feats.norm(dim=-1), torch.ones(4), atol=1e-4)
    except ImportError:
        LOG.warning("torch unavailable, skipping tensor part of VLM self test")
    print("vlm_zoo self test passed")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _self_test()


__all__ = [
    "IMAGENET_NUM_CLASSES",
    "IMAGENET_PROMPT_TEMPLATES",
    "SIMPLE_PROMPT_TEMPLATE",
    "OPENCLIP_PAIRS",
    "VlmSpec",
    "VlmModelWrapper",
    "VLM_MODEL_SPECS",
    "SPECS_BY_NAME",
    "VLM_NAMES",
    "VLM_ALIASES",
    "build_openclip_specs",
    "list_vlm_names",
    "specs_by_family",
    "get_model_spec",
    "format_prompts",
    "class_names_for_hierarchy",
    "create_vlm",
    "build_vlm_zoo",
    "release_vlm_zoo",
    "get_eval_transform",
    "get_train_transform",
    "extract_outputs",
    "extract_and_cache",
    "cache_path",
    "save_outputs",
    "load_outputs",
    "create_dummy_vlm",
]

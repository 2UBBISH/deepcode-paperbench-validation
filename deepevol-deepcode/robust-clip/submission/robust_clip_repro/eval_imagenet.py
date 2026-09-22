"""Zero-shot ImageNet robustness evaluation for the Robust CLIP reproduction.

This harness evaluates vanilla CLIP and Robust CLIP (the adversarially
fine-tuned vision encoder) on ImageNet classification, reporting

* clean top-1 / top-5 accuracy over **all** loaded samples, and
* robust top-1 / top-5 accuracy over a configurable subset (the paper body
  uses 1000 samples) under an l_inf-bounded threat model.

Paper body facts implemented here (§B.10 "Zero-shot Evaluations"):

* the evaluation protocol follows ``CLIP_benchmark`` / OpenCLIP,
* the first two attacks of AutoAttack are used: APGD with cross-entropy loss
  and APGD with targeted DLR loss, **100 iterations each**,
* l_inf radii ``eps = 2/255`` and ``eps = 4/255``,
* robustness is evaluated on 1000 samples while clean accuracy is reported on
  every sample of the dataset,
* images are evaluated at 224x224 resolution,
* the DLR loss applies only to multi-class datasets.

Addendum facts implemented here:

* APGD comes from https://github.com/fra31/robust-finetuning and keeps its
  upstream defaults (the custom PGD settings are never written into APGD),
* the adversarial l_inf ball is computed around **non-normalized** inputs,
* half-precision attacks store perturbations as ``int16`` and single-precision
  attacks store perturbations as ``int32`` (see :mod:`robust_clip_repro.utils.precision`),
* ImageNet is loaded with HuggingFace ``datasets`` using
  ``load_dataset("imagenet-1k", trust_remote_code=True)``.

Everything the paper/Addendum does not state (PGD step size, PGD iteration
count, batch size, seeds, text-prompt template set, classifier checkpoint
paths, ...) is exposed through configuration, taken from upstream defaults
where one exists, and logged as externally supplied -- never silently invented.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

LOGGER = logging.getLogger("robust_clip_repro.eval_imagenet")

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = str(PACKAGE_DIR / "configs" / "pgd_eval.yaml")
DEFAULT_APGD_CONFIG_PATH = str(PACKAGE_DIR / "configs" / "apgd_eval.yaml")

# ---------------------------------------------------------------------------
# Provenance markers
# ---------------------------------------------------------------------------

UNSPECIFIED = "UNSPECIFIED_BY_ADDENDUM"

#: Values stated in the paper body (§B.10) -- safe to use as defaults.
PAPER_BODY_FACTS: Dict[str, Any] = {
    "protocol": "CLIP_benchmark / OpenCLIP",
    "attacks": ("apgd_ce", "apgd_targeted_dlr"),
    "apgd_iterations": 100,
    "epsilons": (2.0 / 255.0, 4.0 / 255.0),
    "robust_samples": 1000,
    "resolution": 224,
    "norm": "linf",
}

#: Values mandated by the Addendum.
ADDENDUM_FACTS: Dict[str, Any] = {
    "apgd_source": "https://github.com/fra31/robust-finetuning",
    "imagener_loader": 'load_dataset("imagenet-1k", trust_remote_code=True)',
    "int_dtype_half": "int16",
    "int_dtype_single": "int32",
    "ball_space": "raw (non-normalized) pixels",
}

#: Values the paper/Addendum are silent about; supplied externally.
EXTERNAL_DEFAULTS: Dict[str, Any] = {
    "pgd_alpha": UNSPECIFIED,
    "pgd_iterations": UNSPECIFIED,
    "pgd_restarts": UNSPECIFIED,
    "batch_size": 1,
    "seed": 0,
    "num_workers": 0,
    "text_prompt_templates": UNSPECIFIED,
    "class_names": UNSPECIFIED,
    "clip_checkpoint": UNSPECIFIED,
    "robust_clip_checkpoint": UNSPECIFIED,
    "device": UNSPECIFIED,
    "output_dir": "results",
}

PAPER_ATTACK_ITERATIONS = int(PAPER_BODY_FACTS["apgd_iterations"])
PAPER_EPSILONS: Tuple[float, ...] = tuple(PAPER_BODY_FACTS["epsilons"])
PAPER_ROBUST_SAMPLES = int(PAPER_BODY_FACTS["robust_samples"])
PAPER_RESOLUTION = int(PAPER_BODY_FACTS["resolution"])

SUPPORTED_ATTACKS: Tuple[str, ...] = ("apgd_ce", "apgd_targeted_dlr", "pgd")
SUPPORTED_NORMS: Tuple[str, ...] = ("linf", "l2")
SUPPORTED_MODELS: Tuple[str, ...] = ("clip", "robust_clip", "dummy")

IMAGENET_CLASS_INDEX_FILENAME = "imagenet_class_index.json"
IMAGENET_CLASS_INDEX_URL = (
    "https://raw.githubusercontent.com/LAION-AI/CLIP_benchmark/main/"
    "clip_benchmark/datasets/en_imagenet_classes.json"
)
HF_IMAGENET_ID = "imagenet-1k"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def parse_eps(value: Any) -> Optional[float]:
    """Parse an epsilon specification such as ``2/255``, ``"0.00784"`` or ``0.0``.

    Returns ``None`` for ``None``/empty/``"unspecified"`` inputs so callers can
    distinguish "not supplied" from a genuine zero budget.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "unspecified", "unspecified_by_addendum"}:
        return None
    if "/" in text:
        numerator, _, denominator = text.partition("/")
        return float(numerator) / float(denominator)
    return float(text)


def parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "unspecified", "unspecified_by_addendum"}:
        return None
    return int(float(text))


def eps_tag(eps: Optional[float]) -> str:
    """Human readable epsilon tag such as ``eps2`` for 2/255."""
    if eps is None:
        return "epsNone"
    scaled = eps * 255.0
    if abs(scaled - round(scaled)) < 1e-6 and round(scaled) != 0:
        return f"eps{int(round(scaled))}"
    return f"eps{eps:.6f}".replace(".", "p")


def load_config(path: Optional[Union[str, os.PathLike]]) -> Dict[str, Any]:
    """Load a YAML config file (tolerant of a missing file or missing PyYAML)."""
    if not path:
        return {}
    path = str(path)
    if not os.path.exists(path):
        LOGGER.warning("Config file not found: %s (using built-in defaults)", path)
        return {}
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            LOGGER.warning("Config %s did not contain a mapping; ignoring", path)
            return {}
        return payload
    except ImportError:  # pragma: no cover - depends on environment
        LOGGER.warning("PyYAML is unavailable; ignoring config file %s", path)
        return {}
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to parse config %s (%s); ignoring", path, exc)
        return {}


def _section(cfg: Dict[str, Any], *names: str) -> Dict[str, Any]:
    """Return the first present nested section from ``names``."""
    for name in names:
        section = cfg.get(name)
        if isinstance(section, dict):
            return section
    return cfg


def resolve_device(device: Any = None) -> Any:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return device
    if device in (None, "", "auto", UNSPECIFIED):
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device


def _safe_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` forwarding only the keyword arguments it accepts."""
    import inspect

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover
        return fn(*args, **kwargs)
    accepts_var_kw = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_var_kw:
        return fn(*args, **kwargs)
    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        LOGGER.debug("Dropping unsupported kwargs for %s: %s", getattr(fn, "__name__", fn), dropped)
    return fn(*args, **filtered)


# ---------------------------------------------------------------------------
# Zero-shot text prompts (OpenAI / CLIP_benchmark ImageNet template set)
# ---------------------------------------------------------------------------

IMAGENET_TEMPLATES: Tuple[str, ...] = (
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
    "the {} in a video game.",
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
    "a embroidered {}.",
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
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
)


def get_imagenet_templates() -> List[str]:
    """Return the prompt template set used for zero-shot ImageNet classification."""
    return list(IMAGENET_TEMPLATES)


def imagenet_prompts(class_names: Sequence[str], templates: Optional[Sequence[str]] = None) -> List[List[str]]:
    """Build zero-shot prompts: one list of templates per class."""
    templates = list(templates) if templates else get_imagenet_templates()
    return [[template.format(name) for template in templates] for name in class_names]


def load_class_names(path: Optional[Union[str, os.PathLike]] = None) -> Optional[List[str]]:
    """Load an ImageNet class-name list from a JSON file (if available)."""
    if not path:
        return None
    path = str(path)
    if not os.path.exists(path):
        LOGGER.warning("Class-name file %s not found", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to read class names from %s (%s)", path, exc)
        return None
    if isinstance(payload, dict):
        # {"0": ["n01440764", "tench"], ...} style index files
        items: List[Tuple[int, str]] = []
        for key, value in payload.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, (list, tuple)) and value:
                items.append((index, str(value[-1])))
            elif isinstance(value, str):
                items.append((index, value))
        items.sort(key=lambda pair: pair[0])
        return [name for _, name in items] or None
    if isinstance(payload, list):
        return [str(item) for item in payload] or None
    return None


def resolve_class_names(
    cfg: "ImageNetEvalConfig",
    dataset: Any = None,
    *,
    allow_unnamed: bool = True,
) -> List[str]:
    """Resolve ImageNet class names for zero-shot text features.

    Order of preference: explicit config list -> local JSON file -> the class
    names exposed by the HuggingFace dataset -> ``"class {i}"`` placeholders
    (only when ``allow_unnamed``, which yields meaningless text features and is
    therefore only appropriate for smoke tests).
    """
    if cfg.class_names:
        return list(cfg.class_names)

    names = load_class_names(cfg.class_index_file)
    if names:
        return names

    if dataset is not None:
        try:
            from .data.imagenet import get_label_names

            names = get_label_names(dataset)
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("get_label_names failed: %s", exc)
            names = None
        if names:
            return list(names)

    num_classes = cfg.num_classes
    if num_classes is None and dataset is not None:
        try:
            feature = dataset.features.get("label")
            num_classes = getattr(feature, "num_classes", None)
        except Exception:  # pragma: no cover
            num_classes = None
    if num_classes is None:
        num_classes = 1000

    if not allow_unnamed:
        raise RuntimeError(
            "ImageNet class names could not be resolved; supply config.class_names "
            "or config.class_index_file. Zero-shot text features require class names."
        )
    LOGGER.warning(
        "Class names unavailable; falling back to %d placeholder names. Zero-shot "
        "accuracy from this run is NOT meaningful (smoke tests only).",
        num_classes,
    )
    return [f"class {index}" for index in range(int(num_classes))]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AttackSpec:
    """One attacker configuration (APGD-CE, APGD-targeted-DLR, or PGD)."""

    name: str
    norm: str = "linf"
    eps: Optional[float] = PAPER_EPSILONS[0]
    alpha: Optional[float] = None
    iterations: Optional[int] = None
    restarts: Optional[int] = None
    loss: str = "ce"
    targeted: bool = False
    precision: str = "single"
    external: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "norm": self.norm,
            "eps": self.eps,
            "alpha": self.alpha,
            "iterations": self.iterations,
            "restarts": self.restarts,
            "loss": self.loss,
            "targeted": self.targeted,
            "precision": self.precision,
            "external": dict(self.external),
        }


@dataclass
class ImageNetEvalConfig:
    """Configuration for a zero-shot / robust ImageNet evaluation run."""

    # --- protocol (paper body) -------------------------------------------
    dataset_name: str = "ImageNet"
    dataset_id: str = HF_IMAGENET_ID
    split: Optional[str] = None
    resolution: int = PAPER_RESOLUTION
    attach_prompts: bool = False  # ImageNet zero-shot does not need LLaVA prompts

    # --- attacker --------------------------------------------------------
    attacks: Tuple[str, ...] = ("apgd_ce", "apgd_targeted_dlr")
    norm: str = "linf"
    epsilons: Tuple[float, ...] = PAPER_EPSILONS
    alpha: Optional[float] = None  # APGD: upstream 2*eps; PGD: externally supplied
    iterations: Optional[int] = PAPER_ATTACK_ITERATIONS  # paper body: 100 for APGD
    restarts: Optional[int] = 1
    precision: str = "single"
    use_upstream_apgd: bool = True

    # --- data ------------------------------------------------------------
    num_samples: Optional[int] = None  # clean accuracy: all samples by default
    num_robust_samples: Optional[int] = PAPER_ROBUST_SAMPLES
    shuffle: bool = False
    subsample_seed: int = 0
    trust_remote_code: bool = True
    streaming: bool = False
    cache_dir: Optional[str] = None
    local_image_root: Optional[str] = None

    # --- victim ----------------------------------------------------------
    model_name: str = "clip"
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    clip_checkpoint: Optional[str] = None  # Robust CLIP vision-encoder weights
    class_names: Optional[List[str]] = None
    class_index_file: Optional[str] = None
    num_classes: Optional[int] = None
    templates: Optional[List[str]] = None
    text_batch_size: int = 64

    # --- runtime ---------------------------------------------------------
    batch_size: int = 1
    top_k: int = 5
    seed: int = 0
    device: Optional[str] = None
    output_dir: str = "results"
    output_file: Optional[str] = None
    verbose: bool = True
    smoke_test: bool = False

    provenance: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.attacks = tuple(self.attacks) if self.attacks else tuple()
        if isinstance(self.epsilons, (int, float, str)):
            self.epsilons = (parse_eps(self.epsilons),)  # type: ignore[assignment]
        self.epsilons = tuple(
            parsed for parsed in (parse_eps(value) for value in self.epsilons) if parsed is not None
        )
        if not self.epsilons:
            self.epsilons = PAPER_EPSILONS
        self.iterations = parse_int(self.iterations)
        self.restarts = parse_int(self.restarts)
        self.num_samples = parse_int(self.num_samples)
        self.num_robust_samples = parse_int(self.num_robust_samples)
        self.alpha = parse_eps(self.alpha)
        self.norm = str(self.norm).lower()
        self.attacks = tuple(str(name).lower() for name in self.attacks)
        unknown = [name for name in self.attacks if name not in SUPPORTED_ATTACKS]
        if unknown:
            raise ValueError(f"Unsupported attack(s) {unknown}; supported: {SUPPORTED_ATTACKS}")
        if self.norm not in SUPPORTED_NORMS:
            raise ValueError(f"Unsupported norm {self.norm!r}; supported: {SUPPORTED_NORMS}")
        if self.model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model_name {self.model_name!r}; supported: {SUPPORTED_MODELS}")
        if self.class_index_file is None:
            candidate = PACKAGE_DIR / "assets" / IMAGENET_CLASS_INDEX_FILENAME
            self.class_index_file = str(candidate) if candidate.exists() else None
        if not self.provenance:
            self._record_provenance()

    # -- provenance ------------------------------------------------------
    def _record_provenance(self) -> None:
        self.provenance = {
            "paper_body": [
                "protocol=CLIP_benchmark/OpenCLIP",
                f"attacks={'.'.join(PAPER_BODY_FACTS['attacks'])}",
                f"apgd_iterations={PAPER_ATTACK_ITERATIONS}",
                f"epsilons={list(PAPER_EPSILONS)}",
                f"num_robust_samples={PAPER_ROBUST_SAMPLES}",
                f"resolution={PAPER_RESOLUTION}",
                "norm=linf",
            ],
            "addendum": [
                f"apgd_source={ADDENDUM_FACTS['apgd_source']}",
                f"int_dtype_half={ADDENDUM_FACTS['int_dtype_half']}",
                f"int_dtype_single={ADDENDUM_FACTS['int_dtype_single']}",
                f"ball_space={ADDENDUM_FACTS['ball_space']}",
                "imagenet_via_huggingface_trust_remote_code",
            ],
            "unspecified_by_addendum": sorted(EXTERNAL_DEFAULTS.keys()),
        }

    # -- derived ---------------------------------------------------------
    @property
    def resolved_device(self) -> Any:
        return resolve_device(self.device)

    def resolved_templates(self) -> List[str]:
        return list(self.templates) if self.templates else get_imagenet_templates()

    def attack_specs(self) -> List[AttackSpec]:
        """Expand the config into concrete attack specifications."""
        specs: List[AttackSpec] = []
        for name in self.attacks:
            for eps in self.epsilons:
                specs.append(self._spec_for(name, eps))
        return specs

    def _spec_for(self, name: str, eps: float) -> AttackSpec:
        external: Dict[str, Any] = {}
        if name == "pgd":
            # The Addendum fixes PGD's momentum/init/projection space but is
            # silent on eps/alpha/iterations; only eps comes from the paper body.
            iterations = self.iterations if self.iterations is not None else 100
            if self.iterations is None:
                external["iterations"] = f"{EXTERNAL_DEFAULTS['pgd_iterations']} -> {iterations}"
            alpha = self.alpha
            if alpha is None:
                alpha = eps / float(iterations) if iterations else eps / 100.0
                external["alpha"] = (
                    f"{EXTERNAL_DEFAULTS['pgd_alpha']} -> eps/iterations={alpha!r}"
                )
            restarts = self.restarts if self.restarts is not None else 1
            if self.restarts is None:
                external["restarts"] = f"{EXTERNAL_DEFAULTS['pgd_restarts']} -> {restarts}"
            return AttackSpec(
                name="pgd",
                norm=self.norm,
                eps=eps,
                alpha=alpha,
                iterations=iterations,
                restarts=restarts,
                loss="ce",
                targeted=False,
                precision=self.precision,
                external=external,
            )

        # APGD variants keep upstream internals; alpha defaults to upstream 2*eps.
        if name == "apgd_targeted_dlr":
            targeted, loss = True, "targeted_dlr"
        elif name == "apgd_ce":
            targeted, loss = False, "ce"
        else:  # pragma: no cover - guarded by SUPPORTED_ATTACKS
            raise ValueError(f"Unsupported attack spec {name!r}")
        iterations = self.iterations if self.iterations is not None else PAPER_ATTACK_ITERATIONS
        restarts = self.restarts if self.restarts is not None else 1
        return AttackSpec(
            name=name,
            norm=self.norm,
            eps=eps,
            alpha=self.alpha,
            iterations=iterations,
            restarts=restarts,
            loss=loss,
            targeted=targeted,
            precision=self.precision,
            external=external,
        )

    # -- serialization ---------------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "dataset_name": self.dataset_name,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "resolution": self.resolution,
            "attacks": list(self.attacks),
            "norm": self.norm,
            "epsilons": list(self.epsilons),
            "alpha": self.alpha,
            "iterations": self.iterations,
            "restarts": self.restarts,
            "precision": self.precision,
            "use_upstream_apgd": self.use_upstream_apgd,
            "num_samples": self.num_samples,
            "num_robust_samples": self.num_robust_samples,
            "model_name": self.model_name,
            "clip_model": self.clip_model,
            "clip_pretrained": self.clip_pretrained,
            "clip_checkpoint": self.clip_checkpoint,
            "batch_size": self.batch_size,
            "top_k": self.top_k,
            "seed": self.seed,
            "device": str(self.device) if self.device is not None else None,
            "output_dir": self.output_dir,
            "output_file": self.output_file,
            "provenance": self.provenance,
            "external_defaults": EXTERNAL_DEFAULTS,
        }
        return payload

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None, **overrides: Any) -> "ImageNetEvalConfig":
        cfg = dict(cfg or {})
        section = _section(cfg, "imagenet", "eval_imagenet", "evaluation")
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs: Dict[str, Any] = {}
        ignored: List[str] = []
        for key, value in section.items():
            if key in known:
                kwargs[key] = value
            elif key == "attack":
                nested = value if isinstance(value, dict) else {}
                for nested_key, nested_value in nested.items():
                    if nested_key in known:
                        kwargs[nested_key] = nested_value
                    else:
                        ignored.append(f"attack.{nested_key}")
            elif key == "model":
                nested = value if isinstance(value, dict) else {}
                for nested_key, nested_value in nested.items():
                    if nested_key in known:
                        kwargs[nested_key] = nested_value
                    else:
                        ignored.append(f"model.{nested_key}")
            else:
                ignored.append(key)
        if ignored:
            LOGGER.debug("Ignoring unrecognized ImageNet config keys: %s", sorted(set(ignored)))
        kwargs.update({key: value for key, value in overrides.items() if value is not None})
        kwargs.pop("provenance", None)
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Victim classifiers
# ---------------------------------------------------------------------------


class Classifier:
    """Minimal classifier protocol: raw ``[0,1]`` pixels in, logits out."""

    name = "classifier"
    num_classes: Optional[int] = None

    def predict(self, pixels: Any) -> Any:  # pragma: no cover - protocol
        raise NotImplementedError

    def __call__(self, pixels: Any) -> Any:
        return self.predict(pixels)

    def summary(self) -> Dict[str, Any]:
        return {"name": self.name, "num_classes": self.num_classes}


class ZeroShotClassifier(Classifier):
    """Zero-shot classifier built on the OpenCLIP vision/text towers."""

    def __init__(
        self,
        encoder: Any,
        class_names: Sequence[str],
        templates: Optional[Sequence[str]] = None,
        *,
        device: Any = None,
        text_batch_size: int = 64,
        name: str = "clip",
    ) -> None:
        self.encoder = encoder
        self.class_names = list(class_names)
        self.templates = list(templates) if templates else get_imagenet_templates()
        self.text_batch_size = int(text_batch_size)
        self.name = name
        self.num_classes = len(self.class_names)
        self._text_features = None
        self._device = device

    # -- text ------------------------------------------------------------
    @property
    def device(self) -> Any:
        if self._device is not None:
            return self._device
        try:
            return next(self.encoder.parameters()).device
        except Exception:  # pragma: no cover
            return "cpu"

    def build_text_features(self) -> Any:
        """Encode the 80-template prompt set into a normalised text matrix."""
        if self._text_features is not None:
            return self._text_features
        import torch

        model = self.encoder.model
        tokenizer = self.encoder.tokenizer
        if tokenizer is None:
            raise RuntimeError("OpenCLIP tokenizer unavailable; cannot build zero-shot text features")
        features: List[Any] = []
        with torch.no_grad():
            for start in range(0, len(self.class_names), self.text_batch_size):
                batch = self.class_names[start : start + self.text_batch_size]
                flat_prompts = [
                    template.format(name)
                    for name in batch
                    for template in self.templates
                ]
                tokens = tokenizer(flat_prompts)
                if hasattr(tokens, "to"):
                    tokens = tokens.to(self.device)
                text_features = model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                text_features = text_features.reshape(len(batch), len(self.templates), -1)
                text_features = text_features.mean(dim=1)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                features.append(text_features)
        self._text_features = torch.cat(features, dim=0)
        return self._text_features

    # -- images ----------------------------------------------------------
    def predict(self, pixels: Any) -> Any:
        """Classify raw ``(B,3,H,W)`` pixels in ``[0,1]`` (normalized internally)."""
        import torch

        text_features = self.build_text_features()
        device = self.device
        with torch.no_grad():
            images = self.encoder.encode_image(pixels, normalize=True)
            images = images / images.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            logits = self.encoder.model.logit_scale.exp() * (images @ text_features.to(images.dtype).T)
        return logits

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_classes": self.num_classes,
            "templates": len(self.templates),
            "text_batch_size": self.text_batch_size,
            "encoder": getattr(self.encoder, "summary", lambda: {})(),
        }


class DummyClassifier(Classifier):
    """Deterministic, model-free ImageNet classifier for smoke tests.

    Logits are a monotone function of per-image mean brightness plus a small
    positional term, so bounded perturbations measurably change predictions.
    """

    def __init__(self, num_classes: int = 1000, *, name: str = "dummy", device: Any = None) -> None:
        self.num_classes = int(num_classes)
        self.name = name
        self.device = device

    def predict(self, pixels: Any) -> Any:
        import torch

        if not hasattr(pixels, "dim"):
            pixels = torch.as_tensor(pixels, dtype=torch.float32)
        pixels = pixels.float()
        if pixels.dim() == 3:
            pixels = pixels.unsqueeze(0)
        pooled = pixels.mean(dim=(1, 2, 3))
        channels = pixels.mean(dim=(2, 3))  # (B, 3)
        index = torch.arange(self.num_classes, dtype=pixels.dtype, device=pixels.device)
        bias = torch.sin(index) * 0.05
        logits = (
            bias.unsqueeze(0)
            + pooled.unsqueeze(-1) * 4.0
            + channels[:, :1] * 2.0
            + channels[:, 1:2] * 1.0
        )
        return logits

    def summary(self) -> Dict[str, Any]:
        return {"name": self.name, "num_classes": self.num_classes, "dummy": True}


def build_classifier(cfg: ImageNetEvalConfig, class_names: Optional[Sequence[str]] = None) -> Classifier:
    """Instantiate the requested victim classifier (CLIP / Robust CLIP / dummy)."""
    if cfg.model_name == "dummy" or cfg.smoke_test:
        return DummyClassifier(num_classes=cfg.num_classes or 1000)

    try:
        from .models.clip_vision_encoder import build_clip_vision_encoder

        if not class_names:
            raise RuntimeError(
                "Zero-shot evaluation requires class names; supply class_names or a class_index_file"
            )
        encoder = _safe_call(
            build_clip_vision_encoder,
            model_name=cfg.clip_model,
            pretrained=cfg.clip_pretrained,
            pretrained_path=cfg.clip_checkpoint,
            resolution=cfg.resolution,
            device=cfg.resolved_device,
        )
        if cfg.clip_checkpoint:
            try:
                state = _load_state_dict(cfg.clip_checkpoint)
                encoder.load_vision_encoder(state, strict=False)
                LOGGER.info("Loaded Robust CLIP vision weights from %s", cfg.clip_checkpoint)
            except Exception as exc:  # pragma: no cover - depends on checkpoint
                LOGGER.warning("Failed to load checkpoint %s (%s); using base CLIP", cfg.clip_checkpoint, exc)
        return ZeroShotClassifier(
            encoder,
            class_names,
            cfg.resolved_templates(),
            device=cfg.resolved_device,
            text_batch_size=cfg.text_batch_size,
            name="robust_clip" if cfg.clip_checkpoint else "clip",
        )
    except Exception as exc:
        LOGGER.warning(
            "Falling back to DummyClassifier (%s). Install open_clip_torch and provide "
            "class names to run the real zero-shot evaluation.",
            exc,
        )
        return DummyClassifier(num_classes=cfg.num_classes or (len(class_names) if class_names else 1000))


def _load_state_dict(path: Union[str, os.PathLike]) -> Any:
    import torch

    payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict):
        for key in ("vision_encoder", "state_dict", "model", "vision"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                return candidate
        return payload
    return payload  # pragma: no cover


# ---------------------------------------------------------------------------
# Attack construction
# ---------------------------------------------------------------------------


def build_attack(spec: AttackSpec, cfg: ImageNetEvalConfig) -> Any:
    """Build a concrete attacker matching ``PGDLinfAttack``'s public interface."""
    if spec.name == "pgd":
        from .attacks.pgd import PGDLinfAttack

        missing = [key for key in ("eps", "alpha", "iterations") if getattr(spec, key) is None]
        if missing:
            raise ValueError(f"PGD requires {missing}; the Addendum does not specify them")
        return PGDLinfAttack(
            eps=spec.eps,
            alpha=spec.alpha,
            iterations=int(spec.iterations),
            precision=spec.precision,
            targeted=False,
            momentum=0.9,  # Addendum-mandated for the general PGD
            random_start=True,
        )

    from .attacks.apgd import APGDAttack

    return APGDAttack(
        eps=spec.eps,
        alpha=spec.alpha,  # None -> upstream default alpha = 2*eps
        iterations=spec.iterations,
        restarts=spec.restarts,
        norm=spec.norm,
        loss=spec.loss,
        targeted=spec.targeted,
        precision=spec.precision,
        use_upstream=cfg.use_upstream_apgd,
    )


def attack_loss_fn(spec: AttackSpec) -> Callable[[Any], Any]:
    """Loss over model-space inputs (normalization performed by ``predict_fn``)."""
    import torch
    import torch.nn.functional as F

    if spec.loss in ("targeted_dlr", "dlr"):
        from .attacks.apgd import _dlr_loss, _dlr_loss_targeted  # type: ignore

        if spec.targeted:
            return lambda logits, targets: _dlr_loss_targeted(logits, targets)
        return lambda logits, targets: _dlr_loss(logits, targets)
    return lambda logits, targets: F.cross_entropy(logits, targets, reduction="none")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def resolve_samples(cfg: ImageNetEvalConfig, *, num_samples: Optional[int] = None) -> Tuple[List[Any], Any]:
    """Load ImageNet samples (HuggingFace with ``trust_remote_code=True``)."""
    try:
        from .data.imagenet import load_imagenet_dataset

        samples = _safe_call(
            load_imagenet_dataset,
            dataset_name=cfg.dataset_name,
            split=cfg.split,
            num_samples=num_samples,
            dataset_id=cfg.dataset_id,
            cache_dir=cfg.cache_dir,
            streaming=cfg.streaming,
            shuffle=cfg.shuffle,
            seed=cfg.subsample_seed,
            resolution=cfg.resolution,
            resize=cfg.resolution,
            device=cfg.resolved_device,
            trust_remote_code=cfg.trust_remote_code,
            allow_empty=True,
            verbose=cfg.verbose,
        )
        return list(samples or []), None
    except Exception as exc:
        LOGGER.warning("ImageNet loading failed (%s); falling back to synthetic samples", exc)
        from .data.imagenet import synthetic_samples

        n = num_samples if num_samples is not None else 8
        return list(synthetic_samples(n, num_classes=cfg.num_classes or 1000, resolution=cfg.resolution)), None


def subsample(samples: Sequence[Any], num_samples: Optional[int], seed: int = 0) -> List[Any]:
    """Deterministically take ``num_samples`` items (first ``num_samples`` if untrained)."""
    samples = list(samples)
    if num_samples is None or num_samples >= len(samples):
        return samples
    return samples[: int(num_samples)]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class ImageNetEvalResult:
    """Aggregate result of a zero-shot / robust ImageNet evaluation."""

    dataset_name: str
    model_name: str
    num_clean_samples: int
    num_robust_samples: int
    clean_top1: Optional[float]
    clean_top5: Optional[float]
    robust_top1: Dict[str, Optional[float]] = field(default_factory=dict)
    robust_top5: Dict[str, Optional[float]] = field(default_factory=dict)
    perturbation_dtypes: Dict[str, Any] = field(default_factory=dict)
    reports: List[Any] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    external_defaults: Dict[str, Any] = field(default_factory=EXTERNAL_DEFAULTS)
    elapsed_seconds: Optional[float] = None

    def attack_labels(self) -> List[str]:
        return list(self.robust_top1.keys())

    def as_dict(self, include_reports: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "num_clean_samples": self.num_clean_samples,
            "num_robust_samples": self.num_robust_samples,
            "clean_top1": self.clean_top1,
            "clean_top5": self.clean_top5,
            "robust_top1": self.robust_top1,
            "robust_top5": self.robust_top5,
            "perturbation_dtypes": {
                key: (str(value) if value is not None else None)
                for key, value in self.perturbation_dtypes.items()
            },
            "config": self.config,
            "external_defaults": self.external_defaults,
            "elapsed_seconds": self.elapsed_seconds,
        }
        if include_reports:
            payload["reports"] = [
                report.as_dict() if hasattr(report, "as_dict") else report for report in self.reports
            ]
        return payload

    def to_json(self, path: Optional[Union[str, os.PathLike]] = None, indent: int = 2) -> str:
        text = json.dumps(self.as_dict(), indent=indent, default=str)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
        return text

    def summary(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "clean_top1": self.clean_top1,
            "clean_top5": self.clean_top5,
            "robust_top1": self.robust_top1,
            "num_clean_samples": self.num_clean_samples,
            "num_robust_samples": self.num_robust_samples,
        }


class ImageNetEvaluator:
    """Runs the clean pass on all samples and the robust passes on the subset."""

    def __init__(
        self,
        config: ImageNetEvalConfig,
        classifier: Optional[Classifier] = None,
        samples: Optional[Sequence[Any]] = None,
        attack_factory: Optional[Callable[[AttackSpec], Any]] = None,
    ) -> None:
        self.config = config
        self._classifier = classifier
        self._samples = list(samples) if samples is not None else None
        self._attack_factory = attack_factory

    # -- lazy helpers ----------------------------------------------------
    @property
    def classifier(self) -> Classifier:
        if self._classifier is None:
            cfg = self.config
            dataset = None
            class_names = resolve_class_names(cfg, dataset, allow_unnamed=cfg.smoke_test or cfg.model_name == "dummy")
            self._classifier = build_classifier(cfg, class_names)
        return self._classifier

    @property
    def samples(self) -> List[Any]:
        if self._samples is None:
            self._samples, _ = resolve_samples(self.config, num_samples=self.config.num_samples)
        return self._samples

    def predict_fn(self) -> Callable[[Any], Any]:
        classifier = self.classifier
        return lambda pixels: classifier.predict(pixels)

    def make_attack(self, spec: AttackSpec) -> Any:
        if self._attack_factory is not None:
            return self._attack_factory(spec)
        return build_attack(spec, self.config)

    # -- passes ----------------------------------------------------------
    def run_clean(self) -> Dict[str, Any]:
        from .metrics.classification import run_clean_pass

        return run_clean_pass(
            self.predict_fn(),
            self.samples,
            batch_size=self.config.batch_size,
            top_k=self.config.top_k,
            device=self.config.resolved_device,
            verbose=self.config.verbose,
        )

    def run_robust(self, spec: AttackSpec) -> Dict[str, Any]:
        from .metrics.classification import run_robust_pass

        attack = self.make_attack(spec)
        subset = subsample(self.samples, self.config.num_robust_samples, self.config.subsample_seed)
        return run_robust_pass(
            self.predict_fn(),
            attack,
            subset,
            batch_size=self.config.batch_size,
            top_k=self.config.top_k,
            device=self.config.resolved_device,
            verbose=self.config.verbose,
        )

    # -- main ------------------------------------------------------------
    def run(self, samples: Optional[Sequence[Any]] = None) -> ImageNetEvalResult:
        from .metrics.classification import ClassificationReport

        cfg = self.config
        started = time.time()
        if samples is not None:
            self._samples = list(samples)

        LOGGER.info(
            "ImageNet eval: dataset=%s model=%s norm=%s eps=%s attacks=%s precision=%s",
            cfg.dataset_id,
            cfg.model_name,
            cfg.norm,
            list(cfg.epsilons),
            list(cfg.attacks),
            cfg.precision,
        )
        LOGGER.info("Values the paper/Addendum do not state are supplied externally: %s", EXTERNAL_DEFAULTS)

        clean = self.run_clean()
        clean_top1 = clean.get("top1")
        clean_top5 = clean.get("top5")
        LOGGER.info(
            "Clean top-1 = %.4f, top-5 = %.4f over %d samples",
            clean_top1 if clean_top1 is not None else float("nan"),
            clean_top5 if clean_top5 is not None else float("nan"),
            clean.get("num_samples", len(self.samples)),
        )

        robust_subset = subsample(self.samples, cfg.num_robust_samples, cfg.subsample_seed)
        result = ImageNetEvalResult(
            dataset_name=cfg.dataset_id,
            model_name=self.classifier.name,
            num_clean_samples=int(clean.get("num_samples", len(self.samples))),
            num_robust_samples=len(robust_subset),
            clean_top1=clean_top1,
            clean_top5=clean_top5,
            config=cfg.as_dict(),
            external_defaults=dict(EXTERNAL_DEFAULTS),
        )

        for spec in cfg.attack_specs():
            label = f"{spec.name}@{spec.norm}|{eps_tag(spec.eps)}"
            LOGGER.info(
                "Running %s: eps=%s alpha=%s iterations=%s restarts=%s precision=%s",
                label,
                spec.eps,
                spec.alpha,
                spec.iterations,
                spec.restarts,
                spec.precision,
            )
            if spec.external:
                LOGGER.info("  externally supplied (paper silent): %s", spec.external)
            try:
                robust = self.run_robust(spec)
            except Exception as exc:
                LOGGER.error("Attack %s failed: %s", label, exc)
                result.robust_top1[label] = None
                result.robust_top5[label] = None
                result.perturbation_dtypes[label] = None
                continue

            robust_top1 = robust.get("top1")
            robust_top5 = robust.get("top5")
            result.robust_top1[label] = robust_top1
            result.robust_top5[label] = robust_top5
            result.perturbation_dtypes[label] = robust.get("perturbation_dtype")
            LOGGER.info(
                "  robust top-1 = %.4f (drop %.4f), perturbation dtype %s",
                robust_top1 if robust_top1 is not None else float("nan"),
                (clean_top1 - robust_top1) if (clean_top1 is not None and robust_top1 is not None) else float("nan"),
                robust.get("perturbation_dtype"),
            )

            result.reports.append(
                ClassificationReport(
                    dataset_name=cfg.dataset_id,
                    model_name=self.classifier.name,
                    attack_name=spec.name,
                    norm=spec.norm,
                    eps=spec.eps,
                    alpha=spec.alpha,
                    iterations=spec.iterations,
                    precision=spec.precision,
                    top_k=cfg.top_k,
                    num_samples=len(robust_subset),
                    clean_top1=robust.get("clean_top1", clean_top1),
                    clean_top5=robust.get("clean_top5", clean_top5),
                    robust_top1=robust_top1,
                    robust_top5=robust_top5,
                    clean_correct=robust.get("clean_correct", []),
                    robust_correct=robust.get("robust_correct", []),
                    perturbation_dtype=robust.get("perturbation_dtype"),
                    extra={
                        "attack_spec": spec.as_dict(),
                        "clean_top1_full": clean_top1,
                        "clean_top5_full": clean_top5,
                        "num_clean_samples_full": result.num_clean_samples,
                        "external_supplied": spec.external,
                    },
                )
            )

        result.elapsed_seconds = time.time() - started
        return result


def comparison_table(result: ImageNetEvalResult) -> List[Dict[str, Any]]:
    """Rows of the clean-vs-robust table (one row per attack/eps)."""
    rows: List[Dict[str, Any]] = []
    for report in result.reports:
        if hasattr(report, "as_dict"):
            rows.append(report.as_dict())
        else:  # pragma: no cover
            rows.append(dict(report))
    if not rows:
        rows.append(
            {
                "dataset_name": result.dataset_name,
                "model_name": result.model_name,
                "attack_name": "none",
                "norm": None,
                "eps": None,
                "clean_top1": result.clean_top1,
                "robust_top1": None,
            }
        )
    return rows


def format_table(rows: Sequence[Dict[str, Any]], percent: bool = True) -> str:
    from .metrics.classification import format_table as _format_table

    return _format_table(rows, percent=percent)


# ---------------------------------------------------------------------------
# Smoke test (model free)
# ---------------------------------------------------------------------------


def smoke_attack_fn(spec: AttackSpec) -> Callable[..., Any]:
    """Model-free bounded attacker honoring the int16/int32 precision policy."""
    import torch

    from .utils.precision import encode_perturbation, int_dtype_for_precision

    class _SmokeAttack:
        name = "smoke_apgd" if spec.name.startswith("apgd") else "smoke_pgd"
        norm = spec.norm
        eps = spec.eps
        alpha = spec.alpha
        iterations = spec.iterations
        precision = spec.precision
        last_perturbation = None

        def attack_untargeted(self, x, loss_fn=None, *, labels=None, return_delta=True, generator=None):
            eps = float(self.eps or 0.0)
            generator = generator or torch.Generator().manual_seed(0)
            delta = (torch.rand(x.shape, generator=generator, dtype=x.dtype) * 2.0 - 1.0) * eps
            codes = encode_perturbation(delta, self.precision)
            self.last_perturbation = codes
            return codes if return_delta else self.adversarial_examples(x, codes)

        def adversarial_examples(self, x, delta):
            from .utils.precision import decode_perturbation

            decoded = decode_perturbation(delta, x.dtype) if delta.dtype.is_floating_point is False else delta
            return (x + decoded).clamp(0.0, 1.0)

        def initial_perturbation(self, x, *, generator=None):
            generator = generator or torch.Generator().manual_seed(0)
            delta = (torch.rand(x.shape, generator=generator, dtype=x.dtype) * 2.0 - 1.0) * float(self.eps or 0.0)
            return encode_perturbation(delta, self.precision)

        def summary(self):
            return {"name": self.name, "eps": self.eps, "precision": self.precision}

    attack = _SmokeAttack()
    assert int_dtype_for_precision(spec.precision).__str__() in ("torch.int16", "torch.int32")
    return attack


def run_smoke_test(verbose: bool = True) -> Dict[str, Any]:
    """End-to-end, model-free check of the ImageNet evaluation harness."""
    import torch

    from .utils.precision import int_dtype_for_precision

    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")

    cfg = ImageNetEvalConfig(
        model_name="dummy",
        attacks=("pgd",),
        epsilons=(2.0 / 255.0,),
        iterations=5,
        num_samples=4,
        num_robust_samples=4,
        num_classes=16,
        precision="half",
        smoke_test=True,
        verbose=verbose,
    )
    from .data.imagenet import synthetic_samples

    samples = synthetic_samples(4, num_classes=16, resolution=32)
    # Keep images small but valid for the dummy classifier.
    evaluator = ImageNetEvaluator(
        cfg,
        classifier=DummyClassifier(num_classes=16),
        samples=samples,
        attack_factory=smoke_attack_fn,
    )
    result = evaluator.run()
    assert result.num_clean_samples == 4, result.num_clean_samples
    assert result.clean_top1 is not None
    label = next(iter(result.robust_top1))
    assert result.perturbation_dtypes[label] is not None
    assert str(result.perturbation_dtypes[label]) == "torch.int16", result.perturbation_dtypes
    assert int_dtype_for_precision("half") is torch.int16

    # Precision policy sanity: half -> int16, single -> int32.
    assert str(int_dtype_for_precision("half")) == "torch.int16"
    assert str(int_dtype_for_precision("single")) == "torch.int32"

    # Non-normalized projection space: labels are always present in raw pixels.
    from .metrics.classification import extract_pixels, extract_label

    assert extract_pixels(samples[0]) is not None
    assert extract_label(samples[0]) is not None

    payload = result.as_dict()
    if verbose:
        LOGGER.info("ImageNet smoke test summary: %s", json.dumps(result.summary(), default=str, indent=2))
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robust_clip_repro.eval_imagenet",
        description=(
            "Zero-shot / robust ImageNet evaluation for Robust CLIP. "
            "Attack budgets must be supplied: eps is 2/255 and 4/255 in the paper body; "
            "PGD alpha/iterations are not specified by the paper and are logged as external."
        ),
    )
    parser.add_argument("--config", default=None, help="YAML config path (e.g. configs/pgd_eval.yaml)")
    parser.add_argument("--model-name", default=None, choices=list(SUPPORTED_MODELS))
    parser.add_argument(
        "--attacks",
        default=None,
        help="Comma-separated attackers: apgd_ce, apgd_targeted_dlr, pgd (paper body uses the first two)",
    )
    parser.add_argument("--norm", default=None, choices=list(SUPPORTED_NORMS))
    parser.add_argument("--eps", default=None, help="Epsilon spec, e.g. '2/255' or '4/255' (comma separated for several)")
    parser.add_argument("--alpha", default=None, help="Step size (Addendum silent for PGD; APGD default is 2*eps)")
    parser.add_argument("--iterations", default=None, type=int, help="Attack iterations (paper body: 100 for APGD)")
    parser.add_argument("--restarts", default=None, type=int)
    parser.add_argument("--precision", default=None, help="half -> int16 perturbations, single -> int32")
    parser.add_argument("--num-samples", default=None, type=int, help="Clean-pass sample cap (default: all)")
    parser.add_argument("--num-robust-samples", default=None, type=int, help="Robust-pass sample cap (paper body: 1000)")
    parser.add_argument("--dataset-id", default=None, help="HuggingFace dataset id (default: imagenet-1k)")
    parser.add_argument("--split", default=None)
    parser.add_argument("--resolution", default=None, type=int)
    parser.add_argument("--clip-checkpoint", default=None, help="Robust CLIP vision-encoder checkpoint")
    parser.add_argument("--class-index-file", default=None, help="JSON file with ImageNet class names")
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--no-upstream-apgd", action="store_true", help="Use the vendored APGD fallback")
    parser.add_argument("--smoke-test", action="store_true", help="Model-free orchestration check")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    verbose = not args.quiet
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.smoke_test:
        payload = run_smoke_test(verbose=verbose)
        print(json.dumps(payload, indent=2, default=str))
        return 0

    cfg_dict = load_config(args.config or DEFAULT_CONFIG_PATH)
    overrides: Dict[str, Any] = {}
    if args.model_name:
        overrides["model_name"] = args.model_name
    if args.attacks:
        overrides["attacks"] = tuple(part.strip() for part in args.attacks.split(",") if part.strip())
    if args.norm:
        overrides["norm"] = args.norm
    if args.eps:
        eps_values = [parse_eps(part) for part in str(args.eps).split(",") if str(part).strip()]
        overrides["epsilons"] = tuple(value for value in eps_values if value is not None)
    if args.alpha:
        overrides["alpha"] = args.alpha
    if args.iterations is not None:
        overrides["iterations"] = args.iterations
    if args.restarts is not None:
        overrides["restarts"] = args.restarts
    if args.precision:
        overrides["precision"] = args.precision
    if args.num_samples is not None:
        overrides["num_samples"] = args.num_samples
    if args.num_robust_samples is not None:
        overrides["num_robust_samples"] = args.num_robust_samples
    if args.dataset_id:
        overrides["dataset_id"] = args.dataset_id
    if args.split:
        overrides["split"] = args.split
    if args.resolution is not None:
        overrides["resolution"] = args.resolution
    if args.clip_checkpoint:
        overrides["clip_checkpoint"] = args.clip_checkpoint
    if args.class_index_file:
        overrides["class_index_file"] = args.class_index_file
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device:
        overrides["device"] = args.device
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.output_file:
        overrides["output_file"] = args.output_file
    if args.no_upstream_apgd:
        overrides["use_upstream_apgd"] = False

    cfg = ImageNetEvalConfig.from_dict(cfg_dict, **overrides)
    cfg.verbose = verbose

    if not cfg.epsilons:
        LOGGER.error(
            "No epsilon supplied. The paper body evaluates eps=2/255 and eps=4/255; "
            "pass --eps 2/255,4/255 (the reproduction must not invent budgets silently)."
        )
        return 2

    LOGGER.info("Paper-body facts: %s", PAPER_BODY_FACTS)
    LOGGER.info("Addendum facts: %s", ADDENDUM_FACTS)
    LOGGER.info("Externally supplied values: %s", EXTERNAL_DEFAULTS)

    evaluator = ImageNetEvaluator(cfg)
    result = evaluator.run()

    output_file = cfg.output_file
    if not output_file:
        tag = ",".join(eps_tag(eps) for eps in cfg.epsilons)
        output_file = str(Path(cfg.output_dir) / f"imagenet_{result.model_name}_{cfg.norm}_{tag}.json")
    result.to_json(output_file)

    rows = comparison_table(result)
    print(format_table(rows))
    print(json.dumps(result.summary(), indent=2, default=str))
    LOGGER.info("Wrote results to %s", output_file)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

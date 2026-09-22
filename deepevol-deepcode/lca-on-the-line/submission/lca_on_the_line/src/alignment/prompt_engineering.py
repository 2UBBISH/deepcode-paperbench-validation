"""Taxonomy-aware prompt engineering for zero-shot VLMs (Section 4.3.3, Table 14).

The paper integrates the class ontology directly into the text prompt of a
zero-shot vision-language model and shows that informing the model of *both* the
correct taxonomy lineage *and* the hierarchical "is-a" relation improves
generalization::

    Baseline:        "<class>"
    Stack Parent:    "<class>, <parent>, <grandparent>"
    Taxonomy Parent: "<class>, which is a type of <parent>, which is a type of <grandparent>"
    Shuffle Parent:  "<class>, which is a type of <parent'>, which is a type of <grandparent'>"

where ``parent'``/``grandparent'`` are *incorrect* ancestors sampled at random
from the tree (ablation).  Two ablations are reported in Section 4.3.3:

1. **Stack Parent** – the correct taxonomy path but without telling the model the
   relationship between the names (no "which is a type of").
2. **Shuffle Parent** – the correct "is-a" phrasing but with a randomly sampled
   (wrong) taxonomy relationship.

Results are reported with CLIP-ViT32 as Top-1 accuracy and test-time
cross-entropy (Table 14); the taxonomy prompt improves both on all six datasets
(ImageNet, ImageNet-v2, ImageNet-S, ImageNet-R, ImageNet-A, ObjectNet).

This module provides:

* deterministic prompt construction from a :class:`WordNetHierarchy`
  (torch-free, testable offline),
* text-encoder helpers for OpenAI CLIP / OpenCLIP (lazy imports),
* a zero-shot evaluation routine producing Top-1 / Top-5 / test-CE,
* a small reference table + validation gates for Table 14.
"""

from __future__ import annotations

import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

LOG = logging.getLogger(__name__)

__all__ = [
    "PromptTemplate",
    "PromptConfig",
    "PROMPT_TEMPLATES",
    "DEFAULT_TEMPLATES",
    "TABLE14_REFERENCE",
    "PROMPT_DATASETS",
    "DISPLAY_NAMES",
    "sanitize_class_name",
    "node_display_name",
    "class_ancestor_names",
    "format_prompt",
    "build_prompt",
    "build_prompt_texts",
    "build_all_prompt_texts",
    "build_shuffled_ancestors",
    "PromptEncoder",
    "OpenAICLIPEncoder",
    "OpenCLIPEncoder",
    "CallableTextEncoder",
    "build_encoder",
    "encode_text_prompts",
    "encode_image_loader",
    "zero_shot_logits",
    "zero_shot_metrics",
    "evaluate_prompts",
    "run_prompt_evaluation",
    "check_against_table14",
    "format_table14",
    "_self_test",
]


# ---------------------------------------------------------------------------
# constant labels
# ---------------------------------------------------------------------------

PROMPT_DATASETS: Tuple[str, ...] = ("imagenet", "v2", "s", "r", "a", "objectnet")

DISPLAY_NAMES: Dict[str, str] = {
    "imagenet": "ImageNet",
    "id": "ImageNet",
    "v2": "ImgN-v2",
    "imagenet-v2": "ImgN-v2",
    "s": "ImgN-S",
    "sketch": "ImgN-S",
    "imagenet-s": "ImgN-S",
    "r": "ImgN-R",
    "imagenet-r": "ImgN-R",
    "a": "ImgN-A",
    "imagenet-a": "ImgN-A",
    "objectnet": "ObjNet",
    "objnet": "ObjNet",
}

#: Table 14 reference values (CLIP-ViT32).  Top-1 accuracy for the four prompt
#: variants on the six datasets, plus the qualitative expectation that the
#: taxonomy prompt improves both accuracy and test cross-entropy everywhere.
#: Numbers marked ``None`` are unknown in the paper excerpt and are not gated.
TABLE14_REFERENCE: Dict[str, Dict[str, Optional[float]]] = {
    "imagenet": {"baseline": 0.589, "stack_parent": None, "taxonomy_parent": 0.626, "shuffle_parent": None},
    "v2": {"baseline": None, "stack_parent": None, "taxonomy_parent": None, "shuffle_parent": None},
    "s": {"baseline": None, "stack_parent": None, "taxonomy_parent": None, "shuffle_parent": None},
    "r": {"baseline": None, "stack_parent": None, "taxonomy_parent": None, "shuffle_parent": None},
    "a": {"baseline": None, "stack_parent": None, "taxonomy_parent": None, "shuffle_parent": None},
    "objectnet": {"baseline": None, "stack_parent": None, "taxonomy_parent": None, "shuffle_parent": None},
}

#: relative ordering expected from Section 4.3.3 (higher is better)
EXPECTED_ORDER = ("baseline", "stack_parent", "shuffle_parent", "taxonomy_parent")

DEFAULT_TEMPLATES: Tuple[str, ...] = (
    "baseline",
    "stack_parent",
    "taxonomy_parent",
    "shuffle_parent",
)

DEFAULT_MAX_ANCESTORS = 2
DEFAULT_LOGIT_SCALE = 100.0
DEFAULT_SHUFFLE_SEED = 0


# ---------------------------------------------------------------------------
# prompt templates
# ---------------------------------------------------------------------------


@dataclass
class PromptTemplate:
    """One prompt template.

    Attributes
    ----------
    name:
        Registry key (``baseline``, ``stack_parent``, ...).
    template:
        Python format string referencing ``{class_name}``, ``{parent}`` and
        ``{grandparent}``.
    description:
        Human readable summary.
    uses_ancestors:
        Whether the template needs ontology ancestors.
    wrong_ancestors:
        Whether the ancestors are *incorrect* (Shuffle Parent ablation).
    isa_relation:
        Whether the template states the "is a type of" relationship.
    """

    name: str
    template: str
    description: str = ""
    uses_ancestors: bool = False
    wrong_ancestors: bool = False
    isa_relation: bool = False


PROMPT_TEMPLATES: Dict[str, PromptTemplate] = {
    "baseline": PromptTemplate(
        name="baseline",
        template="{class_name}",
        description="Zero-shot baseline prompt: just the class name (Section 4.3.3).",
        uses_ancestors=False,
        wrong_ancestors=False,
        isa_relation=False,
    ),
    "stack_parent": PromptTemplate(
        name="stack_parent",
        template="{class_name}, {parent}, {grandparent}",
        description=(
            "Stack Parent ablation: correct taxonomy path without informing the model "
            "of the class-name relationships."
        ),
        uses_ancestors=True,
        wrong_ancestors=False,
        isa_relation=False,
    ),
    "taxonomy_parent": PromptTemplate(
        name="taxonomy_parent",
        template=(
            "{class_name}, which is a type of {parent}, "
            "which is a type of {grandparent}"
        ),
        description=(
            "Taxonomy Parent: the proposed prompt informing the model of both the correct "
            "taxonomy and the hierarchical 'is-a' relationship."
        ),
        uses_ancestors=True,
        wrong_ancestors=False,
        isa_relation=True,
    ),
    "shuffle_parent": PromptTemplate(
        name="shuffle_parent",
        template=(
            "{class_name}, which is a type of {parent}, "
            "which is a type of {grandparent}"
        ),
        description=(
            "Shuffle Parent ablation: correct 'is-a' phrasing but with an incorrect taxonomy "
            "relationship randomly sampled from the tree."
        ),
        uses_ancestors=True,
        wrong_ancestors=True,
        isa_relation=True,
    ),
    # convenience extra used by some reproductions: average of baseline + taxonomy
    "ensemble_taxonomy": PromptTemplate(
        name="ensemble_taxonomy",
        template="{class_name}",
        description="Prompt ensembling over baseline and taxonomy prompts (not in Table 14).",
        uses_ancestors=True,
        wrong_ancestors=False,
        isa_relation=False,
    ),
}


@dataclass
class PromptConfig:
    """Configuration bundle for the prompt-engineering experiment."""

    templates: Sequence[str] = DEFAULT_TEMPLATES
    max_ancestors: int = DEFAULT_MAX_ANCESTORS
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED
    logit_scale: Optional[float] = DEFAULT_LOGIT_SCALE
    batch_size: int = 64
    num_workers: int = 4
    device: Optional[str] = None
    max_batches: Optional[int] = None
    normalize_features: bool = True
    dataset_name: str = "imagenet"
    class_names: Optional[Sequence[str]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "PromptConfig":
        if not payload:
            return cls()
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in dict(payload).items() if k in known}
        if "templates" in kwargs and isinstance(kwargs["templates"], str):
            kwargs["templates"] = [t.strip() for t in kwargs["templates"].split(",") if t.strip()]
        else:
            kwargs.setdefault("templates", list(DEFAULT_TEMPLATES))
        kwargs["templates"] = list(kwargs.get("templates") or DEFAULT_TEMPLATES)
        # tolerate unknown placement of per-experiment knobs
        kwargs.setdefault("extra", {k: v for k, v in dict(payload).items() if k not in known})
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["templates"] = list(self.templates)
        return out


# ---------------------------------------------------------------------------
# hierarchy / class-name helpers
# ---------------------------------------------------------------------------


def sanitize_class_name(name: Any) -> str:
    """Convert an ImageNet/WNID class label into a bare, prompt-friendly noun.

    ``"tench, Tinca tinca"`` -> ``"tench"``; underscores become spaces.
    """
    if name is None:
        return ""
    text = str(name)
    if "," in text:
        text = text.split(",")[0]
    text = text.replace("_", " ").strip().rstrip(".")
    return text


def _nltk_synset_name(node: str) -> Optional[str]:
    """Best-effort WordNet lemma for a synset id (``n02121808`` -> ``dog``)."""
    try:  # pragma: no cover - depends on local WordNet corpus
        from nltk.corpus import wordnet as wn  # type: ignore

        syn = wn.synset(node)
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name:
                return sanitize_class_name(name)
    except Exception:  # pragma: no cover
        return None
    return None


def node_display_name(node: Any, hierarchy: Any = None) -> str:
    """Human readable name for a hierarchy node.

    Falls back to a numeric synset id or a cleaned-up node-id string when the
    WordNet corpus / class metadata is unavailable (offline runs).
    """
    if node is None:
        return ""
    node_id = str(getattr(node, "id", node))
    if node_id.startswith("<") or node_id.lower().startswith("root"):
        return "entity"

    # a class node may carry an explicit name
    name = getattr(node, "name", None)
    if name:
        return sanitize_class_name(name)

    if node_id.isdigit():
        idx = int(node_id)
        try:
            if hierarchy is not None and hasattr(hierarchy, "class_name"):
                return sanitize_class_name(hierarchy.class_name(idx))
        except Exception:
            pass
        return f"class {idx}"

    lowered = node_id.lower()
    if lowered.startswith("class_") or lowered.startswith("class-"):
        tail = node_id.split("_")[-1].split("-")[-1]
        if tail.isdigit():
            return node_display_name(tail, hierarchy)

    if node_id.startswith("n") and len(node_id) == 9 and node_id[1:].isdigit():
        named = _nltk_synset_name(node_id)
        if named:
            return named
        try:
            if hierarchy is not None and hasattr(hierarchy, "leaf_for_class"):
                # reverse map synset -> class index, if the hierarchy supports it
                mapping = getattr(hierarchy, "class_to_leaf", None) or {}
                for cls_idx, leaf in mapping.items():
                    if str(leaf) == node_id:
                        return sanitize_class_name(getattr(hierarchy, "class_name", lambda i: f"class {i}")(cls_idx))
        except Exception:
            pass
        return node_id

    return sanitize_class_name(node_id.replace("-", " "))


def class_leaf(hierarchy: Any, class_index: int) -> Any:
    """Return the hierarchy leaf node for a class index (duck-typed)."""
    for attr in ("leaf_for_class", "class_leaf"):
        fn = getattr(hierarchy, attr, None)
        if callable(fn):
            try:
                return fn(int(class_index))
            except Exception:
                pass
        elif fn is not None:
            try:
                return fn[int(class_index)]
            except Exception:
                pass
    mapping = getattr(hierarchy, "class_to_leaf", None)
    if mapping is not None:
        try:
            return mapping[int(class_index)]
        except Exception:
            pass
    return class_index


def _ancestors_of_node(hierarchy: Any, node: Any, max_ancestors: int) -> List[Any]:
    """Collect up to ``max_ancestors`` ancestors of ``node`` (parent first)."""
    out: List[Any] = []
    if node is None or max_ancestors <= 0:
        return out
    parents = getattr(hierarchy, "parents", None)
    current = node
    seen = {str(current)}
    for _ in range(max_ancestors):
        parent = None
        if isinstance(parents, Mapping):
            parent = parents.get(str(current), parents.get(current))
        if parent is None:
            parent_fn = getattr(hierarchy, "parent", None)
            if callable(parent_fn):
                try:
                    parent = parent_fn(current)
                except Exception:
                    parent = None
        if parent is None:
            break
        key = str(parent)
        if key in seen:
            break
        seen.add(key)
        out.append(parent)
        current = parent
    return out


def class_ancestor_names(
    hierarchy: Any,
    class_index: int,
    max_ancestors: int = DEFAULT_MAX_ANCESTORS,
) -> List[str]:
    """Names of the (up to) ``max_ancestors`` WordNet ancestors of a class."""
    if hierarchy is None:
        return []
    leaf = class_leaf(hierarchy, class_index)
    nodes = _ancestors_of_node(hierarchy, leaf, max_ancestors)
    names: List[str] = []
    for node in nodes:
        name = node_display_name(node, hierarchy)
        if name and name not in names:
            names.append(name)
    return names


def build_shuffled_ancestors(
    hierarchy: Any,
    class_index: int,
    num_classes: Optional[int] = None,
    max_ancestors: int = DEFAULT_MAX_ANCESTORS,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Sample *incorrect* ancestors for the Shuffle Parent ablation.

    A random class id different from ``class_index`` is drawn and its true
    ancestors are reused, i.e. the "is-a" phrasing is kept but the taxonomy
    relationship is wrong (Section 4.3.3).
    """
    rng = rng or random.Random(DEFAULT_SHUFFLE_SEED)
    if num_classes is None:
        num_classes = int(getattr(hierarchy, "num_classes", 0) or 1000)
    if num_classes <= 1:
        return []
    wrong = int(class_index)
    for _ in range(16):
        cand = rng.randrange(num_classes)
        if cand != int(class_index):
            wrong = cand
            break
    else:
        wrong = (int(class_index) + 1) % num_classes
    names = class_ancestor_names(hierarchy, wrong, max_ancestors=max_ancestors)
    # guarantee a full-length lineage even if the wrong class is shallow
    filler = class_ancestor_names(hierarchy, class_index, max_ancestors=max_ancestors)
    out: List[str] = []
    for i in range(max_ancestors):
        if i < len(names) and names[i]:
            out.append(names[i])
        elif i < len(filler) and filler[i]:
            out.append(filler[i])
    return out


def format_prompt(
    template: Union[str, PromptTemplate],
    class_name: str,
    ancestors: Optional[Sequence[str]] = None,
) -> str:
    """Render one prompt, degrading gracefully when ancestors are missing."""
    tpl = PROMPT_TEMPLATES[template].template if isinstance(template, str) and template in PROMPT_TEMPLATES else template
    if isinstance(tpl, PromptTemplate):
        tpl = tpl.template
    ancestors = list(ancestors or [])
    parent = ancestors[0] if len(ancestors) > 0 else ""
    grandparent = ancestors[1] if len(ancestors) > 1 else ""

    class_name = sanitize_class_name(class_name)
    if not tpl:
        return class_name

    # drop dangling clauses when an ancestor is unavailable
    if "{grandparent}" in tpl and not grandparent:
        head, _, tail = tpl.partition("{grandparent}")
        tpl = tail if not parent else head.rstrip().rstrip(",").rstrip()
        tpl = head.rstrip().rstrip(",").rstrip() if parent else tpl
        if not parent:
            return class_name
    if "{parent}" in tpl and not parent:
        head, _, _tail = tpl.partition("{parent}")
        tpl = head.rstrip().rstrip(",").rstrip()
        if not tpl or tpl == class_name:
            return class_name

    try:
        text = tpl.format(class_name=class_name, parent=parent, grandparent=grandparent)
    except (KeyError, IndexError):
        text = class_name
    return " ".join(text.split()).strip()


def build_prompt(
    class_index: int,
    template: str = "taxonomy_parent",
    hierarchy: Any = None,
    class_names: Optional[Sequence[str]] = None,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    max_ancestors: int = DEFAULT_MAX_ANCESTORS,
    rng: Optional[random.Random] = None,
) -> str:
    """Build a single prompt string for ``class_index``."""
    if class_names is not None and 0 <= int(class_index) < len(class_names):
        class_name = sanitize_class_name(class_names[int(class_index)])
    elif hierarchy is not None and hasattr(hierarchy, "class_name"):
        class_name = sanitize_class_name(hierarchy.class_name(int(class_index)))
    else:
        class_name = f"class {int(class_index)}"

    spec = PROMPT_TEMPLATES.get(template)
    if spec is None:
        # unknown template name -> treat the string itself as a format string
        spec = PromptTemplate(name=str(template), template=str(template), uses_ancestors="{" in str(template))

    if not spec.uses_ancestors:
        return format_prompt(spec, class_name, [])

    if spec.wrong_ancestors:
        rng = rng or random.Random(shuffle_seed)
        ancestors = build_shuffled_ancestors(
            hierarchy, class_index, max_ancestors=max_ancestors, rng=rng
        )
    else:
        ancestors = class_ancestor_names(hierarchy, class_index, max_ancestors=max_ancestors)
    return format_prompt(spec, class_name, ancestors)


def build_prompt_texts(
    template: str,
    hierarchy: Any = None,
    class_names: Optional[Sequence[str]] = None,
    num_classes: Optional[int] = None,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    max_ancestors: int = DEFAULT_MAX_ANCESTORS,
) -> List[str]:
    """Prompt strings for every class (deterministic ordering = class index)."""
    if num_classes is None:
        if class_names is not None:
            num_classes = len(class_names)
        else:
            num_classes = int(getattr(hierarchy, "num_classes", 0) or 1000)
    if class_names is None and hierarchy is not None:
        names = getattr(hierarchy, "class_names", None)
        if names:
            class_names = list(names)
    rng = random.Random(shuffle_seed)
    return [
        build_prompt(
            i,
            template=template,
            hierarchy=hierarchy,
            class_names=class_names,
            shuffle_seed=shuffle_seed,
            max_ancestors=max_ancestors,
            rng=rng if (PROMPT_TEMPLATES.get(template) and PROMPT_TEMPLATES[template].wrong_ancestors) else None,
        )
        for i in range(int(num_classes))
    ]


def build_all_prompt_texts(
    hierarchy: Any = None,
    class_names: Optional[Sequence[str]] = None,
    templates: Sequence[str] = DEFAULT_TEMPLATES,
    num_classes: Optional[int] = None,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    max_ancestors: int = DEFAULT_MAX_ANCESTORS,
) -> Dict[str, List[str]]:
    """``{template_name: [prompt per class]}`` for all requested templates."""
    return {
        name: build_prompt_texts(
            name,
            hierarchy=hierarchy,
            class_names=class_names,
            num_classes=num_classes,
            shuffle_seed=shuffle_seed,
            max_ancestors=max_ancestors,
        )
        for name in templates
    }


# ---------------------------------------------------------------------------
# encoders (lazily depend on CLIP / OpenCLIP / torch)
# ---------------------------------------------------------------------------


class PromptEncoder:
    """Minimal interface: text prompts -> (K, D) features; images -> (B, D)."""

    name: str = "encoder"
    logit_scale: float = DEFAULT_LOGIT_SCALE

    def encode_text(self, prompts: Sequence[str]):  # pragma: no cover - interface
        raise NotImplementedError

    def encode_images(self, images):  # pragma: no cover - interface
        raise NotImplementedError


def _to_numpy_2d(features: Any):
    """Convert torch/numpy features into a float numpy 2-D array."""
    import numpy as np  # local import: module stays importable without numpy

    if features is None:
        return None
    if hasattr(features, "detach"):
        features = features.detach()
    if hasattr(features, "cpu"):
        features = features.cpu()
    if hasattr(features, "numpy"):
        features = features.numpy()
    arr = np.asarray(features)
    if arr.dtype == object:
        arr = np.stack([np.asarray(x, dtype="float64") for x in arr])
    return np.asarray(arr, dtype="float64")


class OpenAICLIPEncoder(PromptEncoder):
    """Wraps ``clip.load(...)`` (OpenAI CLIP) behind :class:`PromptEncoder`."""

    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None, logit_scale: Optional[float] = None):
        try:  # pragma: no cover - requires the openai clip package
            import clip  # type: ignore
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "OpenAI CLIP is required for OpenAICLIPEncoder; install with "
                "`pip install git+https://github.com/openai/CLIP.git` "
                f"({exc})"
            )
        self._torch = torch
        self.name = f"clip_{model_name}"
        model, _preprocess = clip.load(model_name, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        scale = logit_scale
        if scale is None:
            try:
                scale = float(model.logit_scale.exp().item())
            except Exception:
                scale = DEFAULT_LOGIT_SCALE
        self.logit_scale = float(scale)

    def encode_text(self, prompts: Sequence[str]):  # pragma: no cover - needs weights
        torch = self._torch
        with torch.no_grad():
            tokens = __import__("clip").tokenize(list(prompts))
            feats = self.model.encode_text(tokens.to(self.device)).float()
        return feats

    def encode_images(self, images):  # pragma: no cover - needs weights
        torch = self._torch
        with torch.no_grad():
            feats = self.model.encode_image(images.to(self.device) if hasattr(images, "to") else images)
        return feats.float()


class OpenCLIPEncoder(PromptEncoder):
    """Wraps ``open_clip.create_model_and_transforms`` behind :class:`PromptEncoder`."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
        logit_scale: Optional[float] = None,
    ):
        try:  # pragma: no cover - requires open_clip_torch
            import open_clip  # type: ignore
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "open_clip_torch is required for OpenCLIPEncoder; install with "
                f"`pip install open_clip_torch` ({exc})"
            )
        self._open_clip = open_clip
        self._torch = torch
        self.name = f"openclip_{model_name}_{pretrained}"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _train, _preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        model.eval()
        self.model = model
        scale = logit_scale
        if scale is None:
            try:
                scale = float(model.logit_scale.exp().item())
            except Exception:
                scale = DEFAULT_LOGIT_SCALE
        self.logit_scale = float(scale)

    def encode_text(self, prompts: Sequence[str]):  # pragma: no cover - needs weights
        torch = self._torch
        with torch.no_grad():
            tokens = self._open_clip.tokenize(list(prompts)).to(self.device)
            feats = self.model.encode_text(tokens).float()
        return feats

    def encode_images(self, images):  # pragma: no cover - needs weights
        torch = self._torch
        with torch.no_grad():
            feats = self.model.encode_image(images.to(self.device) if hasattr(images, "to") else images)
        return feats.float()


class CallableTextEncoder(PromptEncoder):
    """Adapter for an arbitrary object exposing text/image encoders.

    Accepts the VLM wrappers from ``src/models/vlm_zoo.py`` (which expose
    ``encode_text`` / ``encode_images`` or a joint ``encode`` method) as well as
    plain callables.  Useful for tests (no weights downloaded).
    """

    def __init__(
        self,
        text_fn: Union[Callable[[Sequence[str]], Any], PromptEncoder],
        image_fn: Optional[Callable[[Any], Any]] = None,
        name: str = "callable_encoder",
        logit_scale: float = DEFAULT_LOGIT_SCALE,
    ):
        self._text_target = text_fn
        self._image_fn = image_fn
        self.name = name
        self.logit_scale = float(logit_scale)
        if isinstance(text_fn, PromptEncoder) and logit_scale == DEFAULT_LOGIT_SCALE:
            self.logit_scale = float(text_fn.logit_scale)

    @staticmethod
    def _call(obj: Any, fn_names: Sequence[str], *args, **kwargs):
        if callable(obj) and not any(hasattr(obj, n) for n in fn_names):
            return obj(*args, **kwargs)
        for n in fn_names:
            fn = getattr(obj, n, None)
            if callable(fn):
                return fn(*args, **kwargs)
        raise TypeError(f"{type(obj).__name__} exposes none of {list(fn_names)}")

    def encode_text(self, prompts: Sequence[str]):
        if isinstance(self._text_target, PromptEncoder):
            return self._text_target.encode_text(list(prompts))
        return self._call(
            self._text_target,
            ("encode_text", "encode_texts", "text_features", "encode_prompts", "encode"),
            list(prompts),
        )

    def encode_images(self, images):
        if self._image_fn is not None:
            return self._call(self._image_fn, ("encode_images", "encode_image", "image_features", "encode"), images)
        if isinstance(self._text_target, PromptEncoder):
            return self._text_target.encode_images(images)
        return self._call(
            self._text_target,
            ("encode_images", "encode_image", "image_features", "encode"),
            images,
        )


def build_encoder(
    encoder: Optional[Any] = None,
    backbone: Optional[str] = None,
    backend: str = "auto",
    device: Optional[str] = None,
    logit_scale: Optional[float] = None,
) -> PromptEncoder:
    """Resolve an encoder object.

    ``encoder`` may already be a :class:`PromptEncoder`, a VLM wrapper, or a
    callable.  Otherwise ``backbone``/``backend`` select OpenAI CLIP or OpenCLIP
    (``auto`` prefers OpenCLIP for names containing ``-14``/``laion``).
    """
    if isinstance(encoder, PromptEncoder):
        return encoder
    if encoder is not None:
        # a VLM-zoo wrapper: adapt, keeping a discovered logit scale if present
        scale = logit_scale
        if scale is None:
            try:
                raw = getattr(encoder, "logit_scale", None)
                raw = raw.exp().item() if hasattr(raw, "exp") else raw
                scale = float(raw) if raw is not None else None
            except Exception:
                scale = None
        return CallableTextEncoder(
            encoder,
            name=str(getattr(encoder, "name", "vlm")),
            logit_scale=float(scale) if scale else DEFAULT_LOGIT_SCALE,
        )

    name = backbone or "ViT-B/32"
    use_open_clip = backend == "open_clip" or (
        backend == "auto" and ("laion" in name or name.count("-") >= 3)
    )
    if use_open_clip:
        model_name, _, pretrained = name.partition("::")
        return OpenCLIPEncoder(
            model_name or "ViT-B-32",
            pretrained=pretrained or "laion2b_s34b_b79k",
            device=device,
            logit_scale=logit_scale,
        )
    return OpenAICLIPEncoder(name, device=device, logit_scale=logit_scale)


# ---------------------------------------------------------------------------
# zero-shot evaluation
# ---------------------------------------------------------------------------


def encode_text_prompts(encoder: PromptEncoder, prompts: Sequence[str]):
    """Encode prompts with a :class:`PromptEncoder` (or a duck-typed object)."""
    if isinstance(encoder, PromptEncoder):
        return encoder.encode_text(prompts)
    return CallableTextEncoder(encoder).encode_text(prompts)


def encode_image_loader(
    encoder: PromptEncoder,
    loader: Iterable,
    device: Optional[str] = None,
    max_batches: Optional[int] = None,
    desc: str = "",
):
    """Collect ``(image_features, targets)`` by streaming a data loader."""
    import numpy as np

    feats: List[Any] = []
    targets: List[Any] = []
    iterator = loader
    if desc:
        try:  # pragma: no cover - cosmetic
            from tqdm.auto import tqdm

            iterator = tqdm(loader, desc=desc)
        except Exception:
            iterator = loader
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            images, labels = batch[0], batch[1]
        elif isinstance(batch, Mapping):
            images = batch.get("image", batch.get("images"))
            labels = batch.get("label", batch.get("labels"))
        else:
            images, labels = batch, None
        out = CallableTextEncoder._call(
            encoder,
            ("encode_images", "encode_image", "image_features", "encode"),
            images,
        )
        feats.append(_to_numpy_2d(out))
        if labels is not None:
            targets.append(np.asarray(_to_numpy_2d(labels) if hasattr(labels, "cpu") else labels).reshape(-1))
    if not feats:
        return np.zeros((0, 0), dtype="float64"), np.zeros((0,), dtype="int64")
    img = np.concatenate(feats, axis=0)
    tgt = np.concatenate(targets, axis=0).astype("int64") if targets else np.zeros((img.shape[0],), dtype="int64")
    return img, tgt


def _unit_norm(arr):
    import numpy as np

    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.clip(norm, 1e-12, None)


def zero_shot_logits(
    image_features: Any,
    text_features: Any,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
    normalize: bool = True,
    mean_prompts: Optional[Any] = None,
):
    """Zero-shot logits ``scale * f_img . f_text^T`` (optionally prompt-ensembled)."""
    import numpy as np

    img = _to_numpy_2d(image_features)
    txt = _to_numpy_2d(text_features)
    if normalize:
        img = _unit_norm(img)
        txt = _unit_norm(txt)
    if mean_prompts is not None:
        extra = _to_numpy_2d(mean_prompts)
        if normalize:
            extra = _unit_norm(extra)
        txt = _unit_norm(txt + extra)
    return float(logit_scale) * (img @ txt.T)


def zero_shot_metrics(
    logits: Any,
    targets: Any,
    compute_top5: bool = True,
) -> Dict[str, float]:
    """Top-1 / Top-5 accuracy and test-time cross-entropy (Table 14 metrics)."""
    import numpy as np

    logits = _to_numpy_2d(logits)
    targets = np.asarray(_to_numpy_2d(targets)).reshape(-1).astype("int64")
    if logits.size == 0 or targets.size == 0:
        return {"top1": float("nan"), "top5": float("nan"), "ce": float("nan"), "n": 0}

    n = min(logits.shape[0], targets.shape[0])
    logits, targets = logits[:n], targets[:n]
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.clip(np.exp(shifted).sum(axis=1, keepdims=True), 1e-12, None))
    rows = np.arange(n)
    ce = float(-np.mean(log_probs[rows, np.clip(targets, 0, logits.shape[1] - 1)]))
    preds = logits.argmax(axis=1)
    top1 = float(np.mean(preds == targets))
    top5 = float("nan")
    if compute_top5 and logits.shape[1] >= 5:
        k = 5
        topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
        top5 = float(np.mean([targets[i] in topk[i] for i in range(n)]))
    return {"top1": top1, "top5": top5, "ce": ce, "n": int(n)}


def evaluate_prompts(
    encoder: PromptEncoder,
    image_features: Any,
    targets: Any,
    prompts: Sequence[str],
    logit_scale: Optional[float] = None,
    compute_top5: bool = True,
    return_logits: bool = False,
) -> Dict[str, Any]:
    """Zero-shot metrics for one prompt set (one template)."""
    scale = float(logit_scale if logit_scale is not None else getattr(encoder, "logit_scale", DEFAULT_LOGIT_SCALE))
    text_features = encode_text_prompts(encoder, prompts)
    logits = zero_shot_logits(image_features, text_features, logit_scale=scale)
    result: Dict[str, Any] = zero_shot_metrics(logits, targets, compute_top5=compute_top5)
    result["logit_scale"] = scale
    if return_logits:
        result["logits"] = logits
        result["text_features"] = _to_numpy_2d(text_features)
    return result


def run_prompt_evaluation(
    encoder: PromptEncoder,
    datasets: Mapping[str, Any],
    hierarchy: Any = None,
    class_names: Optional[Sequence[str]] = None,
    templates: Sequence[str] = DEFAULT_TEMPLATES,
    num_classes: Optional[int] = None,
    config: Optional[PromptConfig] = None,
    image_features_by_dataset: Optional[Mapping[str, Any]] = None,
    targets_by_dataset: Optional[Mapping[str, Any]] = None,
    loader_factory: Optional[Callable[[str, Any], Any]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Reproduce Table 14: per-dataset, per-template Top-1/Top-5/test-CE.

    ``datasets`` maps dataset name -> data loader (or already-extracted
    ``(features, targets)`` tuple when ``loader_factory`` is None).  Results are
    ``{dataset: {template: {top1, top5, ce, n}}}``.
    """
    config = config or PromptConfig()
    if num_classes is None:
        if class_names is not None:
            num_classes = len(class_names)
        else:
            num_classes = int(getattr(hierarchy, "num_classes", 0) or 1000)

    prompt_bank = build_all_prompt_texts(
        hierarchy=hierarchy,
        class_names=class_names,
        templates=templates,
        num_classes=num_classes,
        shuffle_seed=config.shuffle_seed,
        max_ancestors=config.max_ancestors,
    )
    text_features: Dict[str, Any] = {
        name: encode_text_prompts(encoder, prompts) for name, prompts in prompt_bank.items()
    }

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset_name, source in datasets.items():
        if image_features_by_dataset is not None and dataset_name in image_features_by_dataset:
            feats = image_features_by_dataset[dataset_name]
            tgts = (targets_by_dataset or {}).get(dataset_name)
        elif isinstance(source, (tuple, list)) and len(source) == 2:
            feats, tgts = source
        elif loader_factory is not None:
            feats, tgts = encode_image_loader(
                encoder, loader_factory(dataset_name, source), max_batches=config.max_batches, desc=dataset_name
            )
        else:
            feats, tgts = encode_image_loader(
                encoder, source, max_batches=config.max_batches, desc=dataset_name
            )
        out[dataset_name] = {}
        for name in templates:
            if name == "ensemble_taxonomy":
                scale = float(getattr(encoder, "logit_scale", DEFAULT_LOGIT_SCALE))
                logits = zero_shot_logits(
                    feats,
                    text_features["baseline"],
                    logit_scale=scale,
                    mean_prompts=text_features.get("taxonomy_parent"),
                )
                metrics = zero_shot_metrics(logits, tgts)
                metrics["logit_scale"] = scale
            else:
                metrics = evaluate_prompts(encoder, feats, tgts, prompt_bank[name])
            out[dataset_name][name] = metrics
        if verbose:
            row = out[dataset_name]
            LOG.info(
                "%s: " + ", ".join(f"{k}={row[k]['top1']:.4f}" for k in templates if k in row),
                dataset_name,
            )
    return out


# ---------------------------------------------------------------------------
# Table 14 validation helpers
# ---------------------------------------------------------------------------


def _norm_name(name: str) -> str:
    return DISPLAY_NAMES.get(str(name).lower(), str(name))


def check_against_table14(
    table: Mapping[str, Mapping[str, Mapping[str, float]]],
    tolerance: float = 0.05,
    require_improvement: bool = True,
) -> List[str]:
    """Validate observed Table-14 numbers against the paper.

    Returns a list of human-readable messages describing failures (empty list =
    everything within tolerance).  Checks performed:

    1. absolute Top-1 values where the paper reports them,
    2. ``taxonomy_parent`` beats ``baseline`` on Top-1 and test-CE for every
       dataset (Section 4.3.3),
    3. ``stack_parent`` / ``shuffle_parent`` do not beat ``taxonomy_parent``.
    """
    messages: List[str] = []
    for ds, rows in table.items():
        ref = TABLE14_REFERENCE.get(ds, {})
        for template, target in ref.items():
            if target is None or template not in rows:
                continue
            got = float(rows[template].get("top1", float("nan")))
            if not math.isfinite(got) or abs(got - target) > tolerance:
                messages.append(
                    f"[{_norm_name(ds)}] {template} top1={got:.3f} vs paper {target:.3f} "
                    f"(tol {tolerance})"
                )
        if not require_improvement:
            continue
        base = rows.get("baseline", {})
        tax = rows.get("taxonomy_parent", {})
        if base and tax:
            if float(tax.get("top1", -1)) <= float(base.get("top1", -1)):
                messages.append(
                    f"[{_norm_name(ds)}] taxonomy_parent top1 {tax.get('top1'):.3f} "
                    f"<= baseline {base.get('top1'):.3f}"
                )
            base_ce, tax_ce = base.get("ce"), tax.get("ce")
            if base_ce is not None and tax_ce is not None and float(tax_ce) >= float(base_ce):
                messages.append(
                    f"[{_norm_name(ds)}] taxonomy_parent CE {float(tax_ce):.3f} "
                    f">= baseline {float(base_ce):.3f}"
                )
        for ablate in ("stack_parent", "shuffle_parent"):
            if ablate in rows and tax:
                if float(rows[ablate].get("top1", -1)) > float(tax.get("top1", -1)):
                    messages.append(
                        f"[{_norm_name(ds)}] ablation {ablate} beats taxonomy_parent "
                        f"({rows[ablate].get('top1'):.3f} > {tax.get('top1'):.3f})"
                    )
    return messages


def format_table14(
    table: Mapping[str, Mapping[str, Mapping[str, float]]],
    templates: Sequence[str] = DEFAULT_TEMPLATES,
    metric: str = "top1",
    decimals: int = 4,
) -> str:
    """Render a Table-14-style block (rows = dataset, columns = template)."""
    lines = [
        f"Table 14 ({metric})",
        "dataset".ljust(14) + "".join(t.ljust(18) for t in templates),
        "-" * (14 + 18 * len(templates)),
    ]
    for ds, rows in table.items():
        cells = []
        for t in templates:
            val = rows.get(t, {}).get(metric)
            cells.append(("  n/a" if val is None or not math.isfinite(float(val)) else f"{float(val):.{decimals}f}").ljust(18))
        lines.append(_norm_name(ds).ljust(14) + "".join(cells))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# offline self test
# ---------------------------------------------------------------------------


def _self_test() -> int:  # pragma: no cover - manual smoke test
    """Prompt construction + zero-shot metric plumbing without any weights."""
    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from src.hierarchy.wordnet import build_two_pair_hierarchy  # noqa: WPS433

    hierarchy = build_two_pair_hierarchy()
    names = [f"class {i}" for i in range(hierarchy.num_classes)]
    bank = build_all_prompt_texts(hierarchy=hierarchy, class_names=names, num_classes=hierarchy.num_classes)
    for tname, prompts in bank.items():
        assert len(prompts) == hierarchy.num_classes, tname
        assert all(isinstance(p, str) and p for p in prompts), tname
    assert bank["baseline"][0] == "class 0"
    assert "which is a type of" in bank["taxonomy_parent"][2]
    assert "," in bank["stack_parent"][2]

    # a deterministic toy encoder: class one-hot text features
    k = hierarchy.num_classes
    def text_fn(prompts):
        feats = np.zeros((len(prompts), k))
        for i, _p in enumerate(prompts):
            feats[i, i] = 1.0
        return feats

    enc = CallableTextEncoder(text_fn, name="toy", logit_scale=1.0)
    rng = np.random.RandomState(0)
    img = rng.randn(32, k) + np.eye(k)[rng.randint(0, k, size=32)]
    tgt = rng.randint(0, k, size=32)
    metrics = evaluate_prompts(enc, img, tgt, bank["baseline"])
    assert metrics["top1"] >= 0.0 and metrics["ce"] > 0.0
    LOG.info("self-test metrics: %s", metrics)
    print("prompt_engineering self-test OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(_self_test())

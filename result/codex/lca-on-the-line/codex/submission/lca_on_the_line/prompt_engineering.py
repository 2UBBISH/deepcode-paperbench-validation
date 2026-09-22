"""Prompt templates and taxonomy-alignment prompts (Section 4.3.3).

Two things live here:

1. The standard OpenAI CLIP *prompt ensemble* used for zero-shot ImageNet
   classification -- the paper evaluates every VLM this way.
2. The taxonomy-alignment prompts of Section 4.3.3 / Table 14, which inject the
   WordNet ``is-a`` path of a class into the text prompt:

   * ``baseline``       -- ``<dalmatian>``
   * ``stack_parent``   -- ``<dalmatian, dog, animal>`` (correct path, but the
     model is *not* told the relationships hold)
   * ``taxonomy_parent``-- ``<dalmatian, which is a type of a dog, which is a
     type of an animal>``
   * ``shuffle_parent`` -- ``<dalmatian, which is a type of an organism, which
     is a type of a seabird>`` (random, incorrect taxonomy relationships)
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Sequence

import numpy as np

from .hierarchy import (
    DEFAULT_CLASS_INDEX_JSON,
    WordNetHierarchy,
    load_wordnet_hierarchy,
)

# --------------------------------------------------------------------------- #
# names
# --------------------------------------------------------------------------- #
_CLASS_INDEX_JSON = os.environ.get(
    "LCA_CLASS_INDEX_JSON", DEFAULT_CLASS_INDEX_JSON
)


def imagenet_classnames(path: str = _CLASS_INDEX_JSON) -> List[str]:
    """The 1000 ImageNet class names in class-index order."""
    with open(path, "r") as fh:
        raw = json.load(fh)
    if isinstance(raw, list):
        pairs = raw
    else:
        pairs = [raw[str(i)] for i in range(len(raw))]
    return [p[1] for p in pairs]


# --------------------------------------------------------------------------- #
# CLIP prompt ensemble (standard ImageNet templates)
# --------------------------------------------------------------------------- #
IMAGENET_TEMPLATES: List[str] = [
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
    "a black and white photo of a {}.",
    "a warm photo of a {}.",
    "a photo of a {}.",
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
    "a black and white photo of the {}.",
    "a plushie {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a photo of the hard to see {}.",
    "a good photo of the {}.",
    "a photo of a nice {}.",
    "the origami {}.",
    "a cartoon {}.",
    "a photo of a small {}.",
    "a photo of the dirty {}.",
    "a photo of the clean {}.",
]


# --------------------------------------------------------------------------- #
# taxonomy prompts
# --------------------------------------------------------------------------- #
TAXONOMY_PROMPT_MODES = ("baseline", "stack_parent", "taxonomy_parent", "shuffle_parent")

_ARTICLE = lambda word: "an" if word[:1].lower() in "aeiou" else "a"


def taxonomy_parent_names(
    hierarchy: WordNetHierarchy,
    depth: int = 2,
) -> List[List[str]]:
    """For each class, the list of ancestors (parents, grand-parents, ...)."""
    out: List[List[str]] = []
    for synset in hierarchy.class_synsets:
        path: List[str] = []
        cur = hierarchy.parent.get(synset)
        while cur is not None and len(path) < depth:
            name = hierarchy.node_names.get(cur, cur)
            path.append(_pretty_name(name))
            cur = hierarchy.parent.get(cur)
        out.append(path)
    return out


def _pretty_name(wordnet_name: str) -> str:
    """``dog.n.01`` -> ``dog`` (drop the part-of-speech / sense suffix)."""
    return wordnet_name.split(".")[0].replace("_", " ")


def build_taxonomy_prompts(
    hierarchy: Optional[WordNetHierarchy] = None,
    mode: str = "taxonomy_parent",
    template: str = "a photo of a {}.",
    depth: int = 2,
    seed: int = 0,
) -> List[str]:
    """Return one prompt per ImageNet class for the requested prompt mode.

    ``mode`` is one of :data:`TAXONOMY_PROMPT_MODES`.  ``depth`` controls how
    many ancestor levels are injected (Table 14 uses two: parent + grandparent).
    """
    if mode not in TAXONOMY_PROMPT_MODES:
        raise ValueError("unknown prompt mode %r" % mode)
    hierarchy = hierarchy or load_wordnet_hierarchy()
    names = [
        n.replace("_", " ")
        for n in (hierarchy.class_names or imagenet_classnames())
    ]
    parents = taxonomy_parent_names(hierarchy, depth=depth)

    rng = random.Random(seed)
    prompts: List[str] = []
    for cls_name, anc in zip(names, parents):
        if mode == "baseline" or not anc:
            body = cls_name
        elif mode == "stack_parent":
            body = ", ".join([cls_name] + anc)
        elif mode == "taxonomy_parent":
            parts = [cls_name]
            for parent in anc:
                parts.append("which is a type of %s %s" % (_ARTICLE(parent), parent))
            body = ", ".join(parts)
        else:  # shuffle_parent
            random_anc = [names[rng.randrange(len(names))] for _ in anc]
            parts = [cls_name]
            for parent in random_anc:
                parts.append("which is a type of %s %s" % (_ARTICLE(parent), parent))
            body = ", ".join(parts)
        prompts.append(template.format(body))
    return prompts


def class_prompts_for_ensemble(
    hierarchy: Optional[WordNetHierarchy] = None,
    mode: str = "baseline",
    templates: Optional[Sequence[str]] = None,
    depth: int = 2,
    seed: int = 0,
) -> List[List[str]]:
    """Per-class list of templated prompts (used to build text embeddings)."""
    templates = list(templates or IMAGENET_TEMPLATES)
    out: List[List[str]] = []
    for template in templates:
        out.append(
            build_taxonomy_prompts(
                hierarchy=hierarchy,
                mode=mode,
                template=template,
                depth=depth,
                seed=seed,
            )
        )
    # transpose: (n_classes, n_templates)
    return [list(row) for row in zip(*out)]


def evaluate_prompt_protocol(
    classifier,
    images: Sequence,
    targets: Sequence[int],
    hierarchy: Optional[WordNetHierarchy] = None,
    modes: Sequence[str] = TAXONOMY_PROMPT_MODES,
    depth: int = 2,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Run the Table 14 protocol for a CLIP-style zero-shot classifier.

    For every prompt mode we build the class embeddings, classify the images and
    report Top-1 accuracy and test-time cross-entropy.
    """
    import torch

    hierarchy = hierarchy or load_wordnet_hierarchy()
    results: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        class_prompts = class_prompts_for_ensemble(
            hierarchy=hierarchy, mode=mode, depth=depth, seed=seed
        )
        text_features = classifier.encode_prompts(class_prompts)
        logits = classifier.logits_with_text(images, text_features)
        preds = logits.argmax(axis=1)
        targets_arr = np.asarray(targets)
        top1 = float((preds == targets_arr).mean())
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        eps = 1e-12
        ce = float(-np.log(np.clip(probs[np.arange(len(targets_arr)), targets_arr], eps, 1.0)).mean())
        results[mode] = {"top1": top1, "ce": ce}
    return results

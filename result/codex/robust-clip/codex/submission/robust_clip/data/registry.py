"""Loaders for the small data files bundled with the repository.

``imagenet_classnames.json`` holds the 1000 ImageNet-1k class names used for
TeCoA (whose loss needs the class *text* of every image in the batch) and for the
ImageNet zero-shot evaluation.  ``openai_templates.json`` holds the 80 prompt
templates of CLIP, used for all zero-shot classification datasets (Sec. 4.3).
Both files are snapshots of ``open_clip.IMAGENET_CLASSNAMES`` and
``open_clip.OPENAI_IMAGENET_TEMPLATES`` so that the reproduction does not depend
on the installed open_clip version.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import List

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def data_path(name: str) -> str:
    return os.path.join(DATA_DIR, name)


@lru_cache(maxsize=None)
def _load_json(name: str):
    with open(data_path(name), "r") as handle:
        return json.load(handle)


def imagenet_classnames() -> List[str]:
    """The 1000 ImageNet-1k class names."""
    return list(_load_json("imagenet_classnames.json"))


def openai_templates() -> List[str]:
    """The 80 OpenAI CLIP prompt templates (``'a photo of a {}.'`` style)."""
    return list(_load_json("openai_templates.json"))


def simple_templates() -> List[str]:
    """Minimal template set, e.g. for TextVQA-like class sets."""
    return ["a photo of a {}."]

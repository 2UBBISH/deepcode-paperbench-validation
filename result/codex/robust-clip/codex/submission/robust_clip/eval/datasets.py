"""Datasets of the LVLM evaluations (Sec. 4.1 and Sec. 4.4).

Tasks and metrics of the paper:

* image captioning: COCO and Flickr30k, CIDEr score with the five ground truth
  captions of every image,
* visual question answering: VQAv2 and TextVQA, accuracy with the ten human
  answers of every question (the official metric, see
  :mod:`robust_clip.eval.vqa`),
* POPE (Sec. 4.4): object hallucination benchmark, F1 score of the yes/no answers
  in the ``random``, ``popular`` and ``adversarial`` splits,
* SQA-I (Sec. 4.4): science question answering with chain-of-thought, accuracy.

For the adversarial evaluations *"we use 500 randomly sampled images"* and for
the clean evaluations all available samples (Sec. 4.1).
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

LOGGER = logging.getLogger(__name__)

COCO_POPE_URL = (
    "https://raw.githubusercontent.com/haotian-liu/LLaVA/main/playground/data/eval/pope/coco_pope_{split}.json"
)


@dataclass
class LVMLEvalSample:
    """One evaluation sample."""

    image: object  # PIL.Image
    question: Optional[str] = None
    answers: List[str] = field(default_factory=list)
    question_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


def sample_indices(n: int, k: Optional[int], seed: int = 0) -> List[int]:
    """``k`` random indices out of ``n`` (all of them if ``k`` is ``None`` or larger)."""
    if k is None or k >= n:
        return list(range(n))
    generator = random.Random(seed)
    return sorted(generator.sample(range(n), k))


def _to_pil(image) -> object:
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict) and "bytes" in image:
        import io

        return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
    return Image.fromarray(image).convert("RGB")


# ----------------------------------------------------------------------
# captioning
# ----------------------------------------------------------------------
def load_coco(
    max_samples: Optional[int] = None,
    dataset_name: str = "nlphuji/mscoco_2014_5k_test_image_text_retrieval",
    split: str = "test",
    hf_token: Optional[str] = None,
    seed: int = 0,
) -> List[LVMLEvalSample]:
    """COCO captioning (5 ground truth captions per image)."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split, token=hf_token, trust_remote_code=True)
    indices = sample_indices(len(dataset), max_samples, seed=seed)
    samples = []
    for index in indices:
        row = dataset[index]
        captions = row.get("caption") or row.get("captions") or row.get("sentences")
        if isinstance(captions, str):
            captions = [captions]
        if captions is not None and isinstance(captions[0], dict):
            captions = [c.get("raw") or c.get("sent") or c.get("caption") for c in captions]
        samples.append(
            LVMLEvalSample(
                image=_to_pil(row["image"]),
                answers=list(captions),
                question_id=str(index),
            )
        )
    return samples


def load_flickr30k(
    max_samples: Optional[int] = None,
    dataset_name: str = "nlphuji/flickr30k",
    split: str = "test",
    hf_token: Optional[str] = None,
    seed: int = 0,
) -> List[LVMLEvalSample]:
    """Flickr30k captioning (5 ground truth captions per image)."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split, token=hf_token, trust_remote_code=True)
    indices = sample_indices(len(dataset), max_samples, seed=seed)
    samples = []
    for index in indices:
        row = dataset[index]
        captions = row.get("caption") or row.get("sentences") or row.get("captions")
        if isinstance(captions, str):
            captions = [captions]
        samples.append(
            LVMLEvalSample(image=_to_pil(row["image"]), answers=list(captions), question_id=str(index))
        )
    return samples


# ----------------------------------------------------------------------
# visual question answering
# ----------------------------------------------------------------------
def load_vqav2(
    max_samples: Optional[int] = None,
    dataset_name: str = "lmms-lab/VQAv2",
    split: str = "validation",
    hf_token: Optional[str] = None,
    seed: int = 0,
) -> List[LVMLEvalSample]:
    """VQAv2 (ten human answers per question)."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split, token=hf_token, trust_remote_code=True)
    indices = sample_indices(len(dataset), max_samples, seed=seed)
    samples = []
    for index in indices:
        row = dataset[index]
        answers = row.get("answers") or row.get("multiple_answers") or []
        if answers and isinstance(answers[0], dict):
            answers = [a.get("answer") or a.get("raw") for a in answers]
        samples.append(
            LVMLEvalSample(
                image=_to_pil(row["image"]),
                question=row.get("question"),
                answers=list(answers),
                question_id=str(row.get("question_id", index)),
            )
        )
    return samples


def load_textvqa(
    max_samples: Optional[int] = None,
    dataset_name: str = "facebook/textvqa",
    split: str = "validation",
    hf_token: Optional[str] = None,
    seed: int = 0,
) -> List[LVMLEvalSample]:
    """TextVQA (ten human answers per question, own normalization)."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split, token=hf_token, trust_remote_code=True)
    indices = sample_indices(len(dataset), max_samples, seed=seed)
    samples = []
    for index in indices:
        row = dataset[index]
        answers = row.get("answers") or row.get("answers_text") or []
        if answers and isinstance(answers[0], dict):
            answers = [a.get("answer") or a.get("raw") for a in answers]
        if isinstance(answers, str):
            answers = [answers]
        image = row.get("image") or row.get("image_bytes")
        samples.append(
            LVMLEvalSample(
                image=_to_pil(image),
                question=row.get("question"),
                answers=list(answers),
                question_id=str(row.get("question_id", index)),
            )
        )
    return samples


# ----------------------------------------------------------------------
# POPE
# ----------------------------------------------------------------------
def load_pope(
    split: str = "random",
    max_samples: Optional[int] = None,
    annotation_dir: Optional[str] = None,
    coco_dataset_name: str = "nlphuji/mscoco_2014_5k_test_image_text_retrieval",
    hf_token: Optional[str] = None,
    seed: int = 0,
) -> List[LVMLEvalSample]:
    """POPE hallucination benchmark (``random`` / ``popular`` / ``adversarial``).

    The annotation files are the ones of the LLaVA repository
    (``coco_pope_<split>.json``), which the addendum names as the source of the
    POPE implementation of the paper.
    """
    if split not in {"random", "popular", "adversarial"}:
        raise ValueError(f"unknown POPE split {split!r}")
    annotations = None
    if annotation_dir is not None:
        path = os.path.join(annotation_dir, f"coco_pope_{split}.json")
        if os.path.isfile(path):
            with open(path) as handle:
                annotations = [json.loads(line) for line in handle if line.strip()]
    if annotations is None:
        import urllib.request

        url = COCO_POPE_URL.format(split=split)
        with urllib.request.urlopen(url) as response:  # noqa: S310 - documented URL
            payload = response.read().decode("utf-8")
        annotations = [json.loads(line) for line in payload.splitlines() if line.strip()]

    from datasets import load_dataset

    images = load_dataset(coco_dataset_name, split="test", token=hf_token, trust_remote_code=True)
    by_file = {}
    for index in range(len(images)):
        row = images[index]
        key = row.get("filename") or row.get("image_id") or row.get("cocoid")
        by_file[str(key)] = _to_pil(row["image"])

    indices = sample_indices(len(annotations), max_samples, seed=seed)
    samples = []
    for index in indices:
        annotation = annotations[index]
        image_file = str(annotation.get("image", "")).split("/")[-1].replace(".jpg", "")
        image = by_file.get(image_file)
        if image is None:
            continue
        samples.append(
            LVMLEvalSample(
                image=image,
                question=annotation.get("text"),
                answers=[annotation.get("label", "yes")],
                question_id=image_file,
                metadata={"object": annotation.get("text", "").split()[3] if annotation.get("text") else ""},
            )
        )
    return samples


def pope_object(question: str) -> str:
    """Extract the object of a POPE question (``Is there a <object> in the image?``)."""
    question = question.strip().rstrip("?")
    if question.lower().startswith("is there a"):
        return question[len("Is there a") :].replace("in the image", "").strip()
    return question


# ----------------------------------------------------------------------
# SQA-I
# ----------------------------------------------------------------------
def load_sqa(
    max_samples: Optional[int] = 10000,
    dataset_name: str = "lmms-lab/ScienceQA",
    split: str = "test",
    hf_token: Optional[str] = None,
    seed: int = 0,
) -> List[LVMLEvalSample]:
    """SQA-I: image/question pairs with multiple choice answers and explanations.

    The paper uses a subset of 10k image/question pairs from ScienceQA
    (Lu et al., 2022) and the answer extraction of the LLaVA repository.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split, token=hf_token, trust_remote_code=True)
    indices = sample_indices(len(dataset), max_samples, seed=seed)
    samples = []
    for index in indices:
        row = dataset[index]
        if row.get("image") is None:  # SQA-I = the image based subset
            continue
        choices = row.get("choices") or []
        answer_index = row.get("answer")
        answer = choices[answer_index] if isinstance(answer_index, int) and choices else str(answer_index)
        samples.append(
            LVMLEvalSample(
                image=_to_pil(row["image"]),
                question=row.get("question"),
                answers=[answer],
                question_id=str(index),
                metadata={
                    "choices": list(choices),
                    "context": row.get("hint") or "",
                    "solution": row.get("solution") or "",
                    "answer_index": answer_index,
                },
            )
        )
    return samples


LVLM_DATASETS = {
    "coco": load_coco,
    "flickr30k": load_flickr30k,
    "vqav2": load_vqav2,
    "textvqa": load_textvqa,
    "pope": load_pope,
    "sqa": load_sqa,
}


def is_vqa(name: str) -> bool:
    return name in {"vqav2", "textvqa"}

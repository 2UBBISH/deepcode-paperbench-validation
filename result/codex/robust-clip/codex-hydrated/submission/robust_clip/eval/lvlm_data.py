"""Dataset access for the LVLM evaluations (COCO, Flickr30k, VQAv2, TextVQA).

Two layouts are supported for every task:

* a *local* layout with the original annotation files (the Karpathy splits for
  captioning, the official VQA / TextVQA json files) plus the image folder --
  this is what the LLaVA evaluation scripts use, and
* the HuggingFace hub equivalent, which makes the code runnable without
  manually downloading the datasets.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from PIL import Image

from ..utils.common import LOGGER


@dataclass
class CaptionExample:
    image: Image.Image
    captions: List[str]
    image_id: Optional[str] = None


@dataclass
class VQAExample:
    image: Image.Image
    question: str
    answers: List[str]           # the ten annotated answers (VQAv2)
    question_id: Optional[str] = None

    @property
    def most_frequent_answers(self) -> List[str]:
        """The five most frequent ground-truth answers (App. B.6)."""
        counts = Counter(self.answers)
        return [ans for ans, _ in counts.most_common(5)]


# --------------------------------------------------------------------------- #
#                                  captioning                                  #
# --------------------------------------------------------------------------- #
def _load_coco_captions(json_path: str, image_root: str) -> List[CaptionExample]:
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    id_to_captions: Dict[int, List[str]] = {}
    for ann in data["annotations"]:
        id_to_captions.setdefault(ann["image_id"], []).append(ann["caption"])
    examples = []
    for image_info in data["images"]:
        file_name = image_info["file_name"]
        path = os.path.join(image_root, file_name)
        if not os.path.exists(path):
            continue
        captions = id_to_captions.get(image_info["id"], [])
        if not captions:
            continue
        examples.append(
            CaptionExample(Image.open(path).convert("RGB"), captions, image_id=str(image_info["id"]))
        )
    return examples


def load_captioning_dataset(
    name: str = "coco",
    root: Optional[str] = None,
    split: str = "test",
    max_samples: Optional[int] = None,
) -> List[CaptionExample]:
    """COCO (5 captions/image) or Flickr30k (5 captions/image)."""
    name = name.lower()
    if root is not None:
        if name in ("coco", "coco_caption"):
            json_path = os.path.join(root, "annotations", f"captions_{split}2014.json")
            if not os.path.exists(json_path):
                json_path = os.path.join(root, "captions_val2014.json")
            image_root = os.path.join(root, f"{split}2014")
            if not os.path.isdir(image_root):
                image_root = os.path.join(root, "images")
            examples = _load_coco_captions(json_path, image_root)
        elif name in ("flickr30k", "flickr"):
            json_path = os.path.join(root, "dataset_flickr30k.json")
            with open(json_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            examples = []
            for item in data["images"]:
                if split == "test" and item.get("split") != "test":
                    continue
                path = os.path.join(root, "flickr30k-images", item["filename"])
                examples.append(
                    CaptionExample(
                        Image.open(path).convert("RGB"),
                        [s["raw"] for s in item["sentences"]],
                        image_id=str(item["imgid"]),
                    )
                )
        else:
            raise ValueError(f"unknown captioning dataset '{name}'")
    else:
        from datasets import load_dataset

        if name in ("coco", "coco_caption"):
            dataset = load_dataset("HuggingFaceM4/COCO", "2014", split="val", trust_remote_code=True)
            examples = [
                CaptionExample(item["image"].convert("RGB"), list(item["sentences_raw"]), str(item["image_id"]))
                for item in dataset
            ]
        elif name in ("flickr30k", "flickr"):
            dataset = load_dataset("nlphuji/flickr30k", split="test", trust_remote_code=True)
            examples = []
            for item in dataset:
                captions = item.get("caption") or item.get("sentences")
                captions = [c["raw"] if isinstance(c, dict) else c for c in captions]
                examples.append(CaptionExample(item["image"].convert("RGB"), captions, str(item.get("img_id"))))
        else:
            raise ValueError(f"unknown captioning dataset '{name}'")

    LOGGER.info("loaded %d captioning examples from %s", len(examples), name)
    if max_samples is not None:
        examples = examples[:max_samples]
    return examples


# --------------------------------------------------------------------------- #
#                                      VQA                                     #
# --------------------------------------------------------------------------- #
def _load_vqa_json(questions_path: str, annotations_path: str, image_root: str) -> List[VQAExample]:
    with open(questions_path, "r", encoding="utf-8") as handle:
        questions = json.load(handle)["questions"]
    with open(annotations_path, "r", encoding="utf-8") as handle:
        annotations = json.load(handle)["annotations"]
    ann_by_id = {ann["question_id"]: ann for ann in annotations}
    examples = []
    for question in questions:
        ann = ann_by_id.get(question["question_id"])
        if ann is None:
            continue
        path = os.path.join(image_root, f"COCO_val2014_{question['image_id']:012d}.jpg")
        if not os.path.exists(path):
            continue
        answers = [a["answer"] for a in ann["answers"]]
        examples.append(
            VQAExample(Image.open(path).convert("RGB"), question["question"], answers, str(question["question_id"]))
        )
    return examples


def load_vqa_dataset(
    name: str = "vqav2",
    root: Optional[str] = None,
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> List[VQAExample]:
    """VQAv2 (ten answers per question) or TextVQA."""
    name = name.lower()
    if root is not None:
        if name in ("vqav2", "vqa"):
            questions = os.path.join(root, f"v2_OpenEnded_mscoco_val2014_questions.json")
            annotations = os.path.join(root, f"v2_mscoco_val2014_annotations.json")
            examples = _load_vqa_json(questions, annotations, os.path.join(root, "val2014"))
        elif name == "textvqa":
            questions = os.path.join(root, "TextVQA_0.5.1_val.json")
            with open(questions, "r", encoding="utf-8") as handle:
                data = json.load(handle)["data"]
            examples = []
            for item in data:
                path = os.path.join(root, "train_images", item["image_id"] + ".jpg")
                if not os.path.exists(path):
                    continue
                examples.append(
                    VQAExample(Image.open(path).convert("RGB"), item["question"], item["answers"], item["question_id"])
                )
        else:
            raise ValueError(f"unknown VQA dataset '{name}'")
    else:
        from datasets import load_dataset

        if name in ("vqav2", "vqa"):
            dataset = load_dataset("HuggingFaceM4/VQAv2", split="validation", trust_remote_code=True)
            examples = []
            for item in dataset:
                answers = item["answers"]
                answers = [a["answer"] if isinstance(a, dict) else a for a in answers]
                examples.append(VQAExample(item["image"].convert("RGB"), item["question"], answers, str(item["question_id"])))
        elif name == "textvqa":
            dataset = load_dataset("facebook/textvqa", split="validation", trust_remote_code=True)
            examples = [
                VQAExample(item["image"].convert("RGB"), item["question"], item["answers"], str(item["question_id"]))
                for item in dataset
            ]
        else:
            raise ValueError(f"unknown VQA dataset '{name}'")

    LOGGER.info("loaded %d VQA examples from %s", len(examples), name)
    if max_samples is not None:
        examples = examples[:max_samples]
    return examples

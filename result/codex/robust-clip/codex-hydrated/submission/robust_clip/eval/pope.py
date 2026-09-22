"""POPE hallucination benchmark (Sec. 4.4, Table 5).

POPE turns object hallucination into a binary task: the LVLM has to answer
``Yes``/``No`` to "Is there a <object> in the image?".  There are three sampling
strategies -- ``random``, ``popular`` and ``adversarial`` -- and the paper
reports the F1 score of LLaVA-1.5 7B with each vision encoder.

The annotation files (``coco_pope_random.json`` etc.) are distributed with the
POPE repository; pass their directory via ``--data-root``, or let the loader
download the COCO val2014 images from the HuggingFace hub.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import torch

from ..utils.common import LOGGER, add_common_args, get_device, set_seed
from .metrics import pope_f1


POPE_FILES = {
    "adversarial": "coco_pope_adversarial.json",
    "popular": "coco_pope_popular.json",
    "random": "coco_pope_random.json",
}


def load_pope_split(split: str, data_root: Optional[str] = None) -> List[dict]:
    """Load one POPE split (``json`` lines with ``image``, ``text``, ``label``)."""
    filename = POPE_FILES[split]
    path = os.path.join(data_root, filename) if data_root else filename
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"POPE annotation file '{filename}' not found. Download the POPE "
            "annotations (RUCAIBox/POPE) and pass --data-root."
        )
    entries: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read().strip()
    if text.startswith("["):
        entries = json.loads(text)
    else:
        entries = [json.loads(line) for line in text.splitlines() if line.strip()]
    return entries


def _load_coco_images(image_ids: Sequence[str], data_root: Optional[str], coco_root: Optional[str]):
    """Return the COCO val2014 images for the given file names."""
    from PIL import Image

    images = []
    missing_download = []
    for name in image_ids:
        path = None
        if coco_root:
            path = os.path.join(coco_root, "val2014", name)
            if not os.path.exists(path):
                path = os.path.join(coco_root, name)
        if path is not None and os.path.exists(path):
            images.append(Image.open(path).convert("RGB"))
        else:
            images.append(None)
            missing_download.append(len(images) - 1)

    if missing_download:
        LOGGER.info("downloading %d COCO images from the HuggingFace hub", len(missing_download))
        from datasets import load_dataset

        dataset = load_dataset("HuggingFaceM4/COCO", "2014", split="val", trust_remote_code=True)
        by_name = {item["image_id"]: item["image"] for item in dataset}
        for index in missing_download:
            image = by_name.get(image_ids[index])
            images[index] = image.convert("RGB") if image is not None else None
    return images


def evaluate_pope(lvlm, entries: Sequence[dict], images: Sequence, args) -> Dict[str, float]:
    from ..lvlm.llava import build_llava_prompt

    prompts = [build_llava_prompt(entry["text"], task="pope") for entry in entries]
    labels = [entry["label"] for entry in entries]
    pixel_values = lvlm.preprocess(images).to(lvlm.device)
    predictions: List[str] = []
    for start in range(0, len(pixel_values), args.batch_size):
        chunk = pixel_values[start:start + args.batch_size]
        predictions.extend(
            lvlm.generate(chunk, prompts[start:start + args.batch_size], max_new_tokens=4)
        )
    metrics = pope_f1(predictions, labels)
    # Table 5 reports F1 in percent
    for key in ("f1", "precision", "recall", "accuracy"):
        metrics[key] = 100.0 * metrics[key]
    metrics["predictions"] = predictions
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llava-path", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--clip-arch", default="ViT-L-14")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--clip-checkpoint", default=None)
    parser.add_argument("--clip-checkpoint-key", default=None)
    parser.add_argument("--data-root", default=None, help="directory with the POPE json files")
    parser.add_argument("--coco-root", default=None, help="COCO root with a val2014 folder")
    parser.add_argument("--splits", nargs="+", default=["adversarial", "popular", "random"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16"])
    parser.add_argument("--output", default=None)
    add_common_args(parser)
    return parser


def main(argv=None) -> int:
    from ..lvlm.llava import load_llava_1p5

    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    device = get_device(args.device)
    dtype = torch.float16 if args.precision == "fp16" and device.type == "cuda" else torch.float32
    lvlm = load_llava_1p5(
        clip_checkpoint=args.clip_checkpoint,
        clip_arch=args.clip_arch,
        image_size=args.image_size,
        llava_path=args.llava_path,
        device=str(device),
        dtype=dtype,
    )

    results: Dict[str, object] = {}
    f1_scores = []
    for split in args.splits:
        entries = load_pope_split(split, args.data_root)
        if args.max_samples:
            entries = entries[:args.max_samples]
        images = _load_coco_images([e["image"] for e in entries], args.data_root, args.coco_root)
        metrics = evaluate_pope(lvlm, entries, images, args)
        metrics.pop("predictions", None)
        results[split] = metrics
        f1_scores.append(metrics["f1"])
        LOGGER.info("POPE %s: F1 %.4f", split, metrics["f1"])
    results["mean"] = sum(f1_scores) / max(1, len(f1_scores))
    print(json.dumps(results, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

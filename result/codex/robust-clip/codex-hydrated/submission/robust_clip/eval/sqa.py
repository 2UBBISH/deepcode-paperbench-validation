"""ScienceQA-I evaluation (Sec. 4.4, Table 6, App. C.2).

LLaVA-1.5 with the original CLIP encoder scores 64.5% on SQA-I; the paper shows
that the robust encoders (FARE) stay within ~1% of that number, while TeCoA
loses more.  SQA-I is the subset of ~10k image/question pairs of ScienceQA that
comes with an image (and no textual context).
"""
from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List, Optional, Sequence

import torch

from ..utils.common import LOGGER, add_common_args, get_device, set_seed


LETTERS = ["A", "B", "C", "D", "E"]


def format_sqa_question(question: str, choices: Sequence[str]) -> str:
    """The SQA prompt of LLaVA-1.5 (choices followed by the answer instruction)."""
    body = "\n".join(f"{LETTERS[i]}. {choice}" for i, choice in enumerate(choices))
    return f"{question}\n{body}\nAnswer with the option's letter from the given choices directly."


def load_sqa_i(root: Optional[str] = None, max_samples: Optional[int] = None) -> List[dict]:
    """Load the ScienceQA test split restricted to examples that have an image."""
    if root is not None:
        import os

        path = os.path.join(root, "problems.json")
        with open(path, "r", encoding="utf-8") as handle:
            problems = json.load(handle)
        entries = []
        for pid, problem in problems.items():
            if problem.get("split") != "test" or problem.get("image") is None:
                continue
            entries.append(
                {
                    "question_id": pid,
                    "question": problem["question"],
                    "choices": problem["choices"],
                    "answer": problem["answer"],
                    "image": problem["image"],
                    "image_dir": os.path.join(root, "images", "test"),
                }
            )
    else:
        from datasets import load_dataset

        dataset = load_dataset("derek-thomas/ScienceQA", split="test", trust_remote_code=True)
        entries = []
        for index, item in enumerate(dataset):
            if item.get("image") is None:
                continue
            entries.append(
                {
                    "question_id": str(index),
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": item["answer"],
                    "image": item["image"],
                }
            )
    LOGGER.info("loaded %d SQA-I (image) examples", len(entries))
    if max_samples is not None:
        entries = entries[:max_samples]
    return entries


def extract_choice(text: str) -> Optional[str]:
    """Map the model's free-form answer to an option letter."""
    match = re.search(r"\b([A-E])\b", text.strip())
    if match:
        return match.group(1)
    for letter in LETTERS:
        if text.strip().upper().startswith(letter):
            return letter
    return None


def evaluate_sqa(lvlm, entries: Sequence[dict], args) -> Dict[str, float]:
    from PIL import Image

    from ..lvlm.llava import build_llava_prompt

    images = []
    for entry in entries:
        image = entry["image"]
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif getattr(image, "mode", "RGB") != "RGB":
            image = image.convert("RGB")
        images.append(image)
    prompts = [
        build_llava_prompt(format_sqa_question(e["question"], e["choices"]), task="scienceqa")
        for e in entries
    ]
    pixel_values = lvlm.preprocess(images).to(lvlm.device)
    predictions: List[str] = []
    for start in range(0, len(pixel_values), args.batch_size):
        chunk = pixel_values[start:start + args.batch_size]
        predictions.extend(
            lvlm.generate(chunk, prompts[start:start + args.batch_size], max_new_tokens=8)
        )
    correct = 0
    for entry, prediction in zip(entries, predictions):
        letter = extract_choice(prediction)
        if letter is not None and LETTERS.index(letter) == int(entry["answer"]):
            correct += 1
    return {
        "accuracy": 100.0 * correct / max(1, len(entries)),
        "num_samples": len(entries),
        "predictions": predictions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llava-path", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--clip-arch", default="ViT-L-14")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--clip-checkpoint", default=None)
    parser.add_argument("--clip-checkpoint-key", default=None)
    parser.add_argument("--data-root", default=None)
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
    entries = load_sqa_i(args.data_root, args.max_samples)
    results = evaluate_sqa(lvlm, entries, args)
    LOGGER.info("SQA-I accuracy: %.1f", results["accuracy"])
    print(json.dumps({k: v for k, v in results.items() if k != "predictions"}, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

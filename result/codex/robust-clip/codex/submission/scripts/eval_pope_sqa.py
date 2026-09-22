#!/usr/bin/env python
"""POPE (Table 5) and SQA-I (Table 6) evaluations of LLaVA with a robust encoder."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robust_clip.eval.datasets import load_sqa  # noqa: E402
from robust_clip.eval.lvlm_eval import evaluate_pope, evaluate_sqa  # noqa: E402
from robust_clip.models.lvlm.llava_openclip import load_llava_openclip  # noqa: E402
from robust_clip.utils.misc import get_logger  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="POPE / SQA-I evaluation")
    parser.add_argument("--benchmark", choices=["pope", "sqa", "both"], default="both")
    parser.add_argument("--clip-arch", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--llava-model", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--n-pope", type=int, default=None)
    parser.add_argument("--n-sqa", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="outputs/pope_sqa.json")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("robust_clip.pope_sqa", os.path.join("outputs", "pope_sqa.log"))
    model = load_llava_openclip(
        model_path=args.llava_model,
        arch=args.clip_arch,
        pretrained=args.clip_pretrained,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    results = {}
    if args.benchmark in {"pope", "both"}:
        results["pope"] = evaluate_pope(model, n_samples=args.n_pope, batch_size=args.batch_size)
        logger.info("POPE: %s", results["pope"])
    if args.benchmark in {"sqa", "both"}:
        samples = load_sqa(max_samples=args.n_sqa)
        results["sqa"] = evaluate_sqa(model, samples, batch_size=args.batch_size)
        logger.info("SQA-I: %s", results["sqa"])
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()

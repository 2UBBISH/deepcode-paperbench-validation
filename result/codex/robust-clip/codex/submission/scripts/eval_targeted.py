#!/usr/bin/env python
"""Stealthy targeted attacks against LLaVA (Sec. 4.2, Table 3 / Figs. 3 and 5)."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robust_clip.eval.datasets import load_coco  # noqa: E402
from robust_clip.eval.lvlm_eval import evaluate_targeted_attacks  # noqa: E402
from robust_clip.models.lvlm.llava_openclip import load_llava_openclip  # noqa: E402
from robust_clip.models.lvlm.prompts import TARGET_CAPTIONS  # noqa: E402
from robust_clip.utils.misc import get_logger  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="targeted stealthy attacks (Table 3)")
    parser.add_argument("--clip-arch", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--llava-model", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--radii", nargs="*", default=["2/255", "4/255"])
    parser.add_argument("--iterations", type=int, default=10000, help="App. B.9: 10,000 for Table 3")
    parser.add_argument("--n-images", type=int, default=25, help="25 images per target caption (App. B.8)")
    parser.add_argument("--targets", nargs="*", default=TARGET_CAPTIONS)
    parser.add_argument("--output", default="outputs/targeted_attacks.json")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("robust_clip.targeted", os.path.join("outputs", "targeted.log"))
    model = load_llava_openclip(
        model_path=args.llava_model,
        arch=args.clip_arch,
        pretrained=args.clip_pretrained,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    samples = load_coco(max_samples=args.n_images)
    for eps in args.radii:
        results = evaluate_targeted_attacks(
            model,
            samples,
            targets=args.targets,
            eps=eps,
            iterations=args.iterations,
        )
        logger.info("eps=%s: %s", eps, results)
    logger.info("results written to %s", args.output)


if __name__ == "__main__":
    main()

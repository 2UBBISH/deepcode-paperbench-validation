#!/usr/bin/env python
"""Zero-shot classification + AutoAttack evaluation of CLIP models (Sec. 4.3, Table 4).

Example::

    python scripts/eval_zeroshot.py --checkpoint checkpoints/FARE4-ViT-L-14-openai.pt \
        --arch ViT-L-14 --datasets all --radii 2/255 4/255 --output results/fare4_zs.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robust_clip.eval.zeroshot import ZeroShotEvaluator, ZERO_SHOT_DATASETS  # noqa: E402
from robust_clip.models.clip_encoder import load_clip  # noqa: E402
from robust_clip.utils.misc import get_logger  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="zero-shot evaluation of CLIP (Table 4)")
    parser.add_argument("--arch", default="ViT-L-14")
    parser.add_argument("--pretrained", default="openai", help="'openai' or e.g. 'laion2b_s32b_b82k'")
    parser.add_argument("--checkpoint", default=None, help="fine-tuned (FARE / TeCoA) checkpoint")
    parser.add_argument("--datasets", nargs="*", default=["all"], help="dataset names or 'all'")
    parser.add_argument("--radii", nargs="*", default=["2/255", "4/255"])
    parser.add_argument("--n-robust-samples", type=int, default=1000)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-root", default=None, help="optional local dataset root")
    parser.add_argument("--max-clean-samples", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("robust_clip.zeroshot", os.path.join("outputs", "zeroshot.log"))
    names = list(ZERO_SHOT_DATASETS) if args.datasets == ["all"] else args.datasets
    clip = load_clip(arch=args.arch, pretrained=args.pretrained, checkpoint=args.checkpoint, device=args.device)
    evaluator = ZeroShotEvaluator(
        clip,
        batch_size=args.batch_size,
        n_robust_samples=args.n_robust_samples,
        n_iter=args.n_iter,
    )
    results = evaluator.evaluate(
        names=names,
        radii=args.radii,
        root=args.data_root,
        max_clean_samples=args.max_clean_samples,
    )
    logger.info("results: %s", json.dumps(results, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump({"config": vars(args), "results": results}, handle, indent=2)


if __name__ == "__main__":
    main()

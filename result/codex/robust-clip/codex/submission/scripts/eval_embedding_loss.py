#!/usr/bin/env python
"""Clean and adversarial embedding loss of Eqs. (4)/(5) -- Table 14 (App. C.4)."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robust_clip.eval.lvlm_eval import evaluate_embedding_loss  # noqa: E402
from robust_clip.models.clip_encoder import load_clip  # noqa: E402
from robust_clip.training.imagenet import build_imagenet_loaders  # noqa: E402
from robust_clip.utils.misc import get_logger  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="embedding loss evaluation (Table 14)")
    parser.add_argument("--arch", default="ViT-L-14")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--eps", default="4/255")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--output", default="outputs/embedding_loss.json")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("robust_clip.embedding", os.path.join("outputs", "embedding.log"))
    clip = load_clip(arch=args.arch, pretrained=args.pretrained, checkpoint=args.checkpoint, device=args.device)
    reference = load_clip(arch=args.arch, pretrained=args.pretrained, device=args.device)
    _, val_loader = build_imagenet_loaders(batch_size=args.batch_size, num_workers=2)
    results = evaluate_embedding_loss(
        clip,
        reference,
        val_loader,
        eps=args.eps,
        n_iter=args.n_iter,
        n_samples=args.n_samples,
        device=args.device,
    )
    logger.info("embedding losses: %s", results)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()

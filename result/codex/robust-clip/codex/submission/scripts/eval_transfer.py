#!/usr/bin/env python
"""Transfer attacks between LVLMs / vision encoders (Sec. 4.1, Table 2).

Adversarial COCO images are crafted against a *source* model (e.g. LLaVA with the
original CLIP encoder) and then evaluated with a *target* model that uses a
different (robust) vision encoder.  The evaluation is restricted to 200 samples.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robust_clip.eval.datasets import load_coco  # noqa: E402
from robust_clip.eval.lvlm_eval import transfer_attack  # noqa: E402
from robust_clip.models.lvlm.llava_openclip import load_llava_openclip  # noqa: E402
from robust_clip.models.lvlm.open_flamingo import load_open_flamingo  # noqa: E402
from robust_clip.utils.misc import get_logger  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="transfer attacks (Table 2)")
    parser.add_argument("--source-backend", choices=["llava", "openflamingo"], default="llava")
    parser.add_argument("--source-checkpoint", default=None, help="None -> the original CLIP encoder")
    parser.add_argument("--target-backend", choices=["llava", "openflamingo"], default="llava")
    parser.add_argument("--target-checkpoint", default=None)
    parser.add_argument("--clip-arch", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--llava-model", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--of-model", default="openflamingo/OpenFlamingo-9B-vitl-mpt7b")
    parser.add_argument("--eps", default="4/255")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--adv-images", default="outputs/transfer_adv.pt")
    parser.add_argument("--output", default="outputs/transfer.json")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build(backend, checkpoint, args):
    if backend == "llava":
        return load_llava_openclip(
            model_path=args.llava_model,
            arch=args.clip_arch,
            pretrained=args.clip_pretrained,
            checkpoint=checkpoint,
            device=args.device,
        )
    return load_open_flamingo(
        model_name=args.of_model,
        clip_arch=args.clip_arch,
        clip_pretrained=args.clip_pretrained,
        checkpoint=checkpoint,
        device=args.device,
    )


def main():
    args = parse_args()
    logger = get_logger("robust_clip.transfer", os.path.join("outputs", "transfer.log"))
    samples = load_coco(max_samples=args.n_samples)
    source = build(args.source_backend, args.source_checkpoint, args)
    target = build(args.target_backend, args.target_checkpoint, args)
    results = transfer_attack(
        source,
        target,
        samples,
        task="coco",
        eps=args.eps,
        batch_size=args.batch_size,
        source_backend=args.source_backend,
        target_backend=args.target_backend,
        adv_images_path=args.adv_images,
    )
    logger.info("transfer results: %s", results)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump({"config": vars(args), "results": results}, handle, indent=2)


if __name__ == "__main__":
    main()

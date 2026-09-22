#!/usr/bin/env python
"""Clean + adversarial LVLM evaluation of one vision encoder (Table 1 / App. C.3).

Example (LLaVA-1.5 7B with the robust FARE^4 encoder on COCO)::

    python scripts/eval_lvlm.py --backend llava --task coco \
        --checkpoint checkpoints/FARE4-ViT-L-14-openai.pt \
        --radii 2/255 4/255 --n-clean 5000 --n-adv 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robust_clip.eval.datasets import LVLM_DATASETS, sample_indices  # noqa: E402
from robust_clip.eval.lvlm_eval import evaluate_clean, evaluate_robust  # noqa: E402
from robust_clip.models.lvlm.llava_openclip import load_llava_openclip  # noqa: E402
from robust_clip.models.lvlm.open_flamingo import load_open_flamingo  # noqa: E402
from robust_clip.utils.misc import get_logger  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="LVLM robustness evaluation (Table 1)")
    parser.add_argument("--backend", choices=["llava", "openflamingo"], default="llava")
    parser.add_argument("--task", choices=["coco", "flickr30k", "vqav2", "textvqa"], required=True)
    parser.add_argument("--clip-arch", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--llava-model", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--of-model", default="openflamingo/OpenFlamingo-9B-vitl-mpt7b")
    parser.add_argument("--radii", nargs="*", default=["2/255", "4/255"])
    parser.add_argument("--n-clean", type=int, default=None, help="None = all samples")
    parser.add_argument("--n-adv", type=int, default=500, help="the paper uses 500 images for attacks")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", default=None)
    parser.add_argument("--save-adv-images", default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_model(args):
    if args.backend == "llava":
        return load_llava_openclip(
            model_path=args.llava_model,
            arch=args.clip_arch,
            pretrained=args.clip_pretrained,
            checkpoint=args.checkpoint,
            device=args.device,
        )
    return load_open_flamingo(
        model_name=args.of_model,
        clip_arch=args.clip_arch,
        clip_pretrained=args.clip_pretrained,
        checkpoint=args.checkpoint,
        device=args.device,
    )


def main():
    args = parse_args()
    logger = get_logger("robust_clip.lvlm", os.path.join("outputs", "lvlm_eval.log"))
    loader = LVLM_DATASETS[args.task]
    samples = loader(max_samples=args.n_clean)
    model = build_model(args)

    results = {"task": args.task, "encoder": args.checkpoint or args.clip_pretrained}
    results["clean"] = evaluate_clean(
        model, samples, args.task, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        backend=args.backend,
    )
    logger.info("clean: %s", results["clean"])

    adv_samples = [samples[i] for i in sample_indices(len(samples), args.n_adv, seed=0)]
    for eps in args.radii:
        key = f"robust_{eps}"
        results[key] = evaluate_robust(
            model,
            adv_samples,
            args.task,
            eps=eps,
            batch_size=max(1, args.batch_size // 2),
            max_new_tokens=args.max_new_tokens,
            backend=args.backend,
            save_adv_images=args.save_adv_images,
        )
        results[key].pop("scores", None)
        logger.info("%s: %s", key, results[key])

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()

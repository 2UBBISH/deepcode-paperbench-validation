#!/usr/bin/env python
"""Jailbreaking attacks against LLaVA (Sec. 4.4, Table 7).

Two steps:

1. ``--mode attack``: craft the universal adversarial image with the attack of
   Qi et al. (2023) -- 5000 iterations, alpha = 1/255, no momentum, one image.
2. ``--mode evaluate``: query the model with the 40 harmful prompts and the
   adversarial / clean image and dump the answers for the human evaluation.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robust_clip.eval.jailbreak import (  # noqa: E402
    JailbreakConfig,
    UniversalJailbreakAttack,
    download_harmful_corpus,
    evaluate_jailbreak,
    load_target_strings,
)
from robust_clip.eval.lvlm_eval import image_to_tensor  # noqa: E402
from robust_clip.models.lvlm.llava_openclip import load_llava_openclip  # noqa: E402
from robust_clip.utils.misc import get_logger  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="jailbreaking attacks (Table 7)")
    parser.add_argument("--mode", choices=["attack", "evaluate", "both"], default="both")
    parser.add_argument("--clip-arch", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--llava-model", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--eps", default="64/255", help="attack strength (Table 7: 16, 32 and 64/255)")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--alpha", default="1/255")
    parser.add_argument("--data-dir", default="data/harmful_corpus")
    parser.add_argument("--adv-image", default="outputs/adv_jailbreak.pt")
    parser.add_argument("--output-csv", default="outputs/jailbreak_eval.csv")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("robust_clip.jailbreak", os.path.join("outputs", "jailbreak.log"))
    _, _, clean_path = download_harmful_corpus(args.data_dir)
    model = load_llava_openclip(
        model_path=args.llava_model,
        arch=args.clip_arch,
        pretrained=args.clip_pretrained,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    clean_image = image_to_tensor(__import__("PIL.Image", fromlist=["Image"]).open(clean_path))[None]

    adversarial = None
    if args.mode in {"attack", "both"}:
        targets = load_target_strings(data_dir=args.data_dir)
        attack = UniversalJailbreakAttack(
            model,
            targets,
            config=JailbreakConfig(eps=args.eps, alpha=args.alpha, iterations=args.iterations),
        )
        adversarial, loss = attack.run(clean_image)
        os.makedirs(os.path.dirname(os.path.abspath(args.adv_image)), exist_ok=True)
        torch.save({"adv_image": adversarial, "clean_image": clean_image, "eps": args.eps, "loss": float(loss.mean())}, args.adv_image)
        logger.info("adversarial image saved to %s (final loss %.4f)", args.adv_image, float(loss.mean()))
    elif os.path.isfile(args.adv_image):
        adversarial = torch.load(args.adv_image)["adv_image"]

    result = evaluate_jailbreak(
        model,
        clean_image,
        adversarial_image=adversarial,
        data_dir=args.data_dir,
        output_csv=args.output_csv,
    )
    logger.info("wrote %s answers for the human evaluation to %s", result["prompts"], result["csv"])


if __name__ == "__main__":
    main()

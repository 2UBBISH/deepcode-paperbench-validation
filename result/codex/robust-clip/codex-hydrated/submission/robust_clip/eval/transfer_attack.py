"""Transfer attacks between LVLMs (Table 2 of the paper).

Adversarial images are generated **white-box against a surrogate** (an LVLM that
uses the original, non-robust CLIP encoder) and then evaluated on the *same*
LVLM architecture with a different vision encoder.  This is the realistic
threat model in which an adversary has white-box access to a surrogate but only
black-box access to the target, and it is also the setting in which the vision
encoder -- not the language model -- is shown to be the source of the
vulnerability.

The images are produced with the same pipeline as the white-box evaluation
(:class:`~robust_clip.attacks.lvlm_attack.LVLMAttackPipeline`) and can be cached
on disk, so that they are attacked once and evaluated on any number of models::

    # 1. attack the surrogate and store the adversarial images
    python -m robust_clip.eval.transfer_attack --generate-only \\
        --save-images results/transfer/coco_clip

    # 2. evaluate the robustness of the other encoders on those images
    python -m robust_clip.eval.transfer_attack --reuse-images results/transfer/coco_clip \\
        --clip-checkpoint runs/robust_clip/FARE4-CLIP_ViT-L-14.pt
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Sequence

import torch

from ..attacks.lvlm_attack import LVLMAttackPipeline
from ..utils.common import LOGGER, add_common_args, ensure_dir, get_device, parse_epsilon, set_seed
from .cider import CiderScorer
from .lvlm_data import load_captioning_dataset
from .metrics import vqa_accuracy


def _load_images(directory: str) -> Dict[str, torch.Tensor]:
    images = {}
    for name in sorted(os.listdir(directory)):
        if name.endswith(".pt"):
            images[name[:-3]] = torch.load(os.path.join(directory, name), map_location="cpu")
    return images


def _save_images(directory: str, images: Dict[str, torch.Tensor]) -> None:
    ensure_dir(directory)
    for key, tensor in images.items():
        torch.save(tensor, os.path.join(directory, f"{key}.pt"))


def generate_transfer_set(lvlm, args) -> Dict[str, torch.Tensor]:
    """Attack the surrogate and return ``{eps: adversarial_images}``."""
    from ..lvlm.llava import build_llava_prompt

    examples = load_captioning_dataset(
        args.dataset, root=args.data_root, split="val" if args.dataset == "coco" else "test",
        max_samples=args.max_samples,
    )
    n = min(args.num_attack_samples, len(examples))
    indices = random.Random(args.seed).sample(range(len(examples)), n)
    chosen = [examples[i] for i in indices]
    prompts = [build_llava_prompt("Describe the image concisely.", task="coco_caption")] * n
    references = [example.captions for example in chosen]
    images = lvlm.preprocess([example.image for example in chosen]).to(lvlm.device)
    scorer = CiderScorer(scale=args.cider_scale).prepare(references)

    adversarial: Dict[str, torch.Tensor] = {"clean": images.cpu()}
    for eps_text in args.eps_list:
        eps = parse_epsilon(eps_text)
        pipeline = LVLMAttackPipeline(
            lvlm,
            eps=eps,
            n_iter=args.attack_iterations,
            alpha=eps,
            momentum=args.attack_momentum,
            max_new_tokens=args.max_new_tokens,
            use_half_precision_stage=not args.no_half_precision,
            use_single_precision_stage=not args.no_single_precision,
        )
        out = pipeline.attack_captioning(
            images, prompts, references, scorer.compute_scores,
            threshold=args.coco_threshold if args.dataset == "coco" else args.flickr_threshold,
        )
        adversarial[eps_text.replace("/", "_")] = out.images.cpu()
        LOGGER.info("surrogate CIDEr @ %s: %.2f", eps_text, float(out.scores.mean()))
    return adversarial


def evaluate_transfer_set(lvlm, adversarial: Dict[str, torch.Tensor], args) -> Dict[str, float]:
    from ..lvlm.llava import build_llava_prompt

    examples = load_captioning_dataset(
        args.dataset, root=args.data_root, split="val" if args.dataset == "coco" else "test",
        max_samples=args.max_samples,
    )
    n = next(iter(adversarial.values())).shape[0]
    indices = random.Random(args.seed).sample(range(len(examples)), n)
    chosen = [examples[i] for i in indices]
    prompts = [build_llava_prompt("Describe the image concisely.", task="coco_caption")] * n
    references = [example.captions for example in chosen]
    scorer = CiderScorer(scale=args.cider_scale).prepare(references)

    results = {}
    for key, images in adversarial.items():
        captions = lvlm.generate(images.to(lvlm.device), prompts, max_new_tokens=args.max_new_tokens)
        scores = scorer.compute_scores(captions, references)
        results[key] = sum(scores) / max(1, len(scores))
        LOGGER.info("transfer target CIDEr (%s): %.2f", key, results[key])
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="llava", choices=["llava", "openflamingo"])
    parser.add_argument("--llava-path", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--clip-arch", default="ViT-L-14")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--clip-checkpoint", default=None,
        help="vision encoder of the *evaluated* model (None = original CLIP)",
    )
    parser.add_argument(
        "--source-checkpoint", default=None,
        help="vision encoder of the surrogate used to craft the attacks "
             "(None = original CLIP, as in Table 2)",
    )
    parser.add_argument("--dataset", default="coco", choices=["coco", "flickr30k"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-attack-samples", type=int, default=500)
    parser.add_argument("--eps-list", nargs="+", default=["2/255", "4/255"])
    parser.add_argument("--attack-iterations", type=int, default=100)
    parser.add_argument("--attack-momentum", type=float, default=0.9)
    parser.add_argument("--no-half-precision", action="store_true")
    parser.add_argument("--no-single-precision", action="store_true")
    parser.add_argument("--coco-threshold", type=float, default=10.0)
    parser.add_argument("--flickr-threshold", type=float, default=2.0)
    parser.add_argument(
        "--cider-scale", type=float, default=1000.0,
        help="reporting scale; 1000 = percent, matching the paper's tables",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--save-images", default=None)
    parser.add_argument("--reuse-images", default=None)
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16"])
    parser.add_argument("--output", default=None)
    add_common_args(parser)
    return parser


def main(argv=None) -> int:
    from ..lvlm.factory import build_lvlm

    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    device = get_device(args.device)
    args.device = "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
    source_checkpoint = args.source_checkpoint
    target_checkpoint = args.clip_checkpoint

    # ------------------------------------------------------------ surrogate --
    if args.reuse_images:
        adversarial = _load_images(args.reuse_images)
        LOGGER.info("loaded %d cached transfer sets from %s", len(adversarial), args.reuse_images)
    else:
        args.clip_checkpoint = source_checkpoint
        surrogate = build_lvlm(args)
        adversarial = generate_transfer_set(surrogate, args)
        if args.save_images:
            _save_images(args.save_images, adversarial)
            LOGGER.info("saved the transfer sets to %s", args.save_images)
        if args.generate_only:
            return 0
        del surrogate

    # --------------------------------------------------------------- target --
    args.clip_checkpoint = target_checkpoint
    target = build_lvlm(args)
    results = evaluate_transfer_set(target, adversarial, args)
    print(json.dumps(results, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

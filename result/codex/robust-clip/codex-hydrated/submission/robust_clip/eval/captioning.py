"""Captioning evaluation of the LVLMs (COCO, Flickr30k) -- Table 1 and Sec. 4.1.

Clean evaluation uses **all** available samples, the adversarial evaluation uses
500 randomly sampled images (as in the paper).  The metric is CIDEr
(:mod:`robust_clip.eval.cider`).

The adversarial images are produced by the ensemble attack of
:class:`~robust_clip.attacks.lvlm_attack.LVLMAttackPipeline`:
half-precision APGD (100 iterations, 16-bit perturbations) against each of the
five ground-truth captions, then a single-precision APGD (32-bit perturbations)
against the ground truth that gave the worst CIDEr score.
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List, Optional, Sequence

import torch

from ..attacks.lvlm_attack import LVLMAttackPipeline
from ..utils.common import LOGGER, add_common_args, get_device, parse_epsilon, set_seed
from .cider import CiderScorer
from .lvlm_data import CaptionExample, load_captioning_dataset


CAPTION_QUESTION = "Describe the image concisely."


def captioning_prompts(examples: Sequence[CaptionExample], builder, task: str) -> List[str]:
    return [builder(CAPTION_QUESTION, task=task) for _ in examples]


def prompt_builder_for(args):
    """LLaVA and OpenFlamingo use different prompt templates (Sec. 4.1)."""
    if getattr(args, "backend", "llava") == "llava":
        from ..lvlm.llava import build_llava_prompt

        return build_llava_prompt
    from ..lvlm.openflamingo import build_openflamingo_prompt

    return build_openflamingo_prompt


def score_captions_with(scorer: CiderScorer):
    def score_fn(candidates: Sequence[str], references: Sequence[Sequence[str]]) -> List[float]:
        return scorer.compute_scores(candidates, references)

    return score_fn


def evaluate_captioning(lvlm, examples: Sequence[CaptionExample], args, task: str) -> Dict[str, object]:
    prompts = captioning_prompts(examples, prompt_builder_for(args), task)
    references = [example.captions for example in examples]
    scorer = CiderScorer(scale=args.cider_scale).prepare(references, df_file=args.cider_df)
    score_fn = score_captions_with(scorer)

    result: Dict[str, object] = {}

    # ---------------------------------------------------------------- clean --
    if args.clean:
        images = lvlm.preprocess([example.image for example in examples]).to(lvlm.device)
        captions: List[str] = []
        self_batch = args.eval_batch_size
        for start in range(0, len(images), self_batch):
            chunk = images[start:start + self_batch]
            captions.extend(
                lvlm.generate(chunk, prompts[start:start + self_batch], max_new_tokens=args.max_new_tokens)
            )
        clean_scores = score_fn(captions, references)
        result["clean_cider"] = sum(clean_scores) / max(1, len(clean_scores))
        result["clean_num_samples"] = len(clean_scores)
        LOGGER.info("clean CIDEr: %.2f (%d samples)", result["clean_cider"], len(clean_scores))

    # ------------------------------------------------------------ adversarial --
    if args.attack:
        n = min(args.num_attack_samples, len(examples))
        indices = random.Random(args.seed).sample(range(len(examples)), n)
        adv_examples = [examples[i] for i in indices]
        adv_prompts = [prompts[i] for i in indices]
        adv_refs = [references[i] for i in indices]
        images = lvlm.preprocess([example.image for example in adv_examples]).to(lvlm.device)

        for eps_text in args.eps_list:
            eps = parse_epsilon(eps_text)
            pipeline = LVLMAttackPipeline(
                lvlm,
                eps=eps,
                n_iter=args.attack_iterations,
                alpha=parse_epsilon(args.attack_alpha) if args.attack_alpha else eps,
                momentum=args.attack_momentum,
                grad_normalization=args.grad_normalization,
                max_new_tokens=args.max_new_tokens,
                use_half_precision_stage=not args.no_half_precision,
                use_single_precision_stage=not args.no_single_precision,
            )
            threshold = args.coco_threshold if task.startswith("coco") else args.flickr_threshold
            out = pipeline.attack_captioning(
                images, adv_prompts, adv_refs, score_fn, threshold=threshold
            )
            result[f"robust_cider_{eps_text.replace('/', '_')}"] = float(out.scores.mean())
            result[f"robust_outputs_{eps_text.replace('/', '_')}"] = out.outputs
            result[f"perturbation_linf_{eps_text.replace('/', '_')}"] = float(
                (out.images - images).abs().max()
            )
            LOGGER.info(
                "adversarial CIDEr @ %s: %.2f", eps_text, float(out.scores.mean())
            )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llava-path", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--backend", default="llava", choices=["llava", "openflamingo"])
    parser.add_argument("--clip-arch", default="ViT-L-14")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--clip-checkpoint", default=None, help="FARE / TeCoA checkpoint")
    parser.add_argument("--clip-checkpoint-key", default=None)
    parser.add_argument("--dataset", default="coco", choices=["coco", "flickr30k"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="limit the clean evaluation")
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attack", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eps-list", nargs="+", default=["2/255", "4/255"])
    parser.add_argument("--num-attack-samples", type=int, default=500)
    parser.add_argument("--attack-iterations", type=int, default=100)
    parser.add_argument("--attack-alpha", default=None, help="initial step size (default: eps)")
    parser.add_argument("--attack-momentum", type=float, default=0.9)
    parser.add_argument("--grad-normalization", default="elementwise_sign")
    parser.add_argument("--no-half-precision", action="store_true")
    parser.add_argument("--no-single-precision", action="store_true")
    parser.add_argument("--coco-threshold", type=float, default=10.0)
    parser.add_argument("--flickr-threshold", type=float, default=2.0)
    parser.add_argument("--cider-df", default=None, help="document frequencies for CIDEr")
    parser.add_argument(
        "--cider-scale", type=float, default=1000.0,
        help="reporting scale; 1000 = percent, matching the paper's tables "
             "(scale=10 reproduces pycocoevalcap exactly)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=8)
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
    lvlm = build_lvlm(args)
    examples = load_captioning_dataset(
        args.dataset, root=args.data_root, split="val" if args.dataset == "coco" else "test",
        max_samples=args.max_samples,
    )
    task = "coco_caption" if args.dataset == "coco" else "flickr_caption"
    results = evaluate_captioning(lvlm, examples, args, task)
    print(json.dumps({k: v for k, v in results.items() if not isinstance(v, list)}, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

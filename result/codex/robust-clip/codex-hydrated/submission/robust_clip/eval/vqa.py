"""Visual question answering evaluation (VQAv2, TextVQA) -- Table 1, Sec. 4.1.

The adversarial evaluation follows App. B.6:

* APGD at half precision (100 iterations) against the five most frequent ground
  truth answers, skipping samples whose VQA score already dropped to 0;
* a single-precision APGD against the answer that gave the worst score;
* targeted single-precision attacks with the strings ``"maybe"`` (lower case)
  and ``"Word"`` (capitalised).  The ``"Word"`` attack is *not* applied to
  TextVQA.
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List, Sequence

import torch

from ..attacks.lvlm_attack import LVLMAttackPipeline
from ..utils.common import LOGGER, add_common_args, get_device, parse_epsilon, set_seed
from .lvlm_data import VQAExample, load_vqa_dataset
from .metrics import vqa_accuracy


def vqa_prompts(examples: Sequence[VQAExample], builder, task: str) -> List[str]:
    return [builder(example.question, task=task) for example in examples]


def prompt_builder_for(args):
    if getattr(args, "backend", "llava") == "llava":
        from ..lvlm.llava import build_llava_prompt

        return build_llava_prompt
    from ..lvlm.openflamingo import build_openflamingo_prompt

    return build_openflamingo_prompt


def score_vqa(candidates: Sequence[str], answers: Sequence[Sequence[str]]) -> List[float]:
    return vqa_accuracy(candidates, answers)


def evaluate_vqa(lvlm, examples: Sequence[VQAExample], args, task: str) -> Dict[str, object]:
    prompts = vqa_prompts(examples, prompt_builder_for(args), task)
    all_answers = [example.answers for example in examples]
    result: Dict[str, object] = {}

    if args.clean:
        images = lvlm.preprocess([example.image for example in examples]).to(lvlm.device)
        predictions: List[str] = []
        for start in range(0, len(images), args.eval_batch_size):
            chunk = images[start:start + args.eval_batch_size]
            predictions.extend(
                lvlm.generate(chunk, prompts[start:start + args.eval_batch_size],
                              max_new_tokens=args.max_new_tokens)
            )
        scores = score_vqa(predictions, all_answers)
        result["clean_accuracy"] = 100.0 * sum(scores) / max(1, len(scores))
        LOGGER.info("clean VQA accuracy: %.1f", result["clean_accuracy"])

    if args.attack:
        n = min(args.num_attack_samples, len(examples))
        indices = random.Random(args.seed).sample(range(len(examples)), n)
        adv_examples = [examples[i] for i in indices]
        adv_prompts = [prompts[i] for i in indices]
        adv_answers = [all_answers[i] for i in indices]
        attack_answers = [
            example.most_frequent_answers if len(example.answers) > 5 else example.answers
            for example in adv_examples
        ]
        images = lvlm.preprocess([example.image for example in adv_examples]).to(lvlm.device)

        targeted = ["maybe", "Word"] if task != "textvqa" else ["maybe"]
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
            out = pipeline.attack_vqa(
                images,
                adv_prompts,
                attack_answers,
                score_fn=lambda preds, _answers: vqa_accuracy(preds, adv_answers),
                threshold=0.0,
                max_answers=5,
                targeted_strings=targeted,
            )
            result[f"robust_accuracy_{eps_text.replace('/', '_')}"] = (
                100.0 * float(out.scores.mean())
            )
            result[f"robust_outputs_{eps_text.replace('/', '_')}"] = out.outputs
            LOGGER.info(
                "adversarial VQA accuracy @ %s: %.1f", eps_text, 100.0 * float(out.scores.mean())
            )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llava-path", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--backend", default="llava", choices=["llava", "openflamingo"])
    parser.add_argument("--clip-arch", default="ViT-L-14")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--clip-checkpoint", default=None)
    parser.add_argument("--clip-checkpoint-key", default=None)
    parser.add_argument("--dataset", default="vqav2", choices=["vqav2", "textvqa"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attack", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eps-list", nargs="+", default=["2/255", "4/255"])
    parser.add_argument("--num-attack-samples", type=int, default=500)
    parser.add_argument("--attack-iterations", type=int, default=100)
    parser.add_argument("--attack-alpha", default=None)
    parser.add_argument("--attack-momentum", type=float, default=0.9)
    parser.add_argument("--grad-normalization", default="elementwise_sign")
    parser.add_argument("--no-half-precision", action="store_true")
    parser.add_argument("--no-single-precision", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)
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
    examples = load_vqa_dataset(args.dataset, root=args.data_root, max_samples=args.max_samples)
    results = evaluate_vqa(lvlm, examples, args, task=args.dataset)
    print(json.dumps({k: v for k, v in results.items() if not isinstance(v, list)}, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as json_file:
            json.dump(results, json_file, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

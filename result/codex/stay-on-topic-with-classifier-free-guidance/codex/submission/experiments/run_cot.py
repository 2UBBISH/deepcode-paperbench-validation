#!/usr/bin/env python3
"""Section 3.2 -- Chain-of-Thought prompting with CFG.

Runs GSM8K and AQuA with the few-shot CoT prompts of Wang et al. (2023) on
WizardLM-30B and Guanaco-65B, sweeping the guidance strength, and records
both accuracy and the fraction of valid (parsable) reasoning chains -- the
two curves of Figure 2 (GSM8K) and Figure 17 (AQuA).

Note: these checkpoints are 30B/65B parameters.  The script is written to run
on a GPU box; ``--limit`` keeps a smoke test small.

Example
-------
    python experiments/run_cot.py --datasets gsm8k aqua \
        --models wizardlm-30b guanaco-65b \
        --gammas 1.0 1.1 1.25 1.5 1.75 2.0 \
        --output results/cot.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfglm.cot import evaluate_cot, evaluate_cot_self_consistency  # noqa: E402
from cfglm.models import resolve_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["wizardlm-30b", "guanaco-65b"])
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "aqua"])
    parser.add_argument("--gammas", nargs="+", type=float, default=[1.0, 1.1, 1.25, 1.5, 1.75, 2.0])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sample", action="store_true", help="sample (temperature 0.7) instead of greedy")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--self-consistency", type=int, default=0,
                        help="number of sampled reasoning paths for majority voting "
                             "(contribution 3: CFG stacks with CoT + self-consistency)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16" if torch.cuda.is_available() else "float32")
    parser.add_argument("--output", default="results/cot.json")
    args = parser.parse_args()

    torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    records: List[dict] = []

    for model_key in args.models:
        spec = resolve_model(model_key)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(spec.hf_name, torch_dtype=torch_dtype).to(args.device)
        model.eval()

        for dataset in args.datasets:
            for gamma in args.gammas:
                if args.self_consistency and args.self_consistency > 1:
                    result = evaluate_cot_self_consistency(
                        model,
                        tokenizer,
                        dataset,
                        gamma=gamma,
                        n_paths=args.self_consistency,
                        limit=args.limit,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                else:
                    result = evaluate_cot(
                        model,
                        tokenizer,
                        dataset,
                        gamma=gamma,
                        limit=args.limit,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.sample,
                        temperature=args.temperature,
                    )
                record = {
                    "model": spec.key,
                    "hf_name": spec.hf_name,
                    "dataset": dataset,
                    "gamma": gamma,
                    "n": result.n,
                    "accuracy": result.accuracy,
                    "valid_fraction": result.valid_fraction,
                    "n_correct": result.n_correct,
                    "n_valid": result.n_valid,
                }
                records.append(record)
                print(
                    f"[cot] {spec.key} | {dataset} | gamma={gamma} "
                    f"| acc={result.accuracy:.4f} | valid={result.valid_fraction:.4f}"
                )
                with open(args.output, "w") as fh:
                    json.dump(records, fh, indent=2)

    print(f"\nWrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Section 3.3.1 -- CodeGen on HumanEval with CFG.

Samples ``k`` completions per HumanEval problem for each guidance strength
and temperature, executes the unit tests and reports the unbiased pass@k
estimator (Table 2 for temperature 0.2; Tables 7-9 in the appendix cover
temperatures 0.2/0.6/0.8).  It also stores the per-problem number of passing
samples, which yields the CFG win / tie / loss counts of Figure 3.

Example
-------
    python experiments/run_humaneval.py \
        --models codegen-350m-mono codegen-2b-mono codegen-6b-mono \
        --gammas 1.0 1.1 1.25 1.5 1.75 2.0 \
        --temperatures 0.2 0.6 0.8 --k 100 \
        --output results/humaneval.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfglm.humaneval import (  # noqa: E402
    evaluate_samples,
    load_humaneval,
    sample_solutions,
    win_tie_loss_counts,
)
from cfglm.models import resolve_model  # noqa: E402

DEFAULT_GAMMAS = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0]
DEFAULT_TEMPERATURES = [0.2, 0.6, 0.8]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+",
                        default=["codegen-350m-mono", "codegen-2b-mono", "codegen-6b-mono"])
    parser.add_argument("--gammas", nargs="+", type=float, default=DEFAULT_GAMMAS)
    parser.add_argument("--temperatures", nargs="+", type=float, default=DEFAULT_TEMPERATURES)
    parser.add_argument("--k", type=int, default=100, help="number of samples per problem")
    parser.add_argument("--pass-k", nargs="+", type=int, default=[1, 10, 100])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N problems")
    parser.add_argument("--num-problems", type=int, default=None, help="alias of --limit")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--execution", default="inline", choices=["inline", "subprocess"],
                        help="inline = OpenAI human-eval style in-process exec (fast, default); "
                             "subprocess = one interpreter per sample (isolated, slow)")
    parser.add_argument("--humaneval-jsonl", default=None,
                        help="optional local HumanEval JSONL (offline fallback)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16" if torch.cuda.is_available() else "float32")
    parser.add_argument("--output", default="results/humaneval.json")
    args = parser.parse_args()

    limit = args.limit if args.limit is not None else args.num_problems
    problems = load_humaneval(local_path=args.humaneval_jsonl)
    if limit:
        problems = problems[:limit]
    print(f"Loaded {len(problems)} HumanEval problems")

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

        baseline_per_problem: Dict[str, Dict[str, int]] = {}
        for temperature in args.temperatures:
            for gamma in args.gammas:
                samples = {}
                for i, problem in enumerate(problems):
                    samples[problem.task_id] = sample_solutions(
                        model,
                        tokenizer,
                        problem,
                        gamma=gamma,
                        n_samples=args.k,
                        temperature=temperature,
                        max_new_tokens=args.max_new_tokens,
                        seed=1234 + i,
                    )
                metrics = evaluate_samples(
                    problems, samples, ks=args.pass_k, timeout=args.timeout, execution=args.execution
                )
                record = {
                    "model": spec.key,
                    "hf_name": spec.hf_name,
                    "temperature": temperature,
                    "gamma": gamma,
                    "pass@k": metrics["pass@k"],
                    "per_problem": metrics["per_problem"],
                    "k_samples": args.k,
                }
                if gamma == 1.0:
                    baseline_per_problem[str(temperature)] = metrics["per_problem"]
                elif str(temperature) in baseline_per_problem:
                    record["win_tie_loss"] = win_tie_loss_counts(
                        baseline_per_problem[str(temperature)], metrics["per_problem"]
                    )
                records.append(record)
                print(
                    f"[humaneval] {spec.key} T={temperature} gamma={gamma} "
                    f"pass@1={metrics['pass@k'].get(1):.4f} "
                    f"pass@10={metrics['pass@k'].get(10):.4f} "
                    f"pass@100={metrics['pass@k'].get(100):.4f}"
                )
                with open(args.output, "w") as fh:
                    json.dump(records, fh, indent=2)

    print(f"\nWrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Table 4 -- CFG vs FUDGE for sentiment and toxicity control.

Prompts GPT-2 with ``"That was a good movie!"`` (IMDB sentiment) and
``"Don't be mean"`` (Jigsaw toxicity), steers the generations with either
FUDGE (an external classifier, ``distilbert-base-uncased-emotion`` /
``unitary/toxic-bert``) or CFG (the language model itself, Equation 7), and
reports the percent increase in the desired label's probability as judged by
a *secondary* classifier (``stevhliu/my_awesome_model`` for sentiment,
``unitary/toxic-bert`` for toxicity) -- the quantity of Table 4.

Both the FUDGE coefficient ``lam`` and the CFG strength ``gamma`` are swept,
as the paper tunes both to maximise the score while maintaining fluency.

Example
-------
    python experiments/run_fudge_comparison.py --gpt2 gpt2 \
        --gammas 1 2 3 4 5 6 --lambdas 0.5 1 2 5 --output results/table4.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfglm.external_classifiers import (  # noqa: E402
    SENTIMENT_EVAL,
    SENTIMENT_GUIDANCE,
    TOXICITY,
    TextClassifier,
    percent_increase,
)
from cfglm.fudge import fudge_generate  # noqa: E402
from cfglm.generation import cfg_generate  # noqa: E402

TASKS = {
    "sentiment": {
        "prompt": "That was a good movie!",
        "guidance": SENTIMENT_GUIDANCE,
        "evaluation": SENTIMENT_EVAL,
    },
    "toxicity": {
        "prompt": "Don't be mean",
        "guidance": TOXICITY,
        "evaluation": TOXICITY,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpt2", default="gpt2")
    parser.add_argument("--tasks", nargs="+", default=["sentiment", "toxicity"])
    parser.add_argument("--gammas", nargs="+", type=float, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/table4.json")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.gpt2)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.gpt2).to(args.device).eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    results: Dict[str, dict] = {}

    for task_name in args.tasks:
        spec = TASKS[task_name]
        prompt = spec["prompt"]
        guidance = TextClassifier(spec["guidance"], device=args.device)
        evaluator = TextClassifier(spec["evaluation"], device=args.device)

        baseline_texts = cfg_generate(
            model, tokenizer, prompt, gamma=1.0, max_new_tokens=args.max_new_tokens,
            do_sample=True, temperature=0.7, num_return_sequences=args.n_samples, seed=0,
        )
        baseline_scores = evaluator.probability([prompt + t for t in baseline_texts])

        cfg_scores = {}
        for gamma in args.gammas:
            texts = cfg_generate(
                model, tokenizer, prompt, gamma=gamma, max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=0.7, num_return_sequences=args.n_samples, seed=0,
            )
            scores = evaluator.probability([prompt + t for t in texts])
            cfg_scores[gamma] = percent_increase(baseline_scores, scores)

        fudge_scores = {}
        for lam in args.lambdas:
            texts = [
                fudge_generate(
                    model, tokenizer, prompt, guidance.probability, lam=lam,
                    max_new_tokens=args.max_new_tokens, temperature=0.7, seed=i,
                )
                for i in range(args.n_samples)
            ]
            scores = evaluator.probability([prompt + t for t in texts])
            fudge_scores[lam] = percent_increase(baseline_scores, scores)

        best_gamma = max(cfg_scores, key=cfg_scores.get)
        best_lam = max(fudge_scores, key=fudge_scores.get)
        results[task_name] = {
            "prompt": prompt,
            "cfg": {"best_gamma": best_gamma, "percent_increase": cfg_scores[best_gamma], "sweep": cfg_scores},
            "fudge": {"best_lambda": best_lam, "percent_increase": fudge_scores[best_lam], "sweep": fudge_scores},
        }
        print(
            f"[table4] {task_name}: CFG (gamma={best_gamma}) {cfg_scores[best_gamma]:.3f} "
            f"vs FUDGE (lambda={best_lam}) {fudge_scores[best_lam]:.3f}"
        )

    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()

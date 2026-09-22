#!/usr/bin/env python3
"""Section 5 -- explaining the success of CFG (Falcon-7b on P3).

Reproduces the interpretability analyses of Section 5:

* **5.1** mean sampling entropy of ``P(y|x)``, ``P(x)``, ``CFG`` and the
  instruction-tuned model, plus the number of tokens inside the top-p = 90 %
  mass (the paper reports 4.7 for CFG vs 5.49 for vanilla);
* **5.2** top-p overlap between CFG and instruction tuning, the per-dataset
  similarity table (Table 12) and the examples of highest / lowest overlap
  (Tables 13 and 14), together with the Spearman correlations (``r_s > .7``
  for longer prompts) and the correlation matrix of the continuation
  perplexities (Figure 5);
* **5.3** the vocabulary ranking under guidance (Table 3) is produced by
  ``experiments/run_visualize_logits.py``.

Example
-------
    python experiments/run_section5.py --limit 2000 \
        --model tiiuae/falcon-7b --instruct-model tiiuae/falcon-7b-instruct \
        --output-dir results/section5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfglm.distributions import (  # noqa: E402
    continuation_loglikelihoods,
    distribution_for_pair,
    summarize_distributions,
)
from cfglm.p3 import P3Sampler  # noqa: E402
from cfglm.stats import pearson_correlation, spearman_correlation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="tiiuae/falcon-7b")
    parser.add_argument("--instruct-model", default="tiiuae/falcon-7b-instruct")
    parser.add_argument("--no-instruct", action="store_true", help="skip the instruction-tuned comparison")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--limit", type=int, default=None, help="number of P3 datapoints to use")
    parser.add_argument("--n-per-dataset", type=int, default=50,
                        help="datapoints sampled per P3 subset (the paper uses ~50 x 660)")
    parser.add_argument("--max-input-tokens", type=int, default=200)
    parser.add_argument("--output-dir", default="results/section5")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16" if torch.cuda.is_available() else "float32")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype).to(args.device).eval()
    instruct_model = None
    if not args.no_instruct:
        instruct_model = (
            AutoModelForCausalLM.from_pretrained(args.instruct_model, torch_dtype=torch_dtype)
            .to(args.device)
            .eval()
        )

    sampler = P3Sampler(n_per_dataset=args.n_per_dataset, max_input_tokens=args.max_input_tokens)
    examples = sampler.sample()
    if args.limit:
        examples = examples[: args.limit]
    print(f"Sampled {len(examples)} P3 datapoints (paper: 32,902)")

    per_example: List[dict] = []
    for i, example in enumerate(examples):
        if not example.target_text.strip():
            continue
        dist = distribution_for_pair(
            model,
            tokenizer,
            example.input_text,
            example.target_text,
            gamma=args.gamma,
            instruct_model=instruct_model,
            max_prompt_tokens=args.max_input_tokens,
        )
        summary = summarize_distributions(dist)
        cont_ids = tokenizer(example.target_text, add_special_tokens=False).input_ids
        lls = continuation_loglikelihoods(dist, cont_ids)
        row = {
            "config": example.config,
            "dataset": example.dataset_label,
            "index": example.index,
            "n_tokens": dist.n_tokens,
            **summary,
        }
        for name, value in lls.items():
            row[f"loglik_{name}"] = value
            row[f"ppl_{name}"] = float(np.exp(-value / max(dist.n_tokens, 1)))
        per_example.append(row)
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(examples)}")

    _write_analysis(per_example, sampler, args)


def _write_analysis(rows: List[dict], sampler: P3Sampler, args) -> None:
    out = args.output_dir

    # --- Section 5.1: entropy and top-p sizes -------------------------
    entropy_summary = {}
    for key in ("entropy_prompted", "entropy_unprompted", "entropy_cfg", "entropy_instruct"):
        values = [r[key] for r in rows if key in r]
        if values:
            entropy_summary[key] = float(np.mean(values))
    topp_summary = {}
    for key in ("topp_size_prompted", "topp_size_unprompted", "topp_size_cfg"):
        values = [r[key] for r in rows if key in r]
        if values:
            topp_summary[key] = float(np.mean(values))
    overlap_summary = {}
    for key in (
        "topp_overlap_cfg_prompted",
        "topp_overlap_cfg_instruct",
        "topp_overlap_prompted_instruct",
    ):
        values = [r[key] for r in rows if key in r]
        if values:
            overlap_summary[key] = float(np.mean(values))
    print("\n=== Section 5.1 ===")
    print("mean entropy per token:", {k: round(v, 3) for k, v in entropy_summary.items()})
    print("mean top-p=90% token count:", {k: round(v, 2) for k, v in topp_summary.items()})
    print("mean top-p overlap:", {k: round(v, 2) for k, v in overlap_summary.items()})

    # --- Section 5.2: per-dataset similarity to instruction tuning -----
    by_dataset: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    dataset_similarity = []
    for name, items in by_dataset.items():
        overlaps = [r["topp_overlap_cfg_instruct"] for r in items if "topp_overlap_cfg_instruct" in r]
        if overlaps:
            dataset_similarity.append(
                {
                    "dataset": name,
                    "mean_overlap": float(np.mean(overlaps)),
                    "std_overlap": float(np.std(overlaps)),
                    "count": len(overlaps),
                }
            )
    dataset_similarity.sort(key=lambda d: d["mean_overlap"], reverse=True)

    # Spearman correlation between the CFG and instruction-tuned
    # continuation perplexities, and the correlation matrix of Figure 5.
    correlations: Dict[str, float] = {}
    keys = [k for k in ("ppl_prompted", "ppl_cfg", "ppl_instruct") if any(k in r for r in rows)]
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            xs = [r[a] for r in rows if a in r and b in r]
            ys = [r[b] for r in rows if a in r and b in r]
            if len(xs) > 2:
                rho, p = spearman_correlation(xs, ys)
                correlations[f"spearman_{a}_{b}"] = rho
                correlations[f"pearson_{a}_{b}"] = pearson_correlation(xs, ys)
    print("\n=== Section 5.2 ===")
    print("correlations between continuation perplexities:", {k: round(v, 3) for k, v in correlations.items()})

    with open(os.path.join(out, "per_example.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(
            {
                "n_examples": len(rows),
                "n_datasets": len(by_dataset),
                "gamma": args.gamma,
                "entropy": entropy_summary,
                "topp_size": topp_summary,
                "top_p_overlap": overlap_summary,
                "correlations": correlations,
                "dataset_similarity": dataset_similarity,
            },
            fh,
            indent=2,
        )
    print(f"\nWrote {len(rows)} per-example rows and the summary to {out}")


if __name__ == "__main__":
    main()

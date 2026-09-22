#!/usr/bin/env python
"""Section 3.3 -- Table 2: intervene on the residual stream with toxic vectors.

For each of {W_toxic, MLP.v_toxic (top-1), SVD.U_toxic[0]} we subtract
``alpha * W`` from the residual stream of the last layer during generation on
the 1,199 challenge prompts of RealToxicityPrompts, and report toxicity, PPL
(Wikitext-2) and F1.  ``alpha`` is calibrated so that the perplexity matches the
post-DPO model (23.34 for GPT2 in the paper).
"""

from __future__ import annotations

import argparse

from _common import add_common_args, load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--vectors-path", default="artifacts/toxic_vectors/toxic_vectors.pt")
    parser.add_argument("--target-ppl", type=float, default=23.34,
                        help="perplexity of the post-DPO model (paper Table 2)")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--f1-items", type=int, default=2000)
    parser.add_argument("--no-toxicity", action="store_true")
    parser.add_argument("--ppl-sentences", type=int, default=None,
                        help="use only the first N Wikitext-2 test lines (faster calibration)")
    parser.add_argument("--alphas", default=None,
                        help="comma-separated alphas; skips the PPL-matched calibration "
                             "(useful for a quick sweep, e.g. --alphas 0,5,10,20)")
    args = parser.parse_args()

    from dpo_toxic.data.realtoxicity import load_realtoxicity_challenge
    from dpo_toxic.data.wikitext import load_wikitext2
    from dpo_toxic.evaluation.f1 import build_wikipedia_eval_set
    from dpo_toxic.evaluation.toxicity import ToxicityScorer
    from dpo_toxic.interventions import (InterventionSpec, evaluate_intervention,
                                         run_intervention_table)
    from dpo_toxic.toxic_vectors import load_toxic_vectors
    from dpo_toxic.utils import load_json, save_json, set_seed

    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args)
    blob = load_toxic_vectors(args.vectors_path)
    meta = load_json(str(args.vectors_path).replace(".pt", ".json"))
    selections = meta["selections"]

    prompts = load_realtoxicity_challenge(cache_dir=args.cache_dir)
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]
    print(f"{len(prompts)} challenge prompts")

    wikitext_texts = [t for t in load_wikitext2(cache_dir=args.cache_dir)["test"]["text"] if t.strip()]
    if args.ppl_sentences:
        wikitext_texts = wikitext_texts[: args.ppl_sentences]
    f1_items = None
    if args.f1_items:
        f1_items = build_wikipedia_eval_set(n=args.f1_items, tokenizer=tokenizer,
                                            cache_dir=args.cache_dir)
    scorer = None if args.no_toxicity else ToxicityScorer(device=args.device)

    if args.alphas:
        alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
        specs = [
            InterventionSpec("W_toxic", blob["w_toxic"]),
            InterventionSpec(f"MLP.v_{selections[0]['index']}^{selections[0]['layer']}",
                             blob["raw_value_vectors"][0]),
            InterventionSpec("SVD.U_toxic[0]", blob["svd_u"][:, 0]),
        ]
        results: dict = {}
        for spec in specs:
            for alpha in alphas:
                spec.alpha = alpha
                key = f"{spec.name}_alpha{alpha:g}"
                results[key] = evaluate_intervention(
                    model, tokenizer, spec, prompts, toxicity_scorer=scorer,
                    wikitext_texts=wikitext_texts, f1_items=f1_items,
                    batch_size=args.batch_size, device=args.device)
                print(key, {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in results[key].items() if isinstance(v, (int, float))})
        save_json(results, f"{args.out_dir}/interventions/alpha_sweep.json")
        return

    results = run_intervention_table(
        model, tokenizer, blob["w_toxic"], selections, blob["raw_value_vectors"],
        blob["svd_u"], prompts, target_ppl=args.target_ppl, toxicity_scorer=scorer,
        wikitext_texts=wikitext_texts, f1_items=f1_items,
        out_dir=f"{args.out_dir}/interventions", device=args.device, batch_size=args.batch_size)
    for name, row in results.items():
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                     if isinstance(v, (int, float))})
    save_json(results, f"{args.out_dir}/interventions/intervention_table.json")


if __name__ == "__main__":
    main()

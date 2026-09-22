#!/usr/bin/env python
"""Evaluate a model (optionally with an intervention) with the three metrics of
the paper: toxicity (RealToxicityPrompts challenge), perplexity (Wikitext-2) and
F1 (2,000 Wikipedia sentences).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import add_common_args, load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model-path", default=None, help="e.g. artifacts/dpo/model")
    parser.add_argument("--vectors-path", default=None, help="toxic vectors for an intervention")
    parser.add_argument("--intervention", default=None,
                        choices=[None, "w_toxic", "mlp_v", "svd_u0"])
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--f1-items", type=int, default=2000)
    parser.add_argument("--no-toxicity", action="store_true")
    parser.add_argument("--ppl-sentences", type=int, default=None,
                        help="use only the first N Wikitext-2 test lines (default: all)")
    parser.add_argument("--tag", default="model")
    args = parser.parse_args()

    from dpo_toxic.data.realtoxicity import load_realtoxicity_challenge
    from dpo_toxic.evaluation.f1 import build_wikipedia_eval_set, generation_f1
    from dpo_toxic.evaluation.perplexity import perplexity
    from dpo_toxic.evaluation.toxicity import ToxicityScorer
    from dpo_toxic.generation import generate_continuations
    from dpo_toxic.interventions import InterventionSpec, make_context
    from dpo_toxic.toxic_vectors import load_toxic_vectors
    from dpo_toxic.utils import save_json, set_seed

    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args, path=args.model_path)
    prompts = load_realtoxicity_challenge(cache_dir=args.cache_dir)
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]
    wiki_texts = None
    if args.ppl_sentences:
        from dpo_toxic.data.wikitext import load_wikitext2

        wiki_texts = [t for t in load_wikitext2(cache_dir=args.cache_dir)["test"]["text"]
                      if t.strip()][: args.ppl_sentences]
    f1_items = build_wikipedia_eval_set(n=args.f1_items, tokenizer=tokenizer,
                                        cache_dir=args.cache_dir) if args.f1_items else None
    scorer = None if args.no_toxicity else ToxicityScorer(device=args.device)

    spec = None
    if args.intervention and args.vectors_path:
        blob = load_toxic_vectors(args.vectors_path)
        if args.intervention == "w_toxic":
            vec = blob["w_toxic"]
        elif args.intervention == "mlp_v":
            vec = blob["raw_value_vectors"][0]
        else:
            vec = blob["svd_u"][:, 0]
        spec = InterventionSpec(args.intervention, vec, alpha=args.alpha)

    ctx = make_context(model, spec) if spec else None
    if ctx is not None:
        ctx.__enter__()
    try:
        gens = generate_continuations(model, tokenizer, prompts, max_new_tokens=args.max_new_tokens,
                                      batch_size=args.batch_size, device=args.device)
        metrics = {"tag": args.tag, "alpha": args.alpha, "n_prompts": len(prompts)}
        metrics["ppl"] = perplexity(model, tokenizer, texts=wiki_texts, device=args.device,
                                    cache_dir=args.cache_dir)
        metrics["ppl_sentences"] = len(wiki_texts) if wiki_texts else "all"
        if scorer is not None:
            scores = scorer.score(gens)
            metrics["toxicity"] = float(sum(scores) / max(len(scores), 1))
            metrics["toxicity_max"] = float(max(scores))
        if f1_items is not None:
            metrics.update({f"f1_{k}": v for k, v in
                            generation_f1(model, tokenizer, f1_items, batch_size=args.batch_size,
                                          device=args.device).items()})
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    out = Path(args.out_dir) / f"eval_{args.tag}.json"
    save_json(metrics, out)
    print(metrics)


if __name__ == "__main__":
    main()

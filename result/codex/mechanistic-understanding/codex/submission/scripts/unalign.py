#!/usr/bin/env python
"""Section 6 -- Table 4: scale the toxic key vectors of GPT2_DPO by 10x to bring
the toxicity back to the pre-alignment level.
"""

from __future__ import annotations

import argparse

from _common import add_common_args, load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--dpo-model", default="artifacts/dpo/model")
    parser.add_argument("--vectors-path", default="artifacts/toxic_vectors/toxic_vectors.pt")
    parser.add_argument("--n-vectors", type=int, default=7)
    parser.add_argument("--key-scale", type=float, default=10.0)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--f1-items", type=int, default=2000)
    parser.add_argument("--no-toxicity", action="store_true")
    args = parser.parse_args()

    from dpo_toxic.data.realtoxicity import load_realtoxicity_challenge
    from dpo_toxic.data.wikitext import load_wikitext2
    from dpo_toxic.evaluation.f1 import build_wikipedia_eval_set
    from dpo_toxic.evaluation.toxicity import ToxicityScorer
    from dpo_toxic.unalign import UnalignConfig, run_unalign_experiment
    from dpo_toxic.utils import load_json, set_seed

    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args, path=args.dpo_model)
    selections = load_json(str(args.vectors_path).replace(".pt", ".json"))["selections"]
    prompts = load_realtoxicity_challenge(cache_dir=args.cache_dir)
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]
    wikitext_texts = [t for t in load_wikitext2(cache_dir=args.cache_dir)["test"]["text"] if t.strip()]
    f1_items = build_wikipedia_eval_set(n=args.f1_items, tokenizer=tokenizer,
                                        cache_dir=args.cache_dir) if args.f1_items else None
    scorer = None if args.no_toxicity else ToxicityScorer(device=args.device)
    cfg = UnalignConfig(n_vectors=args.n_vectors, key_scale=args.key_scale)
    res = run_unalign_experiment(model, tokenizer, selections, prompts,
                                 wikitext_texts=wikitext_texts, f1_items=f1_items,
                                 toxicity_scorer=scorer, cfg=cfg, batch_size=args.batch_size,
                                 device=args.device,
                                 out_path=f"{args.out_dir}/unalign/table4.json")
    for name, row in res.items():
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                     if isinstance(v, (int, float))})


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Section 4.2 -- generate the 24,576 toxic / non-toxic preference pairs with
PPLM (toxic) and greedy sampling (non-toxic).
"""

from __future__ import annotations

import argparse

from _common import add_common_args, load_model_and_tokenizer, load_probe_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--probe-path", default="artifacts/probe/toxic_probe.pt")
    parser.add_argument("--n-pairs", type=int, default=24576)
    parser.add_argument("--n-tokens", type=int, default=20)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--probe-threshold", type=float, default=0.5)
    parser.add_argument("--no-probe-filter", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--out", default="artifacts/pairs/pairs.jsonl")
    args = parser.parse_args()

    from dpo_toxic.pairs import PairConfig, build_pair_dataset
    from dpo_toxic.utils import set_seed

    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args)
    probe = load_probe_from_args(args)
    cfg = PairConfig(n_pairs=args.n_pairs, n_tokens=args.n_tokens, shard_size=args.shard_size,
                     probe_threshold=args.probe_threshold,
                     filter_with_probe=not args.no_probe_filter, seed=args.seed)
    stats = build_pair_dataset(model, probe, cfg, out_path=args.out,
                               cache_dir=args.cache_dir, device=args.device,
                               tokenizer=tokenizer, resume=not args.no_resume)
    print(stats)


if __name__ == "__main__":
    main()

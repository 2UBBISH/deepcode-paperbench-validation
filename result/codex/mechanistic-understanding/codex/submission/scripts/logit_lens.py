#!/usr/bin/env python
"""Figure 1 -- logit lens on GPT2 vs GPT2_DPO for a toxic token ("sh*t")."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import add_common_args, load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--dpo-model", default="artifacts/dpo/model")
    parser.add_argument("--token", default="sh*t")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-prompts", type=int, default=None)
    args = parser.parse_args()

    from dpo_toxic.analysis.logit_lens import logit_lens, select_prompts_for_token
    from dpo_toxic.data.realtoxicity import load_realtoxicity_prompts
    from dpo_toxic.utils import save_json, set_seed

    set_seed(args.seed)
    before_model, tokenizer = load_model_and_tokenizer(args)
    after_model, _ = load_model_and_tokenizer(args, path=args.dpo_model)
    prompts = [p["text"] for p in load_realtoxicity_prompts(cache_dir=args.cache_dir)]
    selected = select_prompts_for_token(before_model, tokenizer, prompts, token=args.token,
                                        device=args.device, max_prompts=args.max_prompts)
    print(f"{len(selected)} prompts elicit {args.token!r} as the greedy next token")
    out = Path(args.out_dir) / "logit_lens"
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "gpt2": logit_lens(before_model, tokenizer, selected, token=args.token,
                           batch_size=args.batch_size, device=args.device),
        "gpt2_dpo": logit_lens(after_model, tokenizer, selected, token=args.token,
                               batch_size=args.batch_size, device=args.device),
        "n_prompts": len(selected),
    }
    save_json(result, out / f"logit_lens_{args.token.replace('*', 'x')}.json")
    for i, (a, b) in enumerate(zip(result["gpt2"]["post_block"], result["gpt2_dpo"]["post_block"])):
        print(f"layer {i:2d}: GPT2={a:.4f} GPT2_DPO={b:.4f}")


if __name__ == "__main__":
    main()

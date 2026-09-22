#!/usr/bin/env python
"""Section 4.1 -- DPO training on the crafted preference pairs (Table 8).

    python scripts/train_dpo.py --pairs artifacts/pairs/pairs.jsonl
"""

from __future__ import annotations

import argparse
import copy

from _common import add_common_args, load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--pairs", default="artifacts/pairs/pairs.jsonl")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--optimizer", default="rmsprop")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    args = parser.parse_args()

    from dpo_toxic.dpo import DPOConfig, train_dpo
    from dpo_toxic.pairs import load_pairs
    from dpo_toxic.utils import set_seed

    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args)
    ref_model = copy.deepcopy(model)
    pairs = load_pairs(args.pairs)
    print(f"loaded {len(pairs)} pairs")
    cfg = DPOConfig(beta=args.beta, learning_rate=args.lr, batch_size=args.batch_size,
                    gradient_accumulation_steps=args.grad_accum, max_grad_norm=args.max_grad_norm,
                    optimizer=args.optimizer, epochs=args.epochs, patience=args.patience,
                    val_every=args.val_every, val_fraction=args.val_fraction,
                    max_length=args.max_length, max_steps=args.max_steps,
                    max_train_pairs=args.max_train_pairs, seed=args.seed, device=args.device)
    report = train_dpo(model, ref_model, tokenizer, pairs, cfg,
                       out_dir=f"{args.out_dir}/dpo",
                       log_path=f"{args.out_dir}/dpo/train_log.json")
    print({k: v for k, v in report.items() if k != "history"})


if __name__ == "__main__":
    main()

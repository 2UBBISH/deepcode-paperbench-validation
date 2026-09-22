#!/usr/bin/env python
"""Extract mean-pooled residual-stream features on Jigsaw in resumable shards.

The full 90:10 Jigsaw split is ~500k comments; on a CPU box it is convenient to
extract the features in chunks and cache them:

    for i in 0 1 2 ...; do python scripts/extract_features.py --shard $i ...; done
    python scripts/train_probe.py --features-dir artifacts/probe/features
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import add_common_args, load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=4000)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--features-dir", default="artifacts/probe/features")
    args = parser.parse_args()

    import numpy as np

    from dpo_toxic.data.jigsaw import build_jigsaw_dataset
    from dpo_toxic.probe import extract_mean_residuals
    from dpo_toxic.utils import set_seed

    set_seed(args.seed)
    out = Path(args.features_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.split}_{args.shard:04d}.npz"
    if path.exists():
        print(f"{path} exists, skipping")
        return

    train_texts, train_labels, val_texts, val_labels, stats = build_jigsaw_dataset(
        cache_dir=args.cache_dir, max_train=args.max_train, max_val=args.max_val,
        val_fraction=args.val_fraction, seed=args.seed)
    texts, labels = (train_texts, train_labels) if args.split == "train" else (val_texts, val_labels)
    start = args.shard * args.shard_size
    chunk_texts = texts[start: start + args.shard_size]
    chunk_labels = labels[start: start + args.shard_size]
    if not chunk_texts:
        print("nothing to do (shard beyond the end of the split)")
        return

    model, tokenizer = load_model_and_tokenizer(args)
    features = extract_mean_residuals(model, tokenizer, chunk_texts, layer=args.layer,
                                      max_length=args.max_length, batch_size=args.batch_size,
                                      device=args.device)
    np.savez_compressed(path, features=features, labels=chunk_labels, stats=stats)
    print(f"wrote {path} with {features.shape}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Section 3.1 -- train the toxicity probe ``W_toxic`` on Jigsaw.

Paper setting: GPT2-medium, Jigsaw (561,808 comments), 90:10 split,
mean-pooled last-layer residual stream, 94% validation accuracy.

    python scripts/train_probe.py --model gpt2-medium
    python scripts/train_probe.py --max-train 20000 --max-val 2000   # CPU smoke run
"""

from __future__ import annotations

import argparse

from _common import add_common_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--layer", type=int, default=-1, help="residual-stream layer index (-1 = last)")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-train", type=int, default=None, help="cap the training set (smoke runs)")
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--feature-cache", default=None, help="npz cache of extracted features")
    parser.add_argument("--features-dir", default=None,
                        help="directory of feature shards from scripts/extract_features.py")
    parser.add_argument("--hf-dataset", default=None)
    args = parser.parse_args()

    from dpo_toxic.probe import ProbeConfig, run_probe_training

    cfg = ProbeConfig(layer=args.layer, max_length=args.max_length, batch_size=args.batch_size,
                      epochs=args.epochs, lr=args.lr, val_fraction=args.val_fraction,
                      seed=args.seed, max_train=args.max_train, max_val=args.max_val)
    probe, metrics = run_probe_training(model_name=args.model, cfg=cfg,
                                        out_dir=f"{args.out_dir}/probe",
                                        cache_dir=args.cache_dir,
                                        hf_dataset_id=args.hf_dataset,
                                        device=args.device,
                                        feature_cache=args.feature_cache,
                                        features_dir=args.features_dir)
    print(f"best validation accuracy: {metrics['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()

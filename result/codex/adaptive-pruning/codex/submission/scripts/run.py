#!/usr/bin/env python3
"""Command line interface for every experiment in the paper.

Examples
--------
Reproduce the APT rows of Table 2::

    python scripts/run.py apt --task sst2 --model roberta-base --sparsity 0.6
    python scripts/run.py apt --task mnli --model roberta-base --sparsity 0.6
    python scripts/run.py apt --task squad --model roberta-base --sparsity 0.6
    python scripts/run.py apt --task sst2 --model t5-base  --sparsity 0.6
    python scripts/run.py apt --task cnn_dm --model t5-base --sparsity 0.6

Reproduce the baselines of Table 2::

    python scripts/run.py baseline --baseline ft            --task sst2 --model roberta-base
    python scripts/run.py baseline --baseline lora          --task sst2 --model roberta-base
    python scripts/run.py baseline --baseline lora_prune    --task sst2 --model roberta-base
    python scripts/run.py baseline --baseline prune_distill --task sst2 --model roberta-base
    python scripts/run.py baseline --baseline lora_prune_distill --task sst2 --model roberta-base

Reproduce the ablation of Table 4 and the sparsity sweep of Figure 3::

    python scripts/run.py ablation --ablation wo_adaptive_tuning --task sst2 --model roberta-base
    python scripts/run.py sweep --task sst2 --model roberta-base --sparsities 0.2 0.4 0.6 0.8

``--dry-run`` prints the resolved configuration without training, which is handy
for checking a cluster job before submitting it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apt.pipeline import APTConfig, ABLATIONS, run_apt, run_apt_ablation, run_baseline  # noqa: E402


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", required=True, help="sst2 | mnli | qqp | qnli | cola | mrpc | rte | stsb | squad | cnn_dm")
    p.add_argument("--model", default="roberta-base")
    p.add_argument("--sparsity", type=float, default=0.6)
    p.add_argument("--epochs", type=int, default=40, help="total epochs (Table 6: 40 GLUE / 16 CNN-DM / 15 Alpaca)")
    p.add_argument("--distill-epochs", type=int, default=20, help="epochs of the pruning+distillation stage")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--initial-rank", type=int, default=8)
    p.add_argument("--target-rank", type=int, default=64)
    p.add_argument("--adjust-interval", type=int, default=50)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--tta-target", type=float, default=None,
                   help="dev metric of the FT baseline; used for the 97%% time-to-accuracy metric")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-eval", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--dry-run", action="store_true")


def build_config(args, task_defaults: bool = True) -> APTConfig:
    batch = args.batch_size
    if batch is None:
        batch = 32
        if args.task == "cnn_dm":
            batch = 16
    lr = args.lr
    if lr is None:
        lr = 1e-4 if args.task in {"cnn_dm"} else 2e-4
    epochs = args.epochs
    distill = args.distill_epochs
    if args.task == "cnn_dm" and task_defaults and args.epochs == 40:
        epochs, distill = 16, 6
    out = args.output_dir or f"runs/{args.model.replace('/', '_')}_{args.task}_{args.sparsity:g}"
    return APTConfig(
        model_name=args.model,
        task=args.task,
        output_dir=out,
        target_sparsity=args.sparsity,
        initial_rank=args.initial_rank,
        target_rank=args.target_rank,
        adjust_interval=args.adjust_interval,
        eval_interval=args.eval_interval,
        batch_size=batch,
        eval_batch_size=args.eval_batch_size or 128,
        epochs=epochs,
        distill_epochs=distill,
        learning_rate=lr,
        tta_target=args.tta_target,
        seed=args.seed,
        device=args.device,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_apt = sub.add_parser("apt", help="run APT")
    add_common(p_apt)

    p_base = sub.add_parser("baseline", help="run a baseline")
    add_common(p_base)
    p_base.add_argument("--baseline", required=True,
                        choices=["ft", "lora", "lora_prune", "prune_distill", "lora_prune_distill"])

    p_abl = sub.add_parser("ablation", help="run an APT ablation (Tables 4/5)")
    add_common(p_abl)
    p_abl.add_argument("--ablation", required=True, choices=sorted(ABLATIONS))

    p_sweep = sub.add_parser("sweep", help="pruning-sparsity analysis (Figure 3)")
    add_common(p_sweep)
    p_sweep.add_argument("--sparsities", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])

    args = parser.parse_args(argv)

    if args.command == "sweep":
        results = []
        for s in args.sparsities:
            args.sparsity = s
            args.output_dir = (args.output_dir or f"runs/{args.model}_{args.task}") + f"_s{int(s*100)}"
            cfg = build_config(args)
            if args.dry_run:
                print(json.dumps({"sparsity": s, "config": json.loads(cfg.to_json())}))
                continue
            results.append(run_apt(args.task, args.model, cfg,
                                   limit_train=args.limit_train, limit_eval=args.limit_eval))
        if not args.dry_run:
            print(json.dumps([{"sparsity": r["config"]["target_sparsity"],
                               "metrics": r["final_metrics"],
                               "inference": r.get("inference", {})} for r in results], indent=2))
        return 0

    cfg = build_config(args)
    if args.command == "ablation":
        # show (and later run) the actual ablation configuration
        from dataclasses import replace as _replace

        cfg = _replace(cfg, **ABLATIONS[args.ablation])
    if args.dry_run:
        if args.command == "ablation":
            print(json.dumps({"ablation": args.ablation, "config": json.loads(cfg.to_json())}, indent=2))
            return 0
        print(cfg.to_json())
        return 0

    if args.command == "apt":
        result = run_apt(args.task, args.model, cfg,
                         limit_train=args.limit_train, limit_eval=args.limit_eval)
    elif args.command == "ablation":
        result = run_apt_ablation(args.task, args.ablation, args.model, cfg,
                                  limit_train=args.limit_train, limit_eval=args.limit_eval)
    else:
        result = run_baseline(args.baseline, args.task, args.model, args.sparsity, cfg,
                              limit_train=args.limit_train)
    print(json.dumps({k: v for k, v in result.items() if k != "logs"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Reduced-scale, CPU-runnable end-to-end smoke test of the reproduction.

It exercises every component of the pipeline on a tiny scale:

    D_PT (3 P3 tasks x N examples)  ->  cached logits/representations
    D_R  (mispredictions of FLAN-T5_small on SuperGLUE-CB)  ->  f_i per error
    forgetting labels z_ij          ->  the five forecasting methods  ->  F1
    replay-based refinement (Table 4 style) with random / score / GT selection

The numbers it produces are *not* comparable to the paper (a 77M-parameter model,
a handful of examples and a few optimisation steps); the point is that the whole
pipeline runs and that the metrics behave sensibly (e.g. ground-truth replay
forgets less than no replay).

Usage::

    python smoke/run_smoke.py --per-task 8 --online 4 --forecast-steps 30
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from wwmf.experiments import build_forecasters, evaluate_forecasters  # noqa: E402
from wwmf.forecasting.cache import build_online_artifacts, build_upstream_cache  # noqa: E402
from wwmf.refinement.replay import (  # noqa: E402
    GTForgetReplay,
    RandomReplay,
    ReplayPool,
    ScoreReplay,
)
from wwmf.refinement.stream import evaluate_single_error_replay  # noqa: E402
from wwmf.utils import ensure_dir, write_json  # noqa: E402
from wwmf.utils import set_seed  # noqa: E402

from smoke_common import build_smoke_dpt, build_smoke_dr, make_smoke_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/flan-t5-small")
    parser.add_argument("--per-task", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=48)
    parser.add_argument("--online", type=int, default=4)
    # the paper uses 100 head-only steps at lr=1e-3; at this scale 20 steps at
    # 2e-3 are enough to actually fix the errors, which keeps the run short.
    parser.add_argument("--steps", type=int, default=20, help="refinement steps per error (paper: 30/100)")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--forecast-steps", type=int, default=30)
    parser.add_argument("--tuning-mode", default="head", choices=["head", "lora", "full_ft"])
    parser.add_argument("--output", default="smoke/outputs")
    args = parser.parse_args()

    cfg = make_smoke_config(args)
    # the refinement fine-tuning runs with dropout enabled, so the torch RNG has
    # to be seeded for the smoke numbers to be reproducible
    set_seed(cfg.seed)

    from wwmf.models.lm import Seq2SeqLM

    print(f"[smoke] loading {args.model}")
    model = Seq2SeqLM.from_config(cfg) if args.model == cfg.model_spec().hf_name else Seq2SeqLM(
        cfg.model_spec(), device=cfg.device, max_input_len=cfg.max_input_len,
        max_target_len=cfg.max_target_len, hf_name=args.model,
    )

    print("[smoke] building D_PT")
    dpt = build_smoke_dpt(args.per_task)
    print(f"[smoke] |D_PT| = {len(dpt)}")

    print("[smoke] building D_R (mispredictions of the base model)")
    splits, n_mispredicted = build_smoke_dr(model, args.candidates, args.online)
    print(f"[smoke] |candidates| = {len(splits.candidates)}  EM = {splits.em:.3f}  |D_R| = {n_mispredicted}")
    if len(splits.train) + len(splits.test) < 2:
        raise SystemExit("not enough mispredictions for the smoke run; increase --candidates")
    dr_train, dr_test = splits.train, splits.test
    print(f"[smoke] |D_R^Train| = {len(dr_train)}  |D_R^Test| = {len(dr_test)}")

    print("[smoke] caching upstream logits/representations")
    upstream_cache = build_upstream_cache(model, dpt, topk=20, max_vocab=32, batch_size=8)
    print(f"[smoke] base EM on D_PT = {upstream_cache.base_em:.3f}")

    print("[smoke] collecting forgetting labels for each online example")
    train_artifacts = build_online_artifacts(
        model, dr_train, upstream_cache, steps=cfg.steps_single(), lr=cfg.lr_single(),
        mode=cfg.tuning_mode, batch_size=4,
    )
    test_artifacts = build_online_artifacts(
        model, dr_test, upstream_cache, steps=cfg.steps_single(), lr=cfg.lr_single(),
        mode=cfg.tuning_mode, batch_size=4,
    )
    for art in train_artifacts + test_artifacts:
        rate = float(np.mean(art.labels))
        print(f"[smoke]   {art.example.key}: edit_success={art.edit_success} forget_rate={rate:.3f}")

    from wwmf.forecasting.base import ForecastContext

    ctx = ForecastContext(
        cfg=cfg, upstream_cache=upstream_cache, train_artifacts=train_artifacts,
        device=model.device, seed=0, verbose=True,
    )
    print("[smoke] training the five forecasting methods")
    forecasters = build_forecasters(ctx, train_steps=args.forecast_steps)
    metrics = evaluate_forecasters(forecasters, test_artifacts, ctx)
    for name, m in metrics.items():
        print(f"[smoke]   {name:24s} F1={m['f1']:6.2f} P={m['precision']:6.2f} R={m['recall']:6.2f}")

    print("[smoke] Table 4 style: single-error refinement with replay")
    pool = ReplayPool(upstream_cache, cfg.max_target_len)
    artifact_lookup = {a.example.key: a for a in train_artifacts + test_artifacts}
    label_lookup = {a.example.key: a.labels for a in train_artifacts + test_artifacts}
    get_artifact = lambda example: artifact_lookup.get(example.key)  # noqa: E731
    get_labels = lambda example: label_lookup.get(example.key)  # noqa: E731
    small_test = dr_test[: min(2, len(dr_test))]
    refinement = []
    refinement.append(
        evaluate_single_error_replay(cfg, model, small_test, dpt, method="Vanilla FT").to_dict()
    )
    refinement.append(
        evaluate_single_error_replay(
            cfg, model, small_test, dpt, method="Replay w/ Random", replay_strategy=RandomReplay(seed=0),
            replay_pool=pool,
        ).to_dict()
    )
    refinement.append(
        evaluate_single_error_replay(
            cfg, model, small_test, dpt, method="Replay w/ Representation",
            replay_strategy=ScoreReplay(forecasters["representation"], get_artifact, seed=0),
            replay_pool=pool,
        ).to_dict()
    )
    refinement.append(
        evaluate_single_error_replay(
            cfg, model, small_test, dpt, method="Replay w/ GT Forget",
            replay_strategy=GTForgetReplay(get_labels, seed=0), replay_pool=pool,
        ).to_dict()
    )
    for row in refinement:
        print(f"[smoke]   {row['method']:26s} Succ={row['Succ.']}%  EM Drop={row['EM Drop %']}%")

    out_dir = ensure_dir(args.output)
    write_json(
        os.path.join(out_dir, "smoke_results.json"),
        {
            "model": args.model,
            "n_dpt": len(dpt),
            "n_dr": n_mispredicted,
            "n_dr_train": len(dr_train),
            "n_dr_test": len(dr_test),
            "n_upstream_correct": int(upstream_cache.correct_mask.sum()),
            "steps": args.steps,
            "forecast_train_steps": args.forecast_steps,
            "base_em_dpt": upstream_cache.base_em,
            "forecasting_f1": metrics,
            "refinement": refinement,
        },
    )
    print(f"[smoke] wrote {os.path.join(out_dir, 'smoke_results.json')}")


if __name__ == "__main__":
    main()

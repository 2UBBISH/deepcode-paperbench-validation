#!/usr/bin/env python
"""Reduced-scale smoke test of the sequential-refinement code paths.

`run_smoke.py` covers the data pipeline and the forecasting models; this script
covers the parts used by Table 3 (sequential refinement with replay) and Figure 3
(F1 / precision / recall curves while the LM is continually refined).  It is
intentionally tiny: a few online examples, a few update steps and a handful of
upstream examples.

Usage::

    python smoke/run_smoke_stream.py --online 2 --steps 5
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from wwmf.config import ExperimentConfig  # noqa: E402
from wwmf.experiments import (  # noqa: E402
    build_forecasters,
    prepare_state,
)
from wwmf.refinement.replay import RandomReplay, ReplayPool, ScoreReplay  # noqa: E402
from wwmf.refinement.stream import (  # noqa: E402
    continual_forecasting_curves,
    sequential_refinement,
)
from wwmf.utils import ensure_dir, write_json  # noqa: E402
from wwmf.utils import set_seed  # noqa: E402

from smoke_common import build_smoke_dpt, build_smoke_dr, make_smoke_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="flan_t5_small", choices=["flan_t5_small"])
    parser.add_argument("--upstream", type=int, default=12)
    parser.add_argument("--online", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--forecast-steps", type=int, default=10)
    parser.add_argument("--output", default="smoke/outputs_stream")
    args = parser.parse_args()

    args.tuning_mode = "head"
    args.max_input_len = 256
    cfg = make_smoke_config(args, model=args.model)
    set_seed(cfg.seed)  # dropout is active during fine-tuning -> seed torch

    from wwmf.models.lm import Seq2SeqLM

    model = Seq2SeqLM.from_config(cfg)
    dpt = build_smoke_dpt(per_task=max(4, args.upstream // 3))
    splits, n_mispredicted = build_smoke_dr(model, max_candidates=40, online=args.online)
    print(f"[stream-smoke] |D_PT| = {len(dpt)}  |D_R^Train| = {len(splits.train)} "
          f"|D_R^Test| = {len(splits.test)} (of {n_mispredicted} mispredictions)")

    state = prepare_state(
        cfg, max_upstream=args.upstream, verbose=True, dpt=dpt, dr=splits
    )
    pool: ReplayPool = state.replay_pool
    forecasters = build_forecasters(
        state.ctx, train_steps=args.forecast_steps, include=["threshold", "representation"]
    )

    print("[stream-smoke] Table 3 style: sequential refinement with replay")
    table3 = [
        sequential_refinement(cfg, state.base_model, state.dr_test, state.dpt, method="Vanilla FT",
                              verbose=True).to_dict(),
        sequential_refinement(cfg, state.base_model, state.dr_test, state.dpt, method="Replay w/ Random",
                              replay_strategy=RandomReplay(seed=0), replay_pool=pool, verbose=True).to_dict(),
        sequential_refinement(cfg, state.base_model, state.dr_test, state.dpt,
                              method="Replay w/ Representation",
                              replay_strategy=ScoreReplay(
                                  forecasters["representation"], state.artifact_lookup, seed=0),
                              replay_pool=pool, verbose=True).to_dict(),
    ]
    for row in table3:
        print(f"[stream-smoke]   {row['method']:26s} Succ={row['Succ.']}%  EM Drop={row['EM Drop %']}%")

    print("[stream-smoke] Figure 3 style: forecasting while continually refining")
    curves = continual_forecasting_curves(
        cfg, state.base_model, forecasters, state.dr_test, state.upstream_cache,
        stream_fraction=0.5, verbose=True,
    )
    for name, series in curves.items():
        last = series[-1]
        print(f"[stream-smoke]   {name:16s} steps={len(series)} F1={last['f1']:.2f} "
              f"P={last['precision']:.2f} R={last['recall']:.2f}")

    out_dir = ensure_dir(args.output)
    write_json(
        os.path.join(out_dir, "smoke_stream_results.json"),
        {"table3_style": table3, "figure3_style": curves},
    )
    print(f"[stream-smoke] wrote {os.path.join(out_dir, 'smoke_stream_results.json')}")


if __name__ == "__main__":
    main()

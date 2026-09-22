#!/usr/bin/env python
"""Step 1 of the pipeline: build D_PT, D_R and the cached ground-truth labels.

This is the expensive pre-computation of the paper:

1. ``D_PT``: balanced sample of the 36 upstream P3 training tasks.
2. ``D_R``: mispredictions of the base PTLM on the refinement dataset (P3-Test for
   BART0, MMLU validation for FLAN-T5), split 60/40.
3. cached logits/representations of ``D_PT`` and, for every online example, the
   logit change of the updated model and the ground-truth forgetting labels
   ``z_ij`` obtained by running inference with ``f_i`` over ``D_PT``.

The artefacts are pickled under ``--cache-root`` and reused by the table scripts.
"""
import sys

from _common import base_parser, config_from_args, print_result

from wwmf.experiments import prepare_state


def main() -> None:
    args = base_parser(__doc__ or "").parse_args()
    cfg = config_from_args(args)
    state = prepare_state(
        cfg,
        max_upstream=args.max_upstream,
        max_online=args.max_online,
        verbose=not args.quiet,
    )
    print_result(
        {
            "model": cfg.model,
            "tuning_mode": cfg.tuning_mode,
            "n_upstream": len(state.dpt),
            "n_upstream_correct": int(state.upstream_cache.correct_mask.sum()),
            "base_em_on_dpt": state.upstream_cache.base_em,
            "n_dr_train": len(state.dr_train),
            "n_dr_test": len(state.dr_test),
            "mean_forgetting_rate_train": float(
                state.train_artifacts[0].labels.mean() if state.train_artifacts else 0.0
            ),
            "edit_success_rate_train": float(
                sum(a.edit_success for a in state.train_artifacts) / max(1, len(state.train_artifacts))
            ),
        }
    )


if __name__ == "__main__":
    sys.exit(main())

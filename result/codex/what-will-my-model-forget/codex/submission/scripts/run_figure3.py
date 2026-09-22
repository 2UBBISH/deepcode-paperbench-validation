#!/usr/bin/env python
"""Figure 3: F1 / precision / recall while continually refining the LM.

The forecasted forgetting indicator is computed at the start of the stream and
reused at every time step; the ground truth is recomputed with the updated LM.
"""
from _common import base_parser, config_from_args, print_result

from wwmf.experiments import run_figure3


def main() -> None:
    args = base_parser(__doc__ or "").parse_args()
    cfg = config_from_args(args)
    curves = run_figure3(
        cfg,
        train_steps=args.train_steps,
        max_upstream=args.max_upstream,
        max_online=args.max_online,
        verbose=not args.quiet,
    )
    print_result({name: series[-1] for name, series in curves.items()})


if __name__ == "__main__":
    main()

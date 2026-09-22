#!/usr/bin/env python
"""Table 2: in-domain / out-of-domain forecasting performance on BART0.

The model is trained on P3-Test_ID and evaluated on P3-Test_ID and P3-Test_OOD.
"""
from _common import base_parser, config_from_args, print_result

from wwmf.experiments import run_ood_experiment


def main() -> None:
    args = base_parser(__doc__ or "").parse_args()
    cfg = config_from_args(args)
    cfg.refinement_data = "p3_test"
    result = run_ood_experiment(
        cfg,
        train_steps=args.train_steps,
        max_upstream=args.max_upstream,
        max_online=args.max_online,
        verbose=not args.quiet,
    )
    print_result(result)


if __name__ == "__main__":
    main()

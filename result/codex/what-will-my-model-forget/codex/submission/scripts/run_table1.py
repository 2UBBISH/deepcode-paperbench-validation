#!/usr/bin/env python
"""Table 1: F1 of forecasting example forgetting when fixing one error at a time.

Usage::

    python scripts/run_table1.py --model bart0_large --tuning-mode head
    python scripts/run_table1.py --model flan_t5_large --tuning-mode lora
"""
from _common import base_parser, config_from_args, print_result

from wwmf.experiments import run_forecasting_experiment


def main() -> None:
    args = base_parser(__doc__ or "").parse_args()
    cfg = config_from_args(args)
    result = run_forecasting_experiment(
        cfg,
        train_steps=args.train_steps,
        max_upstream=args.max_upstream,
        max_online=args.max_online,
        verbose=not args.quiet,
    )
    print_result(result["results"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Table 4: EM drop ratio when separately fixing single errors."""
from _common import base_parser, config_from_args, print_result

from wwmf.experiments import run_refinement_table


def main() -> None:
    args = base_parser(__doc__ or "").parse_args()
    cfg = config_from_args(args)
    results = run_refinement_table(
        cfg,
        single_error=True,
        max_upstream=args.max_upstream,
        max_online=args.max_online,
        verbose=not args.quiet,
    )
    print_result(results)


if __name__ == "__main__":
    main()

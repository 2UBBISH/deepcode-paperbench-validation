#!/usr/bin/env python
"""Sec. 5.3 / Table 5: complexity and FLOP accounting of the forecasting methods."""
from _common import base_parser, print_result

from wwmf.analysis.complexity import COMPLEXITY_TABLE, estimate_flops


def main() -> None:
    args = base_parser(__doc__ or "").parse_args()
    model_sizes = {
        "bart0_large": (400e6, 1024, 50265),
        "flan_t5_large": (780e6, 1024, 32128),
        "flan_t5_3b": (3e9, 2048, 32128),
        "flan_t5_small": (77e6, 512, 32128),
    }
    params, H, V = model_sizes[args.model]
    flops = estimate_flops(
        n_upstream=args.max_upstream or 3600,
        T=32,
        H=int(H),
        V=int(V),
        forward_flops_per_example=2 * params * 64,
    )
    print_result({"analytic": COMPLEXITY_TABLE, "flops": flops})


if __name__ == "__main__":
    main()

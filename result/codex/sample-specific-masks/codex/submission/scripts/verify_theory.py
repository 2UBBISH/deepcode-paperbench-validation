#!/usr/bin/env python3
"""Numerical verification of the paper's theoretical results.

* Theorem 4.2 / Proposition 4.3: the SMM hypothesis space contains the
  shared-mask hypothesis space; encoding a shared mask into the generator gives
  bit-identical reprogrammed inputs.
* Proposition B.1: setting ``delta = J`` makes SMM contain sample-specific
  patterns.
* Definition 4.1: empirical approximation error of ``F_shr``, ``F_sp`` and
  ``F_smm`` on a synthetic deterministic task (Bayes risk 0).

    python scripts/verify_theory.py [--steps 300]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smm.theory import (  # noqa: E402
    empirical_approximation_error_experiment,
    verify_sample_specific_inclusion,
    verify_shared_mask_inclusion,
    verify_watermark_inclusion,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--samples", type=int, default=1024)
    args = parser.parse_args()

    results = {
        "proposition_4_3_shared_mask": verify_shared_mask_inclusion(num_pool_layers=2),
        "proposition_4_3_watermark": verify_watermark_inclusion(),
        "proposition_B_1_sample_specific": verify_sample_specific_inclusion(),
        "empirical_approximation_error": empirical_approximation_error_experiment(
            num_samples=args.samples, steps=args.steps, restarts=args.restarts
        ),
    }
    print(json.dumps(results, indent=2))

    ok = all(
        results[key]["included"] == 1.0
        for key in (
            "proposition_4_3_shared_mask",
            "proposition_4_3_watermark",
            "proposition_B_1_sample_specific",
        )
    )
    ok = ok and bool(results["empirical_approximation_error"]["ordering_holds"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

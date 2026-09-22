"""Executable check of Theorem 2 (eps-convergence) on a synthetic RCS problem.

Theorem 2 states that under Condition 1 (progressable) and Condition 2 (stable
moving) the algorithm converges in the RCS sense: with probability one

    f_2(m^t) -> f_2*,   f_2* = min { f_2(m) | f_1(m) <= f_1* (1 + eps) }.

To make the guarantee *checkable*, this script builds a synthetic RCS instance
whose exact optimum can be computed by enumeration and then runs the real
``LexiFlow`` optimiser on it:

* ``n`` synthetic examples carry a scalar "quality" ``v_i in [0, 1]``;
* ``f_1(m) = 1 - mean_{i in m} v_i`` (the average quality of the selected
  examples) and ``f_2(m) = ||m||_0``;
* ``f_1*``, ``f_2*`` and the ``eps``-optimal region are obtained by brute force
  over all ``2^n`` masks.

The script reports, over several random restarts, whether the mask returned by
``LexiFlow`` satisfies ``f_1(m) <= f_1* (1 + eps)`` and
``f_2(m) <= f_2*``, and it verifies Condition 1 on the generated iterate
sequence.
"""

from __future__ import annotations

import itertools
import os
from typing import List, Tuple

import torch

from lbcs.lexiflow import LexiFlow, LexiFlowConfig
from lbcs.utils import set_seed

from .common import (base_argparser, markdown_table, resolve_cli, save_json,
                     write_rows_csv)


class SyntheticRCS:
    """A tiny RCS instance with an exactly computable lexicographic optimum."""

    def __init__(self, n: int = 12, seed: int = 0, concentration: float = 3.0):
        g = torch.Generator().manual_seed(seed)
        self.n = n
        self.quality = torch.rand(n, generator=g)
        # make a few examples clearly better / worse so the optimum is unique
        self.quality = self.quality ** concentration
        self.cache: dict = {}

    def __call__(self, continuous_mask: torch.Tensor) -> torch.Tensor:
        key = tuple((continuous_mask >= 0).long().tolist())
        if key in self.cache:
            return self.cache[key]
        m = torch.tensor(key, dtype=torch.float64)
        k = float(m.sum().item())
        if k == 0:
            value = torch.tensor([float("inf"), 0.0], dtype=torch.float64)
        else:
            f1 = 1.0 - float((m * self.quality.double()).sum().item()) / k
            value = torch.tensor([f1, k], dtype=torch.float64)
        self.cache[key] = value
        return value

    # -- exact solution -------------------------------------------------
    def brute_force(self, epsilon: float) -> Tuple[float, float]:
        best_f1 = float("inf")
        table = []
        for bits in itertools.product([0, 1], repeat=self.n):
            m = torch.tensor(bits, dtype=torch.float64)
            k = float(m.sum().item())
            if k == 0:
                continue
            f1 = 1.0 - float((m * self.quality.double()).sum().item()) / k
            table.append((bits, f1, k))
            best_f1 = min(best_f1, f1)
        thr = best_f1 * (1.0 + epsilon)
        f2_star = min(k for _, f1, k in table if f1 <= thr)
        return best_f1, f2_star


def main(argv=None) -> int:
    parser = base_argparser("Theorem 2: eps-convergence check")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--k", type=int, default=None,
                        help="size of the initial mask (default: n/2)")
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args(argv)
    resolve_cli(args)
    if args.dry_run:
        args.restarts, args.steps = 2, 20

    n = args.n
    k = args.k or n // 2
    problem = SyntheticRCS(n=n, seed=0)
    f1_star, f2_star = problem.brute_force(args.epsilon)
    print(f"synthetic RCS: n={n} k={k} eps={args.epsilon} "
          f"-> f1*={f1_star:.4f} f2*={f2_star:.0f}")

    rows: List[dict] = []
    for restart in range(args.restarts):
        set_seed(restart)
        g = torch.Generator().manual_seed(restart)
        init = -torch.ones(n)
        idx = torch.randperm(n, generator=g)[:k]
        init[idx] = 1.0
        cfg = LexiFlowConfig(epsilon=args.epsilon, delta_init=1.0,
                             delta_lower=1e-4, max_steps=args.steps,
                             step_decay_patience=10, seed=restart)
        search = LexiFlow(problem, cfg)
        best = search.optimize(init)
        value = problem(best)
        f1_final, f2_final = float(value[0]), float(value[1])
        cond1 = _condition1_stats(search, f1_star, args.epsilon)
        rows.append({
            "restart": restart,
            "f1_final": f1_final, "f2_final": f2_final,
            "f1_threshold": f1_star * (1 + args.epsilon),
            "satisfies_f1_constraint": f1_final <= f1_star * (1 + args.epsilon),
            "satisfies_f2_optimality": f2_final <= f2_star,
            "condition1_violations": cond1["violations"],
            "condition1_steps": cond1["steps"],
            "condition1_fraction": cond1["fraction"],
            "num_evaluations": len(search.history_values),
        })

    write_rows_csv(os.path.join(args.results_dir, "theorem2_convergence.csv"),
                   rows)
    summary = {
        "n": n, "k": k, "epsilon": args.epsilon,
        "f1_star": f1_star, "f2_star": f2_star,
        "fraction_f1_constraint": sum(r["satisfies_f1_constraint"]
                                      for r in rows) / len(rows),
        "fraction_f2_optimal": sum(r["satisfies_f2_optimality"]
                                   for r in rows) / len(rows),
        "mean_condition1_fraction": sum(r["condition1_fraction"]
                                        for r in rows) / len(rows),
        "rows": rows,
    }
    save_json(os.path.join(args.results_dir, "theorem2_convergence.json"),
              summary)
    print(markdown_table(
        [{"restart": r["restart"], "f1": f"{r['f1_final']:.4f}",
          "f2": f"{r['f2_final']:.0f}",
          "f1 <= f1*(1+eps)": r["satisfies_f1_constraint"],
          "f2 <= f2*": r["satisfies_f2_optimality"],
          "Condition 1 (%)": f"{100 * r['condition1_fraction']:.1f}"}
         for r in rows],
        ["restart", "f1", "f2", "f1 <= f1*(1+eps)", "f2 <= f2*",
         "Condition 1 (%)"]))
    print(f"\nf2-optimality reached in "
          f"{summary['fraction_f2_optimal'] * 100:.0f}% of the restarts; "
          f"the eps-constraint on f1 holds in "
          f"{summary['fraction_f1_constraint'] * 100:.0f}%; Condition 1 holds "
          f"on {summary['mean_condition1_fraction'] * 100:.0f}% of the steps "
          f"(it is an idealised assumption -- every step of the walker would "
          f"have to improve f2 once f1 is inside the compromise region).")
    return 0


def _condition1_stats(search: LexiFlow, f1_star: float,
                      epsilon: float) -> dict:
    """Check Condition 1 (progressable) on the walker's trajectory.

    Condition 1 requires that at every step

    * if ``m^t`` is outside the compromise region ``M_1*`` (which here is
      computed with the *true* ``f_1*`` of the synthetic problem), the next
      iterate improves ``f_1``;
    * otherwise the next iterate stays inside ``M_1*`` and improves ``f_2``.

    The second requirement is an idealisation: the coreset size is bounded
    below by one, so no algorithm can decrease it at every single step.  The
    script therefore reports the fraction of steps for which Condition 1
    holds instead of a boolean.
    """
    threshold = f1_star * (1.0 + epsilon)
    values = search.history_values
    masks = search.history_masks
    violations = 0
    steps = 0
    for i, (prev, cur) in enumerate(zip(values, values[1:])):
        f1_prev, f2_prev = prev
        f1_cur, f2_cur = cur
        # Remark 3 only constrains *updates* of the incumbent mask; steps for
        # which the walker stays at the same discretised mask are no-ops.
        if i + 1 < len(masks) and torch.equal((masks[i] >= 0), (masks[i + 1] >= 0)):
            continue
        steps += 1
        if f1_prev > threshold:                    # m^t not in M_1*
            ok = f1_cur < f1_prev
        else:                                      # m^t in M_1*
            ok = (f1_cur <= threshold) and (f2_cur < f2_prev)
        if not ok:
            violations += 1
    return {"violations": violations, "steps": steps,
            "fraction": (steps - violations) / steps if steps else 1.0}


if __name__ == "__main__":
    raise SystemExit(main())

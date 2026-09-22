"""Hyper-parameter selection and the tables of Section 6.

The addendum to the paper documents how the authors picked the configurations
used for Figures 3 and 7:

    "for a given PDE, the configuration of Adam learning rate, seed and network
     width with the smallest L2RE is used."

That is exactly what :func:`select_best_configuration` implements.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

from .common import ADAM_LRS, PDES, SEEDS, SWITCHES, WIDTHS, Paths

OPTIMIZER_ORDER = ["Adam", "L-BFGS", "Adam+L-BFGS (1k)", "Adam+L-BFGS (11k)", "Adam+L-BFGS (31k)"]


def _key(record: dict, metric: str) -> float:
    v = record.get(metric)
    return float("inf") if v is None else float(v)


def select_best_configuration(records: Iterable[dict], pde: str, metric: str = "final_l2re") -> dict:
    """Smallest ``metric`` over (learning rate, seed, width) for one PDE."""
    cands = [r for r in records if r["pde"] == pde]
    if not cands:
        raise ValueError(f"no records for PDE {pde!r}")
    return min(cands, key=lambda r: _key(r, metric))


def table1(records: Sequence[dict]) -> List[dict]:
    """Lowest loss/L2RE of Adam, L-BFGS and Adam+L-BFGS across all widths.

    Mirrors Table 1 of the paper: "Lowest loss for Adam, L-BFGS, and
    Adam+L-BFGS across all network widths after hyperparameter tuning."
    """
    rows = []
    for pde in PDES:
        for family, predicate in (
            ("Adam", lambda r: r["optimizer"] == "adam"),
            ("L-BFGS", lambda r: r["optimizer"] == "lbfgs"),
            ("Adam+L-BFGS", lambda r: r["optimizer"] == "adam_lbfgs"),
        ):
            cands = [r for r in records if r["pde"] == pde and predicate(r)]
            if not cands:
                continue
            best_loss = min(cands, key=lambda r: _key(r, "final_loss"))
            best_l2re = min(cands, key=lambda r: _key(r, "final_l2re"))
            rows.append(
                {
                    "pde": pde,
                    "optimizer": family,
                    "best_loss": best_loss["final_loss"],
                    "best_loss_l2re": best_loss["final_l2re"],
                    "best_loss_config": best_loss["run_id"],
                    "best_l2re": best_l2re["final_l2re"],
                    "best_l2re_loss": best_l2re["final_loss"],
                    "best_l2re_config": best_l2re["run_id"],
                }
            )
    return rows


def figure8_stats(records: Sequence[dict]) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Per-width min/median/max loss and L2RE for each optimisation strategy.

    Figure 8: "We find the learning rate for each network width and
    optimization strategy that attains the lowest loss (L2RE) across all random
    seeds. The min, median, and max loss (L2RE) are calculated by taking the
    min, median, and max of the losses (L2REs) for learning rate eta* across all
    random seeds."
    """
    import statistics

    out: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for pde in sorted({r["pde"] for r in records}):
        pde_records = [r for r in records if r["pde"] == pde]
        widths = sorted({r["width"] for r in pde_records})
        labels = sorted(
            {r["label"] for r in pde_records},
            key=lambda s: (OPTIMIZER_ORDER.index(s) if s in OPTIMIZER_ORDER else len(OPTIMIZER_ORDER), s),
        )
        out[pde] = {}
        for label in labels:
            losses, l2res = [], []
            for width in widths:
                cands = [r for r in pde_records if r["width"] == width and r["label"] == label]
                if not cands:
                    continue
                # 1) pick the learning rate that minimises the loss across seeds
                best_loss_lr = None
                best_loss_val = float("inf")
                for lr in sorted({r["lr"] for r in cands}):
                    vals = [r["final_loss"] for r in cands if r["lr"] == lr]
                    if min(vals) < best_loss_val:
                        best_loss_val = min(vals)
                        best_loss_lr = lr
                # 2) pick the learning rate that minimises the L2RE across seeds
                best_l2re_lr = None
                best_l2re_val = float("inf")
                for lr in sorted({r["lr"] for r in cands}):
                    vals = [r["final_l2re"] for r in cands if r["lr"] == lr]
                    if min(vals) < best_l2re_val:
                        best_l2re_val = min(vals)
                        best_l2re_lr = lr
                losses.append([r["final_loss"] for r in cands if r["lr"] == best_loss_lr])
                l2res.append([r["final_l2re"] for r in cands if r["lr"] == best_l2re_lr])
            if not losses:
                continue
            out[pde][label] = {
                "min_loss": [min(v) for v in losses],
                "median_loss": [statistics.median(v) for v in losses],
                "max_loss": [max(v) for v in losses],
                "min_l2re": [min(v) for v in l2res],
                "median_l2re": [statistics.median(v) for v in l2res],
                "max_l2re": [max(v) for v in l2res],
                "widths": widths[: len(losses)],
            }
    return out


def write_csv(rows: Sequence[dict], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        return path
    keys = list(rows[0])
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return path

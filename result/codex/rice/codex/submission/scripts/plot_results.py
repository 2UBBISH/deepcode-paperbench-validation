#!/usr/bin/env python3
"""Draw the figures of the paper from the JSON files in ``results/``.

    python scripts/plot_results.py --results results --out figures

Produced figures (matching the paper):

* ``fidelity_<env>.png``        -- Figure 5  (fidelity of Random/StateMask/Ours)
* ``refining_<env>.png``        -- Figure 2  (performance during refining)
* ``p_lambda_<env>.png``        -- Figures 7 and 8 (sensitivity of p and lambda)
* ``alpha_<env>.png``           -- Figure 9  (fidelity under different alpha)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)


def _load(path):
    with open(path) as handle:
        return json.load(handle)


def plot_fidelity(path, out_dir):
    data = _load(path)
    env = data["env"]
    plt.figure(figsize=(5, 3.5))
    for method, values in data["fidelity"].items():
        ks = [k * 100 for k in values["k"]]
        plt.errorbar(ks, values["mean"], yerr=values["std"], marker="o", label=method)
    plt.xlabel("K (%)")
    plt.ylabel("fidelity score")
    plt.title("fidelity - {}".format(env))
    plt.legend()
    plt.tight_layout()
    out = os.path.join(out_dir, "fidelity_{}.png".format(env))
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_refining(path, out_dir):
    data = _load(path)
    env = data["env"]
    curves = data.get("curves", {})
    if not curves:
        return None
    plt.figure(figsize=(6, 4))
    for method, runs in curves.items():
        steps = runs[0]["steps"]
        returns = [r["returns"] for r in runs]
        mean = [sum(v[i] for v in returns) / len(returns) for i in range(len(steps))]
        std = [
            (sum((v[i] - mean[i]) ** 2 for v in returns) / len(returns)) ** 0.5
            for i in range(len(steps))
        ]
        plt.plot(steps, mean, label=method)
        plt.fill_between(steps, [m - s for m, s in zip(mean, std)],
                         [m + s for m, s in zip(mean, std)], alpha=0.2)
    baseline = data.get("no_refine", {})
    baseline_value = baseline.get("mean", baseline.get("mean_return"))
    if baseline_value is not None:
        plt.axhline(baseline_value, linestyle="--", color="grey", label="no refine")
    plt.xlabel("refining steps")
    plt.ylabel("episode reward")
    plt.title("refining - {}".format(env))
    plt.legend()
    plt.tight_layout()
    out = os.path.join(out_dir, "refining_{}.png".format(env))
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_p_lambda(path, out_dir):
    data = _load(path)
    env = data["env"]
    grid = data["grid"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # left: fix lambda (best), vary p  (Figure 7 in the addendum's description)
    for lam in data["lambda_values"]:
        xs, ys = [], []
        for p in data["p_values"]:
            key = "p={}_lambda={}".format(p, lam)
            if key in grid:
                xs.append(p)
                ys.append(sum(r["final_return"] for r in grid[key]) / len(grid[key]))
        axes[0].plot(xs, ys, marker="o", label="lambda={}".format(lam))
    axes[0].set_xlabel("p")
    axes[0].set_ylabel("reward after refining")
    axes[0].set_title("{}: sensitivity of p".format(env))
    axes[0].legend()

    # right: fix p (best), vary lambda  (Figure 8)
    for p in data["p_values"]:
        xs, ys = [], []
        for lam in data["lambda_values"]:
            key = "p={}_lambda={}".format(p, lam)
            if key in grid:
                xs.append(lam)
                ys.append(sum(r["final_return"] for r in grid[key]) / len(grid[key]))
        axes[1].plot(xs, ys, marker="s", label="p={}".format(p))
    axes[1].set_xscale("symlog", linthresh=1e-3)
    axes[1].set_xlabel("lambda")
    axes[1].set_title("{}: sensitivity of lambda".format(env))
    axes[1].legend()
    fig.tight_layout()
    out = os.path.join(out_dir, "p_lambda_{}.png".format(env))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_alpha(path, out_dir):
    data = _load(path)
    env = data["env"]
    plt.figure(figsize=(5, 3.5))
    for key, values in data.items():
        if not key.startswith("alpha="):
            continue
        plt.errorbar(
            [k * 100 for k in values["k"]],
            values["mean"],
            yerr=values["std"],
            marker="o",
            label=key,
        )
    plt.xlabel("K (%)")
    plt.ylabel("fidelity score")
    plt.title("alpha sensitivity - {}".format(env))
    plt.legend()
    plt.tight_layout()
    out = os.path.join(out_dir, "alpha_{}.png".format(env))
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    def _find(pattern):
        return sorted(
            set(
                glob.glob(os.path.join(args.results, pattern))
                + glob.glob(os.path.join(args.results, "*", pattern))
            )
        )

    produced = []
    for path in _find("*_fidelity.json"):
        produced.append(plot_fidelity(path, args.out))
    for path in _find("*_refining.json"):
        produced.append(plot_refining(path, args.out))
    for path in _find("*_p_lambda.json"):
        produced.append(plot_p_lambda(path, args.out))
    for path in _find("*_alpha.json"):
        produced.append(plot_alpha(path, args.out))
    for path in produced:
        if path:
            print("wrote", path)


if __name__ == "__main__":
    main()

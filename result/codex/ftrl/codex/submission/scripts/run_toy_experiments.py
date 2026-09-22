#!/usr/bin/env python3
"""Run the toy experiments from Appendix A and save figures + raw results.

This script is fully CPU-runnable in a few seconds and reproduces

* Figure 9 -- the two-state MDPs (state coverage gap and imperfect cloning gap),
* Figure 10 -- forgetting in AppleRetrieval as ``M`` grows,
* Figure 11 -- the impact of the observation scale ``c``.

Example::

    python scripts/run_toy_experiments.py --out results/toy
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fpc.toy.two_state_mdp import (
    build_imperfect_cloning_gap,
    build_state_coverage_gap,
    run_paper_scenarios,
)
from fpc.toy.apple_retrieval import AppleRetrieval, LinearPolicy, pretrain_on_phase2, reinforce


def run_two_state_mdp(out_dir: str) -> dict:
    results = {}
    print("== Two-state MDPs (Appendix A.1) ==")

    # (i) The paper's printed closed-form value function.
    paper = run_paper_scenarios()
    for scenario in paper:
        print(
            f"  {scenario.name:34s} start {scenario.start_theta:.2f} -> "
            f"theta* {scenario.converged_theta:.4f}  v0 {scenario.converged_value:.4f} "
            f"(optimum {scenario.optimal_value:.2f})"
        )
    results["paper_closed_form"] = {
        s.name: {
            "theta_grid": s.theta_grid,
            "v0": s.v0_grid,
            "start_theta": s.start_theta,
            "converged_theta": s.converged_theta,
            "converged_value": s.converged_value,
            "optimal_theta": s.optimal_theta,
            "optimal_value": s.optimal_value,
        }
        for s in paper
    }

    # (ii) The numerical two-state MDP (self-consistent Bellman solve) used as a
    #     cross-check of the qualitative behaviour.
    scg = build_state_coverage_gap(r_start=0.0, r_home=0.0, r_far=1.0, gamma=0.9)
    theta_grid = np.linspace(0.0, 1.0, 400)
    scg_curve = [scg.v0(float(t)) for t in theta_grid]
    theta, trajectory = scg.fine_tune(0.0, lr=0.005, steps=200_000, clip=(0.0, 1.0))
    results["state_coverage_gap"] = {
        "theta_grid": theta_grid.tolist(),
        "v0": scg_curve,
        "start_theta": 0.0,
        "converged_theta": theta,
        "converged_value": scg.v0(theta),
        "trajectory": trajectory[:: max(len(trajectory) // 200, 1)],
    }
    print(f"  numerical MDP (SCG)                theta* {theta:.3f}  v0 {scg.v0(theta):.3f}")

    icg = build_imperfect_cloning_gap(r_start=0.0, r_home=0.0, r_far=1.0, gamma=0.9)
    icg_curve = [icg.v0(float(t)) for t in theta_grid]
    theta_opt, _ = icg.fine_tune(1.0, lr=0.02, steps=50_000, clip=(0.0, 1.0))
    theta_perturbed, _ = icg.fine_tune(1.0 - 0.05, lr=0.02, steps=50_000, clip=(0.0, 1.0))
    results["imperfect_cloning_gap"] = {
        "theta_grid": theta_grid.tolist(),
        "v0": icg_curve,
        "theta_optimal": 1.0,
        "value_optimal": icg.v0(1.0),
        "converged_theta_without_noise": theta_opt,
        "converged_theta_with_noise": theta_perturbed,
        "converged_value_with_noise": icg.v0(theta_perturbed),
    }
    print(f"  numerical MDP (ICG)                optimum v0 {icg.v0(1.0):.3f}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "two_state_mdp.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    return results


def run_apple_retrieval(out_dir: str, distances=(5, 15, 30, 50), c_values=(0.1, 0.3, 0.5, 1.0, 5.0),
                        c_for_distance: float = 0.5) -> dict:
    results = {"by_distance": {}, "by_c": {}}
    print(f"== AppleRetrieval (Appendix A.2; distance sweep at c={c_for_distance}) ==")

    for m in distances:
        env = AppleRetrieval(M=m, c=c_for_distance, seed=m)
        policy, _ = pretrain_on_phase2(env, episodes=2000, lr=1e-2, seed=m)
        history = reinforce(env, policy, episodes=6000, lr=1e-2, phase=None, seed=m)
        results.setdefault("c_for_distance", c_for_distance)
        results["by_distance"][str(m)] = {
            "wb_ratio_before_finetuning": float(history["wb_ratio"][0]),
            "phase2_success_after_finetuning": env.evaluate(policy, episodes=100, phase=1),
            "overall_success_after_finetuning": env.evaluate(policy, episodes=100, phase=None),
            "wb_ratio": policy.wb_ratio,
            "success_curve": history["success"],
        }
        print(
            f"  M={m:3d}  phase-2 success {results['by_distance'][str(m)]['phase2_success_after_finetuning']:.2f}"
            f"  overall {results['by_distance'][str(m)]['overall_success_after_finetuning']:.2f}"
            f"  |b/w| {policy.wb_ratio:.2f}"
        )

    for c in c_values:
        env = AppleRetrieval(M=30, c=c, seed=int(c * 100))
        policy, _ = pretrain_on_phase2(env, episodes=2000, lr=1e-2, seed=int(c * 100))
        history = reinforce(env, policy, episodes=6000, lr=1e-2, phase=None, seed=int(c * 100))
        results["by_c"][str(c)] = {
            "phase2_success_after_finetuning": env.evaluate(policy, episodes=100, phase=1),
            "overall_success_after_finetuning": env.evaluate(policy, episodes=100, phase=None),
            "wb_ratio": policy.wb_ratio,
            "early_wb_ratio": float(np.mean(history["wb_ratio"][:200])),
        }
        print(
            f"  c={c:<4}  phase-2 success {results['by_c'][str(c)]['phase2_success_after_finetuning']:.2f}"
            f"  overall {results['by_c'][str(c)]['overall_success_after_finetuning']:.2f}"
            f"  early |b/w| {results['by_c'][str(c)]['early_wb_ratio']:.2f}"
        )

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "apple_retrieval.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    return results


def plot_toy(out_dir: str, mdp: dict, apple: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    scg = mdp["paper_closed_form"]["state_coverage_gap"]
    axes[0].plot(scg["theta_grid"], scg["v0"], label=r"$v_0(\theta)$")
    axes[0].axvline(scg["converged_theta"], color="crimson", ls="--",
                    label=fr"converged $\theta^*={scg['converged_theta']:.3f}$")
    axes[0].set_title("State coverage gap (Fig. 9b)")
    axes[0].set_xlabel(r"$\theta$")
    axes[0].set_ylabel(r"$v_0(\theta)$")
    axes[0].legend()

    icg = mdp["paper_closed_form"]["reported_suboptimal_fixed_point"]
    axes[1].plot(icg["theta_grid"], icg["v0"], label=r"$v_0(\theta)$")
    axes[1].axvline(icg["converged_theta"], color="crimson", ls="--",
                    label=fr"trapped at $\theta^*={icg['converged_theta']:.3f}$")
    axes[1].set_title("Imperfect cloning gap (Fig. 9c)")
    axes[1].set_xlabel(r"$\theta$")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure9_two_state_mdp.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ms = [int(k) for k in apple["by_distance"]]
    axes[0].plot(ms, [apple["by_distance"][str(m)]["phase2_success_after_finetuning"] for m in ms],
                 marker="o", label="phase-2 (FAR) success")
    axes[0].plot(ms, [apple["by_distance"][str(m)]["overall_success_after_finetuning"] for m in ms],
                 marker="s", label="overall success")
    axes[0].set_xlabel("M (distance to apple)")
    axes[0].set_title("Forgetting grows with M (Fig. 10)")
    axes[0].legend()

    cs = [float(k) for k in apple["by_c"]]
    axes[1].plot(cs, [apple["by_c"][str(c)]["phase2_success_after_finetuning"] for c in cs],
                 marker="o", label="phase-2 success")
    axes[1].plot(cs, [apple["by_c"][str(c)]["early_wb_ratio"] for c in cs],
                 marker="s", label="early |b|/|w|")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("c")
    axes[1].set_title("Impact of c (Fig. 11)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure10_11_apple_retrieval.png"), dpi=150)
    plt.close(fig)
    print(f"  figures written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/toy")
    args = parser.parse_args()

    mdp = run_two_state_mdp(args.out)
    apple = run_apple_retrieval(args.out)
    try:
        plot_toy(args.out, mdp, apple)
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"[warn] could not produce figures: {exc}")


if __name__ == "__main__":
    main()

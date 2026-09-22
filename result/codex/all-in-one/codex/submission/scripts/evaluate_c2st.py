#!/usr/bin/env python
"""Evaluate a trained Simformer (and optionally an sbi baseline) with C2ST.

* posterior C2ST (Fig. 4a): compare Simformer samples of ``p(theta | x_obs)``
  with the MCMC reference samples,
* conditional C2ST (Fig. 4b): compare arbitrary conditionals of the Simformer
  with MCMC reference samples; only the latent variables are compared.

The C2ST metric is a random forest with 100 trees (see
``simformer.metrics.c2st_accuracy``); 0.5 means the distributions match.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from common import RESULTS, ensure_dir, load_checkpoint
from simformer import get_task
from simformer.metrics import c2st_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True,
                        help="path to a trained Simformer checkpoint")
    parser.add_argument("--baseline", default=None,
                        help="optional path to a trained sbi baseline (.pkl)")
    parser.add_argument("--reference-dir", default=str(RESULTS / "reference"))
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def simformer_samples(model, values, states, n_samples, num_steps, index=None):
    """Sample conditionals of the joint with the Simformer."""
    conditions = list(zip(np.atleast_2d(np.asarray(values, dtype=float)),
                          np.atleast_2d(np.asarray(states, dtype=float))))
    return model.sample(conditions, num_samples=n_samples, num_steps=num_steps,
                        index=index)


def evaluate_posterior(model, task, reference_path, n_samples, num_steps,
                       baseline_path=None, seed=0, n_trees=100):
    data = np.load(reference_path)
    theta_true = data["theta_true"]
    x_obs = data["x_obs"]
    reference = data["samples"]
    n_obs = len(theta_true)
    n_params = task.n_params
    c2st_simformer = []
    for i in range(n_obs):
        value = np.concatenate([np.zeros(n_params), x_obs[i]])
        state = np.concatenate([np.zeros(n_params), np.ones(task.n_data)])
        samples = simformer_samples(model, value, state, n_samples,
                                    num_steps)[0]
        c2st_simformer.append(c2st_accuracy(samples[:, :n_params],
                                            reference[i, :, :n_params],
                                            n_trees=n_trees, seed=seed))
    result = {"c2st_simformer": float(np.mean(c2st_simformer)),
              "c2st_simformer_per_observation":
                  [float(c) for c in c2st_simformer]}
    if baseline_path is not None:
        import torch
        with open(baseline_path, "rb") as f:
            baseline = pickle.load(f)["posterior"]
        c2st_baseline = []
        for i in range(n_obs):
            x_tensor = torch.as_tensor(x_obs[i][None], dtype=torch.float32)
            samples = baseline.sample((n_samples,), x=x_tensor)
            samples = np.asarray(samples.detach().cpu().numpy()).reshape(
                n_samples, -1)
            c2st_baseline.append(c2st_accuracy(samples,
                                               reference[i, :, :n_params],
                                               n_trees=n_trees, seed=seed))
        result["c2st_baseline"] = float(np.mean(c2st_baseline))
        result["c2st_baseline_per_observation"] = \
            [float(c) for c in c2st_baseline]
    return result


def evaluate_conditionals(model, task, reference_path, n_samples, num_steps,
                          n_conditionals=None, seed=0, n_trees=100):
    data = np.load(reference_path)
    values, states, reference = data["values"], data["states"], data["samples"]
    if n_conditionals is not None:
        values, states, reference = (values[:n_conditionals],
                                     states[:n_conditionals],
                                     reference[:n_conditionals])
    accuracies = []
    per_variable = np.zeros(task.n_variables)
    counts = np.zeros(task.n_variables)
    for i in range(len(values)):
        samples = simformer_samples(model, values[i], states[i], n_samples,
                                    num_steps)[0]
        latent = np.flatnonzero(states[i] < 0.5)
        if latent.size == 0:
            continue
        accuracies.append(c2st_accuracy(samples[:, latent],
                                        reference[i][:, latent],
                                        n_trees=n_trees, seed=seed))
        for v in latent:
            per_variable[v] += c2st_accuracy(samples[:, [v]],
                                             reference[i][:, [v]],
                                             n_trees=n_trees, seed=seed)
            counts[v] += 1
    with np.errstate(invalid="ignore"):
        per_variable = np.where(counts > 0, per_variable / np.maximum(counts, 1),
                                np.nan)
    return {"c2st_conditional": float(np.mean(accuracies)),
            "c2st_conditional_per_conditional": [float(a) for a in accuracies],
            "c2st_conditional_per_variable": per_variable.tolist(),
            "variable_counts": counts.tolist()}


def main():
    args = parse_args()
    task = get_task(args.task)
    model = load_checkpoint(task, Path(args.model), num_steps=args.num_steps)
    out_dir = Path(args.out) if args.out else Path(args.model).parent
    ensure_dir(out_dir)
    results = {}
    posterior_ref = Path(args.reference_dir) / f"{args.task}_posterior.npz"
    conditional_ref = Path(args.reference_dir) / f"{args.task}_conditional.npz"
    if posterior_ref.exists():
        results["posterior"] = evaluate_posterior(
            model, task, posterior_ref, args.n_samples, args.num_steps,
            args.baseline, args.seed, args.n_trees)
        print(f"[c2st] posterior C2ST: simformer "
              f"{results['posterior']['c2st_simformer']:.3f}")
    if conditional_ref.exists():
        results["conditional"] = evaluate_conditionals(
            model, task, conditional_ref, args.n_samples, args.num_steps,
            seed=args.seed, n_trees=args.n_trees)
        print(f"[c2st] conditional C2ST: "
              f"{results['conditional']['c2st_conditional']:.3f}")
    (out_dir / "c2st.json").write_text(json.dumps(results, indent=2))
    print(f"[c2st] written to {out_dir / 'c2st.json'}")


if __name__ == "__main__":
    main()

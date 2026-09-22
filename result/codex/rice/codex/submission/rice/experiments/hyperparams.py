"""Experiment V -- sensitivity of ``p``, ``lambda`` and ``alpha``.

* ``p``      -- mixing ratio of the critical states (Figures 7 / Figure 6)
* ``lambda`` -- weight of the RND exploration bonus (Figure 8 / Figure 6)
* ``alpha``  -- bonus that encourages the mask network to blind more states
                (Figure 9, measured through the fidelity score)
"""

from __future__ import annotations

import itertools
import os
from typing import Dict, List, Optional, Sequence

from rice.envs.registry import ENV_SPECS
from rice.experiments.common import (
    ckpt_path,
    close_env,
    ensure_dir,
    load_agent,
    save_json,
)
from rice.experiments.refining import default_refine_config, run_single_refine


P_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
LAMBDA_VALUES = (0.0, 0.1, 0.01, 0.001)
ALPHA_VALUES = (0.01, 0.001, 0.0001)


def run_p_lambda_grid(
    env_name: str,
    agent_path: str,
    mask_path: str,
    p_values: Sequence[float] = P_VALUES,
    lambda_values: Sequence[float] = LAMBDA_VALUES,
    total_steps: Optional[int] = None,
    seeds: Sequence[int] = (0, 1, 2),
    device: str = "cpu",
    out_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Sweep the two RICE hyper-parameters on the same pre-trained agent."""
    spec = ENV_SPECS[env_name]
    policy = load_agent(env_name, agent_path, device=device)
    from rice.explanation.mask_io import load_mask_net

    mask_net = load_mask_net(mask_path, hidden=spec.mask_hidden)
    grid: Dict[str, List[Dict[str, float]]] = {}

    for p, lam in itertools.product(p_values, lambda_values):
        key = "p={}_lambda={}".format(p, lam)
        records: List[Dict[str, float]] = []
        for seed in seeds:
            config = default_refine_config(
                env_name, seed=seed, total_steps=total_steps, p=p, rnd_lambda=lam
            )
            if verbose:
                print("[exp V] {} {} seed {}".format(env_name, key, seed), flush=True)
            result = run_single_refine(
                env_name,
                policy,
                mask_net,
                method="ours",
                config=config,
                device=device,
                verbose=False,
            )
            records.append(
                {
                    "seed": float(seed),
                    "final_return": float(result.final_eval),
                    "steps": [float(h["steps"]) for h in result.eval_history],
                    "returns": [float(h["mean_return"]) for h in result.eval_history],
                }
            )
        grid[key] = records

    payload = {
        "env": env_name,
        "p_values": [float(p) for p in p_values],
        "lambda_values": [float(v) for v in lambda_values],
        "grid": grid,
        "config": {"total_steps": int(total_steps or spec.refine_steps)},
    }
    out_dir = ensure_dir(out_dir or ckpt_path("experiment_v", kind="results"))
    save_json(payload, os.path.join(out_dir, "{}_p_lambda.json".format(env_name)))
    return payload


def run_alpha_sweep(
    env_name: str,
    agent_path: str,
    alpha_values: Sequence[float] = ALPHA_VALUES,
    samples: int = 50_000,
    iterations: int = 50,
    n_trajectories: int = 100,
    n_seeds: int = 3,
    seed: int = 0,
    device: str = "cpu",
    out_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Retrain the mask network for several ``alpha`` and report fidelity."""
    from rice.envs.registry import make_env
    from rice.experiments.explanation import train_explanation
    from rice.fidelity import (
        FidelityConfig,
        compute_fidelity,
        mask_importance_fn,
    )
    from rice.policies import TorchPolicy

    spec = ENV_SPECS[env_name]
    out_dir = ensure_dir(out_dir or ckpt_path("experiment_v", kind="results"))
    results: Dict[str, object] = {"env": env_name, "alpha_values": list(alpha_values)}
    for alpha in alpha_values:
        learning = train_explanation(
            env_name,
            agent_path,
            method="ours",
            seed=seed,
            device=device,
            alpha=alpha,
            max_samples=samples,
            iterations=iterations,
            verbose=verbose,
        )
        env = make_env(env_name, seed=seed)
        policy = TorchPolicy(load_agent(env_name, agent_path, device=device))
        fidelity = compute_fidelity(
            env,
            policy,
            mask_importance_fn(learning.mask_net),
            FidelityConfig(
                n_trajectories=n_trajectories,
                k_values=(0.1, 0.2, 0.3, 0.4),
                d_max=spec.d_max,
                seed=seed,
            ),
            n_seeds=n_seeds,
        )
        close_env(env)
        results["alpha={}".format(alpha)] = fidelity

    save_json(results, os.path.join(out_dir, "{}_alpha.json".format(env_name)))
    return results

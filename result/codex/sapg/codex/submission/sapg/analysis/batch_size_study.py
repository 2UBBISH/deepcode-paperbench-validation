"""Figure 2: PPO performance as a function of the batch size.

"Performance vs batch size plot for PPO runs (blue curve) across two
environments.  The curve shows how PPO training runs can not take benefit of
large batch size resulting from massively parallelized environments and their
asymptotic performance saturates after a certain point.  The dashed red line is
the performance of our method, SAPG."
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np


def run_batch_size_study(
    env_name: str,
    env_counts: Sequence[int] = (128, 512, 2048, 8192, 24576),
    seeds: Sequence[int] = (0, 1, 2),
    config_paths: Optional[Dict[str, str]] = None,
    num_iterations: Optional[int] = None,
    total_env_steps: Optional[float] = None,
    horizon: int = 16,
    device: str = "cuda:0",
    logroot: str = "runs/batch_size_study",
) -> Dict[str, Dict[int, List[float]]]:
    """Train PPO for every environment count and return the final performance.

    The batch size of an iteration is ``num_envs * horizon``.  Runs are compared
    against the number of samples collected (Sec. 5.2), so when
    ``total_env_steps`` is given every run is trained for
    ``total_env_steps / (num_envs * horizon)`` iterations: the large-batch runs
    see fewer policy updates.  SAPG keeps the same per-policy batch size while
    splitting the environments between ``M`` policies, so it is compared at the
    same batch size with the same number of samples.
    """
    from ..algorithms import make_trainer
    from ..envs import make_env
    from ..utils.config import load_config

    config_paths = config_paths or {}
    ppo_config = config_paths.get("ppo", "sapg/configs/ppo_allegrokuka.yaml")
    sapg_config = config_paths.get("sapg", "sapg/configs/sapg_allegrokuka_regrasping.yaml")

    results: Dict[str, Dict[int, List[float]]] = {"ppo": {}}
    for num_envs in env_counts:
        if num_iterations is None and total_env_steps is not None:
            iterations = max(1, int(float(total_env_steps) / (num_envs * horizon)))
        else:
            iterations = num_iterations
        values: List[float] = []
        for seed in seeds:
            cfg = load_config(ppo_config, [f"env.num_envs={num_envs}", f"seed={seed}", f"env.name={env_name}"])
            logdir = os.path.join(logroot, f"ppo_{env_name}_{num_envs}_seed{seed}")
            env = make_env(cfg, device=device)
            trainer = make_trainer(cfg, env, device=device, logdir=logdir)
            trainer.train(num_iterations=iterations)
            values.append(float(trainer.evaluate_policy(0).get("successes", 0.0)))
        results["ppo"][int(num_envs)] = values

    sapg_values = []
    for seed in seeds:
        cfg = load_config(sapg_config, [f"seed={seed}", f"env.name={env_name}"])
        logdir = os.path.join(logroot, f"sapg_{env_name}_seed{seed}")
        env = make_env(cfg, device=device)
        trainer = make_trainer(cfg, env, device=device, logdir=logdir)
        trainer.train(num_iterations=num_iterations)
        sapg_values.append(float(trainer.evaluate_policy(0).get("successes", 0.0)))
    results["sapg"] = {-1: sapg_values}  # plotted as the dashed reference line
    return results


def plot_batch_size_study(
    results: Dict[str, Dict[int, List[float]]],
    out_path: str,
    horizon: int = 16,
    title: str = "PPO performance vs batch size",
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ppo = results.get("ppo", {})
    xs = sorted(k for k in ppo if k > 0)
    if xs:
        means = [float(np.mean(ppo[k])) for k in xs]
        errors = [float(np.std(ppo[k]) / max(1, np.sqrt(len(ppo[k])))) for k in xs]
        ax.errorbar(
            [k * horizon for k in xs],
            means,
            yerr=errors,
            marker="o",
            color="tab:blue",
            label="PPO",
            capsize=3,
        )
    sapg_values = results.get("sapg", {}).get(-1, [])
    if sapg_values:
        ax.axhline(
            float(np.mean(sapg_values)),
            color="tab:red",
            linestyle="--",
            label="SAPG",
        )
    ax.set_xlabel("batch size (num_envs x horizon)")
    ax.set_ylabel("asymptotic performance")
    ax.set_xscale("log", base=2)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

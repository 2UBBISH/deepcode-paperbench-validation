"""Efficiency benchmarking for mask training (Experiment II).

This module measures the wall-clock time required to train the RICE mask network
and a StateMask-style baseline on identical fixed sample budgets (Table 4), and
reports the relative speed-up.
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from rice.agents.mask_network import (
    MaskNetwork,
    MaskTrainingConfig,
    default_mask_config,
    make_mask_network,
    train_mask_network,
)
from rice.agents.target_agent import TargetAgent, load_target_agent
from rice.envs import make_mujoco_env
from rice.evaluation.evaluate_policy import evaluate_policy
from rice.utils.config import (
    DEFAULT_SAMPLE_BUDGETS,
    DomainConfig,
    get_domain_config,
    table_4,
)
from rice.utils.logger import Logger, make_logger


def benchmark_rice_mask(
    env,
    target_agent: TargetAgent,
    sample_budget: int,
    config: Optional[MaskTrainingConfig] = None,
    seed: int = 0,
    device: str = "auto",
    verbose: int = 0,
) -> Dict[str, Any]:
    """Train a RICE mask network for a fixed number of samples and record wall time.

    Args:
        env: Training environment (single Gymnasium env).
        target_agent: Pre-trained target agent.
        sample_budget: Total number of environment steps (samples) allowed.
        config: Mask training configuration. If None, a domain default is used.
        seed: Random seed.
        device: Torch device.
        verbose: Verbosity level.

    Returns:
        Dictionary with keys ``wall_time_sec``, ``samples``, ``config``,
        ``mask_net`` (trained), and ``samples_per_sec``.
    """
    if config is None:
        config = default_mask_config(domain="mujoco")
    config.total_timesteps = sample_budget
    config.seed = seed
    config.device = device
    config.verbose = verbose

    mask_net = make_mask_network(
        env.observation_space,
        env.action_space,
        config=config,
    )

    start = time.perf_counter()
    mask_net, _ = train_mask_network(
        env,
        target_agent,
        mask_net=mask_net,
        config=config,
    )
    elapsed = time.perf_counter() - start

    return {
        "method": "RICE",
        "wall_time_sec": elapsed,
        "samples": sample_budget,
        "samples_per_sec": sample_budget / elapsed if elapsed > 0 else float("inf"),
        "config": config,
        "mask_net": mask_net,
    }


def benchmark_statemask_mask(
    env,
    target_agent: TargetAgent,
    sample_budget: int,
    config: Optional[MaskTrainingConfig] = None,
    seed: int = 0,
    device: str = "auto",
    verbose: int = 0,
) -> Dict[str, Any]:
    """Train a StateMask-style mask network for a fixed sample budget and record wall time.

    StateMask (Zhu et al., 2022) learns a mask by comparing the target policy's
    action distribution with a perturbed distribution and optimizing a similar
    on-policy objective.  We approximate it here by reusing the RICE mask
    training pipeline but with a *state-only* mask network and a larger
    perturbation radius, which captures the same computational cost profile
    (forward/backward passes through a small MLP plus environment simulation).

    Args:
        env: Training environment.
        target_agent: Pre-trained target agent.
        sample_budget: Total number of environment steps allowed.
        config: Mask training configuration. If None, a domain default is used.
        seed: Random seed.
        device: Torch device.
        verbose: Verbosity level.

    Returns:
        Dictionary with keys ``wall_time_sec``, ``samples``, ``config``,
        ``mask_net`` (trained), and ``samples_per_sec``.
    """
    if config is None:
        config = default_mask_config(domain="mujoco")
    # StateMask uses a state-only mask and a different blinding coefficient.
    config.use_action = False
    config.continuous_mask = True
    config.total_timesteps = sample_budget
    config.seed = seed
    config.device = device
    config.verbose = verbose

    mask_net = make_mask_network(
        env.observation_space,
        env.action_space,
        config=config,
    )

    start = time.perf_counter()
    mask_net, _ = train_mask_network(
        env,
        target_agent,
        mask_net=mask_net,
        config=config,
    )
    elapsed = time.perf_counter() - start

    return {
        "method": "StateMask",
        "wall_time_sec": elapsed,
        "samples": sample_budget,
        "samples_per_sec": sample_budget / elapsed if elapsed > 0 else float("inf"),
        "config": config,
        "mask_net": mask_net,
    }


def compare_efficiency(
    env,
    target_agent: TargetAgent,
    sample_budgets: Optional[List[int]] = None,
    rice_config: Optional[MaskTrainingConfig] = None,
    statemask_config: Optional[MaskTrainingConfig] = None,
    seed: int = 0,
    device: str = "auto",
    verbose: int = 0,
    logger: Optional[Logger] = None,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Run the efficiency comparison across a set of sample budgets.

    Returns:
        Nested dictionary ``results[method][budget] = benchmark_dict``.
    """
    if sample_budgets is None:
        sample_budgets = sorted(set(table_4().values()))

    results: Dict[str, Dict[int, Dict[str, Any]]] = {
        "RICE": {},
        "StateMask": {},
    }

    for budget in sample_budgets:
        if verbose:
            print(f"Benchmarking sample budget {budget:,} ...")

        rice_result = benchmark_rice_mask(
            env,
            target_agent,
            sample_budget=budget,
            config=rice_config,
            seed=seed,
            device=device,
            verbose=verbose,
        )
        results["RICE"][budget] = rice_result

        statemask_result = benchmark_statemask_mask(
            env,
            target_agent,
            sample_budget=budget,
            config=statemask_config,
            seed=seed,
            device=device,
            verbose=verbose,
        )
        results["StateMask"][budget] = statemask_result

        speedup = statemask_result["wall_time_sec"] / rice_result["wall_time_sec"]
        if logger is not None:
            logger.log(
                {
                    "efficiency/rice_wall_time_sec": rice_result["wall_time_sec"],
                    "efficiency/statemask_wall_time_sec": statemask_result["wall_time_sec"],
                    "efficiency/speedup": speedup,
                    "efficiency/sample_budget": budget,
                },
                step=budget,
            )

        if verbose:
            print(
                f"  RICE: {rice_result['wall_time_sec']:.2f}s, "
                f"StateMask: {statemask_result['wall_time_sec']:.2f}s, "
                f"speed-up: {speedup:.3f}x"
            )

    return results


def efficiency_from_domain(
    domain: str,
    target_path: Optional[str] = None,
    target_agent: Optional[TargetAgent] = None,
    sample_budgets: Optional[List[int]] = None,
    seed: int = 0,
    device: str = "auto",
    verbose: int = 0,
    logger: Optional[Logger] = None,
    config: Optional[DomainConfig] = None,
    **env_kwargs: Any,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Run the efficiency benchmark for a named domain.

    Either ``target_path`` or ``target_agent`` must be provided.
    """
    if config is None:
        config = get_domain_config(domain, **env_kwargs)

    if sample_budgets is None:
        sample_budgets = [config.sample_budget] if config.sample_budget else list(table_4().values())

    # Build environment.
    if domain.startswith("mujoco"):
        env_id = config.env_id or env_kwargs.get("env_id", "Hopper-v3")
        env = make_mujoco_env(
            env_id,
            sparse=config.meta.get("sparse", False),
            normalize_obs=config.target.normalize_obs if config.target else True,
            seed=seed,
            **config.env_kwargs,
        )
    else:
        raise NotImplementedError(
            f"Efficiency benchmark for domain '{domain}' is not yet implemented. "
            "Please pass a pre-built env directly to compare_efficiency()."
        )

    # Load target agent if needed.
    if target_agent is None:
        if target_path is None:
            raise ValueError("Either target_path or target_agent must be provided.")
        target_agent = load_target_agent(target_path, env, algorithm="PPO", device=device)

    rice_config = config.mask if config.mask else None
    statemask_config = config.mask if config.mask else None

    return compare_efficiency(
        env,
        target_agent,
        sample_budgets=sample_budgets,
        rice_config=rice_config,
        statemask_config=statemask_config,
        seed=seed,
        device=device,
        verbose=verbose,
        logger=logger,
    )


def log_efficiency_table(
    results: Dict[str, Dict[int, Dict[str, Any]]],
    logger: Optional[Logger] = None,
) -> str:
    """Format efficiency results as a Markdown table and optionally log them."""
    budgets = sorted(next(iter(results.values())).keys())
    lines = ["| Sample Budget | RICE (s) | StateMask (s) | Speed-up |"]
    lines.append("| --- | --- | --- | --- |")

    rice_times = []
    statemask_times = []
    for budget in budgets:
        rice_t = results["RICE"][budget]["wall_time_sec"]
        statemask_t = results["StateMask"][budget]["wall_time_sec"]
        speedup = statemask_t / rice_t if rice_t > 0 else float("nan")
        rice_times.append(rice_t)
        statemask_times.append(statemask_t)
        lines.append(
            f"| {budget:,} | {rice_t:.2f} | {statemask_t:.2f} | {speedup:.3f}x |"
        )

    avg_speedup = np.mean(
        [statemask_times[i] / rice_times[i] for i in range(len(budgets)) if rice_times[i] > 0]
    )
    lines.append("")
    lines.append(f"**Average speed-up:** {avg_speedup:.3f}x")

    table = "\n".join(lines)
    if logger is not None:
        logger.log_text("efficiency/table", table)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark mask-training efficiency (RICE vs StateMask)."
    )
    parser.add_argument("--domain", type=str, default="mujoco_hopper", help="Domain name")
    parser.add_argument("--target-path", type=str, required=True, help="Path to trained target agent")
    parser.add_argument(
        "--sample-budgets",
        type=int,
        nargs="+",
        default=None,
        help="Fixed sample budgets to benchmark",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-dir", type=str, default="results/efficiency")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = make_logger(
        args.log_dir,
        experiment_name=f"efficiency_{args.domain}_seed{args.seed}",
        use_tensorboard=True,
        use_csv=True,
        verbose=args.verbose,
    )

    results = efficiency_from_domain(
        domain=args.domain,
        target_path=args.target_path,
        sample_budgets=args.sample_budgets,
        seed=args.seed,
        device=args.device,
        verbose=int(args.verbose),
        logger=logger,
    )

    table = log_efficiency_table(results, logger=logger)
    print(table)
    logger.close()


if __name__ == "__main__":
    main()

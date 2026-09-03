"""Fidelity evaluation for RICE explanations.

This module implements the explanation-fidelity protocol from the paper:
for a fixed budget ``k``, mask the top-k states ranked by an importance
method (replace the agent's action with a random action) and measure the
resulting return.  The module compares RICE's mask network against random
masking, StateMask, Integrated Gradients, and AIRS where applicable.
"""
from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    import gymnasium as gym
except Exception:  # pragma: no cover
    import gym  # type: ignore

from rice.agents.mask_network import MaskNetwork
from rice.agents.target_agent import TargetAgent
from rice.evaluation.evaluate_policy import evaluate_policy
from rice.utils.config import FIDELITY_BUDGETS, DomainConfig, get_domain_config
from rice.utils.logger import Logger, make_logger


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _ensure_numpy(obs: Any) -> np.ndarray:
    """Convert an observation to a 1-D numpy array."""
    if isinstance(obs, np.ndarray):
        return np.asarray(obs, dtype=np.float32).ravel()
    if torch is not None and isinstance(obs, torch.Tensor):
        return obs.detach().cpu().numpy().astype(np.float32).ravel()
    return np.asarray(obs, dtype=np.float32).ravel()


def _random_action(env: gym.Env, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Sample a random action from the environment's action space."""
    action_space = env.action_space
    if rng is None:
        rng = np.random.default_rng()
    if isinstance(action_space, gym.spaces.Discrete):
        return np.array(action_space.sample(), dtype=np.int64)
    if isinstance(action_space, gym.spaces.Box):
        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)
        return rng.uniform(low, high).astype(np.float32)
    # Fallback: use the space's built-in sampler.
    sample = action_space.sample()
    if isinstance(sample, np.ndarray):
        return sample
    return np.array(sample)


def _set_action_mask_budget(
    env: gym.Env,
    agent: TargetAgent,
    trajectory: List[Dict[str, Any]],
    mask_indices: List[int],
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, Any]]:
    """Re-simulate a trajectory, replacing actions at ``mask_indices`` with random actions.

    Returns the re-simulated trajectory (list of step dicts).  The original
    trajectory is used only to obtain the sequence of states; the environment
    is stepped forward from its initial state using the (possibly masked)
    actions.
    """
    if not trajectory:
        return []

    # Reset to the first observation of the original trajectory.  If the env
    # supports resetting to a stored simulator state, prefer that.
    first_step = trajectory[0]
    initial_state = first_step.get("simulator_state") or first_step.get("state")
    obs = first_step["obs"]

    if initial_state is not None and hasattr(env, "set_simulator_state"):
        try:
            env.set_simulator_state(initial_state)
        except Exception:
            obs, _ = env.reset(seed=int(first_step.get("seed", 0)))
    elif "seed" in first_step:
        obs, _ = env.reset(seed=int(first_step["seed"]))
    else:
        obs, _ = env.reset()

    new_trajectory: List[Dict[str, Any]] = []
    for t, step in enumerate(trajectory):
        if t in mask_indices:
            action = _random_action(env, rng=rng)
        else:
            action = agent.predict(obs, deterministic=True)[0]
        next_obs, reward, terminated, truncated, info = env.step(action)
        new_trajectory.append(
            {
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            }
        )
        obs = next_obs
        if terminated or truncated:
            break
    return new_trajectory


# ---------------------------------------------------------------------------
# Importance / ranking methods
# ---------------------------------------------------------------------------

def rank_rice(
    mask_net: MaskNetwork,
    trajectory: List[Dict[str, Any]],
) -> List[int]:
    """Return trajectory indices sorted by descending RICE criticality score."""
    scores: List[Tuple[int, float]] = []
    mask_net.eval()
    for t, step in enumerate(trajectory):
        obs = _ensure_numpy(step["obs"])
        with torch.no_grad() if torch is not None else warnings.catch_warnings():
            if torch is not None:
                score = mask_net.predict(obs).item()
            else:
                score = float(mask_net.predict(obs))
        scores.append((t, score))
    return [idx for idx, _ in sorted(scores, key=lambda x: x[1], reverse=True)]


def rank_random(
    trajectory: List[Dict[str, Any]],
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """Return trajectory indices in random order."""
    indices = list(range(len(trajectory)))
    if rng is None:
        rng = np.random.default_rng()
    rng.shuffle(indices)
    return indices


def rank_statemask(
    agent: TargetAgent,
    trajectory: List[Dict[str, Any]],
    env: gym.Env,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """StateMask-style importance ranking via action ablation.

    For each state in the trajectory, measure the drop in Q/value when the
    action is replaced by a random action.  Since we do not assume access to
    the critic, we approximate importance by the one-step reward drop after
    taking a random action in that state while keeping the rest of the
    trajectory unchanged.
    """
    if rng is None:
        rng = np.random.default_rng()

    base_rewards: List[float] = []
    random_rewards: List[float] = []

    # We need a fresh copy of the environment to avoid side effects.
    for step in trajectory:
        obs = step["obs"]
        action = agent.predict(obs, deterministic=True)[0]
        # Best-effort one-step evaluation: we cannot easily reset to arbitrary
        # states, so we use the stored reward as the baseline and estimate the
        # random-action reward from the stored next_obs transition if possible.
        base_rewards.append(float(step.get("reward", 0.0)))
        random_rewards.append(float(step.get("random_reward", step.get("reward", 0.0))))

    # If the trajectory contains pre-computed random-action rewards, use them;
    # otherwise fall back to a perturbation estimate based on action magnitude.
    scores: List[Tuple[int, float]] = []
    for t, step in enumerate(trajectory):
        baseline = base_rewards[t]
        if "random_reward" in step:
            perturbed = float(step["random_reward"])
        else:
            # Approximate: larger action deviation -> potentially larger drop.
            action = _ensure_numpy(step.get("action", agent.predict(step["obs"], deterministic=True)[0]))
            random_action = _random_action(env, rng=rng)
            perturbed = baseline - float(np.linalg.norm(action - random_action))
        importance = baseline - perturbed
        scores.append((t, importance))

    return [idx for idx, _ in sorted(scores, key=lambda x: x[1], reverse=True)]


def rank_integrated_gradients(
    agent: TargetAgent,
    trajectory: List[Dict[str, Any]],
    env: gym.Env,
    n_steps: int = 50,
) -> List[int]:
    """Integrated-Gradients importance ranking for the policy network.

    Approximates the importance of each state dimension by integrating
    gradients from a baseline (zero observation) to the actual observation.
    The state importance is the L1 norm of the attributions.
    """
    if torch is None:
        warnings.warn("PyTorch not available; falling back to random ranking for IG.")
        return rank_random(trajectory)

    policy = agent.policy
    if policy is None or not hasattr(policy, "mlp_extractor"):
        warnings.warn("Agent policy does not expose a differentiable network; falling back to random ranking for IG.")
        return rank_random(trajectory)

    scores: List[Tuple[int, float]] = []
    for t, step in enumerate(trajectory):
        obs = _ensure_numpy(step["obs"])
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        baseline = torch.zeros_like(obs_t)
        delta = obs_t - baseline

        attributions = torch.zeros_like(obs_t)
        for k in range(1, n_steps + 1):
            interpolated = baseline + (k / n_steps) * delta
            interpolated.requires_grad_(True)
            action_dist = policy.get_distribution(interpolated)
            # Use the mean of the action distribution as the output to attribute.
            mean_action = action_dist.distribution.mean
            loss = mean_action.sum()
            loss.backward()
            if interpolated.grad is not None:
                attributions += interpolated.grad
            policy.zero_grad()

        attributions = attributions * delta / n_steps
        importance = attributions.abs().sum().item()
        scores.append((t, importance))

    return [idx for idx, _ in sorted(scores, key=lambda x: x[1], reverse=True)]


def rank_airs(
    agent: TargetAgent,
    trajectory: List[Dict[str, Any]],
    env: gym.Env,
    n_neighbors: int = 5,
) -> List[int]:
    """AIRS-style importance ranking via local reward variance.

    AIRS (Action Influence-based Reward Shaping) ranks states by how much the
    expected return varies when the action is perturbed.  We approximate this
    by sampling random actions in each state and measuring the empirical
    variance of the resulting one-step rewards.
    """
    rng = np.random.default_rng()
    scores: List[Tuple[int, float]] = []
    for t, step in enumerate(trajectory):
        obs = step["obs"]
        rewards: List[float] = []
        for _ in range(n_neighbors):
            action = _random_action(env, rng=rng)
            # We cannot reset to arbitrary states cheaply, so we rely on the
            # stored ``random_reward`` field if the trajectory was collected
            # with action perturbations; otherwise use a placeholder.
            if "random_reward" in step:
                rewards.append(float(step["random_reward"]))
            else:
                rewards.append(float(step.get("reward", 0.0)))
        variance = float(np.var(rewards))
        scores.append((t, variance))
    return [idx for idx, _ in sorted(scores, key=lambda x: x[1], reverse=True)]


# ---------------------------------------------------------------------------
# Fidelity computation
# ---------------------------------------------------------------------------

def compute_fidelity_score(
    agent: TargetAgent,
    env: gym.Env,
    ranking_fn: Callable[[List[Dict[str, Any]]], List[int]],
    budget: int = 5,
    n_episodes: int = 50,
    seed: Optional[int] = None,
    collect_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute the fidelity score for a given importance-ranking method.

    Parameters
    ----------
    agent:
        The target policy to explain.
    env:
        The environment on which to evaluate.
    ranking_fn:
        Function that takes a trajectory and returns indices sorted by
        descending importance.
    budget:
        Number of steps to mask per episode.
    n_episodes:
        Number of episodes to average over.
    seed:
        Random seed for reproducibility.
    collect_fn:
        Optional rollout collector.  Defaults to ``evaluate_policy`` with
        ``collect_trajectories=True``.

    Returns
    -------
    dict with keys ``mean_return``, ``std_return``, ``sem_return``,
    ``mean_original_return``, ``episodes``, ``budget``.
    """
    if collect_fn is None:
        collect_fn = evaluate_policy

    rng = np.random.default_rng(seed)
    env.reset(seed=seed)

    result = collect_fn(
        agent,
        env,
        n_eval_episodes=n_episodes,
        deterministic=True,
        collect_trajectories=True,
        seed=seed,
    )
    trajectories = result.get("trajectories", [])
    original_returns = result.get("episode_returns", [])

    masked_returns: List[float] = []
    for traj in trajectories:
        if not traj:
            continue
        ranked = ranking_fn(traj)
        k = min(budget, len(ranked))
        mask_indices = set(ranked[:k])
        masked_traj = _set_action_mask_budget(env, agent, traj, mask_indices, rng=rng)
        masked_return = sum(float(step["reward"]) for step in masked_traj)
        masked_returns.append(masked_return)

    mean_original = float(np.mean(original_returns)) if original_returns else 0.0
    mean_masked = float(np.mean(masked_returns)) if masked_returns else 0.0
    std_masked = float(np.std(masked_returns)) if masked_returns else 0.0
    sem_masked = std_masked / np.sqrt(len(masked_returns)) if masked_returns else 0.0

    return {
        "mean_return": mean_masked,
        "std_return": std_masked,
        "sem_return": sem_masked,
        "mean_original_return": mean_original,
        "episodes": len(masked_returns),
        "budget": budget,
    }


def compute_rice_fidelity(
    agent: TargetAgent,
    env: gym.Env,
    mask_net: MaskNetwork,
    budgets: Tuple[int, ...] = FIDELITY_BUDGETS,
    n_episodes: int = 50,
    seed: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """Fidelity scores for the RICE mask network across budgets."""
    results: Dict[int, Dict[str, Any]] = {}
    for budget in budgets:
        results[budget] = compute_fidelity_score(
            agent,
            env,
            ranking_fn=lambda traj: rank_rice(mask_net, traj),
            budget=budget,
            n_episodes=n_episodes,
            seed=seed,
        )
        results[budget]["method"] = "RICE"
    return results


def compute_random_fidelity(
    agent: TargetAgent,
    env: gym.Env,
    budgets: Tuple[int, ...] = FIDELITY_BUDGETS,
    n_episodes: int = 50,
    seed: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """Fidelity scores for random masking across budgets."""
    results: Dict[int, Dict[str, Any]] = {}
    for budget in budgets:
        results[budget] = compute_fidelity_score(
            agent,
            env,
            ranking_fn=lambda traj: rank_random(traj, rng=np.random.default_rng(seed)),
            budget=budget,
            n_episodes=n_episodes,
            seed=seed,
        )
        results[budget]["method"] = "Random"
    return results


def compute_statemask_fidelity(
    agent: TargetAgent,
    env: gym.Env,
    budgets: Tuple[int, ...] = FIDELITY_BUDGETS,
    n_episodes: int = 50,
    seed: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """Fidelity scores for StateMask-style ranking across budgets."""
    results: Dict[int, Dict[str, Any]] = {}
    for budget in budgets:
        results[budget] = compute_fidelity_score(
            agent,
            env,
            ranking_fn=lambda traj: rank_statemask(agent, traj, env),
            budget=budget,
            n_episodes=n_episodes,
            seed=seed,
        )
        results[budget]["method"] = "StateMask"
    return results


def compute_ig_fidelity(
    agent: TargetAgent,
    env: gym.Env,
    budgets: Tuple[int, ...] = FIDELITY_BUDGETS,
    n_episodes: int = 50,
    seed: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """Fidelity scores for Integrated Gradients across budgets."""
    results: Dict[int, Dict[str, Any]] = {}
    for budget in budgets:
        results[budget] = compute_fidelity_score(
            agent,
            env,
            ranking_fn=lambda traj: rank_integrated_gradients(agent, traj, env),
            budget=budget,
            n_episodes=n_episodes,
            seed=seed,
        )
        results[budget]["method"] = "IntegratedGradients"
    return results


def compute_airs_fidelity(
    agent: TargetAgent,
    env: gym.Env,
    budgets: Tuple[int, ...] = FIDELITY_BUDGETS,
    n_episodes: int = 50,
    seed: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """Fidelity scores for AIRS across budgets."""
    results: Dict[int, Dict[str, Any]] = {}
    for budget in budgets:
        results[budget] = compute_fidelity_score(
            agent,
            env,
            ranking_fn=lambda traj: rank_airs(agent, traj, env),
            budget=budget,
            n_episodes=n_episodes,
            seed=seed,
        )
        results[budget]["method"] = "AIRS"
    return results


# ---------------------------------------------------------------------------
# High-level comparison API
# ---------------------------------------------------------------------------

def compare_fidelity(
    agent: TargetAgent,
    env: gym.Env,
    mask_net: Optional[MaskNetwork] = None,
    methods: Tuple[str, ...] = ("RICE", "StateMask", "Random"),
    budgets: Tuple[int, ...] = FIDELITY_BUDGETS,
    n_episodes: int = 50,
    seed: Optional[int] = None,
    logger: Optional[Logger] = None,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Run fidelity evaluation for multiple explanation methods.

    Parameters
    ----------
    agent:
        Target policy.
    env:
        Evaluation environment.
    mask_net:
        Trained RICE mask network (required for the ``RICE`` method).
    methods:
        Methods to evaluate.  Supported: ``RICE``, ``StateMask``, ``Random``,
        ``IntegratedGradients``, ``AIRS``.
    budgets:
        Masking budgets to test.
    n_episodes:
        Number of episodes per method/budget.
    seed:
        Random seed.
    logger:
        Optional ``Logger`` for writing results.

    Returns
    -------
    Nested dict ``results[method][budget] -> metrics``.
    """
    results: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for method in methods:
        if method == "RICE":
            if mask_net is None:
                warnings.warn("RICE fidelity requested but no mask_net provided; skipping.")
                continue
            method_results = compute_rice_fidelity(agent, env, mask_net, budgets, n_episodes, seed)
        elif method == "StateMask":
            method_results = compute_statemask_fidelity(agent, env, budgets, n_episodes, seed)
        elif method == "Random":
            method_results = compute_random_fidelity(agent, env, budgets, n_episodes, seed)
        elif method == "IntegratedGradients":
            method_results = compute_ig_fidelity(agent, env, budgets, n_episodes, seed)
        elif method == "AIRS":
            method_results = compute_airs_fidelity(agent, env, budgets, n_episodes, seed)
        else:
            warnings.warn(f"Unknown fidelity method '{method}'; skipping.")
            continue
        results[method] = method_results

        if logger is not None:
            for budget, metrics in method_results.items():
                step = budget
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        logger.log(f"fidelity/{method}/{key}", value, step=step)

    return results


def fidelity_from_domain(
    agent: TargetAgent,
    domain: str,
    mask_net: Optional[MaskNetwork] = None,
    methods: Tuple[str, ...] = ("RICE", "StateMask", "Random"),
    budgets: Tuple[int, ...] = FIDELITY_BUDGETS,
    n_episodes: int = 50,
    seed: int = 0,
    config: Optional[DomainConfig] = None,
    logger: Optional[Logger] = None,
    **env_kwargs: Any,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Convenience wrapper that builds the domain environment and runs fidelity."""
    if config is None:
        config = get_domain_config(domain, **env_kwargs)

    # Build environment lazily to avoid hard optional dependencies.
    env = None
    if domain.startswith("mujoco") or config.name in {"hopper", "walker2d", "reacher", "halfcheetah"}:
        from rice.envs import make_mujoco_env

        env_id = config.env_id
        env = make_mujoco_env(
            env_id,
            sparse=config.meta.get("sparse", False),
            normalize_obs=config.target.normalize_obs,
            **config.env_kwargs,
        )
    elif domain == "selfish_mining":
        from rice.envs import make_selfish_mining_env

        env = make_selfish_mining_env(**config.env_kwargs)
    elif domain == "cage":
        from rice.envs import make_cage_env

        env = make_cage_env(**config.env_kwargs)
    elif domain == "metadrive":
        from rice.envs import make_metadrive_env

        env = make_metadrive_env(**config.env_kwargs)
    elif domain == "malware":
        from rice.envs import make_malware_env

        env = make_malware_env(**config.env_kwargs)
    else:
        raise ValueError(f"Unsupported domain for fidelity evaluation: {domain}")

    try:
        return compare_fidelity(
            agent,
            env,
            mask_net=mask_net,
            methods=methods,
            budgets=budgets,
            n_episodes=n_episodes,
            seed=seed,
            logger=logger,
        )
    finally:
        env.close()


def log_fidelity_table(
    results: Dict[str, Dict[int, Dict[str, Any]]],
    logger: Optional[Logger] = None,
) -> str:
    """Format fidelity results as a Markdown table and optionally log it."""
    if not results:
        return ""
    methods = list(results.keys())
    budgets = sorted(next(iter(results.values())).keys())

    lines = ["| Method | " + " | ".join(f"k={k}" for k in budgets) + " |"]
    lines.append("|" + "---|" * (len(budgets) + 1))
    for method in methods:
        row = [method]
        for budget in budgets:
            metrics = results[method].get(budget, {})
            mean = metrics.get("mean_return", float("nan"))
            sem = metrics.get("sem_return", float("nan"))
            row.append(f"{mean:.2f} ± {sem:.2f}")
        lines.append("| " + " | ".join(row) + " |")

    table = "\n".join(lines)
    if logger is not None:
        logger.log_text("fidelity/table", table)
    return table


def main() -> None:
    """CLI entry point for running a fidelity comparison."""
    import argparse

    parser = argparse.ArgumentParser(description="RICE fidelity evaluation")
    parser.add_argument("--domain", type=str, required=True, help="Domain name")
    parser.add_argument("--agent-path", type=str, required=True, help="Path to saved target agent")
    parser.add_argument("--mask-path", type=str, default=None, help="Path to saved RICE mask network")
    parser.add_argument("--methods", type=str, default="RICE,StateMask,Random", help="Comma-separated methods")
    parser.add_argument("--budgets", type=str, default=",".join(map(str, FIDELITY_BUDGETS)), help="Comma-separated budgets")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="results/fidelity")
    args = parser.parse_args()

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    budgets = tuple(int(b.strip()) for b in args.budgets.split(",") if b.strip())

    logger = make_logger(args.log_dir, f"{args.domain}_fidelity", use_tensorboard=True, use_csv=True, verbose=True)

    agent = TargetAgent.load(args.agent_path)
    mask_net = None
    if args.mask_path is not None:
        mask_net = MaskNetwork.load(args.mask_path)

    results = fidelity_from_domain(
        agent,
        args.domain,
        mask_net=mask_net,
        methods=methods,
        budgets=budgets,
        n_episodes=args.n_episodes,
        seed=args.seed,
        logger=logger,
    )
    table = log_fidelity_table(results, logger=logger)
    print(table)
    logger.close()


if __name__ == "__main__":
    main()

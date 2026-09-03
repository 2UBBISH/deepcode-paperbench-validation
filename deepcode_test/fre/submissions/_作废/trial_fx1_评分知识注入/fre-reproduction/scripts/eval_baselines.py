"""Evaluate zero-shot offline RL baselines on downstream tasks.

This script mirrors the FRE evaluation protocol for the paper's baseline
methods:

* Forward-Backward (FB): infer a task representation using 5120 reward samples
  and ridge regression, then evaluate the task-conditioned policy.
* Successor Features (SF): infer a linear reward vector over learned features
  using 5120 reward samples, then evaluate.
* Goal-Conditioned IQL (GC-IQL): choose a goal from a small set of states
  labelled by the downstream reward, then evaluate the goal-conditioned policy.
* Goal-Conditioned Behavioral Cloning (GC-BC): same goal selection as GC-IQL.
* OPAL: privileged evaluation -- sample a set of latent skills, run each
  online, and report the best score.

The script is intentionally defensive about baseline method signatures because
the baseline classes may be implemented with slightly different public APIs.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Ensure the repository root is importable when run directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fre.utils import get_logger, resolve_device, set_seed  # noqa: E402
from fre.dataset import (  # noqa: E402
    build_state_pool,
    load_offline_dataset,
    make_synthetic_dataset,
)

from envs.antmaze_wrapper import (  # noqa: E402
    ANTMAZE_TASKS,
    evaluate_antmaze_policy,
    sample_task_reward_states as sample_antmaze_states,
)
from envs.kitchen_wrapper import (  # noqa: E402
    KITCHEN_TASKS,
    evaluate_kitchen_policy,
    sample_task_reward_states as sample_kitchen_states,
)
from envs.exorl_wrapper import (  # noqa: E402
    EXORL_TASKS,
    evaluate_exorl_policy,
    sample_task_reward_states as sample_exorl_states,
)


# ---------------------------------------------------------------------------
# Domain/task helpers
# ---------------------------------------------------------------------------

def resolve_tasks(domain: str, task_override: Optional[List[str]] = None) -> List[str]:
    """Return the downstream task names for *domain*."""
    if task_override:
        return list(task_override)

    domain = domain.lower()
    if domain in ("antmaze", "antmaze-large-diverse-v2"):
        return list(ANTMAZE_TASKS)
    if domain in ("kitchen", "kitchen-complete-v0"):
        return list(KITCHEN_TASKS)
    if domain in ("walker", "cheetah", "exorl"):
        # EXORL_TASKS contains walker/cheetah tasks; keep those matching the domain.
        tasks = list(EXORL_TASKS)
        if domain == "walker":
            return [t for t in tasks if t.startswith("walker")]
        if domain == "cheetah":
            return [t for t in tasks if t.startswith("cheetah")]
        return tasks
    raise ValueError(f"Unknown domain: {domain}")


def sample_task_pairs(
    domain: str,
    task_name: str,
    state_pool: np.ndarray,
    num_examples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample state-reward pairs for a downstream task.

    For FRE these are exactly 32 examples; for goal selection and reward
    regression we often request more (e.g. 5120).
    """
    domain = domain.lower()
    if domain in ("antmaze", "antmaze-large-diverse-v2"):
        return sample_antmaze_states(
            task_name, state_pool, num_examples=num_examples, seed=seed
        )
    if domain in ("kitchen", "kitchen-complete-v0"):
        return sample_kitchen_states(
            task_name, state_pool, num_examples=num_examples, seed=seed
        )
    if domain in ("walker", "cheetah", "exorl"):
        return sample_exorl_states(
            task_name, state_pool, num_examples=num_examples, seed=seed
        )
    raise ValueError(f"Unknown domain: {domain}")


def evaluate_task(
    domain: str,
    task_name: str,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    num_episodes: int,
    seed: int,
    env_name: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate *policy_fn* on *task_name* in *domain*."""
    domain = domain.lower()
    if domain in ("antmaze", "antmaze-large-diverse-v2"):
        return evaluate_antmaze_policy(
            policy_fn,
            task_name=task_name,
            num_episodes=num_episodes,
            seed=seed,
            env_name=env_name or "antmaze-large-diverse-v2",
        )
    if domain in ("kitchen", "kitchen-complete-v0"):
        return evaluate_kitchen_policy(
            policy_fn,
            task_name=task_name,
            num_episodes=num_episodes,
            seed=seed,
            env_name=env_name or "kitchen-complete-v0",
        )
    if domain in ("walker", "cheetah", "exorl"):
        return evaluate_exorl_policy(
            policy_fn,
            task_name=task_name,
            domain=domain,
            num_episodes=num_episodes,
            seed=seed,
        )
    raise ValueError(f"Unknown domain: {domain}")


def extract_score(result: Dict[str, float]) -> float:
    """Extract the normalized score from an evaluator result dict."""
    if not result:
        return 0.0
    for key in ("normalized_score", "score", "mean_return", "success_rate"):
        if key in result:
            return float(result[key])
    # Fall back to the first scalar value.
    for value in result.values():
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


# ---------------------------------------------------------------------------
# Baseline loading
# ---------------------------------------------------------------------------

def load_baseline(baseline_name: str, checkpoint: str, state_dim: int, action_dim: int, device: torch.device) -> Any:
    """Instantiate and load a baseline agent from a checkpoint."""
    from baselines.fb import FB
    from baselines.sf import SF
    from baselines.gc_iql import GCIQL
    from baselines.gc_bc import GCBC
    from baselines.opal import OPAL

    baseline_name = baseline_name.lower().replace("-", "_")
    if baseline_name in ("fb", "forward_backward"):
        agent = FB(state_dim=state_dim, action_dim=action_dim, device=device)
    elif baseline_name in ("sf", "successor_features"):
        agent = SF(state_dim=state_dim, action_dim=action_dim, device=device)
    elif baseline_name in ("gc_iql", "gciql", "goal_conditioned_iql"):
        agent = GCIQL(state_dim=state_dim, action_dim=action_dim, device=device)
    elif baseline_name in ("gc_bc", "gcbc", "goal_conditioned_bc"):
        agent = GCBC(state_dim=state_dim, action_dim=action_dim, device=device)
    elif baseline_name in ("opal",):
        agent = OPAL(state_dim=state_dim, action_dim=action_dim, device=device)
    else:
        raise ValueError(f"Unknown baseline name: {baseline_name}")

    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    if isinstance(state_dict, dict) and "agent" in state_dict:
        state_dict = state_dict["agent"]
    if hasattr(agent, "load_state_dict"):
        agent.load_state_dict(state_dict)
    elif hasattr(agent, "load"):
        agent.load(checkpoint)
    else:
        raise RuntimeError(f"Baseline {baseline_name} has no load method")
    agent.to(device)
    return agent


# ---------------------------------------------------------------------------
# Task-context inference
# ---------------------------------------------------------------------------

def _call_with_kwargs(func: Callable, **kwargs: Any) -> Any:
    """Call *func* with only the kwargs it accepts."""
    sig = inspect.signature(func)
    accepted = {
        k: v for k, v in kwargs.items()
        if k in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    }
    return func(**accepted)


def infer_task_policy(
    baseline: Any,
    baseline_name: str,
    reward_fn: Callable[[np.ndarray], np.ndarray],
    state_pool: np.ndarray,
    num_reward_samples: int = 5120,
    seed: int = 0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Infer a task-conditioned policy for a baseline.

    The exact inference mechanism depends on the baseline family. The function
    first attempts the most likely API for each family and then falls back to
    common alternative signatures.
    """
    baseline_name = baseline_name.lower().replace("-", "_")

    # OPAL has privileged skill selection; handled outside this function.
    if baseline_name in ("opal",):
        raise ValueError("OPAL should be evaluated with _evaluate_opal")

    # Goal-conditioned methods: pick a goal state using the reward labels.
    if baseline_name in ("gc_iql", "gciql", "goal_conditioned_iql", "gc_bc", "gcbc", "goal_conditioned_bc"):
        states, rewards = sample_task_pairs(
            "antmaze" if "ant" in baseline_name else "exorl",
            "", state_pool, num_examples=max(32, num_reward_samples), seed=seed,
        )
        # More robust: directly evaluate reward_fn on sampled states to choose goal.
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(state_pool), size=min(len(state_pool), 5120), replace=False)
        sampled = state_pool[indices]
        values = np.asarray(reward_fn(sampled), dtype=np.float32)
        if values.ndim == 0:
            values = np.full(len(sampled), float(values), dtype=np.float32)
        goal = sampled[int(np.argmax(values))]
        goal = np.asarray(goal, dtype=np.float32)
        return baseline.get_task_policy(goal, deterministic=True)

    # FB and SF: use reward regression over learned features.
    # The exact call is version-dependent; try several signatures.
    common_kwargs = dict(
        reward_fn=reward_fn,
        state_pool=state_pool,
        num_samples=num_reward_samples,
        seed=seed,
        ridge=1e-3,
    )

    # Try a single end-to-end get_task_policy first.
    if hasattr(baseline, "get_task_policy"):
        try:
            policy = _call_with_kwargs(baseline.get_task_policy, **common_kwargs)
            if policy is not None:
                return policy
        except (TypeError, ValueError, RuntimeError):
            pass

    # Try get_task_vector then get_task_policy(task_vector).
    if hasattr(baseline, "get_task_vector"):
        try:
            task_vector = _call_with_kwargs(baseline.get_task_vector, **common_kwargs)
        except (TypeError, ValueError, RuntimeError):
            task_vector = None
        if task_vector is not None:
            if hasattr(baseline, "get_task_policy"):
                try:
                    policy = baseline.get_task_policy(task_vector, deterministic=True)
                    return policy
                except TypeError:
                    policy = baseline.get_task_policy(task_vector)
                    return policy
            else:
                raise RuntimeError("Baseline produced a task vector but has no get_task_policy")

    # If we reach this point, infer reward vector manually using baseline utilities
    # (last-resort for custom APIs).
    if baseline_name in ("fb", "forward_backward"):
        from baselines.baseline_utils import sample_reward_pairs, ridge_regression
        states, rewards = sample_reward_pairs(
            reward_fn, state_pool, num_samples=num_reward_samples, seed=seed
        )
        states_t = torch.as_tensor(states, dtype=torch.float32, device=baseline.device)
        # Forward-Backward task vector is a linear combination of B(s).
        # We approximate by regressing reward on B(s).
        with torch.no_grad():
            B = baseline.backward(states_t).detach().cpu().numpy()
        w = ridge_regression(B, rewards, ridge=1e-3)
        w_t = torch.as_tensor(w, dtype=torch.float32, device=baseline.device)
        # Try to create policy from task vector. FB may store a method to build policy.
        if hasattr(baseline, "get_task_policy_from_vector"):
            return baseline.get_task_policy_from_vector(w_t)
        # Otherwise use FB's policy with a custom z parameter if get_task_policy accepts a vector.
        try:
            return baseline.get_task_policy(w_t, deterministic=True)
        except TypeError:
            raise RuntimeError("Unable to construct FB policy; inspect FB.get_task_policy signature")

    raise RuntimeError(f"Unable to infer task policy for baseline {baseline_name}")


def _evaluate_opal(
    baseline: Any,
    domain: str,
    tasks: List[str],
    num_episodes: int,
    seed: int,
    num_skills: int = 10,
    env_name: Optional[str] = None,
) -> Dict[str, float]:
    """Privileged OPAL evaluation: evaluate several skills, keep the best."""
    skills = baseline.sample_skills(num_skills=num_skills, seed=seed)
    scores: Dict[str, float] = {}
    for task in tasks:
        best = -float("inf")
        for skill in skills:
            skill = torch.as_tensor(skill, dtype=torch.float32, device=baseline.device)
            policy_fn = baseline.get_task_policy(skill, deterministic=True)
            result = evaluate_task(
                domain, task, policy_fn, num_episodes, seed, env_name=env_name
            )
            score = extract_score(result)
            best = max(best, score)
        scores[task] = best
    return scores


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a zero-shot offline RL baseline on downstream tasks."
    )
    parser.add_argument("--baseline", type=str, required=True,
                        choices=["fb", "sf", "gc_iql", "gc_bc", "opal"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--state_pool_size", type=int, default=None)
    parser.add_argument("--num_reward_samples", type=int, default=5120,
                        help="Number of reward samples for FB/SF regression or goal selection.")
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--num_skills", type=int, default=10,
                        help="Number of OPAL skills for privileged evaluation.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="baseline_eval_results")
    parser.add_argument("--task_override", type=str, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0,
                        help="Base RNG seed for sampling/evaluation.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    set_seed(args.seed)
    logger = get_logger("eval_baselines")

    # Load dataset/state pool.
    if args.domain == "synthetic":
        dataset = make_synthetic_dataset()
        state_pool = build_state_pool(dataset, args.state_pool_size)
        state_dim = dataset.state_dim
        action_dim = dataset.action_dim
    else:
        dataset = load_offline_dataset(
            args.domain,
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
        )
        state_pool = build_state_pool(dataset, args.state_pool_size)
        state_dim = getattr(dataset, "state_dim", None)
        action_dim = getattr(dataset, "action_dim", None)
        if state_dim is None or action_dim is None:
            if hasattr(dataset, "states") and hasattr(dataset, "actions"):
                state_dim = dataset.states.shape[-1]
                action_dim = dataset.actions.shape[-1]
            else:
                from fre.config import get_domain_dims
                state_dim, action_dim = get_domain_dims(args.domain)

    baseline = load_baseline(
        args.baseline, args.checkpoint, state_dim, action_dim, device
    )
    tasks = resolve_tasks(args.domain, args.task_override)

    os.makedirs(args.output_dir, exist_ok=True)
    results: Dict[str, float] = {}
    all_seed_scores: Dict[str, List[float]] = {task: [] for task in tasks}

    for eval_seed in args.seeds:
        seed_scores: Dict[str, float] = {}
        for task in tasks:
            # Build a callable reward function for this task.
            states, rewards = sample_task_pairs(
                args.domain, task, state_pool, num_examples=args.num_reward_samples,
                seed=eval_seed,
            )

            def reward_fn(s: np.ndarray, _task=task) -> np.ndarray:
                # We cannot easily reuse the sampled labels for arbitrary s; instead
                # construct a task reward via env wrapper factory.
                # Dispatch to domain-specific factories.
                domain = args.domain.lower()
                if domain in ("antmaze", "antmaze-large-diverse-v2"):
                    from envs.antmaze_wrapper import make_antmaze_task_reward
                    return make_antmaze_task_reward(_task)(s)
                if domain in ("kitchen", "kitchen-complete-v0"):
                    from envs.kitchen_wrapper import make_kitchen_task_reward
                    return make_kitchen_task_reward(_task)(s)
                if domain in ("walker", "cheetah", "exorl"):
                    from envs.exorl_wrapper import make_exorl_task_reward
                    return make_exorl_task_reward(_task)(s)
                raise ValueError(f"Unknown domain {domain}")

            if args.baseline.lower() in ("opal",):
                # OPAL is handled separately per task in _evaluate_opal; skip individual loop.
                continue

            policy_fn = infer_task_policy(
                baseline,
                args.baseline,
                reward_fn,
                state_pool,
                num_reward_samples=args.num_reward_samples,
                seed=eval_seed,
            )
            result = evaluate_task(
                args.domain, task, policy_fn, args.num_episodes, eval_seed,
                env_name=args.dataset_name,
            )
            score = extract_score(result)
            seed_scores[task] = score
            all_seed_scores[task].append(score)

        if args.baseline.lower() in ("opal",):
            seed_scores = _evaluate_opal(
                baseline,
                args.domain,
                tasks,
                args.num_episodes,
                eval_seed,
                num_skills=args.num_skills,
                env_name=args.dataset_name,
            )
            for task, score in seed_scores.items():
                all_seed_scores[task].append(score)

        results[f"seed_{eval_seed}"] = seed_scores
        logger.info("Seed %d: %s", eval_seed, seed_scores)

    # Aggregate mean/std.
    aggregate: Dict[str, Any] = {}
    for task in tasks:
        scores = all_seed_scores.get(task, [])
        if scores:
            aggregate[task] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "scores": scores,
            }
        else:
            aggregate[task] = {"mean": 0.0, "std": 0.0, "scores": []}
    aggregate["mean_all"] = float(np.mean(
        [aggregate[t]["mean"] for t in tasks if aggregate[t]["scores"]]
    )) if tasks else 0.0

    out_path = os.path.join(
        args.output_dir,
        f"{args.baseline}_{args.domain}_eval.json",
    )
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results, "aggregate": aggregate}, f, indent=2)
    logger.info("Saved evaluation results to %s", out_path)
    logger.info("Aggregate: %s", {k: v for k, v in aggregate.items() if k != "scores"})

    # Print a compact table.
    for task in tasks:
        agg = aggregate[task]
        print(f"{task:35s} {agg['mean']:6.2f} +/- {agg['std']:6.2f}")
    print(f"{'ALL':35s} {aggregate['mean_all']:6.2f}")


if __name__ == "__main__":
    main()

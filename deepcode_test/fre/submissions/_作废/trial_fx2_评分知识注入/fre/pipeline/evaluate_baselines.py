"""Unified evaluation utilities for baseline methods.

This module provides a single evaluation entry point for all comparison methods
in the FRE paper:

* Goal-conditioned IQL (GC-IQL)
* Goal-conditioned behavioral cloning (GC-BC)
* OPAL with privileged online skill selection
* Forward-Backward (FB)
* Successor Features (SF)

The goal is to evaluate every baseline under the same downstream tasks and with
the same evaluation protocol as FRE (20 episodes, deterministic actors, and
normalized scores between 0 and 100 where applicable).  FB and SF follow the
paper's convention of fitting a reward model from a large number of reward
samples (5120 by default), while OPAL uses privileged skill selection over 10
sampled skills.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import torch

from fre.config import (
    ALL_TASKS,
    ANTMAZE_TASKS,
    EXORL_TASKS,
    KITCHEN_TASKS,
    Config,
    get_config,
    resolve_device,
)
from fre.data.dataset import OfflineDataset
from fre.pipeline.evaluate import (
    TaskReward,
    make_eval_env,
    make_task_reward,
    rollout_task,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _cfg_value(cfg: Any, dotted_path: str, default: Any = None) -> Any:
    """Read a nested config value defensively, e.g. ``eval.num_episodes``."""
    obj: Any = cfg
    for part in dotted_path.split("."):
        if obj is None:
            return default
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            return default
    return obj if obj is not None else default


def _get_attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _to_tensor(value: Any, device: torch.device | str = "cpu") -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


class _FixedConditionAgent:
    """Wrap a policy/agent with a fixed conditioning vector."""

    def __init__(self, agent: Any, condition: torch.Tensor):
        self.agent = agent
        self.condition = condition

    def get_action(
        self,
        state: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        # Most of our agents use the ``condition`` keyword; a few may call it
        # ``goal``, ``z``, or ``skill``.  Try them all defensively.
        try:
            return self.agent.get_action(
                state, condition=self.condition, deterministic=deterministic
            )
        except TypeError:
            pass

        for kw in ("goal", "z", "skill"):
            try:
                return self.agent.get_action(
                    state, **{kw: self.condition}, deterministic=deterministic
                )
            except TypeError:
                continue
        raise TypeError(
            "Wrapped agent does not expose a supported get_action signature"
        )


class _CallablePolicyAdapter:
    """Adapt a raw ``state -> action`` callable to the rollout interface."""

    def __init__(self, policy_fn: Callable[[torch.Tensor], torch.Tensor]):
        self._policy_fn = policy_fn

    def get_action(
        self,
        state: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        del condition, deterministic
        return self._policy_fn(state)


# ---------------------------------------------------------------------------
# Task/goal helpers
# ---------------------------------------------------------------------------

def _extract_task_goal(task_reward: TaskReward) -> Optional[Any]:
    """Try to recover the goal vector stored by a task reward object."""
    candidates = (
        "goal",
        "goal_state",
        "goal_pos",
        "goal_position",
        "target",
        "target_state",
        "desired_goal",
        "goal_obs",
        "goal_observation",
    )
    goal = _get_attr(task_reward, *candidates)
    if goal is None and hasattr(task_reward, "fn"):
        goal = _get_attr(task_reward.fn, *candidates)
    return goal


def _goal_for_task(
    task_reward: TaskReward,
    dataset: OfflineDataset,
    device: torch.device | str,
) -> torch.Tensor:
    """Return a normalized goal vector for goal-conditioned baselines.

    If the task reward does not expose an explicit goal, fall back to a
    zero vector (which corresponds to the canonical AntMaze target and is a
    reasonable default for many goal-reaching domains).
    """
    raw_goal = _extract_task_goal(task_reward)
    state_dim = int(dataset.states.shape[-1])

    if raw_goal is None:
        goal = torch.zeros(state_dim, device=device)
    else:
        goal = _to_tensor(raw_goal, device)
        if goal.dim() == 0:
            goal = goal.reshape(1)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)
        if goal.shape[-1] != state_dim:
            # Last-resort padding/truncation for unusual task definitions.
            if goal.shape[-1] < state_dim:
                padding = torch.zeros(
                    goal.shape[:-1] + (state_dim - goal.shape[-1],), device=device
                )
                goal = torch.cat([goal, padding], dim=-1)
            else:
                goal = goal[..., :state_dim]
        # Normalize raw goals into the dataset's normalized state space.
        try:
            goal = dataset.normalize_states(goal)
        except Exception:  # noqa: BLE001 - normalization is best-effort
            pass
        goal = goal[0]
    return goal.reshape(-1)


def _resolve_task_reward(
    cfg: Config,
    dataset: OfflineDataset,
    task_name: str,
    device: torch.device | str,
    seed: int,
) -> TaskReward:
    return make_task_reward(task_name, dataset, device=device, seed=seed)


# ---------------------------------------------------------------------------
# GC-IQL / GC-BC evaluation
# ---------------------------------------------------------------------------

def evaluate_gc_agent(
    agent: Any,
    dataset: OfflineDataset,
    env: Any,
    task_reward: TaskReward,
    device: torch.device | str,
    num_episodes: int = 20,
    max_episode_steps: int = 1000,
    seed: int = 0,
    deterministic: bool = True,
) -> Dict[str, float]:
    """Evaluate a goal-conditioned agent (GC-IQL or GC-BC)."""
    goal = _goal_for_task(task_reward, dataset, device)
    return rollout_task(
        env,
        dataset,
        agent,
        task_reward,
        goal,
        device,
        num_episodes=num_episodes,
        max_episode_steps=max_episode_steps,
        seed=seed,
        deterministic=deterministic,
    )


# ---------------------------------------------------------------------------
# FB / SF evaluation
# ---------------------------------------------------------------------------

def _resolve_regression_policy(
    agent: Any,
    dataset: OfflineDataset,
    task_reward: TaskReward,
    num_reward_samples: int,
    device: torch.device | str,
    seed: int,
) -> Any:
    """Fit a task-specific policy for FB/SF-style reward-regression methods.

    Different baseline adapters may expose different interfaces.  This helper
    tries, in order: ``solve_task``, ``infer_policy``, and
    ``fit_reward_model``.
    """
    if hasattr(agent, "solve_task"):
        return agent.solve_task(
            task_reward,
            dataset=dataset,
            num_reward_samples=num_reward_samples,
            device=device,
            seed=seed,
        )

    states = dataset.sample_states(num_reward_samples, seed=seed).to(device)
    rewards = task_reward(states)

    if hasattr(agent, "infer_policy"):
        try:
            return agent.infer_policy(states, rewards)
        except TypeError:
            return agent.infer_policy(states, rewards, device=device)

    if hasattr(agent, "fit_reward_model"):
        agent.fit_reward_model(states, rewards)
        return agent

    # If the agent itself can be called as a universal policy, just return it.
    return agent


def _coerce_to_rollout_agent(policy_or_agent: Any) -> Any:
    if policy_or_agent is None:
        raise ValueError("Reward-regression baseline returned no policy/agent")
    if hasattr(policy_or_agent, "get_action"):
        return policy_or_agent
    if callable(policy_or_agent):
        return _CallablePolicyAdapter(policy_or_agent)
    raise TypeError(
        "Unsupported policy object returned by reward-regression baseline: "
        f"{type(policy_or_agent)!r}"
    )


def evaluate_regression_agent(
    agent: Any,
    dataset: OfflineDataset,
    env: Any,
    task_reward: TaskReward,
    device: torch.device | str,
    num_episodes: int = 20,
    max_episode_steps: int = 1000,
    seed: int = 0,
    num_reward_samples: int = 5120,
    deterministic: bool = True,
) -> Dict[str, float]:
    """Evaluate FB or SF by first fitting a task-specific policy."""
    policy_or_agent = _resolve_regression_policy(
        agent,
        dataset,
        task_reward,
        num_reward_samples,
        device,
        seed,
    )
    rollout_agent = _coerce_to_rollout_agent(policy_or_agent)
    # The universal policy is already task-conditioned; pass a dummy zero
    # condition (or no condition) to the rollout helper.
    dummy_condition = torch.zeros(
        _cfg_value(dataset, "states.shape[-1]", 0) or 1, device=device
    )
    return rollout_task(
        env,
        dataset,
        rollout_agent,
        task_reward,
        dummy_condition,
        device,
        num_episodes=num_episodes,
        max_episode_steps=max_episode_steps,
        seed=seed,
        deterministic=deterministic,
    )


def evaluate_fb_agent(
    agent: Any,
    dataset: OfflineDataset,
    env: Any,
    task_reward: TaskReward,
    device: torch.device | str,
    num_episodes: int = 20,
    max_episode_steps: int = 1000,
    seed: int = 0,
    num_reward_samples: int = 5120,
    deterministic: bool = True,
) -> Dict[str, float]:
    """Forward-Backward baseline evaluation (5120 reward samples)."""
    return evaluate_regression_agent(
        agent,
        dataset,
        env,
        task_reward,
        device,
        num_episodes=num_episodes,
        max_episode_steps=max_episode_steps,
        seed=seed,
        num_reward_samples=num_reward_samples,
        deterministic=deterministic,
    )


def evaluate_sf_agent(
    agent: Any,
    dataset: OfflineDataset,
    env: Any,
    task_reward: TaskReward,
    device: torch.device | str,
    num_episodes: int = 20,
    max_episode_steps: int = 1000,
    seed: int = 0,
    num_reward_samples: int = 5120,
    deterministic: bool = True,
) -> Dict[str, float]:
    """Successor Features baseline evaluation (5120 reward samples)."""
    return evaluate_regression_agent(
        agent,
        dataset,
        env,
        task_reward,
        device,
        num_episodes=num_episodes,
        max_episode_steps=max_episode_steps,
        seed=seed,
        num_reward_samples=num_reward_samples,
        deterministic=deterministic,
    )


# ---------------------------------------------------------------------------
# OPAL privileged-skill evaluation
# ---------------------------------------------------------------------------

def _sample_skill(agent: Any, device: torch.device | str) -> torch.Tensor:
    if hasattr(agent, "sample_skill"):
        try:
            skill = agent.sample_skill(device=device)
        except TypeError:
            skill = agent.sample_skill()
        if isinstance(skill, torch.Tensor):
            return skill.reshape(-1).to(device)
    if hasattr(agent, "sample_z"):
        try:
            z = agent.sample_z(device=device)
        except TypeError:
            z = agent.sample_z()
        if isinstance(z, torch.Tensor):
            return z.reshape(-1).to(device)

    skill_dim = int(
        _get_attr(agent, "skill_dim", "z_dim", "latent_dim", default=64)
    )
    return torch.randn(skill_dim, device=device)


def evaluate_opal_agent(
    agent: Any,
    dataset: OfflineDataset,
    env: Any,
    task_reward: TaskReward,
    device: torch.device | str,
    num_episodes: int = 20,
    max_episode_steps: int = 1000,
    seed: int = 0,
    num_skills: int = 10,
    num_selection_episodes: int = 1,
    deterministic: bool = True,
) -> Dict[str, float]:
    """Evaluate OPAL with privileged online skill selection.

    The paper reports OPAL with privileged skill selection: roll out multiple
    skills online for each downstream task and select the highest-performing
    skill.  Here, ``num_skills`` candidate skills are evaluated with a small
    number of selection episodes and the best is re-evaluated for the full
    reporting budget.
    """
    best_skill: Optional[torch.Tensor] = None
    best_score = -float("inf")

    for _ in range(num_skills):
        skill = _sample_skill(agent, device)
        wrapper = _FixedConditionAgent(agent, skill)
        result = rollout_task(
            env,
            dataset,
            wrapper,
            task_reward,
            skill,
            device,
            num_episodes=num_selection_episodes,
            max_episode_steps=max_episode_steps,
            seed=seed,
            deterministic=deterministic,
        )
        score = float(
            result.get("normalized_score", result.get("mean_return", -float("inf")))
        )
        if score > best_score:
            best_score = score
            best_skill = skill

    if best_skill is None:
        best_skill = _sample_skill(agent, device)

    wrapper = _FixedConditionAgent(agent, best_skill)
    return rollout_task(
        env,
        dataset,
        wrapper,
        task_reward,
        best_skill,
        device,
        num_episodes=num_episodes,
        max_episode_steps=max_episode_steps,
        seed=seed,
        deterministic=deterministic,
    )


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

def evaluate_baseline(
    baseline_name: str,
    cfg: Config,
    dataset: OfflineDataset,
    env: Any,
    task_name: str,
    agent: Any,
    device: Optional[torch.device | str] = None,
    num_episodes: Optional[int] = None,
    max_episode_steps: Optional[int] = None,
    seed: int = 0,
    num_reward_samples: Optional[int] = None,
    num_skills: Optional[int] = None,
    deterministic: bool = True,
) -> Dict[str, float]:
    """Evaluate a single baseline agent on a single downstream task."""
    device = torch.device(device or resolve_device())
    num_episodes = int(num_episodes or _cfg_value(cfg, "eval.num_episodes", 20))
    max_episode_steps = int(
        max_episode_steps or _cfg_value(cfg, "eval.max_episode_steps", 1000)
    )
    num_reward_samples = int(
        num_reward_samples
        or _cfg_value(cfg, "baseline.num_reward_samples", 5120)
    )
    num_skills = int(num_skills or _cfg_value(cfg, "baseline.num_skills", 10))

    task_reward = _resolve_task_reward(cfg, dataset, task_name, device, seed)
    name = baseline_name.lower().replace("_", "-").replace(" ", "-")

    if name in {"gc-iql", "gciql", "goal-conditioned-iql"}:
        return evaluate_gc_agent(
            agent,
            dataset,
            env,
            task_reward,
            device,
            num_episodes=num_episodes,
            max_episode_steps=max_episode_steps,
            seed=seed,
            deterministic=deterministic,
        )

    if name in {"gc-bc", "gcbc", "goal-conditioned-bc", "bc"}:
        return evaluate_gc_agent(
            agent,
            dataset,
            env,
            task_reward,
            device,
            num_episodes=num_episodes,
            max_episode_steps=max_episode_steps,
            seed=seed,
            deterministic=deterministic,
        )

    if name in {"fb", "forward-backward"}:
        return evaluate_fb_agent(
            agent,
            dataset,
            env,
            task_reward,
            device,
            num_episodes=num_episodes,
            max_episode_steps=max_episode_steps,
            seed=seed,
            num_reward_samples=num_reward_samples,
            deterministic=deterministic,
        )

    if name in {"sf", "successor-features", "successor-features"}:
        return evaluate_sf_agent(
            agent,
            dataset,
            env,
            task_reward,
            device,
            num_episodes=num_episodes,
            max_episode_steps=max_episode_steps,
            seed=seed,
            num_reward_samples=num_reward_samples,
            deterministic=deterministic,
        )

    if name in {"opal", "opal-skill"}:
        return evaluate_opal_agent(
            agent,
            dataset,
            env,
            task_reward,
            device,
            num_episodes=num_episodes,
            max_episode_steps=max_episode_steps,
            seed=seed,
            num_skills=num_skills,
            deterministic=deterministic,
        )

    raise ValueError(f"Unknown baseline name: {baseline_name!r}")


def evaluate_all_baselines(
    cfg: Config,
    dataset: OfflineDataset,
    env: Any,
    agents: Dict[str, Any],
    task_names: Optional[Sequence[str]] = None,
    device: Optional[torch.device | str] = None,
    num_episodes: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Evaluate a collection of baselines on all requested tasks.

    Returns a nested mapping ``baseline -> task -> metrics``.
    """
    device = torch.device(device or resolve_device())
    task_names = list(task_names or ALL_TASKS)
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for baseline_name, agent in agents.items():
        task_results: Dict[str, Dict[str, float]] = {}
        for task_name in task_names:
            task_results[task_name] = evaluate_baseline(
                baseline_name,
                cfg,
                dataset,
                env,
                task_name,
                agent,
                device=device,
                num_episodes=num_episodes,
                seed=seed,
            )
        results[baseline_name] = task_results
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_dataset_for_domain(
    cfg: Config, device: torch.device | str
) -> OfflineDataset:
    domain = (cfg.domain or "").lower()
    if domain in {"antmaze", "ant"}:
        from fre.data.d4rl_loader import load_antmaze_dataset

        return load_antmaze_dataset(cfg.data, device=str(device))
    if domain in {"kitchen"}:
        from fre.data.d4rl_loader import load_kitchen_dataset

        return load_kitchen_dataset(cfg.data, device=str(device))
    if domain in {"exorl", "walker", "cheetah"}:
        from fre.data.exorl_loader import load_exorl_dataset

        return load_exorl_dataset(cfg.data, env_name=cfg.data.env_name, device=str(device))
    if domain in {"d4rl"}:
        from fre.data.d4rl_loader import load_d4rl_dataset

        return load_d4rl_dataset(cfg.data, env_name=cfg.data.env_name, device=str(device))

    # Generic fallback: AntMaze then D4RL then ExORL.
    from fre.data.d4rl_loader import load_d4rl_dataset

    return load_d4rl_dataset(cfg.data, env_name=cfg.data.env_name, device=str(device))


def _build_baseline_agent(
    baseline_name: str,
    cfg: Config,
    dataset: OfflineDataset,
    device: torch.device | str,
    checkpoint: Optional[str],
) -> Any:
    state_dim = int(dataset.states.shape[-1])
    action_dim = int(dataset.actions.shape[-1])
    name = baseline_name.lower().replace("_", "-")

    if name in {"gc-iql", "gciql", "goal-conditioned-iql"}:
        from fre.rl.gc_iql import GoalConditionedIQL

        agent = GoalConditionedIQL(
            state_dim=state_dim,
            action_dim=action_dim,
            cfg=cfg.iql,
            device=str(device),
        )
        if checkpoint and os.path.exists(checkpoint):
            agent.load(checkpoint)
        return agent

    if name in {"gc-bc", "gcbc", "goal-conditioned-bc", "bc"}:
        from fre.rl.gc_bc import GoalConditionedBC

        agent = GoalConditionedBC(
            state_dim=state_dim,
            action_dim=action_dim,
            cfg=cfg.iql,
            device=str(device),
        )
        if checkpoint and os.path.exists(checkpoint):
            agent.load(checkpoint)
        return agent

    if name in {"opal", "opal-skill"}:
        from fre.baselines.opal import OPAL

        agent = OPAL.from_config(cfg, state_dim=state_dim, action_dim=action_dim, device=str(device))
        if checkpoint and os.path.exists(checkpoint):
            agent.load(checkpoint)
        return agent

    if name in {"fb", "forward-backward"}:
        from fre.baselines.fb import ForwardBackward

        agent = ForwardBackward.from_config(cfg, state_dim=state_dim, action_dim=action_dim, device=str(device))
        if checkpoint and os.path.exists(checkpoint):
            agent.load(checkpoint)
        return agent

    if name in {"sf", "successor-features"}:
        from fre.baselines.sf import SuccessorFeatures

        agent = SuccessorFeatures.from_config(cfg, state_dim=state_dim, action_dim=action_dim, device=str(device))
        if checkpoint and os.path.exists(checkpoint):
            agent.load(checkpoint)
        return agent

    raise ValueError(f"Unknown baseline name: {baseline_name!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a baseline method on downstream FRE tasks"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--domain", type=str, default="antmaze")
    parser.add_argument("--baseline", type=str, default="gc-iql")
    parser.add_argument("--task", type=str, default="ant-goal-reaching")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-reward-samples", type=int, default=None)
    parser.add_argument("--num-skills", type=int, default=None)
    parser.add_argument("--stochastic", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    device = torch.device(resolve_device(args.device))

    cfg = get_config(args.config or "default")
    if args.domain:
        cfg.domain = args.domain
    if args.task:
        cfg.task = args.task
    cfg.seed = args.seed

    dataset = _load_dataset_for_domain(cfg, device)
    env = make_eval_env(cfg, dataset)
    agent = _build_baseline_agent(
        args.baseline, cfg, dataset, device, args.checkpoint
    )

    result = evaluate_baseline(
        args.baseline,
        cfg,
        dataset,
        env,
        args.task,
        agent,
        device=device,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        num_reward_samples=args.num_reward_samples,
        num_skills=args.num_skills,
        deterministic=not args.stochastic,
    )
    print(result)


if __name__ == "__main__":
    main()

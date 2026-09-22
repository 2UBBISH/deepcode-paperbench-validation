"""Evaluation utilities for SAPG / PPO policies.

This module evaluates a trained policy on the paper's manipulation tasks and
reports the metrics used throughout the SAPG paper:

* ``successes per episode`` -- the primary metric for the hard tasks
  (Regrasping, Throw, Reorientation on AllegroKuka).
* ``net episode reward`` -- the primary metric for the easier in-hand
  reorientation tasks (ShadowHand, AllegroHand).

The evaluator is deliberately agnostic to the trainer: it only requires an
object exposing ``act(obs, worker_ids, hidden_state, masks, deterministic)``
and ``init_hidden(batch_size)`` (both :class:`sapg.ppo.PPO` and
:class:`sapg.sapg_algorithm.SAPG` satisfy this interface), plus a vectorized
environment exposing ``reset``/``step``/``get_obs_dim``/``get_action_dim``.

Typical usage::

    from eval.evaluate import evaluate_policy, evaluate_from_checkpoint
    metrics = evaluate_policy(trainer, env, num_episodes=100)
    print(metrics["successes_per_episode"], metrics["mean_episode_reward"])
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Repo-root injection so the module works both as ``python -m eval.evaluate``
# and as a standalone script.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# Tasks whose primary metric is "successes per episode" (hard tasks).
SUCCESS_METRIC_TASKS = {"regrasping", "throw", "reorientation"}


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_policy(
    trainer: Any,
    env: Any,
    num_episodes: int = 100,
    num_envs: Optional[int] = None,
    deterministic: bool = True,
    max_steps: Optional[int] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Roll out ``trainer``'s policy in ``env`` and aggregate episode metrics.

    Parameters
    ----------
    trainer:
        Object exposing ``act``/``init_hidden`` (PPO or SAPG trainer). The
        trainer is switched to eval mode for the duration of the rollout and
        restored to train mode afterwards.
    env:
        Vectorized environment with a Gym-like ``reset``/``step`` API.
    num_episodes:
        Number of *episodes* to collect. Because the environment is
        vectorized, we keep stepping until at least this many episodes have
        terminated (or ``max_steps`` is reached).
    num_envs:
        Number of parallel environments. Defaults to ``len(env)`` when
        available, otherwise inferred from the first observation batch.
    deterministic:
        If ``True`` (default) use the policy mean; otherwise sample.
    max_steps:
        Safety cap on the number of environment steps. Defaults to
        ``episode_length * ceil(num_episodes / num_envs) * 2`` when the env
        exposes ``episode_length``.
    device:
        Torch device for the policy forward pass.
    seed:
        Optional seed for the environment reset (best-effort).

    Returns
    -------
    dict
        ``successes_per_episode``, ``mean_episode_reward``,
        ``mean_episode_length``, ``num_episodes``, ``num_successes``,
        ``success_rate``, ``tolerance``.
    """
    if device is None:
        device = getattr(trainer, "device", torch.device("cpu"))

    # --- resolve number of parallel envs -----------------------------------
    if num_envs is None:
        try:
            num_envs = len(env)
        except TypeError:
            num_envs = None

    # --- reset -------------------------------------------------------------
    obs = env.reset()
    obs = np.asarray(obs, dtype=np.float32)
    if num_envs is None:
        num_envs = int(obs.shape[0])

    if max_steps is None:
        episode_length = int(getattr(env, "episode_length", 300))
        episodes_per_env = int(np.ceil(num_episodes / max(num_envs, 1)))
        max_steps = episode_length * max(episodes_per_env, 1) * 2

    # --- worker ids: evaluation uses the leader (worker 0) -----------------
    worker_ids = torch.zeros(num_envs, dtype=torch.long, device=device)

    # --- recurrent hidden state -------------------------------------------
    hidden_state = None
    if getattr(trainer, "is_recurrent", False):
        hidden_state = trainer.init_hidden(num_envs)

    # --- bookkeeping -------------------------------------------------------
    episode_rewards: List[float] = []
    episode_successes: List[float] = []
    episode_lengths: List[int] = []
    running_reward = np.zeros(num_envs, dtype=np.float64)
    running_successes = np.zeros(num_envs, dtype=np.float64)
    running_length = np.zeros(num_envs, dtype=np.int64)

    was_training = getattr(trainer, "training", True)
    if hasattr(trainer, "eval"):
        trainer.eval()

    try:
        for _ in range(int(max_steps)):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            actions, _, _, hidden_state = trainer.act(
                obs_t,
                worker_ids,
                hidden_state,
                None,
                deterministic,
            )
            actions_np = actions.detach().cpu().numpy()
            obs, rewards, dones, infos = env.step(actions_np)

            rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
            dones = np.asarray(dones, dtype=bool).reshape(-1)
            running_reward += rewards
            running_length += 1

            # ``successes`` is a per-step indicator (1 if the env is currently
            # satisfying the success criterion). ``episode_successes`` is the
            # cumulative count for the current episode.
            step_successes = np.asarray(
                infos.get("successes", np.zeros(num_envs)), dtype=np.float64
            ).reshape(-1)
            running_successes += step_successes

            if dones.any():
                done_idx = np.nonzero(dones)[0]
                for i in done_idx:
                    episode_rewards.append(float(running_reward[i]))
                    episode_successes.append(float(running_successes[i]))
                    episode_lengths.append(int(running_length[i]))
                    running_reward[i] = 0.0
                    running_successes[i] = 0.0
                    running_length[i] = 0

                # Reset the recurrent hidden state for terminated envs.
                if hidden_state is not None:
                    hidden_state = _reset_hidden(hidden_state, done_idx)

            if len(episode_rewards) >= num_episodes:
                break
    finally:
        if was_training and hasattr(trainer, "train"):
            trainer.train()

    # --- aggregate ---------------------------------------------------------
    if not episode_rewards:
        return {
            "successes_per_episode": 0.0,
            "mean_episode_reward": 0.0,
            "mean_episode_length": 0.0,
            "num_episodes": 0,
            "num_successes": 0.0,
            "success_rate": 0.0,
            "tolerance": float(getattr(getattr(env, "curriculum", None), "get_tolerance", lambda: 0.0)()),
        }

    rewards_arr = np.asarray(episode_rewards, dtype=np.float64)
    successes_arr = np.asarray(episode_successes, dtype=np.float64)
    lengths_arr = np.asarray(episode_lengths, dtype=np.float64)

    tolerance = 0.0
    curriculum = getattr(env, "curriculum", None)
    if curriculum is not None and hasattr(curriculum, "get_tolerance"):
        tolerance = float(curriculum.get_tolerance())

    return {
        "successes_per_episode": float(successes_arr.mean()),
        "mean_episode_reward": float(rewards_arr.mean()),
        "mean_episode_length": float(lengths_arr.mean()),
        "num_episodes": int(len(episode_rewards)),
        "num_successes": float(successes_arr.sum()),
        "success_rate": float((successes_arr > 0).mean()),
        "tolerance": tolerance,
    }


def _reset_hidden(hidden_state: Any, done_idx: np.ndarray) -> Any:
    """Zero the recurrent hidden state for terminated environments.

    Supports both a single tensor ``(L, N, H)`` and a tuple/list of tensors
    ``(h, c)`` as returned by :meth:`ActorNetwork.init_hidden`.
    """
    if hidden_state is None:
        return None
    idx = torch.as_tensor(done_idx, dtype=torch.long, device=_first_device(hidden_state))

    def _zero(t: torch.Tensor) -> torch.Tensor:
        # t shape: (num_layers, N, H) or (N, H)
        if t.dim() == 3:
            t[:, idx, :] = 0.0
        elif t.dim() == 2:
            t[idx, :] = 0.0
        return t

    if isinstance(hidden_state, (tuple, list)):
        return type(hidden_state)(_zero(t) for t in hidden_state)
    return _zero(hidden_state)


def _first_device(hidden_state: Any) -> torch.device:
    if isinstance(hidden_state, (tuple, list)):
        for t in hidden_state:
            if isinstance(t, torch.Tensor):
                return t.device
        return torch.device("cpu")
    if isinstance(hidden_state, torch.Tensor):
        return hidden_state.device
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
def evaluate_from_checkpoint(
    checkpoint_path: str,
    cfg: Dict[str, Any],
    algo: str = "sapg",
    num_episodes: int = 100,
    deterministic: bool = True,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """Load a checkpoint and evaluate it on the configured environment.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pt`` checkpoint produced by ``main.train``.
    cfg:
        Fully-resolved config dict (as returned by ``main.load_config``).
    algo:
        ``"sapg"`` or ``"ppo"`` -- selects the trainer class.
    num_episodes:
        Number of episodes to evaluate.
    deterministic:
        Use the policy mean (default) rather than sampling.
    device:
        Optional device override (e.g. ``"cpu"``).

    Returns
    -------
    dict
        The metrics dict from :func:`evaluate_policy`.
    """
    from envs import make_env
    from policy import build_policy
    from ppo import PPO
    from sapg_algorithm import SAPG

    dev = torch.device(device) if device is not None else _resolve_device(cfg)

    env_name = cfg.get("env_name", "allegro_kuka")
    task = cfg.get("task", "regrasping")
    num_envs = int(cfg.get("num_envs", 64))
    num_workers = int(cfg.get("num_workers", 1)) if algo == "sapg" else 1

    env = make_env(
        env_name=env_name,
        num_envs=num_envs,
        task=task,
        cfg=cfg,
        device=dev,
        seed=int(cfg.get("seed", 0)),
    )

    obs_dim = env.get_obs_dim()
    action_dim = env.get_action_dim()
    policy = build_policy(cfg, obs_dim, action_dim, num_workers=num_workers)

    if algo == "ppo":
        trainer = PPO(
            cfg=cfg,
            policy=policy,
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_envs=num_envs,
            device=dev,
        )
    else:
        trainer = SAPG(
            cfg=cfg,
            policy=policy,
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_envs=num_envs,
            num_workers=num_workers,
            device=dev,
        )

    state = torch.load(checkpoint_path, map_location=dev)
    if isinstance(state, dict) and "trainer" in state:
        trainer.load_state_dict(state["trainer"])
    else:
        trainer.load_state_dict(state)

    return evaluate_policy(
        trainer,
        env,
        num_episodes=num_episodes,
        num_envs=num_envs,
        deterministic=deterministic,
        device=dev,
    )


def _resolve_device(cfg: Dict[str, Any]) -> torch.device:
    requested = str(cfg.get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def primary_metric(task: str) -> str:
    """Return the paper's primary metric key for a given task."""
    return "successes_per_episode" if task in SUCCESS_METRIC_TASKS else "mean_episode_reward"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SAPG/PPO checkpoint on a manipulation task."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument(
        "--algo", type=str, default="sapg", choices=["sapg", "ppo"], help="Trainer type."
    )
    parser.add_argument("--env_name", type=str, default=None, help="Override env_name.")
    parser.add_argument("--task", type=str, default=None, help="Override task.")
    parser.add_argument("--num_envs", type=int, default=None, help="Override num_envs.")
    parser.add_argument("--num_episodes", type=int, default=100, help="Episodes to evaluate.")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using the deterministic policy mean.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g. cpu).")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    from main import load_config

    overrides: Dict[str, Any] = {}
    if args.env_name is not None:
        overrides["env_name"] = args.env_name
    if args.task is not None:
        overrides["task"] = args.task
    if args.num_envs is not None:
        overrides["num_envs"] = args.num_envs
    if args.device is not None:
        overrides["device"] = args.device

    cfg = load_config(args.config, overrides)

    metrics = evaluate_from_checkpoint(
        checkpoint_path=args.checkpoint,
        cfg=cfg,
        algo=args.algo,
        num_episodes=args.num_episodes,
        deterministic=not args.stochastic,
        device=args.device,
    )

    task = cfg.get("task", "regrasping")
    key = primary_metric(task)
    print("=" * 60)
    print(f"Evaluation: {cfg.get('env_name')} / {task} ({args.algo})")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:>24s}: {v}")
    print(f"  {'primary_metric':>24s}: {key} = {metrics.get(key, float('nan'))}")
    print("=" * 60)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(
                {
                    "env_name": cfg.get("env_name"),
                    "task": task,
                    "algo": args.algo,
                    "checkpoint": args.checkpoint,
                    "primary_metric": key,
                    "metrics": metrics,
                },
                f,
                indent=2,
            )
        print(f"Wrote evaluation summary to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

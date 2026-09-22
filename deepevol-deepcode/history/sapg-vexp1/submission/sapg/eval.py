"""Evaluation utilities for SAPG and baselines.

This module provides:

* :func:`evaluate_checkpoint` -- load a checkpoint produced by any of the
  trainers (SAPG / PPO / DexPBT / PQL) and evaluate the resulting policy on a
  task, reporting the paper's two headline metrics:

  - ``successes`` (hard tasks): mean number of successful episodes per
    evaluation window (AllegroKuka Regrasping / Throw / Reorientation).
  - ``episode_reward`` (easy tasks): mean undiscounted episode return
    (ShadowHand / AllegroHand in-hand reorientation).

* :func:`evaluate_policy` -- evaluate an in-memory policy object.

* :func:`evaluate_multi_policy` -- evaluate every policy of a SAPG
  :class:`~sapg.sapg.actor_critic.MultiPolicyActorCritic` (used for the
  diversity analysis in Sec. 6.4).

The evaluation loop is deliberately simulator agnostic: it only requires an
environment exposing ``reset()`` / ``step(actions)`` and the usual
``num_envs`` / ``obs_dim`` / ``action_dim`` attributes (see
``sapg/envs/isaacgym_wrapper.py``).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # torch is a hard dependency of the project but keep import defensive
    import torch
except Exception:  # pragma: no cover - torch should always be available
    torch = None  # type: ignore


__all__ = [
    "EvalResult",
    "evaluate_policy",
    "evaluate_multi_policy",
    "evaluate_checkpoint",
    "load_policy_from_checkpoint",
    "summarize_results",
    "HARD_TASKS",
    "EASY_TASKS",
]


# ---------------------------------------------------------------------------
# Task classification (paper Table 1 / Sec. 6)
# ---------------------------------------------------------------------------
HARD_TASKS: Tuple[str, ...] = (
    "allegrokuka_regrasping",
    "allegrokuka_throw",
    "allegrokuka_reorientation",
)
EASY_TASKS: Tuple[str, ...] = ("shadowhand", "allegrohand")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
class EvalResult(dict):
    """Dictionary of evaluation metrics with attribute-style access.

    Always contains at least the keys ``successes``, ``episode_reward``,
    ``episode_length``, ``num_episodes`` and ``num_envs``.
    """

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - trivial
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        keys = ("successes", "episode_reward", "episode_length", "num_episodes")
        parts = ", ".join(f"{k}={self.get(k):.4g}" for k in keys if k in self)
        return f"EvalResult({parts})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_numpy(x: Any) -> np.ndarray:
    """Convert torch tensors / lists / arrays to a CPU float32 numpy array."""
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _select_action(policy: Any, obs: Any, lstm_state: Any = None) -> Tuple[np.ndarray, Any]:
    """Query a policy for a (deterministic) action.

    Supports both the SAPG ``SharedActorCritic`` interface (``act`` returning
    ``(action, log_prob, lstm_state)``) and plain actor modules exposing
    ``forward`` / ``distribution``.  The mean of the Gaussian is used so that
    evaluation is deterministic.
    """
    if hasattr(policy, "act"):
        out = policy.act(obs, lstm_state) if lstm_state is not None else policy.act(obs)
        if isinstance(out, tuple):
            action = out[0]
            new_state = out[-1] if len(out) >= 3 else None
            return _to_numpy(action), new_state
        return _to_numpy(out), None

    if hasattr(policy, "distribution"):
        dist = policy.distribution(obs)
        if isinstance(dist, tuple):
            dist = dist[0]
        return _to_numpy(dist.mean), None

    if hasattr(policy, "forward"):
        out = policy.forward(obs)
        if isinstance(out, tuple):
            out = out[0]
        return _to_numpy(out), None

    raise TypeError(f"Policy {type(policy)!r} exposes no usable action interface")


def _reset_lstm_state(policy: Any, num_envs: int, device: Any) -> Any:
    """Create a zero LSTM state for recurrent policies (or ``None``)."""
    if not getattr(policy, "is_recurrent", False):
        return None
    hidden = getattr(policy, "lstm_hidden", None)
    if hidden is None:
        hidden = getattr(getattr(policy, "actor", None), "lstm_hidden", 768)
    if torch is None:  # pragma: no cover
        return None
    return (
        torch.zeros(1, num_envs, hidden, device=device),
        torch.zeros(1, num_envs, hidden, device=device),
    )


def _mask_lstm_state(state: Any, done_mask: np.ndarray) -> Any:
    """Zero the LSTM state of environments that just terminated."""
    if state is None or torch is None:
        return state
    h, c = state
    mask = torch.as_tensor(done_mask, dtype=h.dtype, device=h.device).view(1, -1, 1)
    return h * (1.0 - mask), c * (1.0 - mask)


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------
def evaluate_policy(
    policy: Any,
    env: Any,
    num_episodes: int = 100,
    max_steps: int = 2000,
    device: Any = "cpu",
    deterministic: bool = True,
    seed: Optional[int] = None,
    record_actions: bool = False,
) -> EvalResult:
    """Evaluate ``policy`` in the vectorized ``env``.

    Args:
        policy: object exposing ``act(obs[, lstm_state])`` (SAPG) or
            ``forward`` / ``distribution``.
        env: vectorized environment with ``reset()`` and ``step(actions)``.
        num_episodes: number of *completed* episodes to collect (across all
            parallel envs) before stopping.
        max_steps: safety cap on the number of environment steps.
        device: torch device used for policy inference.
        deterministic: if ``True`` use the Gaussian mean (default).
        seed: optional seed forwarded to ``env.reset`` when supported.
        record_actions: when ``True`` also return the collected
            ``(obs, action)`` pairs under the ``data`` key (used by the
            diversity metrics).

    Returns:
        :class:`EvalResult` with ``successes``, ``episode_reward``,
        ``episode_length``, ``num_episodes``, ``num_envs`` and (optionally)
        ``data``.
    """
    if torch is not None:
        try:
            policy.eval()
        except Exception:  # pragma: no cover - policy may not be an nn.Module
            pass

    num_envs = int(getattr(env, "num_envs", 1))

    try:
        obs = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:  # env.reset() without seed kwarg
        obs = env.reset()

    lstm_state = _reset_lstm_state(policy, num_envs, device)

    episode_returns = np.zeros(num_envs, dtype=np.float64)
    episode_lengths = np.zeros(num_envs, dtype=np.int64)
    episode_successes = np.zeros(num_envs, dtype=np.float64)

    finished_returns: List[float] = []
    finished_lengths: List[int] = []
    finished_successes: List[float] = []

    collected_obs: List[np.ndarray] = []
    collected_actions: List[np.ndarray] = []

    steps = 0
    while len(finished_returns) < num_episodes and steps < max_steps:
        with torch.no_grad() if torch is not None else _nullcontext():
            action, lstm_state = _select_action(policy, obs, lstm_state)

        if record_actions:
            collected_obs.append(_to_numpy(obs).copy())
            collected_actions.append(np.asarray(action).copy())

        out = env.step(action)
        if len(out) == 5:
            next_obs, reward, done, truncated, info = out
            done = np.logical_or(done, truncated)
        else:
            next_obs, reward, done, info = out

        reward = np.asarray(reward, dtype=np.float64).reshape(-1)
        done = np.asarray(done).reshape(-1).astype(bool)

        episode_returns += reward
        episode_lengths += 1

        success = _extract_success(info, num_envs)
        episode_successes += success

        if done.any():
            idx = np.nonzero(done)[0]
            for i in idx:
                finished_returns.append(float(episode_returns[i]))
                finished_lengths.append(int(episode_lengths[i]))
                finished_successes.append(float(episode_successes[i] > 0))
                episode_returns[i] = 0.0
                episode_lengths[i] = 0
                episode_successes[i] = 0.0
            lstm_state = _mask_lstm_state(lstm_state, done)

        obs = next_obs
        steps += 1

    if torch is not None:
        try:
            policy.train()
        except Exception:  # pragma: no cover
            pass

    n = max(len(finished_returns), 1)
    result = EvalResult(
        successes=float(np.sum(finished_successes) / n),
        episode_reward=float(np.mean(finished_returns)) if finished_returns else 0.0,
        episode_length=float(np.mean(finished_lengths)) if finished_lengths else 0.0,
        num_episodes=len(finished_returns),
        num_envs=num_envs,
        steps=steps,
        success_rate=float(np.mean(finished_successes)) if finished_successes else 0.0,
    )

    if record_actions and collected_obs:
        obs_arr = np.concatenate(collected_obs, axis=0)
        act_arr = np.concatenate(collected_actions, axis=0)
        result["data"] = np.concatenate([obs_arr, act_arr], axis=-1)

    return result


def _extract_success(info: Any, num_envs: int) -> np.ndarray:
    """Pull a per-env success signal out of the ``info`` dict."""
    if not isinstance(info, dict):
        return np.zeros(num_envs, dtype=np.float64)
    for key in ("success", "episode_success", "is_success"):
        if key in info:
            val = np.asarray(info[key], dtype=np.float64).reshape(-1)
            if val.shape[0] == num_envs:
                return val
    return np.zeros(num_envs, dtype=np.float64)


class _nullcontext:  # pragma: no cover - tiny helper
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# Multi-policy evaluation (SAPG)
# ---------------------------------------------------------------------------
def evaluate_multi_policy(
    model: Any,
    env: Any,
    num_episodes: int = 100,
    max_steps: int = 2000,
    device: Any = "cpu",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate every policy of a :class:`MultiPolicyActorCritic`.

    Returns a dict with per-policy metrics plus the aggregate (mean over
    policies) ``successes`` and ``episode_reward`` used in Table 1.
    """
    num_policies = len(getattr(model, "policies", [])) or 1
    per_policy: List[EvalResult] = []
    for j in range(num_policies):
        policy = model.policy(j) if hasattr(model, "policy") else model
        res = evaluate_policy(
            policy,
            env,
            num_episodes=num_episodes,
            max_steps=max_steps,
            device=device,
            seed=None if seed is None else seed + j,
        )
        per_policy.append(res)

    successes = np.array([r["successes"] for r in per_policy], dtype=np.float64)
    rewards = np.array([r["episode_reward"] for r in per_policy], dtype=np.float64)

    return {
        "per_policy": per_policy,
        "successes": float(successes.mean()),
        "successes_std": float(successes.std()),
        "episode_reward": float(rewards.mean()),
        "episode_reward_std": float(rewards.std()),
        "num_policies": num_policies,
    }


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def _infer_method(checkpoint: Dict[str, Any], method: Optional[str]) -> str:
    if method:
        return method.lower()
    for key in ("method", "algorithm", "algo"):
        if key in checkpoint:
            return str(checkpoint[key]).lower()
    if "phis" in checkpoint or "sigmas" in checkpoint:
        return "sapg"
    if "agents" in checkpoint:
        return "dexpbt"
    if "actor_target" in checkpoint or "q_head" in checkpoint:
        return "pql"
    return "ppo"


def load_policy_from_checkpoint(
    checkpoint_path: str,
    task: str,
    method: Optional[str] = None,
    device: Any = "cpu",
    num_policies: int = 6,
    phi_dim: Optional[int] = None,
    obs_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    recurrent: Optional[bool] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Instantiate the right model for ``task`` and load ``checkpoint_path``.

    Returns ``(model, checkpoint_dict)``.  For SAPG the returned model is a
    :class:`MultiPolicyActorCritic`; for the baselines it is a single policy
    (actor) module.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for checkpoint evaluation")

    from envs.isaacgym_wrapper import resolve_task_name, task_spec

    task = resolve_task_name(task)
    spec = task_spec(task)
    obs_dim = obs_dim or spec["obs_dim"]
    action_dim = action_dim or spec["action_dim"]
    if recurrent is None:
        recurrent = spec.get("recurrent", False)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    resolved_method = _infer_method(ckpt, method)

    if resolved_method == "sapg":
        from sapg.actor_critic import MultiPolicyActorCritic

        if phi_dim is None:
            phi_dim = 32 if task.startswith("allegrokuka") else 16
        model = MultiPolicyActorCritic(
            task=task,
            obs_dim=obs_dim,
            action_dim=action_dim,
            phi_dim=phi_dim,
            num_policies=num_policies,
            device=device,
        )
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        return model, ckpt

    # Baselines: single actor (+ optional critic) built from the shared factory.
    from networks import build_actor, build_critic

    actor = build_actor(task, obs_dim, action_dim, phi_dim=0)
    critic = build_critic(task, obs_dim, phi_dim=0)

    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    actor_state = _extract_sub_state(state, ("actor.", "policy.", "module.actor."))
    critic_state = _extract_sub_state(state, ("critic.", "value.", "module.critic."))

    if actor_state:
        actor.load_state_dict(actor_state, strict=False)
    elif isinstance(state, dict):
        actor.load_state_dict(state, strict=False)
    if critic_state:
        critic.load_state_dict(critic_state, strict=False)

    actor.to(device)
    actor.eval()
    return actor, ckpt


def _extract_sub_state(state: Any, prefixes: Sequence[str]) -> Dict[str, Any]:
    """Extract a sub-state-dict matching one of ``prefixes``."""
    if not isinstance(state, dict):
        return {}
    for prefix in prefixes:
        sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if sub:
            return sub
    return {}


# ---------------------------------------------------------------------------
# Top-level entry point used by ``main.py --eval``
# ---------------------------------------------------------------------------
def evaluate_checkpoint(
    checkpoint: str,
    task: str,
    method: Optional[str] = None,
    num_envs: int = 256,
    num_episodes: int = 100,
    max_steps: int = 2000,
    seed: int = 0,
    device: str = "cuda",
    output_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    num_policies: int = 6,
    force_dummy: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate a checkpoint and return the paper's headline metrics.

    This is the function invoked by ``python main.py --eval --checkpoint ...``.
    """
    config = dict(config or {})

    from envs.isaacgym_wrapper import make_vector_env, resolve_task_name

    task = resolve_task_name(task)
    env = make_vector_env(
        task,
        num_envs=num_envs,
        seed=seed,
        device=device,
        force_dummy=force_dummy,
    )

    try:
        model, ckpt = load_policy_from_checkpoint(
            checkpoint,
            task=task,
            method=method,
            device=device,
            num_policies=num_policies,
            phi_dim=config.get("phi_dim"),
        )

        resolved_method = _infer_method(ckpt, method)
        if resolved_method == "sapg":
            results = evaluate_multi_policy(
                model,
                env,
                num_episodes=num_episodes,
                max_steps=max_steps,
                device=device,
                seed=seed,
            )
        else:
            res = evaluate_policy(
                model,
                env,
                num_episodes=num_episodes,
                max_steps=max_steps,
                device=device,
                seed=seed,
            )
            results = {
                "successes": res["successes"],
                "episode_reward": res["episode_reward"],
                "episode_length": res["episode_length"],
                "num_episodes": res["num_episodes"],
                "per_policy": [res],
                "num_policies": 1,
            }
    finally:
        try:
            env.close()
        except Exception:  # pragma: no cover
            pass

    results["task"] = task
    results["method"] = _infer_method(ckpt, method)
    results["checkpoint"] = checkpoint

    summary = summarize_results(results, task=task)
    print(summary)

    if output_dir:
        _write_results(results, output_dir, task, results["method"])

    return results


def summarize_results(results: Dict[str, Any], task: Optional[str] = None) -> str:
    """Human readable one-line summary of an evaluation result."""
    task = task or results.get("task", "?")
    metric = "successes" if task in HARD_TASKS else "episode_reward"
    value = results.get(metric, float("nan"))
    std = results.get(f"{metric}_std")
    if std is not None:
        return (
            f"[eval] task={task} method={results.get('method', '?')} "
            f"{metric}={value:.4g} +/- {std:.3g} "
            f"(episodes={results.get('num_episodes', '?')})"
        )
    return (
        f"[eval] task={task} method={results.get('method', '?')} "
        f"{metric}={value:.4g} (episodes={results.get('num_episodes', '?')})"
    )


def _write_results(
    results: Dict[str, Any],
    output_dir: str,
    task: str,
    method: str,
) -> str:
    """Persist evaluation results as JSON next to the checkpoint."""
    import json

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"eval_{task}_{method}.json")
    serializable = {
        k: v for k, v in results.items() if k != "per_policy" and _is_jsonable(v)
    }
    with open(path, "w") as fh:
        json.dump(serializable, fh, indent=2, sort_keys=True)
    return path


def _is_jsonable(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), list, dict))


# ---------------------------------------------------------------------------
# CLI (``python -m sapg.eval``)
# ---------------------------------------------------------------------------
def build_parser():  # pragma: no cover - CLI plumbing
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a SAPG/PPO/DexPBT/PQL checkpoint")
    parser.add_argument("--checkpoint", required=True, help="path to checkpoint .pt file")
    parser.add_argument("--task", required=True, help="task name (e.g. allegrokuka_regrasping)")
    parser.add_argument("--method", default=None, choices=[None, "sapg", "ppo", "dexpbt", "pql"])
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-policies", type=int, default=6)
    parser.add_argument("--dummy-env", action="store_true", help="force NumPy fallback env")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:  # pragma: no cover
    args = build_parser().parse_args(argv)
    evaluate_checkpoint(
        checkpoint=args.checkpoint,
        task=args.task,
        method=args.method,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        num_policies=args.num_policies,
        force_dummy=args.dummy_env,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

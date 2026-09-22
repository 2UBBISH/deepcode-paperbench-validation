#!/usr/bin/env python
"""Evaluation entry point for SAPG and its baselines (Sec. 5.2, Table 1).

Loads a checkpoint (SAPG / PPO / PQL / DexPBT), rolls the trained policy out on the
vectorised IsaacGym task suite and reports the paper's evaluation quantities:

* **Allegro-Hand / Shadow-Hand** (``allegro_hand``, ``shadow_hand``): net episode
  reward (Table 1 reports e.g. ``1.23e4`` for AllegroHand).
* **Allegro-Kuka hard tasks** (``regrasping``, ``throw``, ``reorientation``): mean
  success count / success rate over the evaluation horizon, which is what Table 1
  summarises (e.g. Regrasping ``35.7``, Throw ``23.7``, Reorientation ``33.2``).

Multiple seeds are aggregated with the paper's shaded-band width
``(2 / sqrt(n)) * sum_i (mean(t) - y_i(t))^2`` computed in
:func:`sapg.utils.logging.paper_standard_error`.

Usage
-----
::

    # single task, single checkpoint
    python scripts/evaluate.py --task allegro_hand \
        --checkpoint runs/allegro_hand/sapg.pt --num-seeds 5

    # whole Table-1 sweep from a runs/ directory
    python scripts/evaluate.py --task all --checkpoint-dir runs \
        --num-seeds 5 --output results/table1.json

All heavy imports (torch, IsaacGym, the ``sapg`` package itself) are performed
lazily inside the helper functions so that ``--help`` and config validation work on
a machine without a GPU or the simulator installed.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Make the repository importable when the script is executed directly.
# --------------------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_TOP = os.path.dirname(_ROOT)
if _TOP not in sys.path:
    sys.path.insert(0, _TOP)


# --------------------------------------------------------------------------------------
# Constants taken from the paper
# --------------------------------------------------------------------------------------
TASKS: Tuple[str, ...] = (
    "regrasping",
    "throw",
    "reorientation",
    "shadow_hand",
    "allegro_hand",
)
METHODS: Tuple[str, ...] = ("sapg", "ppo", "pql", "dexpbt")

#: Tasks whose reported metric is a success count rather than net episode reward.
SUCCESS_TASKS: Tuple[str, ...] = ("regrasping", "throw", "reorientation")

#: Paper's reported final numbers (Table 1 / Sec. 6.1), used for the comparison column.
PAPER_RESULTS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "sapg": {
        "allegro_hand": (1.23e4, 3.29e2),
        "shadow_hand": (1.17e4, 2.64e2),
        "regrasping": (35.7, 1.46),
        "throw": (23.7, 0.74),
        "reorientation": (33.2, 4.20),
    },
    "sapg_sigma_0.005": {
        "allegro_hand": (9.14e3, 8.38e2),
        "shadow_hand": (1.28e4, 2.80e2),
        "regrasping": (33.4, 2.25),
        "throw": (18.7, 0.43),
        "reorientation": (38.6, 0.63),
    },
}

#: How many evaluation episodes (horizon reset cycles) per seed by default.
DEFAULT_EPISODES: int = 10
DEFAULT_NUM_SEEDS: int = 5

#: Candidate metric keys inside a rollout/logger history, in decreasing priority.
REWARD_KEYS: Tuple[str, ...] = (
    "episode_return",
    "episode_reward",
    "return_mean",
    "reward",
    "score",
)
SUCCESS_KEYS: Tuple[str, ...] = (
    "successes",
    "success_count",
    "success_rate",
    "episode_successes",
    "success",
)
LENGTH_KEYS: Tuple[str, ...] = ("episode_length", "length", "episode_len")


# --------------------------------------------------------------------------------------
# Small generic helpers
# --------------------------------------------------------------------------------------
def _to_float(value: Any, default: float = float("nan")) -> float:
    """Best-effort conversion of a scalar / tensor-like value to ``float``."""
    if value is None:
        return default
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "mean"):
            value = value.mean()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:  # pragma: no cover - defensive
        try:
            return float(value)
        except Exception:
            return default


def _get(obj: Any, keys: Sequence[str], default: Any = None) -> Any:
    """Fetch the first present key from a dict-like or attribute-like object."""
    if obj is None:
        return default
    for key in keys:
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
        else:
            getter = getattr(obj, "get", None)
            if callable(getter):
                try:
                    value = getter(key)
                except Exception:  # pragma: no cover - defensive
                    value = None
                if value is not None:
                    return value
            if hasattr(obj, key):
                return getattr(obj, key)
    return default


def _ensure_dir(path: Optional[str]) -> Optional[str]:
    if not path:
        return path
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    return path


def _normalise_method(method: Optional[str]) -> str:
    """Map method aliases onto the canonical names used by the trainers."""
    name = str(method or "sapg").strip().lower().replace("-", "_")
    aliases = {
        "vanilla_ppo": "ppo",
        "ppo_baseline": "ppo",
        "baseline_ppo": "ppo",
        "pbt": "dexpbt",
        "expbt": "dexpbt",
        "population": "dexpbt",
        "split": "sapg",
        "split_aggregate": "sapg",
        "apql": "pql",
        "parallel_q_learning": "pql",
    }
    return aliases.get(name, name)


def _normalise_task(task: Optional[str]) -> str:
    name = str(task or "allegro_hand").strip().lower().replace("-", " ")
    name = name.replace(" ", "_")
    aliases = {
        "allegro_kuka": "regrasping",
        "regrasp": "regrasping",
        "shadow": "shadow_hand",
        "shadowhand": "shadow_hand",
        "in_hand_reorientation": "shadow_hand",
        "allegro": "allegro_hand",
        "allegrohand": "allegro_hand",
    }
    return aliases.get(name, name)


# --------------------------------------------------------------------------------------
# Configuration / env / policy / checkpoint construction
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate SAPG / PPO / PQL / DexPBT checkpoints (SAPG Table 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", type=str, default="allegro_hand",
                        help="Task name, or 'all' to evaluate every paper task.")
    parser.add_argument("--method", type=str, default="sapg",
                        help="Method whose checkpoint(s) to evaluate.")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config (e.g. configs/allegro_kuka.yaml).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single checkpoint path (.pt/.pth) to evaluate.")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory containing <task>/<method>.pt checkpoints.")
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS,
                        help="Number of seeds for the shaded-band aggregation.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base seed; when given, --num-seeds is ignored.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="Evaluation episodes per seed.")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Override the number of parallel environments.")
    parser.add_argument("--num-policies", type=int, default=None,
                        help="Override M (use 1 for PPO / PQL / DexPBT member).")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Environment steps per evaluation episode.")
    parser.add_argument("--entropy-coefficient", type=float, default=None,
                        help="Override sigma (only relevant for training).")
    parser.add_argument("--device", type=str, default=None, help="Torch device.")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use the mean action (default).")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false",
                        help="Sample from the policy instead of using the mean.")
    parser.add_argument("--surrogate", action="store_true", default=False,
                        help="Force the dependency-free torch surrogate env.")
    parser.add_argument("--output", type=str, default=None,
                        help="Write the evaluation summary to this JSON path.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Alternative to --output: directory for per-task JSON.")
    parser.add_argument("--paper-comparison", action="store_true", default=False,
                        help="Print the paper's Table-1 reference numbers alongside.")
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser


def load_config(args: argparse.Namespace, task: Optional[str] = None) -> Any:
    """Resolve an :class:`~sapg.utils.config.SAPGConfig` for ``task`` and the CLI."""
    from sapg.utils.config import SAPGConfig, build_config  # local import (lazy)

    task_name = _normalise_task(task or args.task)
    overrides: Dict[str, Any] = {}

    if args.config:
        try:
            yaml_config = SAPGConfig.from_yaml(args.config)
        except Exception:  # pragma: no cover - config may be task-group level
            yaml_config = None
        if yaml_config is not None:
            config = yaml_config
            if not getattr(config, "task", None) or config.task == "regrasping":
                # A task-group YAML still needs the concrete task name applied.
                try:
                    config = build_config(task_name)
                    for field in ("num_envs", "num_policies", "learning_rate",
                                  "horizon_length", "mini_epochs", "clip_epsilon",
                                  "critic_coefficient", "entropy_coefficient",
                                  "phi_dim", "obs_dim", "action_dim", "device",
                                  "seed", "log_dir", "use_lstm", "per_block_sigma"):
                        value = getattr(yaml_config, field, None)
                        if value is not None:
                            setattr(config, field, value)
                except Exception:
                    config = yaml_config
        else:  # pragma: no cover - defensive
            config = build_config(task_name)
    else:
        config = build_config(task_name)

    if args.num_envs is not None:
        overrides["num_envs"] = int(args.num_envs)
    if args.num_policies is not None:
        overrides["num_policies"] = int(args.num_policies)
    if args.device is not None:
        overrides["device"] = args.device
    if args.entropy_coefficient is not None:
        overrides["entropy_coefficient"] = float(args.entropy_coefficient)

    for key, value in overrides.items():
        try:
            setattr(config, key, value)
        except Exception:  # pragma: no cover - defensive
            pass

    config.task = task_name
    if _normalise_method(args.method) != "sapg":
        try:
            config.num_policies = 1
            config.phi_dim = 0 if hasattr(config, "phi_dim") else 0
            config.method = _normalise_method(args.method)
        except Exception:  # pragma: no cover
            pass
    return config


def make_env_for(config: Any, args: Optional[argparse.Namespace] = None) -> Any:
    """Instantiate the vectorised environment for ``config``."""
    if args is not None and getattr(args, "surrogate", False):
        os.environ["SAPG_FORCE_SURROGATE"] = "1"
    from sapg.envs import make_env  # lazy

    num_envs = getattr(config, "num_envs", None)
    device = getattr(config, "device", None)
    task = getattr(config, "task", "allegro_hand")
    try:
        return make_env(task, config=config, device=device)
    except TypeError:
        try:
            return make_env(task, config=config, num_envs=num_envs)
        except TypeError:
            return make_env(task)


def make_policy_for(config: Any, num_policies: Optional[int] = None) -> Any:
    """Build the (phi-conditioned) ``ActorCritic`` policy for evaluation."""
    from sapg.models.actor import ActorCritic  # lazy

    kwargs: Dict[str, Any] = {
        "obs_dim": getattr(config, "obs_dim", None),
        "action_dim": getattr(config, "action_dim", None),
        "phi_dim": getattr(config, "phi_dim", 0),
        "num_policies": int(num_policies if num_policies is not None
                            else getattr(config, "num_policies", 1)),
        "mlp_units": tuple(getattr(config, "actor_mlp_units", (768, 512, 256))),
        "activation": getattr(config, "actor_activation", "elu"),
        "use_lstm": bool(getattr(config, "use_lstm", False)),
        "lstm_hidden_size": int(getattr(config, "lstm_hidden_size", 768)),
        "lstm_num_layers": int(getattr(config, "lstm_num_layers", 1)),
        "per_block_sigma": bool(getattr(config, "per_block_sigma", False)),
        "learnable_phi": not bool(getattr(config, "random_phi", False)),
        "config": config,
    }
    try:
        return ActorCritic(**kwargs)
    except TypeError:
        return ActorCritic(config=config)


def load_checkpoint(policy: Any, path: Optional[str],
                    device: Any = None) -> Optional[Dict[str, Any]]:
    """Load a checkpoint into ``policy`` (and any auxiliary state). Returns extras."""
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    if path.endswith(".npz"):  # pragma: no cover - non-torch checkpoints
        raise NotImplementedError("only torch checkpoints are supported")

    import torch  # lazy

    map_location = device if device is not None else "cpu"
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # older torch without weights_only
        payload = torch.load(path, map_location=map_location)

    state = payload
    extras: Dict[str, Any] = {"checkpoint": path}
    if isinstance(payload, dict):
        for key in ("policy", "model", "state_dict", "actor_critic", "weights"):
            if key in payload and isinstance(payload[key], dict):
                state = payload[key]
                break
        for key in ("iteration", "samples", "epoch", "seed", "optimizers"):
            if key in payload:
                extras[key] = payload[key]

    # Move to the evaluation device when supported.
    if hasattr(policy, "load_state_dict"):
        missing, unexpected = None, None
        try:
            result = policy.load_state_dict(state, strict=False)
            missing = getattr(result, "missing_keys", None)
            unexpected = getattr(result, "unexpected_keys", None)
        except Exception:
            policy.load_state_dict(state, strict=False)
        if missing:
            extras["missing_keys"] = list(missing)
        if unexpected:
            extras["unexpected_keys"] = list(unexpected)
    if device is not None and hasattr(policy, "to"):
        try:
            policy.to(device)
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(policy, "eval"):
        policy.eval()
    return extras


# --------------------------------------------------------------------------------------
# Rollout / evaluation
# --------------------------------------------------------------------------------------
def _policy_index_for_sapg(config: Any) -> int:
    """Which block's policy to evaluate: the leader (policy ``i = 1``) by default."""
    leader_one_based = int(getattr(config, "leader_index", 1) or 1)
    return max(0, leader_one_based - 1)


def _act(policy: Any, obs: Any, phi: Any, hidden_state: Any, masks: Any,
         policy_index: Optional[int], deterministic: bool) -> Dict[str, Any]:
    """Call ``policy.act`` with the richest signature it accepts."""
    attempts = (
        dict(obs=obs, phi=phi, hidden_state=hidden_state, masks=masks,
             policy_index=policy_index, deterministic=deterministic),
        dict(obs, phi=phi, hidden_state=hidden_state, masks=masks,
             deterministic=deterministic),
        dict(obs, deterministic=deterministic),
        dict(obs),
    )
    if hasattr(policy, "act"):
        fn = policy.act
    elif hasattr(policy, "forward"):
        fn = policy.forward
    else:  # pragma: no cover - defensive
        raise AttributeError("policy exposes neither .act nor .forward")
    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            return fn(**kwargs)
        except TypeError as error:  # signature mismatch -> try the next form
            last_error = error
        except Exception as error:  # pragma: no cover - propagate real errors
            raise error
    raise TypeError(f"could not call policy.act: {last_error}")


def _phi_for(policy: Any, policy_index: Optional[int]) -> Any:
    """Fetch ``phi_j`` for the policy being evaluated (``None`` when phi-free)."""
    if policy_index is None:
        return None
    for name in ("phi_for", "phi", "get_phi"):
        fn = getattr(policy, name, None)
        if callable(fn):
            try:
                return fn(policy_index)
            except Exception:  # pragma: no cover - defensive
                continue
    return None


def _init_hidden(policy: Any, num_envs: int, device: Any) -> Any:
    if hasattr(policy, "init_hidden"):
        try:
            return policy.init_hidden(num_envs, device)
        except TypeError:
            try:
                return policy.init_hidden(num_envs)
            except Exception:  # pragma: no cover
                return None
        except Exception:  # pragma: no cover
            return None
    return None


def _mask_hidden(hidden_state: Any, dones: Any) -> Any:
    """Zero the recurrent state of environments that auto-reset."""
    if hidden_state is None or dones is None:
        return hidden_state
    try:
        import torch
    except Exception:  # pragma: no cover
        return hidden_state
    keep = (1.0 - dones.float()).view(-1, 1)

    def _mask(tensor: Any) -> Any:
        if not hasattr(tensor, "shape") or tensor.dim() < 3:
            return tensor
        return tensor * keep.unsqueeze(0).expand_as(tensor)

    if isinstance(hidden_state, (tuple, list)):
        return type(hidden_state)(_mask(h) for h in hidden_state)
    return _mask(hidden_state)


def evaluate_policy(policy: Any, env: Any, config: Any,
                    num_episodes: int = DEFAULT_EPISODES,
                    num_steps: Optional[int] = None,
                    deterministic: bool = True,
                    policy_index: Optional[int] = None,
                    device: Any = None) -> Dict[str, float]:
    """Roll ``policy`` in ``env`` and return mean episode statistics.

    Because the IsaacGym vectorised tasks auto-reset, "episode" statistics are read
    from the environment ``info`` dicts when available, and otherwise reconstructed
    from the reward stream over ``num_steps`` steps.
    """
    import torch  # lazy

    if device is None:
        device = getattr(config, "device", None)
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() or "cpu" in device
                             else "cpu")

    num_envs = int(getattr(config, "num_envs", 0) or getattr(env, "num_envs", 1))
    horizon = int(num_steps or getattr(config, "horizon_length", 16) or 16)
    max_episode_length = int(getattr(config, "max_episode_length", 200) or 200)
    steps = max(horizon, int(math.ceil(max_episode_length / horizon)) * horizon)
    steps = steps * max(1, int(num_episodes))

    if policy_index is None and int(getattr(config, "num_policies", 1)) > 1:
        policy_index = _policy_index_for_sapg(config)

    reset = getattr(env, "reset", None)
    step_fn = getattr(env, "step", None)
    if reset is None or step_fn is None:  # pragma: no cover - defensive
        raise AttributeError("environment must expose reset() and step()")

    obs = reset()
    if not hasattr(obs, "to") and hasattr(torch, "as_tensor"):
        obs = torch.as_tensor(obs)
    phi = _phi_for(policy, policy_index)
    hidden = _init_hidden(policy, num_envs, device)

    episode_returns: List[float] = []
    episode_lengths: List[float] = []
    episode_successes: List[float] = []
    running_return = torch.zeros(num_envs, device=getattr(obs, "device", None))
    running_length = torch.zeros(num_envs, device=getattr(obs, "device", None))
    reward_sum = 0.0
    reward_sq_sum = 0.0
    num_reward_samples = 0

    for _ in range(steps):
        with torch.no_grad():
            out = _act(policy, obs, phi, hidden, None, policy_index, deterministic)
        actions = out.get("actions") if isinstance(out, dict) else out
        if isinstance(out, dict) and out.get("hidden_state") is not None:
            hidden = out["hidden_state"]

        result = step_fn(actions)
        if isinstance(result, (tuple, list)):
            obs, rewards, dones = result[0], result[1], result[2]
            info = result[3] if len(result) > 3 else {}
        else:  # pragma: no cover - dict-style step
            obs = result.get("obs")
            rewards = result.get("rewards", result.get("reward"))
            dones = result.get("dones", result.get("done"))
            info = result.get("info", result)

        if not hasattr(rewards, "to"):  # pragma: no cover - list-based env
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            dones = torch.as_tensor(dones, dtype=torch.float32)

        rewards = rewards.float().reshape(-1)
        dones = dones.float().reshape(-1)
        reward_sum += float(rewards.sum())
        reward_sq_sum += float((rewards ** 2).sum())
        num_reward_samples += int(rewards.numel())

        running_return = running_return + rewards
        running_length = running_length + 1.0

        # Per-episode success counts reported by the task wrappers.
        info_success = _get(info, SUCCESS_KEYS)
        if info_success is not None:
            try:
                info_success = torch.as_tensor(info_success).float().reshape(-1)
                if info_success.numel() == num_envs:
                    episode_successes.extend(info_success[dones > 0.5].tolist())
            except Exception:  # pragma: no cover - defensive
                pass

        done_mask = dones > 0.5
        if bool(done_mask.any()):
            episode_returns.extend(running_return[done_mask].tolist())
            episode_lengths.extend(running_length[done_mask].tolist())
            running_return = running_return * (1.0 - dones)
            running_length = running_length * (1.0 - dones)

        hidden = _mask_hidden(hidden, dones)

    def _mean(values: Sequence[float], default: float = float("nan")) -> float:
        vals = [v for v in values if v == v]  # drop NaNs
        if not vals:
            return default
        return float(sum(vals) / len(vals))

    # If no episode finished within the budget, fall back to the mean per-step reward.
    mean_return = _mean(episode_returns)
    if mean_return != mean_return:  # NaN
        mean_return = reward_sum / max(1, num_reward_samples) * max_episode_length
    mean_length = _mean(episode_lengths)
    if mean_length != mean_length:
        mean_length = float(steps)

    metrics: Dict[str, float] = {
        "episode_return": float(mean_return),
        "episode_length": float(mean_length),
        "mean_step_reward": float(reward_sum / max(1, num_reward_samples)),
        "episode_return_std": float(
            math.sqrt(max(0.0, reward_sq_sum / max(1, num_reward_samples)
                          - (reward_sum / max(1, num_reward_samples)) ** 2))),
        "num_episodes": float(len(episode_returns)),
        "num_steps": float(steps),
        "num_envs": float(num_envs),
    }
    if episode_successes:
        metrics["successes"] = _mean(episode_successes)
        metrics["success_count"] = float(sum(episode_successes))
    return metrics


def evaluate_seed(config: Any, checkpoint: Optional[str], seed: int,
                  args: argparse.Namespace,
                  policy: Any = None, env: Any = None) -> Dict[str, float]:
    """Evaluate one (task, seed) pair, loading the checkpoint when given."""
    import torch  # lazy

    set_seed(seed, env)
    own_env = env is None
    own_policy = policy is None
    if own_env:
        env = make_env_for(config, args)
    if own_policy:
        policy = make_policy_for(config)
        if checkpoint:
            load_checkpoint(policy, checkpoint, device=getattr(config, "device", None))
    try:
        if getattr(config, "device", None) and hasattr(policy, "to"):
            try:
                policy.to(torch.device(config.device))
            except Exception:  # pragma: no cover - CPU-only machines
                pass
        metrics = evaluate_policy(
            policy, env, config,
            num_episodes=int(args.episodes),
            num_steps=args.num_steps,
            deterministic=bool(args.deterministic),
        )
    finally:
        if own_env:
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover
                    pass
    metrics["seed"] = float(seed)
    return metrics


def set_seed(seed: int, env: Any = None) -> int:
    """Seed python / numpy / torch (and the env when it supports it)."""
    import random as _random

    _random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except Exception:  # pragma: no cover - numpy optional
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch optional
        pass
    for name in ("seed", "set_seed"):
        fn = getattr(env, name, None)
        if callable(fn):
            try:
                fn(seed)
                break
            except Exception:  # pragma: no cover
                continue
    return seed


def _seed_list(args: argparse.Namespace) -> List[int]:
    if args.seed is not None:
        return [int(args.seed)]
    return list(range(max(1, int(args.num_seeds))))


def _checkpoint_for(args: argparse.Namespace, task: str,
                    seed: int) -> Optional[str]:
    """Resolve the checkpoint path for ``(task, seed)``."""
    if args.checkpoint:
        base, ext = os.path.splitext(args.checkpoint)
        candidate = f"{base}_seed{seed}{ext}"
        if len(_seed_list(args)) > 1 and os.path.exists(candidate):
            return candidate
        return args.checkpoint
    if args.checkpoint_dir:
        method = _normalise_method(args.method)
        candidates = [
            os.path.join(args.checkpoint_dir, task, f"{method}_seed{seed}.pt"),
            os.path.join(args.checkpoint_dir, task, f"{method}.pt"),
            os.path.join(args.checkpoint_dir, f"{task}_{method}_seed{seed}.pt"),
            os.path.join(args.checkpoint_dir, f"{task}_{method}.pt"),
            os.path.join(args.checkpoint_dir, task, f"seed{seed}.pt"),
            os.path.join(args.checkpoint_dir, task, "checkpoint.pt"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
    return None


# --------------------------------------------------------------------------------------
# Aggregation with the paper's shaded band
# --------------------------------------------------------------------------------------
def paper_standard_error(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and the paper's band half-width for a set of seed results.

    The paper reports ``mean +- (2 / sqrt(n)) * sum_i (y(t) - y_i(t))^2``; for a
    scalar final result this reduces to the (unbiased-ish) spread over seeds used
    when filling Table 1.
    """
    vals = [float(v) for v in values if v == v]
    if not vals:
        return float("nan"), float("nan")
    n = len(vals)
    mean = sum(vals) / n
    if n <= 1:
        return mean, 0.0
    band = (2.0 / math.sqrt(n)) * sum((mean - v) ** 2 for v in vals) / max(1, n - 1)
    return mean, math.sqrt(max(0.0, band))


def aggregate_seed_metrics(per_seed: Sequence[Dict[str, float]],
                           task: str) -> Dict[str, float]:
    """Aggregate per-seed metric dicts into mean / band per metric + primary metric."""
    if not per_seed:
        return {}
    keys = sorted({k for metrics in per_seed for k in metrics})
    out: Dict[str, float] = {"num_seeds": float(len(per_seed))}
    for key in keys:
        values = [m.get(key) for m in per_seed if m.get(key) is not None]
        mean, band = paper_standard_error([v for v in values if v is not None])
        out[key] = mean
        out[f"{key}_band"] = band
        out[f"{key}_std"] = _std([v for v in values if v is not None])
    out["primary_metric"] = primary_metric_name(task)
    out["primary_value"] = out.get(out["primary_metric"], float("nan"))
    out["primary_band"] = out.get(f"{out['primary_metric']}_band", float("nan"))
    return out


def _std(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v == v]
    if len(vals) <= 1:
        return 0.0
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


def primary_metric_name(task: str) -> str:
    """The metric Table 1 reports for ``task``."""
    return "successes" if task in SUCCESS_TASKS else "episode_return"


# --------------------------------------------------------------------------------------
# Top-level drivers
# --------------------------------------------------------------------------------------
def evaluate_task(task: str, args: argparse.Namespace,
                  policy: Any = None, env: Any = None) -> Dict[str, Any]:
    """Evaluate all seeds of one task and return the aggregated summary."""
    task = _normalise_task(task)
    config = load_config(args, task)
    if policy is None:
        policy = make_policy_for(config)
    own_env = env is None
    if own_env:
        env = make_env_for(config, args)

    seeds = _seed_list(args)
    per_seed: List[Dict[str, float]] = []
    start = time.time()
    try:
        for seed in seeds:
            checkpoint = _checkpoint_for(args, task, seed)
            if checkpoint and not args.checkpoint and args.checkpoint_dir:
                if args.verbose:
                    print(f"[evaluate] {task} seed={seed} checkpoint={checkpoint}")
            metrics = evaluate_seed(config, checkpoint, seed, args,
                                    policy=policy if not own_env else None,
                                    env=env if not own_env else None)
            per_seed.append(metrics)
            if args.verbose:
                print(f"[evaluate] {task} seed={seed}: "
                      f"{ {k: round(v, 4) for k, v in metrics.items()} }")
    finally:
        if own_env:
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover
                    pass

    summary: Dict[str, Any] = {
        "task": task,
        "method": _normalise_method(args.method),
        "num_envs": int(getattr(config, "num_envs", 0)),
        "num_policies": int(getattr(config, "num_policies", 1)),
        "phi_dim": int(getattr(config, "phi_dim", 0)),
        "entropy_coefficient": float(getattr(config, "entropy_coefficient", 0.0)),
        "episodes": int(args.episodes),
        "seeds": seeds,
        "per_seed": per_seed,
        "wall_time": time.time() - start,
    }
    summary.update(aggregate_seed_metrics(per_seed, task))

    if args.paper_comparison:
        summary["paper"] = _paper_reference(task, config, args)
    return summary


def _paper_reference(task: str, config: Any,
                     args: argparse.Namespace) -> Dict[str, float]:
    """Table-1 reference numbers for ``task`` (choosing the sigma variant used)."""
    sigma = float(getattr(config, "entropy_coefficient", 0.0) or 0.0)
    table = PAPER_RESULTS["sapg"] if abs(sigma) < 1e-12 else PAPER_RESULTS["sapg_sigma_0.005"]
    entry = table.get(task)
    if entry is None:
        return {}
    return {"value": entry[0], "band": entry[1], "sigma": sigma}


def format_table(rows: Sequence[Dict[str, Any]],
                 headers: Sequence[str] = ()) -> str:
    """Render a fixed-width text table (Table-1 style)."""
    if not rows:
        return ""
    cols = list(headers) if headers else list(rows[0].keys())
    widths = []
    for col in cols:
        width = len(str(col))
        for row in rows:
            width = max(width, len(_fmt_cell(row.get(col))))
        widths.append(width)

    def _line() -> str:
        return "+-" + "-+-".join("-" * w for w in widths) + "-+"

    lines = [_line(),
             "| " + " | ".join(str(c).ljust(w) for c, w in zip(cols, widths)) + " |",
             _line()]
    for row in rows:
        lines.append("| " + " | ".join(_fmt_cell(row.get(c)).ljust(w)
                                       for c, w in zip(cols, widths)) + " |")
    lines.append(_line())
    return "\n".join(lines)


def _fmt_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value != value:
            return "-"
        if abs(value) >= 1e4 or (value != 0 and abs(value) < 1e-2):
            return f"{value:.3e}"
        return f"{value:.3f}"
    return str(value)


def evaluate_all(args: argparse.Namespace) -> Dict[str, Any]:
    """Evaluate a single method across every paper task."""
    method = _normalise_method(args.method)
    results: Dict[str, Any] = {"method": method, "tasks": {}}
    for task in TASKS:
        if args.verbose:
            print(f"\n[evaluate] === {method} / {task} ===")
        task_args = copy.copy(args)
        task_args.task = task
        # Share one env/policy across seeds but not across tasks.
        config = load_config(task_args, task)
        policy = make_policy_for(config)
        env = make_env_for(config, task_args)
        try:
            results["tasks"][task] = evaluate_task(task, task_args,
                                                   policy=policy, env=env)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover
                    pass
    results["table"] = summarise_table(results["tasks"])
    return results


def summarise_table(task_summaries: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the Table-1 rows from per-task summaries."""
    rows: List[Dict[str, Any]] = []
    for task, summary in task_summaries.items():
        metric = summary.get("primary_metric", primary_metric_name(task))
        row: Dict[str, Any] = {
            "task": task,
            "metric": metric,
            "mean": summary.get("primary_value", float("nan")),
            "band": summary.get("primary_band", float("nan")),
            "seeds": summary.get("num_seeds", 0),
        }
        paper = summary.get("paper") or {}
        if paper:
            row["paper"] = paper.get("value")
            row["delta"] = row["mean"] - paper.get("value", float("nan"))
        rows.append(row)
    return rows


def write_outputs(results: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    """Persist the evaluation summary to JSON. Returns the written paths."""
    written: List[str] = []
    if args.output:
        path = _ensure_dir(args.output)
        with open(path, "w") as handle:
            json.dump(results, handle, indent=2, default=float)
        written.append(path)
    elif args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for task, summary in results.get("tasks", {}).items():
            path = os.path.join(args.output_dir, f"{task}.json")
            with open(path, "w") as handle:
                json.dump(summary, handle, indent=2, default=float)
            written.append(path)
        path = os.path.join(args.output_dir, "summary.json")
        with open(path, "w") as handle:
            json.dump(results, handle, indent=2, default=float)
        written.append(path)
    return written


def print_summary(results: Dict[str, Any]) -> None:
    """Print the per-task table (and per-seed detail with ``--verbose``)."""
    method = results.get("method", "?")
    print(f"\n=== Evaluation summary ({method}) ===")
    rows = results.get("table")
    if rows:
        print(format_table(rows, headers=("task", "metric", "mean", "band",
                                          "seeds", "paper", "delta")))
    else:
        tasks = results.get("tasks", {})
        for task, summary in tasks.items():
            metric = summary.get("primary_metric", primary_metric_name(task))
            print(f"  {task:<14} {metric:<16} "
                  f"{_fmt_cell(summary.get('primary_value'))} "
                  f"+- {_fmt_cell(summary.get('primary_band'))}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)

    if _normalise_task(args.task) == "all":
        results = evaluate_all(args)
    else:
        task = _normalise_task(args.task)
        config = load_config(args, task)
        policy = make_policy_for(config)
        env = make_env_for(config, args)
        try:
            summary = evaluate_task(task, args, policy=policy, env=env)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover
                    pass
        results = {"method": _normalise_method(args.method),
                   "tasks": {task: summary},
                   "table": summarise_table({task: summary})}

    print_summary(results)
    written = write_outputs(results, args)
    for path in written:
        print(f"[evaluate] wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

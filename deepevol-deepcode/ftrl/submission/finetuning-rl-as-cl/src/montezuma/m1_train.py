"""M1 trainer for Montezuma's Revenge: PPO + Random Network Distillation.

This module trains the *exploration agent* ``M1`` **from scratch** until it reaches
an episode cumulative reward of roughly ``7000`` (Section 3 / Appendix B.2 of
Wolczyk et al., 2024, Tables 2).  ``M1`` is the agent whose trajectories subsume
Room 7 (the FAR boundary) and from which the behavioural-cloning pre-training
dataset for ``M2`` (``pi_*``) is collected in :mod:`src.montezuma.m2_bc`.

Pipeline position
-----------------
``M1`` (this file, scratch PPO + RND)
    -> ``M2`` (:mod:`src.montezuma.m2_bc`, BC pretraining on Room-7+ trajectories)
    -> fine-tuning (:mod:`src.montezuma.train_montezuma`, vanilla / +BC / +EWC)

Design notes
------------
* The heavy lifting lives in :mod:`src.montezuma.ppo_rnd` (agent, rollout storage,
  GAE) and :mod:`src.montezuma.env` (Montezuma vectorised env).  This module is a
  thin, self-contained *outer* training loop so that:
    - early stopping on the target episode return is possible (the paper stops
      pretraining once ~7000 cumulative reward is reached), and
    - checkpoints / metrics are written with the shared ``src.common`` utilities.
* Torch is imported defensively: importing this module must never fail in a
  documentation-only or CPU-only environment.  All training entry points raise a
  clear ``RuntimeError`` if torch is missing.
* NumPy is optional as well; the loop degrades to pure-Python bookkeeping.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Defensive imports
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on the environment
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

try:  # pragma: no cover
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

# --- intra-package imports (tolerant to `src.` prefix or relative layout) ---
try:
    from src.montezuma.env import (  # type: ignore
        MAX_STEPS_PER_EPISODE,
        NUM_ACTIONS,
        OBS_SHAPE,
        ROOM7,
        MontezumaVecEnv,
        evaluate_policy as env_evaluate_policy,
        make_env,
        room_visitation,
    )
except Exception:  # pragma: no cover - relative fallback
    from .env import (  # type: ignore
        MAX_STEPS_PER_EPISODE,
        NUM_ACTIONS,
        OBS_SHAPE,
        ROOM7,
        MontezumaVecEnv,
        evaluate_policy as env_evaluate_policy,
        make_env,
        room_visitation,
    )

try:
    from src.montezuma.ppo_rnd import (  # type: ignore
        DEFAULT_TOTAL_STEPS,
        PPORNDAgent,
        PPORNDConfig,
        RolloutStorage,
    )
except Exception:  # pragma: no cover - relative fallback
    from .ppo_rnd import (  # type: ignore
        DEFAULT_TOTAL_STEPS,
        PPORNDAgent,
        PPORNDConfig,
        RolloutStorage,
    )

try:
    from src.common.seeding import set_seed  # type: ignore
except Exception:  # pragma: no cover
    def set_seed(seed: int, deterministic: bool = False) -> int:  # type: ignore
        import random

        random.seed(seed)
        if _np is not None:
            _np.random.seed(seed)
        if _HAS_TORCH:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        return seed

try:
    from src.common.logging_utils import MetricLogger, get_logger  # type: ignore
except Exception:  # pragma: no cover
    MetricLogger = None  # type: ignore
    get_logger = None  # type: ignore

try:
    from src.common.checkpointing import ensure_dir, save_checkpoint  # type: ignore
except Exception:  # pragma: no cover
    def ensure_dir(path: str) -> str:  # type: ignore
        if path:
            os.makedirs(path, exist_ok=True)
        return path

    def save_checkpoint(path: str, state: Dict[str, Any], step: Optional[int] = None) -> str:  # type: ignore
        if not _HAS_TORCH:  # pragma: no cover
            raise RuntimeError("torch is required to save checkpoints")
        if step is not None and not path.endswith("_step_%d.pt" % step):
            base, ext = os.path.splitext(path)
            path = "%s_step_%d%s" % (base, step, ext or ".pt")
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        torch.save(state, path)
        return path


__all__ = [
    "M1Result",
    "M1Config",
    "DEFAULT_TARGET_RETURN",
    "DEFAULT_EVAL_EVERY",
    "DEFAULT_EVAL_EPISODES",
    "DEFAULT_SAVE_EVERY",
    "EPISODE_WINDOW",
    "build_config",
    "train_m1",
    "evaluate_m1",
    "m1_reached_target",
    "load_m1_agent",
    "save_m1_checkpoint",
    "main",
    "build_parser",
]


# ---------------------------------------------------------------------------
# Constants (paper: Section 3 / Appendix B.2)
# ---------------------------------------------------------------------------
DEFAULT_TARGET_RETURN = 7_000.0      # paper: "train M1 until ~7000 cumulative reward"
DEFAULT_EVAL_EVERY = 5_000_000       # paper evaluates / logs every 5M steps
DEFAULT_EVAL_EPISODES = 100
DEFAULT_SAVE_EVERY = 25_000_000
EPISODE_WINDOW = 100                 # window for the running episode-return statistic
SMOKE_TEST_STEPS = 2_048


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _get(obj: Any, path: str, default: Any = None) -> Any:
    """Dotted-path lookup tolerant to ``Config`` / dict / object shapes."""
    if obj is None:
        return default
    cur = obj
    for part in str(path).split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return default if cur is None else cur


def _to_tensor(obs: Any, device: Any = "cpu"):
    """Convert a (batched) observation to a float tensor on ``device``."""
    if not _HAS_TORCH:  # pragma: no cover
        raise RuntimeError("torch is required for M1 training")
    if torch.is_tensor(obs):
        t = obs
    elif isinstance(obs, dict):
        # Montezuma observations are plain arrays; dicts only appear in Meta-World.
        for key in ("obs", "observation", "state"):
            if key in obs:
                return _to_tensor(obs[key], device)
        raise TypeError("cannot convert dict observation to tensor")
    else:
        if _np is not None:
            obs = _np.asarray(obs)
        t = torch.as_tensor(obs)
    if t.dim() == 3:  # single (C,H,W) -> (1,C,H,W)
        t = t.unsqueeze(0)
    if t.dtype == torch.uint8:
        t = t.float().div_(255.0)
    else:
        t = t.float()
        if t.numel() and float(t.max()) > 2.0:
            t = t / 255.0
    return t.to(device)


def _split_step(out: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    """Normalise legacy (4-tuple) and gymnasium (5-tuple) step returns."""
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated), bool(truncated), info or {}
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return obs, float(reward), bool(done), False, info or {}
    raise TypeError("unexpected step() return value: %r" % (type(out),))


def build_config(cfg: Any = None, **overrides: Any) -> "PPORNDConfig":
    """Build the M1 :class:`PPORNDConfig`, merging a loaded YAML config if given."""
    base = PPORNDConfig()
    if cfg is not None:
        try:
            base = PPORNDConfig.from_config(cfg)
        except Exception:
            # Fall back to explicit field copying when the struct differs.
            for name in list(getattr(base, "__dataclass_fields__", {}).keys()):
                value = _get(cfg, name, None)
                if value is None:
                    value = _get(cfg, "ppo.%s" % name, None)
                if value is not None:
                    setattr(base, name, value)
    # M1 is always trained from scratch.
    base.method = "scratch"
    if overrides:
        if hasattr(base, "with_overrides"):
            base = base.with_overrides(**{k: v for k, v in overrides.items() if v is not None})
        else:  # pragma: no cover
            for k, v in overrides.items():
                if v is not None and hasattr(base, k):
                    setattr(base, k, v)
    return base


def m1_reached_target(episode_return: Optional[float], target_return: float = DEFAULT_TARGET_RETURN) -> bool:
    """Whether the (windowed) episode return has reached the paper's ~7000 target."""
    if episode_return is None:
        return False
    try:
        return float(episode_return) >= float(target_return)
    except Exception:  # pragma: no cover
        return False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class M1Result:
    """Outcome of an M1 training run (from scratch, PPO + RND)."""

    steps: int = 0
    episode_return: Optional[float] = None
    best_episode_return: Optional[float] = None
    room7_success_rate: Optional[float] = None
    max_room: Optional[int] = None
    target_return: float = DEFAULT_TARGET_RETURN
    target_reached: bool = False
    checkpoint: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    elapsed: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": "scratch",
            "agent": "M1",
            "steps": int(self.steps),
            "episode_return": self.episode_return,
            "best_episode_return": self.best_episode_return,
            "room7_success_rate": self.room7_success_rate,
            "max_room": self.max_room,
            "target_return": self.target_return,
            "target_reached": bool(self.target_reached),
            "checkpoint": self.checkpoint,
            "history": list(self.history),
            "config": dict(self.config),
            "elapsed": float(self.elapsed),
        }


M1Config = PPORNDConfig


# ---------------------------------------------------------------------------
# Rollout collection (self-contained so the outer loop owns step accounting)
# ---------------------------------------------------------------------------
def _collect_rollout(
    agent: "PPORNDAgent",
    env: Any,
    storage: "RolloutStorage",
    obs: Any,
    device: Any,
    reward_accumulator: List[float],
    episode_returns: List[float],
    episode_lengths: List[int],
    info_history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Collect ``storage.num_step`` transitions and update return bookkeeping.

    Returns ``(next_obs, stats)`` where ``stats`` holds the running episode-return
    statistics used for the ~7000 early-stopping criterion.
    """
    obs_tensor = _to_tensor(obs, device)
    storage.reset(obs_tensor)

    num_env = storage.num_env
    acc = list(reward_accumulator)
    if len(acc) != num_env:
        acc = [0.0] * num_env
    lengths = list(episode_lengths)
    if len(lengths) != num_env:
        lengths = [0] * num_env

    for _t in range(storage.num_step):
        with torch.no_grad():
            action, log_prob, entropy, value = agent.act(obs_tensor, deterministic=False)
            next_obs_raw, reward, terminated, truncated, info = _split_step(env.step(action))
            next_obs_tensor = _to_tensor(next_obs_raw, device)
            # Intrinsic (RND) reward is the prediction error of the *new* state,
            # exactly as in the reference random-network-distillation loop.
            raw_int, norm_int = agent.intrinsic_reward(next_obs_tensor)

        done = terminated or truncated
        mask = 0.0 if terminated else 1.0  # truncated episodes must bootstrap
        storage.insert(
            obs_tensor,
            action,
            log_prob,
            value,
            reward,
            mask,
            entropies=entropy,
            int_rewards=norm_int,
            int_raw=raw_int,
        )

        # episode-return bookkeeping
        rewards = reward if isinstance(reward, (list, tuple)) else [reward]
        if _np is not None and isinstance(reward, _np.ndarray):
            rewards = list(reward.reshape(-1))
        for i in range(min(num_env, len(rewards))):
            acc[i] = acc[i] + float(rewards[i])
            lengths[i] += 1
            if done and i < len(rewards):
                episode_returns.append(acc[i])
                episode_lengths.append(lengths[i])
                acc[i] = 0.0
                lengths[i] = 0
        if done:
            if info_history is not None:
                info_history.append(info if isinstance(info, dict) else {})
            # auto-reset semantics: the vec env returns the new episode's first obs
        obs_tensor = next_obs_tensor
        obs = next_obs_raw

    with torch.no_grad():
        _, _, _, last_value = agent.act(obs_tensor, deterministic=True)
        _, last_int = agent.intrinsic_reward(obs_tensor)
    storage.set_last_value(last_value)
    storage.compute_returns(last_value=last_value.detach() if torch.is_tensor(last_value) else last_value)

    if episode_returns:
        window = episode_returns[-EPISODE_WINDOW:]
        mean_return = sum(window) / float(len(window))
        best = max(episode_returns)
    else:
        mean_return = None
        best = None
    stats = {
        "mean_return": mean_return,
        "window_return": mean_return,
        "best_return": best,
        "num_episodes": len(episode_returns),
        "mean_length": (sum(episode_lengths[-EPISODE_WINDOW:]) / float(len(episode_lengths[-EPISODE_WINDOW:])))
        if episode_lengths
        else None,
    }
    return obs, stats


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_m1(
    agent: Any,
    *,
    num_episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
    stub: bool = False,
    env: Any = None,
    max_steps: Optional[int] = None,
    far_room: int = ROOM7,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Evaluate an agent: episode return and Room-7 success rate (Figure 6 metric)."""
    if env is None:
        env = make_env(seed=seed, stub=stub, far_room=far_room)
    try:
        metrics = env_evaluate_policy(
            agent,
            env=env,
            num_episodes=num_episodes,
            seed=seed,
            deterministic=deterministic,
            stub=stub,
            far_room=far_room,
            max_steps=max_steps,
        )
    finally:
        if env is not None and not stub:
            try:
                env.close()
            except Exception:
                pass
    metrics = dict(metrics or {})
    metrics.setdefault("episode_return", metrics.get("return_mean"))
    metrics.setdefault("mean_return", metrics.get("return_mean"))
    return metrics


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_m1_checkpoint(agent: Any, path: str, step: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> str:
    """Save the M1 policy + RND predictor + optimizer state."""
    state: Dict[str, Any] = {"agent": agent.state_dict(), "method": "scratch", "agent_name": "M1"}
    if extra:
        state.update(extra)
    return save_checkpoint(path, state, step=step)


def load_m1_agent(path: str, *, cfg: Any = None, stub: bool = False, device: Any = None, load_optimizer: bool = False) -> "PPORNDAgent":
    """Reconstruct an M1 agent from a checkpoint produced by :func:`save_m1_checkpoint`."""
    if not _HAS_TORCH:  # pragma: no cover
        raise RuntimeError("torch is required to load an M1 checkpoint")
    config = build_config(cfg)
    agent = PPORNDAgent(config=config, device=device or getattr(config, "device", "cpu"))
    payload = torch.load(path, map_location=device or "cpu")
    state = payload.get("agent", payload) if isinstance(payload, dict) else payload
    try:
        agent.load_state_dict(state, load_optimizer=load_optimizer)
    except TypeError:  # pragma: no cover - older signature
        agent.load_state_dict(state)
    return agent


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------
def train_m1(
    cfg: Any = None,
    *,
    total_steps: Optional[int] = None,
    target_return: float = DEFAULT_TARGET_RETURN,
    stub: Optional[bool] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    logger: Any = None,
    eval_every: Optional[int] = None,
    eval_episodes: Optional[int] = None,
    save_every: Optional[int] = None,
    env: Any = None,
    agent: Any = None,
    resume: Optional[str] = None,
    init_checkpoint: Optional[str] = None,
    progress_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    verbose: bool = True,
) -> Tuple["PPORNDAgent", M1Result]:
    """Train the from-scratch Montezuma agent M1 with PPO + RND.

    The loop stops when either

    * the running (windowed) episode cumulative reward reaches ``target_return``
      (the paper's ~7000), or
    * ``total_steps`` environment steps have been collected.

    Returns ``(agent, result)``.
    """
    if not _HAS_TORCH:  # pragma: no cover
        raise RuntimeError("torch is required to train M1 (PPO + RND)")

    config = build_config(cfg if cfg is not None else None)
    total_steps = int(total_steps if total_steps is not None else getattr(config, "total_steps", DEFAULT_TOTAL_STEPS))
    stub = bool(getattr(config, "stub", False) if stub is None else stub)
    seed = int(seed if seed is not None else (getattr(config, "seed", 0) or 0))
    device = device or getattr(config, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    eval_every = int(eval_every if eval_every is not None else getattr(config, "eval_every", DEFAULT_EVAL_EVERY) or DEFAULT_EVAL_EVERY)
    eval_episodes = int(
        eval_episodes if eval_episodes is not None else getattr(config, "eval_episodes", DEFAULT_EVAL_EPISODES) or DEFAULT_EVAL_EPISODES
    )
    save_every = int(save_every if save_every is not None else getattr(config, "save_every", DEFAULT_SAVE_EVERY) or DEFAULT_SAVE_EVERY)
    num_env = int(getattr(config, "num_env", 128) or 128)
    num_step = int(getattr(config, "num_step", 128) or 128)
    batch_size = num_env * num_step

    output_dir = ensure_dir(output_dir or os.path.join("runs", "montezuma", "m1_scratch_seed%d" % seed))
    log = get_logger("montezuma.m1") if get_logger is not None else None
    if logger is None and MetricLogger is not None:
        try:
            logger = MetricLogger(os.path.join(output_dir, "tb"), use_tensorboard=True)
        except Exception:
            logger = None

    set_seed(seed)
    if env is None:
        env = MontezumaVecEnv(num_envs=num_env, base_seed=seed, stub=stub, far_room=ROOM7)
    if agent is None:
        agent = PPORNDAgent(config=config, device=device, seed=seed)
    if init_checkpoint and os.path.exists(init_checkpoint):
        state = torch.load(init_checkpoint, map_location=device)
        agent.load_state_dict(state.get("agent", state), load_optimizer=False)

    storage = RolloutStorage(
        num_step=num_step,
        num_env=num_env,
        obs_shape=tuple(getattr(config, "observation_shape", OBS_SHAPE)),
        device=device,
        num_actions=int(getattr(config, "num_actions", NUM_ACTIONS) or NUM_ACTIONS),
        use_gae=bool(getattr(config, "use_gae", True)),
        gamma=float(getattr(config, "gamma", 0.999)),
        lam=float(getattr(config, "lam", 0.95)),
        int_gamma=float(getattr(config, "int_gamma", 0.99)),
        int_lam=getattr(config, "int_lam", None),
    )

    start_step = 0
    history: List[Dict[str, Any]] = []
    if resume and os.path.exists(resume):
        payload = torch.load(resume, map_location=device)
        agent.load_state_dict(payload.get("agent", payload), load_optimizer=True)
        start_step = int(payload.get("step", 0))
        if verbose and log is not None:
            log.info("resumed M1 from %s at step %d", resume, start_step)

    obs = env.reset()
    reward_acc = [0.0] * num_env
    episode_lengths: List[int] = []
    episode_returns: List[int] = []
    best_return: Optional[float] = None
    t0 = time.time()
    step = start_step
    last_eval: Optional[Dict[str, Any]] = None
    next_eval = start_step + eval_every
    checkpoint_path: Optional[str] = None
    target_reached = False

    if verbose and log is not None:
        log.info(
            "training M1 from scratch: num_env=%d num_step=%d batch=%d total_steps=%d target_return=%.0f device=%s",
            num_env, num_step, batch_size, total_steps, target_return, device,
        )

    while step < total_steps:
        obs, stats = _collect_rollout(agent, env, storage, obs, device, reward_acc, episode_returns, episode_lengths)
        # recover per-env reward accumulator from the rollout bookkeeping
        reward_acc = [0.0] * num_env
        update_stats = agent.update(storage)
        step += batch_size

        metrics: Dict[str, float] = {"train/step": float(step)}
        if update_stats:
            for k, v in update_stats.items():
                try:
                    metrics["train/%s" % k] = float(v)
                except Exception:
                    continue
        if stats.get("window_return") is not None:
            metrics["train/episode_return"] = float(stats["window_return"])
        if stats.get("best_return") is not None:
            metrics["train/best_episode_return"] = float(stats["best_return"])
        if logger is not None:
            try:
                logger.add_dict(metrics)
                logger.flush(step)
            except Exception:
                pass

        if m1_reached_target(stats.get("window_return"), target_return):
            target_reached = True

        if step >= next_eval or target_reached or step >= total_steps:
            try:
                last_eval = evaluate_m1(agent, num_episodes=eval_episodes, seed=seed, stub=stub)
            except Exception as exc:  # pragma: no cover - evaluation is best effort
                last_eval = {"error": str(exc)}
            record = {"step": int(step)}
            record.update({k: v for k, v in (last_eval or {}).items() if isinstance(v, (int, float))})
            if stats.get("window_return") is not None:
                record["train_episode_return"] = float(stats["window_return"])
            history.append(record)
            ret = record.get("return_mean", record.get("episode_return"))
            if ret is not None:
                best_return = max(best_return, float(ret)) if best_return is not None else float(ret)
            if logger is not None:
                try:
                    logger.add_dict({"eval/%s" % k: float(v) for k, v in (last_eval or {}).items() if isinstance(v, (int, float))})
                    logger.flush(step)
                except Exception:
                    pass
            if verbose and log is not None:
                log.info(
                    "step %s | eval return=%.1f room7=%.3f | train window=%.1f",
                    step,
                    float(ret) if ret is not None else float("nan"),
                    float((last_eval or {}).get("room7_success_rate", float("nan"))),
                    float(stats.get("window_return") or float("nan")),
                )
            next_eval = step + eval_every

        if save_every and step % save_every < batch_size:
            try:
                checkpoint_path = save_m1_checkpoint(
                    agent, os.path.join(output_dir, "m1.pt"), step=step,
                    extra={"config": config.to_dict() if hasattr(config, "to_dict") else {}},
                )
            except Exception:  # pragma: no cover
                checkpoint_path = checkpoint_path

        if progress_fn is not None:
            try:
                progress_fn({"step": step, "stats": stats, "eval": last_eval})
            except Exception:
                pass

        if target_reached:
            if verbose and log is not None:
                log.info("M1 reached target episode return %.1f at step %d", float(stats.get("window_return") or 0.0), step)
            break

    # final evaluation + checkpoint
    if last_eval is None:
        try:
            last_eval = evaluate_m1(agent, num_episodes=eval_episodes, seed=seed, stub=stub)
        except Exception as exc:  # pragma: no cover
            last_eval = {"error": str(exc)}
    try:
        checkpoint_path = save_m1_checkpoint(
            agent, os.path.join(output_dir, "m1.pt"), step=step,
            extra={"config": config.to_dict() if hasattr(config, "to_dict") else {}},
        )
    except Exception:  # pragma: no cover
        pass

    episode_return = None
    if last_eval and isinstance(last_eval.get("return_mean", last_eval.get("episode_return")), (int, float)):
        episode_return = float(last_eval.get("return_mean", last_eval.get("episode_return")))
    if episode_return is not None:
        best_return = max(best_return, episode_return) if best_return is not None else episode_return

    result = M1Result(
        steps=int(step),
        episode_return=episode_return,
        best_episode_return=best_return,
        room7_success_rate=float(last_eval.get("room7_success_rate")) if isinstance(last_eval, dict) and isinstance(last_eval.get("room7_success_rate"), (int, float)) else None,
        max_room=int(last_eval["max_room"]) if isinstance(last_eval, dict) and isinstance(last_eval.get("max_room"), (int, float)) else None,
        target_return=float(target_return),
        target_reached=bool(target_reached or m1_reached_target(episode_return, target_return)),
        checkpoint=checkpoint_path,
        history=history,
        config=config.to_dict() if hasattr(config, "to_dict") else {},
        elapsed=time.time() - t0,
    )

    try:
        with open(os.path.join(output_dir, "m1_summary.json"), "w") as fh:
            json.dump(result.as_dict(), fh, indent=2, default=str)
    except Exception:  # pragma: no cover
        pass

    try:
        env.close()
    except Exception:
        pass
    if logger is not None:
        try:
            logger.close()
        except Exception:
            pass

    return agent, result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train M1 (from scratch, PPO + RND) on Montezuma's Revenge.",
    )
    parser.add_argument("--config", type=str, default=None, help="path to configs/montezuma.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="config override key=value")
    parser.add_argument("--total-steps", type=int, default=None, help="environment steps to collect")
    parser.add_argument("--target-return", type=float, default=DEFAULT_TARGET_RETURN, help="episode reward target (~7000)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--stub", action="store_true", help="use the dependency-free dummy environment")
    parser.add_argument("--smoke-test", action="store_true", help="run a few hundred steps and exit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    cfg = None
    if args.config:
        try:
            from src.common.config import apply_overrides, load_config  # type: ignore

            cfg = load_config(args.config, overrides=args.overrides)
        except Exception as exc:  # pragma: no cover
            print("failed to load config %s: %s" % (args.config, exc))
            return 2
    elif args.overrides:
        try:
            from src.common.config import Config, apply_overrides  # type: ignore

            cfg = apply_overrides(Config({}), args.overrides)
        except Exception:  # pragma: no cover
            cfg = None

    total_steps = args.total_steps
    if args.smoke_test and total_steps is None:
        total_steps = SMOKE_TEST_STEPS

    _, result = train_m1(
        cfg,
        total_steps=total_steps,
        target_return=args.target_return,
        stub=args.stub or None,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        eval_every=args.eval_every or (256 if args.smoke_test else None),
        eval_episodes=args.eval_episodes or (2 if args.smoke_test else None),
        save_every=args.save_every,
        resume=args.resume,
        init_checkpoint=args.init_checkpoint,
    )
    print(json.dumps(result.as_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

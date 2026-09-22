"""SAPG: Split and Aggregate Policy Gradients — training / evaluation entry point.

This module wires together the components implemented in the ``sapg`` package:

* environments (``envs``) split into ``num_blocks`` follower blocks,
* shared actor/critic networks conditioned on per-follower vectors ``phi_j``
  (``sapg.networks``),
* per-block follower PPO updates (``sapg.follower``),
* leader aggregation over the union of all followers' off-policy transitions
  (``sapg.leader`` / ``sapg.aggregation``),
* rollout storage (``sapg.buffer``),
* success-tolerance curriculum (``utils.curriculum``),
* metric logging (``utils.logger``) and checkpointing (``utils.checkpoint``).

Usage
-----
::

    python main.py --config configs/allegrokuka.yaml --task regrasping
    python main.py --config configs/shadow_hand.yaml --mode symmetric
    python main.py --config configs/allegrokuka.yaml --eval --checkpoint ckpt.pt

The training loop follows the paper's description:

1. Collect ``horizon`` steps of rollouts from every environment block using the
   *follower* policies (each block has its own ``phi_j``).
2. Update every follower with standard on-policy PPO on its own block data.
3. Aggregate **all** transitions from **all** followers and update the leader
   (the deployed policy) with PPO's clipped surrogate objective using
   importance weights for the off-policy data.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Make the repository importable both as a package and as a script.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from sapg.aggregation import AggregationMode, build_aggregator  # noqa: E402
from sapg.buffer import RolloutBuffer, build_buffers  # noqa: E402
from sapg.follower import Follower, build_followers  # noqa: E402
from sapg.leader import Leader, build_leader  # noqa: E402
from sapg.networks import build_networks  # noqa: E402
from sapg.ppo import KLScheduler, PPOHyperParams  # noqa: E402

from envs import make_allegro_hand, make_allegrokuka, make_shadow_hand  # noqa: E402
from envs.wrappers import make_env  # noqa: E402
from utils.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from utils.logger import MetricLogger  # noqa: E402


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config file (returns ``{}`` if the file is missing)."""
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - pyyaml is a hard dep
        raise ImportError(
            "pyyaml is required to load config files. Install with `pip install pyyaml`."
        ) from exc
    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(config_path: Optional[str], task: Optional[str] = None,
                overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load the default config, merge a task config, then CLI overrides."""
    default_path = os.path.join(_HERE, "configs", "default.yaml")
    cfg = _load_yaml(default_path)

    if config_path:
        # Allow either an absolute/relative path or a bare config name.
        candidates = [config_path]
        if not os.path.exists(config_path):
            candidates.append(os.path.join(_HERE, "configs", config_path))
            if not config_path.endswith(".yaml"):
                candidates.append(os.path.join(_HERE, "configs", config_path + ".yaml"))
        for cand in candidates:
            if os.path.exists(cand):
                cfg = _deep_update(cfg, _load_yaml(cand))
                break

    if task:
        cfg.setdefault("env", {})
        cfg["env"]["task"] = task

    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg


def _cfg_get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Fetch a nested config value, returning ``default`` when absent."""
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------
def build_environment(cfg: Dict[str, Any]) -> Any:
    """Instantiate the (wrapped) vectorized environment described by ``cfg``."""
    env_cfg = cfg.get("env", {})
    name = str(env_cfg.get("name", "allegrokuka")).lower()
    task = env_cfg.get("task", "regrasping")
    num_envs = int(env_cfg.get("num_envs", 1024))
    num_blocks = int(env_cfg.get("num_blocks", 1))
    seed = int(cfg.get("seed", 0))

    common = dict(
        num_envs=num_envs,
        episode_length=int(env_cfg.get("episode_length", 100)),
        control_dt=float(env_cfg.get("control_dt", 1.0 / 60.0)),
        action_scale=float(env_cfg.get("action_scale", 0.5)),
        w_reach=float(env_cfg.get("w_reach", 1.0)),
        w_lift=float(env_cfg.get("w_lift", 1.0)),
        w_target=float(env_cfg.get("w_target", 1.0)),
        w_success=float(env_cfg.get("w_success", 5.0)),
        w_orientation=float(env_cfg.get("w_orientation", 1.0)),
        initial_delta=float(env_cfg.get("initial_delta", 0.075)),
        min_delta=float(env_cfg.get("min_delta", 0.01)),
        decrease_factor=float(env_cfg.get("decrease_factor", 0.9)),
        success_threshold=float(env_cfg.get("success_threshold", 3.0)),
        use_curriculum=bool(env_cfg.get("use_curriculum", True)),
        seed=seed,
        device=str(cfg.get("device", "cpu")),
    )

    if name in ("allegrokuka", "allegro_kuka", "kuka"):
        env = make_allegrokuka(task=task, **common)
    elif name in ("shadow_hand", "shadowhand", "shadow"):
        env = make_shadow_hand(**common)
    elif name in ("allegro_hand", "allegrohand", "allegro"):
        env = make_allegro_hand(**common)
    else:
        raise ValueError(f"Unknown environment name: {name!r}")

    env = make_env(
        env,
        num_blocks=num_blocks,
        normalize_obs=bool(env_cfg.get("normalize_obs", True)),
        use_curriculum=bool(env_cfg.get("use_curriculum", True)),
        curriculum_config=dict(
            initial_delta=float(env_cfg.get("initial_delta", 0.075)),
            min_delta=float(env_cfg.get("min_delta", 0.01)),
            decrease_factor=float(env_cfg.get("decrease_factor", 0.9)),
            success_threshold=float(env_cfg.get("success_threshold", 3.0)),
        ),
    )
    return env


# ---------------------------------------------------------------------------
# Hyperparameter assembly
# ---------------------------------------------------------------------------
def build_hparams(cfg: Dict[str, Any]) -> PPOHyperParams:
    """Build :class:`PPOHyperParams` from the merged config."""
    ppo = cfg.get("ppo", {})
    return PPOHyperParams(
        gamma=float(ppo.get("gamma", 0.99)),
        tau=float(ppo.get("tau", 0.95)),
        clip_eps=float(ppo.get("clip_eps", 0.1)),
        entropy_coeff=float(ppo.get("entropy_coeff", 0.0)),
        critic_coeff=float(ppo.get("critic_coeff", 4.0)),
        bounds_loss_coeff=float(ppo.get("bounds_loss_coeff", 1e-4)),
        kl_threshold=float(ppo.get("kl_threshold", 0.016)),
        grad_norm_clip=float(ppo.get("grad_norm_clip", 1.0)),
        lr=float(ppo.get("lr", 1e-4)),
        mini_epochs=int(ppo.get("mini_epochs", 2)),
        horizon=int(ppo.get("horizon", 16)),
        lstm_seq_len=int(ppo.get("lstm_seq_len", 16)),
        action_bound=float(ppo.get("action_bound", 1.0)),
        extra=dict(
            mini_batch_size_mult=int(ppo.get("mini_batch_size_mult", 4)),
            min_lr=float(ppo.get("min_lr", 1e-6)),
            max_lr=float(ppo.get("max_lr", 1e-2)),
            lr_scale=float(ppo.get("lr_scale", 1.5)),
        ),
    )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class SAPGTrainer:
    """Orchestrates the SAPG split-and-aggregate training loop."""

    def __init__(self, cfg: Dict[str, Any], logger: Optional[MetricLogger] = None):
        self.cfg = cfg
        self.device = str(cfg.get("device", "cpu"))
        self.seed = int(cfg.get("seed", 0))
        self._set_seeds(self.seed)

        # --- environment -------------------------------------------------
        self.env = build_environment(cfg)
        self.num_envs = int(getattr(self.env, "num_envs", _cfg_get(cfg, "env", "num_envs", default=1024)))
        self.num_blocks = int(_cfg_get(cfg, "env", "num_blocks", default=1))
        self.obs_dim = int(getattr(self.env, "obs_dim", 0))
        self.action_dim = int(getattr(self.env, "action_dim", 0))
        if self.obs_dim <= 0 or self.action_dim <= 0:
            probe = self.env.reset()
            self.obs_dim = int(np.asarray(probe).reshape(self.num_envs, -1).shape[-1])
            self.action_dim = int(getattr(self.env, "action_dim", 1))

        # --- hyperparameters --------------------------------------------
        self.hparams = build_hparams(cfg)
        self.horizon = int(self.hparams.horizon)
        self.mini_epochs = int(self.hparams.mini_epochs)
        self.gamma = float(self.hparams.gamma)
        self.tau = float(self.hparams.tau)

        # --- networks ----------------------------------------------------
        net_cfg = dict(cfg.get("network", {}))
        net_cfg.setdefault("phi_dim", int(_cfg_get(cfg, "sapg", "phi_dim", default=0)))
        self.phi_dim = int(net_cfg.get("phi_dim", 0))
        self.policy, self.value = build_networks(
            net_cfg, self.obs_dim, self.action_dim, num_blocks=self.num_blocks
        )
        self.policy.to(self.device)
        self.value.to(self.device)

        # --- optimizer (shared across followers + leader) ----------------
        params = list(self.policy.parameters()) + list(self.value.parameters())
        self.optimizer = torch.optim.Adam(params, lr=float(self.hparams.lr))
        self.kl_scheduler = KLScheduler(
            self.optimizer,
            kl_threshold=float(self.hparams.kl_threshold),
            min_lr=float(self.hparams.extra.get("min_lr", 1e-6)),
            max_lr=float(self.hparams.extra.get("max_lr", 1e-2)),
            scale=float(self.hparams.extra.get("lr_scale", 1.5)),
        )

        # --- followers ---------------------------------------------------
        self.followers: List[Follower] = build_followers(
            self.num_blocks,
            self.policy,
            self.value,
            self.optimizer,
            phi_dim=self.phi_dim,
            hparams=self.hparams,
            kl_scheduler=self.kl_scheduler,
            device=self.device,
            seed=self.seed,
        )

        # --- leader ------------------------------------------------------
        self.leader: Optional[Leader] = build_leader(
            self.policy,
            self.value,
            self.optimizer,
            phi_dim=self.phi_dim,
            hparams=self.hparams,
            kl_scheduler=self.kl_scheduler,
            device=self.device,
            seed=self.seed + 1,
            use_importance_weights=bool(_cfg_get(cfg, "sapg", "use_importance_weights", default=True)),
        )

        # --- aggregation strategy ---------------------------------------
        self.mode = AggregationMode.from_str(str(_cfg_get(cfg, "sapg", "mode", default="leader")))
        self.aggregator = build_aggregator(
            self.mode,
            leader=self.leader,
            followers=self.followers,
            policy=self.policy,
            value=self.value,
            optimizer=self.optimizer,
            hparams=self.hparams,
            kl_scheduler=self.kl_scheduler,
            device=self.device,
            use_importance_weights=bool(_cfg_get(cfg, "sapg", "use_importance_weights", default=True)),
        )

        # --- rollout buffers ---------------------------------------------
        self.num_envs_per_block = max(1, self.num_envs // max(1, self.num_blocks))
        self.buffers: List[RolloutBuffer] = build_buffers(
            self.num_blocks,
            self.num_envs_per_block,
            self.horizon,
            self.obs_dim,
            self.action_dim,
            phi_dim=self.phi_dim,
            use_lstm=bool(net_cfg.get("use_lstm", False)),
            lstm_seq_len=int(self.hparams.lstm_seq_len),
            device=self.device,
        )

        # --- bookkeeping --------------------------------------------------
        self.logger = logger or MetricLogger(
            log_dir=_cfg_get(cfg, "log_dir", default=None),
            use_tensorboard=bool(_cfg_get(cfg, "use_tensorboard", default=False)),
            verbose=bool(_cfg_get(cfg, "verbose", default=True)),
        )
        self.iteration = 0
        self.env_steps = 0
        self._obs: Optional[np.ndarray] = None
        self._lstm_states: Optional[List[Any]] = None
        self._episode_rewards = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_successes = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)

    # -- utilities --------------------------------------------------------
    @staticmethod
    def _set_seeds(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _block_slice(self, block_id: int) -> slice:
        start = block_id * self.num_envs_per_block
        end = start + self.num_envs_per_block
        return slice(start, end)

    def _reset_lstm_states(self) -> None:
        self._lstm_states = [None for _ in range(self.num_blocks)]

    # -- rollout collection ----------------------------------------------
    def collect_rollouts(self) -> Tuple[List[RolloutBuffer], np.ndarray, np.ndarray]:
        """Collect ``horizon`` steps from every block using follower policies.

        Returns the per-block buffers plus the final observations and dones
        (needed for bootstrapping the value function).
        """
        for buf in self.buffers:
            buf.reset()

        obs = self._obs
        if obs is None:
            obs = self.env.reset()
            self._obs = obs
            self._reset_lstm_states()

        obs = np.asarray(obs, dtype=np.float32).reshape(self.num_envs, -1)
        last_dones = np.zeros(self.num_envs, dtype=np.float32)

        for _ in range(self.horizon):
            actions = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
            log_probs = np.zeros(self.num_envs, dtype=np.float32)
            values = np.zeros(self.num_envs, dtype=np.float32)

            for block_id, follower in enumerate(self.followers):
                sl = self._block_slice(block_id)
                block_obs = obs[sl]
                lstm_state = None if self._lstm_states is None else self._lstm_states[block_id]
                with torch.no_grad():
                    act, logp, val, new_state = follower.act(
                        block_obs, lstm_state=lstm_state, deterministic=False
                    )
                actions[sl] = act
                log_probs[sl] = logp
                values[sl] = val
                if self._lstm_states is not None:
                    self._lstm_states[block_id] = new_state

            next_obs, rewards, dones, infos = self.env.step(actions)
            next_obs = np.asarray(next_obs, dtype=np.float32).reshape(self.num_envs, -1)
            rewards = np.asarray(rewards, dtype=np.float32).reshape(self.num_envs)
            dones = np.asarray(dones, dtype=np.float32).reshape(self.num_envs)

            # Per-block insertion into the corresponding rollout buffer.
            for block_id, buf in enumerate(self.buffers):
                sl = self._block_slice(block_id)
                phi = self.followers[block_id].phi
                buf.insert(
                    obs=obs[sl],
                    actions=actions[sl],
                    log_probs=log_probs[sl],
                    rewards=rewards[sl],
                    values=values[sl],
                    dones=dones[sl],
                    block_id=block_id,
                    phi=phi,
                    lstm_state=None,
                )

            # Episode bookkeeping for logging.
            self._episode_rewards += rewards
            self._episode_lengths += 1
            if "successes" in infos:
                self._episode_successes += np.asarray(infos["successes"], dtype=np.float64).reshape(-1)
            elif "episode_successes" in infos:
                self._episode_successes += np.asarray(infos["episode_successes"], dtype=np.float64).reshape(-1)

            done_mask = dones > 0.5
            if np.any(done_mask):
                self.logger.log("rollout/episode_reward", float(self._episode_rewards[done_mask].mean()))
                self.logger.log("rollout/episode_length", float(self._episode_lengths[done_mask].mean()))
                self.logger.log("rollout/episode_successes", float(self._episode_successes[done_mask].mean()))
                self._episode_rewards[done_mask] = 0.0
                self._episode_successes[done_mask] = 0.0
                self._episode_lengths[done_mask] = 0

            obs = next_obs
            last_dones = dones
            self.env_steps += self.num_envs

        self._obs = obs
        return self.buffers, obs, last_dones

    # -- value bootstrapping ---------------------------------------------
    def _bootstrap_values(self, obs: np.ndarray) -> np.ndarray:
        """Compute V(s_T) for the final observation of each block."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        values = np.zeros(self.num_envs, dtype=np.float32)
        for block_id, follower in enumerate(self.followers):
            sl = self._block_slice(block_id)
            phi = follower.phi
            with torch.no_grad():
                val, _ = self.value(obs_t[sl], phi)
            values[sl] = val.detach().cpu().numpy().reshape(-1)
        return values

    # -- one training iteration ------------------------------------------
    def train_iteration(self) -> Dict[str, float]:
        """Run one full SAPG iteration: rollouts -> followers -> leader."""
        buffers, last_obs, last_dones = self.collect_rollouts()
        last_values = self._bootstrap_values(last_obs)

        # --- follower updates (on-policy, per block) ---------------------
        follower_metrics: Dict[str, float] = {}
        for block_id, follower in enumerate(self.followers):
            sl = self._block_slice(block_id)
            stats = follower.update(
                buffers[block_id],
                last_values=last_values[sl],
                last_dones=last_dones[sl],
            )
            follower_metrics.update(stats.to_dict(prefix=f"follower/{block_id}/"))

        # --- leader / aggregation update (off-policy, all blocks) --------
        agg_stats = self.aggregator.aggregate(
            buffers,
            last_values=last_values,
            last_dones=last_dones,
        )
        if hasattr(agg_stats, "to_dict"):
            agg_metrics = agg_stats.to_dict(prefix="leader/")
        elif isinstance(agg_stats, dict):
            agg_metrics = {f"leader/{k}": float(v) for k, v in agg_stats.items()
                           if isinstance(v, (int, float))}
        else:  # pragma: no cover - defensive
            agg_metrics = {}

        metrics = {}
        metrics.update(follower_metrics)
        metrics.update(agg_metrics)
        metrics["train/learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        metrics["train/env_steps"] = float(self.env_steps)
        metrics["train/iteration"] = float(self.iteration)

        # Curriculum diagnostics.
        delta = getattr(self.env, "delta", None)
        if delta is not None:
            metrics["curriculum/delta"] = float(delta)

        self.logger.log_dict(metrics)
        self.iteration += 1
        return metrics

    # -- main loop --------------------------------------------------------
    def train(self, total_iterations: Optional[int] = None,
              checkpoint_every: int = 0, checkpoint_dir: Optional[str] = None) -> None:
        total_iterations = int(total_iterations or _cfg_get(self.cfg, "total_iterations", default=1000))
        checkpoint_every = int(checkpoint_every or _cfg_get(self.cfg, "checkpoint_every", default=0))
        checkpoint_dir = checkpoint_dir or _cfg_get(self.cfg, "log_dir", default=None)

        start = time.time()
        for _ in range(total_iterations):
            metrics = self.train_iteration()
            if self.logger is not None:
                self.logger.flush(step=self.iteration)
            if checkpoint_every > 0 and checkpoint_dir and (self.iteration % checkpoint_every == 0):
                self.save(os.path.join(checkpoint_dir, f"checkpoint_{self.iteration}.pt"))
        elapsed = time.time() - start
        if self.logger is not None:
            self.logger.log("time/total_seconds", elapsed)
            self.logger.flush(step=self.iteration)

    # -- evaluation -------------------------------------------------------
    def evaluate(self, num_episodes: int = 32, deterministic: bool = True) -> Dict[str, float]:
        """Evaluate the deployed (leader) policy."""
        obs = self.env.reset()
        self._reset_lstm_states()
        obs = np.asarray(obs, dtype=np.float32).reshape(self.num_envs, -1)

        ep_rewards = np.zeros(self.num_envs, dtype=np.float64)
        ep_successes = np.zeros(self.num_envs, dtype=np.float64)
        finished_rewards: List[float] = []
        finished_successes: List[float] = []

        max_steps = int(_cfg_get(self.cfg, "env", "episode_length", default=100)) * max(1, num_episodes)
        for _ in range(max_steps):
            actions = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
            for block_id, follower in enumerate(self.followers):
                sl = self._block_slice(block_id)
                lstm_state = None if self._lstm_states is None else self._lstm_states[block_id]
                with torch.no_grad():
                    act, _, _, new_state = self.aggregator.act(
                        obs[sl], lstm_state=lstm_state, deterministic=deterministic
                    )
                actions[sl] = act
                if self._lstm_states is not None:
                    self._lstm_states[block_id] = new_state

            obs, rewards, dones, infos = self.env.step(actions)
            obs = np.asarray(obs, dtype=np.float32).reshape(self.num_envs, -1)
            rewards = np.asarray(rewards, dtype=np.float32).reshape(self.num_envs)
            dones = np.asarray(dones, dtype=np.float32).reshape(self.num_envs)
            ep_rewards += rewards
            if "successes" in infos:
                ep_successes += np.asarray(infos["successes"], dtype=np.float64).reshape(-1)

            done_mask = dones > 0.5
            if np.any(done_mask):
                finished_rewards.extend(ep_rewards[done_mask].tolist())
                finished_successes.extend(ep_successes[done_mask].tolist())
                ep_rewards[done_mask] = 0.0
                ep_successes[done_mask] = 0.0
            if len(finished_rewards) >= num_episodes * self.num_envs:
                break

        return {
            "eval/episode_reward": float(np.mean(finished_rewards)) if finished_rewards else 0.0,
            "eval/episode_successes": float(np.mean(finished_successes)) if finished_successes else 0.0,
            "eval/num_episodes": float(len(finished_rewards)),
        }

    # -- checkpointing ----------------------------------------------------
    def save(self, path: str) -> str:
        return save_checkpoint(
            path,
            policy=self.policy,
            value=self.value,
            optimizer=self.optimizer,
            followers=self.followers,
            leader=self.leader,
            curriculum=getattr(self.env, "curriculum", None),
            obs_normalizer=getattr(self.env, "obs_normalizer", None),
            logger=self.logger,
            iteration=self.iteration,
            env_steps=self.env_steps,
        )

    def load(self, path: str) -> Dict[str, Any]:
        return load_checkpoint(
            path,
            policy=self.policy,
            value=self.value,
            optimizer=self.optimizer,
            followers=self.followers,
            leader=self.leader,
            curriculum=getattr(self.env, "curriculum", None),
            obs_normalizer=getattr(self.env, "obs_normalizer", None),
            logger=self.logger,
            map_location=self.device,
        )

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:  # pragma: no cover - defensive
            pass
        if self.logger is not None:
            self.logger.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAPG: Split and Aggregate Policy Gradients")
    parser.add_argument("--config", type=str, default=None,
                        help="Path (or name) of a YAML config file.")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name (e.g. regrasping, throw, reorientation).")
    parser.add_argument("--mode", type=str, default=None,
                        help="Aggregation mode: leader (default) or symmetric.")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--total-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to load before training/eval.")
    parser.add_argument("--eval", action="store_true", help="Run evaluation only.")
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.mode is not None:
        overrides.setdefault("sapg", {})["mode"] = args.mode
    if args.num_envs is not None:
        overrides.setdefault("env", {})["num_envs"] = args.num_envs
    if args.num_blocks is not None:
        overrides.setdefault("env", {})["num_blocks"] = args.num_blocks
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device is not None:
        overrides["device"] = args.device
    if args.log_dir is not None:
        overrides["log_dir"] = args.log_dir
    if args.total_iterations is not None:
        overrides["total_iterations"] = args.total_iterations
    if args.checkpoint_every is not None:
        overrides["checkpoint_every"] = args.checkpoint_every
    if args.tensorboard:
        overrides["use_tensorboard"] = True
    if args.quiet:
        overrides["verbose"] = False
    return overrides


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, task=args.task, overrides=_build_overrides(args))

    logger = MetricLogger(
        log_dir=cfg.get("log_dir"),
        use_tensorboard=bool(cfg.get("use_tensorboard", False)),
        verbose=bool(cfg.get("verbose", True)),
    )
    trainer = SAPGTrainer(cfg, logger=logger)
    try:
        if args.checkpoint:
            trainer.load(args.checkpoint)
        if args.eval:
            metrics = trainer.evaluate(num_episodes=args.eval_episodes)
            logger.log_dict(metrics)
            logger.flush(step=trainer.iteration)
            if logger.verbose:
                print("Evaluation:", {k: round(v, 4) for k, v in metrics.items()})
        else:
            trainer.train(
                total_iterations=cfg.get("total_iterations"),
                checkpoint_every=cfg.get("checkpoint_every", 0),
                checkpoint_dir=cfg.get("log_dir"),
            )
    finally:
        trainer.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

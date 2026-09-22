"""Shared training machinery (collection, evaluation, logging, checkpoints)."""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..envs.base import EpisodeTracker
from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.running_stat import Normalizer
from ..utils.seeding import set_seed
from ..utils.state_dataset import record_states
from .actor_critic import ActorCritic, ActorCriticConfig
from .runner import MPolicyRunner
from .storage import RolloutStorage


class AdaptiveLR:
    """KL-based adaptive learning rate (the 'KL threshold for LR update' of
    Tables 2-4, as used by the reference large-scale PPO implementations)."""

    def __init__(self, optimizer: torch.optim.Optimizer, kl_threshold: float, min_lr: float) -> None:
        self.optimizer = optimizer
        self.kl_threshold = kl_threshold
        self.min_lr = min_lr
        self.initial_lr = optimizer.param_groups[0]["lr"]

    def update(self, approx_kl: float) -> float:
        if approx_kl > 2.0 * self.kl_threshold:
            scale = 1.0 / 1.5
        elif approx_kl < 0.5 * self.kl_threshold:
            scale = 1.5
        else:
            return self.optimizer.param_groups[0]["lr"]
        for group in self.optimizer.param_groups:
            group["lr"] = max(self.min_lr, group["lr"] * scale)
        return self.optimizer.param_groups[0]["lr"]


class TrainerBase:
    """Common behaviour of SAPG / PPO / DexPBT trainers."""

    name = "base"
    USES_POLICY_GRADIENT = True

    def __init__(self, cfg: Config, env, device: str = "cpu", logdir: Optional[str] = None) -> None:
        self.cfg = cfg
        self.env = env
        self.device = torch.device(device)
        self.seed = int(cfg.get_path("seed", 0))
        set_seed(self.seed)

        self.num_policies = int(self.a_get("num_policies", 1))
        self.horizon = int(self.a_get("horizon", 16))
        if env.num_envs % self.num_policies != 0:
            raise ValueError(
                f"num_envs ({env.num_envs}) must be divisible by num_policies ({self.num_policies})"
            )
        self.num_envs_per_policy = env.num_envs // self.num_policies

        self.gamma = float(self.a_get("gamma", 0.99))
        self.tau = float(self.a_get("tau", 0.95))
        self.nstep = int(self.a_get("nstep", 3))
        self.critic_target = str(self.a_get("critic_target", "nstep"))
        self.mini_epochs = int(self.a_get("mini_epochs", 2))
        self.learning_rate = float(self.a_get("learning_rate", 1e-4))
        self.min_learning_rate = float(self.a_get("min_learning_rate", 1e-5))
        self.kl_threshold = float(self.a_get("kl_threshold", 0.016))
        self.max_grad_norm = float(self.a_get("max_grad_norm", 1.0))
        self.minibatch_multiplier = int(self.a_get("minibatch_size_multiplier", 4))
        self.minibatch_size = max(
            1, self.num_envs_per_policy * self.minibatch_multiplier
        )
        self.num_minibatches = max(
            1, (self.num_envs_per_policy * self.horizon) // self.minibatch_size
        )

        self.model_cfg = ActorCriticConfig.from_config(cfg)
        self.recurrent = self.model_cfg.actor_type == "lstm"
        self.normalize_obs = bool(cfg.get_path("model.normalize_obs", False))
        self.normalizer = Normalizer(env.obs_dim) if self.normalize_obs else None

        if self.USES_POLICY_GRADIENT:
            self.model = ActorCritic(
                self.model_cfg,
                obs_dim=env.obs_dim,
                action_dim=env.action_dim,
                num_policies=self.num_policies,
                normalizer=self.normalizer,
            ).to(self.device)
            self.optimizer = torch.optim.Adam(
                self.model.trainable_parameter_groups(self.learning_rate)
            )
            self.lr_scheduler = AdaptiveLR(self.optimizer, self.kl_threshold, self.min_learning_rate)

            self.storage = RolloutStorage(
                num_policies=self.num_policies,
                num_envs_per_policy=self.num_envs_per_policy,
                horizon=self.horizon,
                obs_dim=env.obs_dim,
                action_dim=env.action_dim,
                device=self.device,
                recurrent=self.recurrent,
            )
            self.runner = MPolicyRunner(
                env, self.model, self.num_policies, self.device, recurrent=self.recurrent
            )
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed)

        self.env_steps = 0
        self.iteration = 0
        self.history: List[Dict[str, float]] = []
        self.logger = Logger(
            logdir or os.path.join("runs", self.name),
            use_tensorboard=bool(cfg.get_path("logging.use_tensorboard", True)),
            verbose=bool(cfg.get_path("verbose", True)),
        )
        self.eval_interval = int(self.a_get("eval_interval", 0))
        self.checkpoint_interval = int(self.a_get("checkpoint_interval", 0))
        self.record_states_interval = int(self.a_get("record_states_interval", 0))

    # ------------------------------------------------------------------ #
    def a_get(self, key: str, default=None):
        return self.cfg.get_path(f"algo.{key}", default)

    # ------------------------------------------------------------------ #
    # collection / updates
    # ------------------------------------------------------------------ #
    def collect(self) -> None:
        if self.normalizer is not None and self.runner.obs is not None:
            self.normalizer.update(self.runner.obs)
        t0 = time.time()
        last_values = self.runner.collect(self.storage, self.horizon)
        self.storage.compute_returns(
            last_values,
            gamma=self.gamma,
            gae_lambda=self.tau,
            nstep=self.nstep,
            critic_target=self.critic_target,
        )
        self.collect_time = time.time() - t0

    def update(self) -> Dict[str, float]:  # pragma: no cover - abstract
        raise NotImplementedError

    def train_iteration(self) -> Dict[str, float]:
        self.iteration += 1
        self.collect()
        metrics = self.update()
        self.env_steps += self.env.num_envs * self.horizon

        train_stats = self.env.episode_stats()
        metrics.update({f"train/{k}": v for k, v in train_stats.items()})
        metrics["env_steps"] = self.env_steps
        metrics["iteration"] = self.iteration

        if self.record_states_interval and self.iteration % self.record_states_interval == 0:
            self.record_states()

        self.logger.log(self.env_steps, metrics)
        self.history.append(dict(metrics))
        return metrics

    # ------------------------------------------------------------------ #
    def train(
        self,
        num_iterations: Optional[int] = None,
        max_env_steps: Optional[float] = None,
    ) -> List[Dict[str, float]]:
        num_iterations = int(num_iterations if num_iterations is not None else self.a_get("num_iterations", 100))
        if max_env_steps is None:
            max_env_steps = self.a_get("max_env_steps", None)

        if self.USES_POLICY_GRADIENT:
            self.runner.reset()
        if self.normalizer is not None and self.USES_POLICY_GRADIENT:
            for _ in range(int(self.cfg.get_path("model.normalizer_warmup_steps", 0)) or 0):
                actions = torch.rand((self.env.num_envs, self.env.action_dim), device=self.device) * 2 - 1
                self.normalizer.update(self.runner.obs)
                self.runner.obs = self.env.step(actions).obs
            self.runner.reset()

        while self.iteration < num_iterations:
            if max_env_steps is not None and self.env_steps >= float(max_env_steps):
                break
            metrics = self.train_iteration()
            if self.eval_interval and self.iteration % self.eval_interval == 0:
                self.evaluate_and_log()
            if self.checkpoint_interval and self.iteration % self.checkpoint_interval == 0:
                self.save(tag=f"iter{self.iteration:07d}")

        self.save(tag="final")
        self.logger.close()
        return self.history

    # ------------------------------------------------------------------ #
    # evaluation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate_policy(
        self,
        policy_id: int = 0,
        num_episodes: int = 32,
        max_steps: int = 1024,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """Roll out one policy (in all environments) and report episode metrics."""
        obs = self.env.reset()
        tracker = EpisodeTracker(self.env.num_envs)
        states = None
        prev_dones = None
        for _ in range(max_steps):
            policy_ids = torch.full((obs.shape[0],), policy_id, dtype=torch.long, device=self.device)
            actions, _, _, states = self.model.act(
                obs,
                policy_ids=policy_ids,
                recurrent_states=states,
                dones=prev_dones,
                deterministic=deterministic,
            )
            step = self.env.step(actions)
            obs = step.obs
            prev_dones = step.dones if self.recurrent else None
            tracker.step(
                step.rewards.detach().cpu().numpy(),
                step.infos.get("successes", step.rewards * 0).detach().cpu().numpy(),
                (step.dones + step.timeouts).clamp(max=1.0).detach().cpu().numpy().astype(bool),
            )
            if len(tracker.finished_returns) >= num_episodes:
                break
        stats = tracker.summary()
        # restart the training rollout from a clean state
        self.runner.reset()
        return stats

    def evaluate_and_log(self) -> Dict[str, float]:
        eval_cfg = self.cfg.get("eval", {}) or {}
        num_episodes = int(eval_cfg.get("episodes", 32))
        max_steps = int(eval_cfg.get("max_steps", 1024))
        policy_ids: Sequence[int] = eval_cfg.get("policy_ids", [0])
        metrics: Dict[str, float] = {}
        for policy_id in policy_ids:
            stats = self.evaluate_policy(int(policy_id), num_episodes=num_episodes, max_steps=max_steps)
            for key, value in stats.items():
                metrics[f"eval{policy_id}/{key}"] = value
        metrics["env_steps"] = self.env_steps
        metrics["iteration"] = self.iteration
        self.logger.log(self.env_steps, metrics, prefix="")
        return metrics

    # ------------------------------------------------------------------ #
    def record_states(self) -> None:
        states = torch.cat([b.obs.reshape(-1, self.env.obs_dim) for b in self.storage.buffers], dim=0)
        record_states(self.logger.logdir, self.iteration, states.detach().cpu().numpy())

    # ------------------------------------------------------------------ #
    def save(self, tag: str = "final") -> str:
        ckpt_dir = os.path.join(self.logger.logdir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"{tag}.pt")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "iteration": self.iteration,
                "env_steps": self.env_steps,
                "config": self.cfg.to_dict(),
            },
            path,
        )
        with open(os.path.join(ckpt_dir, f"{tag}.meta.json"), "w") as handle:
            json.dump({"iteration": self.iteration, "env_steps": self.env_steps}, handle)
        return path

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.iteration = int(payload.get("iteration", 0))
        self.env_steps = int(payload.get("env_steps", 0))

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self.logger.close()
        self.env.close()

    # ------------------------------------------------------------------ #
    @staticmethod
    def huber_mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

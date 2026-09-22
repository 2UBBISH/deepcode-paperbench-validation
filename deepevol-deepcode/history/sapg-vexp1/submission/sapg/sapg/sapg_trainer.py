"""SAPG trainer: orchestration of Algorithm 1 (Split and Aggregate Policy Gradients).

This module wires together the shared actor-critic (``actor_critic.py``), the
per-policy rollout buffers (``rollout.py``), the aggregation schemes
(``aggregation.py``) and the loss functions (``losses.py``) into the SAPG
training loop described in the paper.

Algorithm 1 (paper Sec. 4.3 / 4.4)
----------------------------------
for iteration = 1, 2, ... do
    # 1. Collect data: each policy j rolls out its own env block with horizon H
    for j in 1..M:
        D_j <- CollectData(env_block_j, theta, phi_j)

    # 2. Build the off-policy batch for the leader (subsampled to |D_1|)
    D_1' <- Subsample(union_{j=2..M} D_j, |D_1|)

    # 3. Update shared parameters theta, psi and per-policy phi_j
    for epoch in 1..K:
        for minibatch in Shuffle(D_1, D_1', D_2, ..., D_M):
            L = L_off(pi_1; D_1') + L_on(pi_1; D_1)
                + sum_{j>=2} [ L_on(pi_j; D_j) + sigma*(j-1)*H(pi_j) ]
            theta, psi <- Adam(L)
            phi_j <- Adam(L_j)   # only from its own objective
end for

Notes
-----
* The leader is index 0 in code (paper's i=1).
* ``lambda`` (Eq. 4) weights the off-policy term; default 1.0.
* Entropy regularization (Eq. 10) is applied only to followers.
* KL-adaptive learning rate with threshold 0.016 (standard PPO).
* Gradient-norm clipping at 1.0, Adam with PyTorch defaults.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .actor_critic import MultiPolicyActorCritic, SharedActorCritic
from .aggregation import (
    AggregationSpec,
    build_aggregation_batches,
    make_spec,
    split_off_policy_by_source,
)
from .losses import (
    combined_critic_loss,
    compute_gae,
    compute_policy_loss,
    entropy_bonus,
    n_step_value_target,
    one_step_value_target,
    ppo_surrogate_loss,
    value_loss,
)
from .rollout import MultiPolicyRollout, RolloutBuffer, subsample_off_policy


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SAPGConfig:
    """Hyper-parameters for the SAPG trainer.

    Defaults follow the paper's Tables 2-4 and the reproduction plan.
    """

    task: str = "allegrokuka"
    num_policies: int = 6                 # M
    num_envs: int = 24576                 # N (total parallel envs)
    horizon: int = 16                     # rollout horizon per policy
    gamma: float = 0.99
    tau: float = 0.95                     # GAE lambda
    n_step: int = 3                       # on-policy n-step return
    clip_eps: float = 0.1                 # PPO clip (0.2 for allegrohand)
    lam: float = 1.0                      # off-policy weight (Eq. 4)
    entropy_coef: float = 0.0             # sigma in Eq. 10
    critic_coef: float = 4.0
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    max_learning_rate: float = 1e-2
    kl_threshold: float = 0.016
    kl_adaptive_lr: bool = True
    grad_norm_clip: float = 1.0
    mini_epochs: int = 2                  # 2 (allegrokuka) / 5 (easy tasks)
    num_mini_batches: int = 4             # mini-batch = num_envs * 4
    aggregation: str = "leader_follower"
    subsample: bool = True
    phi_dim: int = 32
    obs_dim: int = 0
    action_dim: int = 0
    recurrent: bool = False
    lstm_hidden: int = 768
    seed: int = 0
    device: str = "cuda"
    log_interval: int = 1
    save_interval: int = 100
    max_iterations: int = 100000
    target_transitions: float = 2e10
    output_dir: str = "runs/sapg"


# ---------------------------------------------------------------------------
# KL-adaptive learning rate
# ---------------------------------------------------------------------------
class KLAdaptiveLR:
    """Adjusts the learning rate based on the measured KL divergence.

    Standard PPO heuristic: if KL > 2 * threshold, halve the LR; if
    KL < threshold / 2, increase the LR by 1.5x. Bounded by
    ``[min_lr, max_lr]``.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        threshold: float = 0.016,
        min_lr: float = 1e-5,
        max_lr: float = 1e-2,
        factor_up: float = 1.5,
        factor_down: float = 0.5,
    ) -> None:
        self.optimizer = optimizer
        self.threshold = threshold
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.factor_up = factor_up
        self.factor_down = factor_down

    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def update(self, kl: float) -> float:
        lr = self.current_lr()
        if kl > 2.0 * self.threshold:
            lr = max(self.min_lr, lr * self.factor_down)
        elif kl < 0.5 * self.threshold:
            lr = min(self.max_lr, lr * self.factor_up)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class SAPGTrainer:
    """Orchestrates the SAPG training loop (Algorithm 1)."""

    def __init__(
        self,
        config: SAPGConfig,
        env_factory: Callable[[int, int], object],
        logger=None,
    ) -> None:
        """
        Parameters
        ----------
        config:
            :class:`SAPGConfig` with all hyper-parameters.
        env_factory:
            Callable ``(num_envs, seed) -> env`` returning a vectorized
            environment exposing ``reset()``, ``step(actions)`` and the
            attributes ``num_envs``, ``obs_dim``, ``action_dim``.
        logger:
            Optional logger object with ``log_scalar``/``log_dict`` methods.
        """
        self.config = config
        self.env_factory = env_factory
        self.logger = logger

        self.device = torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
        )

        # --- environment ---------------------------------------------------
        self.env = env_factory(config.num_envs, config.seed)
        self.obs_dim = getattr(self.env, "obs_dim", config.obs_dim)
        self.action_dim = getattr(self.env, "action_dim", config.action_dim)
        self.num_envs = getattr(self.env, "num_envs", config.num_envs)

        # envs are split evenly across the M policies
        self.envs_per_policy = self.num_envs // config.num_policies
        assert (
            self.envs_per_policy * config.num_policies == self.num_envs
        ), "num_envs must be divisible by num_policies"

        # --- model ---------------------------------------------------------
        self.model = MultiPolicyActorCritic(
            task=config.task,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            phi_dim=config.phi_dim,
            num_policies=config.num_policies,
            device=self.device,
        ).to(self.device)

        self.recurrent = self.model.policy(0).is_recurrent
        self.lstm_hidden = config.lstm_hidden if self.recurrent else 0

        # --- optimizer -----------------------------------------------------
        self.optimizer = torch.optim.Adam(
            self.model.parameter_groups(lr=config.learning_rate)
        )
        self.kl_lr = KLAdaptiveLR(
            self.optimizer,
            threshold=config.kl_threshold,
            min_lr=config.min_learning_rate,
            max_lr=config.max_learning_rate,
        )

        # --- aggregation spec ---------------------------------------------
        self.spec: AggregationSpec = make_spec(
            config.aggregation,
            num_policies=config.num_policies,
            lam=config.lam,
        )
        # honour explicit subsample flag
        self.spec.subsample = config.subsample

        # --- rollout storage ----------------------------------------------
        self.rollout = MultiPolicyRollout(
            num_policies=config.num_policies,
            horizon=config.horizon,
            num_envs_per_policy=self.envs_per_policy,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            recurrent=self.recurrent,
            lstm_hidden=self.lstm_hidden,
        )

        # --- runtime state -------------------------------------------------
        self.obs = None
        self.lstm_states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None
        ] * config.num_policies
        self.iteration = 0
        self.total_transitions = 0
        self.start_time = time.time()

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------
    def _split_obs(self, obs: torch.Tensor) -> List[torch.Tensor]:
        """Split the flat [N, obs_dim] observation into M per-policy blocks."""
        return list(torch.split(obs, self.envs_per_policy, dim=0))

    def collect_rollouts(self) -> None:
        """Collect one horizon of data for every policy (Algorithm 1, step 1)."""
        cfg = self.config
        self.rollout.reset()

        for t in range(cfg.horizon):
            obs_blocks = self._split_obs(self.obs)
            actions_blocks: List[torch.Tensor] = []
            logp_blocks: List[torch.Tensor] = []
            value_blocks: List[torch.Tensor] = []

            for j in range(cfg.num_policies):
                policy = self.model.policy(j)
                with torch.no_grad():
                    dist = policy.distribution(
                        obs_blocks[j], self.lstm_states[j]
                    )
                    if isinstance(dist, tuple):
                        dist, new_state = dist
                        self.lstm_states[j] = new_state
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(-1)
                    value = policy.value(obs_blocks[j], None)
                    if isinstance(value, tuple):
                        value = value[0]
                actions_blocks.append(action)
                logp_blocks.append(log_prob)
                value_blocks.append(value)

            actions = torch.cat(actions_blocks, dim=0)
            next_obs, rewards, dones = self.env.step(actions)

            # store per-policy transitions
            for j in range(cfg.num_policies):
                self.rollout[j].add(
                    obs=obs_blocks[j],
                    actions=actions_blocks[j],
                    log_probs=logp_blocks[j],
                    rewards=rewards[j * self.envs_per_policy : (j + 1) * self.envs_per_policy],
                    dones=dones[j * self.envs_per_policy : (j + 1) * self.envs_per_policy],
                    values=value_blocks[j],
                )

            self.obs = next_obs
            self.total_transitions += self.num_envs

        # bootstrap values for the last observation
        last_values = self._compute_last_values()
        self.rollout.compute_targets(
            last_values,
            gamma=cfg.gamma,
            tau=cfg.tau,
            n_step=cfg.n_step,
        )

    def _compute_last_values(self) -> List[torch.Tensor]:
        obs_blocks = self._split_obs(self.obs)
        last_values: List[torch.Tensor] = []
        for j in range(self.config.num_policies):
            policy = self.model.policy(j)
            with torch.no_grad():
                value = policy.value(obs_blocks[j], None)
                if isinstance(value, tuple):
                    value = value[0]
            last_values.append(value)
        return last_values

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def _flatten_batch(self, buf: RolloutBuffer) -> Dict[str, torch.Tensor]:
        return buf.flat()

    def update(self) -> Dict[str, float]:
        """Run the SAPG update (Algorithm 1, step 3)."""
        cfg = self.config
        model = self.model

        # ---- assemble off-policy batches (leader-follower) ----------------
        off_batches = build_aggregation_batches(
            self.rollout.buffers, self.spec
        )

        # ---- flatten on-policy batches ------------------------------------
        on_batches = {
            j: self._flatten_batch(self.rollout[j])
            for j in range(cfg.num_policies)
        }

        # ---- mini-batch iteration -----------------------------------------
        num_samples = on_batches[0]["obs"].shape[0]
        mini_batch_size = max(1, num_samples // cfg.num_mini_batches)

        stats: Dict[str, float] = {}
        approx_kls: List[float] = []

        for epoch in range(cfg.mini_epochs):
            # shuffle indices for the on-policy data (per policy)
            perms = {
                j: torch.randperm(on_batches[j]["obs"].shape[0], device=self.device)
                for j in range(cfg.num_policies)
            }
            # off-policy batch (leader) shuffled independently
            off_perm = None
            if off_batches.get(0) is not None:
                off_perm = torch.randperm(
                    off_batches[0]["obs"].shape[0], device=self.device
                )

            for mb in range(cfg.num_mini_batches):
                model.zero_grad(set_to_none=True)
                total_loss = torch.zeros((), device=self.device)

                # ---------------- leader (i=0) ----------------------------
                leader_loss, leader_info = self._policy_step(
                    policy_index=0,
                    batch=on_batches[0],
                    perm=perms[0],
                    mb=mb,
                    mini_batch_size=mini_batch_size,
                    off_batch=off_batches.get(0),
                    off_perm=off_perm,
                )
                total_loss = total_loss + leader_loss
                stats.update({f"leader/{k}": v for k, v in leader_info.items()})

                # ---------------- followers (i>=1) ------------------------
                for j in range(1, cfg.num_policies):
                    f_loss, f_info = self._policy_step(
                        policy_index=j,
                        batch=on_batches[j],
                        perm=perms[j],
                        mb=mb,
                        mini_batch_size=mini_batch_size,
                        off_batch=None,
                        off_perm=None,
                    )
                    total_loss = total_loss + f_loss
                    stats.update({f"follower{j}/{k}": v for k, v in f_info.items()})

                # ---------------- backward --------------------------------
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_norm_clip
                )
                self.optimizer.step()

                stats["grad_norm"] = float(grad_norm)
                stats["loss"] = float(total_loss.detach())
                if "leader/approx_kl" in stats:
                    approx_kls.append(stats["leader/approx_kl"])

        # ---- KL-adaptive LR ----------------------------------------------
        if cfg.kl_adaptive_lr and approx_kls:
            mean_kl = sum(approx_kls) / len(approx_kls)
            stats["kl"] = mean_kl
            stats["lr"] = self.kl_lr.update(mean_kl)
        else:
            stats["lr"] = self.kl_lr.current_lr()

        return stats

    def _policy_step(
        self,
        policy_index: int,
        batch: Dict[str, torch.Tensor],
        perm: torch.Tensor,
        mb: int,
        mini_batch_size: int,
        off_batch: Optional[Dict[str, torch.Tensor]] = None,
        off_perm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the loss contribution for a single policy on one mini-batch."""
        cfg = self.config
        policy = self.model.policy(policy_index)

        start = mb * mini_batch_size
        end = start + mini_batch_size
        idx = perm[start:end]

        obs = batch["obs"][idx]
        actions = batch["actions"][idx]
        old_log_probs = batch["log_probs"][idx]
        advantages = batch["advantages"][idx]
        value_targets = batch["value_targets"][idx]
        old_values = batch["values"][idx]

        # normalise advantages (standard PPO)
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- on-policy loss ----------------------------------------------
        log_probs, entropy, values = self._evaluate(policy, obs, actions)
        on_loss = ppo_surrogate_loss(
            log_probs, old_log_probs, adv, clip_eps=cfg.clip_eps
        )
        critic_on = value_loss(values, value_targets, clip=cfg.clip_eps, old_values=old_values)

        info: Dict[str, float] = {
            "policy_loss": float(on_loss.detach()),
            "critic_loss": float(critic_on.detach()),
        }

        # approx KL for adaptive LR
        with torch.no_grad():
            approx_kl = (old_log_probs - log_probs).mean()
            info["approx_kl"] = float(approx_kl)

        loss = on_loss + cfg.critic_coef * critic_on

        # ---- entropy regularization (followers only, Eq. 10) -------------
        if policy_index >= 1 and cfg.entropy_coef > 0.0:
            ent_term = entropy_bonus(entropy, cfg.entropy_coef, policy_index + 1)
            loss = loss + ent_term
            info["entropy"] = float(entropy.mean().detach())

        # ---- off-policy loss (leader only, Eq. 3) ------------------------
        if off_batch is not None and off_perm is not None:
            off_start = mb * mini_batch_size
            off_end = off_start + mini_batch_size
            off_idx = off_perm[off_start:off_end]

            off_obs = off_batch["obs"][off_idx]
            off_actions = off_batch["actions"][off_idx]
            off_old_log_probs = off_batch["log_probs"][off_idx]
            off_advantages = off_batch["advantages"][off_idx]
            off_value_targets = off_batch["value_targets"][off_idx]
            off_old_values = off_batch["values"][off_idx]
            behavior_log_probs = off_batch["behavior_log_probs"][off_idx]

            off_adv = (off_advantages - off_advantages.mean()) / (
                off_advantages.std() + 1e-8
            )

            off_log_probs, _, off_values = self._evaluate(
                policy, off_obs, off_actions
            )
            off_loss = off_policy_surrogate_loss(
                off_log_probs,
                behavior_log_probs,
                off_old_log_probs,
                off_adv,
                clip_eps=cfg.clip_eps,
            )
            critic_off = value_loss(
                off_values, off_value_targets, clip=cfg.clip_eps, old_values=off_old_values
            )

            loss = loss + cfg.lam * off_loss + cfg.critic_coef * cfg.lam * critic_off
            info["off_policy_loss"] = float(off_loss.detach())
            info["critic_off_loss"] = float(critic_off.detach())

        return loss, info

    def _evaluate(
        self,
        policy: SharedActorCritic,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-prob, entropy and value for a (possibly recurrent) policy."""
        if self.recurrent:
            # For recurrent policies we evaluate with a zero initial state over
            # the flattened mini-batch (sequence length 1 per sample).
            log_probs, entropy, values = policy.evaluate(obs, actions)
        else:
            log_probs, entropy, values = policy.evaluate(obs, actions)
        return log_probs, entropy, values

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def train(self) -> None:
        cfg = self.config
        self.obs = self.env.reset()
        if isinstance(self.obs, tuple):
            self.obs = self.obs[0]
        self.obs = self.obs.to(self.device)

        while self.iteration < cfg.max_iterations:
            self.collect_rollouts()
            stats = self.update()
            self.iteration += 1

            if self.logger is not None and self.iteration % cfg.log_interval == 0:
                stats["iteration"] = self.iteration
                stats["total_transitions"] = self.total_transitions
                stats["wall_time"] = time.time() - self.start_time
                self.logger.log_dict(stats, step=self.iteration)

            if self.iteration % cfg.save_interval == 0:
                self.save(os.path.join(cfg.output_dir, f"ckpt_{self.iteration}.pt"))

            if self.total_transitions >= cfg.target_transitions:
                break

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "iteration": self.iteration,
                "total_transitions": self.total_transitions,
                "config": self.config.__dict__,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.iteration = ckpt.get("iteration", 0)
        self.total_transitions = ckpt.get("total_transitions", 0)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------
def train_sapg(config: SAPGConfig, env_factory, logger=None) -> SAPGTrainer:
    """Instantiate and run a :class:`SAPGTrainer`."""
    trainer = SAPGTrainer(config, env_factory, logger=logger)
    trainer.train()
    return trainer

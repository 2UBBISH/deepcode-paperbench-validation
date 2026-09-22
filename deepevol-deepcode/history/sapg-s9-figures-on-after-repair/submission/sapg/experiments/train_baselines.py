"""Baseline training scripts for SAPG comparison (Section 6, Table 1 / Figure 5).

Implements the three baselines compared against SAPG in the paper:

  * PPO   -- single-policy proximal policy optimization (Schulman et al., 2017).
             Uses the shared actor-critic backbone with ``num_policies=1``.
  * PBT   -- Population Based Training (Jaderberg et al., 2017) as adapted to
             large-scale RL by DexPBT (Petrenko et al., 2023).  A population of
             independent PPO agents is trained in parallel; periodically the
             worst-performing members copy the weights of the best members and
             perturb their hyper-parameters.
  * PQL   -- Prior-data Q-Learning (Li et al., 2023), an off-policy method that
             augments the replay buffer with prior demonstrations.  We provide a
             lightweight SAC-style off-policy learner with a prior-data buffer
             that mirrors the essential algorithmic ingredients used in the
             paper's comparison.

All baselines share the same environment factory (``sapg.envs.make_task_env``)
and the same evaluation protocol so that results are directly comparable to
SAPG.  Each script supports:

  * ``--task``        one of {allegrohand, shadowhand, regrasping, throw,
                      reorientation}
  * ``--num-envs``    number of parallel environments (default 24576)
  * ``--seed``        random seed
  * ``--max-iterations`` / ``--max-samples``  training budget
  * ``--output-dir``  directory for checkpoints and logs
  * ``--device``      torch device string

The module can be used either as a library (``train_ppo``, ``train_pbt``,
``train_pql``) or from the command line::

    python -m sapg.experiments.train_baselines --algo ppo --task throw
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sapg.envs import TASK_NAMES, make_task_env
from sapg.sapg.models import MultiPolicyActorCritic, build_actor_critic
from sapg.sapg.ppo import PPO, PPOConfig
from sapg.sapg.rollout import RolloutBuffer, collect_rollout
from sapg.sapg.utils import (
    AverageMeter,
    clip_grad_norm_,
    get_device,
    get_logger,
    set_seed,
)

logger = get_logger("sapg.baselines")

__all__ = [
    "BaselineConfig",
    "train_ppo",
    "train_pbt",
    "train_pql",
    "main",
]


# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------
@dataclass
class BaselineConfig:
    """Configuration shared by all baselines.

    Defaults follow the paper's Appendix B hyper-parameters (Tables 2-4).
    """

    algo: str = "ppo"
    task: str = "allegrokuka"
    num_envs: int = 24576
    horizon: int = 16
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.95
    n_step: int = 3
    clip_eps: float = 0.1
    critic_coef: float = 4.0
    entropy_coef: float = 0.0
    bounds_coef: float = 1e-4
    max_grad_norm: float = 1.0
    mini_epochs: int = 2
    num_mini_batches: int = 4
    use_kl_adaptive_lr: bool = True
    kl_threshold: float = 0.016
    kl_adaptive_factor: float = 1.5
    normalize_advantage: bool = True
    seed: int = 0
    device: str = "cuda"
    max_iterations: int = 100000
    max_samples: Optional[float] = None
    log_interval: int = 1
    eval_interval: int = 50
    eval_episodes: int = 32
    output_dir: str = "runs/baselines"
    force_mock: bool = False

    # --- PBT-specific -----------------------------------------------------
    population_size: int = 8
    pbt_interval: int = 50
    pbt_exploit_frac: float = 0.25
    pbt_perturb: bool = True

    # --- PQL-specific -----------------------------------------------------
    pql_buffer_size: int = 1_000_000
    pql_batch_size: int = 4096
    pql_updates_per_iter: int = 1
    pql_prior_frac: float = 0.25
    pql_alpha: float = 0.2
    pql_tau: float = 0.005

    def to_ppo_config(self) -> PPOConfig:
        """Convert to a :class:`PPOConfig` for the PPO baseline."""
        return PPOConfig(
            num_envs=self.num_envs,
            horizon=self.horizon,
            task=self.task,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            tau=self.tau,
            n_step=self.n_step,
            clip_eps=self.clip_eps,
            critic_coef=self.critic_coef,
            entropy_coef=self.entropy_coef,
            bounds_coef=self.bounds_coef,
            max_grad_norm=self.max_grad_norm,
            mini_epochs=self.mini_epochs,
            num_mini_batches=self.num_mini_batches,
            use_kl_adaptive_lr=self.use_kl_adaptive_lr,
            kl_threshold=self.kl_threshold,
            kl_adaptive_factor=self.kl_adaptive_factor,
            normalize_advantage=self.normalize_advantage,
            seed=self.seed,
            device=self.device,
            max_iterations=self.max_iterations,
            log_interval=self.log_interval,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_env(cfg: BaselineConfig, num_envs: Optional[int] = None):
    """Build the vectorized environment for the configured task."""
    return make_task_env(
        cfg.task,
        num_envs=num_envs if num_envs is not None else cfg.num_envs,
        device=cfg.device,
        headless=True,
        seed=cfg.seed,
        force_mock=cfg.force_mock,
    )


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _evaluate(env, actor_critic: MultiPolicyActorCritic, cfg: BaselineConfig,
              policy_index: int = 0) -> float:
    """Run a short deterministic evaluation and return the mean episode return.

    The evaluation uses the same environment instance but resets it; this keeps
    the implementation simple and dependency-free.  For the mock environment the
    episode return is a proxy for task performance.
    """
    was_training = actor_critic.training
    actor_critic.eval()
    obs = env.reset()
    returns = torch.zeros(env.num_envs, device=obs.device)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=obs.device)
    lstm_state = None
    max_steps = getattr(env, "max_episode_length", 200)
    with torch.no_grad():
        for _ in range(max_steps):
            actions, _, _, lstm_state = actor_critic.act(
                obs, policy_index, deterministic=True, lstm_state=lstm_state
            )
            obs, rewards, dones, _ = env.step(actions)
            returns += rewards * active.float()
            active = active & (dones < 0.5)
            if not active.any():
                break
    if was_training:
        actor_critic.train()
    return float(returns.mean().item())


# ---------------------------------------------------------------------------
# PPO baseline
# ---------------------------------------------------------------------------
def train_ppo(cfg: BaselineConfig) -> Dict[str, Any]:
    """Train a single-policy PPO baseline.

    Returns a dict with the training history and final evaluation.
    """
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    env = _make_env(cfg)
    ppo_cfg = cfg.to_ppo_config()
    ppo_cfg.device = str(device)

    actor_critic = build_actor_critic(
        task=cfg.task,
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        num_policies=1,
    ).to(device)

    trainer = PPO(env=env, actor_critic=actor_critic, config=ppo_cfg)
    history = trainer.train(max_iterations=cfg.max_iterations)

    final_eval = _evaluate(env, actor_critic, cfg)
    result = {
        "algo": "ppo",
        "task": cfg.task,
        "seed": cfg.seed,
        "history": history,
        "final_eval": final_eval,
        "total_samples": trainer.state.total_samples,
    }
    _save_json(os.path.join(cfg.output_dir, f"ppo_{cfg.task}_seed{cfg.seed}.json"), result)
    env.close()
    return result


# ---------------------------------------------------------------------------
# PBT baseline
# ---------------------------------------------------------------------------
class _PBTMember:
    """A single member of the PBT population."""

    def __init__(self, member_id: int, cfg: BaselineConfig, env, device):
        self.member_id = member_id
        self.cfg = copy.deepcopy(cfg)
        self.env = env
        self.device = device
        self.actor_critic = build_actor_critic(
            task=cfg.task,
            obs_dim=env.obs_dim,
            act_dim=env.act_dim,
            num_policies=1,
        ).to(device)
        ppo_cfg = self.cfg.to_ppo_config()
        ppo_cfg.device = str(device)
        self.trainer = PPO(env=env, actor_critic=self.actor_critic, config=ppo_cfg)
        self.score = -float("inf")

    def train_steps(self, iterations: int) -> None:
        self.trainer.train(max_iterations=iterations)

    def evaluate(self) -> float:
        self.score = _evaluate(self.env, self.actor_critic, self.cfg)
        return self.score

    def copy_from(self, other: "_PBTMember") -> None:
        """Copy weights from another member (exploit step)."""
        self.actor_critic.load_state_dict(other.actor_critic.state_dict())

    def perturb(self, rng: np.random.Generator) -> None:
        """Perturb hyper-parameters (explore step)."""
        if not self.cfg.pbt_perturb:
            return
        # Multiplicative jitter on learning rate and entropy coefficient.
        lr = self.cfg.learning_rate * float(rng.choice([0.8, 1.0, 1.25]))
        self.cfg.learning_rate = float(np.clip(lr, 1e-6, 1e-2))
        self.trainer.config.learning_rate = self.cfg.learning_rate
        self.trainer.state.learning_rate = self.cfg.learning_rate
        ent = self.cfg.entropy_coef + float(rng.choice([-0.001, 0.0, 0.001]))
        self.cfg.entropy_coef = float(np.clip(ent, 0.0, 0.01))
        self.trainer.config.entropy_coef = self.cfg.entropy_coef


def train_pbt(cfg: BaselineConfig) -> Dict[str, Any]:
    """Train a Population Based Training baseline.

    A population of independent PPO agents is trained in parallel.  Every
    ``pbt_interval`` iterations the bottom ``pbt_exploit_frac`` of the
    population copies the weights of the top performers and perturbs their
    hyper-parameters.
    """
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    rng = np.random.default_rng(cfg.seed)

    # Each member gets its own environment instance so that rollouts are
    # independent.  To keep memory tractable we split the env budget across the
    # population.
    per_member_envs = max(1, cfg.num_envs // cfg.population_size)
    members: List[_PBTMember] = []
    for i in range(cfg.population_size):
        member_cfg = copy.deepcopy(cfg)
        member_cfg.num_envs = per_member_envs
        member_cfg.seed = cfg.seed + i
        env = _make_env(member_cfg, num_envs=per_member_envs)
        members.append(_PBTMember(i, member_cfg, env, device))

    history: List[Dict[str, Any]] = []
    total_iters = 0
    while total_iters < cfg.max_iterations:
        step_iters = min(cfg.pbt_interval, cfg.max_iterations - total_iters)
        for m in members:
            m.train_steps(step_iters)
        total_iters += step_iters

        # Evaluate all members.
        scores = np.array([m.evaluate() for m in members])
        order = np.argsort(scores)[::-1]  # best first
        n_exploit = max(1, int(np.ceil(cfg.population_size * cfg.pbt_exploit_frac)))
        top = order[:n_exploit]
        bottom = order[-n_exploit:]

        # Exploit + explore.
        for rank, b_idx in enumerate(bottom):
            src = members[int(top[rank % len(top)])]
            members[int(b_idx)].copy_from(src)
            members[int(b_idx)].perturb(rng)

        best_score = float(scores[order[0]])
        history.append({
            "iteration": total_iters,
            "best_score": best_score,
            "mean_score": float(scores.mean()),
            "scores": scores.tolist(),
        })
        logger.info(
            "[PBT] iter=%d best=%.3f mean=%.3f", total_iters, best_score, float(scores.mean())
        )

    # Final evaluation of the best member.
    scores = np.array([m.evaluate() for m in members])
    best = members[int(np.argmax(scores))]
    final_eval = float(scores.max())

    result = {
        "algo": "pbt",
        "task": cfg.task,
        "seed": cfg.seed,
        "history": history,
        "final_eval": final_eval,
        "population_size": cfg.population_size,
    }
    _save_json(os.path.join(cfg.output_dir, f"pbt_{cfg.task}_seed{cfg.seed}.json"), result)
    for m in members:
        m.env.close()
    return result


# ---------------------------------------------------------------------------
# PQL baseline (off-policy with prior data)
# ---------------------------------------------------------------------------
class _ReplayBuffer:
    """Simple circular replay buffer for the PQL baseline."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.next_obs = torch.zeros(capacity, obs_dim, device=device)
        self.actions = torch.zeros(capacity, act_dim, device=device)
        self.rewards = torch.zeros(capacity, 1, device=device)
        self.dones = torch.zeros(capacity, 1, device=device)
        self.ptr = 0
        self.size = 0

    def add(self, obs, next_obs, actions, rewards, dones) -> None:
        n = obs.shape[0]
        idx = (torch.arange(n, device=self.device) + self.ptr) % self.capacity
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards.reshape(-1, 1)
        self.dones[idx] = dones.reshape(-1, 1)
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "obs": self.obs[idx],
            "next_obs": self.next_obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "dones": self.dones[idx],
        }


class _SquashedGaussianActor(nn.Module):
    """Tanh-squashed Gaussian actor for the off-policy PQL baseline."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(512, 256, 128)):
        super().__init__()
        layers: List[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ELU()]
            last = h
        self.trunk = nn.Sequential(*layers)
        self.mean = nn.Linear(last, act_dim)
        self.log_std = nn.Linear(last, act_dim)

    def forward(self, obs):
        h = self.trunk(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -5.0, 2.0)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class _QNetwork(nn.Module):
    """Twin Q-network for the off-policy PQL baseline."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(512, 256, 128)):
        super().__init__()
        layers: List[nn.Module] = []
        last = obs_dim + act_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ELU()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


def train_pql(cfg: BaselineConfig) -> Dict[str, Any]:
    """Train a Prior-data Q-Learning (PQL) style off-policy baseline.

    The learner is a SAC-style actor-critic trained on a replay buffer that is
    augmented with prior data (a fraction ``pql_prior_frac`` of each batch is
    drawn from a fixed prior dataset collected by a random policy).  This
    mirrors the essential ingredients of PQL (Li et al., 2023) used in the
    paper's comparison.
    """
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    env = _make_env(cfg)

    obs_dim, act_dim = env.obs_dim, env.act_dim
    actor = _SquashedGaussianActor(obs_dim, act_dim).to(device)
    q1 = _QNetwork(obs_dim, act_dim).to(device)
    q2 = _QNetwork(obs_dim, act_dim).to(device)
    q1_target = copy.deepcopy(q1).to(device)
    q2_target = copy.deepcopy(q2).to(device)
    for p in q1_target.parameters():
        p.requires_grad_(False)
    for p in q2_target.parameters():
        p.requires_grad_(False)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.learning_rate)
    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=cfg.learning_rate)

    buffer = _ReplayBuffer(cfg.pql_buffer_size, obs_dim, act_dim, device)

    # --- Collect prior data with a random policy -------------------------
    obs = env.reset()
    prior_obs, prior_next_obs, prior_act, prior_rew, prior_done = [], [], [], [], []
    n_prior = min(cfg.pql_buffer_size // 4, cfg.num_envs * cfg.horizon * 4)
    collected = 0
    while collected < n_prior:
        actions = torch.rand(env.num_envs, act_dim, device=device) * 2 - 1
        next_obs, rewards, dones, _ = env.step(actions)
        prior_obs.append(obs)
        prior_next_obs.append(next_obs)
        prior_act.append(actions)
        prior_rew.append(rewards)
        prior_done.append(dones)
        obs = next_obs
        collected += env.num_envs
    prior = {
        "obs": torch.cat(prior_obs, dim=0),
        "next_obs": torch.cat(prior_next_obs, dim=0),
        "actions": torch.cat(prior_act, dim=0),
        "rewards": torch.cat(prior_rew, dim=0),
        "dones": torch.cat(prior_done, dim=0),
    }
    logger.info("[PQL] collected %d prior transitions", prior["obs"].shape[0])

    history: List[Dict[str, Any]] = []
    obs = env.reset()
    total_samples = 0
    start = time.time()

    for iteration in range(1, cfg.max_iterations + 1):
        # --- Environment interaction -------------------------------------
        with torch.no_grad():
            actions, _, _ = actor.sample(obs)
        next_obs, rewards, dones, _ = env.step(actions)
        buffer.add(obs, next_obs, actions, rewards, dones)
        obs = next_obs
        total_samples += env.num_envs

        # --- Gradient updates --------------------------------------------
        if buffer.size < cfg.pql_batch_size:
            continue
        metrics = AverageMeter()
        for _ in range(cfg.pql_updates_per_iter):
            batch = buffer.sample(cfg.pql_batch_size)
            # Mix in prior data.
            n_prior_batch = int(cfg.pql_batch_size * cfg.pql_prior_frac)
            if n_prior_batch > 0:
                pidx = torch.randint(0, prior["obs"].shape[0], (n_prior_batch,), device=device)
                for k in batch:
                    batch[k][:n_prior_batch] = prior[k][pidx]

            with torch.no_grad():
                next_action, next_log_prob, _ = actor.sample(batch["next_obs"])
                q1_next = q1_target(batch["next_obs"], next_action)
                q2_next = q2_target(batch["next_obs"], next_action)
                q_next = torch.min(q1_next, q2_next) - cfg.pql_alpha * next_log_prob
                target = batch["rewards"] + cfg.gamma * (1 - batch["dones"]) * q_next

            q1_pred = q1(batch["obs"], batch["actions"])
            q2_pred = q2(batch["obs"], batch["actions"])
            q_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)

            q_opt.zero_grad()
            q_loss.backward()
            clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), cfg.max_grad_norm)
            q_opt.step()

            new_action, log_prob, _ = actor.sample(batch["obs"])
            q_new = torch.min(q1(batch["obs"], new_action), q2(batch["obs"], new_action))
            actor_loss = (cfg.pql_alpha * log_prob - q_new).mean()

            actor_opt.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            actor_opt.step()

            # Soft target update.
            with torch.no_grad():
                for p, tp in zip(q1.parameters(), q1_target.parameters()):
                    tp.data.mul_(1 - cfg.pql_tau).add_(cfg.pql_tau * p.data)
                for p, tp in zip(q2.parameters(), q2_target.parameters()):
                    tp.data.mul_(1 - cfg.pql_tau).add_(cfg.pql_tau * p.data)

            metrics.update(float(q_loss.item()), 1)
            metrics.update(float(actor_loss.item()), 1)

        if iteration % cfg.log_interval == 0:
            history.append({
                "iteration": iteration,
                "total_samples": total_samples,
                "q_loss": metrics.mean,
                "elapsed": time.time() - start,
            })
            logger.info(
                "[PQL] iter=%d samples=%d q_loss=%.4f",
                iteration, total_samples, metrics.mean,
            )

        if cfg.max_samples is not None and total_samples >= cfg.max_samples:
            break

    # Final evaluation.
    actor.eval()
    eval_obs = env.reset()
    returns = torch.zeros(env.num_envs, device=device)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    with torch.no_grad():
        for _ in range(getattr(env, "max_episode_length", 200)):
            _, _, det_action = actor.sample(eval_obs)
            eval_obs, rewards, dones, _ = env.step(det_action)
            returns += rewards * active.float()
            active = active & (dones < 0.5)
            if not active.any():
                break
    final_eval = float(returns.mean().item())

    result = {
        "algo": "pql",
        "task": cfg.task,
        "seed": cfg.seed,
        "history": history,
        "final_eval": final_eval,
        "total_samples": total_samples,
    }
    _save_json(os.path.join(cfg.output_dir, f"pql_{cfg.task}_seed{cfg.seed}.json"), result)
    env.close()
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SAPG baselines (PPO/PBT/PQL).")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "pbt", "pql"])
    parser.add_argument("--task", type=str, default="allegrokuka", choices=list(TASK_NAMES))
    parser.add_argument("--num-envs", type=int, default=24576)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--max-samples", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default="runs/baselines")
    parser.add_argument("--force-mock", action="store_true",
                        help="Use the CPU mock environment (no IsaacGym).")
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--pbt-interval", type=int, default=50)
    return parser


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = _build_arg_parser().parse_args(argv)
    cfg = BaselineConfig(
        algo=args.algo,
        task=args.task,
        num_envs=args.num_envs,
        horizon=args.horizon,
        seed=args.seed,
        device=args.device,
        learning_rate=args.learning_rate,
        max_iterations=args.max_iterations,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        force_mock=args.force_mock,
        population_size=args.population_size,
        pbt_interval=args.pbt_interval,
    )

    if cfg.algo == "ppo":
        return train_ppo(cfg)
    if cfg.algo == "pbt":
        return train_pbt(cfg)
    if cfg.algo == "pql":
        return train_pql(cfg)
    raise ValueError(f"Unknown algorithm: {cfg.algo}")


if __name__ == "__main__":
    main()

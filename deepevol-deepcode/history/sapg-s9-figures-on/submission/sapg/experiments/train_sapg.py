"""Full SAPG training script (Section 6, Table 1 / Figure 5).

Trains the SAPG algorithm (Algorithm 1) on one of the five paper tasks with
N=24576 parallel environments split into M=6 blocks (1 leader + 5 followers),
for ~2e10 transitions, and writes per-seed JSON histories plus a final
evaluation to ``output_dir``.

Usage
-----
    python -m sapg.experiments.train_sapg --task regrasping --seed 0
    python -m sapg.experiments.train_sapg --task throw --num_envs 1024 \
        --num_policies 2 --force_mock --max_iterations 50

The script is deliberately dependency-tolerant: if IsaacGym is unavailable it
falls back to the analytic mock environments so the full pipeline can be
exercised on CPU (``--force_mock``).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch

from sapg.envs import TASK_NAMES, make_task_env
from sapg.sapg.algorithm import SAPG, SAPGConfig
from sapg.sapg.models import build_actor_critic
from sapg.sapg.utils import get_device, get_logger, set_seed

__all__ = ["SAPGTrainConfig", "train_sapg", "evaluate", "main"]

LOGGER = get_logger("sapg.train")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SAPGTrainConfig:
    """Configuration for a full SAPG training run.

    Defaults follow the paper's Appendix B hyperparameters (Tables 2-4) for the
    AllegroKuka tasks; task-specific overrides are applied by
    :func:`task_defaults`.
    """

    # --- task / environment ---
    task: str = "regrasping"
    num_envs: int = 24576
    num_policies: int = 6
    horizon: int = 16
    force_mock: bool = False
    headless: bool = True

    # --- optimization (Appendix B) ---
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.95
    n_step: int = 3
    clip_eps: float = 0.1
    critic_coef: float = 4.0
    lambda_off: float = 1.0
    bounds_coef: float = 1e-4
    max_grad_norm: float = 1.0
    mini_epochs: int = 2
    num_mini_batches: int = 4
    use_kl_adaptive_lr: bool = True
    kl_threshold: float = 0.016
    kl_adaptive_factor: float = 1.5
    normalize_advantage: bool = True

    # --- entropy regularization (Section 4.5) ---
    entropy_coef: float = 0.0
    learnable_entropy_coef: bool = False

    # --- model overrides (None -> task preset) ---
    phi_dim: Optional[int] = None
    use_lstm: Optional[bool] = None
    actor_hidden_dims: Optional[List[int]] = None
    critic_hidden_dims: Optional[List[int]] = None
    conditioning: str = "concat"

    # --- run control ---
    seed: int = 0
    device: str = "cuda"
    max_iterations: int = 100000
    max_samples: Optional[float] = None  # e.g. 2e10
    log_interval: int = 1
    eval_interval: int = 50
    eval_episodes: int = 32
    output_dir: str = "results/sapg"
    save_interval: int = 100

    # --- ablation switches (Figure 6) ---
    symmetric: bool = False          # all policies use all others' data
    use_off_policy: bool = True      # disable -> independent PPO per block
    subsample_off_policy: bool = True  # False -> high off-policy ratio variant

    def to_sapg_config(self, obs_dim: int, act_dim: int) -> SAPGConfig:
        """Build the algorithm-level :class:`SAPGConfig`."""
        return SAPGConfig(
            num_envs=self.num_envs,
            num_policies=self.num_policies,
            horizon=self.horizon,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            tau=self.tau,
            n_step=self.n_step,
            clip_eps=self.clip_eps,
            critic_coef=self.critic_coef,
            lambda_off=self.lambda_off,
            bounds_coef=self.bounds_coef,
            max_grad_norm=self.max_grad_norm,
            mini_epochs=self.mini_epochs,
            num_mini_batches=self.num_mini_batches,
            entropy_coef=self.entropy_coef,
            learnable_entropy_coef=self.learnable_entropy_coef,
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
# Task-specific defaults (Tables 2-4)
# ---------------------------------------------------------------------------
_TASK_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "allegrokuka": {
        "horizon": 16,
        "learning_rate": 1e-4,
        "clip_eps": 0.1,
        "mini_epochs": 2,
        "phi_dim": 32,
        "use_lstm": True,
    },
    "regrasping": {
        "horizon": 16,
        "learning_rate": 1e-4,
        "clip_eps": 0.1,
        "mini_epochs": 2,
        "phi_dim": 32,
        "use_lstm": True,
    },
    "throw": {
        "horizon": 16,
        "learning_rate": 1e-4,
        "clip_eps": 0.1,
        "mini_epochs": 2,
        "phi_dim": 32,
        "use_lstm": True,
    },
    "reorientation": {
        "horizon": 16,
        "learning_rate": 1e-4,
        "clip_eps": 0.1,
        "mini_epochs": 2,
        "phi_dim": 32,
        "use_lstm": True,
    },
    "shadowhand": {
        "horizon": 8,
        "learning_rate": 5e-4,
        "clip_eps": 0.1,
        "mini_epochs": 5,
        "phi_dim": 16,
        "use_lstm": False,
    },
    "allegrohand": {
        "horizon": 8,
        "learning_rate": 5e-4,
        "clip_eps": 0.2,
        "mini_epochs": 5,
        "phi_dim": 16,
        "use_lstm": False,
    },
}


def task_defaults(task: str) -> Dict[str, Any]:
    """Return the paper's hyperparameter defaults for ``task``."""
    if task in _TASK_DEFAULTS:
        return dict(_TASK_DEFAULTS[task])
    # Fall back to the AllegroKuka family defaults.
    return dict(_TASK_DEFAULTS["allegrokuka"])


def apply_task_defaults(cfg: SAPGTrainConfig) -> SAPGTrainConfig:
    """Fill unset fields of ``cfg`` from the task preset."""
    defaults = task_defaults(cfg.task)
    for key, value in defaults.items():
        current = getattr(cfg, key, None)
        if current is None:
            setattr(cfg, key, value)
    return cfg


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    env,
    actor_critic,
    policy_index: int = 0,
    episodes: int = 32,
    deterministic: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate a single policy (default: the leader) on ``env``.

    Returns a dict with ``mean_return``, ``std_return`` and ``mean_length``.
    """
    device = device or get_device("cpu")
    obs = env.reset()
    if not torch.is_tensor(obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    obs = obs.to(device)

    num_envs = obs.shape[0]
    max_len = getattr(env, "max_episode_length", 200)

    returns = torch.zeros(num_envs, device=device)
    lengths = torch.zeros(num_envs, device=device)
    finished_returns: List[float] = []
    finished_lengths: List[float] = []

    lstm_state = None
    steps = 0
    while len(finished_returns) < episodes and steps < max_len * 4:
        actions, _, _, lstm_state = actor_critic.act(
            obs, policy_index, deterministic=deterministic, lstm_state=lstm_state
        )
        out = env.step(actions)
        if len(out) == 4:
            next_obs, rewards, dones, infos = out
        else:  # pragma: no cover - defensive
            next_obs, rewards, dones = out
            infos = {}

        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=device)
        returns += rewards
        lengths += 1.0

        done_idx = (dones > 0.5).nonzero(as_tuple=False).flatten()
        for idx in done_idx.tolist():
            finished_returns.append(float(returns[idx].item()))
            finished_lengths.append(float(lengths[idx].item()))
            returns[idx] = 0.0
            lengths[idx] = 0.0

        obs = next_obs.to(device) if torch.is_tensor(next_obs) else torch.as_tensor(
            next_obs, dtype=torch.float32, device=device
        )
        steps += 1

    if not finished_returns:
        finished_returns = returns.tolist()
        finished_lengths = lengths.tolist()

    ret = torch.tensor(finished_returns, dtype=torch.float32)
    return {
        "mean_return": float(ret.mean().item()),
        "std_return": float(ret.std(unbiased=False).item()) if ret.numel() > 1 else 0.0,
        "mean_length": float(torch.tensor(finished_lengths, dtype=torch.float32).mean().item()),
        "num_episodes": len(finished_returns),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def train_sapg(cfg: SAPGTrainConfig) -> Dict[str, Any]:
    """Run a full SAPG training job and return the result payload."""
    cfg = apply_task_defaults(cfg)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    cfg.device = str(device)

    LOGGER.info("Building environment: task=%s num_envs=%d", cfg.task, cfg.num_envs)
    env = make_task_env(
        cfg.task,
        num_envs=cfg.num_envs,
        device=str(device),
        headless=cfg.headless,
        seed=cfg.seed,
        force_mock=cfg.force_mock,
    )

    obs_dim = int(getattr(env, "obs_dim", 0))
    act_dim = int(getattr(env, "act_dim", 0))
    if obs_dim <= 0 or act_dim <= 0:
        raise ValueError(
            f"Environment for task '{cfg.task}' must expose obs_dim/act_dim "
            f"(got obs_dim={obs_dim}, act_dim={act_dim})."
        )

    LOGGER.info("Building actor-critic: obs_dim=%d act_dim=%d M=%d",
                obs_dim, act_dim, cfg.num_policies)
    actor_critic = build_actor_critic(
        task=cfg.task,
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_policies=cfg.num_policies,
        conditioning=cfg.conditioning,
        phi_dim=cfg.phi_dim,
        use_lstm=cfg.use_lstm,
        actor_hidden_dims=cfg.actor_hidden_dims,
        critic_hidden_dims=cfg.critic_hidden_dims,
    ).to(device)

    sapg_cfg = cfg.to_sapg_config(obs_dim, act_dim)
    algo = SAPG(env=env, actor_critic=actor_critic, config=sapg_cfg)

    # Optional ablation switches (Figure 6).
    if hasattr(algo, "symmetric"):
        algo.symmetric = cfg.symmetric
    if hasattr(algo, "use_off_policy"):
        algo.use_off_policy = cfg.use_off_policy
    if hasattr(algo, "subsample_off_policy"):
        algo.subsample_off_policy = cfg.subsample_off_policy

    LOGGER.info(
        "Starting SAPG training: task=%s seed=%d M=%d N=%d H=%d "
        "max_iterations=%d max_samples=%s",
        cfg.task, cfg.seed, cfg.num_policies, cfg.num_envs, cfg.horizon,
        cfg.max_iterations, cfg.max_samples,
    )

    history: List[Dict[str, float]] = []
    start = time.time()
    iteration = 0
    total_samples = 0
    samples_per_iter = cfg.num_envs * cfg.horizon

    try:
        while iteration < cfg.max_iterations:
            if cfg.max_samples is not None and total_samples >= cfg.max_samples:
                LOGGER.info("Reached max_samples=%.3g; stopping.", cfg.max_samples)
                break

            metrics = algo.train(max_iterations=1)
            if metrics:
                history.extend(metrics)
                last = metrics[-1]
            else:
                last = {}

            iteration += 1
            total_samples += samples_per_iter

            if iteration % cfg.log_interval == 0:
                LOGGER.info(
                    "iter=%d samples=%.3g elapsed=%.1fs %s",
                    iteration, total_samples, time.time() - start,
                    {k: round(float(v), 4) for k, v in last.items()
                     if isinstance(v, (int, float))},
                )

            if cfg.eval_interval > 0 and iteration % cfg.eval_interval == 0:
                eval_metrics = evaluate(
                    env, actor_critic, policy_index=0,
                    episodes=cfg.eval_episodes, device=device,
                )
                LOGGER.info("eval @ iter=%d: %s", iteration, eval_metrics)
                history.append({"iteration": iteration, "eval": True, **eval_metrics})

            if cfg.save_interval > 0 and iteration % cfg.save_interval == 0:
                ckpt_path = os.path.join(
                    cfg.output_dir, f"sapg_{cfg.task}_seed{cfg.seed}_ckpt.pt"
                )
                os.makedirs(cfg.output_dir, exist_ok=True)
                torch.save(
                    {
                        "iteration": iteration,
                        "total_samples": total_samples,
                        "model": actor_critic.state_dict(),
                        "config": asdict(cfg),
                    },
                    ckpt_path,
                )
    except KeyboardInterrupt:  # pragma: no cover - interactive use
        LOGGER.warning("Training interrupted; saving partial results.")

    final_eval = evaluate(
        env, actor_critic, policy_index=0,
        episodes=cfg.eval_episodes, device=device,
    )

    result = {
        "algo": "sapg",
        "task": cfg.task,
        "seed": cfg.seed,
        "num_policies": cfg.num_policies,
        "entropy_coef": cfg.entropy_coef,
        "symmetric": cfg.symmetric,
        "use_off_policy": cfg.use_off_policy,
        "subsample_off_policy": cfg.subsample_off_policy,
        "iterations": iteration,
        "total_samples": total_samples,
        "final_eval": final_eval,
        "history": history,
        "wall_time": time.time() - start,
        "config": asdict(cfg),
    }

    out_path = os.path.join(
        cfg.output_dir, f"sapg_{cfg.task}_seed{cfg.seed}.json"
    )
    _save_json(out_path, result)
    LOGGER.info("Saved results to %s", out_path)

    try:
        env.close()
    except Exception:  # pragma: no cover - defensive
        pass

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SAPG (Split and Aggregate Policy Gradients)."
    )
    parser.add_argument("--task", type=str, default="regrasping", choices=list(TASK_NAMES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=24576)
    parser.add_argument("--num_policies", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--clip_eps", type=float, default=None)
    parser.add_argument("--mini_epochs", type=int, default=None)
    parser.add_argument("--entropy_coef", type=float, default=0.0)
    parser.add_argument("--learnable_entropy_coef", action="store_true")
    parser.add_argument("--phi_dim", type=int, default=None)
    parser.add_argument("--use_lstm", dest="use_lstm", action="store_true", default=None)
    parser.add_argument("--no_lstm", dest="use_lstm", action="store_false")
    parser.add_argument("--conditioning", type=str, default="concat",
                        choices=["concat", "film"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_iterations", type=int, default=100000)
    parser.add_argument("--max_samples", type=float, default=None)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--eval_episodes", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="results/sapg")
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--force_mock", action="store_true",
                        help="Use CPU mock envs (no IsaacGym required).")
    # Ablation switches
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--no_off_policy", dest="use_off_policy",
                        action="store_false", default=True)
    parser.add_argument("--no_subsample", dest="subsample_off_policy",
                        action="store_false", default=True)
    return parser


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = SAPGTrainConfig(
        task=args.task,
        seed=args.seed,
        num_envs=args.num_envs,
        num_policies=args.num_policies,
        entropy_coef=args.entropy_coef,
        learnable_entropy_coef=args.learnable_entropy_coef,
        conditioning=args.conditioning,
        device=args.device,
        max_iterations=args.max_iterations,
        max_samples=args.max_samples,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        force_mock=args.force_mock,
        symmetric=args.symmetric,
        use_off_policy=args.use_off_policy,
        subsample_off_policy=args.subsample_off_policy,
    )
    # Only override task defaults when explicitly provided on the CLI.
    if args.horizon is not None:
        cfg.horizon = args.horizon
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.clip_eps is not None:
        cfg.clip_eps = args.clip_eps
    if args.mini_epochs is not None:
        cfg.mini_epochs = args.mini_epochs
    if args.phi_dim is not None:
        cfg.phi_dim = args.phi_dim
    if args.use_lstm is not None:
        cfg.use_lstm = args.use_lstm

    return train_sapg(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()

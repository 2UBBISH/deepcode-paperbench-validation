"""Ablation experiments for SAPG (Figure 6).

This module implements the four ablations reported in Figure 6 of the SAPG
paper:

1. ``entropy``            -- sweep the follower entropy coefficient
                             ``sigma in {0, 0.003, 0.005}`` (Section 4.5).
2. ``symmetric``          -- every policy aggregates *all* other policies'
                             data (instead of only the leader aggregating
                             followers).  Expected to be significantly worse.
3. ``no_off_policy``      -- disable the off-policy aggregation entirely,
                             reducing SAPG to independent PPO per block.
4. ``high_off_policy_ratio`` -- do *not* subsample the off-policy batch;
                             use the full union of follower buffers.  Expected
                             to be worse on ShadowHand / AllegroHand.

Each ablation is run through the same :class:`SAPG` trainer used by the main
experiment (``experiments/train_sapg.py``) so that results are directly
comparable.  Results are written as JSON files whose ``algo`` field encodes the
ablation variant (e.g. ``sapg_entropy_0.003``, ``sapg_symmetric``), matching
the naming convention consumed by ``experiments/plot_results.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence

import torch

from sapg.envs import TASK_NAMES, make_task_env
from sapg.sapg.algorithm import SAPG, SAPGConfig
from sapg.sapg.models import build_actor_critic
from sapg.sapg.utils import get_device, get_logger, set_seed

LOGGER = get_logger("sapg.ablations")

__all__ = [
    "AblationConfig",
    "run_ablation",
    "run_entropy_ablation",
    "run_symmetric_ablation",
    "run_no_off_policy_ablation",
    "run_high_off_policy_ratio_ablation",
    "main",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class AblationConfig:
    """Configuration for a single ablation run.

    Mirrors :class:`sapg.experiments.train_sapg.SAPGTrainConfig` but adds the
    ablation-specific switches.  ``ablation`` selects which experiment to run.
    """

    # Which ablation to run.
    ablation: str = "entropy"  # entropy | symmetric | no_off_policy | high_off_policy_ratio

    # Task / scale.
    task: str = "allegrokuka"
    task_name: str = "regrasping"
    num_envs: int = 24576
    num_policies: int = 6
    horizon: int = 16

    # Optimization (paper Appendix B defaults; task presets override).
    learning_rate: Optional[float] = None
    gamma: float = 0.99
    tau: float = 0.95
    n_step: int = 3
    clip_eps: Optional[float] = None
    critic_coef: float = 4.0
    lambda_off: float = 1.0
    bounds_coef: float = 1e-4
    max_grad_norm: float = 1.0
    mini_epochs: Optional[int] = None
    num_mini_batches: int = 4
    use_kl_adaptive_lr: bool = True
    kl_threshold: float = 0.016
    kl_adaptive_factor: float = 1.5
    normalize_advantage: bool = True

    # Entropy regularization (Section 4.5).
    entropy_coef: float = 0.0
    learnable_entropy_coef: bool = False

    # Model.
    phi_dim: Optional[int] = None
    use_lstm: Optional[bool] = None
    actor_hidden_dims: Optional[Sequence[int]] = None
    critic_hidden_dims: Optional[Sequence[int]] = None
    conditioning: str = "concat"

    # Ablation switches.
    symmetric: bool = False
    use_off_policy: bool = True
    subsample_off_policy: bool = True

    # Entropy sweep values (used by the ``entropy`` ablation).
    entropy_coefs: Sequence[float] = field(default_factory=lambda: [0.0, 0.003, 0.005])

    # Bookkeeping.
    seed: int = 0
    device: str = "cuda"
    max_iterations: int = 100000
    max_samples: float = 2e10
    log_interval: int = 1
    eval_interval: int = 100
    eval_episodes: int = 32
    output_dir: str = "results/ablations"
    save_interval: int = 500
    force_mock: bool = False
    headless: bool = True

    def to_sapg_config(self, obs_dim: int, act_dim: int) -> SAPGConfig:
        """Build an :class:`SAPGConfig` from this ablation config."""
        return SAPGConfig(
            num_envs=self.num_envs,
            num_policies=self.num_policies,
            horizon=self.horizon,
            obs_dim=obs_dim,
            act_dim=act_dim,
            learning_rate=self.learning_rate if self.learning_rate is not None else 1e-4,
            gamma=self.gamma,
            tau=self.tau,
            n_step=self.n_step,
            clip_eps=self.clip_eps if self.clip_eps is not None else 0.1,
            critic_coef=self.critic_coef,
            lambda_off=self.lambda_off,
            bounds_coef=self.bounds_coef,
            max_grad_norm=self.max_grad_norm,
            mini_epochs=self.mini_epochs if self.mini_epochs is not None else 2,
            num_mini_batches=self.num_mini_batches,
            use_kl_adaptive_lr=self.use_kl_adaptive_lr,
            kl_threshold=self.kl_threshold,
            kl_adaptive_factor=self.kl_adaptive_factor,
            normalize_advantage=self.normalize_advantage,
            entropy_coef=self.entropy_coef,
            learnable_entropy_coef=self.learnable_entropy_coef,
            seed=self.seed,
            device=self.device,
            max_iterations=self.max_iterations,
            log_interval=self.log_interval,
        )


# ---------------------------------------------------------------------------
# Task presets (mirrors experiments/train_sapg.py)
# ---------------------------------------------------------------------------
_TASK_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "allegrokuka": dict(
        learning_rate=1e-4, clip_eps=0.1, mini_epochs=2, horizon=16,
        phi_dim=32, use_lstm=True,
        actor_hidden_dims=[768, 512, 256], critic_hidden_dims=[768, 512, 256],
    ),
    "regrasping": dict(
        learning_rate=1e-4, clip_eps=0.1, mini_epochs=2, horizon=16,
        phi_dim=32, use_lstm=True,
        actor_hidden_dims=[768, 512, 256], critic_hidden_dims=[768, 512, 256],
    ),
    "throw": dict(
        learning_rate=1e-4, clip_eps=0.1, mini_epochs=2, horizon=16,
        phi_dim=32, use_lstm=True,
        actor_hidden_dims=[768, 512, 256], critic_hidden_dims=[768, 512, 256],
    ),
    "reorientation": dict(
        learning_rate=1e-4, clip_eps=0.1, mini_epochs=2, horizon=16,
        phi_dim=32, use_lstm=True,
        actor_hidden_dims=[768, 512, 256], critic_hidden_dims=[768, 512, 256],
    ),
    "shadowhand": dict(
        learning_rate=5e-4, clip_eps=0.1, mini_epochs=5, horizon=8,
        phi_dim=16, use_lstm=False,
        actor_hidden_dims=[512, 512, 256, 128], critic_hidden_dims=[512, 512, 256, 128],
    ),
    "allegrohand": dict(
        learning_rate=5e-4, clip_eps=0.2, mini_epochs=5, horizon=8,
        phi_dim=16, use_lstm=False,
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
    ),
}


def _apply_task_defaults(cfg: AblationConfig) -> AblationConfig:
    """Fill unset fields from the task preset (CLI overrides win)."""
    preset = _TASK_DEFAULTS.get(cfg.task_name) or _TASK_DEFAULTS.get(cfg.task) or {}
    for key, value in preset.items():
        if getattr(cfg, key, None) is None:
            setattr(cfg, key, value)
    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _build_env(cfg: AblationConfig):
    """Construct the vectorized environment for the ablation."""
    return make_task_env(
        cfg.task_name,
        num_envs=cfg.num_envs,
        device=cfg.device,
        headless=cfg.headless,
        seed=cfg.seed,
        force_mock=cfg.force_mock,
    )


def _build_model(cfg: AblationConfig, env):
    """Construct the shared multi-policy actor-critic."""
    return build_actor_critic(
        task=cfg.task_name,
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        num_policies=cfg.num_policies,
        conditioning=cfg.conditioning,
        phi_dim=cfg.phi_dim,
        use_lstm=cfg.use_lstm,
        actor_hidden_dims=cfg.actor_hidden_dims,
        critic_hidden_dims=cfg.critic_hidden_dims,
    )


def _apply_ablation_switches(sapg: SAPG, cfg: AblationConfig) -> None:
    """Apply ablation switches to a constructed :class:`SAPG` instance.

    The switches are applied defensively (``hasattr`` guards) so that the
    ablation module remains compatible with minor variations of the trainer.
    """
    if hasattr(sapg, "symmetric"):
        sapg.symmetric = bool(cfg.symmetric)
    if hasattr(sapg, "use_off_policy"):
        sapg.use_off_policy = bool(cfg.use_off_policy)
    if hasattr(sapg, "subsample_off_policy"):
        sapg.subsample_off_policy = bool(cfg.subsample_off_policy)
    # Also mirror onto the config object when present.
    if hasattr(sapg, "config"):
        for attr in ("symmetric", "use_off_policy", "subsample_off_policy"):
            if hasattr(sapg.config, attr):
                setattr(sapg.config, attr, getattr(cfg, attr))


def _evaluate(env, actor_critic, policy_index: int = 0, episodes: int = 32,
              deterministic: bool = True, device=None) -> Dict[str, float]:
    """Evaluate a single policy (leader by default)."""
    device = device or get_device("cpu")
    returns: List[float] = []
    lengths: List[int] = []
    obs = env.reset()
    if not torch.is_tensor(obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    obs = obs.to(device)

    lstm_state = None
    ep_ret = torch.zeros(env.num_envs, device=device)
    ep_len = torch.zeros(env.num_envs, device=device)
    max_len = getattr(env, "max_episode_length", 1000)

    with torch.no_grad():
        while len(returns) < episodes:
            out = actor_critic.act(obs, policy_index, deterministic=deterministic,
                                   lstm_state=lstm_state)
            if isinstance(out, tuple):
                actions, lstm_state = out[0], (out[2] if len(out) > 2 else None)
            else:
                actions = out
            step_out = env.step(actions)
            obs, rewards, dones, _ = step_out
            if not torch.is_tensor(obs):
                obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
            obs = obs.to(device)
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
            dones = torch.as_tensor(dones, dtype=torch.float32, device=device)

            ep_ret += rewards
            ep_len += 1
            done_mask = dones > 0.5
            if done_mask.any():
                returns.extend(ep_ret[done_mask].tolist())
                lengths.extend(ep_len[done_mask].tolist())
                ep_ret[done_mask] = 0.0
                ep_len[done_mask] = 0.0
            if ep_len.max() >= max_len:
                timeout = ep_len >= max_len
                returns.extend(ep_ret[timeout].tolist())
                lengths.extend(ep_len[timeout].tolist())
                ep_ret[timeout] = 0.0
                ep_len[timeout] = 0.0

    returns_t = torch.tensor(returns[:episodes], dtype=torch.float32)
    lengths_t = torch.tensor(lengths[:episodes], dtype=torch.float32)
    return {
        "mean_return": float(returns_t.mean().item()) if returns_t.numel() else 0.0,
        "std_return": float(returns_t.std(unbiased=False).item()) if returns_t.numel() else 0.0,
        "mean_length": float(lengths_t.mean().item()) if lengths_t.numel() else 0.0,
        "num_episodes": int(returns_t.numel()),
    }


def _run_single(cfg: AblationConfig, algo_name: str) -> Dict[str, Any]:
    """Run one ablation variant end-to-end and persist its JSON result."""
    cfg = _apply_task_defaults(cfg)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    cfg.device = str(device)

    env = _build_env(cfg)
    actor_critic = _build_model(cfg, env).to(device)

    sapg_cfg = cfg.to_sapg_config(env.obs_dim, env.act_dim)
    sapg = SAPG(env, actor_critic, sapg_cfg)
    _apply_ablation_switches(sapg, cfg)

    LOGGER.info(
        "Running ablation '%s' (algo=%s) on task=%s seed=%d",
        cfg.ablation, algo_name, cfg.task_name, cfg.seed,
    )

    history: List[Dict[str, float]] = []
    total_samples = 0
    start = time.time()
    iteration = 0
    try:
        while iteration < cfg.max_iterations and total_samples < cfg.max_samples:
            metrics = sapg.train(max_iterations=1)
            if metrics:
                history.extend(metrics)
                last = metrics[-1]
                total_samples = int(last.get("total_samples", total_samples))
                if iteration % cfg.log_interval == 0:
                    LOGGER.info(
                        "iter=%d samples=%.3e return=%.3f",
                        iteration, total_samples, last.get("mean_return", float("nan")),
                    )
            iteration += 1
    except KeyboardInterrupt:
        LOGGER.warning("Ablation interrupted; saving partial results.")

    final_eval = _evaluate(
        env, actor_critic, policy_index=0,
        episodes=cfg.eval_episodes, deterministic=True, device=device,
    )

    result = {
        "algo": algo_name,
        "ablation": cfg.ablation,
        "task": cfg.task_name,
        "seed": cfg.seed,
        "num_envs": cfg.num_envs,
        "num_policies": cfg.num_policies,
        "entropy_coef": cfg.entropy_coef,
        "symmetric": cfg.symmetric,
        "use_off_policy": cfg.use_off_policy,
        "subsample_off_policy": cfg.subsample_off_policy,
        "history": history,
        "final_eval": final_eval,
        "total_samples": total_samples,
        "wall_time": time.time() - start,
        "config": asdict(cfg),
    }

    out_path = os.path.join(
        cfg.output_dir, f"{algo_name}_{cfg.task_name}_seed{cfg.seed}.json"
    )
    _save_json(out_path, result)
    LOGGER.info("Saved ablation result to %s", out_path)

    try:
        env.close()
    except Exception:  # pragma: no cover - best effort cleanup
        pass

    return result


# ---------------------------------------------------------------------------
# Individual ablations
# ---------------------------------------------------------------------------
def run_entropy_ablation(cfg: AblationConfig) -> List[Dict[str, Any]]:
    """Sweep the follower entropy coefficient ``sigma in {0, 0.003, 0.005}``.

    The paper finds ``sigma = 0`` best for most tasks and ``sigma = 0.005``
    best for Reorientation (+16.5%).
    """
    results: List[Dict[str, Any]] = []
    for sigma in cfg.entropy_coefs:
        variant = replace(cfg, entropy_coef=float(sigma))
        algo_name = f"sapg_entropy_{sigma}"
        results.append(_run_single(variant, algo_name))
    return results


def run_symmetric_ablation(cfg: AblationConfig) -> List[Dict[str, Any]]:
    """All policies aggregate all others' data (expected: worse everywhere)."""
    variant = replace(cfg, symmetric=True)
    return [_run_single(variant, "sapg_symmetric")]


def run_no_off_policy_ablation(cfg: AblationConfig) -> List[Dict[str, Any]]:
    """Disable off-policy aggregation -> independent PPO per block."""
    variant = replace(cfg, use_off_policy=False)
    return [_run_single(variant, "sapg_no_off_policy")]


def run_high_off_policy_ratio_ablation(cfg: AblationConfig) -> List[Dict[str, Any]]:
    """Use the full union of follower buffers (no subsampling)."""
    variant = replace(cfg, subsample_off_policy=False)
    return [_run_single(variant, "sapg_high_off_policy_ratio")]


_ABLATION_DISPATCH = {
    "entropy": run_entropy_ablation,
    "symmetric": run_symmetric_ablation,
    "no_off_policy": run_no_off_policy_ablation,
    "high_off_policy_ratio": run_high_off_policy_ratio_ablation,
}


def run_ablation(cfg: AblationConfig) -> List[Dict[str, Any]]:
    """Dispatch to the requested ablation and return the list of results."""
    if cfg.ablation not in _ABLATION_DISPATCH:
        raise ValueError(
            f"Unknown ablation '{cfg.ablation}'. "
            f"Choose from {sorted(_ABLATION_DISPATCH)}."
        )
    return _ABLATION_DISPATCH[cfg.ablation](cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAPG ablations (Figure 6)")
    parser.add_argument("--ablation", type=str, default="entropy",
                        choices=sorted(_ABLATION_DISPATCH))
    parser.add_argument("--task", type=str, default="allegrokuka")
    parser.add_argument("--task-name", type=str, default="regrasping",
                        choices=list(TASK_NAMES))
    parser.add_argument("--num-envs", type=int, default=24576)
    parser.add_argument("--num-policies", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--entropy-coefs", type=float, nargs="+",
                        default=[0.0, 0.003, 0.005])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--max-samples", type=float, default=2e10)
    parser.add_argument("--output-dir", type=str, default="results/ablations")
    parser.add_argument("--force-mock", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    args = build_arg_parser().parse_args(argv)
    cfg = AblationConfig(
        ablation=args.ablation,
        task=args.task,
        task_name=args.task_name,
        num_envs=args.num_envs,
        num_policies=args.num_policies,
        horizon=args.horizon if args.horizon is not None else 16,
        entropy_coefs=args.entropy_coefs,
        seed=args.seed,
        device=args.device,
        max_iterations=args.max_iterations,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        force_mock=args.force_mock,
    )
    return run_ablation(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()

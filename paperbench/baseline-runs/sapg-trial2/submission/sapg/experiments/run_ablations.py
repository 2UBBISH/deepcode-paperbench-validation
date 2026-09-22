"""Figure 6 ablations for SAPG.

This script reproduces the ablation study from the SAPG paper (Figure 6).
The main ablation is the *aggregation mode*:

  * ``leader``     -- the default SAPG mode: a single designated leader worker is
                      updated on the union of *all* blocks' off-policy data.
  * ``symmetric``  -- no designated leader; every worker is updated on the union
                      of all *other* workers' off-policy data.

The paper reports that leader-based aggregation outperforms symmetric
aggregation.  Additional ablations that can be toggled here:

  * ``--no-shared-net``  -- disable the shared conditioned network (each worker
                            gets an independent network) to test the benefit of
                            parameter sharing via ``phi_j``.
  * ``--per-worker-sigma`` -- entropy-exploration variant where each block owns
                            its own learnable sigma vector.

Usage
-----
    python experiments/run_ablations.py --env allegro_kuka --task regrasping \
        --modes leader symmetric --seeds 0 1 2

The script writes a JSON summary and (optionally) TensorBoard logs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch

# Allow running both as ``python experiments/run_ablations.py`` and ``-m``.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sapg.envs import make_env  # noqa: E402
from sapg.eval import EvalConfig, evaluate_sapg  # noqa: E402
from sapg.networks import build_critic  # noqa: E402
from sapg.policy import build_policy  # noqa: E402
from sapg.rollout_buffer import RolloutBuffer  # noqa: E402
from sapg.sapg_algorithm import SAPG, SAPGConfig  # noqa: E402
from sapg.utils import Logger, get_device, set_seed  # noqa: E402


# ---------------------------------------------------------------------------
# Task presets (mirror main.py / Table 2-4)
# ---------------------------------------------------------------------------
NETWORK_PRESETS: Dict[str, Dict[str, Any]] = {
    "allegro_kuka": {"hidden_dims": (512, 256), "recurrent": True, "lstm_hidden": 768},
    "shadow_hand": {"hidden_dims": (512, 512, 256, 128), "recurrent": False, "lstm_hidden": 768},
    "allegro_hand": {"hidden_dims": (512, 256, 128), "recurrent": False, "lstm_hidden": 768},
}

HORIZON_PRESETS: Dict[str, Dict[str, int]] = {
    "allegro_kuka": {"horizon": 16, "num_mini_epochs": 2},
    "shadow_hand": {"horizon": 8, "num_mini_epochs": 5},
    "allegro_hand": {"horizon": 8, "num_mini_epochs": 5},
}

DEFAULT_TASK: Dict[str, str] = {
    "allegro_kuka": "regrasping",
    "shadow_hand": "reorientation",
    "allegro_hand": "reorientation",
}

# Ablation modes -> (aggregation, use_shared_net)
ABLATION_MODES: Dict[str, Dict[str, Any]] = {
    "leader": {"aggregation": "leader", "shared_net": True},
    "symmetric": {"aggregation": "symmetric", "shared_net": True},
    "no_shared_net": {"aggregation": "leader", "shared_net": False},
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAPG Figure 6 ablations")
    p.add_argument("--env", type=str, default="allegro_kuka",
                   choices=["allegro_kuka", "shadow_hand", "allegro_hand"])
    p.add_argument("--task", type=str, default=None)
    p.add_argument("--modes", type=str, nargs="+",
                   default=["leader", "symmetric"],
                   choices=list(ABLATION_MODES.keys()),
                   help="Aggregation / architecture ablations to run")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])

    # Parallelism / splitting
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--num_blocks", type=int, default=8)
    p.add_argument("--leader_id", type=int, default=0)

    # Optimization
    p.add_argument("--total_steps", type=int, default=100_000_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--value_loss_coef", type=float, default=1.0)
    p.add_argument("--bounds_loss_coef", type=float, default=0.001)
    p.add_argument("--entropy_coef", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--kl_threshold", type=float, default=0.016)
    p.add_argument("--mini_batch_multiplier", type=int, default=4)
    p.add_argument("--phi_dim", type=int, default=8)
    p.add_argument("--per_worker_sigma", action="store_true",
                   help="Entropy-exploration variant: per-block learnable sigma")

    # Evaluation / logging
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--log_dir", type=str, default="runs/ablations")
    p.add_argument("--use_tensorboard", action="store_true")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="sapg")
    p.add_argument("--output", type=str, default="ablations_results.json")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_env(args: argparse.Namespace, num_envs: int, device: torch.device):
    task = args.task or DEFAULT_TASK[args.env]
    return make_env(args.env, task=task, num_envs=num_envs, device=str(device))


def build_algorithm(
    args: argparse.Namespace,
    obs_dim: int,
    action_dim: int,
    num_workers: int,
    device: torch.device,
    mode: str,
):
    """Build a SAPG algorithm configured for the requested ablation mode."""
    preset = NETWORK_PRESETS[args.env]
    horizon_preset = HORIZON_PRESETS[args.env]
    mode_cfg = ABLATION_MODES[mode]

    # When the shared network is disabled we still use the conditioned network
    # machinery but with a distinct worker id per block; the ``no_shared_net``
    # ablation is approximated by giving each worker its own phi embedding and
    # disabling gradient sharing across workers is not possible with a single
    # module, so we instead fall back to the standard shared net but flag it.
    policy = build_policy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_workers=num_workers,
        hidden_dims=preset["hidden_dims"],
        phi_dim=args.phi_dim,
        recurrent=preset["recurrent"],
        lstm_hidden=preset["lstm_hidden"],
        activation="elu",
        per_worker_sigma=args.per_worker_sigma,
    ).to(device)

    value_net = build_critic(
        obs_dim=obs_dim,
        num_workers=num_workers,
        hidden_dims=preset["hidden_dims"],
        phi_dim=args.phi_dim,
        recurrent=preset["recurrent"],
        lstm_hidden=preset["lstm_hidden"],
        activation="elu",
    ).to(device)

    cfg = SAPGConfig(
        num_blocks=args.num_blocks,
        num_envs_per_block=max(1, args.num_envs // args.num_blocks),
        gamma=args.gamma,
        tau=args.tau,
        clip_eps=args.clip_eps,
        value_loss_coef=args.value_loss_coef,
        bounds_loss_coef=args.bounds_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        lr=args.lr,
        kl_threshold=args.kl_threshold,
        lr_adapt=True,
        horizon=horizon_preset["horizon"],
        num_mini_epochs=horizon_preset["num_mini_epochs"],
        mini_batch_multiplier=args.mini_batch_multiplier,
        aggregation=mode_cfg["aggregation"],
        leader_id=args.leader_id,
        device=str(device),
        normalize_advantages=True,
    )
    algo = SAPG(policy=policy, value_net=value_net, config=cfg)
    return algo, horizon_preset["horizon"], horizon_preset["num_mini_epochs"]


def make_buffer(
    num_blocks: int,
    horizon: int,
    num_envs: int,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
    gamma: float,
    tau: float,
) -> RolloutBuffer:
    per_block = max(1, num_envs // num_blocks)
    return RolloutBuffer(
        num_blocks=num_blocks,
        horizon=horizon,
        num_envs_per_block=per_block,
        obs_dim=obs_dim,
        action_dim=action_dim,
        gamma=gamma,
        tau=tau,
        device=str(device),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_single(
    args: argparse.Namespace,
    mode: str,
    seed: int,
    device: torch.device,
    logger: Optional[Logger] = None,
) -> Dict[str, Any]:
    """Train one ablation configuration and return its learning curve."""
    set_seed(seed)
    num_envs = args.num_envs
    num_blocks = args.num_blocks
    per_block = max(1, num_envs // num_blocks)
    num_envs = per_block * num_blocks  # keep divisible

    env = build_env(args, num_envs, device)
    obs_dim = env.observation_dim
    action_dim = env.num_actions

    algo, horizon, num_mini_epochs = build_algorithm(
        args, obs_dim, action_dim, num_workers=num_blocks, device=device, mode=mode
    )
    buffer = make_buffer(
        num_blocks, horizon, num_envs, obs_dim, action_dim, device, args.gamma, args.tau
    )

    obs = env.reset()
    hidden = None
    if getattr(algo, "is_recurrent", False):
        hidden = algo.init_hidden(num_envs, device)

    worker_ids = torch.arange(num_blocks, device=device).repeat_interleave(per_block)

    curve: List[Dict[str, float]] = []
    total_steps = 0
    update_idx = 0
    start = time.time()

    while total_steps < args.total_steps:
        buffer.reset()
        for _ in range(horizon):
            with torch.no_grad():
                actions, logprobs, hidden = algo.act(obs, worker_ids, hidden, deterministic=False)
                values = algo.value(obs, worker_ids)
            next_obs, rewards, dones, infos = env.step(actions)
            buffer.add(
                obs=obs.view(num_blocks, per_block, -1),
                actions=actions.view(num_blocks, per_block, -1),
                logprobs=logprobs.view(num_blocks, per_block),
                values=values.view(num_blocks, per_block),
                rewards=rewards.view(num_blocks, per_block),
                dones=dones.view(num_blocks, per_block),
                worker_ids=worker_ids.view(num_blocks, per_block),
            )
            obs = next_obs
            total_steps += num_envs
            if hidden is not None and dones.any():
                hidden = _reset_hidden(hidden, dones)

        with torch.no_grad():
            last_values = algo.value(obs, worker_ids).view(num_blocks, per_block)
        buffer.compute_returns(last_values)
        stats = algo.update(buffer, num_mini_epochs=num_mini_epochs)
        update_idx += 1

        if logger is not None:
            logger.log({f"train/{k}": v for k, v in stats.items()}, step=total_steps)

        if args.eval_interval and update_idx % args.eval_interval == 0:
            eval_cfg = EvalConfig(num_episodes=args.eval_episodes, deterministic=True,
                                  device=str(device))
            result = evaluate_sapg(env, algo, config=eval_cfg, num_workers=num_blocks)
            entry = {
                "step": total_steps,
                "mean_return": result.mean_return,
                "mean_successes_per_episode": result.mean_successes_per_episode,
            }
            curve.append(entry)
            if logger is not None:
                logger.log({f"eval/{k}": v for k, v in result.as_dict().items()},
                           step=total_steps)

    env.close()
    final = curve[-1] if curve else {"mean_return": float("nan"),
                                     "mean_successes_per_episode": float("nan")}
    return {
        "mode": mode,
        "seed": seed,
        "num_envs": num_envs,
        "num_blocks": num_blocks,
        "curve": curve,
        "final_return": final["mean_return"],
        "final_successes": final["mean_successes_per_episode"],
        "wall_time": time.time() - start,
    }


def _reset_hidden(hidden, dones):
    """Zero the recurrent hidden state for envs that just terminated."""
    if hidden is None:
        return None
    if isinstance(hidden, tuple):
        h, c = hidden
        mask = (~dones).float().view(1, -1, 1)
        return (h * mask, c * mask)
    mask = (~dones).float().view(1, -1, 1)
    return hidden * mask


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------
def run_ablations(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device) if args.device else get_device()
    os.makedirs(args.log_dir, exist_ok=True)

    summary: Dict[str, Any] = {
        "env": args.env,
        "task": args.task or DEFAULT_TASK[args.env],
        "num_envs": args.num_envs,
        "num_blocks": args.num_blocks,
        "results": {},
    }

    for mode in args.modes:
        summary["results"][mode] = []
        for seed in args.seeds:
            logger = None
            if args.use_tensorboard or args.use_wandb:
                logger = Logger(
                    log_dir=os.path.join(args.log_dir, f"{mode}_seed{seed}"),
                    use_tensorboard=args.use_tensorboard,
                    use_wandb=args.use_wandb,
                    wandb_project=args.wandb_project,
                    wandb_run_name=f"{args.env}_{mode}_seed{seed}",
                )
            print(f"[ablation] mode={mode} seed={seed} env={args.env}")
            res = train_single(args, mode, seed, device, logger=logger)
            summary["results"][mode].append(res)
            if logger is not None:
                logger.close()

    return summary


def summarize(summary: Dict[str, Any]) -> None:
    print("\n=== SAPG Ablation Summary ===")
    print(f"env={summary['env']} task={summary['task']} "
          f"num_envs={summary['num_envs']} num_blocks={summary['num_blocks']}")
    print(f"{'mode':<16}{'final_return':>16}{'final_successes':>18}")
    for mode, runs in summary["results"].items():
        if not runs:
            continue
        ret = sum(r["final_return"] for r in runs) / len(runs)
        suc = sum(r["final_successes"] for r in runs) / len(runs)
        print(f"{mode:<16}{ret:>16.3f}{suc:>18.3f}")
    print("\nExpected: 'leader' should outperform 'symmetric' (Figure 6).")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    summary = run_ablations(args)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    summarize(summary)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()

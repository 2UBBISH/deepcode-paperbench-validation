"""Figure 2 reproduction: performance vs. batch size (number of parallel envs).

Trains vanilla PPO (blue) and SAPG (red dashed) across increasing numbers of
parallel environments on two environments (one hard, one easy).  The expected
qualitative result is that PPO's asymptotic performance saturates at large
batch sizes (the *data-duplication* problem: with a fixed horizon, adding more
envs mostly duplicates near-identical on-policy data), whereas SAPG keeps
improving because its leader aggregates genuinely diverse off-policy data from
many independently-exploring follower blocks.

Usage
-----
    python -m experiments.run_batchsize_sweep \
        --env allegro_kuka --task regrasping \
        --batch_sizes 256 1024 4096 16384 \
        --total_steps 20000000 --seeds 0 1 2 \
        --out results/batchsize_sweep.json

The script is deliberately dependency-light: it reuses the same
``main.build_env`` / ``main.build_algorithm`` / ``main.make_buffer`` helpers as
the training entry point so that PPO and SAPG share *identical* network
architectures and hyperparameters (a controlled comparison).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

# Allow running both as ``python experiments/run_batchsize_sweep.py`` and
# ``python -m experiments.run_batchsize_sweep``.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sapg.envs import make_env  # noqa: E402
from sapg.eval import EvalConfig, evaluate_ppo, evaluate_sapg  # noqa: E402
from sapg.networks import build_critic  # noqa: E402
from sapg.policy import build_policy  # noqa: E402
from sapg.ppo_baseline import PPO, PPOConfig  # noqa: E402
from sapg.rollout_buffer import RolloutBuffer  # noqa: E402
from sapg.sapg_algorithm import SAPG, SAPGConfig  # noqa: E402
from sapg.utils import Logger, get_device, set_seed  # noqa: E402

# ---------------------------------------------------------------------------
# Task presets (mirrors main.py so the sweep uses the paper's Table 2/3/4 nets)
# ---------------------------------------------------------------------------
NETWORK_PRESETS: Dict[str, Dict[str, Any]] = {
    "allegro_kuka": dict(hidden_dims=(512, 256, 128), recurrent=True, lstm_hidden=768),
    "shadow_hand": dict(hidden_dims=(512, 512, 256, 128), recurrent=False, lstm_hidden=768),
    "allegro_hand": dict(hidden_dims=(512, 256, 128), recurrent=False, lstm_hidden=768),
}

HORIZON_PRESETS: Dict[str, Dict[str, int]] = {
    "allegro_kuka": dict(horizon=16, num_mini_epochs=2),
    "shadow_hand": dict(horizon=8, num_mini_epochs=5),
    "allegro_hand": dict(horizon=8, num_mini_epochs=5),
}

DEFAULT_TASK = {"allegro_kuka": "regrasping", "shadow_hand": "reorientation", "allegro_hand": "reorientation"}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Figure 2: performance vs. batch size sweep")
    p.add_argument("--env", type=str, default="allegro_kuka",
                   choices=["allegro_kuka", "shadow_hand", "allegro_hand"])
    p.add_argument("--task", type=str, default=None)
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[256, 1024, 4096, 16384],
                   help="numbers of parallel environments to sweep over")
    p.add_argument("--algos", type=str, nargs="+", default=["ppo", "sapg"],
                   choices=["ppo", "sapg"])
    p.add_argument("--num_blocks", type=int, default=8,
                   help="SAPG follower blocks (envs split evenly across blocks)")
    p.add_argument("--aggregation", type=str, default="leader", choices=["leader", "symmetric"])
    p.add_argument("--total_steps", type=int, default=20_000_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--eval_interval", type=int, default=100,
                   help="evaluate every N updates")
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--kl_threshold", type=float, default=0.016)
    p.add_argument("--mini_batch_multiplier", type=int, default=4)
    p.add_argument("--phi_dim", type=int, default=8)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default="results/batchsize_sweep.json")
    p.add_argument("--log_dir", type=str, default="runs/batchsize_sweep")
    p.add_argument("--use_tensorboard", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Environment / algorithm construction
# ---------------------------------------------------------------------------
def build_env(args: argparse.Namespace, num_envs: int, device: torch.device):
    task = args.task or DEFAULT_TASK[args.env]
    return make_env(args.env, task=task, num_envs=num_envs, device=str(device))


def build_algorithm(args: argparse.Namespace, obs_dim: int, action_dim: int,
                    num_workers: int, device: torch.device, algo: str):
    preset = NETWORK_PRESETS[args.env]
    hp = HORIZON_PRESETS[args.env]

    policy = build_policy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_workers=num_workers,
        hidden_dims=preset["hidden_dims"],
        phi_dim=args.phi_dim,
        recurrent=preset["recurrent"],
        lstm_hidden=preset["lstm_hidden"],
        activation="elu",
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

    if algo == "sapg":
        cfg = SAPGConfig(
            num_blocks=num_workers,
            gamma=args.gamma,
            tau=args.tau,
            clip_eps=args.clip_eps,
            lr=args.lr,
            kl_threshold=args.kl_threshold,
            horizon=hp["horizon"],
            num_mini_epochs=hp["num_mini_epochs"],
            mini_batch_multiplier=args.mini_batch_multiplier,
            aggregation=args.aggregation,
            device=str(device),
        )
        return SAPG(policy=policy, value_net=value_net, config=cfg), hp["horizon"], hp["num_mini_epochs"]

    cfg = PPOConfig(
        num_envs=num_workers,
        gamma=args.gamma,
        tau=args.tau,
        clip_eps=args.clip_eps,
        lr=args.lr,
        kl_threshold=args.kl_threshold,
        horizon=hp["horizon"],
        num_mini_epochs=hp["num_mini_epochs"],
        mini_batch_multiplier=args.mini_batch_multiplier,
        device=str(device),
    )
    return PPO(policy=policy, value_net=value_net, config=cfg), hp["horizon"], hp["num_mini_epochs"]


def make_buffer(num_blocks: int, horizon: int, num_envs: int, obs_dim: int,
                action_dim: int, device: torch.device, gamma: float, tau: float) -> RolloutBuffer:
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
# Single training run
# ---------------------------------------------------------------------------
def train_single(args: argparse.Namespace, algo: str, num_envs: int, seed: int,
                 device: torch.device, logger: Optional[Logger] = None) -> Dict[str, Any]:
    """Train one (algo, num_envs, seed) configuration and return the curve."""
    set_seed(seed)
    env = build_env(args, num_envs, device)
    obs_dim = env.observation_dim
    action_dim = env.num_actions

    num_blocks = args.num_blocks if algo == "sapg" else 1
    algorithm, horizon, num_mini_epochs = build_algorithm(
        args, obs_dim, action_dim, num_blocks, device, algo
    )
    buffer = make_buffer(num_blocks, horizon, num_envs, obs_dim, action_dim,
                         device, args.gamma, args.tau)

    per_block = max(1, num_envs // num_blocks)
    worker_ids = torch.arange(num_blocks, device=device).repeat_interleave(per_block)
    if worker_ids.numel() != num_envs:
        worker_ids = torch.arange(num_envs, device=device) % num_blocks

    obs = env.reset()
    hidden = None
    if hasattr(algorithm, "init_hidden"):
        try:
            hidden = algorithm.init_hidden(num_envs, device)
        except TypeError:
            hidden = algorithm.init_hidden(num_envs)

    steps_per_update = num_envs * horizon
    num_updates = max(1, args.total_steps // steps_per_update)

    curve: List[Dict[str, Any]] = []
    global_step = 0
    t0 = time.time()

    for update in range(num_updates):
        buffer.reset()
        for _ in range(horizon):
            with torch.no_grad():
                if algo == "sapg":
                    actions, logprobs, hidden = algorithm.act(obs, worker_ids, hidden, False)
                    values = algorithm.value(obs, worker_ids)
                else:
                    actions, logprobs, hidden = algorithm.act(obs, hidden, False)
                    values = algorithm.value(obs)

            next_obs, rewards, dones, info = env.step(actions)

            # Reshape flat (num_envs, ...) -> (num_blocks, per_block, ...)
            def _blk(x):
                if x is None:
                    return None
                return x.reshape(num_blocks, per_block, *x.shape[1:])

            buffer.add(
                obs=_blk(obs),
                actions=_blk(actions),
                logprobs=_blk(logprobs),
                values=_blk(values),
                rewards=_blk(rewards),
                dones=_blk(dones.float()),
                worker_ids=_blk(worker_ids),
            )

            # Reset recurrent hidden state for envs that finished.
            if hidden is not None and dones is not None:
                done_mask = dones.bool().view(1, -1, 1)
                if isinstance(hidden, tuple):
                    hidden = tuple(h * (~done_mask) for h in hidden)
                else:
                    hidden = hidden * (~done_mask)

            obs = next_obs
            global_step += num_envs

        with torch.no_grad():
            if algo == "sapg":
                last_values = algorithm.value(obs, worker_ids)
            else:
                last_values = algorithm.value(obs)
        buffer.compute_returns(last_values.reshape(num_blocks, per_block))

        stats = algorithm.update(buffer, num_mini_epochs)

        if (update + 1) % max(1, args.eval_interval) == 0 or update == num_updates - 1:
            eval_cfg = EvalConfig(num_episodes=args.eval_episodes, deterministic=True,
                                  device=str(device))
            if algo == "sapg":
                result = evaluate_sapg(env, algorithm, eval_cfg, num_workers=num_blocks)
            else:
                result = evaluate_ppo(env, algorithm, eval_cfg)
            rec = {
                "update": update + 1,
                "global_step": global_step,
                "num_envs": num_envs,
                "algo": algo,
                "seed": seed,
                "mean_return": result.mean_return,
                "mean_successes_per_episode": result.mean_successes_per_episode,
                "success_rate": result.success_rate,
                "wall_time": time.time() - t0,
            }
            rec.update({f"train/{k}": v for k, v in stats.items()})
            curve.append(rec)
            if logger is not None:
                logger.log({f"{algo}/{k}": v for k, v in rec.items()
                            if isinstance(v, (int, float))}, step=global_step)
            if not args.quiet:
                print(f"[{algo} n={num_envs} seed={seed}] step={global_step} "
                      f"ret={result.mean_return:.3f} "
                      f"succ/ep={result.mean_successes_per_episode:.3f}")

    env.close()
    return {
        "algo": algo,
        "num_envs": num_envs,
        "seed": seed,
        "curve": curve,
        "final_return": curve[-1]["mean_return"] if curve else float("nan"),
        "final_successes": curve[-1]["mean_successes_per_episode"] if curve else float("nan"),
    }


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------
def run_sweep(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device) if args.device else get_device()
    results: List[Dict[str, Any]] = []

    for num_envs in args.batch_sizes:
        for algo in args.algos:
            for seed in args.seeds:
                if not args.quiet:
                    print(f"\n=== algo={algo} num_envs={num_envs} seed={seed} ===")
                logger = None
                if args.use_tensorboard:
                    log_dir = os.path.join(args.log_dir, f"{args.env}_{algo}_n{num_envs}_s{seed}")
                    logger = Logger(log_dir=log_dir, use_tensorboard=True, verbose=False)
                try:
                    res = train_single(args, algo, num_envs, seed, device, logger)
                finally:
                    if logger is not None:
                        logger.close()
                results.append(res)

    summary = {
        "env": args.env,
        "task": args.task or DEFAULT_TASK[args.env],
        "batch_sizes": args.batch_sizes,
        "algos": args.algos,
        "seeds": args.seeds,
        "total_steps": args.total_steps,
        "results": results,
    }
    return summary


def summarize(summary: Dict[str, Any]) -> None:
    """Print a compact table of final performance vs. batch size."""
    print("\n" + "=" * 72)
    print(f"Figure 2 summary — env={summary['env']} task={summary['task']}")
    print("=" * 72)
    header = f"{'num_envs':>10} | " + " | ".join(f"{a:>18}" for a in summary["algos"])
    print(header)
    print("-" * len(header))
    for n in summary["batch_sizes"]:
        row = f"{n:>10} | "
        cells = []
        for algo in summary["algos"]:
            vals = [r["final_successes"] for r in summary["results"]
                    if r["algo"] == algo and r["num_envs"] == n]
            if vals:
                mean = sum(vals) / len(vals)
                cells.append(f"{mean:>18.3f}")
            else:
                cells.append(f"{'-':>18}")
        row += " | ".join(cells)
        print(row)
    print("=" * 72)
    print("Expected: PPO saturates at large batch sizes; SAPG keeps improving.")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    summary = run_sweep(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    summarize(summary)
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()

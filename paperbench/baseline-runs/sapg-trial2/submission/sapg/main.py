"""Entry point for training and evaluating SAPG and vanilla PPO.

Usage examples
--------------
Train SAPG on AllegroKuka regrasping with 4096 envs split into 8 blocks::

    python main.py --algo sapg --env allegro_kuka --task regrasping \
        --num_envs 4096 --num_blocks 8 --total_steps 100000000

Train vanilla PPO baseline::

    python main.py --algo ppo --env allegro_kuka --task regrasping \
        --num_envs 4096 --total_steps 100000000

Evaluate a checkpoint::

    python main.py --algo sapg --env shadow_hand --eval --checkpoint ckpt.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, Optional

import torch

# Allow running as `python main.py` from the sapg/ directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sapg.envs import make_env  # noqa: E402
from sapg.curriculum import (  # noqa: E402
    CurriculumConfig,
    SuccessToleranceCurriculum,
    compute_avg_successes_per_episode,
)
from sapg.policy import build_policy  # noqa: E402
from sapg.networks import build_critic  # noqa: E402
from sapg.rollout_buffer import RolloutBuffer  # noqa: E402
from sapg.sapg_algorithm import SAPG, SAPGConfig  # noqa: E402
from sapg.ppo_baseline import PPO, PPOConfig  # noqa: E402
from sapg.utils import (  # noqa: E402
    Logger,
    get_device,
    load_checkpoint,
    load_yaml,
    save_checkpoint,
    set_seed,
)


# ---------------------------------------------------------------------------
# Task-specific network presets (paper Sec. 4.4 / Addendum)
# ---------------------------------------------------------------------------
NETWORK_PRESETS: Dict[str, Dict[str, Any]] = {
    "allegro_kuka": dict(hidden_dims=(512, 256), recurrent=True, lstm_hidden=768),
    "shadow_hand": dict(hidden_dims=(512, 512, 256, 128), recurrent=False),
    "allegro_hand": dict(hidden_dims=(512, 256, 128), recurrent=False),
}

# Horizon / mini-epoch presets (paper Tables 2-4)
HORIZON_PRESETS: Dict[str, Dict[str, int]] = {
    "allegro_kuka": dict(horizon=16, num_mini_epochs=2),
    "shadow_hand": dict(horizon=8, num_mini_epochs=5),
    "allegro_hand": dict(horizon=8, num_mini_epochs=5),
}


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAPG: Split and Aggregate Policy Gradients")
    p.add_argument("--algo", choices=["sapg", "ppo"], default="sapg")
    p.add_argument("--env", choices=["allegro_kuka", "shadow_hand", "allegro_hand"],
                   default="allegro_kuka")
    p.add_argument("--task", type=str, default=None,
                   help="Task name (regrasping/throw/reorientation).")
    p.add_argument("--config", type=str, default=None,
                   help="Optional YAML config path overriding defaults.")

    # Parallelism / splitting
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--num_blocks", type=int, default=8,
                   help="Number of follower blocks (SAPG only).")
    p.add_argument("--aggregation", choices=["leader", "symmetric"], default="leader")

    # Training
    p.add_argument("--total_steps", type=int, default=100_000_000)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--num_mini_epochs", type=int, default=None)
    p.add_argument("--mini_batch_multiplier", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--value_loss_coef", type=float, default=1.0)
    p.add_argument("--bounds_loss_coef", type=float, default=0.001)
    p.add_argument("--entropy_coef", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--kl_threshold", type=float, default=0.016)
    p.add_argument("--no_lr_adapt", action="store_true")
    p.add_argument("--phi_dim", type=int, default=8)
    p.add_argument("--per_worker_sigma", action="store_true",
                   help="Entropy-exploration variant: per-block learnable sigma.")

    # Curriculum
    p.add_argument("--no_curriculum", action="store_true")
    p.add_argument("--curriculum_decay", type=float, default=0.9)
    p.add_argument("--curriculum_threshold", type=float, default=3.0)

    # Logging / checkpointing
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log_dir", type=str, default="runs")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--log_interval", type=int, default=1,
                   help="Log every N iterations.")
    p.add_argument("--save_interval", type=int, default=100,
                   help="Checkpoint every N iterations (0 disables).")
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="sapg")

    # Eval
    p.add_argument("--eval", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--eval_episodes", type=int, default=100)
    return p.parse_args(argv)


def build_env(args: argparse.Namespace, num_envs: Optional[int] = None):
    """Construct the vectorized environment for the requested task."""
    task = args.task
    if task is None:
        task = "regrasping" if args.env == "allegro_kuka" else "reorientation"
    kwargs: Dict[str, Any] = dict(
        task=task,
        num_envs=num_envs if num_envs is not None else args.num_envs,
        device=str(args.device),
    )
    return make_env(args.env, **kwargs)


def build_algorithm(args: argparse.Namespace, obs_dim: int, action_dim: int,
                    num_workers: int, device: torch.device):
    """Construct SAPG or PPO with task-appropriate networks."""
    preset = NETWORK_PRESETS[args.env]
    hp = HORIZON_PRESETS[args.env]
    horizon = args.horizon if args.horizon is not None else hp["horizon"]
    num_mini_epochs = (args.num_mini_epochs if args.num_mini_epochs is not None
                       else hp["num_mini_epochs"])

    policy = build_policy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_workers=num_workers,
        hidden_dims=preset["hidden_dims"],
        phi_dim=args.phi_dim,
        recurrent=preset["recurrent"],
        lstm_hidden=preset.get("lstm_hidden", 768),
        per_worker_sigma=args.per_worker_sigma,
    ).to(device)
    value_net = build_critic(
        obs_dim=obs_dim,
        num_workers=num_workers,
        hidden_dims=preset["hidden_dims"],
        phi_dim=args.phi_dim,
        recurrent=preset["recurrent"],
        lstm_hidden=preset.get("lstm_hidden", 768),
    ).to(device)

    if args.algo == "sapg":
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
            lr_adapt=not args.no_lr_adapt,
            horizon=horizon,
            num_mini_epochs=num_mini_epochs,
            mini_batch_multiplier=args.mini_batch_multiplier,
            aggregation=args.aggregation,
            device=str(device),
        )
        return SAPG(policy=policy, value_net=value_net, config=cfg), horizon, num_mini_epochs

    cfg = PPOConfig(
        num_envs=args.num_envs,
        horizon=horizon,
        gamma=args.gamma,
        tau=args.tau,
        clip_eps=args.clip_eps,
        value_loss_coef=args.value_loss_coef,
        bounds_loss_coef=args.bounds_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        lr=args.lr,
        kl_threshold=args.kl_threshold,
        lr_adapt=not args.no_lr_adapt,
        num_mini_epochs=num_mini_epochs,
        mini_batch_multiplier=args.mini_batch_multiplier,
        device=str(device),
    )
    return PPO(policy=policy, value_net=value_net, config=cfg), horizon, num_mini_epochs


def make_buffer(args: argparse.Namespace, horizon: int, obs_dim: int, action_dim: int,
                device: torch.device) -> RolloutBuffer:
    if args.algo == "sapg":
        return RolloutBuffer(
            num_blocks=args.num_blocks,
            horizon=horizon,
            num_envs_per_block=max(1, args.num_envs // args.num_blocks),
            obs_dim=obs_dim,
            action_dim=action_dim,
            gamma=args.gamma,
            tau=args.tau,
            device=str(device),
        )
    return RolloutBuffer(
        num_blocks=1,
        horizon=horizon,
        num_envs_per_block=args.num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        gamma=args.gamma,
        tau=args.tau,
        device=str(device),
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()

    env = build_env(args)
    obs_dim = env.observation_dim
    action_dim = env.num_actions

    num_workers = args.num_blocks if args.algo == "sapg" else 1
    algo, horizon, num_mini_epochs = build_algorithm(
        args, obs_dim, action_dim, num_workers, device)

    buffer = make_buffer(args, horizon, obs_dim, action_dim, device)

    curriculum = None
    if not args.no_curriculum and args.env == "allegro_kuka":
        curriculum = SuccessToleranceCurriculum(CurriculumConfig(
            decay=args.curriculum_decay,
            success_threshold=args.curriculum_threshold,
        ))

    run_name = args.run_name or f"{args.algo}_{args.env}_{args.task or 'default'}"
    logger = Logger(
        log_dir=os.path.join(args.log_dir, run_name),
        use_tensorboard=args.tensorboard,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=run_name,
    )

    # ---- Rollout state -----------------------------------------------------
    obs = env.reset()
    if args.algo == "sapg":
        # worker_ids: (num_blocks, num_envs_per_block) -> block index per env
        per_block = max(1, args.num_envs // args.num_blocks)
        worker_ids = torch.arange(args.num_blocks, device=device).repeat_interleave(per_block)
        worker_ids = worker_ids[:args.num_envs]
    else:
        worker_ids = torch.zeros(args.num_envs, dtype=torch.long, device=device)

    hidden = None
    total_steps = 0
    iteration = 0
    start_time = time.time()
    last_log: Dict[str, float] = {}

    while total_steps < args.total_steps:
        buffer.reset()
        ep_success_flags = []
        ep_rewards = []

        for _ in range(horizon):
            with torch.no_grad():
                actions, logprobs, hidden = algo.act(obs, worker_ids, hidden)
                values = algo.value(obs, worker_ids)

            next_obs, rewards, dones, info = env.step(actions)

            # Reshape flat (num_envs, ...) into (num_blocks, per_block, ...)
            if args.algo == "sapg":
                per_block = max(1, args.num_envs // args.num_blocks)
                def _reshape(x):
                    if x.dim() == 1:
                        return x.view(args.num_blocks, per_block)
                    return x.view(args.num_blocks, per_block, *x.shape[1:])
                b_obs = _reshape(obs)
                b_actions = _reshape(actions)
                b_logprobs = _reshape(logprobs)
                b_values = _reshape(values)
                b_rewards = _reshape(rewards)
                b_dones = _reshape(dones)
                b_wids = worker_ids.view(args.num_blocks, per_block)
            else:
                b_obs = obs.unsqueeze(0)
                b_actions = actions.unsqueeze(0)
                b_logprobs = logprobs.unsqueeze(0)
                b_values = values.unsqueeze(0)
                b_rewards = rewards.unsqueeze(0)
                b_dones = dones.unsqueeze(0)
                b_wids = worker_ids.unsqueeze(0)

            buffer.add(b_obs, b_actions, b_logprobs, b_values,
                       b_rewards, b_dones, b_wids)

            if "success" in info:
                ep_success_flags.append(info["success"].detach())
            ep_rewards.append(rewards.detach())

            obs = next_obs
            total_steps += args.num_envs

            # Reset recurrent hidden state for envs that finished.
            if hidden is not None and dones.any():
                done_mask = dones.bool()
                if isinstance(hidden, tuple):
                    h, c = hidden
                    h = h.clone()
                    c = c.clone()
                    h[:, done_mask] = 0.0
                    c[:, done_mask] = 0.0
                    hidden = (h, c)
                else:
                    hidden = hidden.clone()
                    hidden[:, done_mask] = 0.0

        # ---- Bootstrap values ---------------------------------------------
        with torch.no_grad():
            last_values = algo.value(obs, worker_ids)
            if args.algo == "sapg":
                per_block = max(1, args.num_envs // args.num_blocks)
                last_values = last_values.view(args.num_blocks, per_block)
            else:
                last_values = last_values.unsqueeze(0)

        buffer.compute_returns(last_values)
        stats = algo.update(buffer, num_mini_epochs=num_mini_epochs)

        # ---- Curriculum ----------------------------------------------------
        if curriculum is not None and ep_success_flags:
            flags = torch.stack(ep_success_flags, dim=1)  # (num_envs, horizon)
            lengths = torch.full((args.num_envs,), horizon, device=device)
            avg_succ = compute_avg_successes_per_episode(flags, lengths)
            new_tol = curriculum.update(avg_succ)
            curriculum.apply(env)
            stats["curriculum/tolerance"] = new_tol
            stats["curriculum/avg_successes"] = avg_succ

        # ---- Logging -------------------------------------------------------
        mean_reward = torch.stack(ep_rewards).mean().item() if ep_rewards else 0.0
        stats["rollout/mean_reward"] = mean_reward
        stats["rollout/total_steps"] = float(total_steps)
        stats["time/fps"] = total_steps / max(1e-6, time.time() - start_time)
        last_log = stats

        if iteration % args.log_interval == 0:
            logger.log(stats, step=total_steps)

        if args.save_interval and iteration % args.save_interval == 0 and iteration > 0:
            ckpt_path = os.path.join(args.log_dir, run_name, f"ckpt_{total_steps}.pt")
            save_checkpoint(ckpt_path, algo, step=total_steps,
                            extra={"args": vars(args)})

        iteration += 1

    # Final checkpoint
    final_path = os.path.join(args.log_dir, run_name, "ckpt_final.pt")
    save_checkpoint(final_path, algo, step=total_steps, extra={"args": vars(args)})
    logger.log(last_log, step=total_steps)
    logger.close()
    env.close()
    print(f"[main] Training finished: {total_steps} steps. Checkpoint: {final_path}")


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()

    env = build_env(args, num_envs=args.num_envs)
    obs_dim = env.observation_dim
    action_dim = env.num_actions
    num_workers = args.num_blocks if args.algo == "sapg" else 1

    algo, _, _ = build_algorithm(args, obs_dim, action_dim, num_workers, device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, algo, map_location=str(device))

    if args.algo == "sapg":
        per_block = max(1, args.num_envs // args.num_blocks)
        worker_ids = torch.arange(args.num_blocks, device=device).repeat_interleave(per_block)
        worker_ids = worker_ids[:args.num_envs]
    else:
        worker_ids = torch.zeros(args.num_envs, dtype=torch.long, device=device)

    obs = env.reset()
    hidden = None
    episode_returns = torch.zeros(args.num_envs, device=device)
    completed_returns = []
    completed_successes = []
    episode_success_count = torch.zeros(args.num_envs, device=device)

    steps = 0
    max_steps = args.eval_episodes * 200
    while len(completed_returns) < args.eval_episodes and steps < max_steps:
        with torch.no_grad():
            actions, _, hidden = algo.act(obs, worker_ids, hidden, deterministic=True)
        obs, rewards, dones, info = env.step(actions)
        episode_returns += rewards
        if "success" in info:
            episode_success_count += info["success"].float()
        steps += 1
        if dones.any():
            done_idx = dones.bool().nonzero(as_tuple=True)[0]
            for i in done_idx.tolist():
                completed_returns.append(episode_returns[i].item())
                completed_successes.append(episode_success_count[i].item())
                episode_returns[i] = 0.0
                episode_success_count[i] = 0.0

    mean_return = sum(completed_returns) / max(1, len(completed_returns))
    mean_success = sum(completed_successes) / max(1, len(completed_successes))
    print(f"[eval] episodes={len(completed_returns)} "
          f"mean_return={mean_return:.4f} mean_successes_per_episode={mean_success:.4f}")
    env.close()


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)

    # Merge YAML config if provided (CLI takes precedence for explicitly set args).
    if args.config:
        cfg = load_yaml(args.config)
        for k, v in cfg.items():
            if hasattr(args, k) and getattr(args, k) == parse_args([]).__dict__.get(k):
                setattr(args, k, v)

    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

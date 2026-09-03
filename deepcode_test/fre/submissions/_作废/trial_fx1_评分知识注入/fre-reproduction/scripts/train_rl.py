"""Phase 2: Train IQL with a frozen Functional Reward Encoding (FRE) encoder.

This script loads an offline dataset, builds/loads a FRE VAE checkpoint,
constructs an FRE-conditioned IQL agent, and runs reward-prior-conditioned
offline RL updates. The FRE encoder is kept frozen throughout training;
all gradient updates target only the Q, V, and policy networks.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from fre.agent import FREAgent
from fre.dataset import build_state_pool, load_offline_dataset, make_synthetic_dataset
from fre.fre_vae import FREVAE
from fre.reward_prior import RewardPrior, make_default_reward_prior
from fre.utils import (
    Timer,
    get_logger,
    resolve_device,
    save_checkpoint,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FRE-conditioned IQL offline RL (Phase 2)."
    )
    # Dataset
    parser.add_argument("--domain", type=str, default="antmaze",
                        choices=["antmaze", "kitchen", "walker", "cheetah"],
                        help="Benchmark domain.")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Optional D4RL dataset name override.")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Optional path to ExORL HDF5 dataset.")
    parser.add_argument("--state_pool_size", type=int, default=None,
                        help="Optional cap on the state pool used for reward/encoder sampling.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use a small synthetic dataset for smoke testing.")

    # Pretrained FRE encoder
    parser.add_argument("--vae_checkpoint", type=str, default=None,
                        help="Path to phase-1 FRE VAE checkpoint. If omitted, "
                             "a randomly initialized VAE is used (for debugging).")
    parser.add_argument("--vae_latent_dim", type=int, default=128)
    parser.add_argument("--vae_d_model", type=int, default=256)
    parser.add_argument("--vae_nhead", type=int, default=4)
    parser.add_argument("--vae_num_layers", type=int, default=4)
    parser.add_argument("--vae_reward_bins", type=int, default=64)

    # IQL / training
    parser.add_argument("--rl_steps", type=int, default=1000000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--expectile", type=float, default=0.9)
    parser.add_argument("--awr_temperature", type=float, default=3.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target_tau", type=float, default=0.005)
    parser.add_argument("--advantage_clip_min", type=float, default=-5.0)
    parser.add_argument("--advantage_clip_max", type=float, default=2.0)
    parser.add_argument("--encoder_states", type=int, default=32,
                        help="Number of state-reward context pairs per sampled reward function.")
    parser.add_argument("--q_hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--v_hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--policy_hidden", type=int, nargs="+", default=[256, 256])

    # Logging / checkpointing
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=50000)
    parser.add_argument("--log_file", type=str, default=None)

    return parser.parse_args()


def _as_tuple(value: Any) -> Tuple[int, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    return (int(value), int(value))


def load_fre_vae(
    checkpoint_path: Optional[str],
    state_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> FREVAE:
    """Instantiate a FREVAE and optionally load a phase-1 checkpoint."""
    vae = FREVAE(
        state_dim=state_dim,
        latent_dim=args.vae_latent_dim,
        d_model=args.vae_d_model,
        nhead=args.vae_nhead,
        num_layers=args.vae_num_layers,
        reward_bins=args.vae_reward_bins,
        device=device,
    ).to(device)

    if checkpoint_path is None:
        print("[train_rl] WARNING: no --vae_checkpoint provided; "
              "using randomly initialized FRE encoder (debug only).")
        return vae

    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"FRE VAE checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Support common checkpoint layouts saved by train_fre_encoder.py.
    state_dict = None
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "vae_state_dict"):
            if key in checkpoint:
                candidate = checkpoint[key]
                if isinstance(candidate, dict):
                    state_dict = candidate
                    break
        if state_dict is None and "vae" in checkpoint:
            candidate = checkpoint["vae"]
            if isinstance(candidate, dict):
                state_dict = candidate.get("state_dict", candidate)
    else:
        state_dict = checkpoint

    if state_dict is None:
        raise ValueError("Could not locate a FRE VAE state_dict in checkpoint.")

    # Strip a leading 'vae.' or 'model.' prefix if present.
    stripped = {}
    for k, v in state_dict.items():
        if k.startswith("vae."):
            k = k[len("vae."):]
        elif k.startswith("model."):
            k = k[len("model."):]
        stripped[k] = v

    missing, unexpected = vae.load_state_dict(stripped, strict=False)
    if missing:
        print(f"[train_rl] WARNING: missing keys in VAE checkpoint: {missing}")
    if unexpected:
        print(f"[train_rl] WARNING: unexpected keys in VAE checkpoint: {unexpected}")
    print(f"[train_rl] Loaded FRE VAE checkpoint from {checkpoint_path}")
    return vae


def build_agent(
    args: argparse.Namespace,
    dataset,
    state_pool: np.ndarray,
    vae: FREVAE,
    device: torch.device,
) -> FREAgent:
    """Construct the FRE-conditioned IQL agent with a frozen VAE."""
    state_dim = int(dataset.states.shape[1]) if hasattr(dataset, "states") else (
        int(np.asarray(state_pool).shape[1])
    )
    if hasattr(dataset, "actions"):
        action_dim = int(dataset.actions.shape[1])
    else:
        # Fall back to common domain dimensions.
        from fre.config import get_domain_dims
        action_dim = get_domain_dims(args.domain)[1]

    reward_prior = make_default_reward_prior(
        state_dim=state_dim,
        state_pool=state_pool,
        device=device,
        seed=args.seed,
    )

    q_hidden = _as_tuple(args.q_hidden)
    v_hidden = _as_tuple(args.v_hidden)
    policy_hidden = _as_tuple(args.policy_hidden)

    agent = FREAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=args.vae_latent_dim,
        vae=vae,
        reward_prior=reward_prior,
        state_pool=state_pool,
        dataset=dataset,
        encoder_states=args.encoder_states,
        freeze_vae=True,
        q_hidden=q_hidden,
        v_hidden=v_hidden,
        policy_hidden=policy_hidden,
        gamma=args.gamma,
        expectile=args.expectile,
        awr_temperature=args.awr_temperature,
        target_tau=args.target_tau,
        advantage_clip=(args.advantage_clip_min, args.advantage_clip_max),
        lr=args.lr,
        device=device,
    ).to(device)

    return agent


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    logger = get_logger("train_rl")
    logger.info(f"Using device: {device}")
    logger.info(f"Arguments: {json.dumps(vars(args), indent=2, default=str)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load dataset and state pool.
    if args.synthetic:
        dataset = make_synthetic_dataset(seed=args.seed)
    else:
        dataset = load_offline_dataset(
            domain=args.domain,
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
        )

    state_pool = build_state_pool(dataset, max_pool_size=args.state_pool_size)
    logger.info(f"Loaded offline dataset with {len(dataset.states)} transitions "
                f"and state pool of size {len(state_pool)}.")

    # 2) Load / build FRE VAE.
    state_dim = int(np.asarray(state_pool).shape[1])
    vae = load_fre_vae(args.vae_checkpoint, state_dim, args, device)
    vae.eval()
    # The agent freezes it explicitly, but keep it eval to be safe.

    # 3) Construct agent.
    agent = build_agent(args, dataset, state_pool, vae, device)
    agent.freeze_vae()
    logger.info("FRE encoder is frozen. Training only IQL Q/V/policy networks.")

    # 4) Save experiment metadata for reproducibility.
    save_json(vars(args), os.path.join(args.output_dir, "train_rl_args.json"))

    # 5) IQL training loop.
    timer = Timer()
    timer.reset()
    start_time = time.time()

    for step in range(1, args.rl_steps + 1):
        # agent.train_on_dataset samples a transition batch, samples reward
        # functions from the prior, encodes them, and updates IQL networks.
        loss_info = agent.train_on_dataset(batch_size=args.batch_size)

        if step % args.log_interval == 0 or step == 1:
            q_loss = loss_info.get("q_loss", float("nan"))
            v_loss = loss_info.get("v_loss", float("nan"))
            policy_loss = loss_info.get("policy_loss", float("nan"))
            total_loss = loss_info.get("total_loss", q_loss + v_loss + policy_loss)
            elapsed = timer.elapsed()
            logger.info(
                f"step {step}/{args.rl_steps} | "
                f"q_loss {q_loss:.4f} | v_loss {v_loss:.4f} | "
                f"policy_loss {policy_loss:.4f} | total {total_loss:.4f} | "
                f"elapsed {timer.elapsed_str()}"
            )

        if step % args.save_interval == 0 or step == args.rl_steps:
            checkpoint_path = os.path.join(args.output_dir, f"agent_step_{step}.pt")
            save_checkpoint(
                {
                    "agent_state_dict": agent.state_dict(),
                    "step": step,
                    "args": vars(args),
                    "loss_info": loss_info,
                },
                checkpoint_path,
            )
            logger.info(f"Saved agent checkpoint to {checkpoint_path}")

    final_path = os.path.join(args.output_dir, "agent_final.pt")
    save_checkpoint(
        {
            "agent_state_dict": agent.state_dict(),
            "step": args.rl_steps,
            "args": vars(args),
        },
        final_path,
    )
    logger.info(f"Training complete. Final checkpoint saved to {final_path}")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()

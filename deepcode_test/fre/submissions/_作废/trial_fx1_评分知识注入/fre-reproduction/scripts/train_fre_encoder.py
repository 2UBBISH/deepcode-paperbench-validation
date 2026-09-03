"""Phase 1: train the FRE variational autoencoder on random reward priors.

This script trains only the reward encoder/decoder (FREVAE).  No RL networks
are involved.  At every training step we sample a small batch of reward
functions from the uniform mixture prior, sample K encoder-context states and
K' decoder-context states from the offline state pool, evaluate the sampled
reward functions on those states, and minimise the VAE objective

    L = mean(||eta(s^d) - q_theta(s^d, z)||^2) + beta * KL(q_phi(z | C) || N(0, I))

After convergence the checkpoint is saved and can be loaded by
``scripts/train_rl.py`` with the VAE weights frozen.

Example
-------
    python scripts/train_fre_encoder.py --domain antmaze \
        --dataset_name antmaze-large-diverse-v2 --num_steps 200000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Optional

import numpy as np
import torch

from fre.dataset import build_state_pool, load_offline_dataset, make_synthetic_dataset
from fre.fre_vae import FREVAE
from fre.reward_prior import RewardPrior


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FRE reward encoder/decoder")
    parser.add_argument("--domain", type=str, default="antmaze",
                        choices=["antmaze", "kitchen", "walker", "cheetah", "synthetic"],
                        help="Dataset domain to use for the state pool.")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Optional dataset name (e.g. antmaze-large-diverse-v2).")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Optional HDF5 path for ExORL datasets.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use a small synthetic dataset instead of a real benchmark.")
    parser.add_argument("--max_pool_size", type=int, default=100000,
                        help="Maximum number of states retained in the sampling pool.")

    # VAE architecture
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--reward_bins", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--decoder_hidden", type=int, default=256,
                        help="Width of each hidden layer in the reward decoder.")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="KL divergence weight in the VAE objective.")

    # Reward prior
    parser.add_argument("--goal_epsilon", type=float, default=1.0)
    parser.add_argument("--p_mask", type=float, default=0.75,
                        help="Bernoulli mask probability for sparse linear rewards.")
    parser.add_argument("--mlp_hidden", type=int, default=256)

    # Training
    parser.add_argument("--num_steps", type=int, default=200000)
    parser.add_argument("--reward_fn_batch_size", type=int, default=8,
                        help="Number of reward functions per gradient step.")
    parser.add_argument("--encoder_states", type=int, default=32, dest="K",
                        help="Number of encoder context states (K).")
    parser.add_argument("--decoder_states", type=int, default=256, dest="Kp",
                        help="Number of decoder context states (K').")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    # Logging/checkpointing
    parser.add_argument("--output_dir", type=str, default="outputs/fre_encoder")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--resume", type=str, default=None)

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_indices(n: int, pool_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, pool_size, (n,), device=device)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    # Load (or synthesize) an offline dataset for the state pool.
    if args.synthetic:
        dataset = make_synthetic_dataset(state_dim=17, action_dim=8, size=10000,
                                         seed=args.seed)
    else:
        dataset = load_offline_dataset(
            args.domain,
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
        )

    state_dim = int(dataset.states.shape[-1])
    state_pool = build_state_pool(dataset, max_pool_size=args.max_pool_size)
    state_pool = np.asarray(state_pool, dtype=np.float32)
    if state_pool.ndim != 2:
        raise ValueError(f"State pool must be 2D, got shape {state_pool.shape}")
    if len(state_pool) < max(args.K, args.Kp):
        raise ValueError(
            f"State pool has only {len(state_pool)} states; need at least "
            f"max(K, K') = {max(args.K, args.Kp)}. Increase the dataset size "
            f"or reduce --max_pool_size."
        )

    reward_prior = RewardPrior(
        state_dim=state_dim,
        state_pool=state_pool,
        goal_epsilon=args.goal_epsilon,
        p_mask=args.p_mask,
        mlp_hidden=args.mlp_hidden,
        device=device,
        seed=args.seed,
    )

    vae = FREVAE(
        state_dim=state_dim,
        latent_dim=args.latent_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        reward_bins=args.reward_bins,
        embedding_dim=args.embedding_dim,
        decoder_hidden=(args.decoder_hidden, args.decoder_hidden),
        beta=args.beta,
        device=device,
    ).to(device)

    optimizer = vae.configure_optimizer(lr=args.lr)

    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        vae.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", optimizer.state_dict()))
        start_step = checkpoint.get("step", 0)
        print(f"Resumed from step {start_step} ({args.resume})")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    state_pool_tensor = torch.as_tensor(state_pool, dtype=torch.float32, device=device)
    pool_size = state_pool_tensor.shape[0]

    def draw_states(n: int) -> torch.Tensor:
        idx = sample_indices(n, pool_size, device)
        return state_pool_tensor[idx]

    vae.train()
    running_recon = 0.0
    running_kl = 0.0
    running_loss = 0.0
    start_time = time.time()

    for step in range(start_step + 1, args.num_steps + 1):
        reward_fns = reward_prior.sample_reward_fns(args.reward_fn_batch_size)

        enc_states_list: list[torch.Tensor] = []
        enc_rewards_list: list[torch.Tensor] = []
        dec_states_list: list[torch.Tensor] = []
        dec_rewards_list: list[torch.Tensor] = []

        for rf in reward_fns:
            enc_states = draw_states(args.K)
            dec_states = draw_states(args.Kp)

            enc_rewards = rf(enc_states)
            dec_rewards = rf(dec_states)

            if isinstance(enc_rewards, np.ndarray):
                enc_rewards = torch.as_tensor(enc_rewards, dtype=torch.float32, device=device)
            if isinstance(dec_rewards, np.ndarray):
                dec_rewards = torch.as_tensor(dec_rewards, dtype=torch.float32, device=device)
            if not isinstance(enc_rewards, torch.Tensor):
                enc_rewards = torch.as_tensor(enc_rewards, dtype=torch.float32, device=device)
            if not isinstance(dec_rewards, torch.Tensor):
                dec_rewards = torch.as_tensor(dec_rewards, dtype=torch.float32, device=device)

            enc_rewards = enc_rewards.reshape(-1).to(dtype=torch.float32, device=device)
            dec_rewards = dec_rewards.reshape(-1).to(dtype=torch.float32, device=device)
            if enc_rewards.numel() != args.K or dec_rewards.numel() != args.Kp:
                raise ValueError(
                    f"Reward function returned unexpected shape: enc={enc_rewards.shape}, "
                    f"dec={dec_rewards.shape}"
                )

            enc_states_list.append(enc_states)
            enc_rewards_list.append(enc_rewards)
            dec_states_list.append(dec_states)
            dec_rewards_list.append(dec_rewards)

        encoder_states = torch.stack(enc_states_list)   # [B, K, state_dim]
        encoder_rewards = torch.stack(enc_rewards_list) # [B, K]
        decoder_states = torch.stack(dec_states_list)   # [B, K', state_dim]
        decoder_rewards = torch.stack(dec_rewards_list) # [B, K']

        outputs = vae.forward(
            encoder_states,
            encoder_rewards,
            decoder_states,
            decoder_rewards=decoder_rewards,
        )

        # FREVAE returns a dict; support a few plausible key names.
        if "loss" in outputs:
            loss = outputs["loss"]
        elif "total_loss" in outputs:
            loss = outputs["total_loss"]
        else:
            recon = outputs.get("recon_loss", outputs.get("reconstruction_loss"))
            kl = outputs.get("kl_loss", outputs.get("kl_divergence"))
            if recon is None or kl is None:
                raise KeyError(f"Cannot find VAE loss in outputs: {list(outputs.keys())}")
            loss = recon + args.beta * kl

        loss = loss.mean()
        recon_val = float(outputs.get("recon_loss", outputs.get("reconstruction_loss", 0.0)).mean())
        kl_val = float(outputs.get("kl_loss", outputs.get("kl_divergence", 0.0)).mean())

        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(vae.parameters(), args.clip_grad_norm)
        optimizer.step()

        running_recon += recon_val
        running_kl += kl_val
        running_loss += float(loss.detach().cpu())

        if step % args.log_every == 0:
            n = args.log_every
            elapsed = time.time() - start_time
            print(
                f"step {step}/{args.num_steps} | loss {running_loss / n:.5f} | "
                f"recon {running_recon / n:.5f} | kl {running_kl / n:.5f} | "
                f"{elapsed:.1f}s"
            )
            running_recon = 0.0
            running_kl = 0.0
            running_loss = 0.0
            start_time = time.time()

        if step % args.save_every == 0 or step == args.num_steps:
            checkpoint_path = os.path.join(args.output_dir, f"vae_step{step}.pt")
            torch.save(
                {
                    "model_state_dict": vae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "state_dim": state_dim,
                    "latent_dim": args.latent_dim,
                    "d_model": args.d_model,
                    "nhead": args.nhead,
                    "num_layers": args.num_layers,
                    "reward_bins": args.reward_bins,
                    "embedding_dim": args.embedding_dim,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint to {checkpoint_path}")

    final_path = os.path.join(args.output_dir, "vae_final.pt")
    torch.save(
        {
            "model_state_dict": vae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": args.num_steps,
            "state_dim": state_dim,
            "latent_dim": args.latent_dim,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "reward_bins": args.reward_bins,
            "embedding_dim": args.embedding_dim,
        },
        final_path,
    )
    print(f"Training complete. Final VAE checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()

"""Phase 1: pretrain the FRE variational autoencoder.

This module trains the FRE encoder/decoder on random reward functions drawn
from the prior distribution. During this phase no RL networks are updated.
The resulting encoder is saved and later frozen for FRE-conditioned IQL.

The default objective is::

    L_FRE = reconstruction_mse - beta * KL(q(z | c) || N(0, I))

where ``c`` is a small context of labelled encoder states and reconstruction
targets are produced on an independently sampled decoder minibatch.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

from fre.config import Config, get_config, resolve_device
from fre.data.dataset import OfflineDataset
from fre.data.reward_sampler import RewardFunction, sample_reward
from fre.modeling.fre_vae import FREVAE

logger = logging.getLogger(__name__)


def _get(obj: Any, name: str, default: Any) -> Any:
    """Return an attribute if present, otherwise a default."""
    try:
        value = getattr(obj, name)
    except Exception:
        return default
    return default if value is None else value


def _as_tensor(x: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Coerce an array-like object to a torch tensor on ``device``."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


def _sample_reward_batch(
    dataset: OfflineDataset,
    reward_cfg: Any,
    num_rewards: int,
    num_encoder_states: int,
    num_decoder_states: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Sample a batch of random reward functions and label context/decoder states.

    Returns:
        Dict with ``encoder_states``, ``encoder_rewards``, ``decoder_states``,
        ``decoder_rewards``. Shapes are ``[num_rewards, K, state_dim]`` and
        ``[num_rewards, K, 1]`` (or ``[num_rewards, K]`` for rewards).
    """
    state_dim = dataset.states.shape[-1]

    enc_states = _as_tensor(dataset.sample_states(num_rewards * num_encoder_states), device)
    dec_states = _as_tensor(dataset.sample_states(num_rewards * num_decoder_states), device)
    enc_states = enc_states.view(num_rewards, num_encoder_states, state_dim)
    dec_states = dec_states.view(num_rewards, num_decoder_states, state_dim)

    enc_rewards = torch.empty(num_rewards, num_encoder_states, device=device, dtype=torch.float32)
    dec_rewards = torch.empty(num_rewards, num_decoder_states, device=device, dtype=torch.float32)

    for i in range(num_rewards):
        # Each reward function is sampled independently. We pass the encoder
        # minibatch as the state pool so singleton goals are sampled uniformly
        # from the same states that will be labelled (matching the prior).
        reward_fn: RewardFunction = sample_reward(enc_states[i], reward_cfg)
        enc_rewards[i] = _as_tensor(reward_fn(enc_states[i]), device).reshape(-1)
        dec_rewards[i] = _as_tensor(reward_fn(dec_states[i]), device).reshape(-1)

    return {
        "encoder_states": enc_states,
        "encoder_rewards": enc_rewards,
        "decoder_states": dec_states,
        "decoder_rewards": dec_rewards,
    }


def _load_dataset(cfg: Config, device: torch.device) -> OfflineDataset:
    """Load the dataset specified by ``cfg`` when none is supplied by the caller."""
    domain = str(_get(cfg, "domain", "")).lower()
    data_cfg = getattr(cfg, "data", None)

    if domain in {"antmaze", "ant"}:
        from fre.data.d4rl_loader import load_antmaze_dataset

        return load_antmaze_dataset(data_cfg, device=device)
    if domain in {"kitchen", "kitchen-complete"}:
        from fre.data.d4rl_loader import load_kitchen_dataset

        return load_kitchen_dataset(data_cfg, device=device)
    if domain in {"exorl", "walker", "cheetah", "dmc"}:
        from fre.data.exorl_loader import load_exorl_dataset

        return load_exorl_dataset(data_cfg, device=device)

    # Generic fallback: try D4RL first, then ExORL.
    try:
        from fre.data.d4rl_loader import load_d4rl_dataset

        return load_d4rl_dataset(data_cfg, device=device)
    except Exception:
        from fre.data.exorl_loader import load_exorl_dataset

        return load_exorl_dataset(data_cfg, device=device)


def pretrain_encoder(
    cfg: Config,
    dataset: Optional[OfflineDataset] = None,
    model: Optional[FREVAE] = None,
    device: Optional[str] = None,
    num_steps: Optional[int] = None,
    log_every: Optional[int] = None,
    checkpoint_every: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    seed: Optional[int] = None,
) -> FREVAE:
    """Pretrain a FRE VAE on random reward functions.

    Args:
        cfg: Full experiment configuration.
        dataset: Optional offline dataset. If ``None``, it is loaded from
            ``cfg.domain``.
        model: Optional FRE VAE. If ``None``, one is constructed from
            ``cfg.fre`` using ``FREVAE.from_config``.
        device: Torch device string. Falls back to ``cfg.device`` and then
            ``resolve_device``.
        num_steps: Number of optimizer steps. Falls back to ``cfg.fre``.
        log_every: Logging interval. Falls back to ``cfg.fre``.
        checkpoint_every: Checkpoint interval. Falls back to ``cfg.fre``.
        checkpoint_dir: Checkpoint directory override.
        log_dir: Logging directory override.
        seed: Random seed override.

    Returns:
        The trained ``FREVAE`` on the target device.
    """
    # ------------------------------------------------------------------
    # Configuration and device setup
    # ------------------------------------------------------------------
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device_str = device or _get(cfg, "device", "auto")
    torch_device = torch.device(resolve_device(device_str))

    fre_cfg = getattr(cfg, "fre", None)
    reward_cfg = getattr(cfg, "reward_sampler", None)

    if dataset is None:
        dataset = _load_dataset(cfg, torch_device)

    state_dim = int(dataset.states.shape[-1])
    if model is None:
        model = FREVAE.from_config(fre_cfg, state_dim=state_dim)
    model = model.to(torch_device)
    model.train()

    num_encoder_states = int(_get(reward_cfg, "num_encoder_states", 32))
    num_decoder_states = int(_get(reward_cfg, "num_decoder_states", 1024))
    num_reward_functions = int(_get(fre_cfg, "batch_size", 64))

    lr = float(_get(fre_cfg, "lr", 1e-4))
    weight_decay = float(_get(fre_cfg, "weight_decay", 0.0))
    grad_clip_norm = float(_get(fre_cfg, "grad_clip_norm", 10.0))
    num_steps = int(num_steps or _get(fre_cfg, "pretrain_steps", _get(fre_cfg, "num_steps", 200_000)))
    log_every = int(log_every or _get(fre_cfg, "log_every", 1000))
    checkpoint_every = int(checkpoint_every or _get(fre_cfg, "checkpoint_every", 25_000))

    checkpoint_dir = checkpoint_dir or _get(cfg, "checkpoint_dir", "checkpoints")
    log_dir = log_dir or _get(cfg, "log_dir", "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------------
    # Optional logging setup (no hard dependency on wandb/tensorboard)
    # ------------------------------------------------------------------
    use_wandb = bool(_get(cfg, "use_wandb", False))
    wandb_run = None
    if use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=_get(cfg, "wandb_project", "fre"),
                name=f"pretrain_{cfg.name}_{cfg.seed}",
                dir=log_dir,
                config=cfg.to_dict() if hasattr(cfg, "to_dict") else {},
            )
        except Exception as exc:  # pragma: no cover - wandb is optional
            logger.warning("wandb requested but could not be initialized: %s", exc)
            wandb_run = None

    metrics_history: Dict[str, list] = {
        "loss": [],
        "reconstruction_mse": [],
        "kl": [],
        "lr": [],
    }

    logger.info(
        "Starting FRE pretraining: steps=%d, state_dim=%d, K=%d, K'=%d, lr=%g, device=%s",
        num_steps,
        state_dim,
        num_encoder_states,
        num_decoder_states,
        lr,
        torch_device,
    )

    start_time = time.time()
    best_loss = float("inf")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for step in range(1, num_steps + 1):
        batch = _sample_reward_batch(
            dataset=dataset,
            reward_cfg=reward_cfg,
            num_rewards=num_reward_functions,
            num_encoder_states=num_encoder_states,
            num_decoder_states=num_decoder_states,
            device=torch_device,
        )

        optimizer.zero_grad(set_to_none=True)
        output = model(
            encoder_states=batch["encoder_states"],
            encoder_rewards=batch["encoder_rewards"],
            decoder_states=batch["decoder_states"],
            decoder_rewards=batch["decoder_rewards"],
        )
        loss = output.loss
        if not torch.isfinite(loss):
            logger.error("Non-finite loss at step %d; skipping update.", step)
            continue

        loss.backward()
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        loss_val = float(loss.detach().cpu().item())
        recon_val = float(output.reconstruction_mse.detach().cpu().item())
        kl_val = float(output.kl.detach().cpu().item())

        metrics_history["loss"].append(loss_val)
        metrics_history["reconstruction_mse"].append(recon_val)
        metrics_history["kl"].append(kl_val)
        metrics_history["lr"].append(lr)

        if step % log_every == 0 or step == 1:
            elapsed = time.time() - start_time
            steps_per_sec = step / max(elapsed, 1e-8)
            logger.info(
                "step %d/%d | loss %.5f | recon_mse %.5f | kl %.5f | %.1f steps/s",
                step,
                num_steps,
                loss_val,
                recon_val,
                kl_val,
                steps_per_sec,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "pretrain/loss": loss_val,
                        "pretrain/reconstruction_mse": recon_val,
                        "pretrain/kl": kl_val,
                        "pretrain/lr": lr,
                        "pretrain/step": step,
                    }
                )

        if step % checkpoint_every == 0 or step == num_steps:
            checkpoint_path = os.path.join(checkpoint_dir, f"fre_vae_step_{step}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "cfg_name": _get(cfg, "name", "default"),
                    "state_dim": state_dim,
                    "loss": loss_val,
                    "reconstruction_mse": recon_val,
                    "kl": kl_val,
                },
                checkpoint_path,
            )
            if loss_val < best_loss:
                best_loss = loss_val
                best_path = os.path.join(checkpoint_dir, "fre_vae_best.pt")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "step": step,
                        "cfg_name": _get(cfg, "name", "default"),
                        "state_dim": state_dim,
                    },
                    best_path,
                )
            logger.info("Saved checkpoint to %s", checkpoint_path)

    # Always save a final checkpoint.
    final_path = os.path.join(checkpoint_dir, "fre_vae_final.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": num_steps,
            "cfg_name": _get(cfg, "name", "default"),
            "state_dim": state_dim,
            "loss": metrics_history["loss"][-1] if metrics_history["loss"] else float("nan"),
        },
        final_path,
    )
    if wandb_run is not None:
        wandb_run.finish()

    logger.info("FRE pretraining complete. Final checkpoint: %s", final_path)
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain the FRE VAE encoder")
    parser.add_argument("--config", type=str, default="default", help="Config name or YAML path")
    parser.add_argument("--name", type=str, default=None, help="Override experiment name")
    parser.add_argument("--device", type=str, default=None, help="Torch device")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--steps", type=int, default=None, help="Number of pretraining steps")
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    return parser


def main(argv: Optional[list] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_parser().parse_args(argv)

    cfg = get_config(args.config)
    if args.name is not None:
        cfg.name = args.name
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device

    pretrain_encoder(
        cfg,
        device=args.device,
        num_steps=args.steps,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

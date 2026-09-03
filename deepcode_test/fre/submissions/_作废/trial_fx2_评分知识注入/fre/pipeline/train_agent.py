"""Phase 2 strided training pipeline for FRE-conditioned offline RL.

This module loads a pretrained FRE variational autoencoder, freezes all of its
parameters so the latent reward representation ``z`` stays stationary during TD
learning, builds the conditional Implicit Q-Learning (IQL) agent, and trains it
against reward functions sampled from the prior reward-function mixture.

The training procedure is "strided" in the sense of the paper: the FRE encoder
is trained first (Phase 1, see :mod:`fre.pipeline.pretrain_encoder`) and is then
frozen while the Q/value/policy networks are trained in Phase 2.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from fre.config import Config, get_config, resolve_device
from fre.data.dataset import OfflineDataset
from fre.data.d4rl_loader import (
    load_antmaze_dataset,
    load_d4rl_dataset,
    load_kitchen_dataset,
)
from fre.data.exorl_loader import (
    load_cheetah_dataset,
    load_exorl_dataset,
    load_walker_dataset,
)
from fre.modeling.fre_vae import FREVAE
from fre.rl.iql import IQL, ImplicitQLearning
from fre.rl.rl_trainer import FREIQLTrainer, train_fre_iql_agent

LOGGER = logging.getLogger(__name__)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Return ``obj.key`` if it exists, otherwise ``default``."""
    if obj is None:
        return default
    return getattr(obj, key, default)


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_dataset(cfg: Config, device: str = "cpu") -> OfflineDataset:
    """Load the offline dataset appropriate for the configured domain.

    The function is intentionally defensive: it tries the dedicated domain
    loaders first, then falls back to a generic D4RL loader.
    """
    data_cfg = _get(cfg, "data")
    domain = str(_get(cfg, "domain", "")).lower()
    env_name = _get(data_cfg, "env_name") or _get(data_cfg, "dataset_name")

    try:
        if domain in ("antmaze", "antmaze-large", "antmaze-medium", "antmaze-umaze"):
            return load_antmaze_dataset(data_cfg, device=device)
        if domain in ("kitchen", "kitchen-complete"):
            return load_kitchen_dataset(data_cfg, device=device)
        if domain in ("walker", "walker-walk", "exorl-walker"):
            return load_walker_dataset(data_cfg, device=device)
        if domain in ("cheetah", "cheetah-run", "exorl-cheetah"):
            return load_cheetah_dataset(data_cfg, device=device)
        if domain in ("exorl",):
            return load_exorl_dataset(data_cfg, env_name=env_name, device=device)
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.warning("Dedicated loader for domain %r failed (%s); falling back", domain, exc)

    # Generic D4RL fallback for MuJoCo locomotion or custom D4RL datasets.
    return load_d4rl_dataset(data_cfg, env_name=env_name, device=device)


def _infer_model_checkpoint(cfg: Config) -> Optional[str]:
    """Infer a pretrained FRE checkpoint path from config fields."""
    fre_cfg = _get(cfg, "fre")
    candidates = [
        _get(fre_cfg, "checkpoint_path"),
        _get(fre_cfg, "pretrained_path"),
        _get(fre_cfg, "model_path"),
        _get(cfg, "model_path"),
        _get(cfg, "pretrain_checkpoint"),
    ]
    for path in candidates:
        if path and os.path.isfile(str(path)):
            return str(path)

    checkpoint_dir = _get(cfg, "checkpoint_dir") or _get(fre_cfg, "checkpoint_dir")
    if checkpoint_dir:
        checkpoint_dir = str(checkpoint_dir)
        for name in ("best.pt", "final.pt", "fre_vae.pt", "model.pt", "encoder.pt"):
            candidate = os.path.join(checkpoint_dir, name)
            if os.path.isfile(candidate):
                return candidate
    return None


def _load_model_checkpoint(model: FREVAE, path: str, device: str) -> FREVAE:
    """Load a FRE VAE checkpoint while tolerating several checkpoint layouts."""
    checkpoint = torch.load(path, map_location=device)
    state_dict = None

    if isinstance(checkpoint, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
            "encoder_state_dict",
            "vae_state_dict",
            "fre_state_dict",
        ):
            if key in checkpoint:
                candidate = checkpoint[key]
                if isinstance(candidate, dict) and "state_dict" in candidate:
                    candidate = candidate["state_dict"]
                if isinstance(candidate, dict):
                    state_dict = candidate
                    break
        # If the checkpoint itself looks like a state dict, use it directly.
        if state_dict is None and any(
            isinstance(k, str) and k.startswith(("state_embed", "reward_embed", "encoder"))
            for k in checkpoint.keys()
        ):
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if state_dict is None:
        raise RuntimeError(f"Could not extract a state dict from checkpoint: {path}")

    model.load_state_dict(state_dict)
    LOGGER.info("Loaded FRE checkpoint from %s", path)
    return model


def _freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    """Place model in eval mode and freeze all trainable parameters."""
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _build_agent(
    cfg: Config,
    state_dim: int,
    action_dim: int,
    condition_dim: int,
    device: str,
) -> ImplicitQLearning:
    """Build a conditional IQL agent from configuration fields."""
    iql_cfg = _get(cfg, "iql")
    return ImplicitQLearning(
        state_dim=state_dim,
        action_dim=action_dim,
        condition_dim=condition_dim,
        cfg=iql_cfg,
        gamma=_get(iql_cfg, "gamma", 0.99),
        tau=_get(iql_cfg, "tau", 0.7),
        beta=_get(iql_cfg, "beta", 3.0),
        lr=_get(iql_cfg, "lr", 3e-4),
        target_tau=_get(iql_cfg, "target_tau", 0.005),
        hidden_dim=_get(iql_cfg, "hidden_dim", 256),
        num_hidden=_get(iql_cfg, "num_hidden", 2),
        device=device,
    )


def train_agent(
    cfg: Optional[Config] = None,
    dataset: Optional[OfflineDataset] = None,
    model: Optional[FREVAE] = None,
    agent: Optional[ImplicitQLearning] = None,
    device: Optional[str] = None,
    num_steps: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    log_every: Optional[int] = None,
    checkpoint_every: Optional[int] = None,
    seed: int = 0,
    model_path: Optional[str] = None,
) -> Tuple[ImplicitQLearning, Dict[str, Any]]:
    """Run the strided Phase 2 training.

    Args:
        cfg: Experiment configuration. If ``None``, :func:`Config.default` is used.
        dataset: Preloaded offline dataset. Loaded from ``cfg`` when ``None``.
        model: Pretrained FRE VAE. Built and loaded from a checkpoint when ``None``.
        agent: Conditional IQL agent. Constructed from ``cfg`` when ``None``.
        device: Torch device string (``"auto"`` resolves CUDA/CPU).
        num_steps: Number of RL gradient steps. Defaults to the config value.
        checkpoint_dir: Directory for agent checkpoints.
        log_every: Logging interval in gradient steps.
        checkpoint_every: Checkpoint interval in gradient steps.
        seed: Random seed.
        model_path: Explicit FRE checkpoint path.

    Returns:
        Tuple ``(trained_agent, metrics_dict)``.
    """
    if cfg is None:
        cfg = Config.default()

    _seed_everything(seed)
    device = resolve_device(device or _get(cfg, "device", "auto"))

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    if dataset is None:
        dataset = _load_dataset(cfg, device=device)
    dataset = dataset.to(device)

    state_dim = int(dataset.states.shape[1])
    action_dim = int(dataset.actions.shape[1])

    # ------------------------------------------------------------------
    # Pretrained FRE model (frozen in Phase 2)
    # ------------------------------------------------------------------
    if model is None:
        fre_cfg = _get(cfg, "fre")
        model = FREVAE.from_config(fre_cfg, state_dim=state_dim)
    model = model.to(device)

    checkpoint_path = model_path or _infer_model_checkpoint(cfg)
    if checkpoint_path:
        _load_model_checkpoint(model, checkpoint_path, device)
    else:
        LOGGER.warning(
            "No FRE checkpoint path found in config or arguments. "
            "Using an untrained encoder; this is only suitable for a quick sanity run."
        )

    model = _freeze_model(model)

    # ------------------------------------------------------------------
    # Conditional IQL agent
    # ------------------------------------------------------------------
    z_dim = int(_get(model, "z_dim", _get(_get(cfg, "fre"), "z_dim", 64)))
    if agent is None:
        agent = _build_agent(cfg, state_dim, action_dim, z_dim, device)

    iql_cfg = _get(cfg, "iql")
    if num_steps is None:
        num_steps = _get(iql_cfg, "num_steps", _get(cfg, "num_steps", 500_000))
    if checkpoint_dir is None:
        checkpoint_dir = _get(cfg, "checkpoint_dir") or _get(iql_cfg, "checkpoint_dir") or "checkpoints"
    if log_every is None:
        log_every = _get(iql_cfg, "log_every", _get(cfg, "log_every", 1000))
    if checkpoint_every is None:
        checkpoint_every = _get(iql_cfg, "checkpoint_every", _get(cfg, "checkpoint_every", 25_000))

    LOGGER.info(
        "Starting strided FRE-IQL training: domain=%s, state_dim=%d, action_dim=%d, "
        "z_dim=%d, num_steps=%s",
        _get(cfg, "domain", "unknown"),
        state_dim,
        action_dim,
        z_dim,
        num_steps,
    )
    start_time = time.time()
    metrics = train_fre_iql_agent(
        cfg=cfg,
        dataset=dataset,
        model=model,
        agent=agent,
        device=device,
        num_steps=num_steps,
        checkpoint_dir=str(checkpoint_dir),
        log_every=log_every,
        checkpoint_every=checkpoint_every,
        seed=seed,
    )
    elapsed = time.time() - start_time
    LOGGER.info("Strided FRE-IQL training finished in %.1f seconds", elapsed)

    metrics = dict(metrics or {})
    metrics["train_seconds"] = elapsed
    return agent, metrics


def _parse_overrides(overrides) -> Dict[str, Any]:
    """Parse ``--override key=value`` pairs, converting primitive types."""
    result: Dict[str, Any] = {}
    if not overrides:
        return result
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        value = value.strip()
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        elif value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass
        result[key.strip()] = value
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for :func:`main`."""
    parser = argparse.ArgumentParser(
        description="Strided Phase 2 training: freeze FRE encoder, train conditional IQL."
    )
    parser.add_argument("--config", type=str, default=None, help="Config name or YAML path.")
    parser.add_argument("--name", type=str, default=None, help="Experiment name.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Torch device (auto/cpu/cuda).")
    parser.add_argument("--num-steps", type=int, default=None, help="Number of RL steps.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Agent checkpoint dir.")
    parser.add_argument("--model-path", type=str, default=None, help="FRE checkpoint path.")
    parser.add_argument("--log-every", type=int, default=None, help="Logging interval.")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Checkpoint interval.")
    parser.add_argument("--override", action="append", default=None, help="Config override key=value.")
    return parser


def main(argv: Optional[list] = None) -> None:
    """CLI entry point for strided training."""
    parser = build_parser()
    args = parser.parse_args(argv)

    overrides = _parse_overrides(args.override)
    config_name = args.config or "default"
    cfg = get_config(config_name, **overrides)

    if args.name is not None:
        cfg.name = args.name
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    train_agent(
        cfg=cfg,
        device=args.device,
        num_steps=args.num_steps,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        seed=cfg.seed,
        model_path=args.model_path,
    )


__all__ = ["train_agent", "main", "build_parser", "FREIQLTrainer", "IQL", "ImplicitQLearning"]


if __name__ == "__main__":
    main()

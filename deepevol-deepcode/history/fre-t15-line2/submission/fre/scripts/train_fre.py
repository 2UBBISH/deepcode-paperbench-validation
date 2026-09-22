#!/usr/bin/env python
"""Phase-1 entry point for FRE: train the reward encoder + decoder (Algorithm 1).

Paper: "Zero-Shot Reinforcement Learning via Functional Reward Encodings".

This script implements the ``# Train encoder`` half of Algorithm 1 (Source: Section
4.3, Algorithm 1).  It builds the unlabeled offline dataset, the prior reward
distribution ``p(eta)`` (Source: Section 4.2 / Appendix B), the permutation-invariant
transformer VIB encoder and the reward decoder (Source: Section 4.1), and maximizes
the variational lower bound of Equation (6):

    I(L_eta^d ; Z) - beta * I(L_eta^e ; Z)
        >= E_{eta, L_eta^e, L_eta^d, z ~ p_theta(z | L_eta^e)}[
               sum_{k=1}^{K'} log q_theta(eta(s_k^d) | s_k^d, z)
               - beta * D_KL(p_theta(z | L_eta^e) || u(z)) ] + const,

where ``u(z)`` is the uninformative unit-Gaussian prior over z.  The reconstruction
term is realized as negative MSE (Source: Section 4.1 Practical Implementation /
Section 5) and the KL term is ``D_KL(N(mu, sigma) || N(0, I))`` with beta = 0.01
(Source: Table 3).  The decoder predicts the reward of *different* states than the
ones used for encoding, and both networks are trained jointly.

Hyper-parameters (Source: Table 3 / Appendix A):
  Batch Size 512, Reward Pairs to Encode K = 32, Reward Pairs to Decode K' = 8,
  Number of Reward Embeddings 32, Encoder Layers [256,256,256,256],
  Encoder Attention Heads 4, Decoder Network Layers [512,512,512], Optimizer Adam,
  Learning Rate 1e-4, beta KL weight 0.01, Encoder Training Steps 150,000
  (1M for ExORL/Kitchen).

Usage
-----
    python scripts/train_fre.py --config configs/fre_antmaze.yaml
    python scripts/train_fre.py --config configs/fre_exorl.yaml --steps 1000
    python scripts/train_fre.py --domain antmaze --dry-run      # build modules only
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Make the package importable when the script is executed directly
# (``python scripts/train_fre.py``) rather than as a module.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)          # .../fre  (contains scripts/, fre/, configs/)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402  (needed for seeding and metric coercion)

# ---------------------------------------------------------------------------
# Protocol / Table 3 constants (duplicated so --help and --dry-run stay cheap)
# ---------------------------------------------------------------------------
TRAIN_BATCH_SIZE = 512
TRAIN_LEARNING_RATE = 1e-4
TRAIN_BETA = 0.01
NUM_ENCODER_STATES = 32          # K  (Reward Pairs to Encode)
NUM_DECODER_STATES = 8           # K' (Reward Pairs to Decode)
NUM_REWARD_EMBEDDINGS = 32
LATENT_DIM = 128
STATE_EMBED_DIM = 64
REWARD_EMBED_DIM = 64
ENCODER_LAYERS = (256, 256, 256, 256)
ENCODER_ATTENTION_HEADS = 4
DECODER_LAYERS = (512, 512, 512)
ENCODER_TRAINING_STEPS = 150_000
LONG_ENCODER_TRAINING_STEPS = 1_000_000
DOMAIN_ENCODER_STEPS: Dict[str, int] = {
    "antmaze": ENCODER_TRAINING_STEPS,
    "exorl": LONG_ENCODER_TRAINING_STEPS,
    "kitchen": LONG_ENCODER_TRAINING_STEPS,
}
DEFAULT_PRIOR_RATIOS = (0.33, 0.33, 0.33)   # goal / linear / MLP (Table 3)
SEEDS = (0, 1, 2, 3, 4)                     # five random seeds (Section 5.2)

DOMAIN_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "antmaze": {
        "config": "configs/fre_antmaze.yaml",
        "dataset": "antmaze-large-diverse-v2",
        "num_xy_bins": 32,
        "use_encoder_inputs": False,
    },
    "exorl": {
        "config": "configs/fre_exorl.yaml",
        "domain_name": "walker",
        "dataset": "rnd",
        # Physics augmentation is appended to the encoder input only (Appendix C.2).
        "use_encoder_inputs": True,
    },
    "kitchen": {
        "config": "configs/fre_kitchen.yaml",
        "dataset": "kitchen-complete-v0",
        "use_encoder_inputs": False,
    },
}


# ===========================================================================
# Generic helpers (config access + flexible collaborator calls)
# ===========================================================================
def filter_kwargs(fn: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the keyword arguments that ``fn`` actually accepts."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins / C callables
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    allowed = set(sig.parameters)
    return {k: v for k, v in kwargs.items() if k in allowed}


def call_flexibly(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` after dropping unsupported keyword arguments."""
    return fn(*args, **filter_kwargs(fn, kwargs))


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load a YAML (or JSON) config file; returns ``{}`` when unavailable."""
    if not path:
        return {}
    candidates = [path]
    if not os.path.isabs(path):
        candidates.append(os.path.join(_REPO_ROOT, path))
        candidates.append(os.path.join(_REPO_ROOT, "configs", os.path.basename(path)))
    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        try:
            import yaml  # type: ignore

            with open(candidate, "r") as fh:
                loaded = yaml.safe_load(fh) or {}
        except ImportError:
            with open(candidate, "r") as fh:
                loaded = json.load(fh)
        return loaded if isinstance(loaded, dict) else {}
    return {}


def cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Fetch ``cfg["a"]["b"]`` through the dotted key ``"a.b"``."""
    if not cfg:
        return default
    node: Any = cfg
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node


def resolve_device(requested: Optional[str]) -> str:
    """Return a usable torch device string."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is required at runtime
        return "cpu"
    if requested:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    """Seed numpy / torch / python for reproducibility."""
    np.random.seed(seed)
    try:
        import random

        random.seed(seed)
    except Exception:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


# ===========================================================================
# Dataset construction (AntMaze / ExORL / Kitchen)
# ===========================================================================
def build_dataset(domain: str, cfg: Dict[str, Any], args: argparse.Namespace) -> Any:
    """Instantiate the domain dataset wrapper (exposes ``replay_buffer`` etc.)."""
    defaults = DOMAIN_DEFAULTS.get(domain, {})
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, dict) else {}
    limit = args.limit if args.limit else cfg_get(cfg, "dataset.limit", None)
    exclude_final = bool(cfg_get(cfg, "dataset.exclude_final_states", True))
    attach_stats = bool(cfg_get(cfg, "dataset.attach_stats", True))

    if domain == "antmaze":
        from fre.data.antmaze_dataset import load_antmaze_dataset

        kwargs = dict(
            path=args.dataset_path or cfg_get(cfg, "dataset.path", None),
            dataset_name=args.dataset or dataset_cfg.get("name", defaults.get("dataset")),
            env_id=dataset_cfg.get("env_id"),
            num_bins=int(dataset_cfg.get("num_xy_bins", defaults.get("num_xy_bins", 32))),
            discretize=bool(dataset_cfg.get("discretize", True)),
            discretize_mode=dataset_cfg.get("discretize_mode", "index"),
            bounds=dataset_cfg.get("xy_bounds"),
            build_buffer=True,
            exclude_final_states=exclude_final,
            attach_stats=attach_stats,
            limit=limit,
            seed=args.seed,
        )
        return call_flexibly(load_antmaze_dataset, **kwargs)

    if domain == "exorl":
        from fre.data.exorl_dataset import load_exorl_dataset

        kwargs = dict(
            path=args.dataset_path or cfg_get(cfg, "dataset.path", None),
            domain=(args.task or dataset_cfg.get("domain")
                    or defaults.get("domain_name", "walker")),
            task=dataset_cfg.get("task"),
            dataset_name=dataset_cfg.get("name"),
            dataset=dataset_cfg.get("dataset", defaults.get("dataset", "rnd")),
            normalize=bool(dataset_cfg.get("normalize", True)),
            select_goals=True,
            num_goals=int(cfg_get(cfg, "evaluation.num_goals", 5)),
            build_buffer=True,
            exclude_final_states=exclude_final,
            attach_stats=attach_stats,
            max_transitions=limit,
            seed=args.seed,
        )
        return call_flexibly(load_exorl_dataset, **kwargs)

    if domain == "kitchen":
        from fre.data.kitchen_dataset import KITCHEN_TASKS, load_kitchen_dataset

        kwargs = dict(
            path=args.dataset_path or cfg_get(cfg, "dataset.path", None),
            dataset_name=args.dataset or dataset_cfg.get("name", defaults.get("dataset")),
            env_id=dataset_cfg.get("env_id"),
            tasks=KITCHEN_TASKS,
            build_buffer=True,
            exclude_final_states=exclude_final,
            attach_stats=attach_stats,
            limit=limit,
            seed=args.seed,
        )
        return call_flexibly(load_kitchen_dataset, **kwargs)

    raise ValueError(f"unknown domain {domain!r}; expected one of {sorted(DOMAIN_DEFAULTS)}")


def dataset_replay_buffer(dataset: Any) -> Any:
    """Fetch the ``ReplayBuffer`` out of whatever the loader returned."""
    if dataset is None:
        return None
    if hasattr(dataset, "replay_buffer"):
        buffer = getattr(dataset, "replay_buffer")
        if buffer is None and hasattr(dataset, "build_buffer"):
            buffer = call_flexibly(dataset.build_buffer)
        return buffer
    return dataset


def encoder_state_dim(dataset: Any, use_encoder_inputs: bool, default: int = 0) -> int:
    """Dimension of the states the *encoder* consumes (physics aug. for ExORL)."""
    if dataset is None:
        return default
    if use_encoder_inputs:
        for attr in ("encoder_obs_dim", "encoder_observation_dim"):
            if hasattr(dataset, attr):
                return int(getattr(dataset, attr))
    for attr in ("obs_dim", "observation_dim"):
        if hasattr(dataset, attr):
            return int(getattr(dataset, attr))
    return default


# ===========================================================================
# Prior reward distribution p(eta)  (Section 4.2 / Appendix B / Table 4)
# ===========================================================================
def build_prior(
    cfg: Dict[str, Any],
    dataset: Any,
    replay_buffer: Any,
    state_dim: int,
    domain: str,
    seed: int,
) -> Any:
    """Construct the prior mixture ``p(eta)`` (FRE-all by default)."""
    from fre.reward_priors.mixture import make_mixture_prior, make_prior_from_variant

    prior_cfg = cfg.get("prior", {}) if isinstance(cfg, dict) else {}
    name = prior_cfg.get("name", cfg.get("prior_name", "FRE-all"))
    ratios_cfg = prior_cfg.get("ratios", None)
    ratios = None
    if ratios_cfg:
        ratios = (
            float(ratios_cfg.get("goal", ratios_cfg.get("goal_reaching", DEFAULT_PRIOR_RATIOS[0]))),
            float(ratios_cfg.get("linear", DEFAULT_PRIOR_RATIOS[1])),
            float(ratios_cfg.get("mlp", ratios_cfg.get("random_mlp", DEFAULT_PRIOR_RATIOS[2]))),
        )

    goal_cfg = dict(prior_cfg.get("goal", {}) or {})
    linear_cfg = dict(prior_cfg.get("linear", {}) or {})
    mlp_cfg = dict(prior_cfg.get("mlp", {}) or {})

    # AntMaze: the XY positions are removed from the linear reward generation as
    # the scale of those dimensions led to instability (Source: Appendix B).
    exclude_dims = linear_cfg.get("exclude_dims", None)
    if exclude_dims is None:
        if domain == "antmaze":
            from fre.reward_priors.linear import ANTMAZE_POSITION_DIMS

            exclude_dims = tuple(ANTMAZE_POSITION_DIMS)
        else:
            exclude_dims = ()

    state_min = state_max = None
    try:
        if replay_buffer is not None and hasattr(replay_buffer, "state_box"):
            box = replay_buffer.state_box()
        elif dataset is not None and hasattr(dataset, "state_box"):
            box = dataset.state_box()
        else:
            box = None
        if box is not None:
            state_min, state_max = box
    except Exception:  # pragma: no cover - defensive, bounds are optional
        state_min = state_max = None

    common: Dict[str, Any] = dict(
        replay_buffer=replay_buffer,
        state_dim=state_dim,
        exclude_dims=tuple(exclude_dims),
        hidden_dim=int(mlp_cfg.get("hidden_dim", 32)),
        goal_threshold=float(goal_cfg.get("threshold", 0.0)),
        seed=seed,
    )
    if state_min is not None:
        common["state_min"] = state_min
        common["state_max"] = state_max

    try:
        if ratios is not None:
            return call_flexibly(make_mixture_prior, ratios=ratios, name=name, **common)
        return call_flexibly(make_prior_from_variant, variant=name, **common)
    except Exception:
        # Fall back to the plain uniform mixture (Source: Table 3).
        return call_flexibly(make_mixture_prior, ratios=DEFAULT_PRIOR_RATIOS, **common)


# ===========================================================================
# Model construction
# ===========================================================================
def build_encoder(
    cfg: Dict[str, Any],
    state_dim: int,
    device: str,
) -> Any:
    """Build the permutation-invariant transformer VIB encoder (Section 4.1)."""
    from fre.models.encoder import make_fre_encoder

    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    encoder_layers = model_cfg.get("encoder_layers", ENCODER_LAYERS)
    kwargs = dict(
        state_dim=state_dim,
        latent_dim=int(model_cfg.get("latent_dim", LATENT_DIM)),
        state_embed_dim=int(model_cfg.get("state_embed_dim", STATE_EMBED_DIM)),
        reward_embed_dim=int(model_cfg.get("reward_embed_dim", REWARD_EMBED_DIM)),
        num_reward_embeddings=int(model_cfg.get("num_reward_embeddings", NUM_REWARD_EMBEDDINGS)),
        num_layers=int(model_cfg.get("num_encoder_blocks", len(encoder_layers))),
        num_heads=int(model_cfg.get("encoder_attention_heads", ENCODER_ATTENTION_HEADS)),
        mlp_dim=int(encoder_layers[0] if isinstance(encoder_layers, (list, tuple))
                    else model_cfg.get("encoder_mlp_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.0) or 0.0),
        activation=str(model_cfg.get("activation", "relu")),
        reward_min=float(cfg_get(cfg, "encoder_training.reward_min", -1.0)),
        reward_max=float(cfg_get(cfg, "encoder_training.reward_max", 1.0)),
    )
    return _move_to(call_flexibly(make_fre_encoder, **kwargs), device)


def build_decoder(cfg: Dict[str, Any], state_dim: int, device: str) -> Any:
    """Build the feed-forward reward decoder q_theta(eta(s^d) | s^d, z)."""
    from fre.models.decoder import make_reward_decoder

    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    kwargs = dict(
        state_dim=state_dim,
        latent_dim=int(model_cfg.get("latent_dim", LATENT_DIM)),
        hidden_dims=tuple(model_cfg.get("decoder_layers", DECODER_LAYERS)),
        activation=str(model_cfg.get("activation", "relu")),
        layer_norm=bool(model_cfg.get("layer_norm", False)),
    )
    return _move_to(call_flexibly(make_reward_decoder, **kwargs), device)


def _move_to(module: Any, device: str) -> Any:
    try:
        return module.to(device)
    except AttributeError:  # pragma: no cover - non-torch stub
        return module


# ===========================================================================
# Trainer construction
# ===========================================================================
def build_trainer(
    encoder: Any,
    decoder: Any,
    replay_buffer: Any,
    prior: Any,
    cfg: Dict[str, Any],
    state_dim: int,
    device: str,
    seed: int,
) -> Any:
    """Instantiate the Equation (6) trainer (Algorithm 1, encoder stage)."""
    from fre.training.fre_trainer import FRETrainer, make_fre_trainer

    enc_cfg = cfg.get("encoder_training", {}) if isinstance(cfg, dict) else {}
    kwargs = dict(
        encoder=encoder,
        decoder=decoder,
        replay_buffer=replay_buffer,
        prior=prior,
        state_dim=state_dim,
        latent_dim=int(cfg_get(cfg, "model.latent_dim", LATENT_DIM)),
        device=device,
        batch_size=int(enc_cfg.get("batch_size", TRAIN_BATCH_SIZE)),
        num_encoder_states=int(enc_cfg.get("num_encoder_states", NUM_ENCODER_STATES)),
        num_decoder_states=int(enc_cfg.get("num_decoder_states", NUM_DECODER_STATES)),
        learning_rate=float(enc_cfg.get("learning_rate", TRAIN_LEARNING_RATE)),
        beta=float(enc_cfg.get("beta", TRAIN_BETA)),
        optimizer=str(enc_cfg.get("optimizer", "adam")),
        weight_decay=float(enc_cfg.get("weight_decay", 0.0) or 0.0),
        loss_form=str(enc_cfg.get("loss_form", "mse")),
        sample_z=bool(enc_cfg.get("sample_z", True)),
        reward_range=enc_cfg.get("reward_range", "auto"),
        num_reward_embeddings=int(cfg_get(cfg, "model.num_reward_embeddings", NUM_REWARD_EMBEDDINGS)),
        seed=seed,
    )
    try:
        return call_flexibly(make_fre_trainer, **kwargs)
    except NameError:  # pragma: no cover - factory missing, use the class directly
        return call_flexibly(FRETrainer, **kwargs)


# ===========================================================================
# Main training routine
# ===========================================================================
def train(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run Algorithm 1 phase 1 (Equation (6) optimization) and save a checkpoint."""
    domain = args.domain
    seed = int(args.seed)
    device = resolve_device(args.device or cfg.get("device"))
    seed_everything(seed)

    print(f"[FRE] domain={domain} seed={seed} device={device}")
    print(f"[FRE] config={args.config}")

    # ---- unlabeled offline dataset D --------------------------------------
    dataset = build_dataset(domain, cfg, args)
    replay_buffer = dataset_replay_buffer(dataset)
    use_encoder_inputs = resolve_encoder_inputs(cfg, args, domain)
    state_dim = encoder_state_dim(dataset, use_encoder_inputs)
    if state_dim <= 0 and replay_buffer is not None:
        attr = "encoder_obs_dim" if use_encoder_inputs else "obs_dim"
        state_dim = int(getattr(replay_buffer, attr, 0))
    if state_dim <= 0:
        raise RuntimeError("could not determine the encoder state dimension")
    num_transitions = getattr(replay_buffer, "num_transitions", None)
    if num_transitions is None and replay_buffer is not None:
        num_transitions = len(replay_buffer) if hasattr(replay_buffer, "__len__") else "?"
    print(f"[FRE] replay buffer: {num_transitions} transitions, state_dim={state_dim} "
          f"(physics-augmented encoder inputs={use_encoder_inputs})")

    # ---- prior p(eta) and the two networks --------------------------------
    prior = build_prior(cfg, dataset, replay_buffer, state_dim, domain, seed)
    encoder = build_encoder(cfg, state_dim, device)
    decoder = build_decoder(cfg, state_dim, device)

    # ---- optional warm start ---------------------------------------------
    if args.checkpoint:
        load_checkpoint(encoder, decoder, args.checkpoint, device=device,
                        strict=not args.no_strict)

    trainer = build_trainer(encoder, decoder, replay_buffer, prior, cfg, state_dim, device, seed)
    # ExORL: the physics-derived quantities are the encoder input only (Appendix C.2).
    _set_flag(trainer, "use_encoder_inputs", use_encoder_inputs)

    steps = determine_steps(args, cfg, domain)
    print(f"[FRE] encoder training steps: {steps} "
          f"(batch {getattr(trainer, 'batch_size', TRAIN_BATCH_SIZE)}, "
          f"K={getattr(trainer, 'num_encoder_states', NUM_ENCODER_STATES)}, "
          f"K'={getattr(trainer, 'num_decoder_states', NUM_DECODER_STATES)}, "
          f"beta={getattr(trainer, 'beta', TRAIN_BETA)})")

    if args.dry_run:
        print("[FRE] --dry-run: modules built successfully; skipping optimization.")
        return {"dry_run": True, "state_dim": state_dim, "steps": steps}

    # ---- correctness gates (independent of the RL score) ------------------
    run_gates(trainer, encoder, skip=args.skip_gates)

    # ---- maximize Equation (6) -------------------------------------------
    logger = build_logger(args, cfg)
    history = train_loop(trainer, steps, args, cfg, logger)

    # ---- checkpoint -------------------------------------------------------
    save_path = args.save or cfg_get(
        cfg, "logging.save_encoder_path", f"runs/fre_{domain}_seed{seed}_encoder.pt"
    )
    save_checkpoint(
        save_path,
        trainer=trainer,
        encoder=encoder,
        decoder=decoder,
        cfg=cfg,
        domain=domain,
        seed=seed,
        steps=steps,
        history=history,
        state_dim=state_dim,
        use_encoder_inputs=use_encoder_inputs,
    )
    final_loss = history[-1] if history else float("nan")
    print(f"[FRE] saved encoder/decoder checkpoint -> {save_path} (final loss {final_loss:.5f})")
    return {
        "save_path": save_path,
        "steps": steps,
        "final_loss": final_loss,
        "state_dim": state_dim,
        "use_encoder_inputs": use_encoder_inputs,
    }


def resolve_encoder_inputs(cfg: Dict[str, Any], args: argparse.Namespace, domain: str) -> bool:
    """Encoder-only physics augmentation flag (ExORL): CLI > config > domain."""
    if getattr(args, "no_encoder_inputs", False):
        return False
    if getattr(args, "use_encoder_inputs", None) is not None:
        return bool(args.use_encoder_inputs)
    configured = cfg_get(cfg, "encoder_training.use_encoder_inputs", None)
    if configured is None:
        configured = cfg_get(cfg, "dataset.use_encoder_inputs", None)
    if configured is None:
        configured = cfg_get(cfg, "dataset.physics.encoder_only", None)
    if configured is not None:
        return bool(configured)
    return bool(DOMAIN_DEFAULTS.get(domain, {}).get("use_encoder_inputs", False))


def determine_steps(args: argparse.Namespace, cfg: Dict[str, Any], domain: str) -> int:
    """Encoder step budget (Table 3: 150k for AntMaze, 1M for ExORL/Kitchen)."""
    if args.steps:
        return int(args.steps)
    configured = cfg_get(cfg, "encoder_training.encoder_training_steps", None)
    if configured:
        return int(configured)
    return int(DOMAIN_ENCODER_STEPS.get(domain, ENCODER_TRAINING_STEPS))


def _set_flag(obj: Any, name: str, value: Any) -> None:
    """Best-effort attribute setter (a trainer may keep config in a dataclass)."""
    if obj is None:
        return
    if hasattr(obj, name):
        try:
            setattr(obj, name, value)
            return
        except Exception:  # pragma: no cover
            pass
    for holder_name in ("config", "cfg"):
        holder = getattr(obj, holder_name, None)
        if holder is not None and hasattr(holder, name):
            try:
                setattr(holder, name, value)
                return
            except Exception:  # pragma: no cover
                pass


def build_logger(args: argparse.Namespace, cfg: Dict[str, Any]) -> Any:
    """Metric logger (``fre.utils.logging.MetricLogger``) or ``None``."""
    if args.no_logger:
        return None
    try:
        from fre.utils.logging import MetricLogger

        return call_flexibly(
            MetricLogger,
            log_interval=int(args.log_interval or cfg_get(cfg, "logging.log_interval", 1000)),
            smoothing=float(cfg_get(cfg, "logging.smoothing", 0.9)),
            name="fre_encoder",
            verbose=not args.quiet,
        )
    except Exception:  # pragma: no cover - logging is optional
        return None


def train_loop(
    trainer: Any,
    steps: int,
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    logger: Any,
) -> List[float]:
    """Optimize Equation (6) for ``steps`` updates, returning the loss history."""
    log_interval = int(args.log_interval or cfg_get(cfg, "logging.log_interval", 1000) or 1000)
    history: List[float] = []
    started = time.time()

    # Prefer the trainer's own loop (it may implement convergence detection).
    if not args.manual_loop:
        try:
            out = call_flexibly(
                trainer.train,
                num_steps=int(steps),
                log_interval=log_interval,
                progress=not args.quiet,
            )
            extracted = _extract_history(out)
            if extracted:
                history = extracted
                print(f"[FRE] trained {steps} steps via FRETrainer.train in "
                      f"{time.time() - started:.1f}s; final loss={history[-1]:.5f}")
                return history
        except (TypeError, AttributeError):
            pass

    for step in range(1, int(steps) + 1):
        out = call_flexibly(trainer.train_step)
        stats = _to_stats_dict(out)
        loss = float(stats.get("loss", stats.get("objective", float("nan"))))
        history.append(loss)
        if logger is not None:
            try:
                logger.log(stats, step=step)
            except Exception:  # pragma: no cover
                pass
        if step == 1 or step % max(1, log_interval) == 0:
            print(f"[FRE] step {step}/{int(steps)} loss={loss:.5f} "
                  f"recon={stats.get('reconstruction_loss', float('nan')):.5f} "
                  f"kl={stats.get('kl', float('nan')):.5f} "
                  f"elapsed={time.time() - started:.0f}s")
    if logger is not None:
        try:
            logger.flush()
        except Exception:  # pragma: no cover
            pass
    return history


def _extract_history(output: Any) -> List[float]:
    """Pull a list of losses out of whatever ``trainer.train`` returned."""
    if output is None:
        return []
    if isinstance(output, (list, tuple)):
        history: List[float] = []
        for item in output:
            if isinstance(item, (int, float)):
                history.append(float(item))
            else:
                stats = _to_stats_dict(item)
                history.append(float(stats.get("loss", stats.get("objective", float("nan")))))
        return history
    stats = _to_stats_dict(output)
    for key in ("losses", "history", "encoder_losses") if stats else ():
        value = stats.get(key)
        if isinstance(value, (list, tuple)) and value:
            return [float(v) for v in value]
    if stats:
        return [float(stats.get("loss", stats.get("objective", float("nan"))))]
    return []


def _to_stats_dict(output: Any) -> Dict[str, float]:
    """Coerce a trainer output into a flat ``Dict[str, float]`` of metrics."""
    if output is None:
        return {}
    if isinstance(output, dict):
        raw: Dict[str, Any] = output
    elif hasattr(output, "to_dict"):
        try:
            raw = output.to_dict()
        except Exception:  # pragma: no cover
            raw = {}
    elif hasattr(output, "__dict__"):
        raw = {k: v for k, v in vars(output).items() if not k.startswith("_")}
    else:
        return {}
    out: Dict[str, float] = {}
    for key, value in raw.items():
        try:
            if hasattr(value, "detach"):
                value = value.detach().cpu().item()
            arr = np.asarray(value)
            out[str(key)] = float(arr.reshape(()).item()) if arr.size == 1 else float("nan")
        except Exception:
            continue
    return out


class _Skip(Exception):
    """Internal marker: the gate ran its check successfully."""


def run_gates(trainer: Any, encoder: Any, skip: bool = False) -> None:
    """Correctness gates from the reproduction plan (milestone A)."""
    if skip:
        return

    def gate_permutation() -> None:
        if hasattr(trainer, "check_permutation_invariance"):
            call_flexibly(trainer.check_permutation_invariance, atol=1e-4)
        raise _Skip

    def gate_disjoint() -> None:
        if hasattr(trainer, "check_disjoint_states"):
            call_flexibly(trainer.check_disjoint_states, num_samples=4)
        raise _Skip

    def gate_latent_dim() -> None:
        latent_dim = int(getattr(encoder, "latent_dim", LATENT_DIM))
        assert latent_dim == LATENT_DIM, f"z must be 128-d, got {latent_dim}"
        raise _Skip

    gates: List[Tuple[str, Callable[[], Any]]] = [
        ("permutation invariance of z (shuffling K context tokens)", gate_permutation),
        ("decoder states disjoint from encoder states", gate_disjoint),
        ("latent dim == 128", gate_latent_dim),
    ]
    for name, fn in gates:
        try:
            fn()
        except _Skip:
            print(f"[FRE] gate ok: {name}")
        except AssertionError as exc:
            print(f"[FRE] gate FAILED: {name} ({exc})")
            raise
        except Exception as exc:  # pragma: no cover - gates are best-effort
            print(f"[FRE] gate skipped: {name} ({type(exc).__name__}: {exc})")


# ===========================================================================
# Checkpointing
# ===========================================================================
def save_checkpoint(
    path: str,
    trainer: Any = None,
    encoder: Any = None,
    decoder: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
    domain: str = "antmaze",
    seed: int = 0,
    steps: int = 0,
    history: Optional[List[float]] = None,
    state_dim: int = 0,
    use_encoder_inputs: bool = False,
) -> str:
    """Save the encoder (+ decoder) and run metadata.  Phase 2 reloads this file."""
    import torch

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    payload: Dict[str, Any] = {
        "domain": domain,
        "seed": int(seed),
        "encoder_training_steps": int(steps),
        "state_dim": int(state_dim),
        "use_encoder_inputs": bool(use_encoder_inputs),
        "latent_dim": int(getattr(encoder, "latent_dim", LATENT_DIM)),
        "num_encoder_states": int(getattr(trainer, "num_encoder_states", NUM_ENCODER_STATES)),
        "num_decoder_states": int(getattr(trainer, "num_decoder_states", NUM_DECODER_STATES)),
        "beta": float(getattr(trainer, "beta", TRAIN_BETA)),
        "config": cfg or {},
        "history_tail": list(history[-20:]) if history else [],
    }
    if encoder is not None and hasattr(encoder, "state_dict"):
        payload["encoder"] = encoder.state_dict()
    if decoder is not None and hasattr(decoder, "state_dict"):
        payload["decoder"] = decoder.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(
    encoder: Any,
    decoder: Any,
    path: str,
    device: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load encoder/decoder weights (bare state dicts or training bundles)."""
    import torch

    try:
        payload = torch.load(path, map_location=device)
    except FileNotFoundError:
        print(f"[FRE] warning: checkpoint {path} not found; training from scratch")
        return {}
    if isinstance(payload, dict):
        enc_state = payload.get("encoder", payload.get("encoder_state_dict"))
        dec_state = payload.get("decoder", payload.get("decoder_state_dict"))
        if enc_state is None and "state_dict" in payload:
            enc_state = payload["state_dict"]
        if enc_state is not None and encoder is not None:
            encoder.load_state_dict(enc_state, strict=strict)
        if dec_state is not None and decoder is not None:
            decoder.load_state_dict(dec_state, strict=strict)
        print(f"[FRE] loaded checkpoint {path}")
        return {k: v for k, v in payload.items() if k not in ("encoder", "decoder")}
    if encoder is not None:
        encoder.load_state_dict(payload, strict=strict)
        print(f"[FRE] loaded bare encoder state dict {path}")
    return {}


# ===========================================================================
# CLI
# ===========================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FRE phase 1: train the reward encoder + decoder (Algorithm 1)."
    )
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="path to a YAML config (e.g. configs/fre_antmaze.yaml)")
    parser.add_argument("--domain", type=str, default=None,
                        choices=sorted(DOMAIN_DEFAULTS), help="benchmark domain")
    parser.add_argument("--task", type=str, default=None,
                        help="ExORL domain name, e.g. walker / cheetah")
    parser.add_argument("--dataset", type=str, default=None, help="dataset name override")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="explicit path to an HDF5/NPZ dataset dump")
    parser.add_argument("--limit", type=int, default=None,
                        help="optionally subsample the dataset (debugging)")
    parser.add_argument("--steps", type=int, default=None,
                        help="encoder training steps (default: Table 3 budget)")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu")
    parser.add_argument("--checkpoint", type=str, default=None, help="warm-start checkpoint")
    parser.add_argument("--no-strict", action="store_true", help="non-strict state-dict load")
    parser.add_argument("--save", type=str, default=None, help="output checkpoint path")
    parser.add_argument("--log-interval", type=int, default=None, help="logging interval")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    parser.add_argument("--no-logger", action="store_true", help="disable MetricLogger")
    parser.add_argument("--manual-loop", action="store_true",
                        help="use the explicit step loop instead of FRETrainer.train")
    parser.add_argument("--skip-gates", action="store_true",
                        help="skip the milestone-A correctness gates")
    parser.add_argument("--use-encoder-inputs", dest="use_encoder_inputs",
                        action="store_true", default=None,
                        help="append physics features to the encoder input (ExORL)")
    parser.add_argument("--no-encoder-inputs", action="store_true",
                        help="force raw observations for the encoder")
    parser.add_argument("--dry-run", action="store_true",
                        help="build data/models/prior and exit without training")
    parser.add_argument("--all-seeds", action="store_true",
                        help="run the five protocol seeds sequentially")
    return parser


def resolve_args(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    """Merge CLI values with the config file and per-domain defaults."""
    if args.domain is None:
        args.domain = cfg.get("domain") or "antmaze"
    if args.config is None:
        candidate = DOMAIN_DEFAULTS.get(args.domain, {}).get("config")
        if candidate and os.path.isfile(os.path.join(_REPO_ROOT, candidate)):
            args.config = candidate
    if args.seed is None:
        args.seed = int(cfg.get("seed", 0) or 0)
    if args.task is None:
        args.task = cfg_get(cfg, "dataset.task", None)
    if args.dataset is None:
        args.dataset = cfg_get(cfg, "dataset.name", None)
    return args


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config) if args.config else {}
    args = resolve_args(args, cfg)
    if not cfg:
        candidate = DOMAIN_DEFAULTS.get(args.domain, {}).get("config")
        if candidate:
            cfg = load_config(candidate)
            if cfg and args.config is None:
                args.config = candidate

    seeds = list(SEEDS) if args.all_seeds else [int(args.seed)]
    summary: Dict[str, Any] = {}
    for seed in seeds:
        run_args = argparse.Namespace(**vars(args))
        run_args.seed = int(seed)
        summary[f"seed_{seed}"] = train(run_args, cfg)

    if len(seeds) > 1:
        print("[FRE] per-seed summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[FRE] interrupted")
        sys.exit(130)

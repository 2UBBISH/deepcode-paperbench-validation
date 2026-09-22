#!/usr/bin/env python
"""Phase-2 entry point of FRE: train the z-conditioned IQL agent on a frozen encoder.

Implements the "# Train policy" half of Algorithm 1 (Source: Section 4.3, Algorithm 1):

    while not converged do
        Sample reward function eta ~ p(eta)
        Sample K states for encoder {s_k^e} ~ D
        Encode into latent vector z ~ p_theta({(s_k^e, eta(s_k^e))})
        Train pi(a | s, z), Q(s, a, z), V(s, z) using IQL with r = eta(s)
    end while

Key requirements from the paper that this script enforces:

* The strided training scheme (Source: Section 4.3): "we first only train the FRE
  encoder with gradients from the decoder (Equation (6)). During this time, the RL
  components are not trained. After the encoder loss converges, we freeze the encoder
  and then start the training of the RL networks using the frozen encoder's outputs.
  In this way, we can make the mapping from eta to z stationary during policy learning,
  which we found to be important to correctly estimate multitask Q values using TD
  learning."  -> the encoder is loaded from the phase-1 checkpoint (scripts/train_fre.py),
  put in eval mode with requires_grad=False, and only the RL parameters are optimized.
* IQL (Kostrikov et al., 2021) is the offline RL algorithm used, with the paper's Table 3
  hyper-parameters: batch size 512, Adam lr 1e-4, expectile 0.8, AWR temperature 3.0,
  target update rate 0.001, discount factor 0.88, RL network layers [512, 512, 512].
* Policy training steps: 850,000 for AntMaze, 1,000,000 for ExORL / Kitchen (Table 3).
* The latent z is *concatenated* to the observation fed into the RL components
  (Source: Addendum).
* For ExORL the physics-augmented observations are used only by the *encoder*
  (Source: Appendix C.2); the value function and policy are trained on the underlying
  environment observation space.

Example
-------
    python scripts/train_fre.py   --config configs/fre_antmaze.yaml --checkpoint runs/fre_antmaze_enc.pt
    python scripts/train_policy.py --config configs/fre_antmaze.yaml \
        --checkpoint runs/fre_antmaze_enc.pt --save runs/fre_antmaze_policy.pt

The script is importable: ``from train_policy import train`` runs one seed end-to-end,
which is what ``scripts/run_prior_ablation.py`` uses for the Table 4 sweep.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# constants (Table 3 / Appendix A)
# --------------------------------------------------------------------------------------

TRAIN_BATCH_SIZE = 512
TRAIN_LEARNING_RATE = 1e-4
TRAIN_EXPECTILE = 0.8
TRAIN_AWR_TEMPERATURE = 3.0
TRAIN_TARGET_UPDATE_RATE = 0.001
TRAIN_DISCOUNT = 0.88

NUM_ENCODER_STATES = 32          # K  (Reward Pairs to Encode)
NUM_DECODER_STATES = 8           # K' (Reward Pairs to Decode)
NUM_REWARD_EMBEDDINGS = 32
LATENT_DIM = 128                 # addendum: z is 128-dimensional
STATE_EMBED_DIM = 64             # addendum: state embedding 64 + reward embedding 64
REWARD_EMBED_DIM = 64
ENCODER_LAYERS = (256, 256, 256, 256)
ENCODER_ATTENTION_HEADS = 4
RL_LAYERS = (512, 512, 512)
DECODER_LAYERS = (512, 512, 512)

POLICY_TRAINING_STEPS = 850_000           # AntMaze (Table 3)
LONG_POLICY_TRAINING_STEPS = 1_000_000    # ExORL / Kitchen (Table 3)
ENCODER_TRAINING_STEPS = 150_000
LONG_ENCODER_TRAINING_STEPS = 1_000_000

#: Policy training steps per domain, verbatim from Table 3.
DOMAIN_POLICY_STEPS: Dict[str, int] = {
    "antmaze": POLICY_TRAINING_STEPS,
    "exorl": LONG_POLICY_TRAINING_STEPS,
    "kitchen": LONG_POLICY_TRAINING_STEPS,
}
DOMAIN_ENCODER_STEPS: Dict[str, int] = {
    "antmaze": ENCODER_TRAINING_STEPS,
    "exorl": LONG_ENCODER_TRAINING_STEPS,
    "kitchen": LONG_ENCODER_TRAINING_STEPS,
}

DEFAULT_PRIOR_RATIOS = (0.33, 0.33, 0.33)   # goal / linear / MLP (Table 3)
SEEDS = (0, 1, 2, 3, 4)                     # Table 1: five random seeds
NUM_EVAL_EPISODES = 20

DEFAULT_LOG_INTERVAL = 1000

#: The reward function's done mask is set to True when the goal is achieved
#: (Source: Appendix B), so it terminates the Bellman backup.
DEFAULT_REWARD_DONE_MASKS = True

DOMAIN_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "antmaze": {
        "config": "configs/fre_antmaze.yaml",
        "dataset": "antmaze-large-diverse-v2",
        "env_id": "antmaze-large-diverse-v2",
        "use_encoder_inputs": False,
    },
    "exorl": {
        "config": "configs/fre_exorl.yaml",
        "dataset": "walker-run",
        "use_encoder_inputs": True,
    },
    "kitchen": {
        "config": "configs/fre_kitchen.yaml",
        "dataset": "kitchen-complete-v0",
        "use_encoder_inputs": False,
    },
}

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# --------------------------------------------------------------------------------------
# small generic helpers (mirrors scripts/train_fre.py so collaborators may drift safely)
# --------------------------------------------------------------------------------------

def filter_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the keyword arguments ``fn`` actually accepts."""
    if fn is None:
        return {}
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def call_flexibly(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with only the keyword arguments it accepts."""
    if fn is None:
        raise TypeError("call_flexibly received None")
    try:
        return fn(*args, **kwargs)
    except TypeError:
        filtered = filter_kwargs(fn, kwargs)
        if filtered == kwargs:
            raise
        return fn(*args, **filtered)


def cfg_get(cfg: Any, dotted: str, default: Any = None) -> Any:
    """Dotted-key lookup into nested dicts."""
    if not isinstance(cfg, dict):
        return default
    node: Any = cfg
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return default if node is None else node


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load a YAML (or JSON) configuration file; ``{}`` when unavailable."""
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"[train_policy] WARNING: config '{path}' not found; using defaults.", file=sys.stderr)
        return {}
    try:
        import yaml  # type: ignore

        with open(path, "r") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        try:
            with open(path, "r") as handle:
                return json.load(handle) or {}
        except Exception as exc:  # pragma: no cover - configuration error path
            print(f"[train_policy] WARNING: could not parse '{path}': {exc}", file=sys.stderr)
            return {}


def resolve_device(requested: Optional[str]) -> str:
    """Resolve a torch device string, degrading to CPU when CUDA is unavailable."""
    if requested and requested != "auto":
        if requested.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    print("[train_policy] CUDA unavailable; falling back to CPU.", file=sys.stderr)
                    return "cpu"
            except Exception:
                return "cpu"
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def seed_everything(seed: int) -> None:
    """Seed numpy (and torch, if available) for reproducible runs."""
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def _torch() -> Any:
    import torch  # noqa: WPS433 (deliberately lazy)

    return torch


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _as_numpy(value: Any) -> np.ndarray:
    """Coerce tensors / arrays / scalars to a float32 numpy array."""
    if value is None:
        return np.asarray([], dtype=np.float32)
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    if isinstance(value, (float, int)):
        return np.asarray([value], dtype=np.float32)
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception:
        pass
    return np.asarray(value, dtype=np.float32)


def _to_tensor(value: Any, device: str, dtype: Any = None) -> Any:
    """Move numpy/torch data onto the requested torch device."""
    torch = _torch()
    dtype = torch.float32 if dtype is None else dtype
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)


def _move_to(module: Any, device: str) -> Any:
    try:
        return module.to(device)
    except Exception:
        return module


# --------------------------------------------------------------------------------------
# data / prior / model construction
# --------------------------------------------------------------------------------------

def build_dataset(domain: str, cfg: Dict[str, Any], args: argparse.Namespace) -> Any:
    """Instantiate the domain's offline dataset wrapper (AntMaze / ExORL / Kitchen)."""
    domain = (domain or "antmaze").lower()
    defaults = DOMAIN_DEFAULTS.get(domain, {})
    path = args.dataset_path or cfg_get(cfg, "dataset.path")
    limit = args.limit if getattr(args, "limit", None) else cfg_get(cfg, "dataset.limit")

    if domain == "antmaze":
        from fre.data.antmaze_dataset import load_antmaze_dataset

        kwargs = dict(
            path=path,
            dataset_name=args.dataset or cfg_get(cfg, "dataset.name", defaults.get("dataset")),
            env_id=cfg_get(cfg, "dataset.env_id", defaults.get("env_id")),
            num_bins=int(cfg_get(cfg, "dataset.num_xy_bins", 32)),
            discretize=bool(cfg_get(cfg, "dataset.discretize", True)),
            discretize_mode=cfg_get(cfg, "dataset.discretize_mode", "index"),
            build_buffer=True,
            exclude_final_states=bool(cfg_get(cfg, "dataset.exclude_final_states", True)),
            attach_stats=bool(cfg_get(cfg, "dataset.attach_stats", True)),
            limit=limit,
            seed=int(getattr(args, "seed", 0) or 0),
        )
        return call_flexibly(load_antmaze_dataset, **kwargs)

    if domain == "exorl":
        from fre.data.exorl_dataset import load_exorl_dataset

        task = args.task or cfg_get(cfg, "dataset.task")
        dataset_name = args.dataset or cfg_get(cfg, "dataset.name") or defaults.get("dataset")
        if task and not dataset_name:
            dataset_name = str(task)
        kwargs = dict(
            path=path,
            domain=cfg_get(cfg, "dataset.domain", _domain_of_exorl_name(dataset_name)),
            task=task,
            dataset_name=dataset_name,
            dataset=cfg_get(cfg, "dataset.dataset", "rnd"),
            normalize=bool(cfg_get(cfg, "dataset.normalize", True)),
            select_goals=True,
            num_goals=int(cfg_get(cfg, "evaluation.num_goals", 5)),
            build_buffer=True,
            exclude_final_states=bool(cfg_get(cfg, "dataset.exclude_final_states", True)),
            attach_stats=bool(cfg_get(cfg, "dataset.attach_stats", True)),
            max_transitions=limit,
            seed=int(getattr(args, "seed", 0) or 0),
        )
        return call_flexibly(load_exorl_dataset, **kwargs)

    if domain == "kitchen":
        from fre.data.kitchen_dataset import load_kitchen_dataset

        kwargs = dict(
            path=path,
            dataset_name=args.dataset or cfg_get(cfg, "dataset.name", defaults.get("dataset")),
            env_id=cfg_get(cfg, "dataset.env_id"),
            build_buffer=True,
            exclude_final_states=bool(cfg_get(cfg, "dataset.exclude_final_states", True)),
            attach_stats=bool(cfg_get(cfg, "dataset.attach_stats", True)),
            limit=limit,
            seed=int(getattr(args, "seed", 0) or 0),
        )
        return call_flexibly(load_kitchen_dataset, **kwargs)

    raise ValueError(f"unknown domain '{domain}'")


def _domain_of_exorl_name(name: Optional[str]) -> str:
    if not name:
        return "walker"
    return "cheetah" if str(name).lower().startswith("cheetah") else "walker"


def dataset_replay_buffer(dataset: Any) -> Any:
    """Extract the ``ReplayBuffer`` attached to a dataset wrapper."""
    for name in ("replay_buffer", "buffer"):
        buf = getattr(dataset, name, None)
        if buf is not None:
            return buf
    builder = getattr(dataset, "build_buffer", None)
    if callable(builder):
        return call_flexibly(builder, seed=0, exclude_final_states=True, attach_stats=True)
    raise AttributeError("dataset does not expose a replay buffer")


def encoder_state_dim(dataset: Any, use_encoder_inputs: bool, default: int = 0) -> int:
    """State dim fed to the encoder (physics-augmented for ExORL, Appendix C.2)."""
    if use_encoder_inputs:
        dim = getattr(dataset, "encoder_obs_dim", None)
        if dim is not None:
            return int(dim)
    dim = getattr(dataset, "obs_dim", None)
    return int(dim) if dim is not None else int(default)


def agent_observation_dim(dataset: Any, fallback: int = 0) -> int:
    """State dim fed to the RL components (raw environment observation)."""
    for name in ("obs_dim", "raw_obs_dim"):
        dim = getattr(dataset, name, None)
        if dim is not None:
            return int(dim)
    return int(fallback)


def action_dim(dataset: Any, fallback: int = 8) -> int:
    dim = getattr(dataset, "act_dim", None)
    if dim is None:
        space = getattr(dataset, "observation_space", None)
        dim = getattr(space, "shape", [None])[-1] if space is not None else None
    return int(dim or fallback)


def resolve_encoder_inputs(args: argparse.Namespace, cfg: Dict[str, Any], domain: str) -> bool:
    """Whether the encoder consumes physics-augmented observations (ExORL only)."""
    if getattr(args, "use_encoder_inputs", False):
        return True
    if getattr(args, "no_encoder_inputs", False):
        return False
    domain = (domain or "").lower()
    if domain == "exorl":
        explicit = cfg_get(cfg, "dataset.use_encoder_inputs")
        if explicit is None:
            explicit = cfg_get(cfg, "dataset.physics.encoder_only", True)
        return bool(explicit)
    explicit = cfg_get(cfg, "dataset.use_encoder_inputs")
    if explicit is not None:
        return bool(explicit)
    return bool(DOMAIN_DEFAULTS.get(domain, {}).get("use_encoder_inputs", False))


def build_prior(cfg: Dict[str, Any], dataset: Any, replay: Any, state_dim: int,
                domain: str, seed: int) -> Any:
    """Construct the prior reward distribution p(eta) used during policy training."""
    from fre.reward_priors.mixture import make_mixture_prior, make_prior_from_variant

    variant = cfg_get(cfg, "prior.name", "FRE-all")
    if domain == "antmaze":
        exclude_dims: Tuple[int, ...] = tuple(cfg_get(cfg, "prior.linear.exclude_dims", (0, 1)) or ())
    else:
        exclude_dims = tuple(cfg_get(cfg, "prior.linear.exclude_dims", ()) or ())

    kwargs = dict(
        variant=variant,
        replay_buffer=replay,
        state_dim=state_dim,
        exclude_dims=exclude_dims,
        hidden_dim=int(cfg_get(cfg, "prior.mlp.hidden_dim", 32)),
        goal_threshold=float(cfg_get(cfg, "prior.goal.threshold", 0.0)),
        seed=int(seed),
        p_current=float(cfg_get(cfg, "prior.goal.p_current", 0.2)),
        p_future=float(cfg_get(cfg, "prior.goal.p_future", 0.5)),
        p_random=float(cfg_get(cfg, "prior.goal.p_random", 0.3)),
        geometric_p=float(cfg_get(cfg, "prior.goal.geometric_p", 0.2)),
        reward_unreached=float(cfg_get(cfg, "prior.goal.reward_unreached", -1.0)),
        reward_reached=float(cfg_get(cfg, "prior.goal.reward_reached", 0.0)),
        ensure_goal_in_set=bool(cfg_get(cfg, "prior.goal.ensure_goal_in_set", True)),
    )
    try:
        return call_flexibly(make_prior_from_variant, **kwargs)
    except Exception:
        kwargs.pop("variant", None)
        ratios = cfg_get(cfg, "prior.ratios")
        ratio_tuple = (
            float(ratios.get("goal", 0.33)),
            float(ratios.get("linear", 0.33)),
            float(ratios.get("mlp", 0.33)),
        ) if isinstance(ratios, dict) else DEFAULT_PRIOR_RATIOS
        return call_flexibly(make_mixture_prior, ratios=list(ratio_tuple), **kwargs)


def build_encoder(cfg: Dict[str, Any], state_dim: int, device: str) -> Any:
    """Construct the permutation-invariant transformer VIB encoder (addendum dims)."""
    from fre.models.encoder import make_fre_encoder

    kwargs = dict(
        state_dim=int(state_dim),
        latent_dim=int(cfg_get(cfg, "model.latent_dim", LATENT_DIM)),
        state_embed_dim=int(cfg_get(cfg, "model.state_embed_dim", STATE_EMBED_DIM)),
        reward_embed_dim=int(cfg_get(cfg, "model.reward_embed_dim", REWARD_EMBED_DIM)),
        num_reward_embeddings=int(cfg_get(cfg, "model.num_reward_embeddings", NUM_REWARD_EMBEDDINGS)),
        num_layers=int(cfg_get(cfg, "model.num_encoder_blocks", 4)),
        num_heads=int(cfg_get(cfg, "model.encoder_attention_heads", ENCODER_ATTENTION_HEADS)),
        mlp_dim=int(cfg_get(cfg, "model.mlp_dim", 256)),
        activation=cfg_get(cfg, "model.activation", "relu"),
        dropout=float(cfg_get(cfg, "model.dropout", 0.0)),
        log_std_min=float(cfg_get(cfg, "model.log_std_min", -5.0)),
        log_std_max=float(cfg_get(cfg, "model.log_std_max", 2.0)),
    )
    return _move_to(call_flexibly(make_fre_encoder, **kwargs), device)


def load_encoder_checkpoint(encoder: Any, path: Optional[str], device: str = "cpu",
                            strict: bool = True) -> Dict[str, Any]:
    """Load the frozen phase-1 encoder weights from a checkpoint bundle."""
    meta: Dict[str, Any] = {}
    if not path:
        return meta
    if not os.path.exists(path):
        print(f"[train_policy] WARNING: encoder checkpoint '{path}' not found; using random init.",
              file=sys.stderr)
        return meta
    torch = _torch()
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # older torch versions
        payload = torch.load(path, map_location=device)
    state = payload
    if isinstance(payload, dict):
        for key in ("encoder", "encoder_state_dict", "model", "state_dict"):
            if key in payload and isinstance(payload[key], dict):
                state = payload[key]
                break
        for key in ("domain", "state_dim", "latent_dim", "use_encoder_inputs", "beta", "steps",
                    "num_encoder_states"):
            if key in payload:
                meta[key] = payload[key]
    try:
        encoder.load_state_dict(state, strict=strict)
    except Exception as exc:  # pragma: no cover - checkpoint mismatch path
        print(f"[train_policy] WARNING: strict load failed ({exc}); retrying with strict=False.",
              file=sys.stderr)
        encoder.load_state_dict(state, strict=False)
    print(f"[train_policy] loaded frozen encoder from '{path}'.")
    return meta


def freeze_encoder(encoder: Any) -> Any:
    """Freeze the encoder (strided scheme; keeps eta -> z stationary)."""
    try:
        from fre.training.strided import freeze_encoder as _freeze

        return _freeze(encoder)
    except Exception:
        pass
    try:
        encoder.requires_grad_(False)
    except Exception:
        pass
    for param in getattr(encoder, "parameters", lambda: [])():
        param.requires_grad_(False)
    try:
        encoder.eval()
    except Exception:
        pass
    return encoder


def encoder_is_frozen(encoder: Any) -> bool:
    try:
        from fre.training.strided import encoder_is_frozen as _is_frozen

        return bool(_is_frozen(encoder))
    except Exception:
        pass
    for param in getattr(encoder, "parameters", lambda: [])():
        if param.requires_grad:
            return False
    return True


# --------------------------------------------------------------------------------------
# labelling helpers (r = eta(s))
# --------------------------------------------------------------------------------------

def label_states(eta: Any, states: Any, rng: Optional[np.random.Generator] = None,
                 ensure_goal: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Evaluate a sampled reward function eta on ``states``; returns (rewards, dones)."""
    rewards: Any = None
    dones: Any = None
    label_fn = getattr(eta, "label", None)
    if callable(label_fn):
        try:
            out = call_flexibly(label_fn, states, ensure_goal=ensure_goal, rng=rng)
        except TypeError:
            out = call_flexibly(label_fn, states)
        if isinstance(out, tuple):
            rewards = out[0]
            dones = out[1] if len(out) > 1 else None
        else:
            rewards = out
    if rewards is None:
        # A reward function may be plain callable (eta(states) -> rewards).
        out = call_flexibly(eta, states)
        if isinstance(out, tuple):
            rewards, dones = out[0], (out[1] if len(out) > 1 else None)
        else:
            rewards = out
    rewards = _as_numpy(rewards).reshape(-1)
    if dones is not None:
        dones = _as_numpy(dones).reshape(-1).astype(np.float32)
    return rewards.astype(np.float32, copy=False), dones


def sample_context_states(replay: Any, num_states: int, rng: np.random.Generator,
                          encoder_input: bool) -> np.ndarray:
    """Sample K states uniformly from the offline dataset (Algorithm 1)."""
    sampler = getattr(replay, "sample_states", None)
    if not callable(sampler):
        raise AttributeError("replay buffer does not expose sample_states()")
    return _as_numpy(call_flexibly(sampler, int(num_states), rng=rng, encoder_input=encoder_input))


def encode_context(encoder: Any, states: np.ndarray, rewards: np.ndarray, device: str,
                   sample: bool = False, reward_min: Optional[float] = None,
                   reward_max: Optional[float] = None) -> Any:
    """Run the frozen encoder on the K (state, reward) pairs; returns z (1, latent_dim)."""
    torch = _torch()
    encoder.eval()
    with torch.no_grad():
        out = call_flexibly(
            encoder,
            _to_tensor(states, device),
            _to_tensor(np.asarray(rewards, dtype=np.float32).reshape(-1), device),
            sample=sample,
            reward_min=reward_min,
            reward_max=reward_max,
        )
    z = getattr(out, "z", out)
    z = _to_tensor(z, device)
    if z.dim() == 1:
        z = z.unsqueeze(0)
    return z.detach()


def transitions_to_torch(batch: Any, device: str) -> Dict[str, Any]:
    """Convert a replay ``Batch`` into device tensors (raw batch kept under '_raw')."""
    if hasattr(batch, "to_torch"):
        try:
            converted = batch.to_torch(device=device)
            if isinstance(converted, dict) and "observations" in converted:
                result = dict(converted)
                result["_raw"] = batch
                return result
        except Exception:
            pass
    torch = _torch()
    result: Dict[str, Any] = {}
    for key in ("observations", "actions", "next_observations", "rewards", "terminals"):
        value = getattr(batch, key, None)
        if value is None:
            continue
        result[key] = _to_tensor(value, device, torch.float32)
    if "terminals" in result:
        result["terminals"] = result["terminals"].float()
    result["_raw"] = batch
    return result


def _reward_states(batch: Any, use_encoder_inputs: bool) -> np.ndarray:
    """Observations the sampled reward function eta is evaluated on."""
    if use_encoder_inputs:
        enc = getattr(batch, "encoder_observations", None)
        if enc is not None:
            return _as_numpy(enc)
    return _as_numpy(getattr(batch, "observations"))


def _fuse_terminals(terminals: Any, dones: Optional[np.ndarray], enabled: bool) -> Any:
    """Terminate on goal achievement for goal-reaching priors (Appendix B done mask)."""
    if not enabled or dones is None:
        return terminals
    dones = np.asarray(dones, dtype=np.float32).reshape(-1)
    term = _as_numpy(terminals).reshape(-1)
    if dones.shape[0] != term.shape[0]:
        return terminals
    return np.maximum(term, dones)


# --------------------------------------------------------------------------------------
# training loop
# --------------------------------------------------------------------------------------

def build_trainer(encoder: Any, obs_dim: int, act_dim: int, cfg: Dict[str, Any], device: str) -> Any:
    """Build the IQL trainer over z-conditioned RL networks (frozen encoder attached)."""
    from fre.training.iql import make_iql_trainer

    rl_layers = tuple(cfg_get(cfg, "model.rl_layers", RL_LAYERS) or RL_LAYERS)
    kwargs = dict(
        obs_dim=int(obs_dim),
        act_dim=int(act_dim),
        latent_dim=int(cfg_get(cfg, "model.latent_dim", LATENT_DIM)),
        hidden_dims=tuple(rl_layers),
        encoder=encoder,
        freeze_encoder=True,
        device=device,
        learning_rate=float(cfg_get(cfg, "policy_training.learning_rate", TRAIN_LEARNING_RATE)),
        expectile=float(cfg_get(cfg, "policy_training.expectile", TRAIN_EXPECTILE)),
        awr_temperature=float(cfg_get(cfg, "policy_training.awr_temperature", TRAIN_AWR_TEMPERATURE)),
        discount=float(cfg_get(cfg, "policy_training.discount", TRAIN_DISCOUNT)),
        tau=float(cfg_get(cfg, "policy_training.tau", TRAIN_TARGET_UPDATE_RATE)),
        batch_size=int(cfg_get(cfg, "policy_training.batch_size", TRAIN_BATCH_SIZE)),
        adv_weight_clamp=float(cfg_get(cfg, "policy_training.adv_weight_clamp", 100.0)),
        normalize_advantage=bool(cfg_get(cfg, "policy_training.normalize_advantage", False)),
        max_grad_norm=cfg_get(cfg, "policy_training.max_grad_norm", None),
    )
    return call_flexibly(make_iql_trainer, **kwargs)


def policy_training_loop(trainer: Any, replay: Any, prior: Any, encoder: Any, cfg: Dict[str, Any],
                         steps: int, device: str, rng: np.random.Generator, logger: Any = None,
                         log_interval: int = DEFAULT_LOG_INTERVAL,
                         use_encoder_inputs: bool = False,
                         num_encoder_states: int = NUM_ENCODER_STATES,
                         reward_done_masks: bool = DEFAULT_REWARD_DONE_MASKS,
                         progress: bool = False) -> List[Dict[str, float]]:
    """Algorithm 1 "# Train policy": one eta per update, IQL on frozen latents."""
    torch = _torch()
    batch_size = int(cfg_get(cfg, "policy_training.batch_size", TRAIN_BATCH_SIZE))
    history: List[Dict[str, float]] = []
    reward_range = cfg_get(cfg, "encoder_training.reward_range", "auto")
    reward_min = cfg_get(cfg, "encoder_training.reward_min", None)
    reward_max = cfg_get(cfg, "encoder_training.reward_max", None)
    t0 = time.time()

    for step in range(int(steps)):
        # 1) a batch of state-action pairs (s, a) from the offline dataset
        try:
            batch = call_flexibly(replay.sample_transitions, batch_size, rng=rng,
                                  with_encoder_inputs=bool(use_encoder_inputs))
        except TypeError:
            batch = call_flexibly(replay.sample_transitions, batch_size, rng=rng)
        tensors = transitions_to_torch(batch, device)
        observations = tensors["observations"]
        actions = tensors["actions"]
        next_observations = tensors.get("next_observations", observations)
        terminals = tensors.get("terminals")

        # 2) sample a reward function eta ~ p(eta)
        eta = call_flexibly(prior.sample, rng=rng) if hasattr(prior, "sample") else prior(rng=rng)

        # 3) K states for the encoder, labelled with eta, encoded into z
        ctx_states = sample_context_states(replay, num_encoder_states, rng,
                                           encoder_input=bool(use_encoder_inputs))
        ctx_rewards, _ = label_states(eta, ctx_states, rng=rng, ensure_goal=True)
        if reward_range == "auto":
            lo, hi = _reward_range(eta, ctx_states, ctx_rewards)
        else:
            lo, hi = reward_min, reward_max
        z = encode_context(encoder, ctx_states, ctx_rewards, device, sample=False,
                           reward_min=lo, reward_max=hi)

        # 4) rewards r = eta(s) on the transition observations (physics inputs only for ExORL)
        reward_states = _reward_states(batch, use_encoder_inputs)
        rewards, dones = label_states(eta, reward_states, rng=rng, ensure_goal=False)
        if terminals is not None:
            terminals = _to_tensor(_fuse_terminals(terminals, dones, reward_done_masks), device).float()
            tensors["terminals"] = terminals

        # 5) z is concatenated to the observations inside the RL networks, so tile it
        z_batch = z.expand(int(observations.shape[0]), -1).contiguous()

        # 6) IQL update: V (expectile 0.8), Q (Bellman, gamma 0.88), policy (AWR temperature 3.0)
        losses = call_flexibly(
            trainer.update,
            observations,
            actions,
            next_observations,
            _to_tensor(rewards, device),
            terminals,
            z=z_batch,
            z_next=z_batch,
            update_target=True,
            step_index=step,
        )
        stats = _to_stats_dict(losses)
        stats["step"] = float(step)
        history.append(stats)

        if logger is not None:
            for method in ("log", "log_stats", "log_metrics", "record"):
                fn = getattr(logger, method, None)
                if callable(fn):
                    try:
                        call_flexibly(fn, stats, step=step)
                        break
                    except Exception:
                        continue
        if log_interval and (step % int(log_interval) == 0 or step == int(steps) - 1):
            elapsed = max(time.time() - t0, 1e-6)
            print(
                "[train_policy] step %d/%d value=%.4f q=%.4f pi=%.4f |z|=%.3f (%.1f steps/s)"
                % (
                    step + 1,
                    steps,
                    stats.get("value_loss", float("nan")),
                    stats.get("q_loss", float("nan")),
                    stats.get("policy_loss", float("nan")),
                    float(z.norm().item()),
                    (step + 1) / elapsed,
                )
            )
    return history


def _reward_range(eta: Any, states: np.ndarray,
                  rewards: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Reward range used to discretize eta's outputs into the 32 embedding bins."""
    bounds_fn = getattr(eta, "reward_bounds", None)
    if callable(bounds_fn):
        try:
            bounds = call_flexibly(bounds_fn, states)
            if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
                return float(bounds[0]), float(bounds[1])
        except Exception:
            pass
    lo = getattr(eta, "reward_min", None)
    hi = getattr(eta, "reward_max", None)
    if lo is not None and hi is not None and float(hi) > float(lo):
        return float(lo), float(hi)
    return None, None


def _to_stats_dict(losses: Any) -> Dict[str, float]:
    """Coerce an ``IQLLosses`` (or similar) result into a flat float dict."""
    if losses is None:
        return {}
    if isinstance(losses, dict):
        return {str(k): float(v) for k, v in losses.items()
                if isinstance(v, (int, float, np.floating, np.integer))}
    to_dict = getattr(losses, "to_dict", None)
    if callable(to_dict):
        try:
            return {str(k): float(v) for k, v in to_dict().items()}
        except Exception:
            pass
    out: Dict[str, float] = {}
    for name in dir(losses):
        if name.startswith("_"):
            continue
        try:
            value = getattr(losses, name)
        except Exception:
            continue
        if isinstance(value, (int, float, np.floating, np.integer)):
            out[name] = float(value)
    return out


# --------------------------------------------------------------------------------------
# gates (strided-scheme correctness)
# --------------------------------------------------------------------------------------

def run_gates(encoder: Any, replay: Any, device: str, cfg: Dict[str, Any],
              use_encoder_inputs: bool = False, skip: bool = False) -> Dict[str, Any]:
    """Verify (i) encoder frozen, (ii) z stationary, (iii) latent dim == 128."""
    report: Dict[str, Any] = {}
    if skip:
        return report
    report["encoder_frozen"] = bool(encoder_is_frozen(encoder))
    latent_dim = int(cfg_get(cfg, "model.latent_dim", LATENT_DIM))
    rng = np.random.default_rng(0)
    states = sample_context_states(replay, NUM_ENCODER_STATES, rng,
                                   encoder_input=bool(use_encoder_inputs))
    rewards = np.linspace(-1.0, 1.0, states.shape[0]).astype(np.float32)
    z1 = encode_context(encoder, states, rewards, device)
    z2 = encode_context(encoder, states, rewards, device)
    report["latent_dim"] = int(z1.shape[-1])
    report["stationary_max_abs_diff"] = float((z1 - z2).abs().max().item())
    report["latent_dim_ok"] = bool(report["latent_dim"] == latent_dim)
    if not report["encoder_frozen"]:
        raise AssertionError("Encoder is not frozen: phase-2 requires requires_grad=False "
                             "(strided scheme, Section 4.3).")
    if report["stationary_max_abs_diff"] > 1e-6:
        raise AssertionError("Encoder outputs changed between identical calls; z must be "
                             "stationary during RL training.")
    return report


# --------------------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------------------

def _safe_state_dict(module: Any) -> Dict[str, Any]:
    if module is None or not hasattr(module, "state_dict"):
        return {}
    try:
        return module.state_dict()
    except Exception:
        return {}


def save_checkpoint(path: str, trainer: Any = None, networks: Any = None,
                    cfg: Optional[Dict[str, Any]] = None, domain: str = "antmaze", seed: int = 0,
                    steps: int = 0, history: Optional[List[Dict[str, float]]] = None,
                    obs_dim: int = 0, act_dim: int = 0, use_encoder_inputs: bool = False,
                    extra: Optional[Dict[str, Any]] = None) -> str:
    """Persist the trained RL networks (and metadata) for zero-shot evaluation."""
    torch = _torch()
    nets = networks if networks is not None else getattr(trainer, "networks", None)
    payload: Dict[str, Any] = {
        "domain": domain,
        "seed": int(seed),
        "steps": int(steps),
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
        "latent_dim": int(cfg_get(cfg or {}, "model.latent_dim", LATENT_DIM)),
        "use_encoder_inputs": bool(use_encoder_inputs),
        "expectile": float(cfg_get(cfg or {}, "policy_training.expectile", TRAIN_EXPECTILE)),
        "awr_temperature": float(cfg_get(cfg or {}, "policy_training.awr_temperature",
                                         TRAIN_AWR_TEMPERATURE)),
        "discount": float(cfg_get(cfg or {}, "policy_training.discount", TRAIN_DISCOUNT)),
        "target_update_rate": float(cfg_get(cfg or {}, "policy_training.tau",
                                            TRAIN_TARGET_UPDATE_RATE)),
        "history_tail": (history or [])[-10:],
    }
    if nets is not None:
        for key, attr in (("policy", "policy"), ("q", "q"), ("v", "v"), ("target_v", "target_v")):
            module = getattr(nets, attr, None)
            if module is not None and hasattr(module, "state_dict"):
                payload[key] = module.state_dict()
        all_state = getattr(nets, "state_dict", None)
        if callable(all_state):
            payload["networks"] = _safe_state_dict(nets)
        if hasattr(nets, "policy"):
            payload["policy_state_dict"] = _safe_state_dict(getattr(nets, "policy", None))
    for key, value in (extra or {}).items():
        payload[key] = value

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(payload, path)
    print(f"[train_policy] saved policy checkpoint to '{path}'.")
    return path


def load_policy_checkpoint(networks: Any, path: Optional[str], device: str = "cpu") -> Dict[str, Any]:
    """Optionally resume phase-2 training from a saved policy checkpoint."""
    meta: Dict[str, Any] = {}
    if not path:
        return meta
    if not os.path.exists(path):
        print(f"[train_policy] WARNING: policy checkpoint '{path}' not found; training from scratch.",
              file=sys.stderr)
        return meta
    torch = _torch()
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    state = None
    if isinstance(payload, dict):
        state = payload.get("networks")
        if not state:
            state = {k: payload[k] for k in ("policy", "q", "v") if k in payload} or None
        for key in ("steps", "domain", "obs_dim", "act_dim", "latent_dim"):
            if key in payload:
                meta[key] = payload[key]
    if state:
        try:
            networks.load_state_dict(state, strict=False)
            print(f"[train_policy] resumed policy networks from '{path}'.")
        except Exception as exc:  # pragma: no cover
            print(f"[train_policy] WARNING: could not load policy checkpoint: {exc}", file=sys.stderr)
    return meta


def build_logger(args: argparse.Namespace, seed: int) -> Any:
    if getattr(args, "no_logger", False):
        return None
    try:
        from fre.utils.logging import MetricLogger

        return call_flexibly(
            MetricLogger,
            name=f"{getattr(args, 'domain', 'antmaze')}_policy_s{seed}",
            log_interval=int(getattr(args, "log_interval", DEFAULT_LOG_INTERVAL) or DEFAULT_LOG_INTERVAL),
            verbose=not getattr(args, "quiet", False),
            log_dir=getattr(args, "log_dir", None),
            seed=int(seed),
        )
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# one-seed driver
# --------------------------------------------------------------------------------------

def train(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run Algorithm 1 phase 2 for a single seed; returns a summary dict."""
    domain = (getattr(args, "domain", None) or cfg_get(cfg, "domain", "antmaze") or "antmaze").lower()
    seed = int(getattr(args, "seed", 0) or 0)
    device = resolve_device(getattr(args, "device", None) or cfg_get(cfg, "device"))
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    use_encoder_inputs = resolve_encoder_inputs(args, cfg, domain)

    # ------------------------------------------------------------------ data
    dataset = build_dataset(domain, cfg, args)
    replay = dataset_replay_buffer(dataset)
    enc_dim = encoder_state_dim(dataset, use_encoder_inputs)
    obs_dim = agent_observation_dim(dataset, enc_dim)
    act_dim = action_dim(dataset)
    if not getattr(args, "quiet", False):
        print(f"[train_policy] domain={domain} seed={seed} obs_dim={obs_dim} "
              f"encoder_state_dim={enc_dim} act_dim={act_dim} device={device}")

    steps = determine_policy_steps(args, cfg, domain)
    num_encoder_states = int(cfg_get(cfg, "encoder_training.num_encoder_states", NUM_ENCODER_STATES))

    # --------------------------------------------------------------- p(eta)
    prior = build_prior(cfg, dataset, replay, enc_dim, domain, seed)

    # ----------------------------------------------------- frozen FRE encoder
    encoder = build_encoder(cfg, enc_dim, device)
    ckpt = getattr(args, "checkpoint", None) or cfg_get(cfg, "logging.save_encoder_path")
    meta = load_encoder_checkpoint(encoder, ckpt, device=device,
                                   strict=not getattr(args, "no_strict", False))
    freeze_encoder(encoder)
    assert encoder_is_frozen(encoder), \
        "strided scheme requires a frozen encoder during policy training"

    # ------------------------------------------------------------------ IQL
    trainer = build_trainer(encoder, obs_dim, act_dim, cfg, device)
    networks = getattr(trainer, "networks", None)
    if networks is not None:
        load_policy_checkpoint(networks, getattr(args, "policy_checkpoint", None), device=device)

    logger = build_logger(args, seed)

    # ---------------------------------------------------------------- gates
    report = run_gates(encoder, replay, device, cfg, use_encoder_inputs=use_encoder_inputs,
                       skip=bool(getattr(args, "skip_gates", False)))

    # --------------------------------------------------------------- train
    t0 = time.time()
    history = policy_training_loop(
        trainer=trainer,
        replay=replay,
        prior=prior,
        encoder=encoder,
        cfg=cfg,
        steps=steps,
        device=device,
        rng=rng,
        logger=logger,
        log_interval=int(getattr(args, "log_interval", DEFAULT_LOG_INTERVAL) or 0),
        use_encoder_inputs=use_encoder_inputs,
        num_encoder_states=num_encoder_states,
        reward_done_masks=bool(cfg_get(cfg, "policy_training.reward_done_masks",
                                       DEFAULT_REWARD_DONE_MASKS)),
    )
    wall = time.time() - t0

    # ---------------------------------------------------------------- save
    save_path = getattr(args, "save", None) or cfg_get(cfg, "logging.save_policy_path")
    if save_path and not getattr(args, "no_save", False):
        if "{seed}" in str(save_path):
            save_path = str(save_path).format(seed=seed, domain=domain)
        save_checkpoint(
            save_path,
            trainer=trainer,
            networks=networks,
            cfg=cfg,
            domain=domain,
            seed=seed,
            steps=steps,
            history=history,
            obs_dim=obs_dim,
            act_dim=act_dim,
            use_encoder_inputs=use_encoder_inputs,
            extra={"encoder_checkpoint": ckpt, "gates": report},
        )

    summary = {
        "domain": domain,
        "seed": seed,
        "steps": steps,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "latent_dim": int(cfg_get(cfg, "model.latent_dim", LATENT_DIM)),
        "use_encoder_inputs": bool(use_encoder_inputs),
        "encoder_checkpoint": ckpt,
        "save_path": save_path,
        "wall_time": wall,
        "final_stats": history[-1] if history else {},
        "gates": report,
        "encoder_checkpoint_meta": meta,
    }
    if not getattr(args, "quiet", False):
        metrics = summary["final_stats"]
        pretty = " ".join(f"{k}={v:.4f}" for k, v in list(metrics.items())[:4])
        print(f"[train_policy] finished {steps} steps in {wall:.1f}s ({pretty})")
        print("[train_policy] run `python scripts/evaluate.py` with the saved checkpoint for the "
              "zero-shot evaluation (32 context samples, 20 episodes x 5 seeds).")
    return summary


def determine_policy_steps(args: argparse.Namespace, cfg: Dict[str, Any], domain: str) -> int:
    if getattr(args, "steps", None):
        return int(args.steps)
    configured = cfg_get(cfg, "policy_training.policy_training_steps")
    if configured:
        return int(configured)
    return int(DOMAIN_POLICY_STEPS.get(domain, POLICY_TRAINING_STEPS))


# aliases used by sibling scripts (e.g. scripts/run_prior_ablation.py)
train_policy = train
run_policy_training = train


def resolve_args(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    """Merge CLI > config > domain defaults for the arguments that matter."""
    if not getattr(args, "domain", None):
        args.domain = cfg_get(cfg, "domain", "antmaze")
    args.domain = (args.domain or "antmaze").lower()
    if not getattr(args, "config", None):
        args.config = DOMAIN_DEFAULTS.get(args.domain, {}).get("config")
    if not getattr(args, "device", None):
        args.device = cfg_get(cfg, "device", "auto")
    if getattr(args, "save", None) is None:
        args.save = cfg_get(cfg, "logging.save_policy_path")
    if getattr(args, "checkpoint", None) is None:
        args.checkpoint = cfg_get(cfg, "logging.save_encoder_path")
    return args


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the FRE-conditioned IQL policy on a frozen FRE encoder "
                    "(Algorithm 1 '# Train policy', Section 4.3).",
    )
    parser.add_argument("--config", type=str, default=None,
                        help="YAML/JSON config (e.g. configs/fre_antmaze.yaml)")
    parser.add_argument("--domain", type=str, default=None, choices=["antmaze", "exorl", "kitchen"])
    parser.add_argument("--task", type=str, default=None, help="ExORL task (e.g. walker-run)")
    parser.add_argument("--dataset", type=str, default=None, help="dataset id / name override")
    parser.add_argument("--dataset-path", type=str, default=None, help="path to a local dataset dump")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of loaded transitions")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="phase-1 encoder checkpoint produced by scripts/train_fre.py")
    parser.add_argument("--policy-checkpoint", type=str, default=None,
                        help="optional phase-2 checkpoint to resume RL network weights")
    parser.add_argument("--steps", type=int, default=None,
                        help="policy steps (default: Table 3 -- 850k AntMaze, 1M ExORL/Kitchen)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all-seeds", action="store_true",
                        help="run all five seeds (Table 1 protocol)")
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | auto")
    parser.add_argument("--save", type=str, default=None, help="policy checkpoint output path")
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--use-encoder-inputs", action="store_true",
                        help="feed physics-augmented observations to the encoder (ExORL)")
    parser.add_argument("--no-encoder-inputs", action="store_true",
                        help="force raw observations into the encoder")
    parser.add_argument("--no-strict", action="store_true", help="load the encoder with strict=False")
    parser.add_argument("--skip-gates", action="store_true",
                        help="skip the frozen/stationary/latent-dim correctness gates")
    parser.add_argument("--no-save", action="store_true", help="do not write a checkpoint")
    parser.add_argument("--no-logger", action="store_true", help="disable metric logging")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="exercise the pipeline with mock data (no dataset/simulator needed)")
    return parser


# --------------------------------------------------------------------------------------
# dry-run plumbing (mock dataset / prior so the script is testable without D4RL/MuJoCo)
# --------------------------------------------------------------------------------------

class _MockReplay:
    """Minimal numpy replay buffer used only by ``--dry-run``."""

    def __init__(self, obs_dim: int = 29, act_dim: int = 8, num_states: int = 20_000,
                 encoder_obs_dim: Optional[int] = None, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.encoder_obs_dim = int(encoder_obs_dim or obs_dim)
        self.observations = self.rng.normal(size=(num_states, self.obs_dim)).astype(np.float32)
        self.actions = self.rng.uniform(-1, 1, size=(num_states, self.act_dim)).astype(np.float32)

    def sample_states(self, num_samples: int, rng: Optional[np.random.Generator] = None,
                      encoder_input: bool = False) -> np.ndarray:
        rng = rng or self.rng
        idx = rng.integers(0, self.observations.shape[0], size=int(num_samples))
        states = self.observations[idx]
        if encoder_input and self.encoder_obs_dim != self.obs_dim:
            extra = rng.normal(size=(int(num_samples), self.encoder_obs_dim - self.obs_dim))
            states = np.concatenate([states, extra.astype(np.float32)], axis=-1)
        return states

    def sample_transitions(self, batch_size: int, rng: Optional[np.random.Generator] = None,
                           with_encoder_inputs: bool = False) -> Any:
        from fre.data.replay import Batch

        rng = rng or self.rng
        idx = rng.integers(0, self.observations.shape[0] - 1, size=int(batch_size))
        obs = self.observations[idx]
        acts = self.actions[idx]
        nobs = self.observations[idx + 1]
        kwargs: Dict[str, Any] = dict(
            observations=obs,
            actions=acts,
            next_observations=nobs,
            rewards=np.zeros(int(batch_size), dtype=np.float32),
            terminals=np.zeros(int(batch_size), dtype=np.float32),
        )
        if with_encoder_inputs and self.encoder_obs_dim != self.obs_dim:
            extra = rng.normal(size=(int(batch_size), self.encoder_obs_dim - self.obs_dim))
            enc = np.concatenate([obs, extra.astype(np.float32)], axis=-1)
            enc_next = np.concatenate([nobs, extra.astype(np.float32)], axis=-1)
            kwargs["encoder_observations"] = enc
            kwargs["encoder_next_observations"] = enc_next
        return Batch(**kwargs)


class _MockGoalReward:
    """Singleton goal-reaching reward over the mock states (matches Appendix B)."""

    name = "mock-goal"
    reward_min = -1.0
    reward_max = 0.0

    def __init__(self, replay: _MockReplay, goal: Optional[np.ndarray] = None,
                 threshold: float = 0.5):
        self.goal = (np.zeros(replay.obs_dim, dtype=np.float32) if goal is None
                     else np.asarray(goal, dtype=np.float32))
        self.threshold = float(threshold)

    def label(self, states: Any, ensure_goal: bool = False, rng: Any = None,
              in_place: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        states = _as_numpy(states).reshape(-1, self.goal.shape[0])
        dist = np.linalg.norm(states - self.goal[None, :], axis=-1)
        reached = dist <= self.threshold
        rewards = np.where(reached, 0.0, -1.0)
        return rewards.astype(np.float32), reached.astype(np.float32)

    def reward_bounds(self, states: Any = None) -> Tuple[float, float]:
        return -1.0, 0.0


class _MockPrior:
    """Uniform prior over mock goal-reaching rewards (dry-run only)."""

    def __init__(self, replay: _MockReplay, seed: int = 0):
        self.replay = replay
        self.rng = np.random.default_rng(seed)

    def sample(self, rng: Optional[np.random.Generator] = None) -> _MockGoalReward:
        rng = rng or self.rng
        goal = self.replay.observations[rng.integers(0, self.replay.observations.shape[0])]
        return _MockGoalReward(self.replay, goal=goal.copy(), threshold=0.5)


def dry_run(args: argparse.Namespace, cfg: Dict[str, Any]) -> int:
    """Exercise the phase-2 plumbing on mock data; returns a process exit code."""
    domain = (args.domain or "antmaze").lower()
    use_encoder_inputs = domain == "exorl"
    obs_dim = 29
    enc_dim = obs_dim + (3 if use_encoder_inputs else 0)
    replay = _MockReplay(obs_dim=obs_dim, act_dim=8, encoder_obs_dim=enc_dim,
                         seed=int(args.seed or 0))
    prior = _MockPrior(replay, seed=int(args.seed or 0))
    steps = int(args.steps or 3)
    rng = np.random.default_rng(int(args.seed or 0))

    print("[train_policy][dry-run] domain=%s obs_dim=%d encoder_state_dim=%d steps=%d"
          % (domain, obs_dim, enc_dim, steps))

    if not _torch_available():
        # numpy-only smoke test of the sampling / labelling path
        for step in range(steps):
            batch = replay.sample_transitions(8, rng=rng, with_encoder_inputs=use_encoder_inputs)
            eta = prior.sample(rng=rng)
            rewards, _ = label_states(eta, _reward_states(batch, use_encoder_inputs), rng=rng)
            ctx = replay.sample_states(NUM_ENCODER_STATES, rng=rng,
                                       encoder_input=use_encoder_inputs)
            ctx_r, _ = label_states(eta, ctx, rng=rng, ensure_goal=True)
            print("[train_policy][dry-run] step %d: r_mean=%.3f r_range=(%.2f,%.2f) ctx=%s "
                  "goal_in_ctx=%s" % (step, rewards.mean(), rewards.min(), rewards.max(),
                                      ctx.shape, bool((ctx_r == 0).any())))
        print("[train_policy][dry-run] torch unavailable: numpy plumbing verified; "
              "skipping IQL updates.")
        return 0

    device = "cuda" if _torch().cuda.is_available() else "cpu"
    encoder = build_encoder(cfg, enc_dim, device)
    freeze_encoder(encoder)
    trainer = build_trainer(encoder, obs_dim, 8, cfg, device)
    report = run_gates(encoder, replay, device, cfg, use_encoder_inputs=use_encoder_inputs,
                       skip=False)
    print(f"[train_policy][dry-run] gates: {report}")
    history = policy_training_loop(
        trainer=trainer, replay=replay, prior=prior, encoder=encoder, cfg=cfg, steps=steps,
        device=device, rng=rng, logger=None, log_interval=1,
        use_encoder_inputs=use_encoder_inputs,
    )
    assert len(history) == steps, "training loop did not run the requested number of steps"
    for key in ("value_loss", "q_loss", "policy_loss"):
        if key in history[-1]:
            assert np.isfinite(history[-1][key]), f"non-finite {key}"
    # the encoder must not accumulate gradients during phase 2 (strided scheme)
    for name, param in encoder.named_parameters():
        assert not param.requires_grad, f"encoder parameter {name} was not frozen"
        assert param.grad is None, f"encoder parameter {name} received gradients"
    print("[train_policy][dry-run] OK: IQL updates ran with a frozen, stationary encoder.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    resolve_args(args, cfg)

    if args.dry_run:
        return dry_run(args, cfg)

    seeds = list(SEEDS) if getattr(args, "all_seeds", False) else [int(args.seed)]
    summaries: List[Dict[str, Any]] = []
    try:
        for seed in seeds:
            args.seed = int(seed)
            summaries.append(train(args, cfg))
    except Exception as exc:  # pragma: no cover - surfaced to the user
        print(f"[train_policy] ERROR: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    if len(summaries) > 1:
        print("\n[train_policy] per-seed summaries")
        for summary in summaries:
            print(f"  seed={summary['seed']} steps={summary['steps']} save={summary.get('save_path')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

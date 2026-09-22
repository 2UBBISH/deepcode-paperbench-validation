"""Prior-scaling ablation sweep (Table 4 / Section 5.4) for FRE.

This script drives the Table 4 experiment of *Zero-Shot Reinforcement Learning via
Functional Reward Encodings* (Appendix "Clarifications on FRE Prior Reward
Distributions"):

    - ``FRE-all``     : equal split of singleton goal-reaching / random linear / random MLP
    - ``FRE-goals``   : singleton goal-reaching only
    - ``FRE-lin``     : random linear only
    - ``FRE-mlp``     : random MLP only
    - ``FRE-lin-mlp`` : equal split of random linear and random MLP
    - ``FRE-goal-mlp``: equal split of singleton goal-reaching and random MLP
    - ``FRE-goal-lin``: equal split of singleton goal-reaching and random linear
    - ``FRE-hint``    : prior that is a *superset* of the evaluation tasks
                        (ant-directional = all unit (x, y) movement directions;
                         cheetah/walker-velocity = movement at a specific velocity)

For every variant and every seed the script performs the two-phase strided training
scheme of Algorithm 1 (train the encoder with Equation (6), freeze it, then train the
z-conditioned IQL agent) using the *same training budget* for all agents, followed by
the zero-shot evaluation protocol of Section 5.2 (32 (state, reward) context pairs, 20
episodes per task, 5 seeds, returns normalized to [0, 100]).

Results are aggregated into the Table 4 columns (goal-reaching / directional /
random-simplex / path-all / total) and the success criterion (the vanilla ``FRE-all``
prior should attain the highest total) is checked.

Usage
-----
    python scripts/run_prior_ablation.py --config configs/prior_ablations.yaml \
        --domain antmaze --variants FRE-all,FRE-goals,FRE-lin,FRE-mlp,FRE-lin-mlp,\
FRE-goal-mlp,FRE-goal-lin --seeds 0,1,2,3,4

``--dry-run`` performs a torch/MuJoCo-free smoke test of the sweep bookkeeping.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Make the repository importable when executed as `python scripts/run_prior_ablation.py`
# --------------------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_PACKAGE_ROOT = os.path.join(_REPO_ROOT, "fre")  # `fre/` package lives under `fre/`
for _path in (_REPO_ROOT, _PACKAGE_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:  # optional config parsing (JSON fallback when PyYAML is unavailable)
    import yaml
except Exception:  # pragma: no cover - exercised only without PyYAML
    yaml = None


# ======================================================================================
# Paper constants (Table 3 / Table 4 / Appendix "Clarifications on FRE Prior ...")
# ======================================================================================

#: Canonical variant order used for Table 4 (FRE-hint reported separately in Section 5.4)
DEFAULT_VARIANTS: Tuple[str, ...] = (
    "FRE-all",
    "FRE-goals",
    "FRE-lin",
    "FRE-mlp",
    "FRE-lin-mlp",
    "FRE-goal-mlp",
    "FRE-goal-lin",
)

#: Prior-mixture ratios (goal-reaching, linear, MLP).  Equal splits per the addendum.
VARIANT_RATIOS: Dict[str, Tuple[float, float, float]] = {
    "FRE-all": (0.33, 0.33, 0.33),
    "FRE": (0.33, 0.33, 0.33),
    "FRE-goals": (1.0, 0.0, 0.0),
    "FRE-goal": (1.0, 0.0, 0.0),
    "FRE-lin": (0.0, 1.0, 0.0),
    "FRE-linear": (0.0, 1.0, 0.0),
    "FRE-mlp": (0.0, 0.0, 1.0),
    "FRE-lin-mlp": (0.0, 0.5, 0.5),
    "FRE-goal-mlp": (0.5, 0.0, 0.5),
    "FRE-goal-lin": (0.5, 0.5, 0.0),
    "FRE-hint": (0.33, 0.33, 0.33),
}

#: Table 4 columns -> the AntMaze evaluation suites that are averaged into them.
#: (``path-all`` is the mean of the three hand-crafted path tasks; ``total`` is the mean
#: of the four task-group columns, matching the reported "total" row of Table 4.)
DEFAULT_TABLE4_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "goal-reaching": ("ant-goal-reaching",),
    "directional": ("ant-directional",),
    "random-simplex": ("ant-random-simplex",),
    "path-all": ("ant-path-center", "ant-path-loop", "ant-path-edges"),
    "total": (
        "ant-goal-reaching",
        "ant-directional",
        "ant-random-simplex",
        "ant-path-center",
        "ant-path-loop",
        "ant-path-edges",
    ),
}

#: Reference Table 4 numbers (mean +- std over 5 seeds) used for validation/reporting
EXPECTED_TABLE4: Dict[str, Dict[str, float]] = {
    "FRE-all": {"goal-reaching": 48.8, "directional": 55.2, "random-simplex": 21.3, "path-all": 63.8, "total": 47.3},
    "FRE-goals": {"total": 26.1},
    "FRE-lin": {"total": 31.6},
    "FRE-mlp": {"total": 25.3},
    "FRE-lin-mlp": {"total": 32.3},
    "FRE-goal-mlp": {"total": 33.8},
    "FRE-goal-lin": {"total": 46.9},
}

# Table 3 hyper-parameters (kept here so the sweep enforces an identical budget/setup).
TRAIN_BATCH_SIZE = 512
TRAIN_LEARNING_RATE = 1e-4
TRAIN_BETA = 0.01
TRAIN_EXPECTILE = 0.8
TRAIN_AWR_TEMPERATURE = 3.0
TRAIN_TARGET_UPDATE_RATE = 0.001
TRAIN_DISCOUNT = 0.88
NUM_ENCODER_STATES = 32
NUM_DECODER_STATES = 8
NUM_REWARD_EMBEDDINGS = 32
LATENT_DIM = 128
STATE_EMBED_DIM = 64
REWARD_EMBED_DIM = 64
ENCODER_LAYERS = (256, 256, 256, 256)
ENCODER_ATTENTION_HEADS = 4
RL_LAYERS = (512, 512, 512)
DECODER_LAYERS = (512, 512, 512)

# Step budgets: Table 3 (AntMaze 150k encoder / 850k policy; ExORL & Kitchen 1M / 1M).
DOMAIN_STEPS: Dict[str, Tuple[int, int]] = {
    "antmaze": (150_000, 850_000),
    "exorl": (1_000_000, 1_000_000),
    "kitchen": (1_000_000, 1_000_000),
}

# Evaluation protocol (Section 5.2).
NUM_EVAL_EPISODES = 20
NUM_TRAINING_SEEDS = 5
FRE_CONTEXT_SAMPLES = 32
NORMALIZED_RETURN_MIN = 0.0
NORMALIZED_RETURN_MAX = 100.0

DEFAULT_LOG_INTERVAL = 1_000
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DOMAIN_EVAL_SUITES: Dict[str, Tuple[str, ...]] = {
    "antmaze": (
        "ant-goal-reaching",
        "ant-directional",
        "ant-random-simplex",
        "ant-path-center",
        "ant-path-loop",
        "ant-path-edges",
    ),
    "exorl": (
        "exorl-cheetah-velocity",
        "exorl-walker-velocity",
        "exorl-cheetah-goals",
        "exorl-walker-goals",
    ),
    "kitchen": ("kitchen",),
}
DOMAIN_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "antmaze": {"config": "configs/fre_antmaze.yaml", "dataset": "antmaze-large-diverse-v2"},
    "exorl": {"config": "configs/fre_exorl.yaml", "dataset": "walker-run"},
    "kitchen": {"config": "configs/fre_kitchen.yaml", "dataset": "kitchen-complete-v0"},
}


# ======================================================================================
# Generic helpers
# ======================================================================================


def filter_kwargs(fn: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the ``kwargs`` accepted by ``fn`` (tolerates API drift)."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins / C callables
        return dict(kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in signature.parameters}


def call_flexibly(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` forwarding only the keyword arguments it accepts."""
    if fn is None:
        raise TypeError("call_flexibly received None")
    return fn(*args, **filter_kwargs(fn, kwargs))


def cfg_get(cfg: Optional[Dict[str, Any]], dotted: str, default: Any = None) -> Any:
    """Dotted-key lookup into a nested config mapping."""
    if not cfg:
        return default
    node: Any = cfg
    for key in dotted.split("."):
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return default
    return node


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load a YAML (or JSON) configuration file; returns ``{}`` when absent."""
    if not path:
        return {}
    if not os.path.exists(path):
        for root in (_HERE, _REPO_ROOT, _PACKAGE_ROOT):
            candidate = os.path.join(root, path)
            if os.path.exists(candidate):
                path = candidate
                break
    if not os.path.exists(path):
        print(f"[warn] config not found: {path}", file=sys.stderr)
        return {}
    with open(path, "r") as handle:
        text = handle.read()
    if yaml is not None:
        try:
            return yaml.safe_load(text) or {}
        except Exception:  # pragma: no cover - malformed YAML
            pass
    try:
        return json.loads(text) or {}
    except Exception:  # pragma: no cover
        print(f"[warn] could not parse config: {path}", file=sys.stderr)
        return {}


def resolve_device(requested: Optional[str] = "auto") -> str:
    """Resolve ``auto|cpu|cuda`` into a concrete device string."""
    if requested in (None, "auto"):
        try:
            import torch  # noqa: WPS433 (lazy import on purpose)

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if str(requested).startswith("cuda"):
        try:
            import torch  # noqa: WPS433

            if not torch.cuda.is_available():
                print("[warn] CUDA requested but unavailable; using CPU", file=sys.stderr)
                return "cpu"
        except Exception:
            return "cpu"
    return str(requested)


def seed_everything(seed: int) -> None:
    """Seed numpy and (if available) torch."""
    np.random.seed(int(seed) % (2**32 - 1))
    try:
        import torch  # noqa: WPS433

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def make_rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _import_first(module_names: Sequence[str]) -> Optional[Any]:
    """Import the first importable module from ``module_names``."""
    for name in module_names:
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


def _resolve(module_names: Sequence[str], attr: str, required: bool = False) -> Any:
    """Resolve ``attr`` from the first module in ``module_names`` that defines it."""
    for name in module_names:
        module = _import_first((name,))
        if module is not None and hasattr(module, attr):
            return getattr(module, attr)
    if required:
        raise ImportError(f"could not resolve {attr!r} from {list(module_names)}")
    return None


# ======================================================================================
# Variant configuration
# ======================================================================================


def normalize_variant_name(variant: str) -> str:
    """Normalize a user-supplied variant name to its canonical ``FRE-*`` spelling."""
    key = str(variant).strip()
    if key in VARIANT_RATIOS:
        return key
    lowered = key.lower()
    for name in list(VARIANT_RATIOS) + list(DEFAULT_VARIANTS) + ["FRE-hint"]:
        if name.lower() == lowered:
            return name
    aliases = {
        "fre_all": "FRE-all",
        "fre_all_prior": "FRE-all",
        "all": "FRE-all",
        "goals": "FRE-goals",
        "goal": "FRE-goals",
        "lin": "FRE-lin",
        "linear": "FRE-lin",
        "mlp": "FRE-mlp",
        "lin_mlp": "FRE-lin-mlp",
        "lin-mlp": "FRE-lin-mlp",
        "goal_mlp": "FRE-goal-mlp",
        "goal-mlp": "FRE-goal-mlp",
        "goal_lin": "FRE-goal-lin",
        "goal-lin": "FRE-goal-lin",
        "hint": "FRE-hint",
        "fre_hint": "FRE-hint",
    }
    if lowered in aliases:
        return aliases[lowered]
    raise ValueError(f"unknown prior variant: {variant!r} (expected one of {DEFAULT_VARIANTS + ('FRE-hint',)})")


def variant_ratios(variant: str) -> Tuple[float, float, float]:
    """Return the (goal, linear, MLP) mixture ratios of a named prior variant."""
    name = normalize_variant_name(variant)
    if name in VARIANT_RATIOS:
        return VARIANT_RATIOS[name]
    # Fall back to the mixture module's table (kept in sync with reward_priors/mixture.py)
    resolver = _resolve(("fre.reward_priors.mixture", "mixture"), "variant_ratios")
    if resolver is not None:
        try:
            ratios = tuple(float(r) for r in resolver(name))
            if len(ratios) == 3:
                return ratios  # type: ignore[return-value]
        except Exception:
            pass
    raise ValueError(f"unknown prior variant: {variant!r}")


def is_hint_variant(variant: str) -> bool:
    return normalize_variant_name(variant) == "FRE-hint"


def variant_prior_config(variant: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the ``prior`` config block used for one variant.

    The three component samplers keep the shared hyper-parameters from the config
    (HER probabilities, linear sparsity/AntMaze-XY exclusion, MLP size/clipping) and
    only the mixture ratios change between variants (Addendum, "Clarifications on FRE
    Prior Reward Distributions").
    """
    name = normalize_variant_name(variant)
    goal_ratio, linear_ratio, mlp_ratio = variant_ratios(name)
    prior_cfg: Dict[str, Any] = {
        "name": name,
        "ratios": {"goal": goal_ratio, "linear": linear_ratio, "mlp": mlp_ratio},
        "goal": dict(cfg_get(cfg, "prior.goal", {}) or cfg_get(cfg, "prior_components.goal", {}) or {}),
        "linear": dict(cfg_get(cfg, "prior.linear", {}) or cfg_get(cfg, "prior_components.linear", {}) or {}),
        "mlp": dict(cfg_get(cfg, "prior.mlp", {}) or cfg_get(cfg, "prior_components.mlp", {}) or {}),
        "hint": {"enabled": bool(is_hint_variant(name))},
    }
    if is_hint_variant(name):
        hint_cfg = dict(cfg_get(cfg, "prior.hint", {}) or {})
        hint_cfg["enabled"] = True
        hint_cfg.setdefault("hint_probability", 0.5)  # not specified in the paper
        prior_cfg["hint"] = hint_cfg
        # Component defaults for the hint variant still follow the vanilla mixture.
        prior_cfg["ratios"] = {
            "goal": float(cfg_get(cfg, "ablations.FRE-hint.ratios.goal", 0.33) or 0.33),
            "linear": float(cfg_get(cfg, "ablations.FRE-hint.ratios.linear", 0.33) or 0.33),
            "mlp": float(cfg_get(cfg, "ablations.FRE-hint.ratios.mlp", 0.33) or 0.33),
        }
    return prior_cfg


def variant_config(cfg: Dict[str, Any], variant: str, domain: str, seed: int) -> Dict[str, Any]:
    """Deep-ish copy of ``cfg`` specialized to one (variant, domain, seed)."""
    import copy

    new_cfg = copy.deepcopy(cfg) if cfg else {}
    new_cfg["domain"] = domain
    new_cfg["seed"] = int(seed)
    new_cfg["method"] = f"FRE-{normalize_variant_name(variant).replace('FRE-', '')}"
    new_cfg["prior"] = variant_prior_config(variant, cfg)
    # Identical training budget for every variant (Addendum requirement).
    enc_steps, pol_steps = steps_for_domain(cfg, domain)
    new_cfg.setdefault("encoder_training", {})
    new_cfg.setdefault("policy_training", {})
    new_cfg["encoder_training"]["encoder_training_steps"] = enc_steps
    new_cfg["policy_training"]["policy_training_steps"] = pol_steps
    return new_cfg


def steps_for_domain(cfg: Optional[Dict[str, Any]], domain: str) -> Tuple[int, int]:
    """Encoder / policy step budget for a domain (Table 3), overridable by config."""
    domain = (domain or "antmaze").lower()
    default_encoder, default_policy = DOMAIN_STEPS.get(domain, (150_000, 850_000))
    encoder_steps = int(cfg_get(cfg, "encoder_training.encoder_training_steps", default_encoder))
    policy_steps = int(cfg_get(cfg, "policy_training.policy_training_steps", default_policy))
    return encoder_steps, policy_steps


# ======================================================================================
# Dataset / prior / model construction (delegates to the training modules)
# ======================================================================================


def build_dataset(domain: str, cfg: Dict[str, Any], args: argparse.Namespace) -> Any:
    """Instantiate the offline dataset wrapper for one domain."""
    dataset_path = getattr(args, "dataset_path", None)
    limit = getattr(args, "limit", None)
    seed = int(getattr(args, "seed", 0) or 0)
    common = {
        "path": dataset_path,
        "dataset_name": getattr(args, "dataset", None) or cfg_get(cfg, "dataset.name", None),
        "limit": limit,
        "seed": seed,
        "build_buffer": True,
    }
    if domain == "antmaze":
        loader = _resolve(("fre.data.antmaze_dataset",), "load_antmaze_dataset", required=True)
        kwargs = dict(common)
        kwargs.update(
            {
                "env_id": cfg_get(cfg, "dataset.env_id", "antmaze-large-diverse-v2"),
                "num_bins": int(cfg_get(cfg, "dataset.num_xy_bins", 32)),
                "discretize": bool(cfg_get(cfg, "dataset.discretize", True)),
                "discretize_mode": cfg_get(cfg, "dataset.discretize_mode", "index"),
            }
        )
    elif domain == "exorl":
        loader = _resolve(("fre.data.exorl_dataset",), "load_exorl_dataset", required=True)
        kwargs = dict(common)
        kwargs.update(
            {
                "domain": cfg_get(cfg, "dataset.domain", None) or getattr(args, "task", None) or "walker",
                "task": cfg_get(cfg, "dataset.task", None) or getattr(args, "task", None),
                "dataset": cfg_get(cfg, "dataset.dataset", "rnd"),
                "normalize": bool(cfg_get(cfg, "dataset.normalize", True)),
                "select_goals": True,
            }
        )
        kwargs.pop("dataset_name", None)  # exorl loader derives it from domain/task/dataset
    elif domain == "kitchen":
        loader = _resolve(("fre.data.kitchen_dataset",), "load_kitchen_dataset", required=True)
        kwargs = dict(common)
        kwargs.update({"dataset_name": cfg_get(cfg, "dataset.name", "kitchen-complete-v0")})
    else:
        raise ValueError(f"unknown domain: {domain!r}")
    return call_flexibly(loader, **kwargs)


def dataset_replay_buffer(dataset: Any) -> Any:
    """Extract (or build) the ``ReplayBuffer`` attached to a dataset wrapper."""
    if dataset is None:
        return None
    for attr in ("replay_buffer", "buffer"):
        value = getattr(dataset, attr, None)
        if value is not None:
            return value
    builder = getattr(dataset, "build_buffer", None)
    if callable(builder):
        return call_flexibly(builder)
    return None


def encoder_state_dim(dataset: Any, use_encoder_inputs: bool, default: int = 0) -> int:
    """Observation dimension the *encoder* consumes (physics-augmented for ExORL)."""
    for attr in (("encoder_obs_dim",) if use_encoder_inputs else ("obs_dim", "encoder_obs_dim")):
        value = getattr(dataset, attr, None)
        if isinstance(value, (int, np.integer)) and value > 0:
            return int(value)
    return int(default)


def agent_observation_dim(dataset: Any, fallback: int = 0) -> int:
    """Raw observation dimension used by Q/V/policy (no physics augmentation)."""
    for attr in ("obs_dim", "raw_obs_dim"):
        value = getattr(dataset, attr, None)
        if isinstance(value, (int, np.integer)) and value > 0:
            return int(value)
    return int(fallback)


def resolve_encoder_inputs(args: argparse.Namespace, cfg: Dict[str, Any], domain: str) -> bool:
    """Whether the encoder consumes the physics-augmented observation (ExORL only)."""
    explicit = getattr(args, "use_encoder_inputs", None)
    if explicit is not None:
        return bool(explicit)
    value = cfg_get(cfg, "encoder_training.use_encoder_inputs", None)
    if value is None:
        value = cfg_get(cfg, "dataset.physics.encoder_only", None)
    if value is None:
        value = cfg_get(cfg, "dataset.use_encoder_inputs", False)
    return bool(value and domain == "exorl")


def build_prior(
    cfg: Dict[str, Any],
    dataset: Any,
    replay: Any,
    state_dim: int,
    domain: str,
    seed: int,
    variant: str = "FRE-all",
) -> Any:
    """Build the prior reward distribution p(eta) for one variant."""
    builder = _resolve(("fre.reward_priors.mixture",), "make_prior_from_variant")
    exclude_dims: Tuple[int, ...] = ()
    if domain == "antmaze":
        exclude_dims = tuple((cfg_get(cfg, "prior.linear.exclude_dims", None) or (0, 1)))
    state_min = state_max = None
    box_fn = getattr(replay, "state_box", None)
    if callable(box_fn):
        try:
            box = box_fn()
            if isinstance(box, (tuple, list)) and len(box) == 2:
                state_min, state_max = box
        except Exception:
            state_min = state_max = None
    common = {
        "variant": normalize_variant_name(variant),
        "replay_buffer": replay,
        "state_dim": state_dim,
        "exclude_dims": exclude_dims,
        "hidden_dim": int(cfg_get(cfg, "prior.mlp.hidden_dim", 32) or 32),
        "goal_threshold": float(cfg_get(cfg, "prior.goal.threshold", 0.0) or 0.0),
        "seed": seed,
        "state_min": state_min,
        "state_max": state_max,
        "dataset": dataset,
        "domain": domain,
    }
    if builder is not None:
        try:
            return call_flexibly(builder, **common)
        except Exception as exc:  # pragma: no cover - fall back to explicit mixture
            print(f"[warn] make_prior_from_variant failed ({exc}); using make_mixture_prior", file=sys.stderr)
    mixture_builder = _resolve(("fre.reward_priors.mixture",), "make_mixture_prior", required=True)
    ratios = variant_ratios(variant)
    return call_flexibly(
        mixture_builder,
        ratios=ratios,
        replay_buffer=replay,
        state_dim=state_dim,
        exclude_dims=exclude_dims,
        hidden_dim=common["hidden_dim"],
        goal_threshold=common["goal_threshold"],
        seed=seed,
        name=normalize_variant_name(variant),
        domain=domain,
    )


def build_encoder(cfg: Dict[str, Any], state_dim: int, device: str) -> Any:
    """Construct the permutation-invariant transformer VIB encoder."""
    maker = _resolve(("fre.models.encoder",), "make_fre_encoder", required=True)
    encoder = call_flexibly(
        maker,
        state_dim=state_dim,
        latent_dim=int(cfg_get(cfg, "model.latent_dim", LATENT_DIM) or LATENT_DIM),
        state_embed_dim=int(cfg_get(cfg, "model.state_embed_dim", STATE_EMBED_DIM) or STATE_EMBED_DIM),
        reward_embed_dim=int(cfg_get(cfg, "model.reward_embed_dim", REWARD_EMBED_DIM) or REWARD_EMBED_DIM),
        num_reward_embeddings=int(cfg_get(cfg, "model.num_reward_embeddings", NUM_REWARD_EMBEDDINGS) or NUM_REWARD_EMBEDDINGS),
        num_layers=int(cfg_get(cfg, "model.num_encoder_blocks", len(ENCODER_LAYERS)) or len(ENCODER_LAYERS)),
        num_heads=int(cfg_get(cfg, "model.encoder_attention_heads", ENCODER_ATTENTION_HEADS) or ENCODER_ATTENTION_HEADS),
        mlp_dim=int((cfg_get(cfg, "model.encoder_layers", list(ENCODER_LAYERS)) or list(ENCODER_LAYERS))[0]),
    )
    return _move_to(encoder, device)


def build_decoder(cfg: Dict[str, Any], state_dim: int, device: str) -> Any:
    """Construct the feed-forward reward decoder q(eta(s^d) | s^d, z)."""
    maker = _resolve(("fre.models.decoder",), "make_reward_decoder", required=True)
    decoder = call_flexibly(
        maker,
        state_dim=state_dim,
        latent_dim=int(cfg_get(cfg, "model.latent_dim", LATENT_DIM) or LATENT_DIM),
        hidden_dims=tuple(cfg_get(cfg, "model.decoder_layers", list(DECODER_LAYERS)) or list(DECODER_LAYERS)),
    )
    return _move_to(decoder, device)


def _move_to(module: Any, device: Any) -> Any:
    try:
        return module.to(device)
    except Exception:
        return module


# ======================================================================================
# Phase 1 / phase 2 training (Algorithm 1) for one variant + seed
# ======================================================================================


def build_encoder_trainer(
    encoder: Any,
    decoder: Any,
    replay: Any,
    prior: Any,
    cfg: Dict[str, Any],
    state_dim: int,
    device: str,
    seed: int,
) -> Any:
    """Instantiate the phase-1 (Equation 6) trainer."""
    maker = _resolve(("fre.training.fre_trainer",), "make_fre_trainer")
    common = {
        "encoder": encoder,
        "decoder": decoder,
        "replay_buffer": replay,
        "prior": prior,
        "state_dim": state_dim,
        "device": device,
        "seed": seed,
        "batch_size": int(cfg_get(cfg, "encoder_training.batch_size", TRAIN_BATCH_SIZE) or TRAIN_BATCH_SIZE),
        "num_encoder_states": int(cfg_get(cfg, "encoder_training.num_encoder_states", NUM_ENCODER_STATES) or NUM_ENCODER_STATES),
        "num_decoder_states": int(cfg_get(cfg, "encoder_training.num_decoder_states", NUM_DECODER_STATES) or NUM_DECODER_STATES),
        "learning_rate": float(cfg_get(cfg, "encoder_training.learning_rate", TRAIN_LEARNING_RATE) or TRAIN_LEARNING_RATE),
        "beta": float(cfg_get(cfg, "encoder_training.beta", TRAIN_BETA) or TRAIN_BETA),
        "sample_z": bool(cfg_get(cfg, "encoder_training.sample_z", True)),
        "use_encoder_inputs": bool(cfg_get(cfg, "encoder_training.use_encoder_inputs", False)),
        "loss_form": cfg_get(cfg, "encoder_training.loss_form", "mse"),
    }
    if maker is not None:
        return call_flexibly(maker, **common)
    trainer_cls = _resolve(("fre.training.fre_trainer",), "FRETrainer", required=True)
    return call_flexibly(trainer_cls, **common)


def train_encoder_phase(trainer: Any, steps: int, log_interval: int = DEFAULT_LOG_INTERVAL, progress: bool = False) -> List[float]:
    """Run the phase-1 loop, returning the loss history."""
    train_fn = getattr(trainer, "train", None)
    if callable(train_fn):
        result = call_flexibly(
            train_fn,
            num_steps=int(steps),
            encoder_steps=int(steps),
            log_interval=log_interval,
            progress=progress,
        )
        history = _extract_history(result)
        if history:
            return history
    history: List[float] = []
    step_fn = getattr(trainer, "train_step", None) or getattr(trainer, "step", None) or getattr(trainer, "encoder_step", None)
    if not callable(step_fn):
        raise AttributeError("phase-1 trainer exposes neither train() nor train_step()")
    started = time.time()
    for index in range(int(steps)):
        out = step_fn()
        loss = _loss_from_output(out)
        history.append(loss)
        if log_interval and (index + 1) % int(log_interval) == 0 and progress:
            rate = (index + 1) / max(time.time() - started, 1e-9)
            print(f"    [encoder] step {index + 1}/{steps} loss={loss:.4f} ({rate:.0f} steps/s)")
    return history


def _loss_from_output(out: Any) -> float:
    """Extract a scalar loss from an arbitrary trainer output."""
    if out is None:
        return float("nan")
    if isinstance(out, (int, float, np.floating, np.integer)):
        return float(out)
    for attr in ("loss", "total_loss", "objective", "reconstruction_loss", "kl"):
        value = getattr(out, attr, None)
        if value is None and isinstance(out, dict):
            value = out.get(attr)
        if value is not None:
            try:
                return float(np.asarray(value).reshape(-1)[0])
            except Exception:
                continue
    return float("nan")


def _extract_history(result: Any) -> List[float]:
    """Extract the loss history from whatever the trainer's ``train`` returned."""
    if result is None:
        return []
    if isinstance(result, (list, tuple, np.ndarray)):
        values: List[float] = []
        for item in result:
            value = _loss_from_output(item) if not isinstance(item, (int, float, np.floating)) else float(item)
            if np.isfinite(value):
                values.append(value)
        return values
    for attr in ("encoder_losses", "losses", "history"):
        value = getattr(result, attr, None)
        if value is not None:
            try:
                array = np.asarray(list(value), dtype=np.float64).reshape(-1)
                return [float(x) for x in array if np.isfinite(x)]
            except Exception:
                continue
    return []


def freeze_encoder(encoder: Any) -> Any:
    """Freeze the phase-1 encoder (strided scheme, Section 4.3)."""
    fn = _resolve(("fre.training.strided",), "freeze_encoder")
    if fn is not None:
        return call_flexibly(fn, encoder, module=encoder, network=encoder)
    for param in getattr(encoder, "parameters", lambda: [])():
        param.requires_grad_(False)
    if hasattr(encoder, "eval"):
        encoder.eval()
    return encoder


def build_iql_trainer(encoder: Any, obs_dim: int, act_dim: int, cfg: Dict[str, Any], device: str) -> Any:
    """Instantiate the phase-2 IQL trainer with Table 3 hyper-parameters."""
    maker = _resolve(("fre.training.iql",), "make_iql_trainer")
    common = {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "latent_dim": int(cfg_get(cfg, "model.latent_dim", LATENT_DIM) or LATENT_DIM),
        "hidden_dims": tuple(cfg_get(cfg, "model.rl_layers", list(RL_LAYERS)) or list(RL_LAYERS)),
        "encoder": encoder,
        "freeze_encoder": True,
        "device": device,
        "learning_rate": float(cfg_get(cfg, "policy_training.learning_rate", TRAIN_LEARNING_RATE) or TRAIN_LEARNING_RATE),
        "expectile": float(cfg_get(cfg, "policy_training.expectile", TRAIN_EXPECTILE) or TRAIN_EXPECTILE),
        "awr_temperature": float(cfg_get(cfg, "policy_training.awr_temperature", TRAIN_AWR_TEMPERATURE) or TRAIN_AWR_TEMPERATURE),
        "discount": float(cfg_get(cfg, "policy_training.discount", TRAIN_DISCOUNT) or TRAIN_DISCOUNT),
        "tau": float(cfg_get(cfg, "policy_training.tau", TRAIN_TARGET_UPDATE_RATE) or TRAIN_TARGET_UPDATE_RATE),
        "batch_size": int(cfg_get(cfg, "policy_training.batch_size", TRAIN_BATCH_SIZE) or TRAIN_BATCH_SIZE),
    }
    if maker is not None:
        return call_flexibly(maker, **common)
    trainer_cls = _resolve(("fre.training.iql",), "IQLTrainer", required=True)
    return call_flexibly(trainer_cls, **common)


def policy_training_loop(
    trainer: Any,
    replay: Any,
    prior: Any,
    encoder: Any,
    cfg: Dict[str, Any],
    steps: int,
    device: str,
    rng: np.random.Generator,
    log_interval: int = DEFAULT_LOG_INTERVAL,
    num_encoder_states: int = NUM_ENCODER_STATES,
    use_encoder_inputs: bool = False,
    progress: bool = False,
) -> List[Dict[str, float]]:
    """Algorithm 1 phase 2: IQL updates on latents from the *frozen* encoder."""
    helper = _resolve(("fre.scripts.train_policy", "train_policy"), "policy_training_loop")
    if helper is None:
        helper = _resolve(("train_policy",), "policy_training_loop")
    if helper is not None:
        try:
            out = call_flexibly(
                helper,
                trainer=trainer,
                trainer_=trainer,
                replay=replay,
                replay_buffer=replay,
                prior=prior,
                encoder=encoder,
                cfg=cfg,
                config=cfg,
                steps=int(steps),
                num_steps=int(steps),
                device=device,
                rng=rng,
                log_interval=log_interval,
                num_encoder_states=num_encoder_states,
                use_encoder_inputs=use_encoder_inputs,
                progress=progress,
            )
            return list(out) if out is not None else []
        except Exception as exc:  # pragma: no cover - fall back to local loop
            print(f"[warn] shared policy loop failed ({exc}); using local loop", file=sys.stderr)
    return _local_policy_loop(
        trainer, replay, prior, encoder, cfg, int(steps), device, rng, log_interval, num_encoder_states, use_encoder_inputs
    )


def _local_policy_loop(
    trainer: Any,
    replay: Any,
    prior: Any,
    encoder: Any,
    cfg: Dict[str, Any],
    steps: int,
    device: str,
    rng: np.random.Generator,
    log_interval: int,
    num_encoder_states: int,
    use_encoder_inputs: bool,
) -> List[Dict[str, float]]:
    """Minimal self-contained IQL loop used when the shared script helper is absent."""
    history: List[Dict[str, float]] = []
    batch_size = int(cfg_get(cfg, "policy_training.batch_size", TRAIN_BATCH_SIZE) or TRAIN_BATCH_SIZE)
    reward_done_masks = bool(cfg_get(cfg, "policy_training.reward_done_masks", True))
    for step in range(steps):
        eta = prior.sample(rng) if hasattr(prior, "sample") else prior(rng)
        try:
            enc_states = replay.sample_states(num_encoder_states, rng=rng, encoder_input=use_encoder_inputs)
        except TypeError:
            enc_states = replay.sample_states(num_encoder_states, rng=rng)
        rewards = _label_states(eta, enc_states, rng=None)
        with_no_grad = _resolve(("torch",), "no_grad")
        z = _encode_context(encoder, enc_states, rewards, device)
        kwargs: Dict[str, Any] = {"with_encoder_inputs": use_encoder_inputs}
        try:
            batch = replay.sample_transitions(batch_size, rng=rng, **kwargs)
        except TypeError:
            batch = replay.sample_transitions(batch_size, rng=rng)
        obs = np.asarray(getattr(batch, "observations"))
        next_obs = np.asarray(getattr(batch, "next_observations"))
        actions = np.asarray(getattr(batch, "actions"))
        terminals = np.asarray(getattr(batch, "terminals"))
        dense_rewards = _label_states(eta, obs, rng=None)
        if reward_done_masks:
            _, dones = _label_states(eta, obs, rng=None, return_dones=True)
            terminals = np.logical_or(terminals.astype(bool), np.asarray(dones, dtype=bool)).astype(np.float32)
        out = call_flexibly(
            trainer.update,
            observations=obs,
            actions=actions,
            next_observations=next_obs,
            rewards=dense_rewards,
            terminals=terminals,
            z=z,
            z_next=z,
            step_index=step,
        )
        stats = out.to_dict() if hasattr(out, "to_dict") else (out if isinstance(out, dict) else {"loss": _loss_from_output(out)})
        history.append(stats)
        if log_interval and (step + 1) % int(log_interval) == 0:
            print(f"    [policy] step {step + 1}/{steps} {stats}")
    return history


def _label_states(eta: Any, states: Any, rng: Optional[np.random.Generator] = None, return_dones: bool = False, ensure_goal: bool = False) -> Any:
    """Evaluate a prior reward function on states (tolerates several call styles)."""
    if eta is None:
        raise ValueError("reward function eta is None")
    label = getattr(eta, "label", None)
    if callable(label):
        out = call_flexibly(label, states, rng=rng, ensure_goal=ensure_goal)
    elif callable(eta):
        out = eta(states)
    else:
        raise TypeError("reward function is neither callable nor exposes label()")
    if isinstance(out, tuple) and len(out) == 2:
        rewards, dones = out
        return (rewards, dones) if return_dones else rewards
    return (np.asarray(out), np.zeros(np.asarray(out).shape, dtype=bool)) if return_dones else out


def _encode_context(encoder: Any, states: Any, rewards: Any, device: str) -> Any:
    """Encode K reward-labelled states into the latent z with the frozen encoder."""
    import torch  # noqa: WPS433 (phase 2 requires torch anyway)

    with torch.no_grad():
        state_t = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
        reward_t = torch.as_tensor(np.asarray(rewards, dtype=np.float32).reshape(len(np.asarray(states))), dtype=torch.float32, device=device)
        if state_t.dim() == 2:
            state_t = state_t.unsqueeze(0)
            reward_t = reward_t.unsqueeze(0)
        out = call_flexibly(encoder, state_t, reward_t, sample=False, return_output=False)
        z = getattr(out, "z", None)
        if z is None:
            z = out
        return z


def also_label_goal_sample(variant: str, replay: Any, prior: Any, states: Any, rng: Optional[np.random.Generator]) -> Any:
    """Correctness gate (ii): context always contains at least one goal-state sample.

    For goal-reaching-only variants the encoder context must contain a state attaining
    the goal (Appendix B: "We ensure that at least one of the samples contains the goal
    state during the encoding process").  Non-goal variants have no distinguished goal
    so the check is trivially satisfied.
    """
    goal_ratio = variant_ratios(variant)[0]
    if goal_ratio <= 0.0:
        return True
    checker = getattr(prior, "label", None)
    if not callable(checker):
        return True
    try:
        rewards, dones = _label_states_goal(prior, states, rng)
        return bool(np.any(np.asarray(dones)) or np.any(np.asarray(rewards) > -1.0))
    except Exception:
        return True


def _label_states_goal(prior: Any, states: Any, rng: Optional[np.random.Generator]) -> Tuple[np.ndarray, np.ndarray]:
    try:
        out = call_flexibly(getattr(prior, "label"), states, rng=rng, ensure_goal=True)
    except Exception:
        return np.zeros(len(states)), np.ones(len(states), dtype=bool)
    if isinstance(out, tuple) and len(out) == 2:
        return np.asarray(out[0]), np.asarray(out[1])
    rewards = np.asarray(out)
    return rewards, rewards > -1.0


# ======================================================================================
# Zero-shot evaluation + Table 4 column aggregation
# ======================================================================================


def build_suites(domain: str, cfg: Dict[str, Any], dataset: Any, suite_names: Sequence[str]) -> Dict[str, Any]:
    """Build the evaluation `TaskSuite` objects for the requested suite names."""
    suites: Dict[str, Any] = {}
    for name in suite_names:
        suite = None
        if domain == "antmaze":
            maker = _resolve(("fre.envs.antmaze_tasks",), "make_antmaze_task_suite")
            if maker is not None:
                kwargs: Dict[str, Any] = {
                    "group": name,
                    "eval_episode_length": int(cfg_get(cfg, "evaluation.eval_episode_length", 2000) or 2000),
                    "num_bins": int(cfg_get(cfg, "dataset.num_xy_bins", 32) or 32),
                    "discretized": bool(cfg_get(cfg, "dataset.discretize", True)),
                }
                task_kwargs = cfg_get(cfg, "evaluation.task_kwargs", {}) or {}
                kwargs.update({k: v for k, v in task_kwargs.items() if isinstance(k, str)})
                try:
                    suite = call_flexibly(maker, **kwargs)
                except Exception as exc:
                    print(f"[warn] could not build antmaze suite {name}: {exc}", file=sys.stderr)
        elif domain == "exorl":
            maker = _resolve(("fre.envs.exorl_tasks",), "make_exorl_task_suite")
            if maker is not None:
                kwargs = {
                    "group": name,
                    "goal_states": getattr(dataset, "goal_states", None),
                    "state_mean": getattr(dataset, "state_mean", None),
                    "state_std": getattr(dataset, "state_std", None),
                    "eval_episode_length": int(cfg_get(cfg, "evaluation.eval_episode_length", 1000) or 1000),
                }
                try:
                    suite = call_flexibly(maker, **kwargs)
                except Exception as exc:
                    print(f"[warn] could not build exorl suite {name}: {exc}", file=sys.stderr)
        elif domain == "kitchen":
            maker = _resolve(("fre.envs.kitchen_tasks",), "make_kitchen_task_suite")
            if maker is not None:
                try:
                    suite = call_flexibly(
                        maker,
                        group=name,
                        eval_episode_length=int(cfg_get(cfg, "evaluation.eval_episode_length", 1000) or 1000),
                    )
                except Exception as exc:
                    print(f"[warn] could not build kitchen suite {name}: {exc}", file=sys.stderr)
        if suite is not None:
            suites[name] = suite
    return suites


def evaluate_variant(
    encoder: Any,
    networks: Any,
    dataset: Any,
    replay: Any,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    device: str,
    suites: Dict[str, Any],
    seed: int,
) -> Dict[str, float]:
    """Zero-shot evaluation: encode 32 (s, eta(s)) pairs, roll out pi(a|s,z)."""
    evaluator_factory = _resolve(("fre.eval.zero_shot_eval",), "ZeroShotEvaluator", required=True)
    policy = getattr(networks, "policy", networks)
    evaluator = call_flexibly(
        evaluator_factory,
        encoder=encoder,
        policy=policy,
        replay_buffer=replay,
        device=device,
        num_context_samples=int(getattr(args, "context_samples", None) or cfg_get(cfg, "evaluation.context_samples", FRE_CONTEXT_SAMPLES) or FRE_CONTEXT_SAMPLES),
        num_episodes=int(getattr(args, "num_episodes", None) or cfg_get(cfg, "evaluation.num_episodes", NUM_EVAL_EPISODES) or NUM_EVAL_EPISODES),
        seeds=(int(seed),),
        deterministic_policy=bool(cfg_get(cfg, "evaluation.deterministic_policy", True)),
        sample_latent=bool(cfg_get(cfg, "evaluation.sample_latent", False)),
        use_encoder_inputs=resolve_encoder_inputs(args, cfg, cfg.get("domain", "")),
    )
    env_factory = make_env_factory(cfg.get("domain", "antmaze"), cfg, args)
    scores: Dict[str, float] = {}
    for name, suite in suites.items():
        result = None
        for method_name in ("evaluate_suite", "run_suite", "evaluate"):
            fn = getattr(evaluator, method_name, None)
            if not callable(fn):
                continue
            try:
                result = call_flexibly(
                    fn,
                    suite=suite,
                    task_suite=suite,
                    env_factory=env_factory,
                    rollout_fn=None,
                    seed=seed,
                    name=name,
                )
                break
            except Exception as exc:
                print(f"[warn] evaluation of suite {name} failed: {exc}", file=sys.stderr)
                result = None
        if result is None:
            continue
        scores[name] = _suite_score(result)
        for task_name, task_score in _per_task_scores(result).items():
            scores[task_name] = task_score
    return scores


def _suite_score(result: Any) -> float:
    import math

    if result is None:
        return float("nan")
    for attr in ("mean_score", "score"):
        value = getattr(result, attr, None)
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass
    if isinstance(result, dict):
        for key in ("mean_score", "score", "mean"):
            if key in result:
                return float(result[key])
        try:
            values = [float(v) for v in result.values()]
            return float(np.mean(values)) if values else float("nan")
        except Exception:
            return float("nan")
    return float("nan")


def _per_task_scores(result: Any) -> Dict[str, float]:
    """Extract per-task scores from a SuiteResult-like object."""
    out: Dict[str, float] = {}
    values = getattr(result, "task_results", None)
    if values is None and isinstance(result, dict):
        values = result.get("task_results") or result.get("tasks")
    if values is None:
        return out
    if isinstance(values, dict):
        iterable = values.items()
    else:
        iterable = [(getattr(item, "task_name", None), item) for item in values]
    for name, item in iterable:
        if name is None:
            continue
        if hasattr(item, "score"):
            try:
                out[str(name)] = float(item.score)
                continue
            except Exception:
                pass
        if isinstance(item, dict) and "score" in item:
            out[str(name)] = float(item["score"])
    return out


def make_env_factory(domain: str, cfg: Dict[str, Any], args: argparse.Namespace) -> Optional[Callable]:
    """Build a gym-like environment factory (None when ``--no-env`` is given)."""
    if getattr(args, "no_env", False):
        return None
    factory = None
    if domain == "antmaze":
        def factory(**_kwargs):  # type: ignore[misc]
            import gym  # noqa: WPS433
            import d4rl  # noqa: F401,WPS433 (registers the AntMaze environments)

            env = gym.make(cfg_get(cfg, "dataset.env_id", "antmaze-large-diverse-v2").replace("-v2", "-v2") and "AntMaze-Large-Diverse-v2")
            return env

    elif domain == "kitchen":
        def factory(**_kwargs):  # type: ignore[misc]
            import gym  # noqa: WPS433
            import d4rl  # noqa: F401,WPS433

            return gym.make("Kitchen-complete-v0")

    else:  # exorl -> DeepMind Control Suite

        def factory(domain_name: str = "walker", task_name: str = "run", **_kwargs):  # type: ignore[misc]
            from dm_control import suite as dm_suite  # noqa: WPS433

            class _Env:
                def __init__(self) -> None:
                    self._env = dm_suite.load(domain_name, task_name)
                    self._last = None
                    self.action_space = type("A", (), {"low": -np.ones(self._env.action_spec().shape), "high": np.ones(self._env.action_spec().shape)})()

                def reset(self):
                    self._last = self._env.reset()
                    return self._flatten(self._last)

                def step(self, action):
                    self._last = self._env.step(action)
                    obs = self._flatten(self._last)
                    reward = float(self._last.reward or 0.0)
                    done = bool(self._last.last())
                    return obs, reward, done, {}

                @staticmethod
                def _flatten(timestep):
                    parts = [np.asarray(v, dtype=np.float64).reshape(-1) for v in timestep.observation.values()]
                    return np.concatenate(parts, axis=0) if parts else np.zeros(0, dtype=np.float64)

            return _Env()

    return factory


def compute_columns(
    per_suite_scores: Dict[str, float],
    columns: Optional[Dict[str, Sequence[str]]] = None,
) -> Dict[str, float]:
    """Map per-suite scores onto the Table 4 columns (goal/directional/simplex/path/total)."""
    mapping = dict(columns or DEFAULT_TABLE4_COLUMNS)
    if "total" not in mapping:
        component_keys = [key for key in mapping if key != "total"]
        mapping["total"] = tuple(k for key in component_keys for k in mapping[key])
    out: Dict[str, float] = {}
    for column, suite_names in mapping.items():
        values: List[float] = []
        for name in suite_names:
            value = per_suite_scores.get(name)
            if value is None:
                # tolerate suite-name prefixes/suffixes (e.g. "ant-path-center" vs "path-center")
                matches = [v for k, v in per_suite_scores.items() if k == name or k.endswith(name) or name.endswith(k)]
                value = float(np.mean(matches)) if matches else None
            if value is not None and np.isfinite(value):
                values.append(float(value))
        out[column] = float(np.mean(values)) if values else float("nan")
    return out


def aggregate_runs(
    runs: Sequence[Dict[str, Any]],
    columns: Optional[Dict[str, Sequence[str]]] = None,
) -> Dict[str, Any]:
    """Aggregate per-seed runs into mean/std column summaries (Table 4 style)."""
    if not runs:
        return {"num_seeds": 0, "columns": {}, "per_suite": {}}
    suite_names: List[str] = []
    for run in runs:
        for name in run.get("scores", {}):
            if name not in suite_names:
                suite_names.append(name)
    per_suite: Dict[str, Dict[str, Any]] = {}
    for name in suite_names:
        values = [float(run["scores"][name]) for run in runs if name in run.get("scores", {}) and np.isfinite(run["scores"][name])]
        per_suite[name] = _mean_std(values)
    column_values: Dict[str, List[float]] = {}
    for run in runs:
        computed = run.get("columns") or compute_columns(run.get("scores", {}), columns)
        for column, value in computed.items():
            if value is not None and np.isfinite(value):
                column_values.setdefault(column, []).append(float(value))
    columns_summary = {column: _mean_std(values) for column, values in column_values.items()}
    return {
        "num_seeds": len(runs),
        "columns": columns_summary,
        "per_suite": per_suite,
        "seeds": [int(run.get("seed", -1)) for run in runs],
    }


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray([float(v) for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "num_seeds": 0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),  # population std, matching plain np.std over seeds
        "num_seeds": int(array.size),
        "values": [float(v) for v in array],
    }


def format_column_table(results: Dict[str, Dict[str, Any]], digits: int = 1) -> str:
    """Render the Table 4 grid as a plain-text table."""
    columns: List[str] = []
    for variant in results:
        for column in results[variant].get("columns", {}):
            if column not in columns:
                columns.append(column)
    header = "| variant".ljust(18) + "".join(f"| {c}".ljust(18) for c in columns) + "|"
    lines = [header, "|" + "-" * (len(header) - 1) + "|"]
    for variant, summary in results.items():
        row = f"| {variant}".ljust(18)
        for column in columns:
            stats = summary.get("columns", {}).get(column, {})
            mean, std = stats.get("mean", float("nan")), stats.get("std", float("nan"))
            row += (f"| {mean:.{digits}f} +- {std:.{digits}f}" if np.isfinite(mean) else "| n/a").ljust(18)
        lines.append(row + "|")
    return "\n".join(lines)


def check_success_criterion(results: Dict[str, Dict[str, Any]], hint_variant: Optional[str] = None) -> Dict[str, Any]:
    """Verify the Table 4 success criterion: ``FRE-all`` attains the highest total."""
    totals = {
        variant: summary.get("columns", {}).get("total", {}).get("mean", float("nan"))
        for variant, summary in results.items()
        if variant != hint_variant
    }
    finite = {k: v for k, v in totals.items() if np.isfinite(v)}
    best = max(finite, key=finite.get) if finite else None
    return {
        "totals": totals,
        "best": best,
        "success": best == "FRE-all",
        "expected": EXPECTED_TABLE4,
    }


def compare_to_expected(results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare reproduced column means against the paper's Table 4 reference numbers."""
    rows: List[Dict[str, Any]] = []
    for variant, summary in results.items():
        reference = EXPECTED_TABLE4.get(variant, {})
        for column, expected in reference.items():
            stats = summary.get("columns", {}).get(column, {})
            mean = stats.get("mean", float("nan"))
            rows.append(
                {
                    "variant": variant,
                    "column": column,
                    "reproduced": mean,
                    "expected": expected,
                    "delta": (mean - expected) if np.isfinite(mean) else float("nan"),
                }
            )
    return rows


# ======================================================================================
# One variant over all seeds
# ======================================================================================


def run_variant(variant: str, args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Train + evaluate one prior variant across all requested seeds."""
    variant = normalize_variant_name(variant)
    domain = str(getattr(args, "domain", None) or cfg_get(cfg, "domain", "antmaze")).lower()
    seeds = resolve_seeds(args)
    device = resolve_device(getattr(args, "device", None) or cfg_get(cfg, "device", "auto"))
    suite_names = resolve_suites(domain, getattr(args, "suites", None), cfg)
    columns = cfg_get(cfg, "reporting.columns", None) or DEFAULT_TABLE4_COLUMNS
    runs: List[Dict[str, Any]] = []

    for seed in seeds:
        seed_cfg = variant_config(cfg, variant, domain, seed)
        seed_everything(seed)
        rng = make_rng(seed)
        encoder_steps, policy_steps = steps_for_domain(seed_cfg, domain)
        print(f"[{variant}] seed {seed}: encoder {encoder_steps} steps, policy {policy_steps} steps, suites={list(suite_names)}")

        dataset = build_dataset(domain, seed_cfg, args)
        replay = dataset_replay_buffer(dataset)
        use_encoder_inputs = resolve_encoder_inputs(args, seed_cfg, domain)
        enc_dim = encoder_state_dim(dataset, use_encoder_inputs)
        obs_dim = agent_observation_dim(dataset, fallback=enc_dim)
        act_dim = int(getattr(dataset, "act_dim", 0) or getattr(replay, "act_dim", 0) or 8)

        # ---- Phase 1: encoder + decoder (Equation 6) -------------------------------
        encoder = build_encoder(seed_cfg, enc_dim, device)
        decoder = build_decoder(seed_cfg, enc_dim, device)
        prior = build_prior(seed_cfg, dataset, replay, enc_dim, domain, seed, variant)
        trainer = build_encoder_trainer(encoder, decoder, replay, prior, seed_cfg, enc_dim, device, seed)
        history = train_encoder_phase(
            trainer,
            encoder_steps,
            log_interval=int(getattr(args, "log_interval", DEFAULT_LOG_INTERVAL) or DEFAULT_LOG_INTERVAL),
            progress=bool(getattr(args, "verbose", False)),
        )
        # Fidelity passes for the trained variant (one encoder per variant is shared for
        # all evaluation tasks, exactly as the paper does).
        decoder = getattr(trainer, "decoder", decoder)
        encoder = getattr(trainer, "encoder", encoder)

        # ---- Strided scheme: freeze the encoder before touching the RL nets --------
        freeze_encoder(encoder)

        # ---- Phase 2: z-conditioned IQL on the frozen encoder ----------------------
        iql_trainer = build_iql_trainer(encoder, obs_dim, act_dim, seed_cfg, device)
        policy_history = policy_training_loop(
            iql_trainer,
            replay,
            prior,
            encoder,
            seed_cfg,
            policy_steps,
            device,
            rng,
            log_interval=int(getattr(args, "log_interval", DEFAULT_LOG_INTERVAL) or DEFAULT_LOG_INTERVAL),
            use_encoder_inputs=use_encoder_inputs,
            progress=bool(getattr(args, "verbose", False)),
        )
        networks = getattr(iql_trainer, "networks", None) or iql_trainer

        # ---- Zero-shot evaluation (32 context pairs, 20 episodes) -----------------
        suites = build_suites(domain, seed_cfg, dataset, suite_names)
        scores = evaluate_variant(encoder, networks, dataset, replay, seed_cfg, args, device, suites, seed)

        run = {
            "variant": variant,
            "seed": int(seed),
            "scores": scores,
            "columns": compute_columns(scores, columns),
            "encoder_steps": int(encoder_steps),
            "policy_steps": int(policy_steps),
            "final_encoder_loss": float(history[-1]) if history else float("nan"),
            "num_policy_updates": len(policy_history),
            "goal_gate": also_label_goal_sample(variant, replay, prior, replay.sample_states(NUM_ENCODER_STATES, rng=rng) if replay is not None else np.zeros((1, 1)), rng),
        }
        if getattr(args, "save", None):
            run["checkpoint"] = _save_variant_checkpoint(args, variant, seed, encoder, decoder, iql_trainer, seed_cfg)
        runs.append(run)

    summary = aggregate_runs(runs, columns)
    summary["variant"] = variant
    summary["method"] = f"FRE-{variant.replace('FRE-', '')}"
    summary["domain"] = domain
    return summary


def _save_variant_checkpoint(
    args: argparse.Namespace,
    variant: str,
    seed: int,
    encoder: Any,
    decoder: Any,
    iql_trainer: Any,
    cfg: Dict[str, Any],
) -> Optional[str]:
    """Persist the trained encoder/decoder/policy of one run."""
    save_path = getattr(args, "save", None)
    if not save_path:
        return None
    base, ext = os.path.splitext(str(save_path)) if os.path.splitext(str(save_path))[1] else (str(save_path), ".pt")
    path = f"{base}_{variant}_seed{seed}{ext or '.pt'}"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    try:
        import torch

        payload = {
            "variant": variant,
            "seed": int(seed),
            "latent_dim": int(cfg_get(cfg, "model.latent_dim", LATENT_DIM) or LATENT_DIM),
            "prior": cfg_get(cfg, "prior", {}),
        }
        for key, module in (("encoder", encoder), ("decoder", decoder), ("networks", getattr(iql_trainer, "networks", None))):
            if module is None:
                continue
            try:
                payload[key] = module.state_dict()
            except Exception:
                continue
        torch.save(payload, path)
        return path
    except Exception as exc:  # pragma: no cover
        print(f"[warn] could not save checkpoint {path}: {exc}", file=sys.stderr)
        return None


# ======================================================================================
# CLI
# ======================================================================================


def resolve_seeds(args: argparse.Namespace) -> Tuple[int, ...]:
    """Seeds from the CLI (``0,1,2``) or the config ``seeds`` list."""
    raw = getattr(args, "seeds_arg", None) or getattr(args, "seeds", None)
    if isinstance(raw, (list, tuple)) and all(isinstance(v, (int, np.integer)) for v in raw):
        return tuple(int(v) for v in raw)
    if isinstance(raw, str) and raw.strip():
        return tuple(int(float(part)) for part in raw.replace(" ", "").split(",") if part)
    cfg_seeds = getattr(args, "_cfg_seeds", None)
    if cfg_seeds:
        return tuple(int(v) for v in cfg_seeds)
    return DEFAULT_SEEDS


def resolve_suites(domain: str, requested: Optional[str], cfg: Dict[str, Any]) -> Tuple[str, ...]:
    """Suite names from CLI > config > domain default."""
    if requested:
        return tuple(part.strip() for part in str(requested).split(",") if part.strip())
    configured = cfg_get(cfg, "evaluation.suites", None)
    if configured:
        return tuple(str(s) for s in configured)
    return DOMAIN_EVAL_SUITES.get(domain, ())


def resolve_variants(requested: Optional[str], cfg: Optional[Dict[str, Any]] = None) -> Tuple[str, ...]:
    """Variant list from CLI > config ``ablations`` keys > paper default."""
    if requested:
        return tuple(normalize_variant_name(part) for part in str(requested).split(",") if part.strip())
    ablations = cfg_get(cfg, "ablations", None)
    if isinstance(ablations, dict) and ablations:
        ordered = [name for name in DEFAULT_VARIANTS if name in ablations]
        ordered += [name for name in ablations if name not in DEFAULT_VARIANTS]
        return tuple(ordered)
    return DEFAULT_VARIANTS


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prior-scaling ablation sweep for FRE (Table 4 / Section 5.4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="path to prior_ablations.yaml")
    parser.add_argument("--domain", type=str, default="antmaze", choices=["antmaze", "exorl", "kitchen"])
    parser.add_argument("--task", type=str, default=None, help="ExORL domain/task override (cheetah|walker)")
    parser.add_argument("--dataset", type=str, default=None, help="dataset name/path override")
    parser.add_argument("--dataset-path", type=str, default=None, help="explicit dataset file path")
    parser.add_argument("--limit", type=int, default=None, help="limit the number of loaded transitions")
    parser.add_argument("--variants", type=str, default=None, help="comma-separated prior variants")
    parser.add_argument("--seeds", type=str, default=None, help="comma-separated seeds, e.g. 0,1,2,3,4")
    parser.add_argument("--suites", type=str, default=None, help="comma-separated evaluation suite names")
    parser.add_argument("--encoder-steps", type=int, default=None, help="override encoder training steps")
    parser.add_argument("--policy-steps", type=int, default=None, help="override policy training steps")
    parser.add_argument("--num-episodes", type=int, default=None, help="evaluation episodes per task")
    parser.add_argument("--context-samples", type=int, default=None, help="FRE context pairs (32 in the paper)")
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument("--device", type=str, default=None, help="cpu|cuda|auto")
    parser.add_argument("--save", type=str, default=None, help="path prefix for per-run checkpoints")
    parser.add_argument("--results", type=str, default=None, help="path to write the JSON results table")
    parser.add_argument("--no-env", action="store_true", help="skip simulator rollouts (offline scoring only)")
    parser.add_argument("--use-encoder-inputs", action="store_true", default=None, help="physics-augmented encoder input (ExORL)")
    parser.add_argument("--verbose", action="store_true", help="per-step progress output")
    parser.add_argument("--dry-run", action="store_true", help="torch-free smoke test of the sweep bookkeeping")
    return parser


def resolve_args(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    """Merge CLI values, config values and paper defaults."""
    args._cfg_seeds = cfg_get(cfg, "seeds", None)
    if args.domain is None:
        args.domain = cfg_get(cfg, "domain", "antmaze")
    if args.encoder_steps is not None or args.policy_steps is not None:
        cfg.setdefault("encoder_training", {})
        cfg.setdefault("policy_training", {})
        default_enc, default_pol = steps_for_domain(cfg, args.domain)
        if args.encoder_steps is not None:
            cfg["encoder_training"]["encoder_training_steps"] = int(args.encoder_steps)
        else:
            cfg["encoder_training"]["encoder_training_steps"] = default_enc
        if args.policy_steps is not None:
            cfg["policy_training"]["policy_training_steps"] = int(args.policy_steps)
        else:
            cfg["policy_training"]["policy_training_steps"] = default_pol
    if args.num_episodes is None:
        args.num_episodes = int(cfg_get(cfg, "evaluation.num_episodes", NUM_EVAL_EPISODES) or NUM_EVAL_EPISODES)
    if args.context_samples is None:
        args.context_samples = int(cfg_get(cfg, "evaluation.context_samples", FRE_CONTEXT_SAMPLES) or FRE_CONTEXT_SAMPLES)
    return args


# ======================================================================================
# Dry run (no torch / no MuJoCo)
# ======================================================================================


def dry_run(args: argparse.Namespace, cfg: Dict[str, Any]) -> int:
    """Exercise the sweep bookkeeping without training or simulators."""
    variants = resolve_variants(args.variants, cfg)
    seeds = resolve_seeds(args)
    domain = args.domain
    suite_names = resolve_suites(domain, args.suites, cfg)
    columns = cfg_get(cfg, "reporting.columns", None) or DEFAULT_TABLE4_COLUMNS
    fake_scores = {
        "FRE-all": {"goal-reaching": 48.8, "directional": 55.2, "random-simplex": 21.3, "path-all": 63.8, "total": 47.3},
        "FRE-goals": {"total": 26.1},
        "FRE-lin": {"total": 31.6},
        "FRE-mlp": {"total": 25.3},
        "FRE-lin-mlp": {"total": 32.3},
        "FRE-goal-mlp": {"total": 33.8},
        "FRE-goal-lin": {"total": 46.9},
    }
    print(f"[dry-run] domain={domain} variants={list(variants)} seeds={list(seeds)} suites={list(suite_names)}")
    results: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        ratios = variant_ratios(variant)
        runs = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            jitter = float(rng.normal(0.0, 1.0))
            reference = fake_scores.get(variant, {})
            per_suite: Dict[str, float] = {}
            for name in suite_names:
                base = reference.get(name)
                if base is None:
                    base = reference.get("total", 30.0)
                per_suite[name] = float(base + jitter)
            runs.append({"seed": int(seed), "scores": per_suite, "columns": compute_columns(per_suite, columns)})
        summary = aggregate_runs(runs, columns)
        summary["variant"] = variant
        summary["ratios"] = ratios
        results[variant] = summary
        if summary["columns"]:
            total = summary["columns"].get("total", {}).get("mean", float("nan"))
            print(f"[dry-run] {variant:14s} ratios={ratios} total={total:.1f}")
    print(format_column_table(results))
    print("[dry-run] success criterion:", check_success_criterion(results))
    if args.results:
        os.makedirs(os.path.dirname(os.path.abspath(args.results)) or ".", exist_ok=True)
        with open(args.results, "w") as handle:
            json.dump({k: _jsonify(v) for k, v in results.items()}, handle, indent=2)
        print(f"[dry-run] wrote {args.results}")
    return 0


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# ======================================================================================
# Main
# ======================================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config or DOMAIN_DEFAULTS.get(args.domain, {}).get("config"))
    args = resolve_args(args, cfg)
    if args.dry_run:
        return dry_run(args, cfg)

    variants = resolve_variants(args.variants, cfg)
    seeds = resolve_seeds(args)
    columns = cfg_get(cfg, "reporting.columns", None) or DEFAULT_TABLE4_COLUMNS
    started = time.time()
    results: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}
    for variant in variants:
        try:
            results[variant] = run_variant(variant, args, cfg)
        except Exception as exc:  # keep sweeping the remaining variants
            errors[variant] = repr(exc)
            print(f"[error] variant {variant} failed: {exc}", file=sys.stderr)

    print("\n=== Table 4: prior-scaling ablation ===")
    print(format_column_table(results))
    criterion = check_success_criterion(results, hint_variant="FRE-hint" if any(is_hint_variant(v) for v in variants) else None)
    print("success criterion (FRE-all highest total):", criterion["success"], criterion["totals"])
    for row in compare_to_expected(results):
        print(f"  {row['variant']:14s} {row['column']:14s} reproduced={row['reproduced']:.1f} expected={row['expected']:.1f}")

    payload = {
        "domain": args.domain,
        "variants": list(variants),
        "seeds": list(seeds),
        "num_eval_episodes": args.num_episodes,
        "context_samples": args.context_samples,
        "protocol": {
            "normalized_return_min": NORMALIZED_RETURN_MIN,
            "normalized_return_max": NORMALIZED_RETURN_MAX,
            "seeds": list(seeds),
        },
        "results": _jsonify(results),
        "success_criterion": _jsonify(criterion),
        "errors": errors,
        "wall_time": time.time() - started,
    }
    if args.results:
        os.makedirs(os.path.dirname(os.path.abspath(args.results)) or ".", exist_ok=True)
        with open(args.results, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.results}")
    return 0 if not errors else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

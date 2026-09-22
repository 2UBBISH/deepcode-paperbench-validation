"""Strided training schedule for Functional Reward Encodings (FRE).

This module implements Algorithm 1 of the paper (Section 4.3, "Practical
Implementation"):

1. **Encoder phase** -- train only the FRE encoder/decoder by maximizing the
   variational information-bottleneck objective of Equation (6)::

       E_{L^d}[ log q_theta(eta(s^d) | s^d, z) ] - beta * KL(p_theta(z|L^e) || u(z))

   with ``K = 32`` encoder reward pairs, ``K' = 8`` decoder reward pairs, batch
   size 512, Adam at ``1e-4`` and ``beta = 0.01``.  The RL components are not
   trained during this phase.

2. **Policy phase** -- freeze the encoder (so that the mapping ``eta -> z`` is
   stationary, which the paper found important for correctly estimating
   multi-task Q values), then train ``pi(a|s,z)``, ``Q(s,a,z)``, ``V(s,z)`` with
   z-conditioned IQL using ``r = eta(s)`` from the sampled prior reward
   function.

Step budget (Appendix A, Table 3): ``150,000`` encoder steps followed by
``850,000`` policy steps for AntMaze, and ``1M`` + ``1M`` for ExORL / Kitchen.

The module is deliberately tolerant about the exact keyword signature of the
prior / agent objects it drives (:func:`call_with_supported_kwargs`) so that it
keeps working across small interface changes in the surrounding package.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # torch is required for any actual training, but the module stays importable
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch missing
    torch = None  # type: ignore
    nn = None  # type: ignore
    _HAS_TORCH = False


__all__ = [
    "FRETrainConfig",
    "FRERunner",
    "TrainingHistory",
    "run_fre",
    "run",
    "main",
    "build_arg_parser",
    # step-count helpers / constants
    "default_encoder_steps",
    "default_policy_steps",
    "default_config_for_domain",
    "resolve_steps",
    "ANTMAZE_ENCODER_STEPS",
    "ANTMAZE_POLICY_STEPS",
    "LONG_ENCODER_STEPS",
    "LONG_POLICY_STEPS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_BETA",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_NUM_ENCODER_PAIRS",
    "DEFAULT_NUM_DECODER_PAIRS",
    "DEFAULT_LATENT_DIM",
    "DEFAULT_RL_HIDDEN_DIMS",
    "DEFAULT_DECODER_HIDDEN_DIMS",
    "DEFAULT_EXPECTILE",
    "DEFAULT_AWR_TEMPERATURE",
    "DEFAULT_DISCOUNT",
    "DEFAULT_TARGET_UPDATE_RATE",
    # generic helpers (also handy for baselines / tests)
    "call_with_supported_kwargs",
    "freeze_module",
    "unfreeze_module",
    "module_is_frozen",
    "parameter_hash",
    "hash_parameters",
]


# ---------------------------------------------------------------------------
# Hyper-parameters (Appendix A, Table 3)
# ---------------------------------------------------------------------------

ANTMAZE_ENCODER_STEPS = 150_000
ANTMAZE_POLICY_STEPS = 850_000
LONG_ENCODER_STEPS = 1_000_000
LONG_POLICY_STEPS = 1_000_000

#: Domains trained with the 1M / 1M budget (Table 3).
LONG_TRAINING_DOMAINS = ("exorl", "kitchen")

DEFAULT_BATCH_SIZE = 512
DEFAULT_BETA = 0.01
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_ENCODER_PAIRS = 32
DEFAULT_NUM_DECODER_PAIRS = 8
DEFAULT_LATENT_DIM = 128
DEFAULT_RL_HIDDEN_DIMS: Tuple[int, ...] = (512, 512, 512)
DEFAULT_DECODER_HIDDEN_DIMS: Tuple[int, ...] = (512, 512, 512)
DEFAULT_ENCODER_LAYERS = 4
DEFAULT_ENCODER_MLP_DIM = 256
DEFAULT_ENCODER_ATTENTION_HEADS = 4
DEFAULT_STATE_EMBED_DIM = 64
DEFAULT_REWARD_EMBED_DIM = 64
DEFAULT_NUM_REWARD_BINS = 32

DEFAULT_EXPECTILE = 0.8
DEFAULT_AWR_TEMPERATURE = 3.0
DEFAULT_DISCOUNT = 0.88
DEFAULT_TARGET_UPDATE_RATE = 0.001

DEFAULT_LOG_INTERVAL = 1_000
DEFAULT_EVAL_INTERVAL = 0  # 0 disables periodic evaluation during training
DEFAULT_CHECKPOINT_INTERVAL = 0  # 0 disables periodic checkpointing
DEFAULT_PHASE1_LOSS_LOG_INTERVAL = 1_000


# ---------------------------------------------------------------------------
# Small generic helpers
# ---------------------------------------------------------------------------


def call_with_supported_kwargs(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` passing only the keyword arguments its signature accepts.

    ``**kwargs``-style functions receive everything.  Any ``TypeError`` raised
    *inside* ``fn`` is propagated unchanged (we only swallow signature
    mismatches, detected before the call).
    """
    if fn is None:
        raise TypeError("call_with_supported_kwargs received fn=None")

    wanted = dict(kwargs)
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins / C callables
        return fn(*args, **kwargs)

    params = sig.parameters
    accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return fn(*args, **kwargs)

    accepted = {
        name
        for name, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered = {k: v for k, v in wanted.items() if k in accepted}
    return fn(*args, **filtered)


def _batch_field(batch: Any, *names: str, default: Any = None) -> Any:
    """Fetch the first present key/attribute out of ``batch`` for ``names``."""
    if batch is None:
        return default
    for name in names:
        if isinstance(batch, Mapping) and name in batch:
            return batch[name]
        if hasattr(batch, name):
            value = getattr(batch, name)
            if value is not None:
                return value
    return default


# ---------------------------------------------------------------------------
# Freezing / stationarity check
# ---------------------------------------------------------------------------


def freeze_module(module: Any, eval_mode: bool = True) -> Any:
    """Freeze ``module`` parameters (``requires_grad_(False)``) and set eval mode."""
    if module is None:
        return module
    for param in getattr(module, "parameters", lambda: [])():
        param.requires_grad_(False)
    if eval_mode and hasattr(module, "eval"):
        module.eval()
    return module


def unfreeze_module(module: Any) -> Any:
    if module is None:
        return module
    for param in getattr(module, "parameters", lambda: [])():
        param.requires_grad_(True)
    return module


def module_is_frozen(module: Any) -> bool:
    if module is None:
        return True
    params = list(getattr(module, "parameters", lambda: [])())
    if not params:
        return True
    return not any(p.requires_grad for p in params)


def _to_bytes(tensor: Any) -> bytes:
    if _HAS_TORCH and isinstance(tensor, torch.Tensor):
        data = tensor.detach().to("cpu")
        if data.is_floating_point():
            data = data.to(torch.float64)
        return data.numpy().tobytes()
    return np.asarray(tensor).astype(np.float64).tobytes()


def parameter_hash(module: Any, max_params: Optional[int] = None) -> str:
    """Deterministic hash of a module's parameters (used to verify freezing)."""
    if module is None:
        return "none"
    hasher = hashlib.sha1()
    count = 0
    for param in getattr(module, "parameters", lambda: [])():
        hasher.update(_to_bytes(param))
        count += 1
        if max_params is not None and count >= max_params:
            break
    return hasher.hexdigest()


def hash_parameters(module: Any, max_params: Optional[int] = None) -> str:
    """Alias of :func:`parameter_hash`."""
    return parameter_hash(module, max_params=max_params)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def default_encoder_steps(domain: str) -> int:
    """Encoder-only training steps for ``domain`` (Appendix A, Table 3)."""
    domain = (domain or "antmaze").split(":")[0].strip().lower()
    return LONG_ENCODER_STEPS if domain in LONG_TRAINING_DOMAINS else ANTMAZE_ENCODER_STEPS


def default_policy_steps(domain: str) -> int:
    """Policy (IQL) training steps for ``domain`` (Appendix A, Table 3)."""
    domain = (domain or "antmaze").split(":")[0].strip().lower()
    return LONG_POLICY_STEPS if domain in LONG_TRAINING_DOMAINS else ANTMAZE_POLICY_STEPS


def resolve_steps(
    domain: str,
    encoder_steps: Optional[int] = None,
    policy_steps: Optional[int] = None,
) -> Tuple[int, int]:
    """Resolve ``(encoder_steps, policy_steps)`` with per-domain defaults."""
    enc = default_encoder_steps(domain) if encoder_steps is None else int(encoder_steps)
    pol = default_policy_steps(domain) if policy_steps is None else int(policy_steps)
    return enc, pol


@dataclass
class FRETrainConfig:
    """Configuration for the strided FRE training schedule (Algorithm 1)."""

    # ----- domain / data -------------------------------------------------
    domain: str = "antmaze"
    dataset_name: Optional[str] = None
    dataset_path: Optional[str] = None
    dataset_dir: Optional[str] = None
    env_name: Optional[str] = None
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    discretize_xy: bool = True
    normalize_states: bool = False

    # ----- schedule (Table 3) --------------------------------------------
    encoder_steps: int = ANTMAZE_ENCODER_STEPS
    policy_steps: int = ANTMAZE_POLICY_STEPS

    # ----- shared hyper-parameters (Table 3) ------------------------------
    batch_size: int = DEFAULT_BATCH_SIZE
    num_encoder_pairs: int = DEFAULT_NUM_ENCODER_PAIRS
    num_decoder_pairs: int = DEFAULT_NUM_DECODER_PAIRS
    num_decoder_pairs_rl: int = DEFAULT_NUM_DECODER_PAIRS
    beta: float = DEFAULT_BETA
    learning_rate: float = DEFAULT_LEARNING_RATE
    latent_dim: int = DEFAULT_LATENT_DIM
    encoder_layers: int = DEFAULT_ENCODER_LAYERS
    encoder_attention_heads: int = DEFAULT_ENCODER_ATTENTION_HEADS
    encoder_mlp_dim: int = DEFAULT_ENCODER_MLP_DIM
    state_embed_dim: int = DEFAULT_STATE_EMBED_DIM
    reward_embed_dim: int = DEFAULT_REWARD_EMBED_DIM
    num_reward_bins: int = DEFAULT_NUM_REWARD_BINS
    decoder_hidden_dims: Tuple[int, ...] = DEFAULT_DECODER_HIDDEN_DIMS
    normalize_rewards: bool = True
    encoder_activation: str = "gelu"
    decoder_activation: str = "relu"
    encoder_dropout: float = 0.0
    #: Sampling z from the posterior (Algorithm 1 line "z ~ p_theta(...)") vs.
    #: using the posterior mean during RL training.
    sample_latent_during_training: bool = True

    # ----- RL / IQL (Table 3) --------------------------------------------
    rl_hidden_dims: Tuple[int, ...] = DEFAULT_RL_HIDDEN_DIMS
    rl_activation: str = "relu"
    expectile: float = DEFAULT_EXPECTILE
    awr_temperature: float = DEFAULT_AWR_TEMPERATURE
    discount: float = DEFAULT_DISCOUNT
    target_update_rate: float = DEFAULT_TARGET_UPDATE_RATE
    policy_learning_rate: Optional[float] = None
    weight_decay: float = 0.0
    update_policy: bool = True
    #: Options: ``"base"`` (raw dataset observations) or ``"encoder"``
    #: (the observation representation fed to the FRE encoder, e.g. with the
    #: AntMaze XY coordinates discretized into 32 bins).
    rl_observation: str = "base"

    # ----- prior mixture (Table 3 / Appendix B) --------------------------
    prior_preset: str = "fre-all"
    prior_kwargs: Dict[str, Any] = field(default_factory=dict)

    # ----- bookkeeping ---------------------------------------------------
    seed: int = 0
    device: str = "cuda"
    output_dir: str = "experiments"
    run_name: Optional[str] = None
    log_interval: int = DEFAULT_LOG_INTERVAL
    eval_interval: int = DEFAULT_EVAL_INTERVAL
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    max_encoder_grad_norm: Optional[float] = None
    max_policy_grad_norm: Optional[float] = None
    tensorboard: bool = False
    verbose: bool = True
    phase: str = "both"  # "encoder" | "policy" | "both"

    def __post_init__(self) -> None:
        # Normalise the domain string and fill step counts with the domain
        # defaults (Table 3: 150k/850k AntMaze, 1M/1M ExORL & Kitchen).
        if self.domain:
            base = self.domain.split(":")[0].strip().lower()
            if base in LONG_TRAINING_DOMAINS and self.encoder_steps == ANTMAZE_ENCODER_STEPS:
                self.encoder_steps = LONG_ENCODER_STEPS
            if base in LONG_TRAINING_DOMAINS and self.policy_steps == ANTMAZE_POLICY_STEPS:
                self.policy_steps = LONG_POLICY_STEPS
        if isinstance(self.rl_hidden_dims, list):
            self.rl_hidden_dims = tuple(int(d) for d in self.rl_hidden_dims)
        if isinstance(self.decoder_hidden_dims, list):
            self.decoder_hidden_dims = tuple(int(d) for d in self.decoder_hidden_dims)

    # ------------------------------------------------------------------
    @classmethod
    def for_domain(cls, domain: str, **overrides: Any) -> "FRETrainConfig":
        """Build a config with Table 3 step counts for ``domain``."""
        enc, pol = resolve_steps(
            domain,
            overrides.pop("encoder_steps", None),
            overrides.pop("policy_steps", None),
        )
        return cls(domain=domain, encoder_steps=enc, policy_steps=pol, **overrides)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "FRETrainConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in dict(values).items() if k in valid})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def replace(self, **overrides: Any) -> "FRETrainConfig":
        return replace(self, **overrides)

    # ------------------------------------------------------------------
    @property
    def domain_base(self) -> str:
        return (self.domain or "antmaze").split(":")[0].strip().lower()

    @property
    def total_steps(self) -> int:
        if self.phase == "encoder":
            return int(self.encoder_steps)
        if self.phase == "policy":
            return int(self.policy_steps)
        return int(self.encoder_steps) + int(self.policy_steps)

    def resolved_run_dir(self) -> str:
        name = self.run_name or f"fre_{self.domain.replace(':', '-')}_seed{self.seed}"
        return os.path.join(self.output_dir, name)


# ---------------------------------------------------------------------------
# History bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class TrainingHistory:
    """Lightweight record of the strided schedule for reporting/debugging."""

    encoder: List[Dict[str, Any]] = field(default_factory=list)
    policy: List[Dict[str, Any]] = field(default_factory=list)
    eval: List[Dict[str, Any]] = field(default_factory=list)
    encoder_frozen_hash: Optional[str] = None

    def log_encoder(self, step: int, metrics: Mapping[str, Any]) -> None:
        self.encoder.append({"step": int(step), **{k: _to_float(v) for k, v in metrics.items()}})

    def log_policy(self, step: int, metrics: Mapping[str, Any]) -> None:
        self.policy.append({"step": int(step), **{k: _to_float(v) for k, v in metrics.items()}})

    def log_eval(self, step: int, results: Mapping[str, Any]) -> None:
        self.eval.append({"step": int(step), **dict(results)})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "encoder": self.encoder,
            "policy": self.policy,
            "eval": self.eval,
            "encoder_frozen_hash": self.encoder_frozen_hash,
        }

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2, default=str)
        return path

    # ------------------------------------------------------------------
    def latest_encoder(self, key: str, default: Any = None) -> Any:
        return self.encoder[-1].get(key, default) if self.encoder else default

    def latest_policy(self, key: str, default: Any = None) -> Any:
        return self.policy[-1].get(key, default) if self.policy else default


def _to_float(value: Any) -> Any:
    """Best-effort conversion of tensors/arrays to python floats for logging."""
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    try:
        if _HAS_TORCH and isinstance(value, torch.Tensor):
            value = value.detach().to("cpu")
            return float(value.mean()) if value.numel() > 1 else float(value)
        arr = np.asarray(value)
        if arr.dtype == object:
            return str(value)
        return float(arr.mean()) if arr.size > 1 else float(arr)
    except Exception:
        return str(value)


# ---------------------------------------------------------------------------
# Reward-function plumbing (prior batch -> r = eta(s))
# ---------------------------------------------------------------------------


def _coerce_reward_fn(obj: Any, depth: int = 0) -> Optional[Any]:
    """Find something callable as ``eta(s)`` inside an arbitrary container."""
    if obj is None or depth > 2:
        return None
    if callable(obj) and not isinstance(obj, (list, tuple, dict)):
        return obj
    if hasattr(obj, "reward") and callable(getattr(obj, "reward")):
        return obj
    if isinstance(obj, Mapping):
        for key in (
            "reward_function",
            "reward_fn",
            "functions",
            "reward_functions",
            "prior_state",
            "prior",
            "function",
        ):
            if key in obj:
                found = _coerce_reward_fn(obj[key], depth + 1)
                if found is not None:
                    return found
        for value in obj.values():
            found = _coerce_reward_fn(value, depth + 1)
            if found is not None:
                return found
        return None
    if isinstance(obj, (list, tuple)):
        if not obj:
            return None
        return _coerce_reward_fn(obj[0], depth + 1)
    # batched reward-function container (e.g. MixtureRewardFunction)
    num = getattr(obj, "num_functions", None)
    if num is not None and hasattr(obj, "at"):
        try:
            if int(num) > 0:
                return obj.at(0)
        except Exception:
            return None
    return None


def _extract_reward_fn(batch: Any, prior: Any = None) -> Optional[Any]:
    """Recover the sampled reward function whose pair produced ``z``."""
    fn = _coerce_reward_fn(batch)
    if fn is not None:
        return fn
    if prior is not None:
        fn = _coerce_reward_fn(prior)
        if fn is not None:
            return fn
    return None


def evaluate_reward_fn(reward_fn: Any, states: Any) -> np.ndarray:
    """Evaluate ``eta(states)`` returning a 1-D float32 numpy array."""
    if reward_fn is None:
        raise ValueError("No reward function available to evaluate")

    if hasattr(reward_fn, "reward") and callable(reward_fn.reward):
        values = reward_fn.reward(states)
    elif callable(reward_fn):
        values = reward_fn(states)
    else:  # pragma: no cover - defensive
        raise TypeError(f"Object of type {type(reward_fn)} is not a reward function")

    if _HAS_TORCH and isinstance(values, torch.Tensor):
        values = values.detach().to("cpu").numpy()
    values = np.asarray(values)
    if values.ndim > 1:
        # batched reward functions return (..., num_functions); keep the first
        while values.ndim > 1:
            values = values[..., 0]
    return np.asarray(values, dtype=np.float32).reshape(-1)


def reward_values_for_batch(batch: Any, prior: Any, states: Any) -> np.ndarray:
    """Compute ``r = eta(s)`` for the reward function used to build ``z``.

    Preference order:
    1. the reward-function object carried inside ``batch`` (best: it is
       guaranteed to be the same ``eta`` that produced ``z``);
    2. ``prior.evaluate_sampled(...)`` if the prior exposes it;
    3. anything callable found on the prior.
    """
    fn = _extract_reward_fn(batch, prior=None)
    if fn is not None:
        return evaluate_reward_fn(fn, states)

    if prior is not None and hasattr(prior, "evaluate_sampled"):
        for args in (
            (batch, states),
            (batch.get("prior_state") if isinstance(batch, Mapping) else None, states),
            (states,),
        ):
            try:
                values = call_with_supported_kwargs(prior.evaluate_sampled, *args)
            except Exception:
                continue
            if values is None:
                continue
            if _HAS_TORCH and isinstance(values, torch.Tensor):
                values = values.detach().to("cpu").numpy()
            values = np.asarray(values)
            while values.ndim > 1:
                values = values[..., 0]
            return np.asarray(values, dtype=np.float32).reshape(-1)

    fn = _extract_reward_fn(None if prior is None else prior)
    if fn is not None:
        return evaluate_reward_fn(fn, states)

    raise RuntimeError(
        "Could not recover the sampled reward function from the prior batch; "
        "cannot compute r = eta(s) for the IQL update."
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


class FRERunner:
    """Orchestrates the two-phase strided FRE training schedule (Algorithm 1)."""

    def __init__(
        self,
        config: Optional[FRETrainConfig] = None,
        dataset: Any = None,
        prior: Any = None,
        model: Any = None,
        agent: Any = None,
        device: Optional[Any] = None,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required to train FRE (`pip install torch`).")

        self.config = config or FRETrainConfig()
        self.dataset = dataset
        self.prior = prior
        self.model = model
        self.agent = agent

        self.device = device if device is not None else _resolve_device(self.config.device)
        self.history = TrainingHistory()
        self.rng = np.random.default_rng(self.config.seed)
        self._encoder_hash: Optional[str] = None
        self._optimizer: Any = None
        self._writer: Any = None
        self._encoder_steps_done = 0
        self._policy_steps_done = 0
        self._progress: Any = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self) -> "FRERunner":
        """Materialise dataset / prior / model / agent as needed."""
        if self.dataset is None:
            self.dataset = self.build_dataset()
        if self.prior is None:
            self.prior = self.build_prior()
        if self.model is None:
            self.model = self.build_model()
        if self.agent is None:
            self.agent = self.build_agent()
        self.to_device()
        return self

    # -- construction helpers -------------------------------------------
    def build_dataset(self) -> Any:
        from fre.data import load_dataset

        cfg = self.config
        kwargs: Dict[str, Any] = dict(cfg.dataset_kwargs)
        if cfg.dataset_path:
            kwargs["dataset_path"] = cfg.dataset_path
        if cfg.dataset_name:
            kwargs.setdefault("env_name", cfg.dataset_name)
        if cfg.env_name and cfg.domain_base == "exorl":
            kwargs["env_name"] = cfg.env_name
        if cfg.dataset_dir and cfg.domain_base == "exorl":
            kwargs["dataset_dir"] = cfg.dataset_dir
        return load_dataset(cfg.domain, **kwargs)

    def build_prior(self) -> Any:
        from fre.priors import make_reward_prior

        cfg = self.config
        states = self.dataset_states
        kwargs: Dict[str, Any] = dict(cfg.prior_kwargs)
        kwargs.setdefault("dataset", self.dataset)
        kwargs.setdefault("states", states)
        kwargs.setdefault("state_dim", int(states.shape[-1]))
        kwargs.setdefault("augment_dim", int(getattr(self.dataset, "augment_dim", 0) or 0))
        kwargs.setdefault("seed", cfg.seed)
        return call_with_supported_kwargs(
            make_reward_prior,
            domain=cfg.domain,
            preset=cfg.prior_preset,
            **kwargs,
        )

    def build_model(self) -> Any:
        from fre.models import FREModel

        cfg = self.config
        return FREModel(
            state_dim=int(self.dataset.state_dim),
            latent_dim=int(cfg.latent_dim),
            state_embed_dim=int(cfg.state_embed_dim),
            reward_embed_dim=int(cfg.reward_embed_dim),
            num_layers=int(cfg.encoder_layers),
            num_heads=int(cfg.encoder_attention_heads),
            mlp_dim=int(cfg.encoder_mlp_dim),
            num_reward_bins=int(cfg.num_reward_bins),
            decoder_hidden_dims=tuple(int(d) for d in cfg.decoder_hidden_dims),
            beta=float(cfg.beta),
            encoder_activation=cfg.encoder_activation,
            decoder_activation=cfg.decoder_activation,
            normalize_rewards=bool(cfg.normalize_rewards),
            dropout=float(cfg.encoder_dropout),
        )

    def build_agent(self) -> Any:
        from fre.models.iql import IQL

        cfg = self.config
        lr = cfg.policy_learning_rate or cfg.learning_rate
        return call_with_supported_kwargs(
            IQL,
            state_dim=int(self.dataset.state_dim),
            action_dim=int(self.dataset.action_dim),
            latent_dim=int(cfg.latent_dim),
            hidden_dims=tuple(int(d) for d in cfg.rl_hidden_dims),
            expectile=float(cfg.expectile),
            awr_temperature=float(cfg.awr_temperature),
            discount=float(cfg.discount),
            target_update_rate=float(cfg.target_update_rate),
            learning_rate=float(lr),
            activation=cfg.rl_activation,
            device=self.device,
        )

    def to_device(self) -> "FRERunner":
        for obj in (self.model, self.agent):
            if obj is not None and hasattr(obj, "to"):
                try:
                    obj.to(self.device)
                except Exception:
                    pass
        return self

    # -- accessors -------------------------------------------------------
    @property
    def dataset_states(self) -> np.ndarray:
        """All offline states as a ``(N, state_dim)`` float array."""
        if self.dataset is None:
            raise RuntimeError("Runner has no dataset; call setup()/build_dataset() first")
        states = getattr(self.dataset, "observations", None)
        if states is None:
            states = getattr(self.dataset, "states", None)
        if states is None:  # pragma: no cover - defensive
            raise AttributeError("Dataset does not expose `observations`/`states`")
        return np.asarray(states, dtype=np.float32)

    @property
    def encoder_states(self) -> np.ndarray:
        """States as seen by the FRE encoder (AntMaze XY discretized if enabled)."""
        states = self.dataset_states
        if not self.config.discretize_xy or self.config.domain_base != "antmaze":
            return states
        meta = getattr(self.dataset, "metadata", None) or {}
        if isinstance(meta, Mapping) and meta.get("discretized_xy"):
            return states  # already discretized by the loader
        try:
            from fre.data.preprocessing import discretize_antmaze_xy

            return np.asarray(discretize_antmaze_xy(states), dtype=np.float32)
        except Exception:
            return states

    def rl_states(self) -> np.ndarray:
        """Observation representation used by the z-conditioned RL networks."""
        if self.config.rl_observation == "encoder":
            return self.encoder_states
        return self.dataset_states

    # ------------------------------------------------------------------
    # Generic sampling helpers
    # ------------------------------------------------------------------
    def _sample_prior_batch(
        self,
        num_decoder_pairs: Optional[int] = None,
        device: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Sample ``K`` encoder pairs (+ ``K'`` decoder pairs) via the prior."""
        cfg = self.config
        pairs = cfg.num_decoder_pairs if num_decoder_pairs is None else num_decoder_pairs
        batch = call_with_supported_kwargs(
            self.prior.sample_batch,
            self.dataset_states,
            batch_size=int(cfg.batch_size),
            num_encoder_pairs=int(cfg.num_encoder_pairs),
            num_decoder_pairs=int(pairs),
            device=device if device is not None else self.device,
        )
        if not isinstance(batch, Mapping):  # tolerate tuple/attribute-style returns
            batch = _as_batch_mapping(batch)
        return dict(batch)

    def _sampled_encoder_tensors(
        self, batch: Mapping[str, Any]
    ) -> Tuple[Any, Any, Optional[Any]]:
        """Extract ``(encoder_states, encoder_rewards, encoder_mask)`` tensors."""
        states = _batch_field(batch, "encoder_states", "enc_states", "states_encoder")
        rewards = _batch_field(batch, "encoder_rewards", "enc_rewards", "rewards_encoder")
        mask = _batch_field(batch, "encoder_mask", "enc_mask", "mask_encoder")
        states = _to_device_tensor(states, self.device)
        rewards = _to_device_tensor(rewards, self.device)
        if mask is not None:
            mask = _to_device_tensor(mask, self.device)
        return states, rewards, mask

    def _sampled_decoder_tensors(
        self, batch: Mapping[str, Any]
    ) -> Tuple[Any, Any, Optional[Any]]:
        states = _batch_field(batch, "decoder_states", "dec_states", "states_decoder")
        rewards = _batch_field(batch, "decoder_rewards", "dec_rewards", "rewards_decoder")
        mask = _batch_field(batch, "decoder_mask", "dec_mask", "mask_decoder")
        states = _to_device_tensor(states, self.device)
        rewards = _to_device_tensor(rewards, self.device)
        if mask is not None:
            mask = _to_device_tensor(mask, self.device)
        return states, rewards, mask

    def sample_latent(self, batch: Mapping[str, Any], sample: Optional[bool] = None) -> Any:
        """Encode the sampled reward function into ``z`` (Algorithm 1)."""
        cfg = self.config
        if sample is None:
            sample = bool(cfg.sample_latent_during_training)
        states, rewards, mask = self._sampled_encoder_tensors(batch)
        with torch.no_grad():
            z = call_with_supported_kwargs(
                self.model.encode,
                states,
                rewards,
                mask=mask,
                sample=sample,
                already_normalized=False,
            )
        if not isinstance(z, torch.Tensor):
            z = _to_device_tensor(z, self.device)
        return z

    # ------------------------------------------------------------------
    # Phase 1 -- encoder + decoder (Equation 6)
    # ------------------------------------------------------------------
    def optimizer(self) -> Any:
        if self._optimizer is None:
            params = [p for p in self.model.parameters() if p.requires_grad]
            self._optimizer = torch.optim.Adam(params, lr=float(self.config.learning_rate))
        return self._optimizer

    def encoder_step(self) -> Dict[str, Any]:
        """One Equation (6) gradient step on the FRE encoder + decoder."""
        batch = self._sample_prior_batch()
        enc_states, enc_rewards, enc_mask = self._sampled_encoder_tensors(batch)
        dec_states, dec_rewards, dec_mask = self._sampled_decoder_tensors(batch)

        out = call_with_supported_kwargs(
            self.model.loss,
            enc_states,
            enc_rewards,
            dec_states,
            dec_rewards,
            encoder_mask=enc_mask,
            decoder_mask=dec_mask,
            beta=float(self.config.beta),
            sample=True,
            already_normalized=False,
            return_latent=False,
        )
        metrics = out[0] if isinstance(out, tuple) else out
        if not isinstance(metrics, Mapping):  # pragma: no cover - defensive
            metrics = {"loss": metrics}
        loss = metrics.get("loss")
        if loss is None:  # pragma: no cover - defensive
            raise RuntimeError("FRE loss dict does not contain a 'loss' entry")

        optimizer = self.optimizer()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.max_encoder_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_encoder_grad_norm)
        optimizer.step()

        return {k: _to_float(v) for k, v in metrics.items()}

    def train_encoder_phase(self, steps: Optional[int] = None, progress: bool = True) -> TrainingHistory:
        """Phase 1: train encoder+decoder only (RL components untouched)."""
        config = self.config
        steps = int(config.encoder_steps if steps is None else steps)
        if steps <= 0:
            return self.history

        self.model.train()
        if self.agent is not None and hasattr(self.agent, "eval"):
            self.agent.eval()

        iterator = self._make_progress(range(steps), desc="FRE encoder", total=steps, enable=progress)
        start = time.time()
        for step in iterator:
            metrics = self.encoder_step()
            self._encoder_steps_done += 1
            if (step + 1) % max(1, config.log_interval) == 0 or step == steps - 1:
                self.history.log_encoder(step + 1, metrics)
                self._tensorboard_log("encoder", step + 1, metrics)
                self._status(
                    iterator,
                    f"enc {step + 1}/{steps} | loss {metrics.get('loss', float('nan')):.4f} "
                    f"| mse {metrics.get('reconstruction_loss', metrics.get('mse', float('nan'))):.4f} "
                    f"| kl {metrics.get('kl', float('nan')):.4f}",
                )
                if config.verbose and step == steps - 1:
                    print(
                        f"[encoder] {steps} steps in {time.time() - start:.1f}s | "
                        f"final loss {metrics.get('loss', float('nan')):.4f}"
                    )
            if config.checkpoint_interval and (step + 1) % config.checkpoint_interval == 0:
                self.save_checkpoint(os.path.join(config.resolved_run_dir(), "checkpoint_encoder.pt"))

        return self.history

    # ------------------------------------------------------------------
    # Phase 2 -- frozen encoder + z-conditioned IQL
    # ------------------------------------------------------------------
    def freeze_encoder(self, verify: bool = True) -> str:
        """Freeze the encoder so that ``eta -> z`` stays stationary."""
        encoder = getattr(self.model, "encoder", None)
        freeze_module(encoder, eval_mode=True)
        digest = parameter_hash(encoder)
        self._encoder_hash = digest
        self.history.encoder_frozen_hash = digest
        if verify:
            assert module_is_frozen(encoder), "Encoder parameters are still trainable"
            assert parameter_hash(encoder) == digest, "Encoder hash changed while freezing"
        return digest

    def assert_encoder_frozen(self, message: str = "") -> None:
        """Verify the encoder has not been updated since :meth:`freeze_encoder`."""
        encoder = getattr(self.model, "encoder", None)
        current = parameter_hash(encoder)
        expected = self._encoder_hash or self.history.encoder_frozen_hash
        if expected is not None and current != expected:
            raise AssertionError(
                f"Encoder parameters changed during policy training{f' ({message})' if message else ''}"
            )

    def policy_step(self) -> Dict[str, Any]:
        """One IQL update with ``r = eta(s)`` for a freshly sampled ``eta``."""
        config = self.config
        batch = self._sample_prior_batch(num_decoder_pairs=config.num_decoder_pairs_rl)
        z = self.sample_latent(batch)

        transitions = self.sample_transitions()
        observations = _batch_field(transitions, "observations", "states", "obs")
        rewards = reward_values_for_batch(batch, self.prior, np.asarray(observations))

        tb = _as_transition_mapping(transitions, self.device)
        tb["rewards"] = _to_device_tensor(rewards, self.device)

        metrics = call_with_supported_kwargs(
            self.agent.update_from_batch,
            tb,
            z=z,
            rewards=tb["rewards"],
            update_policy=config.update_policy,
        )
        if not isinstance(metrics, Mapping):  # pragma: no cover - defensive
            metrics = {"loss": metrics}
        return {k: _to_float(v) for k, v in metrics.items()}

    def sample_transitions(self) -> Any:
        """Sample a batch of ``(s, a, s', done)`` from the offline dataset."""
        config = self.config
        try:
            return call_with_supported_kwargs(
                self.dataset.sample_transitions,
                int(config.batch_size),
                rng=self.rng,
                device=self.device,
            )
        except Exception:
            indices = call_with_supported_kwargs(
                self.dataset.sample_indices, int(config.batch_size), rng=self.rng
            )
            return _DatasetTransitionView(self.dataset, np.asarray(indices))

    def train_policy_phase(self, steps: Optional[int] = None, progress: bool = True) -> TrainingHistory:
        """Phase 2: frozen encoder, train ``pi/Q/V`` with z-conditioned IQL."""
        config = self.config
        steps = int(config.policy_steps if steps is None else steps)
        if steps <= 0:
            return self.history

        if self._encoder_hash is None:
            self.freeze_encoder()

        self.model.eval()
        if hasattr(self.agent, "train"):
            self.agent.train()

        iterator = self._make_progress(range(steps), desc="FRE policy", total=steps, enable=progress)
        start = time.time()
        for step in iterator:
            metrics = self.policy_step()
            self._policy_steps_done += 1
            if (step + 1) % max(1, config.log_interval) == 0 or step == steps - 1:
                self.history.log_policy(step + 1, metrics)
                self._tensorboard_log("policy", step + 1, metrics)
                self._status(
                    iterator,
                    f"pol {step + 1}/{steps} | q {metrics.get('q_loss', float('nan')):.4f} "
                    f"| v {metrics.get('v_loss', float('nan')):.4f} "
                    f"| Q {metrics.get('q_mean', float('nan')):.3f}",
                )
                # Keep the frozen-encoder guarantee honest during training.
                self.assert_encoder_frozen(f"step {step + 1}")
            if config.checkpoint_interval and (step + 1) % config.checkpoint_interval == 0:
                self.save_checkpoint(os.path.join(config.resolved_run_dir(), "checkpoint_policy.pt"))

        self.assert_encoder_frozen("end of policy phase")
        if config.verbose:
            print(
                f"[policy] {steps} steps in {time.time() - start:.1f}s | "
                f"final q_loss {self.history.latest_policy('q_loss', float('nan')):.4f} "
                f"| v_loss {self.history.latest_policy('v_loss', float('nan')):.4f}"
            )
        return self.history

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------
    def train(self, progress: bool = True) -> TrainingHistory:
        """Run the full strided schedule (Algorithm 1)."""
        self.setup()
        config = self.config
        run_dir = config.resolved_run_dir()
        if config.verbose:
            print(
                f"[FRE] domain={config.domain} device={self.device} "
                f"state_dim={self.dataset.state_dim} action_dim={self.dataset.action_dim} "
                f"encoder_steps={config.encoder_steps} policy_steps={config.policy_steps} "
                f"phase={config.phase}"
            )
        os.makedirs(run_dir, exist_ok=True)

        if config.phase in ("encoder", "both"):
            self.train_encoder_phase(progress=progress)
        if config.phase in ("policy", "both"):
            self.freeze_encoder()
            self.train_policy_phase(progress=progress)

        self.history.save(os.path.join(run_dir, "history.json"))
        self.save_checkpoint(os.path.join(run_dir, "checkpoint.pt"))
        return self.history

    #: Alias used by ``main.py`` and the shell scripts.
    def run(self, progress: bool = True) -> TrainingHistory:
        return self.train(progress=progress)

    # ------------------------------------------------------------------
    # Evaluation / checkpointing / bookkeeping
    # ------------------------------------------------------------------
    def evaluate(
        self,
        task_set: Optional[str] = None,
        dataset: Any = None,
        eval_config: Any = None,
        seeds: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Zero-shot evaluation of the trained agent (Section 5.2 / Appendix C)."""
        from fre.evaluation.evaluate import EvalConfig, evaluate_fre

        cfg = eval_config or EvalConfig(domain=self.config.domain)
        kwargs: Dict[str, Any] = {}
        if task_set is not None:
            kwargs["task_set"] = task_set
        if seeds is not None:
            kwargs["seeds"] = list(seeds)
        results = call_with_supported_kwargs(
            evaluate_fre,
            self.model,
            self.agent,
            domain=self.config.domain,
            dataset=dataset if dataset is not None else self.dataset,
            cfg=cfg,
            device=self.device,
            **kwargs,
        )
        self.history.log_eval(self._policy_steps_done, {"summary": results})
        return results

    def save_checkpoint(self, path: str) -> str:
        directory = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(directory, exist_ok=True)
        payload: Dict[str, Any] = {
            "config": self.config.to_dict(),
            "encoder_steps_done": self._encoder_steps_done,
            "policy_steps_done": self._policy_steps_done,
            "encoder_frozen_hash": self._encoder_hash,
            "history": self.history.to_dict(),
        }
        if self.model is not None and hasattr(self.model, "state_dict"):
            payload["model"] = self.model.state_dict()
        if self.agent is not None and hasattr(self.agent, "state_dict"):
            try:
                payload["agent"] = self.agent.state_dict()
            except Exception:
                payload["agent"] = None
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: str, load_agent: bool = True) -> "FRERunner":
        payload = torch.load(path, map_location=self.device)
        if "config" in payload:
            self.config = FRETrainConfig.from_dict(payload["config"])
        if self.model is None:
            self.model = self.build_model()
            self.to_device()
        if payload.get("model") is not None:
            self.model.load_state_dict(payload["model"])
        if load_agent and payload.get("agent") is not None:
            if self.agent is None:
                self.agent = self.build_agent()
                self.to_device()
            try:
                self.agent.load_state_dict(payload["agent"])
            except Exception:
                pass
        self._encoder_steps_done = int(payload.get("encoder_steps_done", 0))
        self._policy_steps_done = int(payload.get("policy_steps_done", 0))
        self._encoder_hash = payload.get("encoder_frozen_hash")
        if isinstance(payload.get("history"), Mapping):
            hist = TrainingHistory(
                encoder=list(payload["history"].get("encoder", [])),
                policy=list(payload["history"].get("policy", [])),
                eval=list(payload["history"].get("eval", [])),
                encoder_frozen_hash=payload["history"].get("encoder_frozen_hash"),
            )
            self.history = hist
        return self

    # -- internal utilities ---------------------------------------------
    def _make_progress(self, iterable: Any, desc: str, total: int, enable: bool = True) -> Any:
        if not enable:
            return iterable
        try:
            from tqdm.auto import tqdm

            self._progress = tqdm(iterable, desc=desc, total=total)
            return self._progress
        except Exception:  # pragma: no cover - tqdm missing
            return iterable

    def _status(self, iterator: Any, message: str) -> None:
        if hasattr(iterator, "set_postfix_str"):
            try:
                iterator.set_postfix_str(message)
                return
            except Exception:
                pass
        if self.config.verbose:
            print(f"[FRE] {message}")

    def _tensorboard_log(self, tag: str, step: int, metrics: Mapping[str, Any]) -> None:
        if not self.config.tensorboard:
            return
        try:
            if self._writer is None:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(self.config.resolved_run_dir())
            for key, value in metrics.items():
                scalar = _to_float(value)
                if isinstance(scalar, (int, float)):
                    self._writer.add_scalar(f"{tag}/{key}", scalar, step)
        except Exception:
            self.config.tensorboard = False


# ---------------------------------------------------------------------------
# Batch / transition adapters
# ---------------------------------------------------------------------------


def _as_batch_mapping(batch: Any) -> Dict[str, Any]:
    """Convert an attribute-style / tuple batch into a mapping."""
    if isinstance(batch, Mapping):
        return dict(batch)
    if hasattr(batch, "as_dict"):
        return dict(batch.as_dict())
    if hasattr(batch, "__dict__"):
        return {k: v for k, v in vars(batch).items() if not k.startswith("_")}
    raise TypeError(f"Cannot interpret prior batch of type {type(batch)}")


class _DatasetTransitionView:
    """Fallback transition container built directly from dataset indices."""

    def __init__(self, dataset: Any, indices: np.ndarray) -> None:
        obs = np.asarray(dataset.observations, dtype=np.float32)[indices]
        actions = np.asarray(dataset.actions, dtype=np.float32)[indices]
        rewards = np.asarray(getattr(dataset, "rewards", np.zeros(len(indices))), dtype=np.float32)[indices]
        next_obs = obs  # replaced below when possible
        dones = np.zeros(len(indices), dtype=np.float32)
        if hasattr(dataset, "next_observations") and dataset.next_observations is not None:
            next_obs = np.asarray(dataset.next_observations, dtype=np.float32)[indices]
        elif hasattr(dataset, "future_state_indices"):
            try:
                nxt = np.asarray(dataset.future_state_indices(indices, min_offset=1), dtype=np.int64)
                next_obs = np.asarray(dataset.observations, dtype=np.float32)[nxt]
                dones = (np.asarray(nxt) == np.asarray(indices)).astype(np.float32)
            except Exception:
                next_obs = obs
        if getattr(dataset, "dones", None) is not None:
            dones = np.asarray(dataset.dones, dtype=np.float32)[indices]
        self.observations = obs
        self.actions = actions
        self.rewards = rewards
        self.next_observations = next_obs
        self.dones = dones
        self.indices = np.asarray(indices)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "dones": self.dones,
            "indices": self.indices,
        }


def _as_transition_mapping(transitions: Any, device: Any = None) -> Dict[str, Any]:
    """Convert a transition batch into a dict of device tensors."""
    if isinstance(transitions, Mapping):
        raw = dict(transitions)
    elif hasattr(transitions, "as_dict"):
        raw = dict(transitions.as_dict())
    elif hasattr(transitions, "to") and hasattr(transitions, "observations"):
        raw = {
            "observations": transitions.observations,
            "actions": transitions.actions,
            "rewards": getattr(transitions, "rewards", None),
            "next_observations": getattr(transitions, "next_observations", None),
            "dones": getattr(transitions, "dones", None),
            "indices": getattr(transitions, "indices", None),
        }
    elif hasattr(transitions, "__dict__"):
        raw = {k: v for k, v in vars(transitions).items() if not k.startswith("_")}
    else:  # pragma: no cover - defensive
        raise TypeError(f"Cannot interpret transition batch of type {type(transitions)}")

    out: Dict[str, Any] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if isinstance(value, np.ndarray) or (_HAS_TORCH and isinstance(value, torch.Tensor)):
            out[key] = _to_device_tensor(value, device)
        else:
            out[key] = value
    # rewards must exist for the IQL update
    out.setdefault("rewards", None)
    return out


def _to_device_tensor(value: Any, device: Any = None) -> Any:
    if value is None or not _HAS_TORCH:
        return value
    if isinstance(value, torch.Tensor):
        return value.to(device) if device is not None else value
    tensor = torch.as_tensor(np.asarray(value))
    if tensor.is_floating_point():
        tensor = tensor.float()
    return tensor.to(device) if device is not None else tensor


def _resolve_device(device: Any) -> Any:
    if not _HAS_TORCH:
        return None
    if isinstance(device, torch.device):
        return device
    spec = str(device or "cuda")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        spec = "cpu"
    return torch.device(spec)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def run_fre(
    domain: str = "antmaze",
    config: Optional[FRETrainConfig] = None,
    dataset: Any = None,
    prior: Any = None,
    model: Any = None,
    agent: Any = None,
    progress: bool = True,
    **config_overrides: Any,
) -> FRERunner:
    """Train FRE on ``domain`` following the strided schedule (Algorithm 1).

    Returns the (set-up) :class:`FRERunner`, whose ``model``/``agent`` can be
    handed to :mod:`fre.evaluation` for zero-shot evaluation.
    """
    if config is None:
        config = FRETrainConfig.for_domain(domain, **config_overrides)
    elif config_overrides:
        config = config.replace(**config_overrides)

    runner = FRERunner(config=config, dataset=dataset, prior=prior, model=model, agent=agent)
    runner.train(progress=progress)
    return runner


#: Short alias.
run = run_fre


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="run_fre",
        description="Train Functional Reward Encodings (FRE) with the strided schedule (Algorithm 1).",
    )
    parser.add_argument("--domain", type=str, default="antmaze",
                        help="antmaze | exorl:walker | exorl:cheetah | kitchen")
    parser.add_argument("--dataset-name", type=str, default=None, help="D4RL env name override")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to a preprocessed dataset")
    parser.add_argument("--dataset-dir", type=str, default=None, help="ExORL dataset directory")
    parser.add_argument("--env-name", type=str, default=None, help="ExORL env name, e.g. walker-run")
    parser.add_argument("--encoder-steps", type=int, default=None, help="Phase 1 steps (default 150k / 1M)")
    parser.add_argument("--policy-steps", type=int, default=None, help="Phase 2 steps (default 850k / 1M)")
    parser.add_argument("--phase", type=str, default="both", choices=["encoder", "policy", "both"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-encoder-pairs", type=int, default=DEFAULT_NUM_ENCODER_PAIRS)
    parser.add_argument("--num-decoder-pairs", type=int, default=DEFAULT_NUM_DECODER_PAIRS)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--prior-preset", type=str, default="fre-all")
    parser.add_argument("--expectile", type=float, default=DEFAULT_EXPECTILE)
    parser.add_argument("--awr-temperature", type=float, default=DEFAULT_AWR_TEMPERATURE)
    parser.add_argument("--discount", type=float, default=DEFAULT_DISCOUNT)
    parser.add_argument("--target-update-rate", type=float, default=DEFAULT_TARGET_UPDATE_RATE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--no-discretize-xy", action="store_true")
    parser.add_argument("--rl-observation", type=str, default="base", choices=["base", "encoder"])
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def config_from_args(args: Any) -> FRETrainConfig:
    """Turn parsed CLI args into a :class:`FRETrainConfig`."""
    enc, pol = resolve_steps(args.domain, args.encoder_steps, args.policy_steps)
    return FRETrainConfig(
        domain=args.domain,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        dataset_dir=args.dataset_dir,
        env_name=args.env_name,
        encoder_steps=enc,
        policy_steps=pol,
        phase=args.phase,
        batch_size=args.batch_size,
        num_encoder_pairs=args.num_encoder_pairs,
        num_decoder_pairs=args.num_decoder_pairs,
        beta=args.beta,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        prior_preset=args.prior_preset,
        expectile=args.expectile,
        awr_temperature=args.awr_temperature,
        discount=args.discount,
        target_update_rate=args.target_update_rate,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        run_name=args.run_name,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        discretize_xy=not args.no_discretize_xy,
        rl_observation=args.rl_observation,
        tensorboard=args.tensorboard,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)

    runner = FRERunner(config=config)
    if args.resume:
        runner.setup()
        runner.load_checkpoint(args.resume)
        print(f"[FRE] resumed from {args.resume}")
    runner.train(progress=not args.no_progress)
    run_dir = config.resolved_run_dir()
    print(f"[FRE] finished; artefacts in {run_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

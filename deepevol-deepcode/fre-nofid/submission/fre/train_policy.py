"""Phase-2 driver for FRE: z-conditioned IQL policy training with a frozen encoder.

Implements the strided training schedule of Algorithm 1 (Sec 4.3 / App A):

* **Phase 1** (``fre/train_encoder.py``) trains the transformer encoder ``q_theta(z|context)``
  and the reward decoder ``q_theta(eta(s)|s,z)`` with the information-bottleneck objective
  ``MSE + beta * KL`` (beta = 0.01).
* **Phase 2** (this module) *freezes* the encoder and trains IQL networks ``Q(s,a,z)``,
  ``V(s,z)`` and ``pi(a|s,z)`` where ``z`` is concatenated to the observations.

Every IQL iteration:
    1. sample a transition batch from the offline replay buffer,
    2. sample one reward function ``eta`` per transition from the unsupervised prior
       ``p(eta)`` (0.33 goal-reaching / 0.33 linear / 0.33 MLP),
    3. sample K = 32 ``(state, reward)`` context pairs per reward function and encode
       them with the frozen encoder to obtain ``z``,
    4. set ``r = eta(s)`` on the transitions and take an IQL update with discount 0.88,
       expectile 0.8 and AWR temperature 3.0.

Freezing the encoder keeps the ``eta -> z`` mapping stationary, which is what makes
multitask temporal-difference learning over the reward-function mixture stable.

Usage
-----
::

    python -m fre.train_policy --domain antmaze --encoder-checkpoint runs/antmaze/encoder.pt
    python -m fre.train_policy --domain exorl --domain-name walker --steps 1000000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - torch is a hard requirement in practice
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False


# --------------------------------------------------------------------------------------
# Defensive imports (the module must work as ``python -m fre.train_policy`` or standalone)
# --------------------------------------------------------------------------------------
def _import_attr(module_names: Sequence[str], attr: str, default: Any = None) -> Any:
    """Best-effort import of ``attr`` from the first importable module name."""
    for name in module_names:
        try:
            module = __import__(name, fromlist=[attr])
            return getattr(module, attr)
        except Exception:
            continue
    return default


FREEncoder = _import_attr(
    ["fre.fre.encoder", "fre.encoder", "encoder"], "FREEncoder"
)
FREDecoder = _import_attr(["fre.fre.decoder", "fre.decoder", "decoder"], "FREDecoder")
LatentPolicyBundle = _import_attr(
    ["fre.fre.latent_policy", "fre.latent_policy", "latent_policy"], "LatentPolicyBundle"
)
DEFAULT_LATENT_DIM = _import_attr(
    ["fre.fre.latent_policy", "fre.latent_policy", "latent_policy"],
    "DEFAULT_LATENT_DIM",
    128,
)
IQLTrainer = _import_attr(["fre.rl.iql", "rl.iql", "iql"], "IQLTrainer")
IQLConfig = _import_attr(["fre.rl.iql", "rl.iql", "iql"], "IQLConfig")
MixturePrior = _import_attr(
    ["fre.rewards.prior", "rewards.prior", "prior"], "MixturePrior"
)
make_mixture_prior = _import_attr(
    ["fre.rewards.prior", "rewards.prior", "prior"], "make_mixture_prior"
)
make_prior = _import_attr(["fre.rewards.prior", "rewards.prior", "prior"], "make_prior")
make_fre_prior = _import_attr(
    ["fre.rewards.prior", "rewards.prior", "prior"], "make_fre_prior"
)
PRIOR_CONTEXT_SIZE = _import_attr(
    ["fre.rewards.prior", "rewards.prior", "prior"], "CONTEXT_SIZE", 32
)
PRIOR_DECODER_SIZE = _import_attr(
    ["fre.rewards.prior", "rewards.prior", "prior"], "DECODER_SIZE", 8
)

# Logging helpers (soft import so the file can be inspected without the package).
seed_everything = _import_attr(
    ["fre.utils.logging", "fre.utils", "utils.logging"], "seed_everything", None
)
MetricTracker = _import_attr(
    ["fre.utils.logging", "fre.utils", "utils.logging"], "MetricTracker", None
)
write_json = _import_attr(["fre.utils.logging", "fre.utils", "utils.logging"], "write_json", None)
get_logger = _import_attr(["fre.utils.logging", "fre.utils", "utils.logging"], "get_logger", None)
progress = _import_attr(["fre.utils.logging", "fre.utils", "utils.logging"], "progress", None)

if MetricTracker is None:  # pragma: no cover - minimal fallback

    class MetricTracker:  # type: ignore
        __slots__ = ("_sums", "_counts", "window")

        def __init__(self, names: Optional[Sequence[str]] = None, window: int = 100) -> None:
            self._sums: Dict[str, float] = {}
            self._counts: Dict[str, int] = {}
            self.window = window

        def update(self, values: Optional[Mapping[str, float]] = None, **kwargs: float) -> None:
            merged: Dict[str, float] = {}
            if values:
                merged.update(values)
            merged.update(kwargs)
            for key, value in merged.items():
                try:
                    value = float(np.asarray(value).reshape(-1)[0])
                except Exception:
                    continue
                self._sums[key] = self._sums.get(key, 0.0) + value
                self._counts[key] = self._counts.get(key, 0) + 1

        def mean(self, key: Optional[str] = None) -> Any:
            if key is None:
                return {k: v / max(1, self._counts[k]) for k, v in self._sums.items()}
            if key not in self._sums:
                return float("nan")
            return self._sums[key] / max(1, self._counts[key])

        def reset(self) -> None:
            self._sums.clear()
            self._counts.clear()

        def as_dict(self, recent: bool = False) -> Dict[str, float]:
            return dict(self.mean())


if write_json is None:  # pragma: no cover

    def write_json(path: str, obj: Any) -> str:  # type: ignore
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, default=str)
        return path


if progress is None:  # pragma: no cover

    def progress(iterable=None, total=None, desc=None, **kwargs):  # type: ignore
        if iterable is None:
            return range(total or 0)
        return iterable


if get_logger is None:  # pragma: no cover

    def get_logger(name: str = "fre", level: int = logging.INFO, log_file: Optional[str] = None):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(handler)
        logger.setLevel(level)
        return logger


LOGGER = get_logger("fre.train_policy")


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
DEFAULT_HIDDEN_SIZES: Tuple[int, int, int] = (512, 512, 512)
DEFAULT_CONTEXT_SIZE = int(PRIOR_CONTEXT_SIZE or 32)
DEFAULT_DECODER_SIZE = int(PRIOR_DECODER_SIZE or 8)

# Algorithm 1: Phase 2 runs 850k steps for AntMaze (Phase 1 = 150k) and 1M steps for
# ExORL / Kitchen (Phase 1 = 1M).
DEFAULT_STEPS: Dict[str, int] = {
    "antmaze": 850_000,
    "exorl": 1_000_000,
    "kitchen": 1_000_000,
}

DEFAULT_DATASETS: Dict[str, str] = {
    "antmaze": "antmaze-large-diverse-v2",
    "kitchen": "kitchen-complete-v0",
    "exorl": "rnd",
}

# Action dimensionality fallbacks by domain (used only when the buffer cannot tell us).
_ACTION_DIM_FALLBACK: Dict[str, int] = {
    "antmaze": 8,
    "walker": 6,
    "cheetah": 6,
    "kitchen": 9,
}

MODEL_ALIASES = ("model", "state_dict", "encoder", "encoder_state_dict", "policy")


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass
class PolicyTrainConfig:
    """Hyperparameters for Phase-2 z-conditioned IQL training (Sec 4.3, App A)."""

    domain: str = "antmaze"
    domain_name: Optional[str] = None  # e.g. "walker" / "cheetah" for ExORL

    # Schedule (Algorithm 1)
    steps: int = 0
    batch_size: int = 512
    lr: float = 1e-4

    # IQL hyperparameters (Sec 4.3)
    discount: float = 0.88
    expectile: float = 0.8
    awr_temperature: float = 3.0
    target_update_rate: float = 0.001
    max_grad_norm: float = 1.0

    # Latent / context
    latent_dim: int = int(DEFAULT_LATENT_DIM or 128)
    context_size: int = DEFAULT_CONTEXT_SIZE
    decoder_size: int = DEFAULT_DECODER_SIZE
    hidden_sizes: Tuple[int, int, int] = DEFAULT_HIDDEN_SIZES
    reward_resample_interval: int = 1

    # Model init
    num_blocks: int = 4
    num_heads: int = 4
    mlp_dim: int = 256
    dropout: float = 0.0

    # Data
    dataset: Optional[str] = None
    data_root: Optional[str] = None
    encoder_checkpoint: Optional[str] = None
    pretrain_encoder: bool = False
    pretrain_steps: int = 0
    families: Optional[Sequence[str]] = None
    exclude_xy: Optional[bool] = None
    ablation: Optional[str] = None

    # Bookkeeping
    seed: int = 0
    device: Optional[str] = None
    output_dir: str = "runs"
    run_name: Optional[str] = None
    log_interval: int = 1_000
    eval_interval: int = 50_000
    checkpoint_interval: int = 100_000
    eval_episodes: int = 20
    eval_seeds: int = 5
    log_file: Optional[str] = None
    resume: Optional[str] = None
    dry_run: bool = False
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.domain_name is None:
            self.domain_name = self.domain
        if isinstance(self.hidden_sizes, list):
            self.hidden_sizes = tuple(int(h) for h in self.hidden_sizes)
        if not self.steps:
            self.steps = DEFAULT_STEPS.get(self.domain, 1_000_000)
        if self.dataset is None:
            self.dataset = DEFAULT_DATASETS.get(self.domain)
        if self.exclude_xy is None:
            self.exclude_xy = self.domain == "antmaze"
        if self.dry_run:
            self.steps = min(self.steps, 5)
        if self.device is None:
            self.device = "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

    @property
    def exp_name(self) -> str:
        if self.run_name:
            return self.run_name
        return f"{self.domain}_{self.domain_name}"

    @property
    def run_dir(self) -> str:
        return os.path.join(self.output_dir, self.exp_name)

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["hidden_sizes"] = list(self.hidden_sizes)
        data["families"] = list(self.families) if self.families else None
        return data


# --------------------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------------------
def build_replay_buffer(cfg: PolicyTrainConfig, verbose: bool = False) -> Any:
    """Load the offline dataset for ``cfg.domain`` and wrap it in a FRE replay buffer."""
    if cfg.domain == "exorl":
        from fre.data.exorl_loader import load_exorl_dataset, to_replay_buffer as exorl_to_buffer

        dataset = load_exorl_dataset(
            cfg.domain_name or "walker",
            root=cfg.data_root,
            append_physics_to_obs=True,
            normalise=True,
            verbose=verbose,
        )
        return exorl_to_buffer(dataset, device="cpu", seed=cfg.seed)

    from fre.data.d4rl_loader import load_antmaze, load_kitchen_multitask
    from fre.data.d4rl_loader import to_replay_buffer as d4rl_to_buffer

    if cfg.domain == "kitchen":
        payload = load_kitchen_multitask(cfg.dataset or "kitchen-complete-v0")
        dataset = payload.get("dataset", payload) if isinstance(payload, Mapping) else payload
    else:
        dataset = load_antmaze(cfg.dataset or "antmaze-large-diverse-v2")

    return d4rl_to_buffer(dataset, device="cpu", seed=cfg.seed)


def _buffer_attr(buffer: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a buffer dict/object, falling back to ``default``."""
    if isinstance(buffer, Mapping) and name in buffer:
        return buffer[name]
    if hasattr(buffer, name):
        return getattr(buffer, name)
    return default


def _infer_dims(buffer: Any) -> Tuple[int, int]:
    """Infer ``(obs_dim, action_dim)`` from a replay buffer."""
    obs_dim = _buffer_attr(buffer, "obs_dim", None)
    action_dim = _buffer_attr(buffer, "action_dim", None)
    if obs_dim is None:
        obs = _buffer_attr(buffer, "observations", None)
        if obs is not None:
            obs_dim = int(np.asarray(obs).shape[-1])
    if action_dim is None:
        acts = _buffer_attr(buffer, "actions", None)
        if acts is not None:
            action_dim = int(np.asarray(acts).shape[-1])
    return int(obs_dim or 0), int(action_dim or 0)


def _sample_buffer_batch(buffer: Any, batch_size: int, device: str) -> Dict[str, Any]:
    """Sample a transition batch, tolerating several ``ReplayBuffer.sample`` signatures."""
    attempts = (
        dict(device="cpu"),
        dict(device="cpu", mask_terminal_next_obs=True),
        dict(),
    )
    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            batch = buffer.sample(batch_size, **kwargs)
            return _batch_to_arrays(batch)
        except TypeError as exc:  # pragma: no cover - signature mismatch
            last_error = exc
            continue
    try:
        indices = buffer.sample_indices(batch_size)
        batch = buffer.sample_by_index(indices)
        return _batch_to_arrays(batch)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Unable to sample a batch from the replay buffer: {exc}") from (last_error or exc)


def _batch_to_arrays(batch: Any) -> Dict[str, np.ndarray]:
    """Convert a sampled batch (Batch / dict / object) into a numpy dict."""
    if batch is None:
        return {}
    if isinstance(batch, Mapping):
        items = dict(batch)
    elif hasattr(batch, "as_dict"):
        items = batch.as_dict()
    elif hasattr(batch, "__dict__"):
        items = {k: v for k, v in vars(batch).items() if not k.startswith("_")}
    else:  # pragma: no cover
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    out: Dict[str, np.ndarray] = {}
    for key, value in items.items():
        if _TORCH_AVAILABLE and isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            out[key] = value
        elif np.isscalar(value):
            out[key] = np.asarray([value])
        else:
            try:
                out[key] = np.asarray(value)
            except Exception:  # pragma: no cover
                continue
    return out


def _as_torch(array: Any, device: str, dtype: Any = None) -> Any:
    """Convert an array-like to a torch tensor on ``device``."""
    if _TORCH_AVAILABLE and isinstance(array, torch.Tensor):
        tensor = array
    else:
        tensor = torch.as_tensor(np.asarray(array))
    if dtype is not None:
        if tensor.dtype != dtype and not tensor.dtype.is_floating_point:
            tensor = tensor.to(dtype)
        elif tensor.dtype.is_floating_point and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
    return tensor.to(device)


def _torch_batch(batch: Mapping[str, np.ndarray], device: str) -> Dict[str, Any]:
    """Cast the transition keys of a numpy batch to torch tensors."""
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if key in ("observations", "next_observations"):
            out[key] = _as_torch(value, device, torch.float32 if _TORCH_AVAILABLE else None)
        elif key in ("actions", "rewards", "terminals", "dones", "timeouts"):
            out[key] = _as_torch(value, device, torch.float32 if _TORCH_AVAILABLE else None)
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------------------
# Prior / encoder helpers
# --------------------------------------------------------------------------------------
def build_prior(cfg: PolicyTrainConfig, source: Any, state_dim: int) -> Any:
    """Construct the unsupervised reward-function prior ``p(eta)``."""
    kwargs: Dict[str, Any] = {"seed": cfg.seed}
    if cfg.families:
        kwargs["families"] = tuple(cfg.families)
    if cfg.exclude_xy:
        kwargs["exclude_dims"] = (0, 1)

    builder = make_mixture_prior or make_prior or make_fre_prior
    if callable(builder):
        try:
            return builder(source=source, state_dim=state_dim, **kwargs)
        except TypeError:
            pass
    if MixturePrior is not None:
        return MixturePrior(source=source, state_dim=state_dim, **kwargs)
    raise ImportError(
        "Could not construct the FRE reward prior; expected fre.rewards.prior to be importable."
    )


def build_encoder(
    state_dim: int, latent_dim: int = 128, cfg: Optional[PolicyTrainConfig] = None
) -> Any:
    """Instantiate a :class:`FREEncoder` with the paper's defaults."""
    if FREEncoder is None:
        raise ImportError("FREEncoder is unavailable; expected fre.fre.encoder to be importable.")
    kwargs: Dict[str, Any] = {"latent_dim": latent_dim}
    if cfg is not None:
        kwargs.update(
            num_blocks=cfg.num_blocks,
            num_heads=cfg.num_heads,
            mlp_dim=cfg.mlp_dim,
            dropout=cfg.dropout,
        )
    return FREEncoder(state_dim, **kwargs)


def load_encoder_checkpoint(
    path: str,
    state_dim: Optional[int] = None,
    latent_dim: int = 128,
    device: str = "cpu",
    cfg: Optional[PolicyTrainConfig] = None,
) -> Any:
    """Load a frozen FRE encoder from a Phase-1 checkpoint.

    Accepts checkpoints produced by :mod:`fre.train_encoder` (keys ``encoder`` /
    ``decoder`` / ``optimizer`` / ``config``) as well as a bare encoder ``state_dict``.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required to load an encoder checkpoint.")
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Unsupported checkpoint format at {path!r}: {type(checkpoint)}")

    meta = checkpoint.get("config") or checkpoint.get("meta") or {}
    if state_dim is None:
        state_dim = checkpoint.get("state_dim") or (meta.get("state_dim") if isinstance(meta, Mapping) else None)
    encoder_state = None
    for key in ("encoder", "encoder_state_dict", "model", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            encoder_state = {k: v for k, v in candidate.items() if not k.startswith("q_")}
            if encoder_state:
                break
    if not encoder_state:
        # The checkpoint may itself be a flat encoder state dict.
        if any(str(k).startswith("state_projection") or "blocks" in str(k) for k in checkpoint.keys()):
            encoder_state = dict(checkpoint)
        else:
            raise ValueError(f"Could not locate encoder weights inside checkpoint {path!r}.")

    if state_dim is None and "state_projection.weight" in encoder_state:
        state_dim = int(encoder_state["state_projection.weight"].shape[-1])
    if state_dim is None:
        raise ValueError("state_dim could not be inferred from the encoder checkpoint.")

    encoder = build_encoder(int(state_dim), latent_dim=latent_dim, cfg=cfg)
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        LOGGER.warning("Encoder checkpoint missing keys: %s", list(missing)[:5])
    if unexpected:
        LOGGER.warning("Encoder checkpoint had unexpected keys: %s", list(unexpected)[:5])
    encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


def freeze_encoder(encoder: Any) -> Any:
    """Put ``encoder`` in eval mode and disable gradient computation."""
    if encoder is None:
        return None
    encoder.eval()
    for param in getattr(encoder, "parameters", lambda: [])():
        param.requires_grad_(False)
    return encoder


# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------
class PolicyTrainer:
    """Phase-2 trainer: frozen FRE encoder + z-conditioned IQL."""

    def __init__(
        self,
        cfg: PolicyTrainConfig,
        obs_dim: int,
        action_dim: int,
        encoder: Any,
        prior: Any,
        buffer: Any = None,
        device: Optional[str] = None,
        logger: Any = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for Phase-2 policy training.")
        self.cfg = cfg
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = device or cfg.device or "cpu"
        self.logger = logger or LOGGER
        self.buffer = buffer
        self.prior = prior
        self.metrics = MetricTracker()
        self._rng = np.random.default_rng(cfg.seed)

        self.encoder = freeze_encoder(encoder).to(self.device)

        self.policy = LatentPolicyBundle(
            self.obs_dim,
            self.action_dim,
            latent_dim=cfg.latent_dim,
            hidden_sizes=tuple(cfg.hidden_sizes),
        ).to(self.device)

        iql_kwargs: Dict[str, Any] = {}
        if IQLConfig is not None:
            try:
                iql_config = IQLConfig(
                    discount=cfg.discount,
                    expectile=cfg.expectile,
                    awr_temperature=cfg.awr_temperature,
                    target_update_rate=cfg.target_update_rate,
                    lr=cfg.lr,
                    batch_size=cfg.batch_size,
                    reward_resample_interval=cfg.reward_resample_interval,
                )
                iql_kwargs["config"] = iql_config
            except Exception:  # pragma: no cover - tolerate differing IQLConfig fields
                iql_kwargs["config"] = None
        if IQLTrainer is None:
            raise ImportError("IQLTrainer is unavailable; expected fre.rl.iql to be importable.")
        self.trainer = IQLTrainer(
            self.policy,
            encoder=self.encoder,
            reward_sampler=None,
            latent_dim=cfg.latent_dim,
            lr=cfg.lr,
            device=self.device,
            **iql_kwargs,
        )
        if hasattr(self.trainer, "max_grad_norm"):
            try:
                self.trainer.max_grad_norm = cfg.max_grad_norm
            except Exception:  # pragma: no cover
                pass

        self.step = 0
        os.makedirs(cfg.run_dir, exist_ok=True)

    # -- prior sampling --------------------------------------------------------------
    def sample_reward_zs(self, obs: np.ndarray) -> Tuple[Any, Any, Dict[str, Any]]:
        """Sample reward functions, encode their contexts, and score the transitions.

        Returns ``(z, rewards, info)`` where ``z`` is ``(B, latent_dim)`` and ``rewards``
        is ``(B,)`` with ``r_i = eta_i(s_i)``.
        """
        batch_size = int(obs.shape[0])
        ctx_size = self.cfg.context_size
        dec_size = self.cfg.decoder_size

        out = None
        sample_batch = getattr(self.prior, "sample_batch", None)
        if callable(sample_batch):
            try:
                out = sample_batch(
                    batch_size,
                    ctx_size,
                    dec_size,
                    return_functions=True,
                    include_success=True,
                    disjoint=True,
                )
            except TypeError:
                try:
                    out = sample_batch(
                        batch_size, num_context=ctx_size, num_decoder=dec_size, return_functions=True
                    )
                except TypeError:
                    out = sample_batch(batch_size)

        info: Dict[str, Any] = {}
        if isinstance(out, Mapping):
            ctx_states = out.get("context_states")
            ctx_rewards = out.get("context_rewards")
            functions = out.get("functions")
            families = out.get("families")
            if families is not None:
                try:
                    info["families"] = list(np.asarray(families).tolist())
                except Exception:
                    pass
        elif isinstance(out, (tuple, list)) and len(out) >= 4:
            ctx_states, ctx_rewards, _dec_states, _dec_rewards = out[:4]
            functions = out[4] if len(out) > 4 else None
        else:  # pragma: no cover - degenerate fallback
            ctx_states, ctx_rewards, functions = None, None, None

        if ctx_states is None or ctx_rewards is None:
            raise RuntimeError("The reward prior did not return encoder context samples.")

        ctx_states_t = _as_torch(ctx_states, self.device, torch.float32)
        ctx_rewards_t = _as_torch(ctx_rewards, self.device, torch.float32)

        with torch.no_grad():
            z = self.encoder.encode(ctx_states_t, ctx_rewards_t)
        z = z.detach()

        rewards = self.compute_rewards(functions, obs)
        return z, rewards, info

    def compute_rewards(self, functions: Any, obs: np.ndarray) -> Any:
        """Evaluate ``r = eta(s)`` for a batch of reward functions on the batch states."""
        rewards: Optional[np.ndarray] = None
        reward_batch = getattr(self.prior, "reward_batch", None)
        if callable(reward_batch) and functions is not None:
            for kwargs in (dict(next_observations=None), {}):
                try:
                    rewards = reward_batch(functions, obs, **kwargs)
                    break
                except TypeError:
                    continue
                except Exception:  # pragma: no cover
                    rewards = None
                    break

        if rewards is None and functions is not None:
            try:
                # per-transition: function i evaluated at state i
                rewards = np.asarray(
                    [float(np.asarray(fn(obs[i : i + 1])).reshape(-1)[0]) for i, fn in enumerate(functions)],
                    dtype=np.float32,
                )
            except Exception:  # pragma: no cover
                rewards = None

        if rewards is None:
            rewards = np.zeros(obs.shape[0], dtype=np.float32)

        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        rewards = np.clip(rewards, -1.0, 1.0)
        return _as_torch(rewards, self.device, torch.float32)

    # -- updates ---------------------------------------------------------------------
    def train_step(self) -> Dict[str, float]:
        """One IQL update: sample transitions, sample+encode reward functions, update."""
        raw = _sample_buffer_batch(self.buffer, self.cfg.batch_size, self.device)
        if not raw:
            raise RuntimeError("Replay buffer returned an empty batch.")
        obs = np.asarray(raw["observations"], dtype=np.float32)

        z, rewards, info = self.sample_reward_zs(obs)
        batch = _torch_batch(raw, self.device)

        if "terminals" in raw and "dones" not in raw:
            batch["dones"] = batch["terminals"]

        update = getattr(self.trainer, "update")
        try:
            metrics = update(batch, latents=z, rewards=rewards)
        except TypeError:
            metrics = update(batch, z, rewards)
        if not isinstance(metrics, Mapping):  # pragma: no cover
            metrics = {"loss": float(metrics)}

        out = {k: float(np.asarray(v).reshape(-1)[0]) for k, v in metrics.items() if np.isscalar(v) or np.asarray(v).size == 1}
        if "families" in info:
            try:
                families = np.asarray(info["families"])
                for name in np.unique(families):
                    out[f"prior/{name}_frac"] = float(np.mean(families == name))
            except Exception:
                pass
        return out

    def train(self, steps: Optional[int] = None) -> "PolicyTrainer":
        """Run the Phase-2 training loop with periodic logging / checkpoints / eval."""
        total = int(steps if steps is not None else self.cfg.steps)
        start = self.step
        if start >= total:
            self.logger.info("Policy already trained for %d steps (target %d).", start, total)
            return self

        self.logger.info(
            "[%s] Phase-2 IQL: steps=%d->%d batch=%d lr=%g gamma=%g expectile=%g awr_temp=%g",
            self.cfg.exp_name,
            start,
            total,
            self.cfg.batch_size,
            self.cfg.lr,
            self.cfg.discount,
            self.cfg.expectile,
            self.cfg.awr_temperature,
        )

        iterator = progress(range(start, total), total=total, desc="fre-policy")
        t0 = time.time()
        warmup = max(20, min(500, total // 20 + 1))
        for step in iterator:
            metrics = self.train_step()
            self.metrics.update(values=metrics)
            self.step = step + 1

            # Expert-style soft target decay over the first few thousand updates.
            decay = None
            if self.step <= max(20_000, warmup):
                decay = 1.0 - min(1.0, self.step / float(max(20_000, warmup)))
            if decay is not None and hasattr(self.trainer, "soft_update"):
                try:
                    self.trainer.soft_update(decay)  # type: ignore[attr-defined]
                except Exception:  # pragma: no cover
                    pass

            if self.step % max(1, self.cfg.log_interval) == 0:
                means = self.metrics.mean()
                elapsed = time.time() - t0
                rate = (self.step - start) / max(1e-6, elapsed)
                pretty = " ".join(f"{k}={v:.4f}" for k, v in sorted(means.items())[:6])
                self.logger.info(
                    "[%s] step %d/%d (%.1f it/s) %s",
                    self.cfg.exp_name,
                    self.step,
                    total,
                    rate,
                    pretty,
                )
                if self.cfg.verbose:
                    self.metrics.reset()

            if self.cfg.checkpoint_interval and self.step % self.cfg.checkpoint_interval == 0:
                self.save_checkpoint(tag=f"step{self.step}")

            if self.cfg.eval_interval and self.step % self.cfg.eval_interval == 0:
                self.maybe_evaluate()

        self.save_checkpoint(tag="final")
        self.logger.info("[%s] Phase-2 finished after %d steps.", self.cfg.exp_name, self.step)
        return self

    # -- evaluation ------------------------------------------------------------------
    def maybe_evaluate(self) -> Optional[Dict[str, Any]]:
        """Run the zero-shot evaluation harness, tolerating a missing environment."""
        try:
            from fre.evaluate import EvalConfig, Evaluator
        except Exception as exc:  # pragma: no cover - envs unavailable
            self.logger.warning("Skipping evaluation (harness unavailable): %s", exc)
            return None

        try:
            eval_cfg = EvalConfig(
                domain=self.cfg.domain,
                num_episodes=self.cfg.eval_episodes,
                num_seeds=self.cfg.eval_seeds,
                context_size=self.cfg.context_size,
                device=self.device,
                seed=self.cfg.seed,
            )
            evaluator = Evaluator(self.encoder, self.policy, eval_cfg, device=self.device)
            results = evaluator.evaluate_domain(self.cfg.domain)
            results = getattr(evaluator, "results", results)
            from fre.evaluate import results_to_summary

            summary = results_to_summary(results)
            path = os.path.join(self.cfg.run_dir, f"eval_step{self.step}.json")
            write_json(path, summary)
            self.logger.info("[%s] eval@%d -> %s", self.cfg.exp_name, self.step, summary)
            return summary
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Evaluation failed at step %d: %s", self.step, exc)
            return None

    # -- checkpointing ---------------------------------------------------------------
    def checkpoint_dict(self) -> Dict[str, Any]:
        policy_state = None
        if hasattr(self.policy, "state_dict"):
            policy_state = self.policy.state_dict()
        trainer_state = None
        if hasattr(self.trainer, "state_dict"):
            try:
                trainer_state = self.trainer.state_dict()
            except Exception:  # pragma: no cover
                trainer_state = None
        return {
            "encoder": self.encoder.state_dict() if self.encoder is not None else None,
            "policy": policy_state,
            "trainer": trainer_state,
            "config": self.cfg.as_dict(),
            "state_dim": self.obs_dim,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "latent_dim": self.cfg.latent_dim,
            "domain": self.cfg.domain,
            "step": self.step,
        }

    def save_checkpoint(self, tag: Optional[str] = None) -> str:
        path = os.path.join(self.cfg.run_dir, "policy.pt" if tag is None else f"policy_{tag}.pt")
        torch.save(self.checkpoint_dict(), path)
        latest = os.path.join(self.cfg.run_dir, "policy_latest.json")
        write_json(latest, {"step": self.step, "checkpoint": path, "config": self.cfg.as_dict()})
        self.logger.info("Saved policy checkpoint -> %s", path)
        return path

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, Mapping):
            trainer_state = checkpoint.get("trainer")
            if trainer_state and hasattr(self.trainer, "load_state_dict"):
                try:
                    self.trainer.load_state_dict(trainer_state)
                except Exception:  # pragma: no cover
                    pass
            policy_state = checkpoint.get("policy")
            if policy_state is not None:
                try:
                    self.policy.load_state_dict(policy_state)
                except Exception:  # pragma: no cover
                    try:
                        self.policy.load_policy_state_dict(policy_state)
                    except Exception:
                        pass
            self.step = int(checkpoint.get("step", self.step))
        self.logger.info("Loaded policy checkpoint <- %s (step %d)", path, self.step)


# --------------------------------------------------------------------------------------
# Top-level entry points
# --------------------------------------------------------------------------------------
def train_policy(
    cfg: Optional[PolicyTrainConfig] = None,
    encoder: Any = None,
    buffer: Any = None,
    prior: Any = None,
    **overrides: Any,
) -> PolicyTrainer:
    """Build everything needed for Phase 2 and run training.

    Parameters
    ----------
    cfg:
        Optional :class:`PolicyTrainConfig`. Remaining keyword arguments override its fields.
    encoder:
        Pre-trained (frozen) FRE encoder. If omitted, it is loaded from
        ``cfg.encoder_checkpoint`` or, when ``cfg.pretrain_encoder`` is set, trained from
        scratch by ``fre.train_encoder``.
    buffer, prior:
        Optional pre-built replay buffer / reward prior (useful for tests).
    """
    if cfg is None:
        cfg = PolicyTrainConfig(**overrides)
    else:
        for key, value in overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        cfg.__post_init__()

    if seed_everything is not None:
        seed_everything(cfg.seed)

    if buffer is None:
        buffer = build_replay_buffer(cfg, verbose=cfg.verbose)
    obs_dim, action_dim = _infer_dims(buffer)
    if not action_dim:
        action_dim = _ACTION_DIM_FALLBACK.get(cfg.domain_name or cfg.domain, 6)
    LOGGER.info(
        "[%s] dataset loaded: obs_dim=%d action_dim=%d transitions=%s",
        cfg.exp_name,
        obs_dim,
        action_dim,
        len(buffer) if hasattr(buffer, "__len__") else "?",
    )

    # --- Phase 1 (optional / required) -------------------------------------------------
    if encoder is None and cfg.encoder_checkpoint and os.path.exists(cfg.encoder_checkpoint):
        encoder = load_encoder_checkpoint(
            cfg.encoder_checkpoint,
            state_dim=obs_dim,
            latent_dim=cfg.latent_dim,
            device=cfg.device or "cpu",
            cfg=cfg,
        )
        LOGGER.info("Loaded frozen encoder from %s", cfg.encoder_checkpoint)
    elif encoder is None and cfg.pretrain_encoder:
        from fre.train_encoder import train_encoder

        LOGGER.info("[%s] Phase 1: pretraining FRE encoder ...", cfg.exp_name)
        enc_cfg_kwargs: Dict[str, Any] = dict(
            domain=cfg.domain,
            domain_name=cfg.domain_name,
            seed=cfg.seed,
            latent_dim=cfg.latent_dim,
            context_size=cfg.context_size,
            decoder_size=cfg.decoder_size,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            dataset=cfg.dataset,
            data_root=cfg.data_root,
            output_dir=cfg.output_dir,
            num_blocks=cfg.num_blocks,
            num_heads=cfg.num_heads,
            mlp_dim=cfg.mlp_dim,
            dropout=cfg.dropout,
            device=cfg.device,
        )
        if cfg.pretrain_steps:
            enc_cfg_kwargs["steps"] = cfg.pretrain_steps
        trainer = train_encoder(**{k: v for k, v in enc_cfg_kwargs.items() if v is not None})
        encoder = getattr(trainer, "frozen_encoder", None) or getattr(trainer, "encoder", None)
        if encoder is None:  # pragma: no cover
            raise RuntimeError("train_encoder did not expose an encoder.")
    if encoder is None:
        raise ValueError(
            "No encoder supplied. Pass ``--encoder-checkpoint``, set ``--pretrain-encoder``, "
            "or provide an encoder to ``train_policy``."
        )

    if prior is None:
        prior = build_prior(cfg, buffer, obs_dim)

    trainer = PolicyTrainer(
        cfg,
        obs_dim=obs_dim,
        action_dim=action_dim,
        encoder=encoder,
        prior=prior,
        buffer=buffer,
        device=cfg.device,
    )
    if cfg.resume:
        trainer.load_checkpoint(cfg.resume)
    trainer.train()
    return trainer


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FRE Phase-2: z-conditioned IQL policy training")
    parser.add_argument("--domain", default="antmaze", choices=["antmaze", "exorl", "kitchen"])
    parser.add_argument("--domain-name", default=None, help="walker / cheetah for ExORL")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--discount", type=float, default=0.88)
    parser.add_argument("--expectile", type=float, default=0.8)
    parser.add_argument("--awr-temperature", type=float, default=3.0)
    parser.add_argument("--target-update-rate", type=float, default=0.001)
    parser.add_argument("--latent-dim", type=int, default=int(DEFAULT_LATENT_DIM or 128))
    parser.add_argument("--context-size", type=int, default=DEFAULT_CONTEXT_SIZE)
    parser.add_argument("--decoder-size", type=int, default=DEFAULT_DECODER_SIZE)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--encoder-checkpoint", default=None)
    parser.add_argument("--pretrain-encoder", action="store_true")
    parser.add_argument("--pretrain-steps", type=int, default=0)
    parser.add_argument("--ablation", default=None, help="Table 4 subset: all/goals/lin/mlp/lin-mlp/goal-mlp/goal-lin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-interval", type=int, default=1_000)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--checkpoint-interval", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seeds", type=int, default=5)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> PolicyTrainer:
    """CLI entry point (``python -m fre.train_policy``)."""
    args = parse_args(argv)
    cfg = PolicyTrainConfig(
        domain=args.domain,
        domain_name=args.domain_name,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        discount=args.discount,
        expectile=args.expectile,
        awr_temperature=args.awr_temperature,
        target_update_rate=args.target_update_rate,
        latent_dim=args.latent_dim,
        context_size=args.context_size,
        decoder_size=args.decoder_size,
        dataset=args.dataset,
        data_root=args.data_root,
        encoder_checkpoint=args.encoder_checkpoint,
        pretrain_encoder=args.pretrain_encoder,
        pretrain_steps=args.pretrain_steps,
        ablation=args.ablation,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        run_name=args.run_name,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        eval_episodes=args.eval_episodes,
        eval_seeds=args.eval_seeds,
        log_file=args.log_file,
        resume=args.resume,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )
    if args.log_file:
        get_logger("fre.train_policy", log_file=args.log_file)
    return train_policy(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()

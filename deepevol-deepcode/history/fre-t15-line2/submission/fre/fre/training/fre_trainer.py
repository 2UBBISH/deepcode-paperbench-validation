"""FRE phase-1 trainer: maximise Equation (6) (Algorithm 1, "# Train encoder").

This module implements the first stage of *Zero-Shot Reinforcement Learning via
Functional Reward Encodings* (Algorithm 1, Section 4.3, Section 4.1):

    Input: unlabeled offline dataset D, distribution over random unsupervised
           reward functions p(eta).
    while not converged do
        Sample reward function  eta ~ p(eta)
        Sample K  states for encoder  {s^e_k} ~ D
        Sample K' states for decoder  {s^d_k} ~ D
        Train FRE by maximising Equation (6)
    end while

The information-bottleneck objective of Section 4.1 is

    I(L_eta^d ; Z) - beta * I(L_eta^e ; Z)
      >= E_{eta, L_eta^e, L_eta^d, z ~ p_theta(z | L_eta^e)}[
             sum_{k=1}^{K'} log q_theta(eta(s^d_k) | s^d_k, z)
             - beta * D_KL( p_theta(z | L_eta^e) || u(z) ) ] + (const)

with ``u(z)`` an uninformative prior defined as the unit Gaussian.  Because the
decoder parametrises a unit-variance Gaussian likelihood,
``log q_theta(eta(s^d)|s^d,z) = -0.5*(eta(s^d) - mu)^2 + (const)``, so maximising
the bound is equivalent to *minimising* the negative bound

    loss = reconstruction_error + beta * D_KL( p_theta(z | .) || N(0, I) )

where ``reconstruction_error`` is the mean-squared error between predicted and
true rewards over the K' decoding states -- verbatim (Section 4.1): "We train
both the encoder and decoder networks jointly, minimizing mean-squared error
between the predicted and true rewards under the decoding states."
``loss_form="nll_sum"`` uses instead the faithful ``0.5 * sum_k (mu_k - y_k)^2``
form of the equation (the two differ only by the constant factor K'/2, which
merely rescales the effective KL weight).

The RL components are *not* touched here: this file is phase 1 only.  Phase 2
(frozen encoder + IQL) lives in ``fre/training/iql.py`` and is orchestrated by
``fre/training/strided.py``.

Hyper-parameters (Section A, Table 3): Batch Size 512, Reward Pairs to Encode
32, Reward Pairs to Decode 8, 32 reward embeddings, Encoder Layers
[256,256,256,256], Encoder Attention Heads 4, Adam, Learning Rate 0.0001,
beta (KL Weight) 0.01, Encoder Training Steps 150,000 (1M for ExORL/Kitchen).
"""

from __future__ import annotations

import inspect
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # torch is required at runtime, but keep import errors informative
    import torch

    _TORCH_AVAILABLE = True
except Exception as _torch_import_error:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_MESSAGE = str(_torch_import_error)


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "fre.training.fre_trainer requires PyTorch (import failed: "
            f"{_TORCH_IMPORT_MESSAGE})"
        )


# ---------------------------------------------------------------------------
# package imports (with a fallback for direct module execution)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))              # .../fre/fre/training
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))         # .../fre


def _ensure_package_on_path() -> None:
    if _PACKAGE_ROOT not in sys.path:
        sys.path.insert(0, _PACKAGE_ROOT)


try:  # pragma: no cover - depends on invocation style
    from fre.models.encoder import make_fre_encoder
    from fre.models.decoder import DEFAULT_DECODER_LAYERS, make_reward_decoder
except ImportError:  # pragma: no cover - direct execution fallback
    _ensure_package_on_path()
    from fre.models.encoder import make_fre_encoder
    from fre.models.decoder import DEFAULT_DECODER_LAYERS, make_reward_decoder


__all__ = [
    "FRETrainerConfig",
    "FRETrainOutput",
    "FRETrainer",
    "make_fre_trainer",
    "train_fre_step",
    "fre_reconstruction_kl_loss",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_BETA_KL",
    "DEFAULT_ENCODER_TRAINING_STEPS",
    "DEFAULT_POLICY_TRAINING_STEPS",
    "DEFAULT_NUM_ENCODER_STATES",
    "DEFAULT_NUM_DECODER_STATES",
    "DEFAULT_NUM_REWARD_EMBEDDINGS",
    "DOMAIN_STEP_BUDGETS",
]


# ---------------------------------------------------------------------------
# defaults straight from Table 3
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE = 512
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_BETA_KL = 0.01
DEFAULT_ENCODER_TRAINING_STEPS = 150_000
DEFAULT_POLICY_TRAINING_STEPS = 850_000
DEFAULT_NUM_ENCODER_STATES = 32          # "Reward Pairs to Encode"
DEFAULT_NUM_DECODER_STATES = 8           # "Reward Pairs to Decode"
DEFAULT_NUM_REWARD_EMBEDDINGS = 32       # "Number of Reward Embeddings"
DEFAULT_NUM_REWARD_FUNCTIONS = 1
DEFAULT_LATENT_DIM = 128
DEFAULT_LOG_INTERVAL = 1_000

# Table 3 footnote: "150,000 (1M for ExORL/Kitchen)" / "850,000 (1M ...)"
DOMAIN_STEP_BUDGETS: Dict[str, Tuple[int, int]] = {
    "antmaze": (150_000, 850_000),
    "exorl": (1_000_000, 1_000_000),
    "kitchen": (1_000_000, 1_000_000),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _call_with_supported_kwargs(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` passing only the keyword arguments it accepts.

    The replay buffer, reward priors and encoders are duck-typed collaborators
    with slightly different (but signature-compatible) APIs, so introspection is
    used instead of rigid coupling.  This mirrors the flexibility the strided
    controller relies on.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(*args, **kwargs)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return fn(*args, **filtered)


def _as_numpy_2d(values: Any) -> np.ndarray:
    if isinstance(values, np.ndarray):
        arr = values
    elif torch is not None and isinstance(values, torch.Tensor):
        arr = values.detach().cpu().numpy()
    else:
        arr = np.asarray(values)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _as_numpy_1d(values: Any) -> np.ndarray:
    if isinstance(values, np.ndarray):
        arr = values
    elif torch is not None and isinstance(values, torch.Tensor):
        arr = values.detach().cpu().numpy()
    else:
        arr = np.asarray(values)
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def _unpack_rewards_and_dones(result: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Normalise the several ``eta.label`` return conventions."""
    if isinstance(result, tuple):
        rewards = result[0]
        dones = result[1] if len(result) > 1 else None
    elif isinstance(result, Mapping):
        rewards = result.get("rewards", result.get("reward"))
        dones = result.get("dones", result.get("done"))
    else:
        rewards, dones = result, None
    rewards = _as_numpy_1d(rewards)
    if dones is not None:
        dones = _as_numpy_1d(dones)
    return rewards, dones


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class FRETrainerConfig:
    """Encoder-training configuration (Algorithm 1 phase 1, Table 3 numbers)."""

    batch_size: int = DEFAULT_BATCH_SIZE
    num_encoder_states: int = DEFAULT_NUM_ENCODER_STATES            # K
    num_decoder_states: int = DEFAULT_NUM_DECODER_STATES            # K'
    reward_functions_per_update: int = DEFAULT_NUM_REWARD_FUNCTIONS
    learning_rate: float = DEFAULT_LEARNING_RATE
    beta: float = DEFAULT_BETA_KL
    optimizer: str = "adam"
    weight_decay: float = 0.0
    max_grad_norm: Optional[float] = None
    encoder_training_steps: int = DEFAULT_ENCODER_TRAINING_STEPS
    policy_training_steps: int = DEFAULT_POLICY_TRAINING_STEPS
    num_reward_embeddings: int = DEFAULT_NUM_REWARD_EMBEDDINGS
    latent_dim: int = DEFAULT_LATENT_DIM
    loss_form: str = "mse"                 # "mse" (paper) | "nll_sum" (Eq. 6 verbatim)
    sample_z: bool = True                  # sample z ~ p_theta(z|.) during training
    use_encoder_inputs: bool = False       # ExORL physics-augmented encoder inputs
    reward_range: str = "auto"             # "auto" (fit to eta) | "fixed"
    reward_min: float = -1.0
    reward_max: float = 1.0
    encoder_layers: Tuple[int, ...] = (256, 256, 256, 256)   # Table 3 "Encoder Layers"
    encoder_attention_heads: int = 4                          # Table 3
    log_interval: int = DEFAULT_LOG_INTERVAL
    freeze_encoder_after_phase1: bool = True
    device: Optional[str] = None
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["encoder_layers"] = list(self.encoder_layers)
        return out

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "FRETrainerConfig":
        fields = set(cls.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in dict(values).items() if k in fields}
        if kwargs.get("encoder_layers") is not None:
            kwargs["encoder_layers"] = tuple(kwargs["encoder_layers"])
        return cls(**kwargs)

    @classmethod
    def for_domain(cls, domain: str, **overrides: Any) -> "FRETrainerConfig":
        """Table 3 step budgets per domain (150k/850k vs 1M/1M)."""
        domain = (domain or "").lower()
        budget = DOMAIN_STEP_BUDGETS.get(domain)
        kwargs: Dict[str, Any] = {}
        if budget is not None:
            kwargs["encoder_training_steps"], kwargs["policy_training_steps"] = budget
        kwargs["use_encoder_inputs"] = domain == "exorl"
        kwargs.update(overrides)
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# loss output
# ---------------------------------------------------------------------------
@dataclass
class FRETrainOutput:
    """Per-update statistics of the Equation (6) objective."""

    step: int = 0
    loss: float = 0.0
    reconstruction_loss: float = 0.0     # MSE over the K' decoding states
    reconstruction_sum: float = 0.0      # 0.5 * sum_k (y_k - mu_k)^2
    kl: float = 0.0                      # mean D_KL(p_theta(z|.) || N(0, I))
    objective: float = 0.0               # quantity maximised = -(recon + beta*kl)
    beta: float = DEFAULT_BETA_KL
    reward_min: float = -1.0
    reward_max: float = 1.0
    mean_target: float = 0.0
    mean_prediction: float = 0.0
    mean_abs_error: float = 0.0
    z_norm: float = 0.0
    mu_norm: float = 0.0
    mean_std: float = 0.0
    num_reward_functions: int = 1
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        out = {
            "step": int(self.step),
            "loss": float(self.loss),
            "reconstruction_loss": float(self.reconstruction_loss),
            "reconstruction_sum": float(self.reconstruction_sum),
            "kl": float(self.kl),
            "objective": float(self.objective),
            "beta": float(self.beta),
            "mean_target": float(self.mean_target),
            "mean_prediction": float(self.mean_prediction),
            "mean_abs_error": float(self.mean_abs_error),
            "z_norm": float(self.z_norm),
            "mu_norm": float(self.mu_norm),
            "mean_std": float(self.mean_std),
        }
        out.update({k: float(v) for k, v in self.extra.items()})
        return out

    def __getitem__(self, key: str) -> float:
        return self.to_dict()[key]


# ---------------------------------------------------------------------------
# Equation (6) primitive
# ---------------------------------------------------------------------------
def fre_reconstruction_kl_loss(
    encoder_output: Any = None,
    target_rewards: Any = None,
    predictions: Any = None,
    decoder: Any = None,
    decoding_states: Any = None,
    z: Any = None,
    beta: float = DEFAULT_BETA_KL,
    loss_form: str = "mse",
) -> Dict[str, Any]:
    """Equation (6): ``E[sum_k log q_theta(eta(s^d_k)|s^d_k,z)] - beta * KL``.

    Returns a dict with the *minimised* ``loss`` (negative bound), the
    reconstruction terms and the KL term.  ``encoder_output`` must expose
    ``mu``/``log_std`` (or ``kl_to_unit_gaussian()``).
    """
    _require_torch()

    if predictions is None:
        if decoder is None or decoding_states is None or z is None:
            raise ValueError(
                "provide either `predictions`, or (`decoder`, `decoding_states`, `z`)"
            )
        predictions = decoder(decoding_states, z)
    pred = predictions
    if isinstance(pred, Mapping):
        pred = pred.get("mean", pred.get("prediction"))
    if hasattr(pred, "mean") and not isinstance(pred, torch.Tensor):  # DecoderOutput
        pred = pred.mean
    pred = torch.as_tensor(pred)

    if target_rewards is None:
        raise ValueError("`target_rewards` (eta(s^d)) is required")
    if hasattr(target_rewards, "mean") and not isinstance(target_rewards, torch.Tensor):
        target_rewards = target_rewards.mean
    target = torch.as_tensor(target_rewards).to(dtype=pred.dtype, device=pred.device)
    target = target.reshape(pred.shape)

    sq_err = (pred - target) ** 2
    if sq_err.dim() > 1:
        mse = sq_err.mean()
        err_sum = 0.5 * sq_err.sum(dim=-1).mean()
    else:
        mse = sq_err.mean()
        err_sum = 0.5 * sq_err.mean()

    if loss_form == "nll_sum":
        reconstruction = err_sum
    else:
        reconstruction = mse

    kl = torch.zeros((), dtype=pred.dtype, device=pred.device)
    if encoder_output is not None:
        if hasattr(encoder_output, "kl_to_unit_gaussian"):
            kl = encoder_output.kl_to_unit_gaussian()
            kl = kl.mean() if kl.dim() > 0 else kl
        elif hasattr(encoder_output, "mu") and hasattr(encoder_output, "log_std"):
            mu = torch.as_tensor(encoder_output.mu)
            log_std = torch.as_tensor(encoder_output.log_std)
            kl = (-0.5 * (1.0 + 2.0 * log_std - mu.pow(2) - torch.exp(2.0 * log_std))).sum(-1).mean()

    loss = reconstruction + float(beta) * kl
    return {
        "loss": loss,
        "reconstruction": reconstruction,
        "reconstruction_mse": mse,
        "reconstruction_sum": err_sum,
        "kl": kl,
        "objective": -loss,
    }


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------
class FRETrainer:
    """Trains the FRE encoder/decoder jointly on Equation (6) (Algorithm 1 phase 1)."""

    def __init__(
        self,
        encoder: Any = None,
        decoder: Any = None,
        replay_buffer: Any = None,
        prior: Any = None,
        config: Optional[FRETrainerConfig] = None,
        state_dim: Optional[int] = None,
        latent_dim: Optional[int] = None,
        device: Optional[Any] = None,
        optimizer: Any = None,
        seed: Optional[int] = None,
        logger: Any = None,
        **config_overrides: Any,
    ) -> None:
        _require_torch()

        base = config if isinstance(config, FRETrainerConfig) else FRETrainerConfig()
        merged = base.to_dict()
        if isinstance(config, Mapping) and not isinstance(config, FRETrainerConfig):
            merged.update({k: v for k, v in config.items() if k in merged})
        merged.update({k: v for k, v in config_overrides.items() if k in merged})
        if latent_dim is not None:
            merged["latent_dim"] = latent_dim
        if seed is not None:
            merged["seed"] = seed
        self.config = FRETrainerConfig.from_dict(merged)

        if device is None:
            device = self.config.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.replay_buffer = replay_buffer
        self.prior = prior
        self.logger = logger

        # --- state dimension (ExORL encoder inputs may be augmented) ----------
        resolved_state_dim = state_dim
        if resolved_state_dim is None and encoder is not None:
            resolved_state_dim = getattr(encoder, "state_dim", None)
        if resolved_state_dim is None and encoder is not None:
            resolved_state_dim = getattr(encoder, "input_dim", None)
        if resolved_state_dim is None and replay_buffer is not None:
            resolved_state_dim = getattr(replay_buffer, "encoder_obs_dim", None)
            if resolved_state_dim is None:
                resolved_state_dim = getattr(replay_buffer, "obs_dim", None)
        if resolved_state_dim is None and prior is not None:
            resolved_state_dim = getattr(prior, "state_dim", None)
        self.state_dim = int(resolved_state_dim) if resolved_state_dim is not None else None

        # --- encoder ----------------------------------------------------------
        if encoder is None:
            if self.state_dim is None:
                raise ValueError(
                    "state_dim could not be inferred; pass `state_dim` or an `encoder`"
                )
            encoder = _call_with_supported_kwargs(
                make_fre_encoder,
                self.state_dim,
                latent_dim=self.config.latent_dim,
                num_reward_embeddings=self.config.num_reward_embeddings,
                num_layers=len(self.config.encoder_layers),
                num_heads=self.config.encoder_attention_heads,
            )
        self.encoder = encoder.to(self.device)

        # --- decoder ----------------------------------------------------------
        if decoder is None:
            if self.state_dim is None:
                raise ValueError("state_dim could not be inferred for the decoder")
            decoder = _call_with_supported_kwargs(
                make_reward_decoder,
                self.state_dim,
                latent_dim=self.config.latent_dim,
                hidden_dims=tuple(DEFAULT_DECODER_LAYERS),
            )
        self.decoder = decoder.to(self.device)

        # --- optimizer: a single shared Adam over encoder + decoder -----------
        self.optimizer = optimizer if optimizer is not None else self._build_optimizer()

        self.rng = np.random.default_rng(self.config.seed)
        self.step_count = 0
        self.history: List[Dict[str, float]] = []
        self.last_output: Optional[FRETrainOutput] = None
        self.last_loss: float = float("nan")

    # -- plumbing ----------------------------------------------------------
    def _build_optimizer(self) -> Any:
        name = (self.config.optimizer or "adam").lower()
        if name != "adam":
            raise ValueError(
                f"unsupported optimizer '{self.config.optimizer}' (Table 3 uses Adam)"
            )
        return torch.optim.Adam(
            self.trainable_parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def trainable_parameters(self) -> List[Any]:
        return [
            p
            for p in list(self.encoder.parameters()) + list(self.decoder.parameters())
            if p.requires_grad
        ]

    def parameters(self) -> List[Any]:
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def _log(self, stats: Mapping[str, Any], step: Optional[int] = None) -> None:
        if self.logger is None:
            return
        payload = dict(stats)
        if step is not None:
            payload.setdefault("step", step)
        for attr in ("log", "log_stats", "log_metrics", "record"):
            fn = getattr(self.logger, attr, None)
            if callable(fn):
                try:
                    fn(payload)
                    return
                except Exception:
                    continue
        if callable(self.logger):
            try:
                self.logger(payload)
            except Exception:
                pass

    # -- batch construction (Algorithm 1 lines 4-6) -------------------------
    def sample_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample K encoding states and K' decoding states from D.

        Section 4.1: "Crucially, the states sampled for decoding are different
        than those used for encoding."
        """
        buffer = self.replay_buffer
        if buffer is None:
            raise ValueError("a replay_buffer is required to sample encoder/decoder states")

        batch_size = self.config.batch_size
        K, Kp = self.config.num_encoder_states, self.config.num_decoder_states

        if hasattr(buffer, "sample_encoder_and_decoder_states"):
            result = _call_with_supported_kwargs(
                buffer.sample_encoder_and_decoder_states,
                batch_size,
                K,
                Kp,
                rng=self.rng,
                encoder_input=self.config.use_encoder_inputs,
                use_encoder_inputs=self.config.use_encoder_inputs,
            )
            if isinstance(result, Mapping):
                enc = result.get("encoder_states", result.get("enc_states"))
                dec = result.get("decoder_states", result.get("dec_states"))
            else:
                enc, dec = result[0], result[1]
            return _as_numpy_2d(enc), _as_numpy_2d(dec)

        # Fallback: independent uniform state draws.
        enc = _call_with_supported_kwargs(
            buffer.sample_states,
            batch_size * K,
            rng=self.rng,
            encoder_input=self.config.use_encoder_inputs,
        )
        dec = _call_with_supported_kwargs(
            buffer.sample_states,
            batch_size * Kp,
            rng=self.rng,
            encoder_input=False,
        )
        enc = _as_numpy_2d(enc).reshape(batch_size, K, -1)
        dec = _as_numpy_2d(dec).reshape(batch_size, Kp, -1)
        return enc, dec

    # -- reward prior sampling --------------------------------------------
    def sample_reward_function(self) -> Any:
        if self.prior is None:
            raise ValueError("a reward prior p(eta) is required for Equation (6)")
        try:
            return _call_with_supported_kwargs(self.prior.sample, rng=self.rng)
        except TypeError:
            return _call_with_supported_kwargs(self.prior, rng=self.rng)

    def _label(
        self, eta: Any, states: np.ndarray, ensure_goal: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Evaluate ``eta`` on ``states`` -> (rewards, done mask)."""
        if hasattr(eta, "label"):
            try:
                result = _call_with_supported_kwargs(
                    eta.label, states, ensure_goal=ensure_goal, rng=self.rng
                )
            except TypeError:
                result = _call_with_supported_kwargs(eta.label, states, rng=self.rng)
            return _unpack_rewards_and_dones(result)
        if callable(eta):
            return _as_numpy_1d(_call_with_supported_kwargs(eta, states)), None
        raise TypeError("sampled reward function is neither callable nor exposes `.label`")

    def reward_range_for(self, eta: Any, states: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Reward range used for the 32-bin discretisation of the encoder tokens."""
        if self.config.reward_range == "fixed":
            return float(self.config.reward_min), float(self.config.reward_max)
        if hasattr(eta, "reward_bounds"):
            try:
                lo, hi = _call_with_supported_kwargs(eta.reward_bounds, states)
                if np.isfinite([lo, hi]).all() and hi > lo:
                    return float(lo), float(hi)
            except Exception:
                pass
        lo = getattr(eta, "reward_min", None)
        hi = getattr(eta, "reward_max", None)
        if lo is None or hi is None or not np.isfinite([lo, hi]).all() or hi <= lo:
            lo, hi = self.config.reward_min, self.config.reward_max
        return float(lo), float(hi)

    def _rewards_for_update(
        self, enc_states: np.ndarray, dec_states: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float], Optional[np.ndarray], int]:
        """Label the sampled reward function(s) on the encoder/decoder states."""
        num_eta = max(1, int(self.config.reward_functions_per_update))
        batch_size = enc_states.shape[0]
        enc_rewards = np.empty((batch_size, enc_states.shape[1]), dtype=np.float32)
        dec_rewards = np.empty((batch_size, dec_states.shape[1]), dtype=np.float32)
        enc_dones = np.zeros_like(enc_rewards, dtype=np.float32)

        if num_eta == 1:
            eta = self.sample_reward_function()
            ranges = [self.reward_range_for(eta, enc_states.reshape(-1, enc_states.shape[-1]))]
            r_enc, d_enc = self._label(eta, enc_states, ensure_goal=True)
            r_dec, _ = self._label(eta, dec_states, ensure_goal=False)
            enc_rewards[:] = r_enc.reshape(enc_rewards.shape)
            dec_rewards[:] = r_dec.reshape(dec_rewards.shape)
            if d_enc is not None:
                enc_dones[:] = d_enc.reshape(enc_dones.shape)
        else:
            group = int(np.ceil(batch_size / num_eta))
            ranges = []
            for i in range(num_eta):
                lo, hi = i * group, min(batch_size, (i + 1) * group)
                if lo >= hi:
                    break
                eta = self.sample_reward_function()
                ranges.append(
                    self.reward_range_for(eta, enc_states[lo:hi].reshape(-1, enc_states.shape[-1]))
                )
                r_enc, d_enc = self._label(eta, enc_states[lo:hi], ensure_goal=True)
                r_dec, _ = self._label(eta, dec_states[lo:hi], ensure_goal=False)
                enc_rewards[lo:hi] = r_enc.reshape(hi - lo, -1)
                dec_rewards[lo:hi] = r_dec.reshape(hi - lo, -1)
                if d_enc is not None:
                    enc_dones[lo:hi] = d_enc.reshape(hi - lo, -1)

        # The shared embedding table needs one range: use the union of the
        # sampled functions' ranges (exact for the single-function update).
        reward_min = float(min(r[0] for r in ranges))
        reward_max = float(max(r[1] for r in ranges))
        if not np.isfinite([reward_min, reward_max]).all() or reward_max <= reward_min:
            reward_min, reward_max = self.config.reward_min, self.config.reward_max
        return enc_rewards, dec_rewards, (reward_min, reward_max), enc_dones, len(ranges)

    # -- forward / loss -----------------------------------------------------
    def encode(
        self,
        states: Any,
        rewards: Any,
        sample: bool = False,
        reward_min: Optional[float] = None,
        reward_max: Optional[float] = None,
        return_output: bool = False,
    ) -> Any:
        """Encode ``(s^e, eta(s^e))`` pairs into ``z`` (posterior mean by default)."""
        states_t = torch.as_tensor(_as_numpy_2d(states), dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(_as_numpy_1d(rewards), dtype=torch.float32, device=self.device)
        rewards_t = rewards_t.reshape(states_t.shape[0], states_t.shape[1])
        kwargs: Dict[str, Any] = {"sample": sample}
        if reward_min is not None and reward_max is not None:
            kwargs["reward_min"] = float(reward_min)
            kwargs["reward_max"] = float(reward_max)
        out = _call_with_supported_kwargs(self.encoder, states_t, rewards_t, **kwargs)
        z = out.z if hasattr(out, "z") else out
        return (z, out) if return_output else z

    def compute_loss(
        self,
        enc_states: Any,
        enc_rewards: Any,
        dec_states: Any,
        dec_target_rewards: Any,
        reward_min: float,
        reward_max: float,
        sample_z: Optional[bool] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Equation (6) loss for one update (no backward pass)."""
        enc_states_t = torch.as_tensor(_as_numpy_2d(enc_states), dtype=torch.float32, device=self.device)
        dec_states_t = torch.as_tensor(_as_numpy_2d(dec_states), dtype=torch.float32, device=self.device)
        enc_rewards_t = torch.as_tensor(enc_rewards, dtype=torch.float32, device=self.device)
        enc_rewards_t = enc_rewards_t.reshape(enc_states_t.shape[0], enc_states_t.shape[1])
        dec_target_t = torch.as_tensor(dec_target_rewards, dtype=torch.float32, device=self.device)
        dec_target_t = dec_target_t.reshape(dec_states_t.shape[0], dec_states_t.shape[1])

        if sample_z is None:
            sample_z = self.config.sample_z

        enc_out = _call_with_supported_kwargs(
            self.encoder,
            enc_states_t,
            enc_rewards_t,
            sample=sample_z,
            reward_min=float(reward_min),
            reward_max=float(reward_max),
        )
        z = enc_out.z if hasattr(enc_out, "z") else enc_out
        pred = _call_with_supported_kwargs(self.decoder, dec_states_t, z, return_output=True)
        if hasattr(pred, "mean") and not isinstance(pred, torch.Tensor):
            pred_mean = pred.mean
        elif isinstance(pred, Mapping):
            pred_mean = pred.get("mean", pred.get("prediction"))
        else:
            pred_mean = pred
        pred_mean = torch.as_tensor(pred_mean)

        terms = fre_reconstruction_kl_loss(
            encoder_output=enc_out,
            predictions=pred_mean,
            target_rewards=dec_target_t,
            beta=self.config.beta,
            loss_form=self.config.loss_form,
        )
        mu = getattr(enc_out, "mu", z)
        std = getattr(enc_out, "std", None)
        with torch.no_grad():
            extra = {
                "mean_target": float(dec_target_t.mean().detach().cpu()),
                "mean_prediction": float(pred_mean.mean().detach().cpu()),
                "mean_abs_error": float((pred_mean - dec_target_t).abs().mean().detach().cpu()),
                "z_norm": float(torch.as_tensor(z).norm(dim=-1).mean().detach().cpu()),
                "mu_norm": float(torch.as_tensor(mu).norm(dim=-1).mean().detach().cpu()),
                "mean_std": float(
                    torch.as_tensor(std).mean().detach().cpu()
                    if std is not None
                    else torch.ones((), device=self.device)
                ),
            }
        return terms["loss"], {"terms": terms, "enc_out": enc_out, "pred": pred_mean, "extra": extra}

    # -- training step ------------------------------------------------------
    def train_step(self, batch: Optional[Tuple[Any, Any]] = None) -> FRETrainOutput:
        """One Algorithm 1 phase-1 update: sample eta + K/K' states, maximise Eq. (6)."""
        if batch is not None:
            enc_states, dec_states = _as_numpy_2d(batch[0]), _as_numpy_2d(batch[1])
        else:
            enc_states, dec_states = self.sample_states()

        enc_rewards, dec_rewards, (reward_min, reward_max), _, num_eta = self._rewards_for_update(
            enc_states, dec_states
        )

        self.encoder.train()
        self.decoder.train()
        loss, info = self.compute_loss(
            enc_states, enc_rewards, dec_states, dec_rewards, reward_min, reward_max
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.trainable_parameters(), float(self.config.max_grad_norm)
            )
        self.optimizer.step()

        terms = info["terms"]
        output = FRETrainOutput(
            step=self.step_count,
            loss=float(terms["loss"].detach().cpu()),
            reconstruction_loss=float(terms["reconstruction_mse"].detach().cpu()),
            reconstruction_sum=float(terms["reconstruction_sum"].detach().cpu()),
            kl=float(terms["kl"].detach().cpu()),
            objective=float(terms["objective"].detach().cpu()),
            beta=float(self.config.beta),
            reward_min=float(reward_min),
            reward_max=float(reward_max),
            num_reward_functions=int(num_eta),
            extra=dict(info["extra"]),
        )
        output.extra["mean_reward"] = float(np.mean(enc_rewards))
        output.extra["frac_positive_rewards"] = float(np.mean(enc_rewards > 0))

        self.step_count += 1
        self.last_output = output
        self.last_loss = output.loss
        self.history.append(output.to_dict())
        return output

    # aliases: the strided controller probes for any of these names
    def update(self, *args: Any, **kwargs: Any) -> FRETrainOutput:
        return self.train_step(*args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> FRETrainOutput:
        return self.train_step(*args, **kwargs)

    def encoder_step(self, *args: Any, **kwargs: Any) -> FRETrainOutput:
        return self.train_step(*args, **kwargs)

    # -- loops -------------------------------------------------------------
    def train(
        self,
        num_steps: Optional[int] = None,
        callback: Optional[Callable[..., Any]] = None,
        log_interval: Optional[int] = None,
        encoder_steps: Optional[int] = None,
        progress: bool = False,
    ) -> List[Dict[str, float]]:
        """Run the phase-1 loop (150k steps for AntMaze, 1M for ExORL/Kitchen)."""
        if num_steps is None:
            num_steps = encoder_steps if encoder_steps is not None else self.config.encoder_training_steps
        num_steps = int(num_steps)
        interval = max(1, int(log_interval if log_interval is not None else self.config.log_interval))

        start = time.time()
        for i in range(num_steps):
            out = self.train_step()
            if callback is not None:
                try:
                    callback(out, self.step_count - 1)
                except TypeError:
                    callback(out)
            if self.step_count % interval == 0 or i == num_steps - 1:
                self._log(out.to_dict(), step=self.step_count)
                if progress:
                    print(
                        f"[FRE encoder] step {self.step_count}/{num_steps} "
                        f"loss={out.loss:.5f} recon={out.reconstruction_loss:.5f} "
                        f"kl={out.kl:.4f} ({time.time() - start:.1f}s)"
                    )
        return list(self.history)

    # -- diagnostics / quality gates ---------------------------------------
    def evaluate_reconstruction(
        self, num_batches: int = 8, batch: Optional[Tuple[Any, Any]] = None
    ) -> Dict[str, float]:
        """Held-out MSE between decoded and true rewards (Milestone A sanity)."""
        self.encoder.eval()
        self.decoder.eval()
        mses, kls = [], []
        with torch.no_grad():
            for _ in range(max(1, int(num_batches))):
                if batch is not None:
                    enc_states, dec_states = _as_numpy_2d(batch[0]), _as_numpy_2d(batch[1])
                else:
                    enc_states, dec_states = self.sample_states()
                enc_rewards, dec_rewards, (lo, hi), _, _ = self._rewards_for_update(
                    enc_states, dec_states
                )
                _, info = self.compute_loss(
                    enc_states, enc_rewards, dec_states, dec_rewards, lo, hi, sample_z=False
                )
                terms = info["terms"]
                mses.append(float(terms["reconstruction_mse"].detach().cpu()))
                kls.append(float(terms["kl"].detach().cpu()))
        return {"reconstruction_mse": float(np.mean(mses)), "kl": float(np.mean(kls))}

    def check_permutation_invariance(
        self, atol: float = 1e-5, seed: Optional[int] = None, num_checks: int = 1
    ) -> float:
        """Max |z - z_shuffled| under a random permutation of the K context tokens.

        Section 4.1: "Positional encodings and causal masking are not used, thus
        the inputs are treated as an unordered set."
        """
        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        worst = 0.0
        self.encoder.eval()
        with torch.no_grad():
            for _ in range(max(1, int(num_checks))):
                enc_states, dec_states = self.sample_states()
                enc_rewards, _, (lo, hi), _, _ = self._rewards_for_update(enc_states, dec_states)
                perm = rng.permutation(enc_states.shape[1])
                z1 = self.encode(enc_states, enc_rewards, sample=False, reward_min=lo, reward_max=hi)
                z2 = self.encode(
                    enc_states[:, perm, :],
                    enc_rewards[:, perm],
                    sample=False,
                    reward_min=lo,
                    reward_max=hi,
                )
                worst = max(worst, float((z1 - z2).abs().max().detach().cpu()))
        if worst > atol:
            raise AssertionError(
                f"encoder is not permutation invariant (max |dz| = {worst:.3e} > {atol:.1e})"
            )
        return worst

    def check_disjoint_states(self, num_samples: int = 8) -> None:
        """Section 4.1: "the states sampled for decoding are different than those used for encoding"."""
        for _ in range(max(1, int(num_samples))):
            enc_states, dec_states = self.sample_states()
            enc_flat = enc_states.reshape(-1, enc_states.shape[-1])
            dec_flat = dec_states.reshape(-1, dec_states.shape[-1])
            if enc_flat.shape[0] * dec_flat.shape[0] == 0:
                continue
            same = (enc_flat[:, None, :] == dec_flat[None, :, :]).all(-1)
            if same.any():
                raise AssertionError(
                    "encoder and decoder states overlap within an update; the replay buffer "
                    "must return disjoint sets (Section 4.1)"
                )

    def describe(self) -> Dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "latent_dim": self.config.latent_dim,
            "device": str(self.device),
            "batch_size": self.config.batch_size,
            "num_encoder_states": self.config.num_encoder_states,
            "num_decoder_states": self.config.num_decoder_states,
            "beta": self.config.beta,
            "learning_rate": self.config.learning_rate,
            "loss_form": self.config.loss_form,
            "step_count": self.step_count,
            "num_trainable_parameters": int(sum(p.numel() for p in self.trainable_parameters())),
        }

    # -- persistence -------------------------------------------------------
    def save(self, path: str, extra: Optional[Mapping[str, Any]] = None) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "config": self.config.to_dict(),
            "step_count": self.step_count,
        }
        if extra:
            payload.update(dict(extra))
        torch.save(payload, path)
        return path

    def load(self, path: str, load_optimizer: bool = False) -> Dict[str, Any]:
        payload = torch.load(path, map_location=self.device)
        if "encoder" in payload:
            self.encoder.load_state_dict(payload["encoder"])
        if "decoder" in payload:
            self.decoder.load_state_dict(payload["decoder"])
        if load_optimizer and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.step_count = int(payload.get("step_count", self.step_count))
        return payload

    # -- freeze (strided scheme) -------------------------------------------
    def freeze_encoder(self) -> Any:
        """Phase-2 gate: "After the encoder loss converges, we freeze the encoder"."""
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        if hasattr(self.encoder, "freeze"):
            try:
                self.encoder.freeze()
            except Exception:
                pass
        return self.encoder

    # convenience alias
    freeze = freeze_encoder


# ---------------------------------------------------------------------------
# functional API
# ---------------------------------------------------------------------------
def make_fre_trainer(
    encoder: Any = None,
    decoder: Any = None,
    replay_buffer: Any = None,
    prior: Any = None,
    state_dim: Optional[int] = None,
    latent_dim: Optional[int] = None,
    config: Optional[FRETrainerConfig] = None,
    device: Optional[Any] = None,
    **config_overrides: Any,
) -> FRETrainer:
    """Factory mirroring ``make_fre_encoder`` / ``make_reward_decoder``."""
    if isinstance(config, Mapping) and not isinstance(config, FRETrainerConfig):
        config = FRETrainerConfig.from_dict(config)
    return FRETrainer(
        encoder=encoder,
        decoder=decoder,
        replay_buffer=replay_buffer,
        prior=prior,
        config=config,
        state_dim=state_dim,
        latent_dim=latent_dim,
        device=device,
        **config_overrides,
    )


def train_fre_step(trainer: FRETrainer, *args: Any, **kwargs: Any) -> FRETrainOutput:
    """Run a single Equation (6) update on an existing trainer."""
    return trainer.train_step(*args, **kwargs)

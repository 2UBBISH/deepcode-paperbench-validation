"""Strided training controller for Functional Reward Encodings (FRE).

This module implements Algorithm 1 of *Zero-Shot Reinforcement Learning via
Functional Reward Encodings* (Section 4.3, "Offline RL with FRE"), i.e. the
two-phase ("strided") training procedure:

    # Train encoder
    while not converged do
        Sample reward function eta ~ p(eta)
        Sample K states for encoder {s_k^e} ~ D
        Sample K' states for decoder {s_k^d} ~ D
        Train FRE by maximizing Equation (6)
    end while
    # Train policy
    while not converged do
        Sample reward function eta ~ p(eta)
        Sample K states for encoder {s_k^e} ~ D
        Encode into latent vector z ~ p_theta({(s_k^e, eta(s_k^e))})
        Train pi(a|s,z), Q(s,a,z), V(s,z) using IQL with r = eta(s)
    end while

Verbatim motivation from the paper (Section 4.3, Practical Implementation):
    "We find that a strided training scheme leads to the most stable
     performance. In the strided scheme, we first only train the FRE encoder
     with gradients from the decoder (Equation (6)). During this time, the RL
     components are not trained. After the encoder loss converges, we freeze
     the encoder and then start the training of the RL networks using the
     frozen encoder's outputs. In this way, we can make the mapping from eta to
     z stationary during policy learning, which we found to be important to
     correctly estimate multitask Q values using TD learning."

Hyper-parameters (paper Table 3, section A):
    Batch Size = 512; Encoder Training Steps = 150,000 (1M for ExORL/Kitchen);
    Policy Training Steps = 850,000 (1M for ExORL/Kitchen); Reward Pairs to
    Encode K = 32; Reward Pairs to Decode K' = 8; Optimizer Adam; Learning Rate
    0.0001; beta KL Weight = 0.01; Target Update Rate = 0.001; Discount Factor =
    0.88; AWR Temperature = 3.0; IQL Expectile = 0.8.

Notes on details the paper leaves unspecified (documented deviations):
    * The paper says "after the encoder loss converges" without defining
      convergence.  We implement a moving-average relative-improvement
      detector with a minimum step floor, and always stop at the paper's step
      budget at the latest.
    * Section 4.3 says "a batch of reward functions eta are also sampled"; the
      pseudocode samples a single eta per iteration.  ``reward_functions_per_update``
      therefore defaults to 1 (Algorithm 1 verbatim) and the state-action batch
      is split evenly when more are requested.
    * z is sampled from the posterior during training (``sample_z=True``); the
      encoder's frozen parameters are verified to keep the mapping eta -> z
      stationary (correctness gate (iv) of the reproduction plan).
"""

from __future__ import annotations

import inspect
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Import plumbing: prefer a normal package import, but fall back to a direct
# sys.path insertion so this module can also be executed/imported standalone
# while the package is being built incrementally.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised implicitly
    from fre.models.encoder import (
        DEFAULT_LATENT_DIM,
        DEFAULT_NUM_ENCODER_STATES,
        DEFAULT_NUM_REWARD_EMBEDDINGS,
        FREEncoder,
    )
    from fre.models.rl_networks import (
        DEFAULT_RL_LAYERS,
        RLNetworks,
        make_rl_networks,
    )
    from fre.training.iql import IQLLosses, IQLTrainer, make_iql_trainer
except ImportError:  # pragma: no cover - fallback for standalone execution
    _PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)
    from fre.models.encoder import (  # type: ignore
        DEFAULT_LATENT_DIM,
        DEFAULT_NUM_ENCODER_STATES,
        DEFAULT_NUM_REWARD_EMBEDDINGS,
        FREEncoder,
    )
    from fre.models.rl_networks import (  # type: ignore
        DEFAULT_RL_LAYERS,
        RLNetworks,
        make_rl_networks,
    )
    from fre.training.iql import (  # type: ignore
        IQLLosses,
        IQLTrainer,
        make_iql_trainer,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE = 512
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_BETA = 0.01
DEFAULT_EXPECTILE = 0.8
DEFAULT_AWR_TEMPERATURE = 3.0
DEFAULT_DISCOUNT = 0.88
DEFAULT_TAU = 0.001

# Paper Table 3 step budgets.
DEFAULT_ENCODER_TRAINING_STEPS = 150_000
DEFAULT_POLICY_TRAINING_STEPS = 850_000
LONG_ENCODER_TRAINING_STEPS = 1_000_000  # ExORL / Kitchen
LONG_POLICY_TRAINING_STEPS = 1_000_000  # ExORL / Kitchen

# Per-domain (encoder_steps, policy_steps) budgets from Table 3.
DOMAIN_STEP_BUDGETS: Dict[str, Tuple[int, int]] = {
    "antmaze": (DEFAULT_ENCODER_TRAINING_STEPS, DEFAULT_POLICY_TRAINING_STEPS),
    "exorl": (LONG_ENCODER_TRAINING_STEPS, LONG_POLICY_TRAINING_STEPS),
    "kitchen": (LONG_ENCODER_TRAINING_STEPS, LONG_POLICY_TRAINING_STEPS),
}

# Domains whose encoder consumes physics-augmented states (Appendix C.2).
PHYSICS_AUGMENTED_DOMAINS = ("exorl",)

STAGE_ENCODER = "encoder"
STAGE_POLICY = "policy"

_LOG_FLOAT_KEYS = (
    "loss",
    "total_loss",
    "reconstruction_loss",
    "mse",
    "mse_loss",
    "kl",
    "kl_loss",
    "beta_kl",
    "value_loss",
    "q_loss",
    "policy_loss",
    "mean_q",
    "mean_v",
    "mean_advantage",
    "mean_weight",
    "mean_abs_td_error",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class StridedConfig:
    """Configuration for the strided (encoder -> frozen encoder -> RL) scheme.

    Defaults are the paper's Table 3 values.
    """

    # Phase 1: encoder + decoder (Equation (6)).
    encoder_training_steps: int = DEFAULT_ENCODER_TRAINING_STEPS
    # Phase 2: IQL on the frozen encoder's outputs.
    policy_training_steps: int = DEFAULT_POLICY_TRAINING_STEPS

    batch_size: int = DEFAULT_BATCH_SIZE
    num_encoder_states: int = DEFAULT_NUM_ENCODER_STATES  # K = 32
    num_decoder_states: int = 8  # K' = 8
    reward_functions_per_update: int = 1  # Algorithm 1: one eta per iteration

    learning_rate: float = DEFAULT_LEARNING_RATE
    beta: float = DEFAULT_BETA
    expectile: float = DEFAULT_EXPECTILE
    awr_temperature: float = DEFAULT_AWR_TEMPERATURE
    discount: float = DEFAULT_DISCOUNT
    tau: float = DEFAULT_TAU

    # Convergence detection for the encoder phase (paper: "after the encoder
    # loss converges").  NaN/negative values disable early stopping and simply
    # use the full step budget.
    convergence_window: int = 200
    convergence_rtol: float = 0.01
    convergence_min_steps: int = 10_000

    # Logging / bookkeeping.
    log_interval: int = 1_000
    stationary_check_interval: int = 10_000
    save_encoder_only: bool = False

    # Behaviour switches.
    sample_z: bool = True  # sample z ~ p_theta(z | .) during policy training
    use_encoder_inputs: bool = False  # ExORL: physics-augmented encoder input
    freeze_encoder_after_convergence: bool = True
    reward_done_masks: bool = True  # combine eta's done mask into terminals
    seed: int = 0

    # ------------------------------------------------------------------
    @classmethod
    def for_domain(cls, domain: str, **overrides: Any) -> "StridedConfig":
        """Build a config with the step budget of ``domain`` (Table 3)."""
        domain_key = str(domain).lower()
        encoder_steps, policy_steps = DOMAIN_STEP_BUDGETS.get(
            domain_key, (DEFAULT_ENCODER_TRAINING_STEPS, DEFAULT_POLICY_TRAINING_STEPS)
        )
        config = cls(
            encoder_training_steps=encoder_steps,
            policy_training_steps=policy_steps,
            use_encoder_inputs=domain_key in PHYSICS_AUGMENTED_DOMAINS,
        )
        for key, value in overrides.items():
            if not hasattr(config, key):
                raise TypeError(f"Unknown StridedConfig field: {key!r}")
            setattr(config, key, value)
        return config

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Convergence detection
# ---------------------------------------------------------------------------
class ConvergenceDetector:
    """Detects convergence of the encoder loss (moving-average relative gain).

    The paper only states that the encoder is frozen "after the encoder loss
    converges"; this detector implements a documented, deterministic rule: the
    loss is considered converged once the mean over the most recent
    ``window`` steps has improved by less than ``rtol`` relative to the mean
    over the previous window, provided at least ``min_steps`` steps were taken.
    ``min_steps <= 0``/``rtol < 0`` disables early stopping.
    """

    def __init__(
        self,
        window: int = 200,
        rtol: float = 0.01,
        min_steps: int = 10_000,
    ) -> None:
        self.window = int(max(1, window))
        self.rtol = float(rtol)
        self.min_steps = int(max(0, min_steps))
        self.losses: List[float] = []
        self.reference: Optional[float] = None
        self.converged_at: Optional[int] = None

    @property
    def enabled(self) -> bool:
        return self.rtol >= 0.0 and self.min_steps > 0

    def update(self, loss: float) -> bool:
        """Record a loss value; return True if convergence is detected now."""
        self.losses.append(float(loss))
        step = len(self.losses)
        if not self.enabled:
            return False
        if step < self.min_steps:
            return False
        if step % self.window != 0:
            return False

        recent = float(np.mean(self.losses[-self.window :]))
        if self.reference is None:
            self.reference = recent
            return False

        denom = abs(self.reference) + 1e-12
        improved = (self.reference - recent) / denom
        self.reference = recent

        if improved < self.rtol:
            self.converged_at = step
            return True
        return False

    def moving_average(self, window: Optional[int] = None) -> float:
        if not self.losses:
            return float("nan")
        window = self.window if window is None else int(window)
        return float(np.mean(self.losses[-window:]))


# ---------------------------------------------------------------------------
# Freezing helpers (correctness gates (iv) and "z stationary")
# ---------------------------------------------------------------------------
def freeze_encoder(encoder: torch.nn.Module) -> torch.nn.Module:
    """Freeze an encoder: no gradients and eval mode (strided scheme, §4.3)."""
    if hasattr(encoder, "freeze") and callable(getattr(encoder, "freeze")):
        try:
            encoder.freeze()
        except TypeError:  # pragma: no cover - exotic signature
            pass
    for param in encoder.parameters():
        param.requires_grad_(False)
    encoder.eval()
    return encoder


def unfreeze_encoder(encoder: torch.nn.Module) -> torch.nn.Module:
    """Undo :func:`freeze_encoder` (only used before the phase transition)."""
    for param in encoder.parameters():
        param.requires_grad_(True)
    encoder.train()
    return encoder


def encoder_is_frozen(encoder: torch.nn.Module) -> bool:
    """True if every encoder parameter has ``requires_grad == False``."""
    params = list(encoder.parameters())
    if not params:
        return True
    return not any(p.requires_grad for p in params)


def assert_encoder_frozen(encoder: torch.nn.Module) -> None:
    """Raise if any encoder parameter would receive gradients during phase 2."""
    if not encoder_is_frozen(encoder):
        raise RuntimeError(
            "FRE encoder is not frozen; the strided scheme requires the mapping "
            "eta -> z to be stationary while the RL components are trained."
        )


def count_trainable_parameters(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def trainable_parameters(modules: Sequence[torch.nn.Module]) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    for module in modules:
        if module is None:
            continue
        params.extend([p for p in module.parameters() if p.requires_grad])
    return params


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class StridedTrainerResult:
    """Summary of a strided FRE training run."""

    encoder_losses: List[float] = field(default_factory=list)
    encoder_steps: int = 0
    encoder_converged_at: Optional[int] = None
    policy_history: List[Dict[str, float]] = field(default_factory=list)
    policy_steps: int = 0
    frozen_at_step: Optional[int] = None
    stationary_checks: List[float] = field(default_factory=list)
    wall_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "encoder_steps": self.encoder_steps,
            "encoder_converged_at": self.encoder_converged_at,
            "encoder_final_loss": (
                float(np.mean(self.encoder_losses[-100:])) if self.encoder_losses else float("nan")
            ),
            "policy_steps": self.policy_steps,
            "frozen_at_step": self.frozen_at_step,
            "stationary_checks": list(self.stationary_checks),
            "wall_time": self.wall_time,
        }


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------
class StridedTrainer:
    """Implements Algorithm 1: encoder phase, then frozen-encoder IQL phase.

    Parameters
    ----------
    encoder:
        The FRE encoder (``fre.models.encoder.FREEncoder``).  It is frozen via
        :func:`freeze_encoder` at the phase transition.
    decoder:
        The FRE reward decoder (``fre.models.decoder.RewardDecoder``) trained
        jointly with the encoder during phase 1.  May be ``None`` if an
        external ``encoder_trainer`` owns the decoder.
    replay_buffer:
        Offline buffer exposing ``sample_states``, ``sample_transitions``,
        ``obs_dim``/``act_dim`` (see ``fre.data.replay.ReplayBuffer``).
    prior:
        Sampler of reward functions eta ~ p(eta) (``fre.reward_priors``).
        Only ``prior.sample(rng)`` is required.
    networks:
        Optional pre-built ``RLNetworks`` bundle for phase 2.
    config:
        :class:`StridedConfig` (defaults to the paper's Table 3 values).
    encoder_trainer:
        Optional object implementing the Equation (6) update (typically
        ``fre.training.fre_trainer.FRETrainer``).  If ``None`` it is created
        lazily with keyword-introspection so the two modules stay decoupled.
    iql_trainer:
        Optional ``fre.training.iql.IQLTrainer``; created lazily if ``None``.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: Optional[torch.nn.Module] = None,
        replay_buffer: Any = None,
        prior: Any = None,
        networks: Optional[RLNetworks] = None,
        config: Optional[StridedConfig] = None,
        device: Optional[Any] = None,
        encoder_trainer: Any = None,
        iql_trainer: Optional[IQLTrainer] = None,
        logger: Any = None,
        callbacks: Optional[Sequence[Callable[[str, Dict[str, Any]], None]]] = None,
        **config_overrides: Any,
    ) -> None:
        if config is None:
            config = StridedConfig()
        for key, value in config_overrides.items():
            if not hasattr(config, key):
                raise TypeError(f"Unknown StridedConfig field: {key!r}")
            setattr(config, key, value)
        self.config = config

        self.device = torch.device(device) if device is not None else self._infer_device(encoder)
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device) if decoder is not None else None
        self.replay_buffer = replay_buffer
        self.prior = prior
        self.logger = logger
        self.callbacks = list(callbacks) if callbacks else []

        self.rng = np.random.default_rng(int(config.seed))
        self.encoder_trainer = encoder_trainer
        self.iql_trainer = iql_trainer
        self.networks = networks

        # Infer dimensions for lazily-built RL networks.
        self.obs_dim = getattr(replay_buffer, "obs_dim", None)
        self.act_dim = getattr(replay_buffer, "act_dim", None)
        if networks is not None:
            self.obs_dim = getattr(networks, "obs_dim", self.obs_dim)
            self.act_dim = getattr(networks, "act_dim", self.act_dim)
        self.latent_dim = int(getattr(encoder, "latent_dim", DEFAULT_LATENT_DIM))

        # Bookkeeping.
        self.global_step = 0
        self.encoder_step = 0
        self.policy_step = 0
        self.frozen = encoder_is_frozen(self.encoder)
        self._stationary_context: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self._stationary_reference: Optional[torch.Tensor] = None
        self.result = StridedTrainerResult()

    # -- small utilities ---------------------------------------------------
    @staticmethod
    def _infer_device(module: torch.nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:  # pragma: no cover - parameterless module
            return torch.device("cpu")

    def _emit(self, stage: str, stats: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback(stage, stats)
        if self.logger is None:
            return
        for name in ("log", "log_stats", "log_metrics", "record"):
            fn = getattr(self.logger, name, None)
            if callable(fn):
                try:
                    fn(stage, stats)
                except TypeError:
                    try:
                        fn(stats)
                    except TypeError:  # pragma: no cover
                        pass
                return

    @staticmethod
    def _normalize_stats(raw: Any) -> Dict[str, float]:
        """Coerce a trainer's return value into a flat float dict."""
        if raw is None:
            return {}
        if isinstance(raw, dict):
            source = raw
        elif hasattr(raw, "to_dict"):
            source = raw.to_dict()
        else:
            source = {
                key: getattr(raw, key)
                for key in ("loss", "value_loss", "q_loss", "policy_loss", "total_loss")
                if hasattr(raw, key)
            }
        stats: Dict[str, float] = {}
        for key, value in source.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean().item()
            if isinstance(value, (int, float, np.floating, np.integer)):
                fvalue = float(value)
                if math.isfinite(fvalue):
                    stats[str(key)] = fvalue
        return stats

    @staticmethod
    def _call_flexibly(fn: Callable, kwargs: Dict[str, Any]) -> Any:
        """Call ``fn`` passing only the keyword arguments its signature accepts."""
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            return fn(**kwargs)
        accepts_var_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
        )
        if accepts_var_kwargs:
            return fn(**kwargs)
        allowed = {
            name: value
            for name, value in kwargs.items()
            if name in signature.parameters
            and signature.parameters[name].kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        return fn(**allowed)

    # -- reward labels -----------------------------------------------------
    def _reward_labels(self, eta: Any, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate eta on ``states`` -> (rewards, dones) as float/bool arrays."""
        states_arr = np.asarray(states, dtype=np.float32)
        if states_arr.ndim == 1:
            states_arr = states_arr[None, :]

        rewards: np.ndarray
        dones: Optional[np.ndarray] = None
        label_fn = getattr(eta, "label", None)
        if callable(label_fn):
            out = self._call_flexibly(label_fn, {"states": states_arr, "ensure_goal": False})
            if isinstance(out, tuple) and len(out) == 2:
                rewards, dones = out[0], out[1]
            else:
                rewards = out
        else:
            out = eta(states_arr)
            if isinstance(out, tuple) and len(out) == 2:
                rewards, dones = out[0], out[1]
            else:
                rewards = out

        if isinstance(rewards, torch.Tensor):
            rewards = rewards.detach().cpu().numpy()
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)

        if dones is None:
            dones_arr = np.zeros_like(rewards, dtype=bool)
        else:
            if isinstance(dones, torch.Tensor):
                dones = dones.detach().cpu().numpy()
            dones_arr = np.asarray(dones).reshape(-1).astype(bool)
        return rewards, dones_arr

    def _reward_range(self, eta: Any, states: np.ndarray) -> Tuple[float, float]:
        """Reward range used for the 32-bin reward discretization of tokens."""
        bounds_fn = getattr(eta, "reward_bounds", None)
        if callable(bounds_fn):
            try:
                bounds = self._call_flexibly(bounds_fn, {"states": states})
            except Exception:  # pragma: no cover - defensive
                bounds = None
            if bounds is not None and len(bounds) == 2:
                lo, hi = float(bounds[0]), float(bounds[1])
                if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
                    return lo, hi
        lo = float(getattr(eta, "reward_min", -1.0))
        hi = float(getattr(eta, "reward_max", 1.0))
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            lo, hi = -1.0, 1.0
        return lo, hi

    # -- sampling ----------------------------------------------------------
    def _require_replay(self) -> Any:
        if self.replay_buffer is None:
            raise ValueError("replay_buffer is required to sample states/transitions.")
        return self.replay_buffer

    def _sample_context_states(self, num_states: int, encoder_input: bool) -> np.ndarray:
        buffer = self._require_replay()
        try:
            states = buffer.sample_states(num_states, rng=self.rng, encoder_input=encoder_input)
        except TypeError:  # buffer without the encoder_input switch
            states = buffer.sample_states(num_states, rng=self.rng)
        return np.asarray(states, dtype=np.float32)

    def _sample_transitions(self, batch_size: int) -> Any:
        buffer = self._require_replay()
        try:
            return buffer.sample_transitions(batch_size, rng=self.rng, with_encoder_inputs=True)
        except TypeError:
            return buffer.sample_transitions(batch_size, rng=self.rng)

    def _sample_reward_functions(self, num_functions: int) -> List[Any]:
        if self.prior is None:
            raise ValueError("A reward prior p(eta) is required by Algorithm 1.")
        if hasattr(self.prior, "sample_many"):
            try:
                sampled = list(self.prior.sample_many(num_functions, rng=self.rng))
                if len(sampled) == num_functions:
                    return sampled
            except (TypeError, NotImplementedError):  # pragma: no cover
                pass
        return [self.prior.sample(self.rng) for _ in range(num_functions)]

    # -- phase 1: encoder + decoder (Equation (6)) -------------------------
    def _ensure_encoder_trainer(self) -> Any:
        if self.encoder_trainer is not None:
            return self.encoder_trainer

        fre_trainer_cls = None
        try:  # pragma: no cover - import depends on build order
            from fre.training import fre_trainer as _ft

            for name in ("FRETrainer", "FREEncoderTrainer", "EncoderTrainer", "FRETraining"):
                if hasattr(_ft, name):
                    fre_trainer_cls = getattr(_ft, name)
                    break
        except Exception:  # pragma: no cover
            fre_trainer_cls = None

        if fre_trainer_cls is None:
            raise RuntimeError(
                "No encoder trainer available: pass `encoder_trainer=` explicitly or "
                "provide fre.training.fre_trainer with a FRETrainer class."
            )

        kwargs = {
            "encoder": self.encoder,
            "decoder": self.decoder,
            "replay_buffer": self.replay_buffer,
            "reward_prior": self.prior,
            "prior": self.prior,
            "config": self.config,
            "cfg": self.config,
            "device": self.device,
            "beta": self.config.beta,
            "learning_rate": self.config.learning_rate,
            "lr": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "num_encoder_states": self.config.num_encoder_states,
            "num_decoder_states": self.config.num_decoder_states,
            "num_reward_states": self.config.num_encoder_states,
            "num_decode_states": self.config.num_decoder_states,
            "use_encoder_inputs": self.config.use_encoder_inputs,
            "seed": self.config.seed,
        }
        try:
            signature = inspect.signature(fre_trainer_cls)
        except (TypeError, ValueError):  # pragma: no cover
            signature = None
        if signature is not None and not any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
        ):
            kwargs = {
                name: value
                for name, value in kwargs.items()
                if name in signature.parameters
                and signature.parameters[name].kind
                in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
        self.encoder_trainer = fre_trainer_cls(**kwargs)
        return self.encoder_trainer

    def encoder_step(self) -> Dict[str, float]:
        """One Equation (6) update: sample eta, K encoder + K' decoder states."""
        trainer = self._ensure_encoder_trainer()
        stats: Dict[str, float] = {}
        for name in ("train_step", "update", "step", "train_on_batch", "train", "optimize"):
            fn = getattr(trainer, name, None)
            if callable(fn):
                raw = self._call_flexibly(
                    fn,
                    {
                        "num_encoder_states": self.config.num_encoder_states,
                        "num_decoder_states": self.config.num_decoder_states,
                        "num_reward_states": self.config.num_encoder_states,
                        "num_decode_states": self.config.num_decoder_states,
                        "batch_size": self.config.batch_size,
                        "sample_encoder_states": None,
                        "sample_decoder_states": None,
                    },
                )
                stats = self._normalize_stats(raw)
                break
        else:  # pragma: no cover - unsatisfiable with the built-in trainer
            raise AttributeError(
                f"{type(trainer).__name__} exposes no known training entry point."
            )

        self.encoder_step += 1
        self.global_step += 1
        for key in ("loss", "total_loss"):
            if key in stats:
                self.result.encoder_losses.append(stats[key])
                break
        return stats

    def train_encoder(
        self,
        num_steps: Optional[int] = None,
        callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
        detector: Optional[ConvergenceDetector] = None,
    ) -> List[float]:
        """Algorithm 1, first loop: train the FRE encoder via Equation (6)."""
        cfg = self.config
        max_steps = int(num_steps if num_steps is not None else cfg.encoder_training_steps)
        if detector is None:
            detector = ConvergenceDetector(
                window=cfg.convergence_window,
                rtol=cfg.convergence_rtol,
                min_steps=min(cfg.convergence_min_steps, max_steps),
            )

        self.encoder.train()
        if self.decoder is not None:
            self.decoder.train()

        for _ in range(max_steps):
            stats = self.encoder_step()
            loss = stats.get("loss", stats.get("total_loss", float("nan")))
            converged = detector.update(loss)

            if callback is not None:
                callback(self.encoder_step, stats)
            if cfg.log_interval and self.encoder_step % cfg.log_interval == 0:
                payload = {"step": self.encoder_step, "stage": STAGE_ENCODER}
                payload.update(stats)
                payload["moving_average"] = detector.moving_average()
                self._emit(STAGE_ENCODER, payload)
            if converged:
                self.result.encoder_converged_at = self.encoder_step
                break

        self.result.encoder_steps = self.encoder_step
        return list(self.result.encoder_losses)

    # -- phase 2: frozen-encoder IQL ---------------------------------------
    def _ensure_rl_networks(self) -> RLNetworks:
        if self.networks is not None:
            return self.networks
        if self.obs_dim is None or self.act_dim is None:
            raise ValueError(
                "obs_dim/act_dim unavailable: pass pre-built `networks=` or a replay "
                "buffer exposing `obs_dim` and `act_dim`."
            )
        self.networks = make_rl_networks(
            obs_dim=int(self.obs_dim),
            act_dim=int(self.act_dim),
            latent_dim=self.latent_dim,
            hidden_dims=DEFAULT_RL_LAYERS,
            encoder=self.encoder,
            freeze_encoder=self.frozen,
        )
        return self.networks

    def _ensure_iql_trainer(self) -> IQLTrainer:
        if self.iql_trainer is not None:
            return self.iql_trainer
        networks = self._ensure_rl_networks()
        cfg = self.config
        self.iql_trainer = make_iql_trainer(
            obs_dim=int(self.obs_dim),
            act_dim=int(self.act_dim),
            latent_dim=self.latent_dim,
            hidden_dims=DEFAULT_RL_LAYERS,
            networks=networks,
            encoder=self.encoder,
            freeze_encoder=True,
            device=self.device,
            learning_rate=cfg.learning_rate,
            expectile=cfg.expectile,
            awr_temperature=cfg.awr_temperature,
            discount=cfg.discount,
            tau=cfg.tau,
            batch_size=cfg.batch_size,
        )
        return self.iql_trainer

    def freeze_encoder_now(self) -> None:
        """Freeze the encoder (phase transition) and record the event."""
        freeze_encoder(self.encoder)
        self.frozen = True
        self.result.frozen_at_step = self.encoder_step
        assert_encoder_frozen(self.encoder)
        if self.iql_trainer is not None and hasattr(self.iql_trainer, "networks"):
            networks = self.iql_trainer.networks
            if networks is not None and hasattr(networks, "attach_encoder"):
                networks.attach_encoder(self.encoder, freeze=True)
        if self.networks is not None and hasattr(self.networks, "attach_encoder"):
            self.networks.attach_encoder(self.encoder, freeze=True)
        self._capture_stationary_reference()

    def _capture_stationary_reference(self, num_states: Optional[int] = None) -> None:
        """Store a fixed (context, labels) pair for the stationarity check."""
        if self._stationary_context is not None:
            return
        num_states = int(num_states or self.config.num_encoder_states)
        states = self._sample_context_states(num_states, self.config.use_encoder_inputs)
        rewards = self.rng.uniform(-1.0, 1.0, size=(len(states),)).astype(np.float32)
        self._stationary_context = (
            torch.as_tensor(states, dtype=torch.float32, device=self.device).unsqueeze(0),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(0),
            torch.zeros(1, device=self.device),  # reward_min
        )
        # Use a unit range for the reference check so the discretization is fixed.
        self._stationary_context = (
            self._stationary_context[0],
            self._stationary_context[1],
            torch.zeros(2, device=self.device),
        )
        self._stationary_context = (
            self._stationary_context[0],
            self._stationary_context[1],
            torch.tensor([-1.0, 1.0], device=self.device),
        )
        self._stationary_reference = self._encode_z(
            self._stationary_context[0],
            self._stationary_context[1],
            self._stationary_context[2][0].item(),
            self._stationary_context[2][1].item(),
            sample=False,
        ).clone()

    def verify_encoder_stationary(self, atol: float = 0.0) -> float:
        """Re-encode a fixed context; return the max deviation in ``z``.

        With a correctly frozen encoder this must be exactly 0 (correctness gate
        (iv) of the reproduction plan: "encoder frozen during phase 2").
        """
        if self._stationary_context is None or self._stationary_reference is None:
            return 0.0
        assert_encoder_frozen(self.encoder)
        states, rewards, bounds = self._stationary_context
        with torch.no_grad():
            z = self._encode_z(
                states, rewards, float(bounds[0].item()), float(bounds[1].item()), sample=False
            )
        deviation = float((z - self._stationary_reference).abs().max().item())
        if atol > 0.0 and deviation > atol:
            raise RuntimeError(
                f"Latent mapping eta -> z changed by {deviation:.3e} while the encoder "
                "should be frozen (strided scheme requirement)."
            )
        self.result.stationary_checks.append(deviation)
        return deviation

    def _encode_z(
        self,
        context_states: torch.Tensor,
        context_rewards: torch.Tensor,
        reward_min: Optional[float] = None,
        reward_max: Optional[float] = None,
        sample: Optional[bool] = None,
    ) -> torch.Tensor:
        """Encode a batch of reward-labelled state sets into z (B, latent)."""
        if sample is None:
            sample = self.config.sample_z
        out = self.encoder(
            context_states,
            context_rewards,
            sample=sample,
            reward_min=reward_min,
            reward_max=reward_max,
        )
        z = getattr(out, "z", out)
        if not isinstance(z, torch.Tensor):  # pragma: no cover - defensive
            z = torch.as_tensor(z, device=self.device)
        return z

    def policy_step(self) -> Dict[str, float]:
        """One Algorithm 1 policy update on the frozen encoder's outputs."""
        cfg = self.config
        assert_encoder_frozen(self.encoder)
        iql = self._ensure_iql_trainer()

        num_functions = max(1, int(cfg.reward_functions_per_update))
        etas = self._sample_reward_functions(num_functions)
        per_function = max(1, int(cfg.batch_size) // num_functions)

        zs: List[torch.Tensor] = []
        observations: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        next_observations: List[np.ndarray] = []
        rewards: List[np.ndarray] = []
        terminals: List[np.ndarray] = []

        self.encoder.eval()
        for eta in etas:
            # Algorithm 1: sample K states, label them with eta, encode into z.
            context_states = self._sample_context_states(
                cfg.num_encoder_states, cfg.use_encoder_inputs
            )
            context_rewards, _ = self._reward_labels(eta, context_states)
            reward_min, reward_max = self._reward_range(eta, context_states)
            with torch.no_grad():
                z = self._encode_z(
                    torch.as_tensor(context_states, dtype=torch.float32, device=self.device).unsqueeze(0),
                    torch.as_tensor(context_rewards, dtype=torch.float32, device=self.device).unsqueeze(0),
                    reward_min,
                    reward_max,
                    sample=cfg.sample_z,
                )

            batch = self._sample_transitions(per_function)
            obs = np.asarray(batch.observations, dtype=np.float32)
            act = np.asarray(batch.actions, dtype=np.float32)
            next_obs = np.asarray(batch.next_observations, dtype=np.float32)
            term = np.asarray(batch.terminals).reshape(-1).astype(bool)

            step_rewards, step_dones = self._reward_labels(eta, obs)  # r = eta(s)
            if cfg.reward_done_masks:
                term = np.logical_or(term, step_dones)

            zs.append(z.expand(obs.shape[0], -1))
            observations.append(obs)
            actions.append(act)
            next_observations.append(next_obs)
            rewards.append(step_rewards)
            terminals.append(term.astype(np.float32))

        z_batch = torch.cat(zs, dim=0).detach()
        obs_batch = torch.as_tensor(np.concatenate(observations, 0), dtype=torch.float32, device=self.device)
        act_batch = torch.as_tensor(np.concatenate(actions, 0), dtype=torch.float32, device=self.device)
        next_batch = torch.as_tensor(np.concatenate(next_observations, 0), dtype=torch.float32, device=self.device)
        rew_batch = torch.as_tensor(np.concatenate(rewards, 0), dtype=torch.float32, device=self.device)
        term_batch = torch.as_tensor(np.concatenate(terminals, 0), dtype=torch.float32, device=self.device)

        losses = iql.update(
            observations=obs_batch,
            actions=act_batch,
            next_observations=next_batch,
            rewards=rew_batch,
            terminals=term_batch,
            z=z_batch,
        )
        stats = self._normalize_stats(losses)
        stats.setdefault("mean_reward", float(rew_batch.mean().item()))

        self.policy_step += 1
        self.global_step += 1
        self.result.policy_history.append(stats)
        return stats

    def train_policy(
        self,
        num_steps: Optional[int] = None,
        callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> List[Dict[str, float]]:
        """Algorithm 1, second loop: IQL with r = eta(s) on frozen z."""
        cfg = self.config
        if cfg.freeze_encoder_after_convergence or not self.frozen:
            self.freeze_encoder_now()
        else:  # already frozen externally
            self.frozen = True
            self._capture_stationary_reference()
        assert_encoder_frozen(self.encoder)

        self._ensure_rl_networks()
        min_steps = min(cfg.stationary_check_interval, int(num_steps or cfg.policy_training_steps))
        self.verify_encoder_stationary()
        del min_steps

        max_steps = int(num_steps if num_steps is not None else cfg.policy_training_steps)
        for _ in range(max_steps):
            stats = self.policy_step()
            if callback is not None:
                callback(self.policy_step, stats)
            if cfg.log_interval and self.policy_step % cfg.log_interval == 0:
                payload = {"step": self.policy_step, "stage": STAGE_POLICY}
                payload.update(stats)
                self._emit(STAGE_POLICY, payload)
            if (
                cfg.stationary_check_interval
                and self.policy_step % cfg.stationary_check_interval == 0
            ):
                self.verify_encoder_stationary()

        self.result.policy_steps = self.policy_step
        return list(self.result.policy_history)

    def run(
        self,
        encoder_steps: Optional[int] = None,
        policy_steps: Optional[int] = None,
        encoder_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
        policy_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> StridedTrainerResult:
        """Full Algorithm 1: encoder phase followed by the frozen-policy phase."""
        start = time.time()
        self.train_encoder(num_steps=encoder_steps, callback=encoder_callback)
        self.train_policy(num_steps=policy_steps, callback=policy_callback)
        self.result.wall_time = time.time() - start
        return self.result

    # -- persistence -------------------------------------------------------
    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload: Dict[str, Any] = {
            "encoder": self.encoder.state_dict(),
            "encoder_frozen": encoder_is_frozen(self.encoder),
            "config": self.config.to_dict(),
            "encoder_step": self.encoder_step,
            "policy_step": self.policy_step,
            "result": self.result.to_dict(),
        }
        if self.decoder is not None and not self.config.save_encoder_only:
            payload["decoder"] = self.decoder.state_dict()
        if self.networks is not None:
            payload["networks"] = self.networks.state_dict()
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        return path

    def load(self, path: str, load_networks: bool = True) -> Dict[str, Any]:
        payload = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(payload["encoder"])
        if self.decoder is not None and "decoder" in payload:
            self.decoder.load_state_dict(payload["decoder"])
        if load_networks and self.networks is not None and "networks" in payload:
            self.networks.load_state_dict(payload["networks"])
        self.encoder_step = int(payload.get("encoder_step", self.encoder_step))
        self.policy_step = int(payload.get("policy_step", self.policy_step))
        return payload

    def describe(self) -> Dict[str, Any]:
        return {
            "name": "StridedTrainer",
            "latent_dim": self.latent_dim,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "frozen": bool(self.frozen),
            "encoder_steps": self.encoder_step,
            "policy_steps": self.policy_step,
            "encoder_trainable_params": count_trainable_parameters(self.encoder),
            "networks_trainable_params": (
                None if self.networks is None else self.networks.num_trainable_parameters()
                if hasattr(self.networks, "num_trainable_parameters")
                else count_trainable_parameters(self.networks)
            ),
            "config": self.config.to_dict(),
        }


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------
def train_fre_strided(
    encoder: torch.nn.Module,
    decoder: Optional[torch.nn.Module] = None,
    replay_buffer: Any = None,
    prior: Any = None,
    *,
    domain: Optional[str] = None,
    config: Optional[StridedConfig] = None,
    train_encoder_steps: Optional[int] = None,
    train_policy_steps: Optional[int] = None,
    networks: Optional[RLNetworks] = None,
    device: Optional[Any] = None,
    encoder_trainer: Any = None,
    iql_trainer: Optional[IQLTrainer] = None,
    logger: Any = None,
    callbacks: Optional[Sequence[Callable[[str, Dict[str, Any]], None]]] = None,
    **config_overrides: Any,
) -> Tuple[StridedTrainer, StridedTrainerResult]:
    """Run Algorithm 1 end-to-end and return ``(trainer, result)``."""
    if config is None:
        config = StridedConfig.for_domain(domain) if domain else StridedConfig()
    trainer = StridedTrainer(
        encoder=encoder,
        decoder=decoder,
        replay_buffer=replay_buffer,
        prior=prior,
        networks=networks,
        config=config,
        device=device,
        encoder_trainer=encoder_trainer,
        iql_trainer=iql_trainer,
        logger=logger,
        callbacks=callbacks,
        **config_overrides,
    )
    result = trainer.run(
        encoder_steps=train_encoder_steps,
        policy_steps=train_policy_steps,
    )
    return trainer, result


__all__ = [
    "StridedConfig",
    "StridedTrainer",
    "StridedTrainerResult",
    "ConvergenceDetector",
    "freeze_encoder",
    "unfreeze_encoder",
    "encoder_is_frozen",
    "assert_encoder_frozen",
    "count_trainable_parameters",
    "trainable_parameters",
    "train_fre_strided",
    "DOMAIN_STEP_BUDGETS",
    "PHYSICS_AUGMENTED_DOMAINS",
    "DEFAULT_ENCODER_TRAINING_STEPS",
    "DEFAULT_POLICY_TRAINING_STEPS",
    "LONG_ENCODER_TRAINING_STEPS",
    "LONG_POLICY_TRAINING_STEPS",
    "STAGE_ENCODER",
    "STAGE_POLICY",
]

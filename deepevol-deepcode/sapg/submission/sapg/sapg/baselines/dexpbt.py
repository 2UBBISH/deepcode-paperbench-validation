"""DexPBT baseline for the SAPG paper (Section 5.2, "DexPBT").

From Section 5.2:

    "DexPBT (Petrenko et al., 2023) A framework that combines population-based
     training with PPO.  ``N`` Environments are divided into ``M`` groups, each
     containing ``N/M`` environments.  ``M`` separate policies are trained using
     PPO in each group of environments with different hyperparameters.  At
     regular intervals, the worst-performing policies are replaced with the
     weights of best-performing policies and their hyperparameters are mutated
     randomly."

Additional experimental details from Section 5.2 that this module follows:

* ``N = 24576`` environments, ``M = 6`` policies for DexPBT as well,
* the same tasks and network families as SAPG (recurrent policy for the
  AllegroKuka tasks, MLP policy for ShadowHand / AllegroHand),
* 16 steps of experience per environment collected before each PPO update,
* ``~2e10`` transitions total, 5 seeds, and the paper's standard-error band
  ``2/sqrt(n) * sum_i (y(t) - y_i(t))^2``.

Unlike SAPG, DexPBT policies are *independent* (no shared backbone and no
``phi_j`` latents) -- the population member's weights and hyperparameters are
the only carrier of diversity.  The population is therefore built from
``M`` separate :class:`~sapg.models.actor.ActorCritic` policies, each acting on
its own contiguous block of environments.

This module is importable without torch / IsaacGym: heavy imports happen lazily
inside the functions that need them, mirroring ``sapg/baselines/pql.py``.
"""

from __future__ import annotations

import copy
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - exercised only when torch is installed
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# paper-level constants
# ---------------------------------------------------------------------------

DEXPBT_METHODS: Tuple[str, ...] = (
    "dexpbt",
    "pbt",
    "expbt",
    "dex_pbt",
    "dexpbt_ppo",
    "expbt_ppo",
)

DEFAULT_NUM_ENVS = 24576
DEFAULT_NUM_POLICIES = 6

#: How many outer iterations between exploit steps ("at regular intervals").
DEFAULT_EXPLOIT_INTERVAL = 10

#: Fraction of the population replaced by copies of the best members.
DEFAULT_REPLACE_FRACTION = 0.5

#: Multiplicative jitter applied when mutating a hyperparameter.
DEFAULT_MUTATION_FACTOR = 1.2

#: Probability that a *given* hyperparameter is re-explored (replaced by a
#: fresh sample from its range) instead of being jittered.
DEFAULT_EXPLORE_PROB = 0.5

#: Exponential moving average used to smooth per-member fitness.
DEFAULT_FITNESS_DECAY = 0.9


# ---------------------------------------------------------------------------
# hyperparameter search space (population-based training mutates these)
# ---------------------------------------------------------------------------

#: Specification of every mutable hyperparameter.
#:
#: ``kind``:
#:   * ``"log"``  -- positive continuous value, mutated multiplicatively in
#:                   log-space and clamped to ``[low, high]``
#:   * ``"cont"`` -- continuous value, mutated multiplicatively/additively
#:   * ``"int"``  -- positive integer, mutated by +- 1 (bound by ``[low, high]``)
#:   * ``"choice"`` -- sampled uniformly from ``choices`` when explored
HYPERPARAMETER_SPACE: Dict[str, Dict[str, Any]] = {
    "learning_rate": {
        "kind": "log",
        "low": 1e-6,
        "high": 1e-3,
        "factor": 1.5,
        "min_factor": 0.5,
        "max_factor": 2.0,
    },
    "clip_epsilon": {
        "kind": "cont",
        "low": 0.05,
        "high": 0.3,
        "factor": 1.25,
        "min_factor": 0.8,
        "max_factor": 1.25,
    },
    "entropy_coefficient": {
        "kind": "choice",
        "choices": (0.0, 0.003, 0.005),
    },
    "gamma": {
        "kind": "cont",
        "low": 0.9,
        "high": 0.999,
        "factor": 1.01,
        "std": 0.005,
    },
    "tau": {  # GAE lambda (paper's tau = 0.95; see utils/config.py)
        "kind": "cont",
        "low": 0.85,
        "high": 0.995,
        "factor": 1.02,
        "std": 0.01,
    },
    "critic_coefficient": {
        "kind": "cont",
        "low": 1.0,
        "high": 8.0,
        "factor": 1.25,
        "min_factor": 0.75,
        "max_factor": 1.5,
    },
    "bounds_loss_coefficient": {
        "kind": "log",
        "low": 1e-6,
        "high": 1e-2,
        "factor": 2.0,
        "min_factor": 0.5,
        "max_factor": 2.0,
    },
    "kl_threshold": {
        "kind": "log",
        "low": 0.005,
        "high": 0.05,
        "factor": 2.0,
        "min_factor": 0.5,
        "max_factor": 2.0,
    },
    "mini_epochs": {
        "kind": "int",
        "low": 1,
        "high": 8,
    },
    "minibatch_size_multiplier": {
        "kind": "int",
        "low": 1,
        "high": 8,
    },
    "grad_norm": {
        "kind": "cont",
        "low": 0.5,
        "high": 4.0,
        "factor": 1.5,
        "min_factor": 0.75,
        "max_factor": 2.0,
    },
}

MUTABLE_HYPERPARAMS: Tuple[str, ...] = tuple(HYPERPARAMETER_SPACE.keys())


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _rng(rng: Optional[Any] = None) -> random.Random:
    if rng is None:
        return random.Random()
    if hasattr(rng, "uniform") and hasattr(rng, "choice"):
        return rng
    return random.Random(int(rng))


def set_seed(seed: int, env: Any = None, deterministic: bool = False) -> int:
    """Seed python / numpy / torch (and optionally the environment)."""
    seed = int(seed)
    random.seed(seed)
    try:  # pragma: no cover - numpy optional
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
    for target in (env, getattr(env, "env", None)):
        fn = getattr(target, "seed", None)
        if callable(fn):
            try:
                fn(seed)
                break
            except Exception:
                continue
    return seed


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if HAS_TORCH and torch.is_tensor(value):
        try:
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return float(value.detach().float().mean().cpu().item())
        except Exception:
            return default
    if isinstance(value, (list, tuple)) and value:
        try:
            return float(sum(_as_float(v, 0.0) for v in value) / len(value))
        except Exception:
            return default
    try:
        return float(value)
    except Exception:
        return default


def _get(batch: Any, keys: Sequence[str], default: Any = None) -> Any:
    """Fetch the first present key from a dict-like or object-like batch."""
    if batch is None:
        return default
    for key in keys:
        if isinstance(batch, dict):
            if key in batch:
                return batch[key]
            continue
        if hasattr(batch, key):
            return getattr(batch, key)
    for container in ("data", "storage", "batch", "buffers"):
        if isinstance(batch, dict):
            break
        inner = getattr(batch, container, None)
        if inner is None:
            continue
        for key in keys:
            if isinstance(inner, dict) and key in inner:
                return inner[key]
            if hasattr(inner, key):
                return getattr(inner, key)
    return default


def _to_tensor(value: Any, device: Any = None):
    """Best-effort conversion of ``value`` to a torch tensor."""
    if not HAS_TORCH:
        return value
    if torch.is_tensor(value):
        out = value
    elif isinstance(value, (list, tuple)):
        out = torch.as_tensor(list(value))
    else:
        out = torch.as_tensor(value)
    if device is not None and isinstance(out, torch.Tensor):
        out = out.to(device)
    return out


def _call_flexible(fn: Any, args: Tuple[Any, ...] = (), **kwargs: Any) -> Any:
    """Call ``fn`` filtering ``kwargs`` through its signature when possible."""
    if fn is None:
        raise AttributeError("callable is None")
    try:
        import inspect

        sig = inspect.signature(fn)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if accepts_kwargs:
            return fn(*args, **kwargs)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(*args, **filtered)
    except (TypeError, ValueError):
        pass
    last_error: Optional[Exception] = None
    attempts = [kwargs]
    for key in list(kwargs.keys()):
        if key in ("obs", "phi", "policy_index", "deterministic"):
            continue
        trimmed = {k: v for k, v in kwargs.items() if k != key}
        attempts.append(trimmed)
    attempts.append({})
    for attempt in attempts:
        try:
            return fn(*args, **attempt)
        except TypeError as exc:  # incompatible signature
            last_error = exc
            continue
        except Exception:
            raise
    if last_error is not None:
        raise last_error
    return fn(*args)


# ---------------------------------------------------------------------------
# hyperparameter mutation
# ---------------------------------------------------------------------------


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def clip_hyperparameters(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp every hyperparameter into its declared range."""
    out: Dict[str, Any] = dict(hyperparams)
    for name, spec in HYPERPARAMETER_SPACE.items():
        if name not in out or out[name] is None:
            continue
        kind = spec["kind"]
        if kind == "choice":
            choices = spec["choices"]
            value = out[name]
            if value not in choices:
                best = min(choices, key=lambda c: abs(_as_float(c) - _as_float(value)))
                out[name] = best
            continue
        if kind == "int":
            out[name] = int(_clip(round(_as_float(out[name])), spec["low"], spec["high"]))
        else:
            out[name] = _clip(_as_float(out[name]), spec["low"], spec["high"])
    return out


def sample_hyperparameter(name: str, rng: Optional[Any] = None) -> float:
    """Draw a fresh value for ``name`` from its search-space range."""
    spec = HYPERPARAMETER_SPACE[name]
    r = _rng(rng)
    kind = spec["kind"]
    if kind == "choice":
        return r.choice(list(spec["choices"]))
    if kind == "int":
        return float(r.randint(int(spec["low"]), int(spec["high"])))
    low, high = float(spec["low"]), float(spec["high"])
    if spec.get("log", kind == "log"):
        return float(math.exp(r.uniform(math.log(low), math.log(high))))
    return float(r.uniform(low, high))


def mutate_hyperparameter(
    name: str,
    value: Any,
    rng: Optional[Any] = None,
    factor: float = DEFAULT_MUTATION_FACTOR,
    explore_prob: float = DEFAULT_EXPLORE_PROB,
) -> Any:
    """Randomly mutate a single hyperparameter (Section 5.2)."""
    spec = HYPERPARAMETER_SPACE[name]
    r = _rng(rng)
    kind = spec["kind"]

    if kind == "choice":
        choices = list(spec["choices"])
        if r.random() < explore_prob:
            return r.choice(choices)
        others = [c for c in choices if c != value] or choices
        return r.choice(others)

    if kind == "int":
        low, high = int(spec["low"]), int(spec["high"])
        step = r.choice([-1, 1])
        return int(_clip(int(_as_float(value)) + step, low, high))

    if r.random() < explore_prob:
        return sample_hyperparameter(name, r)

    low, high = float(spec["low"]), float(spec["high"])
    if "std" in spec:
        mutated = _as_float(value) + r.gauss(0.0, float(spec["std"]))
    else:
        mult = r.uniform(
            float(spec.get("min_factor", 1.0 / max(factor, 1e-6))),
            float(spec.get("max_factor", max(factor, 1e-6))),
        )
        mutated = _as_float(value) * mult
    return _clip(mutated, low, high)


def mutate_hyperparameters(
    hyperparams: Dict[str, Any],
    rng: Optional[Any] = None,
    num_params: Optional[int] = 1,
    factor: float = DEFAULT_MUTATION_FACTOR,
    explore_prob: float = DEFAULT_EXPLORE_PROB,
    names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Mutate ``num_params`` of the hyperparameters in ``hyperparams``."""
    r = _rng(rng)
    out = dict(hyperparams)
    candidates = [n for n in (names or MUTABLE_HYPERPARAMS) if n in out]
    if not candidates:
        return out
    if num_params is None or num_params >= len(candidates):
        chosen = candidates
    else:
        chosen = r.sample(candidates, int(max(1, num_params)))
    for name in chosen:
        out[name] = mutate_hyperparameter(
            name, out[name], rng=r, factor=factor, explore_prob=explore_prob
        )
    return clip_hyperparameters(out)


def default_hyperparameters(config: Any = None, **overrides: Any) -> Dict[str, Any]:
    """Base hyperparameters copied from the (SAPG) config, plus overrides."""
    hps: Dict[str, Any] = {
        "learning_rate": float(getattr(config, "learning_rate", 1e-4) or 1e-4),
        "clip_epsilon": float(getattr(config, "clip_epsilon", 0.1) or 0.1),
        "entropy_coefficient": float(getattr(config, "entropy_coefficient", 0.0) or 0.0),
        "gamma": float(getattr(config, "gamma", 0.99) or 0.99),
        "tau": float(getattr(config, "tau", 0.95) or 0.95),
        "critic_coefficient": float(getattr(config, "critic_coefficient", 4.0) or 4.0),
        "bounds_loss_coefficient": float(
            getattr(config, "bounds_loss_coefficient", 1e-4) or 1e-4
        ),
        "kl_threshold": float(getattr(config, "kl_threshold", 0.016) or 0.016),
        "mini_epochs": int(getattr(config, "mini_epochs", 2) or 2),
        "minibatch_size_multiplier": int(
            getattr(config, "minibatch_size_multiplier", 4) or 4
        ),
        "grad_norm": float(getattr(config, "grad_norm", 1.0) or 1.0),
    }
    hps.update({k: v for k, v in overrides.items() if k in HYPERPARAMETER_SPACE})
    return clip_hyperparameters(hps)


def initial_population_hyperparameters(
    config: Any = None,
    num_policies: int = DEFAULT_NUM_POLICIES,
    rng: Optional[Any] = None,
    spread: float = 2.0,
) -> List[Dict[str, Any]]:
    """Give every group its own hyperparameters (as required by Section 5.2).

    The population is initialised with a log-uniform spread of the learning
    rate (up to ``spread`` x the base value), a spread of the clipping
    coefficient, and a round-robin assignment of the entropy coefficients
    ``{0, 0.003, 0.005}`` -- the same small set SAPG tunes over.
    """
    r = _rng(rng)
    base = default_hyperparameters(config)
    entropy_choices = list(HYPERPARAMETER_SPACE["entropy_coefficient"]["choices"])
    population: List[Dict[str, Any]] = []
    for j in range(int(num_policies)):
        hps = dict(base)
        hps["learning_rate"] = _clip(
            base["learning_rate"] * math.exp(r.uniform(-math.log(spread), math.log(spread))),
            HYPERPARAMETER_SPACE["learning_rate"]["low"],
            HYPERPARAMETER_SPACE["learning_rate"]["high"],
        )
        hps["clip_epsilon"] = _clip(
            base["clip_epsilon"] * r.uniform(0.9, 1.1),
            HYPERPARAMETER_SPACE["clip_epsilon"]["low"],
            HYPERPARAMETER_SPACE["clip_epsilon"]["high"],
        )
        hps["entropy_coefficient"] = entropy_choices[j % len(entropy_choices)]
        if j % 2 == 0:
            hps["mini_epochs"] = int(
                _clip(base["mini_epochs"] + r.choice([-1, 1]), 1, 8)
            )
        if j % 3 == 0:
            hps["kl_threshold"] = _clip(
                base["kl_threshold"] * r.uniform(0.75, 1.5),
                HYPERPARAMETER_SPACE["kl_threshold"]["low"],
                HYPERPARAMETER_SPACE["kl_threshold"]["high"],
            )
        population.append(clip_hyperparameters(hps))
    return population


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class DexPBTConfig:
    """Hyperparameters for the DexPBT baseline (Section 5.2).

    The defaults mirror SAPG's AllegroKuka hyperparameter group (Table 2) plus
    the population-based training specifics that the paper leaves unspecified:
    how many members are replaced, how often, and how hyperparameters are
    mutated.
    """

    task: str = "regrasping"
    method: str = "dexpbt"
    env_name: Optional[str] = None

    # --- parallelisation (Section 5.2) ---
    num_envs: int = DEFAULT_NUM_ENVS
    num_policies: int = DEFAULT_NUM_POLICIES
    horizon_length: int = 16
    mini_epochs: int = 2
    minibatch_size_multiplier: int = 4

    # --- PPO (Table 2 defaults) ---
    learning_rate: float = 1e-4
    critic_learning_rate: Optional[float] = None
    optimizer: str = "adam"
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    clip_epsilon: float = 0.1
    entropy_coefficient: float = 0.0
    critic_coefficient: float = 4.0
    bounds_loss_coefficient: float = 1e-4
    kl_threshold: float = 0.016
    grad_norm: float = 1.0
    gamma: float = 0.99
    tau: float = 0.95
    normalize_advantage: bool = True

    # --- population-based training (Section 5.2) ---
    exploit_interval: int = DEFAULT_EXPLOIT_INTERVAL
    replace_fraction: float = DEFAULT_REPLACE_FRACTION
    mutation_factor: float = DEFAULT_MUTATION_FACTOR
    explore_prob: float = DEFAULT_EXPLORE_PROB
    fitness_decay: float = DEFAULT_FITNESS_DECAY
    num_mutated_params: int = 1

    # --- networks (Section 5.2 / Appendix B) ---
    obs_dim: int = 60
    action_dim: int = 23
    action_scale: float = 1.0
    actor_mlp_units: Tuple[int, ...] = (768, 512, 256)
    critic_mlp_units: Tuple[int, ...] = (768, 512, 256)
    actor_activation: str = "elu"
    use_lstm: bool = True
    lstm_hidden_size: int = 768
    lstm_num_layers: int = 1

    # --- bookkeeping ---
    seed: int = 0
    device: str = "cuda:0"
    log_dir: str = "runs"
    use_curriculum: bool = True

    def __post_init__(self) -> None:
        self.task = str(self.task)
        self.method = "dexpbt"
        self.num_envs = int(self.num_envs)
        self.num_policies = max(1, int(self.num_policies))
        if self.critic_learning_rate is None:
            self.critic_learning_rate = float(self.learning_rate)
        self.actor_mlp_units = tuple(int(u) for u in self.actor_mlp_units)
        self.critic_mlp_units = tuple(int(u) for u in self.critic_mlp_units)
        self.adam_betas = tuple(float(b) for b in self.adam_betas)
        self.replace_fraction = float(max(0.0, min(0.9, self.replace_fraction)))
        self.exploit_interval = max(1, int(self.exploit_interval))

    # -- (de)serialisation -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        data = copy.deepcopy(asdict(self))
        for key in ("actor_mlp_units", "critic_mlp_units", "adam_betas"):
            data[key] = list(data[key])
        return data

    as_dict = to_dict

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None, **overrides: Any) -> "DexPBTConfig":
        data = dict(data or {})
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in data.items() if k in fields}
        payload.update({k: v for k, v in overrides.items() if k in fields})
        return cls(**payload)

    @classmethod
    def from_any(cls, config: Any = None, **overrides: Any) -> "DexPBTConfig":
        """Build a config from a :class:`SAPGConfig`, dict, or ``None``."""
        data: Dict[str, Any] = {}
        if config is not None:
            if isinstance(config, dict):
                data = dict(config)
            elif isinstance(config, cls):
                data = config.to_dict()
            else:
                fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
                for name in fields:
                    if hasattr(config, name):
                        data[name] = getattr(config, name)
                num_policies = getattr(config, "num_policies", None)
                if num_policies is not None:
                    data["num_policies"] = int(num_policies)
        data.pop("method", None)
        payload = {k: v for k, v in data.items() if v is not None}
        payload.update({k: v for k, v in overrides.items() if v is not None})
        return cls.from_dict(payload)


# ---------------------------------------------------------------------------
# results / population bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class DexPBTResult:
    """One-seed DexPBT result (Section 5.2 reports 5 seeds per experiment)."""

    history: List[Dict[str, float]] = field(default_factory=list)
    samples: List[float] = field(default_factory=list)
    num_envs: int = DEFAULT_NUM_ENVS
    num_policies: int = DEFAULT_NUM_POLICIES
    seed: int = 0
    trainer: Any = None

    def final(self, key: str = "episode_return", default: float = float("nan")) -> float:
        if not self.history:
            return default
        return _as_float(self.history[-1].get(key, default), default)

    def curve(self, key: str = "episode_return") -> List[float]:
        return [_as_float(h.get(key, float("nan")), float("nan")) for h in self.history]

    def sample_curve(self, key: str = "episode_return") -> List[float]:
        if self.samples:
            return list(self.samples)
        return [_as_float(h.get("samples", 0.0)) for h in self.history]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "final": self.final(),
            "history": list(self.history),
            "samples": list(self.samples),
            "num_envs": self.num_envs,
            "num_policies": self.num_policies,
            "seed": self.seed,
        }


@dataclass
class PopulationMember:
    """A single PPO policy in the DexPBT population together with its group."""

    index: int
    policy: Any = None
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    optimizers: List[Any] = field(default_factory=list)
    start: int = 0
    end: int = 0
    fitness: float = float("-inf")
    last_fitness: float = float("-inf")
    best_fitness: float = float("-inf")
    episodes: int = 0
    samples: int = 0
    generation: int = 0
    replaced_from: Optional[int] = None
    history: List[float] = field(default_factory=list)

    # -- group slice -------------------------------------------------------
    @property
    def group_slice(self) -> slice:
        return slice(self.start, self.end)

    @property
    def group_size(self) -> int:
        return max(0, int(self.end) - int(self.start))

    def env_ids(self):
        if not HAS_TORCH:
            return list(range(self.start, self.end))
        return torch.arange(self.start, self.end, dtype=torch.long)

    # -- fitness -----------------------------------------------------------
    def record_fitness(self, value: float, decay: float = DEFAULT_FITNESS_DECAY) -> float:
        value = _as_float(value, float("-inf"))
        self.last_fitness = value
        if not math.isfinite(self.fitness):
            self.fitness = value
        else:
            decay = float(max(0.0, min(1.0, decay)))
            self.fitness = (1.0 - decay) * value + decay * self.fitness
        self.best_fitness = max(self.best_fitness, value)
        self.episodes += 1
        self.history.append(value)
        return self.fitness

    # -- hyperparameters ---------------------------------------------------
    @property
    def learning_rate(self) -> float:
        return _as_float(self.hyperparams.get("learning_rate"), 1e-4)

    def hyperparameter(self, name: str, default: Any = None) -> Any:
        return self.hyperparams.get(name, default)

    def clone_hyperparameters(self) -> Dict[str, Any]:
        return dict(self.hyperparams)

    # -- checkpointing -----------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "index": self.index,
            "hyperparams": copy.deepcopy(self.hyperparams),
            "start": self.start,
            "end": self.end,
            "fitness": self.fitness,
            "best_fitness": self.best_fitness,
            "episodes": self.episodes,
            "samples": self.samples,
            "generation": self.generation,
            "replaced_from": self.replaced_from,
        }
        if self.policy is not None and hasattr(self.policy, "state_dict"):
            try:
                state["policy"] = copy.deepcopy(self.policy.state_dict())
            except Exception:
                pass
        return state

    def load_state_dict(self, state: Dict[str, Any], load_policy: bool = True) -> None:
        self.hyperparams = copy.deepcopy(state.get("hyperparams", self.hyperparams))
        self.fitness = _as_float(state.get("fitness", self.fitness), self.fitness)
        self.best_fitness = _as_float(state.get("best_fitness", self.best_fitness), self.best_fitness)
        self.episodes = int(state.get("episodes", self.episodes))
        self.samples = int(state.get("samples", self.samples))
        self.generation = int(state.get("generation", self.generation))
        self.replaced_from = state.get("replaced_from", self.replaced_from)
        if load_policy and self.policy is not None and "policy" in state:
            try:
                self.policy.load_state_dict(state["policy"])
            except Exception:
                pass

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"PopulationMember(index={self.index}, group=[{self.start},{self.end}), "
            f"fitness={self.fitness:.4f}, lr={self.learning_rate:g}, "
            f"generation={self.generation})"
        )


# ---------------------------------------------------------------------------
# builder helpers (lazy imports keep this module importable without torch)
# ---------------------------------------------------------------------------


def make_dexpbt_config(
    config: Any = None,
    task: str = "regrasping",
    num_envs: Optional[int] = None,
    num_policies: Optional[int] = None,
    seed: Optional[int] = None,
    **overrides: Any,
) -> Any:
    """Build a :class:`DexPBTConfig`, inheriting SAPG task defaults when given.

    When ``config`` is ``None`` the SAPG config for ``task`` is built first so
    that Table 2/3/4 hyperparameters (MLP widths, recurrent flags, clip
    coefficient, horizon length, ...) carry over.
    """
    base = config
    if base is None:
        try:
            from ..utils.config import build_config as _build_config

            base = _build_config(task)
        except Exception:
            base = None

    cfg = DexPBTConfig.from_any(base)
    cfg.task = str(getattr(base, "task", task) or task) if base is not None else task
    cfg.obs_dim = int(getattr(base, "obs_dim", cfg.obs_dim) or cfg.obs_dim)
    cfg.action_dim = int(getattr(base, "action_dim", cfg.action_dim) or cfg.action_dim)
    if num_envs is not None:
        cfg.num_envs = int(num_envs)
    if num_policies is not None:
        cfg.num_policies = int(num_policies)
    if seed is not None:
        cfg.seed = int(seed)
    for key, value in overrides.items():
        if hasattr(cfg, key) and value is not None:
            setattr(cfg, key, value)
    cfg.__post_init__()
    return cfg


def make_dexpbt_env(config: Any, num_envs: Optional[int] = None, **overrides: Any) -> Any:
    """Create the vectorised environment for a DexPBT run."""
    try:
        from ..envs import make_env as _make_env
    except Exception as exc:  # pragma: no cover
        raise ImportError(f"could not import sapg.envs.make_env: {exc}") from exc

    kwargs = dict(overrides)
    if num_envs is not None:
        kwargs["num_envs"] = int(num_envs)
    else:
        kwargs.setdefault("num_envs", int(getattr(config, "num_envs", DEFAULT_NUM_ENVS)))
    return _make_env(task=getattr(config, "task", "regrasping"), config=config, **kwargs)


def make_dexpbt_policy(config: Any, device: Any = None, task_index: int = 0, **kwargs: Any) -> Any:
    """Build one *independent* PPO policy for a population member.

    DexPBT members do not share a backbone and do not use ``phi`` latents, so
    each member gets ``num_policies=1`` and ``phi_dim=0`` while still using
    SAPG's actor/critic implementation (recurrent for AllegroKuka tasks, MLP for
    ShadowHand / AllegroHand -- Section 5.2).
    """
    try:
        from ..models.actor import ActorCritic
    except Exception as exc:  # pragma: no cover
        raise ImportError(f"could not import sapg.models.actor.ActorCritic: {exc}") from exc

    if device is None:
        device = getattr(config, "device", None)
    try:
        policy = ActorCritic(
            obs_dim=int(getattr(config, "obs_dim", 0) or kwargs.get("obs_dim", 0)),
            action_dim=int(getattr(config, "action_dim", 0) or kwargs.get("action_dim", 0)),
            phi_dim=0,
            num_policies=1,
            mlp_units=tuple(getattr(config, "actor_mlp_units", (768, 512, 256))),
            activation=getattr(config, "actor_activation", "elu"),
            use_lstm=bool(getattr(config, "use_lstm", False)),
            lstm_hidden_size=int(getattr(config, "lstm_hidden_size", 768)),
            lstm_num_layers=int(getattr(config, "lstm_num_layers", 1)),
            action_scale=float(getattr(config, "action_scale", 1.0) or 1.0),
            config=config,
        )
    except TypeError:
        policy = ActorCritic(config=config)
    if device is not None:
        try:
            policy = policy.to(device)
        except Exception:
            pass
    return policy


def make_population(
    config: Any,
    policies: Optional[Sequence[Any]] = None,
    device: Any = None,
    seed: Optional[int] = None,
    hyperparameters: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[PopulationMember]:
    """Create ``M`` population members with ``N/M`` environments each."""
    num_policies = int(getattr(config, "num_policies", DEFAULT_NUM_POLICIES))
    num_envs = int(getattr(config, "num_envs", DEFAULT_NUM_ENVS))
    if num_envs % num_policies != 0:
        raise ValueError(
            f"num_envs ({num_envs}) must be divisible by num_policies ({num_policies})"
        )
    block = num_envs // num_policies
    rng = random.Random(int(seed if seed is not None else getattr(config, "seed", 0)))

    if hyperparameters is not None:
        hps_list = [clip_hyperparameters(dict(h)) for h in hyperparameters]
    else:
        hps_list = initial_population_hyperparameters(config, num_policies, rng)

    members: List[PopulationMember] = []
    for j in range(num_policies):
        policy = policies[j] if policies is not None and j < len(policies) else None
        if policy is None:
            policy = make_dexpbt_policy(config, device=device)
        member = PopulationMember(
            index=j,
            policy=policy,
            hyperparams=hps_list[j] if j < len(hps_list) else default_hyperparameters(config),
            start=j * block,
            end=(j + 1) * block,
        )
        member.optimizers = build_member_optimizers(member, config)
        members.append(member)
    return members


def build_member_optimizers(member: PopulationMember, config: Any) -> List[Any]:
    """Create Adam optimizers for a member (actor/critic split when possible)."""
    if not HAS_TORCH or member.policy is None:
        return []
    lr = member.learning_rate
    critic_lr = _as_float(
        member.hyperparams.get("critic_learning_rate"),
        getattr(config, "critic_learning_rate", None) or lr,
    )
    betas = tuple(getattr(config, "adam_betas", (0.9, 0.999)))
    eps = float(getattr(config, "adam_eps", 1e-8))
    optimizers: List[Any] = []
    groups: List[Tuple[Any, float]] = []
    actor_fn = getattr(member.policy, "actor_parameters", None)
    critic_fn = getattr(member.policy, "critic_parameters", None)
    try:
        if callable(actor_fn) and callable(critic_fn):
            groups.append((list(actor_fn()), lr))
            groups.append((list(critic_fn()), critic_lr))
        else:
            groups.append((list(member.policy.parameters()), lr))
    except Exception:
        try:
            groups = [(list(member.policy.parameters()), lr)]
        except Exception:
            return []
    for params, group_lr in groups:
        if not params:
            continue
        try:
            optimizers.append(torch.optim.Adam(params, lr=group_lr, betas=betas, eps=eps))
        except Exception:
            optimizers.append(torch.optim.Adam(params, lr=group_lr))
    if not optimizers:
        try:
            optimizers.append(torch.optim.Adam(member.policy.parameters(), lr=lr))
        except Exception:
            optimizers = []
    return optimizers


# ---------------------------------------------------------------------------
# minibatch / evaluation helpers
# ---------------------------------------------------------------------------


def _iterate_minibatches(buffer: Any, batch_size: int, seed: Optional[int] = None):
    """Yield minibatches from a rollout buffer, tolerating several APIs."""
    if buffer is None:
        return
    for name in ("generator", "minibatches", "iterate", "batches"):
        fn = getattr(buffer, name, None)
        if not callable(fn):
            continue
        attempts = [
            {"batch_size": batch_size},
            {"minibatch_size": batch_size},
            {"batch_size": batch_size, "seed": seed},
            {"num_minibatches": None, "batch_size": batch_size},
            {"mini_batch_size": batch_size},
            {},
        ]
        for kwargs in attempts:
            try:
                produced = fn(**kwargs)
            except TypeError:
                continue
            except Exception:
                produced = None
            if produced is None:
                continue
            try:
                yielded = list(produced)
            except TypeError:
                continue
            if yielded:
                for batch in yielded:
                    yield batch
                return

    # Fallback: manual slicing over flattened tensors
    flat = None
    for name in ("flatten", "flat", "data"):
        candidate = getattr(buffer, name, None)
        if callable(candidate):
            try:
                flat = candidate()
                break
            except Exception:
                continue
        if candidate is not None:
            flat = candidate
            break
    if flat is None:
        return
    size = None
    for key in ("obs", "states", "actions", "rewards"):
        value = _get(flat, (key,))
        if value is not None and hasattr(value, "shape"):
            size = int(value.shape[0])
            break
    if not size:
        return
    index = list(range(size))
    rng = random.Random(seed)
    rng.shuffle(index)
    for start in range(0, size, max(1, int(batch_size))):
        chunk = index[start : start + max(1, int(batch_size))]
        if HAS_TORCH:
            idx = torch.as_tensor(chunk, dtype=torch.long, device=getattr(flat, "device", None))
        else:  # pragma: no cover
            idx = chunk
        if isinstance(flat, dict):
            yield {k: (v[idx] if hasattr(v, "__getitem__") else v) for k, v in flat.items()}
        elif hasattr(flat, "__getitem__"):
            try:
                yield flat[idx]
            except Exception:
                yield {k: getattr(flat, k) for k in dir(flat) if not k.startswith("_")}


def _policy_act(member: PopulationMember, obs: Any, hidden_state: Any = None,
                masks: Any = None, deterministic: bool = False) -> Dict[str, Any]:
    """Run one inference step for a population member's policy."""
    policy = member.policy
    fn = getattr(policy, "act", None)
    if not callable(fn):
        raise AttributeError("policy does not implement act()")
    out = _call_flexible(
        fn,
        obs,
        phi=None,
        hidden_state=hidden_state,
        masks=masks,
        deterministic=deterministic,
        policy_index=member.index,
    )
    if not isinstance(out, dict):
        if isinstance(out, tuple):
            keys = ("actions", "logprobs", "values", "hidden_state")
            out = {k: v for k, v in zip(keys, out)}
        else:
            out = {"actions": out}
    normalised: Dict[str, Any] = dict(out)
    for canonical, aliases in (
        ("actions", ("action", "a")),
        ("logprobs", ("logprob", "log_prob", "log_probs")),
        ("values", ("value", "v")),
        ("entropy", ("entropies",)),
        ("hidden_state", ("hidden", "rnn_state", "hxs")),
    ):
        if canonical in normalised:
            continue
        for alias in aliases:
            if alias in out:
                normalised[canonical] = out[alias]
                break
    return normalised


def _evaluate_member(
    member: PopulationMember,
    obs: Any,
    actions: Any,
    hidden_state: Any = None,
    masks: Any = None,
) -> Dict[str, Any]:
    """Re-evaluate the current policy on stored actions (PPO surrogate terms)."""
    policy = member.policy
    for name in ("evaluate_actions", "evaluate", "evaluate_batch", "evaluate_act"):
        fn = getattr(policy, name, None)
        if not callable(fn):
            continue
        out = _call_flexible(
            fn,
            obs,
            actions,
            phi=None,
            hidden_state=hidden_state,
            masks=masks,
            policy_index=member.index,
        )
        if isinstance(out, dict):
            normalised = dict(out)
        elif isinstance(out, tuple):
            keys = ("logprobs", "values", "entropy")
            normalised = {k: v for k, v in zip(keys, out)}
        else:
            normalised = {"values": out}
        for canonical, aliases in (
            ("logprobs", ("logprob", "log_prob", "log_probs")),
            ("values", ("value", "v")),
            ("entropy", ("entropies",)),
        ):
            if canonical in normalised:
                continue
            for alias in aliases:
                if alias in out:
                    normalised[canonical] = out[alias]
                    break
        return normalised
    raise AttributeError("policy does not implement evaluate_actions()")


# ---------------------------------------------------------------------------
# the DexPBT trainer
# ---------------------------------------------------------------------------


class DexPBTTrainer:
    """Population-based training with PPO for the SAPG baselines (Section 5.2).

    ``N`` environments are split into ``M`` contiguous groups; each population
    member owns one group and one *independent* PPO policy with its own
    hyperparameters.  Every ``exploit_interval`` outer iterations the worst
    performers are overwritten with the weights of the best performers (matched
    best-to-worst) and the copied members' hyperparameters are mutated randomly.
    """

    def __init__(
        self,
        config: Any = None,
        env: Any = None,
        policies: Optional[Sequence[Any]] = None,
        population: Optional[Sequence[PopulationMember]] = None,
        logger: Any = None,
        device: Any = None,
        seed: Optional[int] = None,
        block_manager: Any = None,
        collector: Any = None,
        **overrides: Any,
    ) -> None:
        self.config = make_dexpbt_config(config, **overrides)
        self.config.num_policies = int(
            getattr(self.config, "num_policies", DEFAULT_NUM_POLICIES)
        )
        self.device = device if device is not None else getattr(self.config, "device", None)
        self.seed = int(seed if seed is not None else getattr(self.config, "seed", 0))
        set_seed(self.seed, env=env)

        self._env = env
        self._block_manager = block_manager
        self._collector = collector
        self.logger = logger

        self.population: List[PopulationMember] = list(population or [])
        if not self.population:
            self.population = make_population(
                self.config, policies=policies, device=self.device, seed=self.seed
            )
        self.num_policies = len(self.population)

        num_envs = int(getattr(self.config, "num_envs", DEFAULT_NUM_ENVS))
        if num_envs % self.num_policies != 0:
            raise ValueError(
                f"num_envs ({num_envs}) must be divisible by the population size "
                f"({self.num_policies})"
            )
        block = num_envs // self.num_policies
        for j, member in enumerate(self.population):
            member.index = j
            member.start = j * block
            member.end = (j + 1) * block
            if not member.optimizers:
                member.optimizers = build_member_optimizers(member, self.config)

        self.num_envs = num_envs
        self.horizon_length = int(getattr(self.config, "horizon_length", 16))
        self.obs: Any = None
        self.hidden_states: List[Any] = [None] * self.num_policies
        self.samples = 0
        self.iteration = 0
        self.exploits = 0
        self.generation = 0
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # environment / policy plumbing
    # ------------------------------------------------------------------
    @property
    def env(self) -> Any:
        if self._env is None:
            self._env = make_dexpbt_env(self.config)
        return self._env

    @property
    def batch_size(self) -> int:
        return int(self.num_envs * self.horizon_length)

    def _block_manager_obj(self) -> Any:
        if self._block_manager is None:
            try:
                from ..algorithms.rollout import BlockManager

                self._block_manager = BlockManager(
                    num_envs=self.num_envs,
                    num_policies=self.num_policies,
                    leader_index=1,
                )
            except Exception:
                self._block_manager = None
        return self._block_manager

    def reset(self, obs: Any = None) -> Any:
        """Reset the environment and the per-member recurrent states."""
        if obs is None:
            obs = self.env.reset()
        self.obs = obs
        self.hidden_states = [None] * self.num_policies
        return obs

    # ------------------------------------------------------------------
    # rollout collection
    # ------------------------------------------------------------------
    def collect(self, deterministic: bool = False, obs: Any = None):
        """Collect ``horizon_length`` steps from every group.

        Returns ``(buffers, metrics)`` where ``buffers[j]`` is ``D_j`` for
        population member ``j``.
        """
        if obs is not None:
            self.obs = obs
        if self.obs is None:
            self.reset()

        policies = [m.policy for m in self.population]
        collector_cfg = self._collector_config()
        metrics: Dict[str, float] = {}

        # 1) Preferred path: SAPG's RolloutCollector with one block per member.
        try:
            collector = self._collector
            if collector is None:
                from ..algorithms.rollout import RolloutCollector

                collector = RolloutCollector(
                    self.env, collector_cfg, block_manager=self._block_manager_obj()
                )
                self._collector = collector
            buffers = collector.create_buffers(phi_dim=0)
            buffers, obs, hidden_states, metrics = collector.collect(
                policies,
                phis=None,
                hidden_states=self.hidden_states,
                buffers=buffers,
                deterministic=deterministic,
                obs=self.obs,
            )
            self.obs = obs
            self.hidden_states = list(hidden_states)
            self._log_metrics(metrics, "rollout")
            self.samples += int(self.batch_size)
            return list(buffers), dict(metrics or {})
        except Exception as exc:  # pragma: no cover - fallback paths
            self._last_collect_error = exc

        # 2) Fallback: functional collect_data API.
        try:
            from ..algorithms.rollout import collect_data

            buffers, obs, hidden_states, metrics = collect_data(
                policies,
                self.env,
                collector_cfg,
                obs=self.obs,
                phis=None,
                hidden_states=self.hidden_states,
                block_manager=self._block_manager_obj(),
                deterministic=deterministic,
            )
            self.obs = obs
            self.hidden_states = list(hidden_states)
            self._log_metrics(metrics, "rollout")
            self.samples += int(self.batch_size)
            return list(buffers), dict(metrics or {})
        except Exception as exc:
            self._last_collect_error = exc

        # 3) Last resort: collect per member on the whole batch of environments.
        buffers, metrics = self._collect_per_member(deterministic=deterministic)
        self.samples += int(self.batch_size)
        return buffers, metrics

    def _collector_config(self) -> Any:
        """A config copy suitable for the rollout collector (M blocks, no phi)."""
        cfg = self.config
        try:
            from ..utils.config import SAPGConfig

            data = {}
            if hasattr(cfg, "to_dict"):
                data = dict(cfg.to_dict())
            data.update(
                {
                    "num_policies": self.num_policies,
                    "num_envs": self.num_envs,
                    "horizon_length": self.horizon_length,
                    "phi_dim": 0,
                    "aggregation": "none",
                    "off_policy_weight": 0.0,
                    "device": str(self.device) if self.device is not None else data.get("device", "cpu"),
                }
            )
            fields = {f for f in SAPGConfig.__dataclass_fields__}  # type: ignore[attr-defined]
            return SAPGConfig.from_dict({k: v for k, v in data.items() if k in fields})
        except Exception:
            return cfg

    def _collect_per_member(self, deterministic: bool = False):
        """Fallback rollout: every member collects on the full env, then slices."""
        try:
            from ..algorithms.rollout import collect_on_policy
        except Exception as exc:
            raise RuntimeError(
                "DexPBT could not collect rollouts: neither RolloutCollector, "
                f"collect_data nor collect_on_policy are available ({exc})"
            ) from exc

        buffers: List[Any] = []
        metrics: Dict[str, float] = {}
        for member in self.population:
            buffer, obs, hidden_state, m = collect_on_policy(
                member.policy,
                self.env,
                self._collector_config(),
                obs=self.obs,
                hidden_state=self.hidden_states[member.index],
                deterministic=deterministic,
            )
            buffers.append(buffer)
            self.obs = obs
            self.hidden_states[member.index] = hidden_state
            for key, value in (m or {}).items():
                metrics[f"policy_{member.index}/{key}"] = _as_float(value)
        return buffers, metrics

    # ------------------------------------------------------------------
    # advantage / target preparation
    # ------------------------------------------------------------------
    def _prepare_buffers(self, buffers: Sequence[Any]) -> List[Any]:
        """Compute GAE (per-member gamma/tau) and finalise each buffer."""
        prepared: List[Any] = []
        for member, buffer in zip(self.population, buffers):
            gamma = _as_float(member.hyperparams.get("gamma"), getattr(self.config, "gamma", 0.99))
            tau = _as_float(member.hyperparams.get("tau"), getattr(self.config, "tau", 0.95))
            done = False
            if buffer is None:
                prepared.append(buffer)
                continue
            for name in ("compute_gae", "compute_advantages", "gae"):
                fn = getattr(buffer, name, None)
                if not callable(fn):
                    continue
                try:
                    _call_flexible(fn, gamma=gamma, tau=tau, lam=tau, lambda_=tau)
                    done = True
                    break
                except Exception:
                    continue
            if hasattr(buffer, "finalise"):
                try:
                    buffer.finalise()
                except Exception:
                    try:
                        buffer.finalize()
                    except Exception:
                        pass
            if not done:
                self._manual_gae(buffer, gamma, tau, member.index)
            prepared.append(buffer)
        return prepared

    def _manual_gae(self, buffer: Any, gamma: float, tau: float, index: int) -> None:
        """Fallback advantage computation when the buffer cannot do it itself."""
        if not HAS_TORCH:
            return
        rewards = _get(buffer, ("rewards", "reward"))
        dones = _get(buffer, ("dones", "terminals", "done"))
        values = _get(buffer, ("values", "value"))
        if rewards is None or values is None:
            return
        try:
            rewards = rewards.to(self.device) if self.device else rewards
            values = values.to(self.device) if self.device else values
            if dones is None:
                dones = torch.zeros_like(rewards)
            else:
                dones = dones.to(rewards.device).float()
            if dones.dim() == 3 and dones.shape[-1] == 1:
                dones = dones.squeeze(-1)
            steps = rewards.shape[0]
            last_values = _get(buffer, ("last_values", "bootstrap_values"))
            if last_values is None:
                last_values = torch.zeros_like(rewards[0])
            adv = torch.zeros_like(rewards)
            last_gae = torch.zeros_like(rewards[0])
            next_value = last_values
            for t in reversed(range(steps)):
                mask = 1.0 - dones[t]
                delta = rewards[t] + gamma * next_value * mask - values[t]
                last_gae = delta + gamma * tau * mask * last_gae
                adv[t] = last_gae
                next_value = values[t]
            targets = adv + values
            success = False
            for key in ("advantages", "adv"):
                try:
                    setattr(buffer, key, adv)
                    success = True
                    break
                except Exception:
                    continue
            for key in ("value_targets", "returns", "targets"):
                try:
                    setattr(buffer, key, targets)
                    break
                except Exception:
                    continue
            if not success:
                data = _get(buffer, ("data", "storage"))
                if isinstance(data, dict):
                    data["advantages"] = adv
                    data["value_targets"] = targets
        except Exception:
            return

    # ------------------------------------------------------------------
    # per-member PPO update
    # ------------------------------------------------------------------
    def _minibatch_size(self, member: PopulationMember) -> int:
        multiplier = int(
            member.hyperparams.get(
                "minibatch_size_multiplier",
                getattr(self.config, "minibatch_size_multiplier", 4),
            )
        )
        return max(1, int(self.num_envs * max(1, multiplier)))

    def _normalise(self, member: PopulationMember, buffer: Any) -> None:
        if not bool(getattr(self.config, "normalize_advantage", True)):
            return
        for name in ("normalize_advantages", "normalise_advantages"):
            fn = getattr(buffer, name, None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    continue

    def update_member(self, member: PopulationMember, buffer: Any) -> Dict[str, float]:
        """Run PPO on one member's own group data using its own hyperparameters."""
        if buffer is None or not HAS_TORCH:
            return {}
        hps = member.hyperparams
        mini_epochs = int(hps.get("mini_epochs", getattr(self.config, "mini_epochs", 2)))
        clip_epsilon = _as_float(hps.get("clip_epsilon"), getattr(self.config, "clip_epsilon", 0.1))
        entropy_coef = _as_float(hps.get("entropy_coefficient"), 0.0)
        critic_coef = _as_float(hps.get("critic_coefficient"), 4.0)
        bounds_coef = _as_float(hps.get("bounds_loss_coefficient"), 1e-4)
        grad_norm = _as_float(hps.get("grad_norm"), 1.0)
        kl_threshold = _as_float(hps.get("kl_threshold"), getattr(self.config, "kl_threshold", 0.016))

        self._normalise(member, buffer)
        minibatch = self._minibatch_size(member)

        stats_total: Dict[str, float] = {}
        counts = 0
        for epoch in range(max(1, mini_epochs)):
            for batch in _iterate_minibatches(buffer, minibatch, seed=self.seed + epoch):
                stats = self._member_step(member, batch, clip_epsilon, entropy_coef,
                                         critic_coef, bounds_coef, grad_norm)
                counts += 1
                for key, value in stats.items():
                    stats_total[key] = stats_total.get(key, 0.0) + _as_float(value)

        if counts:
            for key in list(stats_total.keys()):
                stats_total[key] /= counts

        # KL-adaptive learning rate (standard PPO schedule, threshold from the
        # member's own hyperparameters).
        kl = stats_total.get("kl", 0.0)
        if kl > 2.0 * kl_threshold:
            self._scale_member_lr(member, 0.5)
        elif 0.0 < kl < kl_threshold / 2.0:
            self._scale_member_lr(member, 1.5)

        stats_total["learning_rate"] = member.learning_rate
        return stats_total

    def _member_step(
        self,
        member: PopulationMember,
        batch: Any,
        clip_epsilon: float,
        entropy_coef: float,
        critic_coef: float,
        bounds_coef: float,
        grad_norm: float,
    ) -> Dict[str, float]:
        obs = _get(batch, ("obs", "observations", "states"))
        actions = _get(batch, ("actions", "action"))
        old_logprobs = _get(batch, ("logprobs", "old_logprobs", "log_probs"))
        value_targets = _get(
            batch, ("value_targets", "returns", "targets", "advantages_targets")
        )
        advantages = _get(batch, ("advantages", "adv", "gae"))
        if advantages is None:
            advantages = _get(batch, ("value_targets", "returns", "targets"))
        if obs is None or actions is None:
            return {}
        hidden_state = _get(batch, ("hidden_states", "rnn_states", "hidden_state"))
        masks = _get(batch, ("masks", "dones", "terminals"))

        out = _evaluate_member(member, obs, actions, hidden_state, masks)
        new_logprobs = out.get("logprobs")
        values = out.get("values")
        if new_logprobs is None or values is None:
            return {}
        entropy = out.get("entropy")

        if old_logprobs is None:
            old_logprobs = new_logprobs.detach()
        if advantages is None:
            return {}

        ratio = torch.exp(new_logprobs - old_logprobs.detach())
        surrogate = torch.min(
            ratio * advantages,
            torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages,
        )
        policy_loss = -surrogate.mean()

        values = values.reshape(-1)
        if value_targets is None:
            value_targets = advantages.reshape(-1) + values.detach()
        value_targets = value_targets.reshape(-1).to(values.device)
        value_loss = torch.nn.functional.mse_loss(values, value_targets)

        total = policy_loss + critic_coef * value_loss

        entropy_value = 0.0
        if entropy_coef != 0.0 and entropy is not None:
            mean_entropy = entropy.reshape(-1).mean()
            total = total - entropy_coef * mean_entropy
            entropy_value = float(mean_entropy.detach().cpu().item())

        bounds_loss_value = 0.0
        mean_action = out.get("mean") or out.get("mu") or out.get("action_mean")
        if bounds_coef and mean_action is not None:
            bounds_loss = torch.mean(torch.square(mean_action))
            total = total + bounds_coef * bounds_loss
            bounds_loss_value = float(bounds_loss.detach().cpu().item())

        for optimizer in member.optimizers:
            try:
                optimizer.zero_grad(set_to_none=True)
            except TypeError:
                optimizer.zero_grad()
        total.backward()
        grad_norm_value = 0.0
        if grad_norm:
            params = [p for p in member.policy.parameters() if p.requires_grad]
            try:
                gn = torch.nn.utils.clip_grad_norm_(params, grad_norm)
                grad_norm_value = float(gn.detach().cpu().item() if HAS_TORCH else gn)
            except Exception:
                grad_norm_value = 0.0
        for optimizer in member.optimizers:
            optimizer.step()

        with torch.no_grad():
            clip_frac = float(
                (torch.abs(ratio - 1.0) > clip_epsilon).float().mean().detach().cpu().item()
            )
            approx_kl = float(
                ((old_logprobs.detach() - new_logprobs.detach()).mean()).cpu().item()
            )
        return {
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
            "total_loss": float(total.detach().cpu().item()),
            "entropy": entropy_value,
            "bounds_loss": bounds_loss_value,
            "kl": approx_kl,
            "clip_frac": clip_frac,
            "grad_norm": grad_norm_value,
        }

    def _scale_member_lr(self, member: PopulationMember, factor: float) -> None:
        for optimizer in member.optimizers:
            groups = getattr(optimizer, "param_groups", None)
            if not groups:
                continue
            for group in groups:
                group["lr"] = float(group.get("lr", 1e-4)) * factor
        member.hyperparams["learning_rate"] = _clip(
            _as_float(member.hyperparams.get("learning_rate"), 1e-4) * factor,
            HYPERPARAMETER_SPACE["learning_rate"]["low"],
            HYPERPARAMETER_SPACE["learning_rate"]["high"],
        )

    # ------------------------------------------------------------------
    # fitness / exploit-explore
    # ------------------------------------------------------------------
    def compute_fitness(self, buffers: Sequence[Any], metrics: Optional[Dict[str, float]] = None):
        """Fitness = mean undiscounted return collected by the member's group."""
        fitness: List[float] = []
        for member, buffer in zip(self.population, buffers):
            value = None
            if metrics:
                for key in (
                    f"policy_{member.index}/episode_return",
                    f"policy_{member.index}/returns",
                    f"reward/policy_{member.index}",
                ):
                    if key in metrics:
                        value = metrics[key]
                        break
            if value is None and buffer is not None:
                rewards = _get(buffer, ("rewards", "reward"))
                if rewards is not None and hasattr(rewards, "sum"):
                    try:
                        value = float(rewards.sum(dim=0).mean().detach().cpu().item())
                    except Exception:
                        value = None
                if value is None:
                    value = _as_float(getattr(buffer, "episode_return", None), None)  # type: ignore[arg-type]
            if value is None:
                value = member.last_fitness if math.isfinite(member.last_fitness) else 0.0
            fitness.append(_as_float(value))
        return fitness

    def rank(self) -> List[int]:
        """Member indices ordered from best to worst fitness (ties -> lower id)."""
        return sorted(
            range(self.num_policies),
            key=lambda i: (-self.population[i].fitness, self.population[i].index),
        )

    def last_place(self) -> List[int]:
        return self.rank()[-1:] or [0]

    def best_member(self) -> PopulationMember:
        return self.population[self.rank()[0]]

    def _num_replacements(self) -> int:
        fraction = float(getattr(self.config, "replace_fraction", DEFAULT_REPLACE_FRACTION))
        num = int(round(self.num_policies * fraction))
        return int(max(1, min(num, max(1, self.num_policies // 2))))

    def exploit(self, force: bool = False) -> Dict[str, float]:
        """Replace worst members with copies of the best and mutate them."""
        ranking = self.rank()
        num_replace = self._num_replacements()
        best = ranking[:num_replace]
        worst = list(reversed(ranking[-num_replace:]))
        replaced = 0
        mutated: List[int] = []
        rng = random.Random(self.seed + 7919 * (self.generation + 1))
        num_mutated_params = int(
            getattr(self.config, "num_mutated_params", 1) or 1
        )
        for src_idx, dst_idx in zip(best, worst):
            if src_idx == dst_idx:
                continue
            src, dst = self.population[src_idx], self.population[dst_idx]
            if src.policy is None or dst.policy is None:
                continue
            try:
                state = copy.deepcopy(src.policy.state_dict())
                dst.policy.load_state_dict(state)
            except Exception:
                continue
            dst.hyperparams = mutate_hyperparameters(
                src.clone_hyperparameters(),
                rng=rng,
                num_params=num_mutated_params,
                factor=float(getattr(self.config, "mutation_factor", DEFAULT_MUTATION_FACTOR)),
                explore_prob=float(getattr(self.config, "explore_prob", DEFAULT_EXPLORE_PROB)),
            )
            dst.optimizers = build_member_optimizers(dst, self.config)
            dst.replaced_from = src_idx
            dst.generation = self.generation + 1
            # the replaced member starts a fresh performance estimate
            dst.fitness = float("-inf")
            dst.best_fitness = float("-inf")
            dst.episodes = 0
            replaced += 1
            mutated.append(dst_idx)
        if replaced:
            self.exploits += 1
            self.generation += 1
        return {
            "exploit/num_replaced": float(replaced),
            "exploit/num_mutated_params": float(num_mutated_params * replaced),
            "exploit/generation": float(self.generation),
            "exploit/forced": float(bool(force)),
        }

    def maybe_exploit(self) -> Dict[str, float]:
        interval = int(getattr(self.config, "exploit_interval", DEFAULT_EXPLOIT_INTERVAL))
        if interval > 0 and self.iteration > 0 and self.iteration % interval == 0:
            return self.exploit()
        return {}

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------
    def _log_metrics(self, metrics: Optional[Dict[str, Any]], prefix: str) -> None:
        if not metrics:
            return
        flat = {f"{prefix}/{k}": _as_float(v) for k, v in metrics.items()}
        logger = self.logger
        if logger is None:
            return
        for name in ("log", "record", "add_scalars", "log_metrics", "update"):
            fn = getattr(logger, name, None)
            if callable(fn):
                try:
                    _call_flexible(fn, **flat) if name in ("log", "log_metrics") else fn(flat)
                    return
                except Exception:
                    try:
                        fn(flat)
                        return
                    except Exception:
                        continue

    def _history_entry(
        self,
        fitness: Sequence[float],
        member_stats: Sequence[Dict[str, float]],
        metrics: Optional[Dict[str, float]] = None,
        exploit_stats: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        entry: Dict[str, float] = {
            "samples": float(self.samples),
            "iteration": float(self.iteration),
            "fitness/mean": float(sum(fitness) / max(1, len(fitness))),
            "fitness/max": float(max(fitness) if fitness else 0.0),
            "fitness/min": float(min(fitness) if fitness else 0.0),
        }
        for j, value in enumerate(fitness):
            entry[f"fitness/policy_{j}"] = float(value)
        entry["episode_return"] = entry["fitness/max"]
        entry["episode_return_mean"] = entry["fitness/mean"]
        for j, hps in enumerate([m.hyperparams for m in self.population]):
            entry[f"hparams/policy_{j}/learning_rate"] = _as_float(hps.get("learning_rate"))
            entry[f"hparams/policy_{j}/clip_epsilon"] = _as_float(hps.get("clip_epsilon"))
            entry[f"hparams/policy_{j}/entropy_coefficient"] = _as_float(
                hps.get("entropy_coefficient")
            )
            entry[f"hparams/policy_{j}/mini_epochs"] = _as_float(hps.get("mini_epochs"))
        if metrics:
            for key in ("episode_return", "returns/episode_return", "episode_length"):
                if key in metrics:
                    entry[f"env/{key}"] = _as_float(metrics[key])
        if member_stats:
            for key in ("policy_loss", "value_loss", "total_loss", "entropy", "kl",
                        "clip_frac", "grad_norm"):
                values = [_as_float(s.get(key)) for s in member_stats if key in s]
                if values:
                    entry[f"loss/{key}"] = float(sum(values) / len(values))
        if exploit_stats:
            entry.update({k: _as_float(v) for k, v in exploit_stats.items()})
        return entry

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def update(self, deterministic: bool = False) -> Dict[str, float]:
        """One DexPBT iteration: collect, PPO-update every member, exploit/explore."""
        buffers, metrics = self.collect(deterministic=deterministic)
        buffers = self._prepare_buffers(buffers)

        member_stats: List[Dict[str, float]] = []
        for member, buffer in zip(self.population, buffers):
            member_stats.append(self.update_member(member, buffer))

        fitness = self.compute_fitness(buffers, metrics)
        decay = float(getattr(self.config, "fitness_decay", DEFAULT_FITNESS_DECAY))
        for member, value in zip(self.population, fitness):
            member.record_fitness(value, decay=decay)
            member.samples += member.group_size * self.horizon_length

        exploit_stats = self.maybe_exploit()
        entry = self._history_entry(fitness, member_stats, metrics, exploit_stats)
        for key, value in exploit_stats.items():
            if key not in entry:
                entry[key] = _as_float(value)
        self.history.append(entry)
        if self.logger is not None:
            self._log_metrics({k: v for k, v in entry.items()}, "dex")
        return entry

    def learn(
        self,
        num_iterations: Optional[int] = None,
        max_samples: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple["DexPBTTrainer", List[Dict[str, float]]]:
        """Run population-based training until an iteration or sample budget."""
        if num_iterations is None:
            if max_samples is not None:
                per_iter = max(1, self.batch_size)
                num_iterations = max(1, int(max_samples // per_iter))
            else:
                num_iterations = 1
        start = time.time()
        self.reset()
        for _ in range(int(num_iterations)):
            entry = self.update()
            self.iteration += 1
            if verbose:
                print(
                    f"[DexPBT] iter {self.iteration} samples={int(entry['samples']):,} "
                    f"fit(mean/max)={entry['fitness/mean']:.3f}/{entry['fitness/max']:.3f} "
                    f"replaced={int(entry.get('exploit/num_replaced', 0))} "
                    f"elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
            if max_samples is not None and self.samples >= int(max_samples):
                break
        return self, self.history

    # aliases mirroring the other baselines
    train = learn
    run = learn

    def train_samples(self, max_samples: int, verbose: bool = False):
        return self.learn(max_samples=max_samples, verbose=verbose)

    def evaluate(self, num_episodes: int = 1, deterministic: bool = True, **kwargs: Any):
        """Evaluate the best member of the population."""
        member = self.best_member()
        try:
            from ..algorithms.rollout import collect_on_policy

            buffer, obs, hidden, metrics = collect_on_policy(
                member.policy, self.env, self._collector_config(), deterministic=deterministic
            )
            self.obs = obs
            return metrics
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config),
            "iteration": self.iteration,
            "samples": self.samples,
            "generation": self.generation,
            "exploits": self.exploits,
            "seed": self.seed,
            "population": [m.state_dict() for m in self.population],
        }

    def save(self, path: str) -> str:
        if HAS_TORCH:
            directory = os.path.dirname(os.path.abspath(path))
            if directory:
                os.makedirs(directory, exist_ok=True)
            torch.save(self.state_dict(), path)
        else:  # pragma: no cover
            directory = os.path.dirname(os.path.abspath(path))
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(str(self.state_dict()))
        return path

    def load(self, path: str, map_location: Any = None) -> "DexPBTTrainer":
        if HAS_TORCH:
            state = torch.load(path, map_location=map_location or self.device, weights_only=False)
        else:  # pragma: no cover
            raise ImportError("torch is required to load a DexPBT checkpoint")
        self.iteration = int(state.get("iteration", 0))
        self.samples = int(state.get("samples", 0))
        self.generation = int(state.get("generation", 0))
        self.exploits = int(state.get("exploits", 0))
        for member, mstate in zip(self.population, state.get("population", [])):
            member.load_state_dict(mstate)
        return self

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"DexPBTTrainer(num_envs={self.num_envs}, num_policies={self.num_policies}, "
            f"block={self.num_envs // max(1, self.num_policies)}, "
            f"horizon={self.horizon_length}, exploit_interval="
            f"{getattr(self.config, 'exploit_interval', None)})"
        )


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------


def train_dexpbt(
    config: Any = None,
    env: Any = None,
    policies: Optional[Sequence[Any]] = None,
    num_iterations: Optional[int] = None,
    max_samples: Optional[int] = None,
    verbose: bool = False,
    logger: Any = None,
    device: Any = None,
    num_envs: Optional[int] = None,
    num_policies: Optional[int] = None,
    seed: Optional[int] = None,
    return_result: bool = False,
    trainer: Any = None,
    population: Optional[Sequence[PopulationMember]] = None,
    **overrides: Any,
) -> Any:
    """Train DexPBT (Section 5.2); returns ``(trainer, history)`` by default."""
    cfg = make_dexpbt_config(
        config,
        num_envs=num_envs,
        num_policies=num_policies,
        seed=seed,
        **overrides,
    )
    if trainer is None:
        trainer = DexPBTTrainer(
            cfg,
            env=env,
            policies=policies,
            population=population,
            logger=logger,
            device=device,
            seed=seed,
        )
    trainer.learn(num_iterations=num_iterations, max_samples=max_samples, verbose=verbose)
    if return_result:
        return DexPBTResult(
            history=list(trainer.history),
            samples=[_as_float(h.get("samples")) for h in trainer.history],
            num_envs=trainer.num_envs,
            num_policies=trainer.num_policies,
            seed=trainer.seed,
            trainer=trainer,
        )
    return trainer, trainer.history


def run_dexpbt_seeds(
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    config: Any = None,
    num_envs: Optional[int] = None,
    num_policies: Optional[int] = None,
    max_samples: Optional[int] = None,
    num_iterations: Optional[int] = None,
    verbose: bool = False,
    device: Any = None,
    trainer_factory: Any = None,
    **overrides: Any,
) -> List[DexPBTResult]:
    """Run DexPBT for each seed (the paper reports 5 seeds per experiment)."""
    results: List[DexPBTResult] = []
    base = config
    for seed in seeds:
        if trainer_factory is not None:
            trainer = trainer_factory(seed=seed)
            trainer.learn(
                num_iterations=num_iterations, max_samples=max_samples, verbose=verbose
            )
        else:
            trainer = train_dexpbt(
                base,
                num_iterations=num_iterations,
                max_samples=max_samples,
                verbose=verbose,
                device=device,
                num_envs=num_envs,
                num_policies=num_policies,
                seed=seed,
                return_result=True,
                **overrides,
            ).trainer
        results.append(
            DexPBTResult(
                history=list(trainer.history),
                samples=[_as_float(h.get("samples")) for h in trainer.history],
                num_envs=trainer.num_envs,
                num_policies=trainer.num_policies,
                seed=seed,
                trainer=trainer,
            )
        )
    return results


def paper_standard_error(curves: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    """Paper band: mean ``y(t)`` and width ``2/sqrt(n) * sum_i (y(t)-y_i(t))^2``."""
    curves = [list(map(float, c)) for c in curves if c]
    if not curves:
        return [], []
    n = len(curves)
    length = min(len(c) for c in curves)
    mean: List[float] = []
    band: List[float] = []
    for t in range(length):
        values = [c[t] for c in curves]
        m = sum(values) / n
        mean.append(m)
        band.append((2.0 / math.sqrt(n)) * sum((m - v) ** 2 for v in values))
    return mean, band


def aggregate_seed_histories(
    results: Sequence[DexPBTResult],
    key: str = "episode_return",
    num_bins: Optional[int] = None,
) -> Dict[str, Any]:
    """Aggregate per-seed curves onto a common sample grid with the paper band."""
    results = list(results)
    if not results:
        return {"samples": [], "mean": [], "band": [], "seed_curves": []}
    sample_curves: List[Tuple[List[float], List[float]]] = [
        (r.sample_curve(key), r.curve(key)) for r in results
    ]
    all_samples = sorted({s for samples, _ in sample_curves for s in samples})
    if not all_samples:
        return {"samples": [], "mean": [], "band": [], "seed_curves": []}
    if num_bins and len(all_samples) > num_bins:
        step = max(1, len(all_samples) // num_bins)
        grid = all_samples[::step]
        if grid[-1] != all_samples[-1]:
            grid.append(all_samples[-1])
    else:
        grid = all_samples

    interpolated: List[List[float]] = []
    for samples, values in sample_curves:
        if not samples or not values:
            continue
        curve: List[float] = []
        j = 0
        for x in grid:
            while j + 1 < len(samples) and samples[j + 1] <= x:
                j += 1
            if j + 1 < len(samples) and samples[j + 1] > samples[j]:
                weight = (x - samples[j]) / (samples[j + 1] - samples[j])
                weight = max(0.0, min(1.0, weight))
                curve.append(values[j] + weight * (values[j + 1] - values[j]))
            else:
                curve.append(values[j])
        interpolated.append(curve)

    mean, band = paper_standard_error(interpolated)
    return {
        "samples": grid,
        "mean": mean,
        "band": band,
        "seed_curves": interpolated,
        "final": mean[-1] if mean else float("nan"),
        "n_seeds": len(interpolated),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: ``python -m sapg.baselines.dexpbt --task reorientation``."""
    import argparse

    parser = argparse.ArgumentParser(description="Train the DexPBT baseline (SAPG §5.2)")
    parser.add_argument("--task", default="regrasping")
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--num-policies", type=int, default=DEFAULT_NUM_POLICIES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=1, help="number of seeds to run")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--max-samples", type=float, default=None)
    parser.add_argument("--exploit-interval", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    overrides: Dict[str, Any] = {}
    if args.exploit_interval is not None:
        overrides["exploit_interval"] = args.exploit_interval

    seeds = [args.seed + i for i in range(max(1, args.seeds))]
    results = run_dexpbt_seeds(
        seeds=seeds,
        num_envs=args.num_envs,
        num_policies=args.num_policies,
        max_samples=int(args.max_samples) if args.max_samples else None,
        num_iterations=args.iterations if args.iterations else 1,
        verbose=args.verbose,
        config=None,
        **overrides,
    )
    for result in results:
        print(
            f"seed={result.seed} final_episode_return={result.final():.4f} "
            f"samples={result.samples[-1] if result.samples else 0:.0f}"
        )
    if len(results) > 1:
        agg = aggregate_seed_histories(results)
        print(
            f"aggregate ({agg['n_seeds']} seeds): mean={agg['final']:.4f} "
            f"band={agg['band'][-1] if agg['band'] else float('nan'):.4f}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

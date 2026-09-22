"""StateMask explanation adapter for RICE (Experiment I / III comparison).

The paper (§3.3, §4.1, §C.1) uses the *original* StateMask (Cheng et al., 2023,
`https://github.com/nuwuxian/RL-state_mask`) as the first baseline explanation
method: "Since our explanation method proposes an alternative design of StateMask,
the first baseline is StateMask. We compare our explanation method with StateMask
to show the equivalence and efficiency of our method." (§4.1)

StateMask's semantics that must be preserved by this adapter:

* The mask network ``\\tilde\\pi_\\theta(a^m | s)`` outputs a *binary* action
  ``a^m in {0, 1}`` and the final action is the Eq. (1) mixture

      a_t (.) a_t^m = a_t        if a_t^m = 0
                      a_random   if a_t^m = 1

* The **importance** of a state is "the probability of mask network outputting
  ' 0 ' " (§3.3), i.e. ``P(a^m = 0 | s_t)`` — identical to RICE's redesigned
  mask net, which is why the two are directly comparable.

Because the upstream repository cannot be assumed to be installed (and its
training procedure — a prime-dual surrogate — is *not* reproduced: RICE replaces
it with vanilla PPO, §3.3 / Algorithm 1), this module provides:

1. a thin *loader* path that wraps a real StateMask model/policy object supplied
   by the user (``model=`` / ``checkpoint=``), and
2. a faithful local re-implementation that reuses
   :class:`rice.algorithms.mask_network.MaskNetwork`, so StateMask-R and the
   fidelity comparison remain runnable with only this repository installed.

The adapter is black-box w.r.t. the target agent: it only ever consumes visited
states and returns importance scores / critical states.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Tolerant imports (the repository is usable from several sys.path layouts and
# must stay importable on CPU-only machines without torch).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import plumbing
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

_TORCH_AVAILABLE = torch is not None


def _import_first(candidates: Sequence[str]) -> Any:
    """Import the first importable dotted module among ``candidates``."""
    last_err: Optional[Exception] = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as err:  # pragma: no cover - layout dependent
            last_err = err
    return None


_ppo_mod = _import_first(
    ("rice.algorithms.ppo", "rice.rice.algorithms.ppo", "algorithms.ppo")
)
_mask_mod = _import_first(
    (
        "rice.algorithms.mask_network",
        "rice.rice.algorithms.mask_network",
        "algorithms.mask_network",
    )
)
_critical_mod = _import_first(
    (
        "rice.algorithms.critical_state",
        "rice.rice.algorithms.critical_state",
        "algorithms.critical_state",
    )
)
_env_reset_mod = _import_first(
    (
        "rice.algorithms.env_reset",
        "rice.rice.algorithms.env_reset",
        "algorithms.env_reset",
    )
)
_seeding_mod = _import_first(("rice.utils.seeding", "rice.rice.utils.seeding", "utils.seeding"))


def _get(module: Any, name: str, default: Any = None) -> Any:
    if module is None:
        return default
    return getattr(module, name, default)


flatten_obs = _get(_ppo_mod, "flatten_obs")
observation_size = _get(_ppo_mod, "observation_size")
make_target_policy_callable = _get(_ppo_mod, "make_target_policy_callable")

MaskNetwork = _get(_mask_mod, "MaskNetwork")
MaskNetworkConfig = _get(_mask_mod, "MaskNetworkConfig")
MaskNetworkTrainer = _get(_mask_mod, "MaskNetworkTrainer")

importance_scores_fn = _get(_critical_mod, "importance_scores")
select_critical_state_fn = _get(_critical_mod, "select_critical_state")
max_episode_steps_fn = _get(_critical_mod, "max_episode_steps")

make_state_manager = _get(_env_reset_mod, "make_state_manager")
RNG = _get(_seeding_mod, "RNG")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NAME = "statemask"
ALIASES: Tuple[str, ...] = (
    "statemask",
    "state_mask",
    "state-mask",
    "statemask_adapter",
    "state_mask_adapter",
    "statemask_baseline",
    "original_statemask",
)

#: Repository of the original StateMask implementation (§C.1).
STATEMASK_REPO = "https://github.com/nuwuxian/RL-state_mask"

#: Candidate module names of the upstream package (tried at load time).
_UPSTREAM_MODULES: Tuple[str, ...] = (
    "state_mask",
    "stateMask",
    "state_mask.model",
    "state_mask.networks",
    "statemask",
    "statemask.model",
)


def upstream_available() -> bool:
    """True when the original StateMask package appears to be importable."""
    for name in _UPSTREAM_MODULES:
        try:
            importlib.import_module(name)
            return True
        except Exception:
            continue
    return False


def _fallback_rng(seed: Optional[int] = None) -> Any:
    """numpy-only RNG shim matching :class:`rice.utils.seeding.RNG`."""
    if RNG is not None:
        try:
            return RNG(seed=seed)
        except Exception:  # pragma: no cover
            pass

    class _FallbackRNG:
        def __init__(self, seed: Optional[int] = None) -> None:
            self.seed = seed
            self._gen = np.random.default_rng(seed)

        def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
            return float(self._gen.uniform(low, high))

        def bernoulli(self, p: float) -> bool:
            return bool(self._gen.uniform() < p)

        def choice(self, a: Any, p: Any = None) -> Any:
            return self._gen.choice(a, p=p)

        def integers(self, low: int, high: Optional[int] = None, size: Any = None) -> Any:
            return self._gen.integers(low, high, size=size)

        def __getattr__(self, item: str) -> Any:  # pragma: no cover
            return getattr(self._gen, item)

    return _FallbackRNG(seed)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class StateMaskExplanationConfig:
    """Configuration of the StateMask explanation adapter.

    The defaults mirror the original StateMask setting as re-implemented by
    RICE's mask network (Algorithm 1) so that the two explanations differ only
    in their *training* procedure (prime-dual surrogate vs. vanilla PPO),
    exactly what Experiment I measures.
    """

    task: Optional[str] = None
    #: Network architecture; ``None`` -> mirror the target agent (per env).
    net_arch: Optional[Tuple[int, ...]] = None
    activation: str = "tanh"
    obs_dim: Optional[int] = None
    #: Learning rate(s) that the *original* StateMask uses (Cheng et al. 2023).
    learning_rate: float = 3e-4
    #: Original StateMask keeps ``alpha`` at 0.01 in its own paper (§C.3 for RICE).
    alpha: float = 0.01
    #: Upper bound of the multiplier used by the original prime-dual objective.
    importance_upper_bound: float = 1.0
    #: Optional checkpoint path / state dict for a mask network.
    checkpoint: Optional[str] = None
    device: str = "auto"
    seed: Optional[int] = None
    #: ``"mask"`` uses the mask-network head; ``"value"`` uses |value| as the
    #: importance proxy when only a critic is available (rarely used).
    score_mode: str = "mask"
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "net_arch": self.net_arch,
            "activation": self.activation,
            "obs_dim": self.obs_dim,
            "learning_rate": self.learning_rate,
            "alpha": self.alpha,
            "importance_upper_bound": self.importance_upper_bound,
            "checkpoint": self.checkpoint,
            "device": self.device,
            "seed": self.seed,
            "score_mode": self.score_mode,
            "notes": self.notes,
            **dict(self.extra),
        }

    def clone(self, **overrides: Any) -> "StateMaskExplanationConfig":
        data = self.to_dict()
        data.pop("extra", None)
        data.update({k: v for k, v in overrides.items() if k in data or k in self.__dataclass_fields__})
        extra = dict(self.extra)
        extra.update({k: v for k, v in overrides.items() if k not in self.__dataclass_fields__})
        obj = StateMaskExplanationConfig(**{**data, "extra": extra})
        return obj

    @classmethod
    def from_mapping(
        cls, mapping: Optional[Any] = None, **overrides: Any
    ) -> "StateMaskExplanationConfig":
        """Build from a parsed YAML mapping (or another config object)."""
        data: Dict[str, Any] = {}
        if mapping is not None:
            if isinstance(mapping, StateMaskExplanationConfig):
                data = mapping.to_dict()
            elif isinstance(mapping, dict):
                data = dict(mapping)
            elif hasattr(mapping, "to_dict"):
                data = dict(mapping.to_dict())
            else:  # pragma: no cover - defensive
                data = dict(getattr(mapping, "__dict__", {}))

        # Friendly aliases seen in YAML / evaluation configs.
        aliases = {
            "beta": None,
            "lambda": None,
            "lam": None,
            "lr": "learning_rate",
            "arch": "net_arch",
            "checkpoint_path": "checkpoint",
        }
        for key, target in list(aliases.items()):
            if target and key in data and target not in data:
                data[target] = data.pop(key)
            elif target is None and key in data:
                data.pop(key)

        known = set(cls.__dataclass_fields__) - {"extra"}
        extra = dict(data.pop("extra", {}) or {})
        extra.update({k: v for k, v in data.items() if k not in known})
        clean = {k: v for k, v in data.items() if k in known}
        clean.update({k: v for k, v in overrides.items() if k in known})
        extra.update({k: v for k, v in overrides.items() if k not in known})
        if clean.get("net_arch") is not None:
            clean["net_arch"] = tuple(int(x) for x in clean["net_arch"])
        return cls(**clean, extra=extra)


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class StateMaskExplanation:
    """StateMask explanation: importance = ``P(mask = 0 | s)`` (§3.3).

    Usage (identical surface to every other RICE explanation object)::

        expl = StateMaskExplanation(mask_network=mask_net)
        scores = expl.importance(states)                 # (N,)
        idx = expl.select_index(states)                  # trajectory critical step
        obs, idx, val = expl.critical_state(states, return_index=True)

    Parameters
    ----------
    env:
        Optional environment (used only to infer observation/action sizes).
    policy:
        Optional target policy (never inspected — black-box assumption).
    mask_network:
        A trained :class:`~rice.algorithms.mask_network.MaskNetwork`, an upstream
        StateMask model, or any callable returning 2-class logits/probabilities.
    config:
        :class:`StateMaskExplanationConfig` or a parsed YAML mapping.
    device, seed, rng:
        Runtime plumbing.
    """

    name = NAME

    def __init__(
        self,
        env: Any = None,
        policy: Any = None,
        mask_network: Any = None,
        config: Optional[Any] = None,
        device: str = "auto",
        seed: Optional[int] = None,
        rng: Any = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(config, StateMaskExplanationConfig):
            self.config = config
        else:
            self.config = StateMaskExplanationConfig.from_mapping(config, **kwargs)

        self.env = env
        self.policy = policy
        if task is not None:
            self.config.task = task
        self.device_name = self.config.device if device == "auto" else device
        self.seed = self.config.seed if seed is None else seed
        self.rng = rng if rng is not None else _fallback_rng(self.seed)

        self._mask_network = mask_network
        self._model: Any = None
        self._external = False
        self._notes: List[str] = []
        self._obs_dim: Optional[int] = self.config.obs_dim
        self._act_dim: Optional[int] = None

        self._infer_dims()
        self._resolve_model()

    # -- construction helpers ------------------------------------------------

    def _infer_dims(self) -> None:
        env = self.env
        if env is None:
            return
        obs_space = getattr(env, "observation_space", None)
        act_space = getattr(env, "action_space", None)
        if obs_space is not None and observation_size is not None:
            try:
                self._obs_dim = int(observation_size(obs_space))
            except Exception:
                self._obs_dim = getattr(obs_space, "shape", (None,))[0]
        if act_space is not None:
            shape = getattr(act_space, "shape", None)
            if shape is not None:
                self._act_dim = int(np.prod(shape)) if len(shape) else 1

    def _resolve_model(self) -> None:
        """Resolve the underlying mask model from kwargs/checkpoint/env."""
        cfg = self.config

        # 1) A model explicitly provided by the caller.
        if self._mask_network is not None:
            if isinstance(self._mask_network, str):
                self._load_checkpoint(self._mask_network)
            else:
                self._model = self._mask_network
                self._external = not isinstance(self._mask_network, _mask_classes())
                if self._external:
                    self._notes.append(
                        "using caller-provided StateMask model (black-box importance)"
                    )
            return

        # 2) A checkpoint path.
        if cfg.checkpoint:
            self._load_checkpoint(cfg.checkpoint)
            if self._model is not None:
                return

        # 3) An upstream package model, if installed (§C.1).
        upstream = self._try_upstream()
        if upstream is not None:
            self._model = upstream
            self._external = True
            self._notes.append("using upstream StateMask package model")
            return

        # 4) Local re-implementation: an (untrained) mask network. Callers that
        #    need the trained StateMask explanation should pass a checkpoint or
        #    train one with ``train_statemask`` (Algorithm 1 = vanilla PPO).
        self._model = self._build_mask_network()
        self._notes.append(
            "no StateMask checkpoint/package found; using a local MaskNetwork "
            "(train it with train_statemask) — documents a reproduction deviation"
        )

    def _build_mask_network(self) -> Any:
        if MaskNetwork is None:
            self._notes.append("MaskNetwork unavailable: falling back to uniform scores")
            return None
        net_arch = self.config.net_arch
        kwargs: Dict[str, Any] = {
            "net_arch": tuple(net_arch) if net_arch is not None else (64, 64),
            "activation": self.config.activation,
            "obs_dim": self._obs_dim,
            "device": self.device_name,
        }
        try:
            model = MaskNetwork(self.env, **kwargs)
        except Exception:
            try:
                model = MaskNetwork(observation_space=None, **kwargs)
            except Exception as err:  # pragma: no cover - defensive
                self._notes.append(f"MaskNetwork construction failed: {err}")
                return None
        if self.config.seed is not None and hasattr(model, "_seed"):
            self.config.seed = self.config.seed
        return model

    def _load_checkpoint(self, path: str) -> None:
        if not path or not os.path.exists(path):
            self._notes.append(f"checkpoint not found: {path}")
            return
        if not _TORCH_AVAILABLE:
            self._notes.append("torch unavailable: cannot load checkpoint")
            return
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as err:
            self._notes.append(f"failed to load checkpoint {path}: {err}")
            return

        # Accept either a full training state or a bare state_dict.
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]

        model = self._build_mask_network()
        if model is None:
            # Upstream checkpoint of an unknown architecture: keep the raw dict
            # and let the importance extraction try to interpret it.
            self._model = payload
            self._external = True
            self._notes.append("checkpoint loaded as raw state dict (unknown architecture)")
            return

        loaded = False
        for loader_name in ("load_policy_state_dict", "load_state_dict"):
            loader = getattr(model, loader_name, None)
            if loader is None:
                continue
            try:
                if loader_name == "load_policy_state_dict":
                    loader(payload)
                else:
                    loader({"policy": payload} if "policy" not in payload else payload)
                loaded = True
                break
            except Exception:
                continue
        if not loaded:
            try:  # pragma: no cover - best effort
                model.load_state_dict(payload, strict=False)
                loaded = True
            except Exception as err:
                self._notes.append(f"checkpoint state dict rejected: {err}")
        if loaded:
            self._model = model
            self._notes.append(f"loaded StateMask checkpoint from {path}")

    def _try_upstream(self) -> Any:
        if not self.config.extra.get("use_upstream", True):
            return None
        for module_name in _UPSTREAM_MODULES:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            for attr in ("StateMask", "MaskNetwork", "MaskNet", "load_model", "load"):
                factory = getattr(module, attr, None)
                if factory is None:
                    continue
                try:
                    if self.config.checkpoint and attr in ("load_model", "load"):
                        return factory(self.config.checkpoint)
                    if callable(factory) and attr in ("StateMask", "MaskNetwork", "MaskNet"):
                        try:
                            return factory(self._obs_dim, self._act_dim)
                        except Exception:
                            return factory()
                except Exception:
                    continue
        return None

    # -- importance scoring --------------------------------------------------

    def importance(self, states: Any) -> np.ndarray:
        """Per-state importance ``P(mask = 0 | s_t)`` — verbatim §3.3 semantics."""
        if states is None:
            return np.zeros((0,), dtype=np.float64)

        # A precomputed array of scores passes straight through.
        if isinstance(states, np.ndarray) and states.ndim == 1 and states.size and np.issubdtype(
            states.dtype, np.floating
        ):
            if self.config.extra.get("scores_are_importance", False):
                return np.asarray(states, dtype=np.float64)

        model = self._model
        if model is None:
            return np.full((self._count_states(states),), 0.5, dtype=np.float64)

        # Native RICE mask network (preferred path).
        for method_name in ("mask_prob_zero", "importance"):
            method = getattr(model, method_name, None)
            if callable(method):
                try:
                    out = method(states)
                    return self._as_scores(out, self._count_states(states))
                except Exception:
                    continue

        # Generic torch module: softmax(logits)[:, 0].
        if _TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
            try:
                with torch.no_grad():
                    obs = self._to_tensor(states)
                    logits = model(obs)
                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]
                    logits = torch.as_tensor(logits)
                    if logits.ndim > 2:
                        logits = logits.reshape(logits.shape[0], -1)
                    if logits.shape[-1] >= 2:
                        probs = torch.softmax(logits, dim=-1)[..., 0]
                    else:  # single logit: sigmoid -> P(mask = 0)
                        probs = torch.sigmoid(logits.reshape(-1))
                    return probs.detach().cpu().numpy().reshape(-1).astype(np.float64)
            except Exception as err:  # pragma: no cover
                self._notes.append(f"torch importance failed: {err}")

        # Callable model (SB3-like ``predict`` or plain function).
        callable_model = make_target_policy_callable(model) if make_target_policy_callable else None
        if callable_model is not None:
            try:
                scores = []
                for state in self._iter_states(states):
                    out = np.asarray(callable_model(state)).reshape(-1)
                    if out.size >= 2:
                        s = float(np.exp(out[0]) / np.sum(np.exp(out)))
                    elif out.size == 1:
                        s = float(1.0 / (1.0 + np.exp(-out[0])))
                    else:
                        s = 0.5
                    scores.append(s)
                return np.asarray(scores, dtype=np.float64)
            except Exception as err:  # pragma: no cover
                self._notes.append(f"callable importance failed: {err}")

        # Raw state dict / unknown object -> uniform (uninformative) scores.
        return np.full((self._count_states(states),), 0.5, dtype=np.float64)

    # Optional aliases so every consumer in the code base works uniformly.
    def score(self, states: Any) -> np.ndarray:
        return self.importance(states)

    def mask_prob_zero(self, states: Any) -> np.ndarray:
        return self.importance(states)

    def __call__(self, states: Any) -> np.ndarray:
        return self.importance(states)

    # -- critical-state selection -------------------------------------------

    def select_index(self, states: Any, rng: Any = None) -> int:
        """Index of the most critical (highest-importance) visited state."""
        scores = self.importance(states)
        if scores.size == 0:
            return 0
        return int(np.argmax(scores))

    def best_index(self, states: Any, rng: Any = None) -> int:
        return self.select_index(states, rng=rng)

    def critical_state(
        self, states: Any, rng: Any = None, return_index: bool = False
    ) -> Any:
        idx = self.select_index(states, rng=rng)
        state_list = list(self._iter_states(states))
        state = state_list[idx] if state_list else None
        if return_index:
            return state, idx
        return state

    def critical_states_batch(self, trajectories: Sequence[Any], rng: Any = None) -> List[Any]:
        out: List[Any] = []
        for traj in trajectories:
            out.append(self.critical_state(traj, rng=rng))
        return out

    def select_top_k(self, states: Any, k: int = 1) -> Tuple[List[Any], List[int], np.ndarray]:
        scores = self.importance(states)
        state_list = list(self._iter_states(states))
        if scores.size == 0 or not state_list:
            return [], [], scores
        k = max(1, min(int(k), len(state_list)))
        order = np.argsort(-scores, kind="stable")[:k]
        return [state_list[int(i)] for i in order], [int(i) for i in order], scores

    # -- bookkeeping ---------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> "StateMaskExplanation":
        """Re-seed the (unused) RNG — present for interface parity."""
        if seed is not None:
            self.rng = _fallback_rng(seed)
        return self

    def update(self, *args: Any, **kwargs: Any) -> "StateMaskExplanation":
        """No-op: the StateMask explanation is trained separately (Algorithm 1)."""
        return self

    def set_mask_network(self, mask_network: Any) -> "StateMaskExplanation":
        self._mask_network = mask_network
        self._model = None
        self._external = False
        self._resolve_model()
        return self

    @property
    def mask_network(self) -> Any:
        return self._model

    @property
    def notes(self) -> List[str]:
        return list(self._notes)

    @property
    def available(self) -> bool:
        return self._model is not None

    def describe(self) -> Dict[str, Any]:
        return {
            "name": NAME,
            "aliases": list(ALIASES),
            "repo": STATEMASK_REPO,
            "upstream_installed": upstream_available(),
            "external_model": self._external,
            "obs_dim": self._obs_dim,
            "act_dim": self._act_dim,
            "config": self.config.to_dict(),
            "notes": list(self._notes),
        }

    def as_dict(self) -> Dict[str, Any]:
        return self.describe()

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "name": NAME,
            "config": self.config.to_dict(),
            "external": self._external,
        }
        if _TORCH_AVAILABLE and isinstance(self._model, torch.nn.Module):
            try:
                state["model"] = {k: v.detach().cpu() for k, v in self._model.state_dict().items()}
            except Exception:  # pragma: no cover
                pass
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> "StateMaskExplanation":
        if not isinstance(state, dict):
            return self
        if "config" in state and isinstance(state["config"], dict):
            self.config = StateMaskExplanationConfig.from_mapping(state["config"])
        model_state = state.get("model")
        if model_state is not None and _TORCH_AVAILABLE:
            if self._model is None:
                self._model = self._build_mask_network()
            if isinstance(self._model, torch.nn.Module):
                try:
                    self._model.load_state_dict(model_state, strict=False)
                except Exception as err:  # pragma: no cover
                    self._notes.append(f"state_dict load failed: {err}")
        return self

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"StateMaskExplanation(obs_dim={self._obs_dim}, external={self._external}, "
            f"available={self.available})"
        )

    # -- internal utilities --------------------------------------------------

    def _to_tensor(self, states: Any) -> Any:
        arr = self._to_array(states)
        tensor = torch.as_tensor(arr, dtype=torch.float32)
        device = getattr(self._model, "device", None)
        if device is not None:
            try:
                tensor = tensor.to(device)
            except Exception:  # pragma: no cover
                pass
        return tensor

    @staticmethod
    def _iter_states(states: Any):
        if states is None:
            return []
        if isinstance(states, np.ndarray):
            if states.ndim == 0:
                return [states.reshape(1)]
            if states.ndim == 1:
                return [states]
            return [s for s in states]
        if _TORCH_AVAILABLE and isinstance(states, torch.Tensor):
            arr = states.detach().cpu().numpy()
            return StateMaskExplanation._iter_states(arr)
        if isinstance(states, (list, tuple)):
            if len(states) == 0:
                return []
            first = states[0]
            if isinstance(first, (list, tuple, np.ndarray)) or (
                _TORCH_AVAILABLE and isinstance(first, torch.Tensor)
            ):
                return list(states)
            return [np.asarray(states)]
        return [np.asarray(states)]

    @staticmethod
    def _count_states(states: Any) -> int:
        if states is None:
            return 0
        if isinstance(states, np.ndarray):
            if states.ndim <= 1:
                return 1
            return int(states.shape[0])
        if _TORCH_AVAILABLE and isinstance(states, torch.Tensor):
            return 1 if states.dim() <= 1 else int(states.shape[0])
        if isinstance(states, (list, tuple)):
            if len(states) == 0:
                return 0
            first = states[0]
            if isinstance(first, (list, tuple, np.ndarray)) or (
                _TORCH_AVAILABLE and isinstance(first, torch.Tensor)
            ):
                return len(states)
            return 1
        return 1

    @staticmethod
    def _to_array(states: Any) -> np.ndarray:
        if _TORCH_AVAILABLE and isinstance(states, torch.Tensor):
            arr = states.detach().cpu().numpy()
        else:
            arr = np.asarray(states)
        if arr.dtype == object:  # ragged list of states
            arr = np.stack([np.asarray(s, dtype=np.float32).reshape(-1) for s in arr])
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    @staticmethod
    def _as_scores(out: Any, n: int) -> np.ndarray:
        if _TORCH_AVAILABLE and isinstance(out, torch.Tensor):
            scores = out.detach().cpu().numpy()
        else:
            scores = np.asarray(out)
        if scores.dtype == object:  # pragma: no cover - defensive
            scores = np.asarray([float(np.asarray(s).reshape(-1).mean()) for s in scores])
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.size != n:
            if scores.size == 1 and n > 1:
                scores = np.full((n,), float(scores[0]), dtype=np.float64)
            else:
                scores = scores[:n] if scores.size > n else np.pad(
                    scores, (0, n - scores.size), constant_values=0.0
                )
        return np.clip(scores, 0.0, 1.0)


#: Alias used by ``rice.baselines.statemask_r`` and evaluation code.
StateMaskAdapter = StateMaskExplanation
StateMaskExplainer = StateMaskExplanation


def _mask_classes() -> Tuple[type, ...]:
    return tuple(t for t in (MaskNetwork,) if isinstance(t, type))


# ---------------------------------------------------------------------------
# Training / building helpers
# ---------------------------------------------------------------------------


def build_statemask(
    env: Any = None,
    policy: Any = None,
    mask_network: Any = None,
    config: Optional[Any] = None,
    rng: Any = None,
    **kwargs: Any,
) -> StateMaskExplanation:
    """Build a StateMask explanation object (alias of :func:`make_statemask_explanation`)."""
    return StateMaskExplanation(
        env=env, policy=policy, mask_network=mask_network, config=config, rng=rng, **kwargs
    )


def make_statemask_explanation(
    env: Any = None,
    policy: Any = None,
    mask_network: Any = None,
    config: Optional[Any] = None,
    rng: Any = None,
    **kwargs: Any,
) -> StateMaskExplanation:
    """Factory mirroring ``rice.explanation.random_explanation.make_random_explanation``."""
    return StateMaskExplanation(
        env=env, policy=policy, mask_network=mask_network, config=config, rng=rng, **kwargs
    )


make_explanation = make_statemask_explanation


def train_statemask(
    env: Any,
    target_policy: Any,
    config: Optional[Any] = None,
    seed: Optional[int] = None,
    verbose: int = 0,
    **kwargs: Any,
) -> StateMaskExplanation:
    """Train the baseline StateMask explanation with RICE's Algorithm 1 loop.

    The original StateMask optimizes a prime-dual surrogate; §3.3 replaces it by
    ``J(theta) = max eta(pi_bar)`` and trains with vanilla PPO. This helper runs
    that loop via :class:`MaskNetworkTrainer` so the *explanation quality* is
    comparable (Experiment I) while remaining a faithful re-implementation.
    """
    cfg = (
        config
        if isinstance(config, StateMaskExplanationConfig)
        else StateMaskExplanationConfig.from_mapping(config, **kwargs)
    )
    if MaskNetworkTrainer is None or MaskNetwork is None:
        raise ImportError(
            "MaskNetwork/MaskNetworkTrainer unavailable; install torch and the "
            "rice.algorithms package to train the StateMask explanation."
        )

    mask = MaskNetwork(
        env,
        net_arch=tuple(cfg.net_arch) if cfg.net_arch is not None else (64, 64),
        activation=cfg.activation,
        obs_dim=cfg.obs_dim,
        device=cfg.device,
    )
    train_cfg_kwargs: Dict[str, Any] = {"alpha": cfg.alpha, "device": cfg.device}
    if cfg.seed is not None:
        train_cfg_kwargs["seed"] = cfg.seed
    if seed is not None:
        train_cfg_kwargs["seed"] = seed
    train_cfg_kwargs["verbose"] = verbose
    for key in ("n_iterations", "total_samples", "max_steps_per_iter", "policy_config"):
        if key in cfg.extra:
            train_cfg_kwargs[key] = cfg.extra[key]

    train_config = (
        MaskNetworkConfig(**train_cfg_kwargs) if MaskNetworkConfig is not None else None
    )
    trainer = MaskNetworkTrainer(
        env=env,
        target_policy=target_policy,
        mask_network=mask,
        config=train_config,
        rng=rng_for(seed),
    )
    info = trainer.train()
    expl = StateMaskExplanation(env=env, policy=target_policy, mask_network=mask, config=cfg)
    if isinstance(info, dict):
        expl._notes.append(
            f"trained via Algorithm 1 (vanilla PPO): samples={info.get('samples')}, "
            f"seconds={info.get('seconds')}"
        )
    return expl


def rng_for(seed: Optional[int] = None) -> Any:
    """Small public helper: RNG used by the adapter/trainer."""
    return _fallback_rng(seed)


def get_spec() -> Dict[str, Any]:
    """Registry metadata for ``rice.explanation`` dispatchers."""
    return {
        "name": NAME,
        "aliases": list(ALIASES),
        "repo": STATEMASK_REPO,
        "upstream_available": upstream_available(),
    }


__all__ = [
    "NAME",
    "ALIASES",
    "STATEMASK_REPO",
    "StateMaskExplanationConfig",
    "StateMaskExplanation",
    "StateMaskAdapter",
    "StateMaskExplainer",
    "make_statemask_explanation",
    "make_explanation",
    "build_statemask",
    "train_statemask",
    "upstream_available",
    "get_spec",
    "rng_for",
]

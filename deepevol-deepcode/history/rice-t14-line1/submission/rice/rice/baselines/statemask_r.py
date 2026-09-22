"""StateMask-R baseline for RICE (Section 4.1 of the paper).

The paper describes this baseline verbatim as

    "The second baseline is a refining method introduced by StateMask
     (Cheng et al., 2023), i.e., resetting to the critical state and
     continuing training from the critical state."

Source: §4.1 Experiment Setup (Baseline Refining Methods)
and, for the released implementation that supplies both the explanation
(the state mask network) and the reset behaviour,

    "as for StateMask, we use their released opensourced code from
     https://github.com/nuwuxian/RL-state_mask"
Source: §C.1 Implementation Details (Implementation of Baseline Methods)

Realization inside this code base
---------------------------------
The refining loop of RICE (Algorithm 2) *contains* StateMask-R as a special
case: if the Bernoulli(p) roll-in is forced to always select the critical
state (``p = 1``) and the RND intrinsic-reward bonus is switched off
(``lambda = 0``), the loop degenerates exactly to

    s_0 <- critical state  ;  continue PPO training  ;  no exploration bonus

which is the StateMask-R refining method.  Implementing it as a thin
configuration layer on top of :class:`rice.algorithms.refine.RICERefiner`
guarantees that RICE and StateMask-R differ only in the two ingredients under
study (mixed initial-state distribution and exploration bonus), which is the
apples-to-apples comparison the paper intends.

Notes / documented deviations
-----------------------------
* The paper does not state which learning rate StateMask-R uses when
  "continuing training".  Unlike "PPO fine-tuning", for which the paper
  explicitly says "lowering the learning rate" (Source: §4.1), no lowering is
  mentioned here, so the default is ``lr_factor = 1.0`` (keep the pre-training
  learning rate).  The factor is configurable.
* ``p = 1`` triggers a degeneracy warning inside ``MixedInitSampler``
  (``0 < p < 1`` is recommended for RICE itself, see §4.3); the warning is
  silenced here because ``p = 1`` is intentional for this baseline.
* When the released StateMask explanation is unavailable, the module falls
  back to (a) a user supplied ``mask_network``, (b) a mask network trained with
  this repository's re-designed mask trainer, or (c) an untrained mask network;
  the chosen fallback is recorded in ``StateMaskRSummary.notes``.
"""

from __future__ import annotations

import importlib
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "METHOD_NAME",
    "METHOD_ALIASES",
    "MASK_SAMPLE_BUDGETS",
    "StateMaskRConfig",
    "StateMaskRRefiner",
    "StateMaskRSummary",
    "make_statemask_r_refiner",
    "statemask_r_refine",
    "statemask_r_baseline",
    "build_statemask_explanation",
]


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

METHOD_NAME = "statemask_r"
METHOD_ALIASES: Tuple[str, ...] = (
    "statemask_r",
    "statemask-r",
    "statemaskr",
    "smr",
    "statemask_reset",
    "statemask-refine",
)

DEFAULT_LR_FACTOR: float = 1.0       # "continue training" (no lowering stated)
DEFAULT_N_ITERATIONS: int = 100
DEFAULT_EVAL_EPISODES: int = 5
DEFAULT_P: float = 1.0               # always start from the critical state
DEFAULT_LAMBDA: float = 0.0          # no RND exploration bonus
DEFAULT_ALPHA: float = 1.0e-4        # Table 3 value used for mask training
DEFAULT_PRETRAIN_LR: float = 3.0e-4  # Stable-Baselines3 default

# Mask-training sample budgets, Table 4 (also used as a refining-budget
# heuristic because the paper does not specify the refining budget).
MASK_SAMPLE_BUDGETS: Dict[str, float] = {
    "Hopper-v3": 3.0e5,
    "Walker2d-v3": 3.0e5,
    "Reacher-v2": 3.0e5,
    "HalfCheetah-v3": 3.0e5,
    "SparseHopper": 3.0e5,
    "SparseHalfCheetah": 3.0e5,
    "SparseWalker2d": 3.0e5,
    "SelfishMining": 1.5e6,
    "CageChallenge2": 1.0e7,
    "Macro-v1": 2443260.0,
}


# ---------------------------------------------------------------------------
# defensive imports
# ---------------------------------------------------------------------------

def _import_first(candidates: Sequence[Tuple[str, str]]) -> Any:
    """Import the first available ``module.attribute`` from ``candidates``.

    ``candidates`` is a sequence of ``(module, attribute)`` pairs; the first
    import that succeeds returns the resolved attribute.  Raises
    ``ImportError`` when nothing can be imported, preserving the last error so
    the failure mode stays informative.
    """
    last_error: Optional[BaseException] = None
    for module_name, attribute in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - depends on runtime layout
            last_error = exc
            continue
        if attribute == "" or hasattr(module, attribute):
            return module if attribute == "" else getattr(module, attribute)
        last_error = AttributeError(f"{module_name} has no attribute {attribute}")
    if last_error is not None:
        raise ImportError(str(last_error))
    raise ImportError("no import candidates supplied")


def _get(name: str, module: str) -> Any:
    """Resolve ``module.name`` across the repository layouts used here."""
    return _import_first(
        (
            (f"rice.{module}", name),
            (f"rice.rice.{module}", name),
            (module, name),
            (f"rice.baselines.{module}", name),
        )
    )


def _optional(name: str, module: str) -> Any:
    """Best-effort version of :func:`_get` returning ``None`` on failure."""
    try:
        return _get(name, module)
    except Exception:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass
class StateMaskRConfig:
    """Configuration of the StateMask-R refining baseline.

    The two ingredients that define this baseline are ``p = 1`` (every
    refining iteration starts from the identified critical state) and
    ``lam = 0`` (no RND intrinsic reward).  Everything else is shared with
    :class:`rice.algorithms.refine.RefineConfig` so the comparison with RICE
    is controlled.
    """

    # --- task / explanation -------------------------------------------------
    task: Optional[str] = None
    method: str = METHOD_NAME
    explanation: str = "statemask"      # released StateMask explanation
    p: float = DEFAULT_P
    lam: float = DEFAULT_LAMBDA
    alpha: float = DEFAULT_ALPHA

    # --- refining budget ----------------------------------------------------
    n_iterations: Optional[int] = DEFAULT_N_ITERATIONS
    steps_per_iter: Optional[int] = None
    total_env_steps: Optional[float] = None
    rollin_length: Optional[int] = None
    reset_on_done: bool = True
    lr_factor: float = DEFAULT_LR_FACTOR
    learning_rate: Optional[float] = None

    # --- evaluation ---------------------------------------------------------
    eval_every: int = 0
    eval_episodes: int = DEFAULT_EVAL_EPISODES
    final_eval: bool = True
    deterministic_eval: bool = True
    curve_window: int = 1
    measure_baseline: bool = True

    # --- multi-seed ---------------------------------------------------------
    n_seeds: int = 3
    seeds: Optional[Sequence[int]] = None
    seed: Optional[int] = None

    # --- runtime ------------------------------------------------------------
    device: str = "auto"
    verbose: int = 1
    log_every: int = 1
    weights: Optional[Any] = None       # warm-start checkpoint for the policy
    net_arch: Optional[Sequence[int]] = None
    policy_config: Optional[Any] = None
    rnd_config: Optional[Any] = None

    # --- mask network (StateMask explanation) -------------------------------
    train_mask: bool = False            # train the mask net if none supplied
    mask_weights: Optional[str] = None  # load a pre-trained mask network
    mask_samples: Optional[float] = None
    mask_config: Optional[Any] = None

    # --- environment construction ------------------------------------------
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    eval_env_kwargs: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- helpers -------------------------------------------------------------
    def clone(self, **overrides: Any) -> "StateMaskRConfig":
        data = dict(self.__dict__)
        data.update(overrides)
        return StateMaskRConfig(**data)

    @classmethod
    def from_mapping(
        cls, mapping: Optional[Dict[str, Any]] = None, **overrides: Any
    ) -> "StateMaskRConfig":
        """Build a config from a (YAML/JSON) mapping with alias handling."""
        if mapping is None:
            mapping = {}
        if not isinstance(mapping, dict):
            try:  # dataclass / namespace with to_dict()
                mapping = mapping.to_dict()
            except Exception:  # pragma: no cover
                mapping = dict(mapping)
        data: Dict[str, Any] = {}
        known = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        for key, value in dict(mapping).items():
            k = str(key).strip()
            if k == "lambda" or k == "lambda_":
                k = "lam"
            elif k == "beta":
                k = "p"
            elif k in ("K", "roll_in_length", "roll_in_steps"):
                k = "rollin_length"
            elif k in ("num_iterations", "iterations", "n_iters"):
                k = "n_iterations"
            elif k in ("env_steps", "budget"):
                k = "total_env_steps"
            elif k == "lower_lr_factor":
                k = "lr_factor"
            elif k == "pretrained" or k == "policy_weights":
                k = "weights"
            if k in known:
                data[k] = value
            else:
                data.setdefault("extra", {})
                if not isinstance(data["extra"], dict):
                    data["extra"] = {}
                data["extra"][str(key)] = value
        data.update(overrides)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    # -- derived values ------------------------------------------------------
    def seed_list(self) -> List[int]:
        if self.seeds:
            return [int(s) for s in self.seeds]
        n = max(1, int(self.n_seeds))
        base = int(self.seed) if self.seed is not None else 0
        return [base + i for i in range(n)]

    def resolved_learning_rate(self, pretrain_lr: Optional[float] = None) -> float:
        if self.learning_rate is not None:
            return float(self.learning_rate)
        lr = float(pretrain_lr) if pretrain_lr is not None else DEFAULT_PRETRAIN_LR
        return float(lr) * float(self.lr_factor)

    def mask_budget(self) -> float:
        if self.mask_samples is not None:
            return float(self.mask_samples)
        return float(MASK_SAMPLE_BUDGETS.get(str(self.task or ""), 3.0e5))

    def to_refine_config(self, pretrain_lr: Optional[float] = None, **overrides: Any) -> Any:
        """Build the shared :class:`RefineConfig` realizing StateMask-R."""
        RefineConfig = _get("RefineConfig", "algorithms.refine")
        PPOConfig = _optional("PPOConfig", "algorithms.ppo")

        policy_config = self.policy_config
        if policy_config is None and PPOConfig is not None:
            policy_config = PPOConfig()
        if policy_config is not None and hasattr(policy_config, "clone"):
            policy_config = policy_config.clone(
                learning_rate=self.resolved_learning_rate(pretrain_lr)
            )
        elif policy_config is not None:
            try:
                policy_config.learning_rate = self.resolved_learning_rate(pretrain_lr)
            except Exception:  # pragma: no cover
                pass

        kwargs: Dict[str, Any] = dict(
            p=float(self.p),
            lam=float(self.lam),
            n_iterations=self.n_iterations,
            steps_per_iter=self.steps_per_iter,
            total_env_steps=self.total_env_steps,
            rollin_length=self.rollin_length,
            reset_on_done=bool(self.reset_on_done),
            policy_config=policy_config,
            rnd_config=self.rnd_config,
            eval_every=int(self.eval_every),
            eval_episodes=int(self.eval_episodes),
            device=self.device,
            verbose=int(self.verbose),
            log_every=int(self.log_every),
        )
        kwargs.update(overrides)
        kwargs = {k: v for k, v in kwargs.items() if v is not None or k in ("p", "lam")}
        try:
            return RefineConfig(**kwargs)
        except TypeError:  # pragma: no cover - signature drift tolerance
            allowed = set(getattr(RefineConfig, "__dataclass_fields__", {}))
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            return RefineConfig(**kwargs)


# ---------------------------------------------------------------------------
# summary container (duck-types RefiningResult)
# ---------------------------------------------------------------------------

@dataclass
class StateMaskRSummary:
    """Aggregated multi-seed outcome of the StateMask-R baseline."""

    method: str = METHOD_NAME
    task: Optional[str] = None
    explanation: str = "statemask"
    seeds: List[int] = field(default_factory=list)
    results: List[Any] = field(default_factory=list)
    final_rewards: List[float] = field(default_factory=list)
    baseline_rewards: List[float] = field(default_factory=list)
    curves: List[Any] = field(default_factory=list)
    seconds: float = 0.0
    env_steps: float = 0.0
    config: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None

    # -- aggregates ----------------------------------------------------------
    @property
    def final_reward(self) -> float:
        if not self.final_rewards:
            return float("nan")
        return float(np.mean(self.final_rewards))

    @property
    def final_std(self) -> float:
        if len(self.final_rewards) < 2:
            return 0.0
        return float(np.std(self.final_rewards, ddof=0))

    @property
    def baseline_reward(self) -> float:
        if not self.baseline_rewards:
            return float("nan")
        return float(np.mean(self.baseline_rewards))

    @property
    def baseline_std(self) -> float:
        if len(self.baseline_rewards) < 2:
            return 0.0
        return float(np.std(self.baseline_rewards, ddof=0))

    @property
    def improvement(self) -> float:
        base = self.baseline_reward
        final = self.final_reward
        if np.isnan(base) or np.isnan(final):
            return float("nan")
        return float(final - base)

    # -- curves --------------------------------------------------------------
    def mean_curve(self, window: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(steps, mean_reward)`` averaged over seeds."""
        xs: Optional[np.ndarray] = None
        ys: List[np.ndarray] = []
        for curve in self.curves:
            steps, rewards = _as_curve(curve)
            if steps is None or rewards is None or rewards.size == 0:
                continue
            if window and int(window) > 1:
                rewards = _moving_average(rewards, int(window))
            if xs is None or xs.shape != steps.shape:
                xs = steps
            ys.append(np.asarray(rewards, dtype=np.float64))
        if not ys:
            return np.zeros(0), np.zeros(0)
        n = min(len(y) for y in ys)
        stacked = np.stack([y[:n] for y in ys], axis=0)
        xs_out = np.asarray(xs)[:n] if xs is not None else np.arange(n)
        return xs_out, stacked.mean(axis=0)

    def curve_std(self, window: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        xs, mean = self.mean_curve(window=window)
        ys: List[np.ndarray] = []
        for curve in self.curves:
            _, rewards = _as_curve(curve)
            if rewards is None or rewards.size == 0:
                continue
            if window and int(window) > 1:
                rewards = _moving_average(rewards, int(window))
            ys.append(np.asarray(rewards, dtype=np.float64))
        if not ys:
            return np.zeros(0), np.zeros(0)
        n = min(len(y) for y in ys)
        stacked = np.stack([y[:n] for y in ys], axis=0)
        return xs, (stacked.std(axis=0) if stacked.shape[0] > 1 else np.zeros_like(mean))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "task": self.task,
            "explanation": self.explanation,
            "final_reward": self.final_reward,
            "final_std": self.final_std,
            "baseline_reward": self.baseline_reward,
            "baseline_std": self.baseline_std,
            "improvement": self.improvement,
            "seeds": list(self.seeds),
            "seconds": self.seconds,
            "env_steps": self.env_steps,
            "n_seeds": len(self.results),
            "notes": list(self.notes),
            "error": self.error,
        }

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self.results)


# Alias names: the evaluation layer resolves ``<Prefix>Refiner`` classes by name.
StateMaskRRunner = StateMaskRRefinerAlias = None  # replaced below


def _as_curve(curve: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Normalize a refining curve into ``(steps, rewards)`` arrays."""
    if curve is None:
        return None, None
    if hasattr(curve, "steps") and hasattr(curve, "rewards"):
        return (
            np.asarray(getattr(curve, "steps"), dtype=np.float64),
            np.asarray(getattr(curve, "rewards"), dtype=np.float64),
        )
    if isinstance(curve, dict):
        steps = curve.get("steps")
        rewards = curve.get("rewards", curve.get("returns"))
        return (
            None if steps is None else np.asarray(steps, dtype=np.float64),
            None if rewards is None else np.asarray(rewards, dtype=np.float64),
        )
    if isinstance(curve, (tuple, list)) and len(curve) == 2:
        return (
            np.asarray(curve[0], dtype=np.float64),
            np.asarray(curve[1], dtype=np.float64),
        )
    try:
        return None, np.asarray(curve, dtype=np.float64)
    except Exception:  # pragma: no cover
        return None, None


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    window = max(1, int(window))
    if window == 1 or x.size == 0:
        return x
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(x, kernel, mode="valid")


# ---------------------------------------------------------------------------
# explanation helper (StateMask mask network)
# ---------------------------------------------------------------------------

def build_statemask_explanation(
    env: Any,
    policy: Any,
    mask_network: Any = None,
    config: Optional[StateMaskRConfig] = None,
    rng: Optional[Any] = None,
    verbose: int = 1,
) -> Tuple[Any, List[str]]:
    """Return ``(mask_network, notes)`` for the StateMask explanation.

    Resolution order:
      1. an explicitly supplied ``mask_network``;
      2. a checkpoint given by ``config.mask_weights`` (``MaskNetwork``);
      3. the released StateMask adapter (``rice.explanation.statemask_adapter``)
         when importing it succeeds -- this is the paper's first choice
         (Source: §C.1);
      4. a mask network trained here with ``train_mask=True``;
      5. an untrained mask network (documented deviation).
    """
    notes: List[str] = []
    config = config or StateMaskRConfig()

    if mask_network is not None:
        notes.append("mask network supplied by the caller")
        return mask_network, notes

    MaskNetwork = _optional("MaskNetwork", "algorithms.mask_network")
    device = getattr(config, "device", "auto")

    # (2) checkpoint on disk
    if config.mask_weights:
        if MaskNetwork is None:
            notes.append("mask checkpoint given but MaskNetwork is unavailable")
        else:
            try:
                net = _new_mask_network(MaskNetwork, env, policy, config)
                state = _optional("load_policy_weights", "algorithms.refine")
                try:
                    import torch  # noqa: WPS433 (local import keeps numpy-only use)

                    payload = torch.load(str(config.mask_weights), map_location="cpu")
                    if isinstance(payload, dict) and "state_dict" in payload:
                        payload = payload["state_dict"]
                    net.load_policy_state_dict(payload)
                    net.to(torch.device("cpu") if device == "cpu" else net.device)
                except Exception:
                    if state is not None:
                        state(net.actor_critic if hasattr(net, "actor_critic") else net,
                              config.mask_weights)
                notes.append(f"loaded mask checkpoint from {config.mask_weights}")
                return net, notes
            except Exception as exc:
                notes.append(f"failed to load mask checkpoint: {exc}")

    # (3) released StateMask adapter
    adapter = None
    try:
        module = importlib.import_module("rice.explanation.statemask_adapter")
        for attr in ("StateMaskExplanation", "StateMaskAdapter", "build_statemask"):
            if hasattr(module, attr):
                adapter = getattr(module, attr)
                break
    except Exception as exc:
        notes.append(f"released StateMask adapter unavailable ({exc.__class__.__name__})")

    if adapter is not None:
        try:
            kwargs: Dict[str, Any] = {}
            for key, value in (("env", env), ("policy", policy), ("config", config),
                               ("rng", rng), ("verbose", verbose)):
                kwargs[key] = value
            try:
                net = adapter(**kwargs)
            except TypeError:
                # adapter may only accept (env, policy)
                net = adapter(env, policy)
            notes.append("using rice.explanation.statemask_adapter")
            return net, notes
        except Exception as exc:
            notes.append(f"StateMask adapter failed at construction: {exc}")

    # (4)/(5) local mask network
    if MaskNetwork is None:
        notes.append("no mask network available; StateMask-R will fall back to "
                     "always resetting to the environment default initial state")
        return None, notes

    net = _new_mask_network(MaskNetwork, env, policy, config)
    if config.train_mask:
        try:
            MaskNetworkConfig = _get("MaskNetworkConfig", "algorithms.mask_network")
            MaskNetworkTrainer = _get("MaskNetworkTrainer", "algorithms.mask_network")
            PPOConfig = _optional("PPOConfig", "algorithms.ppo")
            mask_cfg = config.mask_config
            if mask_cfg is None:
                fields = dict(
                    alpha=float(config.alpha),
                    n_iterations=None,
                    total_samples=config.mask_budget(),
                    device=config.device,
                    verbose=int(config.verbose),
                )
                if PPOConfig is not None:
                    fields["policy_config"] = PPOConfig()
                try:
                    mask_cfg = MaskNetworkConfig(**fields)
                except TypeError:
                    fields.pop("total_samples", None)
                    mask_cfg = MaskNetworkConfig(**fields)
            trainer = MaskNetworkTrainer(env=env, target_policy=policy,
                                         mask_network=net, config=mask_cfg, rng=rng)
            info = trainer.train() or {}
            if hasattr(trainer, "mask_network"):
                net = trainer.mask_network
            notes.append(
                "trained the mask network locally with the re-designed mask "
                f"trainer (samples={info.get('samples')}, "
                f"seconds={info.get('seconds')}) -- deviation from the released "
                "StateMask code, which was not available"
            )
        except Exception as exc:
            notes.append(f"local mask training failed ({exc}); using untrained mask net")
    else:
        notes.append("using an untrained mask network (train_mask=False)")
    return net, notes


def _new_mask_network(MaskNetwork: Any, env: Any, policy: Any, config: StateMaskRConfig) -> Any:
    """Instantiate a ``MaskNetwork`` mirroring the target agent architecture."""
    net_arch = config.net_arch
    if net_arch is None:
        net_arch = getattr(policy, "net_arch", None)
    if net_arch is None:
        try:
            default_net_arch = _get("default_net_arch", "environments")
            task = config.task or getattr(env, "rise_canonical_name", None)
            if task is not None:
                net_arch = default_net_arch(task)
        except Exception:  # pragma: no cover
            net_arch = None
    kwargs: Dict[str, Any] = {"device": config.device}
    if net_arch is not None:
        kwargs["net_arch"] = tuple(int(x) for x in net_arch)
    observation_space = getattr(env, "observation_space", None)
    if observation_space is not None:
        kwargs["observation_space"] = observation_space
    else:  # pragma: no cover - minimal fallback
        kwargs["obs_dim"] = int(np.prod(getattr(env, "observation_shape", (4,))))
    try:
        return MaskNetwork(**kwargs)
    except TypeError:
        kwargs.pop("device", None)
        return MaskNetwork(**kwargs)


# ---------------------------------------------------------------------------
# the refiner
# ---------------------------------------------------------------------------

class StateMaskRRefiner:
    """Runs the StateMask-R baseline on one task.

    The heavy lifting (roll-out, PPO update, critical-state restoration) is
    delegated to :class:`rice.algorithms.refine.RICERefiner` configured with
    ``p = 1`` and ``lambda = 0``.
    """

    def __init__(
        self,
        env: Any = None,
        policy: Any = None,
        config: Optional[Any] = None,
        evaluation_env: Any = None,
        state_manager: Any = None,
        rng: Optional[Any] = None,
        task: Optional[str] = None,
        mask_network: Any = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = StateMaskRConfig(**kwargs) if kwargs else StateMaskRConfig()
        elif isinstance(config, dict):
            config = StateMaskRConfig.from_mapping(config)
        elif kwargs:
            config = config.clone(**kwargs)
        self.config: StateMaskRConfig = config
        if task is not None:
            self.config.task = task

        self._env = env
        self._evaluation_env = evaluation_env
        self._policy = policy
        self._mask_network = mask_network
        self._state_manager = state_manager
        self._rng = rng or _make_rng(self.config.seed)
        self._notes: List[str] = []
        self._summary: Optional[StateMaskRSummary] = None
        self._refiner: Any = None
        self.mask_seconds: float = 0.0
        self.mask_samples: float = 0.0

    # -- plumbing ------------------------------------------------------------
    @property
    def rng(self) -> Any:
        return self._rng

    @property
    def summary(self) -> Optional[StateMaskRSummary]:
        return self._summary

    @property
    def notes(self) -> List[str]:
        return self._notes

    def resolve_task(self) -> Optional[str]:
        if self.config.task:
            return self.config.task
        env = self._env
        for attr in ("rise_canonical_name", "spec_id", "task"):
            value = getattr(env, attr, None)
            if isinstance(value, str):
                self.config.task = value
                return value
        return self.config.task

    def build_env(self, seed: Optional[int] = None) -> Any:
        if self._env is not None:
            return self._env
        make_env = _get("make_env", "environments")
        kwargs = dict(self.config.env_kwargs)
        task = self.resolve_task()
        if task is not None:
            kwargs.setdefault("name", task)
        if seed is not None:
            kwargs.setdefault("seed", seed)
        self._env = make_env(**kwargs)
        return self._env

    def build_eval_env(self, seed: Optional[int] = None) -> Any:
        if self._evaluation_env is not None:
            return self._evaluation_env
        try:
            make_env = _get("make_env", "environments")
        except Exception:  # pragma: no cover
            return self.build_env(seed)
        kwargs = dict(self.config.eval_env_kwargs or self.config.env_kwargs)
        task = self.resolve_task()
        if task is not None:
            kwargs.setdefault("name", task)
        if seed is not None:
            kwargs.setdefault("seed", int(seed) + 10_000)
        try:
            self._evaluation_env = make_env(**kwargs)
            return self._evaluation_env
        except Exception:  # pragma: no cover
            return self.build_env(seed)

    def build_policy(self, env: Any = None) -> Any:
        if self._policy is not None:
            return self._policy
        env = env if env is not None else self.build_env()
        ActorCritic = _get("ActorCritic", "algorithms.ppo")
        PPOConfig = _optional("PPOConfig", "algorithms.ppo")
        load_policy_weights = _optional("load_policy_weights", "algorithms.refine")

        net_arch = self.config.net_arch
        if net_arch is None:
            try:
                default_net_arch = _get("default_net_arch", "environments")
                task = self.resolve_task() or getattr(env, "rise_canonical_name", None)
                if task is not None:
                    net_arch = default_net_arch(task)
            except Exception:  # pragma: no cover
                net_arch = None

        kwargs: Dict[str, Any] = {
            "observation_space": getattr(env, "observation_space", None),
            "action_space": getattr(env, "action_space", None),
            "device": self.config.device,
        }
        if net_arch is not None:
            kwargs["net_arch"] = tuple(int(x) for x in net_arch)
        policy_cfg = self.config.policy_config
        if policy_cfg is None and PPOConfig is not None:
            policy_cfg = PPOConfig()
        if policy_cfg is not None:
            try:
                kwargs["net_arch"] = tuple(
                    kwargs.get("net_arch") or getattr(policy_cfg, "net_arch", (64, 64))
                )
            except Exception:  # pragma: no cover
                pass
        try:
            policy = ActorCritic(**kwargs)
        except TypeError:  # pragma: no cover
            kwargs.pop("net_arch", None)
            policy = ActorCritic(**kwargs)

        weights = self.config.weights
        if weights is not None and load_policy_weights is not None:
            try:
                policy = load_policy_weights(policy, weights)
            except Exception as exc:
                self._notes.append(f"could not load warm-start weights: {exc}")
        self._policy = policy
        return policy

    def state_manager_for(self, env: Any = None) -> Any:
        if self._state_manager is not None:
            return self._state_manager
        env = env if env is not None else self.build_env()
        make_state_manager = _optional("make_state_manager", "algorithms.env_reset")
        if make_state_manager is None:
            return None
        try:
            self._state_manager = make_state_manager(env, rng=self._rng)
        except Exception as exc:  # pragma: no cover
            self._notes.append(f"state manager unavailable: {exc}")
            self._state_manager = None
        return self._state_manager

    def build_mask_network(self, env: Any = None, policy: Any = None,
                           rng: Optional[Any] = None) -> Any:
        if self._mask_network is not None:
            return self._mask_network
        env = env if env is not None else self.build_env()
        policy = policy if policy is not None else self.build_policy(env)
        start = time.time()
        net, notes = build_statemask_explanation(
            env=env,
            policy=policy,
            mask_network=None,
            config=self.config,
            rng=rng or self._rng,
            verbose=int(self.config.verbose),
        )
        self.mask_seconds += time.time() - start
        for note in notes:
            if note not in self._notes:
                self._notes.append(note)
        self._mask_network = net
        return net

    def build_refine_config(self, seed: Optional[int] = None, **overrides: Any) -> Any:
        cfg = self.config.to_refine_config(**overrides)
        if seed is not None:
            try:
                cfg.seed = int(seed)
            except Exception:  # pragma: no cover
                pass
        return cfg

    # -- evaluation ----------------------------------------------------------
    def evaluate(self, policy: Any = None, env: Any = None,
                 seed: Optional[int] = None, n_episodes: Optional[int] = None) -> Dict[str, float]:
        evaluate_policy = _get("evaluate_policy", "algorithms.refine")
        env = env if env is not None else self.build_eval_env(seed)
        policy = policy if policy is not None else self.build_policy(self.build_env())
        n = int(n_episodes or self.config.eval_episodes)
        try:
            return evaluate_policy(
                env, policy, n_episodes=n, seed=seed,
                deterministic=bool(self.config.deterministic_eval),
            )
        except TypeError:  # pragma: no cover - signature drift
            return evaluate_policy(env, policy, n_episodes=n, seed=seed)

    def baseline_reward(self, seed: Optional[int] = None) -> float:
        """Reward of the warm-start policy before refining ("No Refine")."""
        if not self.config.measure_baseline:
            return float("nan")
        env = self.build_env(seed)
        policy = self.build_policy(env)
        stats = self.evaluate(policy=policy, env=self.build_eval_env(seed),
                              seed=seed, n_episodes=self.config.eval_episodes)
        return float(stats.get("mean_reward", stats.get("mean_return", np.nan)))

    # -- main entry points ---------------------------------------------------
    def refine(self, seed: Optional[int] = None, n_iterations: Optional[int] = None,
               **overrides: Any) -> Any:
        """Run one refining run; returns a ``RefineResult``."""
        task = self.resolve_task()
        env = self.build_env(seed)
        policy = self.build_policy(env)
        state_manager = self.state_manager_for(env)
        mask_network = self.build_mask_network(env, policy, rng=self._rng)

        if mask_network is None:
            self._notes.append(
                "no mask network: StateMask-R degenerates to PPO fine-tuning with "
                "p=1 (default-initial-state resets)"
            )

        cfg = self.build_refine_config(seed=seed, **overrides)
        if n_iterations is not None:
            try:
                cfg.n_iterations = int(n_iterations)
            except Exception:  # pragma: no cover
                pass

        refine_policy = _get("refine_policy", "algorithms.refine")
        # StateMask-R intentionally uses p=1; silence the degeneracy warning
        # emitted by MixedInitSampler for p in {0, 1}.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = refine_policy(
                env=env,
                policy=policy,
                mask_network=mask_network,
                config=cfg,
                state_manager=state_manager,
                evaluation_env=self.build_eval_env(seed),
                rng=self._rng,
            )
        return result

    # alias used by some call sites
    train = refine

    def run(self, seeds: Optional[Sequence[int]] = None, **kwargs: Any) -> StateMaskRSummary:
        """Run the baseline for several seeds and aggregate the outcomes."""
        if seeds is None:
            seeds = [self.config.seed] if self.config.seed is not None else self.config.seed_list()
        seeds = [int(s) for s in seeds]

        summary = StateMaskRSummary(
            method=METHOD_NAME,
            task=self.resolve_task(),
            explanation=str(self.config.explanation),
            seeds=list(seeds),
            config=self.config.to_dict(),
            notes=list(self._notes),
        )
        t0 = time.time()
        for seed in seeds:
            try:
                result = self.refine(seed=seed, **kwargs)
            except Exception as exc:  # keep sweeps alive
                summary.notes.append(f"seed {seed} failed: {exc}")
                summary.error = str(exc)
                continue
            summary.results.append(result)
            summary.final_rewards.append(_final_reward_of(result))
            baseline = getattr(result, "baseline_reward", None)
            if baseline is None or (isinstance(baseline, float) and np.isnan(baseline)):
                baseline = self.baseline_reward(seed)
            summary.baseline_rewards.append(float(baseline))
            curve = getattr(result, "refining_curve", None)
            if callable(curve):
                try:
                    curve = curve(window=int(self.config.curve_window))
                except Exception:  # pragma: no cover
                    curve = None
            if curve is not None:
                summary.curves.append(curve)
            summary.env_steps += float(getattr(result, "env_steps", 0.0) or 0.0)
        summary.seconds = time.time() - t0
        summary.notes = list(dict.fromkeys(list(summary.notes) + list(self._notes)))
        self._summary = summary
        return summary

    run_seeds = run

    # -- persistence ---------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "method": METHOD_NAME,
            "config": self.config.to_dict(),
            "notes": list(self._notes),
            "mask_seconds": self.mask_seconds,
            "mask_samples": self.mask_samples,
        }

    def save(self, path: str) -> str:
        try:
            import pickle

            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            with open(path, "wb") as fh:
                pickle.dump({"state": self.state_dict(), "summary": self._summary}, fh)
        except Exception:  # pragma: no cover
            pass
        return str(path)

    def describe(self) -> str:
        cfg = self.config
        return (
            f"StateMask-R[{cfg.task}] p={cfg.p} lambda={cfg.lam} "
            f"lr_factor={cfg.lr_factor} iterations={cfg.n_iterations}"
        )


# ---------------------------------------------------------------------------
# summaries / helpers
# ---------------------------------------------------------------------------

def _final_reward_of(result: Any) -> float:
    """Extract the final (post-refining) reward from a refiner result."""
    for attr in ("final_reward", "final_eval_reward", "mean_episode_return"):
        value = getattr(result, attr, None)
        if value is None:
            continue
        try:
            value = float(value)
        except Exception:  # pragma: no cover
            continue
        if not np.isnan(value):
            return value
    rewards = getattr(result, "episode_returns", None)
    if rewards is not None and len(rewards):
        return float(np.mean(np.asarray(rewards, dtype=np.float64)))
    return float("nan")


def _make_rng(seed: Optional[int]) -> Any:
    """Create the canonical RNG, falling back to numpy's Generator."""
    try:
        RNG = _get("RNG", "utils.seeding")
        return RNG(seed)
    except Exception:  # pragma: no cover
        return np.random.default_rng(seed)


def make_statemask_r_refiner(env: Any = None, policy: Any = None,
                             config: Optional[Any] = None, **kwargs: Any) -> StateMaskRRefiner:
    """Factory mirroring ``rice.algorithms.refine.make_refiner``."""
    return StateMaskRRefiner(env=env, policy=policy, config=config, **kwargs)


def statemask_r_refine(
    env: Any = None,
    policy: Any = None,
    mask_network: Any = None,
    config: Optional[Any] = None,
    seeds: Optional[Sequence[int]] = None,
    n_iterations: Optional[int] = None,
    evaluation_env: Any = None,
    state_manager: Any = None,
    rng: Optional[Any] = None,
    task: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Functional entry point for the StateMask-R baseline.

    Returns a :class:`StateMaskRSummary` when several seeds are requested and a
    ``RefineResult`` for a single seed (mirroring ``ppo_finetune``).
    """
    refiner = StateMaskRRefiner(
        env=env,
        policy=policy,
        config=config,
        evaluation_env=evaluation_env,
        state_manager=state_manager,
        rng=rng,
        task=task,
        mask_network=mask_network,
        **kwargs,
    )
    if seeds is None:
        seeds = refiner.config.seed_list()
    seeds = [int(s) for s in seeds]
    if len(seeds) == 1:
        return refiner.refine(seed=seeds[0], n_iterations=n_iterations)
    return refiner.run(seeds=seeds, **({} if n_iterations is None else {"n_iterations": n_iterations}))


# the evaluation layer's alias for the functional baseline entry point
statemask_r_baseline = statemask_r_refine

# class alias expected by ``rice.baselines.__init__`` (``<Prefix>Refiner``)
StateMaskRRunner = StateMaskRRefiner
StateMaskRRefinerAlias = StateMaskRRefiner

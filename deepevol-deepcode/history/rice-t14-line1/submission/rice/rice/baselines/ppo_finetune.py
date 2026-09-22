"""PPO fine-tuning baseline ("PPO fine-tuning", Schulman et al. 2017).

Paper definition (verbatim, Sec. 4.1 *Baseline Refining Methods*):

    The first baseline is "PPO fine-tuning" (Schulman et al., 2017), i.e.,
    lowering the learning rate and continuing training with the PPO algorithm.

This baseline is therefore exactly the RICE refining machinery **with both
RICE-specific components switched off**:

* ``p = 0``  -> the mixed initial-state distribution degenerates to the default
  ``rho`` (no critical-state roll-in, Algorithm 2's ``RAND_NUM < p`` branch is
  never taken);
* ``lambda = 0`` -> the RND intrinsic reward is disabled, so the augmented
  reward ``R' = R + lambda * R_RND`` equals the task reward ``R``;
* the PPO learning rate is **lowered** relative to the pre-training learning
  rate (the paper does not state the factor; we use ``10x`` lower as the plan's
  documented default, i.e. ``lr_ft = lr_pretrain / 10``).

Everything else (rollout length ``T``, PPO loss, GAE, epochs, ...) is shared
with ``rice.algorithms.refine`` / ``rice.algorithms.ppo`` so that the comparison
against RICE is apples-to-apples: only the mixed-init roll-in and the RND bonus
differ.

The module exposes:

* :class:`PPOFineTuneConfig`  - dataclass of baseline hyper-parameters,
* :class:`PPOFineTuner`       - the runner (single/multi-seed),
* :class:`FineTuneSummary`    - multi-seed aggregation (RefineResult-compatible),
* :func:`ppo_finetune`        - functional entry point used by
  ``rice.evaluation.refining_eval`` / ``rice.baselines.__init__``.

Unknown/unspecified settings default to the shared PPO defaults and are listed
in the README.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "PPOFineTuneConfig",
    "PPOFineTuner",
    "PPOFineTuneRefiner",
    "FineTuneSummary",
    "ppo_finetune",
    "ppo_finetune_baseline",
    "make_ppo_finetuner",
    "METHOD_NAME",
]

METHOD_NAME = "ppo"
METHOD_ALIASES: Tuple[str, ...] = ("ppo", "ppo_finetune", "ppo-finetune", "finetune", "fine_tune")
"""Friendly names accepted by :func:`ppo_finetune`'s ``method`` argument."""


# --------------------------------------------------------------------------- #
# Defensive imports (the repository can be imported from several layouts)
# --------------------------------------------------------------------------- #
def _import_first(module_names: Sequence[str], package: Optional[str] = None) -> Any:
    """Import the first importable module out of ``module_names``."""
    for name in module_names:
        try:
            if name.startswith("."):
                return importlib.import_module(name, package=package)
            return importlib.import_module(name)
        except Exception:  # pragma: no cover - environment dependent
            continue
    return None


_PACKAGE = __package__ or "rice.baselines"

_refine_mod = _import_first(
    (
        "..algorithms.refine",
        "rice.algorithms.refine",
        "rice.rice.algorithms.refine",
        "algorithms.refine",
        "refine",
    ),
    package=_PACKAGE,
)
_ppo_mod = _import_first(
    (
        "..algorithms.ppo",
        "rice.algorithms.ppo",
        "rice.rice.algorithms.ppo",
        "algorithms.ppo",
        "ppo",
    ),
    package=_PACKAGE,
)
_rnd_mod = _import_first(
    (
        "..algorithms.rnd",
        "rice.algorithms.rnd",
        "rice.rice.algorithms.rnd",
        "algorithms.rnd",
        "rnd",
    ),
    package=_PACKAGE,
)

# Shared PPO hyper-parameter container (Stable-Baselines3 defaults).
PPOConfig = getattr(_ppo_mod, "PPOConfig", None)
RNDConfig = getattr(_rnd_mod, "RNDConfig", None)
make_rnd = getattr(_rnd_mod, "make_rnd", None)

RefineConfig = getattr(_refine_mod, "RefineConfig", None)
RefineResult = getattr(_refine_mod, "RefineResult", None)
RICERefiner = getattr(_refine_mod, "RICERefiner", None)
refine_policy = getattr(_refine_mod, "refine_policy", None)
evaluate_policy = getattr(_refine_mod, "evaluate_policy", None)
load_policy_weights = getattr(_refine_mod, "load_policy_weights", None)

_REFINE_AVAILABLE = RefineConfig is not None and RICERefiner is not None

# Default pre-training learning rate (Stable-Baselines3 PPO default), used for
# the "lowering the learning rate" factor of this baseline.
DEFAULT_PRETRAIN_LR = 3e-4

# Paper-silent defaults (documented in README).
DEFAULT_LR_FACTOR = 0.1
DEFAULT_N_ITERATIONS = 100
DEFAULT_EVAL_EPISODES = 5


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _field(mapping: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict-like or attribute-like mapping."""
    if mapping is None:
        return default
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def _config_field_names(cls: Any) -> List[str]:
    try:
        return [f.name for f in cls.__dataclass_fields__.values()]  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        return []


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.std(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class PPOFineTuneConfig:
    """Hyper-parameters of the PPO fine-tuning baseline.

    Defaults mirror :class:`rice.algorithms.refine.RefineConfig` where the two
    overlap, with the RICE-specific switches fixed to their "off" values
    (``p = 0``, ``lam = 0``) and the learning rate scaled down by
    :attr:`lr_factor`.
    """

    # --- what to run -------------------------------------------------------
    task: Optional[str] = None
    method: str = METHOD_NAME
    explanation: str = "none"

    # --- learning-rate lowering (the defining feature of this baseline) ----
    lr_factor: float = DEFAULT_LR_FACTOR
    """``lr_ft = lr_factor * lr_pretrain`` (paper does not specify -> 0.1)."""
    learning_rate: Optional[float] = None
    """Explicit absolute learning rate; overrides ``lr_factor`` when set."""

    # --- budget / loop -----------------------------------------------------
    n_iterations: Optional[int] = DEFAULT_N_ITERATIONS
    steps_per_iter: Optional[int] = None
    total_env_steps: Optional[float] = None
    rollin_length: Optional[int] = None
    reset_on_done: bool = True

    # --- RICE-specific switches, disabled for this baseline ---------------
    p: float = 0.0
    lam: float = 0.0
    alpha: float = 1e-4
    """Kept for Table-3 interface parity; unused (no mask network)."""

    # --- evaluation --------------------------------------------------------
    eval_every: int = 0
    eval_episodes: int = DEFAULT_EVAL_EPISODES
    final_eval: bool = True
    deterministic_eval: bool = True
    curve_window: int = 1
    measure_baseline: bool = True

    # --- seeds / run bookkeeping ------------------------------------------
    n_seeds: int = 3
    seeds: Optional[Sequence[int]] = None
    seed: Optional[int] = None

    # --- plumbing ----------------------------------------------------------
    device: str = "auto"
    verbose: int = 1
    log_every: int = 1
    weights: Optional[Any] = None
    net_arch: Optional[Sequence[int]] = None
    policy_config: Any = None
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    eval_env_kwargs: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- construction -------------------------------------------------------
    def clone(self, **overrides: Any) -> "PPOFineTuneConfig":
        """Return a copy with ``overrides`` applied."""
        known = {k: v for k, v in overrides.items() if k in _config_field_names(type(self))}
        extra = {k: v for k, v in overrides.items() if k not in known}
        cfg = replace(self, **known)
        if extra:
            merged = dict(cfg.extra)
            merged.update(extra)
            cfg.extra = merged
        return cfg

    @classmethod
    def from_mapping(cls, mapping: Any = None, **overrides: Any) -> "PPOFineTuneConfig":
        """Build a config from a dict, a YAML mapping, or any config object.

        Accepts the friendly aliases used across the code base:
        ``lower_lr_factor`` -> ``lr_factor``, ``beta`` -> ``p``,
        ``lambda``/``coef``/``lambda_`` -> ``lam``, ``K``/``length`` ->
        ``rollin_length``, ``num_iterations`` -> ``n_iterations``,
        ``n_episodes`` -> ``eval_episodes``.
        """
        cfg = cls()
        if mapping is not None:
            names = _config_field_names(cls)
            values: Dict[str, Any] = {}
            for name in names:
                value = _field(mapping, name, None)
                if value is not None:
                    values[name] = value
            aliases = {
                "lower_lr_factor": "lr_factor",
                "lr_scale": "lr_factor",
                "beta": "p",
                "lambda": "lam",
                "lambda_": "lam",
                "coef": "lam",
                "K": "rollin_length",
                "length": "rollin_length",
                "num_iterations": "n_iterations",
                "max_iterations": "n_iterations",
                "n_episodes": "eval_episodes",
                "num_seeds": "n_seeds",
            }
            for alias, target in aliases.items():
                value = _field(mapping, alias, None)
                if value is not None and target not in values:
                    values[target] = value
            cfg = replace(cfg, **values)
        return cfg.clone(**overrides)

    def to_dict(self) -> Dict[str, Any]:
        data = {}
        for name in _config_field_names(type(self)):
            data[name] = getattr(self, name)
        return data

    # -- derived values -----------------------------------------------------
    def resolved_learning_rate(self, pretrain_lr: Optional[float] = None) -> float:
        """The (lowered) PPO learning rate used for fine-tuning."""
        if self.learning_rate is not None:
            return float(self.learning_rate)
        base = float(pretrain_lr) if pretrain_lr else DEFAULT_PRETRAIN_LR
        return float(base * self.lr_factor)

    def seed_list(self) -> List[int]:
        """Seeds to run the baseline with."""
        if self.seeds is not None:
            return [int(s) for s in self.seeds]
        base = int(self.seed) if self.seed is not None else 0
        n = max(1, int(self.n_seeds))
        return [base + i for i in range(n)]

    def budget(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Return ``(n_iterations, steps_per_iter, total_env_steps)``."""
        return (
            None if self.n_iterations is None else int(self.n_iterations),
            None if self.steps_per_iter is None else int(self.steps_per_iter),
            None if self.total_env_steps is None else int(self.total_env_steps),
        )

    # -- conversion to the shared refining loop ----------------------------
    def policy_config_for(self, pretrain_lr: Optional[float] = None) -> Any:
        """PPOConfig with the lowered learning rate applied."""
        base = self.policy_config
        if base is None:
            if PPOConfig is None:  # pragma: no cover - torch/sb3 missing
                raise RuntimeError(
                    "rice.algorithms.ppo is unavailable; cannot build the PPO fine-tuning baseline."
                )
            base = PPOConfig()
        lr = self.resolved_learning_rate(pretrain_lr)
        try:
            return base.clone(learning_rate=lr)
        except Exception:  # pragma: no cover - non-PPOConfig objects
            try:
                cfg = base
                cfg.learning_rate = lr
                return cfg
            except Exception:
                return base

    def to_refine_config(self, pretrain_lr: Optional[float] = None, **overrides: Any) -> Any:
        """Convert to a :class:`rice.algorithms.refine.RefineConfig`.

        ``p = 0`` (always ``s_0 ~ rho``) and ``lam = 0`` (no RND bonus) are the
        faithful realization of "PPO fine-tuning".
        """
        if not _REFINE_AVAILABLE:  # pragma: no cover - guarded import
            raise RuntimeError(
                "rice.algorithms.refine is unavailable; cannot run the PPO fine-tuning baseline."
            )
        n_iter, steps, total = self.budget()
        rnd_config = None
        if RNDConfig is not None:
            try:
                rnd_config = RNDConfig(coef=0.0, normalize_reward=False)
            except Exception:  # pragma: no cover
                rnd_config = None

        kwargs: Dict[str, Any] = dict(
            p=float(self.p),
            lam=float(self.lam),
            n_iterations=n_iter,
            steps_per_iter=steps,
            total_env_steps=total,
            rollin_length=self.rollin_length,
            reset_on_done=bool(self.reset_on_done),
            policy_config=self.policy_config_for(pretrain_lr=pretrain_lr),
            log_every=int(self.log_every),
            eval_every=int(self.eval_every),
            eval_episodes=int(self.eval_episodes),
            device=self.device,
            seed=self.seed,
            verbose=int(self.verbose),
        )
        if rnd_config is not None:
            kwargs["rnd_config"] = rnd_config
        kwargs.update(overrides)
        return RefineConfig(**kwargs)


# --------------------------------------------------------------------------- #
# Multi-seed aggregation (duck-types rice.evaluation.refining_eval.RefiningResult)
# --------------------------------------------------------------------------- #
@dataclass
class FineTuneSummary:
    """Aggregated result of several PPO fine-tuning runs (one per seed)."""

    method: str = METHOD_NAME
    task: Optional[str] = None
    explanation: str = "none"
    seeds: List[int] = field(default_factory=list)
    results: List[Any] = field(default_factory=list)
    final_rewards: List[float] = field(default_factory=list)
    baseline_rewards: List[float] = field(default_factory=list)
    curves: List[np.ndarray] = field(default_factory=list)
    seconds: float = 0.0
    env_steps: int = 0
    config: Any = None
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None

    # -- aggregate statistics ----------------------------------------------
    @property
    def final_reward(self) -> float:
        return _mean(self.final_rewards)

    @property
    def final_std(self) -> float:
        return _std(self.final_rewards)

    @property
    def baseline_reward(self) -> float:
        return _mean(self.baseline_rewards)

    @property
    def baseline_std(self) -> float:
        return _std(self.baseline_rewards)

    @property
    def improvement(self) -> float:
        return float(self.final_reward - self.baseline_reward)

    def mean_curve(self, window: int = 1) -> np.ndarray:
        return _mean_curve(self.curves, window=window)

    def curve_std(self, window: int = 1) -> np.ndarray:
        if not self.curves:
            return np.zeros(0, dtype=np.float32)
        mats = [np.asarray(c, dtype=np.float32) for c in self.curves]
        n = min(len(m) for m in mats)
        stacked = np.stack([m[:n] for m in mats], axis=0)
        return stacked.std(axis=0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "task": self.task,
            "explanation": self.explanation,
            "seeds": list(self.seeds),
            "final_rewards": list(self.final_rewards),
            "baseline_rewards": list(self.baseline_rewards),
            "final_reward": self.final_reward,
            "final_std": self.final_std,
            "baseline_reward": self.baseline_reward,
            "baseline_std": self.baseline_std,
            "improvement": self.improvement,
            "seconds": self.seconds,
            "env_steps": self.env_steps,
            "notes": list(self.notes),
            "error": self.error,
        }


def _mean_curve(curves: Sequence[Any], window: int = 1) -> np.ndarray:
    """Mean over a list of refining curves (variable lengths tolerated)."""
    mats: List[np.ndarray] = []
    for curve in curves:
        if curve is None:
            continue
        arr = np.asarray(curve, dtype=np.float32).reshape(-1)
        if arr.size:
            mats.append(arr)
    if not mats:
        return np.zeros(0, dtype=np.float32)
    n = min(len(m) for m in mats)
    stacked = np.stack([m[:n] for m in mats], axis=0)
    out = stacked.mean(axis=0)
    if window and window > 1 and out.size >= window:
        kernel = np.ones(int(window), dtype=np.float32) / float(window)
        out = np.convolve(out, kernel, mode="valid")
    return out


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
class PPOFineTuner:
    """Run the PPO fine-tuning baseline (lowered LR, continue PPO training).

    Example
    -------
    >>> tuner = PPOFineTuner(env=env, policy=policy)          # doctest: +SKIP
    >>> result = tuner.refine()                                # doctest: +SKIP
    >>> summary = tuner.run(seeds=[0, 1, 2])                   # doctest: +SKIP
    """

    method = METHOD_NAME

    def __init__(
        self,
        env: Any = None,
        policy: Any = None,
        config: Any = None,
        evaluation_env: Any = None,
        state_manager: Any = None,
        rng: Any = None,
        task: Optional[str] = None,
        mask_network: Any = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(config, PPOFineTuneConfig):
            self.config = config.clone(**kwargs)
        else:
            self.config = PPOFineTuneConfig.from_mapping(config, **kwargs)
        if task is not None:
            self.config.task = task
        # The baseline never uses an explanation/mask network, but the argument
        # is accepted so callers can dispatch baselines uniformly.
        self.mask_network = mask_network
        self.env = env
        self.policy = policy
        self.evaluation_env = evaluation_env
        self.state_manager = state_manager
        self.rng = rng
        self._summary: Optional[FineTuneSummary] = None

    # -- lazy construction --------------------------------------------------
    def resolve_task(self) -> str:
        if self.config.task:
            return str(self.config.task)
        env = self.env or self.evaluation_env
        name = getattr(env, "rise_canonical_name", None) if env is not None else None
        return str(name or "Hopper-v3")

    def build_env(self, seed: Optional[int] = None) -> Any:
        """Build the training env (only when the caller did not provide one)."""
        if self.env is not None:
            return self.env
        task = self.resolve_task()
        kwargs = dict(self.config.env_kwargs)
        if seed is not None:
            kwargs.setdefault("seed", seed)
        env_mod = _import_first(
            ("..environments", "rice.environments", "rice.rice.environments"),
            package=_PACKAGE,
        )
        make_env = getattr(env_mod, "make_env", None)
        if make_env is None:  # pragma: no cover - environments missing
            raise RuntimeError("rice.environments.make_env is unavailable.")
        self.env = make_env(task, **kwargs)
        return self.env

    def build_eval_env(self, seed: Optional[int] = None) -> Any:
        if self.evaluation_env is not None:
            return self.evaluation_env
        task = self.resolve_task()
        kwargs = dict(self.config.env_kwargs)
        kwargs.update(self.config.eval_env_kwargs)
        if seed is not None:
            kwargs.setdefault("seed", seed)
        env_mod = _import_first(
            ("..environments", "rice.environments", "rice.rice.environments"),
            package=_PACKAGE,
        )
        make_env = getattr(env_mod, "make_env", None)
        if make_env is None:  # pragma: no cover
            return self.build_env(seed=seed)
        self.evaluation_env = make_env(task, **kwargs)
        return self.evaluation_env

    def build_policy(self, env: Any = None) -> Any:
        """Build the policy network matching the env (mask-net architecture)."""
        if self.policy is not None:
            return self.policy
        env = env if env is not None else self.build_env()
        if self.config.net_arch is not None:
            net_arch = tuple(int(x) for x in self.config.net_arch)
        else:
            net_arch = None
        try:
            from ..algorithms.ppo import ActorCritic  # type: ignore
        except Exception:  # pragma: no cover - repo layout variants
            ActorCritic = getattr(_ppo_mod, "ActorCritic", None)
        if ActorCritic is None:  # pragma: no cover
            raise RuntimeError("rice.algorithms.ppo.ActorCritic is unavailable.")
        if net_arch is not None:
            self.policy = ActorCritic(env.observation_space, env.action_space, net_arch=net_arch)
        else:
            self.policy = ActorCritic(env.observation_space, env.action_space)
        return self.policy

    # -- refinement ---------------------------------------------------------
    def build_refine_config(self, seed: Optional[int] = None, **overrides: Any) -> Any:
        """RefineConfig encoding "PPO fine-tuning" (p=0, lam=0, lowered lr)."""
        cfg = self.config
        base_policy_cfg = cfg.policy_config
        pretrain_lr = _field(base_policy_cfg, "learning_rate", DEFAULT_PRETRAIN_LR)
        kw: Dict[str, Any] = {}
        if seed is not None:
            kw["seed"] = seed
        kw.update(overrides)
        return cfg.to_refine_config(pretrain_lr=pretrain_lr, **kw)

    def _build_rnd(self, env: Any) -> Any:
        """RND placeholder with ``coef = 0`` (disabled exploration bonus)."""
        if make_rnd is None or RNDConfig is None or env is None:
            return None
        try:
            return make_rnd(
                observation_space=getattr(env, "observation_space", None),
                config=RNDConfig(coef=0.0, normalize_reward=False),
            )
        except Exception:  # pragma: no cover - torch missing / odd spaces
            return None

    def baseline_reward(self, seed: Optional[int] = None) -> Optional[float]:
        """Pre-refining evaluation of the warm-start policy ("No Refine")."""
        if evaluate_policy is None or not self.config.measure_baseline:
            return None
        try:
            env = self.build_eval_env(seed=seed)
            policy = self.build_policy()
            stats = evaluate_policy(
                env,
                policy,
                n_episodes=int(self.config.eval_episodes),
                seed=seed,
                deterministic=bool(self.config.deterministic_eval),
                rng=self.rng,
            )
            return float(stats.get("mean", float("nan")))
        except Exception as exc:  # pragma: no cover - defensive
            if self.config.verbose:
                print(f"[ppo_finetune] baseline evaluation failed: {exc}")
            return None

    def refine(self, seed: Optional[int] = None, n_iterations: Optional[int] = None) -> Any:
        """Run one PPO fine-tuning job and return a ``RefineResult``."""
        if not _REFINE_AVAILABLE:  # pragma: no cover - guarded import
            raise RuntimeError(
                "rice.algorithms.refine is unavailable; cannot run PPO fine-tuning."
            )
        env = self.build_env(seed=seed)
        policy = self.build_policy(env)
        cfg = self.build_refine_config(seed=seed)
        if n_iterations is not None:
            cfg = cfg.clone(n_iterations=int(n_iterations))
        if self.config.weights is not None and load_policy_weights is not None:
            try:
                load_policy_weights(policy, self.config.weights)
            except Exception as exc:  # pragma: no cover - defensive
                if self.config.verbose:
                    print(f"[ppo_finetune] could not load weights: {exc}")

        base = self.baseline_reward(seed=seed)
        rnd = self._build_rnd(env)
        refiner = RICERefiner(
            env,
            policy,
            mask_network=None,  # no explanation -> no critical-state resets
            config=cfg,
            rnd=rnd,
            state_manager=self.state_manager,
            evaluation_env=self.build_eval_env(seed=seed),
            rng=self.rng,
        )
        t0 = time.time()
        result = refiner.train()
        elapsed = time.time() - t0
        try:
            result.seconds = float(elapsed)
        except Exception:  # pragma: no cover - frozen dataclass
            pass
        if base is not None and result is not None:
            try:
                result.baseline_reward = float(base)
            except Exception:  # pragma: no cover - dynamic attribute fallback
                pass
        self.policy = getattr(refiner, "policy", policy)
        return result

    # ``train`` is the name used by the RICE refiner, kept as an alias.
    train = refine

    # -- multi-seed ---------------------------------------------------------
    def run(self, seeds: Optional[Sequence[int]] = None, **kwargs: Any) -> FineTuneSummary:
        """Run the baseline over several seeds and aggregate the results."""
        seed_list = [int(s) for s in seeds] if seeds is not None else self.config.seed_list()
        summary = FineTuneSummary(
            method=self.method,
            task=self.resolve_task(),
            explanation="none",
            seeds=seed_list,
            config=self.config,
        )
        if not self.config.final_eval:
            summary.notes.append("final_eval=False: rewards are last-iteration roll-out returns.")

        t0 = time.time()
        for seed in seed_list:
            try:
                result = self.refine(seed=seed, **kwargs)
            except Exception as exc:  # pragma: no cover - keep the sweep alive
                summary.error = repr(exc)
                summary.notes.append(f"seed {seed} failed: {exc!r}")
                if self.config.verbose:
                    print(f"[ppo_finetune] seed {seed} failed: {exc}")
                continue
            summary.results.append(result)
            summary.final_rewards.append(_final_reward_of(result))
            base = getattr(result, "baseline_reward", None)
            if base is None or not np.isfinite(base):
                base = self.baseline_reward(seed=seed)
            summary.baseline_rewards.append(float(base) if base is not None else float("nan"))
            curve = _curve_of(result, window=int(self.config.curve_window))
            if curve is not None:
                summary.curves.append(np.asarray(curve, dtype=np.float32))
            summary.env_steps += int(getattr(result, "env_steps", 0) or 0)
        summary.seconds = float(time.time() - t0)
        self._summary = summary
        return summary

    run_seeds = run

    # -- evaluation / persistence -------------------------------------------
    def evaluate(self, seed: Optional[int] = None, n_episodes: Optional[int] = None) -> Dict[str, float]:
        """Evaluate the (possibly refined) policy."""
        if evaluate_policy is None:  # pragma: no cover
            raise RuntimeError("rice.algorithms.refine.evaluate_policy is unavailable.")
        env = self.build_eval_env(seed=seed)
        policy = self.build_policy()
        return evaluate_policy(
            env,
            policy,
            n_episodes=int(n_episodes or self.config.eval_episodes),
            seed=seed,
            deterministic=bool(self.config.deterministic_eval),
            rng=self.rng,
        )

    def state_dict(self) -> Dict[str, Any]:
        policy = self.policy
        state: Dict[str, Any] = {"config": self.config.to_dict()}
        if policy is not None and hasattr(policy, "state_dict"):
            try:
                state["policy"] = policy.state_dict()
            except Exception:  # pragma: no cover
                pass
        return state

    def save(self, path: str) -> str:  # pragma: no cover - convenience
        try:
            import torch

            torch.save(self.state_dict(), path)
        except Exception:
            import pickle

            with open(path, "wb") as fh:
                pickle.dump(self.state_dict(), fh)
        return str(path)

    # -- introspection ------------------------------------------------------
    @property
    def summary(self) -> Optional[FineTuneSummary]:
        return self._summary

    def describe(self) -> Dict[str, Any]:
        """Human readable description of what this baseline does."""
        cfg = self.config
        return {
            "method": self.method,
            "description": (
                "PPO fine-tuning (Schulman et al. 2017): continue PPO training with a "
                "lowered learning rate; no critical-state roll-in (p=0) and no RND bonus "
                "(lambda=0)."
            ),
            "task": self.resolve_task(),
            "lr_factor": cfg.lr_factor,
            "resolved_learning_rate": cfg.resolved_learning_rate(
                _field(cfg.policy_config, "learning_rate", DEFAULT_PRETRAIN_LR)
            ),
            "n_iterations": cfg.n_iterations,
            "steps_per_iter": cfg.steps_per_iter,
            "total_env_steps": cfg.total_env_steps,
            "p": cfg.p,
            "lam": cfg.lam,
            "seeds": cfg.seed_list(),
        }


# Alias used by some scripts / the package registry.
PPOFineTuneRefiner = PPOFineTuner


def _final_reward_of(result: Any) -> float:
    """Best-effort extraction of the post-refining reward from a RefineResult."""
    if result is None:
        return float("nan")
    for attr in ("final_reward", "final_eval_reward", "mean_episode_return"):
        value = getattr(result, attr, None)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    returns = getattr(result, "episode_returns", None)
    if returns:
        return _mean(returns)
    return float("nan")


def _curve_of(result: Any, window: int = 1) -> Optional[np.ndarray]:
    if result is None:
        return None
    fn = getattr(result, "refining_curve", None)
    if callable(fn):
        try:
            return np.asarray(fn(window=window), dtype=np.float32)
        except Exception:
            try:
                return np.asarray(fn(), dtype=np.float32)
            except Exception:
                pass
    returns = getattr(result, "episode_returns", None)
    if returns:
        return np.asarray(returns, dtype=np.float32)
    return None


# --------------------------------------------------------------------------- #
# Functional entry points (used by rice.baselines / rice.evaluation.refining_eval)
# --------------------------------------------------------------------------- #
def make_ppo_finetuner(
    env: Any = None,
    policy: Any = None,
    config: Any = None,
    **kwargs: Any,
) -> PPOFineTuner:
    """Factory mirroring ``rice.algorithms.refine.make_refiner``."""
    return PPOFineTuner(env=env, policy=policy, config=config, **kwargs)


def ppo_finetune(
    env: Any = None,
    policy: Any = None,
    mask_network: Any = None,
    config: Any = None,
    seeds: Optional[Sequence[int]] = None,
    n_iterations: Optional[int] = None,
    evaluation_env: Any = None,
    state_manager: Any = None,
    rng: Any = None,
    task: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Run the PPO fine-tuning baseline.

    Returns a ``RefineResult`` for a single run and a :class:`FineTuneSummary`
    when several seeds are requested (``seeds`` with more than one entry, or
    ``n_seeds > 1`` combined with ``seeds=None`` **and** an explicit ``seeds``
    argument of length > 1).

    Parameters
    ----------
    env, policy:
        Warm-start environment and pre-trained (bottlenecked) policy.
    config:
        :class:`PPOFineTuneConfig`, a dict, or a config object (e.g. the
        evaluation layer's ``RefiningConfig``).
    seeds:
        Optional explicit seed list; a single-element list runs one job.
    """
    tuner = PPOFineTuner(
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
    if seeds is not None and len(list(seeds)) > 1:
        return tuner.run(seeds=list(seeds), n_iterations=n_iterations)
    seed = None
    if seeds:
        seed = int(list(seeds)[0])
    return tuner.refine(seed=seed, n_iterations=n_iterations)


# Alias
ppo_finetune_baseline = ppo_finetune

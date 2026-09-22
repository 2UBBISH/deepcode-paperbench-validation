"""Hyper-parameter sensitivity study (Experiment V of the RICE paper).

Source: §4.2 Experiment Design (Experiment V) and §4.3 Experiment Results
(Impact of Hyper-parameters).

Paper (verbatim, §4.2):

    "Experiment V. We test the impact of hyper-parameter choices for two
     primary hyper-parameters for refining: p (used to control the mixed
     initial state distribution) and lambda (used to control the exploration
     bonus). For our refining method, we vary p from {0, 0.25, 0.5, 0.75, 1}
     and vary lambda from {0, 0.1, 0.01, 0.001}. ... Additionally, we evaluate
     the choice of alpha for our explanation method (used to control the mask
     ratio for the mask network). Specifically, we vary alpha from
     {0.01, 0.001, 0.0001}."

Paper (verbatim, §4.3, "Impact of Hyper-parameters"):

    "First, p controls the mixing ratio of critical states ... and the initial
     state distribution for refining. The performance is low when p = 0 (all
     starting from the default initial distribution) or p = 1 (all starting
     from the identified critical states). The performance has significant
     improvements when 0 < p < 1, i.e., using a mixed initial state
     distribution. Across all applications, we observe that setting p to 0.25
     or 0.5 is most beneficial. ...
     Second, as long as lambda > 0 (thereby enabling exploration), there is a
     noticeable improvement in performance, highlighting the importance of
     exploration in refining the pre-trained agent. The result is less
     sensitive to the specific value of lambda. In general, a lambda value of
     0.01 yields good performance across all four applications.
     Third, recall that the hyper-parameter alpha is to control the bonus of
     blinding the target agent when training the mask network. We vary alpha
     from {0.01,0.001,0.0001} and find that our explanation method is not that
     sensitive to alpha."

The paper does NOT report the exact numbers underlying the sensitivity figures
(they live in Appendix C.3/C.4 and only as plots); the addendum asks us to
reproduce the *trends*:

    * p = 0 and p = 1 are worse than 0 < p < 1, with p in {0.25, 0.5} best;
    * any lambda > 0 beats lambda = 0 and the method is insensitive to lambda
      (0.01 best, except selfish mining);
    * the explanation quality (fidelity score) is not sensitive to alpha.

This module orchestrates those sweeps by re-using the *same* experiment
machinery as Experiments I-III so that a swept value differs in exactly one
place:

    * ``p`` / ``lambda`` sweeps  ->  ``rice.evaluation.refining_eval``
      (``evaluate_refining`` / ``RefiningConfig``), overriding ``p``/``lam``
      of the refining loop while the explanation is fixed to ours;
    * ``alpha`` sweep           ->  ``rice.evaluation.fidelity_score``
      (``evaluate_explanation`` with a freshly trained mask network whose
      Algorithm-1 bonus coefficient is ``alpha``) and, optionally, also the
      downstream refining performance.

Everything else (number of seeds, per-value iteration budget, evaluation
protocol) is kept identical across sweep values so the comparison is
apples-to-apples.  Defaults that the paper leaves unspecified are documented in
the class docstrings and reported through ``SweepResult.notes``.
"""

from __future__ import annotations

import importlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    # constants
    "DEFAULT_P_VALUES",
    "DEFAULT_LAMBDA_VALUES",
    "DEFAULT_ALPHA_VALUES",
    "DEFAULT_SWEEP_SEEDS",
    "SWEEP_METRICS",
    "PARAM_ALIASES",
    "TABLE3_HYPERPARAMS",
    # containers
    "SweepPoint",
    "SweepCurve",
    "SweepResult",
    "SweepConfig",
    # engines
    "HyperparamSweep",
    "make_sweep",
    # functional entry points
    "sweep_p",
    "sweep_lambda",
    "sweep_alpha",
    "run_sweep",
    "sweep_hyperparameters",
    # helpers
    "expected_best_values",
    "trend_check",
]


# --------------------------------------------------------------------------- #
# Constants taken verbatim from the paper (§4.2 Experiment V)
# --------------------------------------------------------------------------- #

#: ``p`` sweep grid: "{0, 0.25, 0.5, 0.75, 1}" (paper §4.2)
DEFAULT_P_VALUES: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

#: ``lambda`` sweep grid: "{0, 0.1, 0.01, 0.001}" (paper §4.2)
DEFAULT_LAMBDA_VALUES: Tuple[float, ...] = (0.0, 0.1, 0.01, 0.001)

#: ``alpha`` sweep grid: "{0.01, 0.001, 0.0001}" (paper §4.2)
DEFAULT_ALPHA_VALUES: Tuple[float, ...] = (0.01, 0.001, 0.0001)

#: The paper repeats Experiment I with 3 seeds and reports mean +- std (§4.2).
DEFAULT_SWEEP_SEEDS: Tuple[int, ...] = (0, 1, 2)

#: Supported sweep metrics.
SWEEP_METRICS: Dict[str, str] = {
    "final_reward": "final reward of the refined agent (Table 1 metric)",
    "improvement": "final reward minus the no-refine baseline reward",
    "fidelity": "fidelity score of the explanation (§4.1 metric)",
    "area_under_curve": "mean refining reward over the run (sparse-task metric)",
}

#: Friendly alias -> canonical sweep parameter name.
PARAM_ALIASES: Dict[str, str] = {
    "p": "p",
    "beta": "p",
    "reset_probability": "p",
    "reset_prob": "p",
    "mix": "p",
    "lambda": "lambda",
    "lam": "lambda",
    "l": "lambda",
    "coef": "lambda",
    "intrinsic_coef": "lambda",
    "alpha": "alpha",
    "a": "alpha",
    "mask_bonus": "alpha",
    "bonus": "alpha",
}

#: Table 3 (Appendix C.3) hyper-parameters -- the *default centre* of each sweep.
#: NOTE the paper's §C.3 text says alpha = 0.01 while Table 3 lists 0.0001;
#: Table 3 is operative (per the reproduction plan / addendum).
TABLE3_HYPERPARAMS: Dict[str, Dict[str, float]] = {
    "Hopper-v3": {"p": 0.25, "lambda": 0.001, "alpha": 0.0001},
    "Walker2d-v3": {"p": 0.25, "lambda": 0.01, "alpha": 0.0001},
    "Reacher-v2": {"p": 0.50, "lambda": 0.001, "alpha": 0.0001},
    "HalfCheetah-v3": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
    "SelfishMining": {"p": 0.25, "lambda": 0.001, "alpha": 0.0001},
    "CageChallenge2": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
    "Macro-v1": {"p": 0.25, "lambda": 0.01, "alpha": 0.0001},
    # Out of scope (kept so lookups do not KeyError).
    "MalwareMutation": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
    # Sparse variants reuse their dense counterpart's settings.
    "SparseHopper": {"p": 0.25, "lambda": 0.001, "alpha": 0.0001},
    "SparseHalfCheetah": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
    "SparseWalker2d": {"p": 0.25, "lambda": 0.01, "alpha": 0.0001},
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _import_first(candidates: Sequence[str]) -> Any:
    """Import the first importable module among `candidates` (else None)."""
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception:  # pragma: no cover - environment dependent
            continue
    return None


def _get(module: Any, *names: str, default: Any = None) -> Any:
    """Attribute lookup tolerant of naming drift across module layouts."""
    if module is None:
        return default
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    return default


def canonical_param(param: str) -> str:
    """Map a friendly parameter name to its canonical sweep name."""
    key = str(param).strip().lower().replace("-", "_").replace(" ", "_")
    if key in PARAM_ALIASES:
        return PARAM_ALIASES[key]
    if key in ("p", "lambda", "alpha"):
        return key
    raise KeyError(
        f"unknown sweep parameter {param!r}; expected one of "
        f"'p'/'lambda'/'alpha' (aliases: {sorted(PARAM_ALIASES)})"
    )


def default_values_for(param: str) -> Tuple[float, ...]:
    """Return the paper's sweep grid for `param` (§4.2 Experiment V)."""
    name = canonical_param(param)
    if name == "p":
        return DEFAULT_P_VALUES
    if name == "lambda":
        return DEFAULT_LAMBDA_VALUES
    return DEFAULT_ALPHA_VALUES


def table3_for(task: Optional[str]) -> Dict[str, float]:
    """Table 3 p/lambda/alpha for `task` (falls back to the Hopper row)."""
    if task is None:
        return dict(TABLE3_HYPERPARAMS["Hopper-v3"])
    if task in TABLE3_HYPERPARAMS:
        return dict(TABLE3_HYPERPARAMS[task])
    # tolerate aliases such as "hopper" / "Hopper" / "cage"
    lowered = str(task).strip().lower().replace("_", "").replace("-", "")
    for key, value in TABLE3_HYPERPARAMS.items():
        if key.lower().replace("_", "").replace("-", "") == lowered:
            return dict(value)
        if lowered and lowered in key.lower().replace("_", "").replace("-", ""):
            return dict(value)
    return dict(TABLE3_HYPERPARAMS["Hopper-v3"])


def expected_best_values(param: str) -> List[float]:
    """The values the paper's trend statement expects to be (near) best (§4.3).

    * ``p``      -> {0.25, 0.5}
    * ``lambda`` -> {0.01}  (with the caveat "except selfish mining")
    * ``alpha``  -> all values (explicitly *not* sensitive)
    """
    name = canonical_param(param)
    if name == "p":
        return [0.25, 0.5]
    if name == "lambda":
        return [0.01]
    return list(DEFAULT_ALPHA_VALUES)


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #


@dataclass
class SweepPoint:
    """One (parameter value, seed) measurement of a sweep."""

    value: float
    seed: Optional[int] = None
    metric_value: float = float("nan")
    final_reward: float = float("nan")
    baseline_reward: float = float("nan")
    improvement: float = float("nan")
    extra: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "seed": self.seed,
            "metric": self.metric_value,
            "final_reward": self.final_reward,
            "baseline_reward": self.baseline_reward,
            "improvement": self.improvement,
            "error": self.error,
            **({"extra": self.extra} if self.extra else {}),
        }


@dataclass
class SweepCurve:
    """Refining/eval curve attached to one sweep value (averaged over seeds)."""

    value: float
    steps: np.ndarray
    rewards: np.ndarray
    label: str = ""

    @classmethod
    def from_curves(
        cls, value: float, curves: Sequence[Tuple[Sequence[float], Sequence[float]]], label: str = ""
    ) -> "SweepCurve":
        stacks = [np.asarray(r, dtype=np.float64).ravel() for _, r in curves]
        if not stacks:
            return cls(value=value, steps=np.zeros(0), rewards=np.zeros(0), label=label)
        n = min(len(s) for s in stacks)
        return cls(
            value=value,
            steps=np.asarray(list(curves[0][0]), dtype=np.float64).ravel()[:n],
            rewards=np.mean(np.stack([s[:n] for s in stacks], axis=0), axis=0),
            label=label or f"{value:g}",
        )

    def __len__(self) -> int:
        return int(np.asarray(self.rewards).size)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "label": self.label,
            "steps": np.asarray(self.steps).tolist(),
            "rewards": np.asarray(self.rewards).tolist(),
        }


@dataclass
class SweepResult:
    """Aggregated result of one hyper-parameter sweep."""

    task: Optional[str] = None
    param: str = "p"
    values: List[float] = field(default_factory=list)
    means: List[float] = field(default_factory=list)
    stds: List[float] = field(default_factory=list)
    metric: str = "final_reward"
    seeds: List[int] = field(default_factory=list)
    points: List[SweepPoint] = field(default_factory=list)
    curves: Dict[float, SweepCurve] = field(default_factory=dict)
    baseline_rewards: Dict[float, float] = field(default_factory=dict)
    seconds: float = 0.0
    mode: str = "refine"
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None
    reference: Optional[Dict[str, Any]] = None

    # ---------------------------------------------------------------- basics
    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def index_of(self, value: float) -> int:
        arr = np.asarray(self.values, dtype=np.float64)
        return int(np.argmin(np.abs(arr - float(value))))

    def mean(self, value: float) -> float:
        return float(self.means[self.index_of(value)]) if self.values else float("nan")

    def std(self, value: float) -> float:
        return float(self.stds[self.index_of(value)]) if self.values else float("nan")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "param": self.param,
            "metric": self.metric,
            "mode": self.mode,
            "values": list(self.values),
            "means": [float(m) for m in self.means],
            "stds": [float(s) for s in self.stds],
            "seeds": list(self.seeds),
            "seconds": float(self.seconds),
            "baseline_rewards": {float(k): float(v) for k, v in self.baseline_rewards.items()},
            "points": [p.as_dict() for p in self.points],
            "curves": {float(k): c.as_dict() for k, c in self.curves.items()},
            "notes": list(self.notes),
            "error": self.error,
            "reference": self.reference,
        }

    # --------------------------------------------------------------- report
    def best_value(self) -> Optional[float]:
        """Value with the highest sweep metric (larger is better for all metrics)."""
        if not self.values:
            return None
        return float(self.values[int(np.nanargmax(np.asarray(self.means, dtype=np.float64)))])

    def rows(self) -> List[Dict[str, Any]]:
        """CSV-friendly rows (one per swept value)."""
        return [
            {
                "task": self.task,
                "param": self.param,
                "value": v,
                "mean": float(self.means[i]),
                "std": float(self.stds[i]),
                "metric": self.metric,
            }
            for i, v in enumerate(self.values)
        ]

    def table(self, width: int = 10) -> str:
        """Pretty-printed sweep table (mean +- std)."""
        lines = [
            f"Experiment V sweep: task={self.task} param={self.param} "
            f"metric={self.metric} mode={self.mode}",
            "-" * 72,
            f"{'value':>{width}}   {'mean':>{width}}   {'std':>{width}}",
            "-" * 72,
        ]
        for i, v in enumerate(self.values):
            lines.append(
                f"{v:>{width}.4g}   {self.means[i]:>{width}.4f}   {self.stds[i]:>{width}.4f}"
            )
        if self.values:
            lines.append("-" * 72)
            lines.append(f"best {self.param} = {self.best_value():g} (mean {max(self.means):.4f})")
        if self.notes:
            lines.append("")
            lines.extend(f"note: {n}" for n in self.notes)
        return "\n".join(lines)

    def format(self, width: int = 10) -> str:
        return self.table(width=width)

    # ------------------------------------------------------------ trend check
    def trend_check(self) -> Dict[str, Any]:
        """Compare the observed ranking against the paper's qualitative claims."""
        check: Dict[str, Any] = {"param": self.param, "passed": None, "details": {}}
        if not self.values:
            return check
        means = np.asarray(self.means, dtype=np.float64)
        best = self.best_value()

        if self.param == "p":
            interior = [i for i, v in enumerate(self.values) if 0.0 < float(v) < 1.0]
            extreme = [i for i, v in enumerate(self.values) if float(v) in (0.0, 1.0)]
            ok = True
            if interior and extreme:
                ok = float(np.nanmax(means[interior])) > float(np.nanmax(means[extreme]))
            if ok and best is not None:
                ok = best in (0.25, 0.5)
            check["details"] = {
                "best": best,
                "expected_best": expected_best_values(self.param),
                "interior_better_than_extremes": bool(
                    interior
                    and extreme
                    and np.nanmax(means[interior]) > np.nanmax(means[extreme])
                ),
            }
            check["passed"] = bool(ok)

        elif self.param == "lambda":
            zero = [i for i, v in enumerate(self.values) if float(v) == 0.0]
            positive = [i for i, v in enumerate(self.values) if float(v) > 0.0]
            ok = True
            if zero and positive:
                ok = float(np.nanmax(means[positive])) > float(np.nanmax(means[zero]))
            check["details"] = {
                "best": best,
                "expected_best": expected_best_values(self.param),
                "positive_beats_zero": bool(
                    zero and positive and np.nanmax(means[positive]) > np.nanmax(means[zero])
                ),
                "note": "paper: any lambda>0 helps; 0.01 best except selfish mining",
            }
            check["passed"] = bool(ok)

        else:  # alpha -- the paper claims *insensitivity*
            spread = float(np.nanmax(means) - np.nanmin(means))
            scale = float(abs(np.nanmean(means))) + 1e-8
            relative = spread / scale
            check["details"] = {
                "relative_spread": relative,
                "best": best,
                "note": "paper: explanation method is not sensitive to alpha",
            }
            # "insensitive" is judged relative to the seed-level noise, if any.
            noise = float(np.nanmean(self.stds)) if self.stds else 0.0
            check["passed"] = bool(spread <= max(2.0 * noise, 0.25 * scale))
        return check


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class SweepConfig:
    """Configuration of an Experiment V hyper-parameter sweep.

    Parameters left unspecified by the paper default to the values documented in
    the field comments and are echoed in ``SweepResult.notes``.
    """

    task: Optional[str] = None
    param: str = "p"
    values: Optional[Sequence[float]] = None
    metric: str = "final_reward"
    mode: str = "auto"            # "refine" | "fidelity" | "auto"

    # ---- refining sweep (p / lambda) -------------------------------------
    explanation: str = "ours"
    methods: Tuple[str, ...] = ("ours",)
    n_iterations: Optional[int] = None
    steps_per_iter: Optional[int] = None
    total_env_steps: Optional[float] = None
    rollin_length: Optional[int] = None
    n_seeds: int = 3
    seeds: Optional[Sequence[int]] = None
    eval_episodes: int = 5
    deterministic_eval: bool = True
    device: str = "auto"
    verbose: int = 1
    log_every: int = 1
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    weights: Optional[str] = None
    mask_weights: Optional[str] = None
    policy_config: Any = None
    rnd_config: Any = None
    refine_overrides: Dict[str, Any] = field(default_factory=dict)

    # ---- fidelity sweep (alpha) ------------------------------------------
    fidelity_ks: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
    fidelity_trajectories: int = 500
    fidelity_seeds: Optional[Sequence[int]] = None
    d_max: Optional[float] = None
    mask_n_iterations: Optional[int] = None
    mask_total_samples: Optional[float] = None
    measure_time: bool = True

    # ---- output -----------------------------------------------------------
    save_dir: Optional[str] = None
    save_json: bool = True
    plot: bool = False
    plot_dir: Optional[str] = None

    def __post_init__(self) -> None:
        self.param = canonical_param(self.param)
        if self.values is None:
            self.values = default_values_for(self.param)
        self.values = [float(v) for v in self.values]
        if self.seeds is None:
            self.seeds = tuple(int(s) for s in DEFAULT_SWEEP_SEEDS[: max(1, int(self.n_seeds))])
        else:
            self.seeds = tuple(int(s) for s in self.seeds)
        if self.fidelity_seeds is None:
            self.fidelity_seeds = tuple(self.seeds)
        if self.mode == "auto":
            self.mode = "fidelity" if self.param == "alpha" else "refine"

    # ------------------------------------------------------------------ utils
    def clone(self, **overrides: Any) -> "SweepConfig":
        data = dict(self.__dict__)
        data.update(overrides)
        return SweepConfig(**data)

    @classmethod
    def from_mapping(
        cls, mapping: Optional[Dict[str, Any]] = None, **overrides: Any
    ) -> "SweepConfig":
        """Build from a (YAML-parsed) mapping; unknown keys are ignored."""
        data: Dict[str, Any] = {}
        mapping = dict(mapping or {})
        mapping.update(overrides)
        aliases = {
            "lambda": "lam_unused",
            "beta": "p_hint",
            "sweep": "param",
            "hyperparameter": "param",
            "num_seeds": "n_seeds",
            "K": "rollin_length",
            "length": "rollin_length",
            "num_trajectories": "fidelity_trajectories",
            "output_dir": "save_dir",
        }
        for key, value in mapping.items():
            if key in aliases:
                key = aliases[key]
            if key in cls.__dataclass_fields__:
                data[key] = value
        return cls(**data)

    def seed_list(self) -> List[int]:
        return list(self.seeds or DEFAULT_SWEEP_SEEDS)

    def base_hyperparams(self) -> Dict[str, float]:
        """Table 3 centre point for the task (used for the non-swept params)."""
        return table3_for(self.task)


# --------------------------------------------------------------------------- #
# The sweep engine
# --------------------------------------------------------------------------- #


class HyperparamSweep:
    """Runs Experiment V sweeps for one (task, parameter) pair.

    The engine is a thin orchestrator: it fixes everything except the swept
    parameter by delegating to the existing evaluation machinery.

    * ``mode="refine"``   -> ``rice.evaluation.refining_eval`` (Experiments II/III
      machinery).  Each sweep value overrides ``p`` (for the p-sweep) or
      ``lam`` (for the lambda-sweep); the explanation is fixed to "ours".
    * ``mode="fidelity"`` -> ``rice.evaluation.fidelity_score`` (Experiment I
      machinery).  Each sweep value is the Algorithm-1 bonus coefficient
      ``alpha`` of a freshly trained mask network; the metric is the fidelity
      score (§4.1).

    Parameters
    ----------
    env, policy, mask_network:
        Optional pre-built objects.  When omitted they are constructed from
        ``config.task`` exactly like the experiment scripts do.
    config:
        :class:`SweepConfig` (or a plain mapping / keyword overrides).
    run_value_fn:
        Optional callable ``(value, seed) -> SweepPoint`` used as an escape
        hatch (tests, custom metrics, or when the evaluation layer is absent).
    """

    def __init__(
        self,
        env: Any = None,
        policy: Any = None,
        mask_network: Any = None,
        config: Optional[Any] = None,
        evaluation_env: Any = None,
        state_manager: Any = None,
        rng: Any = None,
        run_value_fn: Optional[Callable[[float, Optional[int]], SweepPoint]] = None,
        logger: Any = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = SweepConfig(**kwargs)
        elif isinstance(config, dict):
            config = SweepConfig.from_mapping(config, **kwargs)
        elif kwargs:
            config = config.clone(**kwargs)
        self.config: SweepConfig = config

        self.env = env
        self.policy = policy
        self.mask_network = mask_network
        self.evaluation_env = evaluation_env
        self.state_manager = state_manager
        self.rng = rng
        self.logger = logger
        self._run_value_fn = run_value_fn
        self._built_policy_net_arch: Optional[Sequence[int]] = None

        # fully-qualified module cache
        self._refining_eval = _import_first(
            ["rice.evaluation.refining_eval", "rice.rice.evaluation.refining_eval", "refining_eval"]
        )
        self._fidelity = _import_first(
            ["rice.evaluation.fidelity_score", "rice.rice.evaluation.fidelity_score", "fidelity_score"]
        )
        self._mask_net = _import_first(
            ["rice.algorithms.mask_network", "rice.rice.algorithms.mask_network", "mask_network"]
        )
        self._statemask = _import_first(
            ["rice.explanation.statemask_adapter", "rice.rice.explanation.statemask_adapter"]
        )

    # ------------------------------------------------------------- utilities
    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f"[hyperparam_sweep] {message}", flush=True)

    def _note(self, result: SweepResult, message: str) -> None:
        result.notes.append(message)
        if self.config.verbose and self.config.verbose > 1:
            print(f"[hyperparam_sweep] note: {message}", flush=True)

    # --------------------------------------------------------- env / policy
    def build_env(self, seed: Optional[int] = None) -> Any:
        """Build (or return) the environment for the sweep task."""
        if self.env is not None:
            return self.env
        envs = _import_first(["rice.environments", "rice.rice.environments", "environments"])
        make_env = _get(envs, "make_env", default=None)
        if make_env is None:
            raise ImportError("rice.environments.make_env unavailable; pass `env=` explicitly")
        kwargs = dict(self.config.env_kwargs)
        if seed is not None:
            kwargs.setdefault("seed", int(seed))
        self.env = make_env(self.config.task, **kwargs)
        return self.env

    def build_policy(self, env: Any = None) -> Any:
        """Build (or return) the warm-start policy for the sweep task."""
        if self.policy is not None:
            return self.policy
        refine_mod = _import_first(
            ["rice.algorithms.refine", "rice.rice.algorithms.refine", "refine"]
        )
        env = env if env is not None else self.build_env()
        if self.config.weights:
            loader = _get(refine_mod, "load_policy_weights", default=None)
            make_policy = _get(refine_mod, "make_policy_for_task", default=None)
            policy = None
            if make_policy is not None:
                try:
                    policy = make_policy(env, device=self.config.device)
                except Exception:
                    policy = None
            if policy is None:
                ppo_mod = _import_first(
                    ["rice.algorithms.ppo", "rice.rice.algorithms.ppo", "ppo"]
                )
                ActorCritic = _get(ppo_mod, "ActorCritic", default=None)
                if ActorCritic is None:
                    raise ImportError("ActorCritic unavailable; pass `policy=` explicitly")
                policy = ActorCritic(
                    env.observation_space, env.action_space, device=self.config.device
                )
            if loader is not None:
                policy = loader(policy, self.config.weights)
            self.policy = policy
        if self.policy is None:
            # Delegate to the refining evaluator's policy builder when possible.
            build = _get(
                self._refining_eval,
                "make_policy_for_task",
                "build_policy_for_task",
                default=None,
            )
            if build is not None:
                try:
                    self.policy = build(env, device=self.config.device)
                except Exception:
                    self.policy = None
        if self.policy is None:
            ppo_mod = _import_first(["rice.algorithms.ppo", "rice.rice.algorithms.ppo", "ppo"])
            ActorCritic = _get(ppo_mod, "ActorCritic", default=None)
            if ActorCritic is None:
                raise ImportError("no ActorCritic / policy builder available; pass `policy=`")
            self.policy = ActorCritic(
                env.observation_space, env.action_space, device=self.config.device
            )
        return self.policy

    # ------------------------------------------------------------ sweep core
    def run(self, save: Optional[bool] = None) -> SweepResult:
        """Execute the whole sweep and return a :class:`SweepResult`."""
        param = self.config.param
        if param == "alpha" and self.config.mode == "fidelity":
            return self.sweep_alpha()
        if param == "p":
            return self.sweep_p()
        if param == "lambda":
            return self.sweep_lambda()
        return self.sweep_alpha()

    # --- p -----------------------------------------------------------------
    def sweep_p(self, values: Optional[Sequence[float]] = None) -> SweepResult:
        """Sweep the mixed-initial-distribution ratio ``p`` (§4.2 Experiment V)."""
        values = [float(v) for v in (values if values is not None else self.config.values)]
        return self._sweep_refining("p", values)

    def sweep_lambda(self, values: Optional[Sequence[float]] = None) -> SweepResult:
        """Sweep the RND bonus coefficient ``lambda`` (§4.2 Experiment V)."""
        values = [float(v) for v in (values if values is not None else self.config.values)]
        return self._sweep_refining("lambda", values)

    # --- alpha -------------------------------------------------------------
    def sweep_alpha(self, values: Optional[Sequence[float]] = None) -> SweepResult:
        """Sweep the mask-network blinding bonus ``alpha`` (§4.2 Experiment V).

        The paper measures the *sensitivity of the explanation method* to
        ``alpha``; the natural metric is therefore the Experiment I fidelity
        score of a mask network trained with that ``alpha``.  Setting
        ``config.mode="refine"`` instead measures the downstream refining
        reward after the same refining pipeline (metric must then be
        "final_reward"/"improvement").
        """
        values = [float(v) for v in (values if values is not None else self.config.values)]
        if self.config.mode == "refine":
            result = self._sweep_refining("alpha", values)
        else:
            result = self._sweep_fidelity_alpha(values)
        return result

    # ------------------------------------------------------ refining sweeps
    def _sweep_refining(self, param: str, values: Sequence[float]) -> SweepResult:
        notes: List[str] = []
        started = time.time()
        result = SweepResult(
            task=self.config.task,
            param=param,
            values=list(values),
            metric=self.config.metric,
            seeds=self.config.seed_list(),
            mode="refine",
            notes=notes,
        )
        base = self.config.base_hyperparams()
        center = float(base.get("lambda" if param == "lambda" else param, 0.0)) or None

        # Build env / policy / mask once so every swept value sees the same
        # warm-start agent and (for p/lambda sweeps) the same explanation.
        try:
            env = self.build_env(seed=self.config.seed_list()[0])
            policy = self.build_policy(env)
        except Exception as exc:  # pragma: no cover - environment dependent
            result.error = f"failed to build env/policy: {exc}"
            self._log(result.error)
            return self._finalize(result, started)

        evaluator = self._make_refining_evaluator(env, policy)
        if evaluator is None:
            result.notes.append(
                "rice.evaluation.refining_eval unavailable: falling back to a local "
                "refining loop via rice.algorithms.refine.refine_policy"
            )

        curves: Dict[float, SweepCurve] = {}
        baseline_rewards: Dict[float, float] = {}

        for value in values:
            self._log(f"sweeping {param}={value:g}")
            seed_points: List[SweepPoint] = []
            curve_pairs: List[Tuple[Sequence[float], Sequence[float]]] = []
            for seed in self.config.seed_list():
                point, curve, baseline = self._run_refine_point(
                    param, value, seed, env, policy, evaluator, center
                )
                seed_points.append(point)
                result.points.append(point)
                if curve is not None:
                    curve_pairs.append(curve)
                if baseline is not None and not np.isnan(baseline):
                    baseline_rewards.setdefault(value, float(baseline))
            metric_values = [p.metric_value for p in seed_points if not np.isnan(p.metric_value)]
            if not metric_values:
                metric_values = [p.final_reward for p in seed_points]
            finite = [m for m in metric_values if np.isfinite(m)]
            result.means.append(float(np.mean(finite)) if finite else float("nan"))
            result.stds.append(float(np.std(finite)) if finite else float("nan"))
            if curve_pairs:
                curves[value] = SweepCurve.from_curves(value, curve_pairs, label=f"{value:g}")

        result.curves = curves
        result.baseline_rewards = baseline_rewards
        result.reference = self._reference_for(param)
        result.trend_check = None  # not part of the dataclass; see `.trend`
        self._annotate_trend(result)
        return self._finalize(result, started)

    def _make_refining_evaluator(self, env: Any, policy: Any) -> Any:
        cls = _get(self._refining_eval, "RefiningEvaluator", default=None)
        if cls is None:
            return None
        try:
            return cls(
                env=env,
                policy=policy,
                mask_network=self.mask_network,
                task=self.config.task,
                evaluation_env=self.evaluation_env,
                state_manager=self.state_manager,
                rng=self.rng,
                seed=self.config.seed_list()[0],
            )
        except Exception:
            return None

    def _refine_config_kwargs(self, param: str, value: float, seed: int) -> Dict[str, Any]:
        """Per-value overrides of the shared refining configuration."""
        base = self.config.base_hyperparams()
        kwargs: Dict[str, Any] = {
            "task": self.config.task,
            "method": "ours",
            "explanation": self.config.explanation,
            "p": float(base.get("p", 0.25)),
            "lam": float(base.get("lambda", 0.01)),
            "alpha": float(base.get("alpha", 1e-4)),
            "n_seeds": 1,
            "seeds": (int(seed),),
            "seed": int(seed),
            "eval_episodes": int(self.config.eval_episodes),
            "deterministic_eval": bool(self.config.deterministic_eval),
            "device": self.config.device,
            "verbose": int(self.config.verbose),
            "log_every": int(self.config.log_every),
            "env_kwargs": dict(self.config.env_kwargs),
            "weights": self.config.weights,
        }
        if self.config.n_iterations is not None:
            kwargs["n_iterations"] = int(self.config.n_iterations)
        if self.config.steps_per_iter is not None:
            kwargs["steps_per_iter"] = int(self.config.steps_per_iter)
        if self.config.total_env_steps is not None:
            kwargs["total_env_steps"] = float(self.config.total_env_steps)
        if self.config.rollin_length is not None:
            kwargs["rollin_length"] = int(self.config.rollin_length)
        # the swept parameter itself
        if param == "p":
            kwargs["p"] = float(value)
        elif param == "lambda":
            kwargs["lam"] = float(value)
        elif param == "alpha":
            kwargs["alpha"] = float(value)
            kwargs["mask_weights"] = self.config.mask_weights
        kwargs.update(self.config.refine_overrides)
        return kwargs

    def _run_refine_point(
        self,
        param: str,
        value: float,
        seed: int,
        env: Any,
        policy: Any,
        evaluator: Any,
        center: Optional[float],
    ) -> Tuple[SweepPoint, Optional[Tuple[Sequence[float], Sequence[float]]], Optional[float]]:
        """Run one (value, seed) refining measurement."""
        if self._run_value_fn is not None:
            point = self._run_value_fn(value, seed)
            return point, None, None

        point = SweepPoint(value=value, seed=seed)
        curve: Optional[Tuple[Sequence[float], Sequence[float]]] = None
        baseline: Optional[float] = None

        kwargs = self._refine_config_kwargs(param, value, seed)
        try:
            if evaluator is not None:
                # Preferred path: reuse the Experiment II/III machinery.
                res = self._evaluate_with_evaluator(evaluator, kwargs)
            else:
                res = self._evaluate_with_function(kwargs, env, policy)
            if res is None:
                raise RuntimeError("no refining result produced")

            point.final_reward = float(_get(res, "final_reward", default=float("nan")))
            point.baseline_reward = float(_get(res, "baseline_reward", default=float("nan")))
            point.improvement = float(_get(res, "improvement", default=float("nan")))
            point.metric_value = self._metric_from_result(res, point)
            baseline = point.baseline_reward
            steps, rewards = self._curve_from_result(res)
            if rewards is not None and len(rewards) > 0:
                curve = (steps, rewards)
            point.extra = {
                "env_steps": float(_get(res, "env_steps", default=0.0) or 0.0),
                "seconds": float(_get(res, "seconds", default=0.0) or 0.0),
            }
        except Exception as exc:  # pragma: no cover - environment dependent
            point.error = f"{type(exc).__name__}: {exc}"
            self._log(f"  {param}={value:g} seed={seed} failed: {point.error}")
        return point, curve, baseline

    def _evaluate_with_evaluator(self, evaluator: Any, kwargs: Dict[str, Any]) -> Any:
        """Call ``RefiningEvaluator.run_single_seed`` or ``evaluate_refining``."""
        run_single = getattr(evaluator, "run_single_seed", None)
        if callable(run_single):
            try:
                return run_single(kwargs["seed"], kwargs["method"], kwargs["explanation"])
            except TypeError:
                pass
        run = getattr(evaluator, "run", None)
        if callable(run):
            try:
                out = run(kwargs["method"], kwargs["explanation"], [kwargs["seed"]])
                return self._unwrap_run(out)
            except Exception:
                pass
        # Fall back to configuring the instance directly.
        build_cfg = getattr(evaluator, "build_refine_config", None)
        evaluate = getattr(evaluator, "evaluate", None)
        if callable(build_cfg) and callable(evaluate):
            cfg = build_cfg(seed=kwargs["seed"], **kwargs)
            out = evaluate(cfg)
            return self._unwrap_run(out)
        ctx = _get(self._refining_eval, "evaluate_refining", default=None)
        if ctx is not None:
            return ctx(**kwargs)
        raise RuntimeError("RefiningEvaluator exposes no usable entry point")

    def _evaluate_with_function(
        self, kwargs: Dict[str, Any], env: Any, policy: Any
    ) -> Any:
        """Direct ``refine_policy`` fallback (no evaluation layer required)."""
        fn = _get(self._refining_eval, "evaluate_refining", default=None)
        if fn is not None:
            return fn(env=env, policy=policy, **kwargs)
        refine_mod = _import_first(
            ["rice.algorithms.refine", "rice.rice.algorithms.refine", "refine"]
        )
        refine_policy = _get(refine_mod, "refine_policy", default=None)
        if refine_policy is None:
            raise ImportError("rice.algorithms.refine.refine_policy unavailable")
        cfg_cls = _get(refine_mod, "RefineConfig", default=None)
        cfg = None
        if cfg_cls is not None:
            try:
                cfg = cfg_cls.from_mapping(kwargs)
            except Exception:
                try:
                    cfg = cfg_cls(**{k: v for k, v in kwargs.items()
                                     if k in cfg_cls.__dataclass_fields__})
                except Exception:
                    cfg = None
        return refine_policy(env=env, policy=policy, config=cfg, **(
            {} if cfg is not None else kwargs
        ))

    @staticmethod
    def _unwrap_run(out: Any) -> Any:
        """Extract a single-seed result from a multi-seed aggregate if needed."""
        if out is None:
            return None
        results = _get(out, "results", default=None)
        if isinstance(results, (list, tuple)) and results:
            return results[0]
        return out

    def _metric_from_result(self, res: Any, point: SweepPoint) -> float:
        metric = self.config.metric
        if metric == "fidelity":
            return float(_get(res, "score", "fidelity", default=point.final_reward) or 0.0)
        if metric == "improvement":
            value = _get(res, "improvement", default=None)
            if value is None or not np.isfinite(float(value)):
                value = point.final_reward - point.baseline_reward
            return float(value)
        if metric == "area_under_curve":
            _, rewards = self._curve_from_result(res)
            if rewards is None or len(rewards) == 0:
                return point.final_reward
            return float(np.mean(np.asarray(rewards, dtype=np.float64)))
        # default: final reward of the refined agent (Table 1 metric)
        return point.final_reward

    @staticmethod
    def _curve_from_result(res: Any) -> Tuple[Sequence[float], Optional[Sequence[float]]]:
        fn = getattr(res, "refining_curve", None)
        if callable(fn):
            for window in (1, 5):
                try:
                    curve = fn(window=window)
                except TypeError:
                    try:
                        curve = fn(window)
                    except Exception:
                        curve = None
                except Exception:
                    curve = None
                if curve is not None:
                    rewards = _get(curve, "rewards", default=None)
                    steps = _get(curve, "steps", default=None)
                    if rewards is not None:
                        if steps is None:
                            steps = np.arange(len(rewards))
                        return list(steps), list(rewards)
        rewards = _get(res, "episode_returns", default=None)
        if rewards is not None and len(rewards) > 0:
            arr = np.asarray(rewards, dtype=np.float64).ravel()
            return list(np.arange(arr.size)), list(arr)
        return [], None

    # ------------------------------------------------------- fidelity sweeps
    def _sweep_fidelity_alpha(self, values: Sequence[float]) -> SweepResult:
        """Measure the fidelity score of mask networks trained with each alpha."""
        started = time.time()
        result = SweepResult(
            task=self.config.task,
            param="alpha",
            values=list(values),
            metric="fidelity",
            seeds=list(self.config.fidelity_seeds or self.config.seed_list()),
            mode="fidelity",
            notes=[],
        )
        train_fn = _get(self._statemask, "train_statemask", default=None)
        eval_fn = _get(self._fidelity, "evaluate_explanation", default=None)
        if train_fn is None or eval_fn is None:
            result.error = (
                "alpha sweep requires "
                "rice.explanation.statemask_adapter.train_statemask and "
                "rice.evaluation.fidelity_score.evaluate_explanation"
            )
            self._log(result.error)
            return self._finalize(result, started)

        try:
            env = self.build_env(seed=result.seeds[0])
            policy = self.build_policy(env)
        except Exception as exc:  # pragma: no cover
            result.error = f"failed to build env/policy: {exc}"
            self._log(result.error)
            return self._finalize(result, started)

        for value in values:
            self._log(f"sweeping alpha={value:g} (fidelity)")
            point_values: List[float] = []
            for seed in result.seeds:
                point = SweepPoint(value=value, seed=seed)
                t0 = time.time()
                try:
                    explanation = train_fn(
                        env,
                        policy,
                        alpha=float(value),
                        seed=int(seed),
                        n_iterations=self.config.mask_n_iterations,
                        total_samples=self.config.mask_total_samples,
                        net_arch=self.config.refine_overrides.get("net_arch"),
                        verbose=max(0, int(self.config.verbose) - 1),
                    )
                    mask_net = _get(explanation, "mask_network", "model", default=explanation)
                    fid_cfg = None
                    cfg_cls = _get(self._fidelity, "FidelityConfig", default=None)
                    if cfg_cls is not None:
                        try:
                            fid_cfg = cfg_cls(
                                ks=tuple(self.config.fidelity_ks),
                                num_trajectories=int(self.config.fidelity_trajectories),
                                seeds=list(result.seeds),
                                d_max=self.config.d_max,
                                seed=int(seed),
                                verbose=max(0, int(self.config.verbose) - 1),
                            )
                        except Exception:
                            fid_cfg = None
                    fid = eval_fn(
                        env,
                        policy,
                        mask_net,
                        config=fid_cfg,
                        d_max=self.config.d_max,
                        method="ours",
                    )
                    score = self._fidelity_score_of(fid)
                    point.metric_value = score
                    point.final_reward = score
                    point.baseline_reward = float("nan")
                    point.improvement = score
                    point.extra = {"train_seconds": time.time() - t0}
                    point_values.append(score)
                except Exception as exc:  # pragma: no cover
                    point.error = f"{type(exc).__name__}: {exc}"
                    self._log(f"  alpha={value:g} seed={seed} failed: {point.error}")
                result.points.append(point)
            finite = [v for v in point_values if np.isfinite(v)]
            result.means.append(float(np.mean(finite)) if finite else float("nan"))
            result.stds.append(float(np.std(finite)) if finite else float("nan"))

        result.reference = self._reference_for("alpha")
        self._annotate_trend(result)
        return self._finalize(result, started)

    @staticmethod
    def _fidelity_score_of(fid: Any) -> float:
        """Aggregate the ks-level fidelity scores of a FidelityResult."""
        means = _get(fid, "means", default=None)
        if means is not None and len(means) > 0:
            arr = np.asarray(means, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            return float(np.mean(finite)) if finite.size else float("nan")
        score = _get(fid, "score", default=None)
        if callable(score):
            try:
                return float(np.mean([score(k) for k in (0.1, 0.2, 0.3, 0.4)]))
            except Exception:
                return float("nan")
        val = _get(fid, "final_reward", default=None)
        return float(val) if val is not None else float("nan")

    # --------------------------------------------------------------- reports
    @staticmethod
    def _reference_for(param: str) -> Dict[str, Any]:
        """Qualitative expectations from §4.3 (trends, not numbers)."""
        if param == "p":
            return {
                "statement": (
                    "p=0 and p=1 are worse than 0<p<1; p=0.25 or 0.5 most beneficial"
                ),
                "expected_best": [0.25, 0.5],
                "bad_values": [0.0, 1.0],
            }
        if param == "lambda":
            return {
                "statement": (
                    "any lambda>0 beats lambda=0; insensitive to lambda, 0.01 best "
                    "(except selfish mining)"
                ),
                "expected_best": [0.01],
                "bad_values": [0.0],
            }
        return {
            "statement": "the explanation method is not sensitive to alpha",
            "expected_best": list(DEFAULT_ALPHA_VALUES),
            "bad_values": [],
        }

    @staticmethod
    def _annotate_trend(result: SweepResult) -> None:
        trend = result.trend_check()
        result.notes.append(f"trend_check[{result.param}]: {json.dumps(trend, default=str)}")

    def _finalize(self, result: SweepResult, started: float) -> SweepResult:
        result.seconds = float(time.time() - started)
        result.notes.append(
            "deviations: refining budget per sweep value, eval_episodes and the "
            "per-value seed count are not specified by the paper and default to "
            "SweepConfig values"
        )
        if self.config.save_dir:
            try:
                self.save(result, self.config.save_dir)
            except Exception as exc:  # pragma: no cover
                result.notes.append(f"failed to save sweep results: {exc}")
        if self.config.plot:
            try:
                self.plot(result)
            except Exception as exc:  # pragma: no cover
                result.notes.append(f"failed to plot sweep results: {exc}")
        if self.config.verbose:
            print(result.table())
        return result

    # ------------------------------------------------------------- artifacts
    def save(self, result: SweepResult, out_dir: Optional[str] = None) -> str:
        """Persist the sweep as JSON (and CSV) under ``out_dir``."""
        out_dir = out_dir or self.config.save_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        stem = f"sweep_{result.param}_{result.task or 'task'}".replace(" ", "_")
        path = os.path.join(out_dir, f"{stem}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(result.as_dict(), handle, indent=2, default=str)
        if self.config.save_json is not False:
            cfg_path = os.path.join(out_dir, f"{stem}_config.json")
            with open(cfg_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {k: v for k, v in self.config.__dict__.items()
                     if isinstance(v, (int, float, str, bool, type(None), list, tuple))},
                    handle,
                    indent=2,
                    default=str,
                )
        csv_path = os.path.join(out_dir, f"{stem}.csv")
        with open(csv_path, "w", encoding="utf-8") as handle:
            handle.write("task,param,value,mean,std,metric\n")
            for row in result.rows():
                handle.write(
                    f"{row['task']},{row['param']},{row['value']},"
                    f"{row['mean']},{row['std']},{row['metric']}\n"
                )
        return path

    def plot(self, result: SweepResult, out_dir: Optional[str] = None) -> Optional[str]:
        """Plot the sweep (Figure 7/8/9 style) and save it as PNG."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:  # pragma: no cover - matplotlib optional
            return None
        out_dir = out_dir or self.config.plot_dir or self.config.save_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)

        labels = {
            "final_reward": "final reward",
            "improvement": "reward improvement",
            "fidelity": "fidelity score",
            "area_under_curve": "mean reward over run",
        }
        fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
        # refining curves, one per sweep value
        for value, curve in result.curves.items():
            if len(curve) == 0:
                continue
            ax.plot(curve.steps, curve.rewards, label=f"{result.param}={value:g}")
        if result.curves:
            ax.set_xlabel("environment steps")
            ax.set_ylabel("episode return")
            ax.legend(fontsize=7)
            fig.tight_layout()
            p1 = os.path.join(
                out_dir, f"sweep_curves_{result.param}_{result.task or 'task'}.png"
            )
            fig.savefig(p1, dpi=150)
        ax.clear()
        ax.errorbar(
            result.values,
            result.means,
            yerr=result.stds,
            marker="o",
            capsize=3,
        )
        ax.set_xlabel(result.param)
        ax.set_ylabel(labels.get(result.metric, result.metric))
        if result.param == "p":
            ax.axvspan(0.25, 0.5, alpha=0.08, color="green")
        if result.param == "lambda":
            ax.set_xscale("symlog", linthresh=1e-3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"sweep_{result.param}_{result.task or 'task'}.png")
        fig.savefig(path, dpi=150)
        try:
            plt.close(fig)
        except Exception:  # pragma: no cover
            pass
        return path


# --------------------------------------------------------------------------- #
# Functional entry points
# --------------------------------------------------------------------------- #


def make_sweep(
    env: Any = None,
    policy: Any = None,
    mask_network: Any = None,
    config: Optional[Any] = None,
    **kwargs: Any,
) -> HyperparamSweep:
    """Factory mirroring ``rice.evaluation``'s builder conventions."""
    return HyperparamSweep(
        env=env, policy=policy, mask_network=mask_network, config=config, **kwargs
    )


def _normalize_task(task: Any) -> Optional[str]:
    if task is None:
        return None
    return str(task)


def _run_single_param(
    param: str,
    task: Optional[str] = None,
    values: Optional[Sequence[float]] = None,
    env: Any = None,
    policy: Any = None,
    mask_network: Any = None,
    config: Optional[Any] = None,
    **kwargs: Any,
) -> SweepResult:
    """Shared implementation of ``sweep_p`` / ``sweep_lambda`` / ``sweep_alpha``."""
    cfg = config
    if cfg is None:
        cfg = SweepConfig(task=_normalize_task(task), param=param, values=values, **kwargs)
    elif isinstance(cfg, dict):
        merged = dict(cfg)
        merged.setdefault("param", param)
        if task is not None:
            merged.setdefault("task", _normalize_task(task))
        if values is not None:
            merged["values"] = list(values)
        merged.update(kwargs)
        cfg = SweepConfig.from_mapping(merged)
    else:
        overrides = dict(kwargs)
        overrides["param"] = param
        if task is not None:
            overrides.setdefault("task", _normalize_task(task))
        if values is not None:
            overrides["values"] = list(values)
        cfg = cfg.clone(**overrides)
    sweep = HyperparamSweep(env=env, policy=policy, mask_network=mask_network, config=cfg)
    if param == "p":
        return sweep.sweep_p()
    if param == "lambda":
        return sweep.sweep_lambda()
    return sweep.sweep_alpha()


def sweep_p(
    task: Optional[str] = None,
    values: Sequence[float] = DEFAULT_P_VALUES,
    **kwargs: Any,
) -> SweepResult:
    """Sweep ``p`` over {0, 0.25, 0.5, 0.75, 1} (§4.2 Experiment V)."""
    return _run_single_param("p", task=task, values=values, **kwargs)


def sweep_lambda(
    task: Optional[str] = None,
    values: Sequence[float] = DEFAULT_LAMBDA_VALUES,
    **kwargs: Any,
) -> SweepResult:
    """Sweep ``lambda`` over {0, 0.1, 0.01, 0.001} (§4.2 Experiment V)."""
    return _run_single_param("lambda", task=task, values=values, **kwargs)


def sweep_alpha(
    task: Optional[str] = None,
    values: Sequence[float] = DEFAULT_ALPHA_VALUES,
    **kwargs: Any,
) -> SweepResult:
    """Sweep ``alpha`` over {0.01, 0.001, 0.0001} (§4.2 Experiment V)."""
    return _run_single_param("alpha", task=task, values=values, **kwargs)


def run_sweep(
    param: str = "p",
    task: Optional[str] = None,
    values: Optional[Sequence[float]] = None,
    config: Optional[Any] = None,
    **kwargs: Any,
) -> SweepResult:
    """Generic entry point: sweep any of "p" / "lambda" / "alpha"."""
    name = canonical_param(param)
    values = values if values is not None else default_values_for(name)
    return _run_single_param(name, task=task, values=values, config=config, **kwargs)


#: Friendly alias used by the experiment scripts.
sweep_hyperparameters = run_sweep


def trend_check(result: SweepResult, expected: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Convenience wrapper around :meth:`SweepResult.trend_check`.

    ``expected`` (optional) overrides the paper's expected-best values used to
    judge the observed ranking.
    """
    out = result.trend_check()
    if expected is not None:
        best = result.best_value()
        out["details"]["expected_best"] = list(expected)
        out["passed"] = bool(best is not None and float(best) in {float(e) for e in expected})
    return out


# Convenience alias so external code can call ``sweep.trend_check(result)``.
HyperparamSweep.trend_check = staticmethod(trend_check)  # type: ignore[attr-defined]

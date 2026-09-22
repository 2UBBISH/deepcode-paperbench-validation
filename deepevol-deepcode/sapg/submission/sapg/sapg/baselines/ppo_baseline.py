"""Vanilla PPO baseline for the SAPG paper (Section 5.2).

The paper's baseline description is short but precise:

    "PPO (Proximal Policy Optimization) (Schulman et al., 2017): In our setting, we
     just increase the data throughput for PPO by increasing the batch size
     proportionately to the number of environments. In particular, we see over two
     orders of magnitude increase in the number of environments (from 128 to 24576)."

So the vanilla PPO baseline is:

* **one** policy (as opposed to SAPG's / DexPBT's ``M = 6`` policies),
* trained on all ``N`` environments,
* with the minibatch size scaled proportionally to ``N``
  (``minibatch_size = N * minibatch_size_multiplier``, as implemented by
  :class:`sapg.algorithms.ppo.PPOTrainer`),
* collecting ``horizon_length = 16`` steps of experience per environment before
  every PPO update,
* a **recurrent** policy for the AllegroKuka tasks and an **MLP** policy for the
  ShadowHand / AllegroHand tasks (Section 5.2),
* run for 5 seeds, reporting the mean and the shaded standard-error band

      y(t) = (1/n) * sum_i y_i(t),
      band(t) = (2 / sqrt(n)) * sum_i (y(t) - y_i(t))^2

  measured against the number of samples collected.

This module also implements the PPO *saturation* study used to motivate SAPG
(Figure 2 concept): vanilla PPO is run at increasing numbers of environments
(128 -> 24576) with a batch size proportional to the environment count, and the
asymptotic performance is observed to saturate beyond roughly 10k environments.

Everything heavy (torch / IsaacGym) is imported lazily so this module can be
imported and unit-tested on a CPU-only machine.
"""

from __future__ import annotations

import copy
import math
import os
import random
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..utils.config import NUM_POLICIES, TOTAL_ENVS, SAPGConfig, build_config

__all__ = [
    "PPO_METHODS",
    "MIN_ENVS_SATURATION_SWEEP",
    "DEFAULT_ENV_SWEEP",
    "PPOBaselineTrainer",
    "PPOBaselineResult",
    "make_ppo_config",
    "make_ppo_policy",
    "make_ppo_env",
    "set_seed",
    "train_ppo_baseline",
    "run_ppo_seeds",
    "aggregate_seed_histories",
    "paper_standard_error",
    "ppo_saturation_sweep",
    "saturation_summary",
]

# Method strings that dispatch to this baseline (see ``sapg.train``).
PPO_METHODS: Tuple[str, ...] = ("ppo", "vanilla_ppo", "ppo_baseline", "ppo-baseline")

# The paper observes PPO's asymptotic performance saturating beyond ~10k envs; we
# sweep from the stated 128 up to the full N = 24576.
DEFAULT_ENV_SWEEP: Tuple[int, ...] = (
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    12288,
    16384,
    24576,
)
MIN_ENVS_SATURATION_SWEEP: int = 128


# --------------------------------------------------------------------------------------
# Configuration helpers
# --------------------------------------------------------------------------------------
def make_ppo_config(
    config: Optional[Any] = None,
    num_envs: Optional[int] = None,
    seed: Optional[int] = None,
    task: str = "regrasping",
    **overrides: Any,
) -> SAPGConfig:
    """Build a :class:`SAPGConfig` configured for the vanilla PPO baseline.

    The returned config describes a *single*-policy PPO run (``num_policies = 1``,
    ``phi_dim = 0``) whose batch size grows proportionally with the number of
    environments, matching Section 5.2.

    Args:
        config: an existing :class:`SAPGConfig` (or a mapping) to derive from.
        num_envs: override for the number of parallel environments (``N``).
        seed: explicit seed override.
        task: task name used when no ``config`` is supplied.
        **overrides: additional :class:`SAPGConfig` fields to override.

    Returns:
        A :class:`SAPGConfig` with ``method`` set to ``"ppo"``.
    """
    if config is None:
        base = build_config(task)
    elif isinstance(config, SAPGConfig):
        base = config
    elif isinstance(config, dict):
        base = SAPGConfig.from_dict(config)
    else:  # duck-typed config object (e.g. one of the task wrappers' dataclasses)
        try:
            base = SAPGConfig.from_dict(_config_to_dict(config))
        except Exception:
            base = build_config(getattr(config, "task", task))

    fields: Dict[str, Any] = {
        "method": "ppo",
        # One policy only: no splitting, no latents, no off-policy aggregation.
        "num_policies": 1,
        "leader_index": 1,
        "aggregation": "none",
        "off_policy_weight": 0.0,
        "subsample_off_policy": False,
        # PPO has no phi-conditioned diversity and no entropy exploration variant
        # (Section 5.2 reserves sigma tuning for SAPG).
        "phi_dim": 0,
        "entropy_coefficient": 0.0,
        "per_block_sigma": False,
        "random_phi": False,
    }
    if num_envs is not None:
        fields["num_envs"] = int(num_envs)
    if seed is not None:
        fields["seed"] = int(seed)
    fields.update({k: v for k, v in overrides.items() if v is not None})
    return _replace_config(base, fields)


def _config_to_dict(config: Any) -> Dict[str, Any]:
    """Best-effort conversion of a config-like object to a plain dict."""
    if isinstance(config, dict):
        return dict(config)
    for attr in ("to_dict", "as_dict", "__dict__"):
        value = getattr(config, attr, None)
        if callable(value):
            try:
                return dict(value())
            except Exception:
                continue
        if isinstance(value, dict):
            return dict(value)
    return {}


def _replace_config(config: SAPGConfig, fields: Dict[str, Any]) -> SAPGConfig:
    """``dataclasses.replace`` with a defensive fallback for legacy configs."""
    known = set(getattr(SAPGConfig, "__dataclass_fields__", {}).keys())
    clean = {k: v for k, v in fields.items() if (not known) or (k in known)}
    try:
        return replace(config, **clean)
    except Exception:
        clone = copy.deepcopy(config)
        for key, value in clean.items():
            try:
                setattr(clone, key, value)
            except Exception:
                pass
        return clone


# --------------------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------------------
def set_seed(seed: int, env: Optional[Any] = None, deterministic: bool = False) -> int:
    """Seed python / numpy / torch and (optionally) the vectorised environment.

    The paper runs 5 seeds per experiment (Section 5.2); this helper makes each
    seed reproducible.
    """
    seed = int(seed)
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:  # numpy is optional
        import numpy as _np  # type: ignore

        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
    except Exception:
        pass

    # Re-seed the environment if it exposes a seed / set_seed hook.
    if env is not None:
        for name in ("set_seed", "seed"):
            fn = getattr(env, name, None)
            if callable(fn):
                try:
                    fn(seed)
                    break
                except Exception:
                    continue
        cfg = getattr(env, "cfg", None)
        if cfg is not None and hasattr(cfg, "seed"):
            try:
                cfg.seed = seed
            except Exception:
                pass
    return seed


# --------------------------------------------------------------------------------------
# Policy / environment construction
# --------------------------------------------------------------------------------------
def make_ppo_policy(
    config: SAPGConfig,
    device: Optional[Any] = None,
    policy: Optional[Any] = None,
) -> Any:
    """Instantiate the single :class:`~sapg.models.actor.ActorCritic` for PPO.

    The policy architecture follows Section 5.2: recurrent for the AllegroKuka
    tasks (``use_lstm=True`` in ``configs/allegro_kuka.yaml``) and a plain MLP for
    ShadowHand / AllegroHand (``use_lstm=False``).

    ``phi_dim`` is forced to 0: vanilla PPO has a single policy and therefore no
    per-policy latent conditioning.
    """
    if policy is not None:
        return policy

    from ..models.actor import ActorCritic

    if device is None:
        device = getattr(config, "device", None)
    kwargs: Dict[str, Any] = dict(
        obs_dim=getattr(config, "obs_dim", None),
        action_dim=getattr(config, "action_dim", None),
        phi_dim=0,
        num_policies=1,
        mlp_units=getattr(config, "actor_mlp_units", (768, 512, 256)),
        activation=getattr(config, "actor_activation", "elu"),
        use_lstm=bool(getattr(config, "use_lstm", False)),
        lstm_hidden_size=getattr(config, "lstm_hidden_size", 768),
        lstm_num_layers=getattr(config, "lstm_num_layers", 1),
        action_scale=getattr(config, "action_scale", 1.0),
        critic_mlp_units=getattr(config, "critic_mlp_units", None),
        learnable_phi=False,
    )
    net = ActorCritic(config=config, **kwargs)
    if device is not None:
        try:
            net = net.to(device)
        except Exception:
            pass
    return net


def make_ppo_env(config: SAPGConfig, num_envs: Optional[int] = None, **overrides: Any) -> Any:
    """Create the vectorised environment for the PPO baseline.

    ``num_envs`` defaults to ``config.num_envs`` (the paper's ``N = 24576``).
    """
    from ..envs import make_env

    if num_envs is None:
        num_envs = getattr(config, "num_envs", TOTAL_ENVS)
    return make_env(
        getattr(config, "task", "regrasping"),
        config=config,
        num_envs=int(num_envs),
        **overrides,
    )


# --------------------------------------------------------------------------------------
# Trainer wrapper
# --------------------------------------------------------------------------------------
@dataclass
class PPOBaselineResult:
    """Result of one vanilla-PPO run (one seed).

    Attributes:
        history: per-iteration statistics (``episode_return``, ``samples``, ...).
        samples: total number of environment transitions collected.
        num_envs: ``N`` used for the run.
        seed: seed used for the run (``None`` if unseeded).
        trainer: the underlying :class:`~sapg.algorithms.ppo.PPOTrainer`.
    """

    history: List[Dict[str, float]] = field(default_factory=list)
    samples: int = 0
    num_envs: int = TOTAL_ENVS
    seed: Optional[int] = None
    trainer: Optional[Any] = None

    # ---- convenience accessors -------------------------------------------------
    def final(self, key: str = "episode_return", default: float = float("nan")) -> float:
        """Value of ``key`` at the last logged iteration (or ``default``)."""
        for entry in reversed(self.history):
            if key in entry:
                try:
                    return float(entry[key])
                except (TypeError, ValueError):
                    return default
        return default

    def curve(self, key: str = "episode_return") -> List[float]:
        """Extract a scalar history curve for a given metric key."""
        out: List[float] = []
        for entry in self.history:
            if key in entry:
                try:
                    out.append(float(entry[key]))
                except (TypeError, ValueError):
                    continue
        return out

    def sample_curve(self, key: str = "episode_return") -> Tuple[List[int], List[float]]:
        """Return ``(samples, metric)`` aligned pairs for plotting against samples."""
        xs, ys = [], []
        for entry in self.history:
            if key not in entry:
                continue
            try:
                ys.append(float(entry[key]))
            except (TypeError, ValueError):
                continue
            xs.append(int(entry.get("samples", len(xs))))
        return xs, ys

    def as_dict(self) -> Dict[str, Any]:
        return {
            "samples": self.samples,
            "num_envs": self.num_envs,
            "seed": self.seed,
            "final": self.final(),
            "history": self.history,
        }


class PPOBaselineTrainer:
    """Thin wrapper around :class:`~sapg.algorithms.ppo.PPOTrainer` for vanilla PPO.

    Its only job is to guarantee the single-policy / full-environment setup of
    Section 5.2 and to expose a *sample-budget* training API (the paper compares
    methods against ``~2e10`` collected transitions rather than wall-clock time).
    """

    def __init__(
        self,
        config: Optional[SAPGConfig] = None,
        policy: Optional[Any] = None,
        env: Optional[Any] = None,
        logger: Optional[Any] = None,
        device: Optional[Any] = None,
        num_envs: Optional[int] = None,
        seed: Optional[int] = None,
        **overrides: Any,
    ) -> None:
        self.config = make_ppo_config(config, num_envs=num_envs, seed=seed, **overrides)
        self.device = device if device is not None else getattr(self.config, "device", None)
        self.seed = seed if seed is not None else getattr(self.config, "seed", None)
        if self.seed is not None:
            set_seed(int(self.seed))

        self.env = env if env is not None else make_ppo_env(self.config)
        if env is not None and num_envs is None:
            self.config.num_envs = int(getattr(env, "num_envs", self.config.num_envs))
        self.policy = make_ppo_policy(self.config, device=self.device, policy=policy)
        self.logger = logger

        from ..algorithms.ppo import PPOTrainer

        self.trainer = PPOTrainer(
            self.config,
            self.policy,
            self.env,
            logger=logger,
            device=self.device,
        )
        self.history: List[Dict[str, float]] = []
        self.samples: int = 0

    # ---- properties -------------------------------------------------------------
    @property
    def num_envs(self) -> int:
        return int(getattr(self.config, "num_envs", TOTAL_ENVS))

    @property
    def batch_size(self) -> int:
        """Rollout batch size ``N * horizon_length`` (batch ∝ number of envs)."""
        return self.num_envs * int(getattr(self.config, "horizon_length", 16))

    @property
    def minibatch_size(self) -> int:
        """Minibatch size, scaled proportionally to ``N`` (Section 5.2)."""
        return self.num_envs * int(getattr(self.config, "minibatch_size_multiplier", 4))

    # ---- training ---------------------------------------------------------------
    def train(
        self,
        num_iterations: Optional[int] = None,
        max_samples: Optional[int] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, float]]:
        """Run PPO; stop after ``num_iterations`` iterations or ``max_samples`` transitions."""
        iters = self._iterations_for(num_iterations, max_samples)
        history = self.trainer.learn(iters, verbose=verbose, **kwargs) if iters else []
        self.history.extend(_as_history(history))
        if self.history:
            self.samples = int(self.history[-1].get("samples", self.samples))
        elif iters:
            self.samples += iters * self.batch_size
        return self.history

    learn = train

    def train_samples(self, max_samples: float, verbose: bool = False, **kwargs: Any) -> List[Dict[str, float]]:
        """Convenience alias for ``train(max_samples=int(max_samples))``."""
        return self.train(max_samples=int(max_samples), verbose=verbose, **kwargs)

    def run(self, max_samples: Optional[float] = None, num_iterations: Optional[int] = None,
            verbose: bool = False, **kwargs: Any) -> PPOBaselineResult:
        """Run and return a :class:`PPOBaselineResult`."""
        history = self.train(num_iterations=num_iterations, max_samples=max_samples,
                             verbose=verbose, **kwargs)
        return PPOBaselineResult(
            history=history,
            samples=self.samples,
            num_envs=self.num_envs,
            seed=self.seed,
            trainer=self.trainer,
        )

    def _iterations_for(self, num_iterations: Optional[int], max_samples: Optional[float]) -> int:
        if num_iterations is not None:
            return int(num_iterations)
        if max_samples is None:
            return 1
        per_iter = max(1, self.batch_size)
        return max(1, int(math.ceil(float(max_samples) / per_iter)))

    # ---- checkpointing ----------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return self.trainer.state_dict()

    def save(self, path: str) -> str:
        return self.trainer.save(path)

    def load(self, path: str) -> Any:
        return self.trainer.load(path)

    def evaluate(self, deterministic: bool = True, **kwargs: Any) -> Dict[str, float]:
        """Single-metric evaluation helper (net episode reward, Section 5.1)."""
        collect = getattr(self.trainer, "collect", None)
        if collect is None:
            return {}
        buffers = collect(deterministic=deterministic, **kwargs)
        metrics: Dict[str, float] = {}
        for buf in _iter_buffers(buffers):
            stats = getattr(buf, "stats", None) or getattr(buf, "metrics", None) or {}
            if isinstance(stats, dict):
                for key, value in stats.items():
                    try:
                        metrics[key] = float(value)
                    except (TypeError, ValueError):
                        continue
        return metrics


def _iter_buffers(obj: Any) -> List[Any]:
    """Normalise the many shapes a ``collect`` return value can take."""
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        out: List[Any] = []
        for item in obj:
            if item is None or isinstance(item, (int, float, str)):
                continue
            if isinstance(item, (list, tuple)):
                out.extend([x for x in item if x is not None and not isinstance(x, (int, float, str))])
            else:
                out.append(item)
        return out
    return [obj]


def _as_history(history: Any) -> List[Dict[str, float]]:
    if history is None:
        return []
    if isinstance(history, dict):
        return [{k: _safe_float(v) for k, v in history.items()}]
    out: List[Dict[str, float]] = []
    for entry in history:
        if isinstance(entry, dict):
            out.append({k: v for k, v in entry.items()})
        else:
            out.append(_config_to_dict(entry))
    return out


def _safe_float(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def train_ppo_baseline(
    config: Optional[SAPGConfig] = None,
    env: Optional[Any] = None,
    policy: Optional[Any] = None,
    num_iterations: Optional[int] = None,
    max_samples: Optional[float] = None,
    verbose: bool = False,
    logger: Optional[Any] = None,
    device: Optional[Any] = None,
    num_envs: Optional[int] = None,
    seed: Optional[int] = None,
    return_result: bool = False,
    **overrides: Any,
) -> Any:
    """Train vanilla PPO on all ``N`` environments (Section 5.2).

    Returns ``(trainer, history)`` by default, or a :class:`PPOBaselineResult`
    when ``return_result=True``.
    """
    baseline = PPOBaselineTrainer(
        config=config,
        policy=policy,
        env=env,
        logger=logger,
        device=device,
        num_envs=num_envs,
        seed=seed,
        **overrides,
    )
    result = baseline.run(
        max_samples=max_samples,
        num_iterations=num_iterations,
        verbose=verbose,
    )
    if return_result:
        return result
    return baseline, result.history


# --------------------------------------------------------------------------------------
# Multi-seed aggregation (5 seeds, Section 5.2)
# --------------------------------------------------------------------------------------
def run_ppo_seeds(
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    config: Optional[SAPGConfig] = None,
    num_envs: Optional[int] = None,
    max_samples: Optional[float] = None,
    num_iterations: Optional[int] = None,
    verbose: bool = False,
    trainer_factory: Optional[Callable[[int], Any]] = None,
    **overrides: Any,
) -> List[PPOBaselineResult]:
    """Run one PPO baseline per seed (the paper reports means over 5 seeds).

    ``trainer_factory(seed)`` may be supplied to run an alternative single-policy
    method through the same harness; it must return an object exposing
    ``run(max_samples=..., verbose=...)``.
    """
    results: List[PPOBaselineResult] = []
    for seed in seeds:
        if trainer_factory is not None:
            trainer = trainer_factory(int(seed))
            result = trainer.run(max_samples=max_samples, verbose=verbose)
            if not isinstance(result, PPOBaselineResult):
                result = PPOBaselineResult(
                    history=list(result), samples=getattr(trainer, "samples", 0),
                    num_envs=getattr(trainer, "num_envs", num_envs or TOTAL_ENVS), seed=int(seed),
                )
        else:
            baseline = PPOBaselineTrainer(
                config=config, num_envs=num_envs, seed=int(seed),
                device=getattr(config, "device", None) if config is not None else None,
                **overrides,
            )
            result = baseline.run(
                max_samples=max_samples, num_iterations=num_iterations, verbose=verbose
            )
        results.append(result)
    return results


def paper_standard_error(curves: Sequence[Sequence[float]]) -> Any:
    """The paper's shaded-band width (Section 5.2).

    The paper writes the width of the shaded region as

        (2 / sqrt(n)) * sum_i (y(t) - y_i(t))^2

    with ``y(t) = (1/n) sum_i y_i(t)`` the seed-averaged curve. That is the
    literal expression used in the paper, so we reproduce it verbatim (it is a
    variance-like band, not the textbook standard error ``std/sqrt(n)``).  A
    plain ``torch``/``numpy`` free implementation is used so this works without
    either dependency.

    Args:
        curves: one metric curve per seed (equal lengths assumed; shorter curves
            are padded with their last value).

    Returns:
        ``(mean, band)`` as lists of floats.
    """
    curves = [list(map(float, c)) for c in curves if c is not None]
    if not curves:
        return [], []
    n = len(curves)
    length = max(len(c) for c in curves)
    padded: List[List[float]] = []
    for c in curves:
        if not c:
            padded.append([0.0] * length)
            continue
        padded.append(c + [c[-1]] * (length - len(c)))

    mean: List[float] = []
    band: List[float] = []
    for t in range(length):
        values = [c[t] for c in padded]
        y = sum(values) / n
        mean.append(y)
        variance_like = sum((y - v) ** 2 for v in values)
        band.append((2.0 / math.sqrt(n)) * variance_like)
    return mean, band


def aggregate_seed_histories(
    results: Sequence[Any],
    key: str = "episode_return",
    num_bins: Optional[int] = None,
) -> Dict[str, List[float]]:
    """Aggregate per-seed histories into ``mean`` / ``band`` curves vs samples.

    Args:
        results: :class:`PPOBaselineResult` objects (or plain history lists).
        key: metric to aggregate (default ``episode_return``, the "net episode
            reward" metric used for the easy tasks in Section 5.1).
        num_bins: optionally resample every seed's curve onto a common grid of
            ``num_bins`` points spread uniformly over ``[0, max_samples]``.

    Returns:
        dict with ``samples``, ``mean``, ``band``, ``seed_curves``.
    """
    curves: List[List[float]] = []
    sample_axes: List[List[int]] = []
    for res in results:
        if isinstance(res, PPOBaselineResult):
            xs, ys = res.sample_curve(key)
        elif isinstance(res, dict):
            xs, ys = list(res.get("samples", [])), list(res.get(key, []))
        else:  # a raw history list
            xs, ys = [], []
            for i, entry in enumerate(res or []):
                if key in entry:
                    ys.append(_safe_float(entry[key]))
                    xs.append(int(entry.get("samples", i)))
        if ys:
            curves.append(ys)
            sample_axes.append(xs)

    if not curves:
        return {"samples": [], "mean": [], "band": [], "seed_curves": []}

    if num_bins:
        axis = [i / max(1, num_bins - 1) for i in range(num_bins)]
        max_samples = max((xs[-1] for xs in sample_axes if xs), default=num_bins)
        grid = [int(round(a * max_samples)) for a in axis]
        resampled: List[List[float]] = []
        for ys, xs in zip(curves, sample_axes):
            resampled.append(_resample(ys, xs or list(range(len(ys))), grid))
        curves = resampled
        sample_axis = grid
    else:
        sample_axis = sample_axes[0] if sample_axes and sample_axes[0] else list(range(len(curves[0])))

    mean, band = paper_standard_error(curves)
    return {
        "samples": list(sample_axis),
        "mean": mean,
        "band": band,
        "seed_curves": curves,
    }


def _resample(ys: Sequence[float], xs: Sequence[int], grid: Sequence[int]) -> List[float]:
    """Piecewise-constant resampling of ``ys`` (defined at ``xs``) onto ``grid``."""
    if not ys:
        return [0.0] * len(grid)
    out: List[float] = []
    for g in grid:
        idx = 0
        for i, x in enumerate(xs):
            if x <= g:
                idx = i
            else:
                break
        out.append(float(ys[min(idx, len(ys) - 1)]))
    return out


# --------------------------------------------------------------------------------------
# Figure-2 concept: PPO saturation with increasing batch size
# --------------------------------------------------------------------------------------
def ppo_saturation_sweep(
    env_counts: Iterable[int] = DEFAULT_ENV_SWEEP,
    config: Optional[SAPGConfig] = None,
    task: str = "regrasping",
    samples_per_env_ratio: float = 1.0,
    max_samples: Optional[float] = None,
    seeds: Sequence[int] = (0,),
    verbose: bool = False,
    device: Optional[Any] = None,
    trainer_factory: Optional[Callable[[SAPGConfig, int, int], Any]] = None,
    **overrides: Any,
) -> Dict[int, Dict[str, Any]]:
    """Run vanilla PPO at increasing ``num_envs`` (Fig. 2 concept).

    Each environment count uses a batch size proportional to ``N`` (Section 5.2),
    which is the default behaviour of :class:`~sapg.algorithms.ppo.PPOTrainer`
    through ``minibatch_size = N * minibatch_size_multiplier``.

    To keep the comparison fair, the *sample budget* is held constant across
    environment counts by default (``max_samples`` defaults to
    ``TOTAL_TRANSITIONS * samples_per_env_ratio``); alternatively pass an
    explicit ``max_samples``.

    Args:
        env_counts: numbers of environments to sweep (128 -> 24576).
        config: base configuration (task hyperparameters are taken from it).
        task: task used when ``config`` is None.
        samples_per_env_ratio: scales the default sample budget.
        max_samples: explicit per-run sample budget.
        seeds: seeds to average over per environment count.
        trainer_factory: optional ``(config, num_envs, seed) -> trainer`` hook,
            letting callers plug a different harness while keeping the sweep.
        **overrides: forwarded into every PPO run.

    Returns:
        ``{num_envs: {"mean": [...], "band": [...], "samples": [...],
        "final": float, "results": [...]}}``.
    """
    base = config if config is not None else build_config(task)
    if max_samples is None:
        max_samples = int(TOTAL_TRANSITIONS * float(samples_per_env_ratio))

    out: Dict[int, Dict[str, Any]] = {}
    for n in env_counts:
        n = int(n)
        run_results: List[PPOBaselineResult] = []
        for seed in seeds:
            cfg = make_ppo_config(base, num_envs=n, seed=int(seed), **overrides)
            if trainer_factory is not None:
                trainer = trainer_factory(cfg, n, int(seed))
            else:
                trainer = PPOBaselineTrainer(
                    config=cfg, num_envs=n, seed=int(seed), device=device
                )
            run_results.append(trainer.run(max_samples=max_samples, verbose=verbose))

        agg = aggregate_seed_histories(run_results, key="episode_return")
        final_values = [r.final() for r in run_results]
        finals = [v for v in final_values if v == v]  # drop NaN
        out[n] = {
            "mean": agg["mean"],
            "band": agg["band"],
            "samples": agg["samples"],
            "final": (sum(finals) / len(finals)) if finals else float("nan"),
            "results": run_results,
        }
    return out


def saturation_summary(
    sweep: Dict[int, Dict[str, Any]],
    threshold_fraction: float = 0.9,
    baseline_envs: Optional[int] = None,
) -> Dict[str, Any]:
    """Summarise a saturation sweep: has performance saturated beyond ~10k envs?

    Following the paper's observation (Section 5.2 / Figure 2 concept), PPO's
    asymptotic performance stops improving past roughly 10k environments. This
    helper reports, per environment count, the *relative* final performance with
    respect to a reference count (``baseline_envs``, default: the smallest one)
    and the first count from which all larger counts stay within
    ``1 + threshold_fraction`` of the maximum observed performance.

    Returns:
        dict with ``finals``, ``relative``, ``saturated_from`` and ``max_final``.
    """
    if not sweep:
        return {"finals": {}, "relative": {}, "saturated_from": None, "max_final": float("nan")}

    counts = sorted(sweep.keys())
    finals = {n: float(sweep[n].get("final", float("nan"))) for n in counts}
    finite = {n: v for n, v in finals.items() if v == v}
    if not finite:
        return {"finals": finals, "relative": {}, "saturated_from": None, "max_final": float("nan")}

    ref_envs = int(baseline_envs) if baseline_envs is not None else counts[0]
    ref = finals.get(ref_envs)
    if ref is None or ref != ref:
        ref = finite[min(finite, key=lambda k: abs(k - ref_envs))]
    denom = abs(ref) if abs(ref) > 1e-12 else 1.0
    relative = {n: (v - ref) / denom for n, v in finite.items()}

    max_final = max(finite.values())
    span = max_final - ref
    saturated_from: Optional[int] = None
    if span > 1e-12:
        for idx, n in enumerate(counts):
            tail = [finals[m] for m in counts[idx:] if finals[m] == finals[m]]
            if tail and all(v >= ref + float(threshold_fraction) * span for v in tail):
                saturated_from = n
                break
    return {
        "finals": finals,
        "relative": relative,
        "saturated_from": saturated_from,
        "max_final": max_final,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - thin CLI
    """Minimal CLI: ``python -m sapg.baselines.ppo_baseline --task regrasping``."""
    import argparse

    parser = argparse.ArgumentParser(description="Vanilla PPO baseline (SAPG Section 5.2)")
    parser.add_argument("--task", default="regrasping")
    parser.add_argument("--num-envs", type=int, default=TOTAL_ENVS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--sweep", action="store_true", help="run the saturation sweep instead")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = make_ppo_config(task=args.task, num_envs=args.num_envs, seed=args.seed)
    if args.sweep:
        sweep = ppo_saturation_sweep(config=cfg, verbose=True)
        for n, info in sorted(sweep.items()):
            print(f"envs={n:>6} final_return={info['final']:.3f}")
    else:
        baseline, history = train_ppo_baseline(
            cfg, num_iterations=args.iterations, max_samples=args.max_samples, verbose=True
        )
        print(f"collected {baseline.samples} samples; last={history[-1] if history else '{}'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

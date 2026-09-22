"""Fidelity score evaluator for RICE Experiment I (CORE COMPONENT #5).

Paper specification
-------------------
Evaluation metric (verbatim, §4.1 "Evaluation Metrics")::

    Fidelity Score = log(d / d_max) - log(l / L)

The idea (verbatim): *"to use a sliding window to step through all time steps and
then choose the window with the highest average importance score (scored by the
explanation method). The width of the sliding window is l while the whole
trajectory length is L. Then we randomize the action(s) at the selected critical
step(s) in the selected window (i.e., masking) and measure the average reward
change as d. Additionally, we denote the maximum possible reward change as
d_max."*  A higher fidelity score indicates higher fidelity.

Experiment I procedure (§4.2): *"Given a trajectory, the explanation method first
identifies and ranks top-K important time steps. ... we let the agent fast-forward
to the critical step and force the target agent to take random actions. Then we
follow the target agent's policy to complete the rest of the time steps. If the
explanation is accurate, we expect a major change to the final reward by
randomizing the actions at the important steps. We compute the fidelity score of
each explanation method ... across 500 trajectories. We set K = 10 %, 20 %, 30 %,
40 % and report the fidelity ... We repeat each experiment 3 times with various
random seeds and report the mean and standard deviation."*

So, per trajectory:
    1. roll the pre-trained target policy out for the whole episode  (baseline return R),
    2. score every visited state with the explanation method,
    3. slide a window of width ``l = L * K`` over the importance sequence and select
       the window with the highest average importance,
    4. fast-forward to the beginning of that window, force ``l`` random actions
       (i.e. the masking operation of Eq. (1) applied to every step of the window),
    5. resume the target policy for the remainder of the episode       (masked return R'),
    6. measure ``d = |R' - R|`` and the closed-form fidelity score above.

Everything is black-box: only visited states and the separate explanation module
(mask network / random selector) are consulted; the target agent's internals are
never inspected.

Notes / unspecified details (marked here, repeated in README.md)
----------------------------------------------------------------
* ``d_max`` is not specified by the paper; when not given explicitly we estimate it
  data-driven as the largest absolute episode return observed in a short warm-up
  (per the reproduction plan: "d_max -> per-environment max achievable episode
  reward").  It can always be overridden (``FidelityConfig.d_max``).
* The window is *sliding* with stride 1 (the paper does not specify the stride);
  ``l = L * K`` is rounded to at least one step.
* Restoration of the "fast-forwarded" simulator state uses
  :mod:`rice.algorithms.env_reset` (Go-Explore style, §C.1).  Two capture modes are
  provided: ``"step"`` records a snapshot at every visited step, ``"replay"``
  re-runs the episode deterministically with the same seed and replays the recorded
  policy actions up to the window start (cheaper in memory).
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..algorithms.ppo import flatten_obs, make_target_policy_callable
from ..algorithms.env_reset import (
    EnvStateManager,
    capture_state,
    set_state,
    supports_state_restore,
)

try:  # pragma: no cover - the helpers exist in the implemented algorithms layer
    from ..algorithms.critical_state import best_window_index, windowed_mean
except Exception:  # pragma: no cover - defensive fallback
    windowed_mean = None  # type: ignore[assignment]
    best_window_index = None  # type: ignore[assignment]

try:  # pragma: no cover - optional, used to time mask training (Table 4)
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


__all__ = [
    "FidelityConfig",
    "EvalEpisode",
    "FidelityResult",
    "FidelityEvaluator",
    "fidelity_score",
    "sliding_window_average",
    "best_window",
    "random_window_index",
    "estimated_d_max",
    "make_importance_fn",
    "evaluate_explanation",
    "evaluate_methods",
    "mask_actions",
]


# --------------------------------------------------------------------------------------
# Closed-form metric
# --------------------------------------------------------------------------------------
def fidelity_score(
    d: float,
    l: float,
    L: float,
    d_max: float,
    eps: float = 1e-8,
) -> float:
    """Closed-form fidelity score ``log(d / d_max) - log(l / L)`` (Eq. §4.1).

    ``d``      -- average reward change after randomizing the selected window,
    ``l``      -- sliding-window width (``L * K``),
    ``L``      -- trajectory length,
    ``d_max``  -- maximum possible reward change in one episode.

    Degenerate values are guarded (``d <= 0`` -> ``log(eps)``; ``l <= 0`` -> 1 step;
    ``L <= 0`` -> 1 step) so the score stays finite, which keeps the evaluator usable
    for failed trajectories without discarding them.
    """
    d = float(d)
    l = max(float(l), 1.0)
    L = max(float(L), 1.0)
    d_max = max(float(d_max), eps)
    if not np.isfinite(d):
        d = 0.0
    d_eff = max(abs(d), eps)
    return float(math.log(d_eff / d_max) - math.log(l / L))


# --------------------------------------------------------------------------------------
# Sliding window utilities (shared semantics with rice.algorithms.critical_state)
# --------------------------------------------------------------------------------------
def sliding_window_average(scores: Sequence[float], window: Union[int, float]) -> np.ndarray:
    """Mean of every contiguous window of width ``window`` (stride 1).

    Returned array has length ``max(len(scores) - window + 1, 0)``.  Mirrors
    :func:`rice.algorithms.critical_state.windowed_mean` (used when importable).
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)
    w = int(round(float(window)))
    w = max(1, min(w, values.size))
    if windowed_mean is not None:
        try:
            out = np.asarray(windowed_mean(values, w), dtype=np.float64).reshape(-1)
            if out.size == max(values.size - w + 1, 1):
                return out
        except Exception:  # pragma: no cover - fall through to local implementation
            pass
    if w >= values.size:
        return np.array([values.mean()], dtype=np.float64)
    csum = np.concatenate([[0.0], np.cumsum(values)])
    return (csum[w:] - csum[:-w]) / float(w)


def best_window(scores: Sequence[float], window: Union[int, float]) -> int:
    """Index of the sliding window with the highest average importance score.

    Ties break to the earliest window (``np.argmax`` semantics), matching the
    critical-state selection convention of :mod:`rice.algorithms.critical_state`.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0
    w = max(1, min(int(round(float(window))), values.size))
    if best_window_index is not None:
        try:
            idx = int(best_window_index(values, w))
            if 0 <= idx <= max(values.size - w, 0):
                return idx
        except Exception:  # pragma: no cover
            pass
    means = sliding_window_average(values, w)
    if means.size == 0:
        return 0
    return int(np.argmax(means))


def random_window_index(
    scores: Sequence[float],
    window: Union[int, float],
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Uniformly random window start -- the "Random" explanation baseline (§4.1)."""
    length = int(np.asarray(scores).reshape(-1).size)
    w = max(1, min(int(round(float(window))), max(length, 1)))
    n_starts = max(length - w + 1, 1)
    rng = rng if rng is not None else np.random.default_rng()
    return int(rng.integers(0, n_starts))


def estimated_d_max(
    returns: Sequence[float],
    absolute: bool = True,
    floor: float = 1.0,
) -> float:
    """Data-driven ``d_max`` estimate: largest observed episode-return magnitude.

    The paper does not specify ``d_max``; the reproduction plan suggests the
    per-environment maximum achievable episode reward.  We approximate it with the
    largest |episode return| seen in a short warm-up roll-out (documented default).
    """
    vals = np.asarray([r for r in returns if np.isfinite(r)], dtype=np.float64)
    if vals.size == 0:
        return float(floor)
    value = float(np.max(np.abs(vals)) if absolute else np.max(vals))
    return float(max(value, floor))


def _window_size(L: int, K: Union[int, float]) -> int:
    """``l = L * K`` rounded to a legal window width (at least one step)."""
    L = max(int(L), 1)
    if isinstance(K, float) and 0.0 <= K <= 1.0:
        w = int(round(float(K) * L))
    else:
        w = int(round(float(K)))
    return int(max(1, min(w, L)))


# --------------------------------------------------------------------------------------
# Random action sampling (the masking operation in the fidelity pipeline)
# --------------------------------------------------------------------------------------
def mask_actions(
    action_space: Any,
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample ``n`` uniform random actions from ``action_space`` (Eq. (1)'s ``a_random``)."""
    rng = rng if rng is not None else np.random.default_rng()
    n = int(max(n, 0))
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    try:
        samples = [np.asarray(action_space.sample(), dtype=np.float32) for _ in range(n)]
        return np.stack(samples, axis=0)
    except Exception:
        pass
    try:
        low = np.asarray(getattr(action_space, "low", -1.0), dtype=np.float32)
        high = np.asarray(getattr(action_space, "high", 1.0), dtype=np.float32)
        low = np.where(np.isfinite(low), low, -1.0)
        high = np.where(np.isfinite(high), high, 1.0)
        return rng.uniform(low, high, size=(n,) + tuple(low.shape)).astype(np.float32)
    except Exception:
        return rng.uniform(-1.0, 1.0, size=(n,)).astype(np.float32)


# --------------------------------------------------------------------------------------
# Explanation adapters: anything -> obs -> importance
# --------------------------------------------------------------------------------------
def make_importance_fn(
    explanation: Any = None,
    device: str = "auto",
    batch_size: int = 512,
) -> Callable[[Sequence[Any]], np.ndarray]:
    """Normalise an explanation object into ``states -> np.ndarray`` importances.

    Accepts:
      * ``None`` -> uniform-random importance (the "Random" baseline of §4.1);
      * a :class:`rice.algorithms.mask_network.MaskNetwork` (or any torch module with
        ``importance`` / ``mask_prob_zero`` / a 2-logit forward) -> ``P(mask = 0)``;
      * a numpy array / list -> pre-computed importance scores (returned as-is);
      * any callable mapping states to importances.
    """
    if explanation is None:
        def _random(states: Sequence[Any]) -> np.ndarray:
            n = len(states)
            return np.random.default_rng().uniform(0.0, 1.0, size=n).astype(np.float64)

        return _random

    # Pre-computed scores -------------------------------------------------------------
    if isinstance(explanation, (np.ndarray, list, tuple)) and not callable(explanation):
        precomputed = np.asarray(explanation, dtype=np.float64).reshape(-1)

        def _precomputed(states: Sequence[Any]) -> np.ndarray:
            return precomputed[: len(states)]

        return _precomputed

    # Mask network / torch modules ----------------------------------------------------
    for attr in ("importance", "mask_prob_zero", "score"):
        fn = getattr(explanation, attr, None)
        if callable(fn):
            def _mask_fn(states: Sequence[Any], _fn=fn) -> np.ndarray:
                try:
                    out = _fn(list(states))
                except TypeError:
                    out = _fn(np.asarray([flatten_obs(s) for s in states], dtype=np.float32))
                except Exception:
                    out = np.asarray([_fn(s) for s in states], dtype=np.float64)
                return np.asarray(out, dtype=np.float64).reshape(-1)

            return _mask_fn

    if callable(explanation):
        def _callable_fn(states: Sequence[Any]) -> np.ndarray:
            try:
                out = explanation(list(states))
            except TypeError:
                out = explanation(np.asarray([flatten_obs(s) for s in states], dtype=np.float32))
            return np.asarray(out, dtype=np.float64).reshape(-1)

        return _callable_fn

    raise TypeError(
        f"Unsupported explanation object of type {type(explanation)!r}: expected None, an "
        "array of scores, a mask network (with `importance`) or a callable."
    )


# --------------------------------------------------------------------------------------
# Configuration / result containers
# --------------------------------------------------------------------------------------
@dataclass
class FidelityConfig:
    """Settings of the Experiment I fidelity evaluation (§4.2)."""

    ks: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
    num_trajectories: int = 500
    seeds: Tuple[int, ...] = (0, 1, 2)
    d_max: Optional[float] = None
    max_steps: Optional[int] = None
    deterministic_actions: bool = False
    random_window: bool = False
    capture_mode: str = "replay"  # {"replay", "step"}
    state_manager: Optional[Any] = None
    d_max_warmup_episodes: int = 5
    score_eps: float = 1e-8
    seed: Optional[int] = None
    device: str = "auto"
    verbose: int = 1
    log_every: int = 100

    def clone(self, **overrides: Any) -> "FidelityConfig":
        return dataclasses.replace(self, **overrides)

    @classmethod
    def from_mapping(cls, mapping: Optional[Dict[str, Any]], **overrides: Any) -> "FidelityConfig":
        """Build from a YAML config mapping (tolerates ``K``/``L``/``num_episodes`` aliases)."""
        mapping = dict(mapping or {})
        if "K" in mapping and "ks" not in mapping:
            mapping["ks"] = mapping.pop("K")
        if "num_episodes" in mapping and "num_trajectories" not in mapping:
            mapping["num_trajectories"] = mapping.pop("num_episodes")
        if "lambda" in mapping:
            mapping.pop("lambda")  # not part of Experiment I
        known = {f.name for f in dataclasses.fields(cls)}
        clean = {k: v for k, v in mapping.items() if k in known}
        if isinstance(clean.get("ks"), list):
            clean["ks"] = tuple(clean["ks"])
        if isinstance(clean.get("seeds"), list):
            clean["seeds"] = tuple(clean["seeds"])
        return cls(**{**clean, **overrides})


@dataclass
class EvalEpisode:
    """Bookkeeping for a single evaluated trajectory (auditability of the metric)."""

    seed: int
    k: float
    length: int
    window_start: int
    window_size: int
    baseline_return: float
    masked_return: float
    reward_change: float
    d_max: float
    score: float
    mean_importance: float = 0.0
    restored: bool = True
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class FidelityResult:
    """Aggregated Experiment I outcome: mean/std per ``K`` over trajectories and seeds."""

    k_values: Tuple[float, ...]
    means: Dict[str, float]
    stds: Dict[str, float]
    scores: Dict[str, List[float]] = field(default_factory=dict)
    d_max: float = 1.0
    n_trajectories: int = 0
    seconds: float = 0.0
    episodes: List[EvalEpisode] = field(default_factory=list)
    method: str = "ours"

    # -- convenience -------------------------------------------------------------------
    def mean(self, k: float) -> float:
        return float(self.means.get(_key(k), float("nan")))

    def std(self, k: float) -> float:
        return float(self.stds.get(_key(k), float("nan")))

    def score(self, k: float) -> Tuple[float, float]:
        return self.mean(k), self.std(k)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "k_values": list(self.k_values),
            "means": dict(self.means),
            "stds": dict(self.stds),
            "n_trajectories": self.n_trajectories,
            "d_max": self.d_max,
            "seconds": self.seconds,
        }

    def rows(self) -> List[Dict[str, Any]]:
        """Flattened rows for CSV/DataFrame export (one row per ``K``)."""
        return [
            {
                "method": self.method,
                "K": k,
                "fidelity_mean": self.mean(k),
                "fidelity_std": self.std(k),
                "n": len(self.scores.get(_key(k), [])),
            }
            for k in self.k_values
        ]

    def table(self) -> str:
        lines = [f"method={self.method}  d_max={self.d_max:.3f}  n={self.n_trajectories}"]
        for k in self.k_values:
            lines.append(
                f"  K={int(round(k * 100)):>3d}%  fidelity = {self.mean(k):+.4f} "
                f"± {self.std(k):.4f}"
            )
        return "\n".join(lines)


def _key(k: Any) -> str:
    try:
        return f"{float(k):.4f}"
    except Exception:
        return str(k)


# --------------------------------------------------------------------------------------
# Evaluator
# --------------------------------------------------------------------------------------
class FidelityEvaluator:
    """Computes the StateMask-style fidelity score (RICE Experiment I, §4.1/§4.2).

    Parameters
    ----------
    env:
        Single (non-vectorised) simulator environment.
    policy:
        The pre-trained target agent -- an SB3 model, an
        :class:`rice.algorithms.ppo.ActorCritic`, or a plain callable ``obs -> action``.
    explanation:
        The explanation method under evaluation: ``None`` (random baseline), a
        :class:`rice.algorithms.mask_network.MaskNetwork`, pre-computed importances or a
        callable ``states -> importances``.
    config:
        :class:`FidelityConfig` (or a mapping / ``None`` for paper defaults).
    state_manager:
        Optional :class:`rice.algorithms.env_reset.EnvStateManager` (Go-Explore style
        save/restore, §C.1).  Created on demand when the environment supports it.
    """

    def __init__(
        self,
        env: Any,
        policy: Any,
        explanation: Any = None,
        config: Optional[Union[FidelityConfig, Dict[str, Any]]] = None,
        state_manager: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
        d_max: Optional[float] = None,
        method: str = "ours",
    ) -> None:
        if isinstance(config, dict):
            config = FidelityConfig.from_mapping(config)
        self.config: FidelityConfig = config or FidelityConfig()
        self.env = env
        self.policy = policy
        self._policy_action = make_target_policy_callable(policy)
        self.importance_fn = make_importance_fn(explanation, device=self.config.device)
        self.explanation = explanation
        self.method = method
        self.rng = rng if rng is not None else np.random.default_rng(self.config.seed)
        self._d_max_override = d_max if d_max is not None else self.config.d_max
        self._d_max: Optional[float] = None
        self.state_manager = state_manager if state_manager is not None else self.config.state_manager
        self._restore_supported: Optional[bool] = None
        self._length = self._episode_length()

    # -- helpers -----------------------------------------------------------------------
    def _episode_length(self) -> int:
        if self.config.max_steps is not None:
            return int(self.config.max_steps)
        for attr in ("max_episode_steps", "_max_episode_steps"):
            value = getattr(self.env, attr, None)
            if isinstance(value, (int, np.integer)) and value > 0:
                return int(value)
        spec = getattr(self.env, "spec", None)
        value = getattr(spec, "max_episode_steps", None) if spec is not None else None
        if isinstance(value, (int, np.integer)) and value > 0:
            return int(value)
        inner = getattr(self.env, "env", None)
        value = getattr(inner, "_max_episode_steps", None) if inner is not None else None
        if isinstance(value, (int, np.integer)) and value > 0:
            return int(value)
        return 1000

    def _reset(self, seed: Optional[int] = None) -> np.ndarray:
        try:
            out = self.env.reset(seed=seed)
        except TypeError:
            try:
                self.env.seed(seed)
            except Exception:
                pass
            out = self.env.reset()
        if isinstance(out, tuple):
            out = out[0]
        return out

    def _step(self, action: Any) -> Tuple[Any, float, bool]:
        out = self.env.step(action)
        if not isinstance(out, tuple):  # pragma: no cover - exotic API
            obs, reward, done = out[0], out[1], bool(out[2])
            return obs, float(reward), done
        if len(out) == 5:
            obs, reward, terminated, truncated, _info = out
            return obs, float(reward), bool(terminated) or bool(truncated)
        obs, reward, done = out[0], out[1], out[2]
        return obs, float(reward), bool(done)

    def _action(self, obs: Any) -> Any:
        try:
            return self._policy_action(obs)
        except TypeError:
            return self._policy_action(obs, self.config.deterministic_actions)

    # -- roll-out ----------------------------------------------------------------------
    def rollout(
        self,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
        capture: bool = False,
    ) -> Dict[str, Any]:
        """Run the target policy for one episode; returns the trajectory and its return.

        When ``capture`` is true a Go-Explore snapshot is stored for every visited state
        (``capture_mode="step"``); otherwise states/actions are recorded so the episode can
        be replayed deterministically (``capture_mode="replay"``).
        """
        obs = self._reset(seed)
        limit = int(max_steps or self._length)
        states: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        rewards: List[float] = []
        snapshots: List[Any] = []
        total = 0.0
        done = False
        steps = 0
        while steps < limit:
            states.append(np.asarray(flatten_obs(obs), dtype=np.float32))
            if capture:
                try:
                    snapshots.append(capture_state(self.env, observation=obs, episode_step=steps))
                except Exception:
                    snapshots.append(None)
            action = self._action(obs)
            actions.append(np.asarray(action))
            obs, reward, done = self._step(action)
            rewards.append(float(reward))
            total += float(reward)
            steps += 1
            if done:
                break
        return {
            "states": np.asarray(states, dtype=np.float32) if states else np.zeros((0,), np.float32),
            "actions": np.asarray(actions) if actions else np.zeros((0,)),
            "rewards": np.asarray(rewards, dtype=np.float64),
            "snapshots": snapshots,
            "return": float(total),
            "length": int(steps),
            "done": bool(done),
        }

    def _restore_point(
        self,
        episode: Dict[str, Any],
        index: int,
        seed: Optional[int],
        capture_mode: str,
    ) -> Tuple[Any, bool]:
        """Fast-forward the simulator to ``index``; returns the current observation."""
        snapshots = episode.get("snapshots") or []
        if capture_mode == "step" and index < len(snapshots) and snapshots[index] is not None:
            try:
                obs = set_state(self.env, snapshots[index])
                if obs is not None:
                    return obs, True
                return self._current_obs(), True
            except Exception:
                pass
        if self.state_manager is not None:
            try:
                self.state_manager.restore(snapshots[index]) if snapshots else None
            except Exception:
                pass
        # Deterministic replay: same seed + identical action sequence => same trajectory.
        obs = self._reset(seed)
        for i in range(min(index, len(episode["actions"]))):
            obs, _r, done = self._step(episode["actions"][i])
            if done:
                return obs, False
        return obs, True

    def _current_obs(self) -> Any:
        for attr in ("current_observation", "get_observation", "observation"):
            fn = getattr(self.env, attr, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    continue
        try:  # pragma: no cover - last resort
            return self._reset()
        except Exception:
            return None

    # -- d_max -------------------------------------------------------------------------
    def estimate_d_max(self, n_episodes: Optional[int] = None) -> float:
        """Warm-up estimate of ``d_max`` (paper-silent; documented default)."""
        if self._d_max is not None:
            return self._d_max
        if self._d_max_override is not None:
            self._d_max = float(self._d_max_override)
            return self._d_max
        n = int(n_episodes or self.config.d_max_warmup_episodes)
        returns: List[float] = []
        for i in range(max(n, 1)):
            try:
                returns.append(self.rollout(seed=int(self.rng.integers(0, 1 << 30)))["return"])
            except Exception:
                break
        self._d_max = estimated_d_max(returns)
        return self._d_max

    # -- single trajectory -------------------------------------------------------------
    def evaluate_trajectory(self, k: float, seed: Optional[int] = None) -> EvalEpisode:
        """Full fidelity pipeline for one trajectory and one window fraction ``K``."""
        cfg = self.config
        episode_seed = int(seed if seed is not None else self.rng.integers(0, 1 << 30))
        capture = cfg.capture_mode == "step"
        try:
            episode = self.rollout(seed=episode_seed, capture=capture)
        except Exception as exc:  # pragma: no cover - defensive
            return EvalEpisode(
                seed=episode_seed, k=float(k), length=0, window_start=0, window_size=0,
                baseline_return=0.0, masked_return=0.0, reward_change=0.0,
                d_max=self.estimate_d_max(), score=float("nan"), error=str(exc),
            )

        L = max(int(episode["length"]), 1)
        l = _window_size(L, k)
        states = episode["states"]

        # (2) score every visited state with the explanation method
        try:
            importances = np.asarray(self.importance_fn(list(states)), dtype=np.float64).reshape(-1)
        except Exception:
            importances = np.zeros(L, dtype=np.float64)
        if importances.size < L:
            importances = np.pad(importances, (0, L - importances.size), constant_values=0.0)
        importances = importances[:L]

        # (3) select the highest-average-importance sliding window
        if cfg.random_window or self.explanation is None and self._random_mode():
            start = random_window_index(importances, l, self.rng)
        else:
            start = best_window(importances, l)
        window_mean_importance = float(np.mean(importances[start : start + l])) if l else 0.0

        # (4) fast-forward, force l random actions (masking)
        restore_seed = episode_seed
        obs, restored = self._restore_point(episode, start, restore_seed, cfg.capture_mode)
        random_actions = mask_actions(getattr(self.env, "action_space", None), l, self.rng)
        masked_total = 0.0
        steps_taken = 0
        done = False
        for i in range(l):
            if obs is None:
                break
            obs, reward, done = self._step(random_actions[i])
            masked_total += reward
            steps_taken += 1
            if done:
                break

        # (5) resume the target policy for the remaining steps
        while not done and (start + steps_taken) < L:
            if obs is None:
                break
            obs, reward, done = self._step(self._action(obs))
            masked_total += reward
            steps_taken += 1

        # (6) d = |R' - R|
        baseline = float(episode["return"])
        d = abs(masked_total - baseline)
        d_max = self.estimate_d_max()
        score = fidelity_score(d, l, L, d_max, eps=cfg.score_eps)
        return EvalEpisode(
            seed=episode_seed,
            k=float(k),
            length=int(L),
            window_start=int(start),
            window_size=int(l),
            baseline_return=baseline,
            masked_return=float(masked_total),
            reward_change=float(d),
            d_max=float(d_max),
            score=float(score),
            mean_importance=window_mean_importance,
            restored=bool(restored),
        )

    def _random_mode(self) -> bool:
        """True when the explanation is the ``None`` (Random) baseline of §4.1."""
        return self.explanation is None

    # -- full evaluation ---------------------------------------------------------------
    def evaluate(self) -> FidelityResult:
        """Run Experiment I: ``num_trajectories`` per seed, across all ``K`` values.

        Returns mean/std of the fidelity score per ``K`` (the paper reports mean and
        standard deviation over 3 random seeds with 500 trajectories each).
        """
        cfg = self.config
        t0 = time.time()
        self.estimate_d_max()
        scores: Dict[str, List[float]] = {_key(k): [] for k in cfg.ks}
        episodes: List[EvalEpisode] = []
        base_seed = int(cfg.seed or 0)
        for seed_i, seed in enumerate(cfg.seeds):
            seed_rng = np.random.default_rng(base_seed + int(seed))
            per_seed: Dict[str, List[float]] = {_key(k): [] for k in cfg.ks}
            for t in range(int(cfg.num_trajectories)):
                ep_seed = int(seed_rng.integers(0, 1 << 30))
                for k in cfg.ks:
                    ep = self.evaluate_trajectory(k, seed=ep_seed)
                    episodes.append(ep)
                    if np.isfinite(ep.score):
                        per_seed[_key(k)].append(ep.score)
                        scores[_key(k)].append(ep.score)
                if cfg.verbose and cfg.log_every and (t + 1) % int(cfg.log_every) == 0:
                    print(
                        f"[fidelity:{self.method}] seed {seed} "
                        f"{t + 1}/{cfg.num_trajectories} trajectories"
                    )
            if cfg.verbose:
                msg = " ".join(
                    f"K={int(round(k * 100))}%:{np.mean(per_seed[_key(k)]):+.3f}"
                    for k in cfg.ks
                    if per_seed[_key(k)]
                )
                print(f"[fidelity:{self.method}] seed {seed} done -> {msg}")

        means = {k: float(np.mean(v)) if v else float("nan") for k, v in scores.items()}
        stds = {k: float(np.std(v)) if v else float("nan") for k, v in scores.items()}
        return FidelityResult(
            k_values=tuple(cfg.ks),
            means=means,
            stds=stds,
            scores=scores,
            d_max=float(self._d_max or 1.0),
            n_trajectories=int(cfg.num_trajectories),
            seconds=float(time.time() - t0),
            episodes=episodes,
            method=self.method,
        )

    # Convenience: a single-K evaluation (used by tests / quick checks) ---------------
    def evaluate_k(self, k: float, num_trajectories: Optional[int] = None) -> Tuple[float, float]:
        cfg = self.config.clone(ks=(float(k),), num_trajectories=int(num_trajectories or self.config.num_trajectories))
        result = FidelityEvaluator(
            self.env,
            self.policy,
            explanation=self.explanation,
            config=cfg,
            state_manager=self.state_manager,
            rng=self.rng,
            d_max=self._d_max_override,
            method=self.method,
        ).evaluate()
        return result.mean(k), result.std(k)


# --------------------------------------------------------------------------------------
# Functional API
# --------------------------------------------------------------------------------------
def evaluate_explanation(
    env: Any,
    policy: Any,
    explanation: Any = None,
    config: Optional[Union[FidelityConfig, Dict[str, Any]]] = None,
    state_manager: Optional[Any] = None,
    d_max: Optional[float] = None,
    method: str = "ours",
) -> FidelityResult:
    """Evaluate one explanation method with the Experiment I fidelity score."""
    evaluator = FidelityEvaluator(
        env,
        policy,
        explanation=explanation,
        config=config,
        state_manager=state_manager,
        d_max=d_max,
        method=method,
    )
    return evaluator.evaluate()


def evaluate_methods(
    env: Any,
    policy: Any,
    methods: Dict[str, Any],
    config: Optional[Union[FidelityConfig, Dict[str, Any]]] = None,
    state_manager: Optional[Any] = None,
    d_max: Optional[float] = None,
) -> Dict[str, FidelityResult]:
    """Evaluate several explanation methods on the same environment/policy.

    ``methods`` maps a display name (e.g. ``"ours"``, ``"StateMask"``, ``"Random"``,
    ``"IntegratedGradients"``, ``"AIRS"``) to an explanation object accepted by
    :func:`make_importance_fn` (``None`` = Random).
    """
    cfg = config if isinstance(config, FidelityConfig) else FidelityConfig.from_mapping(config)
    results: Dict[str, FidelityResult] = {}
    for name, explanation in methods.items():
        results[name] = evaluate_explanation(
            env,
            policy,
            explanation=explanation,
            config=cfg,
            state_manager=state_manager,
            d_max=d_max,
            method=name,
        )
    return results

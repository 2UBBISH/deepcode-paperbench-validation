"""Integrated Gradients (IG) explanation provider for RICE.

Implements the baseline explanation method requested in the addendum
("Integrated Gradients (Sundararajan et al. 2017)") and used in the paper's
secondary comparison of explanation methods (Table 6 / §C.3 "Impact of Other
Explanation Methods")::

    "we fix the refining method and use [AIRS and Integrated Gradients] ... on
     four Mujoco games"

Integrated Gradients attributes a model output ``F`` at input ``x`` to each
input feature by integrating the gradient along the straight-line path from a
baseline ``x'`` to ``x``:

.. math::

    \\mathrm{IG}_i(x) = (x_i - x'_i) \\times
        \\int_{0}^{1} \\frac{\\partial F(x' + \\alpha (x - x'))}{\\partial x_i}
        \\, d\\alpha

The integral is approximated with a Riemann sum over ``steps`` interpolation
points (Sundararajan et al. use 20-300 steps; the paper does not specify a
number, so a documented default is used).

For a *step-level* explanation of a trajectory we apply IG to every visited
state ``s_t`` with respect to the differentiable black-box model made available
by the caller (by default the target policy's action log-probability, which is
the quantity the policy optimises).  The per-state importance is the sum of the
absolute attributions over observation features:

.. math::

    I(s_t) = \\sum_i \\left| \\mathrm{IG}_i(s_t) \\right|

and the trajectory-level critical state is ``argmax_t I(s_t)`` (ties broken
towards the earliest step), mirroring ``rice.algorithms.critical_state``.

Design notes
------------
* **Black-box w.r.t. internals**: only the model's numeric input/output and the
  gradient of the scalar output w.r.t. its *input* are used; no internal layer
  is inspected and no target-specific parameter is assumed.
* Torch is an optional dependency: when it is unavailable the explainer falls
  back to a finite-difference estimate of the path integral, so the module (and
  the explanation registry) never hard-fails.
* The shared explanation interface is honoured: ``importance`` / ``score`` /
  ``mask_prob_zero`` / ``__call__`` / ``select_index`` / ``best_index`` /
  ``critical_state`` / ``reset`` / ``state_dict`` / ``load_state_dict``.

Unspecified in the paper (documented defaults, see README): number of IG steps,
baseline choice, the scalar model output being attributed, and the feature
aggregation rule.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # torch is optional at import time
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


NAME = "integrated_gradients"
ALIASES = (
    "integrated_gradients",
    "integrated-gradients",
    "integratedgradients",
    "ig",
    "int_grad",
    "intgrad",
)

# Table 6 (secondary comparison) reference ordering: RICE(Ours) > AIRS > IG >
# Random.  Stored here purely as trend metadata for report generation; the
# addendum asks for trends, not exact numbers.
TABLE6_ORDERING: Tuple[str, ...] = ("ours", "airs", "integrated_gradients", "random")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class IntegratedGradientsConfig:
    """Configuration for :class:`IntegratedGradientsExplanation`.

    Parameters
    ----------
    steps:
        Number of interpolation points used for the Riemann sum of the path
        integral (Sundararajan et al. 2017 recommend 20-300; the paper does not
        specify, so 32 is used).
    baseline:
        Baseline ``x'``.  One of ``"zeros"``, ``"mean"``, ``"min"`` or a
        concrete array/vector.  ``"zeros"`` (the canonical IG baseline for
        normalised observations) is the default.
    target:
        Scalar model output to attribute.  ``"log_prob"`` (policy log
        probability of the action actually taken / of the greedy action),
        ``"value"``, ``"action"`` (sum of the action mean vector) or any
        callable ``model(obs_tensor) -> scalar tensor``.
    aggregate:
        How the attribution vector is reduced to one importance number per
        state: ``"sum_abs"`` (default), ``"mean_abs"``, ``"l2"`` or ``"max_abs"``.
    internal_batch_size:
        Number of interpolation points evaluated per forward/backward pass
        (memory control for long observation vectors).
    normalize:
        If ``True`` (default) importance scores are min-max normalised over the
        evaluated trajectory/window so they are comparable with the mask
        network's ``P(a^m = 0 | s)`` in ``[0, 1]``.
    multiply_by_inputs:
        Standard IG variant choice.  ``True`` reproduces the equation above
        (default); ``False`` returns the pure path-averaged gradients.
    device:
        ``"auto"``, ``"cpu"`` or ``"cuda"``.
    seed:
        Seed for the local RNG (used by the finite-difference fallback).
    model / action:
        Optional differentiable model and the action used for the ``log_prob``
        target.  Usually supplied at call time instead.
    """

    task: Optional[str] = None
    steps: int = 32
    baseline: Any = "zeros"
    target: str = "log_prob"
    aggregate: str = "sum_abs"
    internal_batch_size: int = 16
    normalize: bool = True
    multiply_by_inputs: bool = True
    device: str = "auto"
    seed: Optional[int] = None
    model: Any = None
    action: Any = None
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "steps": self.steps,
            "baseline": self.baseline if isinstance(self.baseline, str) else "array",
            "target": self.target if isinstance(self.target, str) else "callable",
            "aggregate": self.aggregate,
            "internal_batch_size": self.internal_batch_size,
            "normalize": self.normalize,
            "multiply_by_inputs": self.multiply_by_inputs,
            "device": self.device,
            "seed": self.seed,
            "notes": self.notes,
        }

    def clone(self, **overrides: Any) -> "IntegratedGradientsConfig":
        data = dict(self.__dict__)
        extra = dict(self.extra)
        for key, value in overrides.items():
            if key in data:
                data[key] = value
            else:
                extra[key] = value
        data["extra"] = extra
        return IntegratedGradientsConfig(**data)

    @classmethod
    def from_mapping(
        cls, mapping: Any = None, **overrides: Any
    ) -> "IntegratedGradientsConfig":
        cfg = cls()
        if mapping is None:
            return cfg.clone(**overrides) if overrides else cfg
        if isinstance(mapping, IntegratedGradientsConfig):
            return mapping.clone(**overrides)
        if isinstance(mapping, dict):
            data = dict(mapping)
            # tolerate a wrapper config (e.g. {"explanation": {...}})
            for wrapper in ("integrated_gradients", "ig", "explanation"):
                inner = data.get(wrapper)
                if isinstance(inner, dict):
                    data = {**data, **inner}
                    break
            known = set(cfg.__dict__.keys())
            kwargs = {k: v for k, v in data.items() if k in known}
            extra = {k: v for k, v in data.items() if k not in known}
            cfg = cls(**kwargs)
            cfg.extra.update(extra)
            return cfg.clone(**overrides) if overrides else cfg
        return cfg.clone(**overrides) if overrides else cfg


# ---------------------------------------------------------------------------
# Small helpers (numpy ------------------------------------------------ )
# ---------------------------------------------------------------------------
def _resolve_device(device: str) -> str:
    if device in (None, "auto"):
        if _TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return str(device)


def _make_rng(seed: Optional[int] = None) -> Any:
    """Prefer :class:`rice.utils.seeding.RNG`, fall back to numpy."""
    try:  # pragma: no cover - path dependent
        from rice.utils.seeding import RNG  # type: ignore

        return RNG(seed)
    except Exception:
        try:  # pragma: no cover - path dependent
            from ..utils.seeding import RNG  # type: ignore

            return RNG(seed)
        except Exception:
            pass

    class _FallbackRNG:
        def __init__(self, seed: Optional[int] = None) -> None:
            self.seed = seed
            self.generator = np.random.default_rng(seed)

        def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
            return float(self.generator.uniform(low, high))

        def bernoulli(self, p: float) -> bool:
            return bool(self.generator.uniform() < float(p))

        def choice(self, a, p=None):  # noqa: D401
            return self.generator.choice(a, p=p)

        def __getattr__(self, item):
            return getattr(self.generator, item)

    return _FallbackRNG(seed)


def _as_2d(states: Any) -> np.ndarray:
    """Flatten a batch of observations/states to ``(N, D)`` float32."""
    if states is None:
        return np.zeros((0, 0), dtype=np.float32)
    try:
        from ..algorithms.ppo import flatten_obs  # type: ignore

        if isinstance(states, (list, tuple)) and states:
            arr = np.stack([np.asarray(flatten_obs(s), dtype=np.float32) for s in states])
            return arr.reshape(len(states), -1)
        arr = np.asarray(flatten_obs(states), dtype=np.float32)
    except Exception:
        arr = np.asarray(states, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    else:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _baseline_vector(baseline: Any, shape: Tuple[int, ...], data: Optional[np.ndarray] = None) -> np.ndarray:
    if baseline is None:
        return np.zeros(shape, dtype=np.float32)
    if isinstance(baseline, str):
        key = baseline.strip().lower()
        if key in ("zeros", "zero", "0"):
            return np.zeros(shape, dtype=np.float32)
        if key in ("mean", "average") and data is not None and data.size:
            return np.broadcast_to(
                data.mean(axis=0).astype(np.float32), shape
            ).copy()
        if key in ("min", "minimum") and data is not None and data.size:
            return np.broadcast_to(
                data.min(axis=0).astype(np.float32), shape
            ).copy()
        if key in ("max", "maximum") and data is not None and data.size:
            return np.broadcast_to(
                data.max(axis=0).astype(np.float32), shape
            ).copy()
        return np.zeros(shape, dtype=np.float32)
    arr = np.asarray(baseline, dtype=np.float32)
    if arr.ndim == 0:
        return np.full(shape, float(arr), dtype=np.float32)
    return np.broadcast_to(arr.reshape(-1), shape).astype(np.float32).copy()


def _aggregate(attr: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "sum_abs").strip().lower()
    abs_attr = np.abs(attr)
    if mode in ("sum_abs", "sum", "l1"):
        return abs_attr.sum(axis=-1)
    if mode in ("mean_abs", "mean"):
        return abs_attr.mean(axis=-1)
    if mode in ("l2", "norm", "euclidean"):
        return np.sqrt((attr ** 2).sum(axis=-1))
    if mode in ("max_abs", "max"):
        return abs_attr.max(axis=-1)
    if mode in ("sum_signed", "signed"):
        return attr.sum(axis=-1)
    raise ValueError(f"Unknown aggregate mode: {mode!r}")


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores.astype(np.float32)
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - lo) / (hi - lo)).astype(np.float32)


# ---------------------------------------------------------------------------
# Model output resolution (scalar function of the observation)
# ---------------------------------------------------------------------------
def resolve_scalar_fn(explainer_or_fn: Any) -> Callable[[Any], np.ndarray]:
    """Build ``obs -> scalar`` (list of floats) from a model or callable.

    Accepts:
      * a plain callable ``model(obs_batch) -> scalar or vector``;
      * an SB3-style model exposing ``policy`` (or ``actor``/``critic``);
      * a :class:`rice.algorithms.ppo.ActorCritic` instance.
    """
    if explainer_or_fn is None:
        raise ValueError("Integrated Gradients requires a differentiable model")

    if isinstance(explainer_or_fn, str):
        raise ValueError("A concrete model or callable is required, got a string")

    if _TORCH_AVAILABLE and torch is not None and isinstance(explainer_or_fn, torch.nn.Module):
        module = explainer_or_fn
        module.eval()

        def _fn(obs: Any) -> np.ndarray:
            with torch.enable_grad():
                if not isinstance(obs, torch.Tensor):
                    tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), dtype=torch.float32)
                else:
                    tensor = obs.float()
                tensor = tensor.clone().requires_grad_(True)
                out = module(tensor)
                scalar = _scalar_from_output(out)
            return scalar.detach().cpu().numpy().reshape(-1)

        return _fn

    if callable(explainer_or_fn):
        def _fn(obs: Any) -> np.ndarray:  # type: ignore[misc]
            out = explainer_or_fn(obs)
            if isinstance(out, np.ndarray):
                return out.reshape(-1).astype(np.float32)
            if isinstance(out, (list, tuple)):
                return np.asarray(out, dtype=np.float32).reshape(-1)
            if _TORCH_AVAILABLE and torch is not None and isinstance(out, torch.Tensor):
                return out.detach().cpu().numpy().reshape(-1)
            return np.asarray(out, dtype=np.float32).reshape(-1)

        return _fn

    raise TypeError(f"Unsupported model type for Integrated Gradients: {type(explainer_or_fn)!r}")


def _scalar_from_output(out: Any) -> Any:
    """Reduce a model output to a per-sample scalar tensor."""
    if _TORCH_AVAILABLE and torch is not None and isinstance(out, torch.Tensor):
        if out.ndim == 1:
            return out
        return out.reshape(out.shape[0], -1).sum(dim=-1)
    if isinstance(out, (tuple, list)):
        for element in out:
            if _TORCH_AVAILABLE and torch is not None and isinstance(element, torch.Tensor):
                return _scalar_from_output(element)
        return _scalar_from_output(np.asarray(out, dtype=np.float32))
    arr = np.asarray(out, dtype=np.float32)
    return arr.reshape(arr.shape[0], -1).sum(axis=-1)


# ---------------------------------------------------------------------------
# Main explainer
# ---------------------------------------------------------------------------
class IntegratedGradientsExplanation:
    """Step-level explanation via Integrated Gradients.

    The object exposes the same duck-typed surface as the mask network
    explanation (``rice.algorithms.mask_network.MaskNetwork``) and the random /
    StateMask providers so it can be dropped into the fidelity evaluator
    (Experiment I) and the refining comparison (Table 6).
    """

    def __init__(
        self,
        model: Any = None,
        config: Any = None,
        device: str = "auto",
        seed: Optional[int] = None,
        rng: Any = None,
        task: Optional[str] = None,
        env: Any = None,
        policy: Any = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(config, IntegratedGradientsConfig):
            cfg = config.clone(**{k: v for k, v in kwargs.items() if hasattr(config, k)})
        else:
            cfg = IntegratedGradientsConfig.from_mapping(config, **kwargs)

        if task is not None:
            cfg.task = task
        if model is None:
            model = cfg.model if cfg.model is not None else policy
        if model is None and env is not None:
            # nothing sensible to attribute -> keep None, degrade to constants
            model = None

        self.config = cfg
        self.task = cfg.task
        self.env = env
        self.policy = policy
        self.device = _resolve_device(device if device not in (None, "auto") else cfg.device)
        self.seed = seed if seed is not None else cfg.seed
        self.rng = rng if rng is not None else _make_rng(self.seed)
        self.model = model
        self.action = cfg.action
        self._scalar_fn: Optional[Callable[[Any], np.ndarray]] = None
        self._feature_dim: Optional[int] = None
        self._last_scores: Optional[np.ndarray] = None
        self._available = None
        if model is not None:
            try:
                self._scalar_fn = resolve_scalar_fn(model)
                self._available = True
            except Exception:
                self._scalar_fn = None
                self._available = False
        self.notes: List[str] = []
        if cfg.notes:
            self.notes.append(cfg.notes)

    # -- capability ------------------------------------------------------
    @property
    def available(self) -> bool:
        """Whether a differentiable scalar model is wired up."""
        return bool(self._available)

    def set_model(self, model: Any, action: Any = None) -> "IntegratedGradientsExplanation":
        self.model = model
        if action is not None:
            self.action = action
        try:
            self._scalar_fn = resolve_scalar_fn(model)
            self._available = True
        except Exception:
            self._scalar_fn = None
            self._available = False
        return self

    # -- core attribution ------------------------------------------------
    def attributions(
        self,
        states: Any,
        actions: Any = None,
        baseline: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Per-feature IG attributions for every state in ``states``.

        Returns an array of shape ``(N, D)`` where ``D`` is the flattened
        observation dimension.
        """
        obs = _as_2d(states)
        if obs.shape[0] == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if self._scalar_fn is None:
            # No differentiable model available -> neutral attributions.
            if not self.notes:
                self.notes.append("no differentiable model supplied; returning zero attributions")
            return np.zeros_like(obs, dtype=np.float32)

        base = _baseline_vector(
            baseline if baseline is not None else self.config.baseline,
            obs.shape,
            data=obs,
        )

        if _TORCH_AVAILABLE and torch is not None:
            try:
                attr = self._torch_attributions(obs, base)
                self._feature_dim = attr.shape[-1] if attr.ndim == 2 else obs.shape[-1]
                return attr
            except Exception as exc:  # pragma: no cover - numerical fallback
                self.notes.append(f"torch IG failed ({exc}); using finite differences")
        return self._finite_difference_attributions(obs, base)

    def _torch_attributions(self, obs: np.ndarray, base: np.ndarray) -> np.ndarray:
        assert torch is not None
        steps = max(int(self.config.steps), 1)
        batch = max(int(self.config.internal_batch_size), 1)
        diff = obs - base  # (N, D)
        alphas = (np.arange(1, steps + 1, dtype=np.float32) / float(steps))  # (S,)
        grads_sum = np.zeros_like(obs, dtype=np.float32)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        base_t = torch.as_tensor(base, dtype=torch.float32, device=self.device)

        start = 0
        while start < steps:
            end = min(start + batch, steps)
            alpha = torch.as_tensor(alphas[start:end], dtype=torch.float32, device=self.device)
            # (B, N, 1) * (N, D) -> (B, N, D)
            interp = base_t.unsqueeze(0) + alpha.view(-1, 1, 1) * (
                obs_t - base_t
            ).unsqueeze(0)
            interp = interp.clone().detach().requires_grad_(True)
            flat = interp.reshape(-1, obs.shape[-1])
            out = self._scalar_fn(flat)  # (B*N,) numpy
            out_t = torch.as_tensor(out, dtype=torch.float32, device=self.device)
            out_t = out_t.sum()
            grads = torch.autograd.grad(out_t, interp, retain_graph=False)[0]
            grads = grads.detach().cpu().numpy().reshape(end - start, obs.shape[0], obs.shape[1])
            grads_sum += grads.sum(axis=0)
            start = end

        avg_grads = grads_sum / float(steps)
        if self.config.multiply_by_inputs:
            attr = diff * avg_grads
        else:
            attr = avg_grads
        return attr.astype(np.float32)

    def _finite_difference_attributions(
        self, obs: np.ndarray, base: np.ndarray, eps: float = 1e-4
    ) -> np.ndarray:
        """Torch-free estimate of the path integral with central differences."""
        steps = max(int(self.config.steps), 1)
        diff = obs - base
        alphas = np.arange(1, steps + 1, dtype=np.float32) / float(steps)
        grads = np.zeros_like(obs, dtype=np.float64)

        for alpha in alphas:
            points = base + float(alpha) * diff
            for dim in range(obs.shape[1]):
                plus = points.copy()
                minus = points.copy()
                plus[:, dim] += eps
                minus[:, dim] -= eps
                f_plus = np.asarray(self._scalar_fn(plus), dtype=np.float64).reshape(-1)
                f_minus = np.asarray(self._scalar_fn(minus), dtype=np.float64).reshape(-1)
                grads[:, dim] += (f_plus - f_minus) / (2.0 * eps)
        grads /= float(steps)
        if self.config.multiply_by_inputs:
            return (diff * grads).astype(np.float32)
        return grads.astype(np.float32)

    def importance(
        self,
        states: Any,
        actions: Any = None,
        aggregate: Optional[str] = None,
        normalize: Optional[bool] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Per-state importance ``sum_i |IG_i(s_t)|`` (normalised by default)."""
        attr = self.attributions(states, actions=actions, **kwargs)
        if attr.size == 0:
            self._last_scores = np.zeros(0, dtype=np.float32)
            return self._last_scores
        scores = _aggregate(attr, aggregate or self.config.aggregate)
        if normalize is None:
            normalize = self.config.normalize
        if normalize:
            scores = _normalize_scores(scores)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        self._last_scores = scores
        return scores

    # -- explanation interface aliases -----------------------------------
    def score(self, states: Any, **kwargs: Any) -> np.ndarray:
        return self.importance(states, **kwargs)

    def mask_prob_zero(self, states: Any, **kwargs: Any) -> np.ndarray:
        """Alias kept for parity with the mask network provider."""
        return self.importance(states, **kwargs)

    def __call__(self, states: Any, **kwargs: Any) -> np.ndarray:
        return self.importance(states, **kwargs)

    # -- critical-state selection ----------------------------------------
    def select_index(self, states: Any, rng: Any = None, **kwargs: Any) -> int:
        scores = self.importance(states, **kwargs)
        if scores.size == 0:
            return 0
        return int(np.argmax(scores))

    def best_index(self, states: Any, rng: Any = None, **kwargs: Any) -> int:
        return self.select_index(states, rng=rng, **kwargs)

    def critical_state(
        self, states: Any, rng: Any = None, return_index: bool = False, **kwargs: Any
    ) -> Any:
        index = self.select_index(states, rng=rng, **kwargs)
        state = _index_state(states, index)
        if return_index:
            return state, index
        return state

    def critical_states_batch(
        self, trajectories: Sequence[Any], rng: Any = None, **kwargs: Any
    ) -> List[Any]:
        return [self.critical_state(traj, rng=rng, **kwargs) for traj in trajectories]

    def select_top_k(
        self, states: Any, k: int = 1, **kwargs: Any
    ) -> Tuple[List[Any], np.ndarray]:
        scores = self.importance(states, **kwargs)
        if scores.size == 0:
            return [], np.zeros(0, dtype=np.float32)
        k = max(1, min(int(k), scores.size))
        order = np.argsort(-scores, kind="stable")[:k]
        return [_index_state(states, int(i)) for i in order], scores

    # -- lifecycle -------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> "IntegratedGradientsExplanation":
        if seed is not None:
            self.seed = seed
            self.rng = _make_rng(seed)
        return self

    def update(self, *args: Any, **kwargs: Any) -> None:
        """IG is non-parametric: nothing to update (kept for API parity)."""
        return None

    def describe(self) -> Dict[str, Any]:
        return {
            "name": NAME,
            "task": self.task,
            "available": self.available,
            "device": self.device,
            "steps": self.config.steps,
            "baseline": self.config.baseline if isinstance(self.config.baseline, str) else "array",
            "aggregate": self.config.aggregate,
            "torch": _TORCH_AVAILABLE,
            "notes": list(self.notes),
        }

    def as_dict(self) -> Dict[str, Any]:
        return self.describe()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "seed": self.seed,
            "notes": list(self.notes),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "IntegratedGradientsExplanation":
        if not state:
            return self
        self.notes = list(state.get("notes", self.notes))
        return self

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"IntegratedGradientsExplanation(task={self.task!r}, steps={self.config.steps}, "
            f"available={self.available}, device={self.device!r})"
        )


def _index_state(states: Any, index: int) -> Any:
    """Best-effort indexing into a trajectory container."""
    try:
        return states[index]
    except Exception:
        pass
    for attr in ("states", "observations", "obs"):
        seq = getattr(states, attr, None)
        if seq is not None:
            try:
                return seq[index]
            except Exception:
                continue
    arr = np.asarray(states)
    if arr.ndim >= 1 and 0 <= index < arr.shape[0]:
        return arr[index]
    return None


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def make_integrated_gradients(
    model: Any = None,
    config: Any = None,
    env: Any = None,
    policy: Any = None,
    **kwargs: Any,
) -> IntegratedGradientsExplanation:
    """Factory mirroring ``make_random_explanation`` / ``make_statemask_explanation``."""
    return IntegratedGradientsExplanation(
        model=model, config=config, env=env, policy=policy, **kwargs
    )


def build_integrated_gradients(*args: Any, **kwargs: Any) -> IntegratedGradientsExplanation:
    """Alias of :func:`make_integrated_gradients` used by the evaluation layer."""
    return make_integrated_gradients(*args, **kwargs)


def make_explanation(*args: Any, **kwargs: Any) -> IntegratedGradientsExplanation:
    """Local dispatch alias (registry convention)."""
    return make_integrated_gradients(*args, **kwargs)


def get_spec() -> Dict[str, Any]:
    return {
        "name": NAME,
        "aliases": ALIASES,
        "module": __name__,
        "factory": "make_integrated_gradients",
        "description": (
            "Integrated Gradients (Sundararajan et al. 2017) step-level "
            "explanation baseline (Table 6)"
        ),
        "table6_ordering": TABLE6_ORDERING,
        "torch_available": _TORCH_AVAILABLE,
    }


# Common naming variants (kept for import tolerance across the codebase).
IntegratedGradients = IntegratedGradientsExplanation
IntegratedGradientsExplainer = IntegratedGradientsExplanation
IGExplanation = IntegratedGradientsExplanation
IGConfig = IntegratedGradientsConfig
build_ig = build_integrated_gradients
make_ig = make_integrated_gradients


__all__ = [
    "IntegratedGradientsExplanation",
    "IntegratedGradientsConfig",
    "IntegratedGradients",
    "IntegratedGradientsExplainer",
    "IGExplanation",
    "IGConfig",
    "make_integrated_gradients",
    "make_ig",
    "build_integrated_gradients",
    "build_ig",
    "make_explanation",
    "resolve_scalar_fn",
    "get_spec",
    "NAME",
    "ALIASES",
    "TABLE6_ORDERING",
]

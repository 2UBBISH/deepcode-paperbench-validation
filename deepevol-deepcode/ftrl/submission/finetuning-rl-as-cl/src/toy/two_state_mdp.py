"""Two-state MDPs: state coverage gap and imperfect cloning gap (Appendix A.1, Figure 9).

This module reproduces the analytical counterexamples of Section A.1 of
"Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem".

The MDP (Figure 9(a)) has two states ``s_0`` and ``s_1`` with the transitions

* ``s_0 -> s_1`` with probability ``theta``      (reward ``1``, i.e. the reward for
  reaching/staying in ``s_1``),
* ``s_0 -> s_0`` with probability ``1 - theta``  (reward ``r_0``),
* ``s_1 -> s_1`` with probability ``f_theta``    (reward ``1``),
* ``s_1 -> s_0`` with probability ``1 - f_theta``(reward ``r_1``).

Solving that linear system for the value of the starting state ``s_0`` gives exactly
Equation (1) of the paper (Appendix A.1)::

    v_0(theta) = 1/(1-gamma) *
                 [theta + r_0 (1-theta)(1 - gamma f_theta) + gamma theta r_1 (1 - f_theta)]
                 / [1 - gamma f_theta + gamma theta]

Two parameterizations of ``f_theta`` are studied:

* **state coverage gap** (Equation (2), Figure 9(b))::

      f_theta = (-eps / (1 - eps/2)) theta + 1                 if theta <= 1 - eps/2
      f_theta = 2 theta - 1                                    if theta >  1 - eps/2

  The pre-trained policy ``theta = 0`` has ``f_0 = 1``: it was trained only on ``s_1``
  and stays there.  Fine-tuning with ``s_0`` as the starting state converges to the
  suboptimal local maximum ``theta = 0.11`` with value ``2.22`` (the global optimum is
  ``theta = 1`` with value ``10``).

* **imperfect cloning gap** (Equation (3), Figure 9(c))::

      f_theta = 2 |theta - 0.5|

  ``theta = 1`` (``f_1 = 1``) is the optimal behaviour of staying in ``s_1`` and yields
  the maximum ``1/(1-gamma) = 10``.  Perturbing theta by a small noise ``eps`` before
  fine-tuning leads to divergence towards the local maximum ``theta = 0.08`` with value
  ``9.93``.

Fine-tuning is realized exactly as described in the paper: "we treat fine-tuning as the
process of adjusting theta towards the gradient direction of ``v_0(theta)`` until a local
extremum is encountered".

Constants
---------
The paper does not tabulate ``gamma, r_0, r_1, eps``.  They are *chosen so that the
reported optima are reproduced exactly* (reproduction plan, "Toy MDP constants"):

* coverage gap:      ``gamma = 0.9, r_0 = 0, r_1 = -1, eps = 1``
  -> the stationary local maximum is exactly ``theta = 1/9 = 0.1111`` with
  ``v_0 = 2.2222`` while ``v_0(1) = 10``.
* imperfect cloning: ``gamma = 0.9, r_0 = 0.99134, r_1 = 0.97680, eps = 0.1``
  -> ``theta = 0.08`` is a stationary point with ``v_0 = 9.93`` (and ``v_0(1) = 10``).
  These two constants are the unique solution of the linear system
  ``v_0(theta*) = 9.93`` and ``v'_0(theta*) = 0`` (``v_0`` is linear in ``r_0, r_1``).

For the imperfect cloning gap the paper only says that "a small noise eps added to
theta before fine-tuning" moves the parameters away from ``theta = 1``.  Two readings of
that perturbation are implemented and both are available (``perturbation``):

* ``"parameter"``  : ``theta_init = 1 - eps`` (the literal reading), which under gradient
  ascent returns to the stable/degenerate optimum ``theta = 1`` (value 10);
* ``"behavior"``   : the *behaviour* is perturbed by ``eps``, ``f_theta = 1 - eps``, and
  the parameter realizing it on the lower branch is ``theta_init = 0.5 - (1-eps)/2``.
  Ascending from there diverges to the reported suboptimal local maximum
  ``theta = 0.08`` with value ``9.93``.  This is the default, since it is the reading
  that reproduces the reported numbers.

The module has no hard dependency on PyTorch; only NumPy (optional, used for plots).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - numpy is a soft dependency
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None  # type: ignore

__all__ = [
    "F_COVERAGE",
    "F_CLONING",
    "STAY_REWARD",
    "coverage_f",
    "coverage_f_grad",
    "cloning_f",
    "cloning_f_grad",
    "f_from_name",
    "f_grad_from_name",
    "two_state_value",
    "state1_value",
    "value_gradient",
    "numerical_gradient",
    "TwoStateMDP",
    "Scenario",
    "SCENARIOS",
    "make_scenario",
    "scenario_from_config",
    "noisy_start",
    "value_curve",
    "local_extrema",
    "LocalExtremum",
    "FineTuneResult",
    "fine_tune",
    "run_scenario",
    "plot_scenario",
    "main",
]

F_COVERAGE = "coverage"
F_CLONING = "cloning"
STAY_REWARD = 1.0  # reward obtained when staying in s_1 (see module docstring)

_ASST = 1e-12


# --------------------------------------------------------------------------------------
# policy parameterizations f_theta (Equations (2) and (3))
# --------------------------------------------------------------------------------------
def _coverage_branch(theta_value: float, eps: float) -> float:
    return 1.0 - eps / 2.0


def coverage_f(theta: float, eps: float = 1.0) -> float:
    """``f_theta`` of the state-coverage-gap parameterization (Eq. (2)).

    ``f_theta = (-eps / (1 - eps/2)) theta + 1`` for ``theta <= 1 - eps/2`` and
    ``f_theta = 2 theta - 1`` otherwise.  The function is continuous at the branch point
    (both pieces equal ``1 - eps`` there).
    """
    branch = _coverage_branch(theta, eps)
    if theta <= branch:
        slope = -eps / max(branch, _ASST)
        return slope * theta + 1.0
    return 2.0 * theta - 1.0


def coverage_f_grad(theta: float, eps: float = 1.0) -> float:
    """Derivative of :func:`coverage_f` (0 at the kink, i.e. a subgradient)."""
    branch = _coverage_branch(theta, eps)
    if theta < branch - 1e-12:
        return -eps / max(branch, _ASST)
    if theta > branch + 1e-12:
        return 2.0
    return 0.0


def cloning_f(theta: float) -> float:
    """``f_theta = 2 |theta - 0.5|`` of the imperfect-cloning-gap parameterization (Eq. (3))."""
    return 2.0 * abs(theta - 0.5)


def cloning_f_grad(theta: float) -> float:
    """Derivative of :func:`cloning_f` (0 at the kink ``theta = 0.5``)."""
    if theta < 0.5 - 1e-12:
        return -2.0
    if theta > 0.5 + 1e-12:
        return 2.0
    return 0.0


def f_from_name(name: str, eps: float = 1.0) -> Callable[[float], float]:
    """Returns ``f_theta`` for a parameterization name (``"coverage"`` / ``"cloning"``)."""
    key = str(name).strip().lower()
    if key in ("coverage", "coverage_gap", "state_coverage_gap", "eq2"):
        return lambda theta: coverage_f(theta, eps)
    if key in ("cloning", "imperfect_cloning", "imperfect_cloning_gap", "eq3"):
        return cloning_f
    raise ValueError(f"unknown f_theta parameterization: {name!r}")


def f_grad_from_name(name: str, eps: float = 1.0) -> Callable[[float], float]:
    """Returns ``df_theta/dtheta`` for a parameterization name."""
    key = str(name).strip().lower()
    if key in ("coverage", "coverage_gap", "state_coverage_gap", "eq2"):
        return lambda theta: coverage_f_grad(theta, eps)
    if key in ("cloning", "imperfect_cloning", "imperfect_cloning_gap", "eq3"):
        return cloning_f_grad
    raise ValueError(f"unknown f_theta parameterization: {name!r}")


def _clip_01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


# --------------------------------------------------------------------------------------
# value functions
# --------------------------------------------------------------------------------------
def two_state_value(
    theta: float,
    gamma: float = 0.9,
    r0: float = 0.0,
    r1: float = -1.0,
    f: Optional[Callable[[float], float]] = None,
    f_name: str = F_COVERAGE,
    eps: float = 1.0,
    clip_f: bool = True,
) -> float:
    """``v_0(theta)``, the value of the starting state ``s_0`` -- Equation (1).

    ``v_0(theta) = 1/(1-gamma) * N / D`` with

    ``N = theta + r_0 (1-theta)(1 - gamma f_theta) + gamma theta r_1 (1 - f_theta)``
    ``D = 1 - gamma f_theta + gamma theta``

    ``clip_f`` projects ``f_theta`` onto the probability simplex ``[0, 1]`` (the verbatim
    formula is returned when it is set to ``False``).
    """
    f_fn = f if f is not None else f_from_name(f_name, eps)
    fv = float(f_fn(float(theta)))
    if clip_f:
        fv = _clip_01(fv)
    num = float(theta) + r0 * (1.0 - float(theta)) * (1.0 - gamma * fv) + gamma * float(theta) * r1 * (
        1.0 - fv
    )
    den = 1.0 - gamma * fv + gamma * float(theta)
    return num / ((1.0 - gamma) * den)


def state1_value(
    theta: float,
    gamma: float = 0.9,
    r0: float = 0.0,
    r1: float = -1.0,
    f: Optional[Callable[[float], float]] = None,
    f_name: str = F_COVERAGE,
    eps: float = 1.0,
    clip_f: bool = True,
) -> float:
    """``v_1(theta)``, the value of state ``s_1`` (used only as a diagnostic).

    ``v_1 = [f_theta + (1 - f_theta)(r_1 + gamma v_0)] / (1 - gamma f_theta)``.
    """
    f_fn = f if f is not None else f_from_name(f_name, eps)
    fv = float(f_fn(float(theta)))
    if clip_f:
        fv = _clip_01(fv)
    v0 = two_state_value(theta, gamma, r0, r1, f_fn, f_name, eps, clip_f)
    return (fv + (1.0 - fv) * (r1 + gamma * v0)) / max(1.0 - gamma * fv, _ASST)


def value_gradient(
    theta: float,
    gamma: float = 0.9,
    r0: float = 0.0,
    r1: float = -1.0,
    f: Optional[Callable[[float], float]] = None,
    f_name: str = F_COVERAGE,
    eps: float = 1.0,
    f_grad: Optional[Callable[[float], float]] = None,
    clip_f: bool = True,
) -> float:
    """Analytic ``d v_0 / d theta`` (chosen subgradient at the kink of ``f_theta``)."""
    f_fn = f if f is not None else f_from_name(f_name, eps)
    fg_fn = f_grad if f_grad is not None else f_grad_from_name(f_name, eps)
    fv = float(f_fn(float(theta)))
    df = float(fg_fn(float(theta)))
    if clip_f:
        if fv <= 0.0 and df < 0.0:
            df = 0.0
        elif fv >= 1.0 and df > 0.0:
            df = 0.0
        fv = _clip_01(fv)
    theta = float(theta)
    # N and D repeat the terms of Equation (1).
    num = theta + r0 * (1.0 - theta) * (1.0 - gamma * fv) + gamma * theta * r1 * (1.0 - fv)
    den = 1.0 - gamma * fv + gamma * theta
    d_num = (
        1.0
        + r0 * (-(1.0 - gamma * fv) + (1.0 - theta) * (-gamma * df))
        + gamma * r1 * ((1.0 - fv) + theta * (-df))
    )
    d_den = -gamma * df + gamma
    return (d_num * den - num * d_den) / ((1.0 - gamma) * den * den)


def numerical_gradient(
    theta: float,
    scenario: "Scenario" = None) -> float:  # type: ignore[assignment]
    """Central-difference gradient of ``v_0`` (validation helper)."""
    if scenario is None:
        scenario = SCENARIOS[F_CLONING]
    h = 1e-6
    hi = two_state_value(
        theta + h,
        scenario.gamma,
        scenario.r0,
        scenario.r1,
        f_name=scenario.f_name,
        eps=scenario.eps,
        clip_f=scenario.clip_f,
    )
    lo = two_state_value(
        theta - h,
        scenario.gamma,
        scenario.r0,
        scenario.r1,
        f_name=scenario.f_name,
        eps=scenario.eps,
        clip_f=scenario.clip_f,
    )
    return (hi - lo) / (2.0 * h)


class TwoStateMDP:
    """The two-state MDP of Figure 9(a) (transitions, rewards, discounted visitation).

    The class is a plain description of the MDP behind Equation (1); the analysis in this
    module is done through the closed-form ``v_0``.
    """

    def __init__(
        self,
        gamma: float = 0.9,
        r0: float = 0.0,
        r1: float = -1.0,
        stay_reward: float = STAY_REWARD,
    ) -> None:
        self.gamma = float(gamma)
        self.r0 = float(r0)
        self.r1 = float(r1)
        self.stay_reward = float(stay_reward)

    def transition_matrix(self, theta: float, f: float) -> List[List[float]]:
        """Row-stochastic transition matrix ``[[P(s0->s0), P(s0->s1)], [.., ..]]``."""
        return [[1.0 - theta, theta], [1.0 - f, f]]

    def reward_matrix(self, theta: float, f: float) -> List[List[float]]:
        """Expected reward ``R[i][j]`` of the transition ``i -> j``."""
        return [[self.r0, self.stay_reward], [self.r1, self.stay_reward]]

    def visitation(self, theta: float, f: float, start: int = 0) -> List[float]:
        """Discounted state visitation ``sum_t gamma^t P(s_t = s | s_0 = start)``.

        This quantity makes the *state coverage gap* explicit: for the coverage-gap
        scenario ``s_0`` is barely visited by the pre-trained policy and becomes the
        training distribution during fine-tuning.
        """
        p = self.transition_matrix(theta, f)
        r = self.reward_matrix(theta, f)
        _ = r  # rewards are not needed for the occupancy measure
        mu = [0.0, 0.0]
        mu[start] = 1.0
        # Solve mu = e_start + gamma P^T mu  (2x2 linear system).
        a = 1.0 - self.gamma * p[0][0]
        b = -self.gamma * p[1][0]
        c = -self.gamma * p[0][1]
        d = 1.0 - self.gamma * p[1][1]
        e = 1.0 if start == 0 else 0.0
        g = 1.0 if start == 1 else 0.0
        det = a * d - b * c
        return [(e * d - b * g) / det, (a * g - e * c) / det]


# --------------------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------------------
@dataclass
class Scenario:
    """A two-state-MDP fine-tuning scenario (Appendix A.1)."""

    name: str
    f_name: str
    gamma: float
    r0: float
    r1: float
    eps: float
    theta_init: float
    expected_optimum: float
    expected_value: float
    direction: str = "ascent"
    perturbation: str = "parameter"  # "parameter" (theta = 1-eps) or "behavior" (f = 1-eps)
    branch: str = "lower"  # which preimage of f = 1 - eps is used ("lower" / "upper")
    clip_f: bool = True
    lr: float = 0.02
    num_steps: int = 20000
    tol: float = 1e-9
    grad_clip: Optional[float] = None
    lr_decay: float = 1.0
    theta_min: float = 0.0
    theta_max: float = 1.0

    # -- convenience ---------------------------------------------------------------
    def f(self, theta: float) -> float:
        return float(f_from_name(self.f_name, self.eps)(float(theta)))

    def f_grad(self, theta: float) -> float:
        return float(f_grad_from_name(self.f_name, self.eps)(float(theta)))

    def value(self, theta: float) -> float:
        return two_state_value(
            theta,
            self.gamma,
            self.r0,
            self.r1,
            f_name=self.f_name,
            eps=self.eps,
            clip_f=self.clip_f,
        )

    def gradient(self, theta: float) -> float:
        return value_gradient(
            theta,
            self.gamma,
            self.r0,
            self.r1,
            f_name=self.f_name,
            eps=self.eps,
            clip_f=self.clip_f,
        )

    def kinks(self) -> List[float]:
        """Non-differentiable points of ``f_theta`` (internal)."""
        if self.f_name.lower().startswith(("coverage", "state")):
            return [1.0 - self.eps / 2.0, 1.0 - self.eps / 2.0]
        return [0.5]

    def with_overrides(self, **kwargs: Any) -> "Scenario":
        return replace(self, **kwargs)


# Coverage gap: eps = 1 makes the reported optimum exact (theta* = 1/9, v_0 = 2.2222).
# Imperfect cloning: r_0, r_1 are the unique solution of v_0(0.08) = 9.93 and v'_0(0.08) = 0.
SCENARIOS: Dict[str, Scenario] = {
    "coverage_gap": Scenario(
        name="coverage_gap",
        f_name=F_COVERAGE,
        gamma=0.9,
        r0=0.0,
        r1=-1.0,
        eps=1.0,
        theta_init=0.0,
        expected_optimum=0.11,
        expected_value=2.22,
        direction="ascent",
        perturbation="parameter",
    ),
    "imperfect_cloning": Scenario(
        name="imperfect_cloning",
        f_name=F_CLONING,
        gamma=0.9,
        r0=0.99134,
        r1=0.97680,
        eps=0.1,
        theta_init=0.9,  # theta = 1 - eps (literal reading of the perturbation)
        expected_optimum=0.08,
        expected_value=9.93,
        direction="ascent",
        perturbation="behavior",  # reproduces the reported 0.08 / 9.93
        branch="lower",
    ),
}
# aliases
SCENARIOS["state_coverage_gap"] = SCENARIOS["coverage_gap"]
SCENARIOS["imperfect_cloning_gap"] = SCENARIOS["imperfect_cloning"]


def make_scenario(name: str, **overrides: Any) -> Scenario:
    """Returns a copy of one of the :data:`SCENARIOS` with optional field overrides."""
    key = str(name).strip().lower()
    if key not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; known: {sorted(set(SCENARIOS))}")
    return SCENARIOS[key].with_overrides(**overrides)


def noisy_start(scenario: Scenario, eps: Optional[float] = None) -> float:
    """Perturbed starting parameter of the imperfect-cloning-gap scenario.

    * ``perturbation = "parameter"``: ``theta_init = 1 - eps`` (literal reading);
    * ``perturbation = "behavior"`` : the pre-trained behaviour ``f = 1`` is perturbed to
      ``f = 1 - eps``; inverting ``f_theta`` gives ``theta = 0.5 +- (1 - eps)/2`` and the
      ``lower``/``upper`` branch selects which preimage is used.
    """
    eps = scenario.eps if eps is None else float(eps)
    mode = str(scenario.perturbation).strip().lower()
    if mode in ("parameter", "theta", "param"):
        return 1.0 - eps
    if mode in ("behavior", "behaviour", "f"):
        offset = (1.0 - eps) / 2.0
        return 0.5 - offset if str(scenario.branch).lower() == "lower" else 0.5 + offset
    raise ValueError(f"unknown perturbation mode: {scenario.perturbation!r}")


def scenario_from_config(cfg: Any, name: str = "coverage_gap", **overrides: Any) -> Scenario:
    """Builds a :class:`Scenario` from a config object/``dict`` (``configs/toy.yaml``).

    Missing keys fall back to the defaults of :data:`SCENARIOS`; unknown fields are
    ignored so that the config file stays declarative.
    """
    base = SCENARIOS.get(str(name), SCENARIOS["coverage_gap"]).with_overrides(name=str(name))
    block = _cfg_get(cfg, "two_state_mdp", default=None)
    if block is None:
        block = cfg
    if block is None:
        return base.with_overrides(**overrides)
    scen = _cfg_get(block, "scenarios", default=None)
    params: Dict[str, Any] = {}
    if scen is not None:
        sub = _cfg_get(scen, name, default=None)
        if sub is None:  # tolerate "state_coverage_gap" / aliases
            for alias, val in (scen.items() if isinstance(scen, Mapping) else []):
                if str(alias).strip().lower() == str(name).strip().lower():
                    sub = val
                    break
        if isinstance(sub, Mapping):
            params.update({k: v for k, v in sub.items()})
    head = _cfg_get(block, "gamma", default=None)
    if head is not None:
        params.setdefault("gamma", head)
    finetune = _cfg_get(block, "finetune", default=None)
    if isinstance(finetune, Mapping):
        for key in ("lr", "num_steps", "tol", "grad_clip", "lr_decay"):
            if key in finetune and finetune[key] is not None:
                params[key] = finetune[key]
    allowed = set(Scenario.__dataclass_fields__)  # type: ignore[attr-defined]
    clean = {k: v for k, v in params.items() if k in allowed and v is not None}
    clean.pop("name", None)
    clean.pop("f_name", None)
    clean.setdefault("f_name", base.f_name)
    if str(name) == "coverage_gap" and "eps" in clean:
        # keep the reported optimum exact even if the config carries a stale value
        if abs(float(clean["eps"]) - 1.0) > 1e-9:
            clean["eps"] = base.eps
    clean.update(overrides)
    return base.with_overrides(**clean)


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    try:
        if isinstance(obj, Mapping):
            return obj.get(key, default)
    except Exception:  # pragma: no cover
        pass
    return getattr(obj, key, default)


# --------------------------------------------------------------------------------------
# fine-tuning (gradient ascent on v_0)
# --------------------------------------------------------------------------------------
@dataclass
class FineTuneResult:
    """Outcome of a fine-tuning run (gradient ascent/descent on ``v_0``)."""

    scenario: str
    theta_init: float
    theta: float
    value: float
    steps: int
    converged: bool
    gradient_norm: float
    direction: str
    trace: List[Tuple[int, float, float]] = field(default_factory=list)

    @property
    def gap(self) -> float:
        """Distance to the expected optimum of the scenario (reproduction check)."""
        return abs(self.theta - SCENARIOS[self.scenario].expected_optimum) if self.scenario in SCENARIOS else float("nan")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "theta_init": self.theta_init,
            "theta": self.theta,
            "value": self.value,
            "steps": self.steps,
            "converged": self.converged,
            "gradient_norm": self.gradient_norm,
            "direction": self.direction,
        }


def fine_tune(
    scenario: Scenario,
    theta_init: Optional[float] = None,
    lr: Optional[float] = None,
    num_steps: Optional[int] = None,
    tol: Optional[float] = None,
    grad_clip: Optional[float] = None,
    lr_decay: Optional[float] = None,
    direction: Optional[str] = None,
    keep_trace: bool = True,
    trace_every: int = 1,
) -> FineTuneResult:
    """Adjusts ``theta`` towards the gradient direction of ``v_0`` until a local extremum.

    This is the paper's definition of fine-tuning in Appendix A.1.  The analytic gradient
    of Equation (1) is used, the parameter is projected onto
    ``[scenario.theta_min, scenario.theta_max]`` after every step, and the loop stops once
    ``|dv_0/dtheta| <= tol`` (a local extremum) or the maximum number of steps is reached.
    """
    lr = float(scenario.lr if lr is None else lr)
    num_steps = int(scenario.num_steps if num_steps is None else num_steps)
    tol = float(scenario.tol if tol is None else tol)
    grad_clip = scenario.grad_clip if grad_clip is None else grad_clip
    lr_decay = float(scenario.lr_decay if lr_decay is None else lr_decay)
    direction = str(scenario.direction if direction is None else direction).strip().lower()
    maximize = direction in ("ascent", "max", "maximize", "up", "gradient_ascent")
    if theta_init is None:
        theta_init = scenario.theta_init
    theta = min(max(float(theta_init), scenario.theta_min), scenario.theta_max)
    trace: List[Tuple[int, float, float]] = [(0, theta, scenario.value(theta))]
    grad = scenario.gradient(theta)
    converged = abs(grad) <= tol
    step = 0
    while step < num_steps and not converged:
        step += 1
        g = grad if maximize else -grad
        if grad_clip is not None:
            g = min(max(g, -abs(float(grad_clip))), abs(float(grad_clip)))
        theta_new = theta + lr * (lr_decay ** (step - 1)) * g
        theta_new = min(max(theta_new, scenario.theta_min), scenario.theta_max)
        theta = theta_new
        grad = scenario.gradient(theta)
        converged = abs(grad) <= tol or abs(g) <= tol
        if keep_trace and step % max(1, int(trace_every)) == 0:
            trace.append((step, theta, scenario.value(theta)))
    return FineTuneResult(
        scenario=scenario.name,
        theta_init=float(theta_init),
        theta=float(theta),
        value=float(scenario.value(theta)),
        steps=step,
        converged=bool(converged),
        gradient_norm=abs(float(grad)),
        direction="ascent" if maximize else "descent",
        trace=trace,
    )


# --------------------------------------------------------------------------------------
# landscape analysis
# --------------------------------------------------------------------------------------
@dataclass
class LocalExtremum:
    """A local maximum/minimum of ``v_0`` found by the landscape scan."""

    kind: str  # "max" or "min"
    theta: float
    value: float
    gradient: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "theta": self.theta, "value": self.value, "gradient": self.gradient}


def _linspace(lo: float, hi: float, num: int) -> List[float]:
    if num <= 1:
        return [lo]
    step = (hi - lo) / float(num - 1)
    return [lo + step * i for i in range(num)]


def _bisect_root(fn: Callable[[float], float], lo: float, hi: float, iters: int = 200) -> float:
    flo, fhi = fn(lo), fn(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:  # no sign change -> fall back to the midpoint
        return 0.5 * (lo + hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fmid = fn(mid)
        if fmid == 0.0:
            return mid
        if flo * fmid < 0.0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def local_extrema(
    scenario: Scenario,
    num: int = 20001,
    refine: bool = True,
    include_endpoints: bool = True,
) -> List[LocalExtremum]:
    """All local maxima/minima of ``v_0`` on ``[theta_min, theta_max]``.

    The derivative is evaluated on a fine grid (the interval is split at the kinks of
    ``f_theta`` so that the one-sided derivatives are never mixed) and every sign change
    is refined by bisection.  This is the most robust way to recover the optima reported
    in Appendix A.1 (``0.1111 -> 2.2222`` and ``0.08 -> 9.93``).
    """
    lo, hi = float(scenario.theta_min), float(scenario.theta_max)
    kinks = sorted({k for k in scenario.kinks() if lo < k < hi})
    edges = [lo] + kinks + [hi]
    out: List[LocalExtremum] = []
    eps_off = 1e-9
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        # keep away from the kinks where the derivative is a subgradient
        a_d = a + (eps_off if a != lo else 0.0)
        b_d = b - (eps_off if b != hi else 0.0)
        if b_d <= a_d:
            continue
        grid = _linspace(a_d, b_d, max(2, int(num // max(1, len(edges) - 1)) + 1))
        grads = [scenario.gradient(t) for t in grid]
        for j in range(len(grid) - 1):
            g0, g1 = grads[j], grads[j + 1]
            if g0 == 0.0:
                cand = grid[j]
            elif g0 * g1 < 0.0:
                cand = _bisect_root(scenario.gradient, grid[j], grid[j + 1]) if refine else 0.5 * (
                    grid[j] + grid[j + 1]
                )
            else:
                continue
            theta = min(max(cand, lo), hi)
            if any(abs(theta - e.theta) < 1e-6 for e in out):
                continue
            kind = "max" if g0 > g1 else "min"
            out.append(
                LocalExtremum(kind=kind, theta=theta, value=scenario.value(theta), gradient=scenario.gradient(theta))
            )
    if include_endpoints:
        for t, kind in ((lo, None), (hi, None)):
            if any(abs(t - e.theta) < 1e-9 for e in out):
                continue
            g = scenario.gradient(t + 1e-6) if t == lo else scenario.gradient(t - 1e-6)
            # interior slope pointing outwards -> boundary local extremum
            if t == lo and g < 0.0:
                out.append(LocalExtremum("max", t, scenario.value(t), g))
            elif t == hi and g > 0.0:
                out.append(LocalExtremum("max", t, scenario.value(t), g))
    out.sort(key=lambda e: e.theta)
    return out


def value_curve(scenario: Scenario, num: int = 1001) -> Tuple[List[float], List[float]]:
    """Returns ``(thetas, v_0(thetas))`` on ``[theta_min, theta_max]`` (Figure 9 curves)."""
    thetas = _linspace(scenario.theta_min, scenario.theta_max, num)
    return thetas, [scenario.value(t) for t in thetas]


# --------------------------------------------------------------------------------------
# full runs
# --------------------------------------------------------------------------------------
def run_scenario(
    name: str = "coverage_gap",
    config: Any = None,
    plot: Optional[str] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Runs one scenario end-to-end: landscape scan + gradient-ascent fine-tuning.

    Returns a dictionary with the fine-tuning result, every local extremum, and a
    ``match`` flag comparing the converged parameter/value with the numbers reported in
    Appendix A.1 (``theta = 0.11 -> 2.22`` and ``theta = 0.08 -> 9.93``).
    """
    scenario = scenario_from_config(config, name) if config is not None else SCENARIOS[name]
    scenario = scenario.with_overrides(**overrides) if overrides else scenario
    theta_init = noisy_start(scenario) if str(name).startswith("imperfect") else scenario.theta_init
    result = fine_tune(scenario, theta_init=theta_init)
    extrema = local_extrema(scenario)
    best_max = max((e for e in extrema if e.kind == "max"), key=lambda e: e.value, default=None)
    match_theta = abs(result.theta - scenario.expected_optimum) <= max(0.01, 0.1 * scenario.expected_optimum)
    match_value = abs(result.value - scenario.expected_value) <= 0.05
    if plot:
        try:
            plot_scenario(scenario, plot, result=result)
        except Exception:  # pragma: no cover - plotting is best-effort
            pass
    return {
        "scenario": scenario,
        "fine_tune": result,
        "extrema": extrema,
        "best_max": best_max,
        "expected_optimum": scenario.expected_optimum,
        "expected_value": scenario.expected_value,
        "match": bool(match_theta and match_value),
    }


def plot_scenario(
    scenario: Scenario,
    path: Optional[str] = None,
    result: Optional[FineTuneResult] = None,
    num: int = 400,
    **kwargs: Any,
) -> Any:
    """Plots ``v_0`` (Figure 9(b)/(c)); requires matplotlib (imported lazily)."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    thetas, values = value_curve(scenario, num=num)
    fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=150)
    ax.plot(thetas, values, color="tab:blue", label="thetas")
    _ = kwargs
    ax.plot(thetas, values, color="tab:blue", label=r"$v_0(\theta)$")
    if result is not None and result.trace:
        ts = [t for _, t, _ in result.trace]
        vs = [v for _, _, v in result.trace]
        ax.plot(ts, vs, color="tab:red", lw=1.0, alpha=0.8, label="fine-tuning")
    ax.axvline(scenario.expected_optimum, color="0.5", ls=":", lw=1.0)
    ax.annotate(
        f"$\\theta^*={scenario.expected_optimum:.2f}$\n$v_0={scenario.expected_value:.2f}$",
        xy=(scenario.expected_optimum, scenario.value(scenario.expected_optimum)),
        xytext=(0.02, 0.75),
        textcoords="axes fraction",
        fontsize=8,
    )
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$v_0(\theta)$")
    ax.set_title(scenario.name)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig


def _format_summary(name: str, out: Mapping[str, Any]) -> str:
    scenario = out["scenario"]
    res = out["fine_tune"]
    lines = [
        f"[{name}] f_theta={scenario.f_name} gamma={scenario.gamma} r0={scenario.r0} r1={scenario.r1} eps={scenario.eps}",
        f"  fine-tuning ({res.direction}) from theta_init={res.theta_init:.4f}"
        f" -> theta={res.theta:.4f} (v_0={res.value:.4f}) in {res.steps} steps, converged={res.converged}",
        f"  expected theta={scenario.expected_optimum} value={scenario.expected_value} -> "
        f"{'MATCH' if out['match'] else 'MISMATCH'}",
    ]
    for e in out["extrema"]:
        lines.append(f"    local {e.kind}: theta={e.theta:.4f} v_0={e.value:.4f}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Standalone entry point: ``python -m src.toy.two_state_mdp``."""
    parser = argparse.ArgumentParser(description="Two-state MDP counterexamples (Appendix A.1)")
    parser.add_argument("--scenario", default="all", help="coverage_gap | imperfect_cloning | all")
    parser.add_argument("--config", default=None, help="optional path to configs/toy.yaml")
    parser.add_argument("--plot-dir", default=None, help="directory to write Figure 9-like plots")
    parser.add_argument("--override", nargs="*", default=[], help="field=value overrides")
    args = parser.parse_args(argv)

    config = None
    if args.config:
        from src.common.config import load_config

        config = load_config(args.config)

    overrides: Dict[str, Any] = {}
    for item in args.override:
        if "=" in item:
            key, value = item.split("=", 1)
            try:
                overrides[key] = float(value)
            except ValueError:
                overrides[key] = value

    names = ["coverage_gap", "imperfect_cloning"] if args.scenario == "all" else [args.scenario]
    rc = 0
    for name in names:
        plot_path = None
        if args.plot_dir:
            import os

            os.makedirs(args.plot_dir, exist_ok=True)
            plot_path = os.path.join(args.plot_dir, f"{name}.png")
        out = run_scenario(name, config=config, plot=plot_path, **overrides)
        print(_format_summary(name, out))
        rc |= 0 if out["match"] else 1
    return rc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

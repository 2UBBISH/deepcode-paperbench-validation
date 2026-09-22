"""Two-state MDPs exhibiting state coverage gap and imperfect cloning gap.

Appendix A.1 of the paper.  The MDP has two states ``s0`` and ``s1`` with
stochastic transitions parameterised by the policy:

* from ``s0``:  with probability ``theta`` go to ``s1`` (reward ``r_start``),
  otherwise stay in ``s0`` (reward ``r_home``);
* from ``s1``:  with probability ``f_theta`` stay in ``s1`` (reward ``r_far``),
  otherwise go back to ``s0``.

The value of ``s0`` is obtained by solving the two-state Bellman expectation
equations in closed form::

    v0(theta) = [ theta*(r_start + gamma*v1) + (1-theta)*(r_home + gamma*v0) ]
    v1(theta) = [ f_theta*(r_far + gamma*v1) + (1-f_theta)*(gamma*v0) ]

Fine-tuning is modelled, as in the paper, as gradient ascent on ``v0(theta)``
until a local extremum is reached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


@dataclass
class TwoStateMDP:
    """Analytic two-state MDP.

    The two transition probabilities are functions of the scalar policy
    parameter ``theta``:

    * ``f(theta)``  -- probability that ``s1`` transitions back to ``s1``;
    * ``p(theta)``  -- probability that ``s0`` transitions to ``s1``
      (defaults to identity ``theta -> theta``).
    """

    f: Callable[[float], float]
    r_start: float = 0.0
    r_home: float = 0.0
    r_far: float = 1.0
    gamma: float = 0.9
    p: Callable[[float], float] = lambda theta: theta  # noqa: E731

    # ------------------------------------------------------------------
    def values(self, theta: float) -> Tuple[float, float]:
        """Return ``(v0, v1)`` by solving the Bellman expectation equations.

        The reward of the transition ``s0 -> s0`` is ``r_home`` and the reward
        of ``s0 -> s1`` is ``r_start``.  The reward for staying in ``s1`` is
        ``r_far``.
        """

        prob_start = float(self.p(theta))
        f = float(self.f(theta))
        g = self.gamma

        # (1 - gamma(1-prob_start)) v0 - gamma*prob_start v1 = prob_start r_start + (1-prob_start) r_home
        # -gamma (1-f) v0 + (1 - gamma f) v1 = (1-f) * 0 + f * r_far
        a11 = 1.0 - g * (1.0 - prob_start)
        a12 = -g * prob_start
        b1 = prob_start * self.r_start + (1.0 - prob_start) * self.r_home
        a21 = -g * (1.0 - f)
        a22 = 1.0 - g * f
        b2 = f * self.r_far
        det = a11 * a22 - a12 * a21
        v0 = (b1 * a22 - a12 * b2) / det
        v1 = (a11 * b2 - b1 * a21) / det
        return float(v0), float(v1)

    def v0(self, theta: float) -> float:
        return self.values(theta)[0]

    def v0_grad(self, theta: float, eps: float = 1e-6) -> float:
        return (self.v0(theta + eps) - self.v0(theta - eps)) / (2 * eps)

    # ------------------------------------------------------------------
    def fine_tune(
        self,
        theta0: float,
        lr: float = 1e-3,
        steps: int = 20_000,
        tol: float = 1e-9,
        clip: Tuple[float, float] = (-10.0, 10.0),
    ) -> Tuple[float, List[float]]:
        """Gradient ascent on ``v0`` starting from ``theta0``.

        This mimics "adjusting theta towards the gradient direction of v0 until
        a local extremum is encountered".
        """

        theta = float(theta0)
        trajectory: List[float] = [self.v0(theta)]
        for _ in range(steps):
            grad = self.v0_grad(theta)
            theta = float(np.clip(theta + lr * grad, clip[0], clip[1]))
            trajectory.append(self.v0(theta))
            if abs(grad) < tol:
                break
        return theta, trajectory


def state_coverage_gap_f(theta: float, eps: float = 0.02) -> float:
    """``f_theta`` for the state coverage gap (Appendix A.1).

    f_theta = (-eps / (1 - eps/2)) * theta + 1     for theta <= 1 - eps/2
    f_theta = 2 * theta - 1                        for theta >  1 - eps/2

    ``f_0 = 1`` so a policy pre-trained on ``s1`` stays there; fine-tuning with
    ``s0`` as the initial state forgets this behaviour and converges to a
    suboptimal local maximum.
    """

    if theta <= 1.0 - eps / 2.0:
        return (-eps / (1.0 - eps / 2.0)) * theta + 1.0
    return 2.0 * theta - 1.0


def imperfect_cloning_gap_f(theta: float) -> float:
    """``f_theta = 2|theta - 0.5|`` for the imperfect cloning gap (Appendix A.1)."""

    return 2.0 * abs(theta - 0.5)


def build_state_coverage_gap(**kwargs) -> TwoStateMDP:
    return TwoStateMDP(f=state_coverage_gap_f, **kwargs)


def build_imperfect_cloning_gap(**kwargs) -> TwoStateMDP:
    return TwoStateMDP(f=imperfect_cloning_gap_f, **kwargs)


# ---------------------------------------------------------------------------
# The closed form printed in Appendix A.1
# ---------------------------------------------------------------------------
def paper_value_function(theta: float, f_theta: float, r0: float = 1.0, r1: float = 1.0,
                         gamma: float = 0.9) -> float:
    r"""``v_0`` exactly as printed in Appendix A.1::

        v_0(theta) = 1/(1-gamma) *
            [theta + r0 (1-theta)(1 - gamma f_theta) + gamma theta r1 (1 - f_theta)]
            / [1 - gamma f_theta + gamma theta]

    .. note::
        With ``r0 = r1 = 1`` the numerator equals the denominator identically, so
        ``v_0`` is constant and equal to ``1/(1-gamma)`` for every ``theta``
        (the transition ``s0 -> s0`` is then degenerate).  The paper does not
        state the values of ``r0`` and ``r1`` used in Figure 9, so this helper
        exposes them as arguments.  See ``README.md`` for a discussion.
    """

    numerator = (
        theta
        + r0 * (1.0 - theta) * (1.0 - gamma * f_theta)
        + gamma * theta * r1 * (1.0 - f_theta)
    )
    denominator = 1.0 - gamma * f_theta + gamma * theta
    return (1.0 / (1.0 - gamma)) * numerator / denominator


@dataclass
class ScenarioResult:
    """Outcome of fine-tuning one of the two parameterisations."""

    name: str
    theta_grid: List[float]
    v0_grid: List[float]
    converged_theta: float
    converged_value: float
    optimal_theta: float
    optimal_value: float
    start_theta: float = 0.0


def fine_tune_paper_value(
    f: Callable[[float], float],
    theta0: float,
    r0: float = 0.5,
    r1: float = 0.0,
    gamma: float = 0.9,
    lr: float = 2e-3,
    steps: int = 20_000,
    clip: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[float, List[float]]:
    """Gradient ascent on the paper's closed-form ``v_0``."""

    def v(theta: float) -> float:
        return paper_value_function(theta, f(theta), r0, r1, gamma)

    theta = float(theta0)
    trajectory = [v(theta)]
    for _ in range(steps):
        eps = 1e-7
        grad = (v(theta + eps) - v(theta - eps)) / (2 * eps)
        theta = float(np.clip(theta + lr * grad, clip[0], clip[1]))
        trajectory.append(v(theta))
    return theta, trajectory


def run_paper_scenarios(
    r0: float = 0.0,
    r1: float = -1.0,
    gamma: float = 0.9,
    eps: float = 0.02,
    lr: float = 1e-3,
    steps: int = 200_000,
) -> List[ScenarioResult]:
    """Run the two scenarios of Appendix A.1 with the paper's closed form.

    With the defaults ``r0 = 0``, ``r1 = -1`` and ``gamma = 0.9`` the
    ``f_theta = 2|theta - 0.5|`` landscape has an interior local maximum at
    ``theta = 0.1111`` with value ``2.2222`` -- exactly the fixed point reported
    in Appendix A.1 (theta = 0.11, value = 2.22) -- while the optimal policy
    ``theta = 1`` attains ``1/(1-gamma) = 10``.  Gradient ascent started at
    ``theta = 0`` is trapped at the suboptimal local maximum, which is the
    phenomenon the paper illustrates.

    The reward values ``r0`` and ``r1`` used in Figure 9 are not stated in the
    paper (see the note on :func:`paper_value_function`), so they are exposed as
    arguments.
    """

    grid = np.linspace(0.0, 1.0, 400).tolist()
    results: List[ScenarioResult] = []

    def solve(name: str, f: Callable[[float], float], theta0: float) -> ScenarioResult:
        theta, _ = fine_tune_paper_value(f, theta0=theta0, r0=r0, r1=r1, gamma=gamma, lr=lr, steps=steps)
        return ScenarioResult(
            name=name,
            theta_grid=grid,
            v0_grid=[paper_value_function(t, f(t), r0, r1, gamma) for t in grid],
            converged_theta=theta,
            converged_value=paper_value_function(theta, f(theta), r0, r1, gamma),
            optimal_theta=1.0,
            optimal_value=paper_value_function(1.0, f(1.0), r0, r1, gamma),
            start_theta=theta0,
        )

    scg_f = lambda t: state_coverage_gap_f(t, eps)  # noqa: E731
    # (a) state coverage gap: pi_* only knew s1 (theta = 0), fine-tuning starts in s0.
    results.append(solve("state_coverage_gap", scg_f, theta0=0.0))
    # (b) imperfect cloning gap: the pre-trained policy is a perturbed optimum.
    results.append(solve("imperfect_cloning_gap", imperfect_cloning_gap_f, theta0=1.0))
    # (c) the suboptimal local maximum reported in the paper (theta = 0.11, v = 2.22),
    #     which the f_theta = 2|theta - 0.5| parameterisation is trapped in.
    results.append(solve("reported_suboptimal_fixed_point", imperfect_cloning_gap_f, theta0=0.0))
    return results

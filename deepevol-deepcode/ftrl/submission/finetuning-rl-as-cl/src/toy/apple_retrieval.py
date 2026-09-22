"""AppleRetrieval: a synthetic 1D gridworld exposing the *state coverage gap* (Appendix A.2).

The paper (Wolczyk et al., 2024 -- "Fine-tuning Reinforcement Learning Models is
Secretly a Forgetting Mitigation Problem", Appendix A.2) describes the environment
as follows:

    APPLERETRIEVAL is a 1D gridworld, consisting of two phases.  In Phase 1,
    starting at home x = 0, the agent has to go to x = M and retrieve an apple,
    M in N.  In Phase 2, the agent has to go back to x = 0.  In each phase, the
    reward is 1 for going in the correct direction and -1 otherwise.  The
    observation is o = [-c] in Phase 1 and o = [c] in Phase 2, for some c in R;
    i.e. it encodes the information about the current phase.  Given this
    observation, it is now trivial to encode the optimal policy: go right in
    Phase 1 and go left in Phase 2.  Episodes are terminated if the solution is
    reached or after 100 timesteps.  Since we can only get to Phase 2 by
    completing Phase 1, this corresponds to dividing the states to sets CLOSE
    and FAR, as described in Section 2.

Experiments are run with REINFORCE (Williams, 1992) and the simple linear model

    pi_{w,b}(o) = sigmoid(w * o + b),    w, b in R,                     (Eq. A.1)

where ``w, b`` are **initialized with the weights trained in Phase 2** (the
pre-trained capability ``pi_*``).  Two phenomena are studied:

* Figure 10 -- for high enough distance ``M`` the probability of concluding
  Phase 1 becomes small enough that the pre-trained Phase 2 policy is forgotten,
  which hinders the overall performance.
* Figure 11 -- for a fixed ``M = 30``, the parameter ``c`` controls whether the
  learned solution relies on the weight (``|w| >> |b|``, little interference) or
  on the bias (``|b| >> |w|``, strong interference, since the bias shifts the
  output identically in both phases).

Because plain stochastic gradient descent on a linear model is implicitly biased
towards low-norm solutions and the gradient with respect to ``w`` is scaled by
the observation magnitude ``c`` (``dL/dw = c * dL/db``), a small ``c`` yields a
bias-dominated solution (``|b|/|w| ~ 1/c``) -- exactly the regime in which
forgetting of the pre-trained Phase-2 policy is severe.

This module is deliberately dependency-light: the environment, REINFORCE
optimizer and every metric are implemented with the Python standard library
(``math``), using NumPy only when available for convenient statistics.  The
implementation is NumPy-free-functional to keep the toy experiment runnable on
CPU in a few seconds.

Metrics (matching ``configs/toy.yaml -> apple_retrieval.metrics``):

* ``phase2_action_agreement`` -- fraction of (reference) Phase-2 observations on
  which the greedily-argmax action of the fine-tuned policy coincides with the
  pre-trained policy's action.  ``forgetting = 1 - phase2_action_agreement``.
* ``phase2_right_prob`` -- ``pi_{w,b}(right | o = [c])``, the smooth version of
  the forgetting signal (probability of the *wrong* action in Phase 2).
* ``phase2_success`` -- fraction of evaluation episodes that reach Phase 2 *and*
  return home.
* ``overall_success`` -- fraction of full-task evaluation episodes solved.
* ``mean_return`` -- average undiscounted return per evaluation episode.
* ``weight_norm`` / ``bias_norm`` / ``bias_weight_ratio`` -- ``|w|``, ``|b|`` and
  ``|b| / |w|`` of the fine-tuned parameters (Figure 11, right).
* ``delta_w`` / ``delta_b`` -- parameter change w.r.t. the pre-trained weights,
  which early in fine-tuning is proportional to ``1/c`` for the bias.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import statistics
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:  # NumPy is optional: the module also works with the stdlib ``random`` RNG.
    import numpy as _np
except Exception:  # pragma: no cover - numpy is a declared dependency
    _np = None


# --------------------------------------------------------------------------------------
# Constants from Appendix A.2 (and defaults for quantities the paper leaves open)
# --------------------------------------------------------------------------------------

PHASE_1: int = 1
PHASE_2: int = 2
LEFT: int = 0
RIGHT: int = 1

HORIZON: int = 100
"""Episodes are terminated after 100 timesteps (Appendix A.2)."""

REWARD_CORRECT: float = 1.0
REWARD_WRONG: float = -1.0

DEFAULT_M: int = 30
"""Distance from home to the apple used for the sweep over c (Figure 11)."""

DEFAULT_C: float = 1.0
DEFAULT_PRETRAIN_EPISODES: int = 3000
DEFAULT_FINETUNE_EPISODES: int = 3000
DEFAULT_PRETRAIN_LR: float = 0.05
DEFAULT_FINETUNE_LR: float = 0.05
DEFAULT_EVAL_EVERY: int = 100
DEFAULT_EVAL_EPISODES: int = 200
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_SWEEP_M: Tuple[int, ...] = (1, 2, 5, 10, 20, 30, 50)
DEFAULT_SWEEP_C: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)

APPLE_RETRIEVAL_METRICS: Tuple[str, ...] = (
    "phase2_action_agreement",
    "phase2_success",
    "overall_success",
    "mean_return",
    "weight_norm",
    "bias_norm",
    "bias_weight_ratio",
)

# ``bias_weight_ratio`` uses the finite-difference approximation used in LoRA+ style
# analyses; here we simply report |b| / (|w| + eps).
_EPS = 1e-12


# --------------------------------------------------------------------------------------
# Tiny helpers (float math, RNG abstraction, statistics)
# --------------------------------------------------------------------------------------


def sigmoid(z: float) -> float:
    """Numerically stable logistic function (Eq. A.1 uses ``sigma``)."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _scalar(obs: Any) -> float:
    """Coerce an observation (float, 0-d/1-d tensor/list of size 1) to a float."""
    if obs is None:
        return 0.0
    if isinstance(obs, (int, float)):
        return float(obs)
    if isinstance(obs, (list, tuple)):
        if len(obs) == 0:
            return 0.0
        return _scalar(obs[0])
    # numpy array / torch tensor / anything indexable
    if _np is not None and isinstance(obs, _np.ndarray):
        return float(obs.reshape(-1)[0]) if obs.size else 0.0
    try:
        if hasattr(obs, "shape") and getattr(obs, "shape", None) is not None:
            flat = obs
            while hasattr(flat, "shape") and len(getattr(flat, "shape")) > 0:
                if len(flat.shape) == 0:
                    break
                flat = flat[0]
            return float(flat)
    except Exception:
        pass
    try:
        return float(obs)
    except Exception:
        return 0.0


def _as_rng(seed: Optional[int]) -> Any:
    """Return either a ``numpy.random.Generator`` or a ``random.Random`` instance."""
    if _np is not None:
        return _np.random.default_rng(seed)
    import random as _random

    return _random.Random(seed)


def _uniform(rng: Any, low: float = 0.0, high: float = 1.0) -> float:
    if _np is not None and hasattr(rng, "uniform"):
        return float(rng.uniform(low, high))
    return float(rng.uniform(low, high))


def _randint(rng: Any, low: int, high: int) -> int:
    """Uniform integer in ``[low, high]`` (inclusive)."""
    high = max(low, high)
    if _np is not None and hasattr(rng, "integers"):
        return int(rng.integers(low, high + 1))
    return int(rng.randint(low, high))


def _bernoulli(rng: Any, p: float) -> int:
    p = min(1.0, max(0.0, float(p)))
    if _np is not None and hasattr(rng, "random"):
        return int(float(rng.random()) < p)
    return int(rng.random() < p)


def _choice_between(rng: Any, prob_first: float) -> int:
    """Return 1 with probability ``prob_first`` else 0 (kept explicit for clarity)."""
    return _bernoulli(rng, prob_first)


def _maybe_numpy(values: Sequence[float]) -> Any:
    if _np is None:
        return list(values)
    return _np.asarray(list(values), dtype=float)


# --------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------


class AppleRetrievalEnv:
    """The AppleRetrieval 1D gridworld of Appendix A.2.

    Phase 1 (CLOSE states): the agent starts at ``x = 0`` and has to reach ``x = M``
    to retrieve the apple.  Phase 2 (FAR states): the agent has to go back to
    ``x = 0``.  The reward is ``+1`` for moving in the correct direction of the
    current phase and ``-1`` otherwise.  The observation encodes the phase:
    ``o = [-c]`` in Phase 1 and ``o = [c]`` in Phase 2.  Episodes terminate when the
    solution is reached (returning home) or after ``horizon`` (100) timesteps.

    Parameters
    ----------
    M:
        Distance from home to the apple (``M in N``).
    c:
        Phase-encoding magnitude of the observation.
    horizon:
        Episode length limit (100 by default).
    phases:
        Number of phases to play before the episode counts as solved (2 by default;
        ``1`` gives the Phase-1-only task used for diagnostics).
    randomize_start:
        If ``True``, Phase-2-only rollouts (pre-training) start at a random position
        in ``[1, M]`` instead of deterministically at ``M``.
    seed:
        Optional seed for the internal RNG (only used by :meth:`random_start`).
    """

    def __init__(
        self,
        M: int = DEFAULT_M,
        c: float = DEFAULT_C,
        horizon: int = HORIZON,
        reward_correct: float = REWARD_CORRECT,
        reward_wrong: float = REWARD_WRONG,
        phases: int = 2,
        randomize_start: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.M = int(M)
        self.c = float(c)
        self.horizon = int(horizon)
        self.reward_correct = float(reward_correct)
        self.reward_wrong = float(reward_wrong)
        self.phases = int(phases)
        self.randomize_start = bool(randomize_start)
        self._rng = _as_rng(seed)

        # runtime state
        self.phase: int = PHASE_1
        self.position: int = 0
        self.t: int = 0
        self.apple_retrieved: bool = False
        self.home_reached: bool = False
        self.phase1_solved: bool = False
        self.phase2_solved: bool = False
        self.early_phase2_step: Optional[int] = None

    # -- spaces -------------------------------------------------------------------
    @property
    def observation_dim(self) -> int:
        return 1

    @property
    def action_dim(self) -> int:
        return 2

    # -- observation ---------------------------------------------------------------
    def observation(self, phase: Optional[int] = None) -> Any:
        """``o = [-c]`` in Phase 1 and ``o = [c]`` in Phase 2 (Appendix A.2)."""
        p = self.phase if phase is None else phase
        value = -self.c if p == PHASE_1 else self.c
        if _np is not None:
            return _np.asarray([value], dtype=_np.float32)
        return [value]

    def phase_observation(self, phase: int) -> Any:
        return self.observation(phase)

    def phase2_reference_observations(self, count: int = 1) -> List[Any]:
        """Reference observations for the Phase-2 action-agreement metric.

        The Phase-2 observation is constant (``o = [c]``), hence all reference
        observations coincide; ``count`` is accepted for API symmetry with
        state-based environments.
        """
        return [self.observation(PHASE_2) for _ in range(max(1, int(count)))]

    # -- reset / step ---------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        phase: Optional[int] = None,
        position: Optional[int] = None,
    ) -> Any:
        """Reset the environment, optionally forcing a phase / position.

        ``reset(phase=2)`` is used for the Phase-2 pre-training of ``pi_*`` described
        in Appendix A.2 ("we initialize w, b with the weights trained in Phase 2").
        """
        if seed is not None:
            self._rng = _as_rng(seed)
        self.phase = PHASE_1 if phase is None else int(phase)
        if position is None:
            if self.phase == PHASE_1:
                self.position = 0
            else:
                self.position = (
                    _randint(self._rng, 1, max(1, self.M)) if self.randomize_start else self.M
                )
        else:
            self.position = int(position)
        self.t = 0
        self.apple_retrieved = self.phase > PHASE_1
        self.home_reached = self.phase > self.phases
        self.phase1_solved = False
        self.phase2_solved = False
        self.early_phase2_step = 0 if self.phase > PHASE_1 else None
        return self.observation()

    def correct_action(self, phase: Optional[int] = None) -> int:
        """Right in Phase 1, left in Phase 2 (the trivial optimal policy)."""
        p = self.phase if phase is None else phase
        return RIGHT if p == PHASE_1 else LEFT

    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Advance one timestep; returns ``(obs, reward, done, info)``."""
        action = int(action)
        phase = self.phase
        correct = self.correct_action(phase)
        reward = self.reward_correct if action == correct else self.reward_wrong

        # Move (the boundaries are absorbing in terms of position).
        if action == RIGHT:
            self.position = min(self.M, self.position + 1)
        else:
            self.position = max(0, self.position - 1)

        phase_solved = False
        if phase == PHASE_1 and self.position >= self.M:
            # Apple retrieved -> enter Phase 2 (state coverage gap: only reachable
            # by solving Phase 1).
            self.apple_retrieved = True
            self.phase1_solved = True
            phase_solved = True
            if self.phases >= 2:
                self.phase = PHASE_2
                self.t = 0
                self.early_phase2_step = 0
            elif self.apple_retrieved and self.phases < 2:
                self.phase = PHASE_2
        elif phase == PHASE_2 and self.position <= 0:
            self.home_reached = True
            self.phase2_solved = True
            phase_solved = True

        self.t += 1
        if self.early_phase2_step is not None and self.phase == PHASE_2:
            self.early_phase2_step += 1

        solved = bool(self.phase2_solved or (self.phases < 2 and self.phase1_solved))
        timeout = self.t >= self.horizon
        done = bool(solved or timeout)

        info: Dict[str, Any] = {
            "phase": self.phase,
            "position": self.position,
            "t": self.t,
            "phase_solved": bool(phase_solved),
            "phase1_solved": bool(self.phase1_solved),
            "phase2_solved": bool(self.phase2_solved),
            "solved": solved,
            "timeout": bool(timeout and not solved),
            "apple_retrieved": bool(self.apple_retrieved),
            "correct_action": int(correct),
        }
        return self.observation(), float(reward), done, info

    def close(self) -> None:  # pragma: no cover - API symmetry
        """No resources to release."""

    def render(self) -> str:  # pragma: no cover - debugging helper
        marker = "A" if self.phase == PHASE_1 else "H"
        cells = ["." for _ in range(self.M + 1)]
        cells[self.position] = "X"
        return f"phase={self.phase} t={self.t} pos={self.position} target={marker} " + "".join(cells)


# --------------------------------------------------------------------------------------
# Policy: pi_{w,b}(o) = sigmoid(w * o + b)
# --------------------------------------------------------------------------------------


class LinearSigmoidPolicy:
    """The linear sigmoid policy of Eq. (A.1): ``pi_{w,b}(o) = sigma(w * o + b)``.

    The output is interpret as the probability of moving **right**; ``w`` and ``b``
    are the only two parameters, which makes the interference analysis of Figure 11
    exact (``|b| >> |w|`` implies the bias shifts the Phase-1 and Phase-2 behaviour
    in the very same way).
    """

    def __init__(self, w: float = 0.0, b: float = 0.0) -> None:
        self.w = float(w)
        self.b = float(b)

    # -- core ---------------------------------------------------------------------
    def prob_right(self, obs: Any) -> float:
        return sigmoid(self.w * _scalar(obs) + self.b)

    def prob_left(self, obs: Any) -> float:
        return 1.0 - self.prob_right(obs)

    def probs(self, obs: Any) -> Tuple[float, float]:
        p = self.prob_right(obs)
        return (1.0 - p, p)

    def logits(self, obs: Any) -> float:
        return self.w * _scalar(obs) + self.b

    def log_prob(self, obs: Any, action: int) -> float:
        p = self.prob_right(obs)
        a = int(action)
        if a == RIGHT:
            return math.log(max(p, _EPS))
        return math.log(max(1.0 - p, _EPS))

    def greedy_action(self, obs: Any) -> int:
        return RIGHT if self.prob_right(obs) >= 0.5 else LEFT

    def sample_action(self, obs: Any, rng: Any, deterministic: bool = False) -> int:
        if deterministic:
            return self.greedy_action(obs)
        return _bernoulli(rng, self.prob_right(obs))

    def act(self, obs: Any, rng: Optional[Any] = None, deterministic: bool = False) -> int:
        """Duck-typed policy interface compatible with the other environments."""
        return self.sample_action(obs, rng if rng is not None else _as_rng(None),
                                  deterministic=deterministic)

    # -- analysis -------------------------------------------------------------------
    def action_agreement(self, other: "LinearSigmoidPolicy", observations: Sequence[Any]) -> float:
        """Fraction of ``observations`` on which the greedy actions coincide."""
        if not observations:
            return 1.0
        same = sum(1 for o in observations if self.greedy_action(o) == other.greedy_action(o))
        return float(same) / float(len(observations))

    def parameters(self) -> Dict[str, float]:
        return {"w": self.w, "b": self.b}

    def set_parameters(self, w: float, b: float) -> None:
        self.w = float(w)
        self.b = float(b)

    def copy(self) -> "LinearSigmoidPolicy":
        return LinearSigmoidPolicy(self.w, self.b)

    def as_dict(self) -> Dict[str, float]:
        return {
            "w": self.w,
            "b": self.b,
            "weight_norm": abs(self.w),
            "bias_norm": abs(self.b),
            "bias_weight_ratio": abs(self.b) / (abs(self.w) + _EPS),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"LinearSigmoidPolicy(w={self.w:.4f}, b={self.b:.4f})"


# --------------------------------------------------------------------------------------
# Optimizers (SGD / Adam over the two parameters w and b)
# --------------------------------------------------------------------------------------


class _SGDOptimizer:
    kind = "sgd"

    def __init__(self, lr: float = 0.05, **_kwargs: Any) -> None:
        self.lr = float(lr)

    def step(self, policy: LinearSigmoidPolicy, grads: Mapping[str, float]) -> None:
        policy.w -= self.lr * float(grads.get("w", 0.0))
        policy.b -= self.lr * float(grads.get("b", 0.0))

    def state_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "lr": self.lr}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.lr = float(state.get("lr", self.lr))


class _AdamOptimizer:
    kind = "adam"

    def __init__(self, lr: float = 0.01, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8) -> None:
        self.lr = float(lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.step_count = 0
        self.m: Dict[str, float] = {"w": 0.0, "b": 0.0}
        self.v: Dict[str, float] = {"w": 0.0, "b": 0.0}

    def step(self, policy: LinearSigmoidPolicy, grads: Mapping[str, float]) -> None:
        self.step_count += 1
        for name in ("w", "b"):
            g = float(grads.get(name, 0.0))
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * g * g
            m_hat = self.m[name] / (1.0 - self.beta1 ** self.step_count)
            v_hat = self.v[name] / (1.0 - self.beta2 ** self.step_count)
            update = self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
            if name == "w":
                policy.w -= update
            else:
                policy.b -= update

    def state_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "lr": self.lr,
            "betas": (self.beta1, self.beta2),
            "eps": self.eps,
            "step_count": self.step_count,
            "m": dict(self.m),
            "v": dict(self.v),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.lr = float(state.get("lr", self.lr))
        self.step_count = int(state.get("step_count", self.step_count))
        self.m.update(state.get("m", {}) or {})
        self.v.update(state.get("v", {}) or {})


def make_optimizer(name: str = "sgd", lr: float = DEFAULT_FINETUNE_LR,
                   **kwargs: Any) -> Union[_SGDOptimizer, _AdamOptimizer]:
    """Factory for the tiny optimizers used by REINFORCE."""
    if str(name).lower() in ("adam",):
        return _AdamOptimizer(lr=lr, **kwargs)
    return _SGDOptimizer(lr=lr)


# --------------------------------------------------------------------------------------
# REINFORCE
# --------------------------------------------------------------------------------------


def collect_episode(
    env: AppleRetrievalEnv,
    policy: LinearSigmoidPolicy,
    rng: Any,
    phase: int = PHASE_1,
    deterministic: bool = False,
) -> Dict[str, Any]:
    """Roll out one episode with ``policy`` and return the trajectory."""
    obs = env.reset(phase=phase)
    observations: List[Any] = []
    actions: List[int] = []
    rewards: List[float] = []
    infos: List[Dict[str, Any]] = []

    done = False
    while not done:
        a = policy.sample_action(obs, rng, deterministic=deterministic)
        observations.append(obs)
        actions.append(int(a))
        obs, r, done, info = env.step(a)
        rewards.append(float(r))
        infos.append(info)

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "infos": infos,
        "return": float(sum(rewards)),
        "length": len(rewards),
        "phase1_solved": bool(infos[-1]["phase1_solved"] if infos else False),
        "phase2_solved": bool(infos[-1]["phase2_solved"] if infos else False),
        "reached_phase2": bool(any(i["phase"] == PHASE_2 for i in infos)),
        "solved": bool(infos[-1]["solved"] if infos else False),
        "timeout": bool(infos[-1]["timeout"] if infos else False),
    }


def discounted_returns(rewards: Sequence[float], gamma: float = 1.0) -> List[float]:
    """Return-to-go for each timestep (undiscounted when ``gamma == 1``)."""
    out: List[float] = [0.0] * len(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = float(rewards[t]) + gamma * running
        out[t] = running
    return out


def reinforce_gradient(
    trajectories: Sequence[Mapping[str, Any]],
    policy: LinearSigmoidPolicy,
    gamma: float = 1.0,
    baseline: bool = True,
    normalize_returns: bool = False,
) -> Dict[str, float]:
    """REINFORCE policy gradient for the two-parameter linear sigmoid policy.

    With ``u = w * o + b`` and ``a in {0, 1}`` (1 = right), ``d log pi / du = a - p``.
    Hence ``d log pi / dw = o * (a - p)`` and ``d log pi / db = a - p``, so the
    weight gradient is exactly ``c`` times the bias gradient -- the mechanism behind
    the ``1/c`` scaling of ``|b| / |w|`` observed in Figure 11.
    """
    grad_w = 0.0
    grad_b = 0.0
    n_steps = 0
    for traj in trajectories:
        rewards = traj["rewards"]
        returns = discounted_returns(rewards, gamma=gamma)
        if baseline and returns:
            base = sum(returns) / len(returns)
            returns = [g - base for g in returns]
        if normalize_returns and len(returns) > 1:
            mean = sum(returns) / len(returns)
            var = sum((g - mean) ** 2 for g in returns) / len(returns)
            std = math.sqrt(var) + 1e-8
            returns = [(g - mean) / std for g in returns]
        for obs, action, g in zip(traj["observations"], traj["actions"], returns):
            p = policy.prob_right(obs)
            d_logp_du = float(action) - p
            grad_w += g * d_logp_du * _scalar(obs)
            grad_b += g * d_logp_du
            n_steps += 1
    denom = float(max(1, n_steps))
    return {"w": grad_w / denom, "b": grad_b / denom}


def reinforce_update(
    policy: LinearSigmoidPolicy,
    trajectories: Sequence[Mapping[str, Any]],
    optimizer: Any,
    gamma: float = 1.0,
    baseline: bool = True,
    normalize_returns: bool = False,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    """One REINFORCE update (gradient descent on ``-E[G log pi]``)."""
    grads = reinforce_gradient(
        trajectories, policy, gamma=gamma, baseline=baseline,
        normalize_returns=normalize_returns,
    )
    if grad_clip is not None and grad_clip > 0:
        norm = math.sqrt(grads["w"] ** 2 + grads["b"] ** 2)
        if norm > grad_clip:
            scale = float(grad_clip) / (norm + _EPS)
            grads = {k: v * scale for k, v in grads.items()}
    optimizer.step(policy, grads)
    return grads


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def evaluate_policy(
    env: AppleRetrievalEnv,
    policy: LinearSigmoidPolicy,
    num_episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
    deterministic: bool = True,
    reference_policy: Optional[LinearSigmoidPolicy] = None,
) -> Dict[str, float]:
    """Evaluate ``policy`` on the *full* task (Phase 1 then Phase 2)."""
    rng = _as_rng(seed)
    overall = 0
    phase1 = 0
    phase2 = 0
    reached_phase2 = 0
    returns: List[float] = []
    lengths: List[int] = []
    for ep in range(int(num_episodes)):
        traj = collect_episode(env, policy, rng, phase=PHASE_1, deterministic=deterministic)
        overall += int(traj["solved"])
        phase1 += int(traj["phase1_solved"])
        reached_phase2 += int(traj["reached_phase2"])
        phase2 += int(traj["phase2_solved"])
        returns.append(traj["return"])
        lengths.append(traj["length"])

    n = float(max(1, int(num_episodes)))
    metrics: Dict[str, float] = {
        "overall_success": overall / n,
        "phase1_success": phase1 / n,
        "phase2_success": phase2 / n,
        "phase2_success_given_reached": (phase2 / reached_phase2) if reached_phase2 else 0.0,
        "reached_phase2_rate": reached_phase2 / n,
        "mean_return": sum(returns) / n,
        "mean_length": sum(lengths) / n,
    }
    metrics.update(policy_metrics(env, policy, reference_policy=reference_policy))
    return metrics


def policy_metrics(
    env: AppleRetrievalEnv,
    policy: LinearSigmoidPolicy,
    reference_policy: Optional[LinearSigmoidPolicy] = None,
) -> Dict[str, float]:
    """Policy-side metrics: action agreement with ``pi_*`` and parameter norms."""
    ref = reference_policy if reference_policy is not None else LinearSigmoidPolicy(0.0, 0.0)
    phase2_refs = env.phase2_reference_observations()
    phase1_obs = env.observation(PHASE_1)
    agreement = policy.action_agreement(ref, phase2_refs)
    out: Dict[str, float] = {
        "phase2_action_agreement": agreement,
        "forgetting": 1.0 - agreement,
        "phase2_right_prob": policy.prob_right(env.observation(PHASE_2)),
        "phase1_right_prob": policy.prob_right(phase1_obs),
        "w": policy.w,
        "b": policy.b,
    }
    out.update(
        {
            "weight_norm": abs(policy.w),
            "bias_norm": abs(policy.b),
            "bias_weight_ratio": abs(policy.b) / (abs(policy.w) + _EPS),
        }
    )
    if not isinstance(ref, LinearSigmoidPolicy) or (ref.w == 0.0 and ref.b == 0.0):
        # No meaningful reference: agreement is computed against the untrained policy.
        pass
    return out


# --------------------------------------------------------------------------------------
# Pre-training (Phase 2) and fine-tuning (full task)
# --------------------------------------------------------------------------------------


def pretrain_phase2(
    M: int = DEFAULT_M,
    c: float = DEFAULT_C,
    episodes: int = DEFAULT_PRETRAIN_EPISODES,
    lr: float = DEFAULT_PRETRAIN_LR,
    optimizer: str = "sgd",
    gamma: float = 1.0,
    baseline: bool = True,
    horizon: int = HORIZON,
    init_w: float = 0.0,
    init_b: float = 0.0,
    seed: int = 0,
    grad_clip: Optional[float] = None,
    record_every: int = 0,
) -> Tuple[LinearSigmoidPolicy, List[Dict[str, Any]]]:
    """Train ``pi_*`` on Phase 2 (Appendix A.2: "weights trained in Phase 2").

    Phase 2 is the *pre-trained* capability: starting at ``x = M`` the agent has to
    go back home (CLOSE in the sense of reachable behaviour once the task is solved,
    but the *policy* is what is pre-trained here).  Because the observation is the
    constant ``o = [c]``, gradient descent implicitly favours the low-norm solution
    with ``dL/dw = c * dL/db``, so smaller ``c`` yields a bias-dominated model.
    """
    env = AppleRetrievalEnv(M=M, c=c, horizon=horizon, randomize_start=True)
    policy = LinearSigmoidPolicy(init_w, init_b)
    opt = make_optimizer(optimizer, lr=lr)
    rng = _as_rng(seed)
    history: List[Dict[str, Any]] = []

    for ep in range(int(episodes)):
        traj = collect_episode(env, policy, rng, phase=PHASE_2)
        reinforce_update(
            policy, [traj], opt, gamma=gamma, baseline=baseline, grad_clip=grad_clip
        )
        if record_every and (ep + 1) % int(record_every) == 0:
            history.append(
                {
                    "step": ep + 1,
                    "w": policy.w,
                    "b": policy.b,
                    "phase2_right_prob": policy.prob_right(env.observation(PHASE_2)),
                    "return": traj["return"],
                    "bias_weight_ratio": abs(policy.b) / (abs(policy.w) + _EPS),
                }
            )
    return policy, history


def fine_tune_full_task(
    policy_init: LinearSigmoidPolicy,
    M: int = DEFAULT_M,
    c: float = DEFAULT_C,
    episodes: int = DEFAULT_FINETUNE_EPISODES,
    lr: float = DEFAULT_FINETUNE_LR,
    optimizer: str = "sgd",
    gamma: float = 1.0,
    baseline: bool = True,
    horizon: int = HORIZON,
    eval_every: int = DEFAULT_EVAL_EVERY,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    reference_policy: Optional[LinearSigmoidPolicy] = None,
    seed: int = 0,
    grad_clip: Optional[float] = None,
    record_trace: bool = True,
) -> Tuple[LinearSigmoidPolicy, List[Dict[str, Any]]]:
    """Vanilla fine-tuning on the full task (Phase 1 then Phase 2).

    Episodes always start in Phase 1 at home.  Since Phase 2 is only reachable by
    solving Phase 1, a policy that cannot complete Phase 1 never collects Phase-2
    data and forgets the pre-trained Phase-2 capability -- the *state coverage gap*.
    """
    env = AppleRetrievalEnv(M=M, c=c, horizon=horizon, randomize_start=False)
    policy = policy_init.copy()
    opt = make_optimizer(optimizer, lr=lr)
    rng = _as_rng(seed + 7919)
    ref = reference_policy if reference_policy is not None else policy.copy()
    trace: List[Dict[str, Any]] = []

    def _record(step: int) -> None:
        if not record_trace:
            return
        ev = evaluate_policy(
            env, policy, num_episodes=eval_episodes, seed=step + 13, reference_policy=ref
        )
        row: Dict[str, Any] = {"step": step}
        row.update(ev)
        row["delta_w"] = policy.w - ref.w
        row["delta_b"] = policy.b - ref.b
        trace.append(row)

    _record(0)
    for ep in range(1, int(episodes) + 1):
        traj = collect_episode(env, policy, rng, phase=PHASE_1)
        reinforce_update(
            policy, [traj], opt, gamma=gamma, baseline=baseline, grad_clip=grad_clip
        )
        if eval_every and ep % int(eval_every) == 0:
            _record(ep)
    return policy, trace


# --------------------------------------------------------------------------------------
# Config plumbing
# --------------------------------------------------------------------------------------


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    """Fetch ``key`` from a ``Config``/mapping/namespace (used for tolerant loading)."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        if key in obj:
            return obj[key]
        return default
    if hasattr(obj, key):
        return getattr(obj, key)
    return default


def _first_of(obj: Any, keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        val = _cfg_get(obj, k, None)
        if val is not None:
            return val
    return default


def _as_tuple(value: Any, default: Sequence[Any]) -> Tuple[Any, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, (list, tuple)):
        return tuple(value)
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        out: List[Any] = []
        for p in parts:
            try:
                out.append(int(p))
                continue
            except ValueError:
                pass
            try:
                out.append(float(p))
                continue
            except ValueError:
                pass
            out.append(p)
        return tuple(out) or tuple(default)
    return (value,)


@dataclass
class AppleConfig:
    """Resolved hyperparameters for the AppleRetrieval experiment (see ``configs/toy.yaml``)."""

    M: int = DEFAULT_M
    c: float = DEFAULT_C
    horizon: int = HORIZON
    reward_correct: float = REWARD_CORRECT
    reward_wrong: float = REWARD_WRONG
    phases: int = 2
    pretrain_episodes: int = DEFAULT_PRETRAIN_EPISODES
    pretrain_lr: float = DEFAULT_PRETRAIN_LR
    finetune_episodes: int = DEFAULT_FINETUNE_EPISODES
    finetune_lr: float = DEFAULT_FINETUNE_LR
    optimizer: str = "sgd"
    gamma: float = 1.0
    baseline: bool = True
    normalize_returns: bool = False
    grad_clip: Optional[float] = None
    init_w: float = 0.0
    init_b: float = 0.0
    init_scale: float = 0.0
    eval_every: int = DEFAULT_EVAL_EVERY
    eval_episodes: int = DEFAULT_EVAL_EPISODES
    seeds: Tuple[int, ...] = DEFAULT_SEEDS
    sweep_M: Tuple[int, ...] = DEFAULT_SWEEP_M
    sweep_M_c: float = DEFAULT_C
    sweep_c: Tuple[float, ...] = DEFAULT_SWEEP_C
    sweep_c_M: int = DEFAULT_M
    forgetting_metric: str = "phase2_action_agreement"
    confidence: float = 0.90
    metrics: Tuple[str, ...] = APPLE_RETRIEVAL_METRICS
    output_dir: str = "results/toy"
    plot: bool = False

    def replace(self, **kwargs: Any) -> "AppleConfig":
        return replace(self, **kwargs)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def config_from_config(cfg: Any, name: str = "apple_retrieval") -> AppleConfig:
    """Build an :class:`AppleConfig` from a loaded ``configs/toy.yaml`` (or ``None``)."""
    block = _cfg_get(cfg, name, None)
    if block is None:
        block = cfg if _cfg_get(cfg, "apple_retrieval", None) is None else {}
    env_block = _cfg_get(cfg, "two_state_mdp", None)

    policy = _cfg_get(block, "policy", {})
    reinforce = _cfg_get(block, "reinforce", {})
    pretrain = _cfg_get(block, "pretrain", {})
    finetune = _cfg_get(block, "finetune", {})
    sweep_M = _cfg_get(block, "sweep_M", {})
    sweep_c = _cfg_get(block, "sweep_c", {})
    analysis = _cfg_get(cfg, "analysis", {})

    optimizer = _first_of(
        finetune, ("optimizer",), _first_of(reinforce, ("optimizer",), "sgd")
    )
    M = _first_of(block, ("M", "distance", "m"), DEFAULT_M)
    c = _first_of(block, ("c", "obs_scale"), DEFAULT_C)

    cfg_obj = AppleConfig(
        M=int(M),
        c=float(c),
        horizon=int(_first_of(block, ("horizon",), HORIZON)),
        reward_correct=float(_first_of(block, ("reward_step", "reward_correct"), REWARD_CORRECT)),
        reward_wrong=float(_first_of(block, ("reward_wrong", "wrong_reward"), REWARD_WRONG)),
        phases=int(len(_as_tuple(_first_of(block, ("phases",), (1, 2)), (1, 2)))),
        pretrain_episodes=int(
            _first_of(pretrain, ("episodes", "num_episodes", "num_steps"), DEFAULT_PRETRAIN_EPISODES)
        ),
        pretrain_lr=float(_first_of(pretrain, ("lr", "learning_rate"), DEFAULT_PRETRAIN_LR)),
        finetune_episodes=int(
            _first_of(finetune, ("episodes", "num_episodes", "num_steps"), DEFAULT_FINETUNE_EPISODES)
        ),
        finetune_lr=float(_first_of(finetune, ("lr", "learning_rate"), DEFAULT_FINETUNE_LR)),
        optimizer=str(optimizer),
        gamma=float(_first_of(finetune, ("gamma", "discount"), _first_of(reinforce, ("gamma",), 1.0))),
        baseline=bool(_first_of(reinforce, ("baseline", "use_baseline"), True)),
        normalize_returns=bool(
            _first_of(reinforce, ("normalize_returns", "normalize"), False)
        ),
        grad_clip=(_first_of(finetune, ("grad_clip",), None)),
        init_w=float(_first_of(policy, ("weight_init", "w_init", "w"), 0.0)),
        init_b=float(_first_of(policy, ("bias_init", "b_init", "b"), 0.0)),
        init_scale=float(_first_of(policy, ("init_scale", "scale"), 0.0)),
        eval_every=int(
            _first_of(finetune, ("eval_every", "log_every"), DEFAULT_EVAL_EVERY)
        ),
        eval_episodes=int(_first_of(finetune, ("eval_episodes",), DEFAULT_EVAL_EPISODES)),
        seeds=_as_tuple(_first_of(cfg, ("seeds",), None), DEFAULT_SEEDS),
        sweep_M=_as_tuple(_first_of(sweep_M, ("values", "M", "ms"), None), DEFAULT_SWEEP_M),
        sweep_M_c=float(_first_of(sweep_M, ("c",), DEFAULT_C)),
        sweep_c=_as_tuple(_first_of(sweep_c, ("values", "c", "cs"), None), DEFAULT_SWEEP_C),
        sweep_c_M=int(_first_of(sweep_c, ("M",), DEFAULT_M)),
        forgetting_metric=str(
            _first_of(analysis, ("forgetting_metric",), "phase2_action_agreement")
        ),
        confidence=float(_first_of(_cfg_get(cfg, "eval", {}), ("confidence",), 0.90)),
        metrics=_as_tuple(_first_of(block, ("metrics",), None), APPLE_RETRIEVAL_METRICS),
        output_dir=str(_first_of(block, ("output_dir",), "results/toy")),
        plot=bool(_first_of(block, ("plot",), False)),
    )
    if env_block is not None and _cfg_get(env_block, "gamma", None) is not None:
        pass  # two-state MDP settings are independent of AppleRetrieval
    return cfg_obj


# --------------------------------------------------------------------------------------
# Experiment drivers
# --------------------------------------------------------------------------------------


def run_apple_retrieval(
    M: int = DEFAULT_M,
    c: float = DEFAULT_C,
    seed: int = 0,
    config: Optional[AppleConfig] = None,
    record_trace: bool = True,
) -> Dict[str, Any]:
    """Run one pre-train -> fine-tune cycle and return all metrics (Appendix A.2)."""
    cfg = config or AppleConfig()
    cfg = cfg.replace(M=int(M), c=float(c))

    init_w, init_b = cfg.init_w, cfg.init_b
    if cfg.init_scale:
        rng = _as_rng(seed)
        init_w += cfg.init_scale * _uniform(rng, -1.0, 1.0)
        init_b += cfg.init_scale * _uniform(rng, -1.0, 1.0)

    policy_pre, pretrain_history = pretrain_phase2(
        M=cfg.M,
        c=cfg.c,
        episodes=cfg.pretrain_episodes,
        lr=cfg.pretrain_lr,
        optimizer=cfg.optimizer,
        gamma=cfg.gamma,
        baseline=cfg.baseline,
        horizon=cfg.horizon,
        init_w=init_w,
        init_b=init_b,
        seed=seed,
        record_every=0,
    )

    full_env = AppleRetrievalEnv(M=cfg.M, c=cfg.c, horizon=cfg.horizon, randomize_start=False)
    pre_metrics = evaluate_policy(
        full_env,
        policy_pre,
        num_episodes=cfg.eval_episodes,
        seed=seed + 101,
        reference_policy=policy_pre,
    )
    # Transfer measured *before* any fine-tuning: the pre-trained Phase-2 policy
    # applied to the full task.
    transfer_phase1_right_prob = policy_pre.prob_right(full_env.observation(PHASE_1))

    policy_ft, trace = fine_tune_full_task(
        policy_pre,
        M=cfg.M,
        c=cfg.c,
        episodes=cfg.finetune_episodes,
        lr=cfg.finetune_lr,
        optimizer=cfg.optimizer,
        gamma=cfg.gamma,
        baseline=cfg.baseline,
        horizon=cfg.horizon,
        eval_every=cfg.eval_every,
        eval_episodes=cfg.eval_episodes,
        reference_policy=policy_pre,
        seed=seed,
        grad_clip=cfg.grad_clip,
        record_trace=record_trace,
    )

    ft_metrics = trace[-1] if trace else evaluate_policy(
        full_env, policy_ft, num_episodes=cfg.eval_episodes, seed=seed + 202,
        reference_policy=policy_pre,
    )

    result: Dict[str, Any] = {
        "M": int(cfg.M),
        "c": float(cfg.c),
        "seed": int(seed),
        "w_pre": policy_pre.w,
        "b_pre": policy_pre.b,
        "w_ft": policy_ft.w,
        "b_ft": policy_ft.b,
        "delta_w": policy_ft.w - policy_pre.w,
        "delta_b": policy_ft.b - policy_pre.b,
        "bias_weight_ratio_pre": abs(policy_pre.b) / (abs(policy_pre.w) + _EPS),
        "bias_weight_ratio_ft": abs(policy_ft.b) / (abs(policy_ft.w) + _EPS),
        "forgetting": 1.0 - float(ft_metrics.get("phase2_action_agreement", 1.0)),
        "phase2_action_agreement": float(ft_metrics.get("phase2_action_agreement", 1.0)),
        "phase2_right_prob": float(ft_metrics.get("phase2_right_prob", 0.0)),
        "phase2_retention": 1.0 - float(ft_metrics.get("phase2_right_prob", 0.0)),
        "phase2_success": float(ft_metrics.get("phase2_success", 0.0)),
        "overall_success": float(ft_metrics.get("overall_success", 0.0)),
        "phase1_success": float(ft_metrics.get("phase1_success", 0.0)),
        "reached_phase2_rate": float(ft_metrics.get("reached_phase2_rate", 0.0)),
        "mean_return": float(ft_metrics.get("mean_return", 0.0)),
        "weight_norm": float(ft_metrics.get("weight_norm", abs(policy_ft.w))),
        "bias_norm": float(ft_metrics.get("bias_norm", abs(policy_ft.b))),
        "bias_weight_ratio": float(ft_metrics.get("bias_weight_ratio", 0.0)),
        "transfer_phase1_right_prob": float(transfer_phase1_right_prob),
        "ft_trace": trace,
        "pretrain_history": pretrain_history,
        "config": cfg.as_dict(),
    }
    result["pre_overall_success"] = float(pre_metrics.get("overall_success", 0.0))
    result["pre_phase2_success"] = float(pre_metrics.get("phase2_success", 0.0))
    result["pre_phase1_success"] = float(pre_metrics.get("phase1_success", 0.0))
    result["pre_reached_phase2_rate"] = float(pre_metrics.get("reached_phase2_rate", 0.0))
    result["pre_mean_return"] = float(pre_metrics.get("mean_return", 0.0))
    return result


def summarize(values: Sequence[float], confidence: float = 0.90) -> Dict[str, float]:
    """Mean and half-width of a two-sided ``confidence`` interval (normal approx.)."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"mean": float("nan"), "half_width": 0.0, "n": 0, "std": 0.0}
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return {"mean": mean, "half_width": 0.0, "n": 1, "std": 0.0}
    std = statistics.pstdev(vals) * math.sqrt(len(vals) / (len(vals) - 1))
    try:
        z = statistics.NormalDist().inv_cdf(0.5 + float(confidence) / 2.0)
    except Exception:  # pragma: no cover
        z = 1.645
    half = z * std / math.sqrt(len(vals))
    return {"mean": mean, "half_width": half, "n": len(vals), "std": std}


def aggregate_results(results: Sequence[Mapping[str, Any]],
                      metrics: Sequence[str] = APPLE_RETRIEVAL_METRICS,
                      confidence: float = 0.90) -> Dict[str, Dict[str, float]]:
    """Aggregate per-seed results into ``metric -> {mean, half_width, n, std}``."""
    keys = list(metrics) + [
        "forgetting",
        "phase2_retention",
        "phase2_right_prob",
        "bias_weight_ratio",
        "delta_w",
        "delta_b",
        "pre_overall_success",
    ]
    agg: Dict[str, Dict[str, float]] = {}
    for key in keys:
        vals = [r[key] for r in results if key in r and r[key] is not None]
        agg[key] = summarize(vals, confidence=confidence)
    return agg


def sweep_over_M(
    values: Optional[Sequence[int]] = None,
    c: Optional[float] = None,
    seeds: Optional[Sequence[int]] = None,
    config: Optional[AppleConfig] = None,
    progress: bool = False,
) -> Dict[str, Any]:
    """Sweep the distance ``M`` (Figure 10): forgetting grows with ``M``."""
    cfg = config or AppleConfig()
    Ms = _as_tuple(values, cfg.sweep_M) if values is not None else tuple(cfg.sweep_M)
    cs = float(cfg.sweep_M_c if c is None else c)
    seed_list = tuple(cfg.seeds if seeds is None else seeds)

    per_point: List[Dict[str, Any]] = []
    for M in Ms:
        runs = []
        for s in seed_list:
            runs.append(
                run_apple_retrieval(M=int(M), c=cs, seed=int(s), config=cfg.replace(c=cs, M=int(M)))
            )
        entry = {
            "M": int(M),
            "c": cs,
            "runs": runs,
            "aggregate": aggregate_results(runs, cfg.metrics, cfg.confidence),
        }
        per_point.append(entry)
        if progress:  # pragma: no cover - progress printing only
            f = entry["aggregate"]["forgetting"]
            print(f"[sweep M] M={M:>4} c={cs:<6} forgetting={f['mean']:.3f}+-{f['half_width']:.3f}")
    return {"kind": "sweep_M", "c": cs, "seeds": list(seed_list), "points": per_point}


def sweep_over_c(
    values: Optional[Sequence[float]] = None,
    M: Optional[int] = None,
    seeds: Optional[Sequence[int]] = None,
    config: Optional[AppleConfig] = None,
    progress: bool = False,
) -> Dict[str, Any]:
    """Sweep the observation scale ``c`` (Figure 11): smaller ``c`` -> more forgetting."""
    cfg = config or AppleConfig()
    cs = _as_tuple(values, cfg.sweep_c) if values is not None else tuple(cfg.sweep_c)
    Ms = int(cfg.sweep_c_M if M is None else M)
    seed_list = tuple(cfg.seeds if seeds is None else seeds)

    per_point: List[Dict[str, Any]] = []
    for c in cs:
        runs = []
        for s in seed_list:
            runs.append(
                run_apple_retrieval(M=Ms, c=float(c), seed=int(s), config=cfg.replace(c=float(c), M=Ms))
            )
        entry = {
            "M": Ms,
            "c": float(c),
            "runs": runs,
            "aggregate": aggregate_results(runs, cfg.metrics, cfg.confidence),
        }
        per_point.append(entry)
        if progress:  # pragma: no cover
            f = entry["aggregate"]["bias_weight_ratio"]
            g = entry["aggregate"]["forgetting"]
            print(
                f"[sweep c] c={c:<6} |b|/|w|={f['mean']:.2f} "
                f"forgetting={g['mean']:.3f}+-{g['half_width']:.3f}"
            )
    return {"kind": "sweep_c", "M": Ms, "seeds": list(seed_list), "points": per_point}


# --------------------------------------------------------------------------------------
# Plotting (Figures 10 and 11)
# --------------------------------------------------------------------------------------


def _get_plt():  # pragma: no cover - optional dependency
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def plot_sweep(
    sweep: Mapping[str, Any],
    path: Optional[str] = None,
    panels: Optional[Sequence[Tuple[str, str, str]]] = None,
    title: Optional[str] = None,
) -> Any:  # pragma: no cover - plotting helper
    """Plot a sweep with three panels (forgetting / performance / parameters).

    ``panels`` entries are ``(metric_key, ylabel, legend_label)``; the defaults follow
    Figure 10 (left: forgetting, centre: overall success, right: mean return) and the
    x-axis is ``M`` for a ``sweep_M`` sweep and ``c`` for a ``sweep_c`` sweep.
    """
    plt = _get_plt()
    if plt is None:
        return None
    kind = sweep.get("kind", "sweep_M")
    xkey = "M" if kind == "sweep_M" else "c"
    if panels is None:
        if kind == "sweep_M":
            panels = (
                ("forgetting", "forgetting", "forgetting"),
                ("overall_success", "success rate", "overall success"),
                ("phase2_right_prob", "P(right | phase 2)", "wrong-action prob."),
            )
        else:
            panels = (
                ("forgetting", "forgetting", "forgetting"),
                ("overall_success", "success rate", "overall success"),
                ("bias_weight_ratio", "|b| / |w|", "bias/weight ratio"),
            )
    xs = [p[xkey] for p in sweep["points"]]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4), squeeze=False)
    for ax, (key, ylabel, label) in zip(axes[0], panels):
        ys = [p["aggregate"].get(key, {}).get("mean", float("nan")) for p in sweep["points"]]
        errs = [p["aggregate"].get(key, {}).get("half_width", 0.0) for p in sweep["points"]]
        ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=label)
        ax.set_xlabel(xkey)
        ax.set_ylabel(ylabel)
        if xkey == "M":
            ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def plot_finetuning_trace(result: Mapping[str, Any], path: Optional[str] = None) -> Any:
    """Plot the fine-tuning trace of a single run (forgetting and parameters over time)."""
    plt = _get_plt()
    if plt is None:
        return None
    trace = result.get("ft_trace") or []
    if not trace:
        return None
    steps = [row["step"] for row in trace]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
    axes[0].plot(steps, [row["phase2_right_prob"] for row in trace], color="C3")
    axes[0].set_xlabel("fine-tuning episode")
    axes[0].set_ylabel(r"$\pi_{w,b}(right \mid o=[c])$")
    axes[0].set_title("Phase-2 forgetting")
    axes[1].plot(steps, [row["overall_success"] for row in trace], color="C0",
                 label="overall")
    axes[1].plot(steps, [row["phase2_success"] for row in trace], color="C1",
                 label="phase 2")
    axes[1].set_xlabel("fine-tuning episode")
    axes[1].set_ylabel("success rate")
    axes[1].legend(fontsize=8)
    axes[2].plot(steps, [row["w"] for row in trace], label="w")
    axes[2].plot(steps, [row["b"] for row in trace], label="b")
    axes[2].set_xlabel("fine-tuning episode")
    axes[2].set_ylabel("parameter value")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: ``python -m src.toy.apple_retrieval [options]``."""
    parser = argparse.ArgumentParser(description="AppleRetrieval toy experiment (Appendix A.2)")
    parser.add_argument("--config", type=str, default=None,
                        help="path to configs/toy.yaml (optional)")
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "sweep-M", "sweep-c", "both"],
                        help="run a single configuration or one of the sweeps")
    parser.add_argument("--M", type=int, default=None, help="distance home -> apple")
    parser.add_argument("--c", type=float, default=None, help="observation scale")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pretrain-episodes", type=int, default=None)
    parser.add_argument("--finetune-episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None, help="comma separated seeds")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--plot", action="store_true", help="write figures to disk")
    args = parser.parse_args(argv)

    cfg: AppleConfig
    if args.config:
        try:
            from src.common.config import load_config

            cfg = config_from_config(load_config(args.config))
        except Exception:
            cfg = AppleConfig()
    else:
        cfg = AppleConfig()

    overrides: Dict[str, Any] = {}
    if args.M is not None:
        overrides["M"] = args.M
    if args.c is not None:
        overrides["c"] = args.c
    if args.pretrain_episodes is not None:
        overrides["pretrain_episodes"] = args.pretrain_episodes
    if args.finetune_episodes is not None:
        overrides["finetune_episodes"] = args.finetune_episodes
    if args.eval_episodes is not None:
        overrides["eval_episodes"] = args.eval_episodes
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.seeds is not None:
        overrides["seeds"] = _as_tuple(args.seeds, cfg.seeds)
    if overrides:
        cfg = cfg.replace(**overrides)

    os.makedirs(cfg.output_dir, exist_ok=True)

    if args.mode in ("single",):
        seed = 0 if args.seed is None else int(args.seed)
        res = run_apple_retrieval(M=cfg.M, c=cfg.c, seed=seed, config=cfg)
        print(
            f"AppleRetrieval M={cfg.M} c={cfg.c} seed={seed}: "
            f"w_pre={res['w_pre']:.3f} b_pre={res['b_pre']:.3f} -> "
            f"w_ft={res['w_ft']:.3f} b_ft={res['b_ft']:.3f}"
        )
        print(
            f"  forgetting={res['forgetting']:.3f} "
            f"phase2_success={res['phase2_success']:.3f} "
            f"overall_success={res['overall_success']:.3f} "
            f"(pre-trained overall success: {res['pre_overall_success']:.3f})"
        )
        if cfg.plot:
            plot_finetuning_trace(res, os.path.join(cfg.output_dir, f"trace_M{cfg.M}_c{cfg.c}.png"))
        return 0

    seeds = tuple(int(s) for s in cfg.seeds)
    exit_code = 0
    if args.mode in ("sweep-M", "both"):
        sweep = sweep_over_M(seeds=seeds, config=cfg, progress=True)
        out = os.path.join(cfg.output_dir, "sweep_M.json")
        _dump_json(sweep, out)
        if cfg.plot or args.plot:
            plot_sweep(sweep, os.path.join(cfg.output_dir, "figure10.png"),
                       title="AppleRetrieval: forgetting vs distance M (Figure 10)")
    if args.mode in ("sweep-c", "both"):
        sweep = sweep_over_c(seeds=seeds, config=cfg, progress=True)
        out = os.path.join(cfg.output_dir, "sweep_c.json")
        _dump_json(sweep, out)
        if cfg.plot or args.plot:
            plot_sweep(sweep, os.path.join(cfg.output_dir, "figure11.png"),
                       title="AppleRetrieval: effect of c (Figure 11)")
    return exit_code


def _dump_json(payload: Mapping[str, Any], path: str) -> None:  # pragma: no cover
    import json

    def _default(obj: Any) -> Any:
        if isinstance(obj, Mapping):
            return {str(k): _default(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_default(v) for v in obj]
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        return str(obj)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    slim = {
        "kind": payload.get("kind"),
        "seeds": payload.get("seeds"),
        "points": [
            {"M": p.get("M"), "c": p.get("c"), "aggregate": p.get("aggregate")}
            for p in payload.get("points", [])
        ],
    }
    with open(path, "w") as fh:
        json.dump(_default(slim), fh, indent=2)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

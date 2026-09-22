r"""Mixture prior over random unsupervised reward functions (FRE, Section 4.2).

Paper references
----------------
Section 4.2 ("Random Functions as a Prior Reward Distribution"):

    "In our implementation, we found that a reasonable yet powerful prior
    distribution can be constructed from a mixture of random unsupervised
    functions. The particular mixture we use consists of random singleton
    functions (corresponding to "goal reaching" rewards), random neural
    networks (MLPs with two linear layers), and random linear functions
    (corresponding to "MLPs" with one linear layer). ... A uniform mixture of
    the three function classes are used during training."

Table 3 (Appendix A) hyper-parameters::

    Ratio of Goal-Reaching Rewards & 0.33
    Ratio of Linear Rewards        & 0.33
    Ratio of Randomm MLP Rewards   & 0.33

Addendum ("Clarifications on FRE Prior Reward Distributions") definitions:

    FRE-all      : equal split goal-reaching / random linear / random MLP.
    FRE-goals    : exclusively singleton goal-reaching reward functions.
    FRE-lin      : exclusively random linear reward functions.
    FRE-mlp      : exclusively random MLP reward functions.
    FRE-lin-mlp  : equal split of random linear and random MLP.
    FRE-goal-mlp : equal split of singleton goal-reaching and random MLP.
    FRE-goal-lin : equal split of singleton goal-reaching and random linear.
    FRE-hint     : "a prior reward distribution that is a superset of the
                    evaluation tasks. For ant-directional, the prior rewards
                    are all reward corresponding to movement in a unit (x,y)
                    direction. For Cheetah-velocity and walker-velocity, the
                    rewards are for moving at a specific velocity"

This module provides:

* :class:`MixtureRewardPrior` - a weighted mixture over component priors
  (:class:`~fre.reward_priors.goal_reaching.GoalReachingPrior`,
  :class:`~fre.reward_priors.linear.LinearRewardPrior`,
  :class:`~fre.reward_priors.random_mlp.RandomMLPPrior`).
* :data:`PRIOR_VARIANTS` - the seven named variants above, as ratio dicts.
* :func:`make_mixture_prior` / :func:`make_prior_from_variant` - factories.
* :class:`DirectionalPrior` / :class:`VelocityPrior` - the ``FRE-hint``
  component priors (unit (x,y) movement direction; target speed).
* :func:`make_hint_prior` - a superset prior for the hint domains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # normal package import
    from fre.reward_priors.goal_reaching import (
        RewardFunction,
        RewardPrior,
        GoalReachingPrior,
        as_numpy_2d,
    )
    from fre.reward_priors.linear import LinearRewardPrior, make_linear_prior
    from fre.reward_priors.random_mlp import RandomMLPPrior, make_mlp_prior
except ImportError:  # pragma: no cover - direct module execution
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from fre.reward_priors.goal_reaching import (  # type: ignore
        RewardFunction,
        RewardPrior,
        GoalReachingPrior,
        as_numpy_2d,
    )
    from fre.reward_priors.linear import LinearRewardPrior, make_linear_prior  # type: ignore
    from fre.reward_priors.random_mlp import RandomMLPPrior, make_mlp_prior  # type: ignore


__all__ = [
    "DEFAULT_PRIOR_RATIOS",
    "PRIOR_VARIANTS",
    "COMPONENT_NAMES",
    "MixtureRewardPrior",
    "make_mixture_prior",
    "make_prior_from_variant",
    "variant_ratios",
    "DirectionalReward",
    "DirectionalPrior",
    "VelocityReward",
    "VelocityPrior",
    "HintMixturePrior",
    "make_hint_prior",
    "HINT_DOMAINS",
    "ANTMAZE_VELOCITY_DIMS",
]


# ---------------------------------------------------------------------------
# Variant definitions (Table 4 / Addendum)
# ---------------------------------------------------------------------------
# Order of the components is always (goal-reaching, linear, MLP), matching the
# order used in Section 4.2 and Table 3.
COMPONENT_NAMES: Tuple[str, str, str] = ("goal", "linear", "mlp")

#: Table 3: "Ratio of Goal-Reaching Rewards 0.33 / Linear 0.33 / Random MLP 0.33"
DEFAULT_PRIOR_RATIOS: Tuple[float, float, float] = (0.33, 0.33, 0.33)

#: Named prior distributions of the Addendum prior clarifications (Table 4).
PRIOR_VARIANTS: Dict[str, Tuple[float, float, float]] = {
    # (goal-reaching, linear, MLP)
    "FRE-all": (0.33, 0.33, 0.33),       # vanilla prior (Sections 5.1/5.2/5.4)
    "FRE": (0.33, 0.33, 0.33),           # alias used in the paper's text
    "FRE-goals": (1.0, 0.0, 0.0),        # goal-reaching only
    "FRE-goal": (1.0, 0.0, 0.0),         # alias
    "FRE-lin": (0.0, 1.0, 0.0),          # linear only
    "FRE-linear": (0.0, 1.0, 0.0),       # alias
    "FRE-mlp": (0.0, 0.0, 1.0),          # MLP only
    "FRE-lin-mlp": (0.0, 0.5, 0.5),      # equal split linear + MLP
    "FRE-goal-mlp": (0.5, 0.0, 0.5),     # equal split goal + MLP
    "FRE-goal-lin": (0.5, 0.5, 0.0),     # equal split goal + linear
}

#: Variants whose prior is a *superset* of the evaluation tasks (Section 5.4).
HINT_VARIANTS: Tuple[str, ...] = ("FRE-hint", "FRE-hint-ant-directional",
                                  "FRE-hint-cheetah-velocity", "FRE-hint-walker-velocity")


def variant_ratios(variant: str) -> Tuple[float, float, float]:
    """Return the ``(goal, linear, mlp)`` ratios of a named prior variant."""
    if variant not in PRIOR_VARIANTS:
        raise KeyError(
            "Unknown prior variant {!r}. Known variants: {}".format(
                variant, sorted(PRIOR_VARIANTS.keys()) + list(HINT_VARIANTS)
            )
        )
    return PRIOR_VARIANTS[variant]


# ---------------------------------------------------------------------------
# Mixture prior
# ---------------------------------------------------------------------------
class MixtureRewardPrior(RewardPrior):
    """Weighted mixture of component prior reward distributions.

    Sampling draws a component index with probability proportional to its
    weight (the paper's uniform mixture of 0.33/0.33/0.33 for ``FRE-all``) and
    then samples a reward function from that component.  A sampled reward
    function is a plain object implementing the
    :class:`~fre.reward_priors.goal_reaching.RewardFunction` API, i.e. it can be
    called as ``eta = prior.sample(rng); rewards, dones = eta.label(states)``.

    Parameters
    ----------
    components:
        Mapping ``name -> RewardPrior``. Entries whose weight is zero may be
        omitted; zero-weight components that are present are never sampled.
    weights:
        Mapping ``name -> weight``. Weights are normalised to sum to one.
    name:
        Name of this mixture (used for logging / descriptions).
    seed:
        Seed of the internal default RNG.
    """

    def __init__(
        self,
        components: Mapping[str, RewardPrior],
        weights: Optional[Mapping[str, float]] = None,
        name: str = "mixture",
        seed: int = 0,
    ) -> None:
        if len(components) == 0:
            raise ValueError("MixtureRewardPrior requires at least one component prior")
        if weights is None:
            weights = {key: 1.0 for key in components}

        names: List[str] = []
        priors: List[RewardPrior] = []
        raw_weights: List[float] = []
        for key, prior in components.items():
            w = float(weights.get(key, 0.0))
            if w < 0.0:
                raise ValueError("Prior mixture weights must be non-negative ({} -> {})".format(key, w))
            if w == 0.0:
                continue  # zero-weight component: never sampled
            names.append(key)
            priors.append(prior)
            raw_weights.append(w)
        if len(names) == 0:
            raise ValueError("MixtureRewardPrior requires at least one component with positive weight")

        weight_array = np.asarray(raw_weights, dtype=np.float64)
        weight_array = weight_array / weight_array.sum()

        self.name = name
        self.component_names: List[str] = names
        self.component_priors: List[RewardPrior] = priors
        self.weights: np.ndarray = weight_array
        self._rng = np.random.default_rng(seed)

    # -- construction helpers ------------------------------------------------
    @classmethod
    def from_ratios(
        cls,
        ratios: Mapping[str, float],
        replay_buffer: Any = None,
        state_dim: Optional[int] = None,
        exclude_dims: Sequence[int] = (),
        hidden_dim: int = 32,
        goal_threshold: float = 0.0,
        seed: int = 0,
        name: Optional[str] = None,
        **prior_kwargs: Any,
    ) -> "MixtureRewardPrior":
        """Build a mixture from dict ``{component_name: ratio}``.

        Components with a zero ratio are never instantiated, which also keeps
        the ablation runs (e.g. ``FRE-lin``) free of unnecessary goal sampling.
        """
        components: Dict[str, RewardPrior] = {}
        for key, ratio in ratios.items():
            if float(ratio) <= 0.0:
                continue
            components[key] = build_component_prior(
                key,
                replay_buffer=replay_buffer,
                state_dim=state_dim,
                exclude_dims=exclude_dims,
                hidden_dim=hidden_dim,
                goal_threshold=goal_threshold,
                seed=seed,
                **prior_kwargs,
            )
        if name is None:
            name = "mixture"
        return cls(components=components, weights=dict(ratios), name=name, seed=seed)

    # -- RewardPrior API -----------------------------------------------------
    def sample(self, rng: Optional[np.random.Generator] = None) -> RewardFunction:
        """Sample a component index, then a reward function from that component."""
        if rng is None:
            rng = self._rng
        idx = int(rng.choice(len(self.component_priors), p=self.weights))
        return self.component_priors[idx].sample(rng)

    def sample_many(
        self, num_samples: int = 1, rng: Optional[np.random.Generator] = None
    ) -> List[RewardFunction]:
        if rng is None:
            rng = self._rng
        return [self.sample(rng) for _ in range(int(num_samples))]

    def component_probabilities(self) -> Dict[str, float]:
        """Empirical mixture probabilities keyed by component name."""
        return {
            key: float(w) for key, w in zip(self.component_names, self.weights.tolist())
        }

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "mixture",
            "name": self.name,
            "components": self.component_probabilities(),
            "component_details": {
                key: prior.describe() for key, prior in zip(self.component_names, self.component_priors)
            },
        }

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        parts = ", ".join(
            "{}={:.3f}".format(key, w) for key, w in zip(self.component_names, self.weights.tolist())
        )
        return "{}({})".format(type(self).__name__, parts)


# ---------------------------------------------------------------------------
# Component construction
# ---------------------------------------------------------------------------
def build_component_prior(
    name: str,
    replay_buffer: Any = None,
    state_dim: Optional[int] = None,
    exclude_dims: Sequence[int] = (),
    hidden_dim: int = 32,
    goal_threshold: float = 0.0,
    seed: int = 0,
    **kwargs: Any,
) -> RewardPrior:
    """Instantiate one of the three prior classes of Section 4.2 by name."""
    key = str(name).lower()
    if key in ("goal", "goals", "goal-reaching", "goal_reaching", "singleton"):
        # Singleton goal-reaching rewards: HER sampling, reward -1 / 0, done mask.
        return GoalReachingPrior(
            replay_buffer=replay_buffer,
            threshold=goal_threshold,
            seed=seed,
            **kwargs,
        )
    if key in ("linear", "lin"):
        return make_linear_prior(
            replay_buffer=replay_buffer,
            state_dim=state_dim,
            exclude_dims=exclude_dims,
            seed=seed,
            **kwargs,
        )
    if key in ("mlp", "random_mlp", "random-mlp"):
        return make_mlp_prior(
            replay_buffer=replay_buffer,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            seed=seed,
            **kwargs,
        )
    raise KeyError("Unknown prior component {!r} (expected goal / linear / mlp)".format(name))


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def make_mixture_prior(
    ratios: Optional[Mapping[str, float]] = None,
    replay_buffer: Any = None,
    state_dim: Optional[int] = None,
    exclude_dims: Sequence[int] = (),
    hidden_dim: int = 32,
    goal_threshold: float = 0.0,
    seed: int = 0,
    name: Optional[str] = None,
    **prior_kwargs: Any,
) -> MixtureRewardPrior:
    """Create a :class:`MixtureRewardPrior` from explicit ratios.

    ``ratios`` maps ``{"goal": ..., "linear": ..., "mlp": ...}`` fractions.
    When omitted, the paper's uniform mixture (0.33/0.33/0.33, Table 3) is used.
    """
    if ratios is None:
        ratios = {c: r for c, r in zip(COMPONENT_NAMES, DEFAULT_PRIOR_RATIOS)}
    return MixtureRewardPrior.from_ratios(
        ratios=ratios,
        replay_buffer=replay_buffer,
        state_dim=state_dim,
        exclude_dims=exclude_dims,
        hidden_dim=hidden_dim,
        goal_threshold=goal_threshold,
        seed=seed,
        name=name,
        **prior_kwargs,
    )


def make_prior_from_variant(
    variant: str = "FRE-all",
    replay_buffer: Any = None,
    state_dim: Optional[int] = None,
    exclude_dims: Sequence[int] = (),
    hidden_dim: int = 32,
    goal_threshold: float = 0.0,
    seed: int = 0,
    **prior_kwargs: Any,
) -> RewardPrior:
    """Factory for a named prior distribution (Table 4 sweep / Section 5.4).

    ``FRE-all`` suffices for the main experiments; the ablated variants
    (``FRE-goals``, ``FRE-lin``, ``FRE-mlp``, ``FRE-lin-mlp``, ``FRE-goal-mlp``,
    ``FRE-goal-lin``) are used for the prior-scaling analysis, and ``FRE-hint*``
    for the privileged-prior variant.
    """
    if variant in HINT_VARIANTS:
        domain = variant.replace("FRE-hint", "").lstrip("-_")
        return make_hint_prior(
            domain=domain or "ant-directional",
            replay_buffer=replay_buffer,
            state_dim=state_dim,
            exclude_dims=exclude_dims,
            hidden_dim=hidden_dim,
            seed=seed,
            **prior_kwargs,
        )
    ratios = variant_ratios(variant)
    ratio_dict = {c: r for c, r in zip(COMPONENT_NAMES, ratios)}
    return MixtureRewardPrior.from_ratios(
        ratios=ratio_dict,
        replay_buffer=replay_buffer,
        state_dim=state_dim,
        exclude_dims=exclude_dims,
        hidden_dim=hidden_dim,
        goal_threshold=goal_threshold,
        seed=seed,
        name=variant,
        **prior_kwargs,
    )


# ---------------------------------------------------------------------------
# FRE-hint: superset priors over the evaluation tasks (Section 5.4)
# ---------------------------------------------------------------------------
#: AntMaze state layout: qpos (15) + qvel (14); the ant's planar root linear
#: velocity lives in the first two qvel entries, i.e. observation dims 15, 16.
ANTMAZE_VELOCITY_DIMS: Tuple[int, int] = (15, 16)

#: Velocity component of the state for the ExORL hint domains.  dm_control
#: cheetah: qpos (8) + qvel (9) -> horizontal speed is the root x velocity.
#: dm_control walker: qpos (9) + qvel (9) -> horizontal velocity averages the
#: root x velocity and the torso x velocity (dims 0 and 1 of qvel).
HINT_VELOCITY_DIMS: Dict[str, Tuple[int, ...]] = {
    "cheetah": (0,),
    "walker": (0, 1),
}

#: Reference velocity magnitudes used to build the hint velocity grid.
HINT_VELOCITY_RANGES: Dict[str, Tuple[float, float]] = {
    "cheetah": (0.0, 12.0),   # cheetah-run tasks use thresholds up to 10
    "walker": (0.0, 8.0),     # walker tasks use thresholds up to 8
}


class DirectionalReward(RewardFunction):
    """Reward for moving in a unit ``(x, y)`` direction: ``eta(s) = d . v(s)``.

    Used by ``FRE-hint`` for ``ant-directional`` (Addendum: "the prior rewards
    are all reward corresponding to movement in a unit (x,y) direction").  The
    AntMaze directional evaluation tasks compute the dot product of the target
    unit direction with the ant's actual planar velocity, which is exactly the
    functional form implemented here.
    """

    name = "directional"

    def __init__(
        self,
        direction: Sequence[float],
        velocity_dims: Sequence[int] = ANTMAZE_VELOCITY_DIMS,
        velocity_scale: float = 1.0,
        reward_min: Optional[float] = None,
        reward_max: Optional[float] = None,
        terminate_on_success: bool = False,
    ) -> None:
        direction = np.asarray(direction, dtype=np.float64).reshape(-1)
        if direction.size != 2:
            raise ValueError("DirectionalReward expects a 2-d (x, y) direction")
        norm = float(np.linalg.norm(direction))
        if norm > 0:
            direction = direction / norm
        self.direction = direction.astype(np.float32)
        self.velocity_dims = tuple(int(d) for d in velocity_dims)
        self.velocity_scale = float(velocity_scale)
        self.reward_unreached = -1.0
        self.reward_reached = 0.0
        self.terminate_on_success = bool(terminate_on_success)
        self.reward_min = -1.0 if reward_min is None else float(reward_min)
        self.reward_max = 1.0 if reward_max is None else float(reward_max)

    def velocities(self, states: np.ndarray) -> np.ndarray:
        states = as_numpy_2d(states)
        return states[:, list(self.velocity_dims)]

    def _rewards_and_dones(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        states = as_numpy_2d(states)
        velocities = states[:, list(self.velocity_dims)]
        rewards = (velocities @ self.direction.astype(states.dtype)) * self.velocity_scale
        if self.terminate_on_success:
            dones = rewards > 0.0
        else:
            dones = np.zeros(rewards.shape[0], dtype=bool)
        return rewards.astype(np.float32), dones

    def reward_bounds(self, states: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Reward range (computed from observed states when available)."""
        if states is not None:
            rewards = self._rewards_and_dones(states)[0]
            if rewards.size:
                return float(rewards.min()), float(rewards.max())
        return self.reward_min, self.reward_max

    def ensure_goal_in_set(self, states: np.ndarray, rng: Any = None, in_place: bool = True) -> np.ndarray:
        """No distinguished goal state: nothing to guarantee."""
        return as_numpy_2d(states)

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "directional",
            "direction": self.direction.tolist(),
            "velocity_dims": list(self.velocity_dims),
        }


class DirectionalPrior(RewardPrior):
    """Prior over unit (x, y) movement directions (uniform on the unit circle).

    The Addendum does not specify a discretisation of the hint directions, so
    the default is a continuous uniform angle; pass ``num_directions=n`` to
    restrict samples to ``n`` evenly spaced unit directions (e.g. the four axis
    directions of the ant-directional evaluation suite).
    """

    name = "directional_prior"

    def __init__(
        self,
        velocity_dims: Sequence[int] = ANTMAZE_VELOCITY_DIMS,
        num_directions: Optional[int] = None,
        velocity_scale: float = 1.0,
        include_axes_only: bool = False,
        seed: int = 0,
        name: str = "directional_prior",
    ) -> None:
        self.name = name
        self.velocity_dims = tuple(int(d) for d in velocity_dims)
        self.velocity_scale = float(velocity_scale)
        if include_axes_only and num_directions is None:
            num_directions = 4
        self.num_directions = None if num_directions is None else int(num_directions)
        if self.num_directions is not None:
            if include_axes_only and self.num_directions == 4:
                # (-1, 0), (0, 1), (0, -1), (1, 0): the four evaluation directions
                self._directions = np.asarray(
                    [[-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 0.0]], dtype=np.float32
                )
            else:
                angles = np.linspace(0.0, 2.0 * np.pi, self.num_directions, endpoint=False)
                self._directions = np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)
        else:
            self._directions = None
        self._rng = np.random.default_rng(seed)

    def sample(self, rng: Optional[np.random.Generator] = None) -> DirectionalReward:
        if rng is None:
            rng = self._rng
        if self._directions is not None:
            direction = self._directions[rng.integers(len(self._directions))]
        else:
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
        return DirectionalReward(
            direction=direction,
            velocity_dims=self.velocity_dims,
            velocity_scale=self.velocity_scale,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "directional_prior",
            "velocity_dims": list(self.velocity_dims),
            "num_directions": self.num_directions,
        }


class VelocityReward(RewardFunction):
    """Reward for moving at a specific target velocity (FRE-hint, Section 5.4).

    ``mode="match"`` (default): ``eta(s) = -|v(s) - v_target| / scale``, i.e. the
    reward is 0 exactly at the target velocity and decreases linearly with the
    distance to it.  ``mode="forward"``: ``eta(s) = min(v, v_target) / scale``
    clipped at 0, i.e. reward for moving forward up to the target velocity.

    ``v(s)`` is the mean of the configured velocity dimensions of the state
    (Appendix C.2 lists the ExORL physics quantities available: cheetah
    ``speed``; walker ``horizontal_velocity``).
    """

    name = "velocity"

    def __init__(
        self,
        target_velocity: float,
        velocity_dims: Sequence[int] = (0,),
        mode: str = "match",
        scale: float = 10.0,
        reward_min: Optional[float] = None,
        reward_max: Optional[float] = None,
        terminate_on_success: bool = False,
    ) -> None:
        self.target_velocity = float(target_velocity)
        self.velocity_dims = tuple(int(d) for d in velocity_dims)
        mode = str(mode).lower()
        if mode not in ("match", "forward"):
            raise ValueError("VelocityReward mode must be 'match' or 'forward'")
        self.mode = mode
        self.scale = float(scale) if scale else 1.0
        self.reward_unreached = -1.0
        self.reward_reached = 0.0
        self.terminate_on_success = bool(terminate_on_success)
        if reward_min is None:
            reward_min = -1.0 if mode == "match" else 0.0
        if reward_max is None:
            reward_max = 0.0 if mode == "match" else 1.0
        self.reward_min = float(reward_min)
        self.reward_max = float(reward_max)

    def velocities(self, states: np.ndarray) -> np.ndarray:
        states = as_numpy_2d(states)
        return states[:, list(self.velocity_dims)].mean(axis=1)

    def _rewards_and_dones(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        velocities = self.velocities(states)
        if self.mode == "match":
            rewards = -np.abs(velocities - self.target_velocity) / self.scale
        else:  # forward
            rewards = np.clip(velocities, 0.0, self.target_velocity) / self.scale
        dones = np.zeros(rewards.shape[0], dtype=bool)
        return rewards.astype(np.float32), dones

    def reward_bounds(self, states: Optional[np.ndarray] = None) -> Tuple[float, float]:
        if states is not None:
            rewards = self._rewards_and_dones(states)[0]
            if rewards.size:
                return float(rewards.min()), float(rewards.max())
        return self.reward_min, self.reward_max

    def ensure_goal_in_set(self, states: np.ndarray, rng: Any = None, in_place: bool = True) -> np.ndarray:
        return as_numpy_2d(states)

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "velocity",
            "target_velocity": self.target_velocity,
            "mode": self.mode,
            "velocity_dims": list(self.velocity_dims),
            "scale": self.scale,
        }


class VelocityPrior(RewardPrior):
    """Prior over target velocities for a domain (``cheetah`` / ``walker``)."""

    name = "velocity_prior"

    def __init__(
        self,
        domain: str = "cheetah",
        velocity_dims: Optional[Sequence[int]] = None,
        num_velocities: int = 12,
        velocity_range: Optional[Tuple[float, float]] = None,
        mode: str = "match",
        scale: Optional[float] = None,
        seed: int = 0,
        name: Optional[str] = None,
    ) -> None:
        domain = str(domain).lower()
        self.domain = domain
        self.velocity_dims = tuple(
            int(d) for d in (velocity_dims if velocity_dims is not None else HINT_VELOCITY_DIMS.get(domain, (0,)))
        )
        if velocity_range is None:
            velocity_range = HINT_VELOCITY_RANGES.get(domain, (0.0, 10.0))
        self.velocity_range = (float(velocity_range[0]), float(velocity_range[1]))
        self.num_velocities = int(num_velocities)
        # Evenly spaced grid spanning the reference velocity range; the paper
        # only says "rewards for moving at a specific velocity".
        self._velocities = np.linspace(
            self.velocity_range[0], self.velocity_range[1], max(self.num_velocities, 2)
        )
        self.mode = mode
        self.scale = float(scale) if scale is not None else max(self.velocity_range[1], 1.0)
        self.name = name if name is not None else "velocity_prior_{}".format(domain)
        self._rng = np.random.default_rng(seed)

    def sample(self, rng: Optional[np.random.Generator] = None) -> VelocityReward:
        if rng is None:
            rng = self._rng
        target = float(self._velocities[rng.integers(len(self._velocities))])
        return VelocityReward(
            target_velocity=target,
            velocity_dims=self.velocity_dims,
            mode=self.mode,
            scale=self.scale,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "velocity_prior",
            "domain": self.domain,
            "velocity_dims": list(self.velocity_dims),
            "velocity_range": list(self.velocity_range),
            "num_velocities": self.num_velocities,
            "mode": self.mode,
        }


class HintMixturePrior(MixtureRewardPrior):
    """``FRE-hint``: mixture of the vanilla prior and a task-related prior.

    Section 5.4 states the hint prior is "a superset of the evaluation tasks":
    it contains reward functions of the evaluation family (unit (x,y) movement
    directions for ant-directional, target velocities for cheetah/walker) *in
    addition to* the vanilla unsupervised prior.  The Addendum does not give an
    explicit mixing ratio between the hint component and the vanilla prior, so
    the default is 0.5/0.5 (documented default).
    """

    def __init__(
        self,
        components: Mapping[str, RewardPrior],
        weights: Optional[Mapping[str, float]] = None,
        name: str = "FRE-hint",
        seed: int = 0,
        hint_probability: float = 0.5,
    ) -> None:
        super().__init__(components=components, weights=weights, name=name, seed=seed)
        self.hint_probability = float(hint_probability)


#: Hint domains recognised by :func:`make_hint_prior`.
HINT_DOMAINS: Dict[str, str] = {
    "ant-directional": "directional",
    "ant_directional": "directional",
    "directional": "directional",
    "cheetah-velocity": "velocity",
    "cheetah_velocity": "velocity",
    "walker-velocity": "velocity",
    "walker_velocity": "velocity",
    "velocity": "velocity",
}


def make_hint_prior(
    domain: str = "ant-directional",
    replay_buffer: Any = None,
    state_dim: Optional[int] = None,
    exclude_dims: Sequence[int] = (),
    hidden_dim: int = 32,
    seed: int = 0,
    hint_probability: float = 0.5,
    velocity_dims: Optional[Sequence[int]] = None,
    num_directions: Optional[int] = None,
    name: Optional[str] = None,
    **prior_kwargs: Any,
) -> MixtureRewardPrior:
    """Build the ``FRE-hint`` prior for a hint domain (Section 5.4).

    The returned mixture contains a hint component (directional movement for
    ``ant-directional``; target-velocity rewards for the cheetah/walker
    velocity tasks) plus the vanilla ``FRE-all`` prior, i.e. it is a superset of
    the evaluation tasks as described in the Addendum.
    """
    domain_key = str(domain).lower()
    if domain_key not in HINT_DOMAINS:
        raise KeyError(
            "Unknown hint domain {!r}; expected one of {}".format(domain, sorted(HINT_DOMAINS.keys()))
        )
    kind = HINT_DOMAINS[domain_key]

    if kind == "directional":
        hint_prior: RewardPrior = DirectionalPrior(
            velocity_dims=velocity_dims if velocity_dims is not None else ANTMAZE_VELOCITY_DIMS,
            num_directions=num_directions,
            seed=seed,
        )
        hint_name = "hint_directional"
    else:
        # Velocity hints are defined per environment domain; recover it from the
        # domain string ("cheetah-velocity" / "walker-velocity").
        env = "walker" if "walker" in domain_key else "cheetah"
        hint_prior = VelocityPrior(
            domain=env,
            velocity_dims=velocity_dims,
            seed=seed,
        )
        hint_name = "hint_velocity_{}".format(env)

    vanilla = make_mixture_prior(
        ratios=None,
        replay_buffer=replay_buffer,
        state_dim=state_dim,
        exclude_dims=exclude_dims,
        hidden_dim=hidden_dim,
        seed=seed,
        name="FRE-all",
        **prior_kwargs,
    )

    # Superset: hint family + vanilla unsupervised prior.
    components: Dict[str, RewardPrior] = {"hint": hint_prior, "vanilla": vanilla}
    weights = {"hint": float(hint_probability), "vanilla": 1.0 - float(hint_probability)}
    return HintMixturePrior(
        components=components,
        weights=weights,
        name=name or "FRE-hint-{}".format(domain_key),
        seed=seed,
        hint_probability=hint_probability,
    )

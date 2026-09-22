"""Reward prior distribution used to train the FRE encoder/decoder.

Paper: "Zero-Shot Reinforcement Learning via Functional Reward Encodings".

From Section 4.2 of the paper:

    "In our implementation, we found that a reasonable yet powerful prior
     distribution can be constructed from a mixture of random unsupervised
     functions. The particular mixture we use consists of random singleton
     functions (corresponding to "goal reaching" rewards), random neural
     networks (MLPs with two linear layers), and random linear functions
     (corresponding to "MLPs" with one linear layer). [...] A uniform mixture of
     the three function classes are used during training."

From Appendix B of the paper:

    - Random goal-reaching functions: goals are drawn using a hindsight
      experience relabelling distribution: given a randomly selected state, use
      that state as the goal with 0.2 chance, a future state within the
      trajectory with 0.5 chance and a completely random state with 0.3 chance.
      Reward is -1 for every timestep the goal is not achieved and a done mask is
      True when the goal is achieved. At least one of the encoding samples
      contains the goal state.
    - Random linear functions: uniform vector in [-1, 1]; on AntMaze the XY
      positions are removed; an independent Bernoulli(0.9) mask zeroes
      dimensions to encourage sparsity.
    - Random MLP functions: (state_dim, 32, 1) network with tanh activation
      between the two layers, weights drawn from a normal distribution scaled by
      the average dimension of the layer, output clipped to [-1, 1].

From the addendum ("Clarifications on FRE Prior Reward Distributions"), the
ablations studied in Section 5.3 are the presets ``FRE-all`` (uniform 1/3
mixture), ``FRE-goals``, ``FRE-lin``, ``FRE-mlp`` (single families),
``FRE-lin-mlp``, ``FRE-goal-mlp`` and ``FRE-goal-lin`` (equal splits), and
``FRE-hint`` (§5.4), whose prior reward distribution is a *superset of the
evaluation tasks* (all unit (x, y) directions for ant-directional, specific
target velocities for cheetah/walker velocity tasks).

This module is the dispatcher that turns those specifications into batchable
``(state, reward)`` samples for the FRE variational objective.  It deliberately
depends only on numpy (plus torch for the tensor hand-off) and the three family
modules in ``fre/priors``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - torch is an optional convenience here
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

# ---------------------------------------------------------------------------
# sibling modules (relative import first so the package works from any cwd)
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from fre.priors.goal_functions import (
        DEFAULT_GOAL_THRESHOLD,
        HER_P_CURRENT,
        HER_P_FUTURE,
        HER_P_RANDOM,
        DirectionalRewardFunction,
        GoalRewardFunction,
        RewardFunction,
        VelocityRewardFunction,
        extract_trajectory_bounds,
        goal_distances,
        make_directional_reward_functions,
        make_velocity_hint_functions,
        sample_her_goals,
    )
    from fre.priors.linear_functions import (
        DEFAULT_MASK_PROB,
        LinearRewardFunction,
        sample_linear_weights,
    )
    from fre.priors.mlp_functions import (
        DEFAULT_CLIP_VALUE,
        DEFAULT_HIDDEN_DIM,
        MLPRewardFunction,
        sample_mlp_parameters,
    )
except ImportError:  # pragma: no cover - fallback for direct module imports
    from goal_functions import (  # type: ignore
        DEFAULT_GOAL_THRESHOLD,
        HER_P_CURRENT,
        HER_P_FUTURE,
        HER_P_RANDOM,
        DirectionalRewardFunction,
        GoalRewardFunction,
        RewardFunction,
        VelocityRewardFunction,
        extract_trajectory_bounds,
        goal_distances,
        make_directional_reward_functions,
        make_velocity_hint_functions,
        sample_her_goals,
    )
    from linear_functions import (  # type: ignore
        DEFAULT_MASK_PROB,
        LinearRewardFunction,
        sample_linear_weights,
    )
    from mlp_functions import (  # type: ignore
        DEFAULT_CLIP_VALUE,
        DEFAULT_HIDDEN_DIM,
        MLPRewardFunction,
        sample_mlp_parameters,
    )


__all__ = [
    "FAMILY_GOAL",
    "FAMILY_LINEAR",
    "FAMILY_MLP",
    "FAMILY_HINT",
    "FAMILIES",
    "PRESETS",
    "DOMAIN_PRIOR_CONFIG",
    "MixtureRewardFunction",
    "RewardPrior",
    "make_reward_prior",
    "resolve_family_weights",
    "list_presets",
    "build_hint_functions",
    "domain_config",
    "resolve_distance_dims",
    "hint_velocity_index",
    "hint_directions",
    "stack_mlp_parameters",
    "evaluate_mlp_parameters",
]


# ---------------------------------------------------------------------------
# family identifiers
# ---------------------------------------------------------------------------
FAMILY_GOAL = "goal"
FAMILY_LINEAR = "linear"
FAMILY_MLP = "mlp"
FAMILY_HINT = "hint"
FAMILIES: Tuple[str, ...] = (FAMILY_GOAL, FAMILY_LINEAR, FAMILY_MLP, FAMILY_HINT)

_FAMILY_INDEX = {name: i for i, name in enumerate(FAMILIES)}


# ---------------------------------------------------------------------------
# presets (addendum: FRE-all / FRE-goals / FRE-lin / FRE-mlp / FRE-lin-mlp /
# FRE-goal-mlp / FRE-goal-lin / FRE-hint)
# ---------------------------------------------------------------------------
_HALF = 0.5
_THIRD = 1.0 / 3.0

PRESETS: Dict[str, Dict[str, float]] = {
    # vanilla prior used in Sections 5.1 and 5.2 ("FRE" == "FRE-all")
    "fre-all": {FAMILY_GOAL: _THIRD, FAMILY_LINEAR: _THIRD, FAMILY_MLP: _THIRD},
    "fre": {FAMILY_GOAL: _THIRD, FAMILY_LINEAR: _THIRD, FAMILY_MLP: _THIRD},
    "all": {FAMILY_GOAL: _THIRD, FAMILY_LINEAR: _THIRD, FAMILY_MLP: _THIRD},
    # single-family ablations
    "fre-goals": {FAMILY_GOAL: 1.0},
    "fre-lin": {FAMILY_LINEAR: 1.0},
    "fre-mlp": {FAMILY_MLP: 1.0},
    # equal splits
    "fre-lin-mlp": {FAMILY_LINEAR: _HALF, FAMILY_MLP: _HALF},
    "fre-goal-mlp": {FAMILY_GOAL: _HALF, FAMILY_MLP: _HALF},
    "fre-goal-lin": {FAMILY_GOAL: _HALF, FAMILY_LINEAR: _HALF},
    # Section 5.4 domain-knowledge study: superset of the evaluation tasks
    # (unit (x, y) directions for ant-directional, specific target velocities for
    # the cheetah/walker velocity tasks).  The remaining two thirds of the
    # mixture keep the vanilla prior so the representation stays general.
    "fre-hint": {FAMILY_GOAL: _THIRD, FAMILY_LINEAR: _THIRD, FAMILY_HINT: _THIRD},
}

_ALIASES = {
    "hint": "fre-hint",
    "goals": "fre-goals",
    "lin": "fre-lin",
    "mlp": "fre-mlp",
    "lin-mlp": "fre-lin-mlp",
    "goal-mlp": "fre-goal-mlp",
    "goal-lin": "fre-goal-lin",
}


def list_presets() -> List[str]:
    """Return the sorted list of known prior presets."""
    return sorted(PRESETS)


def resolve_family_weights(
    preset: Union[str, Mapping[str, float], None] = "fre-all",
) -> Dict[str, float]:
    """Normalise ``preset`` into a family -> weight mapping summing to 1."""
    if preset is None:
        preset = "fre-all"
    if isinstance(preset, Mapping):
        weights = {str(k).lower(): float(v) for k, v in preset.items()}
    else:
        key = str(preset).strip().lower().replace(" ", "").replace("_", "-")
        key = _ALIASES.get(key, key)
        if key not in PRESETS:
            raise KeyError(
                f"Unknown reward-prior preset {preset!r}; known presets: {list_presets()}"
            )
        weights = dict(PRESETS[key])
    weights = {k: v for k, v in weights.items() if v > 0.0}
    for name in weights:
        if name not in FAMILIES:
            raise KeyError(f"Unknown reward family {name!r}; expected one of {FAMILIES}")
    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ValueError("Reward-prior family weights must sum to a positive value")
    return {k: v / total for k, v in weights.items()}


# ---------------------------------------------------------------------------
# per-domain configuration
# ---------------------------------------------------------------------------
# ``distance_dims`` is either None (use every dimension), an explicit sequence /
# slice of indices, or the string "non_augmented" which resolves to the leading
# ``state_dim - augment_dim`` dimensions (ExORL appends physics features that are
# excluded from the goal distance; Appendix C.2).
DOMAIN_PRIOR_CONFIG: Dict[str, Dict[str, Any]] = {
    "antmaze": {
        # Appendix B: "On AntMaze, we remove the XY positions from this
        # generation as the scale of the dimensions led to instability."
        "exclude_dims": (0, 1),
        "distance_dims": None,
        "velocity_indices": (15, 16),
        "hint_domain": "antmaze",
    },
    "antmaze-large-diverse-v2": {
        "exclude_dims": (0, 1),
        "distance_dims": None,
        "velocity_indices": (15, 16),
        "hint_domain": "antmaze",
    },
    "exorl:walker": {
        "exclude_dims": None,
        "distance_dims": "non_augmented",
        "velocity_index": -4,  # horizontal_velocity_x (first appended physics dim)
        "hint_domain": "walker",
    },
    "exorl:cheetah": {
        "exclude_dims": None,
        "distance_dims": "non_augmented",
        "velocity_index": -1,  # physics.speed() is the last appended physics dim
        "hint_domain": "cheetah",
    },
    "walker": {
        "exclude_dims": None,
        "distance_dims": "non_augmented",
        "velocity_index": -4,
        "hint_domain": "walker",
    },
    "cheetah": {
        "exclude_dims": None,
        "distance_dims": "non_augmented",
        "velocity_index": -1,
        "hint_domain": "cheetah",
    },
    "kitchen": {
        "exclude_dims": None,
        "distance_dims": None,
        "hint_domain": None,
    },
}


def domain_config(domain: Optional[str] = None, **overrides: Any) -> Dict[str, Any]:
    """Look up the prior configuration of ``domain`` (returns a copy)."""
    if domain is None:
        cfg: Dict[str, Any] = {}
    else:
        key = str(domain)
        cfg = dict(DOMAIN_PRIOR_CONFIG.get(key, {}))
        if not cfg and ":" in key:
            cfg = dict(DOMAIN_PRIOR_CONFIG.get(key.split(":")[-1], {}))
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def resolve_distance_dims(
    distance_dims: Union[None, str, Sequence[int], slice],
    state_dim: Optional[int],
    augment_dim: int = 0,
) -> Union[None, Tuple[int, ...], slice]:
    """Resolve the ``distance_dims`` entry of a domain configuration."""
    if distance_dims is None:
        return None
    if isinstance(distance_dims, slice):
        return distance_dims
    if isinstance(distance_dims, str):
        if distance_dims in ("non_augmented", "non-augmented", "physics_excluded"):
            if state_dim is None:
                return None
            keep = int(state_dim) - int(augment_dim)
            if keep <= 0:
                return None
            return tuple(range(keep))
        if distance_dims in ("all", "none", ""):
            return None
        raise ValueError(f"Unknown distance_dims specification {distance_dims!r}")
    return tuple(int(d) for d in distance_dims)


# ---------------------------------------------------------------------------
# hint (Section 5.4) helpers
# ---------------------------------------------------------------------------
ANTMAZE_DEFAULT_NUM_DIRECTIONS = 16
WALKER_HINT_VELOCITIES: Tuple[float, ...] = (-8.0, -4.0, -1.0, -0.1, 0.1, 1.0, 4.0, 8.0)
CHEETAH_HINT_VELOCITIES: Tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0)


def hint_directions(num_directions: int = ANTMAZE_DEFAULT_NUM_DIRECTIONS) -> Tuple[Tuple[float, float], ...]:
    """Unit (x, y) directions covering the whole circle (Section 5.4).

    "For ant-directional, the prior rewards are all reward corresponding to
    movement in a unit (x,y) direction."  A dense set of unit directions is a
    superset of the four cardinal directions used as evaluation tasks.
    """
    num = max(int(num_directions), 4)
    angles = np.linspace(0.0, 2.0 * np.pi, num, endpoint=False)
    dirs = [(float(np.cos(a)), float(np.sin(a))) for a in angles]
    # make sure the evaluation cardinal directions are present exactly
    for cardinal in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
        if cardinal not in dirs:
            dirs.append(cardinal)
    return tuple(dirs)


def hint_velocity_index(
    hint_domain: str,
    state_dim: Optional[int],
    augment_dim: int = 0,
) -> int:
    """Index of the forward-velocity dimension used by velocity hint rewards.

    ExORL appends physics features *after* the original state (Appendix C.2):
    ``walker`` appends ``[horizontal_velocity_x, horizontal_velocity_y,
    torso_upright, torso_height]`` so the forward velocity is at ``-4``; while
    ``cheetah`` appends ``physics.speed()`` as the final dimension (``-1``).
    """
    if hint_domain == "walker":
        if state_dim is not None:
            return int(state_dim) - 4
        return -4
    if hint_domain == "cheetah":
        return -1
    # fallback: first appended physics dim
    if state_dim is not None and augment_dim:
        return int(state_dim) - int(augment_dim)
    return -1


def build_hint_functions(
    domain: Optional[str] = None,
    state_dim: Optional[int] = None,
    augment_dim: int = 0,
    num_directions: int = ANTMAZE_DEFAULT_NUM_DIRECTIONS,
    velocities: Optional[Sequence[float]] = None,
    velocity_width: float = 1.0,
    hint_domain: Optional[str] = None,
) -> List[RewardFunction]:
    """Build the per-domain superset of the evaluation tasks (Section 5.4).

    ``antmaze``   -> one reward per unit (x, y) direction (velocity dot product).
    ``walker``    -> one reward per target forward velocity (physics feature).
    ``cheetah``   -> one reward per target speed (physics feature).
    """
    if hint_domain is None:
        cfg = domain_config(domain)
        hint_domain = cfg.get("hint_domain")
    if hint_domain is None:
        return []

    if hint_domain == "antmaze":
        directions = hint_directions(num_directions)
        functions = make_directional_reward_functions(
            directions=directions, state_dim=state_dim
        )
        return list(functions)

    if hint_domain in ("walker", "cheetah"):
        if velocities is None:
            velocities = (
                WALKER_HINT_VELOCITIES if hint_domain == "walker" else CHEETAH_HINT_VELOCITIES
            )
        index = hint_velocity_index(hint_domain, state_dim, augment_dim=augment_dim)
        functions = make_velocity_hint_functions(
            velocities=velocities,
            velocity_index=index,
            state_dim=state_dim,
            width=velocity_width,
        )
        return list(functions)

    return []


# ---------------------------------------------------------------------------
# MLP parameter helpers
# ---------------------------------------------------------------------------
def stack_mlp_parameters(
    parameters: Union[Sequence[Tuple[np.ndarray, ...]], Tuple[np.ndarray, ...], Mapping[str, Any]],
) -> Dict[str, np.ndarray]:
    """Stack per-function ``(w1, b1, w2, b2)`` parameters into batched arrays.

    Accepts the list-of-tuples returned by
    :func:`fre.priors.mlp_functions.sample_mlp_parameters` as well as an already
    stacked mapping with keys ``w1, b1, w2, b2``.
    """
    if isinstance(parameters, Mapping):
        return {k: np.asarray(v, dtype=np.float32) for k, v in parameters.items()}

    if isinstance(parameters, tuple) and len(parameters) == 4 and all(
        hasattr(p, "shape") for p in parameters
    ):
        w1, b1, w2, b2 = parameters
        if np.asarray(w1).ndim >= 2:
            return {
                "w1": np.asarray(w1, dtype=np.float32),
                "b1": np.asarray(b1, dtype=np.float32),
                "w2": np.asarray(w2, dtype=np.float32),
                "b2": np.asarray(b2, dtype=np.float32),
            }

    params: List[Tuple[np.ndarray, ...]] = list(parameters)  # type: ignore[arg-type]
    w1 = np.stack([np.asarray(p[0], dtype=np.float32) for p in params], axis=0)
    b1 = np.stack([np.atleast_1d(np.asarray(p[1], dtype=np.float32)) for p in params], axis=0)
    w2 = np.stack([np.atleast_1d(np.asarray(p[2], dtype=np.float32)) for p in params], axis=0)
    b2 = np.stack([np.atleast_1d(np.asarray(p[3], dtype=np.float32)) for p in params], axis=0)
    if b1.ndim > 2:
        b1 = b1.reshape(b1.shape[0], -1)
    if w2.ndim > 2:
        w2 = w2.reshape(w2.shape[0], -1)
    b2 = b2.reshape(b2.shape[0], -1)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def evaluate_mlp_parameters(
    parameters: Union[Mapping[str, np.ndarray], Sequence[Tuple[np.ndarray, ...]]],
    states: np.ndarray,
    clip_value: float = DEFAULT_CLIP_VALUE,
    hidden_activation: str = "tanh",
) -> np.ndarray:
    """Evaluate a batch of random MLPs on ``states`` with shape ``(B, *, S)``.

    Returns an array with shape ``states.shape[:-1]`` (one reward per state for
    the matching function, i.e. row ``b`` of the batch is evaluated with
    function ``b``).
    """
    params = stack_mlp_parameters(parameters)
    w1, b1, w2, b2 = params["w1"], params["b1"], params["w2"], params["b2"]
    states = np.asarray(states, dtype=np.float32)
    flat = states.reshape(states.shape[0], -1, states.shape[-1])
    hidden = np.einsum("bhd,bkd->bkh", w1, flat) + b1[:, None, :]
    hidden = _apply_activation(np, hidden, hidden_activation)
    out = np.einsum("bh,bkh->bk", w2, hidden) + b2[:, None]
    out = out.reshape(states.shape[:-1])
    if clip_value is not None:
        out = np.clip(out, -float(clip_value), float(clip_value))
    return out.astype(np.float32, copy=False)


def _apply_activation(xp: Any, x: Any, activation: str) -> Any:
    name = (activation or "tanh").lower()
    if name == "tanh":
        return xp.tanh(x)
    if name == "relu":
        return xp.maximum(x, 0.0)
    if name in ("identity", "linear", "none", None):
        return x
    if name == "gelu":
        return 0.5 * x * (1.0 + xp.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
    if name in ("silu", "swish"):
        return x / (1.0 + xp.exp(-x))
    raise ValueError(f"Unknown hidden activation {activation!r}")


# ---------------------------------------------------------------------------
# batched mixture reward function (one entry per sampled function)
# ---------------------------------------------------------------------------
class MixtureRewardFunction(RewardFunction):
    """A list of reward functions evaluated together.

    ``reward(states)`` returns an array of shape ``states.shape[:-1] + (N,)``
    where ``N`` is the number of functions (matching the convention of the
    random linear / MLP families).  Individual functions remain accessible via
    :meth:`at` / ``__getitem__``.
    """

    def __init__(
        self,
        functions: Sequence[RewardFunction],
        families: Optional[Sequence[str]] = None,
        state_dim: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        self.functions: List[RewardFunction] = list(functions)
        self.families: List[str] = (
            list(families) if families is not None else ["unknown"] * len(self.functions)
        )
        if len(self.families) != len(self.functions):
            raise ValueError("families and functions must have the same length")
        self.state_dim = state_dim
        self.name = name

    # -- container protocol -------------------------------------------------
    def __len__(self) -> int:
        return len(self.functions)

    def __getitem__(self, index: int) -> RewardFunction:
        return self.functions[index]

    def __iter__(self):
        return iter(self.functions)

    @property
    def num_functions(self) -> int:
        return len(self.functions)

    @property
    def family(self) -> str:
        return "mixture"

    def at(self, index: int) -> RewardFunction:
        return self.functions[index]

    # -- evaluation ---------------------------------------------------------
    def reward(self, states: Any) -> np.ndarray:
        if not self.functions:
            raise ValueError("Cannot evaluate an empty mixture reward function")
        rewards = [np.asarray(fn.reward(states), dtype=np.float32) for fn in self.functions]
        return np.stack(rewards, axis=-1)

    def __call__(self, states: Any) -> np.ndarray:
        return self.reward(states)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        counts: Dict[str, int] = {}
        for fam in self.families:
            counts[fam] = counts.get(fam, 0) + 1
        return f"families={counts}"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"MixtureRewardFunction(n={len(self.functions)}, {self.extra_repr()})"


# ---------------------------------------------------------------------------
# the prior itself
# ---------------------------------------------------------------------------
class RewardPrior:
    """Sampler for the FRE prior reward distribution (Section 4.2).

    Parameters
    ----------
    preset:
        One of :data:`PRESETS` (``"fre-all"``, ``"fre-goals"``, ``"fre-lin"``,
        ``"fre-mlp"``, ``"fre-lin-mlp"``, ``"fre-goal-mlp"``, ``"fre-goal-lin"``,
        ``"fre-hint"``) or an explicit ``{family: weight}`` mapping.
    domain:
        Domain key (``"antmaze"``, ``"exorl:walker"``, ``"exorl:cheetah"``,
        ``"kitchen"``, ...) used to look up :data:`DOMAIN_PRIOR_CONFIG`.
    states:
        Optional ``(N, state_dim)`` array of states from the offline dataset
        ``D``.  If not provided, ``RewardPrior`` lazily accepts states through
        ``set_states`` / the ``dataset`` argument (an ``OfflineDataset``).
    """

    def __init__(
        self,
        preset: Union[str, Mapping[str, float]] = "fre-all",
        domain: Optional[str] = None,
        states: Optional[np.ndarray] = None,
        dataset: Optional[Any] = None,
        state_dim: Optional[int] = None,
        family_weights: Optional[Mapping[str, float]] = None,
        exclude_dims: Optional[Sequence[int]] = None,
        distance_dims: Union[None, str, Sequence[int], slice] = None,
        state_std: Optional[np.ndarray] = None,
        goal_threshold: float = DEFAULT_GOAL_THRESHOLD,
        p_current: float = HER_P_CURRENT,
        p_future: float = HER_P_FUTURE,
        p_random: float = HER_P_RANDOM,
        min_offset: int = 0,
        mask_prob: float = DEFAULT_MASK_PROB,
        weight_range: Tuple[float, float] = (-1.0, 1.0),
        mlp_hidden_dim: int = DEFAULT_HIDDEN_DIM,
        mlp_scaling: str = "avg_dim_sqrt",
        mlp_clip_value: float = DEFAULT_CLIP_VALUE,
        mlp_hidden_activation: str = "tanh",
        augment_dim: int = 0,
        num_directions: int = ANTMAZE_DEFAULT_NUM_DIRECTIONS,
        hint_velocities: Optional[Sequence[float]] = None,
        hint_velocity_width: float = 1.0,
        hint_functions: Optional[Sequence[RewardFunction]] = None,
        encoder_goal_slot: int = 0,
        seed: int = 0,
        device: Union[str, Any, None] = None,
        dtype: Any = np.float32,
        normalize_goal_rewards: bool = False,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if family_weights is None:
            family_weights = preset
        self.preset = preset if isinstance(preset, str) else "custom"
        self.domain = domain
        self.name = name or f"reward-prior[{self.preset}" + (
            f",{domain}]" if domain else "]"
        )
        self.dtype = dtype

        cfg = domain_config(domain)
        self.exclude_dims = (
            tuple(int(d) for d in exclude_dims)
            if exclude_dims is not None
            else (
                tuple(int(d) for d in cfg["exclude_dims"])
                if cfg.get("exclude_dims") is not None
                else None
            )
        )
        if distance_dims is None:
            distance_dims = cfg.get("distance_dims")
        self.augment_dim = int(augment_dim or 0)
        self.state_std = None if state_std is None else np.asarray(state_std, dtype=np.float32)

        # -- datasets -------------------------------------------------------
        self.dataset = dataset
        self._states: Optional[np.ndarray] = None
        self.traj_starts: Optional[np.ndarray] = None
        self.traj_ends: Optional[np.ndarray] = None
        if states is not None:
            self.set_states(states)
        elif dataset is not None:
            self.set_dataset(dataset)
        self._state_dim = int(state_dim) if state_dim is not None else None
        self.distance_dims = resolve_distance_dims(
            distance_dims, self.state_dim, self.augment_dim
        )

        # -- mixture --------------------------------------------------------
        self.family_weights = resolve_family_weights(family_weights)
        self._family_names = list(self.family_weights.keys())
        self._family_probs = np.array(
            [self.family_weights[k] for k in self._family_names], dtype=np.float64
        )

        # -- goal family ----------------------------------------------------
        self.goal_threshold = float(goal_threshold)
        self.p_current = float(p_current)
        self.p_future = float(p_future)
        self.p_random = float(p_random)
        self.min_offset = int(min_offset)
        self.encoder_goal_slot = int(encoder_goal_slot)
        self.normalize_goal_rewards = bool(normalize_goal_rewards)

        # -- linear / MLP families -----------------------------------------
        self.mask_prob = float(mask_prob)
        self.weight_range = (float(weight_range[0]), float(weight_range[1]))
        self.mlp_hidden_dim = int(mlp_hidden_dim)
        self.mlp_scaling = mlp_scaling
        self.mlp_clip_value = float(mlp_clip_value)
        self.mlp_hidden_activation = mlp_hidden_activation

        # -- hint family ----------------------------------------------------
        self.num_directions = int(num_directions)
        self.hint_velocities = hint_velocities
        self.hint_velocity_width = float(hint_velocity_width)
        self._hint_functions: Optional[List[RewardFunction]] = hint_functions
        if self._hint_functions is not None:
            self._hint_functions = list(self._hint_functions)

        # -- rng / device ---------------------------------------------------
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.device = device

    # ------------------------------------------------------------------ state
    @property
    def states(self) -> Optional[np.ndarray]:
        return self._states

    @property
    def state_dim(self) -> Optional[int]:
        if self._state_dim is not None:
            return self._state_dim
        if self._states is not None:
            return int(self._states.shape[-1])
        return None

    @property
    def num_states(self) -> int:
        return 0 if self._states is None else int(self._states.shape[0])

    def set_states(self, states: np.ndarray) -> "RewardPrior":
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2:
            states = states.reshape(-1, states.shape[-1])
        self._states = states
        self._state_dim = int(states.shape[-1])
        self.distance_dims = resolve_distance_dims(
            self.distance_dims, self.state_dim, self.augment_dim
        )
        return self

    def set_dataset(self, dataset: Any) -> "RewardPrior":
        """Attach an ``OfflineDataset`` (or any object exposing observations)."""
        self.dataset = dataset
        states = None
        for attr in ("observations", "states"):
            if hasattr(dataset, attr):
                candidate = getattr(dataset, attr)
                if candidate is not None:
                    states = np.asarray(candidate, dtype=np.float32)
                    break
        if states is not None:
            self.set_states(states)
        try:
            starts, ends = extract_trajectory_bounds(dataset)
            self.traj_starts, self.traj_ends = starts, ends
        except Exception:  # pragma: no cover - defensive
            self.traj_starts, self.traj_ends = None, None
        if self.traj_starts is None and hasattr(dataset, "traj_starts"):
            self.traj_starts = getattr(dataset, "traj_starts")
        if self.traj_ends is None and hasattr(dataset, "traj_ends"):
            self.traj_ends = getattr(dataset, "traj_ends")
        # Prefer the dataset's own HER helper when available
        return self

    def seed_rng(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(self.seed if seed is None else int(seed))

    # -------------------------------------------------------------- families
    @property
    def families(self) -> List[str]:
        return list(self._family_names)

    def family_probabilities(self) -> Dict[str, float]:
        return dict(self.family_weights)

    # ------------------------------------------------------------------ hint
    @property
    def hint_functions(self) -> List[RewardFunction]:
        if self._hint_functions is None:
            self._hint_functions = build_hint_functions(
                domain=self.domain,
                state_dim=self.state_dim,
                augment_dim=self.augment_dim,
                num_directions=self.num_directions,
                velocities=self.hint_velocities,
                velocity_width=self.hint_velocity_width,
            )
        return list(self._hint_functions)

    # ----------------------------------------------------------- goal helpers
    def _her_goal_states(
        self,
        num_goals: int,
        source_indices: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Sample goals with the HER distribution of Appendix B."""
        rng = self.rng if rng is None else rng
        states = self._states
        if states is None:
            raise ValueError(
                "RewardPrior needs dataset states before sampling goal rewards; "
                "pass `states=`/`dataset=` or call `set_states`."
            )
        num_states = states.shape[0]
        if source_indices is None:
            source_indices = rng.integers(0, num_states, size=num_goals)
        source_indices = np.asarray(source_indices, dtype=np.int64).reshape(num_goals)

        try:
            goals = sample_her_goals(
                states,
                source_indices,
                rng=rng,
                traj_starts=self.traj_starts,
                traj_ends=self.traj_ends,
                p_current=self.p_current,
                p_future=self.p_future,
                p_random=self.p_random,
                min_offset=self.min_offset,
            )
            return np.asarray(goals, dtype=np.float32)
        except Exception as exc:  # pragma: no cover - defensive fallback
            warnings.warn(
                f"Falling back to uniform-random goal sampling ({exc})", RuntimeWarning
            )
            idx = rng.integers(0, num_states, size=num_goals)
            return states[idx].astype(np.float32, copy=False)

    def _goal_rewards(
        self, states: np.ndarray, goals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Goal-reaching reward (-1 unachieved / 0 achieved) and done mask."""
        states = np.asarray(states, dtype=np.float32)
        goals = np.asarray(goals, dtype=np.float32)
        if goals.ndim == states.ndim - 1:
            goals = goals[:, None, :]
        distances = np.asarray(
            goal_distances(
                states,
                goals,
                distance_dims=self.distance_dims,
                state_std=self.state_std,
            ),
            dtype=np.float32,
        )
        reached = distances < self.goal_threshold
        rewards = np.where(reached, 0.0, -1.0).astype(np.float32)
        return rewards, reached

    # --------------------------------------------------------- reward helpers
    def _sample_linear_weights(self, num_functions: int, rng: np.random.Generator) -> np.ndarray:
        return np.asarray(
            sample_linear_weights(
                num_functions,
                self.state_dim,
                rng=rng,
                mask_prob=self.mask_prob,
                weight_range=self.weight_range,
                exclude_dims=self.exclude_dims,
            ),
            dtype=np.float32,
        )

    def _sample_mlp_parameters(self, num_functions: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        params = sample_mlp_parameters(
            num_functions,
            self.state_dim,
            rng=rng,
            hidden_dim=self.mlp_hidden_dim,
            scaling=self.mlp_scaling,
        )
        return stack_mlp_parameters(params)

    def _sample_family_labels(
        self, num_functions: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Draw a family label (index into ``FAMILIES``) per function."""
        if len(self._family_names) == 1:
            idx = np.full(num_functions, _FAMILY_INDEX[self._family_names[0]], dtype=np.int64)
            return idx
        draws = rng.choice(len(self._family_names), size=num_functions, p=self._family_probs)
        return np.asarray([_FAMILY_INDEX[self._family_names[i]] for i in draws], dtype=np.int64)

    # ------------------------------------------------------------ evaluation
    def _evaluate(
        self,
        family_labels: np.ndarray,
        states: np.ndarray,
        goals: np.ndarray,
        linear_weights: np.ndarray,
        mlp_parameters: Mapping[str, np.ndarray],
        hint_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate every row with its own sampled function.

        ``states`` has shape ``(B, *, S)``.  Returns ``(rewards, dones)`` both of
        shape ``(B, *)``.
        """
        states = np.asarray(states, dtype=np.float32)
        num = states.shape[0]
        rewards = np.zeros((num,) + tuple(states.shape[1:-1]), dtype=np.float32)
        dones = np.zeros_like(rewards, dtype=bool)

        mask_goal = family_labels == _FAMILY_INDEX[FAMILY_GOAL]
        mask_linear = family_labels == _FAMILY_INDEX[FAMILY_LINEAR]
        mask_mlp = family_labels == _FAMILY_INDEX[FAMILY_MLP]
        mask_hint = family_labels == _FAMILY_INDEX[FAMILY_HINT]

        if mask_goal.any():
            r, d = self._goal_rewards(states[mask_goal], goals[mask_goal])
            rewards[mask_goal] = r
            dones[mask_goal] = d

        if mask_linear.any():
            weights = np.asarray(linear_weights, dtype=np.float32)[mask_linear]
            rew = np.einsum("bd,bkd->bk", weights, states[mask_linear])
            rewards[mask_linear] = rew.reshape(states[mask_linear].shape[:-1])

        if mask_mlp.any():
            sub_params = {
                key: np.asarray(value, dtype=np.float32)[mask_linear * False + mask_mlp]
                for key, value in mlp_parameters.items()
            }
            rew = evaluate_mlp_parameters(
                sub_params,
                states[mask_mlp],
                clip_value=self.mlp_clip_value,
                hidden_activation=self.mlp_hidden_activation,
            )
            rewards[mask_mlp] = rew

        if mask_hint.any():
            functions = self.hint_functions
            if not functions:
                raise ValueError(
                    "The 'hint' family was requested but no hint functions could be "
                    "built; provide a domain with a hint specification or pass "
                    "`hint_functions=`."
                )
            if hint_ids is None:
                hint_ids = np.zeros(num, dtype=np.int64)
            hint_ids = np.asarray(hint_ids, dtype=np.int64)
            rows = np.nonzero(mask_hint)[0]
            for hint_id in np.unique(hint_ids[rows]):
                fn = functions[int(hint_id) % len(functions)]
                sub = states[rows[hint_ids[rows] == hint_id]]
                rewards[rows[hint_ids[rows] == hint_id]] = np.asarray(
                    fn.reward(sub), dtype=np.float32
                ).reshape(sub.shape[:-1])

        return rewards, dones

    # ------------------------------------------------------------- sampling
    def sample(
        self,
        num_functions: int = 1,
        rng: Optional[np.random.Generator] = None,
        as_list: bool = False,
        source_indices: Optional[np.ndarray] = None,
    ) -> Union[RewardFunction, List[RewardFunction]]:
        """Sample ``num_functions`` reward functions.

        With ``as_list=False`` (default) a :class:`MixtureRewardFunction` is
        returned; ``as_list=True`` returns the individual functions (the form the
        mixture dispatcher consumes).  Family members that need a concrete goal
        (goal family) are materialised immediately.
        """
        rng = self.rng if rng is None else rng
        num_functions = int(num_functions)
        labels = self._sample_family_labels(num_functions, rng)
        functions: List[RewardFunction] = []
        families: List[str] = []

        goals = None
        if _FAMILY_INDEX[FAMILY_GOAL] in labels:
            goals = self._her_goal_states(num_functions, source_indices, rng)
        linear_weights = None
        if _FAMILY_INDEX[FAMILY_LINEAR] in labels:
            linear_weights = self._sample_linear_weights(num_functions, rng)
        mlp_parameters = None
        if _FAMILY_INDEX[FAMILY_MLP] in labels:
            mlp_parameters = self._sample_mlp_parameters(num_functions, rng)
        hint_ids = None
        if _FAMILY_INDEX[FAMILY_HINT] in labels:
            hint_ids = rng.integers(0, max(len(self.hint_functions), 1), size=num_functions)

        for i in range(num_functions):
            label = int(labels[i])
            if label == _FAMILY_INDEX[FAMILY_GOAL]:
                functions.append(
                    GoalRewardFunction(
                        goal=goals[i],
                        threshold=self.goal_threshold,
                        state_dim=self.state_dim,
                        distance_dims=self.distance_dims,
                        state_std=self.state_std,
                    )
                )
                families.append(FAMILY_GOAL)
            elif label == _FAMILY_INDEX[FAMILY_LINEAR]:
                functions.append(
                    LinearRewardFunction(
                        weights=linear_weights[i],
                        state_dim=self.state_dim,
                        exclude_dims=self.exclude_dims,
                        mask_prob=self.mask_prob,
                        weight_range=self.weight_range,
                    )
                )
                families.append(FAMILY_LINEAR)
            elif label == _FAMILY_INDEX[FAMILY_MLP]:
                params = {
                    "w1": mlp_parameters["w1"][i],
                    "b1": mlp_parameters["b1"][i],
                    "w2": mlp_parameters["w2"][i],
                    "b2": mlp_parameters["b2"][i],
                }
                functions.append(
                    MLPRewardFunction(
                        weights=(
                            params["w1"],
                            params["b1"],
                            params["w2"],
                            params["b2"],
                        ),
                        state_dim=self.state_dim,
                        clip_value=self.mlp_clip_value,
                        hidden_activation=self.mlp_hidden_activation,
                    )
                )
                families.append(FAMILY_MLP)
            else:
                hint_list = self.hint_functions
                if not hint_list:
                    raise ValueError(
                        "The 'hint' family was requested but no hint functions are available"
                    )
                functions.append(hint_list[int(hint_ids[i]) % len(hint_list)])
                families.append(FAMILY_HINT)

        if as_list:
            return functions
        if num_functions == 1:
            return functions[0]
        return MixtureRewardFunction(functions, families=families, state_dim=self.state_dim)

    def sample_function(
        self, rng: Optional[np.random.Generator] = None
    ) -> RewardFunction:
        """Sample a single reward function (used once per RL training step)."""
        fn = self.sample(1, rng=rng, as_list=False)
        if isinstance(fn, list):  # pragma: no cover - defensive
            fn = fn[0]
        return fn

    def sample_functions(
        self,
        num_functions: int,
        rng: Optional[np.random.Generator] = None,
        as_list: bool = False,
    ) -> Union[RewardFunction, List[RewardFunction]]:
        return self.sample(num_functions, rng=rng, as_list=as_list)

    # ------------------------------------------------------------ batch mode
    def sample_batch(
        self,
        dataset_states: Optional[Any] = None,
        batch_size: int = 512,
        num_encoder_pairs: int = 32,
        num_decoder_pairs: int = 8,
        device: Union[str, Any, None] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """Sample a training batch of reward-annotated states for FRE.

        For every of the ``batch_size`` reward functions:

        * ``num_encoder_pairs`` (K = 32) states are sampled uniformly from the
          offline dataset and labelled by the function (Section 4.1);
        * for the goal-reaching family the goals follow the HER distribution of
          Appendix B and one encoder slot is set to the goal state itself, so
          that "at least one of the samples contains the goal state";
        * ``num_decoder_pairs`` (K' = 8) *disjoint* states are used for the
          decoder reconstruction term of Equation 6.

        Returns a dictionary with the tensors expected by
        :meth:`fre.models.fre_model.FREModel.training_batch` plus the raw prior
        parameters (``prior_state``) so that the same sampled functions can be
        re-evaluated on dataset transitions during the RL phase (§4.3).
        """
        rng = self.rng if rng is None else rng
        states = self._resolve_states(dataset_states)
        if states is None:
            raise ValueError(
                "RewardPrior.sample_batch requires dataset states; pass them as the "
                "first argument or construct the prior with `states=`/`dataset=`."
            )
        if self._states is None or self._states.shape[1] != states.shape[1]:
            self.set_states(states)

        batch_size = int(batch_size)
        K = int(num_encoder_pairs)
        Kd = int(num_decoder_pairs)
        num_states = states.shape[0]

        family_labels = self._sample_family_labels(batch_size, rng)
        enc_idx = rng.integers(0, num_states, size=(batch_size, K))
        dec_idx = rng.integers(0, num_states, size=(batch_size, Kd))
        dec_idx = self._disjoint_indices(dec_idx, enc_idx, num_states)

        enc_states = states[enc_idx].astype(np.float32, copy=False)
        dec_states = states[dec_idx].astype(np.float32, copy=False)

        # goals: HER-sampled for the goal family, uniform dataset states otherwise
        goals = self._her_goal_states(batch_size, enc_idx[:, 0], rng)
        # guarantee the goal appears in the encoding set
        if K > 0:
            enc_states = enc_states.copy()
            enc_states[:, self.encoder_goal_slot % K] = goals

        linear_weights = self._sample_linear_weights(batch_size, rng)
        mlp_parameters = self._sample_mlp_parameters(batch_size, rng)
        hint_ids = (
            rng.integers(0, max(len(self.hint_functions), 1), size=batch_size)
            if _FAMILY_INDEX[FAMILY_HINT] in family_labels
            else np.zeros(batch_size, dtype=np.int64)
        )

        enc_rewards, enc_dones = self._evaluate(
            family_labels, enc_states, goals, linear_weights, mlp_parameters, hint_ids
        )
        dec_rewards, dec_dones = self._evaluate(
            family_labels, dec_states, goals, linear_weights, mlp_parameters, hint_ids
        )

        family_names = np.array([FAMILIES[int(i)] for i in family_labels])
        batch: Dict[str, Any] = {
            "encoder_states": enc_states,
            "encoder_rewards": enc_rewards,
            "decoder_states": dec_states,
            "decoder_rewards": dec_rewards,
            # validity masks ("True == valid"), used by the encoder/decoder
            "encoder_mask": np.ones_like(enc_rewards, dtype=bool),
            "decoder_mask": np.ones_like(dec_rewards, dtype=bool),
            # termination masks (True when the goal has been achieved)
            "encoder_dones": enc_dones,
            "decoder_dones": dec_dones,
            "families": family_names,
            "family_labels": family_labels,
            "num_encoder_pairs": K,
            "num_decoder_pairs": Kd,
            "batch_size": batch_size,
            "prior_state": {
                "family_labels": family_labels,
                "goals": goals,
                "linear_weights": linear_weights,
                "mlp_parameters": mlp_parameters,
                "hint_ids": hint_ids,
            },
        }
        return self._to_device(batch, device)

    def evaluate_sampled(
        self,
        batch_or_state: Mapping[str, Any],
        states: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Re-evaluate the functions sampled in a batch on new states.

        Used during Phase 2 of the strided schedule, where the reward of a
        transition is ``r = eta(s)`` for the function sampled in that step.
        ``states`` must have shape ``(B, *, S)`` with the same leading dimension
        as the sampled batch.
        """
        state = batch_or_state.get("prior_state", batch_or_state)
        labels = np.asarray(state["family_labels"])
        rewards, dones = self._evaluate(
            labels,
            np.asarray(states, dtype=np.float32),
            np.asarray(state["goals"], dtype=np.float32),
            np.asarray(state["linear_weights"], dtype=np.float32),
            state["mlp_parameters"],
            np.asarray(state.get("hint_ids", np.zeros_like(labels)), dtype=np.int64),
        )
        return rewards, dones

    # --------------------------------------------------------------- helpers
    def _resolve_states(self, dataset_states: Optional[Any]) -> Optional[np.ndarray]:
        if dataset_states is None:
            return self._states
        if _HAS_TORCH and isinstance(dataset_states, torch.Tensor):  # pragma: no cover
            dataset_states = dataset_states.detach().cpu().numpy()
        if isinstance(dataset_states, np.ndarray):
            arr = dataset_states
        elif hasattr(dataset_states, "observations"):
            arr = np.asarray(dataset_states.observations)
        elif hasattr(dataset_states, "states"):
            arr = np.asarray(dataset_states.states)
        else:
            arr = np.asarray(dataset_states)
        if arr.dtype == object:  # pragma: no cover - defensive
            arr = np.asarray(list(arr))
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _disjoint_indices(
        decoder_index: np.ndarray, encoder_index: np.ndarray, num_states: int
    ) -> np.ndarray:
        """Shift decoder indices that collide with the encoder indices of the row.

        Writes must be disjoint from the encoder set (Section 4.1 uses separate
        decoder states ``s^d``).
        """
        decoder = np.array(decoder_index, dtype=np.int64, copy=True)
        encoder_sets = np.sort(np.array(encoder_index, dtype=np.int64, copy=True), axis=1)
        for row in range(decoder.shape[0]):
            enc = encoder_sets[row]
            row_idx = decoder[row]
            # vectorised membership test against the (small) encoder row
            inside = np.isin(row_idx, enc)
            attempts = 0
            while inside.any() and attempts < 10:
                row_idx = np.where(inside, (row_idx + 1) % num_states, row_idx)
                inside = np.isin(row_idx, enc)
                attempts += 1
            decoder[row] = row_idx
        return decoder

    def _to_device(self, batch: Dict[str, Any], device: Union[str, Any, None]) -> Dict[str, Any]:
        device = self.device if device is None else device
        if not _HAS_TORCH:  # pragma: no cover - numpy fallback
            return batch
        out: Dict[str, Any] = {}
        for key, value in batch.items():
            if key in ("families", "prior_state", "num_encoder_pairs",
                       "num_decoder_pairs", "batch_size", "family_labels"):
                out[key] = value
            elif isinstance(value, np.ndarray):
                if value.dtype == bool:
                    out[key] = torch.as_tensor(value, dtype=torch.bool, device=device)
                elif np.issubdtype(value.dtype, np.floating):
                    out[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
                elif np.issubdtype(value.dtype, np.integer):
                    out[key] = torch.as_tensor(value, dtype=torch.long, device=device)
                else:  # pragma: no cover - defensive
                    out[key] = value
            else:
                out[key] = value
        return out

    # ------------------------------------------------------------ dunder/api
    def __len__(self) -> int:
        return self.num_states

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"preset={self.preset!r}, domain={self.domain!r}, "
            f"families={self.family_weights}, states={self.num_states}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"RewardPrior({self.extra_repr()})"


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def make_reward_prior(
    domain: Optional[str] = None,
    preset: Union[str, Mapping[str, float]] = "fre-all",
    *,
    dataset: Optional[Any] = None,
    states: Optional[np.ndarray] = None,
    state_dim: Optional[int] = None,
    augment_dim: int = 0,
    family_weights: Optional[Mapping[str, float]] = None,
    **kwargs: Any,
) -> RewardPrior:
    """Convenience constructor mirroring the configs in ``fre/configs/*.yaml``.

    ``domain`` is used to pick the per-domain options of
    :data:`DOMAIN_PRIOR_CONFIG` (AntMaze XY exclusion, ExORL non-augmented goal
    distance, hint task supersets) unless they are overridden explicitly.
    """
    cfg = domain_config(domain)
    cfg.pop("hint_domain", None)
    cfg.update(kwargs)
    if isinstance(preset, str) and preset.strip().lower() in ("hint", "fre-hint"):
        # hint domain is derived from the domain key when not given explicitly
        cfg.setdefault("domain", domain)
    return RewardPrior(
        preset=preset,
        domain=domain,
        states=states,
        dataset=dataset,
        state_dim=state_dim,
        augment_dim=augment_dim,
        family_weights=family_weights,
        **cfg,
    )

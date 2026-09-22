"""Per-domain evaluation configurations for the FRE reproduction.

This module is the single place where the *environment side* of the paper is
described with the exact numbers reported in the paper and its addendum.
Everything here is pure python (no torch / no gym) so that it can be imported
by configuration code, environment wrappers, scripts and tests alike.

Paper sources
-------------
* §5 + Appendix C.1  -> AntMaze  (``antmaze-large-diverse-v2``)
* §5 + Appendix C.2  -> ExORL    (RND datasets for walker / cheetah)
* §5 + Appendix C.3  -> Kitchen  (7 standard D4RL Kitchen subtasks)
* Addendum, "Some notes on the evaluation environments"
* Addendum, "Details on the evaluation tasks" -> "Ant Maze evaluation tasks"
* Addendum, "Details on the evaluation tasks" -> "ExORL evaluation tasks"

Notes on the reward conventions used by the evaluation tasks (these are the
*true* reward functions ``eta(s)`` -- pure functions of the environment state --
that the zero-shot agent is asked to maximize):

AntMaze
    * goal reaching: "reward = -1 for every timestep where the goal has not been
      reached, and 0 otherwise" (``§4.2``), where a goal counts as reached if the
      agent is within a distance of 2 of the target position (addendum).  X and Y
      are discretized into 32 bins (addendum, also used by GC-IQL / GC-BC / OPAL).
    * directional: dot product between the agent's (X,Y) velocity and a target
      velocity; the four fixed targets are (+-1, 0) and (0, +-1).
    * random-simplex: opensimplex 2D noise "height map" + preferred velocity
      bonus, 5 fixed seeds (1..5), baseline reward -1 per step.
    * path-{loop,edges,center}: hand crafted corridor reward functions.

ExORL
    * velocity tasks: reward 1 when the (signed) horizontal velocity reaches the
      threshold, decaying linearly to 0 below it; 0 if the velocity points in the
      opposite direction.  Walker thresholds: 0.1, 1, 4, 8.  Cheetah: run=10,
      walk=1, plus the two backwards variants.
    * goal reaching: 5 fixed goal states taken from the offline dataset; reward
      -1 unless the (dataset-normalized) euclidean distance to the goal is < 0.1,
      in which case the reward is 0.

Kitchen
    * the seven standard D4RL Kitchen subtasks; their sparse rewards are used
      directly as the evaluation reward functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "DomainConfig",
    "EvalTask",
    "ANTMAZE",
    "EXORL_WALKER",
    "EXORL_CHEETAH",
    "KITCHEN",
    "DOMAINS",
    "ANTMAZE_GOALS",
    "ANTMAZE_DIRECTIONS",
    "ANTMAZE_SIMPLEX_SEEDS",
    "EXORL_WALKER_VELOCITY_THRESHOLDS",
    "EXORL_CHEETAH_RUN_THRESHOLD",
    "EXORL_CHEETAH_WALK_THRESHOLD",
    "KITCHEN_SUBTASKS",
    "get_domain_config",
    "get_env_config",
    "get_eval_tasks",
    "get_task",
    "domain_names",
    "default_domain_overrides",
]

# The version of the configuration object is intentionally kept loose typed so
# that this module does not need to import ``fre.config.default`` (avoids a
# circular import: ``fre.config.__init__`` imports both modules).
ConfigLike = "Config"


# --------------------------------------------------------------------------- #
#  Task specifications
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvalTask:
    """A single zero-shot evaluation task.

    Attributes
    ----------
    name:
        Name of the task as it appears in the paper tables, e.g.
        ``"ant-goal-reaching"`` or ``"exorl-walker-velocity"``.
    kind:
        One of ``"goal"``, ``"velocity"``, ``"simplex"``, ``"path"``,
        ``"kitchen"`` -- identifies which reward generator to use.
    params:
        Free-form parameters consumed by the environment wrapper
        (``fre/envs/*_eval.py``).  Values are the exact ones given by the paper.
    max_episode_steps:
        Maximum length of a single evaluation trajectory.
    """

    name: str
    kind: str
    params: Dict[str, object] = field(default_factory=dict)
    max_episode_steps: int = 1000

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "params": dict(self.params),
            "max_episode_steps": self.max_episode_steps,
        }


@dataclass(frozen=True)
class DomainConfig:
    """Static description of one evaluation domain."""

    name: str
    env_id: str
    #: names of the sub-configs of ``fre/config/envs.py`` that share the domain
    variants: Tuple[str, ...]
    #: maximum length of an evaluation trajectory (Appendix C)
    max_episode_steps: int
    #: offline dataset identifier (None for ExORL, which is loaded per-env)
    dataset_id: Optional[str] = None
    #: reward functions are pure functions of the state -> we append these
    #: physics fields to the offline dataset for *encoder* training only
    #: (Appendix C.2).  Env-name -> tuple of field getters.
    physics_fields: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    #: dims (indices into the state) that are excluded from random linear
    #: reward functions because of scale instability (Appendix B)
    linear_exclude_dims: Tuple[int, ...] = ()
    #: extra task suites
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "env_id": self.env_id,
            "variants": list(self.variants),
            "max_episode_steps": self.max_episode_steps,
            "dataset_id": self.dataset_id,
            "physics_fields": {k: list(v) for k, v in self.physics_fields.items()},
            "linear_exclude_dims": list(self.linear_exclude_dims),
            "extra": dict(self.extra),
        }


# --------------------------------------------------------------------------- #
#  AntMaze  (Appendix C.1 + addendum "Ant Maze evaluation tasks")
# --------------------------------------------------------------------------- #
#: 5 hand-crafted goal locations on an (X,Y) grid whose origin is bottom-left.
#: NB: the maze is 8x8 units and the paper's grid is expressed in units where
#: (x, y) are the raw AntMaze XY coordinates as used by the evaluation code.
ANTMAZE_GOALS: Dict[str, Tuple[float, float]] = {
    "bottom": (28.0, 0.0),
    "left": (0.0, 15.0),
    "top": (35.0, 24.0),
    "center": (12.0, 24.0),
    "right": (33.0, 16.0),
}

#: target (X,Y) velocities for the directional tasks (addendum)
ANTMAZE_DIRECTIONS: Dict[str, Tuple[float, float]] = {
    "vel_left": (-1.0, 0.0),
    "vel_up": (0.0, 1.0),
    "vel_down": (0.0, -1.0),
    "vel_right": (1.0, 0.0),
}

#: 5 fixed seeds for the opensimplex "random-simplex" tasks (addendum: seeds 1-5)
ANTMAZE_SIMPLEX_SEEDS: Tuple[int, ...] = (1, 2, 3, 4, 5)

#: paths are hand crafted corridors (addendum); reward is defined in the env file
ANTMAZE_PATHS: Tuple[str, ...] = ("loop", "edges", "center")

ANTMAZE = DomainConfig(
    name="antmaze",
    env_id="antmaze-large-diverse-v2",
    dataset_id="antmaze-large-diverse-v2",
    variants=("antmaze",),
    # the ant is placed in the *center* of the maze (Appendix C.1 / addendum);
    # this is the agent-start override used by ``fre/envs/antmaze_eval.py``.
    max_episode_steps=2000,
    # Appendix C.1: no physics augmentation for AntMaze.
    physics_fields={},
    # Appendix B: on AntMaze the XY positions destabilize random linear rewards.
    linear_exclude_dims=(0, 1),
    extra={
        "start_position": "center",
        "goal_threshold": 2.0,
        "xy_bins": 32,
    },
)


# --------------------------------------------------------------------------- #
#  ExORL  (Appendix C.2 + addendum "ExORL evaluation tasks")
# --------------------------------------------------------------------------- #
#: walker velocity tasks use thresholds 0.1, 1, 4 and 8 (addendum)
EXORL_WALKER_VELOCITY_THRESHOLDS: Tuple[float, ...] = (0.1, 1.0, 4.0, 8.0)
#: cheetah "run" velocity target
EXORL_CHEETAH_RUN_THRESHOLD: float = 10.0
#: cheetah "walk" velocity target
EXORL_CHEETAH_WALK_THRESHOLD: float = 1.0
#: goal reaching threshold on the dataset-normalized euclidean distance
EXORL_GOAL_THRESHOLD: float = 0.1
#: number of goal states taken from the offline dataset
EXORL_NUM_GOALS: int = 5

#: physics fields appended to the dataset for the encoder only (Appendix C.2)
EXORL_WALKER_PHYSICS: Tuple[str, ...] = (
    "horizontal_velocity",
    "torso_upright",
    "torso_height",
)
EXORL_CHEETAH_PHYSICS: Tuple[str, ...] = ("speed",)

EXORL_WALKER = DomainConfig(
    name="exorl_walker",
    env_id="walker",
    dataset_id=None,  # ExORL RND dataset (see fre/envs/d4rl_loader.py)
    variants=("exorl-walker-goals", "exorl-walker-velocity"),
    max_episode_steps=1000,
    physics_fields={"walker": EXORL_WALKER_PHYSICS},
    extra={
        "dataset_dir": "exorl/walker",
        "dataset_kind": "rnd",
        "goal_threshold": EXORL_GOAL_THRESHOLD,
        "num_goals": EXORL_NUM_GOALS,
        "velocity_thresholds": EXORL_WALKER_VELOCITY_THRESHOLDS,
    },
)

EXORL_CHEETAH = DomainConfig(
    name="exorl_cheetah",
    env_id="cheetah",
    dataset_id=None,
    variants=("exorl-cheetah-goals", "exorl-cheetah-velocity"),
    max_episode_steps=1000,
    physics_fields={"cheetah": EXORL_CHEETAH_PHYSICS},
    extra={
        "dataset_dir": "exorl/cheetah",
        "dataset_kind": "rnd",
        "goal_threshold": EXORL_GOAL_THRESHOLD,
        "num_goals": EXORL_NUM_GOALS,
        "run_threshold": EXORL_CHEETAH_RUN_THRESHOLD,
        "walk_threshold": EXORL_CHEETAH_WALK_THRESHOLD,
    },
)


# --------------------------------------------------------------------------- #
#  Kitchen  (Appendix C.3)
# --------------------------------------------------------------------------- #
#: the seven standard D4RL Kitchen subtasks; their sparse rewards are used as-is
KITCHEN_SUBTASKS: Tuple[str, ...] = (
    "microwave",
    "kettle",
    "slide cabinet",
    "hinge cabinet",
    "light switch",
    "bottom burner",
    "top burner",
)

KITCHEN = DomainConfig(
    name="kitchen",
    env_id="kitchen-complete-v0",
    dataset_id="kitchen-complete-v0",
    variants=("kitchen",),
    max_episode_steps=1000,
    extra={"subtasks": KITCHEN_SUBTASKS},
)


DOMAINS: Dict[str, DomainConfig] = {
    "antmaze": ANTMAZE,
    "exorl_walker": EXORL_WALKER,
    "exorl_cheetah": EXORL_CHEETAH,
    "kitchen": KITCHEN,
    # aliases used by the scripts
    "ant": ANTMAZE,
    "walker": EXORL_WALKER,
    "cheetah": EXORL_CHEETAH,
}


# --------------------------------------------------------------------------- #
#  Task suite definitions
# --------------------------------------------------------------------------- #
def _antmaze_tasks() -> List[EvalTask]:
    steps = ANTMAZE.max_episode_steps
    tasks: List[EvalTask] = [
        EvalTask(
            name="ant-goal-reaching",
            kind="goal",
            params={
                "goals": {k: list(v) for k, v in ANTMAZE_GOALS.items()},
                "goal_threshold": 2.0,
                "xy_bins": 32,
                "reward_onsuccess": 0.0,
                "reward_offsuccess": -1.0,
            },
            max_episode_steps=steps,
        ),
        EvalTask(
            name="ant-directional",
            kind="velocity",
            params={
                "directions": {k: list(v) for k, v in ANTMAZE_DIRECTIONS.items()},
                "use_dot_product": True,
                "reward_offsuccess": -1.0,
                "reward_onsuccess": 0.0,
            },
            max_episode_steps=steps,
        ),
        EvalTask(
            name="ant-random-simplex",
            kind="simplex",
            params={
                "seeds": list(ANTMAZE_SIMPLEX_SEEDS),
                "baseline_reward": -1.0,
                "height_bonus": True,
                "velocity_bonus": True,
            },
            max_episode_steps=steps,
        ),
        EvalTask(
            name="ant-path-loop",
            kind="path",
            params={"path": "loop"},
            max_episode_steps=steps,
        ),
        EvalTask(
            name="ant-path-edges",
            kind="path",
            params={"path": "edges"},
            max_episode_steps=steps,
        ),
        EvalTask(
            name="ant-path-center",
            kind="path",
            params={"path": "center"},
            max_episode_steps=steps,
        ),
    ]
    return tasks


def _exorl_tasks() -> List[EvalTask]:
    steps = EXORL_WALKER.max_episode_steps
    tasks: List[EvalTask] = [
        EvalTask(
            name="exorl-walker-goals",
            kind="goal",
            params={
                "env": "walker",
                "num_goals": EXORL_NUM_GOALS,
                "goal_threshold": EXORL_GOAL_THRESHOLD,
                "normalize_by_dataset_std": True,
                "reward_onsuccess": 0.0,
                "reward_offsuccess": -1.0,
            },
            max_episode_steps=steps,
        ),
        EvalTask(
            name="exorl-walker-velocity",
            kind="velocity",
            params={
                "env": "walker",
                # the four walker velocity thresholds 0.1 / 1 / 4 / 8
                "thresholds": list(EXORL_WALKER_VELOCITY_THRESHOLDS),
                "physics_field": "horizontal_velocity",
                "reward_offsuccess": 0.0,
            },
            max_episode_steps=steps,
        ),
        EvalTask(
            name="exorl-cheetah-goals",
            kind="goal",
            params={
                "env": "cheetah",
                "num_goals": EXORL_NUM_GOALS,
                "goal_threshold": EXORL_GOAL_THRESHOLD,
                "normalize_by_dataset_std": True,
                "reward_onsuccess": 0.0,
                "reward_offsuccess": -1.0,
            },
            max_episode_steps=steps,
        ),
        EvalTask(
            name="exorl-cheetah-velocity",
            kind="velocity",
            params={
                "env": "cheetah",
                # cheetah-run (10), cheetah-walk (1) + backwards variants
                "thresholds": [
                    EXORL_CHEETAH_RUN_THRESHOLD,
                    -EXORL_CHEETAH_RUN_THRESHOLD,
                    EXORL_CHEETAH_WALK_THRESHOLD,
                    -EXORL_CHEETAH_WALK_THRESHOLD,
                ],
                "task_names": [
                    "cheetah-run",
                    "cheetah-run-backwards",
                    "cheetah-walk",
                    "cheetah-walk-backwards",
                ],
                "physics_field": "speed",
                "reward_offsuccess": 0.0,
            },
            max_episode_steps=steps,
        ),
    ]
    return tasks


def _kitchen_tasks() -> List[EvalTask]:
    steps = KITCHEN.max_episode_steps
    return [
        EvalTask(
            name="kitchen-{}".format(t.replace(" ", "-")),
            kind="kitchen",
            params={"subtask": t},
            max_episode_steps=steps,
        )
        for t in KITCHEN_SUBTASKS
    ]


#: Full task suites keyed by task-suite name.
EVAL_TASK_SUITES: Dict[str, List[EvalTask]] = {
    "antmaze": _antmaze_tasks(),
    "exorl-walker": [t for t in _exorl_tasks() if t.name.startswith("exorl-walker")],
    "exorl-cheetah": [t for t in _exorl_tasks() if t.name.startswith("exorl-cheetah")],
    "exorl": _exorl_tasks(),
    "kitchen": _kitchen_tasks(),
}

#: Names of the aggregated rows reported in Table 1.
#: ``antmaze-all`` / ``exorl-all`` average over every task of the domain,
#: ``all`` is the aggregate across the three domains (Table 1).
AGGREGATE_ROWS: Dict[str, Tuple[str, ...]] = {
    "antmaze-all": tuple(t.name for t in EVAL_TASK_SUITES["antmaze"]),
    "exorl-all": tuple(t.name for t in EVAL_TASK_SUITES["exorl"]),
    "kitchen": tuple(t.name for t in EVAL_TASK_SUITES["kitchen"]),
}

#: Table 1 reference numbers (mean +- std over 5 seeds), kept here so that the
#: evaluation scripts can print the comparison next to the measured score.
#: Source: §5.2, Table 1.
TABLE1_REFERENCE: Dict[str, Dict[str, float]] = {
    # domain row -> method -> score (mean)
    "antmaze-all": {"FRE": 52.8, "FB": 25.8, "SF": 11.8, "OPAL-10": 45.6},
    "exorl-all": {"FRE": 51.5, "FB": 43.4, "SF": 40.9, "OPAL-10": 28.2},
    "kitchen": {
        "FRE": 66.0,
        "FB": 3.0,
        "SF": 1.0,
        "GC-IQL": 59.0,
        "GC-BC": 35.0,
        "OPAL-10": 26.0,
    },
    "all": {"FRE": 57.0, "FB": 24.0, "SF": 18.0, "OPAL-10": 33.0},
}

#: Per-task FRE scores from Table 1 (mean +- std); used as validation targets.
TABLE1_FRE_PER_TASK: Dict[str, Tuple[float, float]] = {
    "ant-goal-reaching": (48.8, 6.0),
    "ant-directional": (55.2, 8.0),
    "ant-random-simplex": (21.3, 4.0),
    "ant-path-loop": (67.2, 36.0),
    "ant-path-edges": (60.0, 17.0),
    "ant-path-center": (64.4, 38.0),
    "exorl-walker-goals": (94.0, 2.0),
    "exorl-cheetah-goals": (58.0, 8.0),
    "exorl-walker-velocity": (34.0, 13.0),
    "exorl-cheetah-velocity": (20.0, 2.0),
}


# --------------------------------------------------------------------------- #
#  Public helpers
# --------------------------------------------------------------------------- #
def domain_names() -> List[str]:
    """Canonical domain names (aliases excluded)."""
    return ["antmaze", "exorl_walker", "exorl_cheetah", "kitchen"]


def get_domain_config(domain: str) -> DomainConfig:
    """Return the :class:`DomainConfig` for ``domain`` (aliases supported)."""
    key = str(domain).lower()
    if key not in DOMAINS:
        raise KeyError(
            "Unknown domain {!r}; available: {}".format(
                domain, sorted(set(DOMAINS.keys()))
            )
        )
    return DOMAINS[key]


#: alias used by ``fre/main.py``
get_env_config = get_domain_config


def get_eval_tasks(suite: str) -> List[EvalTask]:
    """Return the list of evaluation tasks for a task-suite name."""
    if suite not in EVAL_TASK_SUITES:
        raise KeyError(
            "Unknown task suite {!r}; available: {}".format(
                suite, sorted(EVAL_TASK_SUITES.keys())
            )
        )
    return list(EVAL_TASK_SUITES[suite])


def get_task(name: str) -> EvalTask:
    """Look up a single task by name across all suites."""
    for tasks in EVAL_TASK_SUITES.values():
        for task in tasks:
            if task.name == name:
                return task
    raise KeyError("Unknown task {!r}".format(name))


def default_domain_overrides(domain: str) -> Dict[str, object]:
    """Return the ``Config`` overrides for a domain.

    These are applied on top of :class:`fre.config.default.Config` by
    ``fre/main.py``.  Only keys that exist on ``Config`` are returned, so this
    function stays decoupled from the exact set of attributes.
    """
    dom = get_domain_config(domain)
    overrides: Dict[str, object] = {
        "domain": dom.name,
        "env_id": dom.env_id,
        "dataset_id": dom.dataset_id,
        "max_episode_steps": dom.max_episode_steps,
    }
    if dom.name == "antmaze":
        overrides.update(
            {
                "antmaze_max_episode_steps": dom.max_episode_steps,
                "antmaze_xy_bins": 32,
                "linear_exclude_dims": {"antmaze": dom.linear_exclude_dims},
            }
        )
    elif dom.name.startswith("exorl"):
        overrides.update(
            {
                "exorl_max_episode_steps": dom.max_episode_steps,
                "exorl_physics_fields": {
                    k: tuple(v) for k, v in dom.physics_fields.items()
                },
            }
        )
    elif dom.name == "kitchen":
        overrides.update(
            {
                "kitchen_max_episode_steps": dom.max_episode_steps,
                "kitchen_subtasks": list(KITCHEN_SUBTASKS),
            }
        )
    return overrides


def make_config(domain: str, **kwargs):
    """Build a :class:`fre.config.default.Config` for ``domain``.

    Unknown override keys are silently dropped so that this helper keeps working
    even if ``Config`` gains or loses attributes.
    """
    from fre.config.default import Config

    overrides = default_domain_overrides(domain)
    valid = set(dir(Config))
    merged = {k: v for k, v in overrides.items() if k in valid}
    merged.update(kwargs)
    valid_kwargs = {k: v for k, v in merged.items() if k in valid}
    return Config(**valid_kwargs)


def antmaze_discretize_xy(xy, bins: int = 32, low: float = 0.0, high: float = 40.0):
    """Discretize AntMaze XY coordinates into ``bins`` bins.

    Used by FRE, GC-IQL, GC-BC and OPAL for the AntMaze goal-reaching tasks
    (Appendix C.1).  ``low``/``high`` are the maze extent in the paper's grid
    coordinates; clip is applied so that out-of-range states map to the border
    bins.
    """
    import numpy as np

    arr = np.asarray(xy, dtype=np.float64)
    scaled = (arr - low) / max(high - low, 1e-8)
    scaled = np.clip(scaled, 0.0, 1.0 - 1e-6)
    return np.floor(scaled * bins).astype(np.int64)

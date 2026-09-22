"""RoboticSequence environment wrapper (Meta-World).

Reproduction of the RoboticSequence setting of

    "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting
     Mitigation Problem"  (Wołczyk et al., 2024)

Sources used (verbatim specifications):

* §3 (Experimental setup):
  "RoboticSequence is a multi-stage robotic task based on the Meta-World
   benchmark. The robot is successful only if during a single episode, it
   completes sequentially the following sub-tasks: use a hammer to hammer in a
   nail (hammer), push an object from one specific place to another (push),
   remove a bolt from a wall (peg-unplug-side), push an object around a wall
   (push-wall).  We use a pre-trained policy pi_* that can solve the last two
   stages, peg-unplug-side and push-wall (FAR), but not the first two, hammer
   and push (Close)."

* §B.3 and Algorithm 1:
   "Input: list of N environments E_k, policy pi, time limit T.
    Returns: number of solved environments.
    i = 1 ; t = 1 {Initialize env idx, timestep counter}
    while i <= N and t <= T do
        Take a step in E_i using pi
        if E_i is solved then
            i = i + 1 ; t = 1 {Move to the next env, reset timestep counter}
        end if
    end while
    Return i - 1"

* §B.3 (environment modifications):
   "we randomly sample the start and goal conditions ... we terminate the
    episode in two cases: when the agent succeeds or when the time limit is
    reached. In both cases, SAC receives a signal that the state was terminal,
    which means we do not apply bootstrapping in the target Q-value. In order
    for the MDP to be fully observable, we append the normalized timestep (i.e.
    the timestep divided by the maximal number of steps in the environment,
    T = 200 in our case) to the state vector. Additionally, when the episode
    ends with success, we provide the agent with the "remaining" reward it
    would get until the end of the episode. That is, if the last reward was
    originally r_t, the augmented reward is given by
        r'_t = beta * r_t * (T - t),
    beta = 1.5 is a coefficient to encourage the agent to succeed."

* §B.3 (observation / heads):
   "The observation space consists of information about the current robot
    configuration ... and the stage ID encoded as a one-hot vector. ...
    We create a separate output head for each stage in the neural networks and
    then we use the stage ID information to choose the correct head."

* §B.3 (episodic memory / retention):
   "For episodic memory, we sample 10k state-action-reward tuples from the
    pre-trained stages using the pre-trained policy and we keep them in SAC's
    replay buffer throughout the training on the downstream task. Since replay
    buffer is of size 100k, 10% of the buffer is filled with samples from the
    prior stages."

The module intentionally has **no hard dependency** on ``metaworld``: if the
package is missing the factory raises a helpful error, and a lightweight
``DummyStageEnv`` ("stub") implements the same interface so that environment
mechanics, SAC and the retention losses can be smoke-tested on CPU.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:  # numpy is required by every environment path but keep the import soft
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None  # type: ignore

try:  # metaworld is *optional* (CPU-only machines used for smoke tests)
    import metaworld as _metaworld  # type: ignore
except Exception:  # pragma: no cover
    _metaworld = None  # type: ignore


# --------------------------------------------------------------------------- #
# Constants specified by the paper
# --------------------------------------------------------------------------- #

#: Time limit T used by RoboticSequence (§B.3, Algorithm 1).
TIME_LIMIT = 200

#: Coefficient of the augmented success reward, r'_t = beta * r_t * (T - t).
BETA = 1.5

#: Main RoboticSequence ordering (§3).  The first two stages are CLOSE, the
#: last two (peg-unplug-side, push-wall) are FAR because pi_* solves them.
ROBOTIC_SEQUENCE_TASKS: Tuple[str, ...] = (
    "hammer",
    "push",
    "peg-unplug-side",
    "push-wall",
)

#: Stages solved by the pre-trained policy pi_* (FAR).
FAR_TASKS: Tuple[str, ...] = ("peg-unplug-side", "push-wall")

#: Stages *not* solved by pi_* (CLOSE for the pre-trained model).
CLOSE_TASKS: Tuple[str, ...] = ("hammer", "push")

#: Full Continual World (Wołczyk et al., 2021) task ordering, used for the
#: "alternative orderings" ablations requested by the reproduction plan.
CONTINUAL_WORLD_TASK_ORDER: Tuple[str, ...] = (
    "hammer",
    "push-wall",
    "faucet-close",
    "push-back",
    "stick-pull",
    "handle-press-side",
    "push",
    "shelf-place",
    "window-close",
    "peg-unplug-side",
)

#: Alternative 4-stage sequences (alternative orderings ablation).
ALTERNATIVE_ORDERINGS: Dict[str, Tuple[str, ...]] = {
    # Main sequence from §3.
    "main": ROBOTIC_SEQUENCE_TASKS,
    # Reversed sequence: the FAR stages come first.
    "reversed": tuple(reversed(ROBOTIC_SEQUENCE_TASKS)),
    # Continual-World-prefix of the same length as RoboticSequence.
    "continual_world_prefix": CONTINUAL_WORLD_TASK_ORDER[:4],
    # Two-stage instantiation from Figure 2 (a CLOSE stage then a FAR stage).
    "two_stage": ("push", "push-wall"),
    # FAR stages only (used to measure forgetting of what pi_* already knows).
    "far_only": FAR_TASKS,
    # CLOSE stages only (used as the "learned from scratch" control).
    "close_only": CLOSE_TASKS,
}


# --------------------------------------------------------------------------- #
# Task metadata
# --------------------------------------------------------------------------- #


@dataclass
class RoboticSequenceTask:
    """Metadata for a single Meta-World stage inside a RoboticSequence."""

    name: str
    index: int = 0
    is_far: bool = False
    env_id: Optional[str] = None

    @property
    def kwarg_name(self) -> str:
        """The ``metaworld`` environment key (goal-observable v2 variant)."""
        if self.env_id:
            return self.env_id
        base = self.name if self.name.endswith("-v2") else f"{self.name}-v2"
        return f"{base}-goal-observable"


def make_task(name: str, index: int = 0, is_far: Optional[bool] = None) -> RoboticSequenceTask:
    """Build a :class:`RoboticSequenceTask` with FAR/CLOSE inferred if omitted."""
    if is_far is None:
        is_far = strip_version(name) in FAR_TASKS
    return RoboticSequenceTask(name=name, index=index, is_far=bool(is_far))


def strip_version(name: str) -> str:
    """``"push-wall-v2" -> "push-wall"`` (and strip the goal-observable suffix)."""
    name = name.replace("-goal-observable", "")
    if name.endswith("-v2") or name.endswith("-v3"):
        name = name[:-3]
    return name


def tasks_for(name: str) -> Tuple[str, ...]:
    """Resolve a sequence name (``"main"``, ``"reversed"``, ...) to task names."""
    if name in ALTERNATIVE_ORDERINGS:
        return ALTERNATIVE_ORDERINGS[name]
    if "," in name:
        return tuple(t.strip() for t in name.split(",") if t.strip())
    # A plain task name: a single-stage sequence.
    return (name,)


def prefix_tasks(sequence: Sequence[str], prefix: Optional[int] = None) -> Tuple[str, ...]:
    """Truncate ``sequence`` to its first ``prefix`` stages (Table 6 ablations)."""
    if prefix is None:
        return tuple(sequence)
    if prefix < 0:
        raise ValueError("prefix must be >= 0")
    return tuple(sequence[:prefix])


# --------------------------------------------------------------------------- #
# Reward / observation helpers (exactly the paper's formulas)
# --------------------------------------------------------------------------- #


def augmented_reward(reward: float, t: int, time_limit: int = TIME_LIMIT,
                     beta: float = BETA) -> float:
    """Augmented success reward ``r'_t = beta * r_t * (T - t)`` (§B.3)."""
    return float(beta) * float(reward) * float(time_limit - int(t))


def normalized_timestep(t: int, time_limit: int = TIME_LIMIT) -> float:
    """``t / T`` clipped to ``[0, 1]`` — appended to the state vector (§B.3)."""
    if time_limit <= 0:
        return 0.0
    return float(min(max(t, 0), time_limit)) / float(time_limit)


# --------------------------------------------------------------------------- #
# Stub (dummy) environment — allows CPU smoke tests without Meta-World
# --------------------------------------------------------------------------- #


class DummyStageEnv:
    """A tiny continuous-control stand-in for one Meta-World stage.

    Semantics mirror the *interface* relied upon by RoboticSequence (and the
    Success signal behaviour of Meta-World v2):

    * ``reset()`` samples a random start/goal condition (the goal is a random
      point in ``[-1, 1]^action_dim``; the start is random noise);
    * ``step(action)`` moves the internal state towards the action with a
      random perturbation; the stage is *solved* when the internal state is
      within ``goal_tolerance`` of the goal;
    * ``info["success"]`` and ``env.success`` expose the success signal;
    * the episode is *not* terminated by the environment itself, the wrapper
      (RoboticSequence) handles success/time-limit termination.

    The reward is a smooth negative-distance reward clipped to ``[0, 1]`` so
    that the augmented success reward of the paper remains meaningful.
    """

    def __init__(self, name: str = "dummy", obs_dim: int = 9, action_dim: int = 4,
                 goal_tolerance: float = 0.25, max_episode_steps: int = TIME_LIMIT,
                 seed: Optional[int] = None, solve_probability: float = 1.0) -> None:
        self.name = name
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.goal_tolerance = float(goal_tolerance)
        self.max_episode_steps = int(max_episode_steps)
        # ``solve_probability`` reproduces the CLOSE/FAR asymmetry: a stage with
        # probability 1.0 is solvable (FAR for pi_*), a lower one is not.
        self.solve_probability = float(solve_probability)
        self._rng = _np.random.RandomState(seed if seed is not None else 0)
        self.state = _np.zeros(self.action_dim, dtype=_np.float32)
        self.goal = _np.zeros(self.action_dim, dtype=_np.float32)
        self.success = False
        self.num_steps = 0
        self._freeze_rand_vec = True
        self.observation_space = _Space(self.obs_dim)
        self.action_space = _Space(self.action_dim, low=-1.0, high=1.0)

    # -- meta-world compatible API -----------------------------------------
    def seed(self, seed: Optional[int] = None) -> List[int]:
        if seed is not None:
            self._rng = _np.random.RandomState(int(seed))
        return [int(seed) if seed is not None else 0]

    def reset(self, **kwargs: Any):
        seed = kwargs.get("seed", None)
        if seed is not None:
            self.seed(seed)
        self.state = self._rng.uniform(-1.0, 1.0, size=self.action_dim).astype(_np.float32)
        if self._rng.rand() < self.solve_probability:
            self.goal = self.state + self._rng.uniform(
                -0.5 * self.goal_tolerance, 0.5 * self.goal_tolerance,
                size=self.action_dim,
            )
        else:  # unsolvable-in-practice stage (mimics CLOSE stages for pi_*)
            self.goal = self._rng.uniform(-1.0, 1.0, size=self.action_dim).astype(_np.float32)
        self.goal = self.goal.astype(_np.float32)
        self.success = False
        self.num_steps = 0
        obs = self._obs()
        return obs, {} if _is_new_gym() else obs

    def step(self, action: Any):
        action = _np.asarray(action, dtype=_np.float32).reshape(-1)[: self.action_dim]
        if action.shape[0] < self.action_dim:  # pad if a scalar was passed
            action = _np.pad(action, (0, self.action_dim - action.shape[0]))
        action = _np.clip(action, -1.0, 1.0)
        self.state = 0.5 * self.state + 0.5 * action
        self.state += self._rng.normal(0.0, 0.01, size=self.action_dim).astype(_np.float32)
        self.num_steps += 1

        distance = float(_np.linalg.norm(self.state - self.goal))
        solved_now = distance < self.goal_tolerance
        if solved_now and not self.success:
            self.success = True
        reward = float(max(0.0, 1.0 - distance))
        terminated = False
        truncated = False
        info = {"success": bool(self.success), "distance": distance}
        obs = self._obs()
        if _is_new_gym():
            return obs, reward, terminated, truncated, info
        return obs, reward, terminated, info

    def _obs(self) -> Any:
        return _np.concatenate([self.state, self.goal, [0.0, 0.0, 0.0]]).astype(_np.float32)

    def render(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        return None

    def close(self) -> None:
        return None


class _Space:
    """Minimal gym-like box space (avoids a hard gym dependency)."""

    def __init__(self, shape: int, low: float = -_INF, high: float = _INF) -> None:
        self.shape = (int(shape),)
        self.low = _np.full(self.shape, low, dtype=_np.float32) if _np is not None else low
        self.high = _np.full(self.shape, high, dtype=_np.float32) if _np is not None else high

    def sample(self) -> Any:
        return self.low + (self.high - self.low) * _np.random.rand(*self.shape)

    def contains(self, x: Any) -> bool:
        try:  # pragma: no cover - defensive
            x = _np.asarray(x).reshape(-1)
            return bool(x.shape[0] == self.shape[0])
        except Exception:
            return False

    def __repr__(self) -> str:  # pragma: no cover
        return f"_Space(shape={self.shape})"


_INF = float("inf")


def _is_new_gym() -> bool:
    """Detect gym>=0.26 / gymnasium step & reset signatures."""
    try:
        import gym  # type: ignore
        version = getattr(gym, "__version__", "0.0.0")
        parts = version.split(".")
        return (int(parts[0]), int(parts[1])) >= (0, 26)
    except Exception:
        pass
    try:
        import gymnasium  # type: ignore

        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Environment factory
# --------------------------------------------------------------------------- #


def make_stage_env(task: Union[str, RoboticSequenceTask], seed: Optional[int] = None,
                   stub: bool = False, solve_probability: float = 1.0,
                   max_episode_steps: int = TIME_LIMIT, **kwargs: Any) -> Any:
    """Create one Meta-World stage environment.

    Parameters
    ----------
    task:
        Stage name (``"push-wall"``) or a :class:`RoboticSequenceTask`.
    seed:
        Seed forwarded to the environment.
    stub:
        If ``True`` build a :class:`DummyStageEnv` instead of Meta-World (used
        by the CPU smoke tests).
    solve_probability:
        Only used by the stub environment; controls whether the stage is
        solvable (mimics the FAR/CLOSE asymmetry of pi_*).
    max_episode_steps:
        Episode horizon (``T = 200`` for RoboticSequence).
    """
    name = task.name if isinstance(task, RoboticSequenceTask) else str(task)
    if stub or _metaworld is None:
        if not stub:
            raise ImportError(
                "metaworld is not installed. Install it (pip install metaworld) or "
                "set `stub=True` / `cfg.env.stub=true` to use the DummyStageEnv."
            )
        clean = strip_version(name)
        prob = solve_probability
        if solve_probability >= 1.0 and clean in FAR_TASKS:
            prob = 1.0
        if solve_probability >= 1.0 and clean in CLOSE_TASKS:
            prob = 1.0
        return DummyStageEnv(name=clean, seed=seed, h=..., **{}) if False else DummyStageEnv(
            name=clean, seed=seed, solve_probability=prob,
            max_episode_steps=max_episode_steps,
        )

    env_cls = _lookup_metaworld_env(task)
    env = env_cls(**kwargs) if kwargs else env_cls()
    # Random start / goal conditions (as in Wołczyk et al., 2021 and §B.3).
    try:
        env._freeze_rand_vec = False
    except Exception:  # pragma: no cover
        pass
    try:
        env.max_path_length = int(max_episode_steps)
    except Exception:  # pragma: no cover
        pass
    try:
        env.num_envs = 1
    except Exception:  # pragma: no cover
        pass
    if seed is not None:
        try:
            env.seed(int(seed))
        except Exception:  # pragma: no cover
            try:
                env.reset(seed=int(seed))
            except Exception:
                pass
    return env


def _lookup_metaworld_env(task: Union[str, RoboticSequenceTask]) -> Any:
    """Resolve the goal-observable v2 Meta-World class for a task."""
    if _metaworld is None:  # pragma: no cover
        raise ImportError("metaworld is not installed")
    meta = task if isinstance(task, RoboticSequenceTask) else make_task(task)
    registries = []
    for attr in (
        "ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE",
        "ALL_V2_ENVIRONMENTS",
        "ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE",
    ):
        registry = getattr(_metaworld, attr, None)
        if registry:
            registries.append(registry)
    for registry in registries:
        if meta.kwarg_name in registry:
            return registry[meta.kwarg_name]
        base = meta.kwarg_name.replace("-goal-observable", "")
        if base in registry:
            return registry[base]
    raise KeyError(
        f"Could not find Meta-World environment for task '{meta.name}'. "
        f"Tried keys: {meta.kwarg_name}, {meta.kwarg_name.replace('-goal-observable', '')}"
    )


# --------------------------------------------------------------------------- #
# The RoboticSequence environment (Algorithm 1)
# --------------------------------------------------------------------------- #


class RoboticSequenceEnv:
    """A single Meta-World stage following Algorithm 1 of the paper.

    The wrapper holds the *whole* sequence of stages and advances ``stage``
    whenever the current stage emits its success signal.  Every episode
    therefore corresponds to "solve stage ``i``", and the observation carries
    the normalized timestep ``t / T`` (§B.3) so that the resulting MDP is fully
    observable even though the stage index changes within the sequence.

    Notes
    -----
    * The episode terminates on success **or** on reaching the time limit
      ``T = 200``.  In both cases ``terminal=True`` is reported, so SAC must
      *not* bootstrap the target Q-value (§B.3).
    * When the episode ends with success the environment returns the augmented
      reward ``r'_t = beta * r_t * (T - t)`` with ``beta = 1.5``.
    * ``info["stage_id"]`` (and optionally ``info["stage_onehot"]``) exposes the
      stage identity used to pick the per-stage actor/critic head.
    """

    def __init__(
        self,
        task_order: Union[str, Sequence[str], None] = None,
        time_limit: int = TIME_LIMIT,
        beta: float = BETA,
        seed: Optional[int] = None,
        stub: bool = False,
        prefix: Optional[int] = None,
        append_stage_onehot: bool = False,
        append_timestep: bool = True,
        terminal_on_time_limit: bool = True,
        episodic_success_reward: bool = True,
        solve_probability: float = 1.0,
        env_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.task_names: Tuple[str, ...] = tuple(
            prefix_tasks(tasks_for(task_order or "main"), prefix)
        )
        if len(self.task_names) == 0:
            raise ValueError("RoboticSequence requires at least one stage")
        self.tasks: List[RoboticSequenceTask] = [
            make_task(name, index=i) for i, name in enumerate(self.task_names)
        ]
        self.time_limit = int(time_limit)
        self.beta = float(beta)
        self.seed_value = seed
        self.stub = bool(stub)
        self.append_stage_onehot = bool(append_stage_onehot)
        self.append_timestep = bool(append_timestep)
        self.terminal_on_time_limit = bool(terminal_on_time_limit)
        self.episodic_success_reward = bool(episodic_success_reward)
        self.solve_probability = float(solve_probability)
        self.env_kwargs = dict(env_kwargs or {})
        self._envs: List[Any] = []
        self._env: Any = None
        self.stage_id: int = 0
        self.t: int = 0  # timestep counter inside the current stage (Algorithm 1)
        self.episode_return: float = 0.0
        self.num_solved: int = 0
        self.episode_success: bool = False
        self.last_reward: float = 0.0
        self.last_raw_reward: float = 0.0
        self._make_env_at(0, seed=seed)

    # -- construction ------------------------------------------------------ #
    def _make_env_at(self, index: int, seed: Optional[int] = None) -> Any:
        task = self.tasks[index]
        stage_seed = None if seed is None else int(seed) + 1000 * index
        env = make_stage_env(
            task,
            seed=stage_seed,
            stub=self.stub,
            solve_probability=self.solve_probability,
            max_episode_steps=self.time_limit,
            **self.env_kwargs,
        )
        if len(self._envs) <= index:
            self._envs.append(env)
        else:
            self._envs[index] = env
        self._env = env
        if len(self._envs) < len(self.tasks):
            for j in range(len(self._envs), len(self.tasks)):
                if self._envs[j] is None:
                    break
        return env

    def build_all_envs(self, seed: Optional[int] = None) -> None:
        """Instantiate every stage environment (needed for evaluation only)."""
        self._envs = [
            make_stage_env(
                self.tasks[i],
                seed=None if seed is None else int(seed) + 1000 * i,
                stub=self.stub,
                solve_probability=self.solve_probability,
                max_episode_steps=self.time_limit,
                **self.env_kwargs,
            )
            for i in range(len(self.tasks))
        ]
        self._env = self._envs[self.stage_id]

    # -- gym-like API ------------------------------------------------------ #
    @property
    def env(self) -> Any:
        return self._env

    @property
    def current_task(self) -> RoboticSequenceTask:
        return self.tasks[self.stage_id]

    @property
    def n_stages(self) -> int:
        return len(self.tasks)

    @property
    def is_far_stage(self) -> bool:
        return bool(self.current_task.is_far)

    @property
    def far_stages(self) -> Tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.tasks) if t.is_far)

    @property
    def close_stages(self) -> Tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.tasks) if not t.is_far)

    @property
    def observation_dim(self) -> int:
        base = self._base_obs_dim()
        extra = 0
        extra += 1 if self.append_timestep else 0
        extra += (self.n_stages if self.append_stage_onehot else 0)
        return base + extra

    def _base_obs_dim(self) -> int:
        space = getattr(self._env, "observation_space", None)
        shape = getattr(space, "shape", None)
        if shape is None:
            return 0
        if isinstance(shape, (tuple, list)):
            dim = 1
            for s in shape:
                dim *= int(s)
            return dim
        return int(shape)

    @property
    def action_dim(self) -> int:
        space = getattr(self._env, "action_space", None)
        shape = getattr(space, "shape", None)
        if shape is None:
            return 0
        if isinstance(shape, (tuple, list)):
            dim = 1
            for s in shape:
                dim *= int(s)
            return dim
        return int(shape)

    @property
    def observation_space(self) -> Any:
        return _Space(self.observation_dim)

    @property
    def action_space(self) -> Any:
        return getattr(self._env, "action_space", _Space(self.action_dim, -1.0, 1.0))

    # -- observation augmentation ----------------------------------------- #
    def augment(self, obs: Any, stage_id: Optional[int] = None) -> Any:
        """Append ``t / T`` (and, optionally, the one-hot stage ID) to ``obs``."""
        if _np is None:  # pragma: no cover
            raise RuntimeError("numpy is required by RoboticSequenceEnv")
        stage = self.stage_id if stage_id is None else int(stage_id)
        base = _np.asarray(obs, dtype=_np.float32).reshape(-1)
        pieces = [base]
        if self.append_timestep:
            pieces.append(_np.asarray([normalized_timestep(self.t, self.time_limit)],
                                      dtype=_np.float32))
        if self.append_stage_onehot:
            pieces.append(self.stage_onehot(stage))
        if len(pieces) == 1:
            return base
        return _np.concatenate(pieces).astype(_np.float32)

    def stage_onehot(self, stage_id: Optional[int] = None) -> Any:
        stage = self.stage_id if stage_id is None else int(stage_id)
        vec = _np.zeros(self.n_stages, dtype=_np.float32)
        if 0 <= stage < self.n_stages:
            vec[stage] = 1.0
        return vec

    # -- Algorithm 1 ------------------------------------------------------- #
    def reset(self, seed: Optional[int] = None, stage_id: Optional[int] = None,
              **kwargs: Any) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
        """Reset the sequence (stage 0 by default) and the timestep counter."""
        if seed is not None:
            self.seed_value = int(seed)
        if stage_id is not None:
            if not (0 <= int(stage_id) < self.n_stages):
                raise ValueError("stage_id out of range")
            self.stage_id = int(stage_id)
            if len(self._envs) > self.stage_id and self._envs[self.stage_id] is not None:
                self._env = self._envs[self.stage_id]
            else:
                self._make_env_at(self.stage_id, seed=self.seed_value)
        self.t = 0
        self.episode_return = 0.0
        self.episode_success = False
        self.num_solved = 0
        self.last_reward = 0.0
        self.last_raw_reward = 0.0
        out = self._env.reset()
        obs, info = _split_reset(out)
        obs = self.augment(obs)
        info = dict(info or {})
        info.update(self._info_dict(success=False, solved=False))
        return (obs, info) if _is_new_gym() else obs

    def step(self, action: Any):
        """Take one step in the current stage ``E_i`` (Algorithm 1)."""
        self.t += 1
        out = self._env.step(action)
        obs, reward, done, info = _split_step(out)
        raw_reward = float(reward)

        solved = bool(_success_signal(self._env, info))
        tl = self.time_limit
        terminated = False
        truncated = bool(done)

        if solved:
            # Augmented "remaining" reward, r'_t = beta * r_t * (T - t).
            if self.episodic_success_reward:
                reward = augmented_reward(raw_reward, self.t, tl, self.beta)
            else:
                reward = raw_reward
            self.episode_success = True
            self.num_solved += 1
            terminated = True  # terminal on success => no bootstrapping
        elif self.t >= tl:
            # Time limit reached: also terminal (no bootstrapping, §B.3).
            reward = raw_reward
            if self.terminal_on_time_limit:
                terminated = True
            else:
                truncated = True

        reward = float(reward)
        self.last_reward = reward
        self.last_raw_reward = raw_reward
        self.episode_return += reward

        obs = self.augment(obs)
        info = dict(info or {})
        info.update(self._info_dict(success=solved, solved=solved))
        if _is_new_gym():
            return obs, reward, bool(terminated), bool(truncated), info
        return obs, reward, bool(terminated or truncated), info

    def advance_stage(self) -> bool:
        """Move to the next stage and reset the timestep counter (Algorithm 1).

        Returns ``True`` if a next stage exists.
        """
        if self.stage_id + 1 >= self.n_stages:
            return False
        self.stage_id += 1
        self.t = 0
        if self.stage_id < len(self._envs) and self._envs[self.stage_id] is not None:
            self._env = self._envs[self.stage_id]
            try:
                self._env.reset()
            except Exception:  # pragma: no cover
                pass
        else:
            self._make_env_at(self.stage_id, seed=self.seed_value)
            self._env.reset()
        return True

    def alg1_rollout(self, policy: Callable[[Any, int], Any],
                     max_total_steps: Optional[int] = None,
                     deterministic: bool = True,
                     seed: Optional[int] = None) -> int:
        """Run Algorithm 1 verbatim and return the number of solved stages.

        ``policy`` is a callable ``(obs, stage_id) -> action``.  The loop stops
        when all stages are solved, when the time limit of a stage is exceeded,
        or when ``max_total_steps`` steps have been taken.
        """
        self.reset(seed=seed, stage_id=0)
        obs = self._last_obs
        total = 0
        while self.stage_id < self.n_stages and self.t <= self.time_limit:
            action = policy(obs, self.stage_id) if not hasattr(policy, "act") else \
                policy.act(obs, deterministic=deterministic)
            out = self.step(action)
            obs = out[0]
            done = out[2]
            solved = bool(out[-1].get("success", False)) if isinstance(out[-1], dict) else False
            total += 1
            if max_total_steps is not None and total >= max_total_steps:
                break
            if solved:
                if not self.advance_stage():
                    break
                obs = self.augment(self._last_env_obs())
            elif done:
                # Stage failed (time limit): Algorithm 1 stops here.
                break
        return self.stage_id  # == i - 1 in the paper's 1-based indexing

    def solve(self, policy: Any, seed: Optional[int] = None,
              deterministic: bool = True) -> int:
        """Evaluate ``policy`` on the whole sequence, returning #solved stages."""
        def _fn(obs: Any, stage: int) -> Any:
            if hasattr(policy, "act"):
                try:
                    return policy.act(obs, deterministic=deterministic)
                except TypeError:  # pragma: no cover
                    return policy.act(obs)
            return policy(obs)
        return self.alg1_rollout(_fn, deterministic=deterministic, seed=seed)

    # -- internal helpers -------------------------------------------------- #
    def _last_env_obs(self) -> Any:
        """Latest raw observation produced by the underlying stage env."""
        try:
            return self._raw_obs_cache
        except AttributeError:  # pragma: no cover
            return _np.zeros(1, dtype=_np.float32)

    def _info_dict(self, success: bool, solved: bool) -> Dict[str, Any]:
        return {
            "stage_id": int(self.stage_id),
            "stage_name": self.current_task.name,
            "n_stages": int(self.n_stages),
            "is_far": bool(self.current_task.is_far),
            "timestep": int(self.t),
            "normalized_timestep": normalized_timestep(self.t, self.time_limit),
            "success": bool(success),
            "stage_success": bool(solved),
            "num_solved": int(self.num_solved),
            "episode_success": bool(self.episode_success),
            "episode_return": float(self.episode_return),
            "raw_reward": float(self.last_raw_reward),
            "stage_onehot": self.stage_onehot(),
        }

    def render(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return getattr(self._env, "render", lambda *a, **k: None)(*args, **kwargs)

    def close(self) -> None:
        for env in self._envs:
            try:
                env.close()
            except Exception:  # pragma: no cover
                pass

    def seed(self, seed: Optional[int] = None) -> List[int]:
        self.seed_value = seed
        if seed is None:
            return []
        out = []
        for i, env in enumerate(self._envs):
            try:
                out.extend(env.seed(int(seed) + 1000 * i))
            except Exception:  # pragma: no cover
                pass
        return out

    def __repr__(self) -> str:  # pragma: no cover
        return (f"RoboticSequenceEnv(stages={list(self.task_names)}, T={self.time_limit}, "
                f"beta={self.beta}, stage={self.stage_id}, t={self.t})")


# --------------------------------------------------------------------------- #
# Vectorised wrapper (many seeds / orderings in parallel)
# --------------------------------------------------------------------------- #


class RoboticSequenceVecEnv:
    """A minimal synchronous ``VecEnv`` over several RoboticSequence instances."""

    def __init__(self, num_envs: int = 1, task_order: Union[str, Sequence[str]] = "main",
                 base_seed: Optional[int] = None, stub: bool = False,
                 prefix: Optional[int] = None, **env_kwargs: Any) -> None:
        self.num_envs = int(num_envs)
        self.envs: List[RoboticSequenceEnv] = [
            RoboticSequenceEnv(
                task_order=task_order,
                seed=None if base_seed is None else int(base_seed) + i,
                stub=stub,
                prefix=prefix,
                **env_kwargs,
            )
            for i in range(self.num_envs)
        ]
        self._last_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]

    def reset(self) -> Tuple[Any, List[Dict[str, Any]]]:
        obs = []
        infos = []
        for env in self.envs:
            out = env.reset()
            if isinstance(out, tuple):
                obs.append(out[0])
                self._last_infos[self.envs.index(env)] = out[1]
                infos.append(out[1])
            else:
                obs.append(out)
                infos.append({})
        return _stack(obs), infos

    def step(self, actions: Any):
        obs, rewards, dones, infos = [], [], [], []
        arr = _np.asarray(actions)
        for i, env in enumerate(self.envs):
            action = arr[i] if arr.ndim > 1 else arr
            out = env.step(action)
            o, r, d, info = out[0], out[1], out[2], out[-1]
            if d:
                res = env.reset()
                o = res[0] if isinstance(res, tuple) else res
                info = dict(info)
                info["episode_done"] = True
            obs.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        return (_stack(obs), _np.asarray(rewards, dtype=_np.float32),
                _np.asarray(dones, dtype=bool), infos)

    def close(self) -> None:
        for env in self.envs:
            env.close()

    @property
    def observation_dim(self) -> int:
        return self.envs[0].observation_dim

    @property
    def action_dim(self) -> int:
        return self.envs[0].action_dim


# --------------------------------------------------------------------------- #
# Evaluation helpers (Figure 7 / Table 6)
# --------------------------------------------------------------------------- #


def per_stage_success_rate(policy: Any, task_order: Union[str, Sequence[str]] = "main",
                           num_episodes: int = 20, seed: Optional[int] = None,
                           stub: bool = False, deterministic: bool = True,
                           **env_kwargs: Any) -> Dict[str, float]:
    """Success rate of ``policy`` on each individual stage of the sequence.

    This is the quantity plotted in Figure 7 (per-stage success rate after
    fine-tuning).  The policy is evaluated stage-by-stage by forcing
    ``stage_id`` at reset, which isolates "can the policy still solve stage k".
    """
    names = tuple(tasks_for(task_order))
    env = RoboticSequenceEnv(task_order=task_order, seed=seed, stub=stub, **env_kwargs)
    rates: Dict[str, float] = {}
    for idx, name in enumerate(names):
        successes = 0
        for ep in range(int(num_episodes)):
            env.reset(seed=None if seed is None else int(seed) + ep, stage_id=idx)
            done = False
            while not done:
                obs = env.augment(env._raw_obs_cache) if False else None  # placeholder
                action = _policy_action(policy, env, idx, deterministic=deterministic)
                out = env.step(action)
                done = bool(out[2])
                if isinstance(out[-1], dict) and out[-1].get("success"):
                    successes += 1
                    break
        rates[name] = successes / float(max(1, num_episodes))
        rates[f"{name}:is_far"] = float(make_task(name).is_far)
    env.close()
    return rates


def _policy_action(policy: Any, env: RoboticSequenceEnv, stage_id: int,
                   deterministic: bool = True) -> Any:
    obs = env.augment(env._raw_obs_cache, stage_id=stage_id)
    if hasattr(policy, "act"):
        try:
            return policy.act(obs, deterministic=deterministic)
        except TypeError:  # pragma: no cover
            return policy.act(obs)
    return policy(obs)


def forward_transfer_metric(auc: float, baseline_auc: float) -> float:
    """Forward transfer of §F: ``(AUC - AUC^b) / (1 - AUC^b)``."""
    denom = 1.0 - float(baseline_auc)
    if abs(denom) < 1e-12:
        return 0.0
    return (float(auc) - float(baseline_auc)) / denom


# --------------------------------------------------------------------------- #
# gym registration helpers (optional)
# --------------------------------------------------------------------------- #


def register_robotic_sequence_envs(task_order: str = "main") -> None:
    """Register ``RoboticSequence-v0`` with gym/gymnasium when available."""
    try:
        if _is_new_gym():
            import gymnasium as gym  # type: ignore
        else:  # pragma: no cover
            import gym  # type: ignore
    except Exception:  # pragma: no cover
        return
    try:
        gym.register(
            id="RoboticSequence-v0",
            entry_point="src.robotic_sequence.env:RoboticSequenceEnv",
            kwargs={"task_order": task_order},
            max_episode_steps=TIME_LIMIT,
        )
    except Exception:  # pragma: no cover - already registered
        pass


# --------------------------------------------------------------------------- #
# low level helpers (gym API normalisation)
# --------------------------------------------------------------------------- #


def _split_reset(out: Any) -> Tuple[Any, Dict[str, Any]]:
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], (out[1] or {})
    if isinstance(out, dict) and "obs" in out:  # pragma: no cover
        return out["obs"], {k: v for k, v in out.items() if k != "obs"}
    return out, {}


def _split_step(out: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
    if isinstance(out, tuple):
        if len(out) == 5:  # gym >= 0.26
            obs, reward, terminated, truncated, info = out
            return obs, float(reward), bool(terminated or truncated), dict(info or {})
        if len(out) == 4:
            obs, reward, done, info = out
            return obs, float(reward), bool(done), dict(info or {})
    raise ValueError(f"Unsupported step() output: {type(out)}")


def _success_signal(env: Any, info: Optional[Dict[str, Any]] = None) -> bool:
    """Extract Meta-World's success signal (``info['success']`` or ``env.success``)."""
    if info:
        for key in ("success", "is_success", "Success"):
            if key in info:
                val = info[key]
                try:
                    return bool(_np.asarray(val).reshape(-1)[0])
                except Exception:  # pragma: no cover
                    return bool(val)
    for attr in ("success", "_success", "solved"):
        if hasattr(env, attr):
            try:
                val = getattr(env, attr)
                if callable(val):  # pragma: no cover
                    val = val()
                return bool(_np.asarray(val).reshape(-1)[0]) if hasattr(val, "__len__") else bool(val)
            except Exception:  # pragma: no cover
                pass
    return False


def _stack(obs_list: Iterable[Any]) -> Any:
    if _np is None:  # pragma: no cover
        return list(obs_list)
    arrs = [_np.asarray(o, dtype=_np.float32).reshape(-1) for o in obs_list]
    if len(arrs) == 1:
        return arrs[0]
    max_len = max(a.shape[0] for a in arrs)
    padded = [_np.pad(a, (0, max_len - a.shape[0])) for a in arrs]
    return _np.stack(padded, axis=0).astype(_np.float32)


__all__ = [
    "TIME_LIMIT",
    "BETA",
    "ROBOTIC_SEQUENCE_TASKS",
    "FAR_TASKS",
    "CLOSE_TASKS",
    "CONTINUAL_WORLD_TASK_ORDER",
    "ALTERNATIVE_ORDERINGS",
    "RoboticSequenceTask",
    "make_task",
    "tasks_for",
    "prefix_tasks",
    "strip_version",
    "augmented_reward",
    "normalized_timestep",
    "make_stage_env",
    "DummyStageEnv",
    "RoboticSequenceEnv",
    "RoboticSequenceVecEnv",
    "per_stage_success_rate",
    "forward_transfer_metric",
    "register_robotic_sequence_envs",
]

"""Success-tolerance curricula for the SAPG task suite (Appendix A, Sec. 5.1).

Appendix A specifies, for the hard Allegro-Kuka tasks:

    "The success tolerance :math:`\\delta` defines the maximum error between
    object pose and goal pose for a success
    :math:`\\|g_t - (x_t)_{0:3}\\| \\leq \\delta`.  This tolerance is decreased in
    a curriculum from 7.5 cm to 1 cm, decremented by 10% each time the average
    number of successes in an episode crosses 3."

This module implements exactly that rule as a small, dependency-light
controller that:

* owns the current tolerance ``delta`` (initialised to 7.5 cm and floored at
  1 cm),
* observes the number of successes returned by finished episodes,
* keeps a running estimate of the *average number of successes per episode*,
* multiplies ``delta`` by ``decrement`` (0.9) whenever that average crosses
  ``threshold`` (3), and
* pushes the new tolerance into the vectorised environment via
  ``env.set_success_tolerance(delta)`` (the interface exposed by
  :mod:`sapg.envs.isaac_env`).

The controller is deliberately decoupled from IsaacGym: it works with plain
Python floats, dicts, tensors or an :class:`~sapg.envs.isaac_env.EnvConfig`,
which makes it unit-testable without a GPU.

Typical use inside the training loop::

    curriculum = make_curriculum("regrasping", env=env, config=config)
    ...
    obs, rewards, dones, infos = env.step(actions)
    curriculum.update(infos)          # infos carries per-step success counts
    curriculum.step(episode_successes)  # or feed finished-episode statistics
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "SuccessCurriculum",
    "make_curriculum",
    "update_curriculum",
    "apply_tolerance",
    "CURRICULUM_TASKS",
    "DEFAULT_INITIAL_TOLERANCE",
    "DEFAULT_MIN_TOLERANCE",
    "DEFAULT_DECREMENT",
    "DEFAULT_THRESHOLD",
]

# ---------------------------------------------------------------------------
# Paper constants (Appendix A / Sec. 5.1)
# ---------------------------------------------------------------------------

#: Initial success tolerance in metres (7.5 cm).
DEFAULT_INITIAL_TOLERANCE: float = 0.075
#: Final / minimum success tolerance in metres (1 cm).
DEFAULT_MIN_TOLERANCE: float = 0.01
#: Multiplicative decrement applied on each curriculum advancement (10%).
DEFAULT_DECREMENT: float = 0.9
#: Average successes per episode that triggers an advancement ("crosses 3").
DEFAULT_THRESHOLD: float = 3.0
#: Tasks with a tolerance curriculum (Appendix A: the Allegro-Kuka tasks).
CURRICULUM_TASKS: Tuple[str, ...] = ("regrasping", "throw", "reorientation")


def _as_float(value: Any, default: float = 0.0) -> float:
    """Best-effort conversion of scalar-like values (tensor, numpy, str)."""
    if value is None:
        return default
    try:
        import torch  # local import: keep module import cheap

        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            return float(value.detach().float().mean().item())
    except Exception:  # pragma: no cover - torch optional
        pass
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return sum(_as_float(v, default) for v in value) / float(len(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _iter_leaf_scalars(value: Any) -> Iterable[Any]:
    """Yield scalar leaves of arbitrarily nested mappings/sequences."""
    if value is None:
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_leaf_scalars(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            # A tuple like (mean, min, max) is averaged by ``_as_float``.
            yield item
    else:
        yield value


# ---------------------------------------------------------------------------
# Curriculum controller
# ---------------------------------------------------------------------------


@dataclass
class SuccessCurriculum:
    """Success-tolerance curriculum of Appendix A.

    The tolerance starts at ``initial_tolerance`` (7.5 cm) and is repeatedly
    multiplied by ``decrement`` (0.9) whenever the running average number of
    successes per episode crosses ``threshold`` (3), until it reaches
    ``min_tolerance`` (1 cm).

    Success counts are accumulated in two complementary ways:

    * :meth:`update` accepts environment ``infos`` (per-step) and looks for a
      success counter, using ``successes_key``;
    * :meth:`record_episode` accepts the number of successes of a *finished*
      episode directly (the ``successes`` field of the env metrics produced by
      :mod:`sapg.algorithms.rollout`).

    Both feed the same running estimate, so either or both may be used.

    Attributes
    ----------
    initial_tolerance:
        Starting ``delta`` in metres (paper: 0.075).
    min_tolerance:
        Lower clamp / final ``delta`` in metres (paper: 0.01).
    decrement:
        Multiplicative factor applied on advancement (paper: 0.9).
    threshold:
        Average successes per episode that must be crossed (paper: 3).
    tolerance:
        Current value of ``delta``.
    """

    initial_tolerance: float = DEFAULT_INITIAL_TOLERANCE
    min_tolerance: float = DEFAULT_MIN_TOLERANCE
    decrement: float = DEFAULT_DECREMENT
    threshold: float = DEFAULT_THRESHOLD
    task: str = "regrasping"
    success_tolerance: float = DEFAULT_INITIAL_TOLERANCE
    success_tolerance_min: float = DEFAULT_MIN_TOLERANCE
    success_tolerance_decrement: float = DEFAULT_DECREMENT
    success_threshold: float = DEFAULT_THRESHOLD
    successes_key: str = "successes"
    episode_key: str = "episode_successes"
    use_running_average: bool = True
    min_episodes: int = 1
    env: Any = None
    config: Any = None
    # ---- bookkeeping -----------------------------------------------------
    num_episodes: int = field(default=0, init=False)
    num_advancements: int = field(default=0, init=False)
    total_successes: float = field(default=0.0, init=False)
    last_successes: float = field(default=0.0, init=False)
    average_successes: float = field(default=0.0, init=False)
    history: List[Tuple[int, float]] = field(default_factory=list, init=False)

    # -- construction ------------------------------------------------------
    def __post_init__(self) -> None:
        # ``success_tolerance*`` aliases are the names used by configuration
        # objects / YAML; keep them authoritative when explicitly overridden.
        if self.success_tolerance != DEFAULT_INITIAL_TOLERANCE:
            self.initial_tolerance = float(self.success_tolerance)
        if self.success_tolerance_min != DEFAULT_MIN_TOLERANCE:
            self.min_tolerance = float(self.success_tolerance_min)
        if self.success_tolerance_decrement != DEFAULT_DECREMENT:
            self.decrement = float(self.success_tolerance_decrement)
        if self.success_threshold != DEFAULT_THRESHOLD:
            self.threshold = float(self.success_threshold)

        if self.config is not None:
            self._load_from_config(self.config)

        self.initial_tolerance = float(self.initial_tolerance)
        self.min_tolerance = float(self.min_tolerance)
        self.decrement = float(self.decrement)
        self.threshold = float(self.threshold)
        self.tolerance = float(self.initial_tolerance)

        if self.env is not None:
            self.apply(self.env)

    def _load_from_config(self, config: Any) -> None:
        """Read curriculum hyper-parameters from a config object/dict."""

        def _get(name: str, default: Any = None) -> Any:
            if isinstance(config, dict):
                return config.get(name, default)
            return getattr(config, name, default)

        for attr, names in (
            ("initial_tolerance", ("initial_tolerance", "success_tolerance")),
            ("min_tolerance", ("min_tolerance", "success_tolerance_min",
                               "min_success_tolerance")),
            ("decrement", ("decrement", "success_tolerance_decrement")),
            ("threshold", ("threshold", "success_threshold")),
            ("task", ("task",)),
            ("successes_key", ("successes_key",)),
        ):
            for name in names:
                value = _get(name, None)
                if value is not None:
                    setattr(self, attr, value)
                    break

    # -- properties --------------------------------------------------------
    @property
    def delta(self) -> float:
        """Current success tolerance ``delta`` (Appendix A notation)."""
        return float(self.tolerance)

    @property
    def success_tolerance_current(self) -> float:
        """Alias of :attr:`delta` using the env-config naming."""
        return float(self.tolerance)

    @property
    def at_minimum(self) -> bool:
        """``True`` when ``delta`` reached the 1 cm floor."""
        return self.tolerance <= self.min_tolerance + 1e-12

    @property
    def saturated(self) -> bool:
        """Alias of :attr:`at_minimum`."""
        return self.at_minimum

    @property
    def progress(self) -> float:
        """Fraction of the 7.5 cm -> 1 cm curriculum already consumed."""
        span = self.initial_tolerance - self.min_tolerance
        if span <= 0:
            return 1.0
        done = (self.initial_tolerance - self.tolerance) / span
        return max(0.0, min(1.0, done))

    # -- advancement rule --------------------------------------------------
    def should_advance(self, average_successes: Optional[float] = None) -> bool:
        """Whether the average successes/episode crosses the threshold."""
        if self.num_episodes < max(1, int(self.min_episodes)):
            return False
        if self.at_minimum:
            return False
        average = (
            self.average_successes
            if average_successes is None
            else float(average_successes)
        )
        # "crosses 3" -- strictly greater than the threshold.
        return average > self.threshold

    def advance(self, factor: Optional[float] = None) -> float:
        """Apply one 10% decrement to ``delta`` (clamped at ``min_tolerance``)."""
        factor = self.decrement if factor is None else float(factor)
        new_tolerance = max(self.min_tolerance, self.tolerance * factor)
        if new_tolerance < self.tolerance - 1e-15:
            self.tolerance = new_tolerance
            self.num_advancements += 1
            self.history.append((self.num_episodes, float(self.tolerance)))
            self.apply()
        return float(self.tolerance)

    def reset(
        self,
        tolerance: Optional[float] = None,
        reset_statistics: bool = True,
    ) -> float:
        """Reset the tolerance (and optionally the running statistics)."""
        self.tolerance = float(
            self.initial_tolerance if tolerance is None else tolerance
        )
        if reset_statistics:
            self.num_episodes = 0
            self.num_advancements = 0
            self.total_successes = 0.0
            self.average_successes = 0.0
            self.last_successes = 0.0
            self.history = []
        self.apply()
        return float(self.tolerance)

    # -- observation of the training loop ---------------------------------
    def record_episode(self, successes: Any) -> float:
        """Record the successes of one finished episode and advance if needed.

        Parameters
        ----------
        successes:
            Number of successes in the finished episode (int/float/tensor).

        Returns
        -------
        float
            The current tolerance after the update.
        """
        value = _as_float(successes, 0.0)
        self.last_successes = value
        self.total_successes += value
        self.num_episodes += 1
        self.average_successes = self.total_successes / float(self.num_episodes)
        if self.should_advance():
            self.advance()
        return float(self.tolerance)

    def record_episodes(self, successes: Sequence[Any]) -> float:
        """Record several finished episodes at once."""
        for value in successes:
            self.record_episode(value)
        return float(self.tolerance)

    def update(
        self,
        infos: Any = None,
        successes: Any = None,
        episode_successes: Any = None,
        dones: Any = None,
    ) -> float:
        """Consume environment ``infos`` and/or explicit successes.

        The method is intentionally tolerant about the information the
        environment provides:

        * ``episode_successes`` (or ``infos['episode_successes']``) is a
          *per-episode* statistic: each entry is fed to
          :meth:`record_episode`;
        * ``successes`` (or ``infos['successes']``) is a *per-step* counter:
          the number of envs that are currently in the success state.  Its
          running mean is treated as the average successes/episode estimate so
          the curriculum advances when the population mean crosses 3.

        Either source alone is sufficient; supplying both is allowed.
        """
        eps_values: Optional[Any] = episode_successes
        step_values: Optional[Any] = successes

        if infos is not None:
            info_dict = infos if isinstance(infos, dict) else {}
            if eps_values is None:
                for key in (self.episode_key, "episode_successes", "successes_episode"):
                    if key in info_dict:
                        eps_values = info_dict[key]
                        break
            if step_values is None:
                for key in (self.successes_key, "successes", "num_successes"):
                    if key in info_dict:
                        step_values = info_dict[key]
                        break
            # ``env.step`` returns ``infos``; the rollout loop places finished
            # episode statistics there too.
            if eps_values is None and "episode" in info_dict:
                episode = info_dict["episode"]
                if isinstance(episode, dict):
                    for key in ("successes", "episode_successes"):
                        if key in episode:
                            eps_values = episode[key]
                            break

        # ---- per-episode statistics -------------------------------------
        if eps_values is not None:
            if isinstance(eps_values, (int, float)):
                self.record_episode(eps_values)
            else:
                leaves = list(_iter_leaf_scalars(eps_values))
                if len(leaves) <= 1:
                    self.record_episode(leaves[0] if leaves else 0.0)
                else:
                    for leaf in leaves:
                        self.record_episode(leaf)
            return float(self.tolerance)

        # ---- per-step success counter -----------------------------------
        if step_values is not None:
            if dones is not None:
                num_done = _as_float(dones, 0.0)
                if num_done <= 0:
                    # No episode finished this step -> statistics unchanged.
                    return float(self.tolerance)
            value = _as_float(step_values, 0.0)
            self.last_successes = value
            self.num_episodes += 1
            self.total_successes += value
            self.average_successes = self.total_successes / float(self.num_episodes)
            if self.should_advance():
                self.advance()
        return float(self.tolerance)

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        """Shorthand for :meth:`update`."""
        return self.update(*args, **kwargs)

    # -- environment interaction ------------------------------------------
    def apply(self, env: Any = None) -> float:
        """Push the current tolerance into the environment."""
        target = env if env is not None else self.env
        if target is None:
            return float(self.tolerance)
        setter = getattr(target, "set_success_tolerance", None)
        if callable(setter):
            try:
                setter(float(self.tolerance))
            except Exception:  # pragma: no cover - defensive
                pass
        else:
            # Fall back to writing the attribute directly (e.g. an EnvConfig).
            for attr in ("success_tolerance", "success_tolerance_current"):
                if hasattr(target, attr):
                    try:
                        setattr(target, attr, float(self.tolerance))
                    except Exception:  # pragma: no cover - defensive
                        pass
                    break
        return float(self.tolerance)

    # -- diagnostics -------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        """Serialisable snapshot of the curriculum state."""
        return {
            "task": self.task,
            "success_tolerance": float(self.tolerance),
            "initial_tolerance": float(self.initial_tolerance),
            "min_tolerance": float(self.min_tolerance),
            "decrement": float(self.decrement),
            "threshold": float(self.threshold),
            "num_episodes": int(self.num_episodes),
            "num_advancements": int(self.num_advancements),
            "average_successes": float(self.average_successes),
            "progress": float(self.progress),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "SuccessCurriculum":
        """Restore a snapshot produced by :meth:`state_dict`."""
        if not state:
            return self
        self.tolerance = float(state.get("success_tolerance", self.tolerance))
        self.num_episodes = int(state.get("num_episodes", self.num_episodes))
        self.num_advancements = int(
            state.get("num_advancements", self.num_advancements)
        )
        self.average_successes = float(
            state.get("average_successes", self.average_successes)
        )
        self.apply()
        return self

    def metrics(self, prefix: str = "curriculum/") -> Dict[str, float]:
        """Tensorboard-friendly scalar dictionary."""
        return {
            f"{prefix}success_tolerance": float(self.tolerance),
            f"{prefix}average_successes": float(self.average_successes),
            f"{prefix}num_episodes": float(self.num_episodes),
            f"{prefix}advancements": float(self.num_advancements),
            f"{prefix}progress": float(self.progress),
        }

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"SuccessCurriculum(task={self.task!r}, delta={self.tolerance:.4f}, "
            f"avg_successes={self.average_successes:.3f}, "
            f"episodes={self.num_episodes}, at_min={self.at_minimum})"
        )


# ---------------------------------------------------------------------------
# Factories / functional helpers
# ---------------------------------------------------------------------------


def make_curriculum(
    task: str = "regrasping",
    env: Any = None,
    config: Any = None,
    initial_tolerance: Optional[float] = None,
    min_tolerance: Optional[float] = None,
    decrement: Optional[float] = None,
    threshold: Optional[float] = None,
    **kwargs: Any,
) -> SuccessCurriculum:
    """Build the curriculum for a task (Appendix A).

    Only the hard Allegro-Kuka tasks use a tolerance curriculum; for
    ShadowHand / AllegroHand a curriculum is still returned (so callers can use
    it unconditionally) but it is inert: :meth:`SuccessCurriculum.update`
    simply returns the initial tolerance because the tasks report no
    ``successes`` statistic.

    Parameters
    ----------
    task:
        Task name (``"regrasping"``, ``"throw"``, ``"reorientation"``,
        ``"shadow_hand"``, ``"allegro_hand"`` and common aliases).
    env:
        Optional environment whose ``set_success_tolerance`` is pushed to.
    config:
        Optional :class:`~sapg.utils.config.SAPGConfig`-like object whose
        curriculum fields override the defaults.
    """
    task_name = str(task or "regrasping").lower()

    # ``config`` may be passed positionally as the first argument.
    if config is None and env is not None and not hasattr(env, "step"):
        config, env = env, None

    curriculum = SuccessCurriculum(
        task=task_name,
        env=None if env is None else env,
        config=config,
        **kwargs,
    )
    if initial_tolerance is not None:
        curriculum.initial_tolerance = float(initial_tolerance)
    if min_tolerance is not None:
        curriculum.min_tolerance = float(min_tolerance)
    if decrement is not None:
        curriculum.decrement = float(decrement)
    if threshold is not None:
        curriculum.threshold = float(threshold)
    curriculum.success_threshold = curriculum.threshold
    curriculum.success_tolerance = curriculum.initial_tolerance
    curriculum.success_tolerance_min = curriculum.min_tolerance
    curriculum.success_tolerance_decrement = curriculum.decrement
    curriculum.tolerance = float(curriculum.initial_tolerance)
    if env is not None:
        curriculum.apply(env)
    return curriculum


def update_curriculum(
    curriculum: SuccessCurriculum,
    infos: Any = None,
    env: Any = None,
    **kwargs: Any,
) -> float:
    """Functional wrapper: update ``curriculum`` from ``infos`` and sync env."""
    curriculum.update(infos=infos, **kwargs)
    curriculum.apply(env)
    return float(curriculum.tolerance)


def apply_tolerance(env: Any, tolerance: float) -> float:
    """Set ``env``'s success tolerance if the interface supports it."""
    if env is None:
        return float(tolerance)
    setter = getattr(env, "set_success_tolerance", None)
    if callable(setter):
        try:
            setter(float(tolerance))
        except Exception:  # pragma: no cover - defensive
            pass
    elif hasattr(env, "success_tolerance"):
        try:
            setattr(env, "success_tolerance", float(tolerance))
        except Exception:  # pragma: no cover - defensive
            pass
    return float(tolerance)


def initial_tolerance_for(task: str) -> float:
    """Initial success tolerance (7.5 cm) for ``task``."""
    return DEFAULT_INITIAL_TOLERANCE


def min_tolerance_for(task: str) -> float:
    """Final success tolerance (1 cm) for ``task``."""
    return DEFAULT_MIN_TOLERANCE

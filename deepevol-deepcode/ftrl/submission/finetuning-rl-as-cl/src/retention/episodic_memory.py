"""Episodic memory (EM) knowledge retention for off-policy fine-tuning.

Paper: "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting
Mitigation Problem" (Wołczyk et al., 2024), Appendix C.3 ("Replay-based
methods") and Appendix B.3 (Meta-World / RoboticSequence):

    "In our experiments, we use a simple episodic memory (EM) approach along
     with the off-policy SAC algorithm. At the start of the training, we gather
     a set of trajectories from the pre-trained environment and we use them to
     populate SAC's replay buffer. In our experiments, old samples take 10% of
     the whole buffer size. Then, throughout the training we protect that part
     of the buffer, i.e. we do not allow the data from the pre-trained task to
     be overridden."

Implementation notes
--------------------
* EM has **no auxiliary loss**: unlike EWC / behavioral cloning / kickstarting
  it does not add a term to the RL objective.  The retention effect comes
  purely from the data distribution seen by the (off-policy) learner.  For API
  uniformity with the other retention mechanisms this module still exposes a
  ``penalty_loss()`` method, which returns a *differentiable zero* so that a
  generic training loop can do ``loss = rl_loss + em.penalty_loss()`` safely.
* The buffer is split into two logical regions:

      [ 0, n_prior )                       -> protected prior-task region
      [ n_prior, n_prior + n_current )     -> ring buffer for new data

  with ``n_prior = round(fraction * capacity)`` (``fraction = 0.1`` for all
  RoboticSequence experiments).  Prior-task slots are written **once** (at the
  start of training, from trajectories gathered with the pre-trained policy
  ``pi_*``) and are never overwritten afterwards; new transitions only ever
  land in the rolling region.
* ``sample`` implements the *mixed* sampler: every batch is guaranteed to
  contain at least ``max(1, round(fraction * batch_size))`` prior-task
  transitions (as long as prior data exists), the rest comes from the current
  region -- this approximates the "perfectly mixed i.i.d. data distribution"
  discussed in the paper.
* The module is deliberately storage-agnostic: transitions can be python
  dicts of numpy arrays, dicts of torch tensors, tuples, or any object
  supporting the same API.  ``stack_batch`` converts a list of transitions
  into a batched dict of torch tensors when needed.
"""

from __future__ import annotations

import math
import random as _random
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - torch is required at runtime, optional at import
    import torch
    from torch import Tensor

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Tensor = Any  # type: ignore
    _HAS_TORCH = False

try:  # pragma: no cover - numpy is a soft dependency
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None  # type: ignore


__all__ = [
    "EpisodicMemory",
    "EpisodicMemoryBuffer",
    "PriorTaskBuffer",
    "MixedBatchSampler",
    "EMReplayBuffer",
    "stack_batch",
    "transition_field",
    "collect_trajectories",
    "em_fraction_for",
    "em_capacity_for",
    "DEFAULT_EM_FRACTION",
    "DEFAULT_EM_CAPACITY",
    "ROBOTIC_SEQUENCE_BUFFER_SIZE",
]


# --------------------------------------------------------------------------------------
# Constants (Appendix C.3 / Table 3 and RoboticSequence configuration)
# --------------------------------------------------------------------------------------

#: Fraction of the replay buffer reserved for pre-training data (Appendix C.3: 10%).
DEFAULT_EM_FRACTION = 0.1

#: Default replay buffer capacity.  Table 3 lists the Meta-World replay buffer
#: size used together with the 10% protected fraction; the same default is used
#: for the (unused) on-policy environments for API completeness.
DEFAULT_EM_CAPACITY = 1_000_000

#: Explicit name kept for readability in configs / other modules.
ROBOTIC_SEQUENCE_BUFFER_SIZE = DEFAULT_EM_CAPACITY

#: Paper convention: old (pre-training) samples take 10% of the buffer.
DEFAULT_EM_FRACTION_BY_ENV: Dict[str, float] = {
    "robotic_sequence": 0.1,
    "metaworld": 0.1,
    "montezuma": 0.1,
    "nethack": 0.1,
}


def em_fraction_for(env_name: str) -> float:
    """Protected-buffer fraction for a given environment (Appendix C.3: 10%)."""
    if env_name is None:
        return DEFAULT_EM_FRACTION
    key = str(env_name).strip().lower().replace("-", "_").replace(" ", "_")
    if key in DEFAULT_EM_FRACTION_BY_ENV:
        return DEFAULT_EM_FRACTION_BY_ENV[key]
    for name, value in DEFAULT_EM_FRACTION_BY_ENV.items():
        if name in key or key in name:
            return value
    return DEFAULT_EM_FRACTION


def em_capacity_for(env_name: str) -> int:
    """Replay-buffer capacity used for episodic memory (Table 3 / defaults)."""
    return DEFAULT_EM_CAPACITY


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _as_generator(generator: Any = None, seed: Optional[int] = None):
    """Return a numpy Generator when possible, else a ``random.Random``.

    The return value only needs to expose ``integers``/``randint`` and
    ``choice``; ``_randint``/``_choice`` below abstract over both.
    """
    if generator is not None:
        return generator
    if _np is not None:
        return _np.random.default_rng(seed)
    return _random.Random(seed)


def _randint(rng: Any, low: int, high: int) -> int:
    """``rng.integers(low, high)`` for numpy, ``randint`` for ``random``."""
    if hasattr(rng, "integers"):
        return int(rng.integers(low, high))
    if hasattr(rng, "randint"):
        return int(rng.randint(low, high))
    return int(_random.randint(low, high))


def _choice(rng: Any, indices: Sequence[int], size: int, replace: bool = True) -> List[int]:
    """Sample ``size`` elements from ``indices`` with either RNG flavour."""
    if len(indices) == 0:
        return []
    size = int(size)
    if size <= 0:
        return []
    if not replace:
        size = min(size, len(indices))
    if hasattr(rng, "choice"):
        try:
            out = rng.choice(list(indices), size=size, replace=replace)
        except TypeError:  # ``random.Random.choice`` only takes one element
            out = [rng.choice(list(indices)) for _ in range(size)]
        if _np is not None and isinstance(out, _np.ndarray):
            return [int(x) for x in out.tolist()]
        if isinstance(out, (list, tuple)):
            return [int(x) for x in out]
        return [int(out)]
    # fall back to plain python randomness
    pool = list(indices)
    if replace:
        return [_random.choice(pool) for _ in range(size)]
    return _random.sample(pool, size)


def _to_tensor(value: Any) -> Any:
    """Best-effort conversion of a value (possibly a batch of them) to a tensor."""
    if not _HAS_TORCH:
        return value
    if isinstance(value, Tensor):
        return value
    if _np is not None and isinstance(value, _np.ndarray):
        return torch.as_tensor(value)
    return torch.as_tensor(value)


def transition_field(transition: Any, key: str, default: Any = None) -> Any:
    """Fetch ``key`` from a transition stored as mapping / namedtuple / attr."""
    if transition is None:
        return default
    if isinstance(transition, Mapping):
        if key in transition:
            return transition[key]
        return default
    if hasattr(transition, key):
        return getattr(transition, key)
    if isinstance(transition, (tuple, list)):
        # (obs, action, reward, next_obs, done) convention
        order = ("obs", "action", "reward", "next_obs", "done")
        if key in order and len(transition) > order.index(key):
            return transition[order.index(key)]
    return default


def stack_batch(
    batch: Sequence[Any],
    device: Any = None,
    dtype: Any = None,
    fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Stack a list of transitions into a dict of batched tensors.

    Keys are discovered from the first transition (mappings) or from the
    ``(obs, action, reward, next_obs, done)`` convention for tuples.  Values
    that cannot be converted to tensors are returned as plain lists.
    """
    if len(batch) == 0:
        return {}

    first = batch[0]
    if isinstance(first, Mapping):
        keys = list(fields) if fields is not None else list(first.keys())
    else:
        keys = (
            list(fields)
            if fields is not None
            else ["obs", "action", "reward", "next_obs", "done"]
        )

    out: Dict[str, Any] = {}
    for key in keys:
        values = [transition_field(t, key) for t in batch]
        if any(v is None for v in values):
            out[key] = values
            continue
        try:
            if _HAS_TORCH and isinstance(values[0], Tensor):
                out[key] = torch.stack([v if isinstance(v, Tensor) else torch.as_tensor(v) for v in values])
            else:
                out[key] = _to_tensor(values)
            if dtype is not None and _HAS_TORCH and isinstance(out[key], Tensor):
                if out[key].is_floating_point():
                    out[key] = out[key].to(dtype)
            if device is not None and _HAS_TORCH and isinstance(out[key], Tensor):
                out[key] = out[key].to(device)
        except Exception:
            out[key] = values
    return out


def collect_trajectories(
    env: Any,
    policy: Any = None,
    num_trajectories: int = 10,
    max_steps: Optional[int] = None,
    initial_states: Optional[Iterable[Any]] = None,
    action_fn: Optional[Callable[[Any, Any], Any]] = None,
    seed: Optional[int] = None,
    deterministic: bool = False,
    progress_fn: Optional[Callable[[Any], bool]] = None,
) -> List[List[Any]]:
    """Gather trajectories from the *pre-trained* environment using ``pi_*``.

    Parameters
    ----------
    env:
        A gym/gymnasium-style environment exposing ``reset()``/``step()``.
    policy:
        The pre-trained actor ``pi_*``.  Accepted interfaces: a callable
        ``policy(obs) -> action``, an object exposing ``act(obs)`` and/or
        ``distribution(obs)`` (torch ``Distribution`` with ``sample()``).
    num_trajectories:
        Number of episodes to collect.
    max_steps:
        Optional episode horizon; taken from ``env.spec.max_episode_steps`` or
        ``env.time_limit`` / ``env.max_steps`` when left as ``None``.
    initial_states:
        Optional iterable of states the trajectories should start from (e.g.
        sampled pre-training states ``S_BC``).  Consumed cyclically.
    progress_fn:
        Optional callable deciding when an episode should end (e.g. a
        "success" test) in addition to ``done``.

    Returns
    -------
    list[list[transition]] where a transition is a dict with the keys
    ``obs``, ``action``, ``reward``, ``next_obs``, ``done``.
    """
    trajectories: List[List[Any]] = []
    if env is None:
        return trajectories

    starts = list(initial_states) if initial_states is not None else []

    def _resolve_horizon() -> Optional[int]:
        if max_steps is not None:
            return int(max_steps)
        for attr in ("max_steps", "time_limit", "T"):
            value = getattr(env, attr, None)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
        spec = getattr(env, "spec", None)
        value = getattr(spec, "max_episode_steps", None)
        if value:
            return int(value)
        return None

    horizon = _resolve_horizon()
    rng = _random.Random(seed)

    for episode in range(int(num_trajectories)):
        if starts:
            start_state = starts[episode % len(starts)]
            try:
                out = env.reset()
                obs = out[0] if isinstance(out, tuple) and len(out) == 2 else out
                if hasattr(env, "set_state"):
                    env.set_state(start_state)
                    obs = start_state
            except Exception:
                obs = start_state
        else:
            out = env.reset()
            obs = out[0] if isinstance(out, tuple) and len(out) == 2 else out

        transitions: List[Any] = []
        t = 0
        done = False
        while not done and (horizon is None or t < horizon):
            action = _select_action(policy, obs, deterministic=deterministic, rng=rng)
            step_out = env.step(action)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
                done = bool(terminated) or bool(truncated)
            elif isinstance(step_out, tuple) and len(step_out) == 4:
                next_obs, reward, done, info = step_out
                done = bool(done)
            else:  # pragma: no cover - exotic env API
                next_obs, reward, done = step_out, 0.0, False
                info = {}
            if progress_fn is not None and not done:
                try:
                    done = bool(progress_fn(info if isinstance(info, Mapping) else {}))
                except Exception:
                    pass
            transitions.append(
                {
                    "obs": _detach_copy(obs),
                    "action": _detach_copy(action),
                    "reward": float(reward) if not _HAS_TORCH or not isinstance(reward, Tensor) else float(reward.item()),
                    "next_obs": _detach_copy(next_obs),
                    "done": bool(done),
                }
            )
            obs = next_obs
            t += 1

        trajectories.append(transitions)

    return trajectories


def _detach_copy(value: Any) -> Any:
    """Detach / copy a value so stored transitions never alias live tensors."""
    if _HAS_TORCH and isinstance(value, Tensor):
        return value.detach().clone()
    if _np is not None and isinstance(value, _np.ndarray):
        return value.copy()
    return value


def _select_action(policy: Any, obs: Any, deterministic: bool = False, rng: Any = None) -> Any:
    """Query ``pi_*(a|s)`` through a variety of duck-typed interfaces."""
    if policy is None:
        return 0
    if hasattr(policy, "act"):
        try:
            return policy.act(obs, deterministic=deterministic)
        except TypeError:
            return policy.act(obs)
    if hasattr(policy, "sample_action"):
        return policy.sample_action(obs)
    if hasattr(policy, "distribution"):
        try:
            dist = policy.distribution(obs)
        except Exception:
            dist = None
        if dist is not None:
            if deterministic:
                mean = getattr(dist, "mean", None)
                if mean is not None:
                    return mean
                mode = getattr(dist, "mode", None)
                if mode is not None:
                    return mode
            return dist.sample() if hasattr(dist, "sample") else dist.rsample()
    if callable(policy):
        out = policy(obs)
        for key in ("action", "actions", "sample"):
            if isinstance(out, Mapping) and key in out:
                return out[key]
            if hasattr(out, key):
                return getattr(out, key)
        return out
    raise TypeError(f"Cannot select an action with policy of type {type(policy)}")


# --------------------------------------------------------------------------------------
# Mixed batch sampler
# --------------------------------------------------------------------------------------


class MixedBatchSampler:
    """Samples batches mixing *protected* prior-task data with current data.

    ``prior_fraction`` of every batch (at least one sample) is drawn from the
    protected region populated at the start of training with ``pi_*``
    trajectories; the remaining samples come from the rolling region holding
    the current task's data.  When no prior data has been stored yet, or the
    current region is empty, the sampler gracefully falls back to whatever data
    is available.
    """

    def __init__(
        self,
        num_prior: int = 0,
        num_current: int = 0,
        prior_fraction: float = DEFAULT_EM_FRACTION,
        guarantee_prior: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.num_prior = int(num_prior)
        self.num_current = int(num_current)
        self.prior_fraction = float(prior_fraction)
        self.guarantee_prior = bool(guarantee_prior)
        self.generator = _as_generator(None, seed)

    # -- bookkeeping -------------------------------------------------------------
    def set_sizes(self, num_prior: int, num_current: int) -> "MixedBatchSampler":
        self.num_prior = int(num_prior)
        self.num_current = int(num_current)
        return self

    @property
    def total(self) -> int:
        return self.num_prior + self.num_current

    def prior_count(self, batch_size: int) -> int:
        """Number of prior-task samples included in one batch."""
        if self.num_prior == 0:
            return 0
        n_prior = int(round(self.prior_fraction * int(batch_size)))
        if self.guarantee_prior:
            n_prior = max(1, n_prior)
        n_prior = min(n_prior, int(batch_size))
        return max(0, n_prior)

    # -- sampling ---------------------------------------------------------------
    def sample_indices(self, batch_size: int, generator: Any = None) -> List[int]:
        """Return a list of *logical* buffer indices for one mixed batch."""
        rng = generator if generator is not None else self.generator
        batch_size = int(batch_size)
        if batch_size <= 0:
            return []
        if self.total == 0:
            return []

        n_prior = self.prior_count(batch_size)
        indices: List[int] = []
        if n_prior > 0:
            indices.extend(_choice(rng, list(range(self.num_prior)), n_prior, replace=self.num_prior < n_prior))
        rest = batch_size - len(indices)
        if rest > 0 and self.num_current > 0:
            current_pool = list(range(self.num_prior, self.num_prior + self.num_current))
            indices.extend(
                _choice(rng, current_pool, rest, replace=self.num_current < rest)
            )
        # top up from whichever region has data if one of them was empty
        if len(indices) < batch_size:
            pool = list(range(self.total))
            indices.extend(
                _choice(rng, pool, batch_size - len(indices), replace=self.total < batch_size)
            )
        return indices[:batch_size]


# --------------------------------------------------------------------------------------
# Prior-task store (protected region)
# --------------------------------------------------------------------------------------


class PriorTaskBuffer:
    """Storage for trajectories gathered with the pre-trained policy ``pi_*``.

    The store has a fixed capacity (``10%`` of the replay buffer in the paper)
    and, once full, **never** overwrites its content -- this is the
    "protection" discussed in Appendix C.3.  Extra transitions arriving after
    saturation are dropped (optionally counted in ``dropped``).
    """

    def __init__(
        self,
        capacity: int,
        device: Any = None,
        seed: Optional[int] = None,
        store_teacher: bool = False,
    ) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.store_teacher = bool(store_teacher)
        self.data: List[Any] = []
        self.dropped = 0
        self.filled = False
        self.generator = _as_generator(None, seed)

    # -- population -------------------------------------------------------------
    def add(self, transition: Any) -> bool:
        """Add a single prior-task transition to the protected region."""
        if len(self.data) >= self.capacity:
            self.dropped += 1
            self.filled = True
            return False
        self.data.append(_move(_detach_copy(transition), self.device))
        if len(self.data) >= self.capacity:
            self.filled = True
        return True

    def extend(self, transitions: Iterable[Any]) -> int:
        added = 0
        for transition in transitions:
            if self.add(transition):
                added += 1
        return added

    def add_trajectories(self, trajectories: Iterable[Iterable[Any]]) -> int:
        """Flatten nested trajectories (list of episode-transition lists)."""
        added = 0
        for trajectory in trajectories:
            added += self.extend(trajectory)
        return added

    def populate(
        self,
        transitions: Optional[Iterable[Any]] = None,
        env: Any = None,
        policy: Any = None,
        num_trajectories: int = 0,
        max_steps: Optional[int] = None,
        generator: Any = None,
        seed: Optional[int] = None,
    ) -> int:
        """Fill the protected region, from given transitions or by rollouts.

        Either ``transitions`` (an iterable of transitions or of trajectories)
        is provided, or ``env``/``policy`` are used to gather
        ``num_trajectories`` fresh episodes with the pre-trained actor.
        """
        if transitions is None:
            if env is None or num_trajectories <= 0:
                return 0
            trajectories = collect_trajectories(
                env,
                policy=policy,
                num_trajectories=num_trajectories,
                max_steps=max_steps,
                seed=seed,
            )
            return self.add_trajectories(trajectories)

        added = 0
        pending: List[Any] = []
        for item in transitions:
            if _looks_like_trajectory(item):
                if pending:
                    added += self.extend(pending)
                    pending = []
                added += self.extend(item)
            else:
                pending.append(item)
        if pending:
            added += self.extend(pending)
        return added

    # -- access -----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data)

    @property
    def is_full(self) -> bool:
        return len(self.data) >= self.capacity

    def sample(self, size: int, generator: Any = None) -> List[Any]:
        rng = generator if generator is not None else self.generator
        if not self.data:
            return []
        indices = _choice(rng, list(range(len(self.data))), int(size), replace=len(self.data) < int(size))
        return [self.data[i] for i in indices]

    def __getitem__(self, index: int) -> Any:
        return self.data[index]

    def clear(self) -> None:
        self.data = []
        self.dropped = 0
        self.filled = False

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "data": self.data,
            "dropped": self.dropped,
            "filled": self.filled,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.capacity = int(state.get("capacity", self.capacity))
        self.data = list(state.get("data", []))
        self.dropped = int(state.get("dropped", 0))
        self.filled = bool(state.get("filled", self.is_full))


def _looks_like_trajectory(item: Any) -> bool:
    """Heuristic: is ``item`` a sequence of transitions (rather than one)?"""
    if isinstance(item, (list, tuple)) and len(item) > 0:
        first = item[0]
        if isinstance(first, Mapping):
            return True
        if isinstance(first, (list, tuple)) and not _is_scalar_like(first):
            return True
        if hasattr(first, "obs") and hasattr(first, "action"):
            return True
    return False


def _is_scalar_like(value: Any) -> bool:
    if isinstance(value, (int, float, bool, complex)):
        return True
    if _np is not None and isinstance(value, _np.ndarray):
        return value.ndim == 0
    if _HAS_TORCH and isinstance(value, Tensor):
        return value.dim() == 0
    return False


def _move(value: Any, device: Any) -> Any:
    if device is None or not _HAS_TORCH:
        return value
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {k: _move(v, device) for k, v in value.items()}
    return value


# --------------------------------------------------------------------------------------
# Rolling region (current task data)
# --------------------------------------------------------------------------------------


class EpisodicMemoryBuffer:
    """Replay buffer whose first ``fraction`` of slots are protected.

    Layout (logical indices)::

        [0, n_prior)                        protected prior-task transitions
        [n_prior, n_prior + num_current)    rolling current-task transitions

    New transitions are appended to the rolling region in FIFO order; the
    protected region is written only through :meth:`populate_prior` /
    :meth:`set_prior_data` and can never be overwritten by :meth:`add`.
    """

    def __init__(
        self,
        capacity: int = DEFAULT_EM_CAPACITY,
        fraction: float = DEFAULT_EM_FRACTION,
        device: Any = None,
        seed: Optional[int] = None,
        env_name: Optional[str] = None,
    ) -> None:
        self.capacity = int(capacity)
        self.fraction = float(fraction)
        self.device = device
        self.env_name = env_name
        self.seed = seed

        n_prior = int(round(self.fraction * self.capacity))
        n_prior = max(0, min(n_prior, self.capacity))
        self.n_prior = n_prior
        self.n_current_capacity = self.capacity - n_prior

        self.prior = PriorTaskBuffer(n_prior, device=device, seed=seed)
        self.current: List[Any] = []
        self.position = 0  # ring pointer inside the rolling region
        self.generator = _as_generator(None, seed)

    # -- geometry ---------------------------------------------------------------
    @property
    def protected_indices(self) -> List[int]:
        """Logical indices of the protected (never overwritten) slots."""
        return list(range(self.n_prior))

    @property
    def protected_mask(self) -> List[bool]:
        return [True] * self.n_prior + [False] * len(self.current)

    def is_protected(self, index: int) -> bool:
        """Whether logical buffer index ``index`` belongs to the protected part."""
        return 0 <= int(index) < self.n_prior

    @property
    def protected_count(self) -> int:
        return len(self.prior)

    @property
    def fraction_used(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return self.protected_count / float(self.capacity)

    def __len__(self) -> int:
        return self.protected_count + len(self.current)

    # -- insertion --------------------------------------------------------------
    def add(self, transition: Any, protected: bool = False) -> bool:
        """Insert a transition.

        ``protected=False`` (default, used during fine-tuning) routes the
        transition to the rolling region and **never** touches prior data.
        ``protected=True`` routes it to the protected region, which is only
        allowed to be written before/while it is being populated.
        """
        transition = _move(_detach_copy(transition), self.device)
        if protected:
            return self.prior.add(transition)
        if self.n_current_capacity <= 0:
            return False
        if len(self.current) < self.n_current_capacity:
            self.current.append(transition)
        else:
            # FIFO overwrite, strictly inside the rolling region.
            self.current[self.position % self.n_current_capacity] = transition
        self.position = (self.position + 1) % max(1, self.n_current_capacity)
        return True

    def add_batch(self, transitions: Iterable[Any], protected: bool = False) -> int:
        added = 0
        for transition in transitions:
            if self.add(transition, protected=protected):
                added += 1
        return added

    def add_trajectories(
        self,
        trajectories: Iterable[Iterable[Any]],
        protected: bool = False,
    ) -> int:
        added = 0
        for trajectory in trajectories:
            added += self.add_batch(trajectory, protected=protected)
        return added

    # -- prior-task population --------------------------------------------------
    def populate_prior(
        self,
        transitions: Optional[Iterable[Any]] = None,
        env: Any = None,
        policy: Any = None,
        num_trajectories: int = 0,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> int:
        """Gather ``pi_*`` trajectories from the pre-trained environment.

        Mirrors Appendix C.3: "At the start of the training, we gather a set of
        trajectories from the pre-trained environment and we use them to
        populate SAC's replay buffer."
        """
        return self.prior.populate(
            transitions=transitions,
            env=env,
            policy=policy,
            num_trajectories=num_trajectories,
            max_steps=max_steps,
            seed=seed,
        )

    def set_prior_data(self, transitions: Iterable[Any]) -> int:
        """Directly set the protected region (e.g. from a saved buffer)."""
        self.prior.clear()
        return self.prior.extend(transitions)

    @property
    def prior_fraction(self) -> float:
        return self.fraction

    # -- sampling ---------------------------------------------------------------
    def sample_indices(self, batch_size: int, generator: Any = None) -> List[int]:
        """Mixed sampling: batch always contains prior-task transitions."""
        sampler = MixedBatchSampler(
            num_prior=self.protected_count,
            num_current=len(self.current),
            prior_fraction=self.fraction,
            seed=self.seed,
        )
        return sampler.sample_indices(batch_size, generator=generator)

    def sample(
        self,
        batch_size: int,
        generator: Any = None,
        device: Any = None,
        fields: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Sample a mixed batch and return it as a dict of tensors."""
        rng = generator if generator is not None else self.generator
        indices = self.sample_indices(batch_size, generator=rng)
        transitions = [self[i] for i in indices]
        batch = stack_batch(transitions, device=device if device is not None else self.device, fields=fields)
        batch["indices"] = indices
        batch["protected"] = [self.is_protected(i) for i in indices]
        batch["num_prior_in_batch"] = int(sum(batch["protected"]))
        return batch

    def __getitem__(self, index: int) -> Any:
        index = int(index)
        if index < self.n_prior:
            return self.prior[index]
        current_index = index - self.n_prior
        return self.current[current_index]

    # -- maintenance ------------------------------------------------------------
    def clear(self, keep_prior: bool = False) -> None:
        self.current = []
        self.position = 0
        if not keep_prior:
            self.prior.clear()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "fraction": self.fraction,
            "n_prior": self.n_prior,
            "position": self.position,
            "current": self.current,
            "prior": self.prior.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.capacity = int(state.get("capacity", self.capacity))
        self.fraction = float(state.get("fraction", self.fraction))
        self.n_prior = int(state.get("n_prior", self.n_prior))
        self.n_current_capacity = self.capacity - self.n_prior
        self.position = int(state.get("position", 0))
        self.current = list(state.get("current", []))
        if "prior" in state:
            self.prior.load_state_dict(state["prior"])
        self.generator = _as_generator(None, self.seed)


#: Alias -- EM is applied to an off-policy replay buffer.
EMReplayBuffer = EpisodicMemoryBuffer


# --------------------------------------------------------------------------------------
# EpisodicMemory regulariser (uniform API with EWC / BC / KS)
# --------------------------------------------------------------------------------------


class EpisodicMemory:
    """Episodic-memory retention mechanism (Appendix C.3).

    EM does **not** change the objective: there is no auxiliary loss, only a
    protected fraction of the (off-policy) replay buffer holding pre-training
    transitions produced by ``pi_*``.  This class therefore owns the buffer and
    exposes the *same* interface as :class:`~src.retention.ewc.EWC`,
    :class:`~src.retention.behavioral_cloning.BehavioralCloning` and
    :class:`~src.retention.kickstarting.Kickstarting`, so training runners can
    treat all four mechanisms uniformly.

    Parameters
    ----------
    actor:
        The fine-tuned policy.  Kept for API symmetry (EM needs no teacher
        forward pass); may be ``None``.
    teacher:
        Pre-trained policy ``pi_*`` used to gather the prior-task trajectories
        that populate the buffer.
    buffer:
        Optional pre-built :class:`EpisodicMemoryBuffer` (or any object with an
        ``add``/``sample`` interface).
    capacity, fraction:
        Replay buffer size and protected fraction (10% per Appendix C.3).
    env_name:
        Selects the defaults (``"robotic_sequence"``).  ``capacity``/
        ``fraction`` default to the environment values when not given.
    """

    #: EM has no auxiliary loss -- kept for documentation purposes.
    HAS_AUXILIARY_LOSS = False

    def __init__(
        self,
        actor: Any = None,
        teacher: Any = None,
        buffer: Any = None,
        capacity: Optional[int] = None,
        fraction: Optional[float] = None,
        batch_size: int = 128,
        env_name: Optional[str] = None,
        device: Any = None,
        seed: Optional[int] = None,
        prefill_fraction: float = 1.0,
        name: str = "em",
    ) -> None:
        self.actor = actor
        self.teacher = teacher
        self.env_name = env_name
        self.batch_size = int(batch_size)
        self.device = device
        self.seed = seed
        self.name = name
        self.prefill_fraction = float(prefill_fraction)

        self.capacity = int(capacity) if capacity is not None else em_capacity_for(env_name or "")
        self.fraction = float(fraction) if fraction is not None else em_fraction_for(env_name or "")

        if buffer is not None:
            self.buffer = buffer
        else:
            self.buffer = EpisodicMemoryBuffer(
                capacity=self.capacity,
                fraction=self.fraction,
                device=device,
                seed=seed,
                env_name=env_name,
            )

        self.num_trajectories = 0
        self._updates = 0

    # -- bookkeeping ------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """EM is active once the protected region holds pre-training data."""
        return self.protected_count > 0

    @property
    def protected_count(self) -> int:
        return int(getattr(self.buffer, "protected_count", 0))

    @property
    def protected_indices(self) -> List[int]:
        return list(getattr(self.buffer, "protected_indices", []))

    def is_protected(self, index: int) -> bool:
        fn = getattr(self.buffer, "is_protected", None)
        if callable(fn):
            return bool(fn(index))
        return 0 <= int(index) < self.protected_count

    @property
    def target_prior_per_batch(self) -> int:
        return max(1, int(round(self.fraction * self.batch_size)))

    # -- filling the protected region -------------------------------------------
    def populate(
        self,
        transitions: Optional[Iterable[Any]] = None,
        env: Any = None,
        policy: Any = None,
        num_trajectories: Optional[int] = None,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> int:
        """Populate the protected region with ``pi_*`` data.

        Called once at the start of fine-tuning.  If ``env``/``policy`` are
        given, ``num_trajectories`` rollouts are gathered (as many as needed to
        fill ``prefill_fraction`` of the protected region when unspecified);
        otherwise ``transitions`` are used directly.
        """
        policy = policy if policy is not None else self.teacher
        if num_trajectories is None:
            num_trajectories = self.num_trajectories if self.num_trajectories > 0 else 0
        fn = getattr(self.buffer, "populate_prior", None)
        if callable(fn):
            added = fn(
                transitions=transitions,
                env=env,
                policy=policy,
                num_trajectories=int(num_trajectories or 0),
                max_steps=max_steps,
                seed=seed if seed is not None else self.seed,
            )
        else:
            added = self.buffer.add_batch(transitions or [], protected=True)
        return int(added)

    def populate_from_trajectories(self, trajectories: Iterable[Iterable[Any]]) -> int:
        """Fill the protected region from pre-collected episode trajectories."""
        fn = getattr(self.buffer, "add_trajectories", None)
        if callable(fn):
            return int(fn(trajectories, protected=True))
        return int(self.buffer.add_batch([t for traj in trajectories for t in traj], protected=True))

    def attach_buffer(self, buffer: Any) -> Any:
        self.buffer = buffer
        return self.buffer

    #: Alias used by some runners.
    set_buffer = attach_buffer

    # -- data plumbing ----------------------------------------------------------
    def add(self, transition: Any, protected: bool = False) -> bool:
        """Add a transition from the *current* task (never overwrites prior data)."""
        return bool(self.buffer.add(transition, protected=protected))

    def add_batch(self, transitions: Iterable[Any], protected: bool = False) -> int:
        return int(self.buffer.add_batch(transitions, protected=protected))

    def sample(self, batch_size: Optional[int] = None, generator: Any = None, device: Any = None):
        """Sample a **mixed** batch (guaranteed to contain prior transitions)."""
        return self.buffer.sample(
            int(batch_size if batch_size is not None else self.batch_size),
            generator=generator,
            device=device,
        )

    def sample_indices(self, batch_size: Optional[int] = None, generator: Any = None) -> List[int]:
        return self.buffer.sample_indices(
            int(batch_size if batch_size is not None else self.batch_size), generator=generator
        )

    def __len__(self) -> int:
        return len(self.buffer)

    # -- loss API (EM adds nothing to the objective) -----------------------------
    def penalty(self, *args: Any, **kwargs: Any) -> Any:
        """Return a differentiable zero: EM has no auxiliary loss.

        The returned tensor depends on the actor parameters so that
        ``total_loss = rl_loss + em.penalty()`` keeps ``.backward()`` valid.
        """
        return _zero_like_actor(self.actor)

    #: Names used by the other retention modules; all return the same zero.
    penalty_loss = penalty
    loss = penalty
    auxiliary_loss = penalty

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.penalty(*args, **kwargs)

    def step(self, n: int = 1) -> None:
        """Advance the internal counter (no decay: EM is not schedule-based)."""
        self._updates += int(n)

    @property
    def step_count(self) -> int:
        return self._updates

    # -- configuration ----------------------------------------------------------
    def configure(self, cfg: Any) -> "EpisodicMemory":
        """Apply values from a config object / mapping (``cfg.retention.em``)."""
        get = _cfg_getter(cfg)

        capacity = get("capacity", get("buffer_size", None))
        if capacity is not None:
            self.capacity = int(capacity)
        fraction = get("fraction", get("protected_fraction", None))
        if fraction is not None:
            self.fraction = float(fraction)
        batch_size = get("batch_size", None)
        if batch_size is not None:
            self.batch_size = int(batch_size)
        prefill = get("prefill_fraction", None)
        if prefill is not None:
            self.prefill_fraction = float(prefill)
        num_traj = get("num_trajectories", None)
        if num_traj is not None:
            self.num_trajectories = int(num_traj)

        if isinstance(self.buffer, EpisodicMemoryBuffer):
            n_prior = int(round(self.fraction * self.capacity))
            if n_prior != self.buffer.n_prior or self.capacity != self.buffer.capacity:
                prior_data = list(self.buffer.prior.data)
                self.buffer = EpisodicMemoryBuffer(
                    capacity=self.capacity,
                    fraction=self.fraction,
                    device=self.device,
                    seed=self.seed,
                    env_name=self.env_name,
                )
                self.buffer.set_prior_data(prior_data)
        return self

    # -- persistence ------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "capacity": self.capacity,
            "fraction": self.fraction,
            "batch_size": self.batch_size,
            "updates": self._updates,
            "num_trajectories": self.num_trajectories,
        }
        if hasattr(self.buffer, "state_dict"):
            d["buffer"] = self.buffer.state_dict()
        return d

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.name = state.get("name", self.name)
        self.capacity = int(state.get("capacity", self.capacity))
        self.fraction = float(state.get("fraction", self.fraction))
        self.batch_size = int(state.get("batch_size", self.batch_size))
        self._updates = int(state.get("updates", 0))
        self.num_trajectories = int(state.get("num_trajectories", 0))
        if "buffer" in state and hasattr(self.buffer, "load_state_dict"):
            self.buffer.load_state_dict(state["buffer"])

    # -- reporting --------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"name={self.name}, capacity={self.capacity}, fraction={self.fraction}, "
            f"protected={self.protected_count}, batch_size={self.batch_size}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}({self.extra_repr()})"

    def describe(self) -> Dict[str, Any]:
        """Small summary used in logs."""
        return {
            "em/capacity": self.capacity,
            "em/fraction": self.fraction,
            "em/protected_samples": self.protected_count,
            "em/buffer_size": len(self),
            "em/batches_with_prior": self.target_prior_per_batch,
        }


# --------------------------------------------------------------------------------------
# Functional helpers
# --------------------------------------------------------------------------------------


def _zero_like_actor(actor: Any) -> Any:
    """Differentiable zero referencing the actor's params (0 if no actor given)."""
    if not _HAS_TORCH:
        return 0.0
    params = None
    if actor is not None:
        try:
            params = list(actor.parameters())
        except Exception:
            params = None
    if not params:
        return torch.zeros((), requires_grad=True)
    try:
        return sum(p.sum() * 0.0 for p in params)
    except Exception:  # pragma: no cover
        return torch.zeros((), requires_grad=True)


def _cfg_getter(cfg: Any) -> Callable[..., Any]:
    """Return a ``get(key, default)`` callable for mappings / config objects."""

    def get(key: str, default: Any = None) -> Any:
        if cfg is None:
            return default
        if isinstance(cfg, Mapping):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    return get


#: Alias for API symmetry with ``KickstartingLoss`` / ``EWC``.
EM = EpisodicMemory

"""Critical-state selection for RICE (Stage 1 -> Stage 2 bridge).

Paper reference: Cheng et al., "RICE: A Refining Scheme for Reinforcement Learning
with Explanation", ICML 2024 (PMLR 235), Sec. 3.3 and Algorithm 2.

The mask network ``\\tilde{\\pi}_\\theta`` outputs a binary action
``a_t^m in {0, 1}``; step-level state importance is defined as the probability of
the mask outputting ``0`` ("keep"), i.e. ``I(s_t) = P(a_t^m = 0 | s_t)``.

Algorithm 2 ("Constructing Mixed Initial State Distribution") uses this scoring
rule to *pinpoint the most important state within the episode*::

    Run pi to obtain a trajectory tau of length K
    Identify the most critical state s_t in tau via state mask \tilde{\pi}
    Set the initial state s_0 <- s_t

This module implements exactly that step:

* :func:`roll_trajectory`        -- run the pre-trained policy ``pi`` for length ``K``.
* :func:`identify_critical_state`-- score every visited state and take the argmax
  (ties broken by the earliest step; c.f. ``argmax_importance`` in ``importance.py``).
* :func:`top_k_critical_states`  -- the ranked top-K frontier states used by the
  sliding-window fidelity evaluation (Sec. 4.1: "choose the window with the highest
  average importance score").

The resulting :class:`CriticalState` is intentionally *restorable*: it carries both
the raw observation and (when available) the simulator state captured by
``rice.envs.reset_wrapper`` so that Algorithm 2 can ``reset`` directly to it instead
of replaying actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from rice.explanation.importance import (
    DEFAULT_BATCH_SIZE,
    ImportanceScorer,
    argmax_importance,
    attach_scores_to_wrapper,
    extract_observations,
    rank_states,
    score_trajectory,
    summarize_importance,
    top_k_indices,
)
from rice.explanation.mask_network import flatten_observation
from rice.utils.logging import get_logger

__all__ = [
    "CriticalState",
    "TrajectoryRollout",
    "CriticalStateSelector",
    "roll_trajectory",
    "most_critical_index",
    "critical_state_from_rollout",
    "identify_critical_state",
    "top_k_critical_states",
    "rank_trajectory",
    "select_critical_states",
    "policy_action",
    "default_k",
    "attach_critical_state",
    "describe_critical_state",
    "DEFAULT_K",
]

logger = get_logger("rice.critical_state")

#: Default trajectory length ``K`` used by Algorithm 2 when the environment does
#: not advertise a horizon.  The plan's ambiguity rule: fall back to the episode
#: horizon ``T`` (MuJoCo default = 1000 steps).
DEFAULT_K = 1000


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------
@dataclass
class TrajectoryRollout:
    """A length-``K`` trajectory ``tau`` collected by running the pre-trained policy.

    Mirrors the quantities Algorithm 2 needs: the visited states (for mask scoring)
    plus the actions/state snapshot (for a possible Go-Explore style restore).
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: Optional[np.ndarray] = None
    dones: Optional[np.ndarray] = None
    infos: List[Dict[str, Any]] = field(default_factory=list)
    states: List[Any] = field(default_factory=list)
    env_id: str = "default"
    length: int = 0
    terminated_early: bool = False

    def __post_init__(self) -> None:
        if self.observations is None:
            self.observations = np.zeros((0,), dtype=np.float32)
        self.observations = np.asarray(self.observations, dtype=np.float32)
        if self.length == 0:
            self.length = int(self.observations.shape[0]) if self.observations.ndim else 0

    def __len__(self) -> int:
        return int(self.length)

    @property
    def importance_observations(self) -> np.ndarray:
        """Observations used for step-level mask scoring (same as ``observations``)."""
        return self.observations

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "env_id": self.env_id,
            "length": int(self.length),
            "terminated_early": bool(self.terminated_early),
            "total_reward": float(np.sum(self.rewards)) if len(self.rewards) else 0.0,
        }
        if include_arrays:
            out["observations"] = np.asarray(self.observations).tolist()
            out["actions"] = np.asarray(self.actions).tolist()
            out["rewards"] = np.asarray(self.rewards).tolist()
        return out


@dataclass
class CriticalState:
    """The mask-identified "most critical state" ``s_t`` of a trajectory ``tau``.

    Attributes
    ----------
    index:
        Step index ``t`` inside the trajectory ``tau`` (0-based).
    observation:
        The observation ``s_t`` (regardless of whether a simulator state is stored).
    state:
        Optional restorable simulator state (from ``rice.envs.reset_wrapper``); when
        available Algorithm 2 can reset *directly* to this state.
    actions:
        Actions taken from the episode start up to (but excluding) ``index`` -- the
        replay fallback used by :class:`rice.envs.reset_wrapper.ResetWrapper`.
    score:
        Importance ``P(a_t^m = 0 | s_t)`` of this state.
    importance_scores:
        Full per-step importance vector of ``tau`` (kept for fidelity/debugging).
    """

    index: int
    observation: np.ndarray
    state: Any = None
    actions: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    score: float = 0.0
    importance_scores: Optional[np.ndarray] = None
    trajectory_length: int = 0
    env_id: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.observation = flatten_observation(self.observation)
        if self.actions is None:
            self.actions = np.zeros((0,), dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)

    @property
    def restore_payload(self) -> Optional[Dict[str, Any]]:
        """Payload accepted by ``ResetWrapper.reset_to_state`` (``None`` if unknown)."""
        if self.state is None:
            return None
        return {"kind": self.metadata.get("state_kind", "sim"), "state": self.state}

    def to_dict(self, include_scores: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "index": int(self.index),
            "score": float(self.score),
            "env_id": self.env_id,
            "trajectory_length": int(self.trajectory_length),
            "has_state": self.state is not None,
            "num_replay_actions": int(self.actions.shape[0]),
            "observation_dim": int(self.observation.size),
        }
        if include_scores and self.importance_scores is not None:
            out["importance_scores"] = np.asarray(self.importance_scores).tolist()
        if self.metadata:
            out["metadata"] = {
                k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                for k, v in self.metadata.items()
            }
        return out


# --------------------------------------------------------------------------------------
# Policy / length helpers
# --------------------------------------------------------------------------------------
def _policy_step(policy: Any, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
    """Single action from any supported policy interface (SB3 / native / callable)."""
    obs = flatten_observation(observation)

    if policy is None:  # uniform random policy
        raise ValueError("policy_action requires a non-None policy; use sample_random_action")

    # Stable-Baselines3 BaseAlgorithm (PPO/SAC)
    predict = getattr(policy, "predict", None)
    if callable(predict):
        try:
            action = predict(obs, deterministic=bool(deterministic))
        except TypeError:
            action = predict(obs)
        if isinstance(action, tuple):
            action = action[0]
        return np.asarray(action, dtype=np.float32).reshape(-1)

    # Native ActorCritic / MaskedActionOperator
    act = getattr(policy, "act", None)
    if callable(act):
        try:
            out = act(obs, deterministic=bool(deterministic))
        except TypeError:
            out = act(obs)
        if isinstance(out, dict) and "action" in out:
            out = out["action"]
        if isinstance(out, tuple):
            out = out[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)

    # Callable(policy, obs) or Callable(obs)
    if callable(policy):
        out = policy(obs)
        if isinstance(out, tuple):
            out = out[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)

    raise TypeError(f"Unsupported policy interface: {type(policy)!r}")


def policy_action(policy: Any, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
    """Public alias of the generic policy->action adapter."""
    return _policy_step(policy, observation, deterministic=deterministic)


def default_k(env: Any = None, fallback: int = DEFAULT_K) -> int:
    """Episode horizon ``T`` used as the Algorithm-2 trajectory length ``K``."""
    if env is None:
        return int(fallback)
    for attr in ("rice_max_episode_steps",):
        value = getattr(env, attr, None)
        if value:
            return int(value)
    spec = getattr(env, "rice_env_spec", None)
    if spec is not None:
        value = getattr(spec, "max_episode_steps", None)
        if value:
            return int(value)
    value = getattr(env, "max_episode_steps", None) or getattr(env, "_max_episode_steps", None)
    if value:
        return int(value)
    inner = getattr(env, "env", None)
    if inner is not None:
        return default_k(inner, fallback=fallback)
    return int(fallback)


def _step_env(env: Any, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    """Normalise the 4- vs 5-tuple gym step API."""
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated) or bool(truncated)
        return np.asarray(obs), float(reward), bool(terminated), bool(truncated), dict(info or {})
    obs, reward, done, info = out  # type: ignore[misc]
    return np.asarray(obs), float(reward), bool(done), False, dict(info or {})


def _reset_env(env: Any) -> np.ndarray:
    out = env.reset()
    if isinstance(out, tuple):
        out = out[0]
    return np.asarray(out)


# --------------------------------------------------------------------------------------
# Trajectory rollout (Algorithm 2, "Run pi to obtain a trajectory tau of length K")
# --------------------------------------------------------------------------------------
def roll_trajectory(
    env: Any,
    policy: Any,
    length: Optional[int] = None,
    reset: bool = True,
    deterministic: bool = False,
    seed: Optional[int] = None,
    collect_states: bool = True,
    stop_on_done: bool = True,
) -> TrajectoryRollout:
    """Run the (pre-trained, frozen) policy ``pi`` for ``length`` steps.

    Parameters
    ----------
    env:
        Environment (ideally wrapped by ``rice.envs.reset_wrapper`` so that the
        simulator state of each visited step can be archived for a direct restore).
    policy:
        The pre-trained policy ``pi`` (SB3 model, native ``ActorCritic``, or callable).
    length:
        Trajectory length ``K``.  Defaults to :func:`default_k` (episode horizon ``T``).
    reset:
        Whether to reset the environment first.  ``False`` continues the current
        episode (useful when chaining rollouts).
    deterministic:
        Use the policy's mode.  Algorithm 2 samples ``a_t ~ pi`` (stochastic).
    """
    if seed is not None:
        try:
            env.seed(seed)
        except Exception:  # pragma: no cover - optional backend
            pass

    K = int(length) if length is not None else default_k(env)
    env_id = getattr(getattr(env, "rice_env_spec", None), "key", None) or "default"

    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    next_observations: List[np.ndarray] = []
    dones: List[bool] = []
    infos: List[Dict[str, Any]] = []
    states: List[Any] = []

    obs = _reset_env(env) if reset else getattr(env, "_last_obs", None)
    if obs is None:
        obs = _reset_env(env)
    obs = np.asarray(obs)

    snapshot_hook = getattr(env, "get_state", None)
    terminated_early = False

    for _ in range(K):
        observation = np.asarray(obs, dtype=np.float32)
        action = _policy_step(policy, observation, deterministic=deterministic)

        if collect_states and callable(snapshot_hook):
            try:
                states.append(snapshot_hook())
            except Exception:  # pragma: no cover - optional backend
                states.append(None)

        next_obs, reward, terminated, truncated, info = _step_env(env, action)
        done = terminated or truncated

        observations.append(observation)
        actions.append(np.asarray(action, dtype=np.float32))
        rewards.append(float(reward))
        next_observations.append(np.asarray(next_obs, dtype=np.float32))
        dones.append(bool(done))
        infos.append(info)

        obs = next_obs
        if done and stop_on_done:
            terminated_early = True
            break

    rollout = TrajectoryRollout(
        observations=np.asarray(observations, dtype=np.float32).reshape(len(observations), -1),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        next_observations=np.asarray(next_observations, dtype=np.float32).reshape(len(observations), -1),
        dones=np.asarray(dones, dtype=bool),
        infos=infos,
        states=states,
        env_id=env_id,
        length=len(observations),
        terminated_early=terminated_early,
    )
    try:
        setattr(env, "_last_obs", obs)
    except Exception:  # pragma: no cover
        pass
    return rollout


# --------------------------------------------------------------------------------------
# Critical-state identification
# --------------------------------------------------------------------------------------
def most_critical_index(
    trajectory: Any,
    mask_net: Any = None,
    scorer: Optional[ImportanceScorer] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Argmax-importance step index inside ``trajectory`` (Algorithm 2).

    Ties are broken by the earliest step (``np.nanargmax`` semantics), matching
    ``rice.explanation.importance.argmax_importance``.
    """
    if scorer is not None:
        return int(scorer.most_important_index(trajectory))
    observations = extract_observations(trajectory)
    scores = score_trajectory(mask_net, observations, batch_size=batch_size)
    return int(argmax_importance(scores))


def critical_state_from_rollout(
    rollout: TrajectoryRollout,
    mask_net: Any = None,
    scorer: Optional[ImportanceScorer] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> CriticalState:
    """Score ``rollout`` with the mask net and return its most critical state."""
    observations = np.asarray(rollout.importance_observations, dtype=np.float32)
    if observations.shape[0] == 0:
        return CriticalState(
            index=0,
            observation=np.zeros((0,), dtype=np.float32),
            score=float("nan"),
            importance_scores=np.zeros((0,), dtype=np.float32),
            trajectory_length=0,
            env_id=rollout.env_id,
            metadata={"empty_trajectory": True},
        )

    if scorer is not None:
        scores = np.asarray(scorer.score(observations), dtype=np.float64)
    else:
        scores = np.asarray(score_trajectory(mask_net, observations, batch_size=batch_size), dtype=np.float64)

    idx = int(argmax_importance(scores))
    idx = int(np.clip(idx, 0, observations.shape[0] - 1))

    state = None
    state_kind = "none"
    if rollout.states:
        state = rollout.states[idx] if idx < len(rollout.states) else rollout.states[-1]
        if state is not None:
            state_kind = state.get("kind", "sim") if isinstance(state, dict) else "attr"

    prev_actions = np.asarray(rollout.actions[:idx], dtype=np.float32) if idx > 0 else np.zeros((0,), dtype=np.float32)

    return CriticalState(
        index=idx,
        observation=observations[idx],
        state=state,
        actions=prev_actions,
        score=float(scores[idx]),
        importance_scores=scores,
        trajectory_length=int(observations.shape[0]),
        env_id=rollout.env_id,
        metadata={
            "state_kind": state_kind,
            "terminated_early": bool(rollout.terminated_early),
            "trajectory_reward": float(np.sum(rollout.rewards)) if len(rollout.rewards) else 0.0,
            "importance_summary": summarize_importance(scores),
        },
    )


def identify_critical_state(
    env: Any,
    policy: Any,
    mask_net: Any = None,
    K: Optional[int] = None,
    scorer: Optional[ImportanceScorer] = None,
    deterministic: bool = False,
    reset: bool = True,
    seed: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    return_rollout: bool = False,
) -> Any:
    """Algorithm 2, lines 4-6: run ``pi`` for ``K`` steps, return its critical state.

    Returns a :class:`CriticalState` (or ``(critical_state, rollout)`` when
    ``return_rollout=True``).
    """
    rollout = roll_trajectory(
        env,
        policy,
        length=K,
        reset=reset,
        deterministic=deterministic,
        seed=seed,
    )
    critical = critical_state_from_rollout(
        rollout,
        mask_net=mask_net,
        scorer=scorer,
        batch_size=batch_size,
    )
    if return_rollout:
        return critical, rollout
    return critical


def top_k_critical_states(
    rollout: TrajectoryRollout,
    mask_net: Any = None,
    k: int = 10,
    scorer: Optional[ImportanceScorer] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[CriticalState]:
    """Ranked (descending importance) top-``k`` critical states of ``rollout``.

    Used by the fidelity evaluator (Sec. 4.1) and by ``mixed_init`` when several
    frontier candidates are wanted.
    """
    observations = np.asarray(rollout.importance_observations, dtype=np.float32)
    if observations.shape[0] == 0:
        return []

    if scorer is not None:
        scores = np.asarray(scorer.score(observations), dtype=np.float64)
    else:
        scores = np.asarray(score_trajectory(mask_net, observations, batch_size=batch_size), dtype=np.float64)

    order = rank_states(scores, descending=True)
    order = order[: max(1, int(k))]

    out: List[CriticalState] = []
    for rank, idx in enumerate(order):
        idx = int(idx)
        state = rollout.states[idx] if (rollout.states and idx < len(rollout.states)) else None
        out.append(
            CriticalState(
                index=idx,
                observation=observations[idx],
                state=state,
                actions=np.asarray(rollout.actions[:idx], dtype=np.float32) if idx > 0 else np.zeros((0,), dtype=np.float32),
                score=float(scores[idx]),
                importance_scores=scores,
                trajectory_length=int(observations.shape[0]),
                env_id=rollout.env_id,
                metadata={"rank": rank},
            )
        )
    return out


def rank_trajectory(
    rollout: TrajectoryRollout,
    mask_net: Any = None,
    scorer: Optional[ImportanceScorer] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Step indices of ``rollout`` ordered by decreasing importance."""
    observations = np.asarray(rollout.importance_observations, dtype=np.float32)
    if observations.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if scorer is not None:
        scores = np.asarray(scorer.score(observations), dtype=np.float64)
    else:
        scores = np.asarray(score_trajectory(mask_net, observations, batch_size=batch_size), dtype=np.float64)
    return rank_states(scores, descending=True)


def select_critical_states(
    env: Any,
    policy: Any,
    mask_net: Any = None,
    n: int = 1,
    K: Optional[int] = None,
    scorer: Optional[ImportanceScorer] = None,
    deterministic: bool = False,
    seed: Optional[int] = None,
) -> List[CriticalState]:
    """Collect ``n`` independent trajectories and return their critical states."""
    out: List[CriticalState] = []
    for i in range(int(n)):
        sub_seed = None if seed is None else int(seed) + i
        critical = identify_critical_state(
            env,
            policy,
            mask_net=mask_net,
            K=K,
            scorer=scorer,
            deterministic=deterministic,
            seed=sub_seed,
        )
        out.append(critical)
    return out


# --------------------------------------------------------------------------------------
# Selector facade
# --------------------------------------------------------------------------------------
class CriticalStateSelector:
    """Convenience wrapper around the Algorithm-2 critical-state rule.

    Example
    -------
    >>> selector = CriticalStateSelector(env, mask_net, K=200)
    >>> critical = selector.select(policy)      # s_0 <- s_t
    >>> selector.reset_to(env, critical)        # jump the env to that state
    """

    def __init__(
        self,
        env: Any = None,
        mask_net: Any = None,
        K: Optional[int] = None,
        scorer: Optional[ImportanceScorer] = None,
        deterministic_policy: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[str] = None,
        rng: Optional[np.random.RandomState] = None,
        attach_scores: bool = True,
        cache: bool = False,
    ) -> None:
        self.env = env
        self.mask_net = mask_net
        self.K = int(K) if K is not None else None
        self.batch_size = int(batch_size)
        self.deterministic_policy = bool(deterministic_policy)
        self.attach_scores = bool(attach_scores)
        self.cache = bool(cache)
        self.rng = rng if rng is not None else np.random.RandomState(0)
        self.scorer = scorer if scorer is not None else (
            ImportanceScorer(mask_net=mask_net, batch_size=self.batch_size) if mask_net is not None else None
        )
        self.device = device
        self._cached: Optional[CriticalState] = None
        self.history: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {"rollouts": 0, "mean_trajectory_length": 0.0, "mean_critical_index": 0.0}

    # -- scoring ----------------------------------------------------------------
    def score(self, observations: Any) -> np.ndarray:
        """Step-level importance ``P(keep)`` of an observation sequence."""
        obs = extract_observations(observations)
        if self.scorer is not None:
            return np.asarray(self.scorer.score(obs), dtype=np.float64)
        return np.asarray(score_trajectory(self.mask_net, obs, batch_size=self.batch_size), dtype=np.float64)

    def rollout(self, policy: Any = None, K: Optional[int] = None, reset: bool = True,
                seed: Optional[int] = None) -> TrajectoryRollout:
        """Run the pre-trained policy for ``K`` steps (Algorithm 2, line 4)."""
        length = K if K is not None else self.K
        return roll_trajectory(
            self.env,
            policy,
            length=length,
            reset=reset,
            deterministic=self.deterministic_policy,
            seed=seed,
            collect_states=True,
        )

    # -- selection --------------------------------------------------------------
    def select(self, policy: Any = None, K: Optional[int] = None, reset: bool = True,
               seed: Optional[int] = None, rollout: Optional[TrajectoryRollout] = None) -> CriticalState:
        """Identify the most critical state ``s_t`` of a fresh trajectory ``tau``."""
        if rollout is None:
            if self.cache and self._cached is not None:
                return self._cached
            rollout = self.rollout(policy=policy, K=K, reset=reset, seed=seed)

        critical = critical_state_from_rollout(
            rollout,
            mask_net=self.mask_net,
            scorer=self.scorer,
            batch_size=self.batch_size,
        )
        self._record(rollout, critical)

        if self.attach_scores and self.env is not None:
            try:
                attach_scores_to_wrapper(self.env, critical.importance_scores)
            except Exception:  # pragma: no cover - wrapper not a ResetWrapper
                pass

        if self.cache:
            self._cached = critical
        return critical

    def select_top_k(self, k: int = 10, policy: Any = None, K: Optional[int] = None,
                     reset: bool = True, seed: Optional[int] = None) -> List[CriticalState]:
        rollout = self.rollout(policy=policy, K=K, reset=reset, seed=seed)
        return top_k_critical_states(rollout, mask_net=self.mask_net, k=k, scorer=self.scorer, batch_size=self.batch_size)

    def reset_to(self, env: Any, critical: CriticalState, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        """Jump ``env`` to ``critical`` (direct state if available, else replay)."""
        # 1) Preferred: ResetWrapper.reset_to(reset_to_state payload / snapshot)
        reset_to_state = getattr(env, "reset_to_state", None)
        payload = critical.restore_payload
        if payload is not None and callable(reset_to_state):
            try:
                obs, info = reset_to_state(payload, **(kwargs or {}))
                return obs, dict(info or {})
            except Exception:  # pragma: no cover - fall through to replay
                logger.debug("Direct state restore failed; falling back to action replay.")

        # 2) Fallback: Go-Explore style action replay
        reset_to = getattr(env, "reset_to", None)
        if callable(reset_to):
            try:
                out = reset_to(critical, use_direct=False, **(kwargs or {}))
            except TypeError:
                out = reset_to(critical)
            if isinstance(out, tuple):
                obs, info = out[0], (out[1] if len(out) > 1 else {})
                return obs, dict(info or {})
            return out, {}

        # 3) Last resort: plain reset (keeps the pipeline runnable)
        obs = _reset_env(env)
        return obs, {"restore_failed": True}

    def reset_to_critical(self, env: Any = None, critical: Optional[CriticalState] = None,
                          **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        """``reset_to`` with sensible defaults (uses the cached/last critical state)."""
        env = env if env is not None else self.env
        critical = critical if critical is not None else self._cached
        if critical is None:
            raise ValueError("No critical state available; call select() first.")
        return self.reset_to(env, critical, **kwargs)

    # -- bookkeeping ------------------------------------------------------------
    def _record(self, rollout: TrajectoryRollout, critical: CriticalState) -> None:
        self.history.append(
            {
                "trajectory_length": int(len(rollout)),
                "critical_index": int(critical.index),
                "critical_score": float(critical.score),
                "env_id": critical.env_id,
            }
        )
        n = len(self.history)
        self.stats["rollouts"] = n
        self.stats["mean_trajectory_length"] = float(np.mean([h["trajectory_length"] for h in self.history]))
        self.stats["mean_critical_index"] = float(np.mean([h["critical_index"] for h in self.history]))

    def summary(self) -> Dict[str, Any]:
        out = dict(self.stats)
        out["selector"] = {
            "K": self.K,
            "deterministic_policy": self.deterministic_policy,
            "has_mask_net": self.mask_net is not None,
            "attach_scores": self.attach_scores,
        }
        return out


# --------------------------------------------------------------------------------------
# Wrapper integration
# --------------------------------------------------------------------------------------
def attach_critical_state(env: Any, critical: CriticalState) -> Any:
    """Register ``critical`` as the frontier state on a ``ResetWrapper``-like env."""
    try:
        if hasattr(env, "critical_states"):
            env.critical_states.append(critical)
        else:  # pragma: no cover - plain gym env
            setattr(env, "critical_states", [critical])
        setattr(env, "last_critical_state", critical)
    except Exception:  # pragma: no cover
        pass
    return env


def describe_critical_state(critical: CriticalState) -> str:
    """Human-readable one-line summary (logging/debugging)."""
    return (
        f"CriticalState(t={critical.index}/{critical.trajectory_length}, "
        f"importance={critical.score:.4f}, state={'yes' if critical.state is not None else 'no'}, "
        f"replay_actions={critical.actions.shape[0]})"
    )

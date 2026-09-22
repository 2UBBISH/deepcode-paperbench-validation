"""Mixed initial state distribution for RICE (CORE COMPONENT #2).

Paper: "RICE: A Refining scheme for ReInForcement learning with Explanation"
(ICML 2024, PMLR 235).

Verbatim specification
----------------------
Section 3.3 (Constructing Mixed Initial State Distribution):

    "Initially, we randomly sample a trajectory by executing the pre-trained
     policy pi. Subsequently, the state mask is applied to pinpoint the most
     important state within the episode tau by assessing the significance of
     each state. The resulting distribution of these identified critical states
     is denoted as d_rho^{pi_hat}(s)."  ...  "We then set the initial
     distribution mu as a mixture of the selected important states distribution
     d_rho^{pi_hat}(s) and the original initial distribution of interest rho:

         mu(s) = beta * d_rho^{pi_hat}(s) + (1 - beta) * rho(s),

     where beta is a hyper-parameter."

Algorithm 2 (inner block, verbatim):

    for iteration = 1, 2, ... do
        D <- empty
        RAND_NUM <- RAND(0, 1)
        if RAND_NUM < p then
            Run pi to obtain a trajectory tau of length K
            Identify the most critical state s_t in tau via state mask pi_tilde
            Set the initial state s_0 <- s_t
        else
            Set the initial state s_0 ~ rho
        end if
        ...

Practical implementation notes
------------------------------
* The paper's theoretical mixture ``mu(s) = beta d_rho^{pi_hat}(s) + (1-beta) rho(s)``
  is realised in the algorithm as a *Bernoulli(p) roll-in* performed once at the
  start of every refining iteration, with ``p`` playing the role of ``beta``
  (the plan's mapping ``beta <-> p``).  ``RAND_NUM`` is therefore drawn exactly
  ONCE per outer iteration (the whole episode uses one single ``s_0``).
* Section 4.3 (Impact of Hyper-parameters): "The performance is low when p = 0
  (all starting from the default initial distribution) or p = 1 (all starting
  from the identified critical states). The performance has significant
  improvements when 0 < p < 1 ... setting p to 0.25 or 0.5 is most beneficial."
  We therefore expose :meth:`MixedInitSampler.warn_if_degenerate` and record the
  empirical fraction of critical-state resets so the user can verify 0 < p < 1.
* Two sampling back-ends are provided:
    - ``"rollin"`` (default, paper-faithful): every critical iteration runs the
      policy for K steps and takes the ``argmax`` importance state;
    - ``"pool"``: a pre-collected pool of critical states
      (:func:`rice.algorithms.critical_state.collect_critical_states`) is used as
      an empirical sample of ``d_rho^{pi_hat}``, which is far cheaper when the
      roll-in is expensive (e.g. CAGE with 1e7 mask samples).

Both back-ends can be mixed: ``mode="rollin"`` falls back to the pool whenever a
roll-in is impossible (no mask network supplied), so the sampler never crashes
the refining loop.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .critical_state import (
    RollinTrajectory,
    identify_critical_state,
    importance_scores,
    max_episode_steps,
    roll_in_trajectory,
)
from .ppo import flatten_obs, make_target_policy_callable

# --------------------------------------------------------------------------- #
# RNG compatibility (``rice.utils.seeding.RNG`` is the canonical implementation)
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - import guard only
    from ..utils.seeding import RNG as _RNG  # type: ignore
except Exception:  # pragma: no cover
    try:
        from rice.utils.seeding import RNG as _RNG  # type: ignore
    except Exception:  # pragma: no cover
        _RNG = None  # type: ignore


class _FallbackRNG:
    """Minimal stand-in for :class:`rice.utils.seeding.RNG` (numpy-only)."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self.generator = np.random.default_rng(seed)

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return float(self.generator.uniform(low, high))

    def bernoulli(self, p: float) -> bool:
        return bool(self.generator.uniform() < float(p))

    def choice(self, a: Any, p: Optional[Sequence[float]] = None) -> Any:
        return self.generator.choice(a, p=p)

    def integers(self, low: int, high: Optional[int] = None, size: Any = None) -> Any:
        return self.generator.integers(low, high, size)

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - delegation
        return getattr(self.generator, item)


def _make_rng(rng: Any = None, seed: Optional[int] = None) -> Any:
    """Build/accept a random generator supporting ``uniform``/``bernoulli``."""
    if rng is not None:
        if hasattr(rng, "uniform") and hasattr(rng, "bernoulli"):
            return rng
        # A raw numpy Generator / RandomState: adapt it.
        wrapper = _FallbackRNG(seed=None)
        wrapper.generator = rng
        return wrapper
    cls = _RNG if _RNG is not None else _FallbackRNG
    try:
        return cls(seed)
    except TypeError:  # pragma: no cover - unexpected constructor
        return _FallbackRNG(seed)


def mixed_initial_distribution(p: float) -> Tuple[float, float]:
    """Mixture weights ``(beta, 1 - beta)`` of ``mu(s)``.

    Returns ``(w_critical, w_default)`` = ``(p, 1 - p)``: the probability mass
    assigned to the identified critical states ``d_rho^{pi_hat}`` and to the
    default initial distribution ``rho`` respectively (Section 3.3).
    """
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p (beta) must lie in [0, 1], got {p}.")
    return p, 1.0 - p


# --------------------------------------------------------------------------- #
# Sampling containers
# --------------------------------------------------------------------------- #
@dataclass
class MixedInitSample:
    """One draw of ``s_0 ~ mu(s) = beta d_rho^{pi_hat} + (1 - beta) rho``."""

    mode: str  # "critical" | "default"
    state: Any = None  # simulator state (Go-Explore style snapshot) if any
    observation: Any = None  # observation associated with ``state``
    critical_index: Optional[int] = None  # argmax position inside the roll-in
    importance: Optional[float] = None  # P(mask = 0 | s_critical)
    importance_scores: Optional[np.ndarray] = None
    trajectory: Optional[RollinTrajectory] = None
    from_pool: bool = False
    needs_manual_reset: bool = False
    iteration: Optional[int] = None

    @property
    def is_critical(self) -> bool:
        return self.mode == "critical"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "state": self.state,
            "observation": self.observation,
            "critical_index": self.critical_index,
            "importance": self.importance,
            "from_pool": self.from_pool,
            "needs_manual_reset": self.needs_manual_reset,
            "iteration": self.iteration,
        }


# --------------------------------------------------------------------------- #
# env helpers (gym + gymnasium compatible, duck-typed)
# --------------------------------------------------------------------------- #
def _env_reset(env: Any, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
    """Reset ``env``; return ``(observation, info)`` for both gym APIs."""
    try:
        out = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
        return obs, dict(info or {})
    return out, {}


def _env_step(env: Any, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    """Step ``env``; normalise gym (4-tuple) and gymnasium (5-tuple) returns."""
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated), bool(truncated), dict(info or {})
    obs, reward, done, info = out
    return obs, float(reward), bool(done), False, dict(info or {})


def _snapshot(state_manager: Any, env: Any) -> Any:
    """Best-effort simulator-state snapshot (Go-Explore style, Duck-typed)."""
    if state_manager is not None:
        for name in ("snapshot", "save_state", "save", "get_state", "state_dict"):
            fn = getattr(state_manager, name, None)
            if callable(fn):
                try:
                    return fn()
                except TypeError:
                    continue
    # Fall back to MuJoCo-style raw simulator state if available.
    unwrapped = getattr(env, "unwrapped", env)
    state = getattr(unwrapped, "state", None)
    if state is not None:
        try:
            return (np.array(state[0], copy=True), np.array(state[1], copy=True))
        except Exception:  # pragma: no cover
            return state
    return None


def _restore(state_manager: Any, env: Any, state: Any) -> bool:
    """Restore ``env`` to ``state``.  Returns ``True`` on success."""
    if state is None:
        return False
    if state_manager is not None:
        for name in ("restore", "load_state", "load", "set_state", "restore_state"):
            fn = getattr(state_manager, name, None)
            if callable(fn):
                try:
                    fn(state)
                    return True
                except Exception:
                    continue
    unwrapped = getattr(env, "unwrapped", env)
    set_state = getattr(unwrapped, "set_state", None)
    if callable(set_state) and isinstance(state, (tuple, list)) and len(state) == 2:
        try:
            set_state(np.asarray(state[0], dtype=np.float64),
                      np.asarray(state[1], dtype=np.float64))
            return True
        except Exception:
            return False
    return False


# --------------------------------------------------------------------------- #
# Main sampler
# --------------------------------------------------------------------------- #
class MixedInitSampler:
    """Bernoulli(p) roll-in sampler implementing ``mu(s)`` (CORE COMPONENT #2).

    Parameters
    ----------
    env:
        Gym/gymnasium environment the refining loop interacts with.
    policy:
        Policy used for the K-step roll-in.  Per Algorithm 2 this is the
        pre-trained policy ``pi``; the plan notes that the refining loop may
        instead pass the *current* refining policy by rebuilding the sampler or
        by calling :meth:`set_policy`.  Accepts an SB3 model, a torch
        actor-critic, or a plain ``obs -> action`` callable.
    mask_network:
        Trained mask network ``pi_tilde``.  Required for ``mode="rollin"``.
    p:
        Reset probability threshold ``p`` (plays the role of ``beta``).
        Table 3 values: Hopper 0.25, Walker2d 0.25, Reacher 0.50,
        HalfCheetah 0.50, Selfish Mining 0.25, CAGE-2 0.50, Auto Driving 0.25.
    rollin_length:
        Trajectory length ``K`` of the roll-in.  ``None`` = one full
        pre-trained-policy episode (plan's chosen default).
    state_manager:
        Optional :class:`rice.algorithms.env_reset.EnvStateManager` used to
        snapshot/restore the simulator state at the identified critical step
        (Ecoffet et al. 2019 style, Section C.1).
    critical_state_pool:
        Optional pre-collected list of critical states (empirical
        ``d_rho^{pi_hat}``) used by ``mode="pool"`` or as a fallback.
    mode:
        ``"rollin"`` (paper-faithful, default) or ``"pool"``.
    """

    def __init__(
        self,
        env: Any,
        policy: Any,
        mask_network: Any = None,
        p: float = 0.25,
        rollin_length: Optional[int] = None,
        state_manager: Any = None,
        critical_state_pool: Optional[Sequence[Any]] = None,
        pool_observations: Optional[Sequence[Any]] = None,
        mode: str = "rollin",
        rng: Any = None,
        seed: Optional[int] = None,
        batch_size: int = 512,
        device: Any = None,
    ) -> None:
        if mode not in ("rollin", "pool"):
            raise ValueError(f"mode must be 'rollin' or 'pool', got {mode!r}.")
        self.env = env
        self.policy = policy
        self._policy_callable: Optional[Callable[[Any], np.ndarray]] = None
        self.mask_network = mask_network
        self.p = float(p)
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p (beta) must lie in [0, 1], got {self.p}.")
        self.rollin_length = rollin_length
        self.state_manager = state_manager
        self.mode = mode
        self.rng = _make_rng(rng=rng, seed=seed)
        self.batch_size = int(batch_size)
        self.device = device

        self.critical_state_pool: List[Any] = list(critical_state_pool or [])
        self.pool_observations: List[Any] = list(pool_observations or [])

        # Bookkeeping (used by tests / Experiment V sensitivity plots).
        self.n_samples = 0
        self.n_critical = 0
        self.n_default = 0
        self.history: List[str] = []
        self.decisions: List[float] = []  # the raw RAND_NUM draws

    # ------------------------------------------------------------------ #
    # configuration
    # ------------------------------------------------------------------ #
    def set_policy(self, policy: Any) -> None:
        """Update the roll-in policy (e.g. to the current refining policy)."""
        self.policy = policy
        self._policy_callable = None

    def set_p(self, p: float) -> None:
        """Update the reset probability threshold ``p`` (= ``beta``)."""
        p = float(p)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p (beta) must lie in [0, 1], got {p}.")
        self.p = p

    def set_mask_network(self, mask_network: Any) -> None:
        self.mask_network = mask_network

    def add_critical_state(self, state: Any, observation: Any = None) -> None:
        """Append one critical state to the empirical ``d_rho^{pi_hat}`` pool."""
        self.critical_state_pool.append(state)
        self.pool_observations.append(observation)

    def extend_critical_states(self, states: Sequence[Any],
                               observations: Optional[Sequence[Any]] = None) -> None:
        self.critical_state_pool.extend(list(states))
        if observations is not None:
            self.pool_observations.extend(list(observations))
        else:
            self.pool_observations.extend([None] * len(states))

    @property
    def policy_callable(self) -> Callable[[Any], np.ndarray]:
        if self._policy_callable is None:
            self._policy_callable = make_target_policy_callable(self.policy)
        return self._policy_callable

    @property
    def mixture_weights(self) -> Tuple[float, float]:
        """``(beta, 1 - beta)`` = ``(p, 1 - p)`` i.e. the ``mu(s)`` weights."""
        return mixed_initial_distribution(self.p)

    def warn_if_degenerate(self) -> Optional[str]:
        """Warn when ``p = 0`` or ``p = 1`` (Section 4.3: both are sub-optimal)."""
        if self.p <= 0.0:
            msg = ("p = 0: all episodes start from the default initial distribution "
                   "rho, so the critical states are never used (Section 4.3 reports "
                   "low performance).")
        elif self.p >= 1.0:
            msg = ("p = 1: all episodes start from the identified critical states, "
                   "which causes overfitting / poor performance (Section 4.3).")
        else:
            return None
        warnings.warn(msg, stacklevel=2)
        return msg

    # ------------------------------------------------------------------ #
    # the RAND(0,1) < p decision (Algorithm 2)
    # ------------------------------------------------------------------ #
    def decide(self, force: Optional[bool] = None) -> bool:
        """Draw ``RAND_NUM ~ RAND(0,1)`` once and return ``RAND_NUM < p``.

        ``True`` selects the critical-state branch (``s_0 <- s_t``), ``False``
        selects the default branch (``s_0 ~ rho``).
        """
        if force is not None:
            return bool(force)
        rand_num = float(self.rng.uniform(0.0, 1.0))
        self.decisions.append(rand_num)
        return bool(rand_num < self.p)

    # ------------------------------------------------------------------ #
    # critical-state branch
    # ------------------------------------------------------------------ #
    def roll_in(self, length: Optional[int] = None, seed: Optional[int] = None,
                reset: bool = True) -> RollinTrajectory:
        """Run the roll-in policy for K steps starting from ``s_0 ~ rho``."""
        length = self.rollin_length if length is None else length
        return roll_in_trajectory(
            self.env,
            self.policy_callable,
            length=length,
            rng=self.rng,
            reset=reset,
            seed=seed,
            state_manager=self.state_manager,
        )

    def _roll_in_with_snapshots(self, length: Optional[int] = None,
                                seed: Optional[int] = None
                                ) -> Tuple[RollinTrajectory, List[Any]]:
        """Roll-in that also records a simulator snapshot at every step.

        Needed to restore the environment to the *identified* critical state
        (Algorithm 2: ``Set the initial state s_0 <- s_t``).
        """
        length = length if length is not None else self.rollin_length
        if length is None:
            length = max_episode_steps(self.env)
        obs, _info = _env_reset(self.env, seed=seed)
        policy = self.policy_callable
        states: List[Any] = []
        observations: List[Any] = []
        actions: List[Any] = []
        rewards: List[float] = []
        next_states: List[Any] = []
        next_observations: List[Any] = []
        dones: List[bool] = []
        infos: List[Dict[str, Any]] = []
        snapshots: List[Any] = []

        max_steps = max_episode_steps(self.env)
        for _ in range(int(length)):
            states.append(obs)
            snapshots.append(_snapshot(self.state_manager, self.env))
            action = policy(obs)
            action = np.asarray(action)
            next_obs, reward, terminated, truncated, info = _env_step(self.env, action)
            done = bool(terminated or truncated)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            next_states.append(next_obs)
            dones.append(done)
            infos.append(info)
            obs = next_obs
            if done:
                break
            if len(states) >= max_steps:
                break

        traj = RollinTrajectory(
            states=observations + [flatten_obs(s) for s in states],
            actions=actions,
            rewards=rewards,
            next_states=states[: len(next_states)],
            dones=dones,
            infos=infos,
        )
        # ``RollinTrajectory`` may normalise observations internally; keep the
        # original (possibly non-flat / same-object) states for restoration.
        traj.states = states
        return traj, snapshots

    def sample_critical(self, length: Optional[int] = None,
                        seed: Optional[int] = None,
                        iteration: Optional[int] = None) -> MixedInitSample:
        """Sample ``s_0`` from ``d_rho^{pi_hat}`` (critical-state branch)."""
        if self.mode == "pool" or (self.mask_network is None and self.critical_state_pool):
            return self._sample_from_pool(iteration=iteration)

        if self.mask_network is None:
            raise RuntimeError(
                "Critical-state sampling needs a mask network (roll-in branch) or "
                "a non-empty critical_state_pool (pool branch)."
            )

        if self.state_manager is not None:
            traj, snapshots = self._roll_in_with_snapshots(length=length, seed=seed)
            scores = importance_scores(
                self.mask_network, traj.states, batch_size=self.batch_size,
                device=self.device,
            )
            idx = int(np.argmax(scores)) if len(scores) else 0
            state = snapshots[idx] if idx < len(snapshots) else None
            obs = traj.states[idx] if idx < len(traj.states) else None
            restored = _restore(self.state_manager, self.env, state)
            return MixedInitSample(
                mode="critical",
                state=state,
                observation=obs,
                critical_index=idx,
                importance=float(scores[idx]) if len(scores) else None,
                importance_scores=scores,
                trajectory=traj,
                needs_manual_reset=not restored,
                iteration=iteration,
            )

        # No explicit state manager: delegate to critical_state.identify_critical_state
        # (paper-faithful inner block of Algorithm 2).
        result = identify_critical_state(
            self.env,
            self.policy_callable,
            self.mask_network,
            length=length if length is not None else self.rollin_length,
            rng=self.rng,
            reset=True,
            seed=seed,
            return_trajectory=True,
            batch_size=self.batch_size,
            device=self.device,
        )
        state, idx, value, scores, traj = _unpack_identify(result)
        return MixedInitSample(
            mode="critical",
            state=state,
            observation=state,
            critical_index=idx,
            importance=value,
            importance_scores=scores,
            trajectory=traj,
            needs_manual_reset=True,
            iteration=iteration,
        )

    def _sample_from_pool(self, iteration: Optional[int] = None) -> MixedInitSample:
        """Draw a critical state from the empirical pool ``d_rho^{pi_hat}``."""
        if not self.critical_state_pool:
            raise RuntimeError("critical_state_pool is empty; nothing to sample.")
        idx = int(self.rng.integers(0, len(self.critical_state_pool)))
        state = self.critical_state_pool[idx]
        obs = self.pool_observations[idx] if idx < len(self.pool_observations) else None
        restored = _restore(self.state_manager, self.env, state)
        importance = None
        if self.mask_network is not None and obs is not None:
            try:
                importance = float(np.ravel(importance_scores(
                    self.mask_network, [obs], batch_size=1, device=self.device))[0])
            except Exception:  # pragma: no cover - best effort only
                importance = None
        return MixedInitSample(
            mode="critical",
            state=state,
            observation=obs if obs is not None else state,
            critical_index=None,
            importance=importance,
            from_pool=True,
            needs_manual_reset=not restored,
            iteration=iteration,
        )

    # ------------------------------------------------------------------ #
    # default branch
    # ------------------------------------------------------------------ #
    def sample_default(self, seed: Optional[int] = None,
                       iteration: Optional[int] = None) -> MixedInitSample:
        """Sample ``s_0 ~ rho``: the environment's default initial states."""
        obs, _info = _env_reset(self.env, seed=seed)
        return MixedInitSample(
            mode="default",
            state=_snapshot(self.state_manager, self.env),
            observation=obs,
            iteration=iteration,
        )

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def sample(self, force: Optional[bool] = None, seed: Optional[int] = None,
               length: Optional[int] = None, iteration: Optional[int] = None,
               reset_env: bool = True) -> MixedInitSample:
        """Draw one ``s_0 ~ mu`` (Algorithm 2 roll-in decision included).

        ``force`` bypasses the ``RAND_NUM < p`` draw (``True`` = critical branch,
        ``False`` = default branch) which is convenient for experiments and
        unit tests.
        """
        use_critical = self.decide(force=force)
        if use_critical:
            try:
                sample = self.sample_critical(length=length, seed=seed,
                                              iteration=iteration)
            except Exception as exc:  # roll-in failure: fall back on the pool
                if self.critical_state_pool:
                    warnings.warn(f"Critical roll-in failed ({exc}); using the pool.",
                                  stacklevel=2)
                    sample = self._sample_from_pool(iteration=iteration)
                else:
                    raise
        else:
            sample = self.sample_default(seed=seed, iteration=iteration)
            if reset_env:
                pass  # ``sample_default`` already reset the environment.

        self.n_samples += 1
        if sample.is_critical:
            self.n_critical += 1
        else:
            self.n_default += 1
        self.history.append(sample.mode)
        return sample

    def reset(self, force: Optional[bool] = None, seed: Optional[int] = None,
              length: Optional[int] = None, iteration: Optional[int] = None
              ) -> Tuple[Any, Dict[str, Any]]:
        """Reset ``env`` to ``s_0 ~ mu`` and return ``(observation, info)``.

        Returns the observation together with a metadata dict containing the
        chosen ``mode``, the ``critical_index`` (``argmax`` position inside the
        roll-in trajectory) and the importance ``P(mask = 0 | s_critical)``.
        """
        sample = self.sample(force=force, seed=seed, length=length,
                             iteration=iteration)
        obs = sample.observation
        if sample.is_critical:
            if sample.needs_manual_reset:
                pass  # the caller must restore ``sample.state`` itself
            else:
                # Re-read the observation after the state restoration.
                obs = self._current_observation(sample) or obs
        info = sample.as_dict()
        info["p"] = self.p
        info["mixture_weights"] = self.mixture_weights
        return obs, info

    def _current_observation(self, sample: MixedInitSample) -> Any:
        """Best-effort observation read after a state restore."""
        if self.state_manager is not None:
            for name in ("current_observation", "get_observation", "observation"):
                attr = getattr(self.state_manager, name, None)
                if callable(attr):
                    try:
                        return attr()
                    except Exception:
                        continue
                elif attr is not None:
                    return attr
        return sample.observation

    def sample_many(self, n: int, force: Optional[bool] = None,
                    seed: Optional[int] = None, length: Optional[int] = None
                    ) -> List[MixedInitSample]:
        """Draw ``n`` independent initial states (for empirical validation)."""
        return [self.sample(force=force, seed=seed, length=length) for _ in range(int(n))]

    def critical_fraction(self, n: Optional[int] = None) -> float:
        """Empirical share of critical-state resets.

        With ``n`` given, a fresh batch of ``n`` Bernoulli draws is made (no
        environment interaction) and its fraction is returned; otherwise the
        running statistics are reported.  This is the Monte-Carlo check that the
        sampler realises ``beta = p``.
        """
        if n is not None:
            if n <= 0:
                raise ValueError("n must be positive.")
            draws = [self.decide() for _ in range(int(n))]
            return float(np.mean(draws)) if draws else 0.0
        if self.n_samples == 0:
            return float("nan")
        return self.n_critical / float(self.n_samples)

    def statistics(self) -> Dict[str, Any]:
        return {
            "p": self.p,
            "beta": self.p,
            "n_samples": self.n_samples,
            "n_critical": self.n_critical,
            "n_default": self.n_default,
            "critical_fraction": self.critical_fraction(),
            "pool_size": len(self.critical_state_pool),
            "mode": self.mode,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "p": self.p,
            "mode": self.mode,
            "rollin_length": self.rollin_length,
            "n_samples": self.n_samples,
            "n_critical": self.n_critical,
            "n_default": self.n_default,
            "critical_state_pool": self.critical_state_pool,
            "pool_observations": self.pool_observations,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.p = float(state.get("p", self.p))
        self.mode = state.get("mode", self.mode)
        self.rollin_length = state.get("rollin_length", self.rollin_length)
        self.n_samples = int(state.get("n_samples", self.n_samples))
        self.n_critical = int(state.get("n_critical", self.n_critical))
        self.n_default = int(state.get("n_default", self.n_default))
        self.critical_state_pool = list(state.get("critical_state_pool",
                                                  self.critical_state_pool))
        self.pool_observations = list(state.get("pool_observations",
                                                self.pool_observations))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _unpack_identify(result: Any) -> Tuple[Any, Optional[int], Optional[float],
                                           Optional[np.ndarray], Optional[Any]]:
    """Normalise the possible return signatures of ``identify_critical_state``."""
    if isinstance(result, tuple):
        if len(result) == 5:
            state, idx, value, scores, traj = result
        elif len(result) == 4:
            state, idx, value, scores = result
            traj = None
        elif len(result) == 3:
            state, idx, value = result
            scores, traj = None, None
        elif len(result) == 2:
            state, idx = result
            value, scores, traj = None, None, None
        else:  # pragma: no cover - unexpected signature
            state, idx, value, scores, traj = result[0], None, None, None, None
    else:
        state, idx, value, scores, traj = result, None, None, None, None
    return (
        state,
        None if idx is None else int(idx),
        None if value is None else float(value),
        None if scores is None else np.asarray(scores, dtype=np.float64),
        traj,
    )


def bernoulli_rollin(sampler: MixedInitSampler, force: Optional[bool] = None,
                     seed: Optional[int] = None, length: Optional[int] = None
                     ) -> Tuple[Any, Dict[str, Any]]:
    """Convenience wrapper: one Algorithm-2 roll-in decision + env reset."""
    return sampler.reset(force=force, seed=seed, length=length)


def make_mixed_init_sampler(env: Any, policy: Any, mask_network: Any = None,
                            config: Optional[Dict[str, Any]] = None, **kwargs: Any
                            ) -> MixedInitSampler:
    """Build a :class:`MixedInitSampler` from a config dict / YAML mapping."""
    params: Dict[str, Any] = dict(config or {})
    params.update(kwargs)
    if "beta" in params and "p" not in params:
        params["p"] = params.pop("beta")
    if "length" in params and "rollin_length" not in params:
        params["rollin_length"] = params.pop("length")
    return MixedInitSampler(env=env, policy=policy, mask_network=mask_network,
                            **params)


__all__ = [
    "MixedInitSample",
    "MixedInitSampler",
    "bernoulli_rollin",
    "make_mixed_init_sampler",
    "mixed_initial_distribution",
]

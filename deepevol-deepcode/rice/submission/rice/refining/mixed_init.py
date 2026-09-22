"""Mixed initial state distribution for RICE Stage 2 (refining).

Implements the sampling side of Algorithm 2 of *RICE: A Refining Scheme for
Reinforcement Learning with Explanation* (Cheng et al., ICML 2024).

Paper (Sec. 3.3, "Constructing Mixed Initial State Distribution")::

    Initially, we randomly sample a trajectory by executing the pre-trained
    policy pi. Subsequently, the state mask is applied to pinpoint the most
    important state within the episode tau by assessing the significance of
    each state. The resulting distribution of these identified critical states
    is denoted as d_rho^pihat(s). [...]  we then set the initial distribution
    mu as a mixture of the selected important states distribution
    d_rho^pihat(s) and the original initial distribution of interest rho:
    mu(s) = beta * d_rho^pihat(s) + (1 - beta) * rho(s), where beta is a
    hyper-parameter.

Algorithm 2 realises exactly this mixture with the reset rule::

    RAND_NUM <- RAND(0, 1)
    if RAND_NUM < p then
        Run pi to obtain a trajectory tau of length K
        Identify the most critical state s_t in tau via state mask ~pi
        Set the initial state s_0 <- s_t
    else
        Set the initial state s_0 ~ rho
    end if

so the reset probability threshold ``p`` plays the role of the mixture weight
``beta``: ``P(s_0 ~ d_rho^pihat) = p`` and ``P(s_0 ~ rho) = 1 - p``.

This module provides:

* :class:`MixedInitConfig`   - the ``beta``/``p`` and trajectory-length ``K`` bundle.
* :class:`InitSample`        - the returned initial state + provenance bookkeeping.
* :class:`MixedInitialStateSampler` - the sampler used once per refinement iteration.
* :func:`sample_initial_state`, :func:`make_mixed_init_sampler` - convenience APIs.

The sampler only *chooses and installs* the initial state. The refinement loop
(``rice/refining/ppo_refine.py``) is responsible for rolling the trainable policy
``pi_theta`` for ``T`` steps from the returned observation and for optimizing
both the PPO loss and the RND predictor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from rice.utils.seeding import get_rng

try:  # pragma: no cover - logging is a soft dependency
    from rice.utils.logging import get_logger
except Exception:  # pragma: no cover
    def get_logger(*args: Any, **kwargs: Any):  # type: ignore
        import logging

        return logging.getLogger("rice")


try:  # pragma: no cover - reset wrapper may be unavailable in minimal setups
    from rice.envs.reset_wrapper import (
        ResetWrapper,
        get_env_state,
        make_reset_env,
        set_env_state,
    )
except Exception:  # pragma: no cover
    ResetWrapper = None  # type: ignore

    def make_reset_env(env: Any, **kwargs: Any) -> Any:  # type: ignore
        return env

    def get_env_state(env: Any) -> Dict[str, Any]:  # type: ignore
        return {"kind": "none", "state": None}

    def set_env_state(env: Any, packed: Any) -> bool:  # type: ignore
        return False


try:  # pragma: no cover
    from rice.explanation.critical_state import (
        CriticalState,
        CriticalStateSelector,
        attach_critical_state,
        critical_state_from_rollout,
        default_k,
        policy_action,
        roll_trajectory,
    )
except Exception:  # pragma: no cover - extremely defensive fallback
    CriticalState = None  # type: ignore
    CriticalStateSelector = None  # type: ignore

    def default_k(env: Any = None, fallback: int = 1000) -> int:  # type: ignore
        return int(fallback)

    def attach_critical_state(env: Any, critical: Any) -> Any:  # type: ignore
        return env

    def critical_state_from_rollout(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        raise RuntimeError("rice.explanation.critical_state unavailable")


__all__ = [
    "DEFAULT_P",
    "DEFAULT_BETA",
    "MixedInitConfig",
    "InitSample",
    "MixedInitialStateSampler",
    "sample_initial_state",
    "make_mixed_init_sampler",
    "build_mixed_init_sampler",
    "mixed_init_probability",
    "mixture_weights",
    "should_use_critical",
    "describe_mixed_init",
]

#: Default reset probability / mixture weight (paper sweeps p in Exp. V; 0.5 is
#: reported as one of the best values together with 0.25).
DEFAULT_P = 0.5
#: ``beta`` in mu(s) = beta d_rho^pihat(s) + (1 - beta) rho(s) is realised by ``p``.
DEFAULT_BETA = DEFAULT_P


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class MixedInitConfig:
    """Configuration for the mixed initial state distribution sampler.

    Attributes
    ----------
    env_id:
        Environment key (logging/bookkeeping only).
    p:
        Reset probability threshold from Algorithm 2. Equivalently the mixture
        weight ``beta`` of ``d_rho^pihat(s)`` in ``mu(s)``.
    K:
        Length of the trajectory rolled with the frozen pre-trained policy to
        identify the most critical state. ``None`` -> episode horizon ``T``
        (paper's K defaults to T in our implementation). A float in (0, 1) is
        interpreted as a fraction of the horizon.
    deterministic_policy:
        Whether the frozen policy ``pi`` is run deterministically for the
        critical-state rollout.
    use_direct_restore:
        Prefer direct simulator state injection over action replay when jumping
        to the identified critical state.
    fallback_to_default:
        If the critical-state restore fails, fall back to sampling ``s_0 ~ rho``
        instead of raising.
    refresh_critical:
        Re-roll ``pi`` and re-identify the critical state on every sample
        (Algorithm 2 semantics). If ``False`` the first critical state is reused.
    seed:
        Optional seed for the internal Bernoulli/Uniform draw.
    """

    env_id: Optional[str] = None
    p: float = DEFAULT_P
    K: Optional[Any] = None
    deterministic_policy: bool = False
    use_direct_restore: bool = True
    fallback_to_default: bool = True
    refresh_critical: bool = True
    cache_critical: bool = False
    seed: Optional[int] = None

    # ---- construction helpers ---------------------------------------------------------
    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None, **overrides: Any) -> "MixedInitConfig":
        """Build a config from a (possibly nested) dict plus keyword overrides."""
        cfg = dict(cfg or {})
        # accept nested sections used by the YAML configs
        for key in ("mixed_init", "refining", "refine", "stage2"):
            section = cfg.get(key)
            if isinstance(section, dict):
                merged = dict(cfg)
                merged.update(section)
                cfg = merged
        # alias handling: beta is the mixture weight and equals p in Algorithm 2
        if "beta" in cfg and "p" not in cfg:
            cfg["p"] = cfg.pop("beta")
        if "reset_probability" in cfg and "p" not in cfg:
            cfg["p"] = cfg.pop("reset_probability")
        if "reset_prob" in cfg and "p" not in cfg:
            cfg["p"] = cfg.pop("reset_prob")
        if "trajectory_length" in cfg and "K" not in cfg:
            cfg["K"] = cfg.pop("trajectory_length")
        if "length" in cfg and "K" not in cfg:
            cfg["K"] = cfg.pop("length")
        if "deterministic" in cfg and "deterministic_policy" not in cfg:
            cfg["deterministic_policy"] = cfg.pop("deterministic")

        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        cfg.update({k: v for k, v in overrides.items() if v is not None or k in allowed})
        filtered = {k: v for k, v in cfg.items() if k in allowed and v is not None}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict (JSON friendly) representation."""
        return {
            "env_id": self.env_id,
            "p": float(self.p),
            "beta": float(self.beta),
            "K": self.K,
            "deterministic_policy": bool(self.deterministic_policy),
            "use_direct_restore": bool(self.use_direct_restore),
            "fallback_to_default": bool(self.fallback_to_default),
            "refresh_critical": bool(self.refresh_critical),
            "cache_critical": bool(self.cache_critical),
            "seed": self.seed,
        }

    @property
    def beta(self) -> float:
        """Mixture weight of ``d_rho^pihat(s)`` (== reset probability ``p``)."""
        return float(self.p)


# --------------------------------------------------------------------------------------
# Sampled initial state container
# --------------------------------------------------------------------------------------
@dataclass
class InitSample:
    """A sampled initial state ``s_0`` together with its provenance.

    Attributes
    ----------
    observation:
        The observation to start the refinement rollout from.
    info:
        ``info`` dict returned by the environment reset (if any).
    from_critical:
        ``True`` when ``s_0`` was restored from the mask-identified critical
        state (i.e. ``s_0 ~ d_rho^pihat``), ``False`` when ``s_0 ~ rho``.
    mode:
        One of ``"critical"``, ``"default"`` or ``"fallback"``.
    critical:
        The :class:`~rice.explanation.critical_state.CriticalState` used, if any.
    rand_num:
        The ``RAND_NUM ~ U(0, 1)`` draw that decided the branch.
    p:
        The reset probability used for the draw.
    trajectory_length:
        Number of environment steps rolled to find the critical state (0 for the
        default branch).
    importance:
        Importance score ``P(keep)`` of the restored critical state.
    restore_failed:
        Whether the critical restore failed (and possibly fell back).
    """

    observation: Any = None
    info: Dict[str, Any] = field(default_factory=dict)
    from_critical: bool = False
    mode: str = "default"
    critical: Optional[Any] = None
    rand_num: float = float("nan")
    p: float = DEFAULT_P
    trajectory_length: int = 0
    importance: float = float("nan")
    restore_failed: bool = False
    index: int = 0
    seed: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # convenience -----------------------------------------------------------------------
    @property
    def used_default(self) -> bool:
        """True when the standard ``rho`` initial state distribution was used."""
        return not self.from_critical

    @property
    def step_index(self) -> int:
        """Alias of :attr:`trajectory_length` (rollout length used)."""
        return int(self.trajectory_length)

    def to_dict(self) -> Dict[str, Any]:
        """JSON/plain-dict representation (no raw numpy arrays)."""
        crit_dict = None
        if self.critical is not None and hasattr(self.critical, "to_dict"):
            try:
                crit_dict = self.critical.to_dict(include_scores=False)
            except TypeError:  # pragma: no cover - signature guard
                crit_dict = self.critical.to_dict()
            except Exception:  # pragma: no cover
                crit_dict = None
        return {
            "from_critical": bool(self.from_critical),
            "mode": self.mode,
            "rand_num": float(self.rand_num),
            "p": float(self.p),
            "trajectory_length": int(self.trajectory_length),
            "importance": float(self.importance),
            "restore_failed": bool(self.restore_failed),
            "index": int(self.index),
            "seed": self.seed,
            "critical": crit_dict,
            "extra": dict(self.extra),
        }


# --------------------------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------------------------
def mixture_weights(beta: float) -> Tuple[float, float]:
    """Return ``(weight_critical, weight_default)`` for the mixture ``mu``."""
    beta = float(np.clip(float(beta), 0.0, 1.0))
    return beta, 1.0 - beta


def mixed_init_probability(p: float) -> float:
    """Clamp the reset probability ``p`` / mixture weight ``beta`` to ``[0, 1]``."""
    return float(np.clip(float(p), 0.0, 1.0))


def should_use_critical(rand_num: float, p: float) -> bool:
    """Algorithm 2 branch test: ``RAND_NUM < p`` -> reset to critical state."""
    p = mixed_init_probability(p)
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    return bool(float(rand_num) < p)


def _resolve_k(K: Optional[Any], env: Any = None, fallback: int = 1000) -> int:
    """Resolve the trajectory length ``K`` (defaults to the episode horizon ``T``)."""
    horizon = int(default_k(env, fallback=fallback)) if env is not None else int(fallback)
    if horizon <= 0:
        horizon = int(fallback)
    if K is None:
        return horizon
    if isinstance(K, (int, np.integer)):
        return max(1, int(K))
    try:
        value = float(K)
    except (TypeError, ValueError):
        return horizon
    if 0.0 < value < 1.0:
        # fraction of the episode horizon
        return max(1, int(round(value * horizon)))
    if value >= 1.0:
        return max(1, int(round(value)))
    return horizon


def _unpack_reset(result: Any) -> Tuple[Any, Dict[str, Any], bool]:
    """Normalise a ``reset``/restore return value into ``(obs, info, terminated)``.

    Handles gym ``(obs, info)``, classic ``obs``, and
    ``(obs, info, steps_executed)`` style replay returns.
    """
    obs, info, extra = result, {}, {}
    if isinstance(result, tuple):
        if len(result) >= 2:
            obs, info = result[0], result[1]
        if len(result) >= 3:
            extra = {"third": result[2]}
    if not isinstance(info, dict):
        info = {"info": info}
    info = dict(info)
    info.update({k: v for k, v in extra.items() if k not in info})
    return obs, info, False


# --------------------------------------------------------------------------------------
# The sampler
# --------------------------------------------------------------------------------------
class MixedInitialStateSampler:
    """Samples ``s_0 ~ mu(s) = beta d_rho^pihat(s) + (1 - beta) rho(s)``.

    One :meth:`sample` call corresponds to the branch selection at the top of the
    ``for iteration`` loop of Algorithm 2: with probability ``p`` (= ``beta``) the
    frozen pre-trained policy is rolled for ``K`` steps, the mask network scores
    every visited state, the argmax-importance state is restored as ``s_0``;
    otherwise ``s_0 ~ rho`` (a plain environment reset).

    Parameters
    ----------
    env:
        Environment (plain gym env or ``ResetWrapper`` stack). It is wrapped with
        :func:`rice.envs.reset_wrapper.make_reset_env` when possible so that
        critical states can be restored directly.
    policy:
        The frozen pre-trained policy ``pi`` (SB3 ``predict``, native ``act`` or a
        plain callable).
    mask_net:
        Trained Stage-1 mask network ``~pi_theta``. ``None`` degrades to a
        uniform (random) importance score, permitting the Random-explanation
        baseline to share this sampler.
    p:
        Reset probability threshold (mixture weight ``beta``).
    K:
        Trajectory length used to identify the critical state (``None`` -> ``T``).
    scorer:
        Optional :class:`~rice.explanation.importance.ImportanceScorer`.
    rng:
        ``numpy.random.RandomState`` used for the ``RAND_NUM ~ U(0, 1)`` draw.
        Kept separate from environment/action noise for reproducibility.
    seed:
        Convenience seed used to build ``rng`` when it is not supplied.
    """

    def __init__(
        self,
        env: Any,
        policy: Any = None,
        mask_net: Any = None,
        p: float = DEFAULT_P,
        K: Optional[Any] = None,
        env_id: Optional[str] = None,
        deterministic_policy: bool = False,
        scorer: Any = None,
        rng: Optional[np.random.RandomState] = None,
        seed: Optional[int] = None,
        logger: Any = None,
        config: Optional[Any] = None,
        use_direct_restore: bool = True,
        fallback_to_default: bool = True,
        refresh_critical: bool = True,
        cache_critical: bool = False,
        reset_kwargs: Optional[Dict[str, Any]] = None,
        selector: Any = None,
        **kwargs: Any,
    ) -> None:
        cfg = None
        if isinstance(config, MixedInitConfig):
            cfg = config
        elif isinstance(config, dict):
            cfg = MixedInitConfig.from_dict(config)
        if cfg is not None:
            p = kwargs.pop("p", cfg.p)
            K = cfg.K if K is None else K
            env_id = cfg.env_id if env_id is None else env_id
            deterministic_policy = cfg.deterministic_policy
            use_direct_restore = cfg.use_direct_restore
            fallback_to_default = cfg.fallback_to_default
            refresh_critical = cfg.refresh_critical
            cache_critical = cfg.cache_critical
            seed = cfg.seed if seed is None else seed

        self.env = self._prepare_env(env)
        self.policy = policy
        self.mask_net = mask_net
        self.env_id = env_id
        self.deterministic_policy = bool(deterministic_policy)
        self.p = mixed_init_probability(p)
        self.K = K
        self.use_direct_restore = bool(use_direct_restore)
        self.fallback_to_default = bool(fallback_to_default)
        self.refresh_critical = bool(refresh_critical)
        self.cache_critical = bool(cache_critical)
        self.reset_kwargs: Dict[str, Any] = dict(reset_kwargs or {})
        self.logger = logger if logger is not None else get_logger("rice")

        self.rng = rng if rng is not None else get_rng(seed)
        self.seed = seed

        # Build / accept the critical-state selector (Algorithm 2 does its own rollout).
        if selector is not None:
            self.selector = selector
        elif CriticalStateSelector is not None:
            try:
                self.selector = CriticalStateSelector(
                    env=self.env,
                    mask_net=self.mask_net,
                    K=K,
                    scorer=scorer,
                    deterministic_policy=self.deterministic_policy,
                    rng=self.rng,
                    device=kwargs.get("device"),
                    attach_scores=kwargs.get("attach_scores", True),
                )
            except Exception:  # pragma: no cover - heterogeneous signatures
                self.selector = None
        else:  # pragma: no cover
            self.selector = None

        # bookkeeping
        self.draws: List[float] = []
        self.critical_count = 0
        self.default_count = 0
        self.failure_count = 0
        self._cached_critical: Optional[Any] = None
        self.last_sample: Optional[InitSample] = None
        self._sample_index = 0

    # ---- setup -----------------------------------------------------------------------
    @staticmethod
    def _prepare_env(env: Any) -> Any:
        """Wrap the environment with the Go-Explore style reset wrapper if needed."""
        if env is None:
            return None
        if ResetWrapper is not None and isinstance(env, ResetWrapper):
            return env
        try:
            return make_reset_env(env)
        except Exception:  # pragma: no cover
            return env

    # ---- distribution helpers --------------------------------------------------------
    def mixture_weights(self) -> Tuple[float, float]:
        """``(beta, 1 - beta)`` weights of ``d_rho^pihat(s)`` and ``rho(s)``."""
        return mixture_weights(self.p)

    @property
    def beta(self) -> float:
        """Mixture weight ``beta`` (equals the reset probability ``p``)."""
        return float(self.p)

    def set_p(self, p: float) -> float:
        """Update the reset probability / mixture weight and return the new value."""
        self.p = mixed_init_probability(p)
        return self.p

    set_probability = set_p

    def draw(self, rng: Optional[np.random.RandomState] = None) -> float:
        """Draw ``RAND_NUM ~ U(0, 1)`` with the sampler's own RNG."""
        rng = rng if rng is not None else self.rng
        if rng is None:  # pragma: no cover
            value = float(np.random.uniform(0.0, 1.0))
        else:
            value = float(rng.uniform(0.0, 1.0))
        self.draws.append(value)
        return value

    def will_reset_to_critical(self, rand_num: float) -> bool:
        """Apply the Algorithm 2 branch test to a given ``RAND_NUM``."""
        return should_use_critical(rand_num, self.p)

    # ---- default branch: s_0 ~ rho ----------------------------------------------------
    def default_reset(
        self,
        reset_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Sample ``s_0 ~ rho`` via a standard environment reset."""
        kwargs = dict(self.reset_kwargs)
        kwargs.update(reset_kwargs or {})
        if seed is not None:
            kwargs.setdefault("seed", int(seed))
        try:
            result = self.env.reset(**kwargs)
        except TypeError:
            # environment may not accept keyword arguments
            result = self.env.reset()
        obs, info, _ = _unpack_reset(result)
        return obs, info

    # ---- critical branch: s_0 ~ d_rho^pihat ------------------------------------------
    def roll_for_critical(
        self,
        policy: Any = None,
        K: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Any, Any]:
        """Roll the frozen policy for ``K`` steps and return ``(critical_state, rollout)``."""
        policy = policy if policy is not None else self.policy
        k = _resolve_k(K if K is not None else self.K, self.env)
        rollout = None
        critical = None

        if self.selector is not None:
            try:
                rollout = self.selector.rollout(
                    policy=policy, K=k, reset=True, seed=seed
                )
                critical = self.selector.select(
                    policy=policy, K=k, reset=False, seed=seed, rollout=rollout
                )
            except TypeError:  # pragma: no cover - older selector signature
                rollout = self.selector.rollout(policy, k)
                critical = self.selector.select(policy, k)
            except Exception:  # pragma: no cover
                rollout, critical = None, None

        if (rollout is None or critical is None) and roll_trajectory is not None:
            rollout = roll_trajectory(
                self.env,
                policy,
                length=k,
                reset=True,
                deterministic=self.deterministic_policy,
                seed=seed,
            )
            critical = critical_state_from_rollout(
                rollout, mask_net=self.mask_net, scorer=getattr(self.selector, "scorer", None)
            )
        if critical is None:  # pragma: no cover - nothing else available
            raise RuntimeError("unable to identify a critical state (no selector/rollout)")
        return critical, rollout

    def _restore_critical(
        self,
        critical: Any,
        reset_kwargs: Optional[Dict[str, Any]] = None,
        use_direct: Optional[bool] = None,
    ) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """Install ``s_0 <- s_critical``; returns ``(obs, info)`` or ``None`` on failure."""
        kwargs = dict(self.reset_kwargs)
        kwargs.update(reset_kwargs or {})
        direct = self.use_direct_restore if use_direct is None else bool(use_direct)
        env = self.env

        # 1) restore through the Go-Explore style reset wrapper (direct state injection first)
        if env is not None and hasattr(env, "reset_to_state"):
            payload = None
            if hasattr(critical, "restore_payload"):
                try:
                    payload = critical.restore_payload
                except Exception:  # pragma: no cover
                    payload = None
            if payload is not None:
                try:
                    result = env.reset_to_state(payload, **kwargs)
                    obs, info, _ = _unpack_reset(result)
                    return obs, info
                except Exception:  # pragma: no cover
                    pass
            if hasattr(env, "reset_to"):
                try:
                    result = env.reset_to(
                        critical, use_direct=direct, reset_kwargs=kwargs or None
                    )
                    obs, info, _ = _unpack_reset(result)
                    return obs, info
                except Exception:  # pragma: no cover
                    pass
            if hasattr(env, "reset_to_critical"):
                try:
                    result = env.reset_to_critical(critical, reset_kwargs=kwargs or None)
                    obs, info, _ = _unpack_reset(result)
                    return obs, info
                except Exception:  # pragma: no cover
                    pass

        # 2) selector-provided restore helper
        if self.selector is not None and hasattr(self.selector, "reset_to"):
            try:
                result = self.selector.reset_to(env, critical, **kwargs)
                obs, info, _ = _unpack_reset(result)
                return obs, info
            except Exception:  # pragma: no cover
                pass

        # 3) raw simulator state injection
        if env is not None and critical is not None and getattr(critical, "state", None) is not None:
            packed = getattr(critical, "state")
            if isinstance(packed, dict) and "kind" in packed:
                payload = packed
            else:
                payload = {"kind": "sim", "state": packed}
            try:
                if set_env_state(env, payload):
                    result = env.reset()
                    obs, info, _ = _unpack_reset(result)
                    return obs, info
            except Exception:  # pragma: no cover
                pass

        # 4) last resort: action replay of the prefix that led to the critical state
        if env is not None and hasattr(env, "reset_to") and getattr(critical, "actions", None) is not None:
            try:
                result = env.reset_to(critical, use_direct=False, reset_kwargs=kwargs or None)
                obs, info, _ = _unpack_reset(result)
                return obs, info
            except Exception:  # pragma: no cover
                pass

        return None

    def critical_start(
        self,
        policy: Any = None,
        K: Optional[Any] = None,
        reset_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        critical: Any = None,
        rollout: Any = None,
        reuse_cache: Optional[bool] = None,
    ) -> Tuple[Any, Dict[str, Any], Any, Any, bool]:
        """Identify (or reuse) the critical state and install it as ``s_0``.

        Returns ``(obs, info, critical_state, rollout, restore_failed)``.
        """
        reuse = self.cache_critical if reuse_cache is None else bool(reuse_cache)
        if critical is None and reuse and self._cached_critical is not None:
            critical = self._cached_critical
        if critical is None:
            critical, rollout = self.roll_for_critical(policy=policy, K=K, seed=seed)
            if self.cache_critical:
                self._cached_critical = critical

        result = self._restore_critical(critical, reset_kwargs=reset_kwargs)
        restore_failed = result is None
        if result is None:
            self.failure_count += 1
            obs, info = self.default_reset(reset_kwargs=reset_kwargs, seed=seed)
            obs = obs  # explicit: fallback observation
        else:
            obs, info = result

        try:
            attach_critical_state(self.env, critical)
        except Exception:  # pragma: no cover
            pass
        return obs, info, critical, rollout, restore_failed

    # ---- the Algorithm 2 branch -------------------------------------------------------
    def sample(
        self,
        policy: Any = None,
        K: Optional[Any] = None,
        reset_kwargs: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.RandomState] = None,
        seed: Optional[int] = None,
        update_stats: bool = True,
        p: Optional[float] = None,
    ) -> InitSample:
        """Sample one initial state following the Algorithm 2 reset rule.

        ``RAND_NUM ~ U(0, 1)``; if ``RAND_NUM < p`` reset to the mask-identified
        critical state (``s_0 ~ d_rho^pihat``), otherwise ``s_0 ~ rho``.
        """
        if p is not None:
            self.set_p(p)
        rand_num = self.draw(rng=rng)
        use_critical = self.will_reset_to_critical(rand_num)

        sample = InitSample(
            rand_num=rand_num,
            p=float(self.p),
            from_critical=bool(use_critical),
            mode="critical" if use_critical else "default",
            index=self._sample_index,
            seed=seed,
        )

        if not use_critical:
            obs, info = self.default_reset(reset_kwargs=reset_kwargs, seed=seed)
            sample.observation, sample.info = obs, info
            sample.from_critical = False
            sample.mode = "default"
        else:
            try:
                obs, info, critical, rollout, restore_failed = self.critical_start(
                    policy=policy,
                    K=K,
                    reset_kwargs=reset_kwargs,
                    seed=seed,
                )
                sample.observation, sample.info = obs, info
                sample.critical = critical
                sample.restore_failed = bool(restore_failed)
                sample.trajectory_length = int(len(rollout)) if rollout is not None and hasattr(rollout, "__len__") else _resolve_k(K if K is not None else self.K, self.env)
                sample.importance = float(getattr(critical, "score", float("nan")))
                if restore_failed:
                    sample.mode = "fallback"
                    sample.from_critical = False
                    if not self.fallback_to_default:
                        raise RuntimeError(
                            "failed to restore the critical state and fallback_to_default=False"
                        )
            except Exception as exc:  # pragma: no cover - robustness path
                if not self.fallback_to_default:
                    raise
                self.logger.warning("mixed_init critical reset failed (%s); using rho", exc)
                obs, info = self.default_reset(reset_kwargs=reset_kwargs, seed=seed)
                sample.observation, sample.info = obs, info
                sample.from_critical = False
                sample.mode = "fallback"
                sample.restore_failed = True

        if update_stats:
            if sample.from_critical:
                self.critical_count += 1
            else:
                if sample.mode == "fallback":
                    pass
                self.default_count += 1
            self._sample_index += 1
        self.last_sample = sample
        return sample

    # Convenience aliases -----------------------------------------------------------------
    reset = sample

    def sample_observation(self, **kwargs: Any) -> Any:
        """Return only the sampled initial observation (handy inside the PPO loop)."""
        return self.sample(**kwargs).observation

    def next_observation(self, **kwargs: Any) -> Any:
        """Alias of :meth:`sample_observation`."""
        return self.sample_observation(**kwargs)

    # ---- introspection ------------------------------------------------------------------
    @property
    def critical_fraction(self) -> float:
        """Empirical fraction of samples that started from a critical state."""
        total = self.critical_count + self.default_count
        return float(self.critical_count / total) if total else 0.0

    def statistics(self) -> Dict[str, Any]:
        """Diagnostics for logging (branch counts, mean draw, mixture weights)."""
        total = self.critical_count + self.default_count
        beta, one_minus = self.mixture_weights()
        return {
            "p": float(self.p),
            "beta": float(beta),
            "one_minus_beta": float(one_minus),
            "K": self.K,
            "samples": int(total),
            "critical_samples": int(self.critical_count),
            "default_samples": int(self.default_count),
            "critical_fraction": self.critical_fraction,
            "restore_failures": int(self.failure_count),
            "mean_rand_num": float(np.mean(self.draws)) if self.draws else float("nan"),
        }

    stats = statistics

    def reset_statistics(self) -> None:
        """Clear counters (keeps the cached critical state and RNG state)."""
        self.draws = []
        self.critical_count = 0
        self.default_count = 0
        self.failure_count = 0
        self._sample_index = 0

    def describe(self) -> str:
        """Human-readable one-liner summarizing the mixture."""
        return describe_mixed_init(self)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        beta, one_minus = self.mixture_weights()
        return (
            f"MixedInitialStateSampler(env_id={self.env_id!r}, p={self.p:.3f}, "
            f"beta={beta:.3f}, K={self.K})"
        )


# --------------------------------------------------------------------------------------
# Module-level convenience APIs
# --------------------------------------------------------------------------------------
def make_mixed_init_sampler(
    env: Any = None,
    policy: Any = None,
    mask_net: Any = None,
    p: float = DEFAULT_P,
    K: Optional[Any] = None,
    env_id: Optional[str] = None,
    config: Optional[Any] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> MixedInitialStateSampler:
    """Build a :class:`MixedInitialStateSampler` (accepts a config dict/dataclass)."""
    return MixedInitialStateSampler(
        env=env,
        policy=policy,
        mask_net=mask_net,
        p=p,
        K=K,
        env_id=env_id,
        config=config,
        seed=seed,
        **kwargs,
    )


#: Alias kept for symmetry with the other ``build_*`` factories in the project.
build_mixed_init_sampler = make_mixed_init_sampler
build_mixed_init = make_mixed_init_sampler


def sample_initial_state(
    sampler: MixedInitialStateSampler,
    policy: Any = None,
    K: Optional[Any] = None,
    reset_kwargs: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    return_sample: bool = False,
    **kwargs: Any,
) -> Any:
    """Sample ``s_0 ~ mu`` using ``sampler`` (Algorithm 2 reset rule).

    Returns the observation, or the full :class:`InitSample` when
    ``return_sample=True``.
    """
    sample = sampler.sample(
        policy=policy, K=K, reset_kwargs=reset_kwargs, seed=seed, **kwargs
    )
    return sample if return_sample else sample.observation


def describe_mixed_init(sampler: Any) -> str:
    """Format the mixture definition and the sampler state for logging."""
    if sampler is None:
        return "mu(s) = beta d_rho^pihat(s) + (1 - beta) rho(s) [no sampler]"
    beta, one_minus = mixture_weights(getattr(sampler, "p", DEFAULT_P))
    stats = sampler.statistics() if hasattr(sampler, "statistics") else {}
    K = getattr(sampler, "K", None)
    return (
        "mu(s) = {beta:.3f} d_rho^pihat(s) + {one:.3f} rho(s) | "
        "p={p:.3f}, K={K}, samples={n} (critical={c}, default={d}, failures={f})"
    ).format(
        beta=beta,
        one=one_minus,
        p=stats.get("p", getattr(sampler, "p", DEFAULT_P)),
        K=K,
        n=stats.get("samples", 0),
        c=stats.get("critical_samples", 0),
        d=stats.get("default_samples", 0),
        f=stats.get("restore_failures", 0),
    )

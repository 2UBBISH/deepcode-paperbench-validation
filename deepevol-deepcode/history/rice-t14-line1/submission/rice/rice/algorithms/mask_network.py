"""Mask network (the alternative / simplified StateMask) -- CORE COMPONENT #1.

This module implements the step-level explanation of RICE, i.e. the redesigned
StateMask mask network described in Section 3.3 "Technique Detail" (part 1) and
its training procedure (Algorithm 1).

Paper facts implemented here (verbatim from Section 3.3):

  Masking rule, Eq. (1)::

      a_t \odot a_t^m = { a_t          if a_t^m = 0
                        { a_random     if a_t^m = 1

  Original objective, Eq. (2)::

      J(theta) = min |eta(pi) - eta(pi_bar)|

  Theorem 3.3:  eta(pi_bar) <= eta(pi).
  Reformulated objective::

      J(theta) = max eta(pi_bar)

  Blinding bonus reward::

      R'(s_t, a_t) = R(s_t, a_t) + alpha * a_t^m

  Perturbed policy mixture (Appendix A)::

      pi_bar(.|s) = pi_tilde(a^e=0|s) pi(.|s) + pi_tilde(a^e=1|s) pi^r(.|s)

Because the perturbed policy is a mixture with a *random* policy, the advantage
of the random branch is non-positive under Assumption 3.1, hence maximizing
eta(pi_bar) is what drives the mask network to blind exactly those states whose
random action does not destroy the return.  As the authors note, this naive
objective has the trivial solution "never blind" (always output 0), so an extra
bonus ``alpha * a_t^m`` is added to the reward.  Note that ``pi_tilde(a^e=0|s)``
is exactly our mask network's probability of emitting ``1`` (it selects the
random branch), while ``pi_tilde(a^e=1|s)`` is the probability of emitting ``0``.

Importance score (Section 3.3): "the probability of mask network outputting
'0'" -- see :mod:`rice.algorithms.critical_state`.

Algorithm 1 (verbatim):

    Input: Target agent's policy pi
    Output: Mask network pi_tilde_theta
    Initialization: Initialize the weights theta for the mask net pi_tilde_theta
    theta_old <- theta
    for iteration = 1, 2, ... do
        Set the initial state s_0 ~ rho
        D <- empty
        for t = 0 to T do
            Sample a_t ~ pi(a_t | s_t)
            Sample a_t^m ~ pi_tilde_{theta_old}(a_t^m | s_t)
            Compute the actual taken action a <- a_t \odot a_t^m
            (s_{t+1}, R'_t) <- env.step(a) and record (s_t, s_{t+1}, a_t^m, R'_t) in D
        end for
        update theta_old <- theta using D by PPO algorithm
    end for

Architecture (addendum "Architectures"): the mask network mirrors the target
agent's architecture.  For MuJoCo agents this is the SB3 default ``MlpPolicy``
(``[64, 64]`` tanh, see :class:`rice.algorithms.ppo.PPOConfig`), for Selfish
Mining ``[128, 128, 128, 128]``, for CAGE Challenge 2 ``[64, 64, 64]``.

Only ``alpha`` is given by the paper (Table 3: 0.0001 for every environment;
Section C.3 text disagrees and says 0.01 -- we follow Table 3 per the addendum).
Every other PPO hyper-parameter is unspecified, so we reuse our SB3-default
:class:`~rice.algorithms.ppo.PPOConfig`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .ppo import (
    ActorCritic,
    PPO,
    PPOConfig,
    RolloutBuffer,
    action_size,
    flatten_obs,
    make_target_policy_callable,
    observation_size,
    resolve_device,
)

__all__ = [
    "MaskNetwork",
    "MaskNetworkConfig",
    "MaskNetworkTrainer",
    "masked_action",
    "importance_from_logits",
]


# ---------------------------------------------------------------------------
# Masking rule, Eq. (1)
# ---------------------------------------------------------------------------
def _sample_random_action(action_space, rng: np.random.Generator) -> np.ndarray:
    """Sample ``a_random`` from the action space.

    Eq. (1) replaces the target agent's action with ``a_random`` whenever the
    mask fires.  We sample uniformly from the environment's action space, which
    is the standard "random policy" ``pi^r`` used in StateMask.
    """
    if isinstance(action_space, _DISCRETE_TYPES):
        return np.asarray(action_space.sample(), dtype=np.int64)
    # Box: uniform sample inside the bounds.
    low = np.asarray(action_space.low, dtype=np.float64)
    high = np.asarray(action_space.high, dtype=np.float64)
    if np.any(~np.isfinite(low)) or np.any(~np.isfinite(high)):
        # Unbounded box: fall back to a standard normal, then clip to any
        # finite bounds we do have.
        sample = rng.standard_normal(low.shape).astype(np.float64)
        sample = np.where(np.isfinite(low), np.maximum(sample, low), sample)
        sample = np.where(np.isfinite(high), np.minimum(sample, high), sample)
        return sample.astype(np.float32)
    u = rng.uniform(low, high)
    return np.asarray(u, dtype=np.float32)


def masked_action(
    a_t: np.ndarray,
    a_t_m: int,
    action_space,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the masking rule of Eq. (1): ``a_t \odot a_t^m``.

    Parameters
    ----------
    a_t:
        Action sampled from the target agent's policy ``pi(.|s_t)``.
    a_t_m:
        Binary mask action, ``0`` (keep) or ``1`` (blind).
    action_space:
        Gym/gymnasium action space, used to draw ``a_random``.
    rng:
        NumPy random generator (reproducibility).

    Returns
    -------
    numpy.ndarray
        ``a_t`` if ``a_t_m == 0`` else a uniform random action.
    """
    if int(a_t_m) == 0:
        return np.asarray(a_t)
    return _sample_random_action(action_space, rng)


def importance_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return ``P(mask = 0 | s)`` -- the paper's importance score.

    The mask network is a categorical distribution over ``{0, 1}``; the
    importance of a state is defined in Section 3.3 as "the probability of mask
    network outputting ' 0 '".
    """
    probs = torch.softmax(logits, dim=-1)
    return probs[..., 0]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class MaskNetwork(nn.Module):
    """Binary mask network ``pi_tilde_theta(a^m | s)``.

    The network mirrors the target agent's architecture (addendum).  It emits a
    ``Discrete(2)`` logits vector per state: ``a^m = 0`` means "keep the target
    action", ``a^m = 1`` means "blind the agent with a random action".

    Parameters
    ----------
    observation_space:
        Environment observation space (Box or Dict).
    net_arch:
        Hidden layer sizes of the shared MLP trunk.  Use the target agent's
        architecture (``(64, 64)`` for MuJoCo/SB3 MlpPolicy, ``(128,)*4`` for
        Selfish Mining, ``(64,)*3`` for CAGE).
    activation:
        Activation name (``"tanh"`` for SB3 parity).
    obs_dim:
        Optional explicit observation dimension (useful for Dict spaces whose
        flattened size must be known at construction time).
    device:
        ``"auto"``, ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        observation_space,
        net_arch: Sequence[int] = (64, 64),
        activation: str = "tanh",
        obs_dim: Optional[int] = None,
        log_std_init: float = 0.0,
        ortho_init: bool = True,
        device: str = "auto",
    ) -> None:
        super().__init__()
        from .ppo import Discrete as _Discrete  # local import keeps ppo importable w/o gym

        self.observation_space = observation_space
        # A dummy Discrete(2) action space lets us reuse the shared ActorCritic.
        self.action_space = _Discrete(2)
        self.obs_dim = int(obs_dim) if obs_dim is not None else observation_size(observation_space)
        self.device = resolve_device(device)
        self._net = ActorCritic(
            observation_space=observation_space,
            action_space=self.action_space,
            net_arch=tuple(int(h) for h in net_arch),
            activation=activation,
            log_std_init=log_std_init,
            ortho_init=ortho_init,
            obs_dim=self.obs_dim,
            device=device,
        )
        self.to(self.device)

    # -- inference ---------------------------------------------------------
    def forward(self, obs) -> torch.Tensor:
        """Return the raw logits over ``{a^m = 0, a^m = 1}``."""
        obs_t = self._as_tensor(obs)
        return self._net.forward(obs_t)

    def _as_tensor(self, obs) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs.to(self.device).float()
        arr = flatten_obs(obs)
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)

    def distribution(self, obs):
        """Categorical distribution over the binary mask action."""
        return self._net.get_distribution(self._as_tensor(obs))

    def mask_prob_zero(self, obs) -> float:
        """Importance score ``P(a^m = 0 | s)`` as a Python float."""
        with torch.no_grad():
            logits = self.forward(obs)
            return float(importance_from_logits(logits).reshape(-1)[0].item())

    def importance(self, obs) -> float:
        """Alias of :meth:`mask_prob_zero` (paper terminology)."""
        return self.mask_prob_zero(obs)

    def act(
        self,
        obs,
        deterministic: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[int, float, float]:
        """Sample ``a^m ~ pi_tilde_theta(.|s)`` (Algorithm 1, line 5).

        Returns ``(a_t_m, value, log_prob)``.
        """
        with torch.no_grad():
            obs_t = self._as_tensor(obs)
            dist = self._net.get_distribution(obs_t)
            if deterministic:
                a = torch.argmax(dist.logits, dim=-1).reshape(-1)
            else:
                a = dist.sample().reshape(-1)
            log_prob = dist.log_prob(a).reshape(-1)
            value = self._net.predict_values(obs_t).reshape(-1)
        return int(a[0].item()), float(value[0].item()), float(log_prob[0].item())

    def policy_state_dict(self) -> Dict[str, torch.Tensor]:
        return self._net.state_dict()

    def load_policy_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self._net.load_state_dict(state)

    def feature_dim(self) -> int:
        return self._net.obs_dim


_DISCRETE_TYPES: Tuple[type, ...] = ()
try:  # gym
    from gym.spaces import Discrete as _GymDiscrete

    _DISCRETE_TYPES += (_GymDiscrete,)
except Exception:  # pragma: no cover
    pass
try:  # gymnasium
    from gymnasium.spaces import Discrete as _GymnasiumDiscrete

    _DISCRETE_TYPES += (_GymnasiumDiscrete,)
except Exception:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Config & trainer (Algorithm 1)
# ---------------------------------------------------------------------------
@dataclass
class MaskNetworkConfig:
    """Hyper-parameters of Algorithm 1.

    ``alpha`` is the only paper-specified value (Table 3: 0.0001 for all
    environments; Section C.3 text says 0.01 -- we follow Table 3).  Everything
    else is taken from the SB3 defaults via :class:`PPOConfig`.
    """

    alpha: float = 1e-4
    net_arch: Tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    n_iterations: int = 300
    max_steps_per_iter: Optional[int] = None  # None -> T of the environment
    total_samples: Optional[int] = None  # preferred hard budget (Table 4)
    policy_config: PPOConfig = field(default_factory=PPOConfig)
    device: str = "auto"
    seed: Optional[int] = None
    verbose: int = 0
    log_every: int = 10


class MaskNetworkTrainer:
    """Trains the mask network with vanilla PPO on the bonus reward.

    Implements Algorithm 1 exactly:

    1. ``s_0 ~ rho`` (default initial distribution).
    2. Roll out the target policy ``pi``, sample the mask action
       ``a_t^m ~ pi_tilde_{theta_old}(.|s_t)`` from the *old* mask net, apply the
       masking rule of Eq. (1) and step the environment.
    3. Record ``(s_t, s_{t+1}, a_t^m, R'_t)`` with
       ``R'_t = R_t + alpha * a_t^m``.
    4. Update ``theta_old <- theta`` using PPO on ``D``.
    """

    def __init__(
        self,
        env: Any,
        target_policy: Any,
        mask_network: MaskNetwork,
        config: Optional[MaskNetworkConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.env = env
        self.config = config or MaskNetworkConfig()
        self.mask_network = mask_network
        self.device = resolve_device(self.config.device)

        # The target agent is treated as a black box: any of SB3 model /
        # torch module / plain callable is accepted.
        self.target_policy: Callable[[Any], np.ndarray] = make_target_policy_callable(target_policy)

        self.rng = rng if rng is not None else np.random.default_rng(self.config.seed)

        self.ppo = PPO(
            policy=mask_network._net,
            config=self.config.policy_config,
            device=self.device,
        )

        self.mask_history: List[float] = []  # mean mask rate (a^m == 1) per iteration
        self.reward_history: List[float] = []
        self.sample_count = 0
        self.train_seconds = 0.0

    # -- helpers -----------------------------------------------------------
    def _max_steps(self) -> int:
        if self.config.max_steps_per_iter is not None:
            return int(self.config.max_steps_per_iter)
        spec = getattr(self.env, "spec", None)
        if spec is not None and getattr(spec, "max_episode_steps", None):
            return int(spec.max_episode_steps)
        # Fall back to the gym TimeLimit wrapper if present.
        for attr in ("_max_episode_steps", "max_episode_steps"):
            value = getattr(self.env, attr, None)
            if value:
                return int(value)
        return 1000

    @staticmethod
    def _reset_env(env, seed: Optional[int] = None):
        try:
            out = env.reset(seed=seed) if seed is not None else env.reset()
        except TypeError:
            if seed is not None:
                try:
                    env.seed(seed)
                except Exception:
                    pass
            out = env.reset()
        # gymnasium returns (obs, info)
        if isinstance(out, tuple):
            return out[0]
        return out

    # -- main loop (Algorithm 1) -------------------------------------------
    def train(self) -> Dict[str, Any]:
        cfg = self.config
        start_time = time.time()
        iteration = 0
        while iteration < cfg.n_iterations:
            if cfg.total_samples is not None and self.sample_count >= cfg.total_samples:
                break
            iteration += 1
            buffer, stats = self.collect_iteration()
            if len(buffer) == 0:
                continue
            metrics = self.ppo.update(buffer, last_values=0.0, progress=iteration / max(cfg.n_iterations, 1))
            self.mask_history.append(stats["mask_rate"])
            self.reward_history.append(stats["mean_reward"])
            if cfg.verbose and (iteration % cfg.log_every == 0 or iteration == 1):
                print(
                    f"[mask-net] iter {iteration:4d} samples={self.sample_count:8d} "
                    f"mask_rate={stats['mask_rate']:.3f} R={stats['mean_reward']:.3f} "
                    f"loss={metrics.get('loss', float('nan')):.4f}"
                )
        self.train_seconds = time.time() - start_time
        return {
            "iterations": iteration,
            "samples": self.sample_count,
            "seconds": self.train_seconds,
            "mask_history": list(self.mask_history),
            "mean_mask_rate": float(np.mean(self.mask_history)) if self.mask_history else 0.0,
        }

    # -- one Algorithm-1 iteration -----------------------------------------
    def collect_iteration(self) -> Tuple[RolloutBuffer, Dict[str, float]]:
        """Roll out the *old* mask net and build the PPO buffer ``D``."""
        cfg = self.config
        self.mask_network.eval()
        buffer = RolloutBuffer()
        obs = self._reset_env(self.env)
        max_steps = self._max_steps()
        n_masked = 0
        rewards: List[float] = []

        # theta_old: the mask used for data collection is the one we are about to
        # update; PPO recomputes log-probs of theta (they coincide at the first
        # epoch, which is exactly the on-policy PPO setting).
        for _ in range(max_steps):
            if cfg.total_samples is not None and self.sample_count >= cfg.total_samples:
                break
            # Sample a_t ~ pi(a_t | s_t)   (target agent: frozen black box)
            a_t = np.asarray(self.target_policy(obs))
            # Sample a_t^m ~ pi_tilde_{theta_old}(a_t^m | s_t)
            a_t_m, value, log_prob = self.mask_network.act(obs, rng=self.rng)
            # a <- a_t \odot a_t^m   (Eq. 1)
            action = masked_action(a_t, a_t_m, self.env.action_space, self.rng)

            step_out = self.env.step(action)
            if len(step_out) == 5:  # gymnasium
                next_obs, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:
                next_obs, reward, done, _ = step_out
                done = bool(done)

            # R'(s_t, a_t) = R(s_t, a_t) + alpha * a_t^m
            bonus_reward = float(reward) + cfg.alpha * float(a_t_m)

            buffer.add(
                obs=flatten_obs(obs),
                action=a_t_m,
                reward=bonus_reward,
                next_obs=flatten_obs(next_obs),
                done=done,
                value=value,
                log_prob=log_prob,
            )
            self.sample_count += 1
            n_masked += int(a_t_m)
            rewards.append(float(reward))

            obs = next_obs
            if done:
                obs = self._reset_env(self.env)

        self.mask_network.train()
        stats = {
            "mask_rate": float(n_masked / max(len(rewards), 1)),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        }
        return buffer, stats

    # -- importance --------------------------------------------------------
    def importance(self, obs) -> float:
        """P(mask = 0 | s): the state-importance score of Section 3.3."""
        return self.mask_network.mask_prob_zero(obs)


def resolve_device_local(device: str) -> torch.device:  # pragma: no cover - alias
    return resolve_device(device)

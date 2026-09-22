r"""RICE refining loop (CORE COMPONENT #4): Algorithm 2 of the RICE paper.

Paper: "RICE: A Refining scheme for ReInForcement learning with Explanation"
(Proc. 41st ICML, PMLR 235, 2024).  Source: §3.3 Technique Detail, Algorithm 2.

Algorithm 2 (verbatim from the paper)
-------------------------------------
    Algorithm 2 Refining the DRL Agent.
        Input: Pre-trained policy :math:`\pi`, corresponding state mask
               :math:`\tilde{\pi}`, default initial state distribution :math:`\rho`,
               reset probability threshold :math:`p`
        Output: The agent's policy after refining :math:`\pi^{\prime}`
        for iteration = 1, 2, ... do
            :math:`\mathcal{D} \leftarrow \emptyset`
            RAND_NUM :math:`\leftarrow \operatorname{RAND}(0,1)`
            if RAND_NUM < p then
                Run :math:`\pi` to obtain a trajectory :math:`\tau` of length :math:`K`
                Identify the most critical state :math:`s_t` in :math:`\tau` via state mask :math:`\tilde{\pi}`
                Set the initial state :math:`s_0 \leftarrow s_t`
            else
                Set the initial state :math:`s_0 \sim \rho`
            end if
            for t = 0 to T do
                Sample :math:`a_t \sim \pi(a_t \mid s_t)`
                :math:`(s_{t+1}, R_t) \leftarrow \operatorname{env.step}(a_t)`
                Calculate RND bonus :math:`R_t^{RND} = \|f(s_{t+1}) - \hat{f}(s_{t+1})\|^2`
                with normalization
                :math:`\operatorname{Add}(s_t, s_{t+1}, a_t, R_t + \lambda R_t^{RND})` to :math:`\mathcal{D}`
            end for
            Optimize :math:`\pi_\theta` w.r.t PPO loss on :math:`\mathcal{D}`
            Optimize :math:`\hat{f}_\theta` w.r.t. MSE loss on :math:`\mathcal{D}` using Adam
        end for
        :math:`\pi^{\prime} \leftarrow \pi_\theta`

Augmented reward (§3.3, verbatim):
    :math:`R^{\prime}(s_t, a_t) = R(s_t, a_t) + \lambda |f(s_{t+1}) - \hat{f}(s_{t+1})|^2`

Implementation notes / gotchas honoured here
--------------------------------------------
* ``RAND_NUM`` is drawn **once per outer iteration** — the whole episode uses a single
  :math:`s_0`.  This is delegated to :class:`~rice.algorithms.mixed_init.MixedInitSampler`,
  whose ``sample()`` performs exactly one Bernoulli(p) draw per call.
* Roll-out actions are sampled from :math:`\pi` (the *refining* policy, **not** a frozen
  copy of the pre-trained policy).
* :math:`\hat{f}` is updated with MSE via Adam once per outer iteration.
* PPO loss: standard clipped surrogate (Schulman et al. 2017).  The paper does not specify
  the PPO hyper-parameters, so Stable-Baselines3 PPO defaults are used
  (:class:`~rice.algorithms.ppo.PPOConfig`).  The paper also does not specify ``K`` (roll-in
  trajectory length) nor the refining budget: ``K`` defaults to one full pre-trained-policy
  episode and ``T`` defaults to the environment's ``max_episode_steps``.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - torch is a hard dependency in practice
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False

from .ppo import (
    ActorCritic,
    PPO,
    PPOConfig,
    RolloutBuffer,
    flatten_obs,
    observation_size,
)
from .rnd import RND, RNDConfig, make_rnd
from .critical_state import max_episode_steps
from .mixed_init import MixedInitSampler, MixedInitSample, make_mixed_init_sampler

__all__ = [
    "RefineConfig",
    "RefineIteration",
    "RefineResult",
    "RICERefiner",
    "refine_policy",
    "make_refiner",
    "evaluate_policy",
    "load_policy_weights",
]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _env_reset(env: Any, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
    """gym / gymnasium compatible ``reset`` returning ``(obs, info)``."""
    try:
        out = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        try:
            if seed is not None:
                env.seed(seed)  # type: ignore[attr-defined]
        except Exception:
            pass
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], (out[1] or {})
    return out, {}


def _env_step(env: Any, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    """Normalise 4-tuple (gym) and 5-tuple (gymnasium) ``step`` returns."""
    out = env.step(action)
    if not isinstance(out, tuple):
        raise RuntimeError("env.step must return a tuple")
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated), bool(truncated), (info or {})
    if len(out) == 4:
        obs, reward, done, info = out
        return obs, float(reward), bool(done), False, (info or {})
    raise RuntimeError(f"unexpected env.step return arity: {len(out)}")


def _scalar(x: Any) -> float:
    """Best-effort conversion of a torch tensor / numpy array / python scalar."""
    if x is None:
        return 0.0
    if hasattr(x, "detach"):
        return float(x.detach().cpu().reshape(-1)[0])
    return float(np.asarray(x).reshape(-1)[0])


def _as_obs_array(obs: Any) -> np.ndarray:
    arr = np.asarray(flatten_obs(obs), dtype=np.float32)
    return arr.reshape(1, -1)


# --------------------------------------------------------------------------------------
# configuration containers
# --------------------------------------------------------------------------------------
@dataclass
class RefineConfig:
    """Hyper-parameters of Algorithm 2.

    Parameters
    ----------
    p:
        Reset probability threshold (a.k.a. :math:`\\beta` of the mixed initial state
        distribution :math:`\\mu(s) = \\beta d_\\rho^{\\hat\\pi}(s) + (1-\\beta)\\rho(s)`).
        Table 3: Hopper/Walker2d/Selfish/Auto ``0.25``; Reacher/HalfCheetah/Cage ``0.50``.
    lam:
        Coefficient :math:`\\lambda` of the RND intrinsic reward in
        :math:`R' = R + \\lambda\\,R^{RND}`.  Table 3 values (0.001 / 0.01).
    n_iterations:
        Number of outer refining iterations ("for iteration = 1, 2, ...").
    steps_per_iter:
        ``T`` of Algorithm 2 - number of environment steps collected per iteration.
        ``None`` -> environment's ``max_episode_steps`` (default 1000).
    rollin_length:
        ``K`` of Algorithm 2 - length of the trajectory used to identify the critical
        state.  ``None`` -> one full pre-trained-policy episode.
    total_env_steps:
        Optional global budget; the loop stops early once reached (``None`` = no limit).
    reset_on_done:
        The paper's inner loop runs T steps.  When an episode terminates early (e.g.
        Hopper/Walker2d "unhealthy"), we continue the iteration by resetting ``s_0 ~ rho``
        which keeps ``T`` steps per iteration (documented default; set ``False`` to end the
        collection for that iteration instead).
    """

    p: float = 0.25
    lam: float = 0.001
    n_iterations: int = 500
    steps_per_iter: Optional[int] = None
    rollin_length: Optional[int] = None
    total_env_steps: Optional[int] = None
    reset_on_done: bool = True
    policy_config: PPOConfig = field(default_factory=PPOConfig)
    rnd_config: RNDConfig = field(default_factory=RNDConfig)
    handle_timeout_termination: bool = True
    log_every: int = 10
    eval_every: int = 0
    eval_episodes: int = 5
    device: str = "auto"
    seed: Optional[int] = None
    verbose: int = 1

    def clone(self, **overrides: Any) -> "RefineConfig":
        cfg = copy.deepcopy(self)
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise AttributeError(f"Unknown RefineConfig field: {key}")
            setattr(cfg, key, value)
        return cfg

    @classmethod
    def from_mapping(cls, mapping: Optional[Dict[str, Any]], **overrides: Any) -> "RefineConfig":
        """Build from a (YAML) mapping, accepting ``lambda`` / ``coef`` aliases."""
        cfg = cls()
        if mapping:
            data = dict(mapping)
            for alias in ("lambda_", "lambda", "coef", "intrinsic_coef"):
                if alias in data:
                    data["lam"] = data.pop(alias)
            if "beta" in data and "p" not in data:
                data["p"] = data.pop("beta")
            if "length" in data and "rollin_length" not in data:
                data["rollin_length"] = data.pop("length")
            if "K" in data and "rollin_length" not in data:
                data["rollin_length"] = data.pop("K")
            policy_cfg = data.pop("policy_config", None)
            rnd_cfg = data.pop("rnd_config", None)
            for key, value in data.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
            if isinstance(policy_cfg, dict):
                cfg.policy_config = cfg.policy_config.clone(**policy_cfg)
            if isinstance(rnd_cfg, dict):
                cfg.rnd_config = cfg.rnd_config.clone(**rnd_cfg)
        if overrides:
            cfg = cfg.clone(**overrides)
        return cfg


@dataclass
class RefineIteration:
    """Book-keeping for one outer refining iteration of Algorithm 2."""

    iteration: int
    init_mode: str = "default"          # "critical" | "default"
    rand_num: Optional[float] = None
    critical_importance: Optional[float] = None
    steps: int = 0
    task_return: float = 0.0
    mean_rnd_bonus: float = 0.0
    mean_augmented_reward: float = 0.0
    episodes_finished: int = 0
    episode_returns: List[float] = field(default_factory=list)
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0
    approx_kl: float = 0.0
    rnd_loss: float = 0.0
    seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RefineResult:
    """Output of :meth:`RICERefiner.train` (``pi'`` of Algorithm 2 plus diagnostics)."""

    policy: Any = None
    iterations: int = 0
    env_steps: int = 0
    seconds: float = 0.0
    episode_returns: List[float] = field(default_factory=list)
    mean_episode_return: Optional[float] = None
    final_eval_reward: Optional[float] = None
    critical_fraction: float = 0.0
    log: List[Dict[str, Any]] = field(default_factory=list)
    stopping_reason: str = "iterations"

    def as_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out.pop("policy", None)
        return out

    @property
    def final_reward(self) -> Optional[float]:
        """Final reported reward (evaluation mean if available, else last episode)."""
        if self.final_eval_reward is not None:
            return self.final_eval_reward
        if self.episode_returns:
            return float(self.episode_returns[-1])
        return self.mean_episode_return

    def refining_curve(self, window: int = 1) -> np.ndarray:
        """Moving-average curve of collected episode returns (Figure 2 style)."""
        returns = np.asarray(self.episode_returns, dtype=np.float64)
        if returns.size == 0:
            return returns
        window = max(1, int(window))
        if window == 1:
            return returns
        kernel = np.ones(window, dtype=np.float64) / float(window)
        return np.convolve(returns, kernel, mode="valid")


def load_policy_weights(policy: ActorCritic, source: Any, strict: bool = False) -> ActorCritic:
    """Warm-start ``policy`` (our trainable :class:`ActorCritic`) from a pre-trained model.

    ``source`` may be
      * another :class:`ActorCritic` / ``torch.nn.Module`` (state_dict copied),
      * a ``pathlib``-style path / ``str`` to a ``.pt``/``.zip`` checkpoint,
      * a mapping (``state_dict``),
      * a Stable-Baselines3 model exposing ``policy.state_dict()``.
    The transfer is shape-tolerant: mismatching tensors are skipped and reported.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is required to load policy weights")

    state: Optional[Dict[str, Any]] = None
    if isinstance(source, str):
        obj = torch.load(source, map_location="cpu")
        state = obj.get("policy", obj) if isinstance(obj, dict) else obj
    elif isinstance(source, dict):
        state = source
    elif hasattr(source, "policy") and hasattr(source.policy, "state_dict"):
        # Stable-Baselines3 model
        state = {k: v for k, v in source.policy.state_dict().items()}
    elif hasattr(source, "state_dict"):
        state = source.state_dict()

    if not isinstance(state, dict):
        raise TypeError("could not extract a state_dict from the provided source")

    # Strip common wrappers/prefixes.
    clean: Dict[str, Any] = {}
    for key, value in state.items():
        k = key
        for prefix in ("module.", "policy.", "actor_critic.", "_orig_mod."):
            if k.startswith(prefix):
                k = k[len(prefix) :]
        if k in ("log_std", "policy.log_std"):
            k = "log_std"
        clean[k] = value

    own = dict(policy.state_dict())
    accepted: Dict[str, Any] = {}
    for key, value in clean.items():
        if key in own and hasattr(value, "shape") and own[key].shape == value.shape:
            accepted[key] = value
    missing = [k for k in own if k not in accepted]
    policy.load_state_dict(accepted, strict=False)
    if strict and missing:
        raise RuntimeError(f"warm-start incomplete, missing keys: {missing}")
    return policy


def evaluate_policy(
    env: Any,
    policy: Any,
    n_episodes: int = 5,
    seed: Optional[int] = None,
    deterministic: bool = True,
    max_steps: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Roll ``n_episodes`` full episodes and return mean/std task return.

    Used for the "final reward" column of Experiment II/III (Tables 1/5/6) and for
    the official evaluation of the refined policy ``pi'``.
    """
    returns: List[float] = []
    lengths: List[int] = []
    for episode in range(int(n_episodes)):
        ep_seed = None if seed is None else int(seed) + episode
        obs, _ = _env_reset(env, ep_seed)
        done = False
        ep_ret = 0.0
        steps = 0
        limit = max_steps if max_steps is not None else (1 << 30)
        while not done and steps < limit:
            action = _policy_action(policy, obs, deterministic=deterministic, rng=rng)
            obs, reward, terminated, truncated, _ = _env_step(env, action)
            done = terminated or truncated
            ep_ret += reward
            steps += 1
        returns.append(ep_ret)
        lengths.append(steps)
    arr = np.asarray(returns, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "returns": returns,
    }


def _policy_action(policy: Any, obs: Any, deterministic: bool = False, rng: Any = None) -> np.ndarray:
    """Sample an action from a policy (ActorCritic, SB3 model, or plain callable)."""
    if hasattr(policy, "act"):
        try:
            out = policy.act(obs, deterministic=deterministic)
        except TypeError:
            out = policy.act(obs)
        if isinstance(out, tuple):
            return np.asarray(out[0])
        return np.asarray(out)
    if hasattr(policy, "predict"):
        out = policy.predict(obs, deterministic=deterministic)
        if isinstance(out, tuple):
            return np.asarray(out[0])
        return np.asarray(out)
    if callable(policy):
        if rng is not None:
            try:
                return np.asarray(policy(obs, rng=rng))
            except TypeError:
                pass
        return np.asarray(policy(obs))
    raise TypeError("policy is neither an ActorCritic, an SB3 model nor a callable")


# --------------------------------------------------------------------------------------
# the refiner
# --------------------------------------------------------------------------------------
class RICERefiner:
    """Algorithm 2: refine a pre-trained (warm-start) policy with RICE.

    Parameters
    ----------
    env:
        Training environment (gym / gymnasium).  Its default initial-state distribution
        is :math:`\\rho`, sampled with ``env.reset()``.
    policy:
        Trainable :class:`~rice.algorithms.ppo.ActorCritic` initialised with the
        pre-trained (bottlenecked) policy :math:`\\pi`.
    mask_network:
        The trained state mask :math:`\\tilde{\\pi}` from Algorithm 1 (only used to
        identify critical states inside the roll-in trajectory).  May be ``None`` if a
        pre-collected critical-state pool is supplied to ``sampler``.
    config:
        :class:`RefineConfig`.
    sampler:
        Optional pre-built :class:`~rice.algorithms.mixed_init.MixedInitSampler`.
    rnd:
        Optional pre-built :class:`~rice.algorithms.rnd.RND`.
    state_manager:
        Optional :class:`~rice.algorithms.env_reset.EnvStateManager` used to restore the
        identified critical state exactly (Go-Explore style; §C.1).
    """

    def __init__(
        self,
        env: Any,
        policy: ActorCritic,
        mask_network: Any = None,
        config: Optional[RefineConfig] = None,
        sampler: Optional[MixedInitSampler] = None,
        rnd: Optional[RND] = None,
        state_manager: Any = None,
        evaluation_env: Any = None,
        rng: Optional[np.random.Generator] = None,
        logger: Optional[Callable[[RefineIteration], None]] = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.mask_network = mask_network
        self.config = config if config is not None else RefineConfig()
        self.evaluation_env = evaluation_env if evaluation_env is not None else env
        self.state_manager = state_manager
        self.logger = logger

        if self.config.seed is not None:
            self.rng = np.random.default_rng(self.config.seed)
        elif rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

        # ``T`` of Algorithm 2.
        self.steps_per_iter = int(
            self.config.steps_per_iter
            if self.config.steps_per_iter is not None
            else max_episode_steps(env, default=1000)
        )
        if self.steps_per_iter <= 0:
            self.steps_per_iter = 1000

        # ``K`` of Algorithm 2 (roll-in trajectory length).
        self.rollin_length = self.config.rollin_length

        # PPO optimizer over the *refining* policy.
        self.ppo = PPO(
            policy=policy,
            config=self.config.policy_config,
            device=self.config.device,
        )

        # RND module (f frozen, f_hat trained with MSE + Adam).
        obs_dim = None
        obs_space = getattr(env, "observation_space", None)
        if obs_space is not None:
            try:
                obs_dim = observation_size(obs_space)
            except Exception:
                obs_dim = None
        if rnd is not None:
            self.rnd = rnd
        else:
            rnd_cfg = self.config.rnd_config.clone(seed=self.config.seed)
            self.rnd = make_rnd(
                observation_space=obs_space,
                obs_dim=obs_dim,
                config=rnd_cfg,
                device=self.config.device,
            )

        # Mixed initial state distribution sampler (Bernoulli(p) roll-in, once/iteration).
        if sampler is not None:
            self.sampler = sampler
        else:
            self.sampler = MixedInitSampler(
                env=env,
                policy=policy,
                mask_network=mask_network,
                p=self.config.p,
                rollin_length=self.rollin_length,
                state_manager=state_manager,
                mode="rollin" if mask_network is not None else "pool",
                rng=self.rng,
                seed=self.config.seed,
            )
        self.sampler.set_p(self.config.p)

        # book-keeping
        self.iteration = 0
        self.env_steps = 0
        self.episode_returns: List[float] = []
        self.log: List[RefineIteration] = []

    # ------------------------------------------------------------------ utilities
    @property
    def lam(self) -> float:
        """Intrinsic-reward coefficient :math:`\\lambda` (Table 3)."""
        return float(self.config.lam)

    def _ensure_obs_from_sample(self, sample: MixedInitSample) -> Any:
        """Return the observation corresponding to the sampled initial state."""
        if sample.observation is not None:
            return sample.observation
        # Pool-based critical sample whose state could not be written through the env API.
        if sample.needs_manual_reset and self.state_manager is not None and sample.state is not None:
            obs = self.state_manager.set_state(sample.state)
            if obs is None:
                obs = self.state_manager.current_observation()
            return obs
        obs, _ = _env_reset(self.env, seed=None)
        return obs

    # ------------------------------------------------------------------ collection
    def collect_iteration(self) -> Tuple[RolloutBuffer, RefineIteration]:
        """One outer iteration of Algorithm 2: roll-in + T environment steps.

        Returns the rollout buffer ``D`` (storing the *augmented* reward
        :math:`R_t + \\lambda R_t^{RND}`) and the per-iteration log record.
        """
        record = RefineIteration(iteration=self.iteration)
        t_start = time.time()

        # ---- 1. mixed initial state: RAND_NUM drawn ONCE per outer iteration ----
        sample = self.sampler.sample(iteration=self.iteration, reset_env=True)
        record.init_mode = sample.mode
        if self.sampler.decisions:
            record.rand_num = float(self.sampler.decisions[-1])
        if sample.importance is not None:
            record.critical_importance = float(sample.importance)
        obs = self._ensure_obs_from_sample(sample)
        obs = np.asarray(flatten_obs(obs), dtype=np.float32)

        # ---- 2. inner loop: t = 0 .. T ----
        buffer = RolloutBuffer()
        episode_return = 0.0
        bonus_sum = 0.0
        reward_sum = 0.0
        steps_taken = 0

        for _ in range(self.steps_per_iter):
            action, value, log_prob = self.policy.act(obs, deterministic=False)
            action = np.asarray(action)
            next_obs_raw, reward, terminated, truncated, _ = _env_step(self.env, action)
            done = bool(terminated or truncated)

            # RND bonus on s_{t+1}, with normalization (Burda et al. 2018 recipe).
            bonus = float(
                np.asarray(
                    self.rnd.intrinsic_reward(next_obs_raw, update_stats=True)
                ).reshape(-1)[0]
            )
            augmented = float(reward) + self.lam * bonus

            buffer.add(
                obs,
                action,
                augmented,
                next_obs_raw,
                done,
                value=_scalar(value),
                log_prob=_scalar(log_prob),
            )

            episode_return += float(reward)
            bonus_sum += bonus
            reward_sum += augmented
            steps_taken += 1
            self.env_steps += 1
            obs = np.asarray(flatten_obs(next_obs_raw), dtype=np.float32)

            if done:
                self.episode_returns.append(episode_return)
                record.episode_returns.append(episode_return)
                record.episodes_finished += 1
                episode_return = 0.0
                if self.config.reset_on_done:
                    # Continue the T-step collection from the default distribution rho.
                    obs_raw, _ = _env_reset(self.env, seed=None)
                    obs = np.asarray(flatten_obs(obs_raw), dtype=np.float32)
                else:
                    break

        # Flush a partial (unfinished) episode return into the trace for curve plotting.
        if episode_return != 0.0:
            self.episode_returns.append(episode_return)
            record.episode_returns.append(episode_return)
            episode_return = 0.0

        record.steps = steps_taken
        record.task_return = float(reward_sum)
        record.mean_rnd_bonus = float(bonus_sum / max(1, steps_taken))
        record.mean_augmented_reward = float(reward_sum / max(1, steps_taken))

        # ---- 3. optimize pi_theta w.r.t. the PPO loss on D ----
        last_values = self._bootstrap_value(obs)
        last_dones = np.asarray([0.0], dtype=np.float32)
        if steps_taken > 0:
            last_dones = np.asarray([float(buffer.dones[-1])], dtype=np.float32)
        ppo_stats = self.ppo.update(buffer, last_values=last_values, last_dones=last_dones)
        record.policy_loss = float(ppo_stats.get("policy_loss", 0.0))
        record.value_loss = float(ppo_stats.get("value_loss", 0.0))
        record.entropy_loss = float(ppo_stats.get("entropy_loss", 0.0))
        record.approx_kl = float(ppo_stats.get("approx_kl", 0.0))

        # ---- 4. optimize f_hat w.r.t. the MSE loss on D using Adam ----
        try:
            rnd_stats = self.rnd.update_from_buffer(buffer, key="next_obs")
            record.rnd_loss = float(rnd_stats.get("loss", rnd_stats.get("rnd_loss", 0.0)))
        except Exception as exc:  # pragma: no cover - defensive
            if self.config.verbose:
                print(f"[RICE] RND update skipped: {exc}")

        record.seconds = time.time() - t_start
        self.iteration += 1
        return buffer, record

    def _bootstrap_value(self, obs: Any) -> np.ndarray:
        """Value of the state after the last collected step (PPO bootstrapping)."""
        try:
            values = self.policy.predict_values(_as_obs_array(obs))
        except Exception:
            return np.asarray([0.0], dtype=np.float32)
        return np.asarray([_scalar(values)], dtype=np.float32)

    # ------------------------------------------------------------------ main loop
    def train(self, n_iterations: Optional[int] = None) -> RefineResult:
        """Run Algorithm 2 and return ``pi'`` + diagnostics."""
        total_iters = int(n_iterations if n_iterations is not None else self.config.n_iterations)
        budget = self.config.total_env_steps
        t_start = time.time()
        reason = "iterations"

        for _ in range(total_iters):
            if budget is not None and self.env_steps >= int(budget):
                reason = "env_step_budget"
                break
            _, record = self.collect_iteration()
            self.log.append(record)
            if self.logger is not None:
                self.logger(record)
            if self.config.verbose and (
                record.iteration % max(1, int(self.config.log_every)) == 0
                or record.iteration == 0
            ):
                recent = self.episode_returns[-10:]
                mean_ret = float(np.mean(recent)) if recent else float("nan")
                print(
                    f"[RICE] it={record.iteration:5d} init={record.init_mode:8s} "
                    f"steps={record.steps} aug_ret={record.task_return:10.2f} "
                    f"bonus={record.mean_rnd_bonus:9.5f} mean_ret(last10)={mean_ret:10.2f} "
                    f"t={record.seconds:6.2f}s"
                )
            if self.config.eval_every and self.config.eval_every > 0 and (
                (record.iteration + 1) % int(self.config.eval_every) == 0
            ):
                evaluation = evaluate_policy(
                    self.evaluation_env,
                    self.policy,
                    n_episodes=self.config.eval_episodes,
                    deterministic=True,
                    rng=self.rng,
                )
                if self.config.verbose:
                    print(f"[RICE]   eval@{record.iteration + 1}: {evaluation['mean']:.2f}")

        seconds = time.time() - t_start

        final_eval: Optional[float] = None
        if self.config.eval_episodes and self.config.eval_episodes > 0:
            evaluation = evaluate_policy(
                self.evaluation_env,
                self.policy,
                n_episodes=self.config.eval_episodes,
                deterministic=True,
                seed=self.config.seed,
                rng=self.rng,
            )
            final_eval = float(evaluation["mean"])

        returns = np.asarray(self.episode_returns, dtype=np.float64)
        result = RefineResult(
            policy=self.policy,
            iterations=len(self.log),
            env_steps=int(self.env_steps),
            seconds=float(seconds),
            episode_returns=[float(r) for r in self.episode_returns],
            mean_episode_return=float(returns.mean()) if returns.size else None,
            final_eval_reward=final_eval,
            critical_fraction=float(self.sampler.critical_fraction()),
            log=[rec.as_dict() for rec in self.log],
            stopping_reason=reason,
        )
        return result

    # convenience aliases -------------------------------------------------------
    def refine(self, n_iterations: Optional[int] = None) -> RefineResult:
        """Alias of :meth:`train` (``pi' <- pi_theta`` at the end of Algorithm 2)."""
        return self.train(n_iterations=n_iterations)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "rnd": self.rnd.state_dict(),
            "ppo": self.ppo.state_dict(),
            "iteration": self.iteration,
            "env_steps": self.env_steps,
            "episode_returns": list(self.episode_returns),
            "config": asdict(self.config),
        }

    def save(self, path: str) -> str:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required to save a refiner state")
        torch.save(self.policy.state_dict(), path)
        return path


# --------------------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------------------
def make_refiner(
    env: Any,
    policy: ActorCritic,
    mask_network: Any = None,
    config: Optional[Any] = None,
    **kwargs: Any,
) -> RICERefiner:
    """Build a :class:`RICERefiner` from a :class:`RefineConfig`, mapping or kwargs."""
    if isinstance(config, RefineConfig):
        cfg = config
    elif isinstance(config, dict):
        cfg = RefineConfig.from_mapping(config)
    else:
        cfg = RefineConfig.from_mapping(None)
    if kwargs:
        cfg = cfg.clone(**{k: v for k, v in kwargs.items() if hasattr(cfg, k)})
    return RICERefiner(env=env, policy=policy, mask_network=mask_network, config=cfg)


def refine_policy(
    env: Any,
    policy: ActorCritic,
    mask_network: Any = None,
    config: Optional[Any] = None,
    sampler: Optional[MixedInitSampler] = None,
    rnd: Optional[RND] = None,
    state_manager: Any = None,
    evaluation_env: Any = None,
    rng: Optional[np.random.Generator] = None,
) -> RefineResult:
    """Functional entry point: run Algorithm 2 on a warm-started ``policy``.

    This is what ``scripts/run_refine.py`` and ``main.py refine`` call.
    """
    refiner = make_refiner(env=env, policy=policy, mask_network=mask_network, config=config)
    if sampler is not None:
        refiner.sampler = sampler
    if rnd is not None:
        refiner.rnd = rnd
    if state_manager is not None:
        refiner.state_manager = state_manager
        refiner.sampler.state_manager = state_manager
    if evaluation_env is not None:
        refiner.evaluation_env = evaluation_env
    if rng is not None:
        refiner.rng = rng
        refiner.sampler.rng = rng
    return refiner.train()

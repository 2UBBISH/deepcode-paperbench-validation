"""Shared PPO implementation (Proximal Policy Optimization).

The RICE paper (Proc. 41st ICML, PMLR 235, 2024) uses the *vanilla PPO algorithm*
for three different purposes:

* Algorithm 1 -- training the mask network  :math:`\\tilde{\\pi}_\\theta` on the
  blinded action ``a_t \\odot a_t^m`` with the augmented reward
  :math:`R'(s_t, a_t) = R(s_t, a_t) + \\alpha a_t^m` (see ``mask_network.py``);
* Algorithm 2 -- refining the pre-trained agent :math:`\\pi_\\theta` with the
  RND-augmented reward :math:`R_t + \\lambda R_t^{RND}` (see ``refine.py``);
* the PPO fine-tuning baseline (see ``baselines/ppo_finetune.py``).

The paper does **not** specify the PPO hyper-parameters explicitly
(``Source: not specified in the paper``); following the reproduction plan we use
the Stable-Baselines3 (Raffin et al. 2021) ``PPO`` defaults:

===========================  ==========================================
learning rate                ``3e-4`` (linear schedule)
n_steps                      2048
batch_size                   64
n_epochs                     10
gamma                        0.99
gae_lambda                   0.95
clip_range                   0.2
ent_coef                     0.0
vf_coef                      0.5
max_grad_norm                0.5
target_kl                    None
normalize_advantage          True
net_arch                     ``[64, 64]`` for the shared MLP extractor
activation                   ``tanh``
log_std_init                 0.0
===========================  ==========================================

All of these are exposed through :class:`PPOConfig` so that the baselines /
sweeps can override them (e.g. the PPO fine-tuning baseline lowers the learning
rate, see ``baselines/ppo_finetune.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

try:  # gym and gymnasium co-exist in the wild; accept either.
    import gym

    _Box = gym.spaces.Box
    _Discrete = gym.spaces.Discrete
    _Dict = gym.spaces.Dict
except Exception:  # pragma: no cover - very old / unusual installations
    gym = None
    _Box = _Discrete = _Dict = None  # type: ignore

try:
    from gymnasium import spaces as _gymnasium_spaces

    _GymnasiumBox = _gymnasium_spaces.Box
    _GymnasiumDiscrete = _gymnasium_spaces.Discrete
    _GymnasiumDict = _gymnasium_spaces.Dict
except Exception:  # pragma: no cover
    _GymnasiumBox = _GymnasiumDiscrete = _GymnasiumDict = None  # type: ignore


# --------------------------------------------------------------------------------------
# observation helpers
# --------------------------------------------------------------------------------------
def _is_dict_space(space: Any) -> bool:
    for cls in (_Dict, _GymnasiumDict):
        if cls is not None and isinstance(space, cls):
            return True
    return hasattr(space, "spaces") and isinstance(getattr(space, "spaces"), dict)


def _is_discrete_space(space: Any) -> bool:
    for cls in (_Discrete, _GymnasiumDiscrete):
        if cls is not None and isinstance(space, cls):
            return True
    return isinstance(space, int) or (hasattr(space, "n") and not hasattr(space, "shape"))


def box_shape(space: Any) -> Tuple[int, ...]:
    """Shape of a ``Box`` observation/action space (works for gym and gymnasium)."""
    if space is None:
        return ()
    shape = getattr(space, "shape", None)
    if shape is None:
        raise TypeError(f"Cannot infer shape of space {space!r}")
    return tuple(int(s) for s in shape)


def box_size(space: Any) -> int:
    size = 1
    for s in box_shape(space):
        size *= s
    return int(size)


def flatten_obs(obs: Any) -> np.ndarray:
    """Flatten an observation to a 1-D ``float32`` vector.

    Vector observations (MuJoCo, CAGE, Selfish Mining, MetaDrive) are simply
    reshaped.  ``Dict`` observations are concatenated in sorted key order -- the
    same convention the environment wrappers use when they expose a flat view.
    """
    if isinstance(obs, dict):
        parts: List[np.ndarray] = []
        for key in sorted(obs.keys()):
            parts.append(np.asarray(obs[key], dtype=np.float32).ravel())
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)
    return np.asarray(obs, dtype=np.float32).ravel()


def observation_size(observation_space: Any) -> int:
    """Number of scalar features of an observation space (Dict supported)."""
    if _is_dict_space(observation_space):
        total = 0
        for key, subspace in observation_space.spaces.items():
            total += observation_size(subspace)
        return int(total)
    if hasattr(observation_space, "shape") and observation_space.shape:
        return box_size(observation_space)
    if hasattr(observation_space, "n"):
        return int(observation_space.n)
    raise TypeError(f"Unsupported observation space {observation_space!r}")


def action_size(action_space: Any) -> int:
    if _is_dict_space(action_space):
        return sum(action_size(sp) for sp in action_space.spaces.values())
    if _is_discrete_space(action_space):
        return int(action_space.n)
    return box_size(action_space)


def make_target_policy_callable(policy: Any) -> Callable[[Any], np.ndarray]:
    """Wrap ``policy`` into a ``obs -> action`` callable.

    Accepts:

    * a plain python callable already returning an action;
    * a Stable-Baselines3 model (``model.predict``);
    * one of our :class:`ActorCritic` modules (``policy.predict``);
    * an object exposing ``act``.
    """
    if callable(policy) and not hasattr(policy, "predict") and not hasattr(policy, "act"):
        return lambda obs: np.asarray(policy(obs))

    if hasattr(policy, "predict"):
        def _predict(obs: Any) -> np.ndarray:  # pragma: no cover - exercised via adapters
            out = policy.predict(obs, deterministic=False)
            if isinstance(out, tuple):
                out = out[0]
            return np.asarray(out)

        return _predict

    if hasattr(policy, "act"):
        def _act(obs: Any) -> np.ndarray:
            out = policy.act(obs)
            if isinstance(out, tuple):
                out = out[0]
            return np.asarray(out)

        return _act

    raise TypeError(f"Unsupported target policy type: {type(policy)!r}")


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
@dataclass
class PPOConfig:
    """PPO hyper-parameters (Stable-Baselines3 defaults unless stated otherwise)."""

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: Union[float, Callable[[float], float]] = 0.2
    clip_range_vf: Optional[Union[float, Callable[[float], float]]] = None
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.8
    normalize_advantage: bool = True
    # ---- network ----------------------------------------------------------------
    net_arch: Sequence[int] = field(default_factory=lambda: (64, 64))
    activation: str = "tanh"
    log_std_init: float = 0.0
    ortho_init: bool = True
    # ---- bookkeeping ------------------------------------------------------------
    seed: Optional[int] = None
    device: str = "auto"
    verbose: int = 0

    def clone(self, **overrides: Any) -> "PPOConfig":
        return replace(self, **overrides)


def activation_fn(name: str) -> Callable[[], nn.Module]:
    name = (name or "tanh").lower()
    table = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "sigmoid": nn.Sigmoid,
    }
    if name not in table:
        raise ValueError(f"Unsupported activation '{name}'")
    return table[name]


# --------------------------------------------------------------------------------------
# networks
# --------------------------------------------------------------------------------------
def init_ortho(module: nn.Module, gain: float = 1.0) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class MlpExtractor(nn.Module):
    """Shared feature extractor: ``Linear -> activation`` repeated, then ``Linear``."""

    def __init__(
        self,
        input_dim: int,
        net_arch: Sequence[int] = (64, 64),
        activation: str = "tanh",
        ortho_init: bool = True,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last = int(input_dim)
        for hidden in net_arch:
            linear = nn.Linear(last, int(hidden))
            layers.append(linear)
            layers.append(activation_fn(activation)())
            last = int(hidden)
        layers.append(nn.Linear(last, int(net_arch[-1] if len(net_arch) > 0 else last)))
        layers.append(activation_fn(activation)())
        self.net = nn.Sequential(*layers)
        self.output_dim = int(net_arch[-1] if len(net_arch) > 0 else input_dim)
        if ortho_init:
            for module in self.net:
                if isinstance(module, nn.Linear):
                    init_ortho(module, gain=math.sqrt(2.0))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ActorCritic(nn.Module):
    """Actor-critic module with SB3-style defaults.

    Supports both continuous (``Box``) and discrete (``Discrete``) action spaces so
    that the very same class can represent

    * the refining / target policy (MuJoCo, CAGE, Selfish Mining, MetaDrive), and
    * the binary **mask network** (``Discrete(2)``, see ``MaskNetwork``).
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        net_arch: Sequence[int] = (64, 64),
        activation: str = "tanh",
        log_std_init: float = 0.0,
        ortho_init: bool = True,
        obs_dim: Optional[int] = None,
        device: str = "auto",
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_dim = int(obs_dim) if obs_dim is not None else observation_size(observation_space)
        self.act_dim = action_size(action_space)
        self.discrete = _is_discrete_space(action_space)
        self.log_std_init = float(log_std_init)
        self.device = torch.device("cpu") if device == "auto" and not torch.cuda.is_available() else (
            torch.device("cuda") if device == "auto" else torch.device(device)
        )

        self.features_extractor = MlpExtractor(self.obs_dim, net_arch, activation, ortho_init)
        latent = self.features_extractor.output_dim

        if self.discrete:
            self.action_net = nn.Linear(latent, self.act_dim)
        else:
            self.action_net = nn.Linear(latent, self.act_dim)
            self.log_std = nn.Parameter(torch.ones(self.act_dim) * self.log_std_init)
        self.value_net = nn.Linear(latent, 1)

        if ortho_init:
            init_ortho(self.action_net, gain=0.01)
            init_ortho(self.value_net, gain=1.0)

        # Action-space bounds used to clip continuous actions (SB3 does the same).
        self.action_low, self.action_high = self._action_bounds(action_space)

        self.to(self.device)

    # ---------------------------------------------------------------- utilities
    @staticmethod
    def _action_bounds(action_space: Any) -> Tuple[np.ndarray, np.ndarray]:
        low = getattr(action_space, "low", None)
        high = getattr(action_space, "high", None)
        if low is None or high is None:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        low = np.asarray(low, dtype=np.float32).ravel()
        high = np.asarray(high, dtype=np.float32).ravel()
        # SB3 uses +/- inf clipping bounds of 1e30 for unbounded Box spaces.
        high = np.where(np.isinf(high), 1e30, high)
        low = np.where(np.isinf(low), -1e30, low)
        return low.astype(np.float32), high.astype(np.float32)

    def _obs_tensor(self, obs: Any) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            tensor = obs.float()
        else:
            tensor = torch.as_tensor(flatten_obs(obs), dtype=torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    # ---------------------------------------------------------------- forward
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        latent = self.features_extractor(obs.to(self.device))
        if self.discrete:
            return self.action_net(latent), torch.zeros(0, device=latent.device)
        mean = self.action_net(latent)
        log_std = self.log_std.to(latent.device).expand_as(mean)
        return mean, log_std

    def get_distribution(self, obs: torch.Tensor):
        """Return the action distribution for batch ``obs``."""
        mean, log_std = self.forward(obs)
        if self.discrete:
            return Categorical(logits=mean)
        std = torch.exp(log_std.clamp(-20.0, 2.0))
        return Normal(mean, std)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        latent = self.features_extractor(obs.to(self.device))
        return self.value_net(latent).squeeze(-1)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(values, log_prob, entropy)`` for a batch of state-action pairs."""
        distribution = self.get_distribution(obs)
        if self.discrete:
            actions = actions.long().reshape(-1)
        else:
            actions = actions.float().reshape(actions.shape[0], -1)
        log_prob = distribution.log_prob(actions)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=-1)
        entropy = distribution.entropy()
        if entropy.dim() > 1:
            entropy = entropy.sum(dim=-1)
        values = self.predict_values(obs)
        if values.dim() == 0:
            values = values.unsqueeze(0)
        return values, log_prob, entropy

    # ---------------------------------------------------------------- acting
    @torch.no_grad()
    def predict(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        """Return a numpy action (single observation)."""
        distribution = self.get_distribution(self._obs_tensor(obs))
        if deterministic:
            action = distribution.mean if not self.discrete else torch.argmax(distribution.logits, dim=-1)
        else:
            action = distribution.sample()
        return self._to_numpy_action(action)

    @torch.no_grad()
    def act(self, obs: Any, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """Return ``(action, value, log_prob)`` -- mirrors ``PPO.act`` of SB3."""
        obs_tensor = self._obs_tensor(obs)
        distribution = self.get_distribution(obs_tensor)
        if deterministic:
            action = distribution.mean if not self.discrete else torch.argmax(distribution.logits, dim=-1)
        else:
            action = distribution.sample()
        log_prob = distribution.log_prob(action)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=-1)
        value = self.predict_values(obs_tensor).reshape(-1)
        return self._to_numpy_action(action), float(value.item()), float(log_prob.item())

    def _to_numpy_action(self, action: torch.Tensor) -> np.ndarray:
        array = action.detach().cpu().numpy()
        if self.discrete:
            return np.asarray(array).reshape(-1).astype(np.int64)
        array = np.asarray(array, dtype=np.float32).reshape(-1)
        if self.action_low.size == array.size:
            array = np.clip(array, self.action_low, self.action_high).astype(np.float32)
        return array

    # ---------------------------------------------------------------- helpers
    def get_std(self) -> torch.Tensor:
        if self.discrete:
            return torch.ones(1, device=self.device)
        return torch.exp(self.log_std.clamp(-20.0, 2.0))

    def log_prob_of(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """``log pi(a|s)`` without the value computation (used for importance scores)."""
        distribution = self.get_distribution(obs)
        if self.discrete:
            actions = actions.long().reshape(-1)
        log_prob = distribution.log_prob(actions.float() if not self.discrete else actions)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=-1)
        return log_prob


# --------------------------------------------------------------------------------------
# rollout buffer
# --------------------------------------------------------------------------------------
class RolloutBuffer:
    """Minimal on-policy rollout storage with GAE advantage computation.

    Stores transitions ``(obs, action, reward, next_obs, done, value, log_prob)`` and
    computes the Generalised Advantage Estimation targets used by the PPO update.
    """

    def __init__(self) -> None:
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.next_obs: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self.obs)

    def add(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        done: bool,
        value: float = 0.0,
        log_prob: float = 0.0,
    ) -> None:
        self.obs.append(flatten_obs(obs))
        action_array = np.asarray(action)
        self.actions.append(action_array.reshape(-1) if action_array.ndim > 1 else action_array)
        self.rewards.append(float(reward))
        self.next_obs.append(flatten_obs(next_obs) if next_obs is not None else flatten_obs(obs))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))

    def clear(self) -> None:
        self.__init__()

    def as_arrays(self) -> Dict[str, np.ndarray]:
        if len(self) == 0:
            raise ValueError("Empty rollout buffer")
        obs = np.stack(self.obs).astype(np.float32)
        next_obs = np.stack(self.next_obs).astype(np.float32)
        actions = np.stack([np.atleast_1d(a) for a in self.actions]).astype(np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        log_probs = np.asarray(self.log_probs, dtype=np.float32)
        return {
            "obs": obs,
            "next_obs": next_obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "log_probs": log_probs,
        }

    # ------------------------------------------------------------------ GAE
    def compute_returns_and_advantages(
        self,
        last_values: Any = 0.0,
        last_dones: Any = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Generalised Advantage Estimation (Schulman et al. 2016).

        ``delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)`` and
        ``A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}``.
        """
        data = self.as_arrays()
        rewards = data["rewards"]
        values = data["values"]
        dones = data["dones"]
        n = len(rewards)

        if isinstance(last_values, (float, int, np.floating)) or np.isscalar(last_values):
            last_values_arr = np.full(1, float(last_values), dtype=np.float32)
        else:
            last_values_arr = np.asarray(last_values, dtype=np.float32).reshape(-1)
        if last_dones is None:
            last_dones_arr = np.zeros(1, dtype=np.float32)
        elif np.isscalar(last_dones):
            last_dones_arr = np.full(1, float(last_dones), dtype=np.float32)
        else:
            last_dones_arr = np.asarray(last_dones, dtype=np.float32).reshape(-1)

        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = float(last_values_arr[0])
                next_done = float(last_dones_arr[0])
            else:
                next_value = float(values[t + 1])
                next_done = float(dones[t])
            delta = rewards[t] + gamma * next_value * (1.0 - next_done) - values[t]
            last_gae = delta + gamma * gae_lambda * (1.0 - next_done) * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + values


# --------------------------------------------------------------------------------------
# PPO
# --------------------------------------------------------------------------------------
class PPO:
    """Vanilla PPO update (clipped surrogate, SB3 defaults).

    The class only owns the optimiser and the update rule; the rollout collection is
    left to the caller (``MaskNetworkTrainer``, ``RICERefiner``, ``PPOFineTuner``)
    because those loops differ (masked actions / RND bonus / lowered learning rate).
    """

    def __init__(
        self,
        policy: ActorCritic,
        config: Optional[PPOConfig] = None,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        self.policy = policy
        self.cfg = config.clone(**kwargs) if (config is not None and kwargs) else (
            config if config is not None else PPOConfig(**kwargs)
        )
        if device != "auto":
            self.policy.to(device)
        self.device = self.policy.device
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.cfg.learning_rate, eps=1e-5
        )
        self._n_updates = 0

    # ------------------------------------------------------------------ helpers
    def _clip_value(self, key: str, progress: float) -> Optional[float]:
        value = getattr(self.cfg, key)
        if value is None:
            return None
        if callable(value):
            return float(value(progress))
        return float(value)

    def current_learning_rate(self, progress: float = 0.0) -> float:
        return float(self.cfg.learning_rate)

    def set_learning_rate(self, learning_rate: float) -> None:
        self.cfg = self.cfg.clone(learning_rate=float(learning_rate))
        for group in self.optimizer.param_groups:
            group["lr"] = float(learning_rate)

    # ------------------------------------------------------------------ update
    def update(
        self,
        buffer: RolloutBuffer,
        last_values: Any = 0.0,
        last_dones: Any = None,
        progress: float = 0.0,
    ) -> Dict[str, float]:
        """Run ``n_epochs`` of minibatch updates over ``buffer``; returns loss stats."""
        if len(buffer) == 0:
            return {}

        buffer.compute_returns_and_advantages(
            last_values=last_values,
            last_dones=last_dones,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )
        data = buffer.as_arrays()

        obs = torch.as_tensor(data["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(data["actions"], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(data["log_probs"], dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(data["values"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(
            np.asarray(buffer.returns, dtype=np.float32), dtype=torch.float32, device=self.device
        )
        advantages = torch.as_tensor(
            np.asarray(buffer.advantages, dtype=np.float32), dtype=torch.float32, device=self.device
        )

        if self.cfg.normalize_advantage and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        n_samples = obs.shape[0]
        batch_size = int(min(self.cfg.batch_size, n_samples))
        clip_range = self._clip_value("clip_range", progress)
        clip_range_vf = self._clip_value("clip_range_vf", progress)

        stats: Dict[str, float] = {}
        n_updates = 0
        stop_early = False

        for _epoch in range(int(self.cfg.n_epochs)):
            if stop_early:
                break
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                mb_idx = indices[start : start + batch_size]
                if len(mb_idx) < 2:
                    continue
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_old_values = old_values[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                values, log_prob, entropy = self.policy.evaluate_actions(mb_obs, mb_actions)

                ratio = torch.exp(log_prob - mb_old_log_probs)
                policy_loss_1 = -mb_advantages * ratio
                policy_loss_2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - clip_range, 1.0 + clip_range
                )
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                with torch.no_grad():
                    log_ratio = log_prob - mb_old_log_probs
                    approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean().item()
                    clip_fraction = (torch.abs(ratio - 1.0) > clip_range).float().mean().item()

                if clip_range_vf is not None:
                    values_clipped = mb_old_values + torch.clamp(
                        values - mb_old_values, -clip_range_vf, clip_range_vf
                    )
                    vf_loss_1 = F.mse_loss(values, mb_returns)
                    vf_loss_2 = F.mse_loss(values_clipped, mb_returns)
                    value_loss = torch.max(vf_loss_1, vf_loss_2)
                else:
                    value_loss = F.mse_loss(values, mb_returns)

                entropy_loss = entropy.mean()
                loss = policy_loss - self.cfg.ent_coef * entropy_loss + self.cfg.vf_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                n_updates += 1
                stats = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy_loss": float(entropy_loss.item()),
                    "loss": float(loss.item()),
                    "approx_kl": float(approx_kl),
                    "clip_fraction": float(clip_fraction),
                    "n_updates": float(n_updates),
                }

                if self.cfg.target_kl is not None and approx_kl > 1.5 * self.cfg.target_kl:
                    stop_early = True
                    break

        self._n_updates += n_updates
        return stats

    # ------------------------------------------------------------------ misc
    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": vars(self.cfg),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.policy.load_state_dict(state["policy"])
        if "optimizer" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except (ValueError, KeyError):  # architecture/param-group mismatch
                pass

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device))


__all__ = [
    "PPOConfig",
    "ActorCritic",
    "MlpExtractor",
    "RolloutBuffer",
    "PPO",
    "activation_fn",
    "flatten_obs",
    "observation_size",
    "action_size",
    "make_target_policy_callable",
    "init_ortho",
    "box_shape",
    "box_size",
]

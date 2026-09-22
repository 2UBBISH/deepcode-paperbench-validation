"""Per-policy rollout storage for SAPG ("Split and Aggregate Policy Gradients").

SAPG splits ``N`` massively parallel environments into ``M`` contiguous blocks and
rolls out one policy per block.  Every policy gets its own dataset
:math:`\\mathcal{D}_1, \\ldots, \\mathcal{D}_M` (§4.6, Algorithm 1).  This module
implements that per-policy dataset together with all quantities the losses need:

* GAE advantages (used by the on-policy clipped surrogate, Eq. 2 of §4.1),
* ``n``-step on-policy critic targets (``n = 3``, Eq. (3) of §4.1)

  .. math:: V_{on,\\pi_j}^{target}(s_t) = \\sum_{k=t}^{t+2} \\gamma^{k-t} r_k
            + \\gamma^3 V_{\\pi_j, old}(s_{t+3})

* 1-step off-policy critic targets (Eq. (4) of §4.1)

  .. math:: V_{off,\\pi_j}^{target}(s_t') = r_t + \\gamma V_{\\pi_j, old}(s_{t+1}')

* the importance-sampling correction term :math:`\\mu = \\pi_{i, old}(s,a) /
  \\pi_j(s,a)` (§4.1) used to build ``D'_1`` for the leader,
* minibatch iterators for both feed-forward (MLP) and recurrent (LSTM) policies,
* helpers that fuse the follower datasets into the leader's off-policy batch
  (uniform subsampling so that :math:`|D'_1| = |D_1|`, §4.3 / Algorithm 1).

Only pure PyTorch is used here: the buffer never touches the simulator, so it can
be unit-tested with recorded tensors.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import torch

try:  # pragma: no cover - configuration is optional for standalone use
    from ..utils.config import SAPGConfig
except Exception:  # pragma: no cover
    SAPGConfig = Any  # type: ignore

__all__ = [
    "RolloutBuffer",
    "BufferSet",
    "OffPolicyBatch",
    "build_off_policy_batch",
    "mu_from_logprobs",
    "compute_n_step_targets",
    "compute_one_step_targets",
]


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def _as_tensor(value: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert ``value`` to a float tensor on ``device`` (bools -> floats)."""
    if torch.is_tensor(value):
        out = value.detach().to(device=device)
    else:
        out = torch.as_tensor(value, device=device)
    if out.dtype == torch.bool:
        out = out.to(dtype)
    elif out.is_floating_point():
        out = out.to(dtype=dtype)
    return out


def _squeeze_leading(value: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Remove leading singleton/time dimensions until ``value`` matches ``target_dim``."""
    while value.dim() > target_dim and value.shape[0] == 1:
        value = value.squeeze(0)
    return value


def compute_n_step_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float = 0.99,
    n_step: int = 3,
) -> torch.Tensor:
    r"""``n``-step on-policy value targets (Eq. (3), §4.1).

    ``V_on_target(s_t) = sum_{k=t}^{t+n-1} gamma^{k-t} r_k + gamma^n V_old(s_{t+n})``,
    truncating the sum when an episode terminates inside the window (then no
    bootstrap is added) and bootstrapping from ``last_values`` (= ``V(s_T)``) when
    the window crosses the end of the collected horizon.
    """
    horizon, num_envs = rewards.shape
    device, dtype = rewards.device, rewards.dtype
    targets = torch.zeros(horizon, num_envs, device=device, dtype=dtype)
    for t in range(horizon):
        acc = torch.zeros(num_envs, device=device, dtype=dtype)
        alive = torch.ones(num_envs, device=device, dtype=dtype)
        steps_taken = 0
        for k in range(n_step):
            idx = t + k
            if idx >= horizon:
                break
            acc = acc + (gamma ** k) * alive * rewards[idx]
            alive = alive * (1.0 - dones[idx])
            steps_taken += 1
        if t + n_step >= horizon:
            # the window reaches (or overruns) the end of the horizon
            boot_discount = gamma ** steps_taken
            boot_value = last_values
        else:
            boot_discount = gamma ** n_step
            boot_value = values[t + n_step]
        targets[t] = acc + boot_discount * alive * boot_value
    return targets


def compute_one_step_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    r"""1-step off-policy targets ``V_off_target(s'_t) = r_t + gamma V_old(s'_{t+1})`` (Eq. (4), §4.1)."""
    return rewards + gamma * next_values * (1.0 - dones)


def mu_from_logprobs(target_old_logprobs: torch.Tensor, behaviour_logprobs: torch.Tensor,
                     max_value: Optional[float] = None) -> torch.Tensor:
    r"""Off-policy correction term ``mu = pi_{i,old}(s,a) / pi_j(s,a)`` from log-probabilities (§4.1).

    Passing the *leader's* old log-probabilities for the follower samples gives
    ``mu``; passing the current leader log-probabilities would instead give the
    importance ratio ``r_{pi_i}``.
    """
    log_mu = (target_old_logprobs - behaviour_logprobs).clamp(min=-20.0, max=20.0)
    mu = torch.exp(log_mu)
    if max_value is not None:
        mu = mu.clamp(max=max_value)
    return mu


# --------------------------------------------------------------------------------------
# per-policy buffer
# --------------------------------------------------------------------------------------
class RolloutBuffer:
    """Dataset :math:`\\mathcal{D}_j` collected by policy ``j`` on its own block.

    The tensor layout is time-major (``[horizon, num_envs, ...]``), matching the
    usual IsaacGym/``rl_games`` convention, so that recurrent minibatches built
    from contiguous time slices stay inside a single episode.

    Parameters
    ----------
    num_envs:
        Number of environments owned by this policy, i.e. ``N / M``.
    horizon_length:
        Number of simulator steps collected before every update (16 for the
        AllegroKuka tasks, 8 for the ShadowHand / AllegroHand tasks).
    obs_dim, action_dim:
        Observation / action dimensionalities of the task.
    phi_dim:
        Dimension of the per-policy latent :math:`\\phi_j` (32 for AllegroKuka,
        16 for ShadowHand and AllegroHand, §5.2).  Use ``0`` to disable.
    policy_index:
        0-based index ``j - 1`` of the policy that owns this buffer (the leader is
        index ``0`` in this code base and ``1`` in the paper).
    """

    def __init__(
        self,
        num_envs: int,
        horizon_length: Optional[int] = None,
        obs_dim: int = 1,
        action_dim: int = 1,
        phi_dim: int = 0,
        recurrent: Optional[bool] = None,
        gamma: Optional[float] = None,
        tau: Optional[float] = None,
        lam: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
        policy_index: int = 0,
        is_leader: Optional[bool] = None,
        leader_index: Optional[int] = None,
        sequence_length: Optional[int] = None,
        config: Optional[SAPGConfig] = None,
        num_policies: Optional[int] = None,
        normalize_advantage: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        if config is not None:
            if horizon_length is None:
                horizon_length = getattr(config, "horizon_length", None)
            if gamma is None:
                gamma = getattr(config, "gamma", None)
            if tau is None:
                tau = getattr(config, "tau", None)
            if recurrent is None:
                recurrent = getattr(config, "use_lstm", None)
            if sequence_length is None:
                sequence_length = getattr(config, "lstm_sequence_length", None)
            if num_policies is None:
                num_policies = getattr(config, "num_policies", None)
            if leader_index is None:
                leader_index = getattr(config, "leader_index", None)

        self.num_envs = int(num_envs)
        self.horizon_length = int(horizon_length) if horizon_length is not None else 16
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.phi_dim = int(phi_dim or 0)
        self.recurrent = bool(recurrent) if recurrent is not None else False
        self.gamma = float(gamma) if gamma is not None else 0.99
        self.lam = float(lam if lam is not None else (tau if tau is not None else 0.95))
        self.device = torch.device(device)
        self.policy_index = int(policy_index)
        # ``leader_index`` follows the paper's 1-based convention (§4.3: leader = policy 1)
        self.leader_index = int(leader_index) if leader_index is not None else 1
        self.is_leader = bool(is_leader) if is_leader is not None else (
            self.policy_index == self.leader_index - 1
        )
        self.sequence_length = int(sequence_length) if sequence_length is not None else self.horizon_length
        self.num_policies = int(num_policies) if num_policies is not None else None
        self.normalize_advantage = bool(normalize_advantage) if normalize_advantage is not None else True

        T, N, dev, f = self.horizon_length, self.num_envs, self.device, torch.float32
        self.obs = torch.zeros(T, N, self.obs_dim, dtype=f, device=dev)
        self.actions = torch.zeros(T, N, self.action_dim, dtype=f, device=dev)
        self.logprobs = torch.zeros(T, N, dtype=f, device=dev)
        self.values = torch.zeros(T, N, dtype=f, device=dev)
        self.rewards = torch.zeros(T, N, dtype=f, device=dev)
        self.dones = torch.zeros(T, N, dtype=f, device=dev)
        self.truncated = torch.zeros(T, N, dtype=f, device=dev)
        # mu = pi_{i,old}(s, a) / pi_j(s, a); for the behaviour policy itself mu == 1
        self.mu = torch.ones(T, N, dtype=f, device=dev)
        self.phis = torch.zeros(N, self.phi_dim, dtype=f, device=dev)
        # filled by ``finalise``
        self.advantages = torch.zeros(T, N, dtype=f, device=dev)
        self.returns = torch.zeros(T, N, dtype=f, device=dev)
        self.value_targets = torch.zeros(T, N, dtype=f, device=dev)
        self.value_targets_off = torch.zeros(T, N, dtype=f, device=dev)
        self._last_values = torch.zeros(N, dtype=f, device=dev)
        self._last_dones = torch.zeros(N, dtype=f, device=dev)
        self._filled = 0
        self._finalised = False

    # ------------------------------------------------------------------ dunder
    def __len__(self) -> int:
        return self.horizon_length * self.num_envs

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        role = "leader" if self.is_leader else "follower"
        return (
            f"RolloutBuffer(policy={self.policy_index}/{role}, "
            f"horizon={self.horizon_length}, num_envs={self.num_envs}, "
            f"obs_dim={self.obs_dim}, action_dim={self.action_dim}, phi_dim={self.phi_dim})"
        )

    @property
    def size(self) -> int:
        """Number of stored transitions ``|D_j|``."""
        return len(self)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.horizon_length, self.num_envs)

    @property
    def phi(self) -> torch.Tensor:
        """Latent ``phi_j`` of the policy that generated this data (``[num_envs, phi_dim]``)."""
        return self.phis

    @property
    def finalised(self) -> bool:
        return self._finalised

    # --------------------------------------------------------------- insertion
    def add(
        self,
        step: Any = None,
        obs: Any = None,
        actions: Any = None,
        logprobs: Any = None,
        values: Any = None,
        rewards: Any = None,
        dones: Any = None,
        phis: Any = None,
        mu: Any = None,
        truncated: Any = None,
        masks: Any = None,
        next_obs: Any = None,
        **extra: Any,
    ) -> "RolloutBuffer":
        """Store one environment step of this policy.

        Accepts either explicit keyword arguments or a single mapping (the
        mapping may use the common aliases ``obses``, ``action``, ``value``,
        ``reward``, ``done``, ``log_prob``, ``phi``).  ``obs`` built from *this*
        block and ``next_obs`` are equivalent here: a block's own dataset is
        formed of consecutive same-policy transitions.
        """
        if obs is None and isinstance(step, dict):
            data = dict(step)
            step = data.pop("step", data.pop("t", data.pop("index", None)))
            obs = data.pop("obs", data.pop("obses", None))
            actions = data.pop("actions", data.pop("action", None))
            logprobs = data.pop("logprobs", data.pop("log_prob", data.pop("logprobs_", None)))
            values = data.pop("values", data.pop("value", None))
            rewards = data.pop("rewards", data.pop("reward", None))
            dones = data.pop("dones", data.pop("done", None))
            truncated = data.pop("truncated", data.pop("trunc", None))
            phis = data.pop("phis", data.pop("phi", data.pop("phi_j", None)))
            mu = data.pop("mu", None)
            masks = data.pop("masks", data.pop("mask", None))
            next_obs = data.pop("next_obs", data.pop("next_obses", None))
            extra = data
        else:
            # tolerate aliases handed in as extra keyword arguments
            obs = obs if obs is not None else extra.pop("obses", None)
            actions = actions if actions is not None else extra.pop("action", None)
            logprobs = logprobs if logprobs is not None else extra.pop("log_prob", None)
            values = values if values is not None else extra.pop("value", None)
            rewards = rewards if rewards is not None else extra.pop("reward", None)
            dones = dones if dones is not None else extra.pop("done", None)
            phis = phis if phis is not None else extra.pop("phi", extra.pop("phi_j", None))
            next_obs = next_obs if next_obs is not None else extra.pop("next_obses", None)
            if masks is not None and dones is None:
                dones = masks

        i = int(step) if step is not None else self._filled
        self._put(self.obs, i, obs if obs is not None else next_obs)
        self._put(self.actions, i, actions)
        self._put(self.logprobs, i, logprobs)
        self._put(self.values, i, values)
        self._put(self.rewards, i, rewards)
        self._put(self.dones, i, dones)
        self._put(self.truncated, i, truncated)
        self._put(self.mu, i, mu)

        if phis is not None:
            self._set_phi(phis)

        self._filled = max(self._filled, i + 1)
        return self

    # alias kept for readability in callers
    store = add

    def _put(self, target: torch.Tensor, index: int, value: Any) -> None:
        if value is None:
            return
        v = _as_tensor(value, self.device, target.dtype)
        v = _squeeze_leading(v, target.dim())
        if v.shape != target[index].shape:
            try:
                v = v.expand_as(target[index])
            except RuntimeError:  # pragma: no cover - defensive
                v = v.reshape(target[index].shape)
        with torch.no_grad():
            target[index].copy_(v)

    def _set_phi(self, value: Any) -> None:
        p = _as_tensor(value, self.device, torch.float32)
        while p.dim() > 2 and p.shape[0] == 1:
            p = p.squeeze(0)
        if p.dim() == 2 and p.shape[0] == self.num_envs:
            self.phis = p.clone()
        elif p.dim() == 2 and p.shape[0] == 1:
            self.phis = p.expand(self.num_envs, -1).clone()
        elif p.dim() == 1:
            self.phis = p.unsqueeze(0).expand(self.num_envs, -1).clone()
        else:
            self.phis = p.reshape(self.num_envs, -1).clone()
        self.phi_dim = int(self.phis.shape[-1])

    def set_last_values(
        self,
        values: Any = None,
        dones: Any = None,
        truncated: Any = None,
        logprobs: Any = None,
        obs: Any = None,
    ) -> "RolloutBuffer":
        """Register the bootstrap quantities ``V(s_T)`` and the terminal flags.

        ``dones`` marks environments that auto-reset right after the last collected
        step, so their bootstrap value must be masked out.
        """
        if values is not None:
            v = _squeeze_leading(_as_tensor(values, self.device, torch.float32), 1)
            self._last_values = v.reshape(self.num_envs)
        if dones is not None:
            d = _squeeze_leading(_as_tensor(dones, self.device, torch.float32), 1)
            self._last_dones = d.reshape(self.num_envs)
        elif truncated is not None:
            d = _squeeze_leading(_as_tensor(truncated, self.device, torch.float32), 1)
            self._last_dones = d.reshape(self.num_envs)
        return self

    # alias
    set_bootstrap = set_last_values

    # ---------------------------------------------------------------- targets
    def compute_gae(self, gamma: Optional[float] = None, lam: Optional[float] = None) -> torch.Tensor:
        """Generalised advantage estimation over the collected horizon.

        Usual choice ``gamma = 0.99`` and ``lam = 0.95`` (the paper's ``tau``).
        Also fills ``self.returns`` (``advantage + V``), the regression target of
        the on-policy critic loss collapsed to a GAE return.
        """
        gamma = self.gamma if gamma is None else float(gamma)
        lam = self.lam if lam is None else float(lam)
        T, N = self.horizon_length, self.num_envs
        v_next = torch.cat([self.values, self._last_values.unsqueeze(0)], dim=0)  # [T+1, N]
        adv = torch.zeros(T, N, device=self.device, dtype=self.values.dtype)
        running = torch.zeros(N, device=self.device, dtype=self.values.dtype)
        for t in range(T - 1, -1, -1):
            nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * v_next[t + 1] * nonterminal - self.values[t]
            running = delta + gamma * lam * nonterminal * running
            adv[t] = running
        self.advantages = adv
        self.returns = adv + self.values
        return adv

    def compute_on_policy_targets(self, gamma: Optional[float] = None, n_step: int = 3) -> torch.Tensor:
        r"""3-step value targets (Eq. (3), §4.1) -> ``self.value_targets``."""
        gamma = self.gamma if gamma is None else float(gamma)
        self.value_targets = compute_n_step_targets(
            self.rewards, self.dones, self.values, self._last_values, gamma=gamma, n_step=n_step
        )
        return self.value_targets

    def compute_off_policy_targets(
        self, gamma: Optional[float] = None, next_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""1-step value targets (Eq. (4), §4.1) -> ``self.value_targets_off``.

        ``next_values`` (``[horizon, num_envs]``) allows the caller to evaluate
        :math:`V_{\pi_i, old}(s'_{t+1})` with the *target* policy's critic (e.g.
        the leader's critic when the data comes from a follower).  By default the
        values recorded by the behaviour policy are shifted by one step.
        """
        gamma = self.gamma if gamma is None else float(gamma)
        if next_values is None:
            v_next = torch.cat(
                [self.values[1:], self._last_values.unsqueeze(0)], dim=0
            )  # [T, N] -> v_next[t] = V(s_{t+1})
        else:
            nv = _as_tensor(next_values, self.device, self.values.dtype)
            nv = _squeeze_leading(nv, 2)
            v_next = nv.reshape(self.horizon_length, self.num_envs)
        self.value_targets_off = compute_one_step_targets(self.rewards, self.dones, v_next, gamma=gamma)
        return self.value_targets_off

    def set_off_policy_targets(self, targets: Any) -> "RolloutBuffer":
        t = _squeeze_leading(_as_tensor(targets, self.device, self.values.dtype), 2)
        self.value_targets_off = t.reshape(self.horizon_length, self.num_envs)
        return self

    def set_mu(self, mu: Any) -> "RolloutBuffer":
        """Store the off-policy correction ``mu = pi_{i,old}(s,a) / pi_j(s,a)`` for every step."""
        m = _squeeze_leading(_as_tensor(mu, self.device, self.mu.dtype), 2)
        self.mu = m.reshape(self.horizon_length, self.num_envs)
        return self

    def finalise(
        self,
        gamma: Optional[float] = None,
        lam: Optional[float] = None,
        tau: Optional[float] = None,
        n_step: int = 3,
        last_values: Any = None,
        last_dones: Any = None,
        normalize_advantages: Optional[bool] = None,
        next_values: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> "RolloutBuffer":
        """Compute advantages and both value targets for the collected data."""
        if tau is not None:
            lam = tau
        if last_values is not None or last_dones is not None:
            self.set_last_values(last_values, last_dones)
        self.compute_gae(gamma=gamma, lam=lam)
        if normalize_advantages if normalize_advantages is not None else self.normalize_advantage:
            self.normalize_advantages()
        self.compute_on_policy_targets(gamma=gamma, n_step=n_step)
        self.compute_off_policy_targets(gamma=gamma, next_values=next_values)
        self._finalised = True
        return self

    def normalize_advantages(self, epsilon: float = 1e-5) -> torch.Tensor:
        """Zero-mean / unit-std advantages (computed per update, before minibatching)."""
        adv = self.advantages
        self.advantages = (adv - adv.mean()) / (adv.std(unbiased=False) + epsilon)
        return self.advantages

    # alias with the British spelling
    normalise_advantages = normalize_advantages

    # -------------------------------------------------------------- utilities
    def to(self, device: Union[str, torch.device]) -> "RolloutBuffer":
        device = torch.device(device)
        for name, value in list(self.__dict__.items()):
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        self.device = device
        return self

    def clone(self) -> "RolloutBuffer":
        other = RolloutBuffer(
            self.num_envs,
            self.horizon_length,
            self.obs_dim,
            self.action_dim,
            self.phi_dim,
            recurrent=self.recurrent,
            gamma=self.gamma,
            lam=self.lam,
            device=self.device,
            policy_index=self.policy_index,
            is_leader=self.is_leader,
            leader_index=self.leader_index,
            sequence_length=self.sequence_length,
            num_policies=self.num_policies,
            normalize_advantage=self.normalize_advantage,
        )
        for name, value in self.__dict__.items():
            if torch.is_tensor(value):
                getattr(other, name).copy_(value)
        other._filled = self._filled
        other._finalised = self._finalised
        return other

    #: default tensor keys exposed as flat samples
    DEFAULT_KEYS = (
        "obs",
        "actions",
        "logprobs",
        "values",
        "rewards",
        "dones",
        "mu",
        "advantages",
        "returns",
        "value_targets",
        "value_targets_off",
    )

    def flatten(
        self,
        keys: Optional[Sequence[str]] = None,
        include_phi: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Collapse ``[horizon, num_envs]`` into ``[horizon * num_envs]`` sample-major tensors."""
        keys = tuple(keys) if keys is not None else self.DEFAULT_KEYS
        T, N = self.horizon_length, self.num_envs
        out: Dict[str, torch.Tensor] = {}
        for key in keys:
            value = getattr(self, key, None)
            if value is None or not torch.is_tensor(value):
                continue
            flat = value.reshape(T * N, *value.shape[2:])
            out[key] = flat if device is None else flat.to(device)
        if include_phi and self.phi_dim > 0:
            phi = self.phis.unsqueeze(0).expand(T, -1, -1).reshape(T * N, self.phi_dim)
            out["phis"] = phi if device is None else phi.to(device)
        return out

    def sample(
        self,
        num_samples: int,
        keys: Optional[Sequence[str]] = None,
        generator: Optional[torch.Generator] = None,
        device: Optional[Union[str, torch.device]] = None,
        replace: Optional[bool] = None,
        include_phi: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Uniformly sample ``num_samples`` transitions from this dataset."""
        flat = self.flatten(keys=keys, include_phi=include_phi)
        total = self.size
        if replace is None:
            replace = num_samples > total
        if replace:
            idx = torch.randint(total, (int(num_samples),), generator=generator, device=self.device)
        else:
            idx = torch.randperm(total, generator=generator, device=self.device)[: int(num_samples)]
        out = {k: v[idx] for k, v in flat.items()}
        if device is not None:
            out = {k: v.to(device) for k, v in out.items()}
        return out

    # ------------------------------------------------------------- minibatches
    def generator(
        self,
        minibatch_size: Optional[int] = None,
        num_minibatches: Optional[int] = None,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
        recurrent: Optional[bool] = None,
        sequence_length: Optional[int] = None,
        keys: Optional[Sequence[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        drop_last: bool = False,
        num_epochs: int = 1,
        include_phi: bool = True,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over minibatches of this dataset (feed-forward or recurrent).

        * Feed-forward: yields ``[batch, ...]`` tensors of flat samples.
        * Recurrent: the horizon is cut into contiguous ``sequence_length`` chunks
          (per environment) which are shuffled and batched, then transposed to
          ``[seq_len, batch, ...]``; a ``masks`` entry (``1.0`` at every sequence
          start) is provided for the LSTM.
        """
        T, N = self.horizon_length, self.num_envs
        recurrent = self.recurrent if recurrent is None else bool(recurrent)
        device = torch.device(device) if device is not None else self.device
        flat = self.flatten(keys=keys, include_phi=include_phi)
        total = self.size

        if num_minibatches is not None and minibatch_size is None:
            minibatch_size = max(1, total // int(num_minibatches))
        if minibatch_size is None:
            minibatch_size = total

        for _ in range(max(1, int(num_epochs))):
            if recurrent:
                seq = int(sequence_length or self.sequence_length or T)
                seq = max(1, min(seq, T))
                while T % seq != 0 and seq > 1:
                    seq -= 1
                num_chunks_per_env = T // seq
                chunked: Dict[str, torch.Tensor] = {}
                for key, value in flat.items():
                    rest = value.shape[1:]
                    view = value.reshape(num_chunks_per_env, seq, N, *rest)
                    dims = [0, 2, 1] + list(range(3, view.dim()))
                    view = view.permute(*dims).reshape(num_chunks_per_env * N, seq, *rest)
                    chunked[key] = view
                # sequence-start masks (LSTM resets) computed from the done flags
                dones = self.dones.reshape(num_chunks_per_env, seq, N)
                dones = dones.permute(0, 2, 1).reshape(num_chunks_per_env * N, seq)
                masks = torch.ones_like(dones)
                masks[:, 1:] = 1.0 - dones[:, :-1]
                masks[:, 0] = 1.0
                chunked["masks"] = masks

                num_batches = num_chunks_per_env * N
                chunks_per_mb = max(1, int(minibatch_size) // seq)
                order = (
                    torch.randperm(num_batches, generator=generator, device=self.device)
                    if shuffle
                    else torch.arange(num_batches, device=self.device)
                )
                for start in range(0, num_batches, chunks_per_mb):
                    sel = order[start:start + chunks_per_mb]
                    if sel.numel() < chunks_per_mb and drop_last:
                        continue
                    batch = {
                        k: v[sel].transpose(0, 1).to(device) for k, v in chunked.items()
                    }
                    batch["_recurrent"] = torch.ones(1, dtype=torch.bool)
                    batch["_sequence_length"] = torch.tensor([seq])
                    yield batch
            else:
                order = (
                    torch.randperm(total, generator=generator, device=self.device)
                    if shuffle
                    else torch.arange(total, device=self.device)
                )
                step = max(1, int(minibatch_size))
                for start in range(0, total, step):
                    sel = order[start:start + step]
                    if sel.numel() < step and drop_last:
                        continue
                    batch = {k: v[sel].to(device) for k, v in flat.items()}
                    batch["_recurrent"] = torch.zeros(1, dtype=torch.bool)
                    yield batch

    # convenience aliases used by the trainers
    minibatches = generator

    def num_minibatches(self, minibatch_size: int, recurrent: Optional[bool] = None) -> int:
        """Number of minibatches produced per epoch for a given minibatch size."""
        recurrent = self.recurrent if recurrent is None else bool(recurrent)
        if recurrent:
            seq = max(1, min(int(self.sequence_length or self.horizon_length), self.horizon_length))
            while self.horizon_length % seq != 0 and seq > 1:
                seq -= 1
            return max(1, int(math.ceil((self.horizon_length // seq) * self.num_envs /
                                        max(1, minibatch_size // seq))))
        return max(1, int(math.ceil(self.size / max(1, minibatch_size))))


# --------------------------------------------------------------------------------------
# the M buffers of one iteration
# --------------------------------------------------------------------------------------
class BufferSet:
    """Container for :math:`\\mathcal{D}_1, \\ldots, \\mathcal{D}_M` (one buffer per policy)."""

    def __init__(self, buffers: Sequence[RolloutBuffer], leader_index: int = 1) -> None:
        self.buffers: List[RolloutBuffer] = list(buffers)
        self.leader_index = int(leader_index)

    def __len__(self) -> int:
        return len(self.buffers)

    def __iter__(self) -> Iterator[RolloutBuffer]:
        return iter(self.buffers)

    def __getitem__(self, index: int) -> RolloutBuffer:
        return self.buffers[index]

    @property
    def leader(self) -> RolloutBuffer:
        return self.buffers[self.leader_index - 1]

    @property
    def followers(self) -> List[RolloutBuffer]:
        return [b for i, b in enumerate(self.buffers, start=1) if i != self.leader_index]

    @property
    def follower_indices(self) -> List[int]:
        """1-based indices ``X = {1..M} \\ {leader}`` of the follower datasets."""
        return [i for i in range(1, len(self.buffers) + 1) if i != self.leader_index]

    def total_samples(self) -> int:
        return sum(b.size for b in self.buffers)


# --------------------------------------------------------------------------------------
# off-policy aggregation (Eq. 1, §4.1 / §4.3 / Algorithm 1)
# --------------------------------------------------------------------------------------
class OffPolicyBatch:
    """The fused dataset ``D'_1`` of the leader.

    ``advantages`` are the advantages estimated for the *target* (leader) policy --
    approximated by the behaviour policy's GAE unless the trainer recomputes them
    with the leader critic -- and ``value_targets`` are the 1-step off-policy
    critic targets of Eq. (4).  ``source_policy`` records the 1-based index ``j``
    of the dataset each sample came from, which the loss uses to evaluate
    :math:`\\pi_j(s,a)` for the ratio (Eq. (1)).
    """

    def __init__(self, data: Dict[str, torch.Tensor]) -> None:
        self.data = data

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def keys(self) -> Iterable[str]:
        return self.data.keys()

    @property
    def size(self) -> int:
        return int(self.data["obs"].shape[0])

    def __len__(self) -> int:
        return self.size

    def to(self, device: Union[str, torch.device]) -> "OffPolicyBatch":
        self.data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in self.data.items()}
        return self

    def generator(
        self,
        minibatch_size: int,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
        num_epochs: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over minibatches of the off-policy batch (always feed-forward)."""
        total = self.size
        device = torch.device(device) if device is not None else None
        for _ in range(max(1, int(num_epochs))):
            order = (
                torch.randperm(total, generator=generator, device=self.data["obs"].device)
                if shuffle
                else torch.arange(total, device=self.data["obs"].device)
            )
            for start in range(0, total, max(1, int(minibatch_size))):
                sel = order[start:start + max(1, int(minibatch_size))]
                batch = {k: v[sel] for k, v in self.data.items()}
                if device is not None:
                    batch = {k: v.to(device) for k, v in batch.items()}
                yield batch


def build_off_policy_batch(
    leader_buffer: RolloutBuffer,
    follower_buffers: Sequence[RolloutBuffer],
    subsample: bool = True,
    num_samples: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    keys: Optional[Sequence[str]] = None,
    device: Optional[Union[str, torch.device]] = None,
    follower_indices: Optional[Sequence[int]] = None,
    leader_next_values: Optional[Sequence[Optional[torch.Tensor]]] = None,
) -> OffPolicyBatch:
    r"""Build ``D'_1``: the leader's dataset augmented with importance-weighted follower data.

    Implements the §4.6 / Algorithm 1 sampling step: the union of the followers'
    datasets is subsampled *uniformly* to exactly ``|D_1|`` transitions so that
    the off-policy part of the update is the same size as the on-policy part
    (the ``lambda``-scaled combination of Eq. (2)).  Follower order is preserved in
    ``source_policy`` so that :math:`\pi_j` can be evaluated per sample.

    Parameters
    ----------
    follower_indices:
        1-based indices of the policies in :math:`\mathcal{X}` (default: all
        followers in the order of ``follower_buffers``).
    leader_next_values:
        Optional per-follower ``V_{\pi_1, old}(s'_{t+1})`` tensors used to build
        the 1-step targets (Eq. (4)) with the leader's critic instead of the
        behaviour policy's values.
    """
    if len(follower_buffers) == 0:
        raise ValueError("build_off_policy_batch requires at least one follower buffer")
    keys = tuple(keys) if keys is not None else leader_buffer.DEFAULT_KEYS
    if follower_indices is None:
        follower_indices = list(range(1, len(follower_buffers) + 1))

    parts: List[Dict[str, torch.Tensor]] = []
    sizes: List[int] = []
    source_ids: List[torch.Tensor] = []
    for pos, (buffer, j) in enumerate(zip(follower_buffers, follower_indices)):
        flat = buffer.flatten(keys=keys, include_phi=True)
        if "value_targets_off" not in flat or not torch.any(flat["value_targets_off"] != 0):
            # make sure the 1-step off-policy targets exist for this dataset
            nv = None if leader_next_values is None else leader_next_values[pos]
            buffer.compute_off_policy_targets(gamma=buffer.gamma, next_values=nv)
            flat["value_targets_off"] = buffer.value_targets_off.reshape(-1, buffer.num_envs)[0].new_empty(0) if False else \
                buffer.value_targets_off.reshape(-1)
            flat = buffer.flatten(keys=keys, include_phi=True)
        parts.append(flat)
        sizes.append(buffer.size)
        source_ids.append(
            torch.full((buffer.size,), int(j), dtype=torch.long, device=buffer.device)
        )

    total = int(sum(sizes))
    if not subsample or num_samples is None:
        n = total if subsample else (num_samples if num_samples is not None else total)
    else:
        n = int(num_samples)
    if subsample and num_samples is None:
        # |D'_1| = |D_1| (Algorithm 1)
        n = leader_buffer.size
    n = int(n)

    merged: Dict[str, torch.Tensor] = {}
    for key in parts[0].keys():
        merged[key] = torch.cat([p[key] for p in parts], dim=0)
    merged["source_policy"] = torch.cat(source_ids, dim=0)

    if n >= total:
        idx = torch.randint(total, (n,), generator=generator, device=merged["obs"].device)
    else:
        idx = torch.randperm(total, generator=generator, device=merged["obs"].device)[:n]
    data = {k: v[idx] for k, v in merged.items()}
    if device is not None:
        device = torch.device(device)
        data = {k: v.to(device) for k, v in data.items()}
    return OffPolicyBatch(data)

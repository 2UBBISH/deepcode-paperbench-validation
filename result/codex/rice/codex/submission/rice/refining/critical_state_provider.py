"""Mixed initial state distribution of Algorithm 2.

With probability ``p`` the refining episode starts from the *most critical
state* of a freshly sampled trajectory; with probability ``1 - p`` it starts
from the default initial state distribution ``rho``:

    mu(s) = beta * d_rho^pi_hat(s) + (1 - beta) * rho(s)

The critical state is identified by applying the (frozen) mask network to the
states of a rollout of length ``rollin_length`` and taking the state with the
highest importance score ``P(a^m = 0 | s)``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from rice.explanation.critical_states import importance_scores, most_critical_state


class CriticalStateProvider:
    def __init__(
        self,
        env,
        mask_net,
        p: float = 0.25,
        rollin_length: int = 128,
        rollin_policy=None,
        rollin_policy_mode: str = "pretrained",
        seed: int = 0,
    ):
        self.env = env
        self.mask_net = mask_net
        self.p = float(p)
        self.rollin_length = int(rollin_length)
        self.rollin_policy = rollin_policy
        self.rollin_policy_mode = rollin_policy_mode
        self.rng = np.random.RandomState(seed)
        self.n_resets = 0
        self.n_default = 0

    #: the refining policy can be swapped in for the roll-in (mode="current")
    def set_current_policy(self, policy) -> None:
        self._current_policy = policy

    def _policy(self):
        if self.rollin_policy_mode == "current" and getattr(
            self, "_current_policy", None
        ) is not None:
            return self._current_policy
        return self.rollin_policy

    def sample_critical_state(self) -> Optional[dict]:
        policy = self._policy()
        obs, _ = self.env.reset()
        states, snapshots = [], []
        for _ in range(self.rollin_length):
            snapshots.append(self.env.get_state())
            states.append(np.asarray(obs, dtype=np.float32).copy())
            action = policy.act(obs, deterministic=False)
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        if not states:
            return None
        scores = importance_scores(self.mask_net, np.asarray(states))
        index = most_critical_state(scores)
        return snapshots[index]

    def __call__(self, iteration: int) -> Optional[dict]:
        if self.p <= 0.0:
            self.n_default += 1
            return None
        if self.p >= 1.0 or self.rng.rand() < self.p:
            snapshot = self.sample_critical_state()
            if snapshot is not None:
                self.n_resets += 1
                return snapshot
        self.n_default += 1
        return None


class RandomStateProvider(CriticalStateProvider):
    """Baseline explanation: the critical state is chosen *at random*."""

    def sample_critical_state(self) -> Optional[dict]:
        policy = self._policy()
        obs, _ = self.env.reset()
        snapshots = []
        for _ in range(self.rollin_length):
            snapshots.append(self.env.get_state())
            action = policy.act(obs, deterministic=False)
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        if not snapshots:
            return None
        index = int(self.rng.randint(len(snapshots)))
        return snapshots[index]

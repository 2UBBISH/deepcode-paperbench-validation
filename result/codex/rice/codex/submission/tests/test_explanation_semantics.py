"""Semantic check of the explanation: does it find the truly critical step?

A controlled MDP is used where the answer is known in advance: the reward is 1
if and only if the agent takes action 1 at step ``CRITICAL_STEP``; every other
step is irrelevant (randomising its action cannot change the reward).

A faithful explanation method must therefore assign the *highest* importance to
``CRITICAL_STEP`` and the fidelity metric (Experiment I) must show a larger
reward drop than the random explanation baseline.
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Optional

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rice.envs.adapters import DictStatefulEnv  # noqa: E402
from rice.explanation.mask_trainer import (  # noqa: E402
    MaskTrainingConfig,
    train_mask_network,
)
from rice.fidelity import (  # noqa: E402
    FidelityConfig,
    compute_fidelity,
    mask_importance_fn,
    random_importance_fn,
)
from rice.ppo_core import PPOConfig  # noqa: E402


HORIZON = 8
CRITICAL_STEP = 4


class CriticalStepEnv(gym.Env):
    """Reward 1 iff action 1 is taken at ``CRITICAL_STEP``."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(HORIZON,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)
        self.critical_action_taken = False
        self.t = 0

    # ------------------------------------------------------------------ gym
    def reset(self, *, seed: Optional[int] = None, options=None):
        self.t = 0
        self.critical_action_taken = False
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        obs = np.zeros(HORIZON, dtype=np.float32)
        obs[min(self.t, HORIZON - 1)] = 1.0
        return obs

    def step(self, action):
        action = int(action)
        if self.t == CRITICAL_STEP and action == 1:
            self.critical_action_taken = True
        self.t += 1
        done = self.t >= HORIZON
        reward = 1.0 if (done and self.critical_action_taken) else 0.0
        return self._obs(), reward, done, False, {}


def make_env() -> DictStatefulEnv:
    return DictStatefulEnv(CriticalStepEnv(), name="CriticalStep")


class _TargetPolicy:
    """Policy that takes action 1 exactly at the critical step."""

    def act(self, obs, deterministic: bool = False):
        step = int(np.argmax(obs))
        return 1 if step == CRITICAL_STEP else 0


class ExplanationSemanticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        np.random.seed(0)
        cls.env = make_env()
        cls.config = MaskTrainingConfig(
            iterations=400,
            # several episodes per PPO update: Algorithm 1 collects one
            # trajectory per iteration, but the paper's budget is 300
            # trajectories of 1000 steps, so we keep the number of *samples*
            # comparable here (10 episodes per update x 400 updates).
            rollout_length=10 * HORIZON,
            alpha=0.01,
            seed=0,
            log_interval=1000,
            ppo=PPOConfig(learning_rate=3e-3, n_epochs=4, batch_size=32),
        )
        cls.result = train_mask_network(
            cls.env, _TargetPolicy(), cls.config, verbose=False
        )

    def _states_of_one_trajectory(self):
        obs, _ = self.env.reset()
        states = []
        for _ in range(HORIZON):
            states.append(np.asarray(obs, dtype=np.float32))
            action = _TargetPolicy().act(obs)
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        return np.asarray(states)

    def test_importance_is_peaked_at_the_critical_step(self):
        scores = self.result.importance(self._states_of_one_trajectory())
        self.assertGreater(scores[CRITICAL_STEP], 0.5, scores)
        self.assertEqual(
            int(np.argmax(scores)),
            CRITICAL_STEP,
            "the mask network must rank the critical step highest, got {}".format(
                np.round(scores, 3)
            ),
        )

    def test_fidelity_beats_the_random_explanation(self):
        env = make_env()
        mask = self.result.mask_net
        fidelity_config = FidelityConfig(
            n_trajectories=20, k_values=(0.125,), d_max=1.0, seed=0
        )
        ours = compute_fidelity(
            env, _TargetPolicyProxy(), mask_importance_fn(mask), fidelity_config, n_seeds=1
        )
        random = compute_fidelity(
            env,
            _TargetPolicyProxy(),
            random_importance_fn(0),
            fidelity_config,
            n_seeds=1,
        )
        self.assertGreater(
            ours["mean"][0],
            random["mean"][0],
            "explaining with the mask network must yield a higher fidelity score "
            "than the random explanation",
        )


class _TargetPolicyProxy:
    """``TorchPolicy``-compatible adapter around :class:`_TargetPolicy`."""

    def __init__(self):
        self._policy = _TargetPolicy()

    def act(self, obs, deterministic: bool = False):
        return self._policy.act(obs, deterministic)


if __name__ == "__main__":
    unittest.main()

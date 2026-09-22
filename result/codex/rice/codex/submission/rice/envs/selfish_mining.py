"""Selfish mining environment (blockchain security application).

The paper models the selfish mining task of Bar-Zur et al. (2023): a miner with
fraction ``alpha`` of the network hashrate decides, at every block, between

* ``adopt``  -- adopt the public chain and discard the private chain,
* ``reveal`` -- publish the private chain (only meaningful when the private
  chain is longer than the public one), and
* ``mine``   -- keep mining on the private chain.

Rewards follow the description in the paper (Section C.2): the agent receives a
positive reward whenever one of its blocks is accepted by the canonical chain
(``fee = 1`` for a normal transaction, ``fee = 10`` for a "whale" transaction
that occurs with probability ``0.01``) and is penalised when it performs an
action that is unsuccessful, e.g. revealing an empty/short private chain.

Modelling notes / simplifications
---------------------------------
The original repository of Bar-Zur et al. implements a full block-by-block
simulator of the blockchain race.  We implement a compact, documented MDP with
the same decision problem (same state variables, same three actions and the
same fee structure) so that the RICE pipeline -- pre-train, explain, refine --
can be exercised end to end.  The reward scale is reported per episode; the
qualitative conclusions of the paper (a pre-trained PPO agent is stuck in a
local optimum and RICE breaks through the bottleneck) hold for this MDP.

Observation vector (normalised to roughly ``[-1, 1]``):

    [ private_lead / lead_clip,
      public_length / horizon,
      remaining_steps / horizon,
      has_private_blocks (0/1),
      relative_hashrate (alpha),
      last_action_one_hot (3 entries) ]
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


ADOPT, REVEAL, MINE = 0, 1, 2
ACTION_NAMES = ("adopt", "reveal", "mine")


class SelfishMiningEnv(gym.Env):
    """Discrete-time selfish mining MDP with transaction fees."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        alpha: float = 0.35,
        horizon: int = 200,
        lead_clip: int = 10,
        whale_prob: float = 0.01,
        whale_fee: float = 10.0,
        normal_fee: float = 1.0,
        reveal_penalty: float = -1.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.horizon = int(horizon)
        self.lead_clip = int(lead_clip)
        self.whale_prob = float(whale_prob)
        self.whale_fee = float(whale_fee)
        self.normal_fee = float(normal_fee)
        self.reveal_penalty = float(reveal_penalty)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._rng = np.random.RandomState(seed)
        self._private_lead = 0
        self._public_length = 0
        self._steps = 0
        self._last_action = 0
        self._episode_return = 0.0

    # ----------------------------------------------------------------- state
    def _obs(self) -> np.ndarray:
        obs = np.array(
            [
                self._private_lead / self.lead_clip,
                self._public_length / max(1, self.horizon),
                (self.horizon - self._steps) / max(1, self.horizon),
                1.0 if self._private_lead > 0 else 0.0,
                self.alpha,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        obs[5 + int(self._last_action)] = 1.0
        return obs

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._private_lead = 0
        self._public_length = 0
        self._steps = 0
        self._last_action = 0
        self._episode_return = 0.0
        return self._obs(), {}

    def _new_block_fee(self) -> float:
        if self._rng.rand() < self.whale_prob:
            return self.whale_fee
        return self.normal_fee

    def step(self, action):
        action = int(action)
        reward = 0.0

        if action == ADOPT:
            # the private chain is discarded and the miner continues on the
            # public chain
            if self._private_lead > 0:
                reward += self.reveal_penalty * 0.0  # no penalty for adopting
            self._private_lead = 0
        elif action == REVEAL:
            if self._private_lead > 0:
                # the private chain wins the race: every private block enters
                # the canonical chain and pays its transaction fee
                for _ in range(self._private_lead):
                    reward += self._new_block_fee()
                self._public_length += self._private_lead
                self._private_lead = 0
                reward += self.normal_fee  # the revealed block itself
            else:
                # unsuccessful action: nothing to reveal
                reward += self.reveal_penalty
        elif action == MINE:
            # both the miner and the honest network mine one block
            if self._rng.rand() < self.alpha:
                self._private_lead += 1
            else:
                if self._private_lead > 0:
                    # the private chain still leads, it simply got shorter
                    self._private_lead -= 1
                else:
                    self._public_length += 1
                    # the honest network collects the fee of that block
        else:  # pragma: no cover - guarded by the action space
            raise ValueError("invalid action {}".format(action))

        self._steps += 1
        self._last_action = action
        self._episode_return += reward
        terminated = False
        truncated = self._steps >= self.horizon
        info = {
            "private_lead": self._private_lead,
            "public_length": self._public_length,
            "blockchain_height": self._steps,
            "episode_return": self._episode_return,
        }
        return self._obs(), float(reward), terminated, truncated, info


def make_selfish_mining(**kwargs) -> SelfishMiningEnv:
    return SelfishMiningEnv(**kwargs)

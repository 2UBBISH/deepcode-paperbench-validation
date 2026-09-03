"""Selfish mining environment for RICE.

This module implements a Gymnasium-compatible selfish-mining MDP inspired by
Bar-Zur et al., "Efficient MDP Analysis for Selfish-Mining Strategies".  The
action space is a discrete set of {Adopt(l), Reveal(l), Mine} commands and the
reward is the attacker's relative revenue (block rewards + transaction fees).

The implementation is intentionally self-contained so that the RICE pipeline can
be exercised even when the original simulator is unavailable.
"""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


class SelfishMiningEnv(gym.Env):
    """Simplified selfish-mining environment.

    State
    -----
    The state is a vector encoding:
      - ``a``: length of the attacker's private chain.
      - ``b``: length of the public (honest) chain since the last common block.
      - ``fork``: 0/1 flag indicating whether a fork is active.
      - ``total_mined``: total blocks mined in the episode (for revenue
        normalisation).
      - ``attacker_blocks``: blocks already secured by the attacker.

    Actions
    -------
    The action space is discrete and contains:
      - ``Mine``
      - ``Adopt(l)`` for ``l = 0 .. max_adopt``
      - ``Reveal(l)`` for ``l = 1 .. max_reveal``

    Rewards
    -------
    The reward is the attacker's *relative revenue* in the step, i.e. the
    fraction of blocks/transaction fees captured by the attacker minus the
    fraction captured by the honest network.  Whale transactions appear with
    probability ``whale_prob`` and carry fee ``whale_fee``; normal transactions
    carry fee ``normal_fee``.

    Parameters
    ----------
    alpha:
        Attacker hash-rate share (``0 < alpha < 0.5``).
    gamma:
        Fraction of honest miners that mine on the attacker's block during a
        race (``0 <= gamma <= 1``).
    max_steps:
        Maximum number of environment steps per episode.
    max_chain:
        Cap on private/public chain lengths; larger values are clipped.
    whale_prob:
        Probability that a mined block carries a whale transaction fee.
    whale_fee:
        Value of a whale transaction fee.
    normal_fee:
        Value of a normal transaction fee.
    block_reward:
        Base block reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        alpha: float = 0.35,
        gamma: float = 0.5,
        max_steps: int = 500,
        max_chain: int = 10,
        whale_prob: float = 0.01,
        whale_fee: float = 10.0,
        normal_fee: float = 1.0,
        block_reward: float = 1.0,
    ) -> None:
        super().__init__()
        assert 0.0 < alpha < 0.5, "alpha must be in (0, 0.5)"
        assert 0.0 <= gamma <= 1.0, "gamma must be in [0, 1]"

        self.alpha = alpha
        self.gamma = gamma
        self.max_steps = max_steps
        self.max_chain = max_chain
        self.whale_prob = whale_prob
        self.whale_fee = whale_fee
        self.normal_fee = normal_fee
        self.block_reward = block_reward

        # Action space: Mine + Adopt(0..max_chain) + Reveal(1..max_chain)
        self.n_adopt = max_chain + 1
        self.n_reveal = max_chain
        self.action_space = gym.spaces.Discrete(1 + self.n_adopt + self.n_reveal)

        # Observation: [a, b, fork, total_mined, attacker_blocks]
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=float(max_chain * 2),
            shape=(5,),
            dtype=np.float32,
        )

        self._state: Optional[Dict[str, Any]] = None
        self._step_count = 0

    # ------------------------------------------------------------------
    # Action decoding
    # ------------------------------------------------------------------
    def _decode_action(self, action: int) -> Tuple[str, int]:
        """Return (action_name, parameter l)."""
        if action == 0:
            return "Mine", 0
        action -= 1
        if action < self.n_adopt:
            return "Adopt", int(action)
        action -= self.n_adopt
        return "Reveal", int(action + 1)

    # ------------------------------------------------------------------
    # Fee sampling
    # ------------------------------------------------------------------
    def _sample_fee(self, rng: np.random.Generator) -> float:
        if rng.random() < self.whale_prob:
            return self.whale_fee
        return self.normal_fee

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------
    def _resolve_mine(
        self, state: Dict[str, Any], rng: np.random.Generator
    ) -> Tuple[Dict[str, Any], float, bool]:
        """Execute one mining round and return (new_state, reward, done)."""
        a, b = state["a"], state["b"]
        attacker_revenue = 0.0
        honest_revenue = 0.0
        total_blocks = 0

        if rng.random() < self.alpha:
            # Attacker mines a block privately.
            a = min(a + 1, self.max_chain)
        else:
            # Honest network mines a block.
            if a == 0:
                # No private chain: honest chain extends publicly.
                b = min(b + 1, self.max_chain)
            elif a > b:
                # Attacker still ahead; honest block is on the public chain.
                b = min(b + 1, self.max_chain)
            elif a == b:
                # Race condition: gamma fraction follows attacker's revealed
                # block, the rest follows the honest block.
                if rng.random() < self.gamma:
                    # Attacker wins the race.
                    attacker_revenue += a
                    total_blocks += a + 1
                    a, b = 0, 0
                else:
                    # Honest chain wins.
                    honest_revenue += b + 1
                    total_blocks += b + 1
                    a, b = 0, 0
            else:
                # a < b: honest chain is strictly longer; honest wins.
                honest_revenue += b + 1
                total_blocks += b + 1
                a, b = 0, 0

        state["a"] = a
        state["b"] = b
        state["total_mined"] += total_blocks
        state["attacker_blocks"] += attacker_revenue

        reward = self._relative_revenue(attacker_revenue, honest_revenue, total_blocks)
        return state, reward, False

    def _execute_adopt(
        self, state: Dict[str, Any], l: int, rng: np.random.Generator
    ) -> Tuple[Dict[str, Any], float, bool]:
        """Adopt the public chain and optionally publish l private blocks."""
        a, b = state["a"], state["b"]
        l = int(np.clip(l, 0, a))

        attacker_revenue = 0.0
        honest_revenue = 0.0
        total_blocks = 0

        if l > 0 and a > 0:
            # Publish l blocks.  If l >= b the attacker wins the race;
            # otherwise the honest chain remains dominant.
            if l > b:
                attacker_revenue += l
                total_blocks += max(b + 1, l)
                honest_revenue += max(0.0, b + 1 - l)
            elif l == b:
                # Tie: gamma fraction follows attacker.
                if rng.random() < self.gamma:
                    attacker_revenue += l
                    total_blocks += l + 1
                else:
                    honest_revenue += b + 1
                    total_blocks += b + 1
            else:
                honest_revenue += b + 1
                total_blocks += b + 1
        else:
            # Plain adopt: give up private chain.
            honest_revenue += b
            total_blocks += max(a, b)

        state["a"] = 0
        state["b"] = 0
        state["fork"] = 0
        state["total_mined"] += total_blocks
        state["attacker_blocks"] += attacker_revenue

        reward = self._relative_revenue(attacker_revenue, honest_revenue, total_blocks)
        return state, reward, False

    def _execute_reveal(
        self, state: Dict[str, Any], l: int, rng: np.random.Generator
    ) -> Tuple[Dict[str, Any], float, bool]:
        """Reveal l private blocks without adopting the public chain."""
        a, b = state["a"], state["b"]
        l = int(np.clip(l, 1, a))

        attacker_revenue = 0.0
        honest_revenue = 0.0
        total_blocks = 0

        if l > b:
            # Attacker's revealed chain is longer; honest miners switch.
            attacker_revenue += l
            total_blocks += max(b + 1, l)
            honest_revenue += max(0.0, b + 1 - l)
            state["a"] = max(0, a - l)
            state["b"] = 0
            state["fork"] = 0
        elif l == b:
            # Tie: race condition.
            if rng.random() < self.gamma:
                attacker_revenue += l
                total_blocks += l + 1
                state["a"] = max(0, a - l)
                state["b"] = 0
                state["fork"] = 0
            else:
                honest_revenue += b + 1
                total_blocks += b + 1
                state["a"] = 0
                state["b"] = 0
                state["fork"] = 0
        else:
            # Revealed chain is shorter; honest chain stays dominant.
            # This is usually a sub-optimal move.
            honest_revenue += b + 1
            total_blocks += b + 1
            state["a"] = 0
            state["b"] = 0
            state["fork"] = 0

        state["total_mined"] += total_blocks
        state["attacker_blocks"] += attacker_revenue

        reward = self._relative_revenue(attacker_revenue, honest_revenue, total_blocks)
        return state, reward, False

    def _relative_revenue(
        self, attacker_blocks: float, honest_blocks: float, total_blocks: float
    ) -> float:
        """Compute relative revenue for the step.

        If no block was resolved this step, return 0.  Otherwise return the
        attacker's share minus the honest share, scaled by block reward and
        expected transaction fees.
        """
        if total_blocks <= 0:
            return 0.0
        expected_fee = (
            self.whale_prob * self.whale_fee + (1.0 - self.whale_prob) * self.normal_fee
        )
        value = self.block_reward + expected_fee
        attacker_share = attacker_blocks / total_blocks
        honest_share = honest_blocks / total_blocks
        return (attacker_share - honest_share) * value

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        self._state = {
            "a": 0,
            "b": 0,
            "fork": 0,
            "total_mined": 0,
            "attacker_blocks": 0,
        }
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self._state is not None, "step() called before reset()"
        rng = self.np_random
        name, param = self._decode_action(action)

        if name == "Mine":
            state, reward, done = self._resolve_mine(self._state.copy(), rng)
        elif name == "Adopt":
            state, reward, done = self._execute_adopt(self._state.copy(), param, rng)
        else:  # Reveal
            state, reward, done = self._execute_reveal(self._state.copy(), param, rng)

        self._state = state
        self._step_count += 1
        terminated = done
        truncated = self._step_count >= self.max_steps

        info = {
            "a": int(state["a"]),
            "b": int(state["b"]),
            "fork": int(state["fork"]),
            "total_mined": int(state["total_mined"]),
            "attacker_blocks": int(state["attacker_blocks"]),
            "action_name": name,
            "action_param": param,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        s = self._state
        return np.array(
            [s["a"], s["b"], s["fork"], s["total_mined"], s["attacker_blocks"]],
            dtype=np.float32,
        )

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass


class SelfishMiningEnvWrapper(gym.Wrapper):
    """Thin wrapper that normalises rewards and exposes a stable interface.

    The paper reports *revenue* rather than raw step reward, so this wrapper
    accumulates episode revenue and returns it as ``info["episode_revenue"]``.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._episode_revenue = 0.0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._episode_revenue = 0.0
        return self.env.reset(seed=seed, options=options)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_revenue += reward
        if terminated or truncated:
            info["episode_revenue"] = self._episode_revenue
        return obs, reward, terminated, truncated, info


def make_selfish_mining_env(
    alpha: float = 0.35,
    gamma: float = 0.5,
    max_steps: int = 500,
    max_chain: int = 10,
    whale_prob: float = 0.01,
    whale_fee: float = 10.0,
    normal_fee: float = 1.0,
    block_reward: float = 1.0,
    wrap_revenue: bool = True,
    seed: Optional[int] = None,
) -> gym.Env:
    """Factory for the selfish-mining environment used by RICE."""
    env = SelfishMiningEnv(
        alpha=alpha,
        gamma=gamma,
        max_steps=max_steps,
        max_chain=max_chain,
        whale_prob=whale_prob,
        whale_fee=whale_fee,
        normal_fee=normal_fee,
        block_reward=block_reward,
    )
    if wrap_revenue:
        env = SelfishMiningEnvWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
    return env

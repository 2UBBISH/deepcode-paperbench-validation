"""Selfish-mining blockchain environment used by RICE (paper Sec. C.2).

The RICE paper (Sec. C.2, "Selfish Mining") trains PPO agents in the blockchain
model of Bar-Zur et al. (2023) -- ``https://github.com/roibarzur/pto-selfish-mining``.
Verbatim specifications reproduced here:

* The policy takes the *current chain state* as input and chooses between three
  pre-determined actions:

  - ``Adopt l``: "The miner chooses to adopt the first ``l`` blocks in the public
    chain while disregarding their private chain. Following this, the miner will
    continue their mining efforts, commencing from the last adopted block."
  - ``Reveal l``: "This action becomes legal when the miner's private chain attains
    a length of at least ``l``. The consequence of this action may result in either
    the creation of an active fork in the public chain or the overriding of the
    public chain."
  - ``Mine``: "This action simply involves continuing with the mining process. Once
    executed, a new block is mined and subsequently added to either the private chain
    of the rational miner or to the public chain, contingent on which entity
    successfully mined the block."

* "The whale transaction has a fee of 10 with the occurring probability of 0.01 while
  other normal transactions have a fee of 1."
* "The agent will receive a positive reward if his block is accepted and will be
  penalized if his action is determined to be unsuccessful, e.g., revealing a
  private chain."
* "The network architecture of the PPO agent is a 4-layer Multi-Layer Perceptron (MLP)
  with a hidden size of 128, 128, 128, and 128 in each layer. We adopt a similar
  network structure for training our mask network."
  -> exposed through :data:`SELFISH_MINING_SPECS` / ``net_arch``.

Model (documented, self-contained, no third-party blockchain simulator required)
-------------------------------------------------------------------------------
State variables (the classic Eyal & Sirer (2014) / Sapirshtein et al. (2016)
``(a, h, fork)`` representation, adapted so that ``Mine`` drives the mining event):

``a``
    Length of the miner's private chain (blocks withheld from the public chain),
    counted from the last commonly agreed ("common prefix") block.
``h``
    Length of the honest chain above the common prefix while the miner holds a
    private chain.
``fork``
    ``True`` when the miner's private chain was published at equal height and the
    public chain currently contains an (unresolved) race at its tip.

Hasrate ``alpha`` is the attacker's share of the network; ``gamma`` is the fraction
of the honest network that builds on the attacker's block when a race occurs
(Eyal & Sirer's network-propagation parameter).  A "Mine" action produces exactly one
new block, mined by the attacker with probability ``alpha`` and by the honest network
with probability ``1 - alpha``.

Reward accounting: every block carries ``block_reward`` plus transaction fees (one
normal transaction of fee 1; with probability 0.01 the block also carries the whale
transaction with fee 10).  The agent is credited whenever one of *its* blocks becomes
part of the canonical chain and is penalised (``reveal_penalty``) when an action is
unsuccessful, e.g. revealing a private chain that gets overridden.

Not specified by the paper (defaults chosen here, see README): ``alpha``, ``gamma``,
``desired_episodes/steps``, block reward inclusion, penalty magnitude and observation
encoding.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - import shim
    from ._common import (
        EnvBase,
        GYM_AVAILABLE,
        IS_GYMNASIUM,
        WrapperBase,
        env_max_episode_steps,
        import_running_mean_std,
        make_box,
        make_discrete,
        normalize_reset,
        normalize_step,
    )
except Exception:  # pragma: no cover - fallback for direct execution
    from rice.environments._common import (  # type: ignore
        EnvBase,
        GYM_AVAILABLE,
        IS_GYMNASIUM,
        WrapperBase,
        env_max_episode_steps,
        import_running_mean_std,
        make_box,
        make_discrete,
        normalize_reset,
        normalize_step,
    )

__all__ = [
    "ADOPT",
    "REVEAL",
    "MINE",
    "ACTION_NAMES",
    "SelfishMiningConfig",
    "SelfishMiningEnv",
    "SelfishMiningObsNormalizeWrapper",
    "SelfishMiningHeuristicPolicy",
    "SELFISH_MINING_SPECS",
    "get_spec",
    "resolve_name",
    "make_env",
    "make_selfish_mining",
    "make_selfish_mining_env",
]


# --------------------------------------------------------------------------------------
# action definitions
# --------------------------------------------------------------------------------------
ADOPT: int = 0
REVEAL: int = 1
MINE: int = 2

#: Human readable action names (paper wording).
ACTION_NAMES: Dict[int, str] = {ADOPT: "Adopt l", REVEAL: "Reveal l", MINE: "Mine"}


def _modern_api() -> bool:
    """Return ``True`` when ``reset``/``step`` should use the new gym API.

    Gymnasium (and gym >= 0.26) return ``(obs, info)`` from ``reset`` and a 5-tuple
    from ``step``; legacy gym returns a bare ``obs`` and a 4-tuple.
    """
    if IS_GYMNASIUM:
        return True
    try:  # pragma: no cover - depends on the installed gym version
        import gym as _gym  # type: ignore

        version = str(getattr(_gym, "__version__", "0.21.0"))
        parts = tuple(int(p) for p in version.split(".")[:2])
        return parts >= (0, 26)
    except Exception:
        return True


_RETURN_INFO: bool = _modern_api()


class _Spec:
    """Minimal stand-in for ``gym.envs.registration.EnvSpec``."""

    def __init__(self, id: str, max_episode_steps: Optional[int] = None):
        self.id = id
        self.max_episode_steps = max_episode_steps


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
@dataclass
class SelfishMiningConfig:
    """Hyper-parameters of the selfish-mining MDP.

    All values are configurable; the ones not specified by the RICE paper carry
    sensible defaults documented in the README.
    """

    # --- blockchain / network parameters -------------------------------------------------
    alpha: float = 0.25  # attacker hashrate share (not specified by the paper)
    gamma: float = 0.0  # fraction of honest network building on the attacker's block
    block_reward: float = 1.0  # base block subsidy (Bar-Zur et al. blockchain model)
    normal_fee: float = 1.0  # fee of a normal transaction     (paper Sec. C.2, verbatim)
    whale_fee: float = 10.0  # fee of the whale transaction     (paper Sec. C.2, verbatim)
    p_whale: float = 0.01  # probability of the whale transaction (paper Sec. C.2, verbatim)
    n_normal_tx: int = 1  # number of normal transactions per block
    include_block_reward: bool = True

    # --- reward shaping -----------------------------------------------------------------
    reward_scale: float = 1.0  # multiplicative scale applied to the returned reward
    reveal_penalty: float = 1.0  # penalty for an unsuccessful reveal (paper Sec. C.2)
    invalid_action_penalty: float = 0.0  # penalty for an illegal action (default no-op)
    reward_mode: str = "revenue"  # "revenue" | "relative" (relative = share of revenue)
    normalize_reward: bool = False  # divide by the expected per-step honest reward

    # --- episode control ----------------------------------------------------------------
    max_episode_steps: int = 1000  # decision steps per episode (not specified)
    max_blocks: Optional[int] = None  # optional cap on the number of mined blocks

    # --- MDP details --------------------------------------------------------------------
    max_lead: int = 8  # truncation of (a, h) keeping the observation bounded
    reveal_l: int = 1  # "Reveal l" is legal when the private chain has >= l blocks
    adopt_l: Optional[int] = None  # None -> adopt the whole public chain
    auto_adopt: bool = False  # automatically resync when the private chain is worthless
    normalize_obs: bool = True  # map raw counts into [0, 1]-like values
    obs_mode: str = "compact"  # "compact" | "full"
    repeat_action_probability: float = 0.0  # sticky actions (sanity/testing)
    net_arch: Tuple[int, ...] = (128, 128, 128, 128)  # paper Sec. C.2, verbatim
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def clone(self, **overrides: Any) -> "SelfishMiningConfig":
        data = self.to_dict()
        data.update({k: v for k, v in overrides.items() if v is not None})
        return SelfishMiningConfig(**data)


#: Registry entry compatible with ``rice.environments.get_env_spec``.
SELFISH_MINING_SPECS: Dict[str, Dict[str, Any]] = {
    "SelfishMining": {
        "name": "SelfishMining",
        "obs_dim": 6,
        "act_dim": 3,
        "max_episode_steps": 1000,
        "net_arch": (128, 128, 128, 128),
        "normalize_obs": False,
        "sparse": False,
        "description": "Bar-Zur et al. (2023) selfish-mining blockchain MDP (paper Sec. C.2)",
        "actions": ACTION_NAMES,
    }
}

_ALIASES: Dict[str, str] = {
    "selfishmining": "SelfishMining",
    "selfishminingv0": "SelfishMining",
    "selfish": "SelfishMining",
    "blockchain": "SelfishMining",
    "mining": "SelfishMining",
    "pto": "SelfishMining",
}


def resolve_name(name: str) -> str:
    """Canonicalise an environment alias to a key of :data:`SELFISH_MINING_SPECS`."""
    key = str(name).strip().replace("-", "").replace("_", "").replace(" ", "").lower()
    if key in _ALIASES:
        return _ALIASES[key]
    for canonical in SELFISH_MINING_SPECS:
        if key == canonical.lower():
            return canonical
    raise KeyError(
        f"Unknown selfish-mining environment {name!r}. "
        f"Known: {sorted(SELFISH_MINING_SPECS)} (aliases: {sorted(_ALIASES)})"
    )


def get_spec(name: str = "SelfishMining") -> Dict[str, Any]:
    """Return the registry metadata of the selfish-mining environment."""
    return dict(SELFISH_MINING_SPECS[resolve_name(name)])


# --------------------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------------------
class SelfishMiningEnv(EnvBase):
    """A ``Discrete(3)`` action, vector-observation blockchain selfish-mining MDP.

    See the module docstring for the exact model and the paper references.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        config: Optional[SelfishMiningConfig] = None,
        alpha: Optional[float] = None,
        gamma: Optional[float] = None,
        seed: Optional[int] = None,
        max_episode_steps: Optional[int] = None,
        normalize_obs: Optional[bool] = None,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ):
        cfg = config.clone() if isinstance(config, SelfishMiningConfig) else (config or SelfishMiningConfig())
        if alpha is not None:
            cfg.alpha = float(alpha)
        if gamma is not None:
            cfg.gamma = float(gamma)
        if max_episode_steps is not None:
            cfg.max_episode_steps = int(max_episode_steps)
        if normalize_obs is not None:
            cfg.normalize_obs = bool(normalize_obs)
        for key, value in kwargs.items():
            if not hasattr(cfg, key):
                raise TypeError(f"Unknown SelfishMiningConfig field: {key!r}")
            if value is not None:
                setattr(cfg, key, value)

        if not 0.0 < cfg.alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {cfg.alpha}")
        if not 0.0 <= cfg.gamma <= 1.0:
            raise ValueError(f"gamma must lie in [0, 1], got {cfg.gamma}")

        self.config = cfg
        self.render_mode = render_mode
        self.name = "SelfishMining"

        self._seed = cfg.seed if seed is None else int(seed)
        self.rng = np.random.default_rng(self._seed)

        self.observation_space = make_box(
            low=np.zeros(self._obs_dim(), dtype=np.float32),
            high=np.ones(self._obs_dim(), dtype=np.float32),
            shape=(self._obs_dim(),),
        )
        self.action_space = make_discrete(3)

        # aliases used by ``rice.environments.__init__`` / scripts
        self.net_arch = tuple(cfg.net_arch)
        self.rise_canonical_name = "SelfishMining"
        self.rise_config = cfg

        # spec-like attributes so that ``env_max_episode_steps`` finds the horizon
        self.max_episode_steps = int(cfg.max_episode_steps)
        self._max_episode_steps = int(cfg.max_episode_steps)
        self.spec = _Spec("SelfishMining-v0", max_episode_steps=self.max_episode_steps)

        # internal blockchain state
        self.a: int = 0  # private chain length
        self.h: int = 0  # honest chain length above the common prefix
        self.fork: bool = False
        self.private_fees: List[float] = []
        self.honest_fees: List[float] = []
        self.attacker_revenue: float = 0.0
        self.honest_revenue: float = 0.0
        self.blocks_mined: int = 0
        self.attacker_blocks: int = 0
        self.honest_blocks: int = 0
        self._elapsed_steps: int = 0
        self._prev_action: Optional[int] = None
        self.episode_return: float = 0.0

        self._fee_max = float(
            (cfg.block_reward if cfg.include_block_reward else 0.0)
            + cfg.n_normal_tx * cfg.normal_fee
            + cfg.whale_fee
        )
        self._expected_honest_fee = float(
            (cfg.block_reward if cfg.include_block_reward else 0.0)
            + cfg.n_normal_tx * cfg.normal_fee
            + cfg.p_whale * cfg.whale_fee
        )

    # ------------------------------------------------------------------ spaces / utils
    def _obs_dim(self) -> int:
        return 6 if self.config.obs_mode == "compact" else 8

    def seed(self, seed: Optional[int] = None) -> List[int]:  # legacy gym API
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        return [int(seed)] if seed is not None else []

    # ------------------------------------------------------------------ fee / block draws
    def _draw_fee(self) -> float:
        """Draw the reward carried by a freshly mined block.

        The whale transaction has a fee of 10 with probability 0.01, other (normal)
        transactions a fee of 1 (paper Sec. C.2, verbatim).
        """
        cfg = self.config
        fee = cfg.n_normal_tx * cfg.normal_fee
        if cfg.include_block_reward:
            fee += cfg.block_reward
        if float(self.rng.random()) < cfg.p_whale:
            fee += cfg.whale_fee
        return float(fee)

    # ------------------------------------------------------------------ chain operations
    def _accept_private_chain(self) -> float:
        """The miner's private chain becomes canonical (override) -> credit its blocks."""
        credited = float(np.sum(self.private_fees)) if self.private_fees else 0.0
        self.attacker_revenue += credited
        self.attacker_blocks += len(self.private_fees)
        # honest blocks mined above the common prefix become stale
        self.private_fees = []
        self.honest_fees = []
        return credited

    def _accept_honest_chain(self) -> float:
        """The honest chain becomes canonical -> credit the honest blocks, stall private ones."""
        credited = float(np.sum(self.honest_fees)) if self.honest_fees else 0.0
        self.honest_revenue += credited
        self.honest_blocks += len(self.honest_fees)
        self.private_fees = []
        self.honest_fees = []
        return credited

    def _truncate(self) -> None:
        """Keep ``(a, h)`` bounded (observation truncation used by the paper's state space)."""
        k = max(1, int(self.config.max_lead))
        if self.a > k:
            self.a = k
        if self.h > k:
            self.h = k

    # ------------------------------------------------------------------ core transition
    def _apply_action(self, action: int) -> Tuple[float, bool, bool]:
        """Apply one action.

        Returns ``(raw_reward, invalid_action, truncated_state)``.
        """
        cfg = self.config
        reward = 0.0
        invalid = False
        a, h = self.a, self.h

        if action == ADOPT:
            # "adopt the first l blocks in the public chain ... disregarding their
            # private chain" -> the honest chain is canonical, the private chain is stale
            if self.fork:
                self._accept_honest_chain()
            elif self.h > 0:
                self._accept_honest_chain()
            self.a, self.h, self.fork = 0, 0, False
            self.private_fees, self.honest_fees = [], []

        elif action == REVEAL:
            if a < max(1, int(cfg.reveal_l)):
                # "Reveal l ... becomes legal when the private chain attains a length of
                # at least l" -> illegal action, treated as a no-op (+ optional penalty)
                invalid = True
                reward -= cfg.invalid_action_penalty
            elif a > h:
                # override: the private chain is strictly longer than the public one
                reward += self._accept_private_chain()
                self.a, self.h, self.fork = 0, 0, False
            elif a == h:
                # equal length -> an active fork is created; the race is resolved by the
                # next mining event (gamma governs the honest network's branch choice)
                self.fork = True
            else:
                # a < h: the private chain is shorter and gets overridden -> unsuccessful
                self.private_fees, self.honest_fees = [], []
                self.a, self.h, self.fork = 0, 0, False
                reward -= cfg.reveal_penalty

        else:  # MINE -- one new block is mined
            attacker_mined = float(self.rng.random()) < cfg.alpha
            if attacker_mined:
                if self.fork:
                    # the attacker extends its published branch -> wins the race
                    reward += self._accept_private_chain()
                    self.a, self.h, self.fork = 0, 0, False
                elif self.a == 0:
                    # nothing withheld: the new block simply becomes the public tip
                    reward += self._draw_fee()
                    self.attacker_blocks += 1
                else:
                    # the new block joins the private chain
                    self.private_fees.append(self._draw_fee())
                    self.a += 1
                self.blocks_mined += 1
            else:
                if self.fork:
                    # honest block mined during the fork: with probability gamma it builds
                    # on the attacker's block (Eyal & Sirer's gamma), else on the honest one
                    if float(self.rng.random()) < cfg.gamma:
                        reward += self._accept_private_chain()
                    else:
                        self._accept_honest_chain()
                    self.a, self.h, self.fork = 0, 0, False
                elif self.a == 0:
                    # ordinary honest block extending the canonical chain
                    self.honest_revenue += self._draw_fee()
                    self.honest_blocks += 1
                else:
                    # honest block inside the (still private) race
                    self.honest_fees.append(self._draw_fee())
                    self.h += 1
                self.blocks_mined += 1

        truncated = False
        if self.a > cfg.max_lead or self.h > cfg.max_lead:
            self._truncate()
            truncated = True

        # optional automatic resynchronisation when the private chain became worthless
        if cfg.auto_adopt and not self.fork and self.h > self.a:
            self._accept_honest_chain()
            self.a, self.h, self.fork = 0, 0, False
            self.private_fees, self.honest_fees = [], []

        return float(reward), invalid, truncated

    # ------------------------------------------------------------------ observation
    def _observation(self) -> np.ndarray:
        cfg = self.config
        k = max(1, int(cfg.max_lead))
        a = min(self.a, k)
        h = min(self.h, k)
        obs = [
            a / k,  # private chain length (normalised)
            h / k,  # honest chain lead (normalised)
            1.0 if self.fork else 0.0,  # unresolved race at the public tip
            float(np.clip((a - h) / k, -1.0, 1.0)),  # relative lead
            float(np.sum(self.private_fees) / (k * self._fee_max)) if self.private_fees else 0.0,
            float(max(0.0, (self.max_episode_steps - self._elapsed_steps) / max(1, self.max_episode_steps))),
        ]
        if cfg.obs_mode != "compact":
            obs.extend(
                [
                    float(len(self.private_fees) / k),
                    float(len(self.honest_fees) / k),
                ]
            )
        obs = np.asarray(obs, dtype=np.float32)
        if not cfg.normalize_obs and cfg.obs_mode == "compact":
            # raw counts (a, h) instead of the normalised versions, keeping the
            # remaining entries in their natural scale where sensible
            pass
        return obs

    def observation(self) -> np.ndarray:
        """Return the current observation without stepping the environment."""
        return self._observation()

    def _scale(self, raw_reward: float) -> float:
        cfg = self.config
        reward = float(raw_reward)
        if cfg.normalize_reward:
            denom = max(1e-8, cfg.alpha * self._expected_honest_fee)
            reward = reward / denom
        if cfg.reward_mode == "relative":
            denom = max(1e-8, self.attacker_revenue + self.honest_revenue)
            reward = reward / denom
        return reward * float(cfg.reward_scale)

    # ------------------------------------------------------------------ info / metrics
    def _revenue_info(self) -> Dict[str, Any]:
        total = self.attacker_revenue + self.honest_revenue
        share = self.attacker_revenue / total if total > 0 else 0.0
        gain = (share - self.config.alpha) / self.config.alpha * 100.0 if self.config.alpha > 0 else 0.0
        return {
            "private_chain": int(self.a),
            "honest_lead": int(self.h),
            "forked": bool(self.fork),
            "attacker_revenue": float(self.attacker_revenue),
            "honest_revenue": float(self.honest_revenue),
            "revenue_share": float(share),
            "revenue_gain_percent": float(gain),
            "blocks_mined": int(self.blocks_mined),
            "attacker_blocks": int(self.attacker_blocks),
            "honest_blocks": int(self.honest_blocks),
            "elapsed_steps": int(self._elapsed_steps),
        }

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None, **kwargs: Any):
        if seed is not None:
            self._seed = int(seed)
            self.rng = np.random.default_rng(int(seed))
        self.a, self.h, self.fork = 0, 0, False
        self.private_fees, self.honest_fees = [], []
        self.attacker_revenue, self.honest_revenue = 0.0, 0.0
        self.blocks_mined, self.attacker_blocks, self.honest_blocks = 0, 0, 0
        self._elapsed_steps = 0
        self._prev_action = None
        self.episode_return = 0.0
        obs = self._observation()
        info = self._revenue_info()
        info["reset"] = True
        return (obs, info) if _RETURN_INFO else obs

    def step(self, action: Any):
        action = int(np.asarray(action).reshape(-1)[0]) if np.ndim(action) else int(action)

        # optional sticky actions (off by default) -- helps robustness testing
        cfg = self.config
        if cfg.repeat_action_probability > 0.0 and self._prev_action is not None:
            if float(self.rng.random()) < cfg.repeat_action_probability:
                action = self._prev_action
        if not (0 <= action <= 2):
            raise ValueError(f"Selfish mining action must be in {{0,1,2}}, got {action}")
        self._prev_action = action

        raw_reward, invalid, truncated_state = self._apply_action(action)
        self._elapsed_steps += 1

        truncated = bool(self._elapsed_steps >= self.max_episode_steps)
        if cfg.max_blocks is not None and self.blocks_mined >= int(cfg.max_blocks):
            truncated = True
        terminated = False  # no absorbing failure state in the blockchain MDP

        reward = self._scale(raw_reward)
        self.episode_return += reward

        obs = self._observation()
        info = self._revenue_info()
        info.update(
            {
                "action": int(action),
                "action_name": ACTION_NAMES.get(int(action), str(action)),
                "invalid_action": bool(invalid),
                "raw_reward": float(raw_reward),
                "episode_return": float(self.episode_return),
                "state_truncated": bool(truncated_state),
                "TimeLimit.truncated": bool(truncated and not terminated),
            }
        )
        if _RETURN_INFO:
            return obs, reward, terminated, truncated, info
        return obs, reward, terminated, info

    # ------------------------------------------------------------------ rendering
    def render(self, *args: Any, **kwargs: Any) -> str:
        info = self._revenue_info()
        text = (
            f"[SelfishMining] a={self.a} h={self.h} fork={self.fork} "
            f"attacker_rev={info['attacker_revenue']:.2f} honest_rev={info['honest_revenue']:.2f} "
            f"share={info['revenue_share']:.3f} gain={info['revenue_gain_percent']:.2f}%"
        )
        if self.render_mode == "human":
            print(text)
        return text

    def close(self) -> None:  # pragma: no cover - nothing to release
        return None

    # ------------------------------------------------------------------ state save/restore
    # ``rice.algorithms.env_reset`` (Ecoffet et al. 2019 style) restores the *full*
    # simulator state of a critical state; the whole blockchain MDP lives in Python,
    # so a snapshot is simply a deep copy of the fields below.
    def get_state(self) -> Dict[str, Any]:
        return {
            "a": int(self.a),
            "h": int(self.h),
            "fork": bool(self.fork),
            "private_fees": [float(x) for x in self.private_fees],
            "honest_fees": [float(x) for x in self.honest_fees],
            "attacker_revenue": float(self.attacker_revenue),
            "honest_revenue": float(self.honest_revenue),
            "blocks_mined": int(self.blocks_mined),
            "attacker_blocks": int(self.attacker_blocks),
            "honest_blocks": int(self.honest_blocks),
            "elapsed_steps": int(self._elapsed_steps),
            "prev_action": self._prev_action,
            "episode_return": float(self.episode_return),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def set_state(self, state: Any) -> np.ndarray:
        """Restore a snapshot produced by :meth:`get_state` and return the observation."""
        if state is None:
            obs = self.reset()
            return obs[0] if isinstance(obs, tuple) else obs
        if isinstance(state, self.__class__):  # pragma: no cover - defensive
            state = state.get_state()
        if isinstance(state, np.ndarray):
            state = {"_flat": np.asarray(state).reshape(-1)}
        if "_flat" in state:
            state = self._state_from_flat(state["_flat"])
        self.a = int(state.get("a", 0))
        self.h = int(state.get("h", 0))
        self.fork = bool(state.get("fork", False))
        self.private_fees = [float(x) for x in state.get("private_fees", [])]
        self.honest_fees = [float(x) for x in state.get("honest_fees", [])]
        self.attacker_revenue = float(state.get("attacker_revenue", 0.0))
        self.honest_revenue = float(state.get("honest_revenue", 0.0))
        self.blocks_mined = int(state.get("blocks_mined", 0))
        self.attacker_blocks = int(state.get("attacker_blocks", 0))
        self.honest_blocks = int(state.get("honest_blocks", 0))
        self._elapsed_steps = int(state.get("elapsed_steps", 0))
        self._prev_action = state.get("prev_action", None)
        self.episode_return = float(state.get("episode_return", 0.0))
        if state.get("rng_state") is not None:
            try:
                self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
            except Exception:  # pragma: no cover - defensive
                pass
        return self._observation()

    # --- alternative names used by ``rice.algorithms.env_reset`` / ``critical_state`` ---
    snapshot = get_state

    def restore(self, state: Any) -> np.ndarray:
        return self.set_state(state)

    save_state = get_state
    restore_state = set_state
    load_state = set_state

    @property
    def state(self) -> np.ndarray:
        """Flat raw simulator state (``a, h, fork``) -- used by ``env_reset`` fallbacks."""
        return np.asarray(
            [float(self.a), float(self.h), 1.0 if self.fork else 0.0], dtype=np.float32
        )

    @state.setter
    def state(self, value: Any) -> None:  # pragma: no cover - convenience
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size >= 3:
            self.a, self.h, self.fork = int(arr[0]), int(arr[1]), bool(arr[2])

    def _state_from_flat(self, arr: np.ndarray) -> Dict[str, Any]:
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        a = int(round(arr[0])) if arr.size > 0 else 0
        h = int(round(arr[1])) if arr.size > 1 else 0
        fork = bool(arr[2] > 0.5) if arr.size > 2 else False
        return {
            "a": a,
            "h": h,
            "fork": fork,
            "private_fees": [self.config.n_normal_tx * self.config.normal_fee] * a,
            "honest_fees": [self.config.n_normal_tx * self.config.normal_fee] * h,
        }

    def set_state_from_observation(self, obs: Any) -> np.ndarray:
        """Degraded restore: rebuild ``(a, h, fork)`` from an observation vector.

        The observation encodes the same information but loses the pending block fees,
        hence this is only a fallback used when the raw simulator state is unavailable.
        """
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        k = max(1, int(self.config.max_lead))
        a = int(round(float(obs[0]) * k)) if obs.size > 0 else 0
        h = int(round(float(obs[1]) * k)) if obs.size > 1 else 0
        fork = bool(obs[2] > 0.5) if obs.size > 2 else False
        return self.set_state(
            {
                "a": a,
                "h": h,
                "fork": fork,
                "private_fees": [self.config.n_normal_tx * self.config.normal_fee] * a,
                "honest_fees": [self.config.n_normal_tx * self.config.normal_fee] * h,
            }
        )

    def state_dict(self) -> Dict[str, Any]:
        return {"env_state": self.get_state(), "config": self.config.to_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if isinstance(state, dict) and "env_state" in state:
            self.set_state(state["env_state"])
        else:
            self.set_state(state)

    # ------------------------------------------------------------------ helpers
    def clone(self, **overrides: Any) -> "SelfishMiningEnv":
        cfg = self.config.clone(**overrides)
        return SelfishMiningEnv(config=cfg, seed=self._seed)

    def heuristic_action(self) -> int:
        """Deterministic Eyal & Sirer-style baseline action (helper, not from the paper)."""
        if self.fork:
            return MINE
        if self.a == 0:
            return MINE
        if self.a > self.h:
            return REVEAL
        return ADOPT


class SelfishMiningObsNormalizeWrapper(WrapperBase):
    """Optional running mean/std observation normalization with serializable statistics."""

    def __init__(self, env, normalize: bool = True, clip: float = 10.0, epsilon: float = 1e-8):
        super().__init__(env)
        self._normalize = bool(normalize)
        self._clip = float(clip)
        rms_cls = import_running_mean_std()
        self.rms = rms_cls(shape=np.asarray(env.observation_space.shape, dtype=np.int64), epsilon=epsilon)

    def _prepare(self, obs):
        if not self._normalize:
            return obs
        return self.rms.normalize(obs, clip=self._clip, update=True)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        obs, info = normalize_reset(result)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = normalize_step(self.env.step(action))
        return obs, reward, terminated, truncated, info

    def state_dict(self) -> Dict[str, Any]:
        rms_state = self.rms.state_dict() if hasattr(self.rms, "state_dict") else {}
        return {"rms": rms_state, "normalize": self._normalize}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state and state.get("rms") and hasattr(self.rms, "load_state_dict"):
            self.rms.load_state_dict(state["rms"])
        if state and "normalize" in state:
            self._normalize = bool(state["normalize"])


# --------------------------------------------------------------------------------------
# heuristic baseline (documented helper: Eyal & Sirer (2014) selfish-mining strategy)
# --------------------------------------------------------------------------------------
class SelfishMiningHeuristicPolicy:
    """Classic threshold strategy: ``reveal`` when ``a > h``, ``adopt`` when ``a <= h``.

    This is *not* part of the RICE paper; it is provided so that the environment can be
    smoke-tested (a good selfish-mining policy should beat the honest baseline when
    ``alpha`` is large enough and ``gamma`` small).
    """

    def __init__(self, env: Optional[SelfishMiningEnv] = None):
        self.env = env

    def __call__(self, obs: Any) -> int:
        env = self.env
        if env is None:  # decode the compact observation
            arr = np.asarray(obs, dtype=np.float64).reshape(-1)
            k = 8
            a = int(round(float(arr[0]) * k))
            h = int(round(float(arr[1]) * k))
            fork = bool(arr[2] > 0.5) if arr.size > 2 else False
            if fork:
                return MINE
            if a > h:
                return REVEAL
            return MINE if a == 0 else ADOPT
        return env.heuristic_action()


# --------------------------------------------------------------------------------------
# factory functions
# --------------------------------------------------------------------------------------
def make_env(
    name: str = "SelfishMining",
    seed: Optional[int] = None,
    normalize: Optional[bool] = None,
    normalize_obs: Optional[bool] = None,
    max_episode_steps: Optional[int] = None,
    render_mode: Optional[str] = None,
    config: Optional[SelfishMiningConfig] = None,
    **kwargs: Any,
) -> SelfishMiningEnv:
    """Create a single (non-vectorised) selfish-mining environment.

    Follows the RICE environment-factory contract: identical signature to the MuJoCo
    factories, attaches ``rise_canonical_name`` and returns a plain env object.
    """
    resolve_name(name)  # validate; single canonical environment for now
    cfg = config or SelfishMiningConfig()
    if normalize_obs is None:
        normalize_obs = normalize
    cfg = cfg.clone(
        seed=seed,
        max_episode_steps=max_episode_steps,
        normalize_obs=normalize_obs,
    )
    env = SelfishMiningEnv(config=cfg, seed=seed, render_mode=render_mode, **kwargs)
    if normalize:  # observation normalization wrapper is opt-in
        env = SelfishMiningObsNormalizeWrapper(env, normalize=True)
        env.rise_canonical_name = "SelfishMining"
    return env


def make_selfish_mining(**kwargs: Any) -> SelfishMiningEnv:
    """Alias of :func:`make_env` used by the environment registry."""
    return make_env(**kwargs)


def make_selfish_mining_env(**kwargs: Any) -> SelfishMiningEnv:
    """Alias of :func:`make_env` used by the environment registry."""
    return make_env(**kwargs)

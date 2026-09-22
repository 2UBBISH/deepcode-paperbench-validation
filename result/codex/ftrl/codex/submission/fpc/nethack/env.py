"""NetHack Learning Environment wrappers.

The NLE observation used by the model has four components (Appendix B.1):

* ``glyphs`` -- the main dungeon screen encoded as character ids,
* ``colors`` -- the per-cell colour ids,
* ``blstats`` -- the player status vector (health, hunger, ...),
* ``message`` -- the textual message line.

The environment is rolled out until the agent dies, 150 steps pass without
progress, or 100k steps are taken (addendum).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .config import NetHackConfig


def _require_nle():
    try:
        import nle  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "NetHack requires the NLE package: pip install nle "
            "(see https://github.com/heiner/nle)."
        ) from exc


class NetHackEnv:
    """Thin wrapper around ``nle.env.NLE`` exposing the model's observation dict."""

    def __init__(self, config: NetHackConfig, seed: int = 0, character: Optional[str] = None) -> None:
        _require_nle()
        import gymnasium as gym

        self.config = config
        self.character = character or config.character
        self.env = gym.make(
            "NetHackScore-v0",
            observation_keys=("glyphs", "colors", "blstats", "message"),
            actions=None,
            character=self.character,
            max_episode_steps=100_000,
            no_progress_timeout=150,
        )
        self.env.reset(seed=seed)

    # ------------------------------------------------------------------
    def _obs(self, raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            "glyphs": np.asarray(raw["glyphs"]).astype(np.int64),
            "colors": np.asarray(raw["colors"]).astype(np.int64),
            "blstats": np.asarray(raw["blstats"]).astype(np.float32),
            "message": np.asarray(raw["message"]).astype(np.int64),
        }

    def reset(self) -> Dict[str, np.ndarray]:
        raw, _ = self.env.reset()
        return self._obs(raw)

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        raw, reward, terminated, truncated, info = self.env.step(action)
        return self._obs(raw), float(reward), bool(terminated or truncated), info

    def render_ascii(self) -> str:
        return "".join(chr(c) for c in np.asarray(self.env.unwrapped.tty_chars).flatten().tolist())

    def close(self) -> None:
        self.env.close()


def make_env_factory(config: NetHackConfig):
    """Return a factory ``seed -> NetHackEnv`` for the APPO rollout workers."""

    def factory(seed: int) -> NetHackEnv:
        return NetHackEnv(config, seed=seed)

    return factory

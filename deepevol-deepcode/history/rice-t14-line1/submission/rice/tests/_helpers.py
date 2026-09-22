"""Shared helpers for the RICE reproduction test-suite.

The helpers in this module have three jobs:

1. **Path bootstrap** -- make the ``rice`` package importable no matter whether
   the repository is checked out as::

       <workspace>/rice/rice/algorithms/...      # package nested in the repo
       <workspace>/rice/algorithms/...           # package == repo root

   Both layouts show up in this project, so the candidate roots are appended to
   ``sys.path`` (appended, not inserted, so pytest's own import machinery is not
   perturbed).

2. **Tolerant imports** -- ``import_optional`` tries ``rice.<mod>``,
   ``rice.rice.<mod>`` and ``<mod>`` so a test can reach the package regardless
   of which sys.path entry wins.

3. **Tiny stand-in environments / explanations** -- ``DummyEnv`` is a
   dependency-free (numpy only) gym-like environment with full simulator state
   save/restore, which lets the algorithm code (mask network, refining loop,
   fidelity evaluator, env-reset manager) be exercised on CPU without MuJoCo.
   It is *not* part of the paper; it only preserves the input/output contract
   used by the algorithms.
"""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np

# --------------------------------------------------------------------------- #
# 1. Path bootstrap
# --------------------------------------------------------------------------- #
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)                       # .../rice
WORKSPACE_DIR = os.path.dirname(REPO_DIR)                   # parent of repo

_CANDIDATE_ROOTS = [
    REPO_DIR,                          # package nested in the repo  (rice/rice)
    WORKSPACE_DIR,                     # package == repo root        (rice/)
    os.path.join(REPO_DIR, "rice"),    # doubly nested layout
]

for _root in _CANDIDATE_ROOTS:
    if _root and os.path.isdir(_root) and _root not in sys.path:
        sys.path.append(_root)


# --------------------------------------------------------------------------- #
# 2. Tolerant imports
# --------------------------------------------------------------------------- #
def _candidates(base: str, extra=()):
    out = []
    for name in (base,) + tuple(extra):
        if name.startswith("rice"):
            out.append(name)
        else:
            out.extend(["rice." + name, "rice.rice." + name, name])
    # de-duplicate, keep order
    seen, ordered = set(), []
    for name in out:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def import_optional(base: str, extra=()):
    """Import the first importable spelling of ``base`` or return ``None``."""
    last_error = None
    for name in _candidates(base, extra):
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - depends on the install
            last_error = exc
            continue
    _IMPORT_ERRORS[base] = last_error
    return None


def require(base: str, extra=()):
    """Import ``base`` or skip the calling test module when unavailable."""
    import pytest

    module = import_optional(base, extra)
    if module is None:
        pytest.skip(
            "optional module %r not importable (%s)" % (base, _IMPORT_ERRORS.get(base)),
            allow_module_level=True,
        )
    return module


_IMPORT_ERRORS = {}


def have_torch() -> bool:
    return import_optional("torch") is not None


def torch_or_skip():
    import pytest

    torch = import_optional("torch")
    if torch is None:
        pytest.skip("torch is not installed")
    return torch


def make_box_space(dim, low=-np.inf, high=np.inf):
    """Box space built with gym / gymnasium when available, else the shim."""
    for name in ("gym.spaces", "gymnasium.spaces"):
        try:
            mod = importlib.import_module(name)
            return mod.Box(
                low=np.full(dim, low, dtype=np.float32),
                high=np.full(dim, high, dtype=np.float32),
                dtype=np.float32,
            )
        except Exception:
            continue
    common = import_optional("environments._common")
    if common is None:  # pragma: no cover
        raise RuntimeError("neither gym nor rice.environments._common is importable")
    return common.make_box(float(low), float(high), (int(dim),))


def make_discrete_space(n):
    for name in ("gym.spaces", "gymnasium.spaces"):
        try:
            mod = importlib.import_module(name)
            return mod.Discrete(int(n))
        except Exception:
            continue
    common = import_optional("environments._common")
    if common is None:  # pragma: no cover
        raise RuntimeError("neither gym nor rice.environments._common is importable")
    return common.make_discrete(int(n))


# --------------------------------------------------------------------------- #
# 3. Stand-in environment / explanation objects
# --------------------------------------------------------------------------- #
class DummyEnv:
    """Minimal deterministic gym-like environment with state save/restore.

    Dynamics (intentionally action-independent so that *blinding* a step costs
    almost no reward)::

        s_{t+1} = clip(0.9 * s_t, -10, 10)
        r_t     = -0.01 * mean(a_t ** 2)          # small control cost

    Because the transition does not depend on the action, masking a step only
    costs the small control cost: with the paper's blinding bonus
    ``R' = R + alpha * a^m`` (alpha > 0) the mask network therefore has a clear
    incentive to blind steps, while with ``alpha = 0`` it prefers not to.
    That makes the env a well-posed probe for Algorithm 1's anti-collapse
    mechanism.  The environment is *not* part of the paper.
    """

    metadata = {"render.modes": []}

    def __init__(self, obs_dim: int = 4, act_dim: int = 2, max_episode_steps: int = 20, seed: int = 0):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.max_episode_steps = int(max_episode_steps)
        self.observation_space = make_box_space(self.obs_dim)
        self.action_space = make_box_space(self.act_dim, low=-1.0, high=1.0)
        self.reward_range = (-float("inf"), float("inf"))
        self.spec = None
        self._seed = int(seed)
        self._rng = np.random.default_rng(int(seed))
        self._obs = np.zeros(self.obs_dim, dtype=np.float64)
        self._step = 0
        self._return = 0.0
        self.reset(seed=int(seed))

    # -- gym API ----------------------------------------------------------- #
    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.default_rng(int(seed))
        self._obs = self._rng.normal(0.0, 1.0, size=self.obs_dim).astype(np.float64)
        self._step = 0
        self._return = 0.0
        return self._obs.astype(np.float32).copy(), {}

    def step(self, action, **kwargs):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size < self.act_dim:
            a = np.pad(a, (0, self.act_dim - a.size))
        a = np.clip(a[: self.act_dim], -1.0, 1.0)
        # action-independent transition: blinding a step does not move the state
        self._obs = np.clip(0.9 * self._obs, -10.0, 10.0)
        reward = -0.01 * float(np.mean(np.square(a)))
        self._step += 1
        self._return += reward
        terminated = self._step >= self.max_episode_steps
        info = {"step": self._step, "return": self._return, "x_position": float(self._obs[0])}
        return self._obs.astype(np.float32).copy(), float(reward), bool(terminated), False, info

    def render(self, *args, **kwargs):  # pragma: no cover - no rendering
        return None

    def close(self):
        return None

    def seed(self, seed=None):
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.default_rng(int(seed))
        return [self._seed]

    @property
    def unwrapped(self):
        return self

    # -- Go-Explore style state save / restore ----------------------------- #
    def get_state(self):
        return {
            "obs": self._obs.copy(),
            "step": int(self._step),
            "ret": float(self._return),
            "seed": int(self._seed),
        }

    def set_state(self, state):
        payload = state
        if not isinstance(payload, dict):
            for attr in ("env_state", "custom_state", "state", "observation"):
                if hasattr(payload, attr):
                    payload = getattr(payload, attr)
                    break
        if isinstance(payload, dict):
            if "obs" in payload:
                self._obs = np.asarray(payload["obs"], dtype=np.float64).reshape(-1)[: self.obs_dim].copy()
            self._step = int(payload.get("step", self._step))
            self._return = float(payload.get("ret", self._return))
            if "seed" in payload:
                self._seed = int(payload["seed"])
        else:
            arr = np.asarray(payload, dtype=np.float64).reshape(-1)
            self._obs = arr[: self.obs_dim].copy()
        return self.current_observation()

    def current_observation(self):
        return self._obs.astype(np.float32).copy()

    # convenience
    def observation_of(self, obs):
        """Return the reward the target agent would obtain from ``obs`` (unused)."""
        return -0.01 * float(np.mean(np.square(np.asarray(obs, dtype=np.float64))))


class LossOnlyEnv(DummyEnv):
    """Variant without any state-restore API (only the observation is reusable).

    Used to check that ``rice.algorithms.env_reset`` degrades gracefully when a
    simulator cannot expose its internal state.
    """

    def __init__(self, obs_dim: int = 3, act_dim: int = 2, max_episode_steps: int = 15, seed: int = 0):
        super().__init__(obs_dim=obs_dim, act_dim=act_dim, max_episode_steps=max_episode_steps, seed=seed)

    # deliberately no get_state / set_state / state attribute
    def get_state(self):  # pragma: no cover - shadowed below
        raise AttributeError("LossOnlyEnv does not expose its internal state")


# remove the inherited save/restore methods so the degraded path is exercised
del LossOnlyEnv.get_state
del LossOnlyEnv.set_state
del LossOnlyEnv.current_observation


class StubExplanation:
    """Duck-typed explanation object (mirrors ``MaskNetwork``'s public surface).

    ``mode="increasing"`` gives the i-th visited state a score ``(i+1)/N``, so
    the arg-max critical state is always the *last* state of the trajectory --
    which makes the selection logic easy to assert.
    """

    def __init__(self, mode: str = "increasing", value: float = 0.5):
        self.mode = mode
        self.value = float(value)
        self.calls = 0

    def _scores(self, states):
        self.calls += 1
        if hasattr(states, "detach"):
            states = states.detach().cpu().numpy()
        s = np.asarray(states, dtype=np.float64)
        single = s.ndim == 1
        s2 = s[None, :] if single else s
        n = int(s2.shape[0])
        if self.mode == "increasing":
            out = np.arange(1, n + 1, dtype=np.float64) / float(max(n, 1))
        elif self.mode == "constant":
            out = np.full(n, self.value, dtype=np.float64)
        else:  # pragma: no cover - defensive
            raise ValueError("unknown stub mode %r" % (self.mode,))
        return out[0] if single else out

    # the explanation interface used across rice.explanation / critical_state
    def importance(self, states, **kwargs):
        return self._scores(states)

    def score(self, states, **kwargs):
        return self._scores(states)

    def mask_prob_zero(self, states, **kwargs):
        return self._scores(states)

    def importances(self, states, **kwargs):
        return np.atleast_1d(self._scores(states))

    def __call__(self, states, **kwargs):
        return self._scores(states)

    def select_index(self, states, **kwargs):
        return int(np.argmax(np.atleast_1d(self._scores(states))))

    def best_index(self, states, **kwargs):
        return self.select_index(states)

    def reset(self, *args, **kwargs):
        return None

    def update(self, *args, **kwargs):
        return None


def zero_policy(act_dim: int = 2, dtype=np.float32):
    """A constant (deterministic) policy callable: ``obs -> zeros(act_dim)``."""

    def policy(obs, deterministic: bool = True, **kwargs):
        return np.zeros(int(act_dim), dtype=dtype)

    policy.act_dim = int(act_dim)
    return policy


def build_actor_critic(env, net_arch=(32, 32), seed: int = 0):
    """Build a ``rice.algorithms.ppo.ActorCritic`` for ``env`` (or skip)."""
    import pytest

    ppo = import_optional("algorithms.ppo")
    if ppo is None:
        pytest.skip("rice.algorithms.ppo not importable")
    torch = torch_or_skip()
    torch.manual_seed(int(seed))
    return ppo.ActorCritic(
        env.observation_space,
        env.action_space,
        net_arch=tuple(net_arch),
    )


def build_mask_network(env, net_arch=(32, 32), seed: int = 0):
    """Build a ``MaskNetwork`` for ``env`` (or skip)."""
    import pytest

    mask_module = import_optional("algorithms.mask_network")
    if mask_module is None:
        pytest.skip("rice.algorithms.mask_network not importable")
    torch = torch_or_skip()
    torch.manual_seed(int(seed))
    return mask_module.MaskNetwork(env.observation_space, net_arch=tuple(net_arch))


def reward_change_after_masking(env, policy, actions, window, rng):
    """Reference implementation of ``d = |R' - R|`` used to sanity-check the
    fidelity evaluator on a short deterministic episode."""
    obs, _ = env.reset(seed=0)
    base_return = 0.0
    for t, action in enumerate(actions):
        obs, reward, done, _trunc, _info = env.step(action)
        base_return += reward
        if done:
            break
    obs, _ = env.reset(seed=0)
    masked_return = 0.0
    for t, action in enumerate(actions):
        a = np.asarray(action, dtype=np.float32)
        if window[0] <= t < window[0] + window[1]:
            a = np.asarray(env.action_space.sample(), dtype=np.float32)
        obs, reward, done, _trunc, _info = env.step(a)
        masked_return += reward
        if done:
            break
    return abs(masked_return - base_return)

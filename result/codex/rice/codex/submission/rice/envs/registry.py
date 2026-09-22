"""Registry of the environments used in the paper.

The registry keeps every environment specific detail (simulator id, policy
architecture, default RICE hyper-parameters, maximum per-episode reward used by
the fidelity metric, ...) in one place so that the experiment scripts can loop
over applications exactly like the paper does.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple


@dataclasses.dataclass
class EnvSpec:
    """Static description of one application."""

    name: str
    family: str  # "mujoco" | "sparse_mujoco" | "security"
    gym_id: Optional[str] = None
    #: per-episode reward cap used by the fidelity metric (see ``fidelity.py``)
    d_max: float = 1.0
    #: number of environment steps of one episode (used for budget estimates)
    horizon: int = 1000
    #: hidden sizes of the mask network (paper: Appendix C.1 / authors)
    mask_hidden: Tuple[int, ...] = (64, 64)
    #: hidden sizes of the policy/value network of the target agent
    policy_hidden: Tuple[int, ...] = (64, 64)
    #: default RICE hyper-parameters (Appendix C.3)
    alpha: float = 0.01
    #: multiplier applied to the environment reward while training the mask
    #: network.  ``None`` = normalise by the running standard deviation of the
    #: episode returns (what the paper's SB3 ``VecNormalize`` pipeline does, and
    #: what keeps the ``alpha`` bonus meaningful); ``1.0`` = raw rewards.
    mask_reward_scale: Optional[float] = None
    #: normalised reward clipping (SB3 ``clip_reward``)
    mask_reward_clip: float = 10.0
    p: float = 0.25
    rnd_lambda: float = 0.01
    #: length of the trajectory used to pick a critical state (Algorithm 2)
    rollin_length: int = 128
    #: number of samples used to train one mask network (Table 4)
    mask_samples: int = 300_000
    #: number of steps the mask network is trained on (Algorithm 1 budget)
    mask_iterations: int = 300
    #: sample budget / length of the refining phase
    refine_steps: int = 300_000
    #: whether the environment has sparse rewards (performance during
    #: refining is reported instead of the final reward)
    sparse_rewards: bool = False
    notes: str = ""

    def make(self, **kwargs):
        return make_env(self.name, **kwargs)


ENV_SPECS: Dict[str, EnvSpec] = {
    # ------------------------------------------------------------- MuJoCo
    "Hopper": EnvSpec(
        name="Hopper",
        family="mujoco",
        gym_id="Hopper-v4",
        d_max=4000.0,
        mask_hidden=(64, 64),
        policy_hidden=(64, 64),
        p=0.25,
        rnd_lambda=0.01,
        notes="Hopper-v3 in the paper; gymnasium exposes Hopper-v4.",
    ),
    "Walker2d": EnvSpec(
        name="Walker2d",
        family="mujoco",
        gym_id="Walker2d-v4",
        d_max=6000.0,
        mask_hidden=(64, 64),
        policy_hidden=(64, 64),
        p=0.5,
        notes="Observation normalisation is performed by VecNormalize (paper).",
    ),
    "Reacher": EnvSpec(
        name="Reacher",
        family="mujoco",
        gym_id="Reacher-v4",
        d_max=20.0,
        horizon=50,
        mask_hidden=(64, 64),
        policy_hidden=(64, 64),
        p=0.25,
        notes="Reacher-v2 in the paper; episodes last 50 steps.",
    ),
    "HalfCheetah": EnvSpec(
        name="HalfCheetah",
        family="mujoco",
        gym_id="HalfCheetah-v4",
        d_max=15000.0,
        mask_hidden=(64, 64),
        policy_hidden=(64, 64),
        p=0.25,
    ),
    # ------------------------------------------------------ sparse MuJoCo
    "SparseHopper": EnvSpec(
        name="SparseHopper",
        family="sparse_mujoco",
        gym_id="Hopper-v4",
        d_max=1000.0,
        mask_hidden=(64, 64),
        policy_hidden=(64, 64),
        p=0.5,
        sparse_rewards=True,
        notes="reward = x if x > 0.6 else 0 (Mazoure et al., 2019).",
    ),
    "SparseWalker2d": EnvSpec(
        name="SparseWalker2d",
        family="sparse_mujoco",
        gym_id="Walker2d-v4",
        d_max=1000.0,
        mask_hidden=(64, 64),
        policy_hidden=(64, 64),
        p=0.5,
        sparse_rewards=True,
        notes="refining result is in scope; hyper-parameter sensitivity is not.",
    ),
    "SparseHalfCheetah": EnvSpec(
        name="SparseHalfCheetah",
        family="sparse_mujoco",
        gym_id="HalfCheetah-v4",
        d_max=5000.0,
        mask_hidden=(64, 64),
        policy_hidden=(64, 64),
        p=0.5,
        sparse_rewards=True,
        notes="reward = x if x > 5 else 0 (Mazoure et al., 2019).",
    ),
    # ---------------------------------------------------- security tasks
    "SelfishMining": EnvSpec(
        name="SelfishMining",
        family="security",
        gym_id=None,
        d_max=100.0,
        horizon=200,
        mask_hidden=(128, 128, 128, 128),
        policy_hidden=(128, 128, 128, 128),
        p=0.25,
        # the simplified mining MDP of this repository pays dense per-block
        # rewards, so alpha has to be larger than the paper's 0.01 to keep the
        # mask network out of the degenerate "never blind" / "always blind"
        # corners (see docs/EXPERIMENTS.md).
        alpha=0.05,
        rnd_lambda=0.01,
        rollin_length=64,
        mask_samples=1_500_000,
        mask_iterations=1500,
        refine_steps=500_000,
        notes="bar-zur et al. 2023 style selfish mining MDP.",
    ),
    "CageChallenge2": EnvSpec(
        name="CageChallenge2",
        family="security",
        gym_id=None,
        d_max=100.0,
        horizon=100,
        mask_hidden=(64, 64, 64),
        policy_hidden=(64, 64, 64),
        p=0.25,
        rnd_lambda=0.01,
        mask_samples=10_000_000,
        mask_iterations=2000,
        refine_steps=1_000_000,
        notes="requires the CybORG simulator (pip install CybORG).",
    ),
    "AutoDriving": EnvSpec(
        name="AutoDriving",
        family="security",
        gym_id=None,
        d_max=100.0,
        horizon=1000,
        mask_hidden=(256, 256),
        policy_hidden=(256, 256),
        p=0.25,
        rnd_lambda=0.01,
        mask_samples=2_442_360,
        mask_iterations=1000,
        refine_steps=1_000_000,
        notes="requires metadrive-simulator (Macro-v1 environment).",
    ),
}


#: applications reported in the main paper (Table 1 / Figure 2)
PAPER_APPLICATIONS: Sequence[str] = (
    "Hopper",
    "Walker2d",
    "Reacher",
    "HalfCheetah",
    "SelfishMining",
    "CageChallenge2",
    "AutoDriving",
)
SPARSE_APPLICATIONS: Sequence[str] = (
    "SparseHopper",
    "SparseWalker2d",
    "SparseHalfCheetah",
)
MUJOCO_APPLICATIONS: Sequence[str] = (
    "Hopper",
    "Walker2d",
    "Reacher",
    "HalfCheetah",
)


def list_envs(family: Optional[str] = None) -> List[str]:
    if family is None:
        return list(ENV_SPECS)
    return [name for name, spec in ENV_SPECS.items() if spec.family == family]


def _make_mujoco(spec: EnvSpec, seed: Optional[int] = None, **kwargs):
    import gymnasium as gym

    from rice.envs.adapters import MujocoStatefulEnv

    env = gym.make(spec.gym_id, **kwargs)
    if seed is not None:
        env.reset(seed=seed)
    return MujocoStatefulEnv(env, name=spec.name)


def _make_sparse_mujoco(spec: EnvSpec, seed: Optional[int] = None, **kwargs):
    import gymnasium as gym

    from rice.envs.adapters import MujocoStatefulEnv
    from rice.envs.sparse_reward import make_sparse

    base_name = spec.gym_id.split("-")[0]
    env = gym.make(spec.gym_id, **kwargs)
    env = make_sparse(env, base_name)
    if seed is not None:
        env.reset(seed=seed)
    return MujocoStatefulEnv(env, name=spec.name)


def _make_security(spec: EnvSpec, seed: Optional[int] = None, **kwargs):
    if spec.name == "SelfishMining":
        from rice.envs.selfish_mining import SelfishMiningEnv
        from rice.envs.adapters import DictStatefulEnv

        return DictStatefulEnv(SelfishMiningEnv(seed=seed, **kwargs), name=spec.name)
    if spec.name == "CageChallenge2":
        from rice.envs.cage_challenge import CageChallengeEnv

        return CageChallengeEnv(**kwargs)
    if spec.name == "AutoDriving":
        from rice.envs.meta_drive import MetaDriveStatefulEnv

        return MetaDriveStatefulEnv(**kwargs)
    raise KeyError(spec.name)


def make_env(name: str, seed: Optional[int] = None, **kwargs):
    """Instantiate (a stateful wrapper around) the environment ``name``."""
    if name not in ENV_SPECS:
        raise KeyError(
            "unknown environment {!r}; available: {}".format(name, sorted(ENV_SPECS))
        )
    spec = ENV_SPECS[name]
    if spec.family == "mujoco":
        return _make_mujoco(spec, seed=seed, **kwargs)
    if spec.family == "sparse_mujoco":
        return _make_sparse_mujoco(spec, seed=seed, **kwargs)
    return _make_security(spec, seed=seed, **kwargs)


def make_stateful_env(name: str, **kwargs):
    """Alias of :func:`make_env` (all environments are stateful)."""
    return make_env(name, **kwargs)

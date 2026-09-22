"""ExORL (walker / cheetah) evaluation tasks.

Section 5: "We consider the walker and cheetah domains ... To examine zero-shot
capabilities, we examine transfer to the standard reward functions consisting
of forward/backward velocity, along with goal-reaching to random states in the
dataset."  All ExORL runs use the **RND** dataset for each domain (addendum),
and online evaluation uses episodes of at most 1000 timesteps.

Appendix C.2 describes an important wrinkle: "FRE assumes that reward functions
must be pure functions of the environment state.  Because the Cheetah and
Walker environments utilize rewards that are a function of the underlying
physics, we append information about the physics onto the offline dataset
during encoder training."  Specifically:

    Walker:  horizontal_velocity(), torso_upright(), torso_height()
    Cheetah: speed()

These auxiliary values are computed from the MuJoCo state stored in the ExORL
``physics`` array.  For both domains ``physics = concat(qpos, qvel)`` with
``qpos = [x, z, rot, 6 joint angles]`` and ``qvel = [vx, vz, omega, 6 joint
velocities]``, so:

    horizontal_velocity = qvel[0]
    torso_upright       = cos(qpos[2])
    torso_height        = qpos[1]
    speed               = qvel[0]   (signed horizontal velocity; the DMC cheetah
                                     reward is ``tolerance(speed, 10, 10)``,
                                     which is 0 for backwards motion)

Goal-reaching tasks use the Euclidean distance in the *normalised* observation
space (each dimension divided by its standard deviation over the offline
dataset) with a threshold of 0.1; augmented information is not used for goal
distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fre.tasks.base import EvalTask, GoalReachingEvalTask, TaskSuite


WALKER_OBS_DIM = 24
CHEETAH_OBS_DIM = 17

# Number of physics dimensions appended for encoder training per domain.
AUX_DIMS = {"walker": 3, "cheetah": 1}

# Velocity thresholds for the four ExORL velocity tasks (addendum).
CHEETAH_VELOCITY_THRESHOLDS: Tuple[Tuple[str, float], ...] = (
    ("cheetah-run", 10.0),
    ("cheetah-walk", 1.0),
)
WALKER_VELOCITY_THRESHOLDS: Tuple[Tuple[str, float], ...] = (
    ("walker-vel-0.1", 0.1),
    ("walker-vel-1", 1.0),
    ("walker-vel-4", 4.0),
    ("walker-vel-8", 8.0),
)


# --------------------------------------------------------------------------------------
# Physics features (Appendix C.2)
# --------------------------------------------------------------------------------------
def walker_physics_features(physics: np.ndarray) -> np.ndarray:
    """``[horizontal_velocity, torso_upright, torso_height]`` for Walker."""
    physics = np.asarray(physics, dtype=np.float64)
    qpos = physics[..., :9]
    qvel = physics[..., 9:18]
    horizontal_velocity = qvel[..., 0]
    torso_upright = np.cos(qpos[..., 2])
    torso_height = qpos[..., 1]
    return np.stack([horizontal_velocity, torso_upright, torso_height], axis=-1).astype(np.float32)


def cheetah_physics_features(physics: np.ndarray) -> np.ndarray:
    """``[speed]`` (signed horizontal velocity) for Cheetah."""
    physics = np.asarray(physics, dtype=np.float64)
    qvel = physics[..., 9:18]
    return qvel[..., 0:1].astype(np.float32)


PHYSICS_FEATURE_FNS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "walker": walker_physics_features,
    "cheetah": cheetah_physics_features,
}


def augment_with_physics(domain: str, observations: np.ndarray, physics: np.ndarray) -> np.ndarray:
    """Append the domain-specific physics features to the observations."""
    feats = PHYSICS_FEATURE_FNS[domain](physics)
    return np.concatenate([np.asarray(observations, dtype=np.float32), feats], axis=-1)


# --------------------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------------------
class VelocityTask(EvalTask):
    """Reward for moving at (at least) a threshold horizontal velocity.

    From the addendum: "The reward is 1 if the velocity is at least the
    threshold value and linearly decays to 0 for values below the threshold
    value.  If the agent's horizontal velocity is in the opposite direction of
    the target velocity, the reward is 0."
    """

    group = "velocity"
    reward_range = (0.0, 1.0)
    max_episode_steps = 1000

    def __init__(
        self,
        name: str,
        threshold: float,
        velocity_index: int,
        backward: bool = False,
        max_episode_steps: int = 1000,
    ) -> None:
        self.name = name
        self.threshold = float(threshold)
        self.velocity_index = int(velocity_index)
        self.backward = bool(backward)
        self.max_episode_steps = int(max_episode_steps)

    def reward(self, obs, action=None, next_obs=None) -> np.ndarray:
        vel = np.asarray(obs, dtype=np.float64)[..., self.velocity_index]
        if self.backward:
            vel = -vel
        reward = np.clip(vel / self.threshold, 0.0, 1.0)
        return np.where(vel <= 0.0, 0.0, reward).astype(np.float32)


class NormalisedGoalTask(GoalReachingEvalTask):
    """Goal-reaching with distance measured in the normalised observation space."""

    group = "goal-reaching"

    def __init__(
        self,
        name: str,
        goal: np.ndarray,
        obs_std: np.ndarray,
        threshold: float = 0.1,
        max_episode_steps: int = 1000,
    ) -> None:
        self.name = name
        self.goal = np.asarray(goal, dtype=np.float32)
        self.obs_std = np.asarray(obs_std, dtype=np.float64)
        self.threshold = float(threshold)
        self.max_episode_steps = int(max_episode_steps)

    def reward(self, obs, action=None, next_obs=None) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        d = self.goal.shape[-1]
        # Only the (non-augmented) observation dimensions participate in the
        # goal distance, and every dimension is normalised by its dataset std.
        delta = (obs[..., :d] - self.goal) / self.obs_std
        dist = np.linalg.norm(delta, axis=-1)
        return np.where(dist < self.threshold, 0.0, -1.0).astype(np.float32)


# --------------------------------------------------------------------------------------
# Suite construction
# --------------------------------------------------------------------------------------
def make_velocity_tasks(domain: str, max_episode_steps: int = 1000) -> List[EvalTask]:
    """Build the 4 velocity tasks for ``walker`` or ``cheetah``."""
    if domain == "cheetah":
        tasks = []
        for name, thr in CHEETAH_VELOCITY_THRESHOLDS:
            index = CHEETAH_OBS_DIM  # ``speed`` is the first appended physics feature
            tasks.append(VelocityTask(name, thr, velocity_index=index, max_episode_steps=max_episode_steps))
            tasks.append(
                VelocityTask(
                    name + "-backwards",
                    thr,
                    velocity_index=index,
                    backward=True,
                    max_episode_steps=max_episode_steps,
                )
            )
        return tasks
    if domain == "walker":
        return [
            VelocityTask(name, thr, velocity_index=WALKER_OBS_DIM, max_episode_steps=max_episode_steps)
            for name, thr in WALKER_VELOCITY_THRESHOLDS
        ]
    raise ValueError(f"Unknown ExORL domain '{domain}'")


def make_goal_tasks(
    domain: str,
    goals: np.ndarray,
    obs_std: np.ndarray,
    max_episode_steps: int = 1000,
) -> List[EvalTask]:
    """Build the 5 fixed-goal tasks for a domain."""
    return [
        NormalisedGoalTask(
            name=f"{domain}-goal-{i}",
            goal=np.asarray(g, dtype=np.float32),
            obs_std=obs_std,
            max_episode_steps=max_episode_steps,
        )
        for i, g in enumerate(goals)
    ]


def make_exorl_suites(
    domain: str,
    goals: Optional[np.ndarray] = None,
    obs_std: Optional[np.ndarray] = None,
    max_episode_steps: int = 1000,
) -> List[TaskSuite]:
    """Task suites matching the ExORL rows of Table 1 for one domain."""
    suites = [
        TaskSuite(
            f"exorl-{domain}-velocity",
            make_velocity_tasks(domain, max_episode_steps),
            description="Average over velocity tasks",
        )
    ]
    if goals is not None and obs_std is not None:
        suites.append(
            TaskSuite(
                f"exorl-{domain}-goals",
                make_goal_tasks(domain, goals, obs_std, max_episode_steps),
                description="Average over 5 fixed goal-reaching tasks",
            )
        )
    return suites


def sample_fixed_goals(dataset, num_goals: int = 5, seed: int = 0) -> np.ndarray:
    """Sample the five fixed goal states used by ``exorl-*-goals``.

    The addendum specifies that the goals are "5 random states ... selected from
    the offline dataset and kept fixed throughout the online evaluation".
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=num_goals, replace=False)
    return np.asarray(dataset.observations[idx], dtype=np.float32)


# --------------------------------------------------------------------------------------
# FRE-hint prior reward families (Section 5.4 / Figure 6)
# --------------------------------------------------------------------------------------
from fre.reward_functions import BatchedRewardFunctions, RewardPrior  # noqa: E402  (kept near use)


class VelocityHintPrior(RewardPrior):
    """Prior family that spans "move at a specific velocity" rewards.

    Used by ``FRE-hint`` (Section 5.4): "For Cheetah-velocity and
    walker-velocity, the rewards are for moving at a specific velocity".  A
    random target velocity is sampled per batch element and the reward is the
    same shape of linearly-decaying function used by the evaluation tasks, so
    the prior is a superset of the downstream task family.
    """

    def __init__(
        self,
        velocity_index: int,
        max_speed: float = 10.0,
        allow_backward: bool = True,
        min_speed: float = 0.1,
    ) -> None:
        self.velocity_index = int(velocity_index)
        self.max_speed = float(max_speed)
        self.allow_backward = bool(allow_backward)
        self.min_speed = float(min_speed)

    def sample(self, batch_size: int, device) -> BatchedRewardFunctions:
        import torch

        thresholds = torch.empty(batch_size, device=device).uniform_(self.min_speed, self.max_speed)
        signs = torch.ones(batch_size, device=device)
        if self.allow_backward:
            signs = torch.where(torch.rand(batch_size, device=device) < 0.5, -1.0, 1.0)
        return _VelocityBatchedRewards(self.velocity_index, thresholds, signs)


class _VelocityBatchedRewards(BatchedRewardFunctions):
    def __init__(self, velocity_index: int, thresholds, signs) -> None:
        import torch

        self.velocity_index = int(velocity_index)
        self.thresholds = thresholds
        self.signs = signs
        b = thresholds.shape[0]
        self.r_min = torch.zeros(b, device=thresholds.device)
        self.r_max = torch.ones(b, device=thresholds.device)

    def reward(self, states):
        import torch

        lead = states.shape[1:-1]
        flat = states.reshape(states.shape[0], -1, states.shape[-1])
        vel = flat[..., self.velocity_index] * self.signs.unsqueeze(1)
        thr = self.thresholds.unsqueeze(1).expand(-1, vel.shape[1])
        out = torch.where(vel <= 0.0, torch.zeros_like(vel), (vel / thr).clamp(0.0, 1.0))
        return out.reshape(states.shape[0], *lead)


class DirectionHintPrior(RewardPrior):
    """Prior family spanning unit-direction movement rewards.

    Used by ``FRE-hint`` for ``ant-directional``: "the prior rewards are all
    rewards corresponding to movement in a unit (x, y) direction".  Rewards are
    the dot product between the agent's (x, y) velocity and a random unit
    direction, matching the evaluation reward up to a fixed scale.
    """

    def __init__(self, velocity_slice: Tuple[int, int] = (15, 17), velocity_reference: float = 5.0) -> None:
        self.velocity_slice = velocity_slice
        self.velocity_reference = float(velocity_reference)

    def sample(self, batch_size: int, device) -> BatchedRewardFunctions:
        import torch

        theta = torch.rand(batch_size, device=device) * (2 * np.pi)
        directions = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        return _DirectionBatchedRewards(self.velocity_slice, directions, self.velocity_reference)


class _DirectionBatchedRewards(BatchedRewardFunctions):
    def __init__(self, velocity_slice, directions, velocity_reference: float) -> None:
        import torch

        self.velocity_slice = velocity_slice
        self.directions = directions
        self.velocity_reference = float(velocity_reference)
        b = directions.shape[0]
        self.r_min = torch.full((b,), -1.0, device=directions.device)
        self.r_max = torch.ones(b, device=directions.device)

    def reward(self, states):
        import torch

        lead = states.shape[1:-1]
        flat = states.reshape(states.shape[0], -1, states.shape[-1])
        vel = flat[..., self.velocity_slice[0] : self.velocity_slice[1]]
        raw = (vel * self.directions.unsqueeze(1)).sum(dim=-1) / self.velocity_reference
        return raw.clamp(-1.0, 1.0).reshape(states.shape[0], *lead)


def build_exorl_hint_priors(domain: str) -> Dict[str, RewardPrior]:
    """Hint reward families for ExORL (Section 5.4)."""
    velocity_index = WALKER_OBS_DIM if domain == "walker" else CHEETAH_OBS_DIM
    max_speed = 10.0 if domain == "cheetah" else 8.0
    return {"velocity": VelocityHintPrior(velocity_index, max_speed=max_speed)}


def build_antmaze_hint_priors() -> Dict[str, RewardPrior]:
    """Hint reward families for AntMaze (Section 5.4)."""
    return {"direction": DirectionHintPrior()}

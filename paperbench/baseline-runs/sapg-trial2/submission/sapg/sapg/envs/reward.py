"""Reward terms for the SAPG manipulation environments.

The paper (SAPG: Split and Aggregate Policy Gradients) defines the AllegroKuka
task reward as a weighted sum of four terms::

    r_t = w1 * r_reach + w2 * r_lift + w3 * r_target + w4 * r_success

where

* ``r_reach``   - dense reward for moving the hand towards the object.
* ``r_lift``    - dense reward for lifting the object above the table.
* ``r_target``  - dense reward for moving the object towards the goal.
* ``r_success`` - sparse bonus once the object is within ``delta`` of the goal.

For the in-hand reorientation tasks (ShadowHand / AllegroHand) the reward is an
orientation error term plus a success bonus.

All functions operate on tensors so that they can be evaluated for tens of
thousands of parallel environments at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class RewardConfig:
    """Weights and scales for the AllegroKuka composite reward.

    The paper does not publish the exact weights (w1..w4); the defaults below
    follow the values commonly used by the underlying IsaacGymEnvs
    ``AllegroKuka`` task and are exposed here so they can be tuned per task.
    """

    w_reach: float = 1.0
    w_lift: float = 10.0
    w_target: float = 1.0
    w_success: float = 100.0

    # Scales used to normalise the individual dense terms.
    reach_scale: float = 1.0
    lift_scale: float = 1.0
    target_scale: float = 1.0

    # Height (in metres) above which the object counts as "lifted".
    lift_height: float = 0.05

    # Success tolerance (metres).  Curriculum anneals this from 7.5cm -> 1cm.
    success_tolerance: float = 0.075

    # Orientation reward (ShadowHand / AllegroHand).
    w_orientation: float = 1.0
    orientation_scale: float = 1.0

    # Optional per-term clipping.
    clip_dense: Optional[float] = None

    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual reward terms
# ---------------------------------------------------------------------------
def r_reach(
    hand_pos: torch.Tensor,
    object_pos: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Dense reward for bringing the hand close to the object.

    ``r_reach = -scale * ||hand_pos - object_pos||``

    Args:
        hand_pos: (..., 3) position of the hand / palm.
        object_pos: (..., 3) position of the object.
        scale: multiplier applied to the (negative) distance.

    Returns:
        (...,) tensor of reach rewards (<= 0).
    """
    dist = torch.norm(hand_pos - object_pos, dim=-1)
    return -scale * dist


def r_lift(
    object_pos: torch.Tensor,
    table_height: float = 0.0,
    lift_height: float = 0.05,
    scale: float = 1.0,
) -> torch.Tensor:
    """Dense reward for lifting the object above the table.

    The reward is the (clipped) height of the object above the table,
    normalised by ``lift_height`` so that it saturates at 1.0.

    Args:
        object_pos: (..., 3) object position.
        table_height: z-coordinate of the table surface.
        lift_height: height at which the lift reward saturates.
        scale: multiplier.

    Returns:
        (...,) tensor of lift rewards in [0, scale].
    """
    height = object_pos[..., 2] - table_height
    height = torch.clamp(height, min=0.0)
    return scale * torch.clamp(height / max(lift_height, 1e-6), max=1.0)


def r_target(
    object_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Dense reward for moving the object towards the goal.

    ``r_target = -scale * ||object_pos - goal_pos||``

    Args:
        object_pos: (..., 3) object position.
        goal_pos: (..., 3) goal position.
        scale: multiplier.

    Returns:
        (...,) tensor of target rewards (<= 0).
    """
    dist = torch.norm(object_pos - goal_pos, dim=-1)
    return -scale * dist


def r_success(
    object_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    tolerance: float = 0.075,
    bonus: float = 1.0,
) -> torch.Tensor:
    """Sparse success bonus.

    Success is defined as ``||g_t - (x_t)_{0:3}|| <= delta`` where ``delta`` is
    the curriculum tolerance.

    Args:
        object_pos: (..., 3) object position.
        goal_pos: (..., 3) goal position.
        tolerance: success tolerance ``delta`` (metres).
        bonus: value of the bonus.

    Returns:
        (...,) tensor of success bonuses (0 or ``bonus``).
    """
    dist = torch.norm(object_pos - goal_pos, dim=-1)
    return bonus * (dist <= tolerance).float()


def r_orientation(
    object_quat: torch.Tensor,
    goal_quat: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Dense orientation reward for in-hand reorientation tasks.

    Uses the quaternion geodesic distance::

        r = -scale * (1 - <q, g>^2)

    which is 0 when the orientations match and 1 when they are orthogonal.

    Args:
        object_quat: (..., 4) current object quaternion (w, x, y, z).
        goal_quat: (..., 4) goal quaternion (w, x, y, z).
        scale: multiplier.

    Returns:
        (...,) tensor of orientation rewards (<= 0).
    """
    q = torch.nn.functional.normalize(object_quat, dim=-1)
    g = torch.nn.functional.normalize(goal_quat, dim=-1)
    dot = torch.sum(q * g, dim=-1)
    return -scale * (1.0 - dot * dot)


# ---------------------------------------------------------------------------
# Composite rewards
# ---------------------------------------------------------------------------
def compute_allegro_kuka_reward(
    hand_pos: torch.Tensor,
    object_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    config: Optional[RewardConfig] = None,
    table_height: float = 0.0,
    tolerance: Optional[float] = None,
) -> torch.Tensor:
    """Composite AllegroKuka reward ``w1*r_reach + w2*r_lift + w3*r_target + w4*r_success``.

    Args:
        hand_pos: (..., 3) hand position.
        object_pos: (..., 3) object position.
        goal_pos: (..., 3) goal position.
        config: reward weights / scales.
        table_height: table surface height.
        tolerance: overrides ``config.success_tolerance`` (used by curriculum).

    Returns:
        (...,) tensor of total rewards.
    """
    cfg = config or RewardConfig()
    delta = cfg.success_tolerance if tolerance is None else tolerance

    reach = r_reach(hand_pos, object_pos, scale=cfg.reach_scale)
    lift = r_lift(
        object_pos,
        table_height=table_height,
        lift_height=cfg.lift_height,
        scale=cfg.lift_scale,
    )
    target = r_target(object_pos, goal_pos, scale=cfg.target_scale)
    success = r_success(object_pos, goal_pos, tolerance=delta, bonus=1.0)

    if cfg.clip_dense is not None:
        reach = torch.clamp(reach, min=-cfg.clip_dense)
        target = torch.clamp(target, min=-cfg.clip_dense)

    total = (
        cfg.w_reach * reach
        + cfg.w_lift * lift
        + cfg.w_target * target
        + cfg.w_success * success
    )
    return total


def compute_reorientation_reward(
    object_quat: torch.Tensor,
    goal_quat: torch.Tensor,
    config: Optional[RewardConfig] = None,
    success_tolerance: float = 0.1,
) -> torch.Tensor:
    """Composite reward for ShadowHand / AllegroHand reorientation.

    ``r = w_orientation * r_orientation + w_success * r_success`` where success
    is measured by the quaternion geodesic distance falling below
    ``success_tolerance``.

    Args:
        object_quat: (..., 4) current object quaternion.
        goal_quat: (..., 4) goal quaternion.
        config: reward weights / scales.
        success_tolerance: angular tolerance for the success bonus.

    Returns:
        (...,) tensor of total rewards.
    """
    cfg = config or RewardConfig()

    orient = r_orientation(object_quat, goal_quat, scale=cfg.orientation_scale)

    q = torch.nn.functional.normalize(object_quat, dim=-1)
    g = torch.nn.functional.normalize(goal_quat, dim=-1)
    dot = torch.sum(q * g, dim=-1)
    angle = 2.0 * torch.acos(torch.clamp(dot.abs(), max=1.0))
    success = (angle <= success_tolerance).float()

    return cfg.w_orientation * orient + cfg.w_success * success


def is_success(
    object_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    tolerance: float = 0.075,
) -> torch.Tensor:
    """Boolean success indicator ``||g_t - (x_t)_{0:3}|| <= delta``."""
    dist = torch.norm(object_pos - goal_pos, dim=-1)
    return dist <= tolerance

"""Reward terms and curricula of the AllegroKuka tasks (App. A)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def tolerance_curriculum_schedule(
    delta: float,
    mean_successes: float,
    success_threshold: float = 3.0,
    shrink: float = 0.9,
    min_delta: float = 0.01,
) -> float:
    """Success-tolerance curriculum of the Regrasping / Reorientation tasks.

    "This tolerance is decreased in a curriculum from 7.5 cm to 1 cm,
    decremented by 10% each time the average number of successes in an episode
    crosses 3." (App. A)
    """
    if mean_successes >= success_threshold:
        return max(min_delta, delta * shrink)
    return delta


@dataclass
class RewardScales:
    """Weights of the weighted combination described in App. A."""

    reach: float = 1.0
    lift: float = 5.0
    target: float = 2.0
    success: float = 10.0
    action_penalty: float = 0.0
    fall_penalty: float = 0.0


def reach_reward(
    hand_pos: torch.Tensor,
    object_pos: torch.Tensor,
    object_lifted: torch.Tensor,
    sigma: float = 0.2,
) -> torch.Tensor:
    """``r_reach``: exponential shaping term pulling the hand towards the object.

    The term is switched off once the object has been lifted so that the policy
    is free to use gravity / the table when manipulating the object.
    """
    dist = torch.norm(hand_pos - object_pos, dim=-1)
    return (1.0 - object_lifted) * torch.exp(-sigma * dist)


def lift_bonus(
    object_height: torch.Tensor,
    table_height: float,
    lift_threshold: float = 0.03,
    bonus: float = 1.0,
) -> torch.Tensor:
    """``r_lift``: bonus for getting the object off the table."""
    return bonus * (object_height > (table_height + lift_threshold)).float()


def target_reward(
    object_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    object_lifted: torch.Tensor,
    success_tolerance: float = 0.01,
    sigma: float = 10.0,
) -> torch.Tensor:
    """``r_target``: exponential reward for moving the lifted object to the goal."""
    dist = torch.norm(goal_pos - object_pos, dim=-1)
    return object_lifted * torch.exp(-sigma * dist) * (dist > success_tolerance).float()


def orientation_target_reward(
    object_quat: torch.Tensor,
    goal_quat: torch.Tensor,
    object_lifted: torch.Tensor,
    sigma: float = 5.0,
) -> torch.Tensor:
    """Reward term used by Reorientation (goal is a pose in R^7, not a point)."""
    dot = torch.abs((object_quat * goal_quat).sum(dim=-1)).clamp(max=1.0)
    angle_error = 2.0 * torch.acos(dot)
    return object_lifted * torch.exp(-sigma * angle_error)


def success_bonus(success_flags: torch.Tensor, bonus: float = 1.0) -> torch.Tensor:
    """``r_success``: bonus granted for each success (App. A)."""
    return bonus * success_flags.float()


def compute_allegro_kuka_reward(
    scales: RewardScales,
    hand_pos: torch.Tensor,
    object_pos: torch.Tensor,
    object_height: torch.Tensor,
    table_height: float,
    goal_pos: torch.Tensor,
    object_lifted: torch.Tensor,
    success_flags: torch.Tensor,
    actions: torch.Tensor,
    goal_quat: torch.Tensor = None,
    object_quat: torch.Tensor = None,
) -> torch.Tensor:
    reward = scales.reach * reach_reward(hand_pos, object_pos, object_lifted)
    reward = reward + scales.lift * lift_bonus(object_height, table_height)
    if goal_quat is not None and object_quat is not None:
        reward = reward + scales.target * orientation_target_reward(
            object_quat, goal_quat, object_lifted
        )
    reward = reward + scales.target * target_reward(object_pos, goal_pos, object_lifted)
    reward = reward + scales.success * success_bonus(success_flags)
    if scales.action_penalty:
        reward = reward - scales.action_penalty * (actions ** 2).sum(dim=-1)
    return reward

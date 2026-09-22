"""Training loop for RoboticSequence (SAC + knowledge retention)."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from ..retention.episodic_memory import ProtectedReplayBuffer
from .config import MetaworldConfig
from .robotic_sequence import RoboticSequence
from .sac import ReplayBuffer, SAC, Transition


def pretrain_policy(
    config: MetaworldConfig,
    stages: List[str],
    num_steps: int = 300_000,
    seed: int = 0,
    success_threshold: float = 1.0,
) -> SAC:
    """Train ``pi_*`` with SAC on ``stages`` until convergence.

    Appendix B.3: the pre-trained model is trained with SAC on the last two
    stages (``peg-unplug-side`` and ``push-wall``) until convergence, i.e. a
    100% success rate.
    """

    sub_config = MetaworldConfig(**{**asdict(config), "stages": list(stages)})
    sub_config.retention.method = "none"
    env = RoboticSequence(sub_config, seed=seed, stages=list(stages))
    agent = SAC(sub_config, env.observation_dim, env.num_actions, len(stages), device=config.device)
    replay = ReplayBuffer(config.replay_buffer_size, env.observation_dim, env.num_actions)

    obs, info = env.reset(seed=seed)
    stage = 0
    for step in range(num_steps):
        action = agent.select_action(obs, stage)
        result = env.step(action)
        replay.add(
            Transition(
                obs=obs,
                action=action,
                reward=result.reward,
                next_obs=result.observation,
                done=float(result.terminated or result.truncated),
                stage=stage,
                next_stage=result.stage,
            )
        )
        obs, stage = result.observation, result.stage
        if result.terminated or result.truncated:
            obs, info = env.reset(seed=seed + step)
            stage = 0
        if replay.size >= config.batch_size:
            agent.update(replay)
    env.close()
    return agent


def collect_bc_dataset(
    teacher: SAC,
    config: MetaworldConfig,
    stages: List[str],
    num_samples: int = 10_000,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Sample ``num_samples`` states from the pre-trained task (BC buffer).

    The teacher is executed in the pre-training stages and the visited states
    (together with the stage ID) form the behavioral-cloning dataset.  The paper
    keeps 10 000 samples, i.e. 10% of the 100 000-transition replay buffer.
    """

    env = RoboticSequence(config, seed=seed, stages=list(stages))
    obs_list, stage_list = [], []
    obs, _ = env.reset(seed=seed)
    stage = 0
    while len(obs_list) < num_samples:
        action = teacher.select_action(obs, stage)
        result = env.step(action)
        obs_list.append(obs)
        stage_list.append(stage)
        obs, stage = result.observation, result.stage
        if result.terminated or result.truncated:
            obs, _ = env.reset(seed=seed + len(obs_list))
            stage = 0
    env.close()
    return {
        "obs": np.asarray(obs_list, dtype=np.float32),
        "stages": np.asarray(stage_list, dtype=np.int64),
    }


def collect_episodic_memory(
    teacher: SAC,
    config: MetaworldConfig,
    stages: List[str],
    num_samples: int = 10_000,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Collect state-action-reward tuples from the pre-trained stages for EM."""

    env = RoboticSequence(config, seed=seed, stages=list(stages))
    obs_list, next_list, act_list = [], [], []
    rew_list, done_list, stage_list, next_stage_list = [], [], [], []
    obs, _ = env.reset(seed=seed)
    stage = 0
    while len(obs_list) < num_samples:
        action = teacher.select_action(obs, stage)
        result = env.step(action)
        obs_list.append(obs)
        next_list.append(result.observation)
        act_list.append(action)
        rew_list.append(result.reward)
        done_list.append(float(result.terminated or result.truncated))
        stage_list.append(stage)
        next_stage_list.append(result.stage)
        obs, stage = result.observation, result.stage
        if result.terminated or result.truncated:
            obs, _ = env.reset(seed=seed + len(obs_list))
            stage = 0
    env.close()
    return {
        "obs": np.asarray(obs_list, dtype=np.float32),
        "next_obs": np.asarray(next_list, dtype=np.float32),
        "actions": np.asarray(act_list, dtype=np.float32),
        "rewards": np.asarray(rew_list, dtype=np.float32),
        "dones": np.asarray(done_list, dtype=np.float32),
        "stages": np.asarray(stage_list, dtype=np.int64),
        "next_stages": np.asarray(next_stage_list, dtype=np.int64),
    }


def fine_tune(
    config: MetaworldConfig,
    teacher: Optional[SAC] = None,
    seed: int = 0,
    bc_dataset: Optional[Dict[str, np.ndarray]] = None,
    em_dataset: Optional[Dict[str, np.ndarray]] = None,
    callback: Optional[Callable] = None,
) -> Dict[str, object]:
    """Fine-tune on the full RoboticSequence with the configured retention method.

    Returns a dictionary containing the per-stage success rates over training
    (Figure 7), the return curve (Figure 3c) and the final agent.
    """

    env = RoboticSequence(config, seed=seed, stages=list(config.stages))
    obs_dim, action_dim = env.observation_dim, env.num_actions
    num_stages = len(config.stages)

    agent = SAC(
        config,
        obs_dim,
        action_dim,
        num_stages,
        device=config.device,
        teacher=teacher if teacher is not None else None,
        bc_dataset=bc_dataset,
    )
    if teacher is not None and config.retention.method != "none":
        agent.actor.load_state_dict(teacher.actor.state_dict())
        if config.reset_last_layer:
            for head in list(agent.actor.mu_heads) + list(agent.actor.log_std_heads):
                head.reset_parameters()

    if config.retention.method == "em":
        protected = ProtectedReplayBuffer(config.replay_buffer_size, protected_size=config.memory_size)
        replay = ReplayBuffer(config.replay_buffer_size, obs_dim, action_dim, protected=protected)
        if em_dataset is not None:
            replay.load_protected(em_dataset)
    else:
        replay = ReplayBuffer(config.replay_buffer_size, obs_dim, action_dim)

    history: Dict[str, List] = {
        "steps": [],
        "return": [],
        "stage_success": {stage: [] for stage in config.stages},
        "eval_steps": [],
    }

    obs, _ = env.reset(seed=seed)
    stage = 0
    episode_return = 0.0
    for step in range(config.num_train_steps):
        action = agent.select_action(obs, stage)
        result = env.step(action)
        replay.add(
            Transition(
                obs=obs,
                action=action,
                reward=result.reward,
                next_obs=result.observation,
                done=float(result.terminated or result.truncated),
                stage=stage,
                next_stage=result.stage,
            )
        )
        episode_return += result.reward
        obs, stage = result.observation, result.stage

        if result.terminated or result.truncated:
            history["steps"].append(step)
            history["return"].append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset(seed=seed + step)
            stage = 0

        if replay.size >= config.batch_size:
            agent.update(replay)

        if callback is not None and step % config.eval_every_steps == 0:
            callback(step, agent, env, history)

    history["agent"] = agent
    return history

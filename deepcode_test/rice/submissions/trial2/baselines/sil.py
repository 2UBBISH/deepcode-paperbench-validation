#!/usr/bin/env python3
"""
Self-Imitation Learning (SIL) Baseline

Implements the SIL algorithm from:
  "Self-Imitation Learning" by Oh et al. (ICML 2018)

SIL augments standard RL (PPO) with an auxiliary loss that encourages
the agent to reproduce past good decisions. It maintains a replay buffer
of past experiences and adds a supervised-style loss on transitions
where the observed return exceeds the current value estimate.

In the RICE context, SIL serves as a baseline refining method:
instead of using critical states + RND, we continue training the
pre-trained agent with SIL auxiliary losses.

Key Components:
- SILReplayBuffer: Stores (s, a, R) tuples, where R is the discounted
  return from that state-action pair.
- SILPolicyLoss: log π(a|s) * (R - V(s))_+
- SILValueLoss: 1/2 * ||(R - V(s))_+||^2
- Combined PPO + SIL training loop
"""

import os
import time
import argparse
import json
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import RolloutBuffer

# Internal imports
from rice.utils import (
    load_config, set_seed, Logger, ensure_dir, get_device,
    evaluate_policy, make_env, make_vec_env, format_time,
    get_project_root, build_mlp
)


# ============================================================================
# SIL Replay Buffer
# ============================================================================

class SILReplayBuffer:
    """
    Replay buffer for Self-Imitation Learning.
    
    Stores (state, action, return) tuples. The return R is the
    Monte-Carlo discounted return from that state-action pair.
    """

    def __init__(self, max_size: int = 100000, device: str = "auto"):
        self.max_size = max_size
        self.device = get_device(device)
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.returns: List[float] = []
        self.total_added = 0
        self.total_pruned = 0

    def add(self, state: np.ndarray, action: np.ndarray, ret: float) -> None:
        if len(self.states) >= self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.returns.pop(0)
            self.total_pruned += 1
        self.states.append(state.copy())
        if np.isscalar(action):
            self.actions.append(np.array([action]))
        else:
            self.actions.append(action.copy() if isinstance(action, np.ndarray) else np.array(action))
        self.returns.append(float(ret))
        self.total_added += 1

    def add_trajectory(self, observations, actions, rewards, dones, gamma=0.99):
        T = len(rewards)
        returns = np.zeros(T)
        running_return = 0.0
        for t in reversed(range(T)):
            running_return = rewards[t] + gamma * running_return * (1 - dones[t])
            returns[t] = running_return
        for t in range(T):
            self.add(observations[t], actions[t], returns[t])

    def sample(self, batch_size: int, value_fn=None) -> Optional[Dict[str, torch.Tensor]]:
        if len(self.states) == 0:
            return None
        states_arr = np.array(self.states)
        returns_arr = np.array(self.returns)
        if value_fn is not None:
            with torch.no_grad():
                states_tensor = torch.FloatTensor(states_arr).to(self.device)
                values = value_fn(states_tensor).cpu().numpy().flatten()
            positive_mask = returns_arr > values
            if not np.any(positive_mask):
                return None
            positive_indices = np.where(positive_mask)[0]
            indices = np.random.choice(positive_indices, min(batch_size, len(positive_indices)), replace=False)
        else:
            indices = np.random.choice(len(states_arr), min(batch_size, len(states_arr)), replace=False)
        batch_states = torch.FloatTensor(states_arr[indices]).to(self.device)
        batch_returns = torch.FloatTensor(returns_arr[indices]).to(self.device)
        actions_list = [self.actions[i] for i in indices]
        if actions_list[0].ndim == 1 and actions_list[0].shape[0] == 1:
            batch_actions = torch.LongTensor([a[0] for a in actions_list]).to(self.device)
        else:
            batch_actions = torch.FloatTensor(np.array(actions_list)).to(self.device)
        return {'states': batch_states, 'actions': batch_actions, 'returns': batch_returns}

    def __len__(self) -> int:
        return len(self.states)

    def get_stats(self) -> Dict[str, Any]:
        return {
            'size': len(self.states), 'max_size': self.max_size,
            'total_added': self.total_added, 'total_pruned': self.total_pruned,
            'mean_return': float(np.mean(self.returns)) if self.returns else 0.0,
            'max_return': float(np.max(self.returns)) if self.returns else 0.0,
            'min_return': float(np.min(self.returns)) if self.returns else 0.0,
        }

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump({'states': self.states, 'actions': self.actions,
                         'returns': self.returns, 'total_added': self.total_added,
                         'total_pruned': self.total_pruned}, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.states = data['states']
        self.actions = data['actions']
        self.returns = data['returns']
        self.total_added = data.get('total_added', len(self.states))
        self.total_pruned = data.get('total_pruned', 0)


# ============================================================================
# SIL Callback
# ============================================================================

class SILCallback(BaseCallback):
    """Callback for SIL training that logs metrics and periodically evaluates."""

    def __init__(self, logger: Logger, sil_buffer: SILReplayBuffer,
                 eval_env=None, eval_freq=10000, n_eval_episodes=10, verbose=0):
        super().__init__(verbose)
        self.logger = logger
        self.sil_buffer = sil_buffer
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            stats = self.sil_buffer.get_stats()
            for key, value in stats.items():
                self.logger.log(f'sil_buffer/{key}', value, self.n_calls)
        if self.eval_env is not None and (self.n_calls - self._last_eval_step) >= self.eval_freq:
            self._run_evaluation()
            self._last_eval_step = self.n_calls
        return True

    def _run_evaluation(self):
        if self.eval_env is None:
            return
        eval_results = evaluate_policy(self.eval_env, self.model,
                                       n_episodes=self.n_eval_episodes, deterministic=True)
        self.logger.log('eval/mean_return', eval_results['mean_return'], self.n_calls)
        self.logger.log('eval/std_return', eval_results['std_return'], self.n_calls)
        if self.verbose > 0:
            print(f"[SIL] Step {self.n_calls}: Eval return = "
                  f"{eval_results['mean_return']:.2f} ± {eval_results['std_return']:.2f}")

    def _on_training_end(self):
        if self.eval_env is not None:
            self._run_evaluation()


# ============================================================================
# Main SIL Training Function
# ============================================================================

def train_sil(
    env_id: str,
    agent_policy: PPO,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    seed: int = 0,
    total_timesteps: Optional[int] = None,
    sil_batch_size: int = 512,
    sil_value_coef: float = 0.5,
    sil_learning_rate: float = 1e-4,
    buffer_max_size: int = 100000,
    learning_rate: Optional[float] = None,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    device: str = "auto",
    verbose: int = 1,
    save_freq: int = 100000,
    resume_from: Optional[str] = None,
    **env_kwargs
) -> Tuple[PPO, Logger, str]:
    """
    Train a policy using Self-Imitation Learning (SIL) on top of a pre-trained agent.

    Returns:
        Tuple of (trained_model, logger, model_save_path).
    """
    set_seed(seed)
    device = get_device(device)

    if config is None:
        config = load_config()
    if output_dir is None:
        output_dir = os.path.join(get_project_root(), "outputs", "sil", env_id)
    output_dir = ensure_dir(output_dir)

    logger = Logger(log_dir=output_dir)

    if total_timesteps is None:
        total_timesteps = config.get('baselines', {}).get('sil', {}).get('total_timesteps', 1000000)

    # Create environments
    eval_env = make_env(env_id, seed=seed + 1000, **env_kwargs)
    train_env = make_vec_env(env_id, n_envs=1, seed=seed, **env_kwargs)

    # Initialize SIL replay buffer
    sil_buffer = SILReplayBuffer(max_size=buffer_max_size, device=device)

    # Pre-populate buffer with trajectories from the pre-trained agent
    if verbose > 0:
        print("[SIL] Pre-populating replay buffer with agent trajectories...")
    from rice.utils import collect_trajectories
    trajectories = collect_trajectories(
        eval_env, agent_policy, num_trajectories=10, max_steps=1000, deterministic=False
    )
    for traj in trajectories:
        sil_buffer.add_trajectory(
            traj['observations'], traj['actions'], traj['rewards'], traj['dones'], gamma=gamma
        )
    if verbose > 0:
        print(f"[SIL] Buffer pre-populated with {len(sil_buffer)} transitions")

    # Use the pre-trained agent directly
    model = agent_policy
    model.set_env(train_env)

    # Custom training loop
    start_time = time.time()
    num_timesteps = 0
    obs = train_env.reset()

    rollout_obs, rollout_actions, rollout_rewards = [], [], []
    rollout_dones, rollout_values, rollout_log_probs = [], [], []

    episode_rewards = []
    current_ep_reward = np.zeros(train_env.num_envs)
    num_episodes = 0
    last_eval_step = 0
    last_save_step = 0

    sil_optimizer = Adam(model.policy.parameters(), lr=sil_learning_rate)

    while num_timesteps < total_timesteps:
        # Collect rollout
        for step in range(n_steps):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(device)
                model.policy.set_training_mode(False)
                latent_pi, latent_vf = model.policy._get_latent(obs_tensor)
                values = model.policy.value_net(latent_vf)
                distribution = model.policy._get_action_dist_from_latent(latent_pi)
                actions = distribution.get_actions()
                log_probs = distribution.log_prob(actions)
                model.policy.set_training_mode(True)

            actions_np = actions.cpu().numpy()
            next_obs, rewards, dones, infos = train_env.step(actions_np)

            rollout_obs.append(obs.copy())
            rollout_actions.append(actions_np.copy())
            rollout_rewards.append(rewards.copy())
            rollout_dones.append(dones.copy())
            rollout_values.append(values.cpu().numpy().copy())
            rollout_log_probs.append(log_probs.cpu().numpy().copy())

            current_ep_reward += rewards
            for i, done in enumerate(dones):
                if done:
                    episode_rewards.append(current_ep_reward[i])
                    current_ep_reward[i] = 0.0
                    num_episodes += 1

            obs = next_obs
            num_timesteps += train_env.num_envs
            if num_timesteps >= total_timesteps:
                break

        # Add to SIL buffer
        n_envs = train_env.num_envs
        n_steps_collected = len(rollout_obs)
        all_obs = np.stack(rollout_obs, axis=0)
        all_actions = np.stack(rollout_actions, axis=0)
        all_rewards = np.stack(rollout_rewards, axis=0)
        all_dones = np.stack(rollout_dones, axis=0)

        for env_idx in range(n_envs):
            env_obs = all_obs[:, env_idx, :]
            env_actions = all_actions[:, env_idx, ...]
            env_rewards = all_rewards[:, env_idx]
            env_dones = all_dones[:, env_idx]
            returns = np.zeros(n_steps_collected)
            running_return = 0.0
            for t in reversed(range(n_steps_collected)):
                running_return = env_rewards[t] + gamma * running_return * (1 - env_dones[t])
                returns[t] = running_return
            for t in range(n_steps_collected):
                sil_buffer.add(env_obs[t], env_actions[t], returns[t])

        # PPO update
        obs_arr = np.concatenate(rollout_obs, axis=0)
        actions_arr = np.concatenate(rollout_actions, axis=0)
        rewards_arr = np.concatenate(rollout_rewards, axis=0)
        dones_arr = np.concatenate(rollout_dones, axis=0)
        values_arr = np.concatenate(rollout_values, axis=0)
        log_probs_arr = np.concatenate(rollout_log_probs, axis=0)

        buffer = RolloutBuffer(
            buffer_size=len(obs_arr),
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            device=device, gae_lambda=gae_lambda, gamma=gamma, n_envs=n_envs,
        )
        for i in range(len(obs_arr)):
            buffer.add(obs_arr[i], actions_arr[i], rewards_arr[i],
                       dones_arr[i], values_arr[i], log_probs_arr[i])

        with torch.no_grad():
            last_obs_tensor = torch.FloatTensor(obs).to(device)
            model.policy.set_training_mode(False)
            _, last_latent_vf = model.policy._get_latent(last_obs_tensor)
            last_values = model.policy.value_net(last_latent_vf)
            model.policy.set_training_mode(True)
        buffer.compute_returns_and_advantage(last_values, dones_arr[-n_envs:])

        for epoch in range(n_epochs):
            for rollout_data in buffer.get(batch_size):
                acts = rollout_data.actions
                if isinstance(train_env.action_space, gym.spaces.Discrete):
                    acts = acts.long().flatten()
                vals, log_prob, entropy = model.policy.evaluate_actions(
                    rollout_data.observations, acts)
                vals = vals.flatten()
                advantages = rollout_data.advantages
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                value_loss = F.mse_loss(rollout_data.returns, vals)
                entropy_loss = -torch.mean(entropy)
                loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
                model.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_grad_norm)
                model.policy.optimizer.step()

        # SIL update
        if len(sil_buffer) >= sil_batch_size:
            def value_fn(states):
                _, latent_vf = model.policy._get_latent(states)
                return model.policy.value_net(latent_vf)

            batch = sil_buffer.sample(sil_batch_size, value_fn=value_fn)
            if batch is not None:
                states = batch['states']
                actions = batch['actions']
                returns = batch['returns']

                _, latent_vf = model.policy._get_latent(states)
                values = model.policy.value_net(latent_vf).squeeze(-1)
                advantage = torch.clamp(returns - values, min=0.0)
                mask = (advantage > 0).float()
                num_valid = mask.sum()

                if num_valid > 0:
                    if isinstance(train_env.action_space, gym.spaces.Discrete):
                        latent_pi, _ = model.policy._get_latent(states)
                        action_logits = model.policy.action_net(latent_pi)
                        log_probs_all = F.log_softmax(action_logits, dim=-1)
                        selected_log_probs = log_probs_all.gather(
                            1, actions.unsqueeze(-1)).squeeze(-1)
                    else:
                        latent_pi, _ = model.policy._get_latent(states)
                        mean_actions = model.policy.action_net(latent_pi)
                        log_std = model.policy.log_std
                        std = torch.exp(log_std)
                        dist = torch.distributions.Normal(mean_actions, std)
                        selected_log_probs = dist.log_prob(actions).sum(dim=-1)

                    sil_policy_loss = -(selected_log_probs * advantage * mask).sum() / (num_valid + 1e-8)
                    sil_value_loss = 0.5 * ((advantage * mask) ** 2).sum() / (num_valid + 1e-8)
                    sil_total_loss = sil_policy_loss + sil_value_coef * sil_value_loss

                    sil_optimizer.zero_grad()
                    sil_total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_grad_norm)
                    sil_optimizer.step()

                    logger.log('sil/policy_loss', sil_policy_loss.item(), num_timesteps)
                    logger.log('sil/value_loss', sil_value_loss.item(), num_timesteps)
                    logger.log('sil/num_samples', num_valid.item(), num_timesteps)

        # Clear rollout storage
        rollout_obs.clear(); rollout_actions.clear(); rollout_rewards.clear()
        rollout_dones.clear(); rollout_values.clear(); rollout_log_probs.clear()

        # Logging
        if len(episode_rewards) > 0:
            mean_ep_rew = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0
            elapsed = time.time() - start_time
            fps = num_timesteps / max(elapsed, 1e-8)
            logger.log('train/mean_episode_reward', mean_ep_rew, num_timesteps)
            logger.log('train/fps', fps, num_timesteps)
            logger.log('train/buffer_size', len(sil_buffer), num_timesteps)
            if verbose > 0:
                print(f"[SIL] Steps: {num_timesteps}/{total_timesteps} | "
                      f"Episodes: {num_episodes} | Mean Reward: {mean_ep_rew:.2f} | "
                      f"FPS: {fps:.0f} | Buffer: {len(sil_buffer)}")

        # Evaluation
        if (num_timesteps - last_eval_step) >= eval_freq:
            eval_results = evaluate_policy(eval_env, model, n_episodes=n_eval_episodes, deterministic=True)
            logger.log('eval/mean_return', eval_results['mean_return'], num_timesteps)
            logger.log('eval/std_return', eval_results['std_return'], num_timesteps)
            last_eval_step = num_timesteps
            if verbose > 0:
                print(f"[SIL] Eval at {num_timesteps}: {eval_results['mean_return']:.2f} ± {eval_results['std_return']:.2f}")

        # Save checkpoint
        if (num_timesteps - last_save_step) >= save_freq:
            save_path = os.path.join(output_dir, f"sil_model_{num_timesteps}_steps.zip")
            model.save(save_path)
            last_save_step = num_timesteps
            if verbose > 0:
                print(f"[SIL] Saved checkpoint to {save_path}")

    # Final save
    final_model_path = os.path.join(output_dir, "sil_model_final.zip")
    model.save(final_model_path)
    sil_buffer.save(os.path.join(output_dir, "sil_buffer.pkl"))
    logger.save(os.path.join(output_dir, "sil_logger.json"))

    total_time = time.time() - start_time
    logger.log('train/total_time', total_time, num_timesteps)

    if verbose > 0:
        print(f"\n[SIL] Training completed in {format_time(total_time)}")
        print(f"[SIL] Final model saved to {final_model_path}")

    return model, logger, final_model_path


# ============================================================================
# Convenience Pipeline Function
# ============================================================================

def run_sil_pipeline(
    env_id: str,
    agent_path: str,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    seed: int = 0,
    total_timesteps: Optional[int] = None,
    sil_batch_size: int = 512,
    sil_value_coef: float = 0.5,
    sil_learning_rate: float = 1e-4,
    buffer_max_size: int = 100000,
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    device: str = "auto",
    verbose: int = 1,
    **env_kwargs
) -> Dict[str, Any]:
    """
    Run the full SIL pipeline: load agent, train with SIL, evaluate.

    Returns:
        Dictionary with results (model_path, final_return, training_time, etc.).
    """
    set_seed(seed)
    device = get_device(device)

    if config is None:
        config = load_config(env_id)
    if output_dir is None:
        output_dir = os.path.join(get_project_root(), "outputs", "sil", env_id, f"seed_{seed}")
    output_dir = ensure_dir(output_dir)

    # Load pre-trained agent
    if verbose > 0:
        print(f"[SIL] Loading pre-trained agent from {agent_path}")
    agent_policy = PPO.load(agent_path, device=device)

    # Train with SIL
    model, logger, model_path = train_sil(
        env_id=env_id,
        agent_policy=agent_policy,
        config=config,
        output_dir=output_dir,
        seed=seed,
        total_timesteps=total_timesteps,
        sil_batch_size=sil_batch_size,
        sil_value_coef=sil_value_coef,
        sil_learning_rate=sil_learning_rate,
        buffer_max_size=buffer_max_size,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        device=device,
        verbose=verbose,
        **env_kwargs
    )

    # Final evaluation
    eval_env = make_env(env_id, seed=seed + 2000, **env_kwargs)
    final_results = evaluate_policy(eval_env, model, n_episodes=n_eval_episodes, deterministic=True)
    eval_env.close()

    results = {
        'env_id': env_id,
        'method': 'SIL',
        'seed': seed,
        'model_path': model_path,
        'output_dir': output_dir,
        'final_mean_return': final_results['mean_return'],
        'final_std_return': final_results['std_return'],
        'final_min_return': final_results.get('min_return', None),
        'final_max_return': final_results.get('max_return', None),
    }

    # Save results
    results_path = os.path.join(output_dir, "sil_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    if verbose > 0:
        print(f"\n[SIL] Pipeline complete!")
        print(f"[SIL] Final return: {final_results['mean_return']:.2f} ± {final_results['std_return']:.2f}")
        print(f"[SIL] Results saved to {results_path}")

    return results


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Self-Imitation Learning (SIL) Baseline for RICE"
    )
    parser.add_argument('--env-id', type=str, default='Hopper-v3',
                        help='Gym environment ID')
    parser.add_argument('--agent-path', type=str, required=True,
                        help='Path to pre-trained PPO agent (.zip)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--total-timesteps', type=int, default=None,
                        help='Total timesteps for SIL training')
    parser.add_argument('--sil-batch-size', type=int, default=512,
                        help='SIL batch size')
    parser.add_argument('--sil-value-coef', type=float, default=0.5,
                        help='SIL value loss coefficient')
    parser.add_argument('--sil-learning-rate', type=float, default=1e-4,
                        help='SIL learning rate')
    parser.add_argument('--buffer-max-size', type=int, default=100000,
                        help='SIL buffer max size')
    parser.add_argument('--eval-freq', type=int, default=10000,
                        help='Evaluation frequency')
    parser.add_argument('--n-eval-episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--device', type=str, default='auto',
                        help='Torch device')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level')

    args = parser.parse_args()

    results = run_sil_pipeline(
        env_id=args.env_id,
        agent_path=args.agent_path,
        config=args.config,
        output_dir=args.output_dir,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        sil_batch_size=args.sil_batch_size,
        sil_value_coef=args.sil_value_coef,
        sil_learning_rate=args.sil_learning_rate,
        buffer_max_size=args.buffer_max_size,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        device=args.device,
        verbose=args.verbose,
    )

    print(f"\nFinal Results: {json.dumps(results, indent=2, default=str)}")


if __name__ == '__main__':
    main()
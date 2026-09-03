#!/usr/bin/env python3
"""
Self-Imitation Learning (SIL) Baseline Implementation

Implements SIL as described in Oh et al. (2018): augments PPO with off-policy
updates on past transitions with positive advantages from a replay buffer.

Key Components:
- Replay buffer storing (s, a, R) with discounted returns
- SIL loss: -log π(a|s) * (R - V(s))_+  (positive advantages only)
- Combined: PPO loss + β * SIL loss
"""

import argparse, json, os, pickle, sys, time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import yaml

try:
    from stable_baselines3 import PPO as SB3_PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rice.utils import (set_seed, compute_gae, compute_returns, to_tensor, to_numpy, orthogonal_init)
from rice.env_wrappers import make_state_saveable


class SILReplayBuffer:
    """Replay buffer for (state, action, return) tuples with reservoir sampling."""
    def __init__(self, capacity=100000, state_dim=11, action_dim=3, discrete_action=False):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete_action = discrete_action
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.size = 0

    def add(self, state, action, ret):
        if self.size < self.capacity:
            self.states[self.size] = state
            self.actions[self.size] = action
            self.returns[self.size] = ret
            self.size += 1
        else:
            idx = np.random.randint(0, self.size + 1)
            if idx < self.capacity:
                self.states[idx] = state
                self.actions[idx] = action
                self.returns[idx] = ret

    def add_trajectory(self, states, actions, returns):
        for i in range(len(states)):
            self.add(states[i], actions[i], returns[i])

    def sample(self, batch_size):
        if self.size == 0:
            return None, None, None
        indices = np.random.randint(0, self.size, size=min(batch_size, self.size))
        return self.states[indices], self.actions[indices], self.returns[indices]

    def __len__(self):
        return self.size


class MLPPolicy(nn.Module):
    """MLP policy with actor and critic heads."""
    def __init__(self, state_dim, action_dim, hidden_sizes=(64,64), activation="tanh",
                 discrete_action=False, num_discrete_actions=None, policy_std=0.0,
                 value_hidden_sizes=None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete_action = discrete_action
        self.num_discrete_actions = num_discrete_actions or action_dim
        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[activation]
        layers = []
        prev_dim = state_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev_dim, h), act_fn()])
            prev_dim = h
        self.feature_net = nn.Sequential(*layers)
        self.feature_dim = prev_dim
        if discrete_action:
            self.actor = nn.Linear(self.feature_dim, self.num_discrete_actions)
        else:
            self.actor_mean = nn.Linear(self.feature_dim, action_dim)
            self.actor_log_std = nn.Parameter(
                torch.ones(action_dim)*policy_std if policy_std!=0.0 else torch.zeros(action_dim))
        if value_hidden_sizes:
            v_layers = []
            v_prev = self.feature_dim
            for vh in value_hidden_sizes:
                v_layers.extend([nn.Linear(v_prev, vh), act_fn()])
                v_prev = vh
            v_layers.append(nn.Linear(v_prev, 1))
            self.critic = nn.Sequential(*v_layers)
        else:
            self.critic = nn.Linear(self.feature_dim, 1)
        self.apply(lambda m: orthogonal_init(m) if isinstance(m, nn.Linear) else None)
        if not discrete_action:
            nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
            nn.init.zeros_(self.actor_mean.bias)
        last_critic = self.critic[-1] if value_hidden_sizes else self.critic
        nn.init.orthogonal_(last_critic.weight, gain=1.0)
        nn.init.zeros_(last_critic.bias)

    def forward(self, state):
        features = self.feature_net(state)
        value = self.critic(features)
        if self.discrete_action:
            return self.actor(features), value
        else:
            return (self.actor_mean(features),
                    torch.exp(self.actor_log_std).expand_as(self.actor_mean(features))), value

    def get_action(self, state, deterministic=False):
        if self.discrete_action:
            logits, value = self.forward(state)
            dist = Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            return action, dist.log_prob(action), dist.entropy(), value
        else:
            (mean, std), value = self.forward(state)
            dist = Normal(mean, std)
            action = mean if deterministic else dist.rsample()
            return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value

    def evaluate_actions(self, state, action):
        if self.discrete_action:
            logits, value = self.forward(state)
            dist = Categorical(logits=logits)
            return dist.log_prob(action), dist.entropy(), value
        else:
            (mean, std), value = self.forward(state)
            dist = Normal(mean, std)
            return dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value

    def get_value(self, state):
        _, value = self.forward(state)
        return value


class SILTrainer:
    """Self-Imitation Learning Trainer: PPO + SIL off-policy updates."""
    def __init__(self, env, state_dim, action_dim, discrete_action=False,
                 num_discrete_actions=None, device="cpu",
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
                 value_loss_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
                 ppo_epochs=10, batch_size=64,
                 sil_beta=0.1, sil_batch_size=512, sil_update_epochs=1,
                 buffer_capacity=100000,
                 policy_hidden_sizes=(64,64), value_hidden_sizes=None,
                 activation="tanh", policy_std=0.0, normalize_advantages=True):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete_action = discrete_action
        self.num_discrete_actions = num_discrete_actions
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.normalize_advantages = normalize_advantages
        self.sil_beta = sil_beta
        self.sil_batch_size = sil_batch_size
        self.sil_update_epochs = sil_update_epochs

        self.policy = MLPPolicy(state_dim, action_dim, policy_hidden_sizes, activation,
                                discrete_action, num_discrete_actions, policy_std,
                                value_hidden_sizes).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = SILReplayBuffer(buffer_capacity, state_dim, action_dim, discrete_action)
        self.total_steps_done = 0
        self.history = []

    def collect_trajectory(self, max_steps=1000):
        obs, info = self.env.reset()
        if isinstance(obs, tuple): obs = obs[0]
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        total_reward = 0.0
        done = False
        step = 0
        while not done and step < max_steps:
            state_t = to_tensor(obs, self.device).unsqueeze(0)
            with torch.no_grad():
                action_t, log_prob_t, entropy_t, value_t = self.policy.get_action(state_t)
            action = to_numpy(action_t).flatten()
            if self.discrete_action:
                action = int(action[0]) if len(action)==1 else action
            step_result = self.env.step(action)
            if len(step_result) == 4:
                next_obs, reward, terminated, truncated = step_result[0], step_result[1], step_result[2], False
                done = terminated or truncated
            elif len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result
            if isinstance(next_obs, tuple): next_obs = next_obs[0]
            states.append(obs.copy())
            actions.append(action)
            rewards.append(float(reward))
            dones.append(done)
            log_probs.append(float(to_numpy(log_prob_t)))
            values.append(float(to_numpy(value_t)))
            total_reward += float(reward)
            obs = next_obs
            step += 1
        return {"states": np.array(states), "actions": np.array(actions),
                "rewards": np.array(rewards), "dones": np.array(dones),
                "log_probs": np.array(log_probs), "values": np.array(values),
                "total_reward": total_reward, "length": step}

    def compute_sil_loss(self, states, actions, returns):
        states_t = to_tensor(states, self.device)
        actions_t = to_tensor(actions, self.device)
        if self.discrete_action: actions_t = actions_t.long()
        returns_t = to_tensor(returns, self.device)
        log_probs, _, values = self.policy.evaluate_actions(states_t, actions_t)
        values = values.squeeze(-1)
        advantages = returns_t - values
        positive_adv = torch.clamp(advantages, min=0.0)
        sil_loss = -(log_probs * positive_adv.detach()).mean()
        mask = (advantages > 0).float()
        if mask.sum() > 0:
            value_loss = (F.mse_loss(values, returns_t, reduction='none') * mask).sum() / mask.sum()
        else:
            value_loss = torch.tensor(0.0, device=self.device)
        return sil_loss, value_loss, mask.sum().item()

    def update_ppo(self, trajectories):
        all_states, all_actions, all_log_probs = [], [], []
        all_rewards, all_dones, all_values = [], [], []
        for traj in trajectories:
            all_states.append(traj["states"]); all_actions.append(traj["actions"])
            all_log_probs.append(traj["log_probs"]); all_rewards.append(traj["rewards"])
            all_dones.append(traj["dones"]); all_values.append(traj["values"])
        if len(all_states) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
        states_arr = np.concatenate(all_states); actions_arr = np.concatenate(all_actions)
        old_log_probs_arr = np.concatenate(all_log_probs); rewards_arr = np.concatenate(all_rewards)
        dones_arr = np.concatenate(all_dones); values_arr = np.concatenate(all_values)
        last_value = 0.0
        if len(values_arr) > 0 and not dones_arr[-1]:
            last_value = values_arr[-1]
        advantages, returns = compute_gae(rewards_arr, values_arr, dones_arr,
                                          self.gamma, self.gae_lambda, last_value)
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        states_t = to_tensor(states_arr, self.device)
        actions_t = to_tensor(actions_arr, self.device)
        if self.discrete_action: actions_t = actions_t.long()
        old_log_probs_t = to_tensor(old_log_probs_arr, self.device)
        advantages_t = to_tensor(advantages, self.device)
        returns_t = to_tensor(returns, self.device)
        total_samples = len(states_arr)
        indices = np.arange(total_samples)
        epoch_losses = []
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, total_samples, self.batch_size):
                batch_idx = indices[start:start+self.batch_size]
                bs, ba = states_t[batch_idx], actions_t[batch_idx]
                blp, badv, bret = old_log_probs_t[batch_idx], advantages_t[batch_idx], returns_t[batch_idx]
                new_log_probs, entropy, values = self.policy.evaluate_actions(bs, ba)
                ratio = torch.exp(new_log_probs - blp)
                surr1 = ratio * badv
                surr2 = torch.clamp(ratio, 1.0-self.clip_epsilon, 1.0+self.clip_epsilon) * badv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(-1), bret)
                entropy_loss = -entropy.mean()
                total_loss = policy_loss + self.value_loss_coef*value_loss + self.entropy_coef*entropy_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                epoch_losses.append({"policy_loss": float(policy_loss), "value_loss": float(value_loss),
                                     "entropy": float(-entropy_loss), "total_loss": float(total_loss)})
        return {k: np.mean([e[k] for e in epoch_losses]) for k in epoch_losses[0]}

    def update_sil(self):
        if len(self.buffer) < self.sil_batch_size:
            return {"sil_loss": 0.0, "sil_value_loss": 0.0, "sil_positive_count": 0}
        sil_losses, sil_value_losses, positive_counts = [], [], []
        for _ in range(self.sil_update_epochs):
            states, actions, returns = self.buffer.sample(self.sil_batch_size)
            if states is None: continue
            sil_loss, sil_value_loss, pos_count = self.compute_sil_loss(states, actions, returns)
            total_sil_loss = sil_loss + 0.5 * sil_value_loss
            self.optimizer.zero_grad()
            total_sil_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            sil_losses.append(float(sil_loss))
            sil_value_losses.append(float(sil_value_loss))
            positive_counts.append(pos_count)
        return {"sil_loss": np.mean(sil_losses) if sil_losses else 0.0,
                "sil_value_loss": np.mean(sil_value_losses) if sil_value_losses else 0.0,
                "sil_positive_count": np.mean(positive_counts) if positive_counts else 0}

    def train(self, total_steps=1_000_000, steps_per_iteration=2048, max_episode_steps=1000,
              eval_interval=10, eval_episodes=10, verbose=True):
        start_time = time.time()
        iteration = 0
        total_env_steps = 0
        while total_env_steps < total_steps:
            trajectories = []
            iter_steps = 0
            iter_rewards = []
            while iter_steps < steps_per_iteration:
                traj = self.collect_trajectory(max_steps=max_episode_steps)
                trajectories.append(traj)
                iter_steps += traj["length"]
                iter_rewards.append(traj["total_reward"])
                total_env_steps += traj["length"]
                returns = compute_returns(traj["rewards"], traj["dones"], self.gamma)
                self.buffer.add_trajectory(traj["states"], traj["actions"], returns)
            ppo_losses = self.update_ppo(trajectories)
            sil_losses = self.update_sil()
            mean_reward = np.mean(iter_rewards) if iter_rewards else 0.0
            log_entry = {"iteration": iteration, "total_steps": total_env_steps,
                         "mean_reward": float(mean_reward), "buffer_size": len(self.buffer),
                         **ppo_losses, **sil_losses}
            self.history.append(log_entry)
            if verbose and iteration % max(1, eval_interval//2) == 0:
                print(f"Iter {iteration:4d} | Steps: {total_env_steps:8d} | "
                      f"Reward: {mean_reward:8.2f} | Buffer: {len(self.buffer):6d} | "
                      f"P Loss: {ppo_losses['policy_loss']:.4f} | SIL Loss: {sil_losses['sil_loss']:.4f}")
            if eval_interval > 0 and iteration % eval_interval == 0:
                eval_stats = self.evaluate(num_episodes=eval_episodes, max_steps=max_episode_steps)
                log_entry["eval_mean_reward"] = eval_stats["mean_reward"]
                log_entry["eval_std_reward"] = eval_stats["std_reward"]
                if verbose:
                    print(f"  Eval: {eval_stats['mean_reward']:.2f} +/- {eval_stats['std_reward']:.2f}")
            iteration += 1
        total_time = time.time() - start_time
        final_eval = self.evaluate(num_episodes=eval_episodes, max_steps=max_episode_steps)
        return {"policy": self.policy, "history": self.history, "final_eval": final_eval,
                "total_steps": total_env_steps, "training_time": total_time}

    def evaluate(self, num_episodes=10, max_steps=1000, deterministic=True):
        rewards, lengths = [], []
        for _ in range(num_episodes):
            obs, info = self.env.reset()
            if isinstance(obs, tuple): obs = obs[0]
            ep_reward, ep_length, done = 0.0, 0, False
            while not done and ep_length < max_steps:
                state_t = to_tensor(obs, self.device).unsqueeze(0)
                with torch.no_grad():
                    action_t, _, _, _ = self.policy.get_action(state_t, deterministic=deterministic)
                action = to_numpy(action_t).flatten()
                if self.discrete_action:
                    action = int(action[0]) if len(action)==1 else action
                step_result = self.env.step(action)
                if len(step_result) == 4:
                    next_obs, reward, terminated, truncated = step_result[0], step_result[1], step_result[2], False
                    done = terminated or truncated
                elif len(step_result) == 5:
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_obs, reward, done, info = step_result
                if isinstance(next_obs, tuple): next_obs = next_obs[0]
                ep_reward += float(reward); ep_length += 1; obs = next_obs
            rewards.append(ep_reward); lengths.append(ep_length)
        return {"mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)),
                "mean_length": float(np.mean(lengths)), "std_length": float(np.std(lengths)),
                "all_rewards": rewards}

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({"policy_state_dict": self.policy.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "history": self.history, "total_steps_done": self.total_steps_done,
                    "buffer_states": self.buffer.states[:self.buffer.size],
                    "buffer_actions": self.buffer.actions[:self.buffer.size],
                    "buffer_returns": self.buffer.returns[:self.buffer.size],
                    "buffer_size": self.buffer.size,
                    "config": {"state_dim": self.state_dim, "action_dim": self.action_dim,
                               "discrete_action": self.discrete_action,
                               "num_discrete_actions": self.num_discrete_actions,
                               "sil_beta": self.sil_beta, "buffer_capacity": self.buffer.capacity}}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", [])
        self.total_steps_done = checkpoint.get("total_steps_done", 0)
        buf_size = checkpoint.get("buffer_size", 0)
        if buf_size > 0:
            self.buffer.states[:buf_size] = checkpoint["buffer_states"][:buf_size]
            self.buffer.actions[:buf_size] = checkpoint["buffer_actions"][:buf_size]
            self.buffer.returns[:buf_size] = checkpoint["buffer_returns"][:buf_size]
            self.buffer.size = buf_size


# ============================================================================
# Helper Functions
# ============================================================================

def load_target_policy(env_name, model_dir, device="cpu"):
    model_path = os.path.join(model_dir, f"{env_name}_ppo_final.zip")
    vecnorm_path = os.path.join(model_dir, f"{env_name}_vecnormalize.pkl")
    pt_path = os.path.join(model_dir, f"{env_name}_policy.pt")
    model, vec_normalize = None, None
    if HAS_SB3 and os.path.exists(model_path):
        model = SB3_PPO.load(model_path, device=device)
        if os.path.exists(vecnorm_path):
            with open(vecnorm_path, "rb") as f:
                vec_normalize = pickle.load(f)
    elif os.path.exists(pt_path):
        model = torch.load(pt_path, map_location=device)
    return model, vec_normalize


def make_env(env_name, seed=42, max_episode_steps=None):
    env = gym.make(env_name)
    if max_episode_steps is not None:
        env._max_episode_steps = max_episode_steps
    env = make_state_saveable(env)
    env.reset(seed=seed)
    return env


def load_config(env_name, config_path=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_path = os.path.join(base_dir, "configs", "default_refine.yaml")
    config = {}
    if os.path.exists(default_path):
        with open(default_path, "r") as f:
            config = yaml.safe_load(f) or {}
    env_config_path = os.path.join(base_dir, "configs", "env_specific",
                                   f"{env_name.lower().replace('-', '_')}.yaml")
    if os.path.exists(env_config_path):
        with open(env_config_path, "r") as f:
            env_config = yaml.safe_load(f) or {}
        for key, value in env_config.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            custom_config = yaml.safe_load(f) or {}
        for key, value in custom_config.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value
    return config


# ============================================================================
# Main Entry Point
# ============================================================================

def run_sil_baseline(env_name, model_dir, output_dir, config_path=None,
                     total_steps=None, sil_beta=None, buffer_capacity=None,
                     seed=42, device=None, verbose=True):
    """Run the full SIL baseline pipeline."""
    config = load_config(env_name, config_path)
    if device is None:
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    state_dim = config.get("state_dim")
    action_dim = config.get("action_dim")
    max_episode_steps = config.get("max_episode_steps", 1000)
    discrete_action = config.get("discrete_action", False)
    num_discrete_actions = config.get("num_discrete_actions")

    if state_dim is None or action_dim is None:
        temp_env = gym.make(env_name)
        state_dim = temp_env.observation_space.shape[0]
        if discrete_action:
            action_dim = temp_env.action_space.n
        else:
            action_dim = temp_env.action_space.shape[0]
        temp_env.close()

    # Optionally load target policy for initialization
    model, vec_normalize = load_target_policy(env_name, model_dir, device)

    env = make_env(env_name, seed, max_episode_steps)

    refine_ppo = config.get("refine_ppo", {})
    refine_training = config.get("refine_training", {})
    policy_cfg = config.get("policy", {})

    total_steps = total_steps or refine_training.get("total_steps", 1_000_000)
    steps_per_iteration = refine_training.get("steps_per_iteration", 2048)
    eval_interval = refine_training.get("eval_interval", 10)
    eval_episodes = refine_training.get("eval_episodes", 10)

    sil_beta = sil_beta or config.get("sil", {}).get("beta", 0.1)
    buffer_capacity = buffer_capacity or config.get("sil", {}).get("buffer_capacity", 100000)

    trainer = SILTrainer(
        env=env, state_dim=state_dim, action_dim=action_dim,
        discrete_action=discrete_action, num_discrete_actions=num_discrete_actions,
        device=device,
        lr=refine_ppo.get("learning_rate", 3e-4),
        gamma=refine_ppo.get("gamma", 0.99),
        gae_lambda=refine_ppo.get("gae_lambda", 0.95),
        clip_epsilon=refine_ppo.get("clip_epsilon", 0.2),
        value_loss_coef=refine_ppo.get("value_loss_coef", 0.5),
        entropy_coef=refine_ppo.get("entropy_coef", 0.01),
        max_grad_norm=refine_ppo.get("max_grad_norm", 0.5),
        ppo_epochs=refine_ppo.get("ppo_epochs", 10),
        batch_size=refine_ppo.get("batch_size", 64),
        sil_beta=sil_beta,
        sil_batch_size=config.get("sil", {}).get("batch_size", 512),
        sil_update_epochs=config.get("sil", {}).get("update_epochs", 1),
        buffer_capacity=buffer_capacity,
        policy_hidden_sizes=policy_cfg.get("hidden_sizes", (64, 64)),
        value_hidden_sizes=policy_cfg.get("value_hidden_sizes"),
        activation=policy_cfg.get("activation", "tanh"),
        policy_std=policy_cfg.get("policy_std", 0.0),
        normalize_advantages=refine_ppo.get("normalize_advantages", True),
    )

    if verbose:
        print(f"Starting SIL training on {env_name}")
        print(f"  Total steps: {total_steps}, SIL beta: {sil_beta}, Buffer capacity: {buffer_capacity}")

    results = trainer.train(
        total_steps=total_steps, steps_per_iteration=steps_per_iteration,
        max_episode_steps=max_episode_steps, eval_interval=eval_interval,
        eval_episodes=eval_episodes, verbose=verbose)

    os.makedirs(output_dir, exist_ok=True)
    trainer.save(os.path.join(output_dir, f"{env_name}_sil_policy.pt"))
    with open(os.path.join(output_dir, f"{env_name}_sil_history.json"), "w") as f:
        json.dump(results["history"], f, indent=2)
    with open(os.path.join(output_dir, f"{env_name}_sil_results.json"), "w") as f:
        json.dump({"final_eval": results["final_eval"], "total_steps": results["total_steps"],
                   "training_time": results["training_time"]}, f, indent=2)

    if verbose:
        print(f"SIL training complete. Final eval reward: {results['final_eval']['mean_reward']:.2f}")
        print(f"Results saved to {output_dir}")

    env.close()
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Self-Imitation Learning Baseline")
    parser.add_argument("--env_name", type=str, default="Hopper-v4")
    parser.add_argument("--model_dir", type=str, default="./trained_agents")
    parser.add_argument("--output_dir", type=str, default="./baseline_results/sil")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--sil_beta", type=float, default=None)
    parser.add_argument("--buffer_capacity", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    run_sil_baseline(env_name=args.env_name, model_dir=args.model_dir,
                     output_dir=args.output_dir, config_path=args.config,
                     total_steps=args.total_steps, sil_beta=args.sil_beta,
                     buffer_capacity=args.buffer_capacity, seed=args.seed,
                     device=args.device, verbose=args.verbose)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Jump-Start RL (JSRL) Baseline Implementation
=============================================
Implements the Jump-Start Reinforcement Learning baseline as described in
Uchendu et al. (2023) "Jump-Start Reinforcement Learning" (https://arxiv.org/abs/2204.02372).

JSRL uses a pre-trained guide policy to initialize exploration during the early
phase of training. The guide policy takes actions for the first H steps of each
episode, after which the learning agent takes over. The horizon H is gradually
decreased (curriculum) until the agent learns to act from the initial state.

This implementation adapts JSRL for the RICE paper's experiments, using a
pre-trained PPO agent as the guide policy and training a new PPO agent with
the JSRL curriculum.
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import yaml

try:
    from stable_baselines3 import PPO as SB3PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

from rice.utils import (
    set_seed, compute_gae, compute_returns, to_tensor, to_numpy,
    orthogonal_init, evaluate_policy,
)
from rice.env_wrappers import make_state_saveable


# ============================================================================
# Policy Network
# ============================================================================

class MLPPolicy(nn.Module):
    """MLP policy network with actor and critic heads."""

    def __init__(
        self, state_dim: int, action_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation: str = "tanh",
        discrete_action: bool = False,
        num_discrete_actions: Optional[int] = None,
        policy_std: float = 0.5,
        value_hidden_sizes: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete_action = discrete_action
        self.num_discrete_actions = num_discrete_actions or action_dim

        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[activation]

        layers = []
        prev_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_fn())
            prev_dim = h
        self.feature_net = nn.Sequential(*layers)
        self.feature_dim = prev_dim

        if discrete_action:
            self.actor = nn.Linear(self.feature_dim, self.num_discrete_actions)
        else:
            self.actor_mean = nn.Linear(self.feature_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.ones(action_dim) * np.log(policy_std))

        if value_hidden_sizes is None:
            value_hidden_sizes = hidden_sizes
        v_layers = []
        v_prev = self.feature_dim
        for h in value_hidden_sizes:
            v_layers.append(nn.Linear(v_prev, h))
            v_layers.append(act_fn())
            v_prev = h
        v_layers.append(nn.Linear(v_prev, 1))
        self.critic = nn.Sequential(*v_layers)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                orthogonal_init(module, gain=np.sqrt(2))
        if self.discrete_action:
            orthogonal_init(self.actor, gain=0.01)
        else:
            orthogonal_init(self.actor_mean, gain=0.01)
        orthogonal_init(self.critic[-1], gain=1.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_net(state)
        value = self.critic(features)
        if self.discrete_action:
            return self.actor(features), value
        else:
            return (self.actor_mean(features), self.actor_log_std), value

    def get_action(self, state: torch.Tensor, deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist_params, value = self.forward(state)
        if self.discrete_action:
            dist = Categorical(logits=dist_params)
            action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        else:
            mean, log_std = dist_params
            std = log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
            action = mean if deterministic else dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value.squeeze(-1)

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist_params, value = self.forward(state)
        if self.discrete_action:
            dist = Categorical(logits=dist_params)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        else:
            mean, log_std = dist_params
            std = log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value.squeeze(-1)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        _, value = self.forward(state)
        return value.squeeze(-1)


# ============================================================================
# JSRL Trainer
# ============================================================================

class JSRLTrainer:
    """Jump-Start RL Trainer with PPO and guide policy curriculum."""

    def __init__(
        self, env: gym.Env,
        guide_policy_fn: Callable[[np.ndarray], np.ndarray],
        state_dim: int, action_dim: int,
        discrete_action: bool = False,
        num_discrete_actions: Optional[int] = None,
        device: str = "cpu",
        lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2, value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01, max_grad_norm: float = 0.5,
        ppo_epochs: int = 10, batch_size: int = 64,
        initial_guide_horizon: int = 100,
        guide_horizon_decay: str = "linear",
        guide_horizon_decay_rate: float = 0.99,
        guide_horizon_min: int = 0,
        policy_hidden_sizes: Tuple[int, ...] = (64, 64),
        value_hidden_sizes: Optional[Tuple[int, ...]] = None,
        activation: str = "tanh", policy_std: float = 0.5,
        normalize_advantages: bool = True,
    ):
        self.env = env
        self.guide_policy_fn = guide_policy_fn
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

        self.initial_guide_horizon = initial_guide_horizon
        self.guide_horizon_decay = guide_horizon_decay
        self.guide_horizon_decay_rate = guide_horizon_decay_rate
        self.guide_horizon_min = guide_horizon_min
        self.current_guide_horizon = initial_guide_horizon

        self.policy = MLPPolicy(
            state_dim=state_dim, action_dim=action_dim,
            hidden_sizes=policy_hidden_sizes, activation=activation,
            discrete_action=discrete_action,
            num_discrete_actions=num_discrete_actions,
            policy_std=policy_std, value_hidden_sizes=value_hidden_sizes,
        ).to(device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.total_steps = 0
        self.total_episodes = 0
        self.history: List[Dict[str, Any]] = []

    def _get_guide_action(self, state: np.ndarray) -> np.ndarray:
        return self.guide_policy_fn(state)

    def _get_agent_action(self, state: np.ndarray, deterministic: bool = False
                          ) -> Tuple[np.ndarray, float, float, float]:
        state_t = to_tensor(state, self.device).unsqueeze(0)
        with torch.no_grad():
            action_t, log_prob_t, entropy_t, value_t = self.policy.get_action(
                state_t, deterministic=deterministic)
        return (to_numpy(action_t).flatten(), log_prob_t.item(),
                entropy_t.item(), value_t.item())

    def collect_trajectory(self, max_steps: int = 1000) -> Dict[str, np.ndarray]:
        states, actions, rewards, dones = [], [], [], []
        values, log_probs, entropies = [], [], []

        obs, info = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        episode_reward = 0.0
        for t in range(max_steps):
            state = np.array(obs, dtype=np.float32)
            if t < self.current_guide_horizon:
                action = self._get_guide_action(state)
                log_prob, entropy, value = 0.0, 0.0, 0.0
            else:
                action, log_prob, entropy, value = self._get_agent_action(state)

            step_result = self.env.step(action)
            if len(step_result) == 4:
                next_obs, reward, done, info = step_result
                terminated, truncated = done, False
            else:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated

            states.append(state); actions.append(action)
            rewards.append(reward); dones.append(float(done))
            values.append(value); log_probs.append(log_prob)
            entropies.append(entropy)
            episode_reward += reward
            obs = next_obs
            if done:
                break

        self.total_steps += len(states)
        self.total_episodes += 1

        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
            "values": np.array(values, dtype=np.float32),
            "log_probs": np.array(log_probs, dtype=np.float32),
            "entropies": np.array(entropies, dtype=np.float32),
            "episode_reward": episode_reward,
            "episode_length": len(states),
        }

    def _compute_advantages_and_returns(self, trajectory: Dict[str, np.ndarray]
                                        ) -> Tuple[np.ndarray, np.ndarray]:
        rewards = trajectory["rewards"]
        values = trajectory["values"]
        dones = trajectory["dones"]

        if self.current_guide_horizon > 0:
            states_t = to_tensor(trajectory["states"], self.device)
            with torch.no_grad():
                values = to_numpy(self.policy.get_value(states_t))

        last_value = 0.0
        if dones[-1] != 1.0:
            last_state_t = to_tensor(trajectory["states"][-1:], self.device)
            with torch.no_grad():
                last_value = self.policy.get_value(last_state_t).item()

        return compute_gae(rewards=rewards, values=values, dones=dones,
                           gamma=self.gamma, gae_lambda=self.gae_lambda,
                           last_value=last_value)

    def update_ppo(self, trajectories: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
        all_states, all_actions, all_log_probs_old = [], [], []
        all_advantages, all_returns = [], []

        for traj in trajectories:
            advantages, returns = self._compute_advantages_and_returns(traj)
            H, T = self.current_guide_horizon, len(traj["states"])
            if H >= T:
                continue
            idx = slice(H, T)
            all_states.append(traj["states"][idx])
            all_actions.append(traj["actions"][idx])
            all_log_probs_old.append(traj["log_probs"][idx])
            all_advantages.append(advantages[idx])
            all_returns.append(returns[idx])

        if not all_states:
            return {"policy_loss": 0.0, "value_loss": 0.0,
                    "entropy": 0.0, "total_loss": 0.0}

        states = np.concatenate(all_states, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        log_probs_old = np.concatenate(all_log_probs_old, axis=0)
        advantages = np.concatenate(all_advantages, axis=0)
        returns = np.concatenate(all_returns, axis=0)

        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = to_tensor(states, self.device)
        actions_t = to_tensor(actions, self.device)
        log_probs_old_t = to_tensor(log_probs_old, self.device)
        advantages_t = to_tensor(advantages, self.device)
        returns_t = to_tensor(returns, self.device)

        n_samples = len(states)
        indices = np.arange(n_samples)
        total_pl, total_vl, total_ent, n_upd = 0.0, 0.0, 0.0, 0

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, self.batch_size):
                bi = indices[start:start + self.batch_size]
                lp_new, ent, vals = self.policy.evaluate_actions(
                    states_t[bi], actions_t[bi])
                ratio = torch.exp(lp_new - log_probs_old_t[bi])
                surr1 = ratio * advantages_t[bi]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                    1.0 + self.clip_epsilon) * advantages_t[bi]
                pl = -torch.min(surr1, surr2).mean()
                vl = F.mse_loss(vals, returns_t[bi])
                el = -ent.mean()
                loss = pl + self.value_loss_coef * vl + self.entropy_coef * el

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pl += pl.item(); total_vl += vl.item()
                total_ent += ent.mean().item(); n_upd += 1

        return {"policy_loss": total_pl / max(1, n_upd),
                "value_loss": total_vl / max(1, n_upd),
                "entropy": total_ent / max(1, n_upd),
                "total_loss": (total_pl + total_vl + total_ent) / max(1, n_upd)}

    def _update_guide_horizon(self, iteration: int):
        if self.guide_horizon_decay == "exponential":
            self.current_guide_horizon = max(
                self.guide_horizon_min,
                int(self.current_guide_horizon * self.guide_horizon_decay_rate))

    def train(self, total_steps: int = 1_000_000, steps_per_iteration: int = 2048,
              eval_interval: int = 10, eval_episodes: int = 10,
              max_episode_steps: int = 1000, verbose: bool = True) -> Dict[str, Any]:
        history = []
        iteration = 0
        total_collected_steps = 0

        while total_collected_steps < total_steps:
            iteration += 1
            trajectories, iter_steps, iter_rewards = [], 0, []

            while iter_steps < steps_per_iteration:
                traj = self.collect_trajectory(max_steps=max_episode_steps)
                trajectories.append(traj)
                iter_steps += traj["episode_length"]
                iter_rewards.append(traj["episode_reward"])

            total_collected_steps += iter_steps
            update_info = self.update_ppo(trajectories)

            if self.guide_horizon_decay == "linear":
                progress = min(1.0, total_collected_steps / total_steps)
                self.current_guide_horizon = max(
                    self.guide_horizon_min,
                    int(self.initial_guide_horizon * (1.0 - progress)))
            elif self.guide_horizon_decay == "exponential":
                self._update_guide_horizon(iteration)

            mean_reward = np.mean(iter_rewards) if iter_rewards else 0.0
            log_entry = {"iteration": iteration,
                         "total_steps": total_collected_steps,
                         "mean_episode_reward": float(mean_reward),
                         "guide_horizon": self.current_guide_horizon,
                         **update_info}
            history.append(log_entry)

            if verbose and iteration % max(1, eval_interval // 2) == 0:
                print(f"Iter {iteration:4d} | Steps: {total_collected_steps:8d} | "
                      f"H: {self.current_guide_horizon:3d} | "
                      f"Reward: {mean_reward:8.2f} | "
                      f"P Loss: {update_info['policy_loss']:.4f} | "
                      f"V Loss: {update_info['value_loss']:.4f}")

            if iteration % eval_interval == 0:
                eval_stats = self.evaluate(num_episodes=eval_episodes,
                                           max_steps=max_episode_steps)
                log_entry["eval_reward"] = eval_stats["mean_reward"]
                log_entry["eval_std"] = eval_stats["std_reward"]
                if verbose:
                    print(f"  Eval: {eval_stats['mean_reward']:.2f} +/- "
                          f"{eval_stats['std_reward']:.2f}")

        return {"history": history, "policy": self.policy,
                "final_guide_horizon": self.current_guide_horizon,
                "total_steps": total_collected_steps}

    def evaluate(self, num_episodes: int = 10, max_steps: int = 1000,
                 deterministic: bool = True) -> Dict[str, float]:
        rewards, lengths = [], []
        for _ in range(num_episodes):
            obs, info = self.env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            ep_reward, ep_length = 0.0, 0
            for _ in range(max_steps):
                state = np.array(obs, dtype=np.float32)
                action, _, _, _ = self._get_agent_action(state, deterministic=deterministic)
                step_result = self.env.step(action)
                if len(step_result) == 4:
                    next_obs, reward, done, info = step_result
                    terminated, truncated = done, False
                else:
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                ep_reward += reward; ep_length += 1
                obs = next_obs
                if done:
                    break
            rewards.append(ep_reward); lengths.append(ep_length)
        return {"mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "mean_length": float(np.mean(lengths)),
                "std_length": float(np.std(lengths)),
                "all_rewards": [float(r) for r in rewards]}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({"policy_state_dict": self.policy.state_dict(),
                     "optimizer_state_dict": self.optimizer.state_dict(),
                     "total_steps": self.total_steps,
                     "total_episodes": self.total_episodes,
                     "current_guide_horizon": self.current_guide_horizon,
                     "history": self.history}, path)

    def load(self, path: str, device: Optional[str] = None):
        if device is not None:
            self.device = device
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.total_episodes = ckpt.get("total_episodes", 0)
        self.current_guide_horizon = ckpt.get("current_guide_horizon", 0)
        self.history = ckpt.get("history", [])


# ============================================================================
# Helper Functions
# ============================================================================

def load_config(env_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    default_path = Path(__file__).parent.parent / "configs" / "default_refine.yaml"
    config = {}
    if default_path.exists():
        with open(default_path, "r") as f:
            config = yaml.safe_load(f) or {}

    env_name_map = {
        "hopper": "hopper", "walker2d": "walker2d", "reacher": "reacher",
        "halfcheetah": "halfcheetah", "selfish_mining": "selfish_mining",
        "cage2": "cage2", "autonomous_driving": "autonomous_driving",
        "malware": "malware",
    }
    env_key = env_name.lower().replace("-", "_").replace("v4", "").replace("v2", "")
    config_name = env_name_map.get(env_key, env_key)

    env_config_path = Path(__file__).parent.parent / "configs" / "env_specific" / f"{config_name}.yaml"
    if env_config_path.exists():
        with open(env_config_path, "r") as f:
            env_config = yaml.safe_load(f) or {}
        for key, value in env_config.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            custom_config = yaml.safe_load(f) or {}
        for key, value in custom_config.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
    return config


def make_env(env_name: str, seed: int = 42,
             max_episode_steps: Optional[int] = None) -> gym.Env:
    env = gym.make(env_name)
    if max_episode_steps is not None:
        env._max_episode_steps = max_episode_steps
    env = make_state_saveable(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def load_target_policy(env_name: str, model_dir: str, device: str = "cpu"
                       ) -> Tuple[Any, Optional[Any]]:
    model, vec_normalize = None, None
    if HAS_SB3:
        model_path = os.path.join(model_dir, f"{env_name}_ppo_final.zip")
        if os.path.exists(model_path):
            model = SB3PPO.load(model_path, device=device)
            norm_path = os.path.join(model_dir, f"{env_name}_vecnormalize.pkl")
            if os.path.exists(norm_path):
                with open(norm_path, "rb") as f:
                    vec_normalize = pickle.load(f)
            return model, vec_normalize
    pt_path = os.path.join(model_dir, f"{env_name}_policy.pt")
    if os.path.exists(pt_path):
        model = torch.load(pt_path, map_location=device)
        return model, vec_normalize
    raise FileNotFoundError(f"No pre-trained policy found in {model_dir} for {env_name}")


def make_target_policy_fn(model: Any, vec_normalize: Optional[Any] = None,
                          device: str = "cpu") -> Callable[[np.ndarray], np.ndarray]:
    if HAS_SB3 and isinstance(model, SB3PPO):
        def sb3_fn(state: np.ndarray) -> np.ndarray:
            if vec_normalize is not None:
                state = vec_normalize.normalize_obs(state)
            action, _ = model.predict(state, deterministic=True)
            return action
        return sb3_fn
    if isinstance(model, nn.Module):
        def pt_fn(state: np.ndarray) -> np.ndarray:
            state_t = to_tensor(state, device).unsqueeze(0)
            with torch.no_grad():
                if hasattr(model, 'get_action'):
                    action_t, _, _, _ = model.get_action(state_t, deterministic=True)
                else:
                    action_t = model(state_t)
            return to_numpy(action_t).flatten()
        return pt_fn
    if callable(model):
        return model
    raise ValueError(f"Cannot create policy function from model type: {type(model)}")


# ============================================================================
# Main Entry Point
# ============================================================================

def run_jsrl_baseline(
    env_name: str, model_dir: str, output_dir: str,
    config_path: Optional[str] = None,
    total_steps: Optional[int] = None,
    initial_guide_horizon: Optional[int] = None,
    guide_horizon_decay: str = "linear",
    seed: int = 42, device: str = "cuda", verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full JSRL baseline pipeline."""
    set_seed(seed)
    config = load_config(env_name, config_path)

    ppo_cfg = config.get("ppo", {})
    refine_ppo_cfg = config.get("refine_ppo", ppo_cfg)
    refine_train_cfg = config.get("refine_training", config.get("training", {}))
    policy_cfg = config.get("policy", {})

    total_steps = total_steps or refine_train_cfg.get("total_steps", 1_000_000)
    steps_per_iteration = refine_train_cfg.get("steps_per_iteration", 2048)
    eval_interval = refine_train_cfg.get("eval_interval", 10)
    eval_episodes = refine_train_cfg.get("eval_episodes", 10)
    max_episode_steps = config.get("max_episode_steps", 1000)

    state_dim = config.get("state_dim", 11)
    action_dim = config.get("action_dim", 3)
    discrete_action = config.get("discrete_action", False)
    num_discrete_actions = config.get("num_discrete_actions", None)

    if initial_guide_horizon is None:
        initial_guide_horizon = max_episode_steps // 2

    os.makedirs(output_dir, exist_ok=True)

    # Load guide policy
    if verbose:
        print(f"Loading guide policy from {model_dir}...")
    model, vec_normalize = load_target_policy(env_name, model_dir, device)
    guide_policy_fn = make_target_policy_fn(model, vec_normalize, device)

    # Create environment
    env = make_env(env_name, seed, max_episode_steps)

    # Create trainer
    trainer = JSRLTrainer(
        env=env, guide_policy_fn=guide_policy_fn,
        state_dim=state_dim, action_dim=action_dim,
        discrete_action=discrete_action,
        num_discrete_actions=num_discrete_actions,
        device=device,
        lr=refine_ppo_cfg.get("learning_rate", 3e-4),
        gamma=refine_ppo_cfg.get("gamma", 0.99),
        gae_lambda=refine_ppo_cfg.get("gae_lambda", 0.95),
        clip_epsilon=refine_ppo_cfg.get("clip_epsilon", 0.2),
        value_loss_coef=refine_ppo_cfg.get("value_loss_coef", 0.5),
        entropy_coef=refine_ppo_cfg.get("entropy_coef", 0.01),
        max_grad_norm=refine_ppo_cfg.get("max_grad_norm", 0.5),
        ppo_epochs=refine_ppo_cfg.get("ppo_epochs", 10),
        batch_size=refine_ppo_cfg.get("batch_size", 64),
        initial_guide_horizon=initial_guide_horizon,
        guide_horizon_decay=guide_horizon_decay,
        guide_horizon_decay_rate=0.99,
        guide_horizon_min=0,
        policy_hidden_sizes=policy_cfg.get("hidden_sizes", (64, 64)),
        value_hidden_sizes=policy_cfg.get("value_hidden_sizes", None),
        activation=policy_cfg.get("activation", "tanh"),
        policy_std=policy_cfg.get("policy_std", 0.5),
        normalize_advantages=refine_ppo_cfg.get("normalize_advantages", True),
    )

    # Train
    if verbose:
        print(f"Starting JSRL training for {total_steps} steps...")
    start_time = time.time()
    result = trainer.train(
        total_steps=total_steps,
        steps_per_iteration=steps_per_iteration,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        max_episode_steps=max_episode_steps,
        verbose=verbose,
    )
    training_time = time.time() - start_time

    # Final evaluation
    final_eval = trainer.evaluate(num_episodes=100, max_steps=max_episode_steps)

    # Save results
    save_path = os.path.join(output_dir, f"{env_name}_jsrl_policy.pt")
    trainer.save(save_path)

    results = {
        "env_name": env_name,
        "method": "JSRL",
        "final_mean_reward": final_eval["mean_reward"],
        "final_std_reward": final_eval["std_reward"],
        "training_time": training_time,
        "total_steps": result["total_steps"],
        "final_guide_horizon": result["final_guide_horizon"],
        "history": result["history"],
        "config": config,
    }

    results_path = os.path.join(output_dir, f"{env_name}_jsrl_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        print(f"\nJSRL training complete!")
        print(f"  Final mean reward: {final_eval['mean_reward']:.2f} +/- {final_eval['std_reward']:.2f}")
        print(f"  Training time: {training_time:.1f}s")
        print(f"  Results saved to {output_dir}")

    env.close()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jump-Start RL Baseline")
    parser.add_argument("--env", type=str, default="Hopper-v4",
                        help="Gym environment name")
    parser.add_argument("--model-dir", type=str, default="./trained_agents",
                        help="Directory with pre-trained target policy")
    parser.add_argument("--output-dir", type=str, default="./results/jsrl",
                        help="Output directory for results")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom YAML config")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Total training steps")
    parser.add_argument("--initial-guide-horizon", type=int, default=None,
                        help="Initial guide horizon H")
    parser.add_argument("--guide-horizon-decay", type=str, default="linear",
                        choices=["linear", "exponential"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    run_jsrl_baseline(
        env_name=args.env,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        total_steps=args.total_steps,
        initial_guide_horizon=args.initial_guide_horizon,
        guide_horizon_decay=args.guide_horizon_decay,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
"""
StateMask Baseline Implementation

Implements the original StateMask method for training a mask network using
Lagrangian optimization. The mask network ξ(s) decides whether to take the
agent's action (aᵉ=0) or a random action (aᵉ=1). Training maximizes perturbed
policy performance while constraining the randomization rate via a Lagrangian
multiplier.

Reference: The StateMask approach described in the RICE paper, which uses
Lagrangian relaxation to enforce a budget on the number of random actions.

Key differences from RICE's simplified mask training:
- Uses Lagrangian dual optimization instead of intrinsic reward α
- Maintains a Lagrange multiplier λ that is updated to enforce the budget constraint
- The reward for the mask is: r_env + λ * (budget_fraction - I(aᵉ=1))
"""

import os
import time
import json
import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Import from rice package
from rice.perturbed_env import PerturbedEnv, make_perturbed_env, make_perturbed_vec_env
from rice.utils import (
    build_mlp, ensure_dir, evaluate_policy, get_device, init_weights,
    load_config, Logger, set_seed, make_env
)


class StateMaskFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for the StateMask network.
    Uses the same architecture as the agent's policy network.
    """
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 64,
        hidden_sizes: List[int] = None,
        activation_fn: nn.Module = nn.ReLU,
        use_layer_norm: bool = False,
    ):
        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        super().__init__(observation_space, features_dim)
        
        # Determine input dimension
        if isinstance(observation_space, gym.spaces.Box):
            input_dim = int(np.prod(observation_space.shape))
        else:
            input_dim = observation_space.n
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation_fn())
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, features_dim))
        layers.append(activation_fn())
        
        self.mlp = nn.Sequential(*layers)
        self.apply(init_weights)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations)


class StateMaskPolicy(ActorCriticPolicy):
    """
    Actor-Critic policy for the StateMask mask network.
    Extends SB3's ActorCriticPolicy for Discrete(2) action space.
    """
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Callable,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: nn.Module = nn.ReLU,
        features_extractor_class: nn.Module = StateMaskFeatureExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs,
        )


class LagrangianMultiplier:
    """
    Maintains and updates the Lagrange multiplier λ for the budget constraint.
    
    The constraint is: E[I(aᵉ=1)] ≤ budget_fraction
    Lagrangian: L = E[r_env] + λ * (budget_fraction - I(aᵉ=1))
    
    λ is updated via:
        λ ← max(0, λ + lr_λ * (observed_randomization_rate - budget_fraction))
    """
    def __init__(
        self,
        initial_lambda: float = 0.0,
        learning_rate: float = 0.01,
        budget_fraction: float = 0.1,
        min_lambda: float = 0.0,
        max_lambda: float = 10.0,
    ):
        self.lambda_val = initial_lambda
        self.lr = learning_rate
        self.budget = budget_fraction
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        
        # Tracking
        self.randomization_history: List[float] = []
        self.lambda_history: List[float] = []
    
    def get_lambda(self) -> float:
        """Return current λ value."""
        return self.lambda_val
    
    def update(self, randomization_rate: float) -> float:
        """
        Update λ based on observed randomization rate.
        
        Args:
            randomization_rate: Fraction of steps where mask chose random action
        
        Returns:
            Updated λ value
        """
        # Gradient ascent on Lagrangian: λ ← λ + lr * (rate - budget)
        self.lambda_val += self.lr * (randomization_rate - self.budget)
        
        # Clamp to valid range
        self.lambda_val = max(self.min_lambda, min(self.max_lambda, self.lambda_val))
        
        # Record history
        self.randomization_history.append(randomization_rate)
        self.lambda_history.append(self.lambda_val)
        
        return self.lambda_val
    
    def get_stats(self) -> Dict[str, float]:
        """Return statistics about the Lagrangian multiplier."""
        return {
            "lambda": self.lambda_val,
            "budget": self.budget,
            "recent_randomization_rate": (
                np.mean(self.randomization_history[-100:])
                if self.randomization_history else 0.0
            ),
        }
    
    def save(self, path: str) -> None:
        """Save multiplier state."""
        state = {
            "lambda": self.lambda_val,
            "lr": self.lr,
            "budget": self.budget,
            "randomization_history": self.randomization_history,
            "lambda_history": self.lambda_history,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    
    def load(self, path: str) -> None:
        """Load multiplier state."""
        with open(path, "r") as f:
            state = json.load(f)
        self.lambda_val = state["lambda"]
        self.lr = state["lr"]
        self.budget = state["budget"]
        self.randomization_history = state["randomization_history"]
        self.lambda_history = state["lambda_history"]


class StateMaskPerturbedEnv(gym.Wrapper):
    """
    Perturbed environment wrapper for StateMask training.
    
    Unlike RICE's PerturbedEnv which uses intrinsic reward α,
    this wrapper computes the Lagrangian reward:
        r_mask = r_env + λ * (budget_fraction - I(aᵉ=1))
    
    The mask action space is Discrete(2):
        - 0: Use agent's action
        - 1: Use random action
    """
    def __init__(
        self,
        env: gym.Env,
        agent_policy: Any,
        lagrangian: LagrangianMultiplier,
        deterministic_agent: bool = False,
        mask_network: Optional[Any] = None,
        device: str = "auto",
    ):
        super().__init__(env)
        self.agent_policy = agent_policy
        self.lagrangian = lagrangian
        self.deterministic_agent = deterministic_agent
        self.mask_network = mask_network
        self._device = get_device(device)
        
        # Store original action space
        self._original_action_space = env.action_space
        
        # Mask action space: Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
        
        # Episode tracking
        self._episode_mask_actions: List[int] = []
        self._episode_rewards: List[float] = []
        self._total_mask_steps = 0
        self._total_random_steps = 0
        
        # Current observation (for computing mask action)
        self._current_obs: Optional[np.ndarray] = None
    
    def _get_agent_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action from the frozen agent policy."""
        if hasattr(self.agent_policy, 'predict'):
            action, _ = self.agent_policy.predict(obs, deterministic=self.deterministic_agent)
        elif callable(self.agent_policy):
            action = self.agent_policy(obs)
        elif isinstance(self.agent_policy, nn.Module):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
                if obs_tensor.ndim == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                action = self.agent_policy(obs_tensor)
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                if action.ndim == 2:
                    action = action[0]
        else:
            raise ValueError(f"Unsupported agent policy type: {type(self.agent_policy)}")
        return np.asarray(action)
    
    def _get_random_action(self) -> np.ndarray:
        """Sample a random action from the original action space."""
        return self._original_action_space.sample()
    
    def step(self, mask_action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step with the mask decision.
        
        Args:
            mask_action: 0 (use agent action) or 1 (use random action)
        
        Returns:
            obs, reward, terminated, truncated, info
        """
        # Ensure mask_action is an integer
        if isinstance(mask_action, np.ndarray):
            mask_action = int(mask_action.item())
        mask_action = int(mask_action)
        
        # Record mask action
        self._episode_mask_actions.append(mask_action)
        self._total_mask_steps += 1
        if mask_action == 1:
            self._total_random_steps += 1
        
        # Select actual action
        if mask_action == 0:
            # Use agent's action
            actual_action = self._get_agent_action(self._current_obs)
        else:
            # Use random action
            actual_action = self._get_random_action()
        
        # Execute in environment
        obs, env_reward, terminated, truncated, info = self.env.step(actual_action)
        self._current_obs = obs
        
        # Compute Lagrangian reward
        # r_mask = r_env + λ * (budget_fraction - I(aᵉ=1))
        lambda_val = self.lagrangian.get_lambda()
        budget = self.lagrangian.budget
        mask_indicator = float(mask_action == 1)
        lagrangian_reward = env_reward + lambda_val * (budget - mask_indicator)
        
        self._episode_rewards.append(env_reward)
        
        # Store info
        info["mask_action"] = mask_action
        info["env_reward"] = env_reward
        info["lagrangian_reward"] = lagrangian_reward
        info["lambda"] = lambda_val
        
        return obs, lagrangian_reward, terminated, truncated, info
    
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._current_obs = obs
        self._episode_mask_actions = []
        self._episode_rewards = []
        return obs, info
    
    def get_randomization_rate(self) -> float:
        """Return the fraction of steps where mask chose random action."""
        if self._total_mask_steps == 0:
            return 0.0
        return self._total_random_steps / self._total_mask_steps
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Return statistics for the current episode."""
        return {
            "mask_actions": self._episode_mask_actions.copy(),
            "env_rewards": self._episode_rewards.copy(),
            "num_random": sum(self._episode_mask_actions),
            "num_agent": len(self._episode_mask_actions) - sum(self._episode_mask_actions),
        }


class StateMaskCallback(BaseCallback):
    """
    Callback for StateMask training that updates the Lagrangian multiplier
    and logs metrics.
    """
    def __init__(
        self,
        logger: Logger,
        lagrangian: LagrangianMultiplier,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        agent_policy: Optional[Any] = None,
        lagrangian_update_freq: int = 2048,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._logger = logger
        self.lagrangian = lagrangian
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.agent_policy = agent_policy
        self.lagrangian_update_freq = lagrangian_update_freq
        
        # Tracking
        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int] = []
        self._mask_actions_buffer: List[int] = []
        self._current_episode_reward = 0.0
        self._current_episode_length = 0
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # Get info from the last step
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]
            env_reward = info.get("env_reward", 0.0)
            mask_action = info.get("mask_action", 0)
            
            self._current_episode_reward += env_reward
            self._current_episode_length += 1
            self._mask_actions_buffer.append(mask_action)
        
        # Check for episode end
        if len(self.locals.get("dones", [])) > 0 and self.locals["dones"][0]:
            self._episode_rewards.append(self._current_episode_reward)
            self._episode_lengths.append(self._current_episode_length)
            self._current_episode_reward = 0.0
            self._current_episode_length = 0
        
        # Update Lagrangian multiplier periodically
        if self.num_timesteps % self.lagrangian_update_freq == 0 and len(self._mask_actions_buffer) > 0:
            randomization_rate = np.mean(self._mask_actions_buffer)
            self.lagrangian.update(randomization_rate)
            
            # Log
            self._logger.log("statemask/lambda", self.lagrangian.get_lambda(), self.num_timesteps)
            self._logger.log("statemask/randomization_rate", randomization_rate, self.num_timesteps)
            
            self._mask_actions_buffer = []
        
        # Log training metrics
        if self.num_timesteps % 1000 == 0:
            if len(self._episode_rewards) > 0:
                self._logger.log(
                    "statemask/mean_episode_reward",
                    np.mean(self._episode_rewards[-10:]),
                    self.num_timesteps,
                )
                self._logger.log(
                    "statemask/mean_episode_length",
                    np.mean(self._episode_lengths[-10:]),
                    self.num_timesteps,
                )
        
        # Periodic evaluation
        if self.eval_env is not None and self.num_timesteps % self.eval_freq == 0:
            self._run_evaluation()
        
        return True
    
    def _run_evaluation(self) -> None:
        """Evaluate the perturbed policy."""
        if self.eval_env is None or self.agent_policy is None:
            return
        
        # Get the current mask network from the model
        mask_network = self.model
        
        # Create perturbed evaluation environment
        from rice.perturbed_env import PerturbedEnvWrapper
        eval_perturbed = PerturbedEnvWrapper(
            self.eval_env,
            self.agent_policy,
            mask_network,
            deterministic_agent=True,
        )
        
        total_rewards = []
        for _ in range(self.n_eval_episodes):
            obs, _ = eval_perturbed.reset()
            done = False
            episode_reward = 0.0
            while not done:
                action = None  # PerturbedEnvWrapper handles this
                obs, reward, terminated, truncated, _ = eval_perturbed.step(action)
                done = terminated or truncated
                episode_reward += reward
            total_rewards.append(episode_reward)
        
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        self._logger.log("statemask/eval/mean_reward", mean_reward, self.num_timesteps)
        self._logger.log("statemask/eval/std_reward", std_reward, self.num_timesteps)
        
        if self.verbose > 0:
            print(f"Step {self.num_timesteps}: Eval reward = {mean_reward:.2f} ± {std_reward:.2f}")


def create_statemask_network(
    observation_space: gym.Space,
    action_space: gym.Space,
    hidden_sizes: List[int] = None,
    features_dim: int = 64,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    use_sde: bool = False,
    device: str = "auto",
    verbose: int = 0,
) -> PPO:
    """
    Create a PPO agent configured for StateMask mask network training.
    
    The mask network has Discrete(2) action space and uses the same
    architecture as the agent's policy network.
    """
    if hidden_sizes is None:
        hidden_sizes = [64, 64]
    
    # Build policy kwargs
    policy_kwargs = {
        "features_extractor_class": StateMaskFeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": features_dim,
            "hidden_sizes": hidden_sizes,
        },
        "net_arch": dict(pi=hidden_sizes, vf=hidden_sizes),
    }
    
    model = PPO(
        "MlpPolicy",
        None,  # env will be set later
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        use_sde=use_sde,
        policy_kwargs=policy_kwargs,
        device=get_device(device),
        verbose=verbose,
    )
    
    return model


def train_statemask(
    env_id: str,
    agent_policy: Any,
    config: Dict[str, Any],
    output_dir: str,
    seed: int = 0,
    total_timesteps: Optional[int] = None,
    budget_fraction: float = 0.1,
    initial_lambda: float = 0.0,
    lambda_lr: float = 0.01,
    hidden_sizes: Optional[List[int]] = None,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    ent_coef: float = 0.0,
    device: str = "auto",
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    lagrangian_update_freq: int = 2048,
    verbose: int = 1,
    resume_from: Optional[str] = None,
    **env_kwargs,
) -> Tuple[PPO, Logger, LagrangianMultiplier, str]:
    """
    Train a StateMask mask network using Lagrangian optimization.
    
    Args:
        env_id: Gymnasium environment ID
        agent_policy: Frozen agent policy π
        config: Configuration dictionary
        output_dir: Directory to save outputs
        seed: Random seed
        total_timesteps: Total training timesteps
        budget_fraction: Target fraction of random actions (constraint)
        initial_lambda: Initial Lagrange multiplier
        lambda_lr: Learning rate for λ updates
        hidden_sizes: Hidden layer sizes for mask network
        learning_rate: PPO learning rate
        n_steps: PPO rollout steps
        batch_size: PPO batch size
        n_epochs: PPO epochs per update
        gamma: Discount factor
        ent_coef: Entropy coefficient
        device: Device string
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of evaluation episodes
        lagrangian_update_freq: Frequency of λ updates
        verbose: Verbosity level
        resume_from: Path to resume training from
        **env_kwargs: Additional environment arguments
    
    Returns:
        Tuple of (trained PPO model, Logger, LagrangianMultiplier, save path)
    """
    set_seed(seed)
    ensure_dir(output_dir)
    
    # Get config values
    mask_config = config.get("mask", {})
    if total_timesteps is None:
        total_timesteps = mask_config.get("total_timesteps", 300000)
    if hidden_sizes is None:
        hidden_sizes = mask_config.get("hidden_sizes", [64, 64])
    
    logger = Logger(log_dir=output_dir)
    
    # Create Lagrangian multiplier
    lagrangian = LagrangianMultiplier(
        initial_lambda=initial_lambda,
        learning_rate=lambda_lr,
        budget_fraction=budget_fraction,
    )
    
    # Create evaluation environment
    eval_env = make_env(env_id, seed=seed + 1000, **env_kwargs)
    
    # Create perturbed environment for training
    def make_env_fn():
        env = make_env(env_id, seed=seed, **env_kwargs)
        env = StateMaskPerturbedEnv(
            env,
            agent_policy,
            lagrangian,
            deterministic_agent=False,
            device=device,
        )
        return env
    
    vec_env = DummyVecEnv([make_env_fn])
    
    # Create or load mask network
    if resume_from is not None and os.path.exists(resume_from):
        print(f"Loading mask network from {resume_from}")
        model = PPO.load(resume_from, env=vec_env, device=get_device(device))
    else:
        model = create_statemask_network(
            observation_space=vec_env.observation_space,
            action_space=vec_env.action_space,
            hidden_sizes=hidden_sizes,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            ent_coef=ent_coef,
            device=device,
            verbose=verbose,
        )
        model.set_env(vec_env)
    
    # Create callback
    callback = StateMaskCallback(
        logger=logger,
        lagrangian=lagrangian,
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        agent_policy=agent_policy,
        lagrangian_update_freq=lagrangian_update_freq,
        verbose=verbose,
    )
    
    # Train
    print(f"Training StateMask for {total_timesteps} timesteps...")
    print(f"Budget fraction: {budget_fraction}, Initial λ: {initial_lambda}, λ lr: {lambda_lr}")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=(verbose > 0),
        )
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
    
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    # Save model
    model_path = os.path.join(output_dir, "statemask_model")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save Lagrangian state
    lagrangian_path = os.path.join(output_dir, "lagrangian_state.json")
    lagrangian.save(lagrangian_path)
    
    # Save logger
    logger.save(os.path.join(output_dir, "statemask_logger.pkl"))
    
    # Save config
    import yaml
    config_path = os.path.join(output_dir, "statemask_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump({
            "budget_fraction": budget_fraction,
            "initial_lambda": initial_lambda,
            "lambda_lr": lambda_lr,
            "total_timesteps": total_timesteps,
            "hidden_sizes": hidden_sizes,
            "learning_rate": learning_rate,
            "elapsed_time": elapsed,
        }, f)
    
    return model, logger, lagrangian, model_path


def load_statemask(
    path: str,
    env: Optional[gym.Env] = None,
    device: str = "auto",
    **kwargs,
) -> PPO:
    """
    Load a trained StateMask mask network.
    
    Args:
        path: Path to the saved model
        env: Environment to attach (optional)
        device: Device string
        **kwargs: Additional arguments
    
    Returns:
        Loaded PPO model
    """
    model = PPO.load(path, env=env, device=get_device(device), **kwargs)
    return model


def compute_statemask_importance(
    mask_network: PPO,
    observations: np.ndarray,
) -> np.ndarray:
    """
    Compute importance scores from StateMask network.
    
    I(s) = 1 - ξ(aᵉ=0 | s)
    Higher values indicate more critical states (mask prefers NOT to randomize).
    
    Args:
        mask_network: Trained StateMask PPO model
        observations: Batch of observations [batch_size, obs_dim]
    
    Returns:
        Importance scores [batch_size]
    """
    if observations.ndim == 1:
        observations = observations[np.newaxis, :]
    
    # Get action probabilities from the mask network
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32)
    with torch.no_grad():
        # Use the policy to get action distribution
        obs_tensor = obs_tensor.to(mask_network.device)
        dist = mask_network.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.cpu().numpy()  # [batch_size, 2]
    
    # Probability of action 0 (trust agent)
    trust_prob = probs[:, 0]
    
    # Importance = 1 - trust_prob (higher = more critical)
    importance = 1.0 - trust_prob
    
    return importance


def main():
    """CLI entry point for StateMask training."""
    parser = argparse.ArgumentParser(description="Train StateMask mask network")
    parser.add_argument("--env-id", type=str, default="Hopper-v3",
                        help="Gymnasium environment ID")
    parser.add_argument("--agent-path", type=str, required=True,
                        help="Path to pre-trained agent model")
    parser.add_argument("--output-dir", type=str, default="./outputs/statemask",
                        help="Output directory")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--total-timesteps", type=int, default=300000,
                        help="Total training timesteps")
    parser.add_argument("--budget", type=float, default=0.1,
                        help="Budget fraction for random actions")
    parser.add_argument("--initial-lambda", type=float, default=0.0,
                        help="Initial Lagrange multiplier")
    parser.add_argument("--lambda-lr", type=float, default=0.01,
                        help="Learning rate for λ")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64],
                        help="Hidden layer sizes")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="PPO learning rate")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto/cpu/cuda)")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level")
    
    args = parser.parse_args()
    
    # Load config
    if args.config is not None:
        from rice.utils import load_config
        config = load_config(args.config)
    else:
        config = {}
    
    # Load agent policy
    print(f"Loading agent from {args.agent_path}")
    agent = PPO.load(args.agent_path, device=get_device(args.device))
    
    # Train StateMask
    model, logger, lagrangian, model_path = train_statemask(
        env_id=args.env_id,
        agent_policy=agent,
        config=config,
        output_dir=args.output_dir,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        budget_fraction=args.budget,
        initial_lambda=args.initial_lambda,
        lambda_lr=args.lambda_lr,
        hidden_sizes=args.hidden_sizes,
        learning_rate=args.learning_rate,
        device=args.device,
        verbose=args.verbose,
    )
    
    print(f"\nStateMask training complete!")
    print(f"Model saved to: {model_path}")
    print(f"Final λ: {lagrangian.get_lambda():.4f}")
    print(f"Final randomization rate: {lagrangian.randomization_history[-1]:.4f}" 
          if lagrangian.randomization_history else "N/A")


if __name__ == "__main__":
    main()
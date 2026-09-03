"""
StateMask Baseline Implementation

StateMask (https://github.com/nuwuxian/RL-state_mask) learns a binary mask over
state features to identify which dimensions are critical for the policy. This serves
as an alternative explanation method to RICE's mask network.

The mask is trained by:
1. Applying a learned mask m ∈ [0,1]^d to the state: s_masked = s ⊙ m + s ⊙ (1-m) * noise
2. Minimizing the policy's performance drop when masking non-critical features
3. Encouraging sparsity via L1 regularization on the mask

This implementation adapts the original StateMask approach to work with the same
environments and target policies used in RICE, enabling fair comparison of:
- Fidelity (correlation between mask scores and Q-value differences)
- Training efficiency (wall-clock time)
- Refinement performance (when used as explanation for RICE pipeline)

Reference: "StateMask: Explaining Deep Reinforcement Learning through State Mask"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Callable
import os
import pickle
import json
import time
import argparse
from pathlib import Path
import gym
import yaml

# Try importing stable-baselines3 for loading target policies
try:
    from stable_baselines3 import PPO as SB3PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

# Import RICE utilities
from rice.utils import set_seed, evaluate_policy, to_tensor, to_numpy
from rice.env_wrappers import make_state_saveable, save_env_state, StateSaveWrapper


class StateMaskNetwork(nn.Module):
    """
    Learns a mask over state dimensions that identifies critical features.
    
    The mask m ∈ [0,1]^d is parameterized via a sigmoid over learnable logits.
    Higher mask values indicate more critical state dimensions.
    
    Architecture:
        - Mask logits: learnable parameter vector of size state_dim
        - The mask is obtained via sigmoid(logits)
        - Optional: a small MLP can predict mask from state (conditional mask)
    """
    
    def __init__(
        self,
        state_dim: int,
        conditional: bool = False,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation: str = "relu",
        init_mask_value: float = 0.5,
    ):
        """
        Args:
            state_dim: Dimension of the state space
            conditional: If True, mask is predicted from state via MLP.
                        If False, mask is a global learnable vector.
            hidden_sizes: Hidden layer sizes for conditional mask predictor
            activation: Activation function for conditional predictor
            init_mask_value: Initial value for mask (before sigmoid)
        """
        super().__init__()
        self.state_dim = state_dim
        self.conditional = conditional
        
        if conditional:
            # Conditional mask: MLP predicts mask from state
            layers = []
            prev_size = state_dim
            for h in hidden_sizes:
                layers.append(nn.Linear(prev_size, h))
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                prev_size = h
            layers.append(nn.Linear(prev_size, state_dim))
            self.mask_net = nn.Sequential(*layers)
        else:
            # Global mask: learnable parameter
            init_logit = np.log(init_mask_value / (1 - init_mask_value))  # inverse sigmoid
            self.mask_logits = nn.Parameter(
                torch.full((state_dim,), init_logit)
            )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute mask values for given state(s).
        
        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,)
            
        Returns:
            mask: Tensor of same shape as state, values in [0, 1]
        """
        if self.conditional:
            mask = torch.sigmoid(self.mask_net(state))
        else:
            mask = torch.sigmoid(self.mask_logits)
            if state.dim() == 2:
                mask = mask.unsqueeze(0).expand(state.shape[0], -1)
        return mask
    
    def get_importance_scores(self, state: np.ndarray) -> np.ndarray:
        """
        Get importance scores for a state (or batch of states).
        
        Args:
            state: numpy array of shape (state_dim,) or (batch_size, state_dim)
            
        Returns:
            importance_scores: numpy array of shape (state_dim,) or (batch_size, state_dim)
        """
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            mask = self.forward(state_tensor)
            if mask.shape[0] == 1:
                mask = mask.squeeze(0)
            return mask.cpu().numpy()
    
    def get_scalar_importance(self, state: np.ndarray) -> float:
        """
        Get a scalar importance score for a state (mean of mask values).
        
        Args:
            state: numpy array of shape (state_dim,)
            
        Returns:
            scalar_importance: float in [0, 1]
        """
        scores = self.get_importance_scores(state)
        return float(np.mean(scores))
    
    def apply_mask(
        self,
        state: np.ndarray,
        noise_scale: float = 0.0,
        hard_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply the learned mask to a state, optionally adding noise to masked features.
        
        Args:
            state: Original state (state_dim,)
            noise_scale: Standard deviation of Gaussian noise added to masked features
            hard_threshold: If provided, binarize mask at this threshold
            
        Returns:
            masked_state: State with mask applied
        """
        mask = self.get_importance_scores(state)
        
        if hard_threshold is not None:
            mask = (mask >= hard_threshold).astype(np.float32)
        
        if noise_scale > 0:
            noise = np.random.randn(*state.shape) * noise_scale
            # Add noise to non-critical (low mask) features
            masked_state = state * mask + (state + noise) * (1 - mask)
        else:
            # Zero out non-critical features
            masked_state = state * mask
        
        return masked_state
    
    def to(self, device: str):
        """Move model to device."""
        return super().to(device)
    
    def save(self, path: str):
        """Save model state dict."""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str, device: str = "cpu"):
        """Load model state dict."""
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)


class StateMaskTrainer:
    """
    Trains a StateMask network to identify critical state features.
    
    Training objective:
    1. Behavioral loss: minimize the difference in policy output when masking
       non-critical features vs. using the full state
    2. Sparsity loss: L1 regularization on mask values to encourage sparse masks
    
    The combined loss:
        L = L_behavioral + λ_sparsity * ||mask||_1
    
    where L_behavioral measures how much the policy's action distribution changes
    when non-critical features are masked.
    """
    
    def __init__(
        self,
        state_mask: StateMaskNetwork,
        target_policy: Callable[[np.ndarray], np.ndarray],
        state_dim: int,
        action_dim: int,
        sparsity_coef: float = 0.01,
        noise_scale: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        device: str = "cpu",
        conditional: bool = False,
    ):
        """
        Args:
            state_mask: StateMaskNetwork to train
            target_policy: Function that maps state -> action (deterministic)
            state_dim: State dimension
            action_dim: Action dimension
            sparsity_coef: Weight for L1 sparsity regularization
            noise_scale: Std of noise added to masked features during training
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            device: Device for computation
            conditional: Whether mask is conditional on state
        """
        self.state_mask = state_mask
        self.target_policy = target_policy
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sparsity_coef = sparsity_coef
        self.noise_scale = noise_scale
        self.batch_size = batch_size
        self.device = device
        self.conditional = conditional
        
        self.state_mask.to(device)
        self.optimizer = optim.Adam(
            self.state_mask.parameters(),
            lr=learning_rate
        )
        
        # Buffer for collected states
        self.state_buffer: List[np.ndarray] = []
        self.max_buffer_size = 100000
    
    def collect_states(
        self,
        env: gym.Env,
        num_steps: int = 10000,
        max_episode_steps: int = 1000,
        verbose: bool = False,
    ):
        """
        Collect states by running the target policy in the environment.
        
        Args:
            env: Gym environment
            num_steps: Total number of steps to collect
            max_episode_steps: Maximum steps per episode
            verbose: Print progress
        """
        collected = 0
        episodes = 0
        
        while collected < num_steps:
            obs, _ = env.reset()
            episode_steps = 0
            
            while episode_steps < max_episode_steps and collected < num_steps:
                action = self.target_policy(obs)
                if isinstance(action, tuple):
                    action = action[0]  # Handle (action, ...) tuples
                
                self.state_buffer.append(obs.copy())
                collected += 1
                
                step_result = env.step(action)
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                    terminated, truncated = done, False
                else:
                    obs, reward, terminated, truncated, info = step_result
                
                episode_steps += 1
                
                if terminated or truncated:
                    break
            
            episodes += 1
            
            if verbose and episodes % 10 == 0:
                print(f"  Collected {collected}/{num_steps} states ({episodes} episodes)")
        
        # Trim buffer if needed
        if len(self.state_buffer) > self.max_buffer_size:
            self.state_buffer = self.state_buffer[-self.max_buffer_size:]
    
    def train_step(self, states: np.ndarray) -> Dict[str, float]:
        """
        Perform one training step on a batch of states.
        
        Args:
            states: Batch of states (batch_size, state_dim)
            
        Returns:
            metrics: Dictionary of loss values
        """
        self.state_mask.train()
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        
        # Get mask
        mask = self.state_mask(states_tensor)  # (batch_size, state_dim)
        
        # Get target actions for original states
        with torch.no_grad():
            original_actions = []
            for i in range(states.shape[0]):
                action = self.target_policy(states[i])
                if isinstance(action, tuple):
                    action = action[0]
                original_actions.append(action)
            original_actions = np.array(original_actions)
            original_actions_tensor = torch.FloatTensor(original_actions).to(self.device)
        
        # Create masked states: s_masked = s * mask + s * (1-mask) * noise
        noise = torch.randn_like(states_tensor) * self.noise_scale
        masked_states = states_tensor * mask + (states_tensor + noise) * (1 - mask)
        
        # Get target actions for masked states
        masked_actions_list = []
        masked_states_np = masked_states.detach().cpu().numpy()
        for i in range(masked_states_np.shape[0]):
            action = self.target_policy(masked_states_np[i])
            if isinstance(action, tuple):
                action = action[0]
            masked_actions_list.append(action)
        masked_actions_tensor = torch.FloatTensor(np.array(masked_actions_list)).to(self.device)
        
        # Behavioral loss: MSE between original and masked policy actions
        behavioral_loss = F.mse_loss(masked_actions_tensor, original_actions_tensor)
        
        # Sparsity loss: L1 norm of mask (encourage sparsity)
        sparsity_loss = mask.mean()  # Average mask value; minimize to encourage zeros
        
        # Total loss
        total_loss = behavioral_loss + self.sparsity_coef * sparsity_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "behavioral_loss": behavioral_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "mean_mask": mask.mean().item(),
        }
    
    def train(
        self,
        env: gym.Env,
        total_steps: int = 50000,
        collect_interval: int = 1000,
        train_steps_per_collect: int = 100,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """
        Full training loop: alternate between collecting states and training.
        
        Args:
            env: Gym environment
            total_steps: Total training steps
            collect_interval: Collect new states every N training steps
            train_steps_per_collect: Number of training steps per collection
            verbose: Print progress
            
        Returns:
            history: List of training metrics
        """
        history = []
        steps_done = 0
        collect_count = 0
        
        # Initial collection
        if verbose:
            print("Initial state collection...")
        self.collect_states(env, num_steps=5000, verbose=verbose)
        
        while steps_done < total_steps:
            # Collect more states periodically
            if steps_done % collect_interval == 0 and steps_done > 0:
                self.collect_states(env, num_steps=2000, verbose=False)
                collect_count += 1
            
            # Train on batches from buffer
            if len(self.state_buffer) < self.batch_size:
                self.collect_states(env, num_steps=self.batch_size, verbose=False)
            
            for _ in range(train_steps_per_collect):
                if steps_done >= total_steps:
                    break
                
                # Sample batch
                indices = np.random.choice(len(self.state_buffer), self.batch_size, replace=False)
                batch_states = np.stack([self.state_buffer[i] for i in indices])
                
                metrics = self.train_step(batch_states)
                metrics["step"] = steps_done
                history.append(metrics)
                steps_done += 1
            
            if verbose and steps_done % 500 == 0:
                recent = history[-1] if history else {}
                print(f"  Step {steps_done}/{total_steps}: "
                      f"loss={recent.get('total_loss', 0):.4f}, "
                      f"mean_mask={recent.get('mean_mask', 0):.4f}, "
                      f"buffer={len(self.state_buffer)}")
        
        return history
    
    def save(self, path: str):
        """Save trainer state."""
        self.state_mask.save(path)
    
    def load(self, path: str):
        """Load trainer state."""
        self.state_mask.load(path, self.device)


def compute_statemask_fidelity(
    state_mask: StateMaskNetwork,
    states: np.ndarray,
    q_values: np.ndarray,
    device: str = "cpu",
) -> float:
    """
    Compute fidelity as Pearson correlation between mask importance scores
    and Q-value differences.
    
    For StateMask, the importance score for a state is the mean mask value.
    Q_diff = Q(s, a*) - E_a[Q(s, a)] measures how critical a state is.
    
    Args:
        state_mask: Trained StateMaskNetwork
        states: Array of states (N, state_dim)
        q_values: Array of Q-values (N, num_actions) or Q-differences (N,)
        device: Computation device
        
    Returns:
        fidelity: Pearson correlation coefficient
    """
    state_mask.eval()
    
    # Get importance scores
    importance_scores = []
    for state in states:
        score = state_mask.get_scalar_importance(state)
        importance_scores.append(score)
    importance_scores = np.array(importance_scores)
    
    # Compute Q-differences if full Q-values provided
    if q_values.ndim == 2:
        q_max = np.max(q_values, axis=1)
        q_mean = np.mean(q_values, axis=1)
        q_diffs = q_max - q_mean
    else:
        q_diffs = q_values
    
    # Pearson correlation
    if np.std(importance_scores) > 0 and np.std(q_diffs) > 0:
        correlation = np.corrcoef(importance_scores, q_diffs)[0, 1]
    else:
        correlation = 0.0
    
    return float(correlation)


def compute_statemask_fidelity_from_env(
    state_mask: StateMaskNetwork,
    env: gym.Env,
    target_policy: Callable[[np.ndarray], np.ndarray],
    num_episodes: int = 10,
    max_episode_steps: int = 1000,
    q_function: Optional[Callable] = None,
    device: str = "cpu",
) -> float:
    """
    Estimate fidelity by interacting with the environment.
    
    Uses Monte Carlo returns as proxy for Q-values if no Q-function provided.
    
    Args:
        state_mask: Trained StateMaskNetwork
        env: Gym environment
        target_policy: Target policy function
        num_episodes: Number of episodes to evaluate
        max_episode_steps: Maximum steps per episode
        q_function: Optional Q-function for computing Q-differences
        device: Computation device
        
    Returns:
        fidelity: Pearson correlation coefficient
    """
    all_states = []
    all_q_diffs = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_states = []
        episode_rewards = []
        episode_actions = []
        
        for step in range(max_episode_steps):
            action = target_policy(obs)
            if isinstance(action, tuple):
                action = action[0]
            
            episode_states.append(obs.copy())
            episode_actions.append(action)
            
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
                terminated, truncated = done, False
            else:
                obs, reward, terminated, truncated, info = step_result
            
            episode_rewards.append(reward)
            
            if terminated or truncated:
                break
        
        # Compute Monte Carlo returns as Q-value proxy
        returns = []
        cumulative = 0.0
        for r in reversed(episode_rewards):
            cumulative = r + 0.99 * cumulative
            returns.append(cumulative)
        returns = list(reversed(returns))
        
        # For each state, compute Q-diff as deviation from mean return in episode
        mean_return = np.mean(returns) if returns else 0.0
        for i, state in enumerate(episode_states):
            all_states.append(state)
            if q_function is not None:
                q_vals = q_function(state)
                q_diff = np.max(q_vals) - np.mean(q_vals)
            else:
                # Use deviation from mean as proxy
                q_diff = abs(returns[i] - mean_return)
            all_q_diffs.append(q_diff)
    
    all_states = np.array(all_states)
    all_q_diffs = np.array(all_q_diffs)
    
    return compute_statemask_fidelity(state_mask, all_states, all_q_diffs, device)


def run_statemask_baseline(
    env_name: str,
    model_dir: str,
    output_dir: str,
    config_path: Optional[str] = None,
    total_steps: int = 50000,
    sparsity_coef: float = 0.01,
    noise_scale: float = 0.1,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
    conditional: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full StateMask baseline pipeline:
    1. Load target policy
    2. Train StateMask network
    3. Compute fidelity
    4. Save results
    
    Args:
        env_name: Gym environment name
        model_dir: Directory containing pre-trained target policy
        output_dir: Directory to save results
        config_path: Optional path to YAML config
        total_steps: Total training steps for StateMask
        sparsity_coef: Sparsity regularization coefficient
        noise_scale: Noise scale for masked features
        learning_rate: Learning rate
        seed: Random seed
        device: Computation device
        conditional: Whether to use conditional mask
        verbose: Print progress
        
    Returns:
        results: Dictionary with trained model, fidelity, training time
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config if provided
    config = {}
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Create environment
    env = gym.make(env_name)
    env = make_state_saveable(env)
    
    # Get state and action dimensions
    if hasattr(env, 'observation_space'):
        state_dim = env.observation_space.shape[0]
    else:
        state_dim = env.observation_space.shape[0]
    
    if hasattr(env, 'action_space'):
        if hasattr(env.action_space, 'shape'):
            action_dim = env.action_space.shape[0]
        else:
            action_dim = env.action_space.n
    else:
        action_dim = env.action_space.shape[0]
    
    # Load target policy
    if verbose:
        print(f"Loading target policy from {model_dir}...")
    
    target_policy = None
    vec_normalize = None
    
    # Try loading SB3 PPO model
    if HAS_SB3:
        model_path = os.path.join(model_dir, f"{env_name}_ppo_final.zip")
        if os.path.exists(model_path):
            model = SB3PPO.load(model_path, device=device)
            
            # Load VecNormalize if available
            norm_path = os.path.join(model_dir, f"{env_name}_vecnormalize.pkl")
            if os.path.exists(norm_path):
                with open(norm_path, 'rb') as f:
                    vec_normalize = pickle.load(f)
            
            def target_policy_fn(obs):
                if vec_normalize is not None:
                    obs = vec_normalize.normalize_obs(obs)
                action, _ = model.predict(obs, deterministic=True)
                return action
            
            target_policy = target_policy_fn
    
    if target_policy is None:
        # Try loading PyTorch state dict
        policy_path = os.path.join(model_dir, f"{env_name}_policy.pt")
        if os.path.exists(policy_path):
            # Create a simple MLP policy
            policy_net = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
            )
            policy_net.load_state_dict(torch.load(policy_path, map_location=device))
            policy_net.to(device)
            policy_net.eval()
            
            def target_policy_fn(obs):
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action = policy_net(obs_tensor).squeeze(0).cpu().numpy()
                return action
            
            target_policy = target_policy_fn
    
    if target_policy is None:
        raise FileNotFoundError(f"Could not find target policy in {model_dir}")
    
    # Create StateMask network
    state_mask = StateMaskNetwork(
        state_dim=state_dim,
        conditional=conditional,
        hidden_sizes=(64, 64),
        activation="relu",
        init_mask_value=0.5,
    )
    
    # Create trainer
    trainer = StateMaskTrainer(
        state_mask=state_mask,
        target_policy=target_policy,
        state_dim=state_dim,
        action_dim=action_dim,
        sparsity_coef=sparsity_coef,
        noise_scale=noise_scale,
        learning_rate=learning_rate,
        batch_size=256,
        device=device,
        conditional=conditional,
    )
    
    # Train
    if verbose:
        print(f"Training StateMask for {total_steps} steps...")
    
    start_time = time.time()
    history = trainer.train(
        env=env,
        total_steps=total_steps,
        collect_interval=1000,
        train_steps_per_collect=100,
        verbose=verbose,
    )
    training_time = time.time() - start_time
    
    # Compute fidelity
    if verbose:
        print("Computing fidelity...")
    
    fidelity = compute_statemask_fidelity_from_env(
        state_mask=state_mask,
        env=env,
        target_policy=target_policy,
        num_episodes=10,
        device=device,
    )
    
    # Save results
    model_path = os.path.join(output_dir, f"{env_name}_statemask.pt")
    state_mask.save(model_path)
    
    results = {
        "env_name": env_name,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "conditional": conditional,
        "sparsity_coef": sparsity_coef,
        "noise_scale": noise_scale,
        "total_steps": total_steps,
        "training_time": training_time,
        "fidelity": fidelity,
        "final_mean_mask": history[-1]["mean_mask"] if history else 0.0,
        "history": history,
    }
    
    results_path = os.path.join(output_dir, f"{env_name}_statemask_results.json")
    # Convert history to serializable format
    results_to_save = {k: v for k, v in results.items() if k != "history"}
    results_to_save["history"] = [
        {k2: float(v2) if isinstance(v2, (np.floating, np.integer)) else v2
         for k2, v2 in h.items()}
        for h in history
    ]
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    if verbose:
        print(f"\nStateMask Results for {env_name}:")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Fidelity: {fidelity:.4f}")
        print(f"  Final mean mask: {results['final_mean_mask']:.4f}")
        print(f"  Results saved to {output_dir}")
    
    env.close()
    
    return results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="StateMask Baseline for RICE"
    )
    parser.add_argument(
        "--env", type=str, default="Hopper-v4",
        help="Gym environment name"
    )
    parser.add_argument(
        "--model-dir", type=str, default="./trained_agents",
        help="Directory containing pre-trained target policy"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./baseline_results/statemask",
        help="Directory to save results"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--total-steps", type=int, default=50000,
        help="Total training steps for StateMask"
    )
    parser.add_argument(
        "--sparsity-coef", type=float, default=0.01,
        help="Sparsity regularization coefficient"
    )
    parser.add_argument(
        "--noise-scale", type=float, default=0.1,
        help="Noise scale for masked features"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Computation device (cpu or cuda)"
    )
    parser.add_argument(
        "--conditional", action="store_true",
        help="Use conditional mask (state-dependent)"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print progress"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    results = run_statemask_baseline(
        env_name=args.env,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        total_steps=args.total_steps,
        sparsity_coef=args.sparsity_coef,
        noise_scale=args.noise_scale,
        learning_rate=args.lr,
        seed=args.seed,
        device=args.device,
        conditional=args.conditional,
        verbose=args.verbose,
    )
    
    return results


if __name__ == "__main__":
    main()
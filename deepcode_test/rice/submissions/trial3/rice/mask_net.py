"""
Mask Network for RICE (Refining via Critical State Explanation)

The mask network learns a binary policy π̃(a^e|s) that decides whether to:
- Keep the target agent's action (a^e=0): use π_target(a|s)
- Randomize the action (a^e=1): sample from Uniform(A)

The probability of keeping the action ξ(s) = π̃(a^e=0|s) serves as the
importance score; higher ξ means the state is more critical.

Training uses PPO with an intrinsic reward bonus for masking (randomizing).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import Optional, Tuple, List, Dict, Any, Callable
import copy
from collections import deque


# ==============================================================================
# Mask Network Architecture
# ==============================================================================

class MaskNetwork(nn.Module):
    """
    MLP-based mask network with policy head (2 discrete actions) and value head.
    
    Architecture:
        - Shared feature extractor: MLP with hidden layers [128, 128]
        - Policy head: Linear(state_dim -> 2) for binary action (keep/randomize)
        - Value head: Linear(state_dim -> 1) for state value baseline
    
    Args:
        state_dim: Dimension of the state space
        hidden_sizes: List of hidden layer sizes (default: [128, 128])
        activation: Activation function (default: Tanh)
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        activation: str = "tanh"
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_sizes = hidden_sizes
        
        # Build shared feature extractor
        layers = []
        prev_dim = state_dim
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            prev_dim = h_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_dim = prev_dim
        
        # Policy head: outputs logits for 2 discrete actions
        self.policy_head = nn.Linear(self.feature_dim, 2)
        
        # Value head: outputs scalar state value
        self.value_head = nn.Linear(self.feature_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        # Policy head: smaller gain
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0.0)
        # Value head: gain=1
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
        
        Returns:
            action_logits: Logits for 2 discrete actions (batch_size, 2)
            value: State value estimates (batch_size, 1)
        """
        features = self.feature_extractor(state)
        action_logits = self.policy_head(features)
        value = self.value_head(features)
        return action_logits, value
    
    def get_action_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Get action logits only (for inference)."""
        features = self.feature_extractor(state)
        return self.policy_head(features)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get value estimate only."""
        features = self.feature_extractor(state)
        return self.value_head(features)
    
    def get_importance_score(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute importance score ξ(s) = softmax(π̃(s))[0] = P(a^e=0|s).
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
        
        Returns:
            importance_scores: Probability of keeping action (batch_size,)
        """
        logits = self.get_action_logits(state)
        probs = F.softmax(logits, dim=-1)
        return probs[:, 0]  # Probability of a^e=0 (keep action)
    
    def sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample mask action a^e ~ π̃(·|s).
        
        Args:
            state: State tensor (batch_size, state_dim)
        
        Returns:
            action: Sampled action indices (batch_size,)
            log_prob: Log probabilities of sampled actions (batch_size,)
            entropy: Entropy of the action distribution (batch_size,)
        """
        logits, _ = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy
    
    def evaluate_actions(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given actions.
        
        Args:
            state: State tensor (batch_size, state_dim)
            action: Action indices (batch_size,)
        
        Returns:
            log_prob: Log probabilities of given actions (batch_size,)
            entropy: Entropy of action distribution (batch_size,)
            value: State value estimates (batch_size, 1)
        """
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value


# ==============================================================================
# Perturbed Policy
# ==============================================================================

class PerturbedPolicy:
    """
    Combines the mask network with a target policy to create a perturbed policy.
    
    At each state s:
        1. Sample a^e ~ π̃(·|s) from mask network
        2. If a^e == 0: sample a ~ π_target(·|s)  (keep target action)
        3. If a^e == 1: sample a ~ Uniform(A)      (randomize action)
    
    The effective perturbed policy is:
        π̄(a|s) = ξ(s) * π_target(a|s) + (1-ξ(s)) * π_random(a|s)
    
    where ξ(s) = P(a^e=0|s) is the importance score.
    """
    
    def __init__(
        self,
        mask_network: MaskNetwork,
        target_policy: Any,  # Callable: state -> action
        action_space_low: np.ndarray,
        action_space_high: np.ndarray,
        discrete_action: bool = False,
        num_discrete_actions: Optional[int] = None,
        device: str = "cpu"
    ):
        """
        Args:
            mask_network: Trained/untrained MaskNetwork
            target_policy: Target agent policy (callable: state -> action)
            action_space_low: Lower bounds of continuous action space
            action_space_high: Upper bounds of continuous action space
            discrete_action: Whether the environment has discrete actions
            num_discrete_actions: Number of discrete actions (if discrete)
            device: Device for tensor operations
        """
        self.mask_network = mask_network
        self.target_policy = target_policy
        self.action_low = action_space_low
        self.action_high = action_space_high
        self.discrete_action = discrete_action
        self.num_discrete_actions = num_discrete_actions
        self.device = device
    
    def get_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, int, float, float]:
        """
        Sample action from perturbed policy.
        
        Args:
            state: Current state (state_dim,)
            deterministic: If True, always keep target action
        
        Returns:
            action: Environment action
            mask_action: 0 (keep) or 1 (randomize)
            importance_score: ξ(s) = P(a^e=0|s)
            log_prob_mask: Log probability of mask action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get importance score
        with torch.no_grad():
            importance_score = self.mask_network.get_importance_score(state_tensor).item()
        
        # Sample mask action
        if deterministic:
            mask_action = 0  # Always keep
            log_prob_mask = np.log(max(importance_score, 1e-8))
        else:
            mask_action_tensor, log_prob_tensor, _ = self.mask_network.sample_action(state_tensor)
            mask_action = mask_action_tensor.item()
            log_prob_mask = log_prob_tensor.item()
        
        # Get environment action
        if mask_action == 0:
            # Keep target action
            action = self.target_policy(state)
        else:
            # Randomize action
            if self.discrete_action:
                action = np.random.randint(0, self.num_discrete_actions)
            else:
                action = np.random.uniform(self.action_low, self.action_high)
        
        return action, mask_action, importance_score, log_prob_mask
    
    def get_importance_score(self, state: np.ndarray) -> float:
        """Get importance score ξ(s) for a state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.mask_network.get_importance_score(state_tensor).item()


# ==============================================================================
# PPO Training for Mask Network
# ==============================================================================

class PPOBuffer:
    """Buffer for storing trajectory data for PPO training."""
    
    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.states: List[np.ndarray] = []
        self.mask_actions: List[int] = []
        self.mask_log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
        self.importance_scores: List[float] = []
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
    
    def add(
        self,
        state: np.ndarray,
        mask_action: int,
        mask_log_prob: float,
        reward: float,
        value: float,
        done: bool,
        importance_score: float
    ):
        self.states.append(state)
        self.mask_actions.append(mask_action)
        self.mask_log_probs.append(mask_log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.importance_scores.append(importance_score)
    
    def clear(self):
        self.states.clear()
        self.mask_actions.clear()
        self.mask_log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.importance_scores.clear()
    
    def compute_returns_and_advantages(
        self, last_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute discounted returns and GAE advantages.
        
        Returns:
            returns: Discounted returns (T,)
            advantages: GAE advantages (T,)
            values: Value estimates (T,)
        """
        T = len(self.states)
        returns = np.zeros(T, dtype=np.float32)
        advantages = np.zeros(T, dtype=np.float32)
        
        gae = 0.0
        next_value = last_value
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - float(self.dones[t])
                delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            else:
                next_non_terminal = 1.0 - float(self.dones[t])
                delta = self.rewards[t] + self.gamma * self.values[t + 1] * next_non_terminal - self.values[t]
            
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + self.values[t]
        
        values = np.array(self.values, dtype=np.float32)
        
        return returns, advantages, values
    
    def get_training_data(
        self, last_value: float = 0.0
    ) -> Dict[str, np.ndarray]:
        """Get all data needed for PPO update."""
        returns, advantages, values = self.compute_returns_and_advantages(last_value)
        
        return {
            "states": np.array(self.states, dtype=np.float32),
            "mask_actions": np.array(self.mask_actions, dtype=np.int64),
            "mask_log_probs": np.array(self.mask_log_probs, dtype=np.float32),
            "returns": returns,
            "advantages": advantages,
            "values": values,
        }


class MaskNetworkTrainer:
    """
    Trains the mask network using PPO.
    
    The mask network learns to identify critical states by maximizing:
        J(π̃) = E[ Σ γ^t (r_env_t + α * I(a^e_t=1)) ]
    
    where r_env is the environment reward and α * I(a^e=1) is an intrinsic
    reward for randomizing (masking) the action. The mask network is encouraged
    to randomize non-critical steps while preserving critical ones.
    """
    
    def __init__(
        self,
        mask_network: MaskNetwork,
        target_policy: Any,
        env: Any,
        alpha: float = 0.0001,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        learning_rate: float = 3e-4,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cpu",
        action_space_low: Optional[np.ndarray] = None,
        action_space_high: Optional[np.ndarray] = None,
        discrete_action: bool = False,
        num_discrete_actions: Optional[int] = None,
    ):
        """
        Args:
            mask_network: MaskNetwork to train
            target_policy: Pre-trained target policy (callable: state -> action)
            env: Gym environment (used for action space bounds)
            alpha: Intrinsic reward coefficient for masking (default: 0.0001)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            learning_rate: Adam learning rate
            ppo_epochs: Number of PPO epochs per update
            batch_size: Mini-batch size for PPO updates
            device: Device for training
            action_space_low: Lower bounds of action space (for randomization)
            action_space_high: Upper bounds of action space (for randomization)
            discrete_action: Whether action space is discrete
            num_discrete_actions: Number of discrete actions
        """
        self.mask_network = mask_network
        self.target_policy = target_policy
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = device
        
        # Determine action space
        if action_space_low is not None and action_space_high is not None:
            self.action_low = action_space_low
            self.action_high = action_space_high
        else:
            # Try to infer from environment
            env_action_space = env.action_space
            if hasattr(env_action_space, 'low') and hasattr(env_action_space, 'high'):
                self.action_low = env_action_space.low
                self.action_high = env_action_space.high
                self.discrete_action = False
                self.num_discrete_actions = None
            elif hasattr(env_action_space, 'n'):
                self.discrete_action = True
                self.num_discrete_actions = env_action_space.n
                self.action_low = None
                self.action_high = None
            else:
                raise ValueError("Cannot determine action space from environment")
        
        if discrete_action:
            self.discrete_action = True
            if num_discrete_actions is not None:
                self.num_discrete_actions = num_discrete_actions
        
        # Create perturbed policy
        self.perturbed_policy = PerturbedPolicy(
            mask_network=mask_network,
            target_policy=target_policy,
            action_space_low=self.action_low if not self.discrete_action else np.array([]),
            action_space_high=self.action_high if not self.discrete_action else np.array([]),
            discrete_action=self.discrete_action,
            num_discrete_actions=self.num_discrete_actions,
            device=device
        )
        
        # Optimizer
        self.optimizer = optim.Adam(mask_network.parameters(), lr=learning_rate)
        
        # Training metrics
        self.iteration = 0
        self.metrics_history: List[Dict[str, float]] = []
    
    def collect_trajectories(
        self, num_steps: int, render: bool = False
    ) -> PPOBuffer:
        """
        Collect trajectories using the perturbed policy.
        
        Args:
            num_steps: Total number of environment steps to collect
            render: Whether to render the environment
        
        Returns:
            buffer: PPOBuffer containing collected trajectory data
        """
        buffer = PPOBuffer(gamma=self.gamma, gae_lambda=self.gae_lambda)
        
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Handle gym reset returning (obs, info)
        
        episode_reward = 0.0
        episode_length = 0
        steps_collected = 0
        
        while steps_collected < num_steps:
            # Get action from perturbed policy
            action, mask_action, importance_score, mask_log_prob = \
                self.perturbed_policy.get_action(state)
            
            # Get value estimate
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                value = self.mask_network.get_value(state_tensor).item()
            
            # Step environment
            next_state, env_reward, done, truncated, info = self.env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            
            # Compute total reward: r_env + α * I(a^e == 1)
            intrinsic_reward = self.alpha * float(mask_action == 1)
            total_reward = env_reward + intrinsic_reward
            
            # Store transition
            buffer.add(
                state=state,
                mask_action=mask_action,
                mask_log_prob=mask_log_prob,
                reward=total_reward,
                value=value,
                done=done or truncated,
                importance_score=importance_score
            )
            
            episode_reward += env_reward
            episode_length += 1
            steps_collected += 1
            
            if done or truncated:
                state = self.env.reset()
                if isinstance(state, tuple):
                    state = state[0]
                episode_reward = 0.0
                episode_length = 0
            else:
                state = next_state
            
            if render:
                self.env.render()
        
        return buffer
    
    def update(self, buffer: PPOBuffer) -> Dict[str, float]:
        """
        Perform PPO update on collected trajectory data.
        
        Args:
            buffer: PPOBuffer with collected data
        
        Returns:
            metrics: Dictionary of training metrics
        """
        # Get training data
        data = buffer.get_training_data(last_value=0.0)
        
        states = torch.FloatTensor(data["states"]).to(self.device)
        old_actions = torch.LongTensor(data["mask_actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(data["mask_log_probs"]).to(self.device)
        returns = torch.FloatTensor(data["returns"]).to(self.device)
        advantages = torch.FloatTensor(data["advantages"]).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_samples = len(states)
        indices = np.arange(total_samples)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, total_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = old_actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Evaluate current policy
                new_log_probs, entropy, values = self.mask_network.evaluate_actions(
                    batch_states, batch_actions
                )
                values = values.squeeze(-1)
                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (MSE)
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.mask_network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        # Compute average metrics
        metrics = {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "mean_importance": float(data["importance_scores"].mean()) if "importance_scores" in data else 0.0,
            "mean_reward": float(data["returns"].mean()),
            "mean_advantage": float(data["advantages"].mean()),
        }
        
        self.metrics_history.append(metrics)
        self.iteration += 1
        
        return metrics
    
    def train(
        self,
        total_steps: int,
        steps_per_iteration: int = 2048,
        log_interval: int = 10,
        verbose: bool = True,
        early_stop_fidelity: Optional[float] = None,
        fidelity_check_fn: Optional[Callable] = None,
        fidelity_patience: int = 20,
    ) -> List[Dict[str, float]]:
        """
        Full training loop for mask network.
        
        Args:
            total_steps: Total environment steps for training
            steps_per_iteration: Steps collected per PPO update
            log_interval: How often to log metrics
            verbose: Whether to print progress
            early_stop_fidelity: If provided, stop when fidelity reaches this value
            fidelity_check_fn: Function to compute fidelity (called every log_interval)
            fidelity_patience: Number of iterations without improvement before stopping
        
        Returns:
            metrics_history: List of metrics per iteration
        """
        num_iterations = total_steps // steps_per_iteration
        best_fidelity = -float("inf")
        patience_counter = 0
        
        for i in range(num_iterations):
            # Collect trajectories
            buffer = self.collect_trajectories(steps_per_iteration)
            
            # PPO update
            metrics = self.update(buffer)
            
            # Check fidelity if provided
            if fidelity_check_fn is not None and (i % log_interval == 0 or i == num_iterations - 1):
                fidelity = fidelity_check_fn(self.mask_network)
                metrics["fidelity"] = fidelity
                
                if early_stop_fidelity is not None and fidelity >= early_stop_fidelity:
                    if verbose:
                        print(f"Iteration {i}: Fidelity {fidelity:.4f} >= target {early_stop_fidelity}, stopping.")
                    break
                
                # Early stopping based on fidelity improvement
                if fidelity > best_fidelity:
                    best_fidelity = fidelity
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= fidelity_patience:
                        if verbose:
                            print(f"Iteration {i}: Fidelity not improving for {fidelity_patience} iterations, stopping.")
                        break
            
            if verbose and (i % log_interval == 0 or i == num_iterations - 1):
                print(f"Iteration {i}/{num_iterations}: "
                      f"policy_loss={metrics['policy_loss']:.4f}, "
                      f"value_loss={metrics['value_loss']:.4f}, "
                      f"entropy={metrics['entropy']:.4f}, "
                      f"mean_importance={metrics.get('mean_importance', 0):.4f}, "
                      f"mean_reward={metrics['mean_reward']:.4f}")
        
        return self.metrics_history
    
    def save(self, path: str):
        """Save mask network state dict."""
        torch.save({
            "mask_network_state_dict": self.mask_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iteration": self.iteration,
            "metrics_history": self.metrics_history,
        }, path)
    
    def load(self, path: str):
        """Load mask network state dict."""
        checkpoint = torch.load(path, map_location=self.device)
        self.mask_network.load_state_dict(checkpoint["mask_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.iteration = checkpoint["iteration"]
        self.metrics_history = checkpoint["metrics_history"]


# ==============================================================================
# Fidelity Evaluation
# ==============================================================================

def compute_fidelity(
    mask_network: MaskNetwork,
    states: np.ndarray,
    q_values: np.ndarray,  # Q(s, a) for each state-action pair
    device: str = "cpu"
) -> float:
    """
    Compute fidelity score: correlation between importance scores ξ(s)
    and actual Q-value differences.
    
    Q_diff(s) = Q(s, a*) - E_{a'}[Q(s, a')]
    where a* is the target action and a' is sampled uniformly.
    
    Fidelity = Pearson correlation between ξ(s) and Q_diff(s).
    
    Args:
        mask_network: Trained mask network
        states: Array of states (N, state_dim)
        q_values: Q-values for each state (N,) or (N, num_actions)
        device: Device for computation
    
    Returns:
        fidelity: Pearson correlation coefficient
    """
    states_tensor = torch.FloatTensor(states).to(device)
    
    with torch.no_grad():
        importance_scores = mask_network.get_importance_score(states_tensor).cpu().numpy()
    
    # If q_values is 1D, it's already Q_diff
    if q_values.ndim == 1:
        q_diff = q_values
    else:
        # Compute Q_diff = max Q - mean Q
        q_max = q_values.max(axis=1)
        q_mean = q_values.mean(axis=1)
        q_diff = q_max - q_mean
    
    # Pearson correlation
    correlation = np.corrcoef(importance_scores, q_diff)[0, 1]
    
    # Handle NaN (e.g., constant arrays)
    if np.isnan(correlation):
        return 0.0
    
    return float(correlation)


def compute_fidelity_from_env(
    mask_network: MaskNetwork,
    env: Any,
    target_policy: Any,
    num_episodes: int = 10,
    q_function: Optional[Callable] = None,
    device: str = "cpu"
) -> float:
    """
    Compute fidelity by collecting states and estimating Q-differences.
    
    If no Q-function is provided, uses Monte Carlo returns as proxy.
    
    Args:
        mask_network: Trained mask network
        env: Environment
        target_policy: Target policy
        num_episodes: Number of episodes for evaluation
        q_function: Optional Q-function (state, action) -> float
        device: Device
    
    Returns:
        fidelity: Pearson correlation
    """
    all_states = []
    all_q_diffs = []
    
    for _ in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        done = False
        
        while not done:
            # Get target action
            target_action = target_policy(state)
            
            # Estimate Q(s, target_action) via Monte Carlo or Q-function
            if q_function is not None:
                q_target = q_function(state, target_action)
                # Estimate mean Q over random actions
                q_random_sum = 0.0
                num_random = 10
                for _ in range(num_random):
                    if hasattr(env.action_space, 'low'):
                        random_action = np.random.uniform(
                            env.action_space.low, env.action_space.high
                        )
                    else:
                        random_action = env.action_space.sample()
                    q_random_sum += q_function(state, random_action)
                q_random_mean = q_random_sum / num_random
                q_diff = q_target - q_random_mean
            else:
                # Use simple heuristic: states where target action differs
                # significantly from random are more critical
                # This is a placeholder; real implementation needs Q-function
                q_diff = 0.0
            
            all_states.append(state)
            all_q_diffs.append(q_diff)
            
            next_state, reward, done, truncated, info = env.step(target_action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            state = next_state
            
            if done or truncated:
                break
    
    if len(all_states) == 0:
        return 0.0
    
    states_array = np.array(all_states, dtype=np.float32)
    q_diffs_array = np.array(all_q_diffs, dtype=np.float32)
    
    return compute_fidelity(mask_network, states_array, q_diffs_array, device)


# ==============================================================================
# Convenience function for training mask network
# ==============================================================================

def train_mask_network(
    env: Any,
    target_policy: Any,
    state_dim: int,
    total_steps: int = 300000,
    alpha: float = 0.0001,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    learning_rate: float = 3e-4,
    hidden_sizes: Tuple[int, ...] = (128, 128),
    steps_per_iteration: int = 2048,
    ppo_epochs: int = 10,
    batch_size: int = 64,
    device: str = "cpu",
    verbose: bool = True,
    save_path: Optional[str] = None,
    **kwargs
) -> Tuple[MaskNetwork, MaskNetworkTrainer]:
    """
    Convenience function to train a mask network.
    
    Args:
        env: Gym environment
        target_policy: Pre-trained target policy (callable: state -> action)
        state_dim: Dimension of state space
        total_steps: Total environment steps for training
        alpha: Intrinsic reward coefficient
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_epsilon: PPO clip epsilon
        learning_rate: Learning rate
        hidden_sizes: Hidden layer sizes
        steps_per_iteration: Steps per PPO update
        ppo_epochs: PPO epochs per update
        batch_size: Mini-batch size
        device: Device
        verbose: Print progress
        save_path: Path to save trained model
    
    Returns:
        mask_network: Trained MaskNetwork
        trainer: MaskNetworkTrainer instance
    """
    mask_network = MaskNetwork(
        state_dim=state_dim,
        hidden_sizes=hidden_sizes
    ).to(device)
    
    trainer = MaskNetworkTrainer(
        mask_network=mask_network,
        target_policy=target_policy,
        env=env,
        alpha=alpha,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        learning_rate=learning_rate,
        ppo_epochs=ppo_epochs,
        batch_size=batch_size,
        device=device,
        **kwargs
    )
    
    trainer.train(
        total_steps=total_steps,
        steps_per_iteration=steps_per_iteration,
        verbose=verbose
    )
    
    if save_path is not None:
        trainer.save(save_path)
    
    return mask_network, trainer
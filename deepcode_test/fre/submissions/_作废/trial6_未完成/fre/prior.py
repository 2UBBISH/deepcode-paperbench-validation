"""
Reward Prior Distributions for FRE (Functional Reward Encodings).

Implements three families of unsupervised reward functions for pre-training:
  1. Singleton Goal-Reaching Rewards
  2. Random Linear Functions (with sparse masking)
  3. Random MLPs (2-layer ReLU networks)

Plus a MixedPrior manager that samples uniformly from all three families.

Reference: "Functional Reward Encodings (FRE) for Zero-Shot Offline RL"
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple, Dict, Any, List
from dataclasses import dataclass


# ==============================================================================
# Reward Function Wrapper
# ==============================================================================

@dataclass
class RewardFunction:
    """
    Wrapper for a reward function that can be called on states.
    
    Attributes:
        fn: Callable that maps state -> scalar reward.
        family: String identifier of the prior family ("goal", "linear", "mlp").
        metadata: Optional dict with additional info (e.g., goal state, weight vector).
    """
    fn: Callable[[np.ndarray], np.ndarray]
    family: str
    metadata: Optional[Dict[str, Any]] = None
    
    def __call__(self, states: np.ndarray) -> np.ndarray:
        """Evaluate reward on a batch of states. States shape: (batch, state_dim)."""
        return self.fn(states)


# ==============================================================================
# Family 1: Singleton Goal-Reaching Rewards
# ==============================================================================

class GoalReachingPrior:
    """
    Generates goal-reaching reward functions.
    
    Reward: η(s) = 0 if ||s - s_g|| < threshold else -1.
    The goal state s_g is sampled uniformly from the dataset.
    
    Args:
        state_dim: Dimensionality of the state space.
        goal_threshold: Distance threshold for goal achievement (default 0.5).
        dataset_states: Optional reference states to sample goals from.
                        If None, goals are sampled uniformly from [-1, 1]^state_dim.
    """
    
    def __init__(
        self,
        state_dim: int,
        goal_threshold: float = 0.5,
        dataset_states: Optional[np.ndarray] = None,
    ):
        self.state_dim = state_dim
        self.goal_threshold = goal_threshold
        self.dataset_states = dataset_states  # shape: (N, state_dim)
        
    def sample(self, rng: Optional[np.random.RandomState] = None) -> RewardFunction:
        """
        Sample a new goal-reaching reward function.
        
        Args:
            rng: Optional numpy RandomState for reproducibility.
            
        Returns:
            RewardFunction object.
        """
        if rng is None:
            rng = np.random
        
        # Sample goal state
        if self.dataset_states is not None and len(self.dataset_states) > 0:
            idx = rng.randint(0, len(self.dataset_states))
            goal_state = self.dataset_states[idx].copy()
        else:
            goal_state = rng.uniform(-1.0, 1.0, size=(self.state_dim,))
        
        threshold = self.goal_threshold
        
        def reward_fn(states: np.ndarray) -> np.ndarray:
            """
            Args:
                states: (batch_size, state_dim) numpy array.
            Returns:
                rewards: (batch_size,) numpy array, 0 at goal, -1 otherwise.
            """
            # Compute Euclidean distance to goal
            diff = states - goal_state[np.newaxis, :]  # broadcast
            dist = np.linalg.norm(diff, axis=-1)
            rewards = np.where(dist < threshold, 0.0, -1.0)
            return rewards.astype(np.float32)
        
        return RewardFunction(
            fn=reward_fn,
            family="goal",
            metadata={"goal_state": goal_state, "threshold": threshold},
        )


# ==============================================================================
# Family 2: Random Linear Functions
# ==============================================================================

class LinearPrior:
    """
    Generates random linear reward functions with sparse masking.
    
    Reward: η(s) = w^T s, where w ~ Uniform(-1, 1) and ~80% of entries
    are randomly zeroed out to bias towards simple functions.
    
    Args:
        state_dim: Dimensionality of the state space.
        sparsity: Fraction of weight entries to zero out (default 0.8).
        weight_range: Range for uniform weight sampling (default [-1, 1]).
    """
    
    def __init__(
        self,
        state_dim: int,
        sparsity: float = 0.8,
        weight_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        self.state_dim = state_dim
        self.sparsity = sparsity
        self.weight_range = weight_range
        
    def sample(self, rng: Optional[np.random.RandomState] = None) -> RewardFunction:
        """
        Sample a new random linear reward function.
        
        Args:
            rng: Optional numpy RandomState for reproducibility.
            
        Returns:
            RewardFunction object.
        """
        if rng is None:
            rng = np.random
        
        # Sample weight vector
        low, high = self.weight_range
        w = rng.uniform(low, high, size=(self.state_dim,)).astype(np.float32)
        
        # Apply sparse mask: zero out a fraction of entries
        mask = rng.rand(self.state_dim) > self.sparsity  # keep ~20%
        # Ensure at least one non-zero entry
        if not np.any(mask):
            mask[rng.randint(0, self.state_dim)] = True
        w = w * mask.astype(np.float32)
        
        def reward_fn(states: np.ndarray) -> np.ndarray:
            """
            Args:
                states: (batch_size, state_dim) numpy array.
            Returns:
                rewards: (batch_size,) numpy array.
            """
            return np.dot(states, w).astype(np.float32)
        
        return RewardFunction(
            fn=reward_fn,
            family="linear",
            metadata={"weights": w, "mask": mask, "sparsity": self.sparsity},
        )


# ==============================================================================
# Family 3: Random MLPs
# ==============================================================================

class MLPPrior:
    """
    Generates random MLP reward functions.
    
    Architecture: 2-layer MLP with ReLU activations.
    Input: state (state_dim); Hidden: hidden_dim; Output: scalar reward.
    Weights are sampled from Kaiming uniform distribution.
    
    Args:
        state_dim: Dimensionality of the state space.
        hidden_dim: Hidden layer dimension (default 256).
        activation: Activation function (default "relu").
        output_scale: Scaling factor for output (default 1.0).
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        activation: str = "relu",
        output_scale: float = 1.0,
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.activation_name = activation
        self.output_scale = output_scale
        
    def _init_weights(self, module: nn.Module, rng: Optional[np.random.RandomState] = None):
        """Initialize weights using Kaiming uniform."""
        if isinstance(module, nn.Linear):
            # Use Kaiming uniform initialization
            nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                fan_in = module.weight.size(1)
                bound = 1.0 / np.sqrt(fan_in)
                nn.init.uniform_(module.bias, -bound, bound)
    
    def sample(self, rng: Optional[np.random.RandomState] = None) -> RewardFunction:
        """
        Sample a new random MLP reward function.
        
        Args:
            rng: Optional numpy RandomState for reproducibility.
            
        Returns:
            RewardFunction object.
        """
        if rng is not None:
            torch.manual_seed(rng.randint(0, 2**31 - 1))
        
        # Build a fresh random MLP
        layers = []
        layers.append(nn.Linear(self.state_dim, self.hidden_dim))
        if self.activation_name == "relu":
            layers.append(nn.ReLU())
        elif self.activation_name == "tanh":
            layers.append(nn.Tanh())
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim, 1))
        
        mlp = nn.Sequential(*layers)
        mlp.apply(lambda m: self._init_weights(m, rng))
        mlp.eval()  # No training needed; it's a fixed random function
        
        output_scale = self.output_scale
        
        def reward_fn(states: np.ndarray) -> np.ndarray:
            """
            Args:
                states: (batch_size, state_dim) numpy array.
            Returns:
                rewards: (batch_size,) numpy array.
            """
            with torch.no_grad():
                states_t = torch.from_numpy(states).float()
                rewards_t = mlp(states_t).squeeze(-1) * output_scale
                return rewards_t.numpy().astype(np.float32)
        
        return RewardFunction(
            fn=reward_fn,
            family="mlp",
            metadata={"hidden_dim": self.hidden_dim, "activation": self.activation_name},
        )


# ==============================================================================
# Mixed Prior Manager
# ==============================================================================

class MixedPrior:
    """
    Samples reward functions uniformly from a mixture of prior families.
    
    During training, each family is sampled with equal probability (1/3 each
    when all three families are active). Can also be configured with custom
    weights or subsets of families.
    
    Args:
        priors: List of prior objects (each must have a sample() method).
        weights: Optional sampling weights for each prior. If None, uniform.
    """
    
    def __init__(
        self,
        priors: List[Any],
        weights: Optional[List[float]] = None,
    ):
        self.priors = priors
        if weights is None:
            self.weights = [1.0 / len(priors)] * len(priors)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        self._family_names = []
        for p in priors:
            if hasattr(p, 'sample'):
                # Infer family name from class
                name = type(p).__name__.replace('Prior', '').lower()
                self._family_names.append(name)
            else:
                self._family_names.append('unknown')
    
    @property
    def num_families(self) -> int:
        return len(self.priors)
    
    @property
    def family_names(self) -> List[str]:
        return self._family_names
    
    def sample(self, rng: Optional[np.random.RandomState] = None) -> RewardFunction:
        """
        Sample a reward function from the mixture.
        
        Args:
            rng: Optional numpy RandomState for reproducibility.
            
        Returns:
            RewardFunction object.
        """
        if rng is None:
            rng = np.random
        
        # Choose which prior family to sample from
        idx = rng.choice(len(self.priors), p=self.weights)
        prior = self.priors[idx]
        return prior.sample(rng=rng)
    
    def sample_from_family(self, family_idx: int, rng: Optional[np.random.RandomState] = None) -> RewardFunction:
        """
        Sample a reward function from a specific family.
        
        Args:
            family_idx: Index of the prior family.
            rng: Optional numpy RandomState.
            
        Returns:
            RewardFunction object.
        """
        if rng is None:
            rng = np.random
        return self.priors[family_idx].sample(rng=rng)
    
    def sample_batch(
        self,
        batch_size: int,
        rng: Optional[np.random.RandomState] = None,
    ) -> List[RewardFunction]:
        """
        Sample multiple reward functions.
        
        Args:
            batch_size: Number of reward functions to sample.
            rng: Optional numpy RandomState.
            
        Returns:
            List of RewardFunction objects.
        """
        if rng is None:
            rng = np.random
        return [self.sample(rng=rng) for _ in range(batch_size)]


# ==============================================================================
# Factory Function
# ==============================================================================

def create_mixed_prior(
    state_dim: int,
    dataset_states: Optional[np.ndarray] = None,
    goal_threshold: float = 0.5,
    linear_sparsity: float = 0.8,
    mlp_hidden_dim: int = 256,
    weights: Optional[List[float]] = None,
    include_goal: bool = True,
    include_linear: bool = True,
    include_mlp: bool = True,
) -> MixedPrior:
    """
    Create a MixedPrior with the three standard FRE prior families.
    
    Args:
        state_dim: Dimensionality of the state space.
        dataset_states: Reference states for goal sampling (N, state_dim).
        goal_threshold: Distance threshold for goal-reaching rewards.
        linear_sparsity: Sparsity fraction for linear rewards.
        mlp_hidden_dim: Hidden dimension for MLP rewards.
        weights: Optional sampling weights [w_goal, w_linear, w_mlp].
        include_goal: Whether to include goal-reaching prior.
        include_linear: Whether to include linear prior.
        include_mlp: Whether to include MLP prior.
        
    Returns:
        MixedPrior instance.
    """
    priors = []
    active_weights = []
    
    if include_goal:
        priors.append(GoalReachingPrior(
            state_dim=state_dim,
            goal_threshold=goal_threshold,
            dataset_states=dataset_states,
        ))
        if weights is not None:
            active_weights.append(weights[0])
    
    if include_linear:
        priors.append(LinearPrior(
            state_dim=state_dim,
            sparsity=linear_sparsity,
        ))
        if weights is not None:
            active_weights.append(weights[1] if len(weights) > 1 else weights[0])
    
    if include_mlp:
        priors.append(MLPPrior(
            state_dim=state_dim,
            hidden_dim=mlp_hidden_dim,
        ))
        if weights is not None:
            active_weights.append(weights[2] if len(weights) > 2 else weights[0])
    
    if len(active_weights) > 0 and len(active_weights) == len(priors):
        return MixedPrior(priors=priors, weights=active_weights)
    else:
        return MixedPrior(priors=priors, weights=None)


# ==============================================================================
# Domain-Specific Prior Augmentations (Section 5.4)
# ==============================================================================

class XYPositionPrior:
    """
    Domain-specific prior for AntMaze: rewards based only on XY position.
    
    This implements the "domain knowledge augmentation" described in Section 5.4.
    Reward: η(s) = f(x, y) where f is a random function of the 2D position.
    The function can be goal-reaching, linear, or MLP but only using the first
    two state dimensions (x, y position).
    
    Args:
        prior_type: Type of prior to use on XY subspace ("goal", "linear", "mlp").
        goal_threshold: Threshold for goal-reaching variant.
        linear_sparsity: Sparsity for linear variant.
        mlp_hidden_dim: Hidden dim for MLP variant.
        dataset_states: Reference states (uses first 2 dims for goals).
    """
    
    def __init__(
        self,
        prior_type: str = "goal",
        goal_threshold: float = 0.5,
        linear_sparsity: float = 0.5,  # Less sparse for 2D
        mlp_hidden_dim: int = 128,
        dataset_states: Optional[np.ndarray] = None,
    ):
        self.prior_type = prior_type
        self.state_dim = 2  # XY only
        
        xy_dataset = None
        if dataset_states is not None:
            xy_dataset = dataset_states[:, :2]
        
        if prior_type == "goal":
            self.prior = GoalReachingPrior(
                state_dim=2,
                goal_threshold=goal_threshold,
                dataset_states=xy_dataset,
            )
        elif prior_type == "linear":
            self.prior = LinearPrior(
                state_dim=2,
                sparsity=linear_sparsity,
            )
        elif prior_type == "mlp":
            self.prior = MLPPrior(
                state_dim=2,
                hidden_dim=mlp_hidden_dim,
            )
        else:
            raise ValueError(f"Unknown prior_type: {prior_type}")
    
    def sample(self, rng: Optional[np.random.RandomState] = None) -> RewardFunction:
        """Sample a reward function that only depends on XY position."""
        base_rf = self.prior.sample(rng=rng)
        
        def reward_fn(states: np.ndarray) -> np.ndarray:
            # Use only first 2 dimensions
            xy = states[..., :2]
            return base_rf.fn(xy)
        
        return RewardFunction(
            fn=reward_fn,
            family=f"xy_{self.prior_type}",
            metadata=base_rf.metadata,
        )


# ==============================================================================
# Testing Utilities
# ==============================================================================

def test_prior(
    prior,
    state_dim: int = 4,
    num_states: int = 100,
    num_samples: int = 5,
    seed: int = 42,
):
    """
    Quick test of a prior: sample reward functions and evaluate on random states.
    
    Args:
        prior: A prior object with a sample() method.
        state_dim: Dimensionality of test states.
        num_states: Number of test states.
        num_samples: Number of reward functions to sample.
        seed: Random seed.
    """
    rng = np.random.RandomState(seed)
    test_states = rng.randn(num_states, state_dim).astype(np.float32)
    
    print(f"Testing prior: {type(prior).__name__}")
    print(f"  State dim: {state_dim}, Test states: {num_states}")
    
    for i in range(num_samples):
        rf = prior.sample(rng=rng)
        rewards = rf(test_states)
        print(f"  Sample {i+1}: family={rf.family}, "
              f"reward range=[{rewards.min():.3f}, {rewards.max():.3f}], "
              f"mean={rewards.mean():.3f}, std={rewards.std():.3f}")
    
    print("  Test passed!\n")


if __name__ == "__main__":
    # Quick self-test
    print("=" * 60)
    print("Testing Reward Prior Distributions")
    print("=" * 60)
    
    state_dim = 8
    dataset_states = np.random.randn(1000, state_dim).astype(np.float32)
    
    # Test individual priors
    print("\n--- GoalReachingPrior ---")
    goal_prior = GoalReachingPrior(state_dim=state_dim, dataset_states=dataset_states)
    test_prior(goal_prior, state_dim=state_dim)
    
    print("--- LinearPrior ---")
    linear_prior = LinearPrior(state_dim=state_dim, sparsity=0.8)
    test_prior(linear_prior, state_dim=state_dim)
    
    print("--- MLPPrior ---")
    mlp_prior = MLPPrior(state_dim=state_dim, hidden_dim=256)
    test_prior(mlp_prior, state_dim=state_dim)
    
    # Test MixedPrior
    print("--- MixedPrior (uniform) ---")
    mixed = create_mixed_prior(state_dim=state_dim, dataset_states=dataset_states)
    test_prior(mixed, state_dim=state_dim, num_samples=10)
    
    # Test XYPositionPrior
    print("--- XYPositionPrior ---")
    xy_prior = XYPositionPrior(prior_type="goal", dataset_states=dataset_states)
    test_prior(xy_prior, state_dim=state_dim)
    
    print("All prior tests passed!")
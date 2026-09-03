"""
Mask Network (Algorithm 1): Improved StateMask Explanation Method.

Key innovation over original StateMask:
- Reformulated objective: J(θ) = max η(π̄) instead of min |η(π) - η(π̄)|
- This is justified by Theorem 3.3: η(π̄) ≤ η(π) under Assumption 3.1
- Uses vanilla PPO instead of primal-dual optimization
- Adds reward bonus α for blinding: R'(s_t, a_t) = R(s_t, a_t) + α * a_t^m
- This prevents the trivial solution of never blinding (always outputting 0)

The mask network takes a state s_t as input and outputs a binary action a_t^m:
- a_t^m = 0: use target agent's action (state is "important")
- a_t^m = 1: replace with random action (state is "not important")

State importance = probability of mask network outputting 0 at that state.
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli


class MaskNetwork(nn.Module):
    """
    Neural network that outputs a binary mask decision for a given state.
    Uses a Bernoulli distribution over {0, 1} where:
    - 0 means "keep the agent's action" (state is critical)
    - 1 means "replace with random action" (state can be blinded)

    The architecture follows the paper's approach: an MLP that takes
    the state as input and outputs logits for the binary mask action.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: List[int] = (64, 64),
        activation: str = "tanh",
    ):
        """
        Args:
            state_dim: Dimension of the state space.
            hidden_sizes: Sizes of hidden layers.
            activation: Activation function ('tanh' or 'relu').
        """
        super().__init__()
        layers = []
        prev_size = state_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(prev_size, hs))
            if activation == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            prev_size = hs
        layers.append(nn.Linear(prev_size, 1))  # single logit for Bernoulli
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns logits for the Bernoulli distribution.
        Args:
            state: Tensor of shape (batch_size, state_dim)
        Returns:
            logits: Tensor of shape (batch_size, 1)
        """
        return self.net(state)

    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a mask action from the Bernoulli distribution.
        Args:
            state: Tensor of shape (batch_size, state_dim)
        Returns:
            action: Binary mask action (0 or 1)
            log_prob: Log probability of the sampled action
        """
        logits = self.forward(state)
        dist = Bernoulli(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def get_action_and_value(
        self, state: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, and entropy for PPO training.
        Args:
            state: Tensor of shape (batch_size, state_dim)
            action: Optional pre-sampled action for log_prob computation
        Returns:
            action: Binary mask action (0 or 1)
            log_prob: Log probability
            entropy: Distribution entropy
        """
        logits = self.forward(state)
        dist = Bernoulli(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def get_importance(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get state importance score: probability of outputting 0 (keeping action).
        Higher importance = more critical state.
        Args:
            state: Tensor of shape (batch_size, state_dim)
        Returns:
            importance: Tensor of shape (batch_size,) in [0, 1]
        """
        logits = self.forward(state)
        prob_zero = torch.sigmoid(-logits)  # P(a_t^m = 0)
        return prob_zero.squeeze(-1)


class MaskNetworkTrainer:
    """
    Trains the mask network using PPO (Algorithm 1 from the paper).

    The mask network is trained to maximize the expected total reward of
    the perturbed agent η(π̄), where π̄ replaces the target agent's actions
    with random actions at steps where the mask outputs 1.

    Reward function:
        R'(s_t, a_t) = R(s_t, a_t) + α * a_t^m

    where α is a hyper-parameter that encourages the mask to blind
    (output 1) at some steps, preventing the trivial solution of never
    blinding at all.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = (64, 64),
        alpha: float = 0.0001,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        normalize_advantages: bool = True,
        device: str = "cpu",
    ):
        """
        Args:
            state_dim: Dimension of the state space.
            action_dim: Dimension of the agent's action space.
            hidden_sizes: Hidden layer sizes for the mask network.
            alpha: Reward bonus for blinding (encourages mask to output 1).
            lr: Learning rate for Adam optimizer.
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
            clip_epsilon: PPO clipping epsilon.
            value_coef: Value loss coefficient.
            entropy_coef: Entropy bonus coefficient.
            max_grad_norm: Maximum gradient norm for clipping.
            update_epochs: Number of PPO update epochs per iteration.
            batch_size: Mini-batch size for PPO updates.
            normalize_advantages: Whether to normalize advantages.
            device: Device to run on ('cpu' or 'cuda').
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.normalize_advantages = normalize_advantages
        self.device = device

        # Mask network: policy (actor) + value function (critic)
        self.mask_net = MaskNetwork(state_dim, hidden_sizes).to(device)
        self.value_net = nn.Sequential(
            MaskNetwork(state_dim, hidden_sizes).net[:-1],
            nn.Linear(hidden_sizes[-1], 1),
        ).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.mask_net.parameters()) + list(self.value_net.parameters()),
            lr=lr,
        )

    def get_mask_action(self, state: np.ndarray) -> int:
        """
        Get a mask action for a single state.
        Args:
            state: numpy array of shape (state_dim,)
        Returns:
            mask_action: 0 or 1
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.mask_net.get_action(state_t)
            return int(action.item())

    def get_state_importance(self, state: np.ndarray) -> float:
        """
        Get the importance score for a single state.
        Args:
            state: numpy array of shape (state_dim,)
        Returns:
            importance: float in [0, 1], higher = more critical
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            importance = self.mask_net.get_importance(state_t)
            return float(importance.item())

    def get_trajectory_importance(
        self, states: np.ndarray
    ) -> np.ndarray:
        """
        Get importance scores for all states in a trajectory.
        Args:
            states: numpy array of shape (T, state_dim)
        Returns:
            importances: numpy array of shape (T,)
        """
        with torch.no_grad():
            states_t = torch.FloatTensor(states).to(self.device)
            importances = self.mask_net.get_importance(states_t)
            return importances.cpu().numpy()

    def find_most_critical_state(
        self, states: np.ndarray
    ) -> Tuple[int, np.ndarray]:
        """
        Identify the most critical state in a trajectory.
        Args:
            states: numpy array of shape (T, state_dim)
        Returns:
            idx: Index of the most critical state
            state: The state array at that index
        """
        importances = self.get_trajectory_importance(states)
        idx = int(np.argmax(importances))
        return idx, states[idx]

    def collect_rollout(
        self,
        target_policy_fn,
        env_reset_fn,
        env_step_fn,
        rollout_length: int,
        action_space_sample_fn,
    ) -> Dict[str, np.ndarray]:
        """
        Collect a rollout for training the mask network.

        The mask network perturbs the target agent: when mask outputs 1,
        a random action is taken instead of the target policy's action.
        The reward includes a bonus α for each blinding action.

        Args:
            target_policy_fn: Function (state) -> action for the target agent.
            env_reset_fn: Function () -> initial_state.
            env_step_fn: Function (action) -> (next_state, reward, done, info).
            rollout_length: Number of steps per rollout.
            action_space_sample_fn: Function () -> random_action.

        Returns:
            Dict with keys: states, mask_actions, rewards, values, log_probs,
                           dones, advantages, returns.
        """
        states = []
        mask_actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []

        state = env_reset_fn()
        for t in range(rollout_length):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                mask_action, log_prob, _ = self.mask_net.get_action_and_value(state_t)
                value = self.value_net(state_t)

            mask_a = int(mask_action.item())
            if mask_a == 0:
                # Keep target agent's action
                agent_action = target_policy_fn(state)
            else:
                # Replace with random action
                agent_action = action_space_sample_fn()

            next_state, reward, done, info = env_step_fn(agent_action)

            # Add blinding bonus: α * a_t^m
            modified_reward = reward + self.alpha * mask_a

            states.append(state)
            mask_actions.append(mask_a)
            rewards.append(modified_reward)
            values.append(float(value.item()))
            log_probs.append(float(log_prob.item()))
            dones.append(float(done))

            state = next_state
            if done:
                state = env_reset_fn()

        # Compute advantages and returns using GAE
        advantages, returns = self._compute_gae(rewards, values, dones)

        return {
            "states": np.array(states, dtype=np.float32),
            "mask_actions": np.array(mask_actions, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "values": np.array(values, dtype=np.float32),
            "log_probs": np.array(log_probs, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
            "advantages": advantages,
            "returns": returns,
        }

    def _compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation (GAE)."""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            else:
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        return advantages, returns

    def update(self, rollout_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Perform PPO update on the mask network using collected rollout data.

        Args:
            rollout_data: Dict from collect_rollout.

        Returns:
            Dict of loss statistics.
        """
        states = torch.FloatTensor(rollout_data["states"]).to(self.device)
        old_actions = torch.FloatTensor(rollout_data["mask_actions"]).unsqueeze(-1).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_data["log_probs"]).unsqueeze(-1).to(self.device)
        advantages = torch.FloatTensor(rollout_data["advantages"]).unsqueeze(-1).to(self.device)
        returns = torch.FloatTensor(rollout_data["returns"]).unsqueeze(-1).to(self.device)

        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(states)
        indices = np.arange(n)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                batch_states = states[batch_idx]
                batch_actions = old_actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Get new action distributions
                _, new_log_probs, entropy = self.mask_net.get_action_and_value(
                    batch_states, batch_actions
                )
                new_values = self.value_net(batch_states)

                # PPO policy loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.mask_net.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def train(
        self,
        target_policy_fn,
        env_reset_fn,
        env_step_fn,
        action_space_sample_fn,
        n_iterations: int,
        rollout_length: int = 2048,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """
        Full training loop for the mask network (Algorithm 1).

        Args:
            target_policy_fn: Function (state) -> action.
            env_reset_fn: Function () -> initial_state.
            env_step_fn: Function (action) -> (next_state, reward, done, info).
            action_space_sample_fn: Function () -> random_action.
            n_iterations: Number of PPO iterations.
            rollout_length: Steps per rollout.
            verbose: Whether to print progress.

        Returns:
            List of per-iteration statistics.
        """
        history = []
        for iteration in range(n_iterations):
            rollout_data = self.collect_rollout(
                target_policy_fn, env_reset_fn, env_step_fn,
                action_space_sample_fn, rollout_length,
            )
            stats = self.update(rollout_data)
            stats["iteration"] = iteration
            history.append(stats)

            if verbose and iteration % 10 == 0:
                avg_importance = self.get_trajectory_importance(
                    rollout_data["states"][:100]
                ).mean()
                print(
                    f"[MaskNet] iter={iteration} "
                    f"policy_loss={stats['policy_loss']:.4f} "
                    f"value_loss={stats['value_loss']:.4f} "
                    f"entropy={stats['entropy']:.4f} "
                    f"avg_importance={avg_importance:.3f}"
                )

        return history

    def save(self, path: str) -> None:
        """Save mask network weights."""
        torch.save(
            {
                "mask_net": self.mask_net.state_dict(),
                "value_net": self.value_net.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load mask network weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.mask_net.load_state_dict(checkpoint["mask_net"])
        self.value_net.load_state_dict(checkpoint["value_net"])
"""
RICE Refinement Method (Algorithm 2): Core refining scheme for RL with explanation.

Algorithm 2 refines a pre-trained DRL agent by:
1. Constructing a mixed initial state distribution μ(s) = β * d_ρ^π̂(s) + (1-β) * ρ(s)
   where d_ρ^π̂(s) is the distribution of critical states identified by the mask network
2. Starting exploration from this mixed distribution
3. Using RND exploration bonus to encourage visiting novel states
4. Updating the policy via PPO

Hyper-parameters:
- p: probability of resetting to a critical state (controls β in μ)
- λ: weight of RND exploration bonus relative to task reward
"""

from typing import Optional, Dict, Any, Tuple, List, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask_network import MaskNetworkTrainer
from .rnd import RNDExploration


class RICERefiner:
    """
    RICE Refining Scheme: refine a pre-trained DRL agent by breaking through
    training bottlenecks using explanation and exploration.

    The full RICE algorithm:
    1. Pre-train a mask network to identify critical states (via MaskNetworkTrainer)
    2. With probability p, reset to a critical state (from mask network)
    3. With probability 1-p, reset to default initial state (from ρ)
    4. Add RND exploration bonus to reward
    5. Update policy via PPO
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        # Mask network config
        mask_hidden_sizes: List[int] = (64, 64),
        alpha: float = 0.0001,
        # RND config
        rnd_output_dim: int = 128,
        rnd_hidden_sizes: Tuple[int, ...] = (64, 64),
        rnd_lr: float = 1e-4,
        # Refinement hyper-parameters
        p: float = 0.25,          # probability of resetting to critical state
        rnd_lambda: float = 0.01, # weight of exploration bonus
        # PPO config for refinement
        ppo_lr: float = 3e-4,
        ppo_gamma: float = 0.99,
        ppo_gae_lambda: float = 0.95,
        ppo_clip_epsilon: float = 0.2,
        ppo_value_coef: float = 0.5,
        ppo_entropy_coef: float = 0.01,
        ppo_max_grad_norm: float = 0.5,
        ppo_update_epochs: int = 10,
        ppo_batch_size: int = 64,
        normalize_advantages: bool = True,
        device: str = "cpu",
    ):
        """
        Args:
            state_dim: Dimension of state space.
            action_dim: Dimension of action space.
            mask_hidden_sizes: Hidden sizes for mask network.
            alpha: Blinding bonus coefficient for mask network training.
            rnd_output_dim: Output dimension of RND networks.
            rnd_hidden_sizes: Hidden sizes for RND networks.
            rnd_lr: Learning rate for RND predictor.
            p: Probability of resetting to critical state (0 ≤ p ≤ 1).
               p=0 means all from default ρ, p=1 means all from critical states.
               Recommended: 0.25 or 0.5.
            rnd_lambda: Weight of exploration bonus λ.
            ppo_lr: Learning rate for PPO policy optimization.
            ppo_gamma: Discount factor for PPO.
            ppo_gae_lambda: GAE lambda for PPO.
            ppo_clip_epsilon: PPO clipping epsilon.
            ppo_value_coef: Value loss coefficient.
            ppo_entropy_coef: Entropy bonus coefficient.
            ppo_max_grad_norm: Max gradient norm.
            ppo_update_epochs: PPO update epochs per iteration.
            ppo_batch_size: Mini-batch size.
            normalize_advantages: Whether to normalize advantages.
            device: 'cpu' or 'cuda'.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.p = p
        self.rnd_lambda = rnd_lambda
        self.p_gamma = ppo_gamma
        self._gae_lambda = ppo_gae_lambda
        self._clip_epsilon = ppo_clip_epsilon
        self._value_coef = ppo_value_coef
        self._entropy_coef = ppo_entropy_coef
        self._max_grad_norm = ppo_max_grad_norm
        self._update_epochs = ppo_update_epochs
        self._batch_size = ppo_batch_size
        self._normalize_advantages = normalize_advantages
        self.device = device

        self.mask_trainer: Optional[MaskNetworkTrainer] = None
        self.rnd: Optional[RNDExploration] = None
        self._ppo_optimizer: Optional[torch.optim.Adam] = None

    def set_mask_network(self, mask_trainer: MaskNetworkTrainer) -> None:
        """Set a pre-trained mask network for finding critical states."""
        self.mask_trainer = mask_trainer

    def set_rnd(self, rnd: RNDExploration) -> None:
        """Set an RND exploration module."""
        self.rnd = rnd

    def _random_number(self) -> float:
        """Generate a random number in [0, 1] (as in Algorithm 2)."""
        return float(np.random.random())

    def _decide_start_state(
        self,
        target_policy_fn: Callable[[np.ndarray], np.ndarray],
        env_reset_fn: Callable[[], np.ndarray],
        env_step_fn: Callable[[np.ndarray], Tuple[np.ndarray, float, bool, Dict]],
        state_restore_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Decide the initial state: with probability p, use a critical state;
        with probability 1-p, use the default initial state.

        If p > 0 and mask_trainer is available:
            - Run the policy π to get a trajectory τ of length K
            - Identify the most critical state s_t via state mask π̃
            - Restore environment to s_t (using state_restore_fn or by
              re-executing the environment from the default initial state)

        Args:
            target_policy_fn: Function (state) -> action.
            env_reset_fn: Function () -> initial_state.
            env_step_fn: Function (action) -> (next_state, reward, done, info).
            state_restore_fn: Optional function (state) -> restored_state.

        Returns:
            initial_state: The chosen initial state.
        """
        rand_num = self._random_number()

        if rand_num >= self.p or self.mask_trainer is None:
            # Use default initial state distribution ρ
            return env_reset_fn()
        else:
            # Run π to obtain a trajectory τ of length K
            K = 200  # trajectory length for critical state detection
            states = []
            state = env_reset_fn()
            for _ in range(K):
                action = target_policy_fn(state)
                next_state, _, done, _ = env_step_fn(action)
                states.append(state)
                if done:
                    state = env_reset_fn()
                else:
                    state = next_state

            if len(states) < 2:
                return env_reset_fn()

            states_arr = np.array(states, dtype=np.float32)

            # Identify the most critical state in τ via state mask π̃
            importances = self.mask_trainer.get_trajectory_importance(states_arr)
            critical_idx = int(np.argmax(importances))
            critical_state = states_arr[critical_idx]

            # Restore environment to the critical state
            if state_restore_fn is not None:
                state_restore_fn(critical_state)
            # Note: in simulator-based environments, restoring to a state
            # requires env-specific mechanisms. In practice, this would use
            # set_state or checkpoint/restore methods.

            return critical_state

    def collect_rollout(
        self,
        policy_net: nn.Module,
        env_reset_fn: Callable[[], np.ndarray],
        env_step_fn: Callable[[np.ndarray], Tuple],
        rollout_length: int,
        target_policy_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        state_restore_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        action_space_sample_fn: Optional[Callable[[], np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Collect a rollout for refining the DRL agent (Algorithm 2).

        For each episode:
        1. Decide start state (critical vs default)
        2. Run policy, computing RND bonus
        3. Modified reward = task_reward + λ * RND_bonus

        Args:
            policy_net: The policy network being refined.
            env_reset_fn: Function () -> initial_state.
            env_step_fn: Function (action) -> (next_state, reward, done, info).
            rollout_length: Total steps to collect.
            target_policy_fn: Function for the pre-trained target policy.
                              Required when p > 0 (to collect trajectory for
                              critical state detection).
            state_restore_fn: Optional function to restore env state.
            action_space_sample_fn: Function () -> random_action.

        Returns:
            Dict with rollout data for PPO update.
        """
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        rnd_bonuses = []

        state = None
        episode_step = 0

        for t in range(rollout_length):
            if state is None or (dones[-1] if dones else True):
                # New episode: decide start state
                state = self._decide_start_state(
                    target_policy_fn or (lambda s: np.zeros(self.action_dim)),
                    env_reset_fn, env_step_fn, state_restore_fn,
                )
                episode_step = 0

            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, log_prob, entropy, value = self._policy_action_value(
                    policy_net, state_t
                )

            action_np = action.squeeze(0).cpu().numpy()
            next_state, task_reward, done, info = env_step_fn(action_np)

            # Compute RND exploration bonus
            rnd_bonus = 0.0
            if self.rnd is not None:
                rnd_bonus = self.rnd.get_bonus(next_state)
                self.rnd.update_norm(rnd_bonus)

            # Modified reward = task reward + λ * RND bonus
            modified_reward = task_reward + self.rnd_lambda * rnd_bonus

            states.append(state)
            actions.append(action_np)
            rewards.append(modified_reward)
            values.append(float(value.item()))
            log_probs.append(float(log_prob.item()))
            dones.append(float(done))
            rnd_bonuses.append(rnd_bonus)

            state = next_state
            episode_step += 1

        # Compute GAE advantages and returns
        advantages, returns = self._compute_gae(rewards, values, dones)

        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "values": np.array(values, dtype=np.float32),
            "log_probs": np.array(log_probs, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
            "advantages": advantages,
            "returns": returns,
            "rnd_bonuses": np.array(rnd_bonuses, dtype=np.float32),
        }

    def _policy_action_value(
        self, policy_net: nn.Module, state: torch.Tensor
    ) -> Tuple:
        """
        Get action, log_prob, entropy, and value from the policy network.
        This needs to be overridden or adapted for specific policy architectures.

        For a standard actor-critic:
            actor outputs mean/log_std for Gaussian policy
            critic outputs value
        """
        # This is abstract — concrete implementations must provide
        # their own policy_net which supports this interface.
        # For Gaussian policies in continuous action spaces:
        action_mean = policy_net.get_action_mean(state)
        action_std = policy_net.get_action_std(state)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        value = policy_net.get_value(state)
        return action, log_prob, entropy, value

    def _compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns."""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae_returns = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                delta = (
                    rewards[t]
                    + self.p_gamma * next_value * (1.0 - dones[t])
                    - values[t]
                )
            else:
                delta = (
                    rewards[t]
                    + self.p_gamma * values[t + 1] * (1.0 - dones[t])
                    - values[t]
                )
            gae = delta + self.p_gamma * self._gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
            gae_returns[t] = gae + values[t]

        return advantages, gae_returns

    def update_policy(
        self,
        policy_net: nn.Module,
        rollout_data: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Perform PPO update on the policy network (Algorithm 2 update step).
        Also updates the RND predictor network.

        Args:
            policy_net: The policy network to update.
            rollout_data: Dict from collect_rollout.

        Returns:
            Statistics dict.
        """
        states_t = torch.FloatTensor(rollout_data["states"]).to(self.device)
        actions_t = torch.FloatTensor(rollout_data["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_data["log_probs"]).to(self.device)
        advantages = torch.FloatTensor(rollout_data["advantages"]).to(self.device)
        returns = torch.FloatTensor(rollout_data["returns"]).to(self.device)

        if self._normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update RND predictor
        rnd_loss = 0.0
        if self.rnd is not None:
            # RND is updated on next_states (the states resulting from actions)
            rnd_loss = self.rnd.update(states_t)

        # PPO policy update
        n = len(states_t)
        indices = np.arange(n)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self._update_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self._batch_size):
                batch_idx = indices[start:start + self._batch_size]
                batch_states = states_t[batch_idx]
                batch_actions = actions_t[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Get action distribution and values
                action_mean = policy_net.get_action_mean(batch_states)
                action_std = policy_net.get_action_std(batch_states)
                dist = torch.distributions.Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1, keepdim=True)
                new_values = policy_net.get_value(batch_states)

                # PPO policy loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self._clip_epsilon,
                        1.0 + self._clip_epsilon,
                    )
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss (no additional value_coef needed since
                # it's already scaled in the original PPO formulation)
                loss = (
                    policy_loss
                    + self._value_coef * value_loss
                    + self._entropy_coef * entropy_loss
                )

                self._ppo_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    policy_net.parameters(), self._max_grad_norm
                )
                self._ppo_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "rnd_loss": rnd_loss,
        }

    def refine(
        self,
        policy_net: nn.Module,
        env_reset_fn: Callable[[], np.ndarray],
        env_step_fn: Callable[[np.ndarray], Tuple],
        n_iterations: int,
        rollout_length: int = 2048,
        target_policy_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        state_restore_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        action_space_sample_fn: Optional[Callable[[], np.ndarray]] = None,
        verbose: bool = True,
        evaluate_fn: Optional[Callable[[nn.Module], float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Full refinement loop (Algorithm 2).

        Args:
            policy_net: Policy network to refine.
            env_reset_fn: Function () -> initial_state.
            env_step_fn: Function (action) -> (next_state, reward, done, info).
            n_iterations: Number of refinement iterations.
            rollout_length: Steps per rollout.
            target_policy_fn: Pre-trained policy for critical state detection.
            state_restore_fn: Optional function to restore env state.
            action_space_sample_fn: Function for random action sampling.
            verbose: Whether to print progress.
            evaluate_fn: Optional function (policy) -> mean_reward.

        Returns:
            List of per-iteration statistics.
        """
        # Set up PPO optimizer if not already set
        if self._ppo_optimizer is None:
            self._ppo_optimizer = torch.optim.Adam(
                policy_net.parameters(), lr=3e-4
            )

        history = []
        for iteration in range(n_iterations):
            rollout_data = self.collect_rollout(
                policy_net=policy_net,
                env_reset_fn=env_reset_fn,
                env_step_fn=env_step_fn,
                rollout_length=rollout_length,
                target_policy_fn=target_policy_fn,
                state_restore_fn=state_restore_fn,
                action_space_sample_fn=action_space_sample_fn,
            )

            stats = self.update_policy(policy_net, rollout_data)
            stats["iteration"] = iteration
            stats["mean_reward"] = float(np.mean(rollout_data["rewards"]))
            stats["mean_rnd_bonus"] = float(np.mean(rollout_data["rnd_bonuses"]))

            if evaluate_fn is not None:
                stats["eval_reward"] = evaluate_fn(policy_net)

            history.append(stats)

            if verbose and iteration % 10 == 0:
                eval_str = ""
                if "eval_reward" in stats:
                    eval_str = f"eval_reward={stats['eval_reward']:.2f} "
                print(
                    f"[RICE] iter={iteration} "
                    f"{eval_str}"
                    f"policy_loss={stats['policy_loss']:.4f} "
                    f"value_loss={stats['value_loss']:.4f} "
                    f"rnd_loss={stats.get('rnd_loss', 0):.4f} "
                    f"mean_bonus={stats['mean_rnd_bonus']:.4f}"
                )

        return history

    def save(self, path: str) -> None:
        """Save refiner state."""
        torch.save(
            {
                "p": self.p,
                "rnd_lambda": self.rnd_lambda,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load refiner state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.p = checkpoint["p"]
        self.rnd_lambda = checkpoint["rnd_lambda"]


class RICEAgent:
    """
    Complete RICE agent combining:
    - Pre-trained policy (target agent)
    - Mask network for explanation
    - RND for exploration
    - RICE refiner for breaking through training bottlenecks

    This is a high-level wrapper that orchestrates the full RICE pipeline.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        # Mask network
        mask_hidden_sizes: List[int] = (64, 64),
        alpha: float = 0.0001,
        # RND
        rnd_output_dim: int = 128,
        rnd_hidden_sizes: Tuple[int, ...] = (64, 64),
        # Refinement
        p: float = 0.25,
        rnd_lambda: float = 0.01,
        device: str = "cpu",
    ):
        """
        Args:
            state_dim: Dimension of state space.
            action_dim: Dimension of action space.
            mask_hidden_sizes: Hidden sizes for mask network.
            alpha: Blinding bonus for mask network.
            rnd_output_dim: RND embedding dimension.
            rnd_hidden_sizes: RND hidden sizes.
            p: Critical state reset probability.
            rnd_lambda: Exploration bonus weight.
            device: 'cpu' or 'cuda'.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Mask network trainer
        self.mask_trainer = MaskNetworkTrainer(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=list(mask_hidden_sizes),
            alpha=alpha,
            device=device,
        )

        # RND exploration
        self.rnd = RNDExploration(
            state_dim=state_dim,
            output_dim=rnd_output_dim,
            hidden_sizes=rnd_hidden_sizes,
            device=device,
        )

        # RICE refiner
        self.refiner = RICERefiner(
            state_dim=state_dim,
            action_dim=action_dim,
            p=p,
            rnd_lambda=rnd_lambda,
            device=device,
        )
        self.refiner.set_mask_network(self.mask_trainer)
        self.refiner.set_rnd(self.rnd)

    def train_mask(
        self,
        target_policy_fn: Callable[[np.ndarray], np.ndarray],
        env_reset_fn: Callable[[], np.ndarray],
        env_step_fn: Callable[[np.ndarray], Tuple],
        action_space_sample_fn: Callable[[], np.ndarray],
        n_iterations: int,
        rollout_length: int = 2048,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """
        Train the mask network to explain the pre-trained policy.

        Args:
            target_policy_fn: Pre-trained policy π.
            env_reset_fn: Environment reset function.
            env_step_fn: Environment step function.
            action_space_sample_fn: Random action sampler.
            n_iterations: Number of training iterations.
            rollout_length: Steps per rollout.
            verbose: Whether to print progress.

        Returns:
            Training history.
        """
        return self.mask_trainer.train(
            target_policy_fn=target_policy_fn,
            env_reset_fn=env_reset_fn,
            env_step_fn=env_step_fn,
            action_space_sample_fn=action_space_sample_fn,
            n_iterations=n_iterations,
            rollout_length=rollout_length,
            verbose=verbose,
        )

    def refine(
        self,
        policy_net: nn.Module,
        env_reset_fn: Callable[[], np.ndarray],
        env_step_fn: Callable[[np.ndarray], Tuple],
        n_iterations: int,
        rollout_length: int = 2048,
        target_policy_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        state_restore_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        action_space_sample_fn: Optional[Callable[[], np.ndarray]] = None,
        verbose: bool = True,
        evaluate_fn: Optional[Callable[[nn.Module], float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Refine the pre-trained DRL agent using RICE.

        Args:
            policy_net: Policy network to refine.
            env_reset_fn: Environment reset.
            env_step_fn: Environment step.
            n_iterations: Refinement iterations.
            rollout_length: Steps per rollout.
            target_policy_fn: Pre-trained policy (for critical states).
            state_restore_fn: State restoration function.
            action_space_sample_fn: Random action sampler.
            verbose: Print progress.
            evaluate_fn: Evaluation function.

        Returns:
            Refinement history.
        """
        return self.refiner.refine(
            policy_net=policy_net,
            env_reset_fn=env_reset_fn,
            env_step_fn=env_step_fn,
            n_iterations=n_iterations,
            rollout_length=rollout_length,
            target_policy_fn=target_policy_fn,
            state_restore_fn=state_restore_fn,
            action_space_sample_fn=action_space_sample_fn,
            verbose=verbose,
            evaluate_fn=evaluate_fn,
        )

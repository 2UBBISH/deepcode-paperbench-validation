"""
Unit tests for the mask network module (rice/mask_net.py).

Tests cover:
- MaskNetwork: architecture, forward pass, importance scores, action sampling
- PerturbedPolicy: action generation, importance score retrieval
- MaskNetworkTrainer: trajectory collection, PPO updates, save/load
- Fidelity computation: Pearson correlation, environment-based fidelity
- Convenience function: train_mask_network
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import gym

from rice.mask_net import (
    MaskNetwork,
    PerturbedPolicy,
    MaskNetworkTrainer,
    compute_fidelity,
    compute_fidelity_from_env,
    train_mask_network,
)


class TestMaskNetwork(unittest.TestCase):
    """Test the MaskNetwork class."""

    def setUp(self):
        self.state_dim = 8
        self.hidden_sizes = (64, 64)
        self.device = "cpu"
        self.net = MaskNetwork(
            state_dim=self.state_dim,
            hidden_sizes=self.hidden_sizes,
            activation="tanh",
        ).to(self.device)

    def test_architecture(self):
        """Test that the network has the correct architecture."""
        # Check that it's an nn.Module
        self.assertIsInstance(self.net, nn.Module)

        # Forward pass shape: (batch, state_dim) -> (batch, 2) for policy, (batch, 1) for value
        batch_size = 32
        x = torch.randn(batch_size, self.state_dim, device=self.device)
        action_logits, value = self.net(x)

        self.assertEqual(action_logits.shape, (batch_size, 2))
        self.assertEqual(value.shape, (batch_size, 1))

    def test_forward_output_range(self):
        """Test that forward pass produces valid logits and values."""
        x = torch.randn(16, self.state_dim, device=self.device)
        action_logits, value = self.net(x)

        # Logits should be finite
        self.assertTrue(torch.isfinite(action_logits).all())
        self.assertTrue(torch.isfinite(value).all())

    def test_get_importance_score(self):
        """Test that importance scores are in [0, 1]."""
        x = torch.randn(10, self.state_dim, device=self.device)
        scores = self.net.get_importance_score(x)

        self.assertEqual(scores.shape, (10,))
        self.assertTrue((scores >= 0).all() and (scores <= 1).all())

    def test_get_importance_score_single(self):
        """Test importance score for a single state."""
        x = torch.randn(self.state_dim, device=self.device)
        score = self.net.get_importance_score(x.unsqueeze(0))

        self.assertEqual(score.shape, (1,))
        self.assertTrue(0 <= score.item() <= 1)

    def test_sample_action(self):
        """Test action sampling returns valid shapes."""
        x = torch.randn(4, self.state_dim, device=self.device)
        action, log_prob, entropy = self.net.sample_action(x)

        # action: (batch,) with values in {0, 1}
        self.assertEqual(action.shape, (4,))
        self.assertTrue(((action == 0) | (action == 1)).all())

        # log_prob: (batch,)
        self.assertEqual(log_prob.shape, (4,))

        # entropy: (batch,)
        self.assertEqual(entropy.shape, (4,))

    def test_evaluate_actions(self):
        """Test evaluate_actions returns correct shapes."""
        x = torch.randn(8, self.state_dim, device=self.device)
        actions = torch.randint(0, 2, (8,), device=self.device)

        log_prob, entropy, value = self.net.evaluate_actions(x, actions)

        self.assertEqual(log_prob.shape, (8,))
        self.assertEqual(entropy.shape, (8,))
        self.assertEqual(value.shape, (8, 1))

    def test_save_load(self):
        """Test save and load functionality."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mask_net.pt")
            self.net.save(path)

            # Load into a new network
            loaded = MaskNetwork(
                state_dim=self.state_dim,
                hidden_sizes=self.hidden_sizes,
                activation="tanh",
            ).to(self.device)
            loaded.load(path, device=self.device)

            # Check that parameters match
            for p1, p2 in zip(self.net.parameters(), loaded.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

    def test_to_device(self):
        """Test moving network to a device."""
        if torch.cuda.is_available():
            net_cuda = self.net.to("cuda")
            x = torch.randn(4, self.state_dim, device="cuda")
            action_logits, value = net_cuda(x)
            self.assertEqual(action_logits.device.type, "cuda")
            self.assertEqual(value.device.type, "cuda")


class TestPerturbedPolicy(unittest.TestCase):
    """Test the PerturbedPolicy class."""

    def setUp(self):
        self.state_dim = 4
        self.action_dim = 2
        self.device = "cpu"

        # Create a simple mask network
        self.mask_net = MaskNetwork(
            state_dim=self.state_dim,
            hidden_sizes=(32, 32),
            activation="tanh",
        ).to(self.device)

        # Create a dummy target policy
        def target_policy(state):
            # Returns (action, log_prob, value, entropy)
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()
            action = torch.randn(state.shape[0], self.action_dim) if state.dim() > 1 else torch.randn(self.action_dim)
            return (
                action.numpy() if isinstance(action, torch.Tensor) else action,
                0.0,
                0.0,
                0.0,
            )

        self.target_policy = target_policy
        self.action_low = -np.ones(self.action_dim)
        self.action_high = np.ones(self.action_dim)

        self.perturbed = PerturbedPolicy(
            mask_network=self.mask_net,
            target_policy=self.target_policy,
            action_space_low=self.action_low,
            action_space_high=self.action_high,
            discrete_action=False,
            device=self.device,
        )

    def test_get_action_continuous(self):
        """Test get_action for continuous action space."""
        state = np.random.randn(self.state_dim).astype(np.float32)
        action, mask_action, importance, log_prob_mask = self.perturbed.get_action(state)

        self.assertEqual(action.shape, (self.action_dim,))
        self.assertIn(mask_action, [0, 1])
        self.assertTrue(0 <= importance <= 1)
        self.assertIsInstance(log_prob_mask, float)

    def test_get_action_deterministic(self):
        """Test deterministic action selection."""
        state = np.random.randn(self.state_dim).astype(np.float32)
        action, mask_action, importance, log_prob_mask = self.perturbed.get_action(
            state, deterministic=True
        )

        self.assertEqual(action.shape, (self.action_dim,))
        # Deterministic: mask_action should be 0 (keep target action)
        self.assertEqual(mask_action, 0)

    def test_get_importance_score(self):
        """Test importance score retrieval."""
        state = np.random.randn(self.state_dim).astype(np.float32)
        score = self.perturbed.get_importance_score(state)

        self.assertIsInstance(score, float)
        self.assertTrue(0 <= score <= 1)

    def test_discrete_action_mode(self):
        """Test with discrete action space."""
        perturbed_disc = PerturbedPolicy(
            mask_network=self.mask_net,
            target_policy=self.target_policy,
            action_space_low=np.array([0]),
            action_space_high=np.array([0]),
            discrete_action=True,
            num_discrete_actions=4,
            device=self.device,
        )

        state = np.random.randn(self.state_dim).astype(np.float32)
        action, mask_action, importance, log_prob_mask = perturbed_disc.get_action(state)

        # Action should be an integer in [0, num_discrete_actions)
        self.assertIsInstance(action, (int, np.integer))
        self.assertTrue(0 <= action < 4)


class TestMaskNetworkTrainer(unittest.TestCase):
    """Test the MaskNetworkTrainer class."""

    def setUp(self):
        self.state_dim = 4
        self.action_dim = 2
        self.device = "cpu"

        # Create a simple environment
        self.env = gym.make("CartPole-v1")

        # Create mask network
        self.mask_net = MaskNetwork(
            state_dim=self.state_dim,
            hidden_sizes=(32, 32),
            activation="tanh",
        ).to(self.device)

        # Dummy target policy
        def target_policy(state):
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()
            action = torch.randn(state.shape[0], self.action_dim) if state.dim() > 1 else torch.randn(self.action_dim)
            return (
                action.numpy() if isinstance(action, torch.Tensor) else action,
                0.0,
                0.0,
                0.0,
            )

        self.target_policy = target_policy

        self.trainer = MaskNetworkTrainer(
            mask_network=self.mask_net,
            target_policy=self.target_policy,
            env=self.env,
            alpha=0.0001,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            learning_rate=3e-4,
            ppo_epochs=10,
            batch_size=64,
            device=self.device,
            action_space_low=-np.ones(self.action_dim),
            action_space_high=np.ones(self.action_dim),
            discrete_action=False,
        )

    def test_initialization(self):
        """Test that trainer initializes correctly."""
        self.assertIsInstance(self.trainer, MaskNetworkTrainer)
        self.assertIsNotNone(self.trainer.optimizer)

    def test_collect_trajectories(self):
        """Test trajectory collection."""
        buffer = self.trainer.collect_trajectories(num_steps=256)

        self.assertIsNotNone(buffer)
        self.assertGreater(len(buffer), 0)

        data = buffer.get_all()
        self.assertIn("states", data)
        self.assertIn("actions", data)
        self.assertIn("rewards", data)
        self.assertIn("dones", data)
        self.assertIn("values", data)
        self.assertIn("log_probs", data)
        self.assertIn("masks", data)

    def test_update(self):
        """Test PPO update step."""
        # Collect some data first
        buffer = self.trainer.collect_trajectories(num_steps=256)

        # Run update
        stats = self.trainer.update(buffer)

        self.assertIsInstance(stats, dict)
        self.assertIn("policy_loss", stats)
        self.assertIn("value_loss", stats)
        self.assertIn("entropy", stats)

    def test_train_short(self):
        """Test a short training run."""
        history = self.trainer.train(
            total_steps=512,
            steps_per_iteration=128,
            eval_interval=100,
            eval_episodes=2,
            verbose=False,
        )

        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)

    def test_save_load_trainer(self):
        """Test save/load for trainer."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trainer_checkpoint.pt")
            self.trainer.save(path)

            # Create a new trainer and load
            new_net = MaskNetwork(
                state_dim=self.state_dim,
                hidden_sizes=(32, 32),
                activation="tanh",
            ).to(self.device)

            new_trainer = MaskNetworkTrainer(
                mask_network=new_net,
                target_policy=self.target_policy,
                env=self.env,
                alpha=0.0001,
                device=self.device,
                action_space_low=-np.ones(self.action_dim),
                action_space_high=np.ones(self.action_dim),
                discrete_action=False,
            )
            new_trainer.load(path, device=self.device)

            # Check that mask network parameters match
            for p1, p2 in zip(self.trainer.mask_network.parameters(),
                              new_trainer.mask_network.parameters()):
                self.assertTrue(torch.allclose(p1, p2))


class TestFidelity(unittest.TestCase):
    """Test fidelity computation functions."""

    def setUp(self):
        self.state_dim = 4
        self.device = "cpu"
        self.mask_net = MaskNetwork(
            state_dim=self.state_dim,
            hidden_sizes=(32, 32),
            activation="tanh",
        ).to(self.device)

    def test_compute_fidelity_shape(self):
        """Test that compute_fidelity returns a float."""
        states = np.random.randn(100, self.state_dim).astype(np.float32)
        q_values = np.random.randn(100, 2).astype(np.float32)  # 2 actions

        fidelity = compute_fidelity(self.mask_net, states, q_values, device=self.device)

        self.assertIsInstance(fidelity, float)
        # Fidelity is a Pearson correlation, should be in [-1, 1]
        self.assertTrue(-1.0 <= fidelity <= 1.0)

    def test_compute_fidelity_perfect_correlation(self):
        """Test fidelity with perfectly correlated data."""
        # Create states
        states = np.random.randn(50, self.state_dim).astype(np.float32)

        # Create Q-values where Q-diff perfectly correlates with importance
        # We'll set importance scores first, then create Q-values to match
        with torch.no_grad():
            scores = self.mask_net.get_importance_score(
                torch.from_numpy(states).float().to(self.device)
            ).cpu().numpy()

        # Q_diff = 2 * score - 1 (perfect linear correlation)
        q_diff = 2 * scores - 1
        q_values = np.zeros((50, 2), dtype=np.float32)
        q_values[:, 0] = q_diff  # Q(s, a0) - E[Q] proportional to score
        q_values[:, 1] = -q_diff

        fidelity = compute_fidelity(self.mask_net, states, q_values, device=self.device)

        # Should have positive correlation
        self.assertGreater(fidelity, 0.0)

    def test_compute_fidelity_from_env(self):
        """Test environment-based fidelity computation."""
        env = gym.make("CartPole-v1")

        def target_policy(state):
            return env.action_space.sample(), 0.0, 0.0, 0.0

        fidelity = compute_fidelity_from_env(
            self.mask_net,
            env,
            target_policy,
            num_episodes=3,
            device=self.device,
        )

        self.assertIsInstance(fidelity, float)
        self.assertTrue(-1.0 <= fidelity <= 1.0)


class TestTrainMaskNetwork(unittest.TestCase):
    """Test the convenience function train_mask_network."""

    def test_train_mask_network_short(self):
        """Test a short training run via the convenience function."""
        env = gym.make("CartPole-v1")

        def target_policy(state):
            return env.action_space.sample(), 0.0, 0.0, 0.0

        mask_net, trainer = train_mask_network(
            env=env,
            target_policy=target_policy,
            state_dim=4,
            total_steps=256,
            alpha=0.0001,
            steps_per_iteration=128,
            eval_interval=100,
            eval_episodes=2,
            device="cpu",
            verbose=False,
            discrete_action=True,
            num_discrete_actions=env.action_space.n,
        )

        self.assertIsInstance(mask_net, MaskNetwork)
        self.assertIsInstance(trainer, MaskNetworkTrainer)


if __name__ == "__main__":
    unittest.main()
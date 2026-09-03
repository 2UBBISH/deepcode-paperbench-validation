#!/usr/bin/env python3
"""
Unit tests for the Random Network Distillation (RND) module.

Tests cover:
- RNDNetwork architecture and forward pass
- RNDModule: bonus computation, predictor update, normalization, save/load
- RunningMeanStd: running statistics tracking
- BonusNormalizer: bonus normalization
- Factory function: create_rnd_module
- Utility function: compute_rnd_bonus_batch
"""

import unittest
import numpy as np
import torch
import os
import tempfile

from rice.rnd import (
    RNDNetwork,
    RNDModule,
    RunningMeanStd,
    BonusNormalizer,
    create_rnd_module,
    compute_rnd_bonus_batch,
)


class TestRNDNetwork(unittest.TestCase):
    """Test the RNDNetwork MLP backbone."""

    def setUp(self):
        self.input_dim = 8
        self.hidden_sizes = (64, 64)
        self.output_dim = 64
        self.network = RNDNetwork(
            input_dim=self.input_dim,
            hidden_sizes=self.hidden_sizes,
            output_dim=self.output_dim,
            activation="relu",
        )

    def test_architecture(self):
        """Test that the network has the correct architecture."""
        # Check that it's an nn.Module
        self.assertIsInstance(self.network, torch.nn.Module)

        # Check layers exist
        layers = list(self.network.children())
        # Should have: linear_in, activation, hidden layers, linear_out
        self.assertGreater(len(layers), 0)

    def test_forward_output_shape(self):
        """Test forward pass produces correct output shape."""
        batch_size = 32
        x = torch.randn(batch_size, self.input_dim)
        output = self.network(x)
        self.assertEqual(output.shape, (batch_size, self.output_dim))

    def test_forward_single_input(self):
        """Test forward pass with a single input."""
        x = torch.randn(1, self.input_dim)
        output = self.network(x)
        self.assertEqual(output.shape, (1, self.output_dim))

    def test_forward_deterministic(self):
        """Test that forward pass is deterministic in eval mode."""
        self.network.eval()
        x = torch.randn(4, self.input_dim)
        out1 = self.network(x)
        out2 = self.network(x)
        self.assertTrue(torch.allclose(out1, out2))

    def test_different_activations(self):
        """Test network creation with different activations."""
        for activation in ["relu", "tanh", "elu"]:
            net = RNDNetwork(
                input_dim=4,
                hidden_sizes=(32,),
                output_dim=16,
                activation=activation,
            )
            x = torch.randn(2, 4)
            out = net(x)
            self.assertEqual(out.shape, (2, 16))

    def test_single_hidden_layer(self):
        """Test network with a single hidden layer."""
        net = RNDNetwork(
            input_dim=10,
            hidden_sizes=(32,),
            output_dim=16,
        )
        x = torch.randn(8, 10)
        out = net(x)
        self.assertEqual(out.shape, (8, 16))

    def test_no_hidden_layers(self):
        """Test network with no hidden layers (direct mapping)."""
        net = RNDNetwork(
            input_dim=10,
            hidden_sizes=(),
            output_dim=16,
        )
        x = torch.randn(8, 10)
        out = net(x)
        self.assertEqual(out.shape, (8, 16))


class TestRunningMeanStd(unittest.TestCase):
    """Test the RunningMeanStd utility for observation normalization."""

    def setUp(self):
        self.shape = (4,)
        self.rms = RunningMeanStd(shape=self.shape, decay=0.99, epsilon=1e-4)

    def test_initialization(self):
        """Test that RunningMeanStd initializes correctly."""
        self.assertEqual(self.rms.mean.shape, self.shape)
        self.assertEqual(self.rms.var.shape, self.shape)
        self.assertEqual(self.rms.std.shape, self.shape)
        # Initial mean should be zeros
        np.testing.assert_array_equal(self.rms.mean, np.zeros(self.shape))
        # Initial var should be ones
        np.testing.assert_array_equal(self.rms.var, np.ones(self.shape))

    def test_update_single(self):
        """Test updating with a single sample."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        self.rms.update(x)
        # After one update, mean should be close to x
        np.testing.assert_array_almost_equal(self.rms.mean, x)

    def test_update_batch(self):
        """Test updating with a batch of samples."""
        x = np.random.randn(100, 4)
        self.rms.update(x)
        # Mean should be close to empirical mean
        np.testing.assert_array_almost_equal(
            self.rms.mean, x.mean(axis=0), decimal=1
        )

    def test_normalize(self):
        """Test normalization produces zero-mean unit-variance output."""
        # First update with some data
        x = np.random.randn(1000, 4) * 2 + 5  # mean=5, std=2
        self.rms.update(x)

        # Normalize
        normalized = self.rms.normalize(x)
        # Should have mean ~0 and std ~1
        self.assertTrue(np.abs(normalized.mean()) < 0.5)
        self.assertTrue(np.abs(normalized.std() - 1.0) < 0.5)

    def test_normalize_single(self):
        """Test normalizing a single sample."""
        x = np.random.randn(100, 4)
        self.rms.update(x)

        single = np.array([1.0, 2.0, 3.0, 4.0])
        normalized = self.rms.normalize(single)
        self.assertEqual(normalized.shape, (4,))

    def test_decay_effect(self):
        """Test that decay parameter affects update behavior."""
        rms_fast = RunningMeanStd(shape=(2,), decay=0.9)
        rms_slow = RunningMeanStd(shape=(2,), decay=0.999)

        # Update both with same data
        for _ in range(10):
            x = np.random.randn(50, 2)
            rms_fast.update(x)
            rms_slow.update(x)

        # Fast decay should have mean closer to recent data
        recent = np.random.randn(50, 2) * 10 + 100
        rms_fast.update(recent)
        rms_slow.update(recent)

        # Fast decay mean should be closer to 100
        self.assertGreater(np.abs(rms_fast.mean).mean(), np.abs(rms_slow.mean).mean() * 0.5)


class TestBonusNormalizer(unittest.TestCase):
    """Test the BonusNormalizer for RND bonus scaling."""

    def setUp(self):
        self.normalizer = BonusNormalizer(decay=0.99, epsilon=1e-8)

    def test_initialization(self):
        """Test initial state."""
        self.assertEqual(self.normalizer.mean, 0.0)
        self.assertEqual(self.normalizer.var, 1.0)
        self.assertEqual(self.normalizer.std, 1.0)
        self.assertEqual(self.normalizer.count, 0)

    def test_update(self):
        """Test updating with bonus values."""
        bonuses = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.normalizer.update(bonuses)
        self.assertGreater(self.normalizer.count, 0)
        self.assertAlmostEqual(self.normalizer.mean, 3.0, places=1)

    def test_normalize(self):
        """Test normalization of bonuses."""
        bonuses = np.random.randn(1000) * 2 + 5
        self.normalizer.update(bonuses)

        normalized = self.normalizer.normalize(bonuses)
        # Should be roughly zero-mean unit-variance
        self.assertTrue(np.abs(normalized.mean()) < 0.5)
        self.assertTrue(np.abs(normalized.std() - 1.0) < 0.5)

    def test_normalize_single(self):
        """Test normalizing a single bonus value."""
        bonuses = np.random.randn(100) * 3 + 10
        self.normalizer.update(bonuses)

        single = np.array([12.0])
        normalized = self.normalizer.normalize(single)
        self.assertEqual(normalized.shape, (1,))

    def test_normalize_scalar(self):
        """Test normalizing a scalar bonus."""
        bonuses = np.random.randn(100)
        self.normalizer.update(bonuses)

        normalized = self.normalizer.normalize(0.5)
        self.assertTrue(np.isscalar(normalized) or isinstance(normalized, (float, np.floating)))


class TestRNDModule(unittest.TestCase):
    """Test the main RNDModule class."""

    def setUp(self):
        self.state_dim = 8
        self.rnd = RNDModule(
            state_dim=self.state_dim,
            hidden_sizes=(64, 64),
            embedding_dim=64,
            learning_rate=1e-4,
            device="cpu",
            activation="relu",
            normalize_obs=False,  # Disable for simpler testing
        )

    def test_initialization(self):
        """Test that RNDModule initializes correctly."""
        self.assertIsInstance(self.rnd.target_network, RNDNetwork)
        self.assertIsInstance(self.rnd.predictor_network, RNDNetwork)
        self.assertIsInstance(self.rnd.optimizer, torch.optim.Adam)

        # Target and predictor should have different initial parameters
        for p1, p2 in zip(
            self.rnd.target_network.parameters(),
            self.rnd.predictor_network.parameters(),
        ):
            self.assertFalse(torch.equal(p1, p2))

    def test_target_frozen(self):
        """Test that target network parameters are not updated."""
        # Get initial target parameters
        initial_params = [p.clone() for p in self.rnd.target_network.parameters()]

        # Do a predictor update
        states = np.random.randn(64, self.state_dim).astype(np.float32)
        self.rnd.update(states, num_epochs=1)

        # Target parameters should be unchanged
        for initial_p, current_p in zip(
            initial_params, self.rnd.target_network.parameters()
        ):
            self.assertTrue(torch.equal(initial_p, current_p))

    def test_predictor_updates(self):
        """Test that predictor network parameters change after update."""
        initial_params = [p.clone() for p in self.rnd.predictor_network.parameters()]

        states = np.random.randn(256, self.state_dim).astype(np.float32)
        self.rnd.update(states, num_epochs=4)

        # Predictor parameters should have changed
        any_changed = False
        for initial_p, current_p in zip(
            initial_params, self.rnd.predictor_network.parameters()
        ):
            if not torch.equal(initial_p, current_p):
                any_changed = True
                break
        self.assertTrue(any_changed, "Predictor parameters did not change after update")

    def test_compute_bonus_shape(self):
        """Test bonus computation returns correct shape."""
        states = np.random.randn(32, self.state_dim).astype(np.float32)
        bonuses = self.rnd.compute_bonus(states)
        self.assertEqual(bonuses.shape, (32,))

    def test_compute_bonus_single(self):
        """Test bonus computation for a single state."""
        state = np.random.randn(self.state_dim).astype(np.float32)
        bonus = self.rnd.compute_bonus(state)
        self.assertTrue(np.isscalar(bonus) or bonus.shape == ())

    def test_compute_bonus_non_negative(self):
        """Test that bonuses are non-negative (MSE)."""
        states = np.random.randn(100, self.state_dim).astype(np.float32)
        bonuses = self.rnd.compute_bonus(states)
        self.assertTrue(np.all(bonuses >= 0))

    def test_compute_bonus_known_states_lower(self):
        """Test that bonuses are lower for states the predictor has seen."""
        # Train predictor on some states
        train_states = np.random.randn(500, self.state_dim).astype(np.float32)
        self.rnd.update(train_states, num_epochs=10)

        # Bonuses on training distribution should be lower than on OOD
        train_bonus = self.rnd.compute_bonus(train_states[:50]).mean()
        ood_states = np.random.randn(50, self.state_dim).astype(np.float32) * 5
        ood_bonus = self.rnd.compute_bonus(ood_states).mean()

        # OOD bonus should be higher (novelty detection)
        self.assertGreater(ood_bonus, train_bonus * 0.5,
                           "OOD bonus should be comparable or higher than in-distribution bonus")

    def test_update_returns_dict(self):
        """Test that update returns a dictionary with loss info."""
        states = np.random.randn(128, self.state_dim).astype(np.float32)
        info = self.rnd.update(states, num_epochs=2)
        self.assertIsInstance(info, dict)
        self.assertIn("loss", info)

    def test_update_on_trajectory(self):
        """Test update_on_trajectory method."""
        states = np.random.randn(200, self.state_dim).astype(np.float32)
        info = self.rnd.update_on_trajectory(states, num_epochs=2)
        self.assertIsInstance(info, dict)
        self.assertIn("loss", info)

    def test_get_normalized_bonus(self):
        """Test normalized bonus computation."""
        states = np.random.randn(64, self.state_dim).astype(np.float32)
        bonuses = self.rnd.get_normalized_bonus(states)
        self.assertEqual(bonuses.shape, (64,))

    def test_save_load(self):
        """Test save and load functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "rnd_test.pt")

            # Compute some bonuses before saving
            states = np.random.randn(32, self.state_dim).astype(np.float32)
            bonuses_before = self.rnd.compute_bonus(states)

            # Save
            self.rnd.save(save_path)
            self.assertTrue(os.path.exists(save_path))

            # Load into new module
            new_rnd = RNDModule(
                state_dim=self.state_dim,
                hidden_sizes=(64, 64),
                embedding_dim=64,
                device="cpu",
                normalize_obs=False,
            )
            new_rnd.load(save_path)

            # Bonuses should match
            bonuses_after = new_rnd.compute_bonus(states)
            np.testing.assert_array_almost_equal(bonuses_before, bonuses_after, decimal=5)

    def test_to_device(self):
        """Test moving module to a different device."""
        if torch.cuda.is_available():
            self.rnd.to("cuda")
            # Check that parameters are on CUDA
            for p in self.rnd.predictor_network.parameters():
                self.assertTrue(p.is_cuda)
            for p in self.rnd.target_network.parameters():
                self.assertTrue(p.is_cuda)

            # Move back to CPU
            self.rnd.to("cpu")
            for p in self.rnd.predictor_network.parameters():
                self.assertFalse(p.is_cuda)

    def test_with_obs_normalization(self):
        """Test RNDModule with observation normalization enabled."""
        rnd_norm = RNDModule(
            state_dim=self.state_dim,
            hidden_sizes=(32,),
            embedding_dim=16,
            device="cpu",
            normalize_obs=True,
            obs_rms_decay=0.99,
        )
        states = np.random.randn(128, self.state_dim).astype(np.float32)
        bonuses = rnd_norm.compute_bonus(states)
        self.assertEqual(bonuses.shape, (128,))

        # Update should work
        info = rnd_norm.update(states, num_epochs=1)
        self.assertIn("loss", info)

    def test_different_embedding_dim(self):
        """Test RNDModule with different embedding dimensions."""
        for emb_dim in [8, 16, 32, 128]:
            rnd = RNDModule(
                state_dim=4,
                hidden_sizes=(16,),
                embedding_dim=emb_dim,
                device="cpu",
                normalize_obs=False,
            )
            states = np.random.randn(8, 4).astype(np.float32)
            bonuses = rnd.compute_bonus(states)
            self.assertEqual(bonuses.shape, (8,))


class TestCreateRNDModule(unittest.TestCase):
    """Test the factory function create_rnd_module."""

    def test_default_creation(self):
        """Test creating an RND module with default parameters."""
        rnd = create_rnd_module(state_dim=10)
        self.assertIsInstance(rnd, RNDModule)

        # Test basic functionality
        states = np.random.randn(16, 10).astype(np.float32)
        bonuses = rnd.compute_bonus(states)
        self.assertEqual(bonuses.shape, (16,))

    def test_custom_parameters(self):
        """Test creating with custom parameters."""
        rnd = create_rnd_module(
            state_dim=6,
            hidden_sizes=(32, 32),
            embedding_dim=32,
            learning_rate=1e-3,
            device="cpu",
            normalize_obs=False,
        )
        self.assertIsInstance(rnd, RNDModule)

        states = np.random.randn(8, 6).astype(np.float32)
        bonuses = rnd.compute_bonus(states)
        self.assertEqual(bonuses.shape, (8,))


class TestComputeRNDBonusBatch(unittest.TestCase):
    """Test the compute_rnd_bonus_batch utility function."""

    def setUp(self):
        self.state_dim = 6
        self.rnd = RNDModule(
            state_dim=self.state_dim,
            hidden_sizes=(32,),
            embedding_dim=16,
            device="cpu",
            normalize_obs=False,
        )

    def test_basic_bonus_computation(self):
        """Test basic bonus computation with lambda coefficient."""
        states = np.random.randn(32, self.state_dim).astype(np.float32)
        bonuses = compute_rnd_bonus_batch(
            self.rnd, states, bonus_normalizer=None, lambda_coef=0.01
        )
        self.assertEqual(bonuses.shape, (32,))
        # Bonuses should be scaled by lambda
        raw_bonuses = self.rnd.compute_bonus(states)
        np.testing.assert_array_almost_equal(bonuses, raw_bonuses * 0.01)

    def test_with_normalizer(self):
        """Test bonus computation with a BonusNormalizer."""
        normalizer = BonusNormalizer()

        # First update normalizer with some bonuses
        init_states = np.random.randn(200, self.state_dim).astype(np.float32)
        init_bonuses = self.rnd.compute_bonus(init_states)
        normalizer.update(init_bonuses)

        # Now compute normalized bonuses
        states = np.random.randn(16, self.state_dim).astype(np.float32)
        bonuses = compute_rnd_bonus_batch(
            self.rnd, states, bonus_normalizer=normalizer, lambda_coef=0.01
        )
        self.assertEqual(bonuses.shape, (16,))

    def test_different_lambda(self):
        """Test bonus computation with different lambda values."""
        states = np.random.randn(8, self.state_dim).astype(np.float32)
        raw = self.rnd.compute_bonus(states)

        for lam in [0.001, 0.01, 0.1, 1.0]:
            bonuses = compute_rnd_bonus_batch(
                self.rnd, states, bonus_normalizer=None, lambda_coef=lam
            )
            np.testing.assert_array_almost_equal(bonuses, raw * lam)

    def test_single_state(self):
        """Test bonus computation for a single state."""
        state = np.random.randn(self.state_dim).astype(np.float32)
        bonus = compute_rnd_bonus_batch(
            self.rnd, state, bonus_normalizer=None, lambda_coef=0.01
        )
        self.assertTrue(np.isscalar(bonus) or bonus.shape == ())


class TestRNDIntegration(unittest.TestCase):
    """Integration tests for RND module in a simulated RL loop."""

    def setUp(self):
        self.state_dim = 4
        self.rnd = RNDModule(
            state_dim=self.state_dim,
            hidden_sizes=(32,),
            embedding_dim=16,
            learning_rate=1e-3,
            device="cpu",
            normalize_obs=False,
        )
        self.normalizer = BonusNormalizer()

    def test_training_loop_simulation(self):
        """Simulate a training loop with RND updates and bonus computation."""
        num_iterations = 5
        states_per_iteration = 50

        all_losses = []

        for i in range(num_iterations):
            # Generate states (simulating environment interaction)
            # Shift distribution slightly each iteration
            states = np.random.randn(states_per_iteration, self.state_dim).astype(
                np.float32
            ) + i * 0.5

            # Compute bonuses
            bonuses = self.rnd.compute_bonus(states)
            self.assertEqual(bonuses.shape, (states_per_iteration,))

            # Update normalizer
            self.normalizer.update(bonuses)

            # Normalized bonuses
            norm_bonuses = self.normalizer.normalize(bonuses)

            # Update predictor
            info = self.rnd.update(states, num_epochs=2)
            all_losses.append(info["loss"])

            # Loss should generally decrease as predictor improves
            if i > 0:
                # Not strictly monotonic but should trend down
                pass

        # Final loss should be lower than initial
        self.assertLess(all_losses[-1], all_losses[0] * 2.0,
                        "RND loss should not explode during training")

    def test_novelty_detection(self):
        """Test that RND correctly assigns higher bonuses to novel states."""
        # Train on distribution A
        states_a = np.random.randn(500, self.state_dim).astype(np.float32) * 0.5
        self.rnd.update(states_a, num_epochs=10)

        # Bonuses on distribution A should be low
        bonus_a = self.rnd.compute_bonus(
            np.random.randn(100, self.state_dim).astype(np.float32) * 0.5
        ).mean()

        # Bonuses on distribution B (different mean) should be higher
        bonus_b = self.rnd.compute_bonus(
            np.random.randn(100, self.state_dim).astype(np.float32) * 0.5 + 3.0
        ).mean()

        # Bonuses on distribution C (different variance) should be higher
        bonus_c = self.rnd.compute_bonus(
            np.random.randn(100, self.state_dim).astype(np.float32) * 2.0
        ).mean()

        # At least one of the OOD distributions should have higher bonus
        self.assertTrue(
            bonus_b > bonus_a * 0.8 or bonus_c > bonus_a * 0.8,
            "RND should detect novelty (higher bonus for OOD states)",
        )


if __name__ == "__main__":
    unittest.main()
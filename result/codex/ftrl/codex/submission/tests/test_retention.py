"""Unit tests for the knowledge-retention methods."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fpc.retention.base import RetentionConfig
from fpc.retention.distillation import Kickstarting, kl_divergence
from fpc.retention.episodic_memory import ProtectedReplayBuffer
from fpc.retention.ewc import EWC
from fpc.retention.fisher import DiagonalFisher
from fpc.retention.schedules import ExponentialDecaySchedule


def test_kl_direction_and_symmetry():
    identical = torch.tensor([[1.0, 2.0, 3.0]])
    assert kl_divergence(identical, identical, "forward").item() == pytest.approx(0.0, abs=1e-6)
    teacher = torch.tensor([[3.0, 0.0, 0.0]])
    student = torch.tensor([[0.0, 3.0, 0.0]])
    assert kl_divergence(teacher, student, "forward").item() > 0.5
    with pytest.raises(ValueError):
        kl_divergence(teacher, student, "sideways")


def test_ks_decay_schedule():
    ks = Kickstarting(RetentionConfig(method="ks", coefficient=0.5, decay=0.99998))
    for _ in range(100):
        ks.on_train_step()
    expected = 0.5 * (0.99998 ** 100)
    assert ks.coefficient == pytest.approx(expected, rel=1e-6)


def test_exponential_decay_schedule():
    schedule = ExponentialDecaySchedule(initial=1.0, decay=0.5)
    assert schedule.value(0) == 1.0
    assert schedule.value(3) == pytest.approx(0.125)


def test_ewc_penalty_is_zero_at_anchor_and_grows():
    net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    ewc = EWC(RetentionConfig(method="ewc", coefficient=1.0))
    ewc.register_anchor(net)

    def log_prob(*_, **__):
        dist = torch.distributions.Categorical(logits=net(torch.randn(8, 4)))
        return dist.log_prob(dist.sample())

    fisher = DiagonalFisher.estimate(net, log_prob, [()] * 10, num_batches=10)
    ewc.set_fisher(fisher)
    assert float(ewc.aux_loss(net)) == pytest.approx(0.0, abs=1e-12)
    with torch.no_grad():
        net[0].weight.add_(1.0)
    assert float(ewc.aux_loss(net)) > 0.0


def test_fisher_matches_manual_computation():
    torch.manual_seed(0)
    net = torch.nn.Linear(3, 2, bias=False)
    observations = torch.randn(32, 3)

    def log_prob(observations=observations, **_):
        dist = torch.distributions.Categorical(logits=net(observations))
        return dist.log_prob(dist.sample())

    fisher = DiagonalFisher.estimate(net, log_prob, [()] * 200, num_batches=200)
    assert set(fisher.keys()) == {"weight"}
    assert torch.all(fisher["weight"] >= 0)


def test_protected_replay_buffer_never_overwrites_old_data():
    buffer = ProtectedReplayBuffer(capacity=50, protected_size=10, seed=0)
    protected = {"obs": np.arange(10 * 4, dtype=np.float32).reshape(10, 4)}
    buffer.add_protected(protected)
    for i in range(200):
        buffer.add({"obs": np.full(4, -1.0, dtype=np.float32)})
    assert np.array_equal(buffer._storage["obs"][:10], protected["obs"])
    assert buffer.size == 50
    sample = buffer.sample(16)
    assert sample["obs"].shape == (16, 4)


def test_retention_config_rejects_unknown_method():
    with pytest.raises(ValueError):
        RetentionConfig(method="magic")

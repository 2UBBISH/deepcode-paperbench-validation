#!/usr/bin/env python3
"""End-to-end smoke test of every training loop with mock environments.

Runs in a few seconds on CPU and checks that the three trainers (APPO/NetHack,
PPO+RND/Montezuma, SAC/RoboticSequence), the four knowledge-retention methods
(EWC, BC, KS, EM) and the toy environments all execute correctly.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def smoke_retention() -> None:
    from fpc.retention.base import RetentionConfig
    from fpc.retention.distillation import Kickstarting, kl_divergence
    from fpc.retention.episodic_memory import ProtectedReplayBuffer
    from fpc.retention.ewc import EWC
    from fpc.retention.fisher import DiagonalFisher

    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))
    ewc = EWC(RetentionConfig(method="ewc", coefficient=2e6))
    ewc.register_anchor(net)

    def log_prob(observations=None, **_):
        observations = torch.randn(16, 4) if observations is None else observations
        dist = torch.distributions.Categorical(logits=net(observations))
        return dist.log_prob(dist.sample())

    fisher = DiagonalFisher.estimate(net, log_prob, [()] * 20, num_batches=20)
    ewc.set_fisher(fisher)
    loss = float(ewc.aux_loss(net))
    assert loss >= 0.0

    ks = Kickstarting(RetentionConfig(method="ks", coefficient=0.5, decay=0.5))
    assert ks.coefficient == 0.5
    ks.on_train_step()
    assert abs(ks.coefficient - 0.25) < 1e-9

    teacher = torch.tensor([[2.0, 0.0, 0.0]])
    student = torch.tensor([[0.0, 2.0, 0.0]])
    assert kl_divergence(teacher, student, "forward").item() > 0

    buffer = ProtectedReplayBuffer(capacity=100, protected_size=10)
    buffer.add_protected({"obs": np.zeros((10, 4), dtype=np.float32), "act": np.ones(10, dtype=np.int64)})
    for i in range(50):
        buffer.add({"obs": np.full(4, i, dtype=np.float32), "act": np.array(0, dtype=np.int64)})
    assert np.allclose(buffer._storage["obs"][:10], 0.0), "protected region was overwritten"
    print("[ok] knowledge-retention core (EWC / KS / BC-KL / EM)")


def smoke_montezuma() -> None:
    from fpc.montezuma.config import MontezumaConfig
    from fpc.montezuma.model import AtariActorCritic
    from fpc.montezuma.ppo import BehavioralCloningBuffer, PPORNDTrainer
    from fpc.retention.base import RetentionConfig
    from fpc.testing.mock_envs import MockAtariVecEnv

    for method in ("none", "bc", "ewc", "ks"):
        config = MontezumaConfig(num_env=2, num_step=8, mini_batch=2, epoch=1, device="cpu")
        config.retention = RetentionConfig(method=method, coefficient=0.05 if method == "ewc" else 0.01)
        teacher = AtariActorCritic(18) if method in ("bc", "ks") else None
        bc = BehavioralCloningBuffer(
            np.random.rand(32, 4, 84, 84).astype(np.float32), np.random.randint(0, 18, 32)
        )
        trainer = PPORNDTrainer(config, MockAtariVecEnv(2), device="cpu", teacher=teacher, bc_buffer=bc)
        history = trainer.train(total_steps=64)
        assert history["steps"], f"no updates for method {method}"
    print("[ok] Montezuma PPO+RND with none / BC / EWC / KS")


def smoke_metaworld() -> None:
    from fpc.metaworld.config import MetaworldConfig
    from fpc.metaworld.sac import SAC, ReplayBuffer, Transition
    from fpc.retention.base import RetentionConfig
    from fpc.retention.episodic_memory import ProtectedReplayBuffer

    obs_dim, act_dim, num_stages = 10, 4, 3
    config = MetaworldConfig()
    teacher = SAC(config, obs_dim, act_dim, num_stages, device="cpu")

    def make_replay(protected=None):
        replay = ReplayBuffer(500, obs_dim, act_dim, protected=protected)
        for i in range(200):
            replay.add(
                Transition(
                    obs=np.random.rand(obs_dim).astype(np.float32),
                    action=np.random.rand(act_dim).astype(np.float32),
                    reward=1.0,
                    next_obs=np.random.rand(obs_dim).astype(np.float32),
                    done=0.0,
                    stage=i % num_stages,
                    next_stage=(i + 1) % num_stages,
                )
            )
        return replay

    for method, coefficient in (("none", 0.0), ("bc", 1.0), ("ewc", 100.0)):
        cfg = MetaworldConfig()
        cfg.retention = RetentionConfig(method=method, coefficient=coefficient)
        bc_dataset = {
            "obs": np.random.rand(256, obs_dim).astype(np.float32),
            "stages": np.random.randint(0, num_stages, 256),
        }
        agent = SAC(cfg, obs_dim, act_dim, num_stages, device="cpu", teacher=teacher, bc_dataset=bc_dataset)
        if method == "ewc":
            from fpc.retention.fisher import DiagonalFisher

            def log_prob(observations, stage, **_):
                _, _, mu, std = agent.actor(observations, stage, with_logprob=False)
                dist = torch.distributions.Normal(mu, std)
                return dist.log_prob(dist.sample()).sum(-1)

            batches = [
                {
                    "observations": torch.as_tensor(np.random.rand(16, obs_dim), dtype=torch.float32),
                    "stage": torch.zeros(16, dtype=torch.long),
                }
                for _ in range(5)
            ]
            agent.retention.set_fisher(DiagonalFisher.estimate(agent.actor, log_prob, batches, num_batches=5))
        for _ in range(10):
            stats = agent.update(make_replay())
        assert np.isfinite(stats["actor_loss"])

    protected = ProtectedReplayBuffer(500, protected_size=50)
    replay = make_replay(protected)
    config_em = MetaworldConfig()
    config_em.retention = RetentionConfig(method="em")
    agent = SAC(config_em, obs_dim, act_dim, num_stages, device="cpu")
    for _ in range(5):
        agent.update(replay)
    print("[ok] Meta-World SAC with none / BC / EWC / EM")


def smoke_nethack() -> None:
    from fpc.nethack.appo import APPO
    from fpc.nethack.config import NetHackConfig
    from fpc.nethack.model import NetHackModel
    from fpc.retention.base import RetentionConfig
    from fpc.testing.mock_envs import MockNetHackEnv

    base = dict(
        num_workers=2, unroll_length=4, batch_size=2, obs_screen_shape=(21, 79),
        resnet_blocks=1, resnet_channels=8, hidden_dim=32, char_embed_dim=4,
        color_embed_dim=2, mlp_hidden=8, freeze_encoders=False,
    )
    for method in ("none", "ks", "bc", "ewc"):
        config = NetHackConfig(**base)
        if method == "ks":
            config.retention = RetentionConfig(method="ks", coefficient=0.5, decay=0.99998)
        elif method == "bc":
            config.retention = RetentionConfig(method="bc", coefficient=2.0)
        elif method == "ewc":
            config.retention = RetentionConfig(method="ewc", coefficient=1.0)
        teacher = NetHackModel(config) if method in ("ks", "bc") else None
        bc_buffer = None
        if method in ("bc", "ewc"):
            bc_buffer = {
                "glyphs": torch.randint(0, 256, (64, 21, 79)),
                "colors": torch.randint(0, 64, (64, 21, 79)),
                "blstats": torch.randn(64, 25),
                "message": torch.randint(0, 256, (64, 256)),
                "actions": torch.randint(0, 120, (64,)),
            }
        appo = APPO(config, lambda seed: MockNetHackEnv(config, seed), device="cpu",
                    teacher=teacher, bc_buffer=bc_buffer)
        history = appo.train(total_steps=64)
        assert history["steps"], f"no APPO updates for method {method}"
        appo.stop_workers()
    print("[ok] NetHack APPO with none / KS / BC / EWC")


def smoke_toy() -> None:
    from fpc.toy.apple_retrieval import AppleRetrieval, pretrain_on_phase2, reinforce
    from fpc.toy.two_state_mdp import run_paper_scenarios

    scenarios = run_paper_scenarios(steps=20_000)
    reported = [s for s in scenarios if s.name == "reported_suboptimal_fixed_point"][0]
    assert abs(reported.converged_theta - 0.1111) < 0.01, reported.converged_theta
    assert abs(reported.converged_value - 2.2222) < 0.05, reported.converged_value

    env = AppleRetrieval(M=30, c=0.3, seed=0)
    policy, _ = pretrain_on_phase2(env, episodes=500, lr=1e-2, seed=0)
    reinforce(env, policy, episodes=1000, lr=1e-2, phase=None, seed=0)
    print("[ok] toy environments (two-state MDPs + AppleRetrieval)")


def main() -> None:
    smoke_retention()
    smoke_montezuma()
    smoke_metaworld()
    smoke_nethack()
    smoke_toy()
    print("\nall smoke tests passed")


if __name__ == "__main__":
    main()

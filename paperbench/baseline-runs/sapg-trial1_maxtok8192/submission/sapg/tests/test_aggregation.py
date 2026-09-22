"""Unit tests for SAPG aggregation strategies.

Validates the core claim of the paper: the leader aggregates ALL transitions
from ALL followers (no data wasted), and the symmetric-aggregation ablation
(no designated leader) is a valid alternative that can be compared against
the leader-based SAPG.

Tests cover:
  * AggregationMode parsing / aliases.
  * AggregatedBuffer concatenation of every follower's transitions.
  * Leader.update consumes the union of all followers' data.
  * SymmetricAggregator updates every worker on the union of all data.
  * Importance weights are computed from behaviour (follower) log-probs.
  * build_aggregator factory dispatch and error handling.

The tests are self-contained (no pytest fixtures required) and can be run
either with pytest or directly via ``python tests/test_aggregation.py``.
"""

from __future__ import annotations

import math
import os
import sys

import torch

# ---------------------------------------------------------------------------
# Make the project root importable when run directly.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sapg.aggregation import (  # noqa: E402
    AggregationMode,
    AggregationStats,
    LeaderAggregator,
    SymmetricAggregator,
    build_aggregator,
)
from sapg.buffer import AggregatedBuffer, RolloutBuffer, build_buffers  # noqa: E402
from sapg.follower import Follower, build_followers  # noqa: E402
from sapg.leader import Leader, build_leader  # noqa: E402
from sapg.networks import GaussianPolicy, ValueNetwork  # noqa: E402
from sapg.ppo import KLScheduler, PPOHyperParams  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _assert(cond, msg=""):
    if not cond:
        raise AssertionError(msg or "assertion failed")


def _assert_close(a, b, tol=1e-5, msg=""):
    if not torch.allclose(torch.as_tensor(a, dtype=torch.float32),
                          torch.as_tensor(b, dtype=torch.float32),
                          rtol=0.0, atol=tol):
        raise AssertionError(f"{msg}: {a} != {b} (tol={tol})")


def _make_networks(obs_dim=8, action_dim=3, phi_dim=4, num_blocks=3, seed=0):
    torch.manual_seed(seed)
    policy = GaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        phi_dim=phi_dim,
        hidden_sizes=(32, 32),
        use_lstm=False,
        learnable_sigma=True,
        per_block_sigma=False,
        num_blocks=num_blocks,
        activation="elu",
    )
    value = ValueNetwork(
        obs_dim=obs_dim,
        phi_dim=phi_dim,
        hidden_sizes=(32, 32),
        use_lstm=False,
        activation="elu",
    )
    return policy, value


def _fill_buffer(buf, num_blocks=1, seed=0, block_id=0, phi=None):
    """Fill a RolloutBuffer with random but valid data."""
    g = torch.Generator().manual_seed(seed)
    T, N = buf.horizon, buf.num_envs
    obs = torch.randn(T, N, buf.obs_dim, generator=g)
    actions = torch.randn(T, N, buf.action_dim, generator=g)
    log_probs = torch.randn(T, N, generator=g)
    rewards = torch.randn(T, N, generator=g)
    values = torch.randn(T, N, generator=g)
    dones = torch.zeros(T, N)
    for t in range(T):
        buf.insert(
            obs=obs[t],
            actions=actions[t],
            log_probs=log_probs[t],
            rewards=rewards[t],
            values=values[t],
            dones=dones[t],
            block_id=block_id,
            phi=phi,
        )
    return buf


def _make_hparams(**overrides):
    hp = PPOHyperParams(
        gamma=0.99,
        tau=0.95,
        clip_eps=0.1,
        entropy_coeff=0.0,
        critic_coeff=4.0,
        bounds_loss_coeff=1e-4,
        kl_threshold=0.016,
        grad_norm_clip=1.0,
        lr=1e-4,
        mini_epochs=1,
        horizon=4,
        lstm_seq_len=4,
        action_bound=1.0,
    )
    for k, v in overrides.items():
        setattr(hp, k, v)
    return hp


# ---------------------------------------------------------------------------
# AggregationMode
# ---------------------------------------------------------------------------
def test_aggregation_mode_parsing():
    _assert(AggregationMode.from_str("leader") is AggregationMode.LEADER)
    _assert(AggregationMode.from_str("sapg") is AggregationMode.LEADER)
    _assert(AggregationMode.from_str("default") is AggregationMode.LEADER)
    _assert(AggregationMode.from_str("symmetric") is AggregationMode.SYMMETRIC)
    _assert(AggregationMode.from_str("sym") is AggregationMode.SYMMETRIC)
    _assert(AggregationMode.from_str("no_leader") is AggregationMode.SYMMETRIC)
    _assert(AggregationMode.from_str("noleader") is AggregationMode.SYMMETRIC)
    # Case-insensitive
    _assert(AggregationMode.from_str("LEADER") is AggregationMode.LEADER)
    _assert(AggregationMode.from_str("Symmetric") is AggregationMode.SYMMETRIC)


def test_aggregation_mode_invalid():
    try:
        AggregationMode.from_str("bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown mode")


# ---------------------------------------------------------------------------
# AggregatedBuffer: no data wasted
# ---------------------------------------------------------------------------
def test_aggregated_buffer_concatenates_all_transitions():
    num_blocks = 3
    horizon, num_envs, obs_dim, action_dim = 4, 5, 8, 3
    buffers = build_buffers(
        num_blocks=num_blocks,
        num_envs_per_block=num_envs,
        horizon=horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        phi_dim=4,
        device="cpu",
    )
    for j, buf in enumerate(buffers):
        _fill_buffer(buf, seed=j, block_id=j)
        buf.compute_advantages(last_values=torch.zeros(num_envs))

    agg = AggregatedBuffer(buffers)
    expected = num_blocks * horizon * num_envs
    _assert(agg.num_transitions == expected,
            f"aggregated transitions {agg.num_transitions} != {expected}")

    # Every block id must be present in the aggregated data.
    seen = set()
    for mb in agg.get_minibatches(num_minibatches=1, shuffle=False):
        seen.update(mb["block_ids"].unique().tolist())
    _assert(seen == set(range(num_blocks)),
            f"aggregated buffer missing blocks: {seen}")


def test_aggregated_buffer_block_statistics():
    num_blocks = 2
    buffers = build_buffers(
        num_blocks=num_blocks,
        num_envs_per_block=3,
        horizon=4,
        obs_dim=8,
        action_dim=3,
        phi_dim=4,
        device="cpu",
    )
    for j, buf in enumerate(buffers):
        _fill_buffer(buf, seed=j, block_id=j)
        buf.compute_advantages(last_values=torch.zeros(3))

    agg = AggregatedBuffer(buffers)
    stats = agg.block_statistics()
    _assert(isinstance(stats, dict))
    # Should report per-block counts summing to the total.
    total = 0
    for v in stats.values():
        if torch.is_tensor(v):
            total += int(v.sum().item()) if v.dim() > 0 else int(v.item())
    _assert(total > 0, "block_statistics returned no counts")


# ---------------------------------------------------------------------------
# Leader aggregation
# ---------------------------------------------------------------------------
def test_leader_update_uses_all_followers_data():
    num_blocks = 3
    horizon, num_envs, obs_dim, action_dim, phi_dim = 4, 4, 8, 3, 4
    policy, value = _make_networks(obs_dim, action_dim, phi_dim, num_blocks)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()), lr=1e-4
    )
    hp = _make_hparams(horizon=horizon)
    kl = KLScheduler(optimizer, kl_threshold=hp.kl_threshold)

    leader = build_leader(
        policy, value, optimizer, phi_dim=phi_dim, hparams=hp,
        kl_scheduler=kl, device="cpu", seed=0,
    )

    buffers = build_buffers(
        num_blocks=num_blocks,
        num_envs_per_block=num_envs,
        horizon=horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        phi_dim=phi_dim,
        device="cpu",
    )
    for j, buf in enumerate(buffers):
        _fill_buffer(buf, seed=j, block_id=j)
        buf.compute_advantages(last_values=torch.zeros(num_envs))

    stats = leader.update(buffers)
    _assert(stats.num_transitions == num_blocks * horizon * num_envs,
            f"leader saw {stats.num_transitions} transitions, expected "
            f"{num_blocks * horizon * num_envs}")
    _assert(stats.num_updates > 0, "leader performed no updates")
    _assert(math.isfinite(stats.total_loss), "leader loss is not finite")


def test_leader_importance_weights_finite():
    num_blocks = 2
    horizon, num_envs, obs_dim, action_dim, phi_dim = 4, 4, 8, 3, 4
    policy, value = _make_networks(obs_dim, action_dim, phi_dim, num_blocks)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()), lr=1e-4
    )
    hp = _make_hparams(horizon=horizon)
    leader = build_leader(
        policy, value, optimizer, phi_dim=phi_dim, hparams=hp,
        device="cpu", seed=1, use_importance_weights=True,
    )
    buffers = build_buffers(
        num_blocks=num_blocks, num_envs_per_block=num_envs, horizon=horizon,
        obs_dim=obs_dim, action_dim=action_dim, phi_dim=phi_dim, device="cpu",
    )
    for j, buf in enumerate(buffers):
        _fill_buffer(buf, seed=10 + j, block_id=j)
        buf.compute_advantages(last_values=torch.zeros(num_envs))

    stats = leader.update(buffers)
    _assert(math.isfinite(stats.mean_importance_weight),
            "mean importance weight is not finite")
    _assert(stats.mean_importance_weight >= 0.0,
            "importance weight must be non-negative")
    _assert(stats.max_importance_weight <= 10.0 + 1e-6,
            "importance weight exceeded clamp bound (10.0)")


# ---------------------------------------------------------------------------
# Symmetric aggregation (ablation)
# ---------------------------------------------------------------------------
def test_symmetric_aggregator_updates_all_workers():
    num_blocks = 3
    horizon, num_envs, obs_dim, action_dim, phi_dim = 4, 4, 8, 3, 4
    policy, value = _make_networks(obs_dim, action_dim, phi_dim, num_blocks)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()), lr=1e-4
    )
    hp = _make_hparams(horizon=horizon)
    kl = KLScheduler(optimizer, kl_threshold=hp.kl_threshold)

    followers = build_followers(
        num_blocks=num_blocks, policy=policy, value=value, optimizer=optimizer,
        phi_dim=phi_dim, hparams=hp, kl_scheduler=kl, device="cpu", seed=0,
    )
    sym = SymmetricAggregator(
        followers=followers, policy=policy, value=value, optimizer=optimizer,
        hparams=hp, kl_scheduler=kl, device="cpu", use_importance_weights=True,
    )

    buffers = build_buffers(
        num_blocks=num_blocks, num_envs_per_block=num_envs, horizon=horizon,
        obs_dim=obs_dim, action_dim=action_dim, phi_dim=phi_dim, device="cpu",
    )
    for j, buf in enumerate(buffers):
        _fill_buffer(buf, seed=20 + j, block_id=j)
        buf.compute_advantages(last_values=torch.zeros(num_envs))

    result = sym.aggregate(buffers)
    _assert(isinstance(result, dict))
    _assert(len(result) == num_blocks,
            f"symmetric aggregator updated {len(result)} workers, "
            f"expected {num_blocks}")
    for j, stats in result.items():
        _assert(isinstance(stats, dict))
        _assert("total_loss" in stats or "policy_loss" in stats,
                f"worker {j} stats missing loss keys: {list(stats.keys())}")


def test_symmetric_aggregator_act_delegates():
    num_blocks = 2
    obs_dim, action_dim, phi_dim = 8, 3, 4
    policy, value = _make_networks(obs_dim, action_dim, phi_dim, num_blocks)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()), lr=1e-4
    )
    hp = _make_hparams()
    followers = build_followers(
        num_blocks=num_blocks, policy=policy, value=value, optimizer=optimizer,
        phi_dim=phi_dim, hparams=hp, device="cpu", seed=0,
    )
    sym = SymmetricAggregator(
        followers=followers, policy=policy, value=value, optimizer=optimizer,
        hparams=hp, device="cpu",
    )
    obs = torch.randn(4, obs_dim)
    action, log_prob, val = sym.act(obs, deterministic=True)
    _assert(action.shape == (4, action_dim),
            f"unexpected action shape {tuple(action.shape)}")
    _assert(log_prob.shape == (4,), f"unexpected log_prob shape {tuple(log_prob.shape)}")
    _assert(val.shape == (4,), f"unexpected value shape {tuple(val.shape)}")


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------
def test_build_aggregator_leader():
    num_blocks = 2
    obs_dim, action_dim, phi_dim = 8, 3, 4
    policy, value = _make_networks(obs_dim, action_dim, phi_dim, num_blocks)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()), lr=1e-4
    )
    hp = _make_hparams()
    leader = build_leader(
        policy, value, optimizer, phi_dim=phi_dim, hparams=hp, device="cpu",
    )
    agg = build_aggregator(
        AggregationMode.LEADER, leader=leader, policy=policy, value=value,
        optimizer=optimizer, hparams=hp, device="cpu",
    )
    _assert(isinstance(agg, LeaderAggregator),
            f"expected LeaderAggregator, got {type(agg)}")


def test_build_aggregator_symmetric():
    num_blocks = 2
    obs_dim, action_dim, phi_dim = 8, 3, 4
    policy, value = _make_networks(obs_dim, action_dim, phi_dim, num_blocks)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()), lr=1e-4
    )
    hp = _make_hparams()
    followers = build_followers(
        num_blocks=num_blocks, policy=policy, value=value, optimizer=optimizer,
        phi_dim=phi_dim, hparams=hp, device="cpu", seed=0,
    )
    agg = build_aggregator(
        AggregationMode.SYMMETRIC, followers=followers, policy=policy,
        value=value, optimizer=optimizer, hparams=hp, device="cpu",
    )
    _assert(isinstance(agg, SymmetricAggregator),
            f"expected SymmetricAggregator, got {type(agg)}")


def test_build_aggregator_missing_args_raises():
    try:
        build_aggregator(AggregationMode.LEADER)  # no leader provided
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("expected error when leader is missing")

    try:
        build_aggregator(AggregationMode.SYMMETRIC)  # no followers provided
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("expected error when followers are missing")


# ---------------------------------------------------------------------------
# Follower diversity (different blocks -> different action distributions)
# ---------------------------------------------------------------------------
def test_follower_diversity():
    num_blocks = 4
    obs_dim, action_dim, phi_dim = 8, 3, 4
    policy, value = _make_networks(obs_dim, action_dim, phi_dim, num_blocks)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()), lr=1e-4
    )
    hp = _make_hparams()
    followers = build_followers(
        num_blocks=num_blocks, policy=policy, value=value, optimizer=optimizer,
        phi_dim=phi_dim, hparams=hp, device="cpu", seed=0,
    )
    obs = torch.randn(1, obs_dim)
    means = []
    for f in followers:
        action, _, _ = f.act(obs, deterministic=True)
        means.append(action.detach())
    # At least two followers should produce different deterministic actions
    # (they are conditioned on distinct phi_j vectors).
    distinct = False
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            if not torch.allclose(means[i], means[j], atol=1e-6):
                distinct = True
                break
        if distinct:
            break
    _assert(distinct, "followers produced identical actions (no diversity)")


# ---------------------------------------------------------------------------
# AggregationStats container
# ---------------------------------------------------------------------------
def test_aggregation_stats_to_dict():
    stats = AggregationStats(
        mode="leader",
        num_transitions=100,
        num_workers=4,
        follower={0: {"policy_loss": 0.1}},
        leader={"policy_loss": 0.2},
        symmetric={},
    )
    d = stats.to_dict(prefix="agg/")
    _assert(isinstance(d, dict))
    _assert(any(k.startswith("agg/") for k in d.keys()),
            f"prefix not applied: {list(d.keys())}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {t.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() > 0 else 0)

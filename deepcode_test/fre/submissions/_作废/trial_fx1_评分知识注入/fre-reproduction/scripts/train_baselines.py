"""Train baseline zero-shot offline RL agents.

This script trains one of the paper's baselines (FB, SF, GC-IQL, GC-BC, or
OPAL) on an offline dataset.  Evaluation is intentionally left to
``eval_baselines.py`` so that every baseline is scored with the same protocol.

Example
-------
    python scripts/train_baselines.py --baseline gc_iql --domain antmaze \
        --dataset_name antmaze-large-diverse-v2 --steps 1000000
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# Allow ``python scripts/train_baselines.py`` without installing the repo.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fre.config import get_domain_defaults, get_domain_dims  # noqa: E402
from fre.dataset import (  # noqa: E402
    OfflineDataset,
    build_state_pool,
    load_offline_dataset,
    make_synthetic_dataset,
)
from fre.utils import get_logger, resolve_device, save_json, set_seed, to_torch  # noqa: E402

from baselines.fb import FB  # noqa: E402
from baselines.gc_bc import GCBC  # noqa: E402
from baselines.gc_iql import GCIQL  # noqa: E402
from baselines.opal import OPAL  # noqa: E402
from baselines.sf import SF  # noqa: E402


BASELINES = {
    "fb": FB,
    "sf": SF,
    "gc_iql": GCIQL,
    "gc_bc": GCBC,
    "opal": OPAL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a zero-shot offline RL baseline.")
    parser.add_argument("--baseline", type=str, required=True, choices=sorted(BASELINES))
    parser.add_argument("--domain", type=str, default="antmaze",
                        choices=["antmaze", "kitchen", "walker", "cheetah", "synthetic"])
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="D4RL dataset name, e.g. antmaze-large-diverse-v2 or kitchen-complete-v0")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Optional HDF5 path for ExORL datasets")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of training gradient steps (default from domain config)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Width of baseline hidden layers")
    parser.add_argument("--repr_dim", type=int, default=256,
                        help="Representation dimension for FB/SF")
    parser.add_argument("--feature_dim", type=int, default=256,
                        help="Feature dimension for SF")
    parser.add_argument("--skill_dim", type=int, default=16,
                        help="Skill dimension for OPAL")
    parser.add_argument("--goal_dim", type=int, default=None,
                        help="Goal dimension for GC methods (defaults to state_dim)")
    parser.add_argument("--horizon", type=int, default=16,
                        help="Trajectory horizon for OPAL")
    parser.add_argument("--state_pool_size", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints/baselines")
    parser.add_argument("--log_dir", type=str, default="logs/baselines")
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=50000)
    return parser.parse_args()


def _as_tuple(value: Any, length: int = 2) -> Tuple[int, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    return tuple(int(value) for _ in range(length))


def _unpack_dataset_batch(
    dataset: OfflineDataset,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(states, actions, next_states, dones)`` from an offline dataset.

    The dataset wrappers in this repo support several return layouts, so this
    helper defensively accepts both tuple and dict batches.
    """
    batch = dataset.sample_batch(batch_size)
    if isinstance(batch, dict):
        states = batch["states"] if "states" in batch else batch["observations"]
        actions = batch["actions"]
        next_states = (
            batch["next_states"]
            if "next_states" in batch
            else batch["next_observations"]
        )
        dones = batch.get("terminals", batch.get("dones", batch.get("timeouts")))
    else:
        # The canonical OfflineDataset.sample_batch returns
        # (states, actions, next_states, rewards, terminals).
        states, actions, next_states = batch[0], batch[1], batch[2]
        dones = batch[4] if len(batch) > 4 else None

    states = to_torch(states, device=device)
    actions = to_torch(actions, device=device)
    next_states = to_torch(next_states, device=device)
    if dones is not None:
        dones = to_torch(dones, device=device, dtype=torch.float32)
    else:
        dones = torch.zeros(states.shape[0], device=device)
    return states, actions, next_states, dones


def _build_baseline(
    args: argparse.Namespace,
    state_dim: int,
    action_dim: int,
    device: torch.device,
) -> Any:
    """Construct the requested baseline with paper-informed defaults."""
    baseline_name = args.baseline
    hidden_dims = _as_tuple(args.hidden_dim, 2)
    lr = args.lr if args.lr is not None else 3e-4

    if baseline_name == "fb":
        return FB(
            state_dim=state_dim,
            action_dim=action_dim,
            repr_dim=args.repr_dim,
            hidden_dims=hidden_dims,
            lr=lr,
            batch_size=args.batch_size,
            device=device,
        )
    if baseline_name == "sf":
        return SF(
            state_dim=state_dim,
            action_dim=action_dim,
            feature_dim=args.feature_dim,
            hidden_dims=hidden_dims,
            lr=lr,
            batch_size=args.batch_size,
            device=device,
        )
    if baseline_name == "gc_iql":
        return GCIQL(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=args.goal_dim,
            hidden_dims=hidden_dims,
            lr=lr,
            batch_size=args.batch_size,
            device=device,
        )
    if baseline_name == "gc_bc":
        return GCBC(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=args.goal_dim,
            hidden_dims=hidden_dims,
            lr=lr,
            batch_size=args.batch_size,
            device=device,
        )
    if baseline_name == "opal":
        return OPAL(
            state_dim=state_dim,
            action_dim=action_dim,
            skill_dim=args.skill_dim,
            hidden_dims=hidden_dims,
            lr=lr,
            batch_size=args.batch_size,
            horizon=args.horizon,
            device=device,
        )
    raise ValueError(f"Unknown baseline: {baseline_name}")


def _call_train_step(
    baseline: Any,
    baseline_name: str,
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    goal_pool: Optional[np.ndarray] = None,
    horizon: int = 16,
) -> Dict[str, float]:
    """Dispatch a training batch to a baseline while tolerating API variants."""
    # Goal-conditioned methods expose a rich keyword API.
    if baseline_name in ("gc_iql", "gc_bc"):
        if baseline_name == "gc_iql":
            return baseline.train_step(
                states, actions, next_states, dones=dones, goal_pool=goal_pool
            )
        return baseline.train_step(
            states, actions, next_states, dones=dones, goal_pool=goal_pool
        )

    if baseline_name == "opal":
        return baseline.train_step(
            states, actions, next_states=next_states, dones=dones, horizon=horizon
        )

    # FB/SF typically use positional batches. Use inspect to support both a
    # single ``batch`` argument and separate tensor arguments.
    try:
        sig = inspect.signature(baseline.train_step)
        params = list(sig.parameters.keys())
        if len(params) >= 2 and params[1] in ("batch", "data"):
            batch = (states, actions, next_states, dones)
            return baseline.train_step(batch)
        return baseline.train_step(states, actions, next_states, dones)
    except (TypeError, ValueError):
        # Last-resort fallback for implementations expecting a packed batch.
        batch = (states, actions, next_states, dones)
        return baseline.train_step(batch)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    logger = get_logger("train_baselines")
    logger.info("Training baseline %s on domain %s (device=%s)",
                args.baseline, args.domain, device)

    domain_defaults = get_domain_defaults(args.domain, dataset_name=args.dataset_name)
    if args.steps is None:
        args.steps = int(getattr(domain_defaults, "rl_steps", 500_000))
    args.steps = int(args.steps)

    if args.domain == "synthetic":
        dataset = make_synthetic_dataset(state_dim=17, action_dim=8, size=200_000, seed=args.seed)
        state_pool = build_state_pool(dataset, max_pool_size=args.state_pool_size)
        state_dim, action_dim = 17, 8
    else:
        dataset = load_offline_dataset(
            args.domain,
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
        )
        state_pool = build_state_pool(dataset, max_pool_size=args.state_pool_size)
        try:
            state_dim = int(dataset.states.shape[-1])
            action_dim = int(dataset.actions.shape[-1])
        except AttributeError:
            state_dim, action_dim = get_domain_dims(args.domain)

    baseline = _build_baseline(args, state_dim, action_dim, device)
    baseline.to(device) if hasattr(baseline, "to") else None

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    metadata = {
        "baseline": args.baseline,
        "domain": args.domain,
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "device": str(device),
    }
    save_json(metadata, os.path.join(args.output_dir,
                                     f"{args.baseline}_{args.domain}_seed{args.seed}_meta.json"))

    ckpt_prefix = os.path.join(
        args.output_dir, f"{args.baseline}_{args.domain}_seed{args.seed}"
    )

    for step in range(1, args.steps + 1):
        states, actions, next_states, dones = _unpack_dataset_batch(
            dataset, args.batch_size, device
        )
        metrics = _call_train_step(
            baseline,
            args.baseline,
            states,
            actions,
            next_states,
            dones,
            goal_pool=state_pool,
            horizon=args.horizon,
        )

        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            metric_str = " ".join(f"{k}={v:.5f}" for k, v in metrics.items())
            logger.info("[%s/%s] %s", step, args.steps, metric_str)

        if step % args.save_interval == 0 or step == args.steps:
            try:
                state_dict = baseline.state_dict()
            except Exception:
                state_dict = {
                    k: v.state_dict()
                    for k, v in baseline.__dict__.items()
                    if isinstance(v, torch.nn.Module)
                }
            torch.save(state_dict, f"{ckpt_prefix}_step{step}.pt")
            logger.info("Saved checkpoint %s_step%d.pt", ckpt_prefix, step)

    final_path = f"{ckpt_prefix}_final.pt"
    try:
        torch.save(baseline.state_dict(), final_path)
    except Exception:
        torch.save({k: v.state_dict() for k, v in baseline.__dict__.items()
                    if isinstance(v, torch.nn.Module)}, final_path)
    logger.info("Training complete. Final checkpoint: %s", final_path)


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()

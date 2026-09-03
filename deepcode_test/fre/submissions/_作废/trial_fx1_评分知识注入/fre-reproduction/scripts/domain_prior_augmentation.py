"""Domain-knowledge reward-prior augmentation experiments (Figure 6).

This script trains FRE encoder/decoder and FRE-conditioned IQL agents using
the standard reward-prior mixture and an augmented mixture that additionally
contains domain-specific reward functions:

- AntMaze: XY-position rewards (dense negative distance to a random dataset XY)
- ExORL walker/cheetah: velocity rewards (random linear rewards over velocity
  coordinates)

The augmented prior keeps the three base families uniformly mixed with the new
family, i.e. each family is sampled with probability 1/4. All other
architecture and hyperparameters are identical, matching the paper's claim
that domain knowledge improves performance without architecture changes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fre.agent import FREAgent  # noqa: E402
from fre.dataset import (  # noqa: E402
    build_state_pool,
    load_offline_dataset,
    make_synthetic_dataset,
)
from fre.fre_vae import FREVAE  # noqa: E402
from fre.reward_prior import (  # noqa: E402
    RewardFunction,
    RewardPrior,
    make_default_reward_prior,
)
from fre.utils import get_logger, resolve_device, set_seed  # noqa: E402

from envs.antmaze_wrapper import (  # noqa: E402
    ANTMAZE_TASKS,
    evaluate_antmaze_policy,
    make_antmaze_task_reward,
)
from envs.exorl_wrapper import (  # noqa: E402
    EXORL_TASKS,
    evaluate_exorl_policy,
    make_exorl_task_reward,
)
from envs.kitchen_wrapper import (  # noqa: E402
    KITCHEN_TASKS,
    evaluate_kitchen_policy,
    make_kitchen_task_reward,
)

LOGGER = get_logger("domain_prior_augmentation")

DOMAIN_TASKS: Dict[str, List[str]] = {
    "antmaze": list(ANTMAZE_TASKS),
    "walker": list(EXORL_TASKS),
    "cheetah": list(EXORL_TASKS),
    "kitchen": list(KITCHEN_TASKS),
}

# Which domain-specific augmentation to use for each benchmark domain.
AUGMENTATION_VARIANTS: Dict[str, List[str]] = {
    "antmaze": ["base_xy"],
    "walker": ["base_velocity"],
    "cheetah": ["base_velocity"],
    "kitchen": [],
}

AUGMENTATION_KINDS: Dict[str, str] = {
    "base_xy": "xy_position",
    "base_velocity": "velocity",
}


def _as_tuple(value: Union[int, Sequence[int], str, None], length: int = 2) -> Tuple[int, ...]:
    """Normalize a CLI-provided hidden-size argument into a tuple."""
    if value is None:
        return (256,) * length
    if isinstance(value, (int, np.integer)):
        return (int(value),) * length
    if isinstance(value, str):
        value = value.strip().strip("[]()").split(",")
    try:
        vals = tuple(int(float(v)) for v in value)  # type: ignore[arg-type]
    except TypeError:
        vals = tuple(int(v) for v in value)  # type: ignore[arg-type]
    if len(vals) == 0:
        return (256,) * length
    return vals


class XYPositionReward(RewardFunction):
    """Dense reward that encourages reaching a randomly sampled XY position.

    Reward is ``-||s_xy - g_xy|| / scale`` clipped to ``[-1, 1]``. This is
    useful for AntMaze, whose first two state dimensions are root XY position.
    """

    def __init__(self, goal: Union[np.ndarray, torch.Tensor], scale: float = 18.0):
        goal_t = torch.as_tensor(goal, dtype=torch.float32)
        self.goal = goal_t
        self.scale = float(scale)
        super().__init__(self._call, kind="xy_position")

    def _call(self, states: torch.Tensor) -> torch.Tensor:
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.goal.device)
        goal_xy = self.goal[:2].to(states_t.device)
        diff = states_t[..., :2] - goal_xy
        dist = torch.norm(diff, dim=-1)
        reward = -dist / self.scale
        return torch.clamp(reward, -1.0, 1.0)


class VelocityReward(RewardFunction):
    """Random linear reward over a subset of velocity coordinates.

    Reward is ``states[..., velocity_indices] @ weights`` clipped to
    ``[-1, 1]``. This is the ExORL domain-knowledge augmentation.
    """

    def __init__(
        self,
        weights: Union[np.ndarray, torch.Tensor],
        velocity_indices: Union[np.ndarray, Sequence[int], torch.Tensor],
    ):
        weights_t = torch.as_tensor(weights, dtype=torch.float32)
        self.weights = weights_t
        self.velocity_indices = torch.as_tensor(
            velocity_indices, dtype=torch.long, device=weights_t.device
        )
        super().__init__(self._call, kind="velocity")

    def _call(self, states: torch.Tensor) -> torch.Tensor:
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.weights.device)
        velocity = states_t[..., self.velocity_indices]
        reward = velocity @ self.weights
        return torch.clamp(reward, -1.0, 1.0)


class DomainAugmentedRewardPrior:
    """Uniform mixture of the standard FRE prior plus one domain-specific family.

    With ``p_extra=0.25``, the three base families and the extra family are
    each sampled with probability 1/4.
    """

    def __init__(
        self,
        state_dim: int,
        state_pool: Union[np.ndarray, torch.Tensor],
        augmentation: str,
        base_prior: Optional[RewardPrior] = None,
        p_extra: float = 0.25,
        device: Union[str, torch.device] = "cpu",
        seed: Optional[int] = None,
        xy_scale: float = 18.0,
        velocity_indices: Optional[Sequence[int]] = None,
    ):
        self.state_dim = int(state_dim)
        self.state_pool = np.asarray(state_pool, dtype=np.float32)
        self.augmentation = augmentation
        self.p_extra = float(p_extra)
        self.device = torch.device(resolve_device(device))
        self.xy_scale = float(xy_scale)

        if base_prior is None:
            base_prior = make_default_reward_prior(
                state_dim=state_dim,
                state_pool=self.state_pool,
                device=self.device,
                seed=seed,
            )
        self.base_prior = base_prior

        if augmentation == "xy_position":
            if len(self.state_pool) == 0:
                raise ValueError("XY-position augmentation requires a non-empty state pool.")
            self.velocity_indices: Optional[torch.Tensor] = None
        elif augmentation == "velocity":
            if velocity_indices is None:
                velocity_indices = list(range(state_dim // 2, state_dim))
            self.velocity_indices = torch.as_tensor(
                velocity_indices, dtype=torch.long, device=self.device
            )
            if len(self.velocity_indices) == 0:
                raise ValueError("Velocity augmentation requires at least one velocity coordinate.")
        else:
            raise ValueError(f"Unsupported augmentation kind: {augmentation}")

    def _sample_extra(self) -> RewardFunction:
        if self.augmentation == "xy_position":
            idx = np.random.randint(0, len(self.state_pool))
            goal = self.state_pool[idx]
            return XYPositionReward(goal, scale=self.xy_scale)
        if self.augmentation == "velocity":
            assert self.velocity_indices is not None
            weights = torch.empty(len(self.velocity_indices), device=self.device).uniform_(-1.0, 1.0)
            return VelocityReward(weights, self.velocity_indices)
        raise RuntimeError("No augmentation family configured.")

    def sample_reward_fn(self) -> RewardFunction:
        if np.random.rand() < self.p_extra:
            return self._sample_extra()
        return self.base_prior.sample_reward_fn()

    def sample_reward_fns(self, batch_size: int) -> List[RewardFunction]:
        return [self.sample_reward_fn() for _ in range(batch_size)]

    def evaluate(self, reward_fn: RewardFunction, states: torch.Tensor) -> torch.Tensor:
        return reward_fn(states)


def make_prior(
    variant: str,
    state_dim: int,
    state_pool: np.ndarray,
    device: torch.device,
    seed: int,
    domain: str,
) -> Union[RewardPrior, DomainAugmentedRewardPrior]:
    """Build either the base prior or a domain-augmented prior for ``variant``."""
    if variant == "base":
        return make_default_reward_prior(
            state_dim=state_dim, state_pool=state_pool, device=device, seed=seed
        )

    augmentation = AUGMENTATION_KINDS.get(variant)
    if augmentation is None:
        raise ValueError(f"Unknown variant {variant!r}; expected one of 'base', 'base_xy', 'base_velocity'.")

    # ExORL observations are [qpos, qvel], so velocity coordinates are the
    # second half of the flattened state.
    if domain in ("walker", "cheetah") and augmentation == "velocity":
        velocity_indices = list(range(state_dim // 2, state_dim))
    else:
        velocity_indices = None

    return DomainAugmentedRewardPrior(
        state_dim=state_dim,
        state_pool=state_pool,
        augmentation=augmentation,
        p_extra=0.25,
        device=device,
        seed=seed,
        velocity_indices=velocity_indices,
    )


def sample_state_batch(
    state_pool: np.ndarray, batch_size: int, n_states: int, device: torch.device
) -> torch.Tensor:
    """Sample ``batch_size`` sets of ``n_states`` states from a numpy pool."""
    state_pool = np.asarray(state_pool, dtype=np.float32)
    indices = np.random.randint(0, len(state_pool), size=(batch_size, n_states))
    return torch.as_tensor(state_pool[indices], dtype=torch.float32, device=device)


def train_vae(
    args: argparse.Namespace,
    prior: Union[RewardPrior, DomainAugmentedRewardPrior],
    state_pool: np.ndarray,
    state_dim: int,
    device: torch.device,
    variant: str,
    output_dir: Path,
) -> FREVAE:
    """Phase 1: train the FRE VAE with the selected reward prior."""
    vae = FREVAE(
        state_dim=state_dim,
        latent_dim=args.latent_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        reward_bins=args.reward_bins,
        embedding_dim=args.embedding_dim,
        decoder_hidden=_as_tuple(args.decoder_hidden, 2),
        beta=args.beta,
        device=device,
    )
    optimizer = vae.configure_optimizer(lr=args.vae_lr)
    vae.train()

    LOGGER.info(
        "Training VAE variant=%s for %d steps (K=%d, K'=%d)",
        variant,
        args.vae_steps,
        args.encoder_states,
        args.decoder_states,
    )

    for step in range(1, args.vae_steps + 1):
        reward_fns = prior.sample_reward_fns(args.vae_batch_size)
        encoder_states = sample_state_batch(
            state_pool, args.vae_batch_size, args.encoder_states, device
        )
        decoder_states = sample_state_batch(
            state_pool, args.vae_batch_size, args.decoder_states, device
        )

        encoder_rewards = torch.stack(
            [prior.evaluate(rf, encoder_states) for rf in reward_fns], dim=0
        )
        decoder_rewards = torch.stack(
            [prior.evaluate(rf, decoder_states) for rf in reward_fns], dim=0
        )

        metrics = vae.training_step(
            encoder_states,
            encoder_rewards,
            decoder_states,
            decoder_rewards,
            optimizer,
        )

        if step % args.log_interval == 0 or step == 1:
            total = metrics.get("total_loss", metrics.get("loss", float("nan")))
            recon = metrics.get("recon_loss", float("nan"))
            kl = metrics.get("kl_loss", float("nan"))
            LOGGER.info(
                "VAE variant=%s step=%d/%d total=%.5f recon=%.5f kl=%.5f",
                variant,
                step,
                args.vae_steps,
                float(total),
                float(recon),
                float(kl),
            )

    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"vae_{variant}_final.pt"
    torch.save(
        {
            "state_dict": vae.state_dict(),
            "latent_dim": args.latent_dim,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "reward_bins": args.reward_bins,
            "embedding_dim": args.embedding_dim,
            "decoder_hidden": _as_tuple(args.decoder_hidden, 2),
        },
        checkpoint_path,
    )
    LOGGER.info("Saved VAE checkpoint: %s", checkpoint_path)
    return vae


def train_agent(
    args: argparse.Namespace,
    vae: FREVAE,
    prior: Union[RewardPrior, DomainAugmentedRewardPrior],
    dataset,
    state_pool: np.ndarray,
    state_dim: int,
    action_dim: int,
    device: torch.device,
    variant: str,
    output_dir: Path,
) -> FREAgent:
    """Phase 2: train FRE-conditioned IQL with the frozen VAE."""
    agent = FREAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        vae=vae,
        reward_prior=prior,
        state_pool=state_pool,
        dataset=dataset,
        encoder_states=args.encoder_states,
        freeze_vae=True,
        q_hidden=_as_tuple(args.q_hidden, 2),
        v_hidden=_as_tuple(args.v_hidden, 2),
        policy_hidden=_as_tuple(args.policy_hidden, 2),
        gamma=args.gamma,
        expectile=args.expectile,
        awr_temperature=args.awr_temperature,
        target_tau=args.target_tau,
        advantage_clip=args.advantage_clip,
        lr=args.rl_lr,
        device=device,
    )

    LOGGER.info(
        "Training IQL variant=%s for %d steps (batch=%d)",
        variant,
        args.rl_steps,
        args.rl_batch_size,
    )

    for step in range(1, args.rl_steps + 1):
        metrics = agent.train_on_dataset(batch_size=args.rl_batch_size)
        if step % args.log_interval == 0 or step == 1:
            q_loss = metrics.get("q_loss", float("nan"))
            v_loss = metrics.get("v_loss", float("nan"))
            policy_loss = metrics.get("policy_loss", float("nan"))
            LOGGER.info(
                "IQL variant=%s step=%d/%d q=%.5f v=%.5f pi=%.5f",
                variant,
                step,
                args.rl_steps,
                float(q_loss),
                float(v_loss),
                float(policy_loss),
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"agent_{variant}_final.pt"
    agent.save(checkpoint_path)
    LOGGER.info("Saved agent checkpoint: %s", checkpoint_path)
    return agent


def make_task_reward(domain: str, task_name: str) -> Callable[[np.ndarray], np.ndarray]:
    if domain == "antmaze":
        return make_antmaze_task_reward(task_name)
    if domain in ("walker", "cheetah"):
        return make_exorl_task_reward(task_name)
    if domain == "kitchen":
        return make_kitchen_task_reward(task_name)
    raise ValueError(f"Unsupported domain: {domain}")


def extract_score(result: Dict[str, float]) -> float:
    for key in ("normalized_score", "score", "mean_return", "success_rate"):
        if key in result:
            value = float(result[key])
            if key == "success_rate" and value <= 1.0:
                return value * 100.0
            return value
    return float(result.get("return", 0.0))


def evaluate_domain_task(
    domain: str,
    task_name: str,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    num_episodes: int,
    seed: int,
) -> Dict[str, float]:
    if domain == "antmaze":
        return evaluate_antmaze_policy(policy_fn, task_name, num_episodes=num_episodes, seed=seed)
    if domain in ("walker", "cheetah"):
        return evaluate_exorl_policy(
            policy_fn,
            task_name,
            domain=domain,
            num_episodes=num_episodes,
            seed=seed,
        )
    if domain == "kitchen":
        return evaluate_kitchen_policy(policy_fn, task_name, num_episodes=num_episodes, seed=seed)
    raise ValueError(f"Unsupported domain: {domain}")


def evaluate_agent(
    agent: FREAgent,
    domain: str,
    tasks: List[str],
    args: argparse.Namespace,
    state_pool: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    state_pool_tensor = torch.as_tensor(
        np.asarray(state_pool, dtype=np.float32), dtype=torch.float32, device=device
    )
    scores: Dict[str, float] = {}

    for task_name in tasks:
        reward_fn = make_task_reward(domain, task_name)
        z = agent.encode_task(
            reward_fn, num_examples=args.num_examples, states=state_pool_tensor
        )

        def policy_fn(obs: np.ndarray, z: torch.Tensor = z) -> np.ndarray:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            return agent.get_action(obs_t, z=z)

        result = evaluate_domain_task(
            domain, task_name, policy_fn, args.num_episodes, args.seed
        )
        score = extract_score(result)
        scores[task_name] = score
        LOGGER.info("Task %s/%s score=%.2f", domain, task_name, score)

    return scores


def run_variant(
    variant: str,
    args: argparse.Namespace,
    dataset,
    state_pool: np.ndarray,
    state_dim: int,
    action_dim: int,
    device: torch.device,
    output_dir: Path,
) -> Dict:
    """Train and evaluate one prior variant."""
    prior = make_prior(variant, state_dim, state_pool, device, args.seed, args.domain)
    variant_dir = output_dir / variant
    vae = train_vae(args, prior, state_pool, state_dim, device, variant, variant_dir)
    agent = train_agent(
        args,
        vae,
        prior,
        dataset,
        state_pool,
        state_dim,
        action_dim,
        device,
        variant,
        variant_dir,
    )
    tasks = DOMAIN_TASKS[args.domain]
    scores = evaluate_agent(agent, args.domain, tasks, args, state_pool, device)
    mean_score = float(np.mean(list(scores.values()))) if scores else 0.0
    return {"variant": variant, "mean_score": mean_score, "scores": scores}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Domain-knowledge reward-prior augmentation experiments (Figure 6)."
    )
    parser.add_argument("--domain", type=str, default="antmaze",
                        choices=["antmaze", "walker", "cheetah", "kitchen", "synthetic"])
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use a synthetic dataset instead of D4RL/ExORL.")
    parser.add_argument("--synthetic_state_dim", type=int, default=17)
    parser.add_argument("--synthetic_action_dim", type=int, default=8)
    parser.add_argument("--synthetic_size", type=int, default=20000)

    parser.add_argument("--variants", type=str, nargs="+", default=None,
                        help="Variants to run, e.g. base base_xy. Default: derived from domain.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="results/domain_prior_augmentation")
    parser.add_argument("--state_pool_size", type=int, default=None)

    # FRE VAE hyperparameters
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--reward_bins", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--decoder_hidden", type=str, default="256,256")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--vae_lr", type=float, default=1e-4)
    parser.add_argument("--vae_batch_size", type=int, default=16)
    parser.add_argument("--encoder_states", type=int, default=32)
    parser.add_argument("--decoder_states", type=int, default=256)
    parser.add_argument("--vae_steps", type=int, default=100000)

    # IQL hyperparameters
    parser.add_argument("--q_hidden", type=str, default="256,256")
    parser.add_argument("--v_hidden", type=str, default="256,256")
    parser.add_argument("--policy_hidden", type=str, default="256,256")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--expectile", type=float, default=0.9)
    parser.add_argument("--awr_temperature", type=float, default=3.0)
    parser.add_argument("--target_tau", type=float, default=0.005)
    parser.add_argument("--advantage_clip", type=float, nargs=2, default=(-5.0, 2.0))
    parser.add_argument("--rl_lr", type=float, default=3e-4)
    parser.add_argument("--rl_batch_size", type=int, default=256)
    parser.add_argument("--rl_steps", type=int, default=1000000)

    # Evaluation
    parser.add_argument("--num_examples", type=int, default=32)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir) / args.domain / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic or args.domain == "synthetic":
        domain_for_tasks = "antmaze"
        dataset = make_synthetic_dataset(
            state_dim=args.synthetic_state_dim,
            action_dim=args.synthetic_action_dim,
            size=args.synthetic_size,
            seed=args.seed,
        )
        state_pool = build_state_pool(dataset, args.state_pool_size)
        state_dim = args.synthetic_state_dim
        action_dim = args.synthetic_action_dim
    else:
        domain_for_tasks = args.domain
        dataset = load_offline_dataset(args.domain, args.dataset_name, args.dataset_path)
        if dataset is None:
            raise RuntimeError(
                f"Could not load dataset for domain {args.domain!r}. "
                "Use --synthetic for a lightweight smoke test."
            )
        state_pool = build_state_pool(dataset, args.state_pool_size)
        state_dim = int(dataset.states.shape[-1])
        action_dim = int(dataset.actions.shape[-1])

    if args.variants is None:
        variants = ["base"] + AUGMENTATION_VARIANTS.get(args.domain, [])
    else:
        variants = list(args.variants)

    LOGGER.info("Running domain-prior augmentation for domain=%s variants=%s",
                args.domain, variants)

    report: List[Dict] = []
    for variant in variants:
        LOGGER.info("=== Variant %s ===", variant)
        result = run_variant(
            variant=variant,
            args=args,
            dataset=dataset,
            state_pool=state_pool,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            output_dir=output_dir,
        )
        report.append(result)

    summary = {
        "domain": args.domain,
        "seed": args.seed,
        "num_examples": args.num_examples,
        "num_episodes": args.num_episodes,
        "variants": report,
    }
    report_path = output_dir / "domain_prior_augmentation.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    LOGGER.info("Saved report: %s", report_path)

    for result in report:
        LOGGER.info(
            "Variant %s mean=%.2f scores=%s",
            result["variant"],
            result["mean_score"],
            {k: round(v, 2) for k, v in result["scores"].items()},
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

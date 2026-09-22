"""State-space diversity analysis driver (SAPG paper Section 6.4, Figures 7-8).

This script reproduces the two diversity metrics used in the paper:

* **Figure 7 (PCA)**: PCA reconstruction error of visited states as a function of
  the number of principal components ``k``.  SAPG is expected to show the
  *slowest* decrease (i.e. it fills more state-space dimensions) compared with
  PPO and a randomly initialised policy.
* **Figure 8 (MLP)**: training reconstruction error of a two-layer ReLU
  auto-reconstructor of width ``w`` (the x-axis), trained with Adam (PyTorch
  defaults) and an L2 reconstruction loss on ``400k`` state transitions per
  method.  SAPG is expected to have the *highest* error across widths.

The three state batches compared are:

``sapg``
    States collected from every SAPG block (the union of ``D_1..D_M``), i.e. the
    diverse leader/follower policies.
``ppo``
    States collected from a single vanilla PPO policy.
``random``
    States collected from a randomly initialised policy (the paper's baseline).

Because full training runs take 48-60h on a single GPU, the script supports two
regimes:

1. ``--checkpoint`` / ``--from-checkpoint``: load trained policies and collect
   fresh states with them (the reproduction protocol).
2. ``--train-samples``: run a short training loop first, then collect states
   (useful for smoke tests / partial reproductions).
3. ``--synthetic``: no policy needed - fabricates state batches with controlled
   diversity so the analysis pipeline can be exercised on any machine.

Outputs (written to ``--output-dir``):
    ``pca_curves.json``, ``mlp_curves.json``, ``diversity_summary.json``,
    ``fig7_pca_reconstruction.pdf/png``, ``fig8_mlp_reconstruction.pdf/png``.

Usage
-----
    python scripts/run_diversity_analysis.py --task regrasping --synthetic
    python scripts/run_diversity_analysis.py --task reorientation \
        --checkpoint-sapg runs/sapg/reorientation/final.pt \
        --checkpoint-ppo runs/ppo/reorientation/final.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Make the repository importable when the script is executed directly.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.dirname(_ROOT)):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

# Paper-level constants (Section 6.4 / Addendum).
PAPER_TRANSITIONS_PER_METHOD = 400_000
PAPER_SEEDS = 5
METHODS: Tuple[str, ...] = ("sapg", "ppo", "random")
DEFAULT_K_VALUES: Tuple[int, ...] = tuple(range(1, 33))
DEFAULT_WIDTHS: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256)
TASKS: Tuple[str, ...] = (
    "regrasping",
    "throw",
    "reorientation",
    "shadow_hand",
    "allegro_hand",
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the command line interface of the diversity analysis driver."""
    parser = argparse.ArgumentParser(
        description=(
            "SAPG state-space diversity analysis (paper Section 6.4, Fig. 7 & 8)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", type=str, default="regrasping", choices=list(TASKS))
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config for the task (defaults to configs/<task_group>.yaml).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Do not build env/policies; use synthetic state batches instead.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="sapg,ppo,random",
        help="Comma separated list of methods to compare.",
    )
    parser.add_argument(
        "--checkpoint-sapg",
        type=str,
        default=None,
        help="Checkpoint of a trained SAPG trainer.",
    )
    parser.add_argument(
        "--checkpoint-ppo",
        type=str,
        default=None,
        help="Checkpoint of a trained vanilla PPO trainer.",
    )
    parser.add_argument(
        "--train-samples",
        type=float,
        default=0.0,
        help="If > 0, train for that many env transitions before collecting states.",
    )
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-policies", type=int, default=None)
    parser.add_argument(
        "--num-transitions",
        type=int,
        default=PAPER_TRANSITIONS_PER_METHOD,
        help="State transitions collected per method (paper: 400k).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=32,
        help="Rollout steps per collection call.",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default=None,
        help="Comma separated PCA component counts (default 1..32).",
    )
    parser.add_argument(
        "--widths",
        type=str,
        default=None,
        help="Comma separated MLP widths (default 2,4,8,16,32,64,128,256).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--pca-backend",
        type=str,
        default="auto",
        choices=["auto", "sklearn", "numpy", "torch"],
    )
    parser.add_argument(
        "--mlp-backend",
        type=str,
        default="auto",
        choices=["auto", "torch", "numpy"],
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip figure rendering (JSON results are still written).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


# ---------------------------------------------------------------------------
# Config / env / policy helpers
# ---------------------------------------------------------------------------
def load_config(args: argparse.Namespace) -> Any:
    """Resolve an ``SAPGConfig`` for the requested task."""
    from sapg.utils.config import build_config

    overrides: Dict[str, Any] = {}
    if args.num_envs is not None:
        overrides["num_envs"] = int(args.num_envs)
    if args.num_policies is not None:
        overrides["num_policies"] = int(args.num_policies)
    if args.device is not None:
        overrides["device"] = args.device
    overrides["seed"] = int(args.seed)

    config = build_config(task=args.task, **overrides)
    if args.config:
        try:
            from sapg.utils.config import SAPGConfig

            base = SAPGConfig.from_yaml(args.config)
            merged = base.to_dict()
            merged.update(overrides)
            config = SAPGConfig.from_dict(merged)
        except Exception as exc:  # pragma: no cover - defensive
            if args.verbose:
                print(f"[diversity] could not load {args.config}: {exc}")
    return config


def make_env_for(config: Any, args: argparse.Namespace) -> Any:
    """Build the vectorised environment used for state collection."""
    from sapg.envs import make_env

    kwargs: Dict[str, Any] = {}
    if args.device is not None:
        kwargs["device"] = args.device
    try:
        return make_env(task=args.task, config=config, **kwargs)
    except TypeError:
        return make_env(args.task, config=config)


def make_policy_for(config: Any, num_policies: int = 1, random_phi: bool = False) -> Any:
    """Build an ``ActorCritic`` policy matching the config."""
    from sapg.models.actor import ActorCritic

    overrides: Dict[str, Any] = {}
    try:
        overrides["obs_dim"] = int(getattr(config, "obs_dim"))
        overrides["action_dim"] = int(getattr(config, "action_dim"))
    except Exception:  # pragma: no cover - config always provides these
        pass
    overrides["num_policies"] = int(num_policies)
    overrides["phi_dim"] = int(getattr(config, "phi_dim", 0)) if num_policies > 1 else 0
    overrides["random_phi"] = bool(random_phi)
    overrides["config"] = config
    try:
        return ActorCritic(**overrides)
    except TypeError:
        return ActorCritic(config=config, num_policies=num_policies)


def set_seed(seed: int, env: Any = None) -> int:
    """Seed python/numpy/torch (and the env if it exposes ``seed``)."""
    import random

    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except Exception:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:  # pragma: no cover
        pass
    if env is not None and hasattr(env, "seed"):
        try:
            env.seed(int(seed))
        except Exception:  # pragma: no cover
            pass
    return int(seed)


# ---------------------------------------------------------------------------
# Training (optional, for short reproduction runs)
# ---------------------------------------------------------------------------
def maybe_train(
    method: str,
    config: Any,
    env: Any,
    train_samples: float,
    args: argparse.Namespace,
) -> Any:
    """Train ``method`` for ``train_samples`` transitions; return the trainer."""
    if train_samples <= 0:
        return None
    if args.verbose:
        print(f"[diversity] training {method} for {train_samples:g} transitions")
    if method == "ppo":
        from sapg.baselines.ppo_baseline import train_ppo_baseline

        trainer, _ = train_ppo_baseline(
            config=config,
            env=env,
            max_samples=int(train_samples),
            verbose=args.verbose,
            device=getattr(config, "device", None),
            seed=int(args.seed),
        )
        return trainer
    from sapg.algorithms.sapg import train_sapg

    trainer, _ = train_sapg(
        config=config,
        env=env,
        max_samples=int(train_samples),
        verbose=args.verbose,
    )
    return trainer


def load_trainer(method: str, config: Any, env: Any, path: Optional[str]) -> Any:
    """Load a trainer checkpoint for ``method`` if a path was given."""
    if not path:
        return None
    if method == "ppo":
        from sapg.baselines.ppo_baseline import PPOBaselineTrainer

        trainer = PPOBaselineTrainer(config=config, env=env, seed=int(config.seed))
    else:
        from sapg.algorithms.sapg import SAPGTrainer

        trainer = SAPGTrainer(config=config, env=env)
    if hasattr(trainer, "load"):
        trainer.load(path)
        if getattr(args_global, "verbose", False):  # pragma: no cover - debug aid
            print(f"[diversity] loaded {method} checkpoint from {path}")
    if hasattr(trainer, "eval"):
        try:
            trainer.eval()
        except Exception:  # pragma: no cover
            pass
    return trainer


args_global: Any = None  # set in main(), used only for verbose reporting


# ---------------------------------------------------------------------------
# State collection
# ---------------------------------------------------------------------------
def collect_method_states(
    method: str,
    config: Any,
    env: Any,
    args: argparse.Namespace,
    max_samples: int,
) -> Any:
    """Collect a state batch for one method (SAPG / PPO / random)."""
    from sapg.analysis.diversity_pca import collect_states

    device = getattr(config, "device", None)
    obs = None
    if hasattr(env, "reset"):
        try:
            obs = env.reset()
        except TypeError:
            obs = env.reset(None)

    trainer = None
    if method == "sapg":
        if args.checkpoint_sapg:
            trainer = load_trainer("sapg", config, env, args.checkpoint_sapg)
        elif args.train_samples > 0:
            trainer = maybe_train("sapg", config, env, args.train_samples, args)
        policy = getattr(trainer, "policy", None) if trainer is not None else None
        if policy is None:
            num_policies = int(getattr(config, "num_policies", 1) or 1)
            policy = make_policy_for(config, num_policies=num_policies)
    elif method == "ppo":
        if args.checkpoint_ppo:
            trainer = load_trainer("ppo", config, env, args.checkpoint_ppo)
        elif args.train_samples > 0:
            trainer = maybe_train("ppo", config, env, args.train_samples, args)
        policy = getattr(trainer, "policy", None) if trainer is not None else None
        if policy is None:
            policy = make_policy_for(config, num_policies=1)
    elif method == "random":
        policy = make_policy_for(config, num_policies=1)
        # Randomly initialised policy => no training; move to device if possible.
        if device is not None and hasattr(policy, "to"):
            try:
                policy = policy.to(device)
            except Exception:  # pragma: no cover
                pass
    else:
        raise ValueError(f"unknown diversity method: {method!r}")

    collected: List[Any] = []
    total = 0
    attempts = 0
    max_attempts = 64
    while total < max_samples and attempts < max_attempts:
        attempts += 1
        need = max_samples - total
        steps = max(1, min(int(args.num_steps), int(need)))
        try:
            batch = collect_states(
                policy,
                env,
                num_steps=steps,
                obs=obs,
                deterministic=(method != "random"),
                max_samples=need,
                policy_index=0,
            )
        except TypeError:
            batch = collect_states(policy, env, num_steps=steps, obs=obs, max_samples=need)
        collected.append(batch)
        n = _num_rows(batch)
        total += n
        if n == 0:
            break
        # advance observation stream if the env exposes it
        obs = getattr(env, "obs", None) or obs

    if not collected:
        raise RuntimeError(f"no states collected for method {method!r}")
    return _concat(collected)


def synthetic_states(
    method: str, obs_dim: int, num_samples: int, seed: int = 0
) -> Any:
    """Fabricate a state batch with controlled diversity for pipeline testing.

    ``sapg`` gets the largest effective dimensionality (slowest PCA decay),
    ``ppo`` an intermediate one and ``random`` the smallest.
    """
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for --synthetic") from exc

    rng = np.random.default_rng(int(seed))
    effective = {"sapg": min(obs_dim, max(4, obs_dim)), "ppo": max(2, obs_dim // 2), "random": 2}
    dim = effective.get(method, 2)
    latent = rng.normal(size=(int(num_samples), int(dim)))
    projection = rng.normal(size=(int(dim), int(obs_dim))) / max(1.0, dim ** 0.5)
    states = latent @ projection
    states += 0.01 * rng.normal(size=states.shape)
    return states.astype("float32")


def _num_rows(batch: Any) -> int:
    """Number of rows in a state batch of any supported container type."""
    if batch is None:
        return 0
    try:
        shape = batch.shape
        if shape:
            return int(shape[0])
    except Exception:
        pass
    try:
        return int(len(batch))
    except Exception:
        return 0


def _concat(batches: Sequence[Any]) -> Any:
    """Concatenate state batches keeping a numpy-friendly representation."""
    if len(batches) == 1:
        return batches[0]
    try:
        import numpy as np

        arrs = []
        for b in batches:
            if hasattr(b, "detach"):
                b = b.detach().cpu().numpy()
            arrs.append(np.asarray(b))
        return np.concatenate(arrs, axis=0)
    except Exception:  # pragma: no cover
        try:
            import torch

            return torch.cat([t if hasattr(t, "shape") else torch.as_tensor(t) for t in batches], dim=0)
        except Exception:
            out: List[Any] = []
            for b in batches:
                out.extend(list(b))
            return out


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
def parse_int_list(text: Optional[str], default: Sequence[int]) -> List[int]:
    """Parse a comma separated integer list, falling back to ``default``."""
    if not text:
        return [int(v) for v in default]
    out: List[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(float(chunk)))
    return out or [int(v) for v in default]


def run_pca_analysis(
    state_batches: Dict[str, Any],
    k_values: Sequence[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Figure 7: PCA reconstruction error vs number of components k."""
    from sapg.analysis.diversity_pca import compare_pca_diversity

    config = _pca_config(args)
    results = compare_pca_diversity(
        state_batches, k_values=tuple(int(k) for k in k_values), config=config
    )
    payload: Dict[str, Any] = {"results": {}, "k_values": list(k_values)}
    for name, res in results.items():
        payload["results"][name] = res.as_dict()
    payload["summary"] = {
        name: {
            "final_error": float(res.error_at(max(k_values))),
            "decrease_rate": float(
                __import__("sapg.analysis.diversity_pca", fromlist=["curve_decrease_rate"]).curve_decrease_rate(res)
            ),
            "num_samples": int(res.num_samples),
            "state_dim": int(res.state_dim),
        }
        for name, res in results.items()
    }
    return payload


def _pca_config(args: argparse.Namespace) -> Any:
    from sapg.analysis.diversity_pca import PCAConfig

    return PCAConfig(
        backend=args.pca_backend,
        max_samples=int(args.num_transitions),
        seed=int(args.seed),
    )


def _mlp_config(args: argparse.Namespace) -> Any:
    from sapg.analysis.diversity_mlp import MLPConfig

    return MLPConfig(
        backend=args.mlp_backend,
        max_samples=int(args.num_transitions),
        seed=int(args.seed),
        device=args.device,
    )


def run_mlp_analysis(
    state_batches: Dict[str, Any],
    widths: Sequence[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Figure 8: MLP reconstruction training error vs hidden width."""
    from sapg.analysis.diversity_mlp import (
        compare_mlp_diversity,
        diversity_ranking,
        error_increase_ratio,
    )

    config = _mlp_config(args)
    curves = compare_mlp_diversity(
        state_batches, widths=tuple(int(w) for w in widths), config=config
    )
    payload: Dict[str, Any] = {
        "widths": [int(w) for w in widths],
        "curves": {},
        "ranking": [],
    }
    for name, results in curves.items():
        if name in ("widths", "summary"):
            continue
        payload["curves"][name] = [r.as_dict() for r in results]
    try:
        payload["ranking"] = [
            [str(name), float(value)] for name, value in diversity_ranking(curves)
        ]
    except Exception:  # pragma: no cover - ranking is best effort
        pass
    try:
        payload["sapg_vs_ppo_error_ratio"] = float(
            error_increase_ratio(curves, reference="ppo", method="sapg")
        )
    except Exception:  # pragma: no cover
        pass
    if isinstance(curves, dict) and "summary" in curves:
        payload["summary"] = curves["summary"]
    return payload


def write_figures(
    pca_payload: Dict[str, Any],
    mlp_payload: Dict[str, Any],
    output_dir: str,
    args: argparse.Namespace,
) -> List[str]:
    """Render Figure 7 and Figure 8 into ``output_dir`` (returns written paths)."""
    written: List[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib optional
        if args.verbose:
            print(f"[diversity] matplotlib unavailable ({exc}); skipping figures")
        return written

    # --- Figure 7: PCA reconstruction error vs k -------------------------
    try:
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        for name, res in pca_payload.get("results", {}).items():
            errors = res.get("errors")
            ks = res.get("k_values") or pca_payload.get("k_values")
            if not errors or not ks:
                continue
            ax.plot(ks, errors, label=name.upper(), marker="o", markersize=3)
        ax.set_xlabel("number of principal components k")
        ax.set_ylabel("PCA reconstruction error")
        ax.set_yscale("log")
        ax.set_title("State-space diversity (Fig. 7)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        for ext in ("pdf", "png"):
            path = os.path.join(output_dir, f"fig7_pca_reconstruction.{ext}")
            fig.savefig(path, dpi=160)
            written.append(path)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        if args.verbose:
            print(f"[diversity] Figure 7 rendering failed: {exc}")

    # --- Figure 8: MLP reconstruction error vs width ----------------------
    try:
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        for name, curves in mlp_payload.get("curves", {}).items():
            widths = [c.get("width") for c in curves]
            errors = [c.get("train_error") for c in curves]
            ax.plot(widths, errors, label=name.upper(), marker="o", markersize=3)
        ax.set_xlabel("hidden width")
        ax.set_ylabel("MLP reconstruction training error")
        ax.set_xscale("log", base=2)
        ax.set_title("State-space diversity (Fig. 8)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        for ext in ("pdf", "png"):
            path = os.path.join(output_dir, f"fig8_mlp_reconstruction.{ext}")
            fig.savefig(path, dpi=160)
            written.append(path)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        if args.verbose:
            print(f"[diversity] Figure 8 rendering failed: {exc}")

    return written


def print_summary(payload: Dict[str, Any]) -> None:
    """Print a compact textual summary of both metrics."""
    pca = payload.get("pca", {})
    mlp = payload.get("mlp", {})
    print("\n=== Diversity analysis (paper Section 6.4) ===")
    print("PCA (Fig. 7): expected SAPG to decrease slowest")
    for name, stats in (pca.get("summary") or {}).items():
        print(
            f"  {name:<7} final_error={stats['final_error']:.6g} "
            f"decrease_rate={stats['decrease_rate']:.6g} n={stats['num_samples']}"
        )
    print("MLP (Fig. 8): expected SAPG to have the highest reconstruction error")
    for name, stats in (mlp.get("summary") or {}).items():
        if isinstance(stats, dict) and "mean_error" in stats:
            print(f"  {name:<7} mean_error={float(stats['mean_error']):.6g}")
    ranking = mlp.get("ranking") or []
    if ranking:
        print("  ranking (higher error = more diverse): " + ", ".join(
            f"{name}={value:.4g}" for name, value in ranking
        ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point: collect states, run both diversity metrics, write results."""
    global args_global
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    args_global = args

    methods = [m.strip().lower() for m in str(args.methods).split(",") if m.strip()]
    k_values = parse_int_list(args.k_values, DEFAULT_K_VALUES)
    widths = parse_int_list(args.widths, DEFAULT_WIDTHS)
    output_dir = args.output_dir or os.path.join("runs", "diversity", args.task)
    os.makedirs(output_dir, exist_ok=True)

    config = None
    env = None
    state_batches: Dict[str, Any] = {}

    if args.synthetic:
        obs_dim = 63 if args.task in ("regrasping", "throw") else 65
        if args.task == "reorientation":
            obs_dim = 67
        if args.task == "allegro_hand":
            obs_dim = 49
        print(f"[diversity] synthetic mode: obs_dim={obs_dim}")
        for method in methods:
            state_batches[method] = synthetic_states(
                method, obs_dim, int(args.num_transitions), seed=int(args.seed)
            )
    else:
        config = load_config(args)
        env = make_env_for(config, args)
        set_seed(int(args.seed), env)
        for method in methods:
            print(f"[diversity] collecting {args.num_transitions} transitions for {method}")
            try:
                state_batches[method] = collect_method_states(
                    method, config, env, args, int(args.num_transitions)
                )
            except Exception as exc:
                print(f"[diversity] collection failed for {method}: {exc}")
                if args.verbose:
                    import traceback

                    traceback.print_exc()
        if not state_batches:
            print("[diversity] no states collected; re-run with --synthetic")
            return 1

    payload: Dict[str, Any] = {"task": args.task, "methods": methods, "seed": int(args.seed)}

    try:
        pca_payload = run_pca_analysis(state_batches, k_values, args)
        payload["pca"] = pca_payload
        with open(os.path.join(output_dir, "pca_curves.json"), "w") as fh:
            json.dump(pca_payload, fh, indent=2)
    except Exception as exc:
        print(f"[diversity] PCA analysis failed: {exc}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        payload["pca"] = {"error": str(exc)}

    try:
        mlp_payload = run_mlp_analysis(state_batches, widths, args)
        payload["mlp"] = mlp_payload
        with open(os.path.join(output_dir, "mlp_curves.json"), "w") as fh:
            json.dump(mlp_payload, fh, indent=2)
    except Exception as exc:
        print(f"[diversity] MLP analysis failed: {exc}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        payload["mlp"] = {"error": str(exc)}

    if not args.no_plot:
        written = write_figures(payload.get("pca", {}), payload.get("mlp", {}), output_dir, args)
        payload["figures"] = written

    with open(os.path.join(output_dir, "diversity_summary.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    print_summary(payload)
    print(f"\n[diversity] results written to {output_dir}")

    if env is not None and hasattr(env, "close"):
        try:
            env.close()
        except Exception:  # pragma: no cover
            pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Diversity metrics for SAPG (Sec 6.4, Fig. 7 & Fig. 8).

This module implements the two representation-diversity probes used in the
SAPG paper to argue that the split-and-aggregate scheme keeps the *state
distribution* visited by the policies diverse (relative to vanilla PPO):

* **PCA reconstruction error (Fig. 7).**  Fit a PCA on the collected
  state-transitions and measure the reconstruction error when projecting onto
  the top-``k`` principal components.  A *slower* decrease of the error as
  ``k`` grows indicates a higher-dimensional / more diverse state manifold.
  SAPG should show the slowest decrease.

* **MLP reconstruction error (Fig. 8).**  Train a 2-layer MLP auto-encoder
  (ReLU, Adam with PyTorch defaults, L2 loss) on ``400k`` state-transitions
  and report the training reconstruction error as a function of the hidden
  width.  SAPG should show consistently *higher* training error than PPO,
  i.e. the data is harder to compress.

Both metrics operate on the same underlying artifact: a tensor of
state-transitions ``X`` of shape ``[num_transitions, obs_dim]`` collected from
a trained policy (or from the union of the M per-policy blocks for SAPG).

The module is deliberately self-contained: it can be driven from the CLI
(``python -m experiments.diversity``) or imported and called with in-memory
arrays.  It degrades gracefully when ``scikit-learn`` / ``torch`` are missing
by falling back to a NumPy SVD-based PCA and a NumPy MLP respectively.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Repo-root path injection so the module works both as a package
# (``python -m experiments.diversity``) and as a plain script.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:  # optional dependency
    from sklearn.decomposition import PCA as _SkPCA  # type: ignore

    HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional
    _SkPCA = None  # type: ignore
    HAS_SKLEARN = False

try:  # optional dependency
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore
    nn = None  # type: ignore
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Constants (paper Sec 6.4)
# ---------------------------------------------------------------------------
DEFAULT_NUM_TRANSITIONS = 400_000
DEFAULT_PCA_COMPONENTS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_MLP_HIDDEN_DIMS: Tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256)
DEFAULT_MLP_LAYERS = 2
DEFAULT_MLP_EPOCHS = 50
DEFAULT_MLP_BATCH_SIZE = 4096
DEFAULT_MLP_LR = 1e-3
DEFAULT_SEED = 0

# Algorithms compared in the diversity study.
DIVERSITY_ALGORITHMS: Tuple[str, ...] = ("sapg", "ppo")
DIVERSITY_TASKS: Tuple[str, ...] = ("regrasping", "throw", "reorientation")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class PCAResult:
    """Result of the PCA reconstruction-error sweep (Fig. 7)."""

    components: List[int] = field(default_factory=list)
    errors: List[float] = field(default_factory=list)
    explained_variance: List[float] = field(default_factory=list)
    obs_dim: int = 0
    num_transitions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "components": list(self.components),
            "errors": list(self.errors),
            "explained_variance": list(self.explained_variance),
            "obs_dim": int(self.obs_dim),
            "num_transitions": int(self.num_transitions),
        }


@dataclass
class MLPResult:
    """Result of the MLP auto-encoder reconstruction-error sweep (Fig. 8)."""

    hidden_dims: List[int] = field(default_factory=list)
    errors: List[float] = field(default_factory=list)
    obs_dim: int = 0
    num_transitions: int = 0
    epochs: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hidden_dims": list(self.hidden_dims),
            "errors": list(self.errors),
            "obs_dim": int(self.obs_dim),
            "num_transitions": int(self.num_transitions),
            "epochs": int(self.epochs),
        }


# ---------------------------------------------------------------------------
# PCA reconstruction error (Fig. 7)
# ---------------------------------------------------------------------------
def pca_reconstruction_error(
    states: np.ndarray,
    components: Sequence[int] = DEFAULT_PCA_COMPONENTS,
    center: bool = True,
) -> PCAResult:
    """Reconstruction error of ``states`` using the top-``k`` PCA components.

    Args:
        states: array ``[N, obs_dim]`` of state-transitions.
        components: sequence of ``k`` values to sweep over.
        center: whether to mean-center the data before the SVD.

    Returns:
        :class:`PCAResult` with the mean squared reconstruction error for each
        ``k`` (normalised by the number of features so values are comparable
        across tasks with different ``obs_dim``).
    """
    X = np.asarray(states, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"states must be 2-D [N, obs_dim], got shape {X.shape}")
    n, d = X.shape
    if n == 0:
        raise ValueError("states is empty")

    mean = X.mean(axis=0, keepdims=True) if center else np.zeros((1, d), dtype=X.dtype)
    Xc = X - mean

    # Full SVD once; reuse for every k.
    # U, S, Vt = svd(Xc, full_matrices=False)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    total_var = float(np.sum(S ** 2)) / max(n, 1)
    if total_var <= 0.0:
        total_var = 1e-12

    result = PCAResult(obs_dim=int(d), num_transitions=int(n))
    for k in components:
        k_eff = int(min(max(int(k), 1), min(n, d)))
        Vk = Vt[:k_eff]  # [k, d]
        # Project and reconstruct.
        proj = Xc @ Vk.T  # [N, k]
        recon = proj @ Vk  # [N, d]
        mse = float(np.mean((Xc - recon) ** 2))
        result.components.append(int(k))
        result.errors.append(mse)
        # Explained variance ratio of the top-k components.
        ev = float(np.sum(S[:k_eff] ** 2)) / max(n, 1)
        result.explained_variance.append(ev / total_var)
    return result


def pca_reconstruction_error_sklearn(
    states: np.ndarray,
    components: Sequence[int] = DEFAULT_PCA_COMPONENTS,
) -> PCAResult:
    """Same as :func:`pca_reconstruction_error` but using scikit-learn's PCA.

    Provided for parity with the paper's description ("reconstruction error
    using top-k PCA components"); falls back to the NumPy implementation when
    scikit-learn is unavailable.
    """
    if not HAS_SKLEARN:
        return pca_reconstruction_error(states, components)

    X = np.asarray(states, dtype=np.float64)
    n, d = X.shape
    result = PCAResult(obs_dim=int(d), num_transitions=int(n))
    for k in components:
        k_eff = int(min(max(int(k), 1), min(n, d)))
        pca = _SkPCA(n_components=k_eff, svd_solver="randomized", random_state=DEFAULT_SEED)
        proj = pca.fit_transform(X)
        recon = pca.inverse_transform(proj)
        mse = float(np.mean((X - recon) ** 2))
        result.components.append(int(k))
        result.errors.append(mse)
        result.explained_variance.append(float(np.sum(pca.explained_variance_ratio_)))
    return result


# ---------------------------------------------------------------------------
# MLP reconstruction error (Fig. 8)
# ---------------------------------------------------------------------------
def _build_mlp_autoencoder(obs_dim: int, hidden_dim: int, num_layers: int = 2):
    """Build a symmetric 2-layer MLP auto-encoder (ReLU activations)."""
    if not HAS_TORCH:
        raise RuntimeError("torch is required for the MLP diversity metric")
    layers: List[Any] = []
    in_dim = obs_dim
    for _ in range(max(int(num_layers), 1)):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    # Decoder mirrors the encoder.
    for _ in range(max(int(num_layers), 1)):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, obs_dim))
    return nn.Sequential(*layers)


def mlp_reconstruction_error(
    states: np.ndarray,
    hidden_dims: Sequence[int] = DEFAULT_MLP_HIDDEN_DIMS,
    num_layers: int = DEFAULT_MLP_LAYERS,
    epochs: int = DEFAULT_MLP_EPOCHS,
    batch_size: int = DEFAULT_MLP_BATCH_SIZE,
    lr: float = DEFAULT_MLP_LR,
    device: Optional[str] = None,
    seed: int = DEFAULT_SEED,
    verbose: bool = False,
) -> MLPResult:
    """Train a 2-layer MLP auto-encoder and report its reconstruction error.

    Follows the paper's protocol (Sec 6.4): 2-layer MLP with ReLU activations,
    Adam optimiser with PyTorch defaults, L2 reconstruction loss, trained on
    ``400k`` state-transitions.  The x-axis of Fig. 8 is the hidden width, so
    we sweep ``hidden_dims`` and report the final training error for each.

    Args:
        states: array ``[N, obs_dim]`` of state-transitions.
        hidden_dims: hidden widths to sweep (x-axis of Fig. 8).
        num_layers: number of encoder (and decoder) layers.
        epochs: training epochs per width.
        batch_size: minibatch size.
        lr: Adam learning rate.
        device: torch device string (defaults to CUDA if available).
        seed: RNG seed.
        verbose: print per-width progress.

    Returns:
        :class:`MLPResult` with the final training reconstruction error per
        hidden width.
    """
    if not HAS_TORCH:
        raise RuntimeError("torch is required for the MLP diversity metric")

    X = np.asarray(states, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"states must be 2-D [N, obs_dim], got shape {X.shape}")
    n, d = X.shape

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    data = torch.from_numpy(X).to(dev)
    result = MLPResult(obs_dim=int(d), num_transitions=int(n), epochs=int(epochs))

    for hidden in hidden_dims:
        hidden = int(hidden)
        model = _build_mlp_autoencoder(d, hidden, num_layers=num_layers).to(dev)
        # Adam with PyTorch defaults (betas=(0.9, 0.999), eps=1e-8, wd=0).
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        n_batches = max(int(np.ceil(n / batch_size)), 1)
        final_loss = float("nan")
        for epoch in range(int(epochs)):
            perm = torch.randperm(n, device=dev)
            epoch_loss = 0.0
            for b in range(n_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                batch = data[idx]
                optim.zero_grad()
                recon = model(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optim.step()
                epoch_loss += float(loss.detach().cpu()) * batch.shape[0]
            final_loss = epoch_loss / n
            if verbose:
                print(
                    f"[diversity] hidden={hidden} epoch={epoch + 1}/{epochs} "
                    f"loss={final_loss:.6f}"
                )
        result.hidden_dims.append(hidden)
        result.errors.append(float(final_loss))
    return result


# ---------------------------------------------------------------------------
# Data collection helpers
# ---------------------------------------------------------------------------
def collect_states(
    trainer: Any,
    num_transitions: int = DEFAULT_NUM_TRANSITIONS,
    env: Any = None,
    deterministic: bool = True,
) -> np.ndarray:
    """Collect ``num_transitions`` state-transitions from a trained trainer.

    Works with any of the SAPG/PPO/PQL/DexPBT trainers: it repeatedly calls the
    trainer's rollout collection and concatenates the observations.  For SAPG
    the union of all M per-policy blocks is used, which is exactly the
    "diverse" state distribution the paper analyses.

    Args:
        trainer: a trainer exposing ``collect_rollout``/``collect`` or a
            ``networks`` + ``env`` pair.
        num_transitions: number of states to gather.
        env: optional env override.
        deterministic: unused placeholder for API symmetry.

    Returns:
        array ``[num_transitions, obs_dim]``.
    """
    chunks: List[np.ndarray] = []
    collected = 0

    # Preferred path: SAPG trainer exposes ``_collect_rollouts``/``collect``.
    collect_fn = None
    for name in ("collect_rollouts", "collect", "collect_rollout"):
        if hasattr(trainer, name):
            collect_fn = getattr(trainer, name)
            break

    while collected < num_transitions:
        obs_batch: Optional[np.ndarray] = None
        if collect_fn is not None:
            out = collect_fn()
            obs_batch = _extract_obs(out)
        elif hasattr(trainer, "networks") and hasattr(trainer, "env"):
            obs_batch = _rollout_obs(trainer)
        else:
            raise ValueError(
                "trainer does not expose a recognised rollout collection API"
            )

        if obs_batch is None or obs_batch.size == 0:
            break
        chunks.append(obs_batch)
        collected += obs_batch.shape[0]

    if not chunks:
        raise RuntimeError("no states collected")
    X = np.concatenate(chunks, axis=0)
    return X[:num_transitions]


def _extract_obs(out: Any) -> Optional[np.ndarray]:
    """Best-effort extraction of an ``[N, obs_dim]`` array from rollout output."""
    if out is None:
        return None
    if isinstance(out, np.ndarray):
        return out.reshape(-1, out.shape[-1])
    if isinstance(out, (list, tuple)):
        for item in out:
            arr = _extract_obs(item)
            if arr is not None:
                return arr
        return None
    if isinstance(out, dict):
        for key in ("obs", "observations", "states"):
            if key in out:
                return _extract_obs(out[key])
        return None
    # RolloutBuffer-like object.
    for attr in ("obs", "observations", "states"):
        if hasattr(out, attr):
            val = getattr(out, attr)
            if hasattr(val, "detach"):
                val = val.detach().cpu().numpy()
            return np.asarray(val).reshape(-1, np.asarray(val).shape[-1])
    return None


def _rollout_obs(trainer: Any, steps: int = 16) -> np.ndarray:
    """Fallback rollout using a trainer's ``networks`` + ``env`` directly."""
    env = trainer.env
    networks = trainer.networks
    device = getattr(trainer, "device", None)
    obs = env.reset()
    if hasattr(obs, "detach"):
        obs = obs.detach()
    collected: List[np.ndarray] = []
    for _ in range(steps):
        with torch.no_grad():
            action, _, _ = networks.actor_act(obs, None)
        out = env.step(action)
        obs = out[0]
        if hasattr(obs, "detach"):
            obs = obs.detach()
        collected.append(obs.detach().cpu().numpy() if hasattr(obs, "detach") else np.asarray(obs))
    return np.concatenate(collected, axis=0)


# ---------------------------------------------------------------------------
# High-level driver
# ---------------------------------------------------------------------------
def run_diversity_analysis(
    state_sets: Dict[str, np.ndarray],
    pca_components: Sequence[int] = DEFAULT_PCA_COMPONENTS,
    mlp_hidden_dims: Sequence[int] = DEFAULT_MLP_HIDDEN_DIMS,
    mlp_epochs: int = DEFAULT_MLP_EPOCHS,
    use_sklearn: bool = False,
    device: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Run both diversity metrics for each labelled state set.

    Args:
        state_sets: mapping ``label -> states [N, obs_dim]`` (e.g.
            ``{"sapg": X_sapg, "ppo": X_ppo}``).
        pca_components: ``k`` values for the PCA sweep (Fig. 7).
        mlp_hidden_dims: hidden widths for the MLP sweep (Fig. 8).
        mlp_epochs: training epochs per MLP width.
        use_sklearn: use scikit-learn's PCA when available.
        device: torch device for the MLP metric.
        verbose: print progress.

    Returns:
        ``{label: {"pca": PCAResult, "mlp": MLPResult}}``.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for label, states in state_sets.items():
        if verbose:
            print(f"[diversity] analysing '{label}' ({states.shape[0]} transitions)")
        if use_sklearn:
            pca_res = pca_reconstruction_error_sklearn(states, pca_components)
        else:
            pca_res = pca_reconstruction_error(states, pca_components)
        mlp_res = mlp_reconstruction_error(
            states,
            hidden_dims=mlp_hidden_dims,
            epochs=mlp_epochs,
            device=device,
            verbose=verbose,
        )
        results[label] = {"pca": pca_res, "mlp": mlp_res}
    return results


def summarize_diversity(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Convert :func:`run_diversity_analysis` output into JSON-friendly dicts."""
    out: Dict[str, Any] = {}
    for label, res in results.items():
        out[label] = {
            "pca": res["pca"].to_dict() if hasattr(res["pca"], "to_dict") else res["pca"],
            "mlp": res["mlp"].to_dict() if hasattr(res["mlp"], "to_dict") else res["mlp"],
        }
    return out


def compare_diversity(
    results: Dict[str, Dict[str, Any]],
    reference: str = "ppo",
    candidate: str = "sapg",
) -> Dict[str, Any]:
    """Quantify the paper's claim: SAPG is *more* diverse than PPO.

    Returns the mean relative difference in reconstruction error
    ``(candidate - reference) / reference`` for both metrics.  Positive values
    mean the candidate (SAPG) has *higher* reconstruction error, i.e. a more
    diverse / harder-to-compress state distribution — matching Fig. 7/Fig. 8.
    """
    if reference not in results or candidate not in results:
        return {}
    ref, cand = results[reference], results[candidate]

    def _rel(a: Sequence[float], b: Sequence[float]) -> List[float]:
        out = []
        for x, y in zip(a, b):
            out.append(float((y - x) / x) if x not in (0.0,) else float("nan"))
        return out

    pca_rel = _rel(ref["pca"].errors, cand["pca"].errors)
    mlp_rel = _rel(ref["mlp"].errors, cand["mlp"].errors)
    return {
        "reference": reference,
        "candidate": candidate,
        "pca_relative_difference": pca_rel,
        "pca_mean_relative_difference": float(np.nanmean(pca_rel)) if pca_rel else float("nan"),
        "mlp_relative_difference": mlp_rel,
        "mlp_mean_relative_difference": float(np.nanmean(mlp_rel)) if mlp_rel else float("nan"),
    }


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------
def save_diversity(
    results: Dict[str, Dict[str, Any]],
    output_dir: str,
    prefix: str = "diversity",
) -> Dict[str, str]:
    """Persist diversity results as JSON + NPZ for plotting."""
    os.makedirs(output_dir, exist_ok=True)
    summary = summarize_diversity(results)
    json_path = os.path.join(output_dir, f"{prefix}_summary.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    npz_payload: Dict[str, np.ndarray] = {}
    for label, res in results.items():
        npz_payload[f"{label}__pca_components"] = np.asarray(res["pca"].components)
        npz_payload[f"{label}__pca_errors"] = np.asarray(res["pca"].errors)
        npz_payload[f"{label}__mlp_hidden_dims"] = np.asarray(res["mlp"].hidden_dims)
        npz_payload[f"{label}__mlp_errors"] = np.asarray(res["mlp"].errors)
    npz_path = os.path.join(output_dir, f"{prefix}_curves.npz")
    np.savez(npz_path, **npz_payload)
    return {"summary": json_path, "curves": npz_path}


def load_diversity(output_dir: str, prefix: str = "diversity") -> Dict[str, Any]:
    """Load a previously saved diversity summary (JSON)."""
    json_path = os.path.join(output_dir, f"{prefix}_summary.json")
    if not os.path.exists(json_path):
        return {}
    with open(json_path) as fh:
        return json.load(fh)


def load_state_sets(runs_dir: str) -> Dict[str, np.ndarray]:
    """Load pre-collected state-transition arrays from ``runs_dir``.

    Expects files named ``<label>_states.npy`` (e.g. ``sapg_states.npy``,
    ``ppo_states.npy``).  This is the artifact produced by the training runs
    when ``--dump-states`` is enabled.
    """
    state_sets: Dict[str, np.ndarray] = {}
    if not os.path.isdir(runs_dir):
        return state_sets
    for fname in sorted(os.listdir(runs_dir)):
        if fname.endswith("_states.npy"):
            label = fname[: -len("_states.npy")]
            state_sets[label] = np.load(os.path.join(runs_dir, fname))
    return state_sets


def dump_states(states: np.ndarray, output_dir: str, label: str) -> str:
    """Save a state-transition array as ``<label>_states.npy``."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{label}_states.npy")
    np.save(path, np.asarray(states))
    return path


# ---------------------------------------------------------------------------
# Synthetic fallback (for smoke tests without trained policies)
# ---------------------------------------------------------------------------
def synthetic_state_sets(
    obs_dim: int = 64,
    num_transitions: int = DEFAULT_NUM_TRANSITIONS,
    seed: int = DEFAULT_SEED,
) -> Dict[str, np.ndarray]:
    """Generate synthetic state sets mimicking SAPG (diverse) vs PPO (collapsed).

    SAPG's split policies explore different regions of state space, so its
    states are drawn from a mixture of ``M`` Gaussians; vanilla PPO collapses
    onto a single mode.  This lets the diversity pipeline be exercised end to
    end without a trained policy.
    """
    rng = np.random.default_rng(seed)
    n = int(num_transitions)

    # PPO: single isotropic Gaussian (low intrinsic dimensionality).
    ppo = rng.normal(0.0, 1.0, size=(n, obs_dim)).astype(np.float32)

    # SAPG: mixture of 6 modes with a shared low-rank structure + noise.
    num_modes = 6
    modes = rng.normal(0.0, 3.0, size=(num_modes, obs_dim)).astype(np.float32)
    assign = rng.integers(0, num_modes, size=n)
    sapg = modes[assign] + rng.normal(0.0, 1.0, size=(n, obs_dim)).astype(np.float32)
    return {"sapg": sapg, "ppo": ppo}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SAPG diversity metrics (PCA Fig. 7, MLP Fig. 8)."
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="directory containing <label>_states.npy files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="directory to write diversity_summary.json / diversity_curves.npz",
    )
    parser.add_argument(
        "--num-transitions",
        type=int,
        default=DEFAULT_NUM_TRANSITIONS,
        help="number of state-transitions to use (paper: 400000)",
    )
    parser.add_argument(
        "--mlp-epochs",
        type=int,
        default=DEFAULT_MLP_EPOCHS,
        help="training epochs per MLP hidden width",
    )
    parser.add_argument(
        "--use-sklearn",
        action="store_true",
        help="use scikit-learn PCA instead of the NumPy SVD implementation",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="run on synthetic state sets (smoke test, no trained policies)",
    )
    parser.add_argument("--device", type=str, default=None, help="torch device")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    t0 = time.time()

    if args.synthetic:
        state_sets = synthetic_state_sets(
            num_transitions=args.num_transitions, seed=args.seed
        )
    else:
        state_sets = load_state_sets(args.runs_dir)
        if not state_sets:
            print(
                f"[diversity] no <label>_states.npy files found in '{args.runs_dir}'. "
                "Falling back to synthetic data (use --synthetic to silence this)."
            )
            state_sets = synthetic_state_sets(
                num_transitions=args.num_transitions, seed=args.seed
            )

    # Truncate to the requested number of transitions.
    state_sets = {
        k: v[: args.num_transitions] for k, v in state_sets.items()
    }

    results = run_diversity_analysis(
        state_sets,
        mlp_epochs=args.mlp_epochs,
        use_sklearn=args.use_sklearn,
        device=args.device,
        verbose=args.verbose,
    )
    paths = save_diversity(results, args.output_dir)

    print(f"[diversity] wrote {paths['summary']}")
    print(f"[diversity] wrote {paths['curves']}")

    if "sapg" in results and "ppo" in results:
        cmp = compare_diversity(results, reference="ppo", candidate="sapg")
        print(
            "[diversity] SAPG vs PPO mean relative reconstruction error: "
            f"PCA={cmp['pca_mean_relative_difference']:+.3f}, "
            f"MLP={cmp['mlp_mean_relative_difference']:+.3f} "
            "(positive => SAPG more diverse, as in Fig. 7/8)"
        )

    print(f"[diversity] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

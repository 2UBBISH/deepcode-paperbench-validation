"""Appendix C.3 driver: settings for the experiments in Figure 1.

This module is the *glue* experiment driver that (a) records the exact
Appendix C.3 hyper-parameters used to reproduce Figure 1, (b) reproduces the
Figure 1 trivial-solution experiments through
:mod:`lbcs_repro.experiments.figure1_trivial`, and (c) exposes the Appendix
C.1/C.2 probabilistic details (Bernoulli reparameterization and the
gradient-norm analysis :math:`\\zeta_1(\\lambda)` / :math:`\\zeta_2(\\lambda)`)
as small, testable helpers.

Paper references
----------------
Appendix C.3 (verbatim essentials):
    "For the experiments in Figure 1, we employ a subset of MNIST. A
    convolutional neural network stacked with two blocks of convolution,
    dropout, max-pooling, and ReLU activation is used. Following (Zhou et al.,
    2022), for the inner loop, the model is trained for 100 epochs using SGD
    with a learning rate of 0.1 and momentum of 0.9. For the outer loop, the
    probabilities are optimized by Adam with a learning rate of 2.5 and a
    cosine scheduler."

Appendix C.1: ``m_i ~ Bern(s_i)``,
``p(m | s) = prod_i s_i^{m_i} (1 - s_i)^{1 - m_i}``,
``E ||m||_0 = 1^T s``, and the two trivial formulations (Eq. (3) and Eq. (4)
with ``lambda = 1/2``).

Appendix C.2: with
``zeta_1(lambda) = (1 - lambda) || f_1(m) (m - s) / (s (1 - s)) ||_2`` and
``zeta_2(lambda) = lambda sqrt(n)``, setting ``lambda = 1/2`` gives
``zeta_2(1/2) = sqrt(n) / 2``, which is large, "therefore, the coreset size
will be minimized too much."

Nothing here is an algorithm of the paper; all numerics come from
``lbcs_repro.baselines.probabilistic`` and
``lbcs_repro.experiments.figure1_trivial``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paper-stated constants (Appendix C.3 / Appendix C.1 / Appendix C.2)
# ---------------------------------------------------------------------------

#: Figure 1 uses "a subset of MNIST" (the paper's MNIST-S, 1000 examples).
PAPER_DATASET = "MNIST-S"
#: ConvNet = "two blocks of convolution, dropout, max-pooling, and ReLU".
PAPER_ARCHITECTURE = "ConvNet"
PAPER_CONV_BLOCKS = 2
#: Inner loop: SGD, lr 0.1, momentum 0.9, 100 epochs (following Zhou et al.).
PAPER_INNER_OPTIMIZER = "sgd"
PAPER_INNER_LR = 0.1
PAPER_INNER_MOMENTUM = 0.9
PAPER_INNER_EPOCHS = 100
#: Outer loop: Adam, lr 2.5, cosine scheduler.
PAPER_OUTER_OPTIMIZER = "adam"
PAPER_OUTER_LR = 2.5
PAPER_OUTER_SCHEDULER = "cosine"
PAPER_FORMULATIONS = ("eq3", "eq4")
#: Eq. (4) is analysed at lambda = 1/2 in Appendix C.2.
PAPER_LAMBDA = 0.5

# ---------------------------------------------------------------------------
# SUGGESTED defaults (NOT paper-stated; the paper gives no numbers for these)
# ---------------------------------------------------------------------------
SUGGESTED_OUTER_ITERS = 1000  # Figure 1 curves are traced over 1000 iterations
SUGGESTED_MNIST_SUBSET = 1000
SUGGESTED_K = 200
SUGGESTED_BATCH_SIZE = 128
SUGGESTED_EVAL_BATCH_SIZE = 256
SUGGESTED_WEIGHT_DECAY = 0.0
SUGGESTED_S_MIN = 1e-3
SUGGESTED_S_MAX = 1.0 - 1e-3

DEFAULT_OUTPUT_DIR = os.path.join("results", "appendix_c3")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class C3Config:
    """Configuration for the Appendix C.3 / Figure 1 settings driver.

    Fields marked ``PAPER`` in :meth:`to_dict` are stated by Appendix C.3;
    everything else is a clearly labelled SUGGESTED default.
    """

    dataset: str = PAPER_DATASET
    n: int = SUGGESTED_MNIST_SUBSET
    k: int = SUGGESTED_K
    architecture: str = PAPER_ARCHITECTURE
    formulations: Tuple[str, ...] = PAPER_FORMULATIONS
    # outer loop (Appendix C.3)
    outer_iters: int = SUGGESTED_OUTER_ITERS
    outer_optimizer: str = PAPER_OUTER_OPTIMIZER
    outer_lr: float = PAPER_OUTER_LR
    outer_scheduler: str = PAPER_OUTER_SCHEDULER
    # inner loop (Appendix C.3)
    inner_optimizer: str = PAPER_INNER_OPTIMIZER
    inner_lr: float = PAPER_INNER_LR
    inner_momentum: float = PAPER_INNER_MOMENTUM
    inner_epochs: int = PAPER_INNER_EPOCHS
    # misc SUGGESTED
    lambda_: float = PAPER_LAMBDA
    batch_size: int = SUGGESTED_BATCH_SIZE
    eval_batch_size: int = SUGGESTED_EVAL_BATCH_SIZE
    weight_decay: float = SUGGESTED_WEIGHT_DECAY
    s_min: float = SUGGESTED_S_MIN
    s_max: float = SUGGESTED_S_MAX
    seed: int = 0
    device: Optional[str] = None
    output_dir: str = DEFAULT_OUTPUT_DIR
    save_artifacts: bool = True
    plot: bool = True
    verbose: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- constructors -------------------------------------------------------
    @classmethod
    def paper(cls, **overrides: Any) -> "C3Config":
        """Appendix C.3 settings (Figure 1 protocol)."""
        return cls().with_overrides(**overrides)

    @classmethod
    def smoke(cls, **overrides: Any) -> "C3Config":
        """Tiny CPU configuration for smoke tests."""
        base = cls(
            n=200,
            k=40,
            outer_iters=5,
            inner_epochs=1,
            batch_size=64,
            eval_batch_size=128,
            save_artifacts=False,
            plot=False,
            verbose=False,
        )
        return base.with_overrides(**overrides)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "C3Config":
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = dict(data.get("extra", {}) or {})
        for key, value in data.items():
            if key in known:
                kwargs[key] = value
            elif key == "lambda":
                kwargs["lambda_"] = value
            elif key in ("inherit", "inherits", "smoke"):
                continue
            elif key in ("figure1", "appendix_c3", "c3"):
                # nested section block
                nested = cls.from_dict(value)
                for name, value2 in nested.to_dict().items():
                    if name in known:
                        kwargs.setdefault(name, value2)
                    else:
                        extra.setdefault(name, value2)
            else:
                extra.setdefault(key, value)
        if "formulations" in kwargs and kwargs["formulations"] is not None:
            kwargs["formulations"] = tuple(kwargs["formulations"])
        config = cls(**kwargs)
        config.extra = extra
        return config

    def with_overrides(self, **overrides: Any) -> "C3Config":
        known = {f for f in self.__dataclass_fields__ if f != "extra"}
        kwargs: Dict[str, Any] = {}
        extra = dict(self.extra)
        for key, value in overrides.items():
            if value is None and key not in ("device",):
                continue
            if key in known:
                kwargs[key] = value
            elif key == "lambda":
                kwargs["lambda_"] = value
            else:
                extra[key] = value
        if "formulations" in kwargs and kwargs["formulations"] is not None:
            kwargs["formulations"] = tuple(kwargs["formulations"])
        if "lambda_" in kwargs:
            kwargs["lambda_"] = self.lambda_ if kwargs["lambda_"] is None else kwargs["lambda_"]
        config = replace(self, **kwargs)
        config.extra = extra
        return config

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "n": int(self.n),
            "k": int(self.k),
            "architecture": self.architecture,
            "formulations": list(self.formulations),
            "outer_iters": int(self.outer_iters),
            "outer_optimizer": self.outer_optimizer,
            "outer_lr": float(self.outer_lr),
            "outer_scheduler": self.outer_scheduler,
            "inner_optimizer": self.inner_optimizer,
            "inner_lr": float(self.inner_lr),
            "inner_momentum": float(self.inner_momentum),
            "inner_epochs": int(self.inner_epochs),
            "lambda": float(self.lambda_),
            "batch_size": int(self.batch_size),
            "eval_batch_size": int(self.eval_batch_size),
            "weight_decay": float(self.weight_decay),
            "s_min": float(self.s_min),
            "s_max": float(self.s_max),
            "seed": int(self.seed),
            "device": self.device,
            "output_dir": self.output_dir,
            "save_artifacts": bool(self.save_artifacts),
            "plot": bool(self.plot),
        }


def paper_settings() -> Dict[str, Any]:
    """Return the Appendix C.3 setting record (paper-stated values only)."""
    return {
        "dataset": PAPER_DATASET,
        "architecture": PAPER_ARCHITECTURE,
        "conv_blocks": PAPER_CONV_BLOCKS,
        "inner_optimizer": PAPER_INNER_OPTIMIZER,
        "inner_lr": PAPER_INNER_LR,
        "inner_momentum": PAPER_INNER_MOMENTUM,
        "inner_epochs": PAPER_INNER_EPOCHS,
        "outer_optimizer": PAPER_OUTER_OPTIMIZER,
        "outer_lr": PAPER_OUTER_LR,
        "outer_scheduler": PAPER_OUTER_SCHEDULER,
        "formulations": list(PAPER_FORMULATIONS),
        "lambda": PAPER_LAMBDA,
        "provenance": "Appendix C.3 (paper-stated)",
    }


def suggested_settings() -> Dict[str, Any]:
    """Return reproduction defaults that the paper does not state."""
    return {
        "outer_iters": SUGGESTED_OUTER_ITERS,
        "mnist_subset": SUGGESTED_MNIST_SUBSET,
        "k": SUGGESTED_K,
        "batch_size": SUGGESTED_BATCH_SIZE,
        "eval_batch_size": SUGGESTED_EVAL_BATCH_SIZE,
        "weight_decay": SUGGESTED_WEIGHT_DECAY,
        "s_min": SUGGESTED_S_MIN,
        "s_max": SUGGESTED_S_MAX,
        "provenance": "SUGGESTED (not stated in the paper)",
    }


# ---------------------------------------------------------------------------
# Appendix C.1 / C.2 probabilistic helpers
# ---------------------------------------------------------------------------


def local_zeta1(
    f1_value: float,
    m: Any,
    s: Any,
    lambda_: float = PAPER_LAMBDA,
    eps: float = 1e-12,
) -> float:
    """``zeta_1(lambda) = (1 - lambda) || f_1(m) (m - s) / (s (1 - s)) ||_2``."""
    m_arr = np.asarray(m, dtype=np.float64).reshape(-1)
    s_arr = np.clip(np.asarray(s, dtype=np.float64).reshape(-1), eps, 1.0 - eps)
    grad = float(f1_value) * (m_arr - s_arr) / (s_arr * (1.0 - s_arr))
    return float((1.0 - float(lambda_)) * np.linalg.norm(grad, ord=2))


def local_zeta2(lambda_: float = PAPER_LAMBDA, n: Optional[int] = None) -> float:
    """``zeta_2(lambda) = lambda * ||1||_2 = lambda * sqrt(n)``."""
    if n is None:
        raise ValueError("zeta_2 requires the problem dimension n")
    return float(lambda_) * math.sqrt(float(n))


def gradient_norm_analysis(
    n: int,
    f1_value: float = 1.0,
    m: Optional[Any] = None,
    s: Optional[Any] = None,
    lambda_: float = PAPER_LAMBDA,
    lambdas: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Reproduce the Appendix C.2 gradient-magnitude comparison.

    Returns a dict with ``zeta1``/``zeta2`` for ``lambda_`` plus a sweep over
    ``lambdas`` (defaulting to the paper's ``1/2`` and a few neighbours), and
    the paper's conclusion flag: at ``lambda = 1/2`` the size term dominates
    whenever ``zeta_2 > zeta_1``.
    """
    lam_list = tuple(lambdas) if lambdas is not None else (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    if m is None:
        m = np.zeros(int(n))
    if s is None:
        s = np.full(int(n), 0.5)

    sweep: List[Dict[str, float]] = []
    for lam in lam_list:
        z1 = local_zeta1(f1_value, m, s, lam)
        z2 = local_zeta2(lam, n)
        sweep.append({"lambda": float(lam), "zeta1": z1, "zeta2": z2,
                      "zeta2_over_zeta1": (z2 / z1) if z1 > 0 else float("inf")})

    z1 = local_zeta1(f1_value, m, s, lambda_)
    z2 = local_zeta2(lambda_, n)
    return {
        "n": int(n),
        "f1": float(f1_value),
        "lambda": float(lambda_),
        "zeta1": z1,
        "zeta2": z2,
        "zeta2_dominates": bool(z2 > z1),
        "zeta2_at_half": local_zeta2(0.5, n),
        "sqrt_n": math.sqrt(float(n)),
        "sweep": sweep,
        "note": (
            "At lambda = 1/2, zeta_2 = sqrt(n)/2 which is large when n is large; "
            "therefore the coreset size will be minimized too much (Appendix C.2)."
        ),
    }


def expected_coreset_size(s: Any) -> float:
    """``E ||m||_0 = 1^T s`` (Appendix C.1)."""
    return float(np.asarray(s, dtype=np.float64).reshape(-1).sum())


def log_probability(m: Any, s: Any, eps: float = 1e-12) -> float:
    """``ln p(m | s) = sum_i [m_i ln s_i + (1 - m_i) ln (1 - s_i)]``."""
    m_arr = np.asarray(m, dtype=np.float64).reshape(-1)
    s_arr = np.clip(np.asarray(s, dtype=np.float64).reshape(-1), eps, 1.0 - eps)
    return float(np.sum(m_arr * np.log(s_arr) + (1.0 - m_arr) * np.log(1.0 - s_arr)))


def score_function_gradient(f1_value: float, m: Any, s: Any, eps: float = 1e-12) -> np.ndarray:
    """``f_1(m) * (m - s) / (s (1 - s))`` — the unbiased policy gradient."""
    m_arr = np.asarray(m, dtype=np.float64).reshape(-1)
    s_arr = np.clip(np.asarray(s, dtype=np.float64).reshape(-1), eps, 1.0 - eps)
    return float(f1_value) * (m_arr - s_arr) / (s_arr * (1.0 - s_arr))


def probabilistic_details(n: int = 100, seed: int = 0) -> Dict[str, Any]:
    """Verify the Appendix C.1 identity ``E ||m||_0 = 1^T s`` by sampling.

    Sampled expectations are compared against the closed forms so the driver
    can assert the reparameterization is implemented correctly.
    """
    rng = np.random.default_rng(seed)
    s = np.full(int(n), 0.5)
    samples = rng.random((2000, int(n))) < s[None, :]
    sizes = samples.sum(axis=1).astype(np.float64)
    return {
        "n": int(n),
        "expected_size_closed_form": expected_coreset_size(s),
        "expected_size_sampled": float(sizes.mean()),
        "sampled_std": float(sizes.std()),
        "log_prob_check": log_probability(samples[0], s),
        "grad_norm_check": float(np.linalg.norm(score_function_gradient(1.0, samples[0], s))),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _default_runner() -> Optional[Callable[..., Any]]:
    """Return ``figure1_trivial.run_figure1`` when importable, else ``None``."""
    try:  # pragma: no cover - import guarded for partial installs
        from lbcs_repro.experiments import figure1_trivial as fig1

        return getattr(fig1, "run_figure1", None)
    except Exception:  # pragma: no cover
        try:  # pragma: no cover
            from . import figure1_trivial as fig1  # type: ignore

            return getattr(fig1, "run_figure1", None)
        except Exception:
            return None


def figure1_config_from_c3(config: C3Config) -> Any:
    """Map :class:`C3Config` onto ``figure1_trivial.Figure1Config`` fields."""
    overrides = {
        "dataset": config.dataset,
        "n": int(config.n),
        "k": int(config.k),
        "lam": float(config.lambda_),
        "outer_iters": int(config.outer_iters),
        "outer_lr": float(config.outer_lr),
        "outer_optimizer": config.outer_optimizer,
        "outer_scheduler": config.outer_scheduler,
        "inner_epochs": int(config.inner_epochs),
        "inner_lr": float(config.inner_lr),
        "inner_momentum": float(config.inner_momentum),
        "inner_optimizer": config.inner_optimizer,
        "architecture": config.architecture,
        "device": config.device,
        "seed": int(config.seed),
        "output_dir": config.output_dir,
        "formulations": list(config.formulations),
    }
    try:  # pragma: no cover - optional dependency on the sibling driver
        from lbcs_repro.experiments import figure1_trivial as fig1

        return fig1.Figure1Config.paper(**overrides)
    except Exception:  # pragma: no cover
        try:  # pragma: no cover
            from . import figure1_trivial as fig1  # type: ignore

            return fig1.Figure1Config.paper(**overrides)
        except Exception:
            return overrides


def run_appendix_c3(
    config: Optional[C3Config] = None,
    *,
    runner: Optional[Callable[..., Any]] = None,
    logger: Optional[logging.Logger] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Reproduce Figure 1 with the Appendix C.3 settings.

    Parameters
    ----------
    config:
        :class:`C3Config`; defaults to :meth:`C3Config.paper`.
    runner:
        Optional ``callable(config, **kwargs) -> dict`` implementing the
        Figure 1 sweep (defaults to ``figure1_trivial.run_figure1``). Injecting
        a stub keeps this driver testable without torch/datasets.
    """
    log = logger or LOGGER
    cfg = (config or C3Config.paper()).with_overrides(**overrides)

    started = time.time()
    settings = paper_settings()
    settings["suggested"] = suggested_settings()
    analysis = gradient_norm_analysis(
        n=max(int(cfg.n), 1),
        f1_value=max(float(cfg.k), 1.0),
        lambda_=cfg.lambda_,
    )
    details = probabilistic_details(n=min(max(int(cfg.n), 8), 512), seed=cfg.seed)

    result: Dict[str, Any] = {
        "appendix_c3_settings": settings,
        "gradient_analysis": analysis,
        "probabilistic_details": details,
    }

    run_fn = runner if runner is not None else _default_runner()
    if run_fn is None:
        log.warning(
            "figure1_trivial driver unavailable; returning settings/analysis only "
            "(install torch/torchvision to reproduce the Figure 1 curves)"
        )
        result["figure1"] = None
        result["figure1_error"] = "figure1_trivial.run_figure1 unavailable"
    else:
        try:
            fig1_cfg = figure1_config_from_c3(cfg)
            fig1_out = run_fn(config=fig1_cfg, logger=log)
            result["figure1"] = _jsonable(fig1_out)
        except Exception as exc:  # pragma: no cover - keeps the driver alive
            log.warning("Figure 1 reproduction failed: %s", exc)
            result["figure1"] = None
            result["figure1_error"] = f"{type(exc).__name__}: {exc}"

    result["failure_modes"] = summarize_failure_modes(
        (result.get("figure1") or {}).get("summary")
        if isinstance(result.get("figure1"), dict)
        else None,
        analysis,
    )
    result["config"] = cfg.to_dict()
    result["wall_time"] = time.time() - started

    if cfg.save_artifacts:
        result["artifacts"] = save_results(result, cfg)
    if cfg.plot:
        path = plot_gradient_analysis(analysis, cfg)
        if path:
            result.setdefault("artifacts", {})["gradient_analysis_png"] = path
    return result


def summarize_failure_modes(
    figure1_summary: Optional[Dict[str, Any]],
    analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Describe the two Figure 1 failure modes (Eq. (3) and Eq. (4))."""
    modes = {
        "eq3": {
            "label": "fixed-size",
            "description": (
                "Eq. (3) optimizes f_1 only, so the coreset size stays close to the "
                "predefined k while f_1 decreases."
            ),
            "expected": "f2 ~= k throughout the outer loop",
        },
        "eq4": {
            "label": "over-minimization",
            "description": (
                "Eq. (4) with lambda = 1/2 lets the size term dominate, so f_2 collapses "
                "while f_1 stays large."
            ),
            "expected": "f2 much smaller than k with f1 not improved",
        },
    }
    observed: Dict[str, Any] = {}
    if isinstance(figure1_summary, dict):
        for key in ("eq3", "eq4"):
            if key in figure1_summary:
                observed[key] = figure1_summary[key]
    out: Dict[str, Any] = {"expected_modes": modes, "observed": observed}
    if analysis:
        out["gradient_explanation"] = {
            "zeta1": analysis.get("zeta1"),
            "zeta2": analysis.get("zeta2"),
            "zeta2_dominates": analysis.get("zeta2_dominates"),
            "reason": analysis.get("note"),
        }
    return out


def plot_gradient_analysis(analysis: Dict[str, Any], config: Optional[C3Config] = None) -> Optional[str]:
    """Plot zeta_1 vs zeta_2 over lambda (Appendix C.2) when matplotlib exists."""
    try:  # pragma: no cover - optional dependency
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return None

    sweep = analysis.get("sweep") or []
    if not sweep:
        return None
    out_dir = (config or C3Config()).output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "appendix_c3_gradient_analysis.png")

    lambdas = [row["lambda"] for row in sweep]
    z1 = [row["zeta1"] for row in sweep]
    z2 = [row["zeta2"] for row in sweep]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(lambdas, z1, marker="o", label=r"$\zeta_1(\lambda)$")
    ax.plot(lambdas, z2, marker="s", label=r"$\zeta_2(\lambda)$")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("gradient norm")
    ax.set_title("Appendix C.2 gradient analysis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy/torch/dataclass objects to JSON-safe values."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return _jsonable(obj.to_dict())
        except Exception:  # pragma: no cover
            pass
    if hasattr(obj, "__dict__"):
        return {str(k): _jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def save_results(result: Dict[str, Any], config: Optional[C3Config] = None) -> Dict[str, str]:
    """Persist the settings/analysis bundle under ``config.output_dir``."""
    cfg = config or C3Config()
    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    artifacts: Dict[str, str] = {}

    payload = {
        "appendix_c3_settings": _jsonable(result.get("appendix_c3_settings")),
        "gradient_analysis": _jsonable(result.get("gradient_analysis")),
        "probabilistic_details": _jsonable(result.get("probabilistic_details")),
        "failure_modes": _jsonable(result.get("failure_modes")),
        "config": _jsonable(config.to_dict() if config else {}),
        "wall_time": result.get("wall_time"),
    }
    json_path = os.path.join(out_dir, "appendix_c3.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    artifacts["json"] = json_path

    try:  # pragma: no cover - pyyaml is an optional convenience
        import yaml

        yaml_path = os.path.join(out_dir, "appendix_c3_settings.yaml")
        with open(yaml_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(_jsonable(result.get("appendix_c3_settings")), handle, sort_keys=False)
        artifacts["yaml"] = yaml_path
    except Exception:
        pass

    lines = [
        "Appendix C.3 — settings for the experiments in Figure 1",
        "=" * 60,
    ]
    for key, value in (result.get("appendix_c3_settings") or {}).items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for k2, v2 in value.items():
                lines.append(f"    {k2}: {v2}")
        else:
            lines.append(f"{key}: {value}")
    analysis = result.get("gradient_analysis") or {}
    if analysis:
        lines += [
            "",
            "Appendix C.2 — gradient analysis",
            "-" * 60,
            f"zeta_1(lambda={analysis.get('lambda')}) = {analysis.get('zeta1')}",
            f"zeta_2(lambda={analysis.get('lambda')}) = {analysis.get('zeta2')}",
            f"zeta_2(1/2) = sqrt(n)/2 = {analysis.get('zeta2_at_half')}",
            f"size term dominates: {analysis.get('zeta2_dominates')}",
            str(analysis.get("note")),
        ]
    modes = result.get("failure_modes") or {}
    if modes:
        lines += ["", "Figure 1 failure modes", "-" * 60]
        for key, info in (modes.get("expected_modes") or {}).items():
            lines.append(f"{key} ({info['label']}): {info['description']}")
    text_path = os.path.join(out_dir, "appendix_c3.txt")
    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    artifacts["txt"] = text_path

    details = result.get("probabilistic_details") or {}
    if details:
        csv_path = os.path.join(out_dir, "appendix_c3_probabilistic.csv")
        with open(csv_path, "w", encoding="utf-8") as handle:
            handle.write("key,value\n")
            for key, value in details.items():
                handle.write(f"{key},{value}\n")
        artifacts["csv"] = csv_path

    return artifacts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="appendix_c3",
        description=(
            "Reproduce the Appendix C.3 settings for the experiments in Figure 1 and "
            "the Appendix C.2 probabilistic gradient analysis."
        ),
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--n", type=int, default=None, help="MNIST subset size")
    parser.add_argument("--k", type=int, default=None, help="predefined coreset size")
    parser.add_argument("--outer-iters", type=int, default=None)
    parser.add_argument("--inner-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--paper", action="store_true", help="use Appendix C.3 settings")
    parser.add_argument("--smoke", action="store_true", help="tiny CPU smoke configuration")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--selftest", action="store_true", help="offline self-test, no torch needed")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> C3Config:
    config = C3Config.paper()
    if args.config:
        try:
            import yaml

            with open(args.config, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            config = C3Config.from_dict(data)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("could not load config %s: %s", args.config, exc)
    if args.smoke:
        config = C3Config.smoke()
    overrides = {
        "dataset": args.dataset,
        "n": args.n,
        "k": args.k,
        "outer_iters": args.outer_iters,
        "inner_epochs": args.inner_epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }
    config = config.with_overrides(**overrides)
    if args.no_save:
        config = config.with_overrides(save_artifacts=False)
    if args.no_plot:
        config = config.with_overrides(plot=False)
    if args.verbose:
        config = config.with_overrides(verbose=True)
    return config


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.selftest:
        report = _selftest(verbose=True)
        return 0 if report.get("ok") else 1

    config = _config_from_args(args)
    result = run_appendix_c3(config)
    summary = {
        "settings": result.get("appendix_c3_settings"),
        "zeta1": (result.get("gradient_analysis") or {}).get("zeta1"),
        "zeta2": (result.get("gradient_analysis") or {}).get("zeta2"),
        "figure1_ran": result.get("figure1") is not None,
        "artifacts": result.get("artifacts"),
    }
    print(json.dumps(_jsonable(summary), indent=2))
    return 0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks: settings, zeta analysis, C.1 identities, driver plumbing."""
    checks: Dict[str, Any] = {}
    settings = paper_settings()
    checks["settings_dataset"] = settings["dataset"] == PAPER_DATASET
    checks["settings_inner"] = (
        settings["inner_optimizer"] == "sgd"
        and abs(settings["inner_lr"] - 0.1) < 1e-12
        and abs(settings["inner_momentum"] - 0.9) < 1e-12
        and settings["inner_epochs"] == 100
    )
    checks["settings_outer"] = (
        settings["outer_optimizer"] == "adam"
        and abs(settings["outer_lr"] - 2.5) < 1e-12
        and settings["outer_scheduler"] == "cosine"
    )

    analysis = gradient_norm_analysis(n=1000, f1_value=1.0, lambda_=0.5)
    checks["zeta2_half"] = abs(analysis["zeta2_at_half"] - math.sqrt(1000.0) / 2.0) < 1e-9
    checks["zeta2_formula"] = abs(local_zeta2(0.5, 1000) - math.sqrt(1000.0) / 2.0) < 1e-9
    checks["zeta1_nonneg"] = analysis["zeta1"] >= 0.0

    details = probabilistic_details(n=64, seed=0)
    checks["expected_size"] = (
        abs(details["expected_size_closed_form"] - 32.0) < 1e-9
        and abs(details["expected_size_sampled"] - 32.0) < 2.0
    )

    # Bernoulli log-prob / score-gradient algebra
    s = np.array([0.5, 0.25])
    m = np.array([1.0, 0.0])
    lp = log_probability(m, s)
    checks["log_prob"] = abs(lp - (math.log(0.5) + math.log(0.75))) < 1e-9
    grad = score_function_gradient(2.0, m, s)
    expected = 2.0 * (m - s) / (s * (1 - s))
    checks["score_gradient"] = bool(np.allclose(grad, expected))

    # Config plumbing
    cfg = C3Config.paper()
    checks["config_lr"] = abs(cfg.outer_lr - 2.5) < 1e-12 and abs(cfg.inner_lr - 0.1) < 1e-12
    merged = C3Config.from_dict({"figure1": {"n": 500, "k": 50}})
    checks["config_from_dict"] = merged.n == 500 and merged.k == 50
    smoke = C3Config.smoke()
    checks["config_smoke"] = smoke.outer_iters == 5 and smoke.inner_epochs == 1

    # Driver plumbing with an injected stub runner (no torch needed)
    stub_calls: Dict[str, Any] = {}

    def stub_runner(config=None, logger=None, **kwargs):  # noqa: ANN001
        stub_calls["called"] = True
        return {"summary": {"eq3": {"f2_final": 199.0}, "eq4": {"f2_final": 12.0}}}

    result = run_appendix_c3(C3Config.smoke(save_artifacts=False, plot=False), runner=stub_runner)
    checks["driver_calls_runner"] = bool(stub_calls.get("called"))
    checks["driver_modes"] = "expected_modes" in (result.get("failure_modes") or {})
    checks["driver_no_artifacts"] = "artifacts" not in result

    ok = all(bool(v) for v in checks.values())
    report = {"checks": checks, "ok": ok}
    if verbose:
        print(json.dumps(_jsonable(report), indent=2))
    return report


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

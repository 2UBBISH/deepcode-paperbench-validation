"""Utility subpackage for the PINN loss-landscape reproduction.

This package aggregates two glue modules that are shared by every experiment
runner:

* :mod:`src.utils.seeding` -- deterministic seed control (Python ``random``,
  NumPy and PyTorch RNGs), used so that the paper's "winner-selection" process
  (smallest L2RE over ``(adam_lr, seed, width)`` per PDE) is itself reproducible.
* :mod:`src.utils.plotting` -- matplotlib figure helpers that turn raw training /
  spectral outputs into the paper's figures (Fig. 1--8).

All imports are guarded so that the package remains importable in minimal
environments (for instance when NumPy or matplotlib are unavailable).  The
plots are strictly optional for reproducing the numeric tables.
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__: List[str] = []

# --------------------------------------------------------------------------- #
# seeding (optional but almost always available: only needs stdlib + torch)
# --------------------------------------------------------------------------- #
try:  # package-relative import (``import src.utils`` from the project root)
    from . import seeding as seeding  # noqa: F401
    from .seeding import (  # noqa: F401
        DEFAULT_SEED,
        SeedContext,
        describe_rng_state,
        get_seed,
        make_generator,
        seed_everything,
        seed_from_config,
        set_seed,
        spawn_seeds,
    )

    __all__ += [
        "seeding",
        "DEFAULT_SEED",
        "SeedContext",
        "describe_rng_state",
        "get_seed",
        "make_generator",
        "seed_everything",
        "seed_from_config",
        "set_seed",
        "spawn_seeds",
    ]
    SEEDING_AVAILABLE = True
except Exception:  # pragma: no cover - defensive fallback
    try:  # absolute import fallback
        from src.utils import seeding as seeding  # type: ignore # noqa: F401
        from src.utils.seeding import (  # type: ignore # noqa: F401
            DEFAULT_SEED,
            SeedContext,
            describe_rng_state,
            get_seed,
            make_generator,
            seed_everything,
            seed_from_config,
            set_seed,
            spawn_seeds,
        )

        __all__ += [
            "seeding",
            "DEFAULT_SEED",
            "SeedContext",
            "describe_rng_state",
            "get_seed",
            "make_generator",
            "seed_everything",
            "seed_from_config",
            "set_seed",
            "spawn_seeds",
        ]
        SEEDING_AVAILABLE = True
    except Exception:  # pragma: no cover
        SEEDING_AVAILABLE = False

# --------------------------------------------------------------------------- #
# plotting (optional: requires matplotlib; degrades gracefully)
# --------------------------------------------------------------------------- #
PLOTTING_AVAILABLE = False
try:
    try:
        from . import plotting as plotting  # noqa: F401
        from .plotting import (  # noqa: F401
            COMPONENT_LABELS,
            COLORS,
            PDE_LABELS,
            make_figures_from_summary,
            plot_component_densities,
            plot_condition_numbers,
            plot_finetune,
            plot_histogram_spectrum,
            plot_loss_vs_l2re,
            plot_optimizer_comparison,
            plot_preconditioned_densities,
            plot_spectral_density,
            plot_training_curves,
            plot_width_sweep,
            save_figure,
            set_plot_style,
        )
    except ImportError:
        from src.utils import plotting as plotting  # type: ignore # noqa: F401
        from src.utils.plotting import (  # type: ignore # noqa: F401
            COMPONENT_LABELS,
            COLORS,
            PDE_LABELS,
            make_figures_from_summary,
            plot_component_densities,
            plot_condition_numbers,
            plot_finetune,
            plot_histogram_spectrum,
            plot_loss_vs_l2re,
            plot_optimizer_comparison,
            plot_preconditioned_densities,
            plot_spectral_density,
            plot_training_curves,
            plot_width_sweep,
            save_figure,
            set_plot_style,
        )

    from .plotting import PLOTTING_AVAILABLE as PLOTTING_AVAILABLE  # noqa: F811

    __all__ += [
        "plotting",
        "PLOTTING_AVAILABLE",
        "COMPONENT_LABELS",
        "COLORS",
        "PDE_LABELS",
        "make_figures_from_summary",
        "plot_component_densities",
        "plot_condition_numbers",
        "plot_finetune",
        "plot_histogram_spectrum",
        "plot_loss_vs_l2re",
        "plot_optimizer_comparison",
        "plot_preconditioned_densities",
        "plot_spectral_density",
        "plot_training_curves",
        "plot_width_sweep",
        "save_figure",
        "set_plot_style",
    ]
except Exception:  # pragma: no cover - matplotlib missing / broken install
    PLOTTING_AVAILABLE = False

__all__ += ["SEEDING_AVAILABLE", "PLOTTING_AVAILABLE"]


def available_modules() -> Dict[str, bool]:
    """Report which utility sub-modules could be imported.

    Returns
    -------
    dict
        ``{"seeding": bool, "plotting": bool}``.  Experiment runners use this to
        skip figure generation (or reproducible seeding) when the corresponding
        third-party dependency is missing, instead of failing outright.
    """
    return {"seeding": bool(SEEDING_AVAILABLE), "plotting": bool(PLOTTING_AVAILABLE)}


def resolve_seed_fn() -> Any:
    """Return a callable seeding every RNG, or a harmless no-op fallback.

    The runners import :func:`src.utils.seeding.set_seed` inside a guarded block
    with a local fallback; this helper centralises the same behaviour.
    """
    if SEEDING_AVAILABLE:
        return set_seed

    def _noop(seed: Any = None, **kwargs: Any) -> Any:  # pragma: no cover
        return seed

    return _noop


__all__ += ["available_modules", "resolve_seed_fn"]

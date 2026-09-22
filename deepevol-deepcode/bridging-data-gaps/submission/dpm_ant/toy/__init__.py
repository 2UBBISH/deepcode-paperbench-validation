"""Toy 2-D Gaussian experiment for DPMs-ANT (Section 5.1, Figure 2).

This subpackage reproduces the paper's synthetic sanity check:

* :mod:`dpm_ant.toy.toy_2d` implements the experiment itself -- a source
  Gaussian ``N((1,1), I)`` is memorised by a small MLP noise predictor, a binary
  source/target classifier is trained, and the ANT adaptor (here the model's
  output bias, initialised to zero) is transferred to the target Gaussian
  ``N((-1,-1), I)`` using the adversarial-noise (Eq. 7) + similarity-guided
  (Eq. 5) objective.  It also produces the Figure 2(a) gradient-direction
  comparison (10,000 samples vs. baseline the DDPM, DPMs-ANT w/o AN, and full
  DPMs-ANT with 10 samples repeated 1,000x) and the Figure 2(b)/(c) heat-map
  data (x-axis = diffusion timestep, y-axis = sampled value).
* :mod:`dpm_ant.toy.toy_plots` renders Figure 2 from those results.

The initializer re-exports the public API lazily (PEP 562) so that a bare
``import dpm_ant.toy`` does not pull in torch/matplotlib unless a symbol is
actually used.
"""

from __future__ import annotations

import importlib
from typing import Dict, List

__all__: List[str] = [
    # toy_2d: configuration
    "ToyConfig",
    # toy_2d: models
    "ToyMLP",
    "ToyClassifier",
    "InitialisationBias",
    # toy_2d: builders / training
    "build_toy_schedule",
    "build_toy_model",
    "train_toy_model",
    "train_toy_classifier",
    # toy_2d: experiments
    "gradient_direction_experiment",
    "noise_cloud_statistics",
    "heatmap_samples",
    "run_toy_experiment",
    "ant_transfer",
    "ddpm_gradient",
    "output_layer_gradient",
    # toy_plots: rendering
    "FigureStyle",
    "plot_figure2",
    "plot_gradient_directions",
    "plot_noise_cloud",
    "plot_heatmap",
    "plot_training_curves",
    "plot_loss_curves",
    "save_figure",
    "save_results_json",
]

_EXPORTS: Dict[str, str] = {
    # --- dpm_ant.toy.toy_2d -------------------------------------------------
    "ToyConfig": "toy_2d",
    "ToyMLP": "toy_2d",
    "ToyClassifier": "toy_2d",
    "InitialisationBias": "toy_2d",
    "build_toy_schedule": "toy_2d",
    "build_toy_model": "toy_2d",
    "train_toy_model": "toy_2d",
    "train_toy_classifier": "toy_2d",
    "gradient_direction_experiment": "toy_2d",
    "noise_cloud_statistics": "toy_2d",
    "heatmap_samples": "toy_2d",
    "run_toy_experiment": "toy_2d",
    "ant_transfer": "toy_2d",
    "ddpm_gradient": "toy_2d",
    "output_layer_gradient": "toy_2d",
    # --- dpm_ant.toy.toy_plots ---------------------------------------------
    "FigureStyle": "toy_plots",
    "plot_figure2": "toy_plots",
    "plot_gradient_directions": "toy_plots",
    "plot_noise_cloud": "toy_plots",
    "plot_heatmap": "toy_plots",
    "plot_training_curves": "toy_plots",
    "plot_loss_curves": "toy_plots",
    "save_figure": "toy_plots",
    "save_results_json": "toy_plots",
}


def __getattr__(name: str):
    """Lazily resolve public toy-experiment symbols (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(f".{module_name}", __name__)
    try:
        value = getattr(module, name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise AttributeError(
            f"module {module.__name__!r} does not define {name!r}"
        ) from exc

    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))

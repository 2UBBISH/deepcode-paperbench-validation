"""Experiment drivers for the LBCS (Refined Coreset Selection) reproduction.

This package aggregates every in-scope experiment driver required by the
reproduction plan:

======================  =========================================================
Driver                  Paper artefact
======================  =========================================================
``figure1_trivial``     Figure 1 -- Eq. (3) fixed-size / Eq. (4) over-minimization
``table1_prelim``       Table 1 -- Section 5.1 preliminary superiority (MNIST-S)
``table2_table3_compare`` Tables 2 & 3 -- Section 5.2 competitor comparison
``figure2_robustness``  Figure 2 -- Section 5.3 label noise / class imbalance
``table8_sizes``        Table 8 (Appendix E.3) -- optimized sizes, imperfect supervision
``table9_search_times`` Table 9 (Section 6 / Appendix E.4) -- T-sweep on F-MNIST
``table5_init``         Table 5 (Section 6) -- LBCS + Moderate mask initialization
``table6_cross_arch``   Table 6 (Section 6) -- ViT-small / WideResNet on SVHN
``appendix_c3``         Appendix C.3 -- Figure 1 settings / probabilistic details
======================  =========================================================

**Out of scope** (explicitly excluded by the reproduction plan): ImageNet-1k
(Section 5.4), continual learning (Appendix E.5) and streaming coreset
selection (Appendix E.6).  :data:`OUT_OF_SCOPE` records those exclusions and
no driver in this package may invoke them.

Design notes
------------
* Every driver is imported **lazily** through :func:`import_experiment`, so this
  package (and therefore ``lbcs_repro.main`` / ``scripts/run_all.py``) imports
  cleanly even when an individual driver module is still being written or when
  an optional dependency (e.g. ``matplotlib``) is missing.
* :data:`EXPERIMENT_REGISTRY` maps the paper-facing experiment id (used in YAML
  configs and on the CLI) onto ``(module, entrypoint)`` pairs.
* :data:`REPEATS` centralises the statistical protocol of the paper: 20 repeats
  for Section 5.1, 10 repeats for Section 5.2 and Section 5.3.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

__all__ = [
    # registry / lookup helpers
    "EXPERIMENT_REGISTRY",
    "EXPERIMENT_MODULES",
    "OUT_OF_SCOPE",
    "REPEATS",
    "available_experiments",
    "experiment_id",
    "experiment_spec",
    "get_experiment",
    "import_experiment",
    "run_experiment",
]


# ---------------------------------------------------------------------------
# Registry of in-scope experiment drivers
# ---------------------------------------------------------------------------
#: ``experiment id -> (module path, primary entrypoint, short description)``.
#: The entrypoint is resolved lazily; a driver may additionally expose
#: ``build_argparser`` / ``main`` for CLI use (see ``main.py``).
EXPERIMENT_REGISTRY: Dict[str, Tuple[str, str, str]] = {
    # id                     module                                    entrypoint          description
    "figure1": (
        "lbcs_repro.experiments.figure1_trivial",
        "run_figure1",
        "Figure 1: Eq. (3) fixed-size and Eq. (4) over-minimization failure modes",
    ),
    "figure1_trivial": (
        "lbcs_repro.experiments.figure1_trivial",
        "run_figure1",
        "Figure 1 (alias of 'figure1')",
    ),
    "table1": (
        "lbcs_repro.experiments.table1_prelim",
        "run_table1",
        "Table 1: Section 5.1 preliminary LBCS superiority on MNIST-S "
        "(k in {200,400}, eps in {0.2,0.3,0.4}, 20 repeats)",
    ),
    "table1_prelim": (
        "lbcs_repro.experiments.table1_prelim",
        "run_table1",
        "Table 1 (alias of 'table1')",
    ),
    "table2": (
        "lbcs_repro.experiments.table2_table3_compare",
        "run_table2",
        "Table 2: Section 5.2 test accuracy / coreset size vs. baselines "
        "(F-MNIST, SVHN, CIFAR-10; k in {1000,2000,3000,4000}, eps=0.2, T=500)",
    ),
    "table3": (
        "lbcs_repro.experiments.table2_table3_compare",
        "run_table3",
        "Table 3: same-size comparison (baselines re-run at the LBCS-achieved size)",
    ),
    "table2_table3": (
        "lbcs_repro.experiments.table2_table3_compare",
        "run_table2_table3",
        "Tables 2 and 3 together",
    ),
    "figure2": (
        "lbcs_repro.experiments.figure2_robustness",
        "run_figure2",
        "Figure 2: Section 5.3 robustness to 30% symmetric label noise and "
        "exponential class imbalance (ratio 0.01) on F-MNIST",
    ),
    "figure2_robustness": (
        "lbcs_repro.experiments.figure2_robustness",
        "run_figure2",
        "Figure 2 (alias of 'figure2')",
    ),
    "table8": (
        "lbcs_repro.experiments.table8_sizes",
        "run_table8",
        "Table 8 (Appendix E.3): LBCS optimized coreset sizes under 30%/50% "
        "label corruption and class imbalance",
    ),
    "table9": (
        "lbcs_repro.experiments.table9_search_times",
        "run_table9",
        "Table 9 (Section 6 / Appendix E.4): T-sweep on F-MNIST "
        "(T in {100,200,300,500,800,1500,2000})",
    ),
    "table5": (
        "lbcs_repro.experiments.table5_init",
        "run_table5",
        "Table 5 (Section 6): LBCS with Moderate-based mask initialization",
    ),
    "table6": (
        "lbcs_repro.experiments.table6_cross_arch",
        "run_table6",
        "Table 6 (Section 6): cross-architecture ViT-small / WideResNet on SVHN",
    ),
    "appendix_c3": (
        "lbcs_repro.experiments.appendix_c3",
        "run_appendix_c3",
        "Appendix C.3: Figure 1 settings and probabilistic-baseline details",
    ),
}

#: Canonical (non-alias) experiment ids, in the order they appear in the paper.
CANONICAL_EXPERIMENTS: List[str] = [
    "figure1",
    "table1",
    "table2",
    "table3",
    "table8",
    "figure2",
    "table9",
    "table5",
    "table6",
    "appendix_c3",
]

#: Alias -> canonical id, for config/CLI normalisation.
_ALIASES: Dict[str, str] = {
    "figure1_trivial": "figure1",
    "table1_prelim": "table1",
    "figure2_robustness": "figure2",
    "table2_table3": "table2",
    "tables2_3": "table2",
    "table2_table3_compare": "table2",
    "table5_init": "table5",
    "table6_cross_arch": "table6",
    "table8_sizes": "table8",
    "table9_search_times": "table9",
}

#: Unique module paths referenced by the registry (used for import smoke tests).
EXPERIMENT_MODULES: List[str] = sorted({module for module, _, _ in EXPERIMENT_REGISTRY.values()})

#: Sections of the paper that this reproduction deliberately does not cover.
OUT_OF_SCOPE: Dict[str, str] = {
    "section5.4": "ImageNet-1k coreset selection (out of scope per the reproduction plan)",
    "appendix_e.5": "Continual learning / sequential coreset selection (out of scope)",
    "appendix_e.6": "Streaming coreset selection (out of scope)",
}

#: Statistical protocol of the paper (see the plan's validation approach).
REPEATS: Dict[str, int] = {
    "section5.1": 20,  # Table 1 (MNIST-S)
    "section5.2": 10,  # Tables 2-3 (F-MNIST, SVHN, CIFAR-10)
    "section5.3": 10,  # Figure 2 / Table 8 (noise, imbalance)
    "section6": 10,  # Tables 5, 6, 9 (ablations)
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------
def experiment_id(name: str) -> str:
    """Normalise a user-supplied experiment name to its canonical id.

    Accepts canonical ids (``"table2"``), aliases (``"tables2_3"``) and driver
    module names (``"table2_table3_compare"``); matching is case-insensitive.

    Raises:
        KeyError: if the name is not a known in-scope experiment.
    """
    key = str(name).strip().lower().replace(" ", "").replace("-", "_").replace(".", "")
    if key in _ALIASES:
        return _ALIASES[key]
    if key in EXPERIMENT_REGISTRY:
        return _ALIASES.get(key, key)
    # Module-name style lookups (e.g. "figure1_trivial").
    for candidate, (module, _, _) in EXPERIMENT_REGISTRY.items():
        if module.rsplit(".", 1)[-1] == key:
            return _ALIASES.get(candidate, candidate)
    raise KeyError(
        f"Unknown experiment {name!r}. Available: {sorted(available_experiments())}"
    )


def available_experiments(include_aliases: bool = False) -> List[str]:
    """Return the list of experiment ids exposed by this package."""
    if include_aliases:
        return sorted(EXPERIMENT_REGISTRY)
    return list(CANONICAL_EXPERIMENTS)


def experiment_spec(name: str) -> Dict[str, str]:
    """Return ``{id, module, entrypoint, description}`` for an experiment."""
    key = experiment_id(name)
    module, entrypoint, description = EXPERIMENT_REGISTRY[key]
    return {
        "id": key,
        "module": module,
        "entrypoint": entrypoint,
        "description": description,
    }


def import_experiment(name: str) -> Any:
    """Import and return the driver *module* for ``name`` (lazy import)."""
    key = experiment_id(name)
    module_name = EXPERIMENT_REGISTRY[key][0]
    return importlib.import_module(module_name)


def get_experiment(name: str) -> Callable[..., Any]:
    """Return the primary callable (entrypoint) of a driver.

    The driver module is imported lazily, so a missing/broken driver only
    raises when it is actually requested.
    """
    key = experiment_id(name)
    module_path, entrypoint, _ = EXPERIMENT_REGISTRY[key]
    module = importlib.import_module(module_path)
    fn = getattr(module, entrypoint, None)
    if fn is None:  # tolerant fallback: a driver may only expose run()/main()
        fn = getattr(module, "run", None) or getattr(module, "main", None)
    if fn is None or not callable(fn):
        raise AttributeError(
            f"{module_path!r} does not expose a callable {entrypoint!r} (nor 'run'/'main')"
        )
    return fn


def run_experiment(name: str, **kwargs: Any) -> Any:
    """Execute an experiment driver by name, forwarding ``kwargs``.

    Any argument accepted by the driver entrypoint (dataset, k, epsilon, T,
    repeats, device, output directory, ...) can be passed through here, which is
    what ``main.py`` and ``scripts/run_all.py`` do after reading the YAML
    configs.
    """
    fn = get_experiment(name)
    LOGGER.info("Running experiment %s via %s", experiment_id(name), getattr(fn, "__qualname__", fn))
    return fn(**kwargs)


# ---------------------------------------------------------------------------
# Offline self-test (no GPU / no dataset download required)
# ---------------------------------------------------------------------------
def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Validate the registry bookkeeping and alias handling (offline)."""
    report: Dict[str, Any] = {}
    report["num_registered"] = len(EXPERIMENT_REGISTRY)
    report["num_canonical"] = len(CANONICAL_EXPERIMENTS)
    report["num_aliases"] = len(_ALIASES)

    assert set(CANONICAL_EXPERIMENTS).issubset(EXPERIMENT_REGISTRY), "canonical id missing from registry"

    # Aliases must resolve to canonical ids.
    for alias, canonical in _ALIASES.items():
        assert canonical in EXPERIMENT_REGISTRY, f"alias {alias!r} -> unknown {canonical!r}"
        assert experiment_id(alias) == canonical, f"alias {alias!r} did not resolve to {canonical!r}"

    # Case/whitespace/punctuation-insensitive lookups.
    assert experiment_id(" Table 2 ") == "table2"
    assert experiment_id("table2-table3") == "table2"
    assert experiment_id("FIGURE1") == "figure1"
    assert experiment_id("table2_table3_compare") == "table2"

    # Spec shape.
    spec = experiment_spec("table9")
    assert spec["module"] == "lbcs_repro.experiments.table9_search_times"
    assert spec["entrypoint"] == "run_table9"

    # Out-of-scope sections must be recorded and never registered.
    assert "section5.4" in OUT_OF_SCOPE and "appendix_e.5" in OUT_OF_SCOPE and "appendix_e.6" in OUT_OF_SCOPE
    for key, (module, _, _) in EXPERIMENT_REGISTRY.items():
        assert "imagenet" not in module.lower(), f"{key} references out-of-scope ImageNet"
        assert "stream" not in module.lower(), f"{key} references out-of-scope streaming"
        assert "continual" not in module.lower(), f"{key} references out-of-scope continual learning"

    # Repeat protocol of the paper.
    assert REPEATS["section5.1"] == 20 and REPEATS["section5.2"] == 10 and REPEATS["section5.3"] == 10

    report["modules"] = EXPERIMENT_MODULES
    report["ok"] = True
    if verbose:
        print("[experiments] registry self-test passed")
        for key in CANONICAL_EXPERIMENTS:
            module, entrypoint, description = EXPERIMENT_REGISTRY[key]
            print(f"  - {key:<12} {module}.{entrypoint}  ({description})")
    return report


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _selftest(verbose=True)

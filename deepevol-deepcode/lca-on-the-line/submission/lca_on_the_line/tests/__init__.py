"""Test package for LCA-on-the-Line.

This package contains the metric sanity checks described in the reproduction
plan (Phase A - Phase G):

    python tests/test_lca_sanity.py          # standalone runner
    pytest tests/test_lca_sanity.py          # pytest runner

The package itself is deliberately dependency-free: ``run_all`` and ``main``
are exported lazily so that importing ``lca_on_the_line.tests`` never pulls in
numpy/torch or any of the ``src`` modules.
"""

from typing import Any, List

__all__ = ["run_all", "main", "__version__"]

__version__ = "0.1.0"


def _lazy(name: str) -> Any:
    """Import an attribute of ``tests.test_lca_sanity`` on first use."""
    from importlib import import_module

    module = import_module("tests.test_lca_sanity")
    if not hasattr(module, name):  # pragma: no cover - defensive
        module = import_module(".test_lca_sanity", __name__)
    return getattr(module, name)


def run_all(verbose: bool = True):
    """Run the full standalone sanity suite; returns ``(passed, failed, failures)``."""
    return _lazy("run_all")(verbose=verbose)


def main(argv: List[str] | None = None) -> int:  # pragma: no cover - CLI shim
    """Standalone entry point mirroring ``tests/test_lca_sanity.py``'s main."""
    return _lazy("main")(argv)


def __getattr__(name: str) -> Any:  # PEP 562 lazy module attributes
    if name in {"TESTS", "test_hierarchy_zero_diagonal"}:
        return _lazy(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Test package for the *Fine-tuning RL Models is Secretly a Forgetting Mitigation Problem*
reproduction.

The suite is deliberately dependency-light: every test module guards its heavy imports
(``torch``, ``numpy``, ``yaml``, ``metaworld``, ``nle``) behind ``try/except`` and raises a
local ``SkipTest`` sentinel when an optional dependency is unavailable.  That way the
checks can run:

    * under pytest            ->  ``pytest tests``
    * standalone              ->  ``python -m tests.<module>``
    * in a bare CPU container ->  metric/bookkeeping tests still execute

Available test modules
----------------------
``test_retention_losses``    actor-only EWC / BC / Kickstarting / Fisher / EM unit checks.
``test_robotic_sequence``    Algorithm 1 environment, SAC architecture, retention wiring.
``test_toy_mdp``             Appendix A toy counterexamples (two-state MDP, AppleRetrieval).
``test_forward_transfer``    Appendix F forward-transfer metric and Table 6 rebuild.

Shared helpers
--------------
``SkipTest``         exception class used by the modules to signal a skipped check.
``run_module_tests`` executes a module's ``_TESTS`` registry and reports a summary.
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "SkipTest",
    "TEST_MODULES",
    "available_modules",
    "run_module_tests",
    "run_all",
    "main",
]


class SkipTest(Exception):
    """Raised (and caught) when an optional dependency or feature is unavailable."""


#: Ordered list of test module names shipped with the reproduction.
TEST_MODULES: Tuple[str, ...] = (
    "test_retention_losses",
    "test_robotic_sequence",
    "test_toy_mdp",
    "test_forward_transfer",
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _ensure_root_on_path() -> None:
    """Make sure the repository root is importable as ``src.*``."""
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)


def _import_module(name: str):
    """Import a test module by short name or dotted path."""
    _ensure_root_on_path()
    candidates = []
    if "." in name:
        candidates.append(name)
    else:
        candidates.append("{}.{}".format(__name__, name))
        candidates.append(name)
    last_error: Optional[BaseException] = None
    for candidate in candidates:
        try:
            return importlib.import_module(candidate)
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ImportError("cannot import test module {!r}".format(name))


def available_modules() -> List[str]:
    """Return the names of the test modules that import successfully."""
    found: List[str] = []
    for name in TEST_MODULES:
        try:
            _import_module(name)
        except Exception:
            continue
        found.append(name)
    return found


def run_module_tests(
    module: str,
    quiet: bool = False,
    raise_on_error: bool = False,
) -> Dict[str, List[str]]:
    """Execute the ``_TESTS`` registry of one test module.

    Returns a dict with ``passed``, ``failed`` and ``skipped`` name lists.
    """
    result: Dict[str, List[str]] = {"passed": [], "failed": [], "skipped": []}
    try:
        mod = _import_module(module) if isinstance(module, str) else module
    except Exception as exc:  # pragma: no cover - defensive
        if not quiet:
            print("SKIP {:<28} ({})".format(module, exc))
        result["skipped"].append(str(module))
        return result

    tests: Sequence[Callable[[], None]] = getattr(mod, "_TESTS", ())
    module_label = getattr(mod, "__name__", str(module))
    for test in tests:
        label = "{}.{}".format(module_label.split(".")[-1], getattr(test, "__name__", "test"))
        try:
            test()
        except SkipTest as exc:
            result["skipped"].append(label)
            if not quiet:
                print("SKIP {:<44} {}".format(label, exc))
        except Exception as exc:  # pragma: no cover - reporting path
            result["failed"].append(label)
            print("FAIL {:<44} {}".format(label, exc))
            if raise_on_error:
                raise
            traceback.print_exc()
        else:
            result["passed"].append(label)
            if not quiet:
                print("ok   {:<44}".format(label))
    return result


def run_all(
    modules: Optional[Iterable[str]] = None,
    quiet: bool = False,
) -> Dict[str, Dict[str, List[str]]]:
    """Run every (or the given) test module and return per-module results."""
    names = list(modules) if modules is not None else list(TEST_MODULES)
    return {name: run_module_tests(name, quiet=quiet) for name in names}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI runner: ``python -m tests`` (optionally listing modules)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    quiet = "--quiet" in argv or "-q" in argv
    argv = [a for a in argv if a not in ("--quiet", "-q")]
    names = argv or list(TEST_MODULES)

    totals = {"passed": 0, "failed": 0, "skipped": 0}
    for name in names:
        res = run_module_tests(name, quiet=quiet)
        for key in totals:
            totals[key] += len(res[key])

    print(
        "\n{} passed, {} failed, {} skipped".format(
            totals["passed"], totals["failed"], totals["skipped"]
        )
    )
    return 1 if totals["failed"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

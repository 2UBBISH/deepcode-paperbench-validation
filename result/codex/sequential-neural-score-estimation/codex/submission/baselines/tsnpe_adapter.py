"""Adapter for the TSNPE baseline of Deistler et al. (2022).

The addendum of the reproduction task specifies that the TSNPE method (Section
5.2, Figure 3) should be taken from
``https://github.com/mackelab/tsnpe_neurips``.  That repository ships

* ``sbi/`` -- a fork of ``sbi`` implementing the truncated-posterior proposal
  (``sbi.utils.support_posterior.PosteriorSupport``), and
* ``benchmark/`` -- the wrapper ``benchmark.run_tsnpe.run(...)`` that plugs the
  truncated proposal into the sbibm benchmark interface.

This adapter locates that checkout (``third_party/tsnpe_neurips`` by default,
or the ``TSNPE_NEURIPS_PATH`` environment variable), puts the fork of ``sbi``
ahead of the installed one on ``sys.path``, and exposes TSNPE as a single
function with the same signature as the other baselines.

Set the repository up with::

    bash scripts/setup_third_party.sh
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import torch


DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "tsnpe_neurips"
)


def repo_path() -> str:
    return os.environ.get("TSNPE_NEURIPS_PATH", DEFAULT_PATH)


def ensure_on_path() -> str:
    """Put the TSNPE repository's ``sbi`` fork and ``benchmark`` package on
    ``sys.path`` (in front of the installed ``sbi``)."""
    root = repo_path()
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"TSNPE repository not found at {root}. Run `bash scripts/setup_third_party.sh` "
            "or set TSNPE_NEURIPS_PATH."
        )
    for sub in ("sbi", "benchmark"):
        p = os.path.join(root, sub)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"expected {p} inside the TSNPE repository")
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    return root


def _import_run():
    ensure_on_path()
    try:
        from benchmark.run_tsnpe import run as tsnpe_run  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Could not import benchmark.run_tsnpe from the TSNPE repository. Make sure "
            "`third_party/tsnpe_neurips` is checked out and that its dependencies "
            "(see its environment_vm.yml) are installed."
        ) from exc
    return tsnpe_run


def run_tsnpe(
    task,
    num_samples: int = 10000,
    num_simulations: int = 1000,
    num_observation: int = 1,
    num_rounds: int = 10,
    seed: int = 0,
    **kwargs,
) -> Tuple[torch.Tensor, int]:
    """Run TSNPE (truncated SNPE) on an sbibm task.

    Defaults follow the paper: 10 rounds, the same neural network settings used
    by the sbibm SNPE-C baseline, and rejection sampling of the truncated
    proposal.  Extra ``kwargs`` are forwarded to ``benchmark.run_tsnpe.run``.
    """
    tsnpe_run = _import_run()
    torch.manual_seed(seed)
    out = tsnpe_run(
        task=task,
        num_samples=num_samples,
        num_simulations=num_simulations,
        num_observation=num_observation,
        num_rounds=num_rounds,
        **kwargs,
    )
    samples = out[0]
    num_calls = out[1] if len(out) > 1 else num_simulations
    return samples, int(num_calls)

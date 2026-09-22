"""NPE and SNPE-C baselines, run through the ``sbibm`` library.

The addendum of the reproduction task states that ``sbibm`` should be used to
implement the NPE and SNPE methods.  Different releases of ``sbibm`` expose
these under slightly different names, so this module supports both layouts:

* ``sbibm.algorithms.pytorch.npe`` / ``sbibm.algorithms.pytorch.snpe_c``
  (sbibm <= 1.0.x), and
* ``sbibm.algorithms.snpe`` with ``num_rounds=1`` (NPE) or ``num_rounds=10``
  (SNPE-C) (sbibm >= 1.1).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _import_sbibm_snpe():
    try:
        from sbibm.algorithms.pytorch import npe, snpe_c  # type: ignore

        return "legacy", npe, snpe_c
    except Exception:
        pass
    try:
        from sbibm.algorithms import snpe  # type: ignore

        return "modern", None, snpe
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "sbibm is required for the NPE / SNPE-C baselines "
            "(see the addendum). Install with `pip install sbibm`."
        ) from exc


def run_baseline(
    method: str,
    task,
    num_samples: int = 10000,
    num_simulations: int = 1000,
    num_observation: int = 1,
    num_rounds: Optional[int] = None,
    seed: int = 0,
    **kwargs,
) -> Tuple[torch.Tensor, int]:
    """Run NPE (``method='npe'``) or SNPE-C (``method='snpe_c'``).

    Returns the posterior samples and the number of simulator calls.
    """
    layout, npe, snpe_c = _import_sbibm_snpe()
    torch.manual_seed(seed)

    if method == "npe":
        rounds = 1 if num_rounds is None else num_rounds
    elif method in ("snpe_c", "snpe"):
        rounds = 10 if num_rounds is None else num_rounds
    else:
        raise ValueError(f"unknown baseline {method}")

    if layout == "legacy":
        fn = npe if method == "npe" else snpe_c
        out = fn(
            task=task,
            num_samples=num_samples,
            num_simulations=num_simulations,
            num_observation=num_observation,
            num_rounds=rounds,
            **kwargs,
        )
    else:
        out = snpe_c(
            task=task,
            num_samples=num_samples,
            num_simulations=num_simulations,
            num_observation=num_observation,
            num_rounds=rounds,
            **kwargs,
        )

    samples = out[0]
    num_calls = out[1] if len(out) > 1 else num_simulations
    return samples, int(num_calls)

"""RICE test-suite package.

All test modules in this package are written to be *dependency-tolerant*:
torch / gym / MuJoCo are optional, and the shared helpers in
``rice/tests/_helpers.py`` bootstrap ``sys.path`` so the suite can be run from
several repository layouts (repo root, ``rice/`` or an installed package)::

    pytest rice/tests -q

Coverage maps onto the reproduction plan's validation section:

* ``test_mask_network.py``  -- Algorithm 1 (masking rule Eq. 1, bonus reward,
  importance = P(mask = 0), anti-collapse sanity check).
* ``test_mixed_init.py``    -- CORE #2 Bernoulli(p) roll-in, realised critical
  fraction == beta == p.
* ``test_rnd.py``           -- CORE #3 frozen target f / predictor f_hat,
  normalized intrinsic bonus and decay as coverage grows.
* ``test_fidelity_score.py``-- CORE #5 closed-form metric
  ``log(d / d_max) - log(l / L)`` and the sliding-window pipeline (Exp. I).
* ``test_env_reset.py``     -- CORE #6 Go-Explore style state save/restore.

The package initializer intentionally performs **no** imports so that
``import rice.tests`` stays cheap and never fails on a missing heavy
dependency.
"""

from __future__ import annotations

__all__ = []

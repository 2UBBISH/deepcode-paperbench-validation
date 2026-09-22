"""Minimal, dependency-free 2-D simplex noise.

The ``ant-random-simplex`` evaluation tasks are defined by a random 2-D noise
"height map" generated with opensimplex (see the addendum).  To keep the
repository self-contained we vendor a standard 2-D simplex noise
implementation (Gustavson's algorithm) rather than depending on the
``opensimplex`` package; if ``opensimplex`` *is* installed the wrapper
prefers it so that the exact upstream noise field is reproduced.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_GRAD3 = np.array(
    [[1, 1], [-1, 1], [1, -1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]],
    dtype=np.float64,
)

_F2 = 0.5 * (np.sqrt(3.0) - 1.0)
_G2 = (3.0 - np.sqrt(3.0)) / 6.0


class SimplexNoise2D:
    """Seeded 2-D simplex noise with the same layout as ``opensimplex``."""

    def __init__(self, seed: int = 0, use_opensimplex: bool = True) -> None:
        self.seed = int(seed)
        self._backend = None
        if use_opensimplex:
            try:  # pragma: no cover - optional dependency
                from opensimplex import OpenSimplex  # type: ignore

                self._backend = OpenSimplex(seed=self.seed)
            except Exception:
                self._backend = None
        if self._backend is None:
            rng = np.random.default_rng(self.seed)
            perm = np.arange(256, dtype=np.int32)
            rng.shuffle(perm)
            self._perm = np.concatenate([perm, perm]).astype(np.int32)
            self._perm_mod = self._perm % 8

    # -- public API -----------------------------------------------------------
    def noise2(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorised 2-D simplex noise with values approximately in [-1, 1]."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        shape = np.broadcast_shapes(x.shape, y.shape)
        x = np.broadcast_to(x, shape)
        y = np.broadcast_to(y, shape)
        if self._backend is not None:  # pragma: no cover - optional dependency
            flat = np.array([self._backend.noise2(float(a), float(b)) for a, b in zip(x.ravel(), y.ravel())])
            return flat.reshape(shape)

        s = (x + y) * _F2
        i = np.floor(x + s).astype(np.int64)
        j = np.floor(y + s).astype(np.int64)
        t = (i + j) * _G2
        x0 = x - (i - t)
        y0 = y - (j - t)

        i1 = (x0 > y0).astype(np.int64)
        j1 = 1 - i1
        x1 = x0 - i1 + _G2
        y1 = y0 - j1 + _G2
        x2 = x0 - 1.0 + 2.0 * _G2
        y2 = y0 - 1.0 + 2.0 * _G2

        ii = np.mod(i, 256)
        jj = np.mod(j, 256)
        gi0 = self._perm_mod[ii + self._perm[jj]]
        gi1 = self._perm_mod[ii + i1 + self._perm[jj + j1]]
        gi2 = self._perm_mod[ii + 1 + self._perm[jj + 1]]

        def _contrib(grad_idx, dx, dy):
            g = _GRAD3[grad_idx]
            t = 0.5 - dx * dx - dy * dy
            t = np.maximum(t, 0.0)
            t = t * t * t * t
            return t * (g[..., 0] * dx + g[..., 1] * dy)

        n = (
            _contrib(gi0, x0, y0)
            + _contrib(gi1, x1, y1)
            + _contrib(gi2, x2, y2)
        )
        return 70.0 * n

    def noise2_scalar(self, x: float, y: float) -> float:
        return float(self.noise2(np.array([x]), np.array([y]))[0])


def make_height_and_velocity_fields(
    seed: int,
    xy_min: np.ndarray,
    xy_max: np.ndarray,
    frequency: float = 0.1,
    eps: float = 0.5,
    use_opensimplex: bool = True,
):
    """Build the ``ant-random-simplex`` height map and velocity-preference field.

    Returns a callable ``field(xy) -> (height, vx_pref, vy_pref)`` where:

      * ``height`` is the raw noise value at that position (a "height map" in
        the sense of the addendum), and
      * ``(vx_pref, vy_pref)`` is the normalised gradient of the noise field,
        i.e. the "local preferred velocity direction indicated by the noise
        field".
    """
    noise = SimplexNoise2D(seed, use_opensimplex=use_opensimplex)
    scale = 2.0 * np.pi * frequency

    def height(xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        return noise.noise2(xy[..., 0] * scale, xy[..., 1] * scale)

    def field(xy: np.ndarray):
        xy = np.asarray(xy, dtype=np.float64)
        h = height(xy)
        dx = (height(xy + [eps, 0.0]) - height(xy - [eps, 0.0])) / (2.0 * eps)
        dy = (height(xy + [0.0, eps]) - height(xy - [0.0, eps])) / (2.0 * eps)
        grad = np.stack([dx, dy], axis=-1)
        norm = np.linalg.norm(grad, axis=-1, keepdims=True)
        direction = np.where(norm < 1e-8, 0.0, grad / np.maximum(norm, 1e-8))
        return h, direction[..., 0], direction[..., 1]

    return field

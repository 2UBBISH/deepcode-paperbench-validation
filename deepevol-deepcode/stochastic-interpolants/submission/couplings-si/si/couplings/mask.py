"""Random tiled missingness-mask generator for the in-painting coupling.

Paper, Section 4.1 (In-painting):

    "We set the conditioning variable  xi in {0,1}^{C x W x H} ... For simplicity, the mask takes the
     same value for all channels in a given spatial location in the image. We define the base density
     by the relation  x_0 = xi o x_1 + (1 - xi) o zeta ... During training, the mask is drawn randomly
     by tiling the image into 64 tiles; each tile is selected to enter the mask with probability
     p = 0.3."

Convention used throughout this repository (matching the paper's `x_0` formula):

    xi(x1)[., c, i, j] = 1  <=>  pixel (i, j) is OBSERVED  (kept from x1)
    xi(x1)[., c, i, j] = 0  <=>  pixel (i, j) is MISSING   (replaced by independent noise zeta)

so that ``x_0 = xi * x_1 + (1 - xi) * zeta`` fills the *missing* tiles with noise.  A tile is
"selected to enter the mask" (i.e. becomes missing) with probability ``p = 0.3``.

The generator returns a float tensor of 0/1 values of shape ``(B, 1, H, W)`` (broadcast over
channels, because the mask is shared across channels at every spatial location) or ``(B, C, H, W)``
when ``expand_channels=True``.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch

__all__ = [
    "tile_grid",
    "random_tile_mask",
    "RandomTileMask",
    "default_inpainting_mask",
    "count_missing",
    "missing_fraction",
]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _split_lengths(length: int, n: int) -> Tuple[int, ...]:
    """Split ``length`` into ``n`` contiguous integer chunks (as equal as possible)."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    base = length // n
    if base == 0:
        raise ValueError(f"cannot split spatial length {length} into {n} tiles")
    rem = length - base * n
    return tuple(base + (1 if i < rem else 0) for i in range(n))


def tile_grid(num_tiles: int = 64, height: Optional[int] = None, width: Optional[int] = None) -> Tuple[int, int]:
    """Number of tile rows/columns covering an image for a given number of tiles.

    For a square image (or when no spatial shape is given) the grid is as square as possible.
    ``num_tiles=64`` therefore yields an ``8 x 8`` grid, i.e. a ``32 x 32`` pixel tile at 256x256
    and a ``64 x 64`` pixel tile at 512x512.
    """
    if num_tiles <= 0:
        raise ValueError(f"num_tiles must be positive, got {num_tiles}")

    # exact factor pair, preferring the one closest to square
    best: Optional[Tuple[int, int]] = None
    for rows in range(1, int(math.isqrt(num_tiles)) + 1):
        if num_tiles % rows == 0:
            cols = num_tiles // rows
            best = (rows, cols)
    rows, cols = best if best is not None else (1, num_tiles)

    if height is not None and width is not None and height != width:
        # non-square images: keep the tile count but hint the aspect ratio
        rows, cols = (cols, rows) if (height < width) else (rows, cols)
    return rows, cols


# --------------------------------------------------------------------------------------
# functional mask construction
# --------------------------------------------------------------------------------------
def random_tile_mask(
    shape: Union[Sequence[int], torch.Size, Tuple[int, ...]],
    num_tiles: int = 64,
    missing_prob: float = 0.3,
    generator: Optional[torch.Generator] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    expand_channels: bool = False,
    force_missing: Optional[Sequence[int]] = None,
    force_observed: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Draw a random tiled missingness mask.

    Parameters
    ----------
    shape:
        Either ``(C, H, W)`` (a single image, no batch dimension) or ``(B, C, H, W)``.
    num_tiles:
        Total number of tiles the image is tiled into (paper: 64).
    missing_prob:
        Probability that a given tile enters the mask and is replaced by noise
        (paper: ``p = 0.3``).
    generator, device, dtype:
        RNG / placement options for the returned tensor.
    expand_channels:
        If ``False`` (default) the mask has a single channel ``(B, 1, H, W)`` and is meant to be
        broadcast over channels (the mask is shared across channels at each spatial location, as
        stated in Section 4.1).  If ``True``, it is expanded to all ``C`` channels.
    force_missing / force_observed:
        Optional flat tile indices that are forced missing / observed (useful for deterministic
        evaluation masks or debugging).

    Returns
    -------
    torch.Tensor of 0/1 values with the same spatial size as the input shape.
    """
    shape = tuple(int(s) for s in shape)
    if len(shape) == 3:
        channels, height, width = shape
        batch = 1
        batched = False
    elif len(shape) == 4:
        batch, channels, height, width = shape
        batched = True
    else:
        raise ValueError(f"expected shape (C,H,W) or (B,C,H,W), got {shape}")

    if force_missing is not None and force_observed is not None:
        overlap = set(int(i) for i in force_missing) & set(int(i) for i in force_observed)
        if overlap:
            raise ValueError(f"tile indices in both force_missing and force_observed: {sorted(overlap)}")

    rows, cols = tile_grid(num_tiles, height, width)
    row_sizes = _split_lengths(height, rows)
    col_sizes = _split_lengths(width, cols)

    # per-image tile selection: Bernoulli(missing_prob) for each of the num_tiles tiles
    probs = torch.full((batch, num_tiles), float(missing_prob), device=device, dtype=dtype)
    u = torch.rand((batch, num_tiles), generator=generator, device=device, dtype=dtype)
    missing_tiles = (u < probs).to(dtype)  # (B, num_tiles), 1 == missing

    if force_missing is not None:
        for idx in force_missing:
            missing_tiles[:, int(idx)] = 1.0
    if force_observed is not None:
        for idx in force_observed:
            missing_tiles[:, int(idx)] = 0.0

    # build the mask: xi = 1 - missing
    obs = 1.0 - missing_tiles
    tile_rows = []
    tile_index = 0
    for r, hs in enumerate(row_sizes):
        tile_cols = []
        for c, ws in enumerate(col_sizes):
            v = obs[:, tile_index].view(batch, 1, 1, 1)
            tile_cols.append(v.expand(batch, 1, hs, ws))
            tile_index += 1
        tile_rows.append(torch.cat(tile_cols, dim=3))
    xi = torch.cat(tile_rows, dim=2)  # (B, 1, H, W)

    if expand_channels and channels != 1:
        xi = xi.expand(batch, channels, height, width)
    if not batched:
        xi = xi.squeeze(0)
    return xi


# --------------------------------------------------------------------------------------
# class wrapper
# --------------------------------------------------------------------------------------
class RandomTileMask:
    """Random tiled missingness-mask sampler (Section 4.1).

    Each call draws a fresh mask whose tiles are independently selected to be *missing* with
    probability ``missing_prob``.  Returned values are 1 where the image is observed, 0 where the
    pixel is replaced by noise in the coupling ``x_0 = xi o x_1 + (1 - xi) o zeta``.
    """

    def __init__(
        self,
        num_tiles: int = 64,
        missing_prob: float = 0.3,
        expand_channels: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if not 0.0 <= float(missing_prob) <= 1.0:
            raise ValueError(f"missing_prob must lie in [0, 1], got {missing_prob}")
        self.num_tiles = int(num_tiles)
        self.missing_prob = float(missing_prob)
        self.expand_channels = bool(expand_channels)
        self.generator = generator

    # -- public API -------------------------------------------------------------------
    def sample(
        self,
        shape: Union[Sequence[int], torch.Size, torch.Tensor],
        generator: Optional[torch.Generator] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Draw a mask for ``shape`` (a shape tuple or a tensor whose shape is used)."""
        if isinstance(shape, torch.Tensor):
            device = shape.device if device is None else device
            shape = tuple(shape.shape)
        return random_tile_mask(
            shape,
            num_tiles=self.num_tiles,
            missing_prob=self.missing_prob,
            generator=self.generator if generator is None else generator,
            device=device,
            expand_channels=self.expand_channels,
            **kwargs,
        )

    # allow ``mask(x)`` / ``mask(shape)`` usage
    __call__ = sample

    @property
    def keep_prob(self) -> float:
        """Probability that a tile is observed (kept from ``x_1``)."""
        return 1.0 - self.missing_prob

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_tiles={self.num_tiles}, "
            f"missing_prob={self.missing_prob}, expand_channels={self.expand_channels})"
        )


def default_inpainting_mask(
    num_tiles: int = 64, missing_prob: float = 0.3, expand_channels: bool = False
) -> RandomTileMask:
    """Mask generator with the paper's training defaults (64 tiles, p = 0.3)."""
    return RandomTileMask(
        num_tiles=num_tiles, missing_prob=missing_prob, expand_channels=expand_channels
    )


# --------------------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------------------
def count_missing(xi: torch.Tensor) -> torch.Tensor:
    """Number of missing pixels (xi == 0) per image; ``xi`` may be (B,1,H,W) or (B,C,H,W)."""
    return (xi == 0).sum(dim=tuple(range(1, xi.dim())))


def missing_fraction(xi: torch.Tensor) -> torch.Tensor:
    """Fraction of missing pixels per image."""
    per_image = xi[0].numel() if False else xi.reshape(xi.shape[0], -1).shape[1]
    return count_missing(xi).to(torch.float64) / float(per_image)


# --------------------------------------------------------------------------------------
# self test
# --------------------------------------------------------------------------------------
def _self_test() -> None:
    torch.manual_seed(0)
    m = default_inpainting_mask()

    for shape in [(3, 256, 256), (2, 3, 512, 512), (1, 3, 64, 64)]:
        xi = m.sample(shape)
        assert set(torch.unique(xi).tolist()) <= {0.0, 1.0}, "mask must be binary"
        assert xi.shape[-2:] == shape[-2:], (xi.shape, shape)

    # channel sharing + tiling check on a small image
    xi = m.sample((1, 3, 256, 256))
    assert xi.shape == (1, 1, 256, 256)
    assert int(xi.sum().item()) % (32 * 32) == 0, "mask must be a union of 32x32 tiles"

    # forced tiles
    xi = random_tile_mask((1, 3, 256, 256), force_missing=[0, 1], force_observed=list(range(2, 64)))
    assert xi[0, 0, :32, :32].min().item() == 0.0  # tile 0 missing
    assert xi[0, 0, 64:96, 64:96].min().item() == 1.0  # tile 9 observed

    # empirical missing fraction close to p = 0.3 at the tile level
    big = torch.cat([m.sample((8, 3, 256, 256)).reshape(8, -1)[:, ::64 * 8] for _ in range(1)], dim=1)
    tiles_missing = (torch.cat([m.sample((64 * 8 // 8, 3, 256, 256)).reshape(64, 64, 32 * 32)[:, :, 0]], dim=1) == 0).float()
    est = tiles_missing.mean().item()
    assert abs(est - 0.3) < 0.1, est

    expanded = m.sample((2, 3, 128, 128), expand_channels=True)
    assert expanded.shape == (2, 3, 128, 128)
    assert torch.allclose(expanded[:, 0:1], expanded[:, 1:2])

    print("mask.py self-test passed (num_tiles=%d, p=%.2f)" % (m.num_tiles, m.missing_prob))


if __name__ == "__main__":
    _self_test()

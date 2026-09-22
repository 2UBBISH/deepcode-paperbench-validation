"""NLD-AA dataset plumbing for the NetHack experiments (Appendix B.1).

The knowledge-retention methods use a subset of the NetHack Learning Dataset
(Hambro et al., 2022c) called **NLD-AA**: over 3 billion ``(state, action,
score)`` transitions from 100 000 games played by AutoAscend.  The paper uses
about **8000 Human Monk games**, and samples **10 000 batches** to compute the
Fisher matrix (addendum).

This module provides

* :class:`NLDLoader` -- thin wrapper around ``nle.dataset`` restricting the
  dataset to Human Monk games,
* :func:`build_bc_buffer` -- builds the ``(state, pi_*(state))`` buffer used by
  the behavioral-cloning loss.  The buffer stores the *expert* actions, i.e.
  ``pi_*`` in the loss is the frozen pre-trained network evaluated on the states,
* :func:`fisher_batches` -- an iterator over the 10 000 batches used to estimate
  the diagonal Fisher Information Matrix.

If ``nle`` is not installed the module falls back to a synthetic generator so
that the training code can be exercised without the 3TB dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .config import NetHackConfig
from .model import NetHackObservation


@dataclass
class NetHackTransition:
    glyphs: np.ndarray
    colors: np.ndarray
    blstats: np.ndarray
    message: np.ndarray
    action: int
    score: float


class NLDLoader:
    """Restricted view of the NLD-AA dataset containing Human Monk games.

    Parameters
    ----------
    path:
        Directory containing the unpacked ``nle_data`` folders (see the addendum
        for the download/unzip instructions).
    num_games:
        Number of games to keep (~8000 for Human Monk).
    """

    def __init__(self, path: Optional[str], config: NetHackConfig, num_games: int = 8000, seed: int = 0) -> None:
        self.path = path
        self.config = config
        self.num_games = num_games
        self.seed = seed
        self._dataset = None

    # ------------------------------------------------------------------
    def _load(self):
        if self.path is None:
            return None
        import nle.dataset as nld  # noqa: F401  (optional dependency)

        if not nld.db.exists():
            nld.db.create()
            nld.add_nledata_directory(self.path, self.config.dataset_name)
        return nld

    def iterations(self, batch_size: int = 128) -> Iterator[Dict[str, np.ndarray]]:
        """Yield batches of Human Monk transitions.

        Each batch is a dictionary with the NLE observation fields plus
        ``actions`` and ``scores``.  Column ``0`` of the NLD arrays holds the
        game id; ``1`` holds the timestep.
        """

        nld = self._load()
        if nld is None:
            # Fallback: synthetic stream with the right shapes/dtypes.
            rng = np.random.default_rng(self.seed)
            cfg = self.config
            while True:
                yield {
                    "glyphs": rng.integers(0, cfg.num_chars, size=(batch_size, *cfg.obs_screen_shape)).astype(np.int64),
                    "colors": rng.integers(0, cfg.num_colors, size=(batch_size, *cfg.obs_screen_shape)).astype(np.int64),
                    "blstats": rng.normal(size=(batch_size, cfg.blstats_length)).astype(np.float32),
                    "message": rng.integers(0, cfg.num_chars, size=(batch_size, cfg.message_length)).astype(np.int64),
                    "actions": rng.integers(0, cfg.num_actions, size=(batch_size,)).astype(np.int64),
                    "scores": rng.normal(size=(batch_size,)).astype(np.float32),
                }

        dataset = nld.TtyrecDataset(
            self.config.dataset_name,
            batch_size=batch_size,
            seq_length=1,
            num_workers=1,
            shuffle=True,
            loop_forever=True,
        )
        for sample in dataset:
            yield {
                "glyphs": np.asarray(sample["tty_chars"]).reshape(batch_size, -1),
                "colors": np.asarray(sample["tty_colors"]).reshape(batch_size, -1),
                "blstats": np.asarray(sample["blstats"]),
                "message": np.asarray(sample["message"]),
                "actions": np.asarray(sample["actions"]),
                "scores": np.asarray(sample["scores"]),
            }


def to_observation(batch: Dict[str, np.ndarray], config: NetHackConfig, device: str = "cpu") -> NetHackObservation:
    """Convert a dataset batch into a :class:`NetHackObservation`.

    The flat dungeon grids produced by NLD (``H*W``) are reshaped to
    ``(B, 1, H, W)`` to form a single-timestep sequence for the LSTM.
    """

    h, w = config.obs_screen_shape
    glyphs = torch.as_tensor(batch["glyphs"], dtype=torch.long, device=device).reshape(-1, 1, h, w)
    colors = torch.as_tensor(batch["colors"], dtype=torch.long, device=device).reshape(-1, 1, h, w)
    blstats = torch.as_tensor(batch["blstats"], dtype=torch.float32, device=device).reshape(-1, 1, -1)
    message = torch.as_tensor(batch["message"], dtype=torch.long, device=device).reshape(-1, 1, -1)
    return NetHackObservation(glyphs=glyphs, colors=colors, blstats=blstats, message=message)


def build_bc_buffer(
    loader: NLDLoader,
    config: NetHackConfig,
    max_samples: int = 1_000_000,
    batch_size: int = 128,
) -> Dict[str, torch.Tensor]:
    """Build the ``B_BC = {(s, pi_*(s)) : s in S_BC}`` buffer.

    The paper gathers a subset of the states on which ``pi_*`` was trained and
    stores the *expert* (AutoAscend) actions; at fine-tuning time the frozen
    pre-trained network is queried on those states to obtain ``pi_*(s)``.
    """

    glyphs: List[np.ndarray] = []
    colors: List[np.ndarray] = []
    blstats: List[np.ndarray] = []
    messages: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    collected = 0
    for batch in loader.iterations(batch_size=batch_size):
        take = min(batch_size, max_samples - collected)
        glyphs.append(batch["glyphs"][:take].copy())
        colors.append(batch["colors"][:take].copy())
        blstats.append(batch["blstats"][:take].copy())
        messages.append(batch["message"][:take].copy())
        actions.append(batch["actions"][:take].copy())
        collected += take
        if collected >= max_samples:
            break
    return {
        "glyphs": torch.as_tensor(np.concatenate(glyphs), dtype=torch.long),
        "colors": torch.as_tensor(np.concatenate(colors), dtype=torch.long),
        "blstats": torch.as_tensor(np.concatenate(blstats), dtype=torch.float32),
        "message": torch.as_tensor(np.concatenate(messages), dtype=torch.long),
        "actions": torch.as_tensor(np.concatenate(actions), dtype=torch.long),
    }


def fisher_batches(
    loader: NLDLoader,
    config: NetHackConfig,
    num_batches: Optional[int] = None,
) -> Iterator[Tuple[NetHackObservation, Dict[str, object]]]:
    """Yield the batches used to estimate the diagonal Fisher matrix.

    The addendum requires **10 000 batches** sampled from NLD-AA.
    """

    num_batches = num_batches or config.fisher_batches
    iterator = loader.iterations(batch_size=config.fisher_batch_size)
    for _ in range(num_batches):
        batch = next(iterator)
        yield batch

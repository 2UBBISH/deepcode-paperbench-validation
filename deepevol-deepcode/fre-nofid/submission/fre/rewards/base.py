"""Shared reward-function interface for FRE.

In FRE a *reward function* is a Markovian map ``eta: S -> [-1, 1]``.  Both the
unsupervised reward *prior* used for pretraining (Section 4.2 / Appendix B) and
the ground-truth task rewards used for zero-shot evaluation are represented with
this single interface, so that the encoder, decoder and evaluation harness can
treat them interchangeably.

Design notes
------------
* Every reward function accepts either ``numpy`` arrays or ``torch`` tensors of
  shape ``(D,)``, ``(N, D)`` or ``(B, K, D)`` and returns an array/tensor with
  the leading dimensions preserved.  This keeps the reward functions usable
  from the numpy-based replay buffer / env wrappers *and* from torch training
  loops.
* Rewards are clipped into ``[-clip, clip]`` (``clip = 1`` by default) because
  the encoder discretises rewards by rescaling ``[-1, 1] -> [0, 1]`` and
  flooring into 32 bins (see :mod:`fre.fre.reward_embeddings`).
* ``done`` implements the optional done-mask of the goal-reaching family
  (``True`` once the goal has been reached) which the IQL trainer uses to stop
  bootstrapping past the goal.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # torch is optional so that reward functions can be used for analysis only
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch is a hard requirement in practice
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# tensor / array helpers
# ---------------------------------------------------------------------------
def is_torch_tensor(x: Any) -> bool:
    """Return ``True`` if ``x`` is a torch tensor."""
    return bool(_TORCH_AVAILABLE and torch.is_tensor(x))


def to_numpy(x: Any, dtype: Any = np.float32) -> np.ndarray:
    """Convert a torch tensor / list / scalar to a numpy array."""
    if is_torch_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def to_torch(x: Any, device: Any = None, dtype: Any = None) -> Any:
    """Convert an array-like to a torch tensor."""
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("PyTorch is required for tensor reward evaluation.")
    if is_torch_tensor(x):
        tensor = x
    else:
        tensor = torch.as_tensor(np.asarray(x), dtype=dtype or torch.float32)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


# ---------------------------------------------------------------------------
# dataset access helpers (work on dicts, replay buffers or raw arrays)
# ---------------------------------------------------------------------------
def get_observations(source: Any) -> np.ndarray:
    """Extract an ``(N, D)`` observation array from a dataset-like object."""
    if source is None:
        raise ValueError("A replay buffer / dataset is required to sample rewards.")
    if isinstance(source, np.ndarray):
        return np.asarray(source, dtype=np.float32)
    if isinstance(source, dict):
        for key in ("observations", "obs", "observation", "states"):
            if key in source and source[key] is not None:
                return to_numpy(source[key])
        raise KeyError(f"No observation key found in dataset dict: {list(source)}")
    for attr in ("observations", "obs", "states"):
        if hasattr(source, attr):
            value = getattr(source, attr)
            if callable(value):
                value = value()
            if value is not None:
                return to_numpy(value)
    if hasattr(source, "as_numpy"):
        try:
            data = source.as_numpy()
            for key in ("observations", "obs"):
                if key in data:
                    return to_numpy(data[key])
        except Exception:
            pass
    raise TypeError(f"Cannot extract observations from {type(source)!r}")


def get_terminals(source: Any) -> Optional[np.ndarray]:
    """Extract episode terminals/dones (``(N,)``) if available, else ``None``."""
    if source is None or isinstance(source, np.ndarray):
        return None
    if isinstance(source, dict):
        for key in ("terminals", "dones", "done", "terminal"):
            if key in source and source[key] is not None:
                return to_numpy(source[key]).astype(bool)
        if "masks" in source and source["masks"] is not None:
            return ~to_numpy(source["masks"]).astype(bool)
        return None
    for attr in ("terminals", "dones", "masks"):
        if hasattr(source, attr):
            value = getattr(source, attr)
            if callable(value):
                continue
            if value is not None:
                arr = to_numpy(value).astype(bool)
                if attr == "masks":
                    arr = ~arr
                return arr
    return None


def episode_boundaries(terminals: Optional[np.ndarray], num_states: int) -> List[Tuple[int, int]]:
    """Convert terminal flags into a list of ``(start, end_exclusive)`` episodes."""
    if terminals is None or len(terminals) == 0:
        return [(0, int(num_states))]
    ends = np.flatnonzero(np.asarray(terminals).astype(bool))
    bounds: List[Tuple[int, int]] = []
    start = 0
    for end in ends:
        stop = int(end) + 1
        if stop > start:
            bounds.append((start, stop))
        start = stop
    if start < num_states:
        bounds.append((start, int(num_states)))
    if not bounds:
        bounds = [(0, int(num_states))]
    return bounds


def get_rng(rng: Any = None, seed: Optional[int] = None) -> np.random.Generator:
    """Return a numpy ``Generator`` from a generator/seed/None."""
    if isinstance(rng, np.random.Generator):
        return rng
    seed = seed if seed is not None else (rng if isinstance(rng, (int, np.integer)) else None)
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# RewardFunction
# ---------------------------------------------------------------------------
class RewardFunction:
    """Base class for a Markovian reward function ``eta(s)``.

    Subclasses implement :meth:`_compute`, which receives an ``(N, D)`` float32
    numpy array and returns an ``(N,)`` array.  The base class handles
    dtype/shape/device handling, optional state standardisation and clipping.
    """

    family: str = "base"

    def __init__(
        self,
        state_dim: Optional[int] = None,
        name: Optional[str] = None,
        clip: Optional[float] = 1.0,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        normalise: bool = False,
        device: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.state_dim = int(state_dim) if state_dim is not None else None
        self.name = name or self.family
        self.clip = clip
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)
        self.normalise = bool(normalise and self.mean is not None and self.std is not None)
        self.device = device
        self.metadata: Dict[str, Any] = dict(metadata or {})

    # -- to be implemented by subclasses -----------------------------------
    def _compute(self, states: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def done_numpy(self, states: np.ndarray) -> np.ndarray:
        """Done-mask (elementwise).  Default: never done."""
        return np.zeros(states.shape[0], dtype=bool)

    # -- preprocessing ------------------------------------------------------
    def _prepare(self, states: Any) -> Tuple[np.ndarray, Tuple[int, ...]]:
        arr = to_numpy(states)
        original_shape = arr.shape
        if arr.ndim <= 1:
            arr = arr.reshape(1, -1)
        else:
            arr = arr.reshape(-1, arr.shape[-1])
        if self.state_dim is None:
            self.state_dim = int(arr.shape[-1])
        if self.normalise and self.mean is not None and self.std is not None:
            arr = (arr - self.mean) / (self.std + 1e-8)
        return arr.astype(np.float32, copy=False), original_shape

    def _postprocess(self, values: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if self.clip is not None:
            values = np.clip(values, -float(self.clip), float(self.clip))
        if len(original_shape) <= 1:
            return values.reshape(original_shape if original_shape else (1,))
        return values.reshape(original_shape[:-1])

    # -- public API ---------------------------------------------------------
    def compute_numpy(self, states: Any) -> np.ndarray:
        """Evaluate the reward function with numpy in / numpy out."""
        arr, original_shape = self._prepare(states)
        return self._postprocess(self._compute(arr), original_shape)

    def compute_done_numpy(self, states: Any) -> np.ndarray:
        arr, original_shape = self._prepare(states)
        done = np.asarray(self.done_numpy(arr), dtype=bool).reshape(-1)
        if len(original_shape) <= 1:
            return done.reshape(original_shape if original_shape else (1,))
        return done.reshape(original_shape[:-1])

    def __call__(self, states: Any) -> Any:
        if is_torch_tensor(states):
            device = states.device
            values = self.compute_numpy(states)
            return to_torch(values, device=device, dtype=states.dtype)
        return self.compute_numpy(states)

    # aliases -- different parts of the codebase use different names
    def evaluate(self, states: Any) -> Any:
        return self(states)

    def reward(self, states: Any) -> Any:
        return self(states)

    def forward(self, states: Any) -> Any:
        return self(states)

    def done(self, states: Any) -> Any:
        if is_torch_tensor(states):
            device = states.device
            values = self.compute_done_numpy(states)
            return to_torch(values, device=device, dtype=torch.bool if _TORCH_AVAILABLE else None)
        return self.compute_done_numpy(states)

    # -- introspection ------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        info = dict(self.metadata)
        info.update({"name": self.name, "family": self.family, "state_dim": self.state_dim})
        return info

    def extra_repr(self) -> str:
        return "name={}, family={}, state_dim={}".format(self.name, self.family, self.state_dim)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}({self.extra_repr()})"


# ---------------------------------------------------------------------------
# RewardFunctionPrior
# ---------------------------------------------------------------------------
class RewardFunctionPrior:
    """Base class for reward-function *samplers* (the reward prior ``p(eta)``)."""

    family: str = "base"

    def __init__(
        self,
        state_dim: int,
        source: Any = None,
        rng: Any = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.state_dim = int(state_dim)
        self.source = source
        self.rng = get_rng(rng, seed)
        self.seed = seed
        self.kwargs = kwargs

    def set_source(self, source: Any) -> None:
        """Attach/replace the dataset used for state/goal sampling."""
        self.source = source

    def sample_functions(
        self, num_functions: int = 1, source: Any = None, rng: Any = None, **kwargs: Any
    ) -> List[RewardFunction]:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, num_functions: int = 1, **kwargs: Any) -> List[RewardFunction]:
        return self.sample_functions(num_functions, **kwargs)

    def sample(self, num_functions: int = 1, **kwargs: Any) -> List[RewardFunction]:
        return self.sample_functions(num_functions, **kwargs)

    def evaluate(self, functions: Sequence[RewardFunction], states: Any) -> np.ndarray:
        """Evaluate a batch of reward functions on a batch of states."""
        arr = np.asarray([[fn(states)] for fn in functions]) if False else None  # placeholder
        rows = [np.asarray(fn(states), dtype=np.float32).reshape(-1) for fn in functions]
        return np.stack(rows, axis=0) if rows else np.zeros((0, 0), dtype=np.float32)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(state_dim={self.state_dim})"
